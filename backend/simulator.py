"""
SustainCluster evaluation wrapper for GreenDispatch.

Provides SustainClusterSimulator, a clean interface around the SustainCluster
multi-datacenter RL training/evaluation framework.
"""

import contextlib
import copy
import datetime
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _cwd(path: str):
    """Temporarily change the working directory (non-thread-safe)."""
    old = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


# Maps user-facing checkpoint names to relative paths inside the sustain-cluster repo.
CHECKPOINT_MAP = {
    "multi_action_enable_defer_2": (
        "checkpoints/train_multiaction_defer_20250527_210534/best_eval_checkpoint.pth"
    ),
    "multi_action_disable_defer_2": (
        "checkpoints/train_multiaction_nodefer_20250527_212105/best_eval_checkpoint.pth"
    ),
    "single_action_enable_defer_2": (
        "checkpoints/train_single_action_enable_defer_20250527_222926/best_eval_checkpoint.pth"
    ),
    "single_action_disable_defer_2": (
        "checkpoints/train_single_action_disable_defer_20250527_223002/best_eval_checkpoint.pth"
    ),
}

# Static per-strategy configuration.
# RL strategies derive env flags from the checkpoint's extra_info at runtime.
_STRATEGY_CFG = {
    "manual_rl":     {"is_rl": True},
    "local_only":    {"is_rl": False, "env_single_action": False, "env_disable_defer": True},
    "lowest_carbon": {"is_rl": False, "env_single_action": False, "env_disable_defer": True},
}


class SustainClusterSimulator:
    """Wraps the SustainCluster evaluation pipeline for GreenDispatch."""

    def __init__(self, project_root: str) -> None:
        """
        Parameters
        ----------
        project_root:
            Absolute (or relative) path to the cloned sustain-cluster repository.
            Must contain configs/, checkpoints/, data/, etc.
        """
        self.project_root = os.path.abspath(project_root)

        # Make SustainCluster importable.
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)

        # Lazy import after sys.path is set.
        from utils.config_loader import load_yaml  # noqa: PLC0415

        cfg_dir = os.path.join(self.project_root, "configs", "env")
        self.base_sim_cfg = load_yaml(os.path.join(cfg_dir, "sim_config.yaml"))
        self.base_dc_cfg = load_yaml(os.path.join(cfg_dir, "datacenters.yaml"))
        self.base_reward_cfg = load_yaml(os.path.join(cfg_dir, "reward_config.yaml"))

        logger.info("SustainClusterSimulator initialised with project_root=%s", self.project_root)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_paths(self, sim_cfg: dict, dc_cfg: dict) -> None:
        """Resolve relative file paths in config dicts to absolute paths in-place."""
        workload = sim_cfg["simulation"].get("workload_path", "")
        if workload and not os.path.isabs(workload):
            sim_cfg["simulation"]["workload_path"] = os.path.join(self.project_root, workload)

        for dc in dc_cfg.get("datacenters", []):
            dc_file = dc.get("dc_config_file", "")
            if dc_file and not os.path.isabs(dc_file):
                dc["dc_config_file"] = os.path.join(self.project_root, dc_file)
            # Resolve optional HVAC policy path.
            hvac_path = dc.get("hvac_policy_path")
            if hvac_path and not os.path.isabs(hvac_path):
                dc["hvac_policy_path"] = os.path.join(self.project_root, hvac_path)

    def _make_eval_env(
        self,
        strategy: str,
        eval_days: int,
        seed: int,
        env_disable_defer: bool,
        env_single_action: bool,
    ):
        """Instantiate a TaskSchedulingEnv ready for evaluation (not yet reset)."""
        from envs.task_scheduling_env import TaskSchedulingEnv  # noqa: PLC0415
        from rewards.predefined.composite_reward import CompositeReward  # noqa: PLC0415
        from simulation.cluster_manager import DatacenterClusterManager  # noqa: PLC0415

        sim_cfg = copy.deepcopy(self.base_sim_cfg)
        dc_cfg = copy.deepcopy(self.base_dc_cfg)
        reward_cfg = copy.deepcopy(self.base_reward_cfg)

        self._resolve_paths(sim_cfg, dc_cfg)

        sim = sim_cfg["simulation"]
        sim["strategy"] = strategy
        sim["duration_days"] = eval_days
        sim["disable_defer_action"] = env_disable_defer
        sim["single_action_mode"] = env_single_action

        start = pd.Timestamp(
            datetime.datetime(
                sim["year"], sim["month"], sim["init_day"], sim["init_hour"], 0,
                tzinfo=datetime.timezone.utc,
            )
        )
        end = start + datetime.timedelta(days=sim["duration_days"])

        # SustainCluster resolves many data paths relative to CWD, so we must
        # run env construction (and later env interactions) from project_root.
        with _cwd(self.project_root):
            cluster = DatacenterClusterManager(
                config_list=dc_cfg["datacenters"],
                simulation_year=sim["year"],
                # init_day encoded as day-of-year (matches original script).
                init_day=int(sim["month"] * 30.5 + sim["init_day"]),
                init_hour=sim["init_hour"],
                strategy=sim["strategy"],
                tasks_file_path=sim["workload_path"],
                shuffle_datacenter_order=False,   # deterministic for eval
                cloud_provider=sim["cloud_provider"],
                logger=logger,
            )

            reward_fn = CompositeReward(
                components=reward_cfg["reward"]["components"],
                normalize=False,
            )

            env = TaskSchedulingEnv(
                cluster_manager=cluster,
                start_time=start,
                end_time=end,
                reward_fn=reward_fn,
                writer=None,
                sim_config=sim,
            )
        return env

    def _load_actor(self, checkpoint_name: str):
        """
        Load an actor network from a named checkpoint.

        Returns
        -------
        actor : nn.Module  (in eval mode, on CPU)
        single_action_mode : bool
        disable_defer : bool
        """
        from rl_components.agent_net import ActorNet, AttentionActorNet  # noqa: PLC0415
        from utils.checkpoint_manager import load_checkpoint_data  # noqa: PLC0415

        rel_path = CHECKPOINT_MAP[checkpoint_name]
        ckpt_path = os.path.join(self.project_root, rel_path)

        checkpoint_data, _ = load_checkpoint_data(path=ckpt_path, device="cpu")
        if checkpoint_data is None:
            raise FileNotFoundError(
                f"Checkpoint not found or failed to load: {ckpt_path}"
            )

        extra = checkpoint_data.get("extra_info", {})
        obs_dim = extra["obs_dim"]
        act_dim = extra["act_dim"]
        hidden_dim = extra.get("hidden_dim", 64)
        use_layer_norm = extra.get("use_layer_norm", False)
        use_attention = extra.get("use_attention", False)
        single_action_mode = extra.get("single_action_mode", True)
        disable_defer = extra.get("disable_defer_action", False)

        if use_attention:
            actor = AttentionActorNet(
                obs_dim, act_dim,
                embed_dim=extra.get("attn_embed_dim", 128),
                num_heads=extra.get("attn_num_heads", 4),
                num_attention_layers=extra.get("attn_num_layers", 2),
                dropout=extra.get("attn_dropout", 0.1),
            )
        else:
            actor = ActorNet(obs_dim, act_dim, hidden_dim, use_layer_norm=use_layer_norm)

        actor.load_state_dict(checkpoint_data["actor_state_dict"])
        actor.eval()

        logger.info(
            "Loaded actor from %s  (single_action=%s, disable_defer=%s)",
            ckpt_path, single_action_mode, disable_defer,
        )
        return actor, single_action_mode, disable_defer

    @staticmethod
    def _select_actions(
        actor,
        obs,
        single_action_mode: bool,
        disable_defer: bool,
    ):
        """
        Run actor forward pass and return the actions to pass to env.step().

        Returns
        -------
        actions : int  (single-action mode)  or  list[int]  (multi-task mode)
        deferred_count : int   number of tasks deferred this step
        """
        deferred_count = 0

        if single_action_mode:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)   # [1, obs_dim]
            with torch.no_grad():
                logits = actor(obs_t)
            action = torch.argmax(logits, dim=-1).item()
            if not disable_defer and action == 0:
                deferred_count = 1   # single action affects all current tasks
            return action, deferred_count

        # Multi-task mode: obs is a list of per-task observation arrays.
        if len(obs) == 0:
            return [], 0

        obs_t = torch.FloatTensor(np.array(obs))   # [k_t, obs_dim]
        with torch.no_grad():
            logits = actor(obs_t)
        actions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

        if not disable_defer:
            deferred_count = sum(1 for a in actions if a == 0)

        return actions, deferred_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_comparison(
        self,
        strategies: List[str],
        eval_days: int,
        checkpoint_name: str,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run evaluation for each strategy and return three DataFrames.

        Parameters
        ----------
        strategies:
            List of controller names, e.g. ["manual_rl", "local_only", "lowest_carbon"].
        eval_days:
            Simulation duration in days (1–14).
        checkpoint_name:
            One of the keys in CHECKPOINT_MAP. Used only when an RL strategy is present.
        seed:
            RNG seed for env.reset().

        Returns
        -------
        per_dc_df:
            One row per (timestep, datacenter, controller).
            Columns: timestep, datacenter, controller, energy_cost_usd, energy_kwh,
                     carbon_kg, water_usage_m3, cpu_util_pct, gpu_util_pct, mem_util_pct,
                     running_tasks_count, pending_tasks_count, tasks_assigned_count,
                     sla_met_count, sla_violated_count, carbon_intensity_gco2_kwh,
                     price_per_kwh, temperature_c.
        global_df:
            One row per (timestep, controller) with cluster-wide transmission metrics.
            Columns: timestep, controller, transmission_cost_usd,
                     transmission_energy_kwh, transmission_emissions_kg.
        summary_df:
            One row per controller with aggregated totals.
            Columns: Controller, Total CO2 (kg), Total Energy (kWh), Total Cost ($),
                     Total Water (m3), SLA Violation Rate (%), Avg CPU Util (%),
                     Avg GPU Util (%), Total Tasks Deferred.
        """
        per_dc_records: list = []
        global_records: list = []
        summary_records: list = []

        num_steps = eval_days * 24 * 4   # 15-minute timesteps

        for strategy in strategies:
            logger.info("Evaluating strategy: %s  (%d days, seed=%d)", strategy, eval_days, seed)

            cfg = _STRATEGY_CFG.get(strategy)
            if cfg is None:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    f"Known strategies: {list(_STRATEGY_CFG)}"
                )

            is_rl = cfg["is_rl"]

            # ----------------------------------------------------------
            # Build environment + optional actor
            # ----------------------------------------------------------
            if is_rl:
                actor, single_action_mode, disable_defer = self._load_actor(checkpoint_name)
                env = self._make_eval_env(
                    strategy="manual_rl",
                    eval_days=eval_days,
                    seed=seed,
                    env_disable_defer=disable_defer,
                    env_single_action=single_action_mode,
                )
            else:
                actor = None
                single_action_mode = cfg["env_single_action"]
                disable_defer = cfg["env_disable_defer"]
                env = self._make_eval_env(
                    strategy=strategy,
                    eval_days=eval_days,
                    seed=seed,
                    env_disable_defer=disable_defer,
                    env_single_action=single_action_mode,
                )

            # ----------------------------------------------------------
            # Evaluation loop
            # SustainCluster loads data files lazily on reset/step, so the
            # entire loop must also run with CWD = project_root.
            # ----------------------------------------------------------
            total_deferred = 0

            with _cwd(self.project_root):
                obs, _ = env.reset(seed=seed)

                for t in range(num_steps):
                    if is_rl and actor is not None:
                        actions, deferred = self._select_actions(
                            actor, obs, single_action_mode, disable_defer
                        )
                        total_deferred += deferred
                    else:
                        # RBC: env handles routing internally; pass empty action list.
                        actions = []

                    obs, _reward, done, truncated, info = env.step(actions)

                    # ---- Per-DC metrics ----
                    for dc_name, dc_info in info.get("datacenter_infos", {}).items():
                        common = dc_info.get("__common__", {})
                        agent_dc = dc_info.get("agent_dc", {})
                        sla = common.get("__sla__", {})

                        per_dc_records.append({
                            "timestep":                 t,
                            "datacenter":               dc_name,
                            "controller":               strategy,
                            "energy_cost_usd":          common.get("energy_cost_USD", 0.0),
                            "energy_kwh":               common.get("energy_consumption_kwh", 0.0),
                            "carbon_kg":                common.get("carbon_emissions_kg", 0.0),
                            # dc_water_usage is reported in litres by SustainDC; convert to m³.
                            "water_usage_m3":           agent_dc.get("dc_water_usage", 0.0) / 1_000.0,
                            "cpu_util_pct":             common.get("cpu_util_percent", 0.0),
                            "gpu_util_pct":             common.get("gpu_util_percent", 0.0),
                            "mem_util_pct":             common.get("mem_util_percent", 0.0),
                            "running_tasks_count":      common.get("running_tasks", 0),
                            "pending_tasks_count":      common.get("pending_tasks", 0),
                            "tasks_assigned_count":     common.get("tasks_assigned", 0),
                            "sla_met_count":            sla.get("met", 0),
                            "sla_violated_count":       sla.get("violated", 0),
                            # ci is already in gCO2/kWh in SustainCluster.
                            "carbon_intensity_gco2_kwh": common.get("ci", 0.0),
                            "price_per_kwh":            common.get("price_USD_kwh", 0.0),
                            # 'weather' holds ambient temperature in °C.
                            "temperature_c":            common.get("weather", float("nan")),
                        })

                    # ---- Global (cluster-wide) metrics ----
                    global_records.append({
                        "timestep":                   t,
                        "controller":                 strategy,
                        "transmission_cost_usd":      info.get("transmission_cost_total_usd", 0.0),
                        "transmission_energy_kwh":    info.get("transmission_energy_total_kwh", 0.0),
                        "transmission_emissions_kg":  info.get("transmission_emissions_total_kg", 0.0),
                    })

                    if done or truncated:
                        logger.info(
                            "Episode ended at step %d for strategy '%s'", t + 1, strategy
                        )
                        break

            # ----------------------------------------------------------
            # Summary row for this strategy
            # ----------------------------------------------------------
            dc_rows = [r for r in per_dc_records if r["controller"] == strategy]
            if dc_rows:
                total_sla_met = sum(r["sla_met_count"] for r in dc_rows)
                total_sla_viol = sum(r["sla_violated_count"] for r in dc_rows)
                denom = total_sla_met + total_sla_viol
                sla_viol_rate = (total_sla_viol / denom * 100.0) if denom > 0 else 0.0

                summary_records.append({
                    "Controller":           strategy,
                    "Total CO2 (kg)":       sum(r["carbon_kg"] for r in dc_rows),
                    "Total Energy (kWh)":   sum(r["energy_kwh"] for r in dc_rows),
                    "Total Cost ($)":       sum(r["energy_cost_usd"] for r in dc_rows),
                    "Total Water (m3)":     sum(r["water_usage_m3"] for r in dc_rows),
                    "SLA Violation Rate (%)": round(sla_viol_rate, 4),
                    "Avg CPU Util (%)":     float(np.mean([r["cpu_util_pct"] for r in dc_rows])),
                    "Avg GPU Util (%)":     float(np.mean([r["gpu_util_pct"] for r in dc_rows])),
                    "Total Tasks Deferred": total_deferred,
                })

            logger.info("Finished strategy '%s'", strategy)

        per_dc_df = pd.DataFrame(per_dc_records)
        global_df = pd.DataFrame(global_records)
        summary_df = pd.DataFrame(summary_records)

        return per_dc_df, global_df, summary_df
