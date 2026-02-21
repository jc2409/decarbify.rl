"""
Performance tracker for GreenDispatch.

Produces per-datacenter time series comparing **local-only** performance
(every job stays at its origin DC) against **RL-optimal** performance
(jobs routed by the agent), with delta metrics at each timestep.

See PLAN_INFERENCE_PIPELINE.md for the full specification.
"""

from typing import Dict, List

# Assumed energy consumption per task per 15-min timestep (kWh).
# This is a simplified constant; the real SustainCluster environment would
# compute this from task resource requirements and duration.
_KWH_PER_TASK = 2.5

DC_IDS = ["DC1", "DC2", "DC3", "DC4", "DC5"]


class PerformanceTracker:
    """Computes local vs optimal performance time series per datacenter."""

    def compute(
        self,
        observations: List[dict],
        decisions: List[dict],
    ) -> Dict[str, List[dict]]:
        """Compute performance comparison.

        Parameters
        ----------
        observations:
            Observation sequence from ``mock_data.generate_mock_sequence()``.
        decisions:
            Decision sequence from ``InferenceEngine.run_sequence()``.

        Returns
        -------
        dict[str, list[dict]]
            Keyed by DC id (``"DC1"``…``"DC5"``).  Each value is a list of
            timestep dicts with ``local``, ``optimal``, and ``delta`` fields.
        """
        if len(observations) != len(decisions):
            raise ValueError(
                f"observations ({len(observations)}) and decisions ({len(decisions)}) "
                "must have the same length"
            )

        # Initialise per-DC result lists.
        results: Dict[str, List[dict]] = {dc: [] for dc in DC_IDS}

        for obs, dec in zip(observations, decisions):
            timestep = obs["timestep"]
            timestamp = obs["timestamp"]
            task = obs["task"]
            dcs = obs["datacenters"]

            origin_dc = task["origin_dc"]
            chosen_dc = dec["decision"]["dc_chosen"]
            was_deferred = dec["decision"]["was_deferred"]

            # Energy varies with task size (approximate).
            task_energy = _KWH_PER_TASK * (
                1.0 + task["cpu_cores_norm"] + task["gpu_cores_norm"]
            )

            for dc_id in DC_IDS:
                ci = dcs[dc_id]["carbon_intensity_gco2_kwh"]
                price = dcs[dc_id]["price_per_kwh"]

                # ── Local: task stays at origin DC ────────────────────────
                if dc_id == origin_dc:
                    local_tasks = 1
                    local_carbon = task_energy * ci / 1000.0  # gCO₂ → kg
                    local_cost = task_energy * price
                else:
                    local_tasks = 0
                    local_carbon = 0.0
                    local_cost = 0.0

                # ── Optimal: task goes wherever agent decided ─────────────
                opt_tasks_processed = 0
                opt_tasks_received = 0
                opt_carbon = 0.0
                opt_cost = 0.0

                if was_deferred:
                    # Task deferred — no DC processes it this timestep.
                    # Origin DC still has baseline idle energy (small).
                    if dc_id == origin_dc:
                        idle_energy = task_energy * 0.05  # ~5% idle overhead
                        opt_carbon = idle_energy * ci / 1000.0
                        opt_cost = idle_energy * price
                else:
                    if dc_id == origin_dc and dc_id == chosen_dc:
                        # Task stays at origin (agent agreed with local).
                        opt_tasks_processed = 1
                        opt_carbon = task_energy * ci / 1000.0
                        opt_cost = task_energy * price
                    elif dc_id == origin_dc and dc_id != chosen_dc:
                        # Task leaves origin — origin has no work this step.
                        opt_tasks_processed = 0
                        opt_carbon = 0.0
                        opt_cost = 0.0
                    elif dc_id == chosen_dc and dc_id != origin_dc:
                        # Task arrives at this DC from elsewhere.
                        opt_tasks_received = 1
                        opt_tasks_processed = 1
                        opt_carbon = task_energy * ci / 1000.0
                        opt_cost = task_energy * price

                # ── Delta ─────────────────────────────────────────────────
                carbon_saved = local_carbon - opt_carbon
                cost_saved = local_cost - opt_cost
                pct_carbon = (
                    (carbon_saved / local_carbon * 100.0) if local_carbon > 0 else 0.0
                )

                results[dc_id].append({
                    "timestep": timestep,
                    "timestamp": timestamp,
                    "local": {
                        "carbon_kg": round(local_carbon, 6),
                        "energy_cost_usd": round(local_cost, 6),
                        "tasks_processed": local_tasks,
                        "carbon_intensity": ci,
                    },
                    "optimal": {
                        "carbon_kg": round(opt_carbon, 6),
                        "energy_cost_usd": round(opt_cost, 6),
                        "tasks_processed": opt_tasks_processed,
                        "tasks_received": opt_tasks_received,
                        "carbon_intensity": ci,
                    },
                    "delta": {
                        "carbon_kg_saved": round(carbon_saved, 6),
                        "cost_saved_usd": round(cost_saved, 6),
                        "pct_carbon_saved": round(pct_carbon, 2),
                    },
                })

        return results

    @staticmethod
    def summarise(results: Dict[str, List[dict]]) -> Dict[str, dict]:
        """Aggregate per-DC time series into summary statistics.

        Returns a dict keyed by DC id with totals for local carbon, optimal
        carbon, carbon saved, cost saved, total tasks processed locally, and
        total tasks received from other DCs under optimal routing.
        """
        summaries: Dict[str, dict] = {}

        for dc_id, series in results.items():
            total_local_carbon = sum(s["local"]["carbon_kg"] for s in series)
            total_opt_carbon = sum(s["optimal"]["carbon_kg"] for s in series)
            total_local_cost = sum(s["local"]["energy_cost_usd"] for s in series)
            total_opt_cost = sum(s["optimal"]["energy_cost_usd"] for s in series)
            total_local_tasks = sum(s["local"]["tasks_processed"] for s in series)
            total_opt_tasks = sum(s["optimal"]["tasks_processed"] for s in series)
            total_received = sum(s["optimal"]["tasks_received"] for s in series)

            pct_carbon = (
                ((total_local_carbon - total_opt_carbon) / total_local_carbon * 100.0)
                if total_local_carbon > 0
                else 0.0
            )

            summaries[dc_id] = {
                "local_carbon_kg": round(total_local_carbon, 4),
                "optimal_carbon_kg": round(total_opt_carbon, 4),
                "carbon_saved_kg": round(total_local_carbon - total_opt_carbon, 4),
                "pct_carbon_saved": round(pct_carbon, 2),
                "local_cost_usd": round(total_local_cost, 4),
                "optimal_cost_usd": round(total_opt_cost, 4),
                "cost_saved_usd": round(total_local_cost - total_opt_cost, 4),
                "local_tasks": total_local_tasks,
                "optimal_tasks": total_opt_tasks,
                "tasks_received_from_others": total_received,
            }

        return summaries
