"""
Inference engine for GreenDispatch.

Runs the SAC actor network (or a mock heuristic) step-by-step on a sequence
of observation dicts and returns per-step decisions with confidence scores
and a human-readable reasoning breakdown.

See PLAN_INFERENCE_PIPELINE.md for the full specification.
"""

import math
import os
import sys
from typing import List, Optional

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

DC_IDS = ["DC1", "DC2", "DC3", "DC4", "DC5"]
ACTION_LABELS = ["DEFER", "DC1", "DC2", "DC3", "DC4", "DC5"]

# DCs whose solar window (10–16 UTC) is meaningful for the reasoning text.
_SOLAR_DCS = {"DC1", "DC3"}

# Normalisation constants matching the 34-dim obs layout.
_MAX_DURATION = 10.0
_MAX_DEADLINE = 24.0
_MAX_CI = 800.0
_MAX_PRICE = 0.30

# Capacity threshold below which a DC is considered blocked.
_CAPACITY_THRESHOLD = 0.08


# ══════════════════════════════════════════════════════════════════════════════
# Observation encoding / decoding
# ══════════════════════════════════════════════════════════════════════════════

def encode_observation(obs_dict: dict) -> np.ndarray:
    """Convert a structured observation dict into a flat 34-dim float vector.

    Index layout follows the "Exact 34-Dim Observation Vector Layout" in the
    plan document.
    """
    from datetime import datetime as _dt

    ts = _dt.fromisoformat(obs_dict["timestamp"])
    hour = ts.hour + ts.minute / 60.0
    day_of_year = ts.timetuple().tm_yday

    vec = [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * day_of_year / 365),
        math.cos(2 * math.pi * day_of_year / 365),
    ]

    task = obs_dict["task"]
    vec.extend([
        task["cpu_cores_norm"],
        task["gpu_cores_norm"],
        task["memory_gb_norm"],
        task["duration_hours"] / _MAX_DURATION,
        task["deadline_hours"] / _MAX_DEADLINE,
    ])

    for dc_id in DC_IDS:
        dc = obs_dict["datacenters"][dc_id]
        vec.extend([
            dc["avail_cpu_ratio"],
            dc["avail_gpu_ratio"],
            dc["avail_mem_ratio"],
            dc["carbon_intensity_gco2_kwh"] / _MAX_CI,
            dc["price_per_kwh"] / _MAX_PRICE,
        ])

    return np.array(vec, dtype=np.float32)


def decode_observation(vector: np.ndarray, timestamp: str) -> dict:
    """Reconstruct a structured observation dict from a flat 34-dim vector.

    This is the inverse of :func:`encode_observation` and is used when
    plugging in real SustainCluster environment observations.
    """
    from datetime import datetime as _dt

    ts = _dt.fromisoformat(timestamp)

    task = {
        "cpu_cores_norm": float(vector[4]),
        "gpu_cores_norm": float(vector[5]),
        "memory_gb_norm": float(vector[6]),
        "duration_hours": round(float(vector[7]) * _MAX_DURATION, 2),
        "deadline_hours": round(float(vector[8]) * _MAX_DEADLINE, 2),
        "origin_dc": "DC1",  # cannot recover from vector; default
    }

    dcs = {}
    for i, dc_id in enumerate(DC_IDS):
        base = 9 + i * 5
        dcs[dc_id] = {
            "avail_cpu_ratio": float(vector[base]),
            "avail_gpu_ratio": float(vector[base + 1]),
            "avail_mem_ratio": float(vector[base + 2]),
            "carbon_intensity_gco2_kwh": round(float(vector[base + 3]) * _MAX_CI, 1),
            "price_per_kwh": round(float(vector[base + 4]) * _MAX_PRICE, 4),
        }

    return {
        "timestep": 0,
        "timestamp": timestamp,
        "task": task,
        "datacenters": dcs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Mock actor (heuristic, no weights)
# ══════════════════════════════════════════════════════════════════════════════

class MockActor:
    """Heuristic actor that mimics smart routing using a weighted score.

    Produces realistic-looking logits without any trained weights.  The score
    combines normalised carbon intensity (weight 0.6) and normalised price
    (weight 0.4) — lower is better — with a capacity penalty for overloaded
    DCs.  A small bonus is added for the solar-window DCs during 10–16 UTC.
    """

    def __call__(self, obs_vector: np.ndarray) -> np.ndarray:
        """Return raw logits [6] for the given 34-dim observation.

        Index 0 is the defer action; indices 1–5 are DC1–DC5.
        """
        logits = np.zeros(6, dtype=np.float32)

        # Extract hour from time features
        sin_h, cos_h = float(obs_vector[0]), float(obs_vector[1])
        hour = (math.atan2(sin_h, cos_h) / (2 * math.pi) * 24) % 24

        # Deadline (un-normalised)
        deadline = float(obs_vector[8]) * _MAX_DEADLINE
        task_needs_gpu = float(obs_vector[5]) > 0.01

        for i, dc_id in enumerate(DC_IDS):
            base = 9 + i * 5
            cpu_r = float(obs_vector[base])
            gpu_r = float(obs_vector[base + 1])
            ci_norm = float(obs_vector[base + 3])   # already /800
            pr_norm = float(obs_vector[base + 4])   # already /0.30

            # Lower score → better DC (will be negated to make high = preferred)
            score = 0.6 * ci_norm + 0.4 * pr_norm

            # Capacity penalty
            if task_needs_gpu and gpu_r < _CAPACITY_THRESHOLD:
                score += 2.0  # effectively block
            if cpu_r < _CAPACITY_THRESHOLD:
                score += 2.0

            # Solar bonus
            if dc_id in _SOLAR_DCS and 10 <= hour <= 16:
                score -= 0.08

            # Convert: lower score → higher logit
            logits[i + 1] = -score * 8.0  # scale for softmax spread

        # Defer logit — only competitive when deadline is long and all DCs
        # look bad (all scores high).
        best_dc_logit = float(np.max(logits[1:]))
        if deadline > 2.0 and best_dc_logit < -2.0:
            logits[0] = best_dc_logit + 0.5  # slightly worse than best DC
        else:
            logits[0] = best_dc_logit - 3.0  # much worse → almost never chosen

        return logits


# ══════════════════════════════════════════════════════════════════════════════
# Reasoning generator (4-stage, pure Python)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_reasoning(obs_dict: dict, action_index: int) -> dict:
    """Build a human-readable reasoning dict for the given decision.

    Runs four stages:
    1. Capacity filter
    2. Deadline urgency check
    3. Rank remaining DCs on carbon, cost, capacity
    4. Build summary text
    """
    from datetime import datetime as _dt

    task = obs_dict["task"]
    dcs = obs_dict["datacenters"]
    ts = _dt.fromisoformat(obs_dict["timestamp"])
    hour = ts.hour + ts.minute / 60.0
    needs_gpu = task["gpu_cores_norm"] > 0.01

    # ── Stage 1: capacity filter ──────────────────────────────────────────
    dc_analysis: dict = {}
    for dc_id in DC_IDS:
        dc = dcs[dc_id]
        gpu_blocked = needs_gpu and dc["avail_gpu_ratio"] < _CAPACITY_THRESHOLD
        cpu_blocked = dc["avail_cpu_ratio"] < _CAPACITY_THRESHOLD
        mem_blocked = dc["avail_mem_ratio"] < _CAPACITY_THRESHOLD
        blocked = gpu_blocked or cpu_blocked or mem_blocked

        dc_analysis[dc_id] = {
            "capacity_blocked": blocked,
            "solar_window": dc_id in _SOLAR_DCS and 10 <= hour <= 16,
            "carbon_intensity": dc["carbon_intensity_gco2_kwh"],
            "price_per_kwh": dc["price_per_kwh"],
            "avail_gpu_ratio": dc["avail_gpu_ratio"],
            "avail_cpu_ratio": dc["avail_cpu_ratio"],
        }

    # ── Stage 2: deadline urgency ─────────────────────────────────────────
    defer_eligible = task["deadline_hours"] >= 1.0

    # ── Stage 3: rank unblocked DCs ───────────────────────────────────────
    unblocked = [dc for dc in DC_IDS if not dc_analysis[dc]["capacity_blocked"]]
    if not unblocked:
        unblocked = list(DC_IDS)  # fallback: rank all

    # Carbon rank (lower CI = rank 1)
    by_ci = sorted(unblocked, key=lambda d: dcs[d]["carbon_intensity_gco2_kwh"])
    for rank, dc_id in enumerate(by_ci, 1):
        dc_analysis[dc_id]["carbon_rank"] = rank

    # Cost rank (lower price = rank 1)
    by_price = sorted(unblocked, key=lambda d: dcs[d]["price_per_kwh"])
    for rank, dc_id in enumerate(by_price, 1):
        dc_analysis[dc_id]["cost_rank"] = rank

    # Capacity rank (higher availability = rank 1)
    cap_key = "avail_gpu_ratio" if needs_gpu else "avail_cpu_ratio"
    by_cap = sorted(unblocked, key=lambda d: dcs[d][cap_key], reverse=True)
    for rank, dc_id in enumerate(by_cap, 1):
        dc_analysis[dc_id]["capacity_rank"] = rank

    # Assign default ranks for blocked DCs
    for dc_id in DC_IDS:
        dc_analysis[dc_id].setdefault("carbon_rank", len(DC_IDS))
        dc_analysis[dc_id].setdefault("cost_rank", len(DC_IDS))
        dc_analysis[dc_id].setdefault("capacity_rank", len(DC_IDS))

    # ── Stage 4: build summary ────────────────────────────────────────────
    is_solar = 10 <= hour <= 16

    if action_index == 0:
        # Defer
        avg_ci = sum(dcs[d]["carbon_intensity_gco2_kwh"] for d in DC_IDS) / len(DC_IDS)
        summary = (
            f"No suitable DC — cluster avg carbon intensity {avg_ci:.0f} gCO₂/kWh. "
            f"Deadline allows deferral ({task['deadline_hours']:.1f}h remaining). "
            f"Waiting 15 min for better grid conditions."
        )
        primary_factor = "defer_wait"
    else:
        chosen = ACTION_LABELS[action_index]  # "DC1"…"DC5"
        ci = dcs[chosen]["carbon_intensity_gco2_kwh"]
        price = dcs[chosen]["price_per_kwh"]
        analysis = dc_analysis[chosen]

        avg_ci = sum(dcs[d]["carbon_intensity_gco2_kwh"] for d in DC_IDS) / len(DC_IDS)
        pct_below = (1 - ci / avg_ci) * 100 if avg_ci > 0 else 0

        # Determine primary factor
        if analysis["carbon_rank"] == 1:
            primary_factor = "carbon_intensity"
            summary = (
                f"{chosen} ({_DC_DISPLAY.get(chosen, chosen)}) chosen: "
                f"cleanest grid ({ci:.0f} gCO₂/kWh, {pct_below:.0f}% below cluster avg). "
                f"Cost {'also lowest' if analysis['cost_rank'] == 1 else 'competitive'} "
                f"(${price:.3f}/kWh)."
            )
        elif analysis["cost_rank"] == 1:
            primary_factor = "price"
            summary = (
                f"{chosen} ({_DC_DISPLAY.get(chosen, chosen)}) chosen: "
                f"cheapest electricity (${price:.3f}/kWh). "
                f"Carbon {'acceptable' if ci < avg_ci else 'above average'} "
                f"({ci:.0f} gCO₂/kWh)."
            )
        elif analysis["capacity_blocked"] is False and analysis["capacity_rank"] == 1:
            primary_factor = "capacity"
            summary = (
                f"{chosen} ({_DC_DISPLAY.get(chosen, chosen)}) chosen: "
                f"best available capacity ({cap_key} = "
                f"{dc_analysis[chosen][cap_key]:.2f}). "
                f"Carbon {ci:.0f} gCO₂/kWh, price ${price:.3f}/kWh."
            )
        else:
            primary_factor = "carbon_intensity"
            summary = (
                f"{chosen} ({_DC_DISPLAY.get(chosen, chosen)}) chosen: "
                f"best overall trade-off (carbon {ci:.0f} gCO₂/kWh, "
                f"price ${price:.3f}/kWh)."
            )

        # Append solar note
        if analysis.get("solar_window"):
            summary += " Solar generation active — low carbon window."

        # Append capacity notes
        blocked_names = [d for d in DC_IDS if dc_analysis[d]["capacity_blocked"]]
        if blocked_names:
            summary += f" Note: {', '.join(blocked_names)} at capacity."

    return {
        "summary": summary,
        "primary_factor": primary_factor,
        "defer_eligible": defer_eligible,
        "solar_window": is_solar,
        "dc_analysis": dc_analysis,
    }


# Display labels for reasoning text.
_DC_DISPLAY = {
    "DC1": "California",
    "DC2": "Germany",
    "DC3": "Chile",
    "DC4": "Singapore",
    "DC5": "Australia",
}


# ══════════════════════════════════════════════════════════════════════════════
# Inference engine
# ══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """Runs the SAC actor on observation sequences and produces decisions.

    Parameters
    ----------
    checkpoint_path:
        Path to a real ``best_eval_checkpoint.pth`` file.  If ``None`` or the
        file does not exist, falls back to :class:`MockActor` silently.
    """

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self._actor = None
        self._is_mock = True

        if checkpoint_path and os.path.isfile(checkpoint_path):
            self._actor = self._load_real_actor(checkpoint_path)
            if self._actor is not None:
                self._is_mock = False

        if self._actor is None:
            self._actor = MockActor()

    @staticmethod
    def _load_real_actor(path: str):
        """Attempt to load a real ActorNet from a SustainCluster checkpoint."""
        try:
            import torch

            # Make sure sustain-cluster is importable.
            sc_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sustain-cluster")
            if sc_root not in sys.path:
                sys.path.insert(0, sc_root)

            from rl_components.agent_net import ActorNet  # noqa: PLC0415

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            extra = checkpoint.get("extra_info", {})
            obs_dim = extra.get("obs_dim", 34)
            act_dim = extra.get("act_dim", 6)
            hidden_dim = extra.get("hidden_dim", 256)
            use_layer_norm = extra.get("use_layer_norm", True)

            actor = ActorNet(obs_dim, act_dim, hidden_dim, use_layer_norm=use_layer_norm)
            actor.load_state_dict(checkpoint["actor_state_dict"])
            actor.eval()
            return actor
        except Exception:
            return None

    def _forward(self, obs_vector: np.ndarray) -> np.ndarray:
        """Run actor forward pass and return raw logits [6]."""
        if self._is_mock:
            return self._actor(obs_vector)

        import torch

        obs_t = torch.FloatTensor(obs_vector).unsqueeze(0)
        with torch.no_grad():
            logits = self._actor(obs_t)
        return logits.squeeze(0).cpu().numpy()

    # ── Public API ────────────────────────────────────────────────────────

    def step(self, obs_dict: dict) -> dict:
        """Process a single observation and return the decision dict."""
        obs_vec = encode_observation(obs_dict)
        logits = self._forward(obs_vec)

        # Softmax → confidence scores
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / exp_l.sum()

        action_index = int(np.argmax(logits))
        dc_chosen = ACTION_LABELS[action_index]
        was_deferred = action_index == 0

        confidence = {label: round(float(probs[i]), 4) for i, label in enumerate(ACTION_LABELS)}
        reasoning = _generate_reasoning(obs_dict, action_index)

        return {
            "timestep": obs_dict.get("timestep", 0),
            "timestamp": obs_dict["timestamp"],
            "decision": {
                "action_index": action_index,
                "dc_chosen": dc_chosen,
                "was_deferred": was_deferred,
                "confidence_scores": confidence,
                "reasoning": reasoning,
            },
        }

    def run_sequence(self, observations: List[dict]) -> List[dict]:
        """Run the actor over a full observation sequence.

        Parameters
        ----------
        observations:
            List of observation dicts (e.g. from
            ``mock_data.generate_mock_sequence()``).

        Returns
        -------
        list[dict]
            One decision dict per timestep.
        """
        return [self.step(obs) for obs in observations]

    @property
    def is_mock(self) -> bool:
        """``True`` if using the heuristic actor, ``False`` if using a real checkpoint."""
        return self._is_mock
