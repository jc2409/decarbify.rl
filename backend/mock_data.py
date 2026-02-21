"""
Mock sequential observation data for GreenDispatch inference pipeline.

Generates 288-timestep (3-day, 15-min interval) sequences with realistic
real-world patterns for carbon intensity, electricity price, resource
availability, and task arrivals.  See PLAN_INFERENCE_PIPELINE.md for the
full pattern specification.
"""

import math
import random
from datetime import datetime, timedelta, timezone
from typing import List

# ── DC metadata ───────────────────────────────────────────────────────────────

DC_IDS = ["DC1", "DC2", "DC3", "DC4", "DC5"]

_DC_LOCATIONS = {
    "DC1": "California",
    "DC2": "Germany",
    "DC3": "Chile",
    "DC4": "Singapore",
    "DC5": "Australia",
}

# Origin-DC distribution for task generation (proportional to DC size).
_ORIGIN_WEIGHTS = {"DC1": 0.30, "DC2": 0.25, "DC3": 0.15, "DC4": 0.20, "DC5": 0.10}


# ── Interpolation helper ─────────────────────────────────────────────────────

def _interp(hour: float, anchors: list[tuple[float, float]]) -> float:
    """Linearly interpolate a value from (hour, value) anchor points.

    *anchors* must be sorted by hour and wrap around midnight.  The last
    anchor is assumed to connect back to the first at hour 24.
    """
    # Wrap to [0, 24)
    hour = hour % 24.0

    for i in range(len(anchors)):
        h0, v0 = anchors[i]
        h1, v1 = anchors[(i + 1) % len(anchors)]
        # Handle wrap-around midnight
        if h1 <= h0:
            h1 += 24.0
        hh = hour if hour >= h0 else hour + 24.0
        if h0 <= hh <= h1:
            t = (hh - h0) / (h1 - h0) if h1 != h0 else 0.0
            return v0 + t * (v1 - v0)
    # Fallback (should not happen with correct anchors)
    return anchors[0][1]


# ══════════════════════════════════════════════════════════════════════════════
# Carbon intensity generators (per DC, per fractional-hour)
# ══════════════════════════════════════════════════════════════════════════════

_CI_ANCHORS_DC1 = [
    (0, 230), (6, 240), (10, 180), (13, 155), (16, 195), (19, 260), (22, 235),
]

_CI_ANCHORS_DC2_CALM = [
    (0, 310), (6, 330), (10, 350), (14, 360), (18, 345), (22, 320),
]
_CI_ANCHORS_DC2_WINDY = [
    (0, 280), (6, 270), (9, 210), (12, 200), (15, 220), (18, 290), (22, 300),
]

_CI_ANCHORS_DC4 = [
    (0, 420), (9, 475), (14, 460), (18, 450), (22, 430),
]

_CI_ANCHORS_DC5 = [
    (0, 600), (8, 660), (13, 620), (18, 670), (23, 610),
]


def _carbon_intensity(dc_id: str, hour: float, day_index: int, rng: random.Random) -> float:
    """Return carbon intensity (gCO₂/kWh) for *dc_id* at fractional *hour*."""
    noise_std: float

    if dc_id == "DC1":
        base = _interp(hour, _CI_ANCHORS_DC1)
        day_scale = 1.0 + rng.gauss(0, 0.03)  # ±5% day-to-day
        noise_std = 10.0
        return max(80.0, base * day_scale + rng.gauss(0, noise_std))

    if dc_id == "DC2":
        # Day 2 (index 1) is the windy day
        anchors = _CI_ANCHORS_DC2_WINDY if day_index == 1 else _CI_ANCHORS_DC2_CALM
        base = _interp(hour, anchors)
        noise_std = 15.0
        return max(120.0, base + rng.gauss(0, noise_std))

    if dc_id == "DC3":
        # Hydro-dominated, very stable
        base = 195.0 + 10.0 * math.sin(2 * math.pi * hour / 24)  # tiny midday bump
        noise_std = 12.0
        return max(100.0, base + rng.gauss(0, noise_std))

    if dc_id == "DC4":
        base = _interp(hour, _CI_ANCHORS_DC4)
        noise_std = 10.0
        return max(350.0, base + rng.gauss(0, noise_std))

    # DC5
    base = _interp(hour, _CI_ANCHORS_DC5)
    noise_std = 15.0
    return max(500.0, base + rng.gauss(0, noise_std))


# ══════════════════════════════════════════════════════════════════════════════
# Electricity price generators
# ══════════════════════════════════════════════════════════════════════════════

_PRICE_ANCHORS_DC1 = [
    (0, 0.12), (6, 0.11), (10, 0.09), (13, 0.07), (16, 0.12),
    (18, 0.22), (20, 0.20), (22, 0.13),
]

_PRICE_ANCHORS_DC2_NORMAL = [
    (0, 0.15), (5, 0.14), (9, 0.20), (13, 0.22), (17, 0.21), (20, 0.18), (23, 0.15),
]
_PRICE_ANCHORS_DC2_WINDY = [
    (0, 0.13), (5, 0.12), (9, 0.13), (12, 0.11), (15, 0.12), (18, 0.16), (23, 0.14),
]

_PRICE_ANCHORS_DC5 = [
    (0, 0.08), (6, 0.09), (10, 0.12), (14, 0.14), (18, 0.16), (22, 0.10),
]


def _electricity_price(dc_id: str, hour: float, day_index: int, rng: random.Random) -> float:
    """Return electricity price ($/kWh) for *dc_id* at fractional *hour*."""
    if dc_id == "DC1":
        base = _interp(hour, _PRICE_ANCHORS_DC1)
        return max(0.03, base + rng.gauss(0, 0.01))

    if dc_id == "DC2":
        anchors = _PRICE_ANCHORS_DC2_WINDY if day_index == 1 else _PRICE_ANCHORS_DC2_NORMAL
        base = _interp(hour, anchors)
        return max(0.05, base + rng.gauss(0, 0.01))

    if dc_id == "DC3":
        # Cheap hydro, almost flat
        base = 0.055 + 0.015 * math.sin(2 * math.pi * (hour - 6) / 24)
        return max(0.03, base + rng.gauss(0, 0.005))

    if dc_id == "DC4":
        # Flat gas-indexed
        base = 0.165 + 0.015 * math.sin(2 * math.pi * (hour - 6) / 24)
        return max(0.10, base + rng.gauss(0, 0.01))

    # DC5
    base = _interp(hour, _PRICE_ANCHORS_DC5)
    return max(0.04, base + rng.gauss(0, 0.008))


# ══════════════════════════════════════════════════════════════════════════════
# Resource availability generators
# ══════════════════════════════════════════════════════════════════════════════

# (overnight_high, midday_trough) for GPU and CPU per DC
_AVAIL_PARAMS = {
    "DC1": {"gpu_high": 0.70, "gpu_low": 0.35, "cpu_high": 0.75, "cpu_low": 0.40},
    "DC2": {"gpu_high": 0.80, "gpu_low": 0.55, "cpu_high": 0.80, "cpu_low": 0.60},
    "DC3": {"gpu_high": 0.60, "gpu_low": 0.28, "cpu_high": 0.65, "cpu_low": 0.35},
    "DC4": {"gpu_high": 0.35, "gpu_low": 0.12, "cpu_high": 0.45, "cpu_low": 0.20},
    "DC5": {"gpu_high": 0.30, "gpu_low": 0.10, "cpu_high": 0.40, "cpu_low": 0.18},
}


def _resource_ratio(high: float, low: float, hour: float, rng: random.Random) -> float:
    """Sinusoidal availability that troughs around 14:00 and peaks around 04:00."""
    # Phase: peak at ~04:00 (offset -10 → cos peaks at 4)
    t = math.cos(2 * math.pi * (hour - 4) / 24)  # 1 at 04:00, -1 at 16:00
    base = low + (high - low) * (t + 1) / 2  # map [-1,1] → [low, high]
    return max(0.05, min(0.95, base + rng.gauss(0, 0.05)))


def _dc_availability(dc_id: str, hour: float, rng: random.Random) -> dict:
    """Return avail_cpu_ratio, avail_gpu_ratio, avail_mem_ratio for one DC."""
    p = _AVAIL_PARAMS[dc_id]
    gpu = _resource_ratio(p["gpu_high"], p["gpu_low"], hour, rng)
    cpu = _resource_ratio(p["cpu_high"], p["cpu_low"], hour, rng)
    mem = _resource_ratio(p["cpu_high"], p["cpu_low"], hour, rng)  # tracks CPU
    return {
        "avail_cpu_ratio": round(cpu, 4),
        "avail_gpu_ratio": round(gpu, 4),
        "avail_mem_ratio": round(mem, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Task arrival generator
# ══════════════════════════════════════════════════════════════════════════════

def _task_arrival_rate(hour: float) -> float:
    """Expected number of tasks per 15-min step given hour-of-day.

    Sinusoidal workday envelope peaking ~11:00, quiet overnight.
    """
    # Bell-shaped curve centred on 11.5, width ~4h
    work = math.exp(-((hour - 11.5) ** 2) / (2 * 4.0**2))
    return 0.2 + 7.8 * work  # range [0.2, 8.0]


def _generate_task(rng: random.Random) -> dict:
    """Generate a single random task dict matching the plan schema."""
    roll = rng.random()
    if roll < 0.60:
        # CPU-only, short
        return {
            "cpu_cores_norm": round(rng.uniform(0.03, 0.10), 4),
            "gpu_cores_norm": 0.0,
            "memory_gb_norm": round(rng.uniform(0.02, 0.10), 4),
            "duration_hours": round(rng.uniform(0.25, 1.0), 2),
            "deadline_hours": round(rng.uniform(1.0, 3.0), 2),
            "origin_dc": rng.choices(DC_IDS, weights=[0.30, 0.25, 0.15, 0.20, 0.10])[0],
        }
    elif roll < 0.90:
        # GPU job, deferrable
        return {
            "cpu_cores_norm": round(rng.uniform(0.05, 0.15), 4),
            "gpu_cores_norm": round(rng.uniform(0.10, 0.40), 4),
            "memory_gb_norm": round(rng.uniform(0.05, 0.20), 4),
            "duration_hours": round(rng.uniform(0.5, 3.0), 2),
            "deadline_hours": round(rng.uniform(2.0, 6.0), 2),
            "origin_dc": rng.choices(DC_IDS, weights=[0.30, 0.25, 0.15, 0.20, 0.10])[0],
        }
    else:
        # GPU job, urgent
        return {
            "cpu_cores_norm": round(rng.uniform(0.05, 0.10), 4),
            "gpu_cores_norm": round(rng.uniform(0.10, 0.30), 4),
            "memory_gb_norm": round(rng.uniform(0.05, 0.15), 4),
            "duration_hours": round(rng.uniform(0.5, 2.0), 2),
            "deadline_hours": round(rng.uniform(0.3, 0.9), 2),
            "origin_dc": rng.choices(DC_IDS, weights=[0.30, 0.25, 0.15, 0.20, 0.10])[0],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_mock_sequence(
    num_timesteps: int = 288,
    seed: int = 42,
    start_date: str = "2025-06-15",
) -> List[dict]:
    """Generate a mock observation sequence.

    Parameters
    ----------
    num_timesteps:
        Number of 15-minute timesteps (default 288 = 3 days).
    seed:
        RNG seed for reproducibility.
    start_date:
        ISO date string for the first timestep (any date works).

    Returns
    -------
    list[dict]
        One observation dict per timestep, matching the schema defined in
        ``PLAN_INFERENCE_PIPELINE.md``.
    """
    rng = random.Random(seed)
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)

    observations: List[dict] = []

    for t in range(num_timesteps):
        ts = start_dt + timedelta(minutes=15 * t)
        hour = ts.hour + ts.minute / 60.0
        day_index = (ts - start_dt).days  # 0, 1, 2

        # Day-to-day amplitude multiplier (day 2 busier)
        day_mult = 1.15 if day_index == 1 else 1.0

        # ── Datacenter states ─────────────────────────────────────────────
        dc_states: dict = {}
        for dc_id in DC_IDS:
            ci = _carbon_intensity(dc_id, hour, day_index, rng)
            price = _electricity_price(dc_id, hour, day_index, rng)
            avail = _dc_availability(dc_id, hour, rng)
            dc_states[dc_id] = {
                "avail_cpu_ratio": avail["avail_cpu_ratio"],
                "avail_gpu_ratio": avail["avail_gpu_ratio"],
                "avail_mem_ratio": avail["avail_mem_ratio"],
                "carbon_intensity_gco2_kwh": round(ci, 1),
                "price_per_kwh": round(price, 4),
            }

        # ── Task for this timestep ────────────────────────────────────────
        # Arrival rate determines whether a task shows up.  For simplicity
        # the pipeline processes one task per timestep (the primary task);
        # arrival rate modulates task resource size and urgency implicitly.
        rate = _task_arrival_rate(hour) * day_mult
        task = _generate_task(rng)

        # Scale task resources slightly with arrival rate (busier periods →
        # bigger jobs on average).
        if rate > 5.0:
            task["cpu_cores_norm"] = min(0.95, task["cpu_cores_norm"] * 1.3)
            task["gpu_cores_norm"] = min(0.95, task["gpu_cores_norm"] * 1.2)

        observations.append({
            "timestep": t,
            "timestamp": ts.isoformat(),
            "task": task,
            "datacenters": dc_states,
        })

    return observations
