"""
mock_data.py — Realistic mock data generator for GreenDispatch demo.

Generates DataFrames that mimic SustainClusterSimulator.run_comparison() output
with zero external dependencies.  Data is deterministic for a given seed and
scales proportionally with eval_days.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ── Data Center Definitions ──────────────────────────────────────────────────

DC_INFO = {
    "DC1": {"name": "US-California", "lat": 37.39, "lon": -122.08,
            "zone": "US-CAL-CISO", "flag": "\U0001f1fa\U0001f1f8", "utc_offset": -8},
    "DC2": {"name": "Germany",       "lat": 50.11, "lon": 8.68,
            "zone": "DE",           "flag": "\U0001f1e9\U0001f1ea", "utc_offset": 1},
    "DC3": {"name": "Chile",         "lat": -33.45, "lon": -70.67,
            "zone": "CL-SEN",       "flag": "\U0001f1e8\U0001f1f1", "utc_offset": -4},
    "DC4": {"name": "Singapore",     "lat": 1.35,  "lon": 103.82,
            "zone": "SG",           "flag": "\U0001f1f8\U0001f1ec", "utc_offset": 8},
    "DC5": {"name": "Australia",     "lat": -33.87, "lon": 151.21,
            "zone": "AU-NSW",       "flag": "\U0001f1e6\U0001f1fa", "utc_offset": 11},
}

CONTROLLERS = {
    "SAC RL Agent (Geo+Time)": "manual_rl",
    "Local Only (Baseline)":   "local_only",
    "Lowest Carbon":           "lowest_carbon",
}

CONTROLLER_LABELS = {v: k for k, v in CONTROLLERS.items()}

DC_IDS = list(DC_INFO.keys())

# ── Carbon intensity profiles ────────────────────────────────────────────────
# Each function returns gCO2/kWh for a given local hour (0-24 float).

def _ci_california(local_hour: float, rng: np.random.RandomState) -> float:
    """80-380, massive solar dip midday, dirty evenings. Can be cleanest ~11-3pm."""
    base = 270.0
    solar = -170.0 * math.exp(-0.5 * ((local_hour - 13.0) / 2.0) ** 2)
    evening = 100.0 * math.exp(-0.5 * ((local_hour - 19.0) / 1.8) ** 2)
    night = -20.0 * math.exp(-0.5 * ((local_hour - 3.0) / 3.0) ** 2)
    event = rng.normal(0, 40)  # cloud cover / grid volatility
    return float(np.clip(base + solar + evening + night + event, 80, 380))


def _ci_germany(local_hour: float, rng: np.random.RandomState, wind_factor: float = 1.0) -> float:
    """100-450, highly volatile — wind gusts can make it very clean."""
    base = 300.0 * wind_factor
    morning = 50.0 * math.exp(-0.5 * ((local_hour - 8.0) / 2.0) ** 2)
    afternoon = 30.0 * math.exp(-0.5 * ((local_hour - 15.0) / 3.0) ** 2)
    night = -60.0 * math.exp(-0.5 * ((local_hour - 3.0) / 3.0) ** 2)
    # Wind gusts — large random swings
    wind_gust = rng.normal(0, 55)
    return float(np.clip(base + morning + afternoon + night + wind_gust, 100, 450))


def _ci_chile(local_hour: float, rng: np.random.RandomState) -> float:
    """120-300, generally clean but evening thermal ramp + random drought spikes."""
    base = 200.0
    solar = -50.0 * math.exp(-0.5 * ((local_hour - 13.0) / 2.5) ** 2)
    evening = 60.0 * math.exp(-0.5 * ((local_hour - 20.0) / 2.0) ** 2)
    # Random hydro variability (drought days spike carbon)
    hydro_event = rng.normal(0, 35)
    return float(np.clip(base + solar + evening + hydro_event, 120, 300))


def _ci_singapore(local_hour: float, rng: np.random.RandomState) -> float:
    """250-520, gas-heavy but LNG spot price swings create big variation."""
    base = 390.0
    demand = 30.0 * math.exp(-0.5 * ((local_hour - 14.0) / 3.0) ** 2)
    night_dip = -50.0 * math.exp(-0.5 * ((local_hour - 4.0) / 3.0) ** 2)
    # LNG price volatility
    lng_event = rng.normal(0, 50)
    return float(np.clip(base + demand + night_dip + lng_event, 250, 520))


def _ci_australia(local_hour: float, rng: np.random.RandomState) -> float:
    """200-600, coal baseline but massive solar midday can make it competitive."""
    base = 420.0
    solar = -160.0 * math.exp(-0.5 * ((local_hour - 12.5) / 2.0) ** 2)
    evening = 80.0 * math.exp(-0.5 * ((local_hour - 19.0) / 2.0) ** 2)
    morning = 40.0 * math.exp(-0.5 * ((local_hour - 7.5) / 2.0) ** 2)
    event = rng.normal(0, 40)
    return float(np.clip(base + solar + evening + morning + event, 200, 600))


_CI_FUNCS = {
    "DC1": _ci_california,
    "DC2": _ci_germany,
    "DC3": _ci_chile,
    "DC4": _ci_singapore,
    "DC5": _ci_australia,
}

# ── Price profiles (USD/kWh) ─────────────────────────────────────────────────

def _price_california(local_hour: float, ci: float, rng: np.random.RandomState) -> float:
    base = 0.12
    evening = 0.08 * math.exp(-0.5 * ((local_hour - 19.0) / 2.0) ** 2)
    solar_dip = -0.04 * math.exp(-0.5 * ((local_hour - 13.0) / 2.5) ** 2)
    corr = (ci - 220) * 0.0001
    return float(np.clip(base + evening + solar_dip + corr + rng.normal(0, 0.005), 0.08, 0.22))


def _price_germany(local_hour: float, ci: float, rng: np.random.RandomState) -> float:
    base = 0.18
    peak = 0.08 * math.exp(-0.5 * ((local_hour - 12.0) / 3.0) ** 2)
    corr = (ci - 310) * 0.00015
    return float(np.clip(base + peak + corr + rng.normal(0, 0.008), 0.12, 0.30))


def _price_chile(local_hour: float, ci: float, rng: np.random.RandomState) -> float:
    base = 0.08
    slight_peak = 0.03 * math.exp(-0.5 * ((local_hour - 14.0) / 4.0) ** 2)
    corr = (ci - 155) * 0.00015
    return float(np.clip(base + slight_peak + corr + rng.normal(0, 0.005), 0.06, 0.14))


def _price_singapore(local_hour: float, ci: float, rng: np.random.RandomState) -> float:
    base = 0.14
    corr = (ci - 460) * 0.0001
    return float(np.clip(base + corr + rng.normal(0, 0.006), 0.10, 0.18))


def _price_australia(local_hour: float, ci: float, rng: np.random.RandomState) -> float:
    base = 0.11
    evening = 0.06 * math.exp(-0.5 * ((local_hour - 18.5) / 2.0) ** 2)
    solar_dip = -0.04 * math.exp(-0.5 * ((local_hour - 12.5) / 2.5) ** 2)
    corr = (ci - 480) * 0.0001
    return float(np.clip(base + evening + solar_dip + corr + rng.normal(0, 0.006), 0.06, 0.20))


_PRICE_FUNCS = {
    "DC1": _price_california,
    "DC2": _price_germany,
    "DC3": _price_chile,
    "DC4": _price_singapore,
    "DC5": _price_australia,
}

# ── Temperature profiles (°C) ───────────────────────────────────────────────

def _temp(dc_id: str, local_hour: float, rng: np.random.RandomState) -> float:
    profiles = {
        "DC1": (20.0, 6.0),   # 12-28
        "DC2": (12.5, 5.5),   # 5-20
        "DC3": (16.0, 4.5),   # 10-22
        "DC4": (30.0, 2.5),   # 26-34
        "DC5": (25.0, 7.0),   # 15-35
    }
    mean, amp = profiles[dc_id]
    # Peak at 15:00 local
    diurnal = amp * math.sin(math.pi * (local_hour - 6.0) / 18.0) if 6 <= local_hour <= 24 else -amp * 0.3
    return float(mean + diurnal + rng.normal(0, 1.0))


# ── Controller task distribution logic ───────────────────────────────────────

def _tasks_local_only(rng: np.random.RandomState) -> dict[str, int]:
    """Even distribution: 20-40 per DC."""
    return {dc: int(rng.randint(20, 40)) for dc in DC_IDS}


def _tasks_lowest_carbon(ci_values: dict[str, float], rng: np.random.RandomState) -> dict[str, int]:
    """Heavy bias to cleanest DCs. Top 2 get bulk, others get scraps."""
    ranked = sorted(ci_values, key=ci_values.get)  # cleanest first
    tasks = {}
    tasks[ranked[0]] = int(rng.randint(55, 80))   # cleanest DC: heavy load
    tasks[ranked[1]] = int(rng.randint(35, 55))   # 2nd cleanest: significant load
    tasks[ranked[2]] = int(rng.randint(10, 20))   # 3rd: some overflow
    tasks[ranked[3]] = int(rng.randint(3, 8))     # 4th: minimal
    tasks[ranked[4]] = int(rng.randint(2, 6))     # dirtiest: almost nothing
    return tasks


def _tasks_rl_agent(ci_values: dict[str, float], local_hours: dict[str, float],
                    rng: np.random.RandomState) -> tuple[dict[str, int], int]:
    """Smart distribution: aggressively routes to the cleanest DCs, shifts over time."""
    # Sharper inverse-CI weighting — cube the inverse so small CI differences
    # produce large routing swings (makes the demo visually dynamic)
    inv_ci = {dc: (1.0 / (ci + 20)) ** 3 for dc, ci in ci_values.items()}
    total_inv = sum(inv_ci.values())
    weights = {dc: v / total_inv for dc, v in inv_ci.items()}

    total_tasks = int(rng.randint(110, 145))
    tasks = {}
    for dc in DC_IDS:
        raw = int(total_tasks * weights[dc])
        # Wide range: cleanest DC can get 60+, dirtiest can get as low as 5
        tasks[dc] = max(5, min(70, raw + int(rng.randint(-3, 3))))

    # Deferral: more during high-carbon hours globally
    avg_ci = np.mean(list(ci_values.values()))
    defer_prob = np.clip((avg_ci - 250) / 350, 0.0, 0.65)
    deferred = int(rng.binomial(15, defer_prob))

    return tasks, deferred


# ── Energy / carbon / cost / water calculations ─────────────────────────────

def _compute_dc_metrics(
    tasks: int, ci: float, price: float, temp: float,
    rng: np.random.RandomState,
    controller: str = "",
) -> dict:
    """Compute per-DC per-timestep metrics from task count and environmental data."""
    # Base energy per task (kWh per 15-min step)
    energy_per_task = 0.08 + rng.normal(0, 0.005)
    # Cooling overhead scales with temperature
    cooling_factor = 1.0 + max(0, (temp - 20.0)) * 0.012

    # Overload penalty: when a DC runs >60 tasks, efficiency drops modestly
    # (contention, context-switching, thermal throttling)
    overload_factor = 1.0
    if tasks > 60:
        overload_factor = 1.0 + (tasks - 60) * 0.005  # mild penalty
    energy = tasks * energy_per_task * cooling_factor * overload_factor

    carbon = energy * ci / 1000.0  # kg
    cost = energy * price

    # Cross-region routing adds transmission costs
    if controller == "lowest_carbon":
        cost += tasks * 0.005  # modest per-task transmission surcharge
    elif controller == "manual_rl":
        cost += tasks * 0.002  # RL routes smarter, lower transmission overhead

    # Water usage: proportional to cooling needs (hot regions use more)
    water_per_kwh = max(0, (temp - 15.0) * 0.0008) + 0.0005
    water = energy * water_per_kwh

    # Utilisation: proportional to task count relative to capacity (~150 tasks = 100%)
    cpu_util = np.clip(tasks / 150.0 * 100.0 + rng.normal(0, 3), 5, 98)
    gpu_util = np.clip(tasks / 170.0 * 100.0 + rng.normal(0, 4), 3, 95)
    mem_util = np.clip(tasks / 160.0 * 100.0 + rng.normal(0, 2), 5, 90)

    return {
        "energy_kwh": max(0, energy),
        "carbon_kg": max(0, carbon),
        "energy_cost_usd": max(0, cost),
        "water_usage_m3": max(0, water),
        "cpu_util_pct": cpu_util,
        "gpu_util_pct": gpu_util,
        "mem_util_pct": mem_util,
    }


def _sla_violations(controller: str, tasks: int, capacity_ratio: float,
                    rng: np.random.RandomState) -> tuple[int, int]:
    """Return (sla_met, sla_violated) for a DC at one timestep."""
    if tasks == 0:
        return 0, 0

    if controller == "local_only":
        viol_rate = 0.06 + rng.normal(0, 0.015)
    elif controller == "lowest_carbon":
        # Overloaded DC gets high violations
        viol_rate = 0.04 + capacity_ratio * 0.18 + rng.normal(0, 0.02)
    else:  # RL agent
        viol_rate = 0.025 + rng.normal(0, 0.008)

    viol_rate = float(np.clip(viol_rate, 0.0, 0.35))
    violated = int(round(tasks * viol_rate))
    met = max(0, tasks - violated)
    return met, violated


def _generate_mock_action_probs(
    avg_ci: float, rng: np.random.RandomState, n_actions: int = 6
) -> list[float]:
    """
    Generate mock action probability distribution for the SAC agent.

    Simulates a reasonable policy where:
    - Action 0 (DEFER) is favored when CI is high (carbon-aware)
    - Actions 1-n (DISPATCH to various DCs) are favored when CI is low
    - Some entropy/uncertainty is always present (stochastic policy)

    Parameters
    ----------
    avg_ci : float
        Average carbon intensity across DCs (used to bias action selection)
    rng : np.random.RandomState
        Random state for reproducibility
    n_actions : int
        Number of actions (default 6: 1 defer + 5 dispatch DCs)

    Returns
    -------
    list[float]
        Probability distribution over actions (sums to ~1.0)
    """
    entropy_factor = np.clip((avg_ci - 200) / 400.0, 0.0, 1.0)
    base_entropy = 0.3 + entropy_factor * 0.4

    defer_bias = np.clip((avg_ci - 250) / 150.0, -1.0, 1.0)

    logits = np.zeros(n_actions)
    logits[0] = defer_bias * 1.5

    dispatch_base = -defer_bias * 0.8
    for i in range(1, n_actions):
        logits[i] = dispatch_base + rng.normal(0, 0.3)

    logits += rng.normal(0, base_entropy)

    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    return probs.tolist()


# ── Main generator ───────────────────────────────────────────────────────────

def generate_mock_comparison(
    strategies: list[str],
    eval_days: int = 3,
    checkpoint_name: str = "multi_action_enable_defer_2",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drop-in replacement for SustainClusterSimulator.run_comparison().

    Returns (per_dc_df, global_df, summary_df) with realistic mock data
    that tells a compelling story where the RL agent outperforms baselines.
    """
    rng = np.random.RandomState(seed)
    num_steps = eval_days * 24 * 4  # 15-min intervals
    start_dt = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)

    per_dc_records = []
    global_records = []
    summary_records = []

    # Pre-compute wind factor variation for Germany (multi-day pattern)
    wind_factors = []
    for d in range(eval_days):
        # Some days windier than others — wide range for visible variation
        wind_factors.append(0.55 + 0.7 * rng.random())

    for strategy in strategies:
        strategy_rng = np.random.RandomState(seed + hash(strategy) % 10000)
        total_deferred = 0

        strat_per_dc = []

        for t in range(num_steps):
            current_dt = start_dt + timedelta(minutes=15 * t)
            day_idx = t // (24 * 4)
            utc_hour = current_dt.hour + current_dt.minute / 60.0

            # Compute environmental signals for each DC
            ci_values = {}
            price_values = {}
            temp_values = {}
            local_hours = {}

            for dc_id in DC_IDS:
                offset = DC_INFO[dc_id]["utc_offset"]
                local_hour = (utc_hour + offset) % 24.0
                local_hours[dc_id] = local_hour

                # Carbon intensity
                if dc_id == "DC2":
                    wf = wind_factors[min(day_idx, len(wind_factors) - 1)]
                    ci = _ci_germany(local_hour, strategy_rng, wind_factor=wf)
                else:
                    ci = _CI_FUNCS[dc_id](local_hour, strategy_rng)
                ci_values[dc_id] = ci

                # Price (correlated with CI)
                price_values[dc_id] = _PRICE_FUNCS[dc_id](local_hour, ci, strategy_rng)

                # Temperature
                temp_values[dc_id] = _temp(dc_id, local_hour, strategy_rng)

            # Task distribution depends on controller
            deferred_this_step = 0
            if strategy == "local_only":
                tasks = _tasks_local_only(strategy_rng)
            elif strategy == "lowest_carbon":
                tasks = _tasks_lowest_carbon(ci_values, strategy_rng)
            else:  # manual_rl
                tasks, deferred_this_step = _tasks_rl_agent(
                    ci_values, local_hours, strategy_rng
                )
                total_deferred += deferred_this_step

            # Per-DC records
            for dc_id in DC_IDS:
                metrics = _compute_dc_metrics(
                    tasks[dc_id], ci_values[dc_id], price_values[dc_id],
                    temp_values[dc_id], strategy_rng, controller=strategy,
                )
                cap_ratio = tasks[dc_id] / 150.0
                sla_met, sla_viol = _sla_violations(strategy, tasks[dc_id], cap_ratio, strategy_rng)

                per_dc_records.append({
                    "timestep": t,
                    "datacenter": dc_id,
                    "controller": strategy,
                    "energy_cost_usd": metrics["energy_cost_usd"],
                    "energy_kwh": metrics["energy_kwh"],
                    "carbon_kg": metrics["carbon_kg"],
                    "water_usage_m3": metrics["water_usage_m3"],
                    "cpu_util_pct": metrics["cpu_util_pct"],
                    "gpu_util_pct": metrics["gpu_util_pct"],
                    "mem_util_pct": metrics["mem_util_pct"],
                    "running_tasks_count": tasks[dc_id],
                    "pending_tasks_count": max(0, int(strategy_rng.randint(0, 8))),
                    "tasks_assigned_count": tasks[dc_id],
                    "sla_met_count": sla_met,
                    "sla_violated_count": sla_viol,
                    "carbon_intensity_gco2_kwh": ci_values[dc_id],
                    "price_per_kwh": price_values[dc_id],
                    "temperature_c": temp_values[dc_id],
                    "deferred_tasks_this_step": deferred_this_step if dc_id == "DC1" else 0,
                })

            # Global record
            transmission_cost = 0.0
            action_probs = []
            if strategy == "manual_rl":
                transmission_cost = float(strategy_rng.uniform(0.5, 2.5))
                # Generate mock action probabilities for RL strategy
                avg_ci = np.mean(list(ci_values.values()))
                action_probs = _generate_mock_action_probs(avg_ci, strategy_rng, n_actions=6)
            elif strategy == "lowest_carbon":
                transmission_cost = float(strategy_rng.uniform(1.0, 4.0))

            global_records.append({
                "timestep": t,
                "controller": strategy,
                "deferred_tasks_this_step": deferred_this_step,
                "transmission_cost_usd": transmission_cost,
                "transmission_energy_kwh": transmission_cost * 0.3,
                "transmission_emissions_kg": transmission_cost * 0.05,
                "reward_this_step": float(strategy_rng.uniform(-1, 1)),
                "action_probs": action_probs,
            })

        # Summary
        dc_rows = [r for r in per_dc_records if r["controller"] == strategy]
        if dc_rows:
            total_sla_met = sum(r["sla_met_count"] for r in dc_rows)
            total_sla_viol = sum(r["sla_violated_count"] for r in dc_rows)
            denom = total_sla_met + total_sla_viol
            sla_viol_rate = (total_sla_viol / denom * 100.0) if denom > 0 else 0.0

            summary_records.append({
                "Controller": strategy,
                "Total CO2 (kg)": sum(r["carbon_kg"] for r in dc_rows),
                "Total Energy (kWh)": sum(r["energy_kwh"] for r in dc_rows),
                "Total Cost ($)": sum(r["energy_cost_usd"] for r in dc_rows),
                "Total Water (m3)": sum(r["water_usage_m3"] for r in dc_rows),
                "SLA Violation Rate (%)": round(sla_viol_rate, 2),
                "Avg CPU Util (%)": float(np.mean([r["cpu_util_pct"] for r in dc_rows])),
                "Avg GPU Util (%)": float(np.mean([r["gpu_util_pct"] for r in dc_rows])),
                "Total Tasks Deferred": total_deferred,
            })

    per_dc_df = pd.DataFrame(per_dc_records)
    global_df = pd.DataFrame(global_records)
    summary_df = pd.DataFrame(summary_records)

    return per_dc_df, global_df, summary_df


# ── Mock live carbon data ───────────────────────────────────────────────────

def get_mock_live_carbon() -> dict[str, float]:
    """Return realistic current carbon intensity for each DC zone.

    Uses current hour to create believable diurnal variation.
    Returns dict like {"DC1 (US-California \U0001f1fa\U0001f1f8)": 234.5, ...}
    """
    now = datetime.now(timezone.utc)
    utc_hour = now.hour + now.minute / 60.0
    # Use minute as micro-seed for slight variation on refresh
    rng = np.random.RandomState(int(now.hour * 60 + now.minute) % 1000)

    result = {}
    for dc_id, info in DC_INFO.items():
        local_hour = (utc_hour + info["utc_offset"]) % 24.0
        if dc_id == "DC2":
            ci = _ci_germany(local_hour, rng, wind_factor=1.0)
        else:
            ci = _CI_FUNCS[dc_id](local_hour, rng)
        label = f"{dc_id} ({info['name']} {info['flag']})"
        result[label] = round(ci, 1)

    return result
