"""
LLM Scheduler Benchmark
=======================
Loads 1 day (96 x 15-min intervals) of Alibaba 2020 workload data and runs
three scheduling strategies:
  - local_only    : all tasks assigned to DC1 (single-DC baseline)
  - lowest_carbon : always pick the DC with the lowest current CI
  - llm           : GPT-4o-mini decides per-task routing via JSON

Computes carbon, cost, and SLA metrics for each strategy, then prints a
comparison table and saves results to llm_results.json.

Requirements:
    pip install openai python-dotenv
    Put your key in a .env file next to this script:
        OPENAI_API_KEY=sk-...

Run:
    cd c:\\Users\\vleou\\Downloads\\hackeurope
    python llm_scheduler.py
"""

import json
import math
import os
import time

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Config ──────────────────────────────────────────────────────────────────

PICKLE_PATH   = "result_df_full_year_2020.pkl"
NUM_INTERVALS = 96          # 1 day; increase to 288 for 3 days
LLM_MODELS    = ["gpt-4o-mini", "gpt-4o"]
TASK_SCALE    = 5           # matches SustainCluster's task_scale=5

# DC definitions: CI in gCO2/kWh, price in $/kWh, timezone offset in hours
DC_CONFIG = {
    "DC1": {"location": "California", "base_ci": 230.0, "price": 0.120, "tz": -8},
    "DC2": {"location": "Germany",    "base_ci": 340.0, "price": 0.180, "tz": +1},
    "DC3": {"location": "Chile",      "base_ci": 195.0, "price": 0.080, "tz": -4},
    "DC4": {"location": "Singapore",  "base_ci": 455.0, "price": 0.150, "tz": +8},
    "DC5": {"location": "Australia",  "base_ci": 630.0, "price": 0.140, "tz": +11},
}

# Power constants (Watts per unit)
CPU_W_PER_CORE = 6.0
GPU_W_PER_GPU  = 500.0
MEM_W_PER_GB   = 2.5


# ── DC state helpers ─────────────────────────────────────────────────────────

def get_dc_ci(dc_id: str, utc_hour: float) -> float:
    """Return carbon intensity with a sinusoidal time-of-day variation (±15%)."""
    base = DC_CONFIG[dc_id]["base_ci"]
    tz   = DC_CONFIG[dc_id]["tz"]
    local_hour = (utc_hour + tz) % 24
    # Solar dip during mid-day; highest CI in evening
    variation = 1.0 - 0.15 * math.sin(math.pi * (local_hour - 6) / 12)
    return round(base * variation, 1)


def get_dc_state(utc_hour: float) -> dict:
    """Return {dc_id: {ci, price}} for all DCs at a given UTC hour."""
    return {
        dc_id: {
            "ci":    get_dc_ci(dc_id, utc_hour),
            "price": DC_CONFIG[dc_id]["price"],
            "location": DC_CONFIG[dc_id]["location"],
        }
        for dc_id in DC_CONFIG
    }


# ── Workload parsing ─────────────────────────────────────────────────────────

def parse_tasks(tasks_matrix) -> list[dict]:
    """
    Convert a tasks_matrix row into a list of task dicts.
    task_data layout (from SustainCluster workload_utils.py):
      [0] job_id  [4] duration_min  [5] cpu_pct  [6] gpu_pct  [7] mem_gb  [8] bandwidth_gb
    TASK_SCALE is applied to CPU and GPU (matching the simulator's task_scale=5).
    """
    tasks = []
    for task_data in tasks_matrix:
        try:
            duration = float(task_data[4])
            cores    = TASK_SCALE * float(task_data[5]) / 100.0
            gpus     = TASK_SCALE * float(task_data[6]) / 100.0
            mem_gb   = float(task_data[7])
            bw_gb    = float(task_data[8])
            if duration <= 0:
                continue
            tasks.append({
                "job_id":   str(task_data[0]),
                "duration": duration,
                "cores":    round(cores, 3),
                "gpus":     round(gpus, 3),
                "mem_gb":   round(mem_gb, 3),
                "bw_gb":    round(bw_gb, 3),
            })
        except (IndexError, ValueError, TypeError):
            continue
    return tasks


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_task_metrics(task: dict, dc_id: str, dc_state: dict, deferred: bool = False) -> dict:
    """Return energy_kwh, carbon_kg, cost_usd for a task at a given DC."""
    if deferred:
        return {"energy_kwh": 0.0, "carbon_kg": 0.0, "cost_usd": 0.0, "sla_violated": True}

    dc = dc_state[dc_id]
    power_w    = task["cores"] * CPU_W_PER_CORE + task["gpus"] * GPU_W_PER_GPU + task["mem_gb"] * MEM_W_PER_GB
    energy_kwh = power_w * (task["duration"] / 60.0) / 1000.0
    # gCO2/kWh * kWh / 1000 = kg CO2
    carbon_kg  = energy_kwh * dc["ci"] / 1000.0
    cost_usd   = energy_kwh * dc["price"]
    return {
        "energy_kwh":   round(energy_kwh, 6),
        "carbon_kg":    round(carbon_kg, 6),
        "cost_usd":     round(cost_usd, 6),
        "sla_violated": False,
    }


# ── Strategies ───────────────────────────────────────────────────────────────

def strategy_local_only(tasks: list[dict], dc_state: dict) -> list[str]:
    """Assign every task to DC1."""
    return ["DC1"] * len(tasks)


def strategy_lowest_carbon(tasks: list[dict], dc_state: dict) -> list[str]:
    """Assign every task to whichever DC currently has the lowest CI."""
    best = min(dc_state, key=lambda dc: dc_state[dc]["ci"])
    return [best] * len(tasks)


def strategy_llm(tasks: list[dict], dc_state: dict, client: OpenAI, model: str = "gpt-4o-mini") -> list[str]:
    """
    Ask GPT-4o-mini to route each task to a DC.
    Returns a list of DC IDs (or 'DEFER') in task order.
    Falls back to lowest_carbon if parsing fails.
    """
    if not tasks:
        return []

    # Build DC summary
    dc_lines = "\n".join(
        f"  {dc_id} ({info['location']}): CI={info['ci']} gCO2/kWh, price=${info['price']:.3f}/kWh"
        for dc_id, info in dc_state.items()
    )

    # Build task summary (trim to avoid huge prompts)
    task_lines = "\n".join(
        f"  task_{i}: duration={t['duration']:.0f}min, CPU={t['cores']:.1f} cores, "
        f"GPU={t['gpus']:.1f}, MEM={t['mem_gb']:.1f}GB"
        for i, t in enumerate(tasks)
    )

    task_keys = ", ".join(f'"task_{i}"' for i in range(len(tasks)))

    user_msg = (
        f"DC STATE:\n{dc_lines}\n\n"
        f"INCOMING TASKS ({len(tasks)}):\n{task_lines}\n\n"
        f"Reply ONLY with a JSON object mapping each task key to a DC ID.\n"
        f"Keys: {{{task_keys}}}\n"
        f"Values: one of DC1, DC2, DC3, DC4, DC5 (or DEFER to skip — counts as SLA violation).\n"
        f"Minimise total carbon emissions. Avoid DEFER unless you have a strong reason."
    )

    system_msg = (
        "You are a carbon-aware datacenter scheduler. "
        "Given the current carbon intensity and price of 5 global datacenters, "
        "route each incoming compute task to the datacenter that minimises "
        "total carbon emissions while keeping costs reasonable. "
        "Respond ONLY with a JSON object — no explanation."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        routing = json.loads(resp.choices[0].message.content)
        valid   = set(DC_CONFIG.keys()) | {"DEFER"}
        result  = []
        for i in range(len(tasks)):
            assigned = routing.get(f"task_{i}", "DEFER").strip().upper()
            result.append(assigned if assigned in valid else "DEFER")
        return result
    except Exception as exc:
        print(f"    [LLM error, falling back to lowest_carbon] {exc}")
        return strategy_lowest_carbon(tasks, dc_state)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_strategy(df: pd.DataFrame, strategy_name: str, client=None, model: str = "gpt-4o-mini") -> dict:
    """
    Run a single strategy over all intervals.
    Returns aggregated metrics dict.
    """
    total_energy   = 0.0
    total_carbon   = 0.0
    total_cost     = 0.0
    total_tasks    = 0
    total_violated = 0

    for idx, row in df.iterrows():
        ts    = row["interval_15m"]
        tasks = parse_tasks(row["tasks_matrix"])
        if not tasks:
            continue

        utc_hour  = ts.hour + ts.minute / 60.0
        dc_state  = get_dc_state(utc_hour)

        if strategy_name == "local_only":
            assignments = strategy_local_only(tasks, dc_state)
        elif strategy_name == "lowest_carbon":
            assignments = strategy_lowest_carbon(tasks, dc_state)
        elif strategy_name == "llm":
            assignments = strategy_llm(tasks, dc_state, client, model=model)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        for task, dc_id in zip(tasks, assignments):
            deferred = (dc_id == "DEFER")
            m = compute_task_metrics(task, dc_id if not deferred else "DC1", dc_state, deferred)
            total_energy   += m["energy_kwh"]
            total_carbon   += m["carbon_kg"]
            total_cost     += m["cost_usd"]
            total_violated += int(m["sla_violated"])
            total_tasks    += 1

        if strategy_name == "llm" and (idx + 1) % 10 == 0:
            print(f"    interval {idx + 1}/{len(df)} done …")

    return {
        "strategy":       strategy_name,
        "model":          model if strategy_name == "llm" else None,
        "total_tasks":    total_tasks,
        "energy_kwh":     round(total_energy, 3),
        "carbon_kg":      round(total_carbon, 3),
        "cost_usd":       round(total_cost, 4),
        "sla_violations": total_violated,
        "sla_viol_pct":   round(total_violated / max(total_tasks, 1) * 100, 2),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_table(all_results: list[dict]) -> None:
    col_w   = [28, 16, 15, 17, 10]
    headers = ["Strategy", "CO2 (kg)", "Energy (kWh)", "Cost (USD)", "SLA Viol"]
    sep     = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    hdr     = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_w)) + " |"

    print(f"\n\n{'Results':^{sum(col_w) + 3 * len(col_w) + 1}}")
    print(sep)
    print(hdr)
    print(sep)
    for r in all_results:
        if r["strategy"] == "llm":
            label = f"LLM ({r['model']})"
        elif r["strategy"] == "local_only":
            label = "Local Only (baseline)"
        else:
            label = "Lowest Carbon (greedy)"
        row = [
            label.ljust(col_w[0]),
            f"{r['carbon_kg']:.4f}".ljust(col_w[1]),
            f"{r['energy_kwh']:.3f}".ljust(col_w[2]),
            f"${r['cost_usd']:.4f}".ljust(col_w[3]),
            f"{r['sla_violations']} ({r['sla_viol_pct']}%)".ljust(col_w[4]),
        ]
        print("| " + " | ".join(row) + " |")
    print(sep)


def print_deltas(all_results: list[dict]) -> None:
    local_r = next((r for r in all_results if r["strategy"] == "local_only"), None)
    green_r = next((r for r in all_results if r["strategy"] == "lowest_carbon"), None)
    llm_rows = [r for r in all_results if r["strategy"] == "llm"]

    for llm_r in llm_rows:
        label = llm_r["model"]
        if local_r:
            co2_saved  = local_r["carbon_kg"] - llm_r["carbon_kg"]
            co2_pct    = co2_saved / local_r["carbon_kg"] * 100
            cost_saved = local_r["cost_usd"] - llm_r["cost_usd"]
            cost_pct   = cost_saved / local_r["cost_usd"] * 100
            print(f"\n{label} vs Local Only:")
            print(f"  CO2  saved : {co2_saved:.4f} kg  ({co2_pct:.1f}%)")
            print(f"  Cost saved : ${cost_saved:.4f}  ({cost_pct:.1f}%)")
        if green_r:
            co2_gap  = llm_r["carbon_kg"] - green_r["carbon_kg"]
            co2_gpct = co2_gap / green_r["carbon_kg"] * 100
            sign = "+" if co2_gap >= 0 else ""
            print(f"{label} vs Lowest Carbon:")
            print(f"  CO2  delta : {sign}{co2_gap:.4f} kg  ({sign}{co2_gpct:.1f}%)")

    # Head-to-head between the two LLMs if both ran
    if len(llm_rows) == 2:
        a, b = llm_rows
        co2_diff  = a["carbon_kg"] - b["carbon_kg"]
        cost_diff = a["cost_usd"]  - b["cost_usd"]
        sign_co2  = "+" if co2_diff >= 0 else ""
        sign_cost = "+" if cost_diff >= 0 else ""
        print(f"\n{a['model']} vs {b['model']} (head-to-head):")
        print(f"  CO2  delta : {sign_co2}{co2_diff:.4f} kg  (positive = {b['model']} is greener)")
        print(f"  Cost delta : {sign_cost}${cost_diff:.4f}  (positive = {b['model']} is cheaper)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GreenDispatch — LLM Scheduler Benchmark")
    print("=" * 60)

    # Load data
    print(f"\nLoading {PICKLE_PATH} …")
    df_full = pd.read_pickle(PICKLE_PATH)
    df = df_full.head(NUM_INTERVALS).copy()
    print(f"Loaded {len(df)} intervals ({NUM_INTERVALS // 96} day(s)).")
    total_tasks_estimate = sum(len(r) for r in df["tasks_matrix"])
    print(f"Estimated tasks: {total_tasks_estimate:,}")

    # OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("\n[WARNING] OPENAI_API_KEY not set. LLM strategies will be skipped.")
    client = OpenAI(api_key=api_key) if api_key else None

    all_results = []

    # Baselines (run once, shared across both LLM comparisons)
    print("\n--- Running: local_only ---")
    all_results.append(run_strategy(df, "local_only"))
    print("  done.")

    print("\n--- Running: lowest_carbon ---")
    all_results.append(run_strategy(df, "lowest_carbon"))
    print("  done.")

    # LLM strategies
    if client:
        for model in LLM_MODELS:
            print(f"\n--- Running: llm ({model}) ---")
            t0 = time.time()
            r = run_strategy(df, "llm", client=client, model=model)
            elapsed = time.time() - t0
            print(f"  done in {elapsed:.1f}s.")
            all_results.append(r)

            # Save per-model JSON
            model_slug = model.replace("/", "-")
            out_path = f"llm_results_{model_slug}.json"
            with open(out_path, "w") as f:
                json.dump(r, f, indent=2)
            print(f"  saved → {out_path}")
    else:
        print("\n[SKIPPED] LLM strategies — no API key.")

    # Save combined results
    with open("llm_results_combined.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nCombined results saved → llm_results_combined.json")

    print_table(all_results)
    print_deltas(all_results)

    print(f"\nTotal tasks processed: {all_results[0]['total_tasks']:,}")
    print("Done.")
