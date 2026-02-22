"""FastAPI backend for GreenDispatch — replaces Streamlit app.py as the data layer."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup so backend modules can import sustain-cluster internals
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "sustain-cluster"))

from backend.mock_data import generate_mock_comparison, DC_INFO, CONTROLLERS  # noqa: E402
from backend.carbon_api import get_live_carbon_intensity  # noqa: E402

# ---------------------------------------------------------------------------
# Thread pool — max_workers=1 for live sim (os.chdir not thread-safe),
# higher for mock (stateless).
# ---------------------------------------------------------------------------
_MOCK_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_LIVE_EXECUTOR = ThreadPoolExecutor(max_workers=1)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="GreenDispatch API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRATEGY_INTERNAL = {
    "SAC RL Agent": "manual_rl",
    "Local Only": "local_only",
    "Lowest Carbon": "lowest_carbon",
}

STRATEGY_DISPLAY = {v: k for k, v in STRATEGY_INTERNAL.items()}
STRATEGY_DISPLAY["manual_rl"] = "SAC RL Agent (Geo+Time)"
STRATEGY_DISPLAY["local_only"] = "Local Only (Baseline)"
STRATEGY_DISPLAY["lowest_carbon"] = "Lowest Carbon"

STRATEGY_COLORS = {
    "SAC RL Agent (Geo+Time)": "#00C853",
    "Local Only (Baseline)": "#FF6B6B",
    "Lowest Carbon": "#4FC3F7",
}

CHECKPOINT_OPTIONS = {
    "Multi-Action + Defer (recommended)": "multi_action_enable_defer_2",
    "Multi-Action + No Defer": "multi_action_disable_defer_2",
    "Single-Action + Defer": "single_action_enable_defer_2",
    "Single-Action + No Defer": "single_action_disable_defer_2",
}

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RunSimulationRequest(BaseModel):
    strategies: List[str]
    eval_days: int = 1
    checkpoint_name: str = "multi_action_enable_defer_2"
    seed: int = 42
    use_live: bool = False


class PerDcRow(BaseModel):
    timestep: int
    datacenter: str
    controller: str
    energy_cost_usd: float
    energy_kwh: float
    carbon_kg: float
    water_usage_m3: float
    cpu_util_pct: float
    gpu_util_pct: float
    mem_util_pct: float
    running_tasks_count: int
    pending_tasks_count: int
    tasks_assigned_count: int
    sla_met_count: int
    sla_violated_count: int
    carbon_intensity_gco2_kwh: float
    price_per_kwh: float
    temperature_c: float
    deferred_tasks_this_step: int = 0


class GlobalRow(BaseModel):
    timestep: int
    controller: str
    deferred_tasks_this_step: int = 0
    transmission_cost_usd: float
    transmission_energy_kwh: float
    transmission_emissions_kg: float


class SummaryRow(BaseModel):
    controller: str
    controller_label: str
    total_co2_kg: float
    total_energy_kwh: float
    total_cost_usd: float
    total_water_m3: float
    sla_violation_rate_pct: float
    avg_cpu_util_pct: float
    avg_gpu_util_pct: float
    total_tasks_deferred: int


class SimulationResponse(BaseModel):
    per_dc: List[PerDcRow]
    global_metrics: List[GlobalRow]
    summary: List[SummaryRow]


class CarbonIntensityEntry(BaseModel):
    dc_id: str
    display_label: str
    ci_gco2_kwh: float


class LiveCarbonResponse(BaseModel):
    entries: List[CarbonIntensityEntry]
    is_live: bool


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUMMARY_RENAME = {
    "Controller": "controller",
    "Total CO2 (kg)": "total_co2_kg",
    "Total Energy (kWh)": "total_energy_kwh",
    "Total Cost ($)": "total_cost_usd",
    "Total Water (m3)": "total_water_m3",
    "SLA Violation Rate (%)": "sla_violation_rate_pct",
    "Avg CPU Util (%)": "avg_cpu_util_pct",
    "Avg GPU Util (%)": "avg_gpu_util_pct",
    "Total Tasks Deferred": "total_tasks_deferred",
}


def _run_mock(req: RunSimulationRequest):
    """Blocking call — runs in thread pool."""
    per_dc_df, global_df, summary_df = generate_mock_comparison(
        strategies=req.strategies,
        eval_days=req.eval_days,
        checkpoint_name=req.checkpoint_name,
        seed=req.seed,
    )
    return per_dc_df, global_df, summary_df


def _run_live(req: RunSimulationRequest):
    """Blocking call for real RL simulation — runs in single-worker pool."""
    try:
        from backend.simulator import SustainClusterSimulator
        sim = SustainClusterSimulator(_ROOT)
        per_dc_df, global_df, summary_df = sim.run_comparison(
            strategies=req.strategies,
            eval_days=req.eval_days,
            checkpoint_name=req.checkpoint_name,
            seed=req.seed,
        )
        return per_dc_df, global_df, summary_df
    except Exception as exc:
        raise RuntimeError(f"Live simulation failed: {exc}") from exc


def _serialize_dfs(per_dc_df, global_df, summary_df) -> SimulationResponse:
    # Normalize per_dc
    per_dc_records = per_dc_df.to_dict(orient="records")
    for row in per_dc_records:
        row.setdefault("deferred_tasks_this_step", 0)
        # Cast integer-ish fields
        for field in (
            "timestep", "running_tasks_count", "pending_tasks_count",
            "tasks_assigned_count", "sla_met_count", "sla_violated_count",
            "deferred_tasks_this_step",
        ):
            if field in row:
                row[field] = int(row[field])

    # Normalize global
    global_records = global_df.to_dict(orient="records")
    for row in global_records:
        row.setdefault("deferred_tasks_this_step", 0)
        for field in ("timestep", "deferred_tasks_this_step"):
            if field in row:
                row[field] = int(row[field])

    # Normalize summary
    summary_df = summary_df.rename(columns=_SUMMARY_RENAME)
    summary_df["controller_label"] = summary_df["controller"].map(
        lambda c: STRATEGY_DISPLAY.get(c, c)
    )
    summary_df["total_tasks_deferred"] = summary_df["total_tasks_deferred"].fillna(0).astype(int)
    summary_records = summary_df.to_dict(orient="records")

    return SimulationResponse(
        per_dc=[PerDcRow(**r) for r in per_dc_records],
        global_metrics=[GlobalRow(**r) for r in global_records],
        summary=[SummaryRow(**r) for r in summary_records],
    )


def _parse_carbon_response(raw: dict, is_live: bool) -> LiveCarbonResponse:
    entries = []
    for label, ci_value in raw.items():
        match = re.match(r"(DC\d)", label)
        dc_id = match.group(1) if match else label
        entries.append(CarbonIntensityEntry(
            dc_id=dc_id,
            display_label=label,
            ci_gco2_kwh=float(ci_value),
        ))
    return LiveCarbonResponse(entries=entries, is_live=is_live)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="2.0.0")


@app.post("/api/simulation/run", response_model=SimulationResponse)
async def run_simulation(req: RunSimulationRequest):
    if not req.strategies:
        raise HTTPException(status_code=422, detail="At least one strategy required")
    if req.eval_days < 1 or req.eval_days > 14:
        raise HTTPException(status_code=422, detail="eval_days must be 1–14")

    loop = asyncio.get_event_loop()
    try:
        if req.use_live:
            per_dc_df, global_df, summary_df = await loop.run_in_executor(
                _LIVE_EXECUTOR, _run_live, req
            )
        else:
            per_dc_df, global_df, summary_df = await loop.run_in_executor(
                _MOCK_EXECUTOR, _run_mock, req
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _serialize_dfs(per_dc_df, global_df, summary_df)


@app.get("/api/carbon/live", response_model=LiveCarbonResponse)
async def carbon_live():
    loop = asyncio.get_event_loop()
    raw, is_live = await loop.run_in_executor(
        _MOCK_EXECUTOR, get_live_carbon_intensity
    )
    return _parse_carbon_response(raw, is_live)


@app.get("/api/constants/strategies")
async def get_strategies():
    return {
        "strategies": [
            {"display": d, "internal": i, "color": STRATEGY_COLORS.get(STRATEGY_DISPLAY.get(i, ""), "#888888")}
            for d, i in STRATEGY_INTERNAL.items()
        ]
    }


@app.get("/api/constants/datacenters")
async def get_datacenters():
    return {"datacenters": DC_INFO}


@app.get("/api/constants/checkpoints")
async def get_checkpoints():
    return {
        "checkpoints": [
            {"display": d, "internal": i}
            for d, i in CHECKPOINT_OPTIONS.items()
        ]
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
