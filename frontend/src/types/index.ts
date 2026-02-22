// ── DataFrame row types (mirrors FastAPI Pydantic models exactly) ──────────

export interface PerDcRow {
  timestep: number;
  datacenter: DcId;
  controller: StrategyId;
  energy_cost_usd: number;
  energy_kwh: number;
  carbon_kg: number;
  water_usage_m3: number;
  cpu_util_pct: number;
  gpu_util_pct: number;
  mem_util_pct: number;
  running_tasks_count: number;
  pending_tasks_count: number;
  tasks_assigned_count: number;
  sla_met_count: number;
  sla_violated_count: number;
  carbon_intensity_gco2_kwh: number;
  price_per_kwh: number;
  temperature_c: number;
  deferred_tasks_this_step: number;
}

export interface GlobalRow {
  timestep: number;
  controller: StrategyId;
  deferred_tasks_this_step: number;
  transmission_cost_usd: number;
  transmission_energy_kwh: number;
  transmission_emissions_kg: number;
}

export interface SummaryRow {
  controller: StrategyId;
  controller_label: string;
  total_co2_kg: number;
  total_energy_kwh: number;
  total_cost_usd: number;
  total_water_m3: number;
  sla_violation_rate_pct: number;
  avg_cpu_util_pct: number;
  avg_gpu_util_pct: number;
  total_tasks_deferred: number;
}

export interface SimulationResponse {
  per_dc: PerDcRow[];
  global_metrics: GlobalRow[];
  summary: SummaryRow[];
}

// ── Carbon live types ─────────────────────────────────────────────────────

export interface CarbonIntensityEntry {
  dc_id: string;
  display_label: string;
  ci_gco2_kwh: number;
}

export interface LiveCarbonResponse {
  entries: CarbonIntensityEntry[];
  is_live: boolean;
}

// ── Domain union types ────────────────────────────────────────────────────

export type DcId = "DC1" | "DC2" | "DC3" | "DC4" | "DC5";
export type StrategyId = "manual_rl" | "local_only" | "lowest_carbon";

// ── DC metadata ───────────────────────────────────────────────────────────

export interface DcLocation {
  name: string;
  lat: number;
  lon: number;
  flag: string;
  utc_offset: number;
  zone: string;
}

// ── Simulation config (sidebar state) ────────────────────────────────────

export interface SimulationConfig {
  evalDays: number;
  strategies: StrategyId[];
  checkpointName: string;
  useLive: boolean;
}

// ── Computed savings (derived, not from API) ──────────────────────────────

export interface Savings {
  co2SavedKg: number;
  co2SavedPct: number;
  energySavedKwh: number;
  energySavedPct: number;
  waterSavedM3: number;
  waterSavedPct: number;
  costDeltaUsd: number;
  costDeltaPct: number;
  totalDeferred: number;
}

