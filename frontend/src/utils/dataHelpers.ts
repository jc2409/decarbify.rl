import { PerDcRow, Savings } from "@/types";

export function computeSavings(filtered: PerDcRow[]): Savings | null {
  const rl = filtered.filter((r) => r.controller === "manual_rl");
  const bl = filtered.filter((r) => r.controller === "local_only");
  if (rl.length === 0 || bl.length === 0) return null;

  const sum = (arr: PerDcRow[], field: keyof PerDcRow) =>
    arr.reduce((s, r) => s + (r[field] as number), 0);

  const rlCo2 = sum(rl, "carbon_kg");
  const blCo2 = sum(bl, "carbon_kg");
  const rlEnergy = sum(rl, "energy_kwh");
  const blEnergy = sum(bl, "energy_kwh");
  const rlWater = sum(rl, "water_usage_m3");
  const blWater = sum(bl, "water_usage_m3");
  const rlCost = sum(rl, "energy_cost_usd");
  const blCost = sum(bl, "energy_cost_usd");
  const totalDeferred = rl.reduce((s, r) => s + r.deferred_tasks_this_step, 0);

  const pct = (saved: number, base: number) => (base === 0 ? 0 : (saved / base) * 100);

  return {
    co2SavedKg: blCo2 - rlCo2,
    co2SavedPct: pct(blCo2 - rlCo2, blCo2),
    energySavedKwh: blEnergy - rlEnergy,
    energySavedPct: pct(blEnergy - rlEnergy, blEnergy),
    waterSavedM3: blWater - rlWater,
    waterSavedPct: pct(blWater - rlWater, blWater),
    costDeltaUsd: rlCost - blCost,
    costDeltaPct: pct(rlCost - blCost, blCost),
    totalDeferred,
  };
}

export function fmtNum(v: number, decimals = 1): string {
  return v.toLocaleString("en-US", { maximumFractionDigits: decimals });
}

export function fmtPct(v: number, decimals = 1): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(decimals)}%`;
}

export function ciColor(ci: number): string {
  if (ci < 150) return "#00C853";
  if (ci < 350) return "#FFD166";
  return "#FF6B6B";
}
