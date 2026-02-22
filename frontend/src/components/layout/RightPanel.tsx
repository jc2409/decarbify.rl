import type { PerDcRow, DcId, Savings, SummaryRow } from "@/types";
import { DC_LOCATIONS, DC_IDS, STRATEGY_COLOR_BY_ID } from "@/constants";
import { ciColor, fmtNum } from "@/utils/dataHelpers";

interface RightPanelProps {
  stepData: PerDcRow[];        // current step, all strategies combined
  allStepData: PerDcRow[];     // current step, map strategy only
  activeDcId: DcId | null;
  savings: Savings | null;
  currentHour: number;
  maxHours: number;
  playbackStep: number;
  maxStep: number;
  summary: SummaryRow[];
}

function Divider() {
  return <hr className="cyber-divider" />;
}

function StatLine({
  label,
  value,
  color,
  mono = true,
}: {
  label: string;
  value: string;
  color?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-1 py-0.5">
      <span className="micro-label truncate max-w-[50%]">{label}</span>
      <span
        className={mono ? "live-num text-xs font-semibold" : "text-xs font-semibold"}
        style={{ color: color ?? "#E6EDF3" }}
      >
        {value}
      </span>
    </div>
  );
}

function SectionHead({ title }: { title: string }) {
  return (
    <div className="tracking-widest text-white/70 pt-1 pb-0.5 text-xs font-semibold" style={{ color: "#E6EDF3" }}>
      {title}
    </div>
  );
}

export default function RightPanel({
  stepData,
  allStepData,
  activeDcId,
  savings,
  currentHour,
  maxHours,
  playbackStep,
  maxStep,
  summary,
}: RightPanelProps) {
  // Per-DC data at current step (map strategy)
  const dcRows = DC_IDS.map((dc_id) => {
    const row = allStepData.find((r) => r.datacenter === dc_id);
    const loc = DC_LOCATIONS[dc_id];
    return { dc_id, loc, row };
  });

  return (
    <aside
      className="w-56 shrink-0 flex flex-col overflow-y-auto"
      style={{
        borderLeft: "1px solid #30363D",
        background: "#161B22",
      }}
    >
      <div className="flex flex-col gap-0 p-3 h-full">
        {/* ── Panel header ──────────────────────────────────────── */}
        <div className="mb-3 pt-1">
          <div
            className="font-semibold tracking-wide text-white/80"
            style={{ fontSize: 15 }}
          >
            ANALYTICS
          </div>
          <div className="micro-label text-white/25 mt-0.5">live simulation metrics</div>
        </div>

        <Divider />

        {/* ── All DCs snapshot ──────────────────────────────────── */}
        <SectionHead title="ALL DATACENTERS" />
        <div className="flex flex-col gap-0.5 mb-1">
          {dcRows.map(({ dc_id, loc, row }) => {
            const ci = row?.carbon_intensity_gco2_kwh ?? null;
            const tasks = row?.running_tasks_count ?? 0;
            const isActive = dc_id === activeDcId;
            const color = ci !== null ? ciColor(ci) : "#3a3f4b";
            return (
              <div
                key={dc_id}
                className="flex items-center justify-between py-0.5 px-1.5 rounded"
                style={{
                  background: isActive ? "rgba(0,255,160,0.06)" : "transparent",
                  border: isActive ? "1px solid rgba(0,255,160,0.15)" : "1px solid transparent",
                }}
              >
                <span className="micro-label flex items-center gap-1">
                  {loc.flag} {dc_id}
                  {isActive && <span className="glow-green text-xs">●</span>}
                </span>
                <div className="flex items-center gap-2">
                  <span className="live-num text-xs font-mono" style={{ color }}>
                    {ci !== null ? ci.toFixed(0) : "—"}
                  </span>
                  <span className="micro-label text-white/30">{tasks}t</span>
                </div>
              </div>
            );
          })}
        </div>

        <Divider />

        {/* ── RL Savings ────────────────────────────────────────── */}
        {savings ? (
          <>
            <SectionHead title="RL SAVINGS vs BASELINE" />
            <StatLine
              label="CO₂ Saved"
              value={`${fmtNum(savings.co2SavedKg)} kg`}
              color={savings.co2SavedKg > 0 ? "#00FFA0" : "#FF4757"}
            />
            <StatLine
              label="Savings %"
              value={`${savings.co2SavedPct.toFixed(1)}%`}
              color={savings.co2SavedPct > 0 ? "#00FFA0" : "#FF4757"}
            />
            <StatLine
              label="Energy Saved"
              value={`${fmtNum(savings.energySavedKwh)} kWh`}
              color={savings.energySavedKwh > 0 ? "#00FFA0" : "#FF4757"}
            />
            <StatLine
              label="Water Saved"
              value={`${fmtNum(savings.waterSavedM3)} m³`}
              color={savings.waterSavedM3 > 0 ? "#00FFA0" : "#FF4757"}
            />
            <StatLine
              label="Cost Delta"
              value={`${savings.costDeltaUsd >= 0 ? "+" : ""}$${fmtNum(Math.abs(savings.costDeltaUsd), 2)}`}
              color={savings.costDeltaUsd <= 0 ? "#00FFA0" : "#FF8C42"}
            />
            <StatLine
              label="Tasks Deferred"
              value={fmtNum(savings.totalDeferred, 0)}
              color="#00FF9F"
            />

            <Divider />
          </>
        ) : null}

        {/* ── Strategy totals ───────────────────────────────────── */}
        {summary.length > 0 && (
          <>
            <div className="mt-4" />
            <SectionHead title="TOTAL EMISSIONS" />
            {summary.map((row) => (
              <StatLine
                key={row.controller}
                label={row.controller_label.replace(" (Geo+Time)", "").replace(" (Baseline)", "")}
                value={`${fmtNum(row.total_co2_kg)} kg`}
                color={STRATEGY_COLOR_BY_ID[row.controller] ?? "#888"}
              />
            ))}

            <Divider />

            <SectionHead title="SLA COMPLIANCE" />
            {summary.map((row) => {
              const compliance = 100 - row.sla_violation_rate_pct;
              const slaColor = compliance >= 95 ? "#00FF9F" : "#F43F5E";
              return (
                <StatLine
                  key={row.controller}
                  label={row.controller_label.replace(" (Geo+Time)", "").replace(" (Baseline)", "")}
                  value={`${compliance.toFixed(1)}%`}
                  color={slaColor}
                />
              );
            })}
          </>
        )}

        {/* Spacer */}
        <div className="flex-1" />
      </div>
    </aside>
  );
}
