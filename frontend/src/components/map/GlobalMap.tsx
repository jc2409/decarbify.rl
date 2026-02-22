/**
 * GlobalMap - world map rendered with react-map-gl + maplibre-gl only.
 * No deck.gl: eliminates the luma.gl v9 WebGLCanvasContext ResizeObserver
 * race condition that crashes before this.device is initialized.
 */

import { useState, useMemo, useCallback } from "react";
import { Map, Marker } from "react-map-gl/maplibre";
import type { DcId, PerDcRow } from "@/types";
import { DC_IDS, DC_LOCATIONS } from "@/constants";
import { ciColor } from "@/utils/dataHelpers";

// ── Constants ──────────────────────────────────────────────────────────────

const MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

const INITIAL_VIEW = {
  longitude: 10,
  latitude: 15,
  zoom: 0.9,
};

// ── Types ──────────────────────────────────────────────────────────────────

interface DcPoint {
  dc_id: DcId;
  name: string;
  flag: string;
  lon: number;
  lat: number;
  tasks: number;
  ci: number;
  cpu: number;
  isActive: boolean;
}

interface TooltipInfo {
  dc: DcPoint;
}

interface Props {
  stepData: PerDcRow[];
  activeDcId: DcId | null;
}

// ── Helpers ────────────────────────────────────────────────────────────────

/** Maps carbon intensity to a dot radius in px (clamped 10-36). */
function dotRadius(tasks: number): number {
  return Math.min(36, Math.max(10, 10 + tasks * 0.4));
}

// ── Component ──────────────────────────────────────────────────────────────

export default function GlobalMap({ stepData, activeDcId }: Props) {
  const [tooltip, setTooltip] = useState<TooltipInfo | null>(null);

  const points = useMemo<DcPoint[]>(
    () =>
      DC_IDS.map((dc_id) => {
        const loc = DC_LOCATIONS[dc_id];
        const row = stepData.find((r) => r.datacenter === dc_id);
        return {
          dc_id,
          name: loc.name,
          flag: loc.flag,
          lon: loc.lon,
          lat: loc.lat,
          tasks: row?.running_tasks_count ?? 20,
          ci: row?.carbon_intensity_gco2_kwh ?? 300,
          cpu: row?.cpu_util_pct ?? 50,
          isActive: dc_id === activeDcId,
        };
      }),
    [stepData, activeDcId]
  );

  const handleMouseEnter = useCallback(
    (dc: DcPoint) => () => setTooltip({ dc }),
    []
  );
  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        borderRadius: 12,
        overflow: "hidden",
        background: "#0a0c10",
      }}
    >
      {/* Map fills parent fully — renderWorldCopies=false prevents tile repetition */}
      <Map
        initialViewState={INITIAL_VIEW}
        mapStyle={MAP_STYLE}
        attributionControl={false}
        style={{ width: "100%", height: "100%" }}
        reuseMaps
        renderWorldCopies={false}
      >
        {points.map((dc) => {
          const radius = dotRadius(dc.tasks);
          const color = dc.isActive ? "#00FFA0" : ciColor(dc.ci);
          const glowColor = dc.isActive
            ? "rgba(0,255,160,0.5)"
            : "rgba(255,255,255,0.15)";

          return (
            <Marker
              key={dc.dc_id}
              longitude={dc.lon}
              latitude={dc.lat}
              anchor="center"
            >
              <div
                style={{ position: "relative", cursor: "pointer" }}
                onMouseEnter={handleMouseEnter(dc)}
                onMouseLeave={handleMouseLeave}
              >
                {/* Outer pulse ring for active DC */}
                {dc.isActive && (
                  <div
                    style={{
                      position: "absolute",
                      top: "50%",
                      left: "50%",
                      transform: "translate(-50%, -50%)",
                      width: radius * 3,
                      height: radius * 3,
                      borderRadius: "50%",
                      border: "2px solid rgba(0,255,160,0.4)",
                      animation: "pulse-ring 1.6s ease-out infinite",
                      pointerEvents: "none",
                    }}
                  />
                )}

                {/* Main dot */}
                <div
                  style={{
                    width: radius,
                    height: radius,
                    borderRadius: "50%",
                    background: color,
                    boxShadow: `0 0 ${radius * 0.7}px ${glowColor}`,
                    border: dc.isActive
                      ? "2px solid rgba(0,255,160,0.9)"
                      : "1px solid rgba(255,255,255,0.18)",
                    transition: "background 0.35s, box-shadow 0.35s",
                  }}
                />

                {/* Label below dot */}
                <div
                  style={{
                    position: "absolute",
                    top: radius + 2,
                    left: "50%",
                    transform: "translateX(-50%)",
                    fontSize: 9,
                    fontFamily: "Roboto Mono, monospace",
                    color: dc.isActive
                      ? "rgba(0,255,160,0.95)"
                      : "rgba(160,180,200,0.7)",
                    whiteSpace: "nowrap",
                    pointerEvents: "none",
                    textShadow: dc.isActive
                      ? "0 0 6px rgba(0,255,160,0.6)"
                      : "none",
                  }}
                >
                  {dc.dc_id}
                </div>
              </div>
            </Marker>
          );
        })}
      </Map>

      {/* Neon border overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: 12,
          pointerEvents: "none",
          zIndex: 10,
          boxShadow:
            "inset 0 0 40px rgba(0,255,160,0.05), inset 0 0 1px rgba(0,255,160,0.18)",
        }}
      />

      {/* Hover tooltip */}
      {tooltip && (
        <div
          style={{
            position: "absolute",
            bottom: 12,
            right: 12,
            zIndex: 20,
            pointerEvents: "none",
          }}
        >
          <div className="glass-card px-3 py-2 text-xs min-w-[150px]">
            <div className="font-semibold text-white/90 mb-1">
              {tooltip.dc.flag} {tooltip.dc.dc_id} &mdash; {tooltip.dc.name}
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="micro-label">
                CI{" "}
                <span className="glow-green font-mono">
                  {tooltip.dc.ci.toFixed(0)}
                </span>{" "}
                gCO2/kWh
              </span>
              <span className="micro-label">
                Tasks{" "}
                <span className="text-white/80">{tooltip.dc.tasks}</span>
              </span>
              <span className="micro-label">
                CPU{" "}
                <span className="text-white/80">
                  {tooltip.dc.cpu.toFixed(1)}%
                </span>
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Active DC badge (bottom-left) */}
      {activeDcId && (() => {
        const loc = DC_LOCATIONS[activeDcId];
        const row = stepData.find((r) => r.datacenter === activeDcId);
        return (
          <div
            style={{ position: "absolute", bottom: 12, left: 12, zIndex: 20 }}
          >
            <div className="glass-card px-3 py-2 flex items-center gap-2">
              <div className="pulse-dot" />
              <span className="micro-label">Active</span>
              <span className="glow-green text-sm font-semibold">
                {loc.flag} {activeDcId}
              </span>
              {row && (
                <span className="micro-label text-white/50">
                  {row.running_tasks_count}t &middot;{" "}
                  {row.carbon_intensity_gco2_kwh.toFixed(0)} gCO2
                </span>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
