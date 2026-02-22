import Plot from "react-plotly.js";
import { useMemo } from "react";
import type { PerDcRow } from "@/types";
import {
  STEPS_PER_HOUR,
  WINDOW_STEPS,
  STRATEGY_COLORS,
  STRATEGY_DISPLAY,
  PLOTLY_DARK_LAYOUT,
  PLOTLY_CONFIG,
} from "@/constants";

interface SlidingCO2ChartProps {
  allPerDc: PerDcRow[];
  playbackStep: number;
  maxStep: number;
  strategies: string[];
}

function buildStepMap(rows: PerDcRow[], ctrl: string): Map<number, number> {
  const m = new Map<number, number>();
  for (const row of rows) {
    if (row.controller !== ctrl) continue;
    m.set(row.timestep, (m.get(row.timestep) ?? 0) + row.carbon_kg);
  }
  return m;
}

const CHART_HEIGHT = 180;

export default function SlidingCO2Chart({
  allPerDc,
  playbackStep,
  maxStep,
  strategies,
}: SlidingCO2ChartProps) {
  const traces = useMemo(() => {
    return strategies.map((ctrl) => {
      const label = STRATEGY_DISPLAY[ctrl] ?? ctrl;
      if (maxStep < 1) {
        return {
          name: label,
          type: "scatter" as const,
          mode: "lines" as const,
          x: [] as number[],
          y: [] as number[],
          line: { color: STRATEGY_COLORS[label] ?? "#888", width: 2 },
        };
      }

      const stepMap = buildStepMap(allPerDc, ctrl);
      const cycleLen = maxStep + 1; // total number of timesteps in one loop
      const xs: number[] = [];
      const ys: number[] = [];
      for (let offset = 0; offset < WINDOW_STEPS; offset++) {
        const virtualStep = playbackStep - (WINDOW_STEPS - 1) + offset;
        // Wrap negatives to the tail of the previous cycle for a seamless loop
        const wrappedStep = ((virtualStep % cycleLen) + cycleLen) % cycleLen;
        xs.push(offset / STEPS_PER_HOUR);
        ys.push(stepMap.get(wrappedStep) ?? 0);
      }

      return {
        name: label,
        type: "scatter" as const,
        mode: "lines" as const,
        x: xs,
        y: ys,
        line: { color: STRATEGY_COLORS[label] ?? "#888", width: 2, shape: "spline" as const, smoothing: 0.7 },
        fill: ctrl === "manual_rl" ? ("tozeroy" as const) : ("none" as const),
        fillcolor: ctrl === "manual_rl" ? "rgba(0,255,160,0.06)" : undefined,
      };
    });
  }, [allPerDc, playbackStep, maxStep, strategies]);

  return (
    <Plot
      data={traces}
      useResizeHandler
      layout={{
        ...PLOTLY_DARK_LAYOUT,
        autosize: true,
        height: CHART_HEIGHT,
        uirevision: "lock",
        // t: 48 gives the legend row + title enough room above the plot area
        margin: { t: 48, b: 32, l: 52, r: 16 },
        legend: {
          orientation: "h" as const,
          // y > 1 + yanchor "bottom" places the legend above the plot area
          x: 0,
          y: 1.18,
          xanchor: "left" as const,
          yanchor: "bottom" as const,
          font: { size: 9, color: "#9ca3af" },
          bgcolor: "rgba(0,0,0,0)",
          borderwidth: 0,
          traceorder: "normal" as const,
        },
        xaxis: {
          ...PLOTLY_DARK_LAYOUT.xaxis,
          range: [0, WINDOW_STEPS / STEPS_PER_HOUR],
          tickformat: ".0f",
          dtick: 2,
          title: { text: "hours (rolling 12h)", font: { color: "#4b5563", size: 8 } },
        },
        yaxis: {
          ...PLOTLY_DARK_LAYOUT.yaxis,
          title: { text: "kg CO2", font: { color: "#6b7280", size: 9 } },
          rangemode: "tozero" as const,
        },
        title: {
          text: "CO2 Emissions",
          font: { color: "#4b5563", size: 10, family: "Roboto Mono, monospace" },
          x: 0.01,
          xanchor: "left" as const,
          yanchor: "top" as const,
          y: 0.99,
        },
      }}
      config={{ ...PLOTLY_CONFIG, responsive: true }}
      style={{ width: "100%", height: `${CHART_HEIGHT}px` }}
    />
  );
}
