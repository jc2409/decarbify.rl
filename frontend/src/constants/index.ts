import type { DcId, DcLocation } from "@/types";
import type { Layout, Config } from "plotly.js";

// ── Strategy mappings ─────────────────────────────────────────────────────

export const STRATEGY_INTERNAL: Record<string, string> = {
  "SAC RL Agent": "manual_rl",
  "Local Only": "local_only",
  "Lowest Carbon": "lowest_carbon",
};

export const STRATEGY_DISPLAY: Record<string, string> = {
  manual_rl: "SAC RL Agent (Geo+Time)",
  local_only: "Local Only (Baseline)",
  lowest_carbon: "Lowest Carbon",
};

// Cyber neon palette for strategies
export const STRATEGY_COLORS: Record<string, string> = {
  "SAC RL Agent (Geo+Time)": "#00FFA0",
  "Local Only (Baseline)":   "#FF4757",
  "Lowest Carbon":           "#00D4FF",
};

export const STRATEGY_COLOR_BY_ID: Record<string, string> = {
  manual_rl:     "#00FFA0",
  local_only:    "#FF4757",
  lowest_carbon: "#00D4FF",
};

export const CHECKPOINT_OPTIONS: { display: string; internal: string }[] = [
  { display: "Multi-Action + Defer (recommended)", internal: "multi_action_enable_defer_2" },
  { display: "Multi-Action + No Defer",            internal: "multi_action_disable_defer_2" },
  { display: "Single-Action + Defer",              internal: "single_action_enable_defer_2" },
  { display: "Single-Action + No Defer",           internal: "single_action_disable_defer_2" },
];

// ── Datacenter metadata ───────────────────────────────────────────────────

export const DC_LOCATIONS: Record<DcId, DcLocation> = {
  DC1: { name: "US-California", lat: 37.39,  lon: -122.08, flag: "🇺🇸", utc_offset: -8,  zone: "US-CAL-CISO" },
  DC2: { name: "Germany",       lat: 50.11,  lon:    8.68, flag: "🇩🇪", utc_offset:  1,  zone: "DE" },
  DC3: { name: "Chile",         lat: -33.45, lon:  -70.67, flag: "🇨🇱", utc_offset: -4,  zone: "CL-SEN" },
  DC4: { name: "Singapore",     lat:  1.35,  lon:  103.82, flag: "🇸🇬", utc_offset:  8,  zone: "SG" },
  DC5: { name: "Australia",     lat: -33.87, lon:  151.21, flag: "🇦🇺", utc_offset: 11,  zone: "AU-NSW" },
};

export const DC_IDS: DcId[] = ["DC1", "DC2", "DC3", "DC4", "DC5"];

export const DC_PALETTE = ["#00FFA0", "#00D4FF", "#FFD166", "#FF8C42", "#A78BFA"];

// ── Simulation / playback constants ──────────────────────────────────────

export const STEPS_PER_HOUR = 4;          // 15-minute timesteps
export const WINDOW_STEPS   = 48;         // 12-hour sliding window (48 × 15min)
export const BASE_INTERVAL_MS = 300;      // base playback speed per step

// ── Plotly cyber dark layout ──────────────────────────────────────────────

export const PLOTLY_DARK_LAYOUT: Partial<Layout> = {
  plot_bgcolor:  "rgba(0,0,0,0)",
  paper_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#94a3b8", family: "Roboto Mono, monospace", size: 11 },
  legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1,
            font: { size: 10 }, bgcolor: "rgba(0,0,0,0)" },
  margin: { t: 32, b: 32, l: 52, r: 12 },
  xaxis: {
    gridcolor: "rgba(0,255,160,0.06)",
    zerolinecolor: "rgba(0,255,160,0.10)",
    linecolor: "rgba(0,255,160,0.12)",
    tickfont: { size: 10, color: "#4b5563" },
  },
  yaxis: {
    gridcolor: "rgba(0,255,160,0.06)",
    zerolinecolor: "rgba(0,255,160,0.10)",
    linecolor: "rgba(0,255,160,0.12)",
    tickfont: { size: 10, color: "#4b5563" },
  },
};

export const PLOTLY_CONFIG: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
};
