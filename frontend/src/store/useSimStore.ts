import { create } from "zustand";
import type {
  SimulationResponse,
  LiveCarbonResponse,
  PerDcRow,
  SimulationConfig,
  DcId,
} from "@/types";
import { STEPS_PER_HOUR } from "@/constants";

export interface LogEntry {
  id: number;
  simTime: string;
  text: string;
  color: string;
}

interface SimState {
  // ── Config ───────────────────────────────────────────────────────────────
  config: SimulationConfig;
  setConfig: (patch: Partial<SimulationConfig>) => void;

  // ── Agent log ─────────────────────────────────────────────────────────────
  logEntries: LogEntry[];
  pushLog: (entry: Omit<LogEntry, "id">) => void;
  clearLog: () => void;

  // ── Simulation results ────────────────────────────────────────────────────
  results: SimulationResponse | null;
  isLoading: boolean;
  error: string | null;
  setResults: (r: SimulationResponse) => void;
  setLoading: (v: boolean) => void;
  setError: (e: string | null) => void;

  // ── Streaming playback ────────────────────────────────────────────────────
  playbackStep: number;          // current timestep (0 → maxStep, loops)
  isPlaying: boolean;
  speed: number;                 // multiplier: 0.5 | 1 | 2 | 4
  activeDcId: DcId | null;       // DC with most tasks this step (manual_rl)
  setPlaybackStep: (n: number) => void;
  setIsPlaying: (v: boolean) => void;
  setSpeed: (s: number) => void;
  setActiveDcId: (id: DcId | null) => void;

  // ── Live carbon ───────────────────────────────────────────────────────────
  liveCarbon: LiveCarbonResponse | null;
  setLiveCarbon: (r: LiveCarbonResponse) => void;
  setCarbonLoading: (v: boolean) => void;

  // ── Computed helpers ──────────────────────────────────────────────────────
  getMaxStep: () => number;
  getMaxHours: () => number;
  getCurrentHour: () => number;
  getStepData: (strategy: string) => PerDcRow[];
  getFilteredPerDc: () => PerDcRow[];
}

export const useSimStore = create<SimState>((set, get) => ({
  // Config
  config: {
    evalDays: 4,
    strategies: ["manual_rl", "local_only"],
    checkpointName: "multi_action_enable_defer_2",
    useLive: false,
  },
  setConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch } })),

  // Agent log
  logEntries: [],
  pushLog: (entry) =>
    set((s) => {
      const next = [...s.logEntries, { ...entry, id: Date.now() }];
      // Keep at most 120 entries so the DOM stays small
      return { logEntries: next.length > 120 ? next.slice(-120) : next };
    }),
  clearLog: () => set({ logEntries: [] }),

  // Results
  results: null,
  isLoading: false,
  error: null,
  setResults: (r) => set({ results: r, error: null, playbackStep: 0, logEntries: [] }),
  setLoading: (v) => set({ isLoading: v }),
  setError: (e) => set({ error: e, isLoading: false }),

  // Playback
  playbackStep: 0,
  isPlaying: false,
  speed: 1,
  activeDcId: null,
  setPlaybackStep: (n) => set({ playbackStep: n }),
  setIsPlaying: (v) => set({ isPlaying: v }),
  setSpeed: (s) => set({ speed: s }),
  setActiveDcId: (id) => set({ activeDcId: id }),

  // Live carbon
  liveCarbon: null,
  setLiveCarbon: (r) => set({ liveCarbon: r }),
  setCarbonLoading: (_v) => { /* retained for App.tsx compatibility — no-op */ },

  // Computed
  getMaxStep: () => {
    const r = get().results;
    if (!r || r.per_dc.length === 0) return 0;
    return Math.max(...r.per_dc.map((row) => row.timestep));
  },
  getMaxHours: () => {
    const r = get().results;
    if (!r || r.per_dc.length === 0) return 0;
    return Math.floor(Math.max(...r.per_dc.map((row) => row.timestep)) / STEPS_PER_HOUR);
  },
  getCurrentHour: () => get().playbackStep / STEPS_PER_HOUR,
  getStepData: (strategy: string) => {
    const { results, playbackStep } = get();
    if (!results) return [];
    return results.per_dc.filter(
      (r) => r.timestep === playbackStep && r.controller === strategy
    );
  },
  getFilteredPerDc: () => {
    const { results, playbackStep } = get();
    if (!results) return [];
    return results.per_dc.filter((r) => r.timestep <= playbackStep);
  },
}));
