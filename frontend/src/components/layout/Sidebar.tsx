import { useEffect, useRef } from "react";
import { useSimStore } from "@/store/useSimStore";
import { runSimulation } from "@/api/client";
import { CHECKPOINT_OPTIONS } from "@/constants";
import type { StrategyId } from "@/types";

const STRATEGIES: { label: string; id: StrategyId }[] = [
  { label: "SAC RL Agent", id: "manual_rl" },
  { label: "Local Only",   id: "local_only" },
  { label: "Lowest Carbon",id: "lowest_carbon" },
];

export default function Sidebar() {
  const {
    config,
    setConfig,
    setLoading,
    setResults,
    setError,
    isLoading,
    setIsPlaying,
    logEntries,
    clearLog,
  } = useSimStore();

  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log to bottom whenever entries change
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logEntries]);

  const handleRun = async () => {
    if (config.strategies.length === 0) return;
    setIsPlaying(false);
    clearLog();
    setLoading(true);
    setError(null);
    try {
      const data = await runSimulation({
        strategies: config.strategies,
        eval_days: config.evalDays,
        checkpoint_name: config.checkpointName,
        seed: 42,
        use_live: config.useLive,
      });
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation failed");
    } finally {
      setLoading(false);
    }
  };

  const toggleStrategy = (id: StrategyId) => {
    const cur = config.strategies;
    const next = cur.includes(id) ? cur.filter((s) => s !== id) : [...cur, id];
    setConfig({ strategies: next as StrategyId[] });
  };

  return (
    <aside
      className="w-52 shrink-0 flex flex-col overflow-hidden"
      style={{ borderRight: "1px solid rgba(0,255,160,0.08)", background: "rgba(10,12,16,0.6)" }}
    >
      {/* ── Fixed config section ────────────────────────────────────── */}
      <div className="flex flex-col gap-3 p-3 shrink-0">
        <div className="micro-label tracking-widest text-center pt-1">CONFIG</div>

        {/* Strategies */}
        <div>
          <div className="micro-label mb-1.5">Strategies</div>
          <div className="flex flex-col gap-1.5">
            {STRATEGIES.map(({ label, id }) => {
              const checked = config.strategies.includes(id);
              return (
                <label key={id} className="flex items-center gap-2 cursor-pointer group">
                  <div
                    onClick={() => toggleStrategy(id)}
                    className="w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-all cursor-pointer"
                    style={{
                      borderColor: checked ? "#00FFA0" : "rgba(255,255,255,0.15)",
                      background: checked ? "rgba(0,255,160,0.18)" : "transparent",
                      boxShadow: checked ? "0 0 6px rgba(0,255,160,0.3)" : "none",
                    }}
                  >
                    {checked && <span className="text-cyber-green text-xs leading-none">✓</span>}
                  </div>
                  <span className="text-xs text-white/60 group-hover:text-white/90 transition-colors">
                    {label}
                  </span>
                </label>
              );
            })}
          </div>
        </div>

        {/* Checkpoint */}
        <div>
          <div className="micro-label mb-1.5">RL Checkpoint</div>
          <select
            value={config.checkpointName}
            onChange={(e) => setConfig({ checkpointName: e.target.value })}
            className="w-full text-xs rounded px-2 py-1.5 focus:outline-none"
            style={{
              background: "rgba(17,19,24,0.8)",
              border: "1px solid rgba(255,255,255,0.10)",
              color: "rgba(255,255,255,0.6)",
            }}
          >
            {CHECKPOINT_OPTIONS.map(({ display, internal }) => (
              <option key={internal} value={internal}>{display}</option>
            ))}
          </select>
        </div>

        {/* Live toggle */}
        <div className="flex items-center justify-between">
          <span className="micro-label">Live Sim</span>
          <button
            onClick={() => setConfig({ useLive: !config.useLive })}
            className="relative w-9 h-5 rounded-full transition-colors"
            style={{
              background: config.useLive ? "rgba(0,255,160,0.5)" : "rgba(255,255,255,0.08)",
              border: `1px solid ${config.useLive ? "#00FFA0" : "rgba(255,255,255,0.12)"}`,
            }}
          >
            <span
              className="absolute top-0.5 left-0.5 w-3.5 h-3.5 rounded-full bg-white shadow transition-transform"
              style={{ transform: config.useLive ? "translateX(16px)" : "translateX(0)" }}
            />
          </button>
        </div>
      </div>

      {/* ── Agent log terminal — fills remaining height ─────────────── */}
      <div className="flex flex-col flex-1 min-h-0 px-3 pb-1">
        <div className="micro-label mb-1 flex items-center justify-between shrink-0">
          <span>AGENT LOG</span>
          {logEntries.length > 0 && (
            <button
              onClick={clearLog}
              className="micro-label text-white/20 hover:text-white/50 transition-colors"
            >
              CLR
            </button>
          )}
        </div>

        <div
          ref={logRef}
          className="flex-1 min-h-0 overflow-y-auto"
          style={{
            background: "rgba(0,0,0,0.45)",
            border: "1px solid rgba(0,255,160,0.08)",
            borderRadius: 6,
            padding: "6px 8px",
            fontFamily: "Roboto Mono, monospace",
            fontSize: 10,
            lineHeight: 1.6,
            scrollbarWidth: "none",
          }}
        >
          {logEntries.length === 0 ? (
            <span style={{ color: "rgba(255,255,255,0.12)" }}>
              awaiting simulation...
            </span>
          ) : (
            logEntries.map((e) => (
              <div key={e.id}>
                <span style={{ color: "rgba(255,255,255,0.22)" }}>[{e.simTime}]</span>
                {" "}
                <span style={{ color: e.color }}>{e.text}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* ── Run button ──────────────────────────────────────────────── */}
      <div className="p-3 pt-2 shrink-0">
        <button
          onClick={handleRun}
          disabled={config.strategies.length === 0 || isLoading}
          className="btn-cyber w-full py-2 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          {isLoading ? "Loading…" : "▶ RUN"}
        </button>
      </div>
    </aside>
  );
}
