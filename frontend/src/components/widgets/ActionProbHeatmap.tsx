/**
 * ActionProbHeatmap — vertical bar chart of the SAC agent's action
 * probability distribution at the current timestep.
 *
 * Action index mapping (5-DC + defer enabled checkpoint):
 *   0   = DEFER
 *   1   = DC1 (US-California)
 *   2   = DC2 (Germany)
 *   3   = DC3 (Chile)
 *   4   = DC4 (Singapore)
 *   5   = DC5 (Australia)
 */

import { useEffect, useRef, useState } from "react";

interface Props {
  actionProbs: number[];
}

// ── Per-action metadata ────────────────────────────────────────────────────

interface ActionMeta {
  label: string;
  sublabel: string;
  color: string;
}

const ACTION_META: ActionMeta[] = [
  { label: "DEFER", sublabel: "wait",  color: "#FBBF24" },
  { label: "DC1",   sublabel: "US",    color: "#00FF9F" },
  { label: "DC2",   sublabel: "DE",    color: "#00FF9F" },
  { label: "DC3",   sublabel: "CL",    color: "#00FF9F" },
  { label: "DC4",   sublabel: "SG",    color: "#00FF9F" },
  { label: "DC5",   sublabel: "AU",    color: "#00FF9F" },
];

// ── Entropy ────────────────────────────────────────────────────────────────

function entropy(probs: number[]): number {
  return -probs
    .filter((p) => p > 0)
    .reduce((s, p) => s + p * Math.log2(p), 0);
}

function entropyLabel(h: number, nActions: number): { text: string; color: string } {
  const maxH = Math.log2(Math.max(nActions, 2));
  const ratio = h / maxH;
  if (ratio < 0.35) return { text: "CONFIDENT", color: "#00FF9F" };
  if (ratio < 0.65) return { text: "MODERATE",  color: "#FBBF24" };
  return                 { text: "UNCERTAIN",   color: "#F43F5E" };
}

// ── Component ──────────────────────────────────────────────────────────────

export default function ActionProbHeatmap({ actionProbs }: Props) {
  const prevProbs = useRef<number[]>([]);
  const [pulsing, setPulsing] = useState<Set<number>>(new Set());

  // Detect significant distribution shifts and trigger pulse animation
  useEffect(() => {
    if (actionProbs.length === 0 || prevProbs.current.length === 0) {
      prevProbs.current = actionProbs;
      return;
    }

    const changed = new Set<number>();
    actionProbs.forEach((p, i) => {
      if (Math.abs(p - (prevProbs.current[i] ?? 0)) > 0.05) changed.add(i);
    });

    if (changed.size > 0) {
      setPulsing(changed);
      const t = setTimeout(() => setPulsing(new Set()), 500);
      prevProbs.current = actionProbs;
      return () => clearTimeout(t);
    }
    prevProbs.current = actionProbs;
  }, [actionProbs]);

  const hasData = actionProbs.length > 0;
  const h = entropy(actionProbs);
  const { text: entropyText, color: entropyColor } = entropyLabel(h, actionProbs.length);
  const maxProb = Math.max(...actionProbs, 0.01);
  const maxIdx = actionProbs.indexOf(Math.max(...actionProbs));

  return (
    <div className="flex flex-col gap-2 pb-1">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div
          className="font-semibold tracking-wide text-white/80"
          style={{ fontSize: 13 }}
        >
          AGENT POLICY
        </div>
        {hasData && (
          <div
            className="font-mono"
            style={{ color: entropyColor, fontSize: 9, letterSpacing: "0.08em" }}
          >
            {entropyText}
          </div>
        )}
      </div>

      {!hasData ? (
        <div
          className="flex flex-col gap-1.5 items-center justify-center rounded"
          style={{
            height: 88,
            background: "rgba(22,27,34,0.5)",
            border: "1px solid #30363D",
          }}
        >
          <span style={{ color: "#8B949E", fontSize: 9, fontFamily: "Roboto Mono, monospace" }}>
            NO DATA YET
          </span>
          <span style={{ color: "#30363D", fontSize: 8, fontFamily: "Roboto Mono, monospace" }}>
            awaiting simulation
          </span>
        </div>
      ) : (
        <>
          {/* Vertical bars */}
          <div className="flex items-end gap-1" style={{ height: 76 }}>
            {actionProbs.map((prob, i) => {
              const meta = ACTION_META[i] ?? { label: `A${i}`, sublabel: "", color: "#00FF9F" };
              const heightPct = (prob / maxProb) * 100;
              const isPrimary = i === maxIdx;

              return (
                <div
                  key={i}
                  className="flex flex-col items-center flex-1 gap-1"
                  style={{ height: "100%" }}
                >
                  {/* Percentage label above bar */}
                  <span
                    className="font-mono"
                    style={{
                      fontSize: 8,
                      color: isPrimary ? meta.color : "#8B949E",
                      transition: "color 300ms ease",
                    }}
                  >
                    {(prob * 100).toFixed(0)}%
                  </span>

                  {/* Bar track */}
                  <div
                    className="flex-1 w-full flex flex-col justify-end rounded-sm overflow-hidden"
                    style={{ background: "#30363D" }}
                  >
                    <div
                      style={{
                        height: `${heightPct}%`,
                        background: isPrimary ? meta.color : meta.color + "44",
                        transition: "height 200ms ease-in-out, background 200ms ease",
                        boxShadow: isPrimary ? `0 0 6px ${meta.color}55` : "none",
                        animation: pulsing.has(i) ? "bar-pulse 0.5s ease-out" : "none",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Labels below bars */}
          <div className="flex gap-1">
            {actionProbs.map((_, i) => {
              const meta = ACTION_META[i] ?? { label: `A${i}`, sublabel: "", color: "#00FF9F" };
              const isPrimary = i === maxIdx;
              return (
                <div key={i} className="flex flex-col items-center flex-1">
                  <span
                    className="font-mono text-center"
                    style={{
                      fontSize: 7,
                      color: isPrimary ? "#E6EDF3" : "#8B949E",
                      letterSpacing: "0.04em",
                    }}
                  >
                    {meta.label}
                  </span>
                  <span
                    className="font-mono text-center"
                    style={{ fontSize: 6, color: "#4b5563" }}
                  >
                    {meta.sublabel}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
