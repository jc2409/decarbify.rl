import { useEffect, useRef } from "react";
import { useSimStore } from "@/store/useSimStore";
import { BASE_INTERVAL_MS, DC_IDS, DC_LOCATIONS, STEPS_PER_HOUR } from "@/constants";
import type { DcId } from "@/types";

/**
 * Drives the streaming playback loop.
 * - Advances playbackStep by 1 on each tick.
 * - Loops when it hits maxStep.
 * - Derives activeDcId (DC with most running tasks for manual_rl at current step).
 * - Auto-starts when results are loaded.
 */
/** Build a one-line agent decision log from per-dc rows at a given step. */
function buildLogEntry(
  results: NonNullable<ReturnType<typeof useSimStore.getState>["results"]>,
  step: number
) {
  const rlRows = results.per_dc.filter(
    (r) => r.timestep === step && r.controller === "manual_rl"
  );
  if (rlRows.length === 0) return null;

  const best = rlRows.reduce((a, b) =>
    a.running_tasks_count > b.running_tasks_count ? a : b
  );
  const totalDeferred = rlRows.reduce((s, r) => s + r.deferred_tasks_this_step, 0);

  const simHour = Math.floor(step / STEPS_PER_HOUR);
  const day = Math.floor(simHour / 24) + 1;
  const hh = String(simHour % 24).padStart(2, "0");
  const mm = String((step % STEPS_PER_HOUR) * 15).padStart(2, "0");
  const ts = `D${day} ${hh}:${mm}`;

  const loc = DC_LOCATIONS[best.datacenter as DcId];
  const ci = best.carbon_intensity_gco2_kwh.toFixed(0);
  const tasks = best.running_tasks_count;

  if (totalDeferred > 0) {
    return {
      simTime: ts,
      text: `DEFER ${totalDeferred}t -> ${best.datacenter} CI:${ci}`,
      color: "#FF8C42",
    };
  }
  return {
    simTime: ts,
    text: `ROUTE ${best.datacenter} (${loc?.name ?? ""}) | CI:${ci} | ${tasks}t`,
    color: best.carbon_intensity_gco2_kwh < 200 ? "#00FFA0" : "#94a3b8",
  };
}

export function usePlayback() {
  const {
    results,
    isPlaying,
    speed,
    playbackStep,
    setPlaybackStep,
    setIsPlaying,
    setActiveDcId,
    getMaxStep,
  } = useSimStore();

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Derive activeDcId from current step data
  const deriveActiveDc = (step: number): DcId | null => {
    if (!results) return null;
    const rlRows = results.per_dc.filter(
      (r) => r.timestep === step && r.controller === "manual_rl"
    );
    if (rlRows.length === 0) {
      // fallback: any strategy's most-loaded DC
      const allRows = results.per_dc.filter((r) => r.timestep === step);
      if (allRows.length === 0) return null;
      const best = allRows.reduce((a, b) =>
        a.running_tasks_count > b.running_tasks_count ? a : b
      );
      return best.datacenter as DcId;
    }
    const best = rlRows.reduce((a, b) =>
      a.running_tasks_count > b.running_tasks_count ? a : b
    );
    return best.datacenter as DcId;
  };

  const tick = () => {
    const maxStep = getMaxStep();
    if (maxStep === 0) return;

    useSimStore.setState((s) => {
      const next = s.playbackStep >= maxStep ? 0 : s.playbackStep + 1;
      return { playbackStep: next };
    });

    // Update active DC and push agent log entry
    const newStep = useSimStore.getState().playbackStep;
    const activeDc = deriveActiveDc(newStep);
    setActiveDcId(activeDc);

    const { results: r, pushLog } = useSimStore.getState();
    if (r) {
      const entry = buildLogEntry(r, newStep);
      if (entry) pushLog(entry);
    }
  };

  // Start/stop interval based on isPlaying + speed
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (!isPlaying || !results) return;

    const ms = BASE_INTERVAL_MS / speed;
    intervalRef.current = setInterval(tick, ms);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, speed, results]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-start when results first load
  useEffect(() => {
    if (results && !isPlaying) {
      setIsPlaying(true);
      // Set initial active DC
      setActiveDcId(deriveActiveDc(0));
    }
  }, [results]); // eslint-disable-line react-hooks/exhaustive-deps

  return null;
}
