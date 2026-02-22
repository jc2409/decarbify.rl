import { useEffect } from "react";
import { useSimStore } from "@/store/useSimStore";
import { runSimulation, fetchLiveCarbon } from "@/api/client";
import { usePlayback } from "@/hooks/usePlayback";
import { computeSavings } from "@/utils/dataHelpers";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import GlobalMap from "@/components/map/GlobalMap";
import SlidingCO2Chart from "@/components/charts/SlidingCO2Chart";
import RightPanel from "@/components/layout/RightPanel";
import LoadingSpinner from "@/components/ui/LoadingSpinner";

export default function App() {
  const {
    isLoading,
    error,
    results,
    setLiveCarbon,
    setCarbonLoading,
    setResults,
    setLoading,
    setError,
    config,
    playbackStep,
    activeDcId,
    getMaxStep,
    getCurrentHour,
    getMaxHours,
    getStepData,
    getFilteredPerDc,
  } = useSimStore();

  usePlayback();

  useEffect(() => {
    const autoRun = async () => {
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
        setError(err instanceof Error ? err.message : "Could not connect to API on :8000");
      } finally {
        setLoading(false);
      }
    };
    autoRun();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const load = async () => {
      setCarbonLoading(true);
      try { setLiveCarbon(await fetchLiveCarbon()); }
      catch { /* silent */ }
      finally { setCarbonLoading(false); }
    };
    load();
    const id = setInterval(load, 300_000);
    return () => clearInterval(id);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const maxStep     = getMaxStep();
  const maxHours    = getMaxHours();
  const curHour     = getCurrentHour();
  const savings     = results ? computeSavings(getFilteredPerDc()) : null;

  const mapStrategy = config.strategies.includes("manual_rl")
    ? "manual_rl"
    : config.strategies[0] ?? "manual_rl";

  const mapStepData = results
    ? results.per_dc.filter(
        (r) => r.timestep === playbackStep && r.controller === mapStrategy
      )
    : [];

  const stepData = results
    ? getStepData("manual_rl").concat(getStepData("local_only"))
    : [];

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {isLoading && <LoadingSpinner />}
      <Header />

      {error && !isLoading && (
        <div
          className="px-4 py-2 text-xs shrink-0 flex items-center gap-2"
          style={{
            background: "rgba(255,71,87,0.10)",
            borderBottom: "1px solid rgba(255,71,87,0.20)",
            color: "#FF4757",
          }}
        >
          <span>⚠</span>
          <span>
            {error} — run{" "}
            <code className="font-mono">uv run uvicorn backend.api:app --port 8000</code>
          </span>
        </div>
      )}

      {/* 3-column body */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left: config sidebar */}
        <Sidebar />

        {/* Centre: map (fills remaining height) + chart strip pinned at bottom */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0 p-2 gap-2">
          {/* Map — fills all available vertical space */}
          <div className="flex-1 min-h-0">
            <GlobalMap
              stepData={mapStepData}
              activeDcId={activeDcId}
            />
          </div>

          {/* Chart strip — fixed height, never shrinks */}
          <div
            className="glass-card shrink-0 overflow-hidden"
            style={{ height: "200px", paddingBottom: "12px" }}
          >
            {results ? (
              <SlidingCO2Chart
                allPerDc={results.per_dc}
                playbackStep={playbackStep}
                maxStep={maxStep}
                strategies={config.strategies}
              />
            ) : (
              <div className="flex items-center justify-center h-full micro-label text-white/20">
                Waiting for simulation…
              </div>
            )}
          </div>
        </div>

        {/* Right: stats panel */}
        <RightPanel
          stepData={stepData}
          allStepData={mapStepData}
          activeDcId={activeDcId}
          savings={savings}
          currentHour={curHour}
          maxHours={maxHours}
          playbackStep={playbackStep}
          maxStep={maxStep}
          summary={results?.summary ?? []}
        />
      </div>
    </div>
  );
}
