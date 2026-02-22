import { useSimStore } from "@/store/useSimStore";
import { DC_IDS, DC_LOCATIONS } from "@/constants";
import { ciColor } from "@/utils/dataHelpers";

const SPEEDS = [0.5, 1, 2, 4];

export default function Header() {
  const { isPlaying, setIsPlaying, speed, setSpeed, results, playbackStep, liveCarbon, activeDcId } =
    useSimStore();

  // Get CI for each DC at current playback step
  const ciBadges = DC_IDS.map((dc_id) => {
    const row = results?.per_dc.find(
      (r) => r.timestep === playbackStep && r.datacenter === dc_id
    );
    const ci = row?.carbon_intensity_gco2_kwh ?? null;
    const loc = DC_LOCATIONS[dc_id];
    return { dc_id, ci, flag: loc.flag };
  });

  return (
    <header className="flex items-center gap-3 px-4 py-2 border-b shrink-0"
      style={{ borderColor: "rgba(0,255,160,0.10)", background: "rgba(10,12,16,0.95)" }}>

      {/* Logo */}
      <div className="flex items-center gap-2 shrink-0">
        <span className="glow-green text-base">⬡</span>
        <span className="font-semibold tracking-tight text-white/90">GreenDispatch</span>
        <span className="micro-label text-white/20 hidden sm:inline">CARBON-AWARE SCHEDULER</span>
      </div>

      {/* CI badges */}
      <div className="flex items-center gap-1.5 flex-1 justify-center flex-wrap">
        {ciBadges.map(({ dc_id, ci, flag }) => {
          const color = ci !== null ? ciColor(ci) : "#3a3f4b";
          const isActive = dc_id === activeDcId;
          return (
            <div
              key={dc_id}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono transition-all"
              style={{
                color,
                border: `1px solid ${isActive ? color : color + "44"}`,
                background: isActive ? `${color}18` : "transparent",
                boxShadow: isActive ? `0 0 10px ${color}30` : "none",
              }}
            >
              <span>{flag}</span>
              <span>{dc_id}</span>
              {ci !== null && (
                <span className="opacity-80">{ci.toFixed(0)}</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 shrink-0">
        {/* Speed */}
        <div className="flex items-center gap-1">
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={speed === s ? "btn-cyber" : "btn-cyber-dim"}
            >
              {s}×
            </button>
          ))}
        </div>

        {/* Play/Pause */}
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          disabled={!results}
          className={results ? "btn-cyber" : "btn-cyber-dim"}
          style={{ minWidth: 64 }}
        >
          {isPlaying ? "⏸ Pause" : "▶ Play"}
        </button>

        {/* Live indicator */}
        {liveCarbon?.is_live && (
          <div className="flex items-center gap-1.5">
            <div className="pulse-dot" />
            <span className="micro-label text-cyber-green/70">LIVE</span>
          </div>
        )}
      </div>
    </header>
  );
}
