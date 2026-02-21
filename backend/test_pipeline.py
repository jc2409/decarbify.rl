"""
Integration test for the GreenDispatch inference pipeline.

Chains all three modules:
    mock_data → inference_engine → performance_tracker
and verifies output shapes, types, and format.

Run from the project root:
    python -m backend.test_pipeline
"""

import json
import sys
import os

# Ensure project root is on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.mock_data import generate_mock_sequence, DC_IDS
from backend.inference_engine import InferenceEngine, encode_observation, ACTION_LABELS
from backend.performance_tracker import PerformanceTracker


def main() -> None:
    print("=" * 70)
    print("GreenDispatch Inference Pipeline — Integration Test")
    print("=" * 70)

    # ── Step 1: Generate mock data ────────────────────────────────────────
    print("\n[1/4] Generating mock observation sequence (288 timesteps, 3 days)...")
    observations = generate_mock_sequence(num_timesteps=288, seed=42)
    assert len(observations) == 288, f"Expected 288 obs, got {len(observations)}"

    obs0 = observations[0]
    assert "timestep" in obs0
    assert "timestamp" in obs0
    assert "task" in obs0
    assert "datacenters" in obs0
    assert len(obs0["datacenters"]) == 5
    for dc_id in DC_IDS:
        dc = obs0["datacenters"][dc_id]
        assert "avail_cpu_ratio" in dc
        assert "avail_gpu_ratio" in dc
        assert "avail_mem_ratio" in dc
        assert "carbon_intensity_gco2_kwh" in dc
        assert "price_per_kwh" in dc
    print(f"  ✓ 288 observation dicts generated")
    print(f"  ✓ First timestamp: {obs0['timestamp']}")
    print(f"  ✓ Last timestamp:  {observations[-1]['timestamp']}")
    print(f"  ✓ Sample DC1 carbon intensity: {obs0['datacenters']['DC1']['carbon_intensity_gco2_kwh']} gCO₂/kWh")
    print(f"  ✓ Sample task origin: {obs0['task']['origin_dc']}")

    # ── Step 2: Test observation encoding ─────────────────────────────────
    print("\n[2/4] Testing observation encoding (dict → 34-dim vector)...")
    vec = encode_observation(obs0)
    assert vec.shape == (34,), f"Expected shape (34,), got {vec.shape}"
    assert vec.dtype.name.startswith("float"), f"Expected float dtype, got {vec.dtype}"
    print(f"  ✓ Encoded vector shape: {vec.shape}")
    print(f"  ✓ Vector range: [{vec.min():.4f}, {vec.max():.4f}]")

    # ── Step 3: Run inference ─────────────────────────────────────────────
    print("\n[3/4] Running inference engine (MockActor, 288 steps)...")
    engine = InferenceEngine()  # No checkpoint → MockActor
    assert engine.is_mock, "Expected MockActor mode"

    decisions = engine.run_sequence(observations)
    assert len(decisions) == 288, f"Expected 288 decisions, got {len(decisions)}"

    dec0 = decisions[0]
    assert "timestep" in dec0
    assert "timestamp" in dec0
    assert "decision" in dec0
    d = dec0["decision"]
    assert "action_index" in d
    assert "dc_chosen" in d
    assert "was_deferred" in d
    assert "confidence_scores" in d
    assert "reasoning" in d
    assert d["dc_chosen"] in ACTION_LABELS
    assert len(d["confidence_scores"]) == 6
    scores_sum = sum(d["confidence_scores"].values())
    assert abs(scores_sum - 1.0) < 0.01, f"Confidence scores sum to {scores_sum}, expected ~1.0"
    assert "summary" in d["reasoning"]
    assert "primary_factor" in d["reasoning"]
    assert "dc_analysis" in d["reasoning"]
    assert len(d["reasoning"]["dc_analysis"]) == 5

    # Count action distribution.
    action_counts = {label: 0 for label in ACTION_LABELS}
    for dec in decisions:
        action_counts[dec["decision"]["dc_chosen"]] += 1

    print(f"  ✓ 288 decision dicts generated (MockActor)")
    print(f"  ✓ First decision: {dec0['decision']['dc_chosen']} "
          f"(confidence {dec0['decision']['confidence_scores'][dec0['decision']['dc_chosen']]:.2%})")
    print(f"  ✓ Confidence scores sum: {scores_sum:.4f}")
    print(f"  ✓ Action distribution:")
    for label, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / 288 * 100
        bar = "█" * int(pct / 2)
        print(f"      {label:6s}: {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n  ✓ Sample reasoning: {dec0['decision']['reasoning']['summary'][:100]}...")

    # ── Step 4: Run performance tracker ───────────────────────────────────
    print("\n[4/4] Running performance tracker (local vs optimal, 5 DCs)...")
    tracker = PerformanceTracker()
    performance = tracker.compute(observations, decisions)

    assert len(performance) == 5, f"Expected 5 DCs, got {len(performance)}"
    for dc_id in DC_IDS:
        assert dc_id in performance
        series = performance[dc_id]
        assert len(series) == 288, f"{dc_id}: expected 288 timesteps, got {len(series)}"
        s0 = series[0]
        assert "local" in s0
        assert "optimal" in s0
        assert "delta" in s0
        assert "carbon_kg" in s0["local"]
        assert "energy_cost_usd" in s0["local"]
        assert "carbon_kg" in s0["optimal"]
        assert "tasks_received" in s0["optimal"]
        assert "carbon_kg_saved" in s0["delta"]
        assert "pct_carbon_saved" in s0["delta"]

    # Summarise.
    summaries = tracker.summarise(performance)
    print(f"  ✓ Performance computed for {len(performance)} DCs × 288 timesteps")
    print(f"\n  Per-DC Summary:")
    print(f"  {'DC':<5s} {'Local CO₂(kg)':>14s} {'Optimal CO₂(kg)':>16s} {'Saved(kg)':>11s} {'Saved(%)':>10s} {'Tasks Recv':>12s}")
    print(f"  {'─'*5} {'─'*14} {'─'*16} {'─'*11} {'─'*10} {'─'*12}")

    total_local = 0.0
    total_optimal = 0.0
    for dc_id in DC_IDS:
        s = summaries[dc_id]
        total_local += s["local_carbon_kg"]
        total_optimal += s["optimal_carbon_kg"]
        print(
            f"  {dc_id:<5s} {s['local_carbon_kg']:>14.4f} {s['optimal_carbon_kg']:>16.4f} "
            f"{s['carbon_saved_kg']:>11.4f} {s['pct_carbon_saved']:>9.1f}% "
            f"{s['tasks_received_from_others']:>12d}"
        )

    total_saved = total_local - total_optimal
    total_pct = (total_saved / total_local * 100) if total_local > 0 else 0
    print(f"\n  Cluster total: {total_local:.4f} kg (local) → {total_optimal:.4f} kg (optimal)")
    print(f"  Total saved: {total_saved:.4f} kg ({total_pct:.1f}%)")

    # ── Dump sample output ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Sample output (timestep 48 = noon day 1):")
    print("─" * 70)
    sample_obs = observations[48]
    sample_dec = decisions[48]
    sample_perf = {dc: performance[dc][48] for dc in DC_IDS}

    print(f"\nObservation (timestep {sample_obs['timestep']}, {sample_obs['timestamp']}):")
    print(f"  Task: origin={sample_obs['task']['origin_dc']}, "
          f"gpu={sample_obs['task']['gpu_cores_norm']:.3f}, "
          f"deadline={sample_obs['task']['deadline_hours']:.1f}h")
    for dc_id in DC_IDS:
        dc = sample_obs["datacenters"][dc_id]
        print(f"  {dc_id}: CI={dc['carbon_intensity_gco2_kwh']:6.1f} gCO₂/kWh, "
              f"price=${dc['price_per_kwh']:.3f}, "
              f"GPU avail={dc['avail_gpu_ratio']:.2f}")

    print(f"\nDecision:")
    print(f"  Chosen: {sample_dec['decision']['dc_chosen']} "
          f"(action {sample_dec['decision']['action_index']})")
    print(f"  Confidence: {sample_dec['decision']['confidence_scores']}")
    print(f"  Reasoning: {sample_dec['decision']['reasoning']['summary']}")
    print(f"  Primary factor: {sample_dec['decision']['reasoning']['primary_factor']}")

    print(f"\nPerformance (origin DC = {sample_obs['task']['origin_dc']}):")
    origin = sample_obs['task']['origin_dc']
    chosen = sample_dec['decision']['dc_chosen']
    if chosen != "DEFER":
        p_origin = sample_perf[origin]
        print(f"  {origin} local:   {p_origin['local']['carbon_kg']:.6f} kg CO₂, "
              f"${p_origin['local']['energy_cost_usd']:.6f}")
        print(f"  {origin} optimal: {p_origin['optimal']['carbon_kg']:.6f} kg CO₂, "
              f"${p_origin['optimal']['energy_cost_usd']:.6f}")
        print(f"  {origin} delta:   {p_origin['delta']['carbon_kg_saved']:.6f} kg saved "
              f"({p_origin['delta']['pct_carbon_saved']:.1f}%)")
        if chosen != origin:
            p_dest = sample_perf[chosen]
            print(f"  {chosen} received task: +{p_dest['optimal']['carbon_kg']:.6f} kg CO₂")

    print("\n" + "=" * 70)
    print("✅ All integration tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
