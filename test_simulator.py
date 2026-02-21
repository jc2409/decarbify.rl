#!/usr/bin/env python3
"""
Quick smoke-test for SustainClusterSimulator.

Runs a 1-day simulation with the local_only (RBC) strategy — no RL model
needed, so it completes quickly.

Usage:
    cd greendispatch
    python test_simulator.py
"""

import os
import sys
import logging

# Ensure backend package is importable from this script's directory.
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from backend.simulator import SustainClusterSimulator  # noqa: E402

SUSTAIN_CLUSTER_PATH = os.path.join(os.path.dirname(__file__), "sustain-cluster")


def main() -> None:
    print("=" * 60)
    print("GreenDispatch — SustainClusterSimulator smoke test")
    print("=" * 60)

    print(f"\nProject root: {os.path.abspath(SUSTAIN_CLUSTER_PATH)}")
    print("Initialising simulator…")
    sim = SustainClusterSimulator(project_root=SUSTAIN_CLUSTER_PATH)

    print("\nRunning 1-day comparison with ['local_only']…")
    per_dc_df, global_df, summary_df = sim.run_comparison(
        strategies=["local_only"],
        eval_days=1,
        # checkpoint_name is required but unused for RBC-only runs.
        checkpoint_name="multi_action_enable_defer_2",
        seed=42,
    )

    print("\n--- Summary ---")
    print(summary_df.to_string(index=False))

    print(f"\nper_dc_df  : {per_dc_df.shape[0]} rows × {per_dc_df.shape[1]} cols")
    print(f"global_df  : {global_df.shape[0]} rows × {global_df.shape[1]} cols")

    print("\n--- per_dc_df (first 5 rows) ---")
    print(per_dc_df.head().to_string(index=False))

    print("\n--- global_df (first 5 rows) ---")
    print(global_df.head().to_string(index=False))

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
