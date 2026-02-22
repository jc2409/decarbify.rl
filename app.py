"""
GreenDispatch — Carbon-Aware AI Workload Scheduler Dashboard
"""

import os
import sys
import time as _time
from datetime import datetime

# Ensure sustain-cluster packages (envs, simulation, etc.) are importable.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_APP_DIR, "sustain-cluster"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from backend.mock_data import (
    DC_INFO,
    CONTROLLER_LABELS,
    generate_mock_comparison,
    get_mock_live_carbon,
)

# ── Constants ────────────────────────────────────────────────────────────────

_SUSTAIN_CLUSTER_PATH = os.path.join(_APP_DIR, "sustain-cluster")

_STRATEGY_INTERNAL: dict[str, str] = {
    "SAC RL Agent":  "manual_rl",
    "Local Only":    "local_only",
    "Lowest Carbon": "lowest_carbon",
}

_STRATEGY_DISPLAY: dict[str, str] = {
    "manual_rl":     "SAC RL Agent (Geo+Time)",
    "local_only":    "Local Only (Baseline)",
    "lowest_carbon": "Lowest Carbon",
}

_CHECKPOINT_OPTIONS: dict[str, str] = {
    "Multi-Action + Defer (recommended)": "multi_action_enable_defer_2",
    "Multi-Action + No Defer":            "multi_action_disable_defer_2",
    "Single-Action + Defer":              "single_action_enable_defer_2",
    "Single-Action + No Defer":           "single_action_disable_defer_2",
}

_DC_LOCATIONS: dict[str, dict] = {
    dc_id: {"name": info["name"], "lat": info["lat"], "lon": info["lon"]}
    for dc_id, info in DC_INFO.items()
}

# ── Strategy color scheme (consistent everywhere) ───────────────────────────

STRATEGY_COLORS = {
    "SAC RL Agent (Geo+Time)": "#00C853",
    "Local Only (Baseline)":   "#FF6B6B",
    "Lowest Carbon":           "#4FC3F7",
}

_COLOR_MAP = STRATEGY_COLORS  # alias for charts

# DC palette
_DC_PALETTE = ["#00C49F", "#FF6B6B", "#4ECDC4", "#FFD166", "#A78BFA"]

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GreenDispatch",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0,200,83,0.10) 0%, rgba(0,200,83,0.03) 100%);
        border: 1px solid rgba(0,200,83,0.30);
        border-left: 3px solid #00C853;
        border-radius: 8px;
        padding: 14px 18px 10px 18px;
    }
    div[data-testid="stMetricValue"] > div {
        color: #00C853 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.82rem !important;
        opacity: 0.85;
    }
    section[data-testid="stSidebar"] {
        border-right: 2px solid rgba(0,200,83,0.25);
    }
    .stTabs [aria-selected="true"] {
        color: #00C853 !important;
        border-bottom-color: #00C853 !important;
        font-weight: 600;
    }
    .gd-callout {
        background: linear-gradient(135deg, rgba(0,200,83,0.15) 0%, rgba(0,200,83,0.04) 100%);
        border: 1px solid rgba(0,200,83,0.40);
        border-radius: 10px;
        padding: 16px 22px;
        margin: 12px 0 18px 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .gd-savings-banner {
        background: linear-gradient(135deg, rgba(0,200,83,0.18) 0%, rgba(0,200,83,0.06) 100%);
        border: 2px solid rgba(0,200,83,0.5);
        border-radius: 12px;
        padding: 18px 24px;
        margin: 8px 0 16px 0;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    .ci-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        margin: 2px 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .ci-green  { background: rgba(0,200,83,0.2);  color: #00C853; }
    .ci-yellow { background: rgba(255,179,0,0.2);  color: #FFB300; }
    .ci-red    { background: rgba(255,23,68,0.2);   color: #FF1744; }
    .recommendation-box {
        background: linear-gradient(135deg, rgba(0,200,83,0.15) 0%, rgba(0,200,83,0.05) 100%);
        border: 2px solid #00C853;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Shared chart layout ─────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=30, b=10),
    xaxis=dict(gridcolor="rgba(150,150,150,0.15)"),
    yaxis=dict(gridcolor="rgba(150,150,150,0.15)"),
)


# ── Cached runners ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_mock_simulation(
    strategies_tuple: tuple,
    eval_days: int,
    checkpoint_name: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run mock comparison; returns (per_dc_df, global_df, summary_df)."""
    per_dc_df, global_df, summary_df = generate_mock_comparison(
        strategies=list(strategies_tuple),
        eval_days=eval_days,
        checkpoint_name=checkpoint_name,
        seed=seed,
    )
    per_dc_df = per_dc_df.copy()
    per_dc_df["controller_label"] = per_dc_df["controller"].map(_STRATEGY_DISPLAY)
    summary_df = summary_df.copy()
    summary_df["controller_label"] = summary_df["Controller"].map(_STRATEGY_DISPLAY)
    return per_dc_df, global_df, summary_df


@st.cache_data(show_spinner=False)
def run_real_simulation(
    strategies_tuple: tuple,
    eval_days: int,
    checkpoint_name: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run real SustainCluster comparison (requires sustain-cluster install)."""
    from backend.simulator import SustainClusterSimulator
    sim = SustainClusterSimulator(project_root=_SUSTAIN_CLUSTER_PATH)
    per_dc_df, global_df, summary_df = sim.run_comparison(
        strategies=list(strategies_tuple),
        eval_days=eval_days,
        checkpoint_name=checkpoint_name,
        seed=seed,
    )
    per_dc_df = per_dc_df.copy()
    per_dc_df["controller_label"] = per_dc_df["controller"].map(_STRATEGY_DISPLAY)
    summary_df = summary_df.copy()
    summary_df["controller_label"] = summary_df["Controller"].map(_STRATEGY_DISPLAY)
    return per_dc_df, global_df, summary_df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_carbon_intensity() -> tuple[dict[str, float], bool]:
    """Fetch live (or mock) carbon intensity; cached for 5 minutes."""
    try:
        from backend.carbon_api import get_live_carbon_intensity
        return get_live_carbon_intensity()
    except Exception:
        return get_mock_live_carbon(), False


# ── Helper: filter data to current time window ──────────────────────────────

def _filter_to_hour(df: pd.DataFrame, current_hour: int, steps_per_hour: int = 4) -> pd.DataFrame:
    """Filter DataFrame to timesteps up to the current hour."""
    max_step = current_hour * steps_per_hour
    return df[df["timestep"] <= max_step]


def _filter_to_step(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Filter to exactly one timestep."""
    return df[df["timestep"] == step]


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("GreenDispatch")

    eval_days = st.slider("Simulation days", min_value=1, max_value=7, value=3)

    selected_labels = st.multiselect(
        "Strategies to compare",
        options=list(_STRATEGY_INTERNAL.keys()),
        default=list(_STRATEGY_INTERNAL.keys()),
    )

    checkpoint_label = st.selectbox(
        "Checkpoint",
        options=list(_CHECKPOINT_OPTIONS.keys()),
    )
    checkpoint_name = _CHECKPOINT_OPTIONS[checkpoint_label]

    st.divider()

    use_live = st.checkbox("Use live simulation", value=False,
                           help="When checked, attempts real SustainCluster backend. Default uses mock data.")

    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    st.divider()
    st.caption("Live Carbon tab refreshes every 5 min.")

# ── Main header ──────────────────────────────────────────────────────────────

st.title("GreenDispatch — Carbon-Aware AI Workload Scheduler")
st.caption(
    "Route ML training jobs across 5 global datacenters using an RL agent or "
    "rule-based strategies, and compare carbon emissions, energy cost, and SLA violations."
)

# ── How It Works expander ────────────────────────────────────────────────────

with st.expander("How GreenDispatch Works", expanded=False):
    st.markdown("""
- We simulate **5 globally distributed data centers** with real carbon intensity,
  electricity pricing, and cooling physics from the **SustainCluster benchmark** (HPE, NeurIPS 2025)
- A **Soft Actor-Critic (SAC) reinforcement learning agent** learns to route AI workloads
  to the cleanest, cheapest data centers and **defer non-urgent tasks** to low-carbon time windows
- We compare against baselines: **"Local Only"** (run where you are) and **"Lowest Carbon"**
  (always pick cleanest grid, ignoring capacity)
- The RL agent finds the **Pareto-optimal balance**: lowest total CO2 while maintaining
  SLA compliance and reasonable cost
    """)

# ── Auto-run on first load with default settings ────────────────────────────

# Use current minute as seed so each session/reload gets slightly different data
_dynamic_seed = int(datetime.now().timestamp() // 60)

if "summary_df" not in st.session_state and not run_btn:
    strategies = [_STRATEGY_INTERNAL[s] for s in list(_STRATEGY_INTERNAL.keys())]
    per_dc_df, global_df, summary_df = run_mock_simulation(
        strategies_tuple=tuple(strategies),
        eval_days=eval_days,
        checkpoint_name=checkpoint_name,
        seed=_dynamic_seed,
    )
    st.session_state.update(
        per_dc_df=per_dc_df,
        global_df=global_df,
        summary_df=summary_df,
        strategies_run=strategies,
        eval_days=eval_days,
    )

# ── Trigger simulation ──────────────────────────────────────────────────────

if run_btn:
    if not selected_labels:
        st.error("Please select at least one strategy.")
    else:
        strategies = [_STRATEGY_INTERNAL[s] for s in selected_labels]
        # Fresh seed each click so re-runs produce different data
        btn_seed = int(datetime.now().timestamp())
        with st.spinner("Running simulation..."):
            try:
                if use_live:
                    per_dc_df, global_df, summary_df = run_real_simulation(
                        strategies_tuple=tuple(strategies),
                        eval_days=eval_days,
                        checkpoint_name=checkpoint_name,
                        seed=btn_seed,
                    )
                else:
                    per_dc_df, global_df, summary_df = run_mock_simulation(
                        strategies_tuple=tuple(strategies),
                        eval_days=eval_days,
                        checkpoint_name=checkpoint_name,
                        seed=btn_seed,
                    )
            except Exception as e:
                st.warning(f"Live simulation failed ({e}). Falling back to mock data.")
                per_dc_df, global_df, summary_df = run_mock_simulation(
                    strategies_tuple=tuple(strategies),
                    eval_days=eval_days,
                    checkpoint_name=checkpoint_name,
                    seed=btn_seed,
                )
        st.session_state.update(
            per_dc_df=per_dc_df,
            global_df=global_df,
            summary_df=summary_df,
            strategies_run=strategies,
            eval_days=eval_days,
        )

# ── Check for results ───────────────────────────────────────────────────────

has_results = "summary_df" in st.session_state

# ── Time Slider (shown when results exist) ───────────────────────────────────

if has_results:
    current_eval_days = st.session_state.get("eval_days", eval_days)
    max_hours = current_eval_days * 24
    max_steps = max_hours * 4

    st.subheader("Simulation Timeline")

    col_slider, col_play = st.columns([6, 1])
    with col_slider:
        current_hour = st.slider(
            "Simulation Hour",
            min_value=0,
            max_value=max_hours,
            value=st.session_state.get("current_hour", max_hours),
            step=1,
            key="time_slider",
            help="Drag to see how metrics evolve over time. All charts update based on this position.",
        )
        st.session_state["current_hour"] = current_hour
    with col_play:
        auto_play = st.button("Play", help="Auto-advance the timeline")

    current_step = current_hour * 4

    # ── Carbon intensity strip ───────────────────────────────────────────────
    per_dc_df = st.session_state.per_dc_df
    global_df = st.session_state.global_df
    summary_df = st.session_state.summary_df
    strategies_run = st.session_state.strategies_run

    # Get CI values at the current step
    first_strategy = strategies_run[0]
    step_data = per_dc_df[
        (per_dc_df["timestep"] == min(current_step, per_dc_df["timestep"].max())) &
        (per_dc_df["controller"] == first_strategy)
    ]

    if not step_data.empty:
        badges_html = f"<div style='margin: 4px 0 12px 0;'>Hour {current_hour} of {max_hours}: "
        for _, row in step_data.iterrows():
            dc_id = row["datacenter"]
            ci = row["carbon_intensity_gco2_kwh"]
            info = DC_INFO[dc_id]
            if ci < 150:
                css_class = "ci-green"
            elif ci <= 350:
                css_class = "ci-yellow"
            else:
                css_class = "ci-red"
            badges_html += (
                f'<span class="ci-badge {css_class}">'
                f'{info["flag"]} {info["name"]}: {ci:.0f} gCO\u2082/kWh</span>'
            )
        badges_html += "</div>"
        st.markdown(badges_html, unsafe_allow_html=True)

    # ── Cumulative savings counter ───────────────────────────────────────────
    filtered = _filter_to_hour(per_dc_df, current_hour)

    has_rl = "manual_rl" in strategies_run
    has_baseline = "local_only" in strategies_run

    if has_rl and has_baseline and not filtered.empty:
        rl_data = filtered[filtered["controller"] == "manual_rl"]
        bl_data = filtered[filtered["controller"] == "local_only"]

        rl_co2 = rl_data["carbon_kg"].sum()
        bl_co2 = bl_data["carbon_kg"].sum()
        co2_saved = bl_co2 - rl_co2
        co2_pct = (co2_saved / bl_co2 * 100) if bl_co2 > 0 else 0

        rl_cost = rl_data["energy_cost_usd"].sum()
        bl_cost = bl_data["energy_cost_usd"].sum()
        cost_delta = rl_cost - bl_cost
        cost_pct = (cost_delta / bl_cost * 100) if bl_cost > 0 else 0

        # Deferred tasks up to current hour
        filtered_global = _filter_to_hour(global_df, current_hour)
        rl_global = filtered_global[filtered_global["controller"] == "manual_rl"]
        total_deferred = int(rl_global["deferred_tasks_this_step"].sum()) if not rl_global.empty else 0

        st.markdown(
            f"""<div class="gd-savings-banner">
            <strong>CO\u2082 Saved: {co2_saved:.1f} kg</strong> (\u2193 {co2_pct:.1f}%)
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Cost Impact: {'+' if cost_delta >= 0 else ''}{cost_delta:.2f} USD</strong> ({'+' if cost_pct >= 0 else ''}{cost_pct:.1f}%)
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Hour: {current_hour}/{max_hours}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Deferred: {total_deferred}
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Auto-play logic ──────────────────────────────────────────────────────
    if auto_play:
        progress_placeholder = st.empty()
        play_start = max(0, current_hour - 1)
        step_size = max(1, max_hours // 100)  # ~100 frames
        for h in range(play_start, max_hours + 1, step_size):
            st.session_state["current_hour"] = h
            progress_placeholder.progress(h / max_hours, text=f"Hour {h}/{max_hours}")
            _time.sleep(0.05)
        st.session_state["current_hour"] = max_hours
        st.rerun()

# ── Five tabs ────────────────────────────────────────────────────────────────

tab_summary, tab_map, tab_ts, tab_tradeoff, tab_live = st.tabs([
    "Summary", "Map", "Time Series", "Trade-offs", "Live Carbon",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Summary
# ══════════════════════════════════════════════════════════════════════════════

with tab_summary:
    if not has_results:
        st.info("Configure your simulation in the sidebar and click **Run Simulation** to begin.")
    else:
        per_dc_df = st.session_state.per_dc_df
        summary_df = st.session_state.summary_df
        strategies_run = st.session_state.strategies_run

        # Compute cumulative summary up to current hour
        filtered = _filter_to_hour(per_dc_df, current_hour)

        # ── Big KPI cards: RL vs Local Only ──
        rl_row_full = summary_df[summary_df["Controller"] == "manual_rl"]
        bl_row_full = summary_df[summary_df["Controller"] == "local_only"]

        if not rl_row_full.empty and not bl_row_full.empty:
            rl_filt = filtered[filtered["controller"] == "manual_rl"]
            bl_filt = filtered[filtered["controller"] == "local_only"]

            rl_co2 = rl_filt["carbon_kg"].sum()
            bl_co2 = bl_filt["carbon_kg"].sum()
            co2_saved = bl_co2 - rl_co2

            rl_energy = rl_filt["energy_kwh"].sum()
            bl_energy = bl_filt["energy_kwh"].sum()
            energy_saved = bl_energy - rl_energy

            rl_water = rl_filt["water_usage_m3"].sum()
            bl_water = bl_filt["water_usage_m3"].sum()
            water_saved = bl_water - rl_water

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "CO\u2082 Saved vs Baseline",
                f"{co2_saved:,.1f} kg",
                delta=f"{(co2_saved / bl_co2 * 100):.1f}% reduction" if bl_co2 > 0 else "N/A",
                delta_color="normal",
            )
            c2.metric(
                "Energy Saved vs Baseline",
                f"{energy_saved:,.1f} kWh",
                delta=f"{(energy_saved / bl_energy * 100):.1f}% reduction" if bl_energy > 0 else "N/A",
                delta_color="normal",
            )
            c3.metric(
                "Water Saved vs Baseline",
                f"{water_saved:,.4f} m\u00b3",
                delta=f"{(water_saved / bl_water * 100):.1f}% reduction" if bl_water > 0 else "N/A",
                delta_color="normal",
            )

        st.divider()

        # ── Grouped bar chart comparing strategies ──
        st.subheader("Strategy Comparison")
        st.caption("Grouped comparison across key metrics. The RL agent balances all dimensions.")

        bar_data = []
        for _, row in summary_df.iterrows():
            label = row["controller_label"]
            bar_data.append({"Strategy": label, "Metric": "CO\u2082 (kg)", "Value": row["Total CO2 (kg)"]})
            bar_data.append({"Strategy": label, "Metric": "Energy (kWh)", "Value": row["Total Energy (kWh)"]})
            bar_data.append({"Strategy": label, "Metric": "Cost ($)", "Value": row["Total Cost ($)"]})
            bar_data.append({"Strategy": label, "Metric": "SLA Viol. (%)", "Value": row["SLA Violation Rate (%)"] * 20})  # scale for visibility

        bar_df = pd.DataFrame(bar_data)
        fig_bar = px.bar(
            bar_df, x="Metric", y="Value", color="Strategy",
            barmode="group",
            color_discrete_map=_COLOR_MAP,
        )
        fig_bar.update_layout(**_CHART_LAYOUT, height=400, showlegend=True)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Strategy Recommendation callout ──
        if not rl_row_full.empty and not bl_row_full.empty:
            rl = rl_row_full.iloc[0]
            bl = bl_row_full.iloc[0]
            co2_reduction_pct = (1 - rl["Total CO2 (kg)"] / bl["Total CO2 (kg)"]) * 100
            cost_change_pct = (rl["Total Cost ($)"] / bl["Total Cost ($)"] - 1) * 100
            sla_rate = rl["SLA Violation Rate (%)"]
            deferred = int(rl["Total Tasks Deferred"])

            # Compute workload routing percentages
            rl_tasks = filtered[filtered["controller"] == "manual_rl"]
            if not rl_tasks.empty:
                dc_totals = rl_tasks.groupby("datacenter")["tasks_assigned_count"].sum()
                total_tasks = dc_totals.sum()
                if total_tasks > 0:
                    top_dcs = dc_totals.nlargest(2)
                    routing_text = " and ".join(
                        f"{DC_INFO[dc]['name']}" for dc in top_dcs.index
                    )
                    routing_pct = top_dcs.sum() / total_tasks * 100
                else:
                    routing_text = "cleaner regions"
                    routing_pct = 67
            else:
                routing_text = "cleaner regions"
                routing_pct = 67

            st.markdown(
                f"""<div class="recommendation-box">
                <strong>Recommended: SAC RL Agent (Geo+Time)</strong><br/><br/>
                Reduces CO\u2082 by <strong>{co2_reduction_pct:.1f}%</strong> vs baseline
                with {'only ' if abs(cost_change_pct) < 5 else ''}<strong>{cost_change_pct:+.1f}%</strong> cost change
                and <strong>{sla_rate:.1f}%</strong> SLA violation rate.<br/>
                The agent intelligently defers <strong>{deferred:,}</strong> tasks to low-carbon windows
                and routes <strong>{routing_pct:.0f}%</strong> of workload to {routing_text}
                during clean grid periods.
                </div>""",
                unsafe_allow_html=True,
            )

        # ── Full summary table ──
        with st.expander("Full Summary Table", expanded=False):
            disp = summary_df.drop(columns=["controller_label"], errors="ignore").copy()
            disp["Controller"] = disp["Controller"].map(_STRATEGY_DISPLAY)
            fmt = {
                "Total CO2 (kg)":         "{:,.1f}",
                "Total Energy (kWh)":     "{:,.1f}",
                "Total Cost ($)":         "{:,.2f}",
                "Total Water (m3)":       "{:,.4f}",
                "SLA Violation Rate (%)": "{:.2f}",
                "Avg CPU Util (%)":       "{:.2f}",
                "Avg GPU Util (%)":       "{:.2f}",
                "Total Tasks Deferred":   "{:,}",
            }
            try:
                styled = (
                    disp.style
                    .format(fmt)
                    .background_gradient(
                        subset=["Total CO2 (kg)", "Total Energy (kWh)", "Total Cost ($)"],
                        cmap="RdYlGn_r",
                    )
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Map
# ══════════════════════════════════════════════════════════════════════════════

with tab_map:
    if not has_results:
        st.info("Run a simulation to populate the map with per-DC metrics.")
    else:
        per_dc_df = st.session_state.per_dc_df
        strategies_run = st.session_state.strategies_run

        st.subheader("Global Datacenter Overview")
        st.caption("Circle size = running tasks at current hour. Color = carbon intensity (green=clean, red=dirty).")

        map_strategy = st.selectbox(
            "Strategy for map",
            options=[_STRATEGY_DISPLAY[s] for s in strategies_run],
            index=0,
            key="map_strategy",
        )
        # Reverse-lookup internal name
        map_strat_internal = {v: k for k, v in _STRATEGY_DISPLAY.items()}[map_strategy]

        # Get data at current timestep
        target_step = min(current_step, per_dc_df["timestep"].max())
        step_data = per_dc_df[
            (per_dc_df["timestep"] == target_step) &
            (per_dc_df["controller"] == map_strat_internal)
        ]

        map_rows = []
        for dc_id, dc_meta in _DC_LOCATIONS.items():
            dc_slice = step_data[step_data["datacenter"] == dc_id]
            if not dc_slice.empty:
                row = dc_slice.iloc[0]
                ci = float(row["carbon_intensity_gco2_kwh"])
                tasks = int(row["running_tasks_count"])
                energy = float(row["energy_kwh"])
            else:
                ci = 0.0
                tasks = 0
                energy = 0.0

            map_rows.append({
                "dc": dc_id,
                "location": dc_meta["name"],
                "lat": dc_meta["lat"],
                "lon": dc_meta["lon"],
                "ci": round(ci, 1),
                "tasks": tasks,
                "energy": round(energy, 2),
                "radius": max(150_000, int(tasks * 8000 + 100_000)),
            })

        map_df = pd.DataFrame(map_rows)

        # Color by CI: green (low) to red (high)
        ci_min = max(map_df["ci"].min(), 1.0)
        ci_max = max(map_df["ci"].max(), ci_min + 1.0)
        norm = (map_df["ci"] - ci_min) / (ci_max - ci_min)
        map_df["r"] = (norm * 215 + 40).astype(int)
        map_df["g"] = ((1 - norm) * 195 + 60).astype(int)
        map_df["b"] = 70

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_color=["r", "g", "b", 190],
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            stroked=True,
            get_line_color=[255, 255, 255, 80],
            line_width_min_pixels=1,
        )

        label_df = map_df.copy()
        label_df["label_lat"] = label_df["lat"] + 4.5
        text_layer = pdk.Layer(
            "TextLayer",
            data=label_df,
            get_position=["lon", "label_lat"],
            get_text="dc",
            get_size=14,
            get_color=[255, 255, 255, 220],
            get_anchor="'middle'",
            get_alignment_baseline="'bottom'",
            billboard=True,
        )

        layers = [scatter_layer, text_layer]

        # ── Arc layer for RL agent task flows ────────────────────────────────
        if map_strat_internal == "manual_rl":
            fair_share = map_df["tasks"].sum() / 5.0 if map_df["tasks"].sum() > 0 else 1
            arcs = []
            # Find DCs receiving more than fair share
            receiving_dcs = map_df[map_df["tasks"] > fair_share * 1.2]
            sending_dcs = map_df[map_df["tasks"] < fair_share * 0.8]

            for _, recv in receiving_dcs.iterrows():
                for _, send in sending_dcs.iterrows():
                    arcs.append({
                        "source_lon": send["lon"],
                        "source_lat": send["lat"],
                        "target_lon": recv["lon"],
                        "target_lat": recv["lat"],
                        "flow": int(recv["tasks"] - fair_share),
                    })

            if arcs:
                arc_df = pd.DataFrame(arcs)
                arc_layer = pdk.Layer(
                    "ArcLayer",
                    data=arc_df,
                    get_source_position=["source_lon", "source_lat"],
                    get_target_position=["target_lon", "target_lat"],
                    get_source_color=[0, 200, 83, 160],
                    get_target_color=[0, 200, 83, 80],
                    get_width=3,
                    pickable=True,
                )
                layers.append(arc_layer)

        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(
                    latitude=15, longitude=15, zoom=1.2, pitch=0,
                ),
                tooltip={
                    "html": (
                        "<b style='font-size:14px'>{dc} — {location}</b><br/>"
                        "Carbon Intensity: <b>{ci}</b> gCO\u2082/kWh<br/>"
                        "Running Tasks: <b>{tasks}</b><br/>"
                        "Energy: <b>{energy}</b> kWh"
                    ),
                    "style": {
                        "backgroundColor": "#1a1a2e",
                        "color": "white",
                        "fontSize": "13px",
                        "borderRadius": "6px",
                        "padding": "10px 14px",
                        "border": "1px solid rgba(0,200,83,0.4)",
                    },
                },
                map_style=pdk.map_styles.CARTO_DARK,
            ),
            use_container_width=True,
            height=460,
        )

        st.dataframe(
            map_df[["dc", "location", "ci", "tasks", "energy"]].rename(columns={
                "dc":       "DC",
                "location": "Location",
                "ci":       "CI (gCO\u2082/kWh)",
                "tasks":    "Running Tasks",
                "energy":   "Energy (kWh)",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Time Series
# ══════════════════════════════════════════════════════════════════════════════

with tab_ts:
    if not has_results:
        st.info("Run a simulation to see time-series charts.")
    else:
        per_dc_df = st.session_state.per_dc_df
        global_df = st.session_state.global_df
        strategies_run = st.session_state.strategies_run

        # Filter to current time window
        filtered_dc = _filter_to_hour(per_dc_df, current_hour)
        filtered_global = _filter_to_hour(global_df, current_hour)

        if filtered_dc.empty:
            st.info("Move the timeline slider to see data.")
        else:
            agg = (
                filtered_dc
                .groupby(["timestep", "controller", "controller_label"], as_index=False)
                .agg(
                    carbon_kg=("carbon_kg", "sum"),
                    energy_cost_usd=("energy_cost_usd", "sum"),
                )
            )

            # Convert timestep to hours for readability
            agg["hour"] = agg["timestep"] / 4.0

            c_l, c_r = st.columns(2)

            with c_l:
                st.subheader("CO\u2082 Emissions Over Time")
                st.caption("Lower is better. The RL agent reduces emissions by shifting work to cleaner grids.")
                fig_co2 = px.line(
                    agg, x="hour", y="carbon_kg",
                    color="controller_label",
                    color_discrete_map=_COLOR_MAP,
                    labels={
                        "hour":             "Hour",
                        "carbon_kg":        "CO\u2082 (kg)",
                        "controller_label": "Strategy",
                    },
                )
                fig_co2.update_layout(**_CHART_LAYOUT)

                # Add annotation for RL deferral
                if "manual_rl" in strategies_run and not filtered_global.empty:
                    rl_global = filtered_global[filtered_global["controller"] == "manual_rl"]
                    if not rl_global.empty:
                        peak_defer = rl_global.loc[rl_global["deferred_tasks_this_step"].idxmax()]
                        if peak_defer["deferred_tasks_this_step"] > 3:
                            fig_co2.add_annotation(
                                x=peak_defer["timestep"] / 4.0,
                                y=agg[agg["controller_label"] == "SAC RL Agent (Geo+Time)"]["carbon_kg"].max() * 0.85,
                                text="Agent defers during peak carbon",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="#00C853",
                                font=dict(size=10, color="#00C853"),
                                bgcolor="rgba(0,0,0,0.6)",
                                bordercolor="#00C853",
                            )

                st.plotly_chart(fig_co2, use_container_width=True)

            with c_r:
                st.subheader("Energy Cost Over Time")
                st.caption("Cost per 15-min interval by strategy.")
                fig_cost = px.line(
                    agg, x="hour", y="energy_cost_usd",
                    color="controller_label",
                    color_discrete_map=_COLOR_MAP,
                    labels={
                        "hour":             "Hour",
                        "energy_cost_usd":  "Cost (USD)",
                        "controller_label": "Strategy",
                    },
                )
                fig_cost.update_layout(**_CHART_LAYOUT)
                st.plotly_chart(fig_cost, use_container_width=True)

            # ── Carbon Intensity per DC ──
            st.subheader("Carbon Intensity per Datacenter")
            st.caption(
                "Carbon intensity is grid data — identical across strategies. "
                "Watch how it varies by time of day and region."
            )
            first_strategy = strategies_run[0]
            ci_df = filtered_dc[filtered_dc["controller"] == first_strategy][
                ["timestep", "datacenter", "carbon_intensity_gco2_kwh"]
            ].copy()
            ci_df["hour"] = ci_df["timestep"] / 4.0

            fig_ci = px.line(
                ci_df, x="hour", y="carbon_intensity_gco2_kwh",
                color="datacenter",
                color_discrete_sequence=_DC_PALETTE,
                labels={
                    "hour":                      "Hour",
                    "carbon_intensity_gco2_kwh": "CI (gCO\u2082/kWh)",
                    "datacenter":                "Datacenter",
                },
            )
            fig_ci.update_layout(**_CHART_LAYOUT)
            st.plotly_chart(fig_ci, use_container_width=True)

            # ── Task Distribution Over Time ──
            st.subheader("Task Distribution by Datacenter")
            st.caption(
                "Watch how the RL agent sends more work to Chile (cleanest grid) while keeping load balanced."
            )

            task_dist_strat = st.selectbox(
                "Strategy for task distribution",
                options=[_STRATEGY_DISPLAY[s] for s in strategies_run],
                index=0,
                key="task_dist_strategy",
            )
            task_strat_internal = {v: k for k, v in _STRATEGY_DISPLAY.items()}[task_dist_strat]

            task_df = filtered_dc[filtered_dc["controller"] == task_strat_internal][
                ["timestep", "datacenter", "running_tasks_count"]
            ].copy()
            task_df["hour"] = task_df["timestep"] / 4.0

            fig_tasks = px.area(
                task_df, x="hour", y="running_tasks_count",
                color="datacenter",
                color_discrete_sequence=_DC_PALETTE,
                labels={
                    "hour":                "Hour",
                    "running_tasks_count": "Running Tasks",
                    "datacenter":          "Datacenter",
                },
            )
            fig_tasks.update_layout(**_CHART_LAYOUT)
            st.plotly_chart(fig_tasks, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Trade-offs (Pareto)
# ══════════════════════════════════════════════════════════════════════════════

with tab_tradeoff:
    if not has_results:
        st.info("Run a simulation to see the cost-vs-carbon trade-off chart.")
    else:
        per_dc_df = st.session_state.per_dc_df
        summary_df = st.session_state.summary_df

        st.subheader("Cost vs Carbon Trade-off (Pareto)")
        st.caption(
            "The ideal strategy is bottom-left (low cost, low carbon). "
            "The RL agent finds this sweet spot — Pareto optimal."
        )

        # Compute cumulative totals up to current hour for Pareto plot
        filtered = _filter_to_hour(per_dc_df, current_hour)
        if not filtered.empty:
            pareto_data = (
                filtered
                .groupby(["controller"], as_index=False)
                .agg(
                    total_co2=("carbon_kg", "sum"),
                    total_cost=("energy_cost_usd", "sum"),
                    total_energy=("energy_kwh", "sum"),
                )
            )
            pareto_data["controller_label"] = pareto_data["controller"].map(_STRATEGY_DISPLAY)

            fig_scatter = px.scatter(
                pareto_data,
                x="total_cost",
                y="total_co2",
                size="total_energy",
                color="controller_label",
                text="controller_label",
                color_discrete_map=_COLOR_MAP,
                size_max=72,
                labels={
                    "total_cost":       "Total Energy Cost (USD)",
                    "total_co2":        "Total CO\u2082 Emissions (kg)",
                    "controller_label": "Strategy",
                    "total_energy":     "Total Energy (kWh)",
                },
            )
            fig_scatter.update_traces(
                textposition="top center",
                marker=dict(opacity=0.88, line=dict(width=1.5, color="white")),
            )
            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=520,
                xaxis=dict(gridcolor="rgba(150,150,150,0.2)", zeroline=False),
                yaxis=dict(gridcolor="rgba(150,150,150,0.2)", zeroline=False),
                margin=dict(t=40),
            )
            fig_scatter.add_annotation(
                text="\u2190 Lower cost & lower carbon is better",
                xref="paper", yref="paper",
                x=0.01, y=0.03,
                showarrow=False,
                font=dict(size=11, color="rgba(180,180,180,0.75)"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Summary table
        tdf = summary_df.copy()
        tdf_display = (
            tdf[["controller_label", "Total CO2 (kg)", "Total Energy (kWh)",
                 "Total Cost ($)", "Total Water (m3)", "SLA Violation Rate (%)",
                 "Total Tasks Deferred"]]
            .rename(columns={"controller_label": "Strategy"})
        )
        st.dataframe(
            tdf_display.style.format({
                "Total CO2 (kg)":         "{:,.1f}",
                "Total Energy (kWh)":     "{:,.1f}",
                "Total Cost ($)":         "{:,.2f}",
                "Total Water (m3)":       "{:,.4f}",
                "SLA Violation Rate (%)": "{:.2f}",
                "Total Tasks Deferred":   "{:,}",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Carbon
# ══════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.subheader("Live Carbon Intensity by Datacenter Zone")

    with st.spinner("Fetching carbon intensity data..."):
        ci_data, is_live = fetch_carbon_intensity()

    if is_live:
        st.success("Live data from Electricity Maps API")
    else:
        st.info(
            "Using demo data — set **ELECTRICITY_MAPS_TOKEN** environment variable "
            "to enable live grid data from Electricity Maps."
        )

    greenest_label = min(ci_data, key=ci_data.get)
    greenest_ci = ci_data[greenest_label]
    worst_label = max(ci_data, key=ci_data.get)
    worst_ci = ci_data[worst_label]

    st.markdown(
        f"""
        <div class="gd-callout">
        <strong>Greenest DC right now:</strong> {greenest_label}
        &nbsp;&middot;&nbsp; <strong>{greenest_ci:.0f} gCO\u2082/kWh</strong>
        &nbsp;&middot;&nbsp; {(1 - greenest_ci / worst_ci) * 100:.0f}% cleaner than
        the highest-carbon DC ({worst_label} at {worst_ci:.0f} gCO\u2082/kWh)<br/>
        <span style="opacity:0.8; font-size:0.95rem;">
        Right now, the RL agent would route work to <strong>{greenest_label}</strong>
        to minimize carbon impact.
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ci_df = pd.DataFrame(
        [{"DC": dc, "ci": ci} for dc, ci in ci_data.items()]
    )
    ci_df["is_greenest"] = ci_df["DC"] == greenest_label

    fig_bar = px.bar(
        ci_df,
        x="DC",
        y="ci",
        color="ci",
        color_continuous_scale=["#00C853", "#FFB300", "#FF1744"],
        range_color=[0, max(ci_data.values()) * 1.05],
        text=ci_df["ci"].map(lambda v: f"{v:.0f}"),
        labels={"DC": "Datacenter Zone", "ci": "Carbon Intensity (gCO\u2082/kWh)"},
        title="Current Carbon Intensity per Datacenter Zone",
    )

    greenest_idx = ci_df[ci_df["is_greenest"]].index[0]
    greenest_dc = ci_df.loc[greenest_idx, "DC"]
    fig_bar.add_annotation(
        x=greenest_dc,
        y=greenest_ci + max(ci_data.values()) * 0.04,
        text="Greenest",
        showarrow=False,
        font=dict(size=12, color="#00C853"),
        bgcolor="rgba(0,200,83,0.12)",
        bordercolor="rgba(0,200,83,0.4)",
        borderpad=4,
        borderwidth=1,
        opacity=0.95,
    )

    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(150,150,150,0.15)", title="gCO\u2082/kWh"),
        margin=dict(t=60, b=20),
        height=420,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    cols = st.columns(len(ci_data))
    sorted_dcs = sorted(ci_data.items(), key=lambda x: x[1])
    for col, (dc_label, ci_val) in zip(cols, sorted_dcs):
        col.metric(
            label=dc_label,
            value=f"{ci_val:.0f}",
            delta=f"{ci_val - greenest_ci:+.0f} vs greenest" if dc_label != greenest_label else "Greenest",
            delta_color="inverse",
        )

    st.caption(
        "Data auto-refreshes every 5 minutes. "
        "Lower carbon intensity = cleaner electricity = less CO\u2082 per unit of compute."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Bottom — About
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
with st.expander("About GreenDispatch", expanded=False):
    st.markdown("""
**GreenDispatch** is a research dashboard for carbon-aware AI workload scheduling.
It simulates routing ML training jobs across **5 geographically distributed
datacenters** using a trained Soft Actor-Critic (SAC) RL agent and compares it
against rule-based baselines.

| Strategy | Description |
|---|---|
| **SAC RL Agent** | SAC policy trained to minimise carbon emissions and energy cost by routing jobs to greener datacenters — and optionally deferring jobs to wait for a cleaner grid window |
| **Local Only** | Zero-transfer baseline — every job stays at its origin datacenter |
| **Lowest Carbon** | Greedy rule that always routes to whichever DC currently has the lowest carbon intensity |

**Simulation backend:** [SustainCluster](https://github.com/HewlettPackard/sustain-cluster) —
an open-source multi-datacenter RL environment modelling real carbon intensity,
electricity prices, weather, and Alibaba 2020 workload traces.

**Live carbon data:** [Electricity Maps API](https://api.electricitymap.org)
(set `ELECTRICITY_MAPS_TOKEN` in your environment for live grid data).

**Metrics tracked:** CO\u2082 emissions, energy cost, water usage,
CPU/GPU/MEM utilisation, SLA violation rate, tasks deferred.
    """)
