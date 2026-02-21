"""
GreenDispatch — Carbon-Aware AI Workload Scheduler Dashboard
"""

import os
import sys

# Ensure sustain-cluster packages (envs, simulation, etc.) are importable.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_APP_DIR, "sustain-cluster"))

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from backend.carbon_api import get_live_carbon_intensity
from backend.simulator import SustainClusterSimulator

# ── Constants ──────────────────────────────────────────────────────────────────

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
    "DC1": {"name": "US-California", "lat":  37.39, "lon": -122.08},
    "DC2": {"name": "Germany",       "lat":  50.11, "lon":    8.68},
    "DC3": {"name": "Chile",         "lat": -33.45, "lon":  -70.67},
    "DC4": {"name": "Singapore",     "lat":   1.35, "lon":  103.82},
    "DC5": {"name": "Australia",     "lat": -33.87, "lon":  151.21},
}

# Teal, coral, mint, amber, violet — consistent across every chart
_PALETTE = ["#00C49F", "#FF6B6B", "#4ECDC4", "#FFD166", "#A78BFA"]

_COLOR_MAP: dict[str, str] = {
    "SAC RL Agent (Geo+Time)": _PALETTE[0],
    "Local Only (Baseline)":   _PALETTE[1],
    "Lowest Carbon":           _PALETTE[2],
}

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GreenDispatch 🌍",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — green theme ───────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Metric cards: green left border + subtle tinted background ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(
            135deg,
            rgba(0, 196, 159, 0.10) 0%,
            rgba(0, 196, 159, 0.03) 100%
        );
        border: 1px solid rgba(0, 196, 159, 0.30);
        border-left: 3px solid #00C49F;
        border-radius: 8px;
        padding: 14px 18px 10px 18px;
    }

    /* Metric value in brand teal */
    div[data-testid="stMetricValue"] > div {
        color: #00C49F !important;
        font-weight: 700 !important;
    }

    /* Label — slightly muted */
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.82rem !important;
        opacity: 0.85;
    }

    /* ── Sidebar: subtle right border accent ── */
    section[data-testid="stSidebar"] {
        border-right: 2px solid rgba(0, 196, 159, 0.25);
    }

    /* ── Active tab indicator ── */
    .stTabs [aria-selected="true"] {
        color: #00C49F !important;
        border-bottom-color: #00C49F !important;
        font-weight: 600;
    }

    /* ── Live carbon: callout box ── */
    .gd-callout {
        background: linear-gradient(
            135deg,
            rgba(0, 196, 159, 0.15) 0%,
            rgba(0, 196, 159, 0.04) 100%
        );
        border: 1px solid rgba(0, 196, 159, 0.40);
        border-radius: 10px;
        padding: 16px 22px;
        margin: 12px 0 18px 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached runners ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_simulation(
    strategies_tuple: tuple,
    eval_days: int,
    checkpoint_name: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run SustainCluster comparison; returns (per_dc_df, global_df, summary_df)."""
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
    return get_live_carbon_intensity()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚡ Job Configuration")

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
    run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    st.divider()
    st.caption("🌐 Live Carbon tab refreshes automatically every 5 min.")

# ── Main header ────────────────────────────────────────────────────────────────

st.title("GreenDispatch — Carbon-Aware AI Workload Scheduler")
st.caption(
    "Route ML training jobs across 5 global datacenters using an RL agent or "
    "rule-based strategies, and compare carbon emissions, energy cost, and SLA violations."
)

# ── Trigger simulation ─────────────────────────────────────────────────────────

if run_btn:
    if not selected_labels:
        st.error("Please select at least one strategy.")
    else:
        strategies = [_STRATEGY_INTERNAL[s] for s in selected_labels]
        with st.spinner("Running simulation… this may take a minute."):
            per_dc_df, global_df, summary_df = run_simulation(
                strategies_tuple=tuple(strategies),
                eval_days=eval_days,
                checkpoint_name=checkpoint_name,
            )
        st.session_state.update(
            per_dc_df=per_dc_df,
            global_df=global_df,
            summary_df=summary_df,
            strategies_run=strategies,
        )

# ── Five tabs — Live Carbon is always available ────────────────────────────────

has_results = "summary_df" in st.session_state

tab_summary, tab_map, tab_ts, tab_tradeoff, tab_live = st.tabs([
    "📊 Summary", "🗺️ Map", "📈 Time Series", "⚖️ Trade-offs", "🌐 Live Carbon",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Summary
# ══════════════════════════════════════════════════════════════════════════════

with tab_summary:
    if not has_results:
        st.info("👈 Configure your simulation in the sidebar and click **🚀 Run Simulation** to begin.")
    else:
        per_dc_df    = st.session_state.per_dc_df
        summary_df   = st.session_state.summary_df
        strategies_run = st.session_state.strategies_run

        st.subheader("Strategy Comparison")

        disp = summary_df.drop(columns=["controller_label"]).copy()
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

        # ── KPI cards: RL vs Local Only ──
        rl_row = summary_df[summary_df["Controller"] == "manual_rl"]
        bl_row = summary_df[summary_df["Controller"] == "local_only"]

        if not rl_row.empty and not bl_row.empty:
            st.divider()
            st.subheader("🤖 RL Agent vs 📍 Local Only Baseline")
            rl, bl = rl_row.iloc[0], bl_row.iloc[0]

            c1, c2, c3, c4 = st.columns(4)

            # CO2 saved: positive = RL is greener → green arrow (delta_color="normal")
            co2_saved = bl["Total CO2 (kg)"] - rl["Total CO2 (kg)"]
            c1.metric(
                "CO₂ Saved (kg)",
                f"{co2_saved:,.1f}",
                delta=f"{co2_saved / bl['Total CO2 (kg)'] * 100:.1f}% vs baseline",
                delta_color="normal",
                help="How much less CO₂ the RL agent emits compared to Local Only.",
            )

            # Cost saved: positive = RL is cheaper → green arrow
            cost_saved = bl["Total Cost ($)"] - rl["Total Cost ($)"]
            c2.metric(
                "Cost Saved ($)",
                f"${cost_saved:,.2f}",
                delta=f"{cost_saved / bl['Total Cost ($)'] * 100:.1f}% vs baseline",
                delta_color="normal",
                help="Energy cost reduction achieved by the RL agent.",
            )

            # Energy delta: negative = RL uses less → show inverse (green if neg)
            energy_delta = rl["Total Energy (kWh)"] - bl["Total Energy (kWh)"]
            c3.metric(
                "Energy Delta (kWh)",
                f"{energy_delta:+,.1f}",
                delta=f"{energy_delta / bl['Total Energy (kWh)'] * 100:.1f}%",
                delta_color="inverse",
                help="Negative means the RL agent uses less total energy.",
            )

            # SLA violation rate: lower is better → inverse colour
            sla_delta = rl["SLA Violation Rate (%)"] - bl["SLA Violation Rate (%)"]
            c4.metric(
                "RL SLA Violation Rate",
                f"{rl['SLA Violation Rate (%)']:.2f}%",
                delta=f"{sla_delta:+.2f}% vs baseline",
                delta_color="inverse",
                help="RL may defer tasks to greener windows, increasing violations.",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Map
# ══════════════════════════════════════════════════════════════════════════════

with tab_map:
    if not has_results:
        st.info("👈 Run a simulation to populate the map with per-DC metrics.")
    else:
        per_dc_df      = st.session_state.per_dc_df
        strategies_run = st.session_state.strategies_run

        st.subheader("🌍 Global Datacenter Locations")
        first_strategy = strategies_run[0]
        first_label    = _STRATEGY_DISPLAY[first_strategy]

        map_rows = []
        for dc_id, dc_meta in _DC_LOCATIONS.items():
            dc_slice = per_dc_df[
                (per_dc_df["datacenter"] == dc_id) &
                (per_dc_df["controller"]  == first_strategy)
            ]
            avg_ci       = float(dc_slice["carbon_intensity_gco2_kwh"].mean()) if not dc_slice.empty else 0.0
            total_tasks  = int(dc_slice["tasks_assigned_count"].sum())          if not dc_slice.empty else 0
            total_energy = float(dc_slice["energy_kwh"].sum())                  if not dc_slice.empty else 0.0
            map_rows.append({
                "dc":           dc_id,
                "location":     dc_meta["name"],
                "lat":          dc_meta["lat"],
                "lon":          dc_meta["lon"],
                "avg_ci":       round(avg_ci, 1),
                "total_tasks":  total_tasks,
                "total_energy": round(total_energy, 1),
                "radius":       max(250_000, int(avg_ci * 600)),
            })

        map_df = pd.DataFrame(map_rows)

        # Green (low CI) → Red (high CI) colour ramp
        ci_min = map_df["avg_ci"].min()
        ci_max = max(map_df["avg_ci"].max(), ci_min + 1.0)
        norm   = (map_df["avg_ci"] - ci_min) / (ci_max - ci_min)
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

        # DC label overlay — slightly offset above each circle
        label_df = map_df.copy()
        label_df["label_lat"] = label_df["lat"] + 4.5   # nudge north
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

        st.pydeck_chart(
            pdk.Deck(
                layers=[scatter_layer, text_layer],
                initial_view_state=pdk.ViewState(
                    latitude=15, longitude=15, zoom=1.2, pitch=0,
                ),
                tooltip={
                    "html": (
                        "<b style='font-size:14px'>{dc} — {location}</b><br/>"
                        "Avg Carbon Intensity: <b>{avg_ci}</b> gCO₂/kWh<br/>"
                        "Tasks Assigned: <b>{total_tasks}</b><br/>"
                        "Total Energy: <b>{total_energy}</b> kWh"
                    ),
                    "style": {
                        "backgroundColor": "#1a1a2e",
                        "color": "white",
                        "fontSize": "13px",
                        "borderRadius": "6px",
                        "padding": "10px 14px",
                        "border": "1px solid rgba(0,196,159,0.4)",
                    },
                },
                map_style=pdk.map_styles.CARTO_DARK,
            ),
            use_container_width=True,
            height=460,
        )

        st.caption(
            f"Circle size and colour reflect average carbon intensity "
            f"(🟢 low → 🔴 high) under the **{first_label}** strategy. "
            "Hover a circle for details."
        )

        st.dataframe(
            map_df[["dc", "location", "avg_ci", "total_tasks", "total_energy"]].rename(columns={
                "dc":           "DC",
                "location":     "Location",
                "avg_ci":       "Avg CI (gCO₂/kWh)",
                "total_tasks":  "Tasks Assigned",
                "total_energy": "Total Energy (kWh)",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Time Series
# ══════════════════════════════════════════════════════════════════════════════

with tab_ts:
    if not has_results:
        st.info("👈 Run a simulation to see time-series charts.")
    else:
        per_dc_df      = st.session_state.per_dc_df
        strategies_run = st.session_state.strategies_run

        agg = (
            per_dc_df
            .groupby(["timestep", "controller", "controller_label"], as_index=False)
            .agg(
                carbon_kg=("carbon_kg", "sum"),
                energy_cost_usd=("energy_cost_usd", "sum"),
            )
        )

        c_l, c_r = st.columns(2)

        _chart_layout = dict(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, b=10),
            xaxis=dict(gridcolor="rgba(150,150,150,0.15)"),
            yaxis=dict(gridcolor="rgba(150,150,150,0.15)"),
        )

        with c_l:
            st.subheader("Total CO₂ Emissions Over Time")
            fig_co2 = px.line(
                agg, x="timestep", y="carbon_kg",
                color="controller_label",
                color_discrete_map=_COLOR_MAP,
                labels={
                    "timestep":         "Timestep (15-min intervals)",
                    "carbon_kg":        "CO₂ Emissions (kg)",
                    "controller_label": "Strategy",
                },
            )
            fig_co2.update_layout(**_chart_layout)
            st.plotly_chart(fig_co2, use_container_width=True)

        with c_r:
            st.subheader("Total Energy Cost Over Time")
            fig_cost = px.line(
                agg, x="timestep", y="energy_cost_usd",
                color="controller_label",
                color_discrete_map=_COLOR_MAP,
                labels={
                    "timestep":         "Timestep (15-min intervals)",
                    "energy_cost_usd":  "Energy Cost (USD)",
                    "controller_label": "Strategy",
                },
            )
            fig_cost.update_layout(**_chart_layout)
            st.plotly_chart(fig_cost, use_container_width=True)

        st.subheader("Carbon Intensity per Datacenter (Environmental Signal)")
        first_strategy = strategies_run[0]
        ci_df = per_dc_df[per_dc_df["controller"] == first_strategy][
            ["timestep", "datacenter", "carbon_intensity_gco2_kwh"]
        ].copy()

        fig_ci = px.line(
            ci_df, x="timestep", y="carbon_intensity_gco2_kwh",
            color="datacenter",
            color_discrete_sequence=_PALETTE,
            labels={
                "timestep":                  "Timestep",
                "carbon_intensity_gco2_kwh": "Carbon Intensity (gCO₂/kWh)",
                "datacenter":                "Datacenter",
            },
        )
        fig_ci.update_layout(**_chart_layout)
        st.plotly_chart(fig_ci, use_container_width=True)
        st.caption(
            "Carbon intensity is grid data — identical across all strategies. "
            f"Displayed here for **{_STRATEGY_DISPLAY[first_strategy]}**."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Trade-offs
# ══════════════════════════════════════════════════════════════════════════════

with tab_tradeoff:
    if not has_results:
        st.info("👈 Run a simulation to see the cost-vs-carbon trade-off chart.")
    else:
        summary_df = st.session_state.summary_df

        st.subheader("Cost vs Carbon Trade-off")
        st.caption("Each point is one strategy. Bubble size = total energy (kWh). Lower-left is optimal.")

        tdf = summary_df.copy()
        fig_scatter = px.scatter(
            tdf,
            x="Total Cost ($)",
            y="Total CO2 (kg)",
            size="Total Energy (kWh)",
            color="controller_label",
            text="controller_label",
            color_discrete_map=_COLOR_MAP,
            size_max=72,
            labels={
                "Total Cost ($)":     "Total Energy Cost (USD)",
                "Total CO2 (kg)":     "Total CO₂ Emissions (kg)",
                "controller_label":   "Strategy",
                "Total Energy (kWh)": "Total Energy (kWh)",
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
            text="← Lower cost & lower carbon is better",
            xref="paper", yref="paper",
            x=0.01, y=0.03,
            showarrow=False,
            font=dict(size=11, color="rgba(180,180,180,0.75)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

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
    st.subheader("🌐 Live Carbon Intensity by Datacenter Zone")

    with st.spinner("Fetching carbon intensity data…"):
        ci_data, is_live = fetch_carbon_intensity()

    # ── Data source banner ──
    if is_live:
        st.success("✅ Live data from Electricity Maps API", icon="📡")
    else:
        st.info(
            "📊 Using demo data — set the **ELECTRICITY_MAPS_TOKEN** environment variable "
            "to enable live grid data from Electricity Maps.",
            icon="ℹ️",
        )

    # ── Greenest DC callout ──
    greenest_label = min(ci_data, key=ci_data.get)
    greenest_ci    = ci_data[greenest_label]
    worst_label    = max(ci_data, key=ci_data.get)
    worst_ci       = ci_data[worst_label]

    st.markdown(
        f"""
        <div class="gd-callout">
        🟢 <strong>Greenest DC right now:</strong> {greenest_label}
        &nbsp;·&nbsp; <strong>{greenest_ci:.0f} gCO₂/kWh</strong>
        &nbsp;·&nbsp; {(1 - greenest_ci / worst_ci) * 100:.0f}% cleaner than
        the highest-carbon DC ({worst_label} at {worst_ci:.0f} gCO₂/kWh)<br/>
        <span style="opacity:0.8; font-size:0.95rem;">
        ⚡ Right now, the RL agent would route work to <strong>{greenest_label}</strong>
        to minimize carbon impact.
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Build DataFrame ──
    ci_df = pd.DataFrame(
        [{"DC": dc, "ci": ci} for dc, ci in ci_data.items()]
    )
    ci_df["is_greenest"] = ci_df["DC"] == greenest_label

    # ── Bar chart — continuous green→red scale ──
    fig_bar = px.bar(
        ci_df,
        x="DC",
        y="ci",
        color="ci",
        color_continuous_scale=["#00C49F", "#FFD166", "#FF6B6B"],
        range_color=[0, max(ci_data.values()) * 1.05],
        text=ci_df["ci"].map(lambda v: f"{v:.0f}"),
        labels={
            "DC":  "Datacenter Zone",
            "ci":  "Carbon Intensity (gCO₂/kWh)",
        },
        title="Current Carbon Intensity per Datacenter Zone",
    )

    # Highlight the greenest bar with a star annotation
    greenest_idx = ci_df[ci_df["is_greenest"]].index[0]
    greenest_dc  = ci_df.loc[greenest_idx, "DC"]
    fig_bar.add_annotation(
        x=greenest_dc,
        y=greenest_ci + max(ci_data.values()) * 0.04,
        text="🏆 Greenest",
        showarrow=False,
        font=dict(size=12, color="#00C49F"),
        bgcolor="rgba(0,196,159,0.12)",
        bordercolor="rgba(0,196,159,0.4)",
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
        yaxis=dict(gridcolor="rgba(150,150,150,0.15)", title="gCO₂/kWh"),
        margin=dict(t=60, b=20),
        height=420,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Per-DC metrics row ──
    cols = st.columns(len(ci_data))
    sorted_dcs = sorted(ci_data.items(), key=lambda x: x[1])
    for col, (dc_label, ci_val) in zip(cols, sorted_dcs):
        icon = "🏆" if dc_label == greenest_label else "🔴" if ci_val == worst_ci else "⚡"
        col.metric(
            label=f"{icon} {dc_label}",
            value=f"{ci_val:.0f}",
            delta=f"{ci_val - greenest_ci:+.0f} vs greenest" if dc_label != greenest_label else "Greenest ✓",
            delta_color="inverse",
            help=f"Carbon intensity in gCO₂/kWh for {dc_label}",
        )

    st.caption(
        "Data auto-refreshes every 5 minutes. "
        "Lower carbon intensity = cleaner electricity = less CO₂ per unit of compute."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Bottom — About expander (always visible)
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
with st.expander("ℹ️ About GreenDispatch", expanded=False):
    st.markdown(
        """
**GreenDispatch** is a research dashboard for carbon-aware AI workload scheduling.
It simulates routing ML training jobs across **5 geographically distributed
datacenters** using a trained Soft Actor-Critic (SAC) RL agent and compares it
against rule-based baselines.

| Strategy | Description |
|---|---|
| 🤖 **SAC RL Agent** | SAC policy trained to minimise carbon emissions and energy cost by routing jobs to greener datacenters—and optionally deferring jobs to wait for a cleaner grid window |
| 📍 **Local Only** | Zero-transfer baseline — every job stays at its origin datacenter |
| 🍃 **Lowest Carbon** | Greedy rule that always routes to whichever DC currently has the lowest carbon intensity |

**Simulation backend:** [SustainCluster](https://github.com/HewlettPackard/sustain-cluster) —
an open-source multi-datacenter RL environment modelling real carbon intensity,
electricity prices, weather, and Alibaba 2020 workload traces.

**Live carbon data:** [Electricity Maps API](https://api.electricitymap.org)
(set `ELECTRICITY_MAPS_TOKEN` in your environment for live grid data).

**Metrics tracked:** CO₂ emissions · energy cost · water usage ·
CPU/GPU/MEM utilisation · SLA violation rate · tasks deferred.
        """
    )
