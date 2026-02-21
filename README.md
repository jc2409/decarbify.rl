# 🌍 GreenDispatch — Carbon-Aware AI Workload Scheduler

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.54-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

GreenDispatch is an interactive research dashboard that demonstrates **carbon-aware
scheduling of AI/ML workloads** across a simulated global datacenter fleet.
It trains a [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) reinforcement
learning agent to route ML training jobs to whichever datacenter currently has the
cheapest, cleanest electricity — and optionally defer jobs to wait for a greener
grid window — then compares the agent head-to-head against two rule-based baselines
using a realistic simulation of carbon intensity, electricity prices, weather, and
Alibaba 2020 production workload traces.
A live carbon overlay fetches real-time grid data from the
[Electricity Maps API](https://api.electricitymap.org) so the dashboard always
reflects today's actual grid conditions.

![Dashboard Screenshot](docs/screenshot.png)

---

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit App (app.py)                   │
│                                                                 │
│  Sidebar config → run_simulation() ──► session_state results   │
│                                                                 │
│  📊 Summary │ 🗺️ Map │ 📈 Time Series │ ⚖️ Trade-offs │ 🌐 Live  │
└──────────────────────┬──────────────────────┬───────────────────┘
                       │                      │
        ┌──────────────▼──────────┐  ┌────────▼──────────────────┐
        │  backend/simulator.py   │  │  backend/carbon_api.py    │
        │  SustainClusterSimulator│  │  get_live_carbon_intensity│
        └──────────────┬──────────┘  └────────┬──────────────────┘
                       │                      │
        ┌──────────────▼──────────┐  ┌────────▼──────────────────┐
        │  sustain-cluster/       │  │  Electricity Maps API     │
        │  TaskSchedulingEnv      │  │  (or time-aware mock)     │
        │  DatacenterClusterMgr   │  └───────────────────────────┘
        │  SAC ActorNet           │
        └─────────────────────────┘
```

### Simulation backend — SustainCluster

[SustainCluster](https://github.com/HewlettPackard/sustain-cluster) (HPE Research)
is a multi-datacenter RL gym environment that simulates five globally distributed
datacenters with real carbon intensity traces, electricity price curves, weather data,
and a full 2020 Alibaba production workload. Each 15-minute timestep, arriving tasks
are routed (or deferred) according to the active strategy.

### RL agent vs baselines

| Strategy | How it works |
|---|---|
| 🤖 **SAC RL Agent** | Pre-trained SAC policy that observes per-DC carbon intensity, electricity price, temperature, and queue state to pick the optimal DC (or defer the job to a cleaner window) |
| 📍 **Local Only** | Zero-transfer baseline — every job stays at its origin datacenter |
| 🍃 **Lowest Carbon** | Greedy rule that routes each job to whichever DC has the lowest current carbon intensity |

### Live carbon data overlay

`backend/carbon_api.py` calls the Electricity Maps API to fetch the **current**
carbon intensity (gCO₂/kWh) for each datacenter's grid zone. If no API key is
configured, realistic mock data with time-of-day solar variation is used instead.
Results are cached for five minutes so the dashboard stays responsive.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/greendispatch.git
cd greendispatch
```

### 2. Clone SustainCluster and install its dependencies

```bash
git clone https://github.com/HewlettPackard/sustain-cluster.git
pip install -r requirements.txt
# SustainCluster also needs its own deps:
pip install -r sustain-cluster/requirements.txt   # if present
```

### 3. (Optional) Get a free Electricity Maps API key

1. Sign up at **https://api.electricitymap.org** — the free tier covers 1,000 requests/month
2. Export your token:

```bash
export ELECTRICITY_MAPS_TOKEN="your_token_here"
```

Without the token the app runs in **demo mode** with realistic mock data and a
visible notice in the Live Carbon tab.

### 4. Run the app

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

### 5. Remote / cloud access (optional)

If running on a remote server, expose the app with ngrok:

```bash
# Download ngrok (Linux x86_64)
curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz -C /tmp
/tmp/ngrok config add-authtoken <your_ngrok_token>
/tmp/ngrok http 8501
```

---

## 📊 What It Shows

| Tab | Description |
|---|---|
| **📊 Summary** | Comparison table of all strategies with green-gradient heatmap on CO₂, energy, and cost. KPI metric cards show how much CO₂ and money the RL agent saves vs the Local Only baseline. |
| **🗺️ Map** | World map of all 5 datacenters rendered with `pydeck`. Circle size and green→red colour encode average carbon intensity under the selected strategy. Hover for per-DC stats. |
| **📈 Time Series** | Side-by-side Plotly line charts of total CO₂ emissions and energy cost over every 15-minute timestep, plus a per-datacenter carbon intensity chart showing the raw environmental signal. |
| **⚖️ Trade-offs** | Bubble scatter plot — X = total cost, Y = total CO₂, bubble size = total energy — one point per strategy. Reveals the frontier between carbon savings and SLA compliance. |
| **🌐 Live Carbon** | Real-time (or mock) carbon intensity bar chart for all 5 grid zones, updated every 5 minutes. Calls out the greenest DC and states where the RL agent would route work right now. |

---

## 🧠 Technical Details

### SAC RL Agent

The agent is a **Soft Actor-Critic** policy (`rl_components/agent_net.py`) trained
with the SustainCluster multi-datacenter environment:

- **Observation space** (per task, 34-dim): time features (sin/cos hour + day-of-year),
  task resource requirements (cores, GPUs, memory, duration, deadline), and per-DC
  state (available resources, carbon intensity, electricity price, temperature).
- **Action space**: assign task to one of 5 DCs, or defer (action 0) to wait for a
  cleaner window.
- **Reward**: weighted combination of energy price (weight 0.9) and carbon emissions
  (weight 0.3) — no reward normalisation at evaluation time.
- **Architecture**: two-layer MLP (hidden dim 256, LayerNorm, ReLU) with separate
  actor and twin-critic heads.
- Four pre-trained checkpoints are included, covering multi-/single-action mode and
  defer-enabled/disabled variants.

### Datacenters

| DC | Location | Grid Zone | Notable characteristic |
|---|---|---|---|
| DC1 | US-California | US-CAL-CISO | Moderate CI, significant solar |
| DC2 | Germany | DE | Coal + growing renewables |
| DC3 | Chile | CL-SEN | Hydro-heavy, often cleanest |
| DC4 | Singapore | SG | Natural-gas dominated |
| DC5 | Australia (NSW) | AU-NSW | Coal-heavy, typically highest CI |

### Metrics tracked

`carbon_kg` · `energy_cost_usd` · `energy_kwh` · `water_usage_m3` ·
`cpu_util_pct` · `gpu_util_pct` · `mem_util_pct` ·
`sla_met_count` · `sla_violated_count` · `tasks_assigned_count` ·
`carbon_intensity_gco2_kwh` · `price_per_kwh` · `temperature_c`

---

## 🏆 Hackathon Context

GreenDispatch was built for the **Sustainability Track** of the hackathon, sponsored
by **Crusoe**. The track challenge: *demonstrate how AI systems can be made
dramatically more carbon-efficient without sacrificing performance*.

Crusoe operates datacenters powered by otherwise-flared or stranded renewable energy,
making geographic and temporal job routing a first-class sustainability lever.
GreenDispatch addresses this directly by showing that a lightweight SAC agent —
without any oracle knowledge of future grid conditions — can cut CO₂ emissions
by ~5% and energy cost by ~4% over a naive local-only baseline, purely by learning
which datacenters are cheapest and cleanest at each hour of the day.

---

## 🔮 Future Work

- **Real cloud provider integration** — replace the SustainCluster simulator with
  live scheduling hooks into AWS, GCP, or Azure spot-instance APIs; route actual
  container jobs based on real-time carbon signals.
- **Custom model training UI** — expose SustainCluster's training loop in the
  dashboard so users can retrain the RL agent with custom reward weights (e.g.,
  prioritise SLA over carbon, or water over cost).
- **Job queue API** — add a REST endpoint (`/submit`) so external MLOps pipelines
  (Argo Workflows, Kubeflow) can query GreenDispatch for the optimal datacenter
  before launching a training run.
- **Multi-objective RL with user-defined weights** — let users drag sliders for
  carbon weight, cost weight, and SLA penalty, then instantly rerun inference with
  a conditioned policy (e.g., using preference-conditioned SAC or Pareto-optimal
  policy sets).
- **Temporal carbon forecasting** — integrate day-ahead carbon intensity forecasts
  to enable smarter deferral decisions (defer now if the grid will be 40% cleaner
  in 3 hours).

---

## 🙏 Acknowledgments

- **[SustainCluster](https://github.com/HewlettPackard/sustain-cluster)** (Hewlett Packard Enterprise Research) —
  the open-source multi-datacenter RL simulation environment powering the backend.
- **[Electricity Maps](https://www.electricitymaps.com/)** — real-time and historical
  carbon intensity data for electricity grids worldwide.
- **[Crusoe](https://crusoe.ai/)** — sustainable cloud computing infrastructure and
  hackathon track sponsor; their mission of eliminating energy waste in AI compute
  directly inspired this project.
- **[Alibaba Cluster Trace 2020](https://github.com/alibaba/clusterdata)** — the
  production workload dataset used in the simulation.
