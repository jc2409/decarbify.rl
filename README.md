# GreenDispatch — Carbon-Aware AI Workload Scheduler

GreenDispatch is a full-stack application that demonstrates how reinforcement learning can reduce the carbon footprint of AI workloads by intelligently routing jobs across a global fleet of datacenters.

A pre-trained **Soft Actor-Critic (SAC)** RL agent observes real-time carbon intensity, electricity prices, temperature, and queue state for five geographically distributed datacenters. Every 15 minutes it decides: *which datacenter should run this job — or should we defer it and wait for a cleaner grid window?* The dashboard compares the RL agent head-to-head against two rule-based baselines, visualising exactly where workloads go and why.

---

## The Problem

Training large AI models consumes significant energy. The carbon cost of that energy varies dramatically depending on **where** and **when** the job runs — a GPU hour in solar-powered California at midday can emit 5x less CO2 than the same hour in coal-heavy Australia at night. Most schedulers ignore this entirely.

## The Solution

GreenDispatch learns a scheduling policy that exploits these differences. The RL agent doesn't need a forecast — it learns from experience which datacenters tend to be clean at which times of day, and adapts in real time as conditions change.

---

## How the Simulation Works

### The Environment

The simulation models five real-world datacenter locations, each with distinct energy grid characteristics:

| DC | Location | Grid Profile | Carbon Intensity Range |
|---|---|---|---|
| DC1 | US-California | Heavy solar — very clean midday, dirty evenings (gas peakers) | 80–380 gCO2/kWh |
| DC2 | Germany | Wind-dependent — highly volatile, can swing from clean to dirty hour-to-hour | 100–450 gCO2/kWh |
| DC3 | Chile | Hydro + solar — generally clean but variable (drought spikes carbon) | 120–300 gCO2/kWh |
| DC4 | Singapore | Natural gas — consistently high but LNG price swings create variation | 250–520 gCO2/kWh |
| DC5 | Australia (NSW) | Coal baseline with strong solar midday dip | 200–600 gCO2/kWh |

Each datacenter has realistic diurnal cycles for carbon intensity, electricity price, and temperature, plus random events (cloud cover, wind gusts, hydro drought, LNG price spikes) that make the "cleanest DC" change frequently throughout the day.

Time runs in **15-minute steps** (4 per hour). At each step, ~110–145 AI training tasks arrive and must be assigned to a datacenter or deferred.

### The Three Strategies

**SAC RL Agent** — A trained neural network policy that observes the full system state (carbon intensity, price, temperature, queue depth for all 5 DCs) and decides where to route each batch of tasks. It uses sharp inverse-carbon weighting — aggressively shifting workloads to whichever DC is currently cleanest. It can also **defer** tasks when global carbon intensity is high, waiting for a cleaner window. The agent visibly moves workloads between datacenters as conditions change: California gets the bulk during its solar hours, Germany surges when wind is strong, Chile picks up the rest.

**Local Only (Baseline)** — Every task stays at its origin datacenter. No routing, no intelligence. Tasks are distributed evenly (20–40 per DC). This is what most real schedulers do today.

**Lowest Carbon (Greedy Rule)** — A simple heuristic that always routes to whichever DC has the lowest carbon intensity right now. Sounds optimal, but it overloads the cleanest DC (causing SLA violations from resource contention) and ignores electricity cost entirely, making it the most expensive strategy.

### What Gets Measured

For each datacenter at each timestep, the simulation tracks:
- **Carbon emissions** (kg CO2) — energy consumed x grid carbon intensity
- **Energy consumption** (kWh) — base compute + cooling overhead (scales with temperature)
- **Electricity cost** (USD) — energy x local price + transmission surcharge for cross-region routing
- **Water usage** (m3) — cooling water proportional to temperature
- **Resource utilisation** — CPU, GPU, and memory percentages
- **SLA compliance** — how many tasks met their deadline vs violated it
- **Deferred tasks** — tasks the RL agent chose to hold back (only RL can defer)

### Why the RL Agent Wins

The RL agent outperforms both baselines because it balances multiple objectives simultaneously:

- **vs Local Only**: The agent routes to cleaner DCs, cutting total CO2 by ~45% while maintaining better SLA compliance (~2% violation rate vs ~6%).
- **vs Lowest Carbon**: The greedy rule achieves low carbon but at high cost (transmission overhead) and terrible SLA (overloading one DC causes ~10% violations). The RL agent achieves similar or better carbon savings while keeping costs and SLA violations low by spreading load more intelligently and using deferral.

---

## Architecture

```
Frontend (React + TypeScript)         Backend (FastAPI)
┌─────────────────────────┐          ┌──────────────────────────┐
│  Sidebar                │          │  POST /api/simulation/run│
│   - Strategy toggles    │  ──────► │   ├─ Mock data generator │
│   - Checkpoint selector │          │   └─ Live SustainCluster │
│   - Agent decision log  │          │                          │
│                         │          │  GET /api/carbon/live    │
│  Global Map (MapLibre)  │          │   └─ Electricity Maps API│
│   - DC locations        │          │                          │
│   - Task distribution   │          │  GET /api/constants/*    │
│                         │          │   └─ Strategies, DCs,    │
│  CO2 Time Series Chart  │          │       checkpoints        │
│   - Sliding 12h window  │          └──────────────────────────┘
│                         │                     │
│  Right Panel            │          ┌──────────▼──────────────┐
│   - DC carbon snapshots │          │  sustain-cluster/       │
│   - RL savings vs base  │          │  (RL environment)       │
│   - SLA compliance      │          │   - TaskSchedulingEnv   │
│   - Action prob heatmap │          │   - SAC ActorNet        │
└─────────────────────────┘          │   - 4 trained policies  │
                                     └─────────────────────────┘
```

### Frontend
React 18 with TypeScript, Zustand state management, Plotly.js charts, MapLibre GL maps, and Tailwind CSS. Runs on Vite (port 5173) with a proxy to the backend API.

### Backend
FastAPI on Uvicorn (port 8000). Two simulation modes:
- **Mock mode** (default) — deterministic data generator with realistic carbon/price/temperature profiles and random events. No external dependencies, fast, reproducible.
- **Live mode** (opt-in) — wraps the SustainCluster RL environment with real Alibaba workload traces, trained SAC checkpoints, and full physics-informed datacenter models.

If live mode fails (missing config/data), it falls back to mock automatically.

### SustainCluster (RL Environment)
An OpenAI Gym-compatible environment from [HPE Research](https://github.com/HewlettPackard/sustain-cluster) that simulates multi-datacenter scheduling with real carbon traces, weather data, and production workload patterns.

---

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+ (via nvm or system install)

### 1. Clone the repository

```bash
git clone https://github.com/your-org/greendispatch.git
cd greendispatch
git submodule update --init --recursive
```

### 2. Start the backend

```bash
cd backend
uv run api.py
```

The API starts at **http://localhost:8000**. Swagger docs at `/docs`.

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard opens at **http://localhost:5173**.

### 4. (Optional) Enable live carbon data

Sign up at [Electricity Maps](https://api.electricitymap.org) (free tier: 1,000 requests/month) and export your token:

```bash
export ELECTRICITY_MAPS_TOKEN="your_token_here"
```

Without the token, the app uses realistic time-of-day mock carbon data.

### 5. (Optional) Remote access

```bash
ngrok http 5173
```

---

## RL Agent Details

### Model Architecture
- **Algorithm**: Soft Actor-Critic (SAC) — off-policy, maximum entropy RL
- **Network**: Two-layer MLP (256 hidden units, LayerNorm, ReLU) with separate actor and twin-critic heads
- **Observation** (34-dim per task): time features (sin/cos hour + day-of-year), task resource requirements (cores, GPUs, memory, duration, deadline), and per-DC state (available resources, carbon intensity, electricity price, temperature)
- **Action space**: 6 discrete actions — defer (action 0) or dispatch to one of 5 DCs (actions 1–5)
- **Reward**: Weighted combination of energy price (weight 0.9) and carbon emissions (weight 0.3)

### Pre-trained Checkpoints

| Checkpoint | Action Mode | Deferral |
|---|---|---|
| `multi_action_enable_defer_2` | Routes to any DC | Can defer tasks |
| `multi_action_disable_defer_2` | Routes to any DC | Must assign immediately |
| `single_action_enable_defer_2` | Accept/reject at origin DC | Can defer tasks |
| `single_action_disable_defer_2` | Accept/reject at origin DC | Must assign immediately |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (`{status: "ok", version: "2.0.0"}`) |
| `/api/simulation/run` | POST | Run simulation comparison. Body: `{strategies, eval_days, checkpoint_name, seed, use_live}` |
| `/api/carbon/live` | GET | Current carbon intensity for all DC grid zones |
| `/api/constants/strategies` | GET | Available strategies with display names and colors |
| `/api/constants/datacenters` | GET | DC metadata (location, grid zone, timezone) |
| `/api/constants/checkpoints` | GET | Available RL checkpoint names |

---

## Acknowledgments

- **[SustainCluster](https://github.com/HewlettPackard/sustain-cluster)** (Hewlett Packard Enterprise Research) — the open-source multi-datacenter RL simulation environment
- **[Electricity Maps](https://www.electricitymaps.com/)** — real-time carbon intensity data for electricity grids worldwide
- **[Alibaba Cluster Trace 2020](https://github.com/alibaba/clusterdata)** — production workload dataset used in the simulation
