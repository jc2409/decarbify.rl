# Backend Inference Pipeline — Implementation Plan

## Overview

Two new backend features to power the hackathon web app demo:

1. **Inference Engine** — Runs the RL actor network step-by-step on a sequence of observations,
   producing per-step decisions with full reasoning breakdown (which DC, why, confidence scores).

2. **Performance Tracker** — Produces 5 parallel time series (one per datacenter) comparing
   each DC's **local-only** performance against the **RL-optimal** performance, so the frontend
   can show how much each DC benefits from smart routing.

Both modules are self-contained additions to `backend/`. Nothing in the existing files
(`simulator.py`, `carbon_api.py`) needs to be modified.

---

## What Already Exists (Do Not Modify)

| File | Role | Status |
|---|---|---|
| `backend/simulator.py` | Bulk comparison runner (runs full episodes, returns DataFrames) | Keep as-is |
| `backend/carbon_api.py` | Live / mock carbon intensity for the Live Carbon tab | Keep as-is |
| `backend/__init__.py` | Empty package marker | Keep as-is |

The new modules sit alongside these and are independent.

---

## New Files to Create

```
backend/
├── mock_data.py            ← Generates mock sequential input data (288 timesteps = 3 days)
├── inference_engine.py     ← Runs actor on sequence, emits decisions + reasoning
├── performance_tracker.py  ← Produces local vs optimal time series per DC
└── PLAN_INFERENCE_PIPELINE.md  ← This file
```

---

## Feature 1 — Inference Engine (`backend/inference_engine.py`)

### Purpose

Given a **sequence of observations** (one per 15-minute timestep), run the SAC actor
network forward and return, for each step:
- Which DC was chosen (or deferred)
- Softmax confidence scores for all 6 actions
- A human-readable reasoning breakdown explaining the decision in terms of the raw input factors

This is what powers the "Decision Theater" card in the frontend.

### Inputs

A list of observation dictionaries, one per timestep:

```python
# Each element in the input sequence:
{
  "timestep": 0,
  "timestamp": "2025-06-15T00:00:00Z",   # any date — see note below

  # Task arriving at this timestep
  "task": {
    "cpu_cores_norm":     0.0625,   # raw / DC_MAX_CPU  (e.g. 4/64 = 0.0625)
    "gpu_cores_norm":     0.25,     # raw / DC_MAX_GPU  (e.g. 2/8  = 0.25)
    "memory_gb_norm":     0.125,    # raw / DC_MAX_MEM  (e.g. 16/128 = 0.125)
    "duration_hours":     0.5,      # raw hours (normalised /10 before encoding)
    "deadline_hours":     2.0,      # raw hours (normalised /24 before encoding)
    "origin_dc":          "DC1"     # used for local-vs-optimal comparison only, not in obs vector
  },

  # Per-datacenter state — all resource values are UTILISATION RATIOS (0.0–1.0)
  # NOT absolute counts. Real DCs have thousands of GPUs; the model only ever
  # sees the fraction that is currently free.
  "datacenters": {
    "DC1": {
      "avail_cpu_ratio":            0.62,   # available / total CPU
      "avail_gpu_ratio":            0.50,   # available / total GPU
      "avail_mem_ratio":            0.55,   # available / total memory
      "carbon_intensity_gco2_kwh":  230.5,  # raw gCO₂/kWh (normalised /800 before encoding)
      "price_per_kwh":              0.124   # raw $/kWh   (normalised /0.30 before encoding)
    },
    "DC2": { ... },
    "DC3": { ... },
    "DC4": { ... },
    "DC5": { ... }
  }
}
```

**Note on resource ratios**: Real hyperscale datacenters have thousands of GPUs
(e.g. DC1 might have 2,000 GPUs with 1,000 currently free → ratio = 0.50). The model
never sees the absolute count — only the fraction available. Mock data should generate
ratios directly in the 0.0–1.0 range; the absolute scale is irrelevant to the model.

**Note on timestamp / start date**: Any date works for mock data. The model only
extracts hour-of-day and day-of-year to encode daily and seasonal patterns (solar at noon,
summer vs winter). The 2020 constraint applies only to the real SustainCluster environment,
which is tied to the Alibaba 2020 workload trace and historical carbon/price traces.
For mock and demo purposes, use any recent date (e.g. today).

These map directly to the 34-dim observation vector the actor expects:
- 4 time features (sin/cos of hour + day-of-year, derived from `timestamp`)
- 5 task features (from `task`)
- 5 × 5 DC features (from `datacenters`) = 25 dims

### Outputs

A list of result dictionaries, one per timestep:

```python
{
  "timestep": 0,
  "timestamp": "2025-06-15T00:00:00Z",

  "decision": {
    "action_index":  3,           # 0 = defer, 1-5 = DC1-DC5
    "dc_chosen":     "DC3",       # "DEFER" if action_index == 0
    "was_deferred":  False,

    # Softmax probabilities over all 6 actions — sums to 1.0.
    # These come from the actual model forward pass and reflect ALL 34 inputs.
    "confidence_scores": {
      "DEFER": 0.02,
      "DC1":   0.15,
      "DC2":   0.05,
      "DC3":   0.68,
      "DC4":   0.06,
      "DC5":   0.04
    },

    # Human-readable post-hoc interpretation of the decision.
    # This is deterministic Python — NOT a re-computation of the model's decision.
    "reasoning": {
      "summary": "DC3 chosen: cleanest grid (185 gCO₂/kWh, 48% below cluster avg). Cost also lowest ($0.05/kWh).",
      "primary_factor": "carbon_intensity",   # carbon_intensity | price | capacity | defer_wait
      "defer_eligible":  False,               # False if deadline < 1h
      "solar_window":    False,               # True if 10–16 UTC for DC1/DC3
      "dc_analysis": {
        # For each DC: rankings + capacity status + raw factor values
        "DC1": {
          "carbon_rank":        2,
          "cost_rank":          3,
          "capacity_rank":      2,
          "capacity_blocked":   False,
          "solar_window":       False,
          "carbon_intensity":   230.5,
          "price_per_kwh":      0.124,
          "avail_gpu_ratio":    0.50
        },
        "DC2": {
          "carbon_rank":        3,
          "cost_rank":          4,
          "capacity_rank":      1,
          "capacity_blocked":   False,
          "solar_window":       False,
          "carbon_intensity":   340.0,
          "price_per_kwh":      0.180,
          "avail_gpu_ratio":    0.80
        },
        "DC3": {
          "carbon_rank":        1,
          "cost_rank":          1,
          "capacity_rank":      3,
          "capacity_blocked":   False,
          "solar_window":       False,
          "carbon_intensity":   185.0,
          "price_per_kwh":      0.050,
          "avail_gpu_ratio":    0.40
        },
        "DC4": {
          "carbon_rank":        4,
          "cost_rank":          5,
          "capacity_rank":      4,
          "capacity_blocked":   False,
          "solar_window":       False,
          "carbon_intensity":   455.0,
          "price_per_kwh":      0.160,
          "avail_gpu_ratio":    0.25
        },
        "DC5": {
          "carbon_rank":        5,
          "cost_rank":          2,
          "capacity_rank":      5,
          "capacity_blocked":   False,
          "solar_window":       False,
          "carbon_intensity":   630.0,
          "price_per_kwh":      0.100,
          "avail_gpu_ratio":    0.10
        }
      }
    }
  }
}
```

### How Reasoning Is Generated

Reasoning is **not an LLM** — it is deterministic Python logic applied to the raw observation
values after the actor has already made its decision. It is a post-hoc human-readable
interpretation of what the model chose, not a re-computation of the decision.

The process runs in four priority stages:

**Stage 1 — Capacity filter (hard constraint)**
Eliminate any DC whose `avail_gpu_ratio < threshold` when the task requires GPUs,
or `avail_cpu_ratio < threshold` for CPU-only tasks, or `avail_mem_ratio` too low.
DCs that fail this check are flagged as `"capacity_blocked": true` in the output.

**Stage 2 — Deadline urgency check**
If `deadline_hours < 1.0`: mark `defer_eligible = False` — the job cannot wait.
If `deadline_hours >= 2.0`: defer is on the table if all remaining DCs look bad.

**Stage 3 — Rank remaining DCs on environmental + cost factors**
For the DCs that passed Stage 1, rank on:
- `carbon_intensity_gco2_kwh` (lower = better) → `carbon_rank`
- `price_per_kwh` (lower = better) → `cost_rank`
- `avail_gpu_ratio` or `avail_cpu_ratio` depending on job type → `capacity_rank`
- Implicit temporal signal: note whether the timestamp is in a known solar window (10–16 UTC)
  for DC1/DC3 — flag as `"solar_window": true` if so, since the model learned this pattern

**Stage 4 — Build summary text from outcome**
Identify which factor the winning DC ranks #1 on → that becomes `primary_factor`.
Build a `summary` string from templates:
- Carbon wins: `"DC3 chosen: cleanest grid (185 gCO₂/kWh, 48% below cluster avg). Cost also competitive ($0.05/kWh)."`
- Cost wins: `"DC1 chosen: cheapest electricity ($0.08/kWh). Carbon acceptable (215 gCO₂/kWh, near-solar window)."`
- Defer: `"No suitable DC — all above 450 gCO₂/kWh and deadline not urgent. Deferring 15 min to wait for solar."`
- Capacity blocked: `"DC4, DC5 at GPU capacity. Routing to DC3 despite higher latency."`

The `confidence_scores` always come directly from `softmax(logits)` of the actor network
and are independent of the reasoning text — they reflect all 34 inputs as weighted by the
learned network, not just the 3–4 factors shown in the human summary.

### Actor Loading Strategy

**When sustain-cluster is available (real checkpoint)**:
```python
# Load ActorNet from checkpoint, obs_dim=34, act_dim=6
actor = ActorNet(obs_dim=34, act_dim=6, hidden_dim=256, use_layer_norm=True)
actor.load_state_dict(checkpoint["actor_state_dict"])
actor.eval()
```

**When sustain-cluster is absent (mock / demo mode)**:
Use a `MockActor` — a heuristic that mimics smart routing by computing a weighted score
per DC based on `(0.6 × normalised_carbon) + (0.4 × normalised_price)` and converting
to pseudo-logits. This produces realistic-looking decisions and confidence scores without
any model weights.

The `InferenceEngine` class accepts an optional `checkpoint_path` argument. If `None`,
it falls back to `MockActor` silently, so the frontend sees identical output format in both modes.

### Public API

```python
from backend.inference_engine import InferenceEngine

engine = InferenceEngine(checkpoint_path="sustain-cluster/checkpoints/.../best_eval_checkpoint.pth")
# or: InferenceEngine()  # uses MockActor

results = engine.run_sequence(observations)   # list[dict], one per timestep
# or step by step:
result = engine.step(obs_dict)                # single dict
```

---

## Feature 2 — Performance Tracker (`backend/performance_tracker.py`)

### Purpose

Produce **5 pairs of time series** (one pair per DC), each pair being:
- `local`: the carbon and cost if every job from that DC stayed at that DC
- `optimal`: the actual carbon and cost under RL routing (some jobs leave, some arrive)

This powers the comparison graph in the frontend.

### How Local vs Optimal Differ

Under **local-only** routing, each DC runs all jobs that originate there:
```
DC5 (Australia, coal) runs all its own jobs → high carbon per job
DC3 (Chile, hydro)   runs all its own jobs → low carbon per job
```

Under **RL routing**, the agent redistributes:
```
DC5 jobs (high-carbon grid) → rerouted to DC3 (low-carbon grid)
  → DC5's actual carbon drops (fewer jobs)
  → DC3's actual carbon rises slightly (more jobs, but still on clean grid)
  → Net effect: overall cluster carbon falls
```

The comparison chart shows this per-DC divergence over time.

### Outputs

```python
{
  "DC1": [
    {
      "timestep":         0,
      "timestamp":        "2025-06-15T00:00:00Z",
      "local": {
        "carbon_kg":        1.20,
        "energy_cost_usd":  0.45,
        "tasks_processed":  3,
        "carbon_intensity": 230.5    # grid CI at this timestep
      },
      "optimal": {
        "carbon_kg":              0.85,   # lower: some tasks rerouted away
        "energy_cost_usd":        0.38,
        "tasks_processed":        2,      # 1 task sent to DC3
        "tasks_received":         0,      # no incoming tasks this step
        "carbon_intensity":       230.5   # same grid — what changed is task count
      },
      "delta": {
        "carbon_kg_saved":   0.35,
        "cost_saved_usd":    0.07,
        "pct_carbon_saved":  29.2
      }
    },
    ... (287 more timesteps — 3 days total)
  ],
  "DC2": [ ... ],
  "DC3": [ ... ],
  "DC4": [ ... ],
  "DC5": [ ... ]
}
```

### Derivation of Optimal Metrics (Mock Mode)

In mock mode, optimal values are derived from the inference engine's decision sequence:

1. For each timestep `t`, the inference engine says "Job X goes to DC_k".
2. If `DC_k != task.origin_dc`:
   - `optimal.tasks_processed[origin_dc]` decreases by 1 (job left)
   - `optimal.tasks_processed[DC_k]` increases by 1 (job arrived)
3. Carbon saved at origin DC = `tasks_rerouted × energy_per_task × origin_dc_carbon_intensity`
4. Carbon added at destination = `tasks_rerouted × energy_per_task × dest_dc_carbon_intensity`
5. Net saving = step 3 − step 4 (always positive when routing from high-CI to low-CI)

### Public API

```python
from backend.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()
results = tracker.compute(
    observations,          # same sequence as InferenceEngine input
    decisions,             # output of engine.run_sequence()
)
# returns: dict[dc_id, list[timestep_dict]]
```

---

## Mock Data (`backend/mock_data.py`)

### Purpose

Generates a **288-timestep (3 days, 15-min intervals)** sequence of observations that the
inference engine and performance tracker can consume. 3 days is the default because it is
long enough to show two full daily carbon cycles repeating (demonstrating that the agent
consistently exploits solar windows), while remaining fast to generate and easy to display.

**The 15-minute interval is mandatory and must not be changed.** The model was trained
at this cadence — the time feature encoding, reward accumulation, and task queue dynamics
are all calibrated to 15-minute steps. For frontend charts that would look cluttered at
288 points, aggregate to hourly (group 4 steps → 1 display point) for visualisation only;
the model and tracker always operate at 15-min resolution.

Produces realistic patterns:
- DC1 carbon dips midday each day (California solar, 10–16 UTC)
- DC3 carbon stays consistently low (Chilean hydro, small daily variation)
- DC2 carbon varies with simulated wind (some days cleaner, some dirtier)
- DC4/DC5 carbon stays persistently high (gas/coal, small random noise only)
- Task arrivals follow a workday sinusoidal curve (peak 9am–6pm, quiet overnight)
- Mix of CPU-only jobs (no GPU requirement) and GPU jobs (GPU ratio checked first)
- Deadlines vary: urgent (0.5–1h) and deferrable (2–6h)

### Output Format

```python
from backend.mock_data import generate_mock_sequence

observations = generate_mock_sequence(
    num_timesteps=288,          # 3 days — recommended default
    seed=42,
    start_date="2025-06-15"     # any date; summer chosen for strong solar signal
)
# returns: list[dict]   each element matches the input schema above
```

`start_date` can be any date. It is used only to compute hour-of-day and day-of-year
for the time feature encoding — no external data is fetched.

### Mock DC Parameters

All resource values are **utilisation ratios (0.0–1.0)**, not raw counts.
Real datacenters have hundreds to thousands of GPUs; the model only ever sees the
fraction currently available. The absolute count is irrelevant to the model.

| DC | Location | Base CI (gCO₂/kWh) | CI Solar Dip | Price ($/kWh) | Avg GPU avail ratio | Avg CPU avail ratio |
|---|---|---|---|---|---|---|
| DC1 | California | 230 | −25% at 10–16 UTC | 0.12 | ~0.55 | ~0.60 |
| DC2 | Germany | 340 | −10% (wind, variable) | 0.18 | ~0.70 | ~0.65 |
| DC3 | Chile | 195 | −5% (hydro, stable) | 0.05 | ~0.45 | ~0.50 |
| DC4 | Singapore | 455 | none (gas baseline) | 0.16 | ~0.30 | ~0.40 |
| DC5 | Australia | 630 | none (coal baseline) | 0.10 | ~0.20 | ~0.35 |

Ratios fluctuate ±0.15 around their averages as tasks arrive and complete each step.
DC4 and DC5 have lower available ratios (busier) reflecting that the agent rarely
routes jobs away from them, so their queues fill up locally.

Task arrivals: sinusoidal workday curve, mix of CPU-only and GPU jobs, deadlines 0.5–6 hours.

---

## Mock Data Generation Guidelines — Real-Life Patterns

All 288 timesteps must follow the realistic patterns below. The goal is that the agent's
decisions look explainable when overlaid on the charts — e.g. judges should be able to see
DC1 carbon drop at noon and see the agent route more jobs there simultaneously.

### Carbon Intensity Patterns (per DC, per hour)

**DC1 — California (duck curve)**
Strong solar dip midday, sharp ramp-up in early evening when solar exits but demand stays high.
```
00:00 → 230   overnight gas baseload
06:00 → 240   morning demand ramp, solar not yet online
10:00 → 180   solar coming online
13:00 → 155   solar peak — deepest trough of day
16:00 → 195   solar fading, demand still elevated
19:00 → 260   evening peak — solar gone, gas peakers ramp
22:00 → 235   demand eases, settling back to baseload
```
Interpolate smoothly between these anchors. Add ±10 gCO₂/kWh Gaussian noise per step.
Pattern repeats each of the 3 days with slight day-to-day noise (±5% amplitude).

**DC2 — Germany (wind-variable)**
Less predictable than solar. Model as two different day types across the 3-day window:
- Day 1 and 3: calm days, coal-heavy. CI hovers 330–360 all day, small overnight dip to 310.
- Day 2: windier. CI drops to a trough of ~200 for a 6-hour window (randomise which 6 hours),
  otherwise 280–320. This day-to-day variation shows the agent adapting to changing conditions.
Add ±15 gCO₂/kWh noise per step.

**DC3 — Chile (hydro-dominated, stable)**
Barely varies intraday. Reservoir dispatch is flat. Slight midday uptick from industrial demand.
```
Flat ~185–210 throughout all 3 days, ±12 noise only.
No meaningful solar or wind pattern.
```

**DC4 — Singapore (gas, demand-tracked)**
Follows commercial and industrial activity. Single smooth hump peaking mid-morning.
```
00:00 → 420   low overnight demand
09:00 → 475   commercial peak
14:00 → 460   slight afternoon dip
18:00 → 450   early evening, gradual decline
22:00 → 430   settling back
```
Add ±10 noise. Pattern nearly identical across all 3 days.

**DC5 — Australia NSW (coal, double-hump)**
Two demand peaks per day — morning (residential + commercial start) and evening (return home).
No meaningful renewable penetration in mock.
```
00:00 → 600   overnight baseload
08:00 → 660   morning peak
13:00 → 620   midday slight dip
18:00 → 670   evening peak — highest of day
23:00 → 610   declining back to baseload
```
Add ±15 noise. Pattern repeats all 3 days.

---

### Electricity Price Patterns (per DC, per hour)

Prices are not simply proportional to carbon — they track supply/demand dynamics,
which creates interesting tension the agent must navigate.

**DC1 — California (duck curve inversion)**
Price *drops* at solar peak due to oversupply, then spikes sharply in early evening
when solar exits and gas peakers come online. This is the California duck curve in prices.
```
Midnight:    $0.12
Solar peak (13:00): $0.07   ← cheapest AND cleanest simultaneously
Evening peak (18:00–20:00): $0.22  ← most expensive AND dirtiest
Late night:  $0.11
```
This means DC1 at noon is the agent's best option on both dimensions at once.
Smooth sinusoidal shape between anchors, ±$0.01 noise per step.

**DC2 — Germany**
Tracks demand; cheapest overnight, expensive during working hours.
```
01:00–05:00: $0.14
09:00–17:00: $0.20–0.22
Evening:     $0.18
```
On DC2's windy day (Day 2), add a brief price dip ($0.11–0.13) during
the wind generation window — cheap AND relatively clean simultaneously.

**DC3 — Chile**
Chronically cheap hydro. Barely varies intraday.
```
All day: $0.04–0.07   always the cheapest DC across all timesteps
```
Slight overnight dip to $0.04, daytime $0.06–0.07. ±$0.005 noise.

**DC4 — Singapore**
Flat gas-indexed price, no meaningful variation.
```
All day: $0.15–0.18   ±$0.01 noise only
```

**DC5 — Australia**
Cheapest overnight (coal surplus), expensive during peak demand.
This creates a cost-vs-carbon tension: DC5 is cheapest at night but always the dirtiest.
```
Midnight:    $0.08   cheap but carbon 600 gCO₂/kWh
Peak (18:00): $0.16
Late night:  $0.09
```
The agent must decide if the cost saving is worth the carbon penalty —
this is exactly the kind of trade-off that distinguishes RL from simple greedy baselines.

---

### Resource Availability Ratios (per DC, intraday)

Availability is driven by task arrival rate. Jobs take time to complete, so availability
lags behind arrivals — it drops through the morning and troughs in mid-to-late afternoon,
then recovers overnight as jobs finish.

**General shape (all DCs)**:
```
00:00–06:00: high availability (jobs finishing overnight, few new arrivals)
06:00–10:00: gradual drop (morning task ramp begins)
10:00–17:00: trough (peak arrivals, jobs still running)
17:00–00:00: gradual recovery (arrivals tail off, jobs complete)
```

**Per-DC specifics**:

| DC | Overnight high | Midday trough | Notes |
|---|---|---|---|
| DC1 | GPU 0.70, CPU 0.75 | GPU 0.35, CPU 0.40 | Moderate churn, daily recovery |
| DC2 | GPU 0.80, CPU 0.80 | GPU 0.55, CPU 0.60 | Agent rarely sends jobs here — stays available |
| DC3 | GPU 0.60, CPU 0.65 | GPU 0.28, CPU 0.35 | Busiest: receives many incoming routed jobs |
| DC4 | GPU 0.35, CPU 0.45 | GPU 0.12, CPU 0.20 | Local queue fills; agent avoids routing here |
| DC5 | GPU 0.30, CPU 0.40 | GPU 0.10, CPU 0.18 | Near-full locally; agent still avoids — barely recovers overnight |

GPU availability drops more slowly than CPU (GPU jobs run longer) and recovers more slowly.
Memory tracks CPU roughly. Add ±0.05 noise per step, clipped to [0.05, 0.95].

---

### Task Arrival Patterns

**Arrival rate envelope** — sinusoidal workday shape, same across all 3 days with ±10% day-to-day amplitude variation:
```
00:00–06:00: 0–1 tasks per 15-min step   (quiet overnight)
06:00–09:00: ramp 1→4 tasks              (morning startup)
09:00–14:00: peak 4–8 tasks              (core working hours)
14:00–18:00: gradual decline 3→6 tasks   (afternoon)
18:00–22:00: tail 1–3 tasks              (evening wind-down)
22:00–00:00: near zero                   (overnight quiet)
```

Day 2 is slightly busier (multiply envelope by 1.15) to prevent the 3-day window looking
like three identical repeating loops.

**Job type mix per arrival**:

| Type | Share | GPU cores | CPU cores | Duration | Deadline | Routing behaviour |
|---|---|---|---|---|---|---|
| CPU-only, short | 60% | 0 | 0.03–0.10 norm | 0.25–1h | 1–3h | Agent can defer moderately |
| GPU job, deferrable | 30% | 0.10–0.40 norm | 0.05–0.15 norm | 0.5–3h | 2–6h | Worth routing across world for clean energy |
| GPU job, urgent | 10% | 0.10–0.30 norm | 0.05–0.10 norm | 0.5–2h | <1h | Cannot defer; agent must route immediately |

`origin_dc` is assigned proportionally to DC size: DC1 30%, DC2 25%, DC3 15%, DC4 20%, DC5 10%.
This reflects DC5 being a smaller facility — it still generates jobs the agent must route away.

---

### What These Patterns Produce in the Demo

With these patterns in place the visualisations tell a coherent narrative without any
manual tuning:

| Time window | What happens | Why visible in charts |
|---|---|---|
| **Midnight** | Agent routes almost everything to DC3 | DC3 cheapest + cleanest; DC1 not in solar window |
| **10:00** | Agent splits between DC1 and DC3 | DC1 enters solar window, becomes competitive on both axes |
| **13:00** | DC1 peaks in routing share | DC1 at CI=155, price=$0.07 — best on both dimensions simultaneously |
| **17:00** | DC1 routing share collapses | Solar exits, DC1 price spikes to $0.22 — agent snaps back to DC3 |
| **Windy Day 2 afternoon** | DC2 briefly gets routed jobs | CI drops to ~200 + price dips — agent notices and adapts |
| **DC3 near capacity** | Agent spills over to DC1 and DC2 | Shows multi-DC balancing, not just always pick DC3 |
| **Urgent GPU job at 18:00** | Agent routes to DC3 despite DC5 origin | Cannot defer — shows deadline constraint in reasoning output |

The local vs optimal comparison chart tells the clearest story for **DC4 and DC5**:
their local lines stay persistently high (coal and gas for every job) while their
optimal lines dip significantly as the agent reroutes their jobs to cleaner grids.
DC3's optimal line rises slightly above its local line — it absorbs incoming jobs but
still at low CI, so the cluster-wide net saving is always positive.

---

## Exact 34-Dim Observation Vector Layout

This is what gets built from the structured dict and fed to the actor network.
The `encode_observation(obs_dict)` function in `inference_engine.py` must produce
exactly this layout. `decode_observation(vector, timestamp)` reverses it for the
real-data swap path.

```
Index   Feature                              Encoding
──────  ───────────────────────────────────  ──────────────────────────────────
0       sin(2π × hour / 24)                  derived from timestamp
1       cos(2π × hour / 24)                  derived from timestamp
2       sin(2π × day_of_year / 365)          derived from timestamp
3       cos(2π × day_of_year / 365)          derived from timestamp

4       task.cpu_cores_norm                  already a ratio — pass through
5       task.gpu_cores_norm                  already a ratio — pass through
6       task.memory_gb_norm                  already a ratio — pass through
7       task.duration_hours / 10.0           normalise by assumed max of 10h
8       task.deadline_hours / 24.0           normalise by 24h

# DC1 block
9       DC1.avail_cpu_ratio                  already a ratio — pass through
10      DC1.avail_gpu_ratio                  already a ratio — pass through
11      DC1.avail_mem_ratio                  already a ratio — pass through
12      DC1.carbon_intensity / 800.0         normalise by 800 gCO₂/kWh (observed max)
13      DC1.price_per_kwh    / 0.30          normalise by $0.30/kWh (observed max)

# DC2 block
14–18   same 5-feature pattern for DC2

# DC3 block
19–23   same 5-feature pattern for DC3

# DC4 block
24–28   same 5-feature pattern for DC4

# DC5 block
29–33   same 5-feature pattern for DC5

Total: 4 + 5 + (5 × 5) = 34 ✓
```

**Note on temperature**: Temperature is recorded in the simulation metrics output
(`temperature_c` in `per_dc_records`) but is **not** a feature in the current 34-dim
observation vector. The 4+5+25 breakdown leaves no room for it, and the README
confirms 34 total dims. Temperature affects carbon intensity and price indirectly through
the simulation (hotter weather → higher cooling load → higher effective energy use), but
the agent sees those downstream effects in the carbon and price signals, not the raw temperature.

---

## What Each Input Actually Contributes to the Model's Decision

The confidence scores reflect all 34 inputs simultaneously through learned network weights.
The reasoning text only surfaces 3–4 factors for human readability. This section explains
what each input group actually does inside the model.

### Time Features — Temporal Pattern Recognition (dims 0–3)

The model does not directly compute "solar = lower carbon." Instead, during training it saw
millions of steps where carbon intensity traces dropped at noon for DC1/DC3 and correlated
those drops with the time encoding. The result is implicit temporal routing:

| Learned pattern | Time signal that triggers it |
|---|---|
| Prefer DC1, DC3 midday | sin/cos hour ≈ solar peak (10–16 UTC) |
| Avoid deferring early morning | Hour signal near 06:00 → "day starting, use capacity" |
| Accept higher carbon overnight | Hour near 02:00 → "demand low, prices low, trade-off shifts" |
| Summer DC3 preference | Day-of-year → summer hydro levels higher in Chile |

These patterns are **not directly exposed in the reasoning text** because they are
implicit signals, not discrete readable values. The reasoning can note
`"solar_window": true` when the timestamp falls in 10–16 UTC for DC1/DC3 as
a hint that the time signal is likely influencing the decision.

### Task Features — Routing Constraints (dims 4–8)

| Feature | What the model learned |
|---|---|
| `gpu_cores_norm > 0` | Only route to DCs with spare GPU capacity; GPU jobs are worth routing farther for a clean grid |
| `deadline_hours` small (<1h) | Cannot defer → pick best available DC now, even if suboptimal |
| `deadline_hours` large (>2h) | Deferral (action 0) becomes viable if all DCs look bad |
| `duration_hours` large | Longer job → carbon savings compound → worth routing to a cleaner DC even with some cost penalty |
| `memory_gb_norm` high | `avail_mem_ratio` becomes a binding constraint; a low-carbon DC with no memory is not an option |

### Per-DC Resource Ratios — Capacity Constraints (dims 9–11, 14–16, 19–21, 24–26, 29–31)

`avail_cpu_ratio`, `avail_gpu_ratio`, `avail_mem_ratio` for each DC.

These are hard-ish constraints that the model learned to respect:
- A DC at near-zero GPU ratio effectively gets ruled out for GPU jobs regardless of carbon
- A DC with high availability gets a moderate preference boost (less queueing delay)
- The model learned that routing to an overloaded DC causes SLA violations → negative reward

### Per-DC Carbon Intensity — Primary Sustainability Signal (dims 12, 17, 22, 27, 32)

`carbon_intensity / 800.0` for each DC.

This is the single most influential feature for the carbon dimension of the reward
(`weight = 0.3`). The model learned to strongly prefer the DC with the lowest current
carbon intensity, all else being equal. Because carbon intensity varies by time of day
(solar) the model correlates this with the time features.

### Per-DC Price — Primary Cost Signal (dims 13, 18, 23, 28, 33)

`price_per_kwh / 0.30` for each DC.

This is the most influential feature overall — the reward weights cost at 0.9 vs carbon
at 0.3. In practice the model often picks the cheapest DC unless it is dramatically
dirtier than the cheapest-carbon option. DC3 (Chile) frequently wins on both dimensions
simultaneously (cheap hydro power), which is why it dominates routing decisions.

### What Temperature Is NOT in the Vector

As noted above, temperature is absent from the obs vector. It affects energy use through
the simulation's internal PUE model, which means its effect is already baked into the
`carbon_kg` and `energy_cost_usd` outputs at each step — but the agent never directly
sees outdoor temperature as an input.

---

## How to Swap to Real Data

When `sustain-cluster` is initialised, replace mock data and mock actor with real ones:

### Real Observations (from SustainCluster environment)

```python
# Instead of generate_mock_sequence(), capture obs from real env:
env = make_eval_env(...)
obs, _ = env.reset(seed=42)

observations = []
for t in range(288):   # 3 days at 15-min intervals
    obs_dict = decode_observation(obs, current_time)  # inverse of encode_observation()
    observations.append(obs_dict)
    obs, _, done, _, info = env.step([])
```

Add `encode_observation(obs_dict) -> np.ndarray` and `decode_observation(obs_vector, timestamp) -> dict`
helpers to `inference_engine.py`. `encode_observation` follows the exact index layout in the
"Exact 34-Dim Observation Vector Layout" section above. `decode_observation` reverses it —
multiply by the normalisation constants to recover raw values, reconstruct the ratio fields directly.

### Real Actor (from checkpoint)

```python
engine = InferenceEngine(
    checkpoint_path="sustain-cluster/checkpoints/train_multiaction_defer_.../best_eval_checkpoint.pth"
)
```

No other code changes needed — same API, same output format.

### Real Carbon Intensity (live)

`carbon_api.py` already provides live CI values. These can be used to override the
carbon_intensity fields in each observation dict before passing to the engine:

```python
ci_data, is_live = get_live_carbon_intensity()
for obs_dict in observations:
    for dc_id in obs_dict["datacenters"]:
        label = f"{dc_id} (US-CA)" if dc_id == "DC1" else ...
        obs_dict["datacenters"][dc_id]["carbon_intensity_gco2_kwh"] = ci_data[label]
```

---

## Implementation Order

1. **`backend/mock_data.py`** — Build first. No dependencies. Can be tested in isolation.
   Output: list of 288 observation dicts (3 days, 15-min intervals) with realistic patterns.
   All resource fields are ratios 0.0–1.0. Start date defaults to today or any recent date.

2. **`backend/inference_engine.py`** — Depends on mock_data for testing.
   - Implement `encode_observation(obs_dict) -> np.ndarray[34]` first — this is the
     critical bridge between the structured dict and what the model consumes.
     Follow the exact index layout in "Exact 34-Dim Observation Vector Layout" above.
   - Build `MockActor` (heuristic weighted score: carbon×0.6 + price×0.4 per DC,
     converted to pseudo-logits) — verify output format and confidence score distribution
   - Implement 4-stage reasoning generator (capacity → deadline → rankings → summary text)
   - Then wire up real `ActorNet` loading path (gated by `checkpoint_path` arg)
   - Output: list of 288 decision dicts with confidence scores + full reasoning

3. **`backend/performance_tracker.py`** — Depends on inference_engine output.
   - Takes observation sequence + decision sequence
   - Derives local vs optimal metrics per DC per timestep
   - Output: dict of 5 DC time series

4. **Integration test** — Simple script that chains all three:
   ```python
   obs = generate_mock_sequence()
   decisions = InferenceEngine().run_sequence(obs)
   performance = PerformanceTracker().compute(obs, decisions)
   # print sample output, verify shapes
   ```

5. **FastAPI route (future)** — Expose as HTTP endpoints for frontend:
   ```
   GET  /api/sequence?days=1&seed=42   → observations (mock or real)
   POST /api/infer                      → decisions + reasoning
   POST /api/performance                → local vs optimal time series
   ```

---

## Existing Code Integration Summary

| Existing code | Relationship to new modules |
|---|---|
| `simulator.py` `run_comparison()` | Still used for the Streamlit bulk-comparison tabs. New modules are a separate, step-by-step path. No conflict. |
| `carbon_api.py` `get_live_carbon_intensity()` | Optional: can inject live CI values into the observation sequence before passing to `InferenceEngine`. |
| `app.py` Streamlit dashboard | Unchanged. New modules are consumed by a separate web app / FastAPI layer. |

---

## Key Design Principles

- **Same output format** whether using real actor or mock actor — frontend never knows the difference
- **No changes to existing files** — purely additive
- **Self-contained mock data** — demo works fully offline with no sustain-cluster and no API keys
- **Real-data swap is a one-line change** — just pass `checkpoint_path` to `InferenceEngine`
- **Reasoning is deterministic Python** — no LLM, no latency, works in real-time per step
