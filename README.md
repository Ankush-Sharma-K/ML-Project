# IDR — AI/ML Intelligent Dead Reckoning for GPS-denied tunnels

Smartphone-only navigation through tunnels: when GPS disappears, the phone's
accelerometer and gyroscope, a 1-D CNN speed model, a Kalman filter and the
OpenStreetMap road network keep estimating where the car is, check whether it
is still on the planned route, and correct the tunnel track as soon as GPS
returns. Built and evaluated on the real IO-VNBD smartphone drives (97 drives,
UK + France).

## Results (held-out TEST drives)

Simulated GPS outages on real drives, entering at a real GPS fix; everything
the method uses comes from before the outage; error = metres to the first GPS
fix after the outage (median, 60 outages on 4 test drives).

| Method | 30 s | 60 s | 120 s |
|---|---|---|---|
| Keep last GPS speed and direction (baseline) | 120 m | 332 m | 929 m |
| ML speed (anchored) + calibrated gyro heading (Day 10) | 94 m | 208 m | 747 m |
| + HMM map matching (Day 12) | 72 m | 106 m | 712 m |
| + EKF GNSS/INS fusion (Day 13, batch) | 66 m | 112 m | 484 m |
| **Streaming edge engine, int8 on-device model (Day 15, deployed)** | **69 m** | **130 m** | **379 m** |

| Other results | |
|---|---|
| Speed model (20 s IMU window -> speed), test MAE | CNN **4.53 m/s**; gradient boosting 5.18; ridge 6.51; predict the mean 8.05 |
| On-device model | int8 **0.71 MB**, 0.6 ms per window (1 CPU thread), test MAE 4.58 m/s |
| Tunnel track rebuilt when GPS returns (RTS smoother), 120 s outage | **43 m** median at the hidden GPS fixes (engine), vs 78 m real time |
| Route verification inside the tunnel | 8% false alarms on the correct route, 65% of real wrong turns detected, median 7.8 s after the car left the route |
| GPS outliers (15% of fixes replaced by 200 m jumps) | chi-square gate: P90 entrance error 97 m vs 198 m |

Honest limits: only 4 of the 9 test drives are long enough for 2-minute
outages (the larger test + val table is in `results/metrics.md`); the 120 s
error is dominated by the speed model's per-drive bias (map matching fixes
direction, not distance); the engine's numbers differ from the batch run by
design (10-minute rolling calibration instead of all data before the outage).
Full narrative, every experiment that did NOT work, and all numbers:
[`PROGRESS.md`](PROGRESS.md); function registry: [`context.md`](context.md).

## Pipeline

```text
phone IMU (10 Hz) ──► gravity removal ─► bias + mount calibration (rolling, GPS-confirmed stops)
                  └─► per-phone gyro axis fit ─► turn rate
20 s window ──► int8 1-D CNN ──► speed (anchored at the tunnel entrance)
speed + turn rate + GPS (when present) ──► EKF [pos, heading, speed, ML bias, gyro bias]
                                           (adaptive GPS noise, chi-square gate)
no GPS ──► HMM map matching on OSM roads ──► position on the road  ──► route monitor (ON / OFF route)
GPS back ──► RTS smoother ──► corrected tunnel track
```

## Quick start

```powershell
python -m venv mlproject
mlproject\Scripts\pip install -r requirements.txt
cd idr-project
python main.py app          # edge engine + mobile web app -> http://localhost:8000
python main.py test         # 27 unit tests (synthetic data, no dataset needed)
```

The app also opens as static files (`mobile_app/index.html` via any web server)
and then plays the bundled demo replays (first outage of every test drive,
correct and wrong planned route). With the server it can replay any test / val
outage live through the engine. Checked in headless Microsoft Edge (idle page
and a full 120 s replay rendered); `?run=1` in the URL starts a replay on load
(handy for presentations). The first replay of an outage takes ~15 s, then it
is cached.

Data: put the IO-VNBD `S-*.csv` files in `data/raw/real/` (vehicle files for
wheel-speed truth in `data/raw/vehicle/`); road maps are built by
`python main.py maps` (Geofabrik extracts, Overpass for large regions) into
`data/external_maps/`.

## Commands (`python main.py <cmd>`)

| Command | What it does | Output |
|---|---|---|
| `app [--port N] [--host H]` | edge engine HTTP API + app (localhost by default) | `results/replays/` cache |
| `test` | unit tests | |
| `train` | train the speed model (fixed split `data/split.json`) | `results/checkpoints/velocity_cnn_best.pt` |
| `compress` | int8 / fp32 on-device export + verification | `velocity_model_int8.ptl`, `results/compression_summary.json` |
| `baselines` | ridge + gradient boosting vs the CNN | `results/simple_baselines.json` |
| `maps` | OSM road corridors | `data/external_maps/*.graphml` |
| `bench day8 / day9 / day10 / day11-12 / day13 / day15` | each day's benchmark | `results/*benchmark*.csv`, `results/metrics.md` |
| `demo-bundle` | regenerate the app's offline replays | `mobile_app/assets/demo/` |

## Project layout

```text
idr-project/
├── main.py                     single entry point
├── run_training.py             speed-model training on the real drives
├── src/
│   ├── io_vnbd_loader.py       IO-VNBD loader (schemas, units, time resets)
│   ├── calibration/            gravity, ZUPT / stops, bias, mount alignment, denoising
│   ├── models/                 windows, CNN, training, simple baselines, on-device export
│   ├── dead_reckoning/         strapdown INS, anchored ML speed, gyro heading + 2-D path
│   ├── map_matching/           OSM corridors (osm_loader), HMM matcher
│   ├── fusion/                 EKF + RTS smoother, GNSS/INS fusion
│   ├── edge_engine/            streaming engine, replay benchmark, HTTP server, demo bundle
│   └── utils/                  wheel-speed truth, synthetic data
├── mobile_app/                 web app (Leaflet), demo replays in assets/demo/
├── tests/                      27 unit tests
├── data/                       raw drives, split.json, external_maps/
└── results/                    metrics.md, benchmark CSVs, logs, checkpoints
```

---

## Original project brief (team README)

## AI-ML-based-Intelligent-Dead-Reckoning-system-for-seamless-navigation
## AI-ML Based Intelligent Tunnel Navigation System

### Problem Statement

When a vehicle enters a tunnel, GPS and network connectivity may become unavailable.

We need to develop a system that uses **vehicle/device sensors and AI/ML** to:

* Estimate vehicle **speed and velocity**
* Determine **movement and direction**
* Estimate the vehicle's **path inside the tunnel**
* Compare the estimated path with the **expected map route**
* Detect whether the vehicle is **following the correct route or has deviated**

### Basic Idea

```text
Vehicle enters tunnel
        ↓
GPS / Network unavailable
        ↓
Sensor Data
(Accelerometer + Gyroscope + other sensors)
        ↓
AI/ML
        ↓
Speed + Velocity + Direction + Movement
        ↓
Estimated Vehicle Path
        ↓
Compare with Map Route
        ↓
Correct Route / Route Deviation
```

### What We Need to Learn

#### Basics

* Python
* NumPy
* Pandas
* Matplotlib
* Basic mathematics and statistics
* Git & GitHub

#### Project Concepts

* Accelerometer and Gyroscope
* GPS and its limitations
* Dead Reckoning
* Sensor data processing
* Time-series data
* Basic Machine Learning
* Path/route comparison

### Project Roadmap

#### Stage 1 — Understand the Problem

Learn:

* What happens to GPS inside tunnels
* How accelerometer and gyroscope work
* Basic dead reckoning
* How vehicle movement can be estimated from sensors

#### Stage 2 — Get Sensor Data

Start with an existing dataset containing:

* Timestamp
* Accelerometer
* Gyroscope
* GPS/reference position
* Speed, if available

Do not start by building a mobile application.

#### Stage 3 — Analyze the Data

Use Python to:

* Load the dataset
* Clean the sensor data
* Plot sensor readings
* Understand how sensor values change with vehicle movement

#### Stage 4 — Build a Basic Baseline

Before using AI/ML, create a basic movement/path estimation method.

```text
Previous Position
      +
Movement Information
      ↓
Estimated New Position
```

This gives us something to compare the ML system against.

#### Stage 5 — Apply AI/ML

Train a model to estimate useful movement information such as:

* Speed
* Distance travelled
* Direction/heading correction
* Position/path correction

Start with simple models before trying neural networks.

#### Stage 6 — Route Verification

Compare:

```text
Expected Map Route
        vs
Estimated Vehicle Path
```

Then determine whether the vehicle is:

* Following the route
* Deviating from the route

#### Stage 7 — Evaluate

Compare the estimated path with the actual/reference path.

Measure:

* Position error
* Speed prediction error
* Path deviation
* Final location error

### Project Structure

```text
AI-ML-Dead-Reckoning/
│
├── README.md
├── requirements.txt
├── main.py
│
├── data/
│   └── raw/
│
├── src/
│
├── results/
│
└── docs/
```

Later, `src/` can contain:

```text
data_processing.py
features.py
dead_reckoning.py
model.py
path_comparison.py
evaluation.py
visualization.py
```

### Team Work

Divide the work into:

* **Sensor/Data Team** → Dataset and sensor understanding
* **Dead Reckoning Team** → Movement and position calculations
* **ML Team** → Features and ML models
* **Map/Path Team** → Route comparison and deviation detection
* **Integration Team** → Combine everything

Everyone should understand the complete workflow.

### GitHub Workflow

Each member works on a separate branch:

```text
main
 ├── sensor-data
 ├── dead-reckoning
 ├── machine-learning
 └── path-comparison
```

Basic workflow:

```bash
git pull
git checkout -b feature-name
git add .
git commit -m "Add feature"
git push
```

### Current Task

**Start with Stage 1 only.**

1. Understand GPS failure inside tunnels.
2. Learn accelerometer and gyroscope basics.
3. Learn the basic idea of dead reckoning.
4. Find a suitable sensor/GPS dataset.
5. Load and visualize the data using Python.

#### Project Principle

> **Understand → Collect Data → Build Baseline → Apply ML → Compare Path → Evaluate**
