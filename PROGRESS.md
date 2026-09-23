# idr-project — Progress Report

**Project:** AI/ML-based Intelligent Dead Reckoning (IDR) for Seamless Navigation
**Plan:** 15-day phased build, deployed as a mobile app + edge-deployable engine
**Approach:** classical strapdown INS (orientation integration + ZUPT) corrected by
AI/ML (learned forward-velocity model, adaptive fusion noise), constrained by
OSM map-matching with Non-Holonomic Constraints, fused with GNSS via UKF/EKF.
Trained/evaluated against the IO-VNBD dataset per the problem statement.

## Overall Plan

| Phase | Days | Focus |
|---|---|---|
| 1 — Setup & Dataset Understanding | 1–2 | Repo scaffold, IO-VNBD loader, EDA, naive drift baseline |
| 2 — Calibration & Preprocessing | 3–5 | Mount alignment, bias estimation, denoising, ZUPT classifier |
| 3 — AI Velocity Estimation | 6–8 | IMU→velocity model, robustness, on-device compression |
| 4 — Core Dead Reckoning | 9–10 | Strapdown INS integration, drift benchmark evaluation |
| 5 — Map Matching | 11–12 | OSM road graph, NHC constraints, HMM map matcher |
| 6 — GNSS+INS Fusion | 13–14 | UKF/EKF fusion, adaptive noise, seamless GNSS↔DR handoff |
| 7 — App & Deployment | 15 | Edge engine API, mobile app UI, packaging, final submission |

**Status: Phase 1, Day 1 of 15 complete.**

---

## Day 1 — Repo Scaffold + IO-VNBD Loader + Synthetic Data

**Goal:** Stand up the project skeleton and a working data-loading path
before touching any DSP/ML — every later phase depends on being able to
load a drive (real or synthetic) into a clean, known DataFrame shape.

**What was built:**
- Full repo scaffold (`data/`, `src/{calibration,models,dead_reckoning,
  map_matching,fusion,edge_engine,utils}`, `mobile_app/`, `tests/`,
  `results/plots/`) per the phase plan above.
- `src/io_vnbd_loader.py` — loads IO-VNBD smartphone drives (`S-*.csv`)
  into a tidy DataFrame, matching the dataset's published 24-column
  schema (10 Hz IMU: accel/gyro/mag/orientation, 1 Hz GPS: lat/lon/speed/
  accuracy/satellites). Includes `drive_summary()` for a quick post-load
  sanity check (sample count, duration, implied sample rate, speed
  range).
- `src/utils/synthetic_data.py` — generates a physically-plausible fake
  drive (accelerate → cruise → brake → stop cycles, IMU noise + constant
  bias, injected pothole shocks) matching the exact same schema. Needed
  because this dev environment has no internet access to download the
  real dataset — this unblocks every downstream day and gets swapped for
  real IO-VNBD CSVs with zero code changes elsewhere.

**Verified:** Generated a synthetic drive (1200 samples, 119.9 s),
loaded it through `io_vnbd_loader.py`, and confirmed: implied sample
rate 10.0 Hz (matches spec), GPS speed range 0–41 km/h (matches the
40 km/h cruise target). Plotted raw `accel_x` (clearly shows
accel/cruise/brake/stop cycles), `accel_z` (shows injected pothole
spikes), and `gyro_yaw` (noisy around a small bias, as expected).
Output saved as `results/plots/day1_imu_sanity_check.png`.

**Why this matters for later phases:** The calibration (Phase 2),
velocity model (Phase 3), and strapdown integrator (Phase 4) all consume
this same DataFrame shape — getting the schema and a realistic noise/bias
profile right now means nothing downstream has to change when real
IO-VNBD data replaces the synthetic drive.

**Decisions made:**
- IO-VNBD schema confirmed against the published Data in Brief paper
  (Onyekpe, Palade, Kanarachos, Szkolnik, Coventry University).
- This sandbox has no internet / can't `pip install` torch, scikit-learn,
  or filterpy — classical DSP/filter/Kalman code gets built and tested
  here on synthetic data; Phase 3's deep-learning training and any run on
  real IO-VNBD data will happen in your own environment (local + internet,
  or Colab), with code handed off ready to run.
- Synthetic drive is straight-line only for now — turning/curved roads
  aren't needed until Phase 5's map matcher requires a real road graph.

**Next (Day 2):** EDA pass + the naive double-integration drift baseline
— integrate raw accelerometer directly to position with zero correction,
and plot how fast it drifts. This number is what every later phase needs
to beat, and gets logged as the first entry in `results/metrics.md`.

**Files:**
```
idr-project/
├── requirements.txt
├── PROGRESS.md
├── SUMMARY.md
├── src/
│   ├── io_vnbd_loader.py
│   ├── utils/
│   │   └── synthetic_data.py
│   ├── calibration/        (empty — Day 3)
│   ├── models/              (empty — Day 6)
│   ├── dead_reckoning/       (empty — Day 9)
│   ├── map_matching/          (empty — Day 11)
│   ├── fusion/                 (empty — Day 13)
│   └── edge_engine/             (empty — Day 15)
├── data/
│   └── raw/
│       └── SYNTH-drive1.csv    (generated, not real IO-VNBD yet)
├── mobile_app/               (empty — Day 15)
├── tests/                     (empty)
└── results/
    └── plots/
        └── day1_imu_sanity_check.png
```
