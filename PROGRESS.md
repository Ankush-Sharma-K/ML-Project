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

Day 2 — EDA + Naive Double-Integration Drift Baseline

Goal: Characterize the raw sensor data (distributions, noise floor, stationary periods) and establish the naive dead-reckoning baseline — the number every later phase (calibration, ZUPT, learned velocity model, strapdown INS, map matching, GNSS fusion) has to beat.

What was built:

src/eda.py — EDA pass over a loaded drive:
channel_stats(): mean/std/min/max for all 6 raw IMU channels (accel_x/y/z, gyro_roll/pitch/yaw).
stationary_mask(): flags samples as stationary using forward-filled GPS speed < 1 km/h — this is the same signal Phase 2's ZUPT classifier will need to detect zero-velocity windows.
stationary_noise_floor(): IMU channel std restricted to stationary windows — the actual sensor noise floor, used later to set Kalman filter R/Q matrices (Phase 6).
Two plots: raw IMU channel histograms, and accel_x/GPS-speed with detected stationary windows overlaid.
src/dead_reckoning/naive_drift_baseline.py — the naive baseline itself:
naive_double_integrate(): raw accel_x → cumulative-sum velocity → cumulative-sum position, with zero bias correction, ZUPT, or any other correction.
evaluate_drift(): compares integrated position against ground truth (synthetic drives carry true distance in df.attrs; real IO-VNBD drives fall back to a GPS-speed-integrated reference track) and reports final position error, drift rate (m/min), and RMSE.
log_metric(): appends the result as the first row of results/metrics.md, which every later phase will add a row to.

Verified: Ran EDA and the drift baseline against the Day 1 synthetic drive (regenerated fresh from the same seed, 1200 samples / 119.9 s):

Channel stats: accel_x mean ≈ 0.005 m/s² (near-zero, as expected for a straight cyclic drive), std ≈ 0.84 m/s² (dominated by the accel/brake cycles, not noise). accel_z mean ≈ 9.82 m/s² (≈ gravity, as expected), with a max of 14.5 m/s² from the injected pothole spikes. Gyro channels all near-zero mean with ~0.01 rad/s std, matching the injected bias/noise config.
Stationary detection: 90 of 1200 samples (7.5%) flagged stationary across the 4 stop segments in the drive — matches the 3-cycle accelerate/cruise/brake/stop profile plus the drive's initial stopped state.
Stationary noise floor: accel_y/z and all 3 gyro channels come out at ~0.01–0.03 std, consistent with the injected sensor noise. accel_x stationary std came out higher (~0.52) than the injected 0.05 — see "Issue found" below.
Naive drift baseline: final position error 41.99 m over a 966.67 m true drive (21.01 m/min drift rate, 20.71 m RMSE), with error growing monotonically and non-linearly (visible acceleration in the error curve during each accel/brake cycle) — the textbook double-integration drift signature. Logged as the first row of results/metrics.md.

Issue found (carries into Day 3): the stationary-window detector relies on 1 Hz GPS speed forward-filled to 10 Hz, which lags behind the true stop instant by up to ~1s — so the "stationary" accel_x std comes out inflated (it's partially catching the tail of the braking ramp, not true zero-velocity samples). Phase 2's real ZUPT classifier (Day 3-5) needs a tighter, IMU-native stationary detector (e.g. windowed gyro-magnitude + accel-variance threshold) rather than relying on sparse GPS speed alone.

Decisions made:

Naive baseline integrates accel_x only (forward axis) since the synthetic drive is straight-line-only; once Phase 5 introduces curved roads the baseline will need heading (yaw) integration too to project forward accel into a 2D track — noted for when the synthetic generator gains turns.
results/metrics.md format locked in now (Phase/Day, Method, final error, drift rate, RMSE columns) — every phase from here appends one row, so the drift-reduction story is directly comparable end-to-end.

Next (Day 3): Start Phase 2 — mount alignment (estimate and correct the phone's orientation relative to the vehicle) and constant-bias estimation per channel, using the stationary windows identified today (with the tighter IMU-native detector noted above). Re-run the Day 2 drift baseline after bias correction to log the first improvement in results/metrics.md.

Files:

idr-project/
├── requirements.txt
├── PROGRESS.md
├── SUMMARY.md
├── src/
│   ├── io_vnbd_loader.py
│   ├── eda.py
│   ├── utils/
│   │   └── synthetic_data.py
│   ├── calibration/        (empty — Day 3)
│   ├── models/              (empty — Day 6)
│   ├── dead_reckoning/
│   │   └── naive_drift_baseline.py
│   ├── map_matching/          (empty — Day 11)
│   ├── fusion/                 (empty — Day 13)
│   └── edge_engine/             (empty — Day 15)
├── data/
│   └── raw/
│       └── SYNTH-drive1.csv    (generated, not real IO-VNBD yet)
├── mobile_app/               (empty — Day 15)
├── tests/                     (empty)
└── results/
    ├── metrics.md              (Day 2: naive baseline logged)
    └── plots/
        ├── day1_imu_sanity_check.png
        ├── day2_eda_imu_distributions.png
        ├── day2_eda_stationary_noise.png
        └── day2_naive_drift_baseline.png