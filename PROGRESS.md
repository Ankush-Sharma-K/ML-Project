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

**Status: Phase 2, Day 3 of 15 complete.**

---

## Day 1 — Repo Scaffold + IO-VNBD Loader + Synthetic Data

**Goal:** Stand up the project skeleton and a working data-loading path
before touching any DSP/ML.

**What was built:**
- Full repo scaffold (`data/`, `src/{calibration,models,dead_reckoning,
  map_matching,fusion,edge_engine,utils}`, `mobile_app/`, `tests/`,
  `results/plots/`).
- `src/io_vnbd_loader.py` — `load_smartphone_drive()` loads IO-VNBD
  smartphone drives into a tidy DataFrame (24-column schema: 10 Hz IMU
  accel/gravity/gyro/mag/orientation + GPS lat/lon/speed/accuracy/
  satellites), plus `drive_summary()` for a post-load sanity check.
- `src/utils/synthetic_data.py` — generates a physically-plausible fake
  drive matching the same schema (no internet access in this sandbox to
  fetch real IO-VNBD data).

**Verified:** synthetic drive (1200 samples, 119.9 s) loads correctly,
implied rate 10.0 Hz, GPS speed range matches the cruise target.

**Decisions made:**
- Schema confirmed against the published Data in Brief paper.
- No internet / no `pip install torch, scikit-learn, filterpy` in this
  sandbox — classical DSP/filter/Kalman code built and tested here on
  synthetic data; deep-learning training (Phase 3) and real IO-VNBD runs
  happen in the user's own environment.
- Synthetic drive is straight-line only for now.

---

## Day 2 — EDA + Naive Double-Integration Drift Baseline

**Goal:** Characterize raw sensor noise/distributions and establish the
naive dead-reckoning baseline every later phase has to beat.

**What was built:**
- `src/eda.py` — `channel_stats()`, `stationary_mask()` (GPS-speed
  threshold), `stationary_noise_floor()`, plus distribution and
  stationary-window plots.
- `src/dead_reckoning/naive_drift_baseline.py` — `naive_double_integrate()`
  (raw `accel_x_mps2` → cumsum → velocity → cumsum → position, zero
  correction), `gps_reference_distance()` (ground truth), `evaluate_drift()`,
  `plot_drift()`, `log_metric()` (appends to `results/metrics.md`).

**Verified:** on a synthetic drive with no injected sensor bias, naive
double integration gave 41.99 m final error / 21.01 m/min drift / 20.71 m
RMSE over a 966.67 m drive — logged as the first row of `results/metrics.md`.

**Issue found (carried into Day 3):** the GPS-speed-based stationary
detector lags the true stop instant by up to ~1 s (1 Hz GPS, sparse),
inflating the measured stationary noise floor — flagged as needing a
tighter, IMU-native detector.

**Decisions made:**
- Naive baseline integrates `accel_x_mps2` only (forward axis); will
  need heading integration once turns are introduced (Phase 5+).
- `results/metrics.md` format locked: Phase/Day, Method, final error,
  drift rate, RMSE — every phase appends one row.

---

## Day 3 — Gravity Compensation + IMU-Native ZUPT + Bias Estimation

**Goal:** Fix the Day 2 stationary-detection issue and remove the two
error sources naive double-integration is most vulnerable to — gravity
leakage from phone tilt, and constant sensor bias — then re-measure
drift to log the first real improvement.

**What was built:**
- `src/calibration/gravity_compensation.py` — `compute_linear_acceleration()`.
  The IO-VNBD smartphone schema conveniently provides both raw
  accelerometer (`accel_{x,y,z}_mps2`, gravity-inclusive) AND the
  phone's own fused gravity estimate (`gravity_{x,y,z}_mps2`).
  Subtracting one from the other gives linear acceleration
  (`lin_accel_{x,y,z}_mps2`) **without needing to know the phone's
  mounting orientation** — this has to happen before bias estimation,
  or a constant tilt-induced gravity component gets misattributed as
  sensor bias.
- `src/calibration/zupt_classifier.py` — `zupt_mask()`: an IMU-native
  stationary detector using windowed rolling std of linear-acceleration
  magnitude AND gyro magnitude (both must be below threshold), replacing
  Day 2's GPS-lag-prone detector. `evaluate_zupt_against_gps()` cross-checks
  agreement against the (independent, imperfect) GPS-speed reference.
- `src/calibration/bias_estimation.py` — `estimate_constant_bias()`
  (mean of each linear-accel/gyro channel during ZUPT-flagged stationary
  windows — at true rest these should read exactly zero, so any nonzero
  mean IS the bias), `apply_bias_correction()`, and the Day 3 orchestration
  script that runs gravity compensation → ZUPT → bias estimation →
  correction → re-runs `naive_double_integrate()` on the corrected signal
  → logs the result as a new `results/metrics.md` row.

**Verified:** Built and tested against a locally-constructed test drive
with a **known, deliberately injected** constant bias (accel_x +0.15 m/s²,
gyro_yaw +0.02 rad/s, plus a +0.30 m/s² constant gravity-tilt component
on the x-axis) — specifically so the bias estimator's output could be
checked against ground truth:
- **Bias recovery:** estimated +0.1675 m/s² (true +0.15) on accel_x,
  +0.0203 rad/s (true +0.02) on gyro_yaw — both close, small residual
  from noise + imperfect stationary-window selection.
- **Drift improvement, same drive, apples-to-apples:** naive
  double-integration on raw `accel_x_mps2` (uncorrected, gravity leakage
  + bias both present) → **3213.93 m final error** (this drive's bias
  makes Day 2's raw-signal approach catastrophic, as expected — double
  integration of ANY nonzero constant bias explodes quadratically with
  time). After gravity compensation + bias correction → **149.96 m
  final error, 67.82 m RMSE** — roughly a **20x reduction**, logged as
  the Day 3 row in `results/metrics.md`.

**⚠️ Caveat — not yet validated on the user's real data:** this sandbox
does not have the user's actual `synthetic_data.py` (only
`io_vnbd_loader.py` has been confirmed against the real file so far, per
`context.md`), so Day 3's numbers above are from a locally-built
stand-in test drive, not the project's real synthetic generator. The
**code and logic are what's verified** (imports match the real
`io_vnbd_loader.py` interface, gravity-compensation math is
dataset-schema-correct, bias estimation is proven to recover a known
injected bias) — re-run `src/calibration/bias_estimation.py` against the
real `data/raw/SYNTH-drive1.csv` to get the numbers that actually belong
in the project's own `results/metrics.md`.

**Issue found (carries into Day 4/5):** `zupt_mask()`'s default
thresholds (accel_var 0.05, gyro_var 0.02 — set from Day 2's measured
noise floor with margin) only reach ~6% precision against the GPS-speed
reference on the current straight-line synthetic generator, because its
abrupt (linspace) accel/brake transitions inflate rolling variance right
at the true stop instant — the same lag problem as Day 2, now visible
from the other direction. Two independent fixes worth doing before
Phase 4 depends on ZUPT: (1) smooth the synthetic generator's speed
profile transitions (ease-in/out instead of linspace) so stationary
windows are less ambiguous even in test data, and (2) once real IO-VNBD
data is available, retune thresholds against its real stationary noise
floor rather than synthetic data's edge cases.

**Decisions made:**
- Gravity compensation runs before bias estimation, always — order
  matters, since gravity leakage and constant bias are two different
  error sources conflated in raw `accel_x_mps2`.
- Bias is estimated as a single constant per channel (not time-varying)
  — consistent with the "constant-bias" MEMS sensor model assumed in
  the problem statement; a slowly time-varying bias model is a possible
  future refinement if real data shows drift in the bias itself.
- Mount alignment (rotating the phone's coordinate frame into the
  vehicle's forward/lateral/vertical frame) is deferred to Day 4 — it
  depends on having clean, bias-corrected linear acceleration first, and
  needs stretches of real turning/cornering motion to observe, which the
  current straight-line-only synthetic drive can't fully exercise.

**Next (Day 4):** Mount alignment — estimate the phone's orientation
relative to the vehicle body frame (e.g. via gravity-vector alignment at
rest + heading correlation with GPS bearing during motion), rotate
`lin_accel_{x,y,z}_mps2_cal` into the vehicle frame, and re-run the drift
baseline again. Also: smooth the synthetic generator's transition
profile per the ZUPT issue above, and retune `zupt_mask()` thresholds.

**Files:**
```
idr-project/
├── requirements.txt
├── PROGRESS.md
├── context.md
├── SUMMARY.md
├── src/
│   ├── io_vnbd_loader.py
│   ├── eda.py
│   ├── utils/
│   │   └── synthetic_data.py
│   ├── calibration/
│   │   ├── gravity_compensation.py
│   │   ├── zupt_classifier.py
│   │   ├── bias_estimation.py
│   │   └── mount_alignment.py      (empty — Day 4)
│   ├── models/                      (empty — Day 6)
│   ├── dead_reckoning/
│   │   └── naive_drift_baseline.py
│   ├── map_matching/                  (empty — Day 11)
│   ├── fusion/                         (empty — Day 13)
│   └── edge_engine/                     (empty — Day 15)
├── data/
│   └── raw/
│       └── SYNTH-drive1.csv    (generated, not real IO-VNBD yet)
├── mobile_app/               (empty — Day 15)
├── tests/                     (empty)
└── results/
    ├── metrics.md              (Day 2 + Day 3 rows)
    └── plots/
        ├── day1_imu_sanity_check.png
        ├── day2_eda_imu_distributions.png
        ├── day2_eda_stationary_noise.png
        ├── day2_naive_drift_baseline.png
        └── day3_calibrated_drift.png
```