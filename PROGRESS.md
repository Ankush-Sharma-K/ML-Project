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

**Status: Phase 2, Day 4 of 15 — IN PROGRESS (part 1 of 2 complete).**

---

## Day 1 — Repo Scaffold + IO-VNBD Loader + Synthetic Data

Full repo scaffold; `src/io_vnbd_loader.py` (`load_smartphone_drive()`,
`drive_summary()`); `src/utils/synthetic_data.py` (schema-matching
synthetic drive generator, no internet in this sandbox). Verified:
1200-sample synthetic drive loads correctly at the expected 10 Hz.

---

## Day 2 — EDA + Naive Double-Integration Drift Baseline

`src/eda.py` (channel stats, GPS-based stationary detection, noise
floor). `src/dead_reckoning/naive_drift_baseline.py` (naive
accel→velocity→position integration, zero correction — the baseline
every later phase must beat). Verified: 41.99 m final error / 21.01
m/min drift on a bias-free synthetic drive, logged as the first
`results/metrics.md` row.

**Issue found:** GPS-speed-based stationary detection lags the true
stop instant — carried into Day 3.

---

## Day 3 — Gravity Compensation + IMU-Native ZUPT + Bias Estimation

`src/calibration/gravity_compensation.py` (`compute_linear_acceleration()`
— subtracts the phone's own fused gravity estimate from raw
accelerometer, no orientation needed). `src/calibration/zupt_classifier.py`
(`zupt_mask()` — IMU-native stationary detection, fixes Day 2's GPS-lag
issue). `src/calibration/bias_estimation.py` (`estimate_constant_bias()`,
`apply_bias_correction()`). Verified on a test drive with a known
injected bias: bias recovered closely (+0.1675 vs true +0.15 m/s² on
accel_x), drift cut ~20x on the same drive (3213.93 m → 149.96 m).

**Caveat:** verified against a locally-built stand-in test drive, not
the user's real `synthetic_data.py` (still not shared — see `context.md`
Section 5).

**Issue found:** `zupt_mask()`'s default thresholds only reach ~6%
precision against GPS on the current straight-line synthetic generator
— carried into Day 4/5.

---

## Day 4 (part 1 of 2) — Mount Alignment: LEVELING

**Goal:** Correct for the phone's mounting orientation so acceleration
is expressed in the vehicle's frame, not the phone's. Full mount
alignment is two independent sub-problems — leveling (roll/pitch, from
gravity) and heading (yaw, needs a second signal) — and rather than
ship both half-verified, **today's scope was leveling only**; heading
estimation + the drift re-run are Day 4 part 2, next session.

**What was built:**
- `src/calibration/mount_alignment.py` —
  - `rotation_from_vectors(a, b)`: general Rodrigues shortest-arc
    rotation between two vectors (small reusable utility, will also be
    needed for the heading step in part 2).
  - `estimate_leveling_rotation(df, stationary_mask)`: aligns the mean
    measured gravity vector (during ZUPT-flagged stationary windows) to
    `VEHICLE_UP`. Correct up to an unknown rotation about the vertical
    axis — exactly the ambiguity part 2 resolves.
  - `recovered_roll_pitch_deg(R_level)`: reports the tilt angles being
    corrected, for sanity-checking/logging only, not used downstream.

**Sign convention resolved (not assumed):** before writing any code,
checked what Android's `TYPE_GRAVITY` sensor convention actually is —
same sign as the accelerometer at rest, i.e. `gravity_z_mps2 ≈ +9.81`
when the phone lies flat screen-up. `VEHICLE_UP = (0, 0, +1)` is the
leveling target for that reason. This matters: an earlier draft of this
file (written before checking) had targeted vehicle-*down* instead,
which would have silently flipped the leveled z-axis. Documented in the
module docstring as something to re-confirm against real IO-VNBD data,
since a flipped sign in a real drive's export would mean the phone was
mounted screen-down for that recording, not a convention error.

**Verified — two separate checks:**
1. **Purpose-built known-tilt test** (roll +12.0°, pitch −7.0°
   deliberately injected into a synthetic gravity vector, not from any
   real or existing drive file): `estimate_leveling_rotation()` recovered
   roll = 11.995°, pitch = −7.0006° — both accurate to well within the
   injected sensor noise. The leveled gravity vector landed at
   `[9.17e-05, -8.3e-04, 9.80999996]` — magnitude and direction correct.
2. **Ran end-to-end against the real Days 1-3 pipeline** (gravity
   compensation → ZUPT → this drive's stationary mask) without error,
   from multiple working directories.

**Caveat on check 2's specific numbers:** the existing
`data/raw/SYNTH-drive1.csv` in this sandbox (left over from earlier
day's ad hoc bias-injection testing, not the user's real
`synthetic_data.py`) has arbitrary gravity values that don't represent
a physically real phone tilt — it produced a nonsensical "177° roll".
This is expected and NOT a bug: check 1 (the purpose-built test) is
what validates correctness; check 2 only confirms the code runs without
error against real pipeline data shapes. Re-run
`src/calibration/mount_alignment.py` once the real `synthetic_data.py`
or actual IO-VNBD data is available to get a meaningful tilt number.

**Explicitly NOT done today (Day 4 part 2):**
- Heading (yaw) estimation from accelerating-phase correlation with GPS
- `apply_mount_alignment()` to actually rotate `lin_accel_*_cal` into
  the vehicle frame
- Re-running the drift baseline on the mount-aligned signal
- Logging a Day 4 row to `results/metrics.md`

**Next (Day 4 part 2):** heading estimation via horizontal
linear-acceleration direction during GPS-speed-increasing windows,
combine with today's `R_level` into a single phone→vehicle rotation,
apply it, re-run `naive_double_integrate()` on
`lin_accel_x_mps2_cal_veh`, and log the Day 4 row.

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
│   │   └── mount_alignment.py      (leveling only — heading pending)
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
    ├── metrics.md              (Day 2 + Day 3 rows; Day 4 pending part 2)
    └── plots/
        ├── day1_imu_sanity_check.png
        ├── day2_eda_imu_distributions.png
        ├── day2_eda_stationary_noise.png
        ├── day2_naive_drift_baseline.png
        └── day3_calibrated_drift.png
```