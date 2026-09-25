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

**Status: Phase 2, Day 4 of 15 complete.**

---

## Day 1 — Repo Scaffold + IO-VNBD Loader + Synthetic Data

Full repo scaffold; `src/io_vnbd_loader.py` (`load_smartphone_drive()`,
`drive_summary()`); `src/utils/synthetic_data.py`. Verified: 1200-sample
synthetic drive loads correctly at 10 Hz.

---

## Day 2 — EDA + Naive Double-Integration Drift Baseline

`src/eda.py`, `src/dead_reckoning/naive_drift_baseline.py`. Verified:
41.99 m final error / 21.01 m/min drift on a bias-free synthetic drive,
logged as the first `results/metrics.md` row.

**Issue found:** GPS-speed-based stationary detection lags the true
stop instant — carried into Day 3.

---

## Day 3 — Gravity Compensation + IMU-Native ZUPT + Bias Estimation

`src/calibration/gravity_compensation.py`, `zupt_classifier.py`,
`bias_estimation.py`. Verified on a test drive with known injected
bias: recovered closely (+0.1675 vs true +0.15 m/s²), drift cut ~20x
on the same drive (3213.93 m → 149.96 m).

**Caveat:** verified against a locally-built stand-in test drive, not
the user's real `synthetic_data.py` (still not shared).

**Issue found:** `zupt_mask()` default thresholds reach only ~6%
precision against GPS on the synthetic generator — carried into Day 4/5.

---

## Day 4 — Mount Alignment: Leveling + Heading

**Goal:** Express acceleration in the vehicle's frame (forward/lateral/
up), not the phone's — the phone can be mounted at any orientation.
Gravity alone can only constrain 2 of 3 rotational degrees of freedom,
so this is genuinely two sub-problems, done across two sessions so
neither shipped half-verified.

**What was built (part 1 — leveling):**
- `src/calibration/mount_alignment.py` — `rotation_from_vectors(a, b)`
  (general Rodrigues shortest-arc rotation), `estimate_leveling_rotation()`
  (aligns mean stationary gravity to `VEHICLE_UP`), `recovered_roll_pitch_deg()`
  (diagnostic only).
- **Sign convention resolved, not assumed:** confirmed Android's
  `TYPE_GRAVITY` uses the same sign as the accelerometer at rest
  (≈+9.81 when level) before writing any code — an earlier draft had
  targeted vehicle-down instead, which would have silently flipped the
  leveled z-axis.
- **Verified:** purpose-built known-tilt test (roll +12.0°, pitch
  −7.0° injected) recovered 11.995°/−7.0006°.

**What was built (part 2 — heading, completes Day 4):**
- Extended `mount_alignment.py` with `rotation_about_z()`,
  `accelerating_mask()` (flags GPS-speed-increasing samples),
  `estimate_heading_offset()` (angle of mean leveled horizontal linear
  acceleration during those windows — requires bias-corrected input),
  `estimate_mount_rotation()` (combines leveling + heading into one
  phone→vehicle rotation), `apply_mount_alignment()` (produces
  `lin_accel_{x,y,z}_mps2_cal_veh`).

**Verified:**
- Built a new test drive with a full 3D mount misalignment injected
  (roll=8.0°, pitch=−5.0°, **yaw=22.0°** — the earlier leveling-only
  test had no yaw, so couldn't test this half of the problem).
  `estimate_heading_offset()` recovered **21.99°**, accurate to 0.01°.
- Full pipeline on that drive: mount-aligned double integration gave
  **209.54 m** final error vs. **254.29 m** on the same drive without
  mount alignment (raw phone-frame x-axis) — a real ~18% improvement.
  Logged as the Day 4 row in `results/metrics.md`.

**Finding investigated rather than just reported:** the ~18%
improvement is smaller than Day 3's ~20x. Rather than leave that
unexplained, checked whether leveling or heading were actually
imprecise — they aren't (leveling verified by applying `R_level`
directly to the true gravity vector: landed within `1e-16` of
`[0,0,9.81]`; heading recovered to 0.01°). The remaining ~210 m is
dominated by plain sensor **noise** compounding through naive double
integration over 120 s — Day 2's noise-only baseline alone produced
~42 m from noise with zero injected bias or misalignment. Mount
alignment can't fix noise-driven drift; that's specifically what
Phase 4's strapdown INS with periodic ZUPT resets (Days 9–10) is for.

**A diagnostic-only bug found and fixed along the way:**
`recovered_roll_pitch_deg()` reported misleading numbers (~9.3° vs
true 8.0° roll) once real yaw was present in test data — isolated to
confirm the actual rotation (`R_level`) was correct and only the
human-readable Euler-angle readout was ambiguous (the minimal Rodrigues
rotation isn't the same as a naive "undo roll, then pitch" inverse when
yaw ≠ 0). This function is never used downstream, so it had zero effect
on any real result — but the docstring was tightened so it doesn't
mislead future debugging.

**A path-resolution bug found and fixed across all 6 script files:**
running a file from an unexpected location broke on
`ModuleNotFoundError`, because every script counted a fixed number of
`.parent` levels to find the project root. Replaced with a function
that walks upward looking for a folder containing both `src/` and
`data/`, verified robust to file location and working directory.

**Decisions made:**
- Heading estimation requires bias-corrected (`_cal`) linear
  acceleration, not gravity-compensated-only — an uncorrected bias
  would skew the estimated angle.
- `estimate_mount_rotation()` combines leveling + heading into a single
  matrix (`R_align = R_yaw @ R_level`) applied once, rather than two
  separate rotation steps — simpler downstream and avoids compounding
  floating-point error across two applications.

**Next (Day 5):** Denoising (the last item in Phase 2's original scope
— "Mount alignment, bias estimation, denoising, ZUPT classifier").
Also worth doing given today's finding: since remaining drift is now
confirmed noise-dominated, a low-pass filter or windowed smoothing on
`lin_accel_x_mps2_cal_veh` before integration is directly motivated by
today's diagnosis, not just the original phase checklist.

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
│   │   └── mount_alignment.py      (leveling + heading, complete)
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
    ├── metrics.md              (Day 2, 3, 4 rows)
    └── plots/
        ├── day1_imu_sanity_check.png
        ├── day2_eda_imu_distributions.png
        ├── day2_eda_stationary_noise.png
        ├── day2_naive_drift_baseline.png
        ├── day3_calibrated_drift.png
        └── day4_mount_aligned_drift.png
```