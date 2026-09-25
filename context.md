# context.md — Project Context & Interface Registry

> **Purpose of this file:** `PROGRESS.md` tells you what was done and why,
> narratively. This file tells you the exact names of every function,
> column, and variable that already exist, so new code plugs into old
> code without guessing or colliding. Upload **both** files (plus any
> source file the next day's code will directly import from) at the
> start of every new day's session.

---

## 1. Project Aim

**Goal:** Build an AI/ML-based Intelligent Dead Reckoning (IDR) system
for seamless vehicle navigation on a smartphone — keep producing an
accurate position estimate using only the phone's IMU during GPS-denied
periods, and hand off seamlessly back to GNSS when signal returns.

**Approach (classical + ML hybrid):** classical strapdown INS
(orientation integration + ZUPT) + a learned forward-velocity model +
adaptive fusion noise, OSM map matching with Non-Holonomic Constraints,
GNSS+INS fusion via UKF/EKF.

**Dataset:** IO-VNBD (Onyekpe, Palade, Kanarachos & Szkolnik, Data in
Brief 2021). This sandbox has no internet, so a synthetic generator
matching the same schema is used for dev/test until real data is run in
a local/Colab environment.

**Plan:**

| Phase | Days | Focus |
|---|---|---|
| 1 — Setup & Dataset Understanding | 1–2 | Repo scaffold, IO-VNBD loader, EDA, naive drift baseline |
| 2 — Calibration & Preprocessing | 3–5 | Mount alignment, bias estimation, denoising, ZUPT classifier |
| 3 — AI Velocity Estimation | 6–8 | IMU→velocity model, robustness, on-device compression |
| 4 — Core Dead Reckoning | 9–10 | Strapdown INS integration, drift benchmark evaluation |
| 5 — Map Matching | 11–12 | OSM road graph, NHC constraints, HMM map matcher |
| 6 — GNSS+INS Fusion | 13–14 | UKF/EKF fusion, adaptive noise, seamless GNSS↔DR handoff |
| 7 — App & Deployment | 15 | Edge engine API, mobile app UI, packaging, final submission |

**Constraint carried through every phase:** no internet / no
`pip install torch/scikit-learn/filterpy` in this sandbox — classical
code built and tested here on synthetic data; Phase 3 training and real
IO-VNBD runs happen in the user's own environment.

---

## 2. Data Schema (locked — do not rename without updating this file)

Source of truth: `src/io_vnbd_loader.py`, `load_smartphone_drive()`.

### Raw columns
`gps_lat_deg`, `gps_lon_deg`, `gps_alt_m`, `gps_speed_kmh` (full 10 Hz
rate, no fix flag), `gps_accuracy_m`, `gps_orientation_deg`,
`gps_satellites_in_range`, `time_since_start_ms`, `datetime`,
`accel_{x,y,z}_mps2` (raw, gravity-inclusive), `gravity_{x,y,z}_mps2`
(phone's fused gravity estimate — **sign convention confirmed:
≈(0,0,+9.81) when level, same as accelerometer at rest**),
`gyro_{yaw,pitch,roll}_rads`, `mag_{x,y,z}_ut`,
`orientation_{yaw,pitch,roll}_deg`.

### Derived columns

| Column | Meaning | Added by |
|---|---|---|
| `t_sec` | seconds since drive start — canonical time axis | `io_vnbd_loader.load_smartphone_drive()` |
| `lin_accel_{x,y,z}_mps2` | gravity-compensated linear accel | `gravity_compensation.compute_linear_acceleration()` (Day 3) |
| `<col>_cal` | bias-corrected version | `bias_estimation.apply_bias_correction()` (Day 3) |
| `lin_accel_{x,y,z}_mps2_cal_veh` | **vehicle-frame** (mount-aligned) linear accel — forward/lateral/up | `mount_alignment.apply_mount_alignment()` (Day 4, now built) |

**Naming convention:** axis suffix + unit suffix, `_cal` once
bias-corrected, `_veh` once rotated into the vehicle frame.

---

## 3. Function Registry

### `src/io_vnbd_loader.py`
`load_smartphone_drive(csv_path, header=None) -> pd.DataFrame`.
`drive_summary(df) -> dict`.

### `src/utils/synthetic_data.py`
**⚠️ Still not verified against the user's real file.** Highest-priority
upload still outstanding — every day so far has been tested against a
locally-built stand-in, not the project's real generator.

### `src/eda.py`
`channel_stats`, `stationary_mask` (GPS-based), `stationary_noise_floor`,
`plot_distributions`, `plot_stationary_noise`, `run_eda`.
`IMU_ACCEL_COLS`, `IMU_GYRO_COLS`.

### `src/dead_reckoning/naive_drift_baseline.py`
| Name | Signature | Notes |
|---|---|---|
| `naive_double_integrate` | `(df, accel_col="accel_x_mps2") -> dict` | generic — reused with a different `accel_col` every day since Day 3 |
| `gps_reference_distance` | `(df) -> np.ndarray` | ground truth |
| `evaluate_drift` | `(df, true_distance_m=None) -> dict` | — |
| `plot_drift` | `(eval_result, out_path, title="Day 2 — ...")` | always pass `title` explicitly |
| `log_metric` | `(metrics_path, eval_result)` | hardcodes "Day 2" label — Day 3+ scripts write their own metrics row |

### `src/calibration/gravity_compensation.py` (Day 3)
`compute_linear_acceleration(df) -> pd.DataFrame`. `AXES`.

### `src/calibration/zupt_classifier.py` (Day 3)
`zupt_mask(df, accel_var_thresh=0.05, gyro_mag_thresh=0.02, window=5) -> np.ndarray`
(defaults not yet tuned against real data). `evaluate_zupt_against_gps()`.
`LIN_ACCEL_COLS`, `GYRO_COLS`.

### `src/calibration/bias_estimation.py` (Day 3)
`estimate_constant_bias(df, mask, cols=BIAS_COLS) -> dict`.
`apply_bias_correction(df, bias) -> pd.DataFrame`. `BIAS_COLS`.

### `src/calibration/mount_alignment.py` (Day 4 — complete)
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `rotation_from_vectors` | `(a, b) -> np.ndarray` | 3×3 rotation matrix | general Rodrigues shortest-arc rotation |
| `estimate_leveling_rotation` | `(df, stationary_mask) -> np.ndarray` | 3×3 rotation (roll+pitch only) | uses raw `gravity_*`, not bias-corrected accel |
| `recovered_roll_pitch_deg` | `(R_level) -> dict` | `{roll_deg, pitch_deg}` | **diagnostic only, unreliable when true yaw ≠ 0 — see docstring; never used downstream** |
| `rotation_about_z` | `(theta_rad) -> np.ndarray` | 3×3 rotation matrix | — |
| `accelerating_mask` | `(df, accel_thresh_ms2=0.3) -> np.ndarray` | bool array | `d(gps_speed)/dt > thresh` |
| `estimate_heading_offset` | `(df, R_level, accel_mask, lin_accel_cal_cols=DEFAULT) -> float` | radians | **requires `_cal` (bias-corrected) columns**, not just `_mps2` |
| `estimate_mount_rotation` | `(df, stationary_mask, accel_mask=None, lin_accel_cal_cols=DEFAULT) -> dict` | `{R_level, heading_offset_rad, heading_offset_deg, R_align, n_accelerating_samples}` | top-level combiner — leveling + heading in one call |
| `apply_mount_alignment` | `(df, R_align, lin_accel_cal_cols=DEFAULT) -> pd.DataFrame` | df + `lin_accel_{x,y,z}_mps2_cal_veh` | — |
| `VEHICLE_UP` | `np.array([0,0,1.0])` | — | leveling target |
| `GRAVITY_COLS`, `DEFAULT_LIN_ACCEL_CAL_COLS` | module constants | — | — |

### Not yet built (reserve these module paths)
| Phase | Expected file(s) |
|---|---|
| 2 (Day 5) | denoising (file not yet named — likely `src/calibration/denoising.py`) |
| 3 (Days 6–8) | `src/models/velocity_model.py` |
| 4 (Days 9–10) | `src/dead_reckoning/strapdown_ins.py` |
| 5 (Days 11–12) | `src/map_matching/osm_graph.py`, `src/map_matching/hmm_matcher.py` |
| 6 (Days 13–14) | `src/fusion/ukf_fusion.py` |
| 7 (Day 15) | `src/edge_engine/`, `mobile_app/` |

---

## 4. Results Log Format (locked)

`results/metrics.md` — `Phase / Day | Method | Final position error (m) | Drift rate (m/min) | RMSE (m)`

Current rows (all from locally-built test drives, NOT the user's real
`SYNTH-drive1.csv` — re-run before trusting for reporting):
- Day 2 — Naive double-integration — 41.99 m — 21.01 m/min — 20.71 m RMSE
- Day 3 — Gravity-comp + bias-corrected — 149.96 m — 75.04 m/min — 67.82 m RMSE (vs. 3213.93 m same-drive uncorrected)
- Day 4 — Mount-aligned (vehicle-frame) — 209.54 m — 104.86 m/min — 92.62 m RMSE (vs. 254.29 m same-drive without mount alignment; separately verified: heading recovered to 0.01° accuracy on a test with 22° injected yaw)

Note: Day 4's *absolute* number (209.54 m) isn't directly comparable to
Day 3's (149.96 m) — they're different test drives with different
injected errors (Day 3's had no mount misalignment, Day 4's did but a
different bias/noise realization). The meaningful comparisons are
same-drive, same-day (each row's own "vs. uncorrected" figure).

---

## 5. Known Open Issues (carry forward until resolved)

- **`synthetic_data.py` still unconfirmed** — highest priority, every
  day so far has been verified against a stand-in, not the real file.
- **ZUPT threshold tuning** — poor precision (~6%) against GPS on the
  synthetic generator; needs smoother transitions + real-data retuning.
- **Straight-line-only synthetic drives** — Day 4's heading test needed
  a purpose-built drive with injected yaw for exactly this reason; the
  main `SYNTH-drive1.csv` generator still doesn't produce turns.
- **Bias model is constant-only** — may need to become time-varying if
  real data shows bias drift.
- **Gravity sign convention on real data unconfirmed** — code assumes
  `gravity_z_mps2 ≈ +9.81` at rest; check against a real stationary
  window before trusting `estimate_leveling_rotation()` on actual data.
- **NEW — remaining drift after Day 4 is noise-dominated, not
  calibration-dominated:** confirmed via direct investigation (leveling
  and heading both verified exact; Day 2's noise-only baseline alone
  produced comparable-magnitude drift). This means Day 5's denoising
  step has a clear, evidence-based motivation, and Days 9–10's ZUPT-reset
  strapdown INS is what actually resolves this class of error — naive
  double integration, however well-calibrated, cannot.
- **`recovered_roll_pitch_deg()` diagnostic limitation** — misleading
  when true mount yaw ≠ 0 (Euler-angle ambiguity, not a rotation-matrix
  bug). Never used downstream. Full explanation in the function's
  docstring and PROGRESS.md Day 4.

---

## How to use this file

1. Upload `PROGRESS.md`, `context.md`, and the specific source file(s)
   the next day's code will import from — `synthetic_data.py` is the
   highest-priority upload still outstanding.
2. After each day, update both: narrative in `PROGRESS.md`, exact
   names/signatures in Section 3 here.
3. Never rename an existing entry without updating every caller.