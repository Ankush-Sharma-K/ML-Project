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
for seamless vehicle navigation on a smartphone — i.e., keep producing
an accurate position estimate using only the phone's IMU during
GPS-denied periods, and hand off seamlessly back to GNSS when signal
returns.

**Approach (classical + ML hybrid, per problem statement):** classical
strapdown INS (orientation integration + ZUPT) + a learned
forward-velocity model + adaptive fusion noise, OSM map matching with
Non-Holonomic Constraints, GNSS+INS fusion via UKF/EKF.

**Dataset:** IO-VNBD (Onyekpe, Palade, Kanarachos & Szkolnik, Data in
Brief 2021). Smartphone recordings (`S-*.csv`) are the primary input;
this sandbox has no internet, so a synthetic generator matching the same
schema is used for dev/test until real data is run in a local/Colab
environment.

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

**Constraint carried through every phase:** dev sandbox has no internet
and can't `pip install torch/scikit-learn/filterpy` — classical
DSP/filter/Kalman code is built and tested here against synthetic data;
deep-learning training (Phase 3) and real IO-VNBD runs happen in the
user's own environment.

---

## 2. Data Schema (locked — do not rename without updating this file)

Source of truth: `src/io_vnbd_loader.py`, `load_smartphone_drive()`.
Real, verified schema (confirmed against the user's actual file).

### Raw columns
`gps_lat_deg`, `gps_lon_deg`, `gps_alt_m`, `gps_speed_kmh` (full 10 Hz
rate, no fix flag), `gps_accuracy_m`, `gps_orientation_deg`,
`gps_satellites_in_range`, `time_since_start_ms`, `datetime`,
`accel_{x,y,z}_mps2` (raw, gravity-inclusive), `gravity_{x,y,z}_mps2`
(phone's fused gravity estimate — **sign convention: ≈(0,0,+9.81) when
level, same convention as accelerometer at rest; re-confirm on real
data, see Section 5**), `gyro_{yaw,pitch,roll}_rads`, `mag_{x,y,z}_ut`,
`orientation_{yaw,pitch,roll}_deg`.

### Derived columns

| Column | Meaning | Added by |
|---|---|---|
| `t_sec` | seconds since drive start — canonical time axis | `io_vnbd_loader.load_smartphone_drive()` |
| `lin_accel_{x,y,z}_mps2` | gravity-compensated linear accel, `= accel - gravity` | `gravity_compensation.compute_linear_acceleration()` (Day 3) |
| `<col>_cal` | bias-corrected version, `= col - estimated_bias[col]` | `bias_estimation.apply_bias_correction()` (Day 3) |
| `lin_accel_{x,y,z}_mps2_cal_veh` | vehicle-frame (mount-aligned) linear accel — **NOT YET PRODUCED**, needs heading (Day 4 part 2) before it can be computed | `mount_alignment.apply_mount_alignment()` — function doesn't exist yet |

**Naming convention:** axis suffix + unit suffix, `_cal` once
bias-corrected, `_veh` once rotated into the vehicle frame.

---

## 3. Function Registry

### `src/io_vnbd_loader.py`
| Name | Signature | Returns |
|---|---|---|
| `load_smartphone_drive` | `(csv_path, header=None) -> pd.DataFrame` | tidy DataFrame + `t_sec` |
| `drive_summary` | `(df) -> dict` | sanity-check stats |

### `src/utils/synthetic_data.py`
**⚠️ Still not verified against the user's real file.** Highest-priority
upload still outstanding.

### `src/eda.py`
`channel_stats`, `stationary_mask` (GPS-based, superseded for
calibration by `zupt_classifier.zupt_mask`), `stationary_noise_floor`,
`plot_distributions`, `plot_stationary_noise`, `run_eda`.
`IMU_ACCEL_COLS`, `IMU_GYRO_COLS`.

### `src/dead_reckoning/naive_drift_baseline.py`
| Name | Signature | Notes |
|---|---|---|
| `naive_double_integrate` | `(df, accel_col="accel_x_mps2") -> dict` | generic — reused with different `accel_col` every day since Day 3 |
| `gps_reference_distance` | `(df) -> np.ndarray` | ground truth |
| `evaluate_drift` | `(df, true_distance_m=None) -> dict` | — |
| `plot_drift` | `(eval_result, out_path, title="Day 2 — ...")` | `title` is a parameter as of Day 3 — always pass one explicitly |
| `log_metric` | `(metrics_path, eval_result)` | hardcodes "Day 2" method label — Day 3+ scripts write their own metrics row instead of calling this |

### `src/calibration/gravity_compensation.py` (Day 3)
`compute_linear_acceleration(df) -> pd.DataFrame`. `AXES = ["x","y","z"]`.

### `src/calibration/zupt_classifier.py` (Day 3)
`zupt_mask(df, accel_var_thresh=0.05, gyro_mag_thresh=0.02, window=5) -> np.ndarray`
(defaults not yet tuned against real data — see Section 5).
`evaluate_zupt_against_gps(mask, df, speed_thresh_kmh=1.0) -> dict`.
`LIN_ACCEL_COLS`, `GYRO_COLS`.

### `src/calibration/bias_estimation.py` (Day 3)
`estimate_constant_bias(df, mask, cols=BIAS_COLS) -> dict`.
`apply_bias_correction(df, bias) -> pd.DataFrame`. `BIAS_COLS` (the 6
lin_accel/gyro channels bias is estimated for).

### `src/calibration/mount_alignment.py` (Day 4 part 1 — LEVELING ONLY)
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `rotation_from_vectors` | `(a: np.ndarray, b: np.ndarray) -> np.ndarray` | 3×3 rotation matrix | general Rodrigues shortest-arc rotation; will be reused for heading in part 2 |
| `estimate_leveling_rotation` | `(df, stationary_mask) -> np.ndarray` | 3×3 rotation matrix (phone→vehicle, roll+pitch only, yaw ambiguous) | uses raw `gravity_{x,y,z}_mps2`, NOT the bias-corrected linear accel — leveling doesn't need bias correction |
| `recovered_roll_pitch_deg` | `(R_level) -> dict` | `{roll_deg, pitch_deg}` | sanity-check/logging only, not used downstream |
| `VEHICLE_UP` | `np.array([0,0,1.0])` | — | leveling target — see Section 2's sign-convention note |
| `GRAVITY_COLS` | `["gravity_x_mps2", "gravity_y_mps2", "gravity_z_mps2"]` | — | — |

**Functions that do NOT exist yet (Day 4 part 2, next session):**
`accelerating_mask()`, `estimate_heading_offset()`,
`estimate_mount_rotation()` (the part-1+part-2 combiner),
`apply_mount_alignment()`. Don't assume these are callable — check this
file's actual content first, since the registry will be updated once
they're built and verified.

### Not yet built (reserve these module paths)
| Phase | Expected file(s) |
|---|---|
| 3 (Days 6–8) | `src/models/velocity_model.py` |
| 4 (Days 9–10) | `src/dead_reckoning/strapdown_ins.py` |
| 5 (Days 11–12) | `src/map_matching/osm_graph.py`, `src/map_matching/hmm_matcher.py` |
| 6 (Days 13–14) | `src/fusion/ukf_fusion.py` |
| 7 (Day 15) | `src/edge_engine/`, `mobile_app/` |

---

## 4. Results Log Format (locked)

`results/metrics.md` — `Phase / Day | Method | Final position error (m) | Drift rate (m/min) | RMSE (m)`

Current rows (Day 2/3 numbers are from locally-built test drives, NOT
the user's real `SYNTH-drive1.csv` — re-run before trusting for
reporting):
- Day 2 — Naive double-integration — 41.99 m — 21.01 m/min — 20.71 m RMSE
- Day 3 — Gravity-comp + bias-corrected — 149.96 m — 75.04 m/min — 67.82 m RMSE (vs. 3213.93 m same-drive uncorrected)
- **Day 4 — not yet logged.** Part 1 (leveling) has no drift number of
  its own — leveling alone doesn't change `accel_x`'s forward-axis
  reading meaningfully until heading (part 2) is also applied and the
  full rotation is used to re-derive the forward axis.

---

## 5. Known Open Issues (carry forward until resolved)

- **`synthetic_data.py` still unconfirmed** — highest priority.
- **ZUPT threshold tuning** — poor precision (~6%) against GPS on the
  synthetic generator; needs smoother transitions + real-data retuning.
- **Straight-line-only synthetic drives** — blocks fully exercising
  heading estimation (Day 4 part 2) and is required before Phase 5.
- **Bias model is constant-only** — may need to become time-varying if
  real data shows bias drift.
- **NEW — gravity sign convention on real data unconfirmed:**
  `mount_alignment.py` assumes `gravity_z_mps2 ≈ +9.81` at rest
  (Android `TYPE_GRAVITY` standard convention). Check this against an
  actual stationary window in real IO-VNBD data before trusting
  `estimate_leveling_rotation()`'s output on it — if the real sign is
  flipped, either the phone was mounted screen-down for that drive, or
  `VEHICLE_UP` needs to become `VEHICLE_DOWN` in the code.
- **NEW — Day 4 part 1's own test drive (`data/raw/SYNTH-drive1.csv` as
  it currently sits in this sandbox) is not physically meaningful for
  leveling** — it's a leftover from Day 3's ad hoc bias-injection test,
  not a real or representative phone orientation. Leveling correctness
  was validated on a separate purpose-built known-tilt test instead
  (see PROGRESS.md Day 4). Re-run against the real `synthetic_data.py`
  or actual IO-VNBD data once available.

---

## How to use this file

1. Upload `PROGRESS.md`, `context.md`, and the specific source file(s)
   the next day's code will import from — `synthetic_data.py` is the
   highest-priority upload still outstanding.
2. After each day, update both: narrative in `PROGRESS.md`, exact
   names/signatures in Section 3 here.
3. Never rename an existing entry without updating every caller.