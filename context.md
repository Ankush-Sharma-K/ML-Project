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
an accurate position estimate using only the phone's IMU (accelerometer
+ gyroscope + magnetometer) during GPS-denied periods (tunnels, urban
canyons, parking structures), and hand off seamlessly back to GNSS when
signal returns.

**Approach (classical + ML hybrid, per problem statement):**
- Classical strapdown INS (orientation integration + Zero-velocity
  UPdaTe / ZUPT) as the mechanical backbone
- AI/ML correction layer: a learned forward-velocity model (replaces
  naive double-integration) + adaptive fusion noise estimation
- OSM road-network map matching with Non-Holonomic Constraints (a
  vehicle can't move sideways) to snap noisy DR output back onto plausible
  roads
- GNSS + INS sensor fusion via UKF/EKF, with seamless handoff between
  GNSS-available and GNSS-denied periods

**Dataset:** IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark
Dataset — Onyekpe, Palade, Kanarachos & Szkolnik, Data in Brief 2021).
Smartphone recordings (`S-*.csv`, AndroSensor app, 10 Hz IMU + 1 Hz GPS)
are the primary input; the sandbox has no internet, so a synthetic
generator matching the same schema is used for all dev/test work until
real data is run in a local/Colab environment.

**Deliverable (Day 15):** an edge-deployable engine + mobile app that
runs this whole pipeline on-device.

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
and can't `pip install torch`/`scikit-learn`/`filterpy` — classical
DSP/filter/Kalman code is built and tested here against synthetic data;
deep-learning training (Phase 3) and any run against real IO-VNBD data
happens in the user's own environment (local + internet, or Colab), with
code handed off ready to run.

---

## 2. Data Schema (locked — do not rename without updating this file)

Source of truth: `src/io_vnbd_loader.py`, function `load_smartphone_drive()`.
This is the **real, verified** schema (confirmed against the user's
actual file on 2026-09-23) — every downstream script must import these
exact names, not invent its own.

### Raw columns (as loaded from CSV, 24 cols + derived)

| Column | Meaning | Unit |
|---|---|---|
| `gps_lat_deg` | GPS latitude | degrees |
| `gps_lon_deg` | GPS longitude | degrees |
| `gps_alt_m` | GPS altitude | meters |
| `gps_speed_kmh` | GPS speed | km/h — present at full 10 Hz IMU rate, no separate fix flag |
| `gps_accuracy_m` | GPS horizontal accuracy | meters |
| `gps_orientation_deg` | GPS-derived heading | degrees |
| `gps_satellites_in_range` | satellite count | count |
| `time_since_start_ms` | raw elapsed time | milliseconds |
| `datetime` | wall-clock timestamp | parsed to pandas Timestamp (best-effort, NaT on failure) |
| `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2` | raw accelerometer (gravity-inclusive) | m/s² |
| `gravity_x_mps2`, `gravity_y_mps2`, `gravity_z_mps2` | phone's fused gravity estimate | m/s² |
| `gyro_yaw_rads`, `gyro_pitch_rads`, `gyro_roll_rads` | raw gyroscope | rad/s |
| `mag_x_ut`, `mag_y_ut`, `mag_z_ut` | raw magnetometer | µT |
| `orientation_yaw_deg`, `orientation_pitch_deg`, `orientation_roll_deg` | phone-fused orientation | degrees |

### Derived columns

| Column | Meaning | Added by |
|---|---|---|
| `t_sec` | seconds since drive start — **the canonical time axis used everywhere downstream, not `datetime` or `time_since_start_ms`** | `io_vnbd_loader.load_smartphone_drive()` |
| `lin_accel_x_mps2`, `lin_accel_y_mps2`, `lin_accel_z_mps2` | gravity-compensated linear acceleration, `= accel_*_mps2 - gravity_*_mps2` | `calibration/gravity_compensation.compute_linear_acceleration()` (Day 3) |
| `<col>_cal` (e.g. `lin_accel_x_mps2_cal`, `gyro_yaw_rads_cal`) | bias-corrected version of any BIAS_COLS channel, `= col - estimated_bias[col]` | `calibration/bias_estimation.apply_bias_correction()` (Day 3) |

**Naming convention to keep:** axis suffix (`_x/_y/_z` or `_yaw/_pitch/_roll`)
+ unit suffix (`_mps2`, `_rads`, `_deg`, `_ut`, `_kmh`, `_m`), derived
columns prefixed by what they are (`lin_`) and suffixed `_cal` once
bias-corrected. A **mount-aligned** (vehicle-frame) version of a channel,
once Day 4 builds it, should follow the same pattern — e.g.
`lin_accel_x_mps2_cal_veh`, not a new naming scheme.

---

## 3. Function Registry (grows every day — append, don't duplicate)

### `src/io_vnbd_loader.py`
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `load_smartphone_drive` | `(csv_path: str \| Path, header: int \| None = None) -> pd.DataFrame` | tidy DataFrame, schema above + `t_sec` | header=0 if CSV already has its own header row |
| `drive_summary` | `(df: pd.DataFrame) -> dict` | `{n_samples, duration_s, duration_min, median_dt_s, implied_rate_hz, gps_speed_max_kmh, gps_speed_mean_kmh}` | sanity check right after loading |
| `SMARTPHONE_COLUMNS` | module-level list | — | the 24 raw column names, in order |
| `SMARTPHONE_SAMPLE_RATE_HZ` / `SMARTPHONE_GPS_RATE_HZ` | module-level float | `10.0` / `1.0` | — |

### `src/utils/synthetic_data.py`
**⚠️ Still not verified against the user's real file — only reconstructed.**
Function name(s) and output shape (module-level constants, `df.attrs`
usage for ground truth, etc.) are unconfirmed. Upload this file before
any code assumes its interface — Day 3's numbers were validated against
a locally-built stand-in drive for exactly this reason (see PROGRESS.md
Day 3 caveat).

### `src/eda.py`
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `channel_stats` | `(df: pd.DataFrame) -> pd.DataFrame` | mean/std/min/max per IMU channel | uses `IMU_ACCEL_COLS`, `IMU_GYRO_COLS` |
| `stationary_mask` | `(df: pd.DataFrame, speed_thresh_kmh: float = 1.0) -> np.ndarray` | bool array | GPS-based; superseded for calibration purposes by `zupt_classifier.zupt_mask()` (Day 3), kept here for EDA/reference |
| `stationary_noise_floor` | `(df: pd.DataFrame, mask: np.ndarray) -> dict` | channel → std | sensor noise floor |
| `plot_distributions` | `(df, out_path)` | writes PNG | histograms of 6 IMU channels |
| `plot_stationary_noise` | `(df, mask, out_path)` | writes PNG | accel_x + GPS speed w/ stationary overlay |
| `run_eda` | `(df, plots_dir="results/plots") -> dict` | `{channel_stats, n_stationary_samples, pct_stationary, stationary_noise_floor_std}` | top-level entry point |
| `IMU_ACCEL_COLS` | `["accel_x_mps2", "accel_y_mps2", "accel_z_mps2"]` | — | module-level constant |
| `IMU_GYRO_COLS` | `["gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads"]` | — | module-level constant |

### `src/dead_reckoning/naive_drift_baseline.py`
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `naive_double_integrate` | `(df, accel_col="accel_x_mps2") -> dict` | `{t, velocity, position}` | zero correction; **reused as-is in Day 3** with `accel_col="lin_accel_x_mps2_cal"` — this is why the function takes `accel_col` as a parameter, don't fork it |
| `gps_reference_distance` | `(df: pd.DataFrame) -> np.ndarray` | cumulative distance from `gps_speed_kmh` | ground-truth fallback |
| `evaluate_drift` | `(df, true_distance_m=None) -> dict` | adds `true_distance_m, error, final_error_m, final_true_distance_m, drift_rate_m_per_min, rmse_position_m` to the integrate dict | pass `true_distance_m` explicitly if a better reference exists |
| `plot_drift` | `(eval_result, out_path, title="Day 2 — Naive Double-Integration Drift Baseline (zero correction)")` | writes PNG | **`title` is a parameter as of Day 3** — always pass an accurate one when reusing this for a later day, don't rely on the default |
| `log_metric` | `(metrics_path, eval_result)` | appends row | writes `results/metrics.md` — hardcodes the "Day 2" method label, so Day 3+ scripts write their own metrics-row `open()`/`write()` instead of calling this (see `bias_estimation.py`'s `__main__`) rather than editing this function's label |

### `src/calibration/gravity_compensation.py` (Day 3)
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `compute_linear_acceleration` | `(df: pd.DataFrame) -> pd.DataFrame` | copy of df + `lin_accel_{x,y,z}_mps2` | must run before `zupt_classifier` or `bias_estimation` |
| `AXES` | `["x", "y", "z"]` | — | module-level constant |

### `src/calibration/zupt_classifier.py` (Day 3)
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `zupt_mask` | `(df, accel_var_thresh=0.05, gyro_mag_thresh=0.02, window=5) -> np.ndarray` | bool array | requires `lin_accel_{x,y,z}_mps2` already present; defaults set from Day 2's noise floor, **not yet tuned against a real drive** — see PROGRESS.md Day 3 issue |
| `evaluate_zupt_against_gps` | `(mask, df, speed_thresh_kmh=1.0) -> dict` | `{n_zupt_flagged, n_gps_stationary, agreement_precision, agreement_recall}` | sanity cross-check only, GPS isn't ground truth either |
| `LIN_ACCEL_COLS` | `["lin_accel_x_mps2", "lin_accel_y_mps2", "lin_accel_z_mps2"]` | — | module-level constant |
| `GYRO_COLS` | `["gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads"]` | — | module-level constant |

### `src/calibration/bias_estimation.py` (Day 3)
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `estimate_constant_bias` | `(df, mask, cols=BIAS_COLS) -> dict` | `{col_name: bias_value}` | mean of each channel during ZUPT-flagged stationary windows |
| `apply_bias_correction` | `(df, bias: dict) -> pd.DataFrame` | copy of df + `<col>_cal` columns | — |
| `BIAS_COLS` | `["lin_accel_x_mps2", "lin_accel_y_mps2", "lin_accel_z_mps2", "gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads"]` | — | module-level constant — the 6 channels bias is estimated for |

### Not yet built (reserve these module paths, don't rename on arrival)
| Phase | Expected file(s) |
|---|---|
| 2 (Day 4–5) | `src/calibration/mount_alignment.py` |
| 3 (Days 6–8) | `src/models/velocity_model.py` |
| 4 (Days 9–10) | `src/dead_reckoning/strapdown_ins.py` |
| 5 (Days 11–12) | `src/map_matching/osm_graph.py`, `src/map_matching/hmm_matcher.py` |
| 6 (Days 13–14) | `src/fusion/ukf_fusion.py` |
| 7 (Day 15) | `src/edge_engine/`, `mobile_app/` |

---

## 4. Results Log Format (locked)

`results/metrics.md` — one row per phase/day, columns:
`Phase / Day | Method | Final position error (m) | Drift rate (m/min) | RMSE (m)`

Current rows (numbers are from a locally-built test drive with a known
injected bias, NOT the project's real `SYNTH-drive1.csv` — see PROGRESS.md
Day 3 caveat; re-run against the real drive before trusting these for
reporting):
- Day 2 — Naive double-integration (zero correction) — 41.99 m — 21.01 m/min — 20.71 m RMSE *(original run, no injected bias)*
- Day 3 — Gravity-compensated + bias-corrected double integration — 149.96 m — 75.04 m/min — 67.82 m RMSE *(on a separate, bias-injected test drive; ~20x better than the same-drive uncorrected baseline of 3213.93 m)*

---

## 5. Known Open Issues (carry forward until resolved)

- **`synthetic_data.py` still unconfirmed:** function names, output
  columns, and any `df.attrs` ground-truth convention are guessed, not
  verified. Highest-priority file to upload next.
- **ZUPT threshold tuning:** `zupt_classifier.zupt_mask()` defaults
  (accel_var 0.05, gyro_var 0.02) reach poor precision (~6%) against
  GPS on the current synthetic generator, because its abrupt
  accel/brake transitions inflate variance right at the true stop
  instant. Needs (a) a smoother synthetic transition profile and (b)
  retuning against real IO-VNBD data.
- **Straight-line-only synthetic drives:** no turning/curved roads yet
  — blocks fully exercising Day 4's mount-alignment heading correlation
  and is required before Phase 5's map matcher.
- **Bias model is constant-only:** if real data shows the bias itself
  drifting slowly over a long drive, `estimate_constant_bias()` will
  need to become windowed/time-varying rather than a single scalar per
  channel.

---

## How to use this file

1. At the start of a new day's session, upload `PROGRESS.md`,
   `context.md`, and the specific source file(s) that day's code will
   import from (per the Function Registry above) — `synthetic_data.py`
   is the highest-priority upload still outstanding.
2. After the day's work, update **both**: add the narrative to
   `PROGRESS.md`, and add any new functions/columns to Section 3 here
   (or move an entry out of "Not yet built" once it exists).
3. Never rename an existing entry in Section 2 or 3 without updating
   every caller — that's the whole point of this file.