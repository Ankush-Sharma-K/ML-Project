# context.md — Project Context & Interface Registry

> **Purpose of this file:** `PROGRESS.md` tells you what was done and why,
> narratively. This file tells you the exact names of every function,
> column, and variable that already exist, so new code plugs into old
> code without guessing or colliding. Upload **both** files (plus any
> source file the next day's code will directly import from) at the
> start of every new day's session.

---

## 1. Project Aim

**Goal:** AI/ML-based Intelligent Dead Reckoning (IDR) — accurate
position on a smartphone using only its IMU during GPS-denied periods,
seamless handoff back to GNSS when signal returns.

**Approach:** classical strapdown INS (orientation integration + ZUPT)
+ a learned forward-velocity model + adaptive fusion noise, OSM map
matching with Non-Holonomic Constraints, GNSS+INS fusion via UKF/EKF.

**Dataset:** IO-VNBD. No internet in this sandbox — a synthetic
generator matching the same schema is used for dev/test.

**Plan:**

| Phase | Days | Focus | Status |
|---|---|---|---|
| 1 — Setup & Dataset Understanding | 1–2 | Repo scaffold, IO-VNBD loader, EDA, naive drift baseline | done |
| 2 — Calibration & Preprocessing | 3–5 | Mount alignment, bias estimation, denoising, ZUPT classifier | done |
| 3 — AI Velocity Estimation | 6–8 | IMU→velocity model, robustness, on-device compression | next |
| 4 — Core Dead Reckoning | 9–10 | Strapdown INS integration, drift benchmark evaluation | — |
| 5 — Map Matching | 11–12 | OSM road graph, NHC constraints, HMM map matcher | — |
| 6 — GNSS+INS Fusion | 13–14 | UKF/EKF fusion, adaptive noise, seamless GNSS↔DR handoff | — |
| 7 — App & Deployment | 15 | Edge engine API, mobile app UI, packaging, final submission | — |

**Constraint carried through every phase:** no internet / no
`pip install torch/scikit-learn/filterpy` in this sandbox — Phase 3
training and real IO-VNBD runs happen in the user's own environment.

---

## 2. Data Schema (locked)

Source of truth: `src/io_vnbd_loader.py`, `load_smartphone_drive()`.

### Raw columns
`gps_lat_deg`, `gps_lon_deg`, `gps_alt_m`, `gps_speed_kmh` (full 10 Hz
rate), `gps_accuracy_m`, `gps_orientation_deg`, `gps_satellites_in_range`,
`time_since_start_ms`, `datetime`, `accel_{x,y,z}_mps2` (raw,
gravity-inclusive), `gravity_{x,y,z}_mps2` (≈(0,0,+9.81) when level,
confirmed convention), `gyro_{yaw,pitch,roll}_rads`, `mag_{x,y,z}_ut`,
`orientation_{yaw,pitch,roll}_deg`.

### Derived columns

| Column | Meaning | Added by |
|---|---|---|
| `t_sec` | canonical time axis | `io_vnbd_loader.load_smartphone_drive()` |
| `lin_accel_{x,y,z}_mps2` | gravity-compensated | `gravity_compensation.compute_linear_acceleration()` (Day 3) |
| `<col>_cal` | bias-corrected | `bias_estimation.apply_bias_correction()` (Day 3) |
| `lin_accel_{x,y,z}_mps2_cal_veh` | vehicle-frame (mount-aligned): forward/lateral/up | `mount_alignment.apply_mount_alignment()` (Day 4) |
| `<col>_dn` | low-pass denoised (any column passed to it) | `denoising.denoise_lowpass()` (Day 5) — e.g. `lin_accel_x_mps2_cal_veh_dn` |

**Naming convention:** axis suffix + unit suffix, `_cal` once
bias-corrected, `_veh` once rotated into the vehicle frame, `_dn` once
low-pass filtered. Suffixes stack in the order they're applied
(`_cal_veh_dn`, not `_dn_cal_veh`).

---

## 3. Function Registry

### `src/io_vnbd_loader.py`
`load_smartphone_drive(csv_path, header=None) -> pd.DataFrame`.
`drive_summary(df) -> dict`.

### `src/utils/synthetic_data.py`
**⚠️ Still not verified against the user's real file.** Every day so
far (1–5) has been tested against a locally-built stand-in, not the
project's real generator. Highest-priority upload still outstanding.

### `src/eda.py`
`channel_stats`, `stationary_mask`, `stationary_noise_floor`,
`plot_distributions`, `plot_stationary_noise`, `run_eda`.

### `src/dead_reckoning/naive_drift_baseline.py`
| Name | Signature | Notes |
|---|---|---|
| `naive_double_integrate` | `(df, accel_col="accel_x_mps2") -> dict` | generic — reused with a different `accel_col` every day since Day 3 |
| `gps_reference_distance` | `(df) -> np.ndarray` | ground truth |
| `evaluate_drift` | `(df, true_distance_m=None) -> dict` | — |
| `plot_drift` | `(eval_result, out_path, title=...)` | always pass `title` explicitly |
| `log_metric` | `(metrics_path, eval_result)` | hardcodes "Day 2" label — later days write their own metrics row |

### `src/calibration/gravity_compensation.py` (Day 3)
`compute_linear_acceleration(df) -> pd.DataFrame`.

### `src/calibration/zupt_classifier.py` (Day 3)
`zupt_mask(df, accel_var_thresh=0.05, gyro_mag_thresh=0.02, window=5) -> np.ndarray`
(defaults not yet tuned against real data). `evaluate_zupt_against_gps()`.

### `src/calibration/bias_estimation.py` (Day 3)
`estimate_constant_bias(df, mask, cols=BIAS_COLS) -> dict`.
`apply_bias_correction(df, bias) -> pd.DataFrame`.

### `src/calibration/mount_alignment.py` (Day 4)
`rotation_from_vectors`, `estimate_leveling_rotation`,
`recovered_roll_pitch_deg` (diagnostic only, unreliable when true
yaw≠0), `rotation_about_z`, `accelerating_mask`,
`estimate_heading_offset` (needs `_cal` columns), `estimate_mount_rotation`
(combines leveling+heading), `apply_mount_alignment` → produces
`lin_accel_{x,y,z}_mps2_cal_veh`.

### `src/calibration/denoising.py` (Day 5 — NEW)
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `characterize_frequency_content` | `(signal: np.ndarray, sample_rate_hz: float) -> dict` | `{f90_hz, f95_hz, f99_hz, nyquist_hz, top_power_bins}` | FFT-based — run this before trusting a cutoff on new/real data |
| `denoise_lowpass` | `(df, cols: list[str], sample_rate_hz=10.0, cutoff_hz=1.0, order=4) -> pd.DataFrame` | df + `<col>_dn` for each input col | zero-phase (`filtfilt`) — deliberately not causal, see docstring |
| `DEFAULT_CUTOFF_HZ` | `1.0` | — | chosen from FFT evidence on the Day 4/5 test drive — **re-derive for real data, don't assume it transfers** |

**⚠️ Important finding, not just a function to know about:**
denoising barely helps double-integration drift (0.2% on the test
drive) — see PROGRESS.md Day 5 for the full explanation (low-frequency/
near-DC noise, not high-frequency jitter, drives integration drift).
Don't expect `denoise_lowpass()` to fix drift numbers in Phase 4 either
— it's still useful for Phase 3's model input and any variance-based
detector, just not for this.

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

Current rows (all from locally-built test drives, NOT the user's real
`SYNTH-drive1.csv` — re-run before trusting for reporting):
- Day 2 — Naive double-integration — 41.99 m — 21.01 m/min — 20.71 m RMSE
- Day 3 — Gravity-comp + bias-corrected — 149.96 m — 75.04 m/min — 67.82 m RMSE (vs. 3213.93 m same-drive uncorrected)
- Day 4 — Mount-aligned (vehicle-frame) — 209.54 m — 104.86 m/min — 92.62 m RMSE (vs. 254.29 m same-drive without mount alignment)
- Day 5 — Mount-aligned + denoised — 209.13 m — 104.65 m/min — 92.39 m RMSE (vs. 209.54 m same-drive without denoising — a 0.2% change, see the Day 5 finding above)

Day-to-day absolute numbers aren't directly comparable (different test
drives, different injected errors) — the meaningful comparisons are
same-drive, same-day "vs. without this step" figures.

---

## 5. Known Open Issues (carry forward until resolved)

- **`synthetic_data.py` still unconfirmed** — highest priority, every
  day (1–5) has been verified against a stand-in, not the real file.
- **ZUPT threshold tuning** — poor precision (~6%) against GPS; needs
  smoother transitions + real-data retuning.
- **Straight-line-only synthetic drives** — main generator still
  doesn't produce turns (Day 4's heading test needed a separate
  purpose-built drive for this reason).
- **Bias model is constant-only** — may need to become time-varying if
  real data shows bias drift.
- **Gravity sign convention on real data unconfirmed** — code assumes
  `gravity_z_mps2 ≈ +9.81` at rest; check against a real stationary
  window before trusting on actual data.
- **RESOLVED/REVISED — Day 4's "noise-dominated drift" claim:**
  confirmed the diagnosis (not a calibration problem) but corrected the
  implied fix (denoising ≠ drift fix — see Section 3's denoising entry
  and PROGRESS.md Day 5). The actual fix is Days 9–10's ZUPT-reset
  strapdown INS.
- **1.0 Hz denoising cutoff is drive-specific** — derived from this
  synthetic generator's smooth drive-cycle timescale; re-run
  `characterize_frequency_content()` on real/different data before
  reusing it.

---

## How to use this file

1. Upload `PROGRESS.md`, `context.md`, and the specific source file(s)
   the next day's code will import from — `synthetic_data.py` is the
   highest-priority upload still outstanding.
2. After each day, update both: narrative in `PROGRESS.md`, exact
   names/signatures in Section 3 here.
3. Never rename an existing entry without updating every caller.