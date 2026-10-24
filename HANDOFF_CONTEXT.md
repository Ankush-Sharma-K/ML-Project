# IDR Project — Handoff Context for New Chat

> Paste this entire file at the start of a new chat. Days 1–10 are done (see the Day 8 audit parts 1–3, Day 9 and Day 10 at the end of PROGRESS.md); continue from Phase 5 (map matching).

---

## Project Identity

**Name:** AI/ML-based Intelligent Dead Reckoning (IDR) — M.Tech 1st Sem ML Project  
**Goal:** Seamless navigation in GPS-denied tunnels using smartphone IMU + ML  
**Root:** `d:\M.Tech\1st Sem\Machine Learning\ML Project\idr-project\`  
**Venv Python:** `& "d:\M.Tech\1st Sem\Machine Learning\ML Project\mlproject\Scripts\python.exe"`  
**Run modules as:** `& "...\mlproject\Scripts\python.exe" -m src.MODULE` from inside `idr-project\`

---

## Current Status: Day 10 complete ✅ (Days 1–10 verified on the real drives)

| Phase | Status |
|-------|--------|
| Phase 1 — Setup, EDA, naive baseline | ✅ Complete |
| Phase 2 — Calibration (gravity, ZUPT, bias, mount, denoise) | ✅ Complete |
| Phase 3 — Dataset pipeline + CNN velocity model architecture | ✅ Complete |
| Model training (run_training.py) | ✅ 20 s-window deep CNN on real drives — test MAE 4.53 m/s (per-drive 4.71) |
| Phase 4 — Strapdown INS + ZUPT | ✅ Complete — honest simulated-outage benchmark |
| Day 9 — Anchored ML velocity in outages | ✅ Complete — `ins_with_ml_velocity.py` |
| Day 10 — Gyro heading + 2-D path in outages | ✅ Complete — `heading.py` |
| Phase 5 — Map Matching | ❌ Not started |
| Phase 6 — GNSS+INS Fusion | ❌ Not started |
| Phase 7 — App & Deployment | ❌ Not started |

---

## All Source Files — Current State

### ✅ `src/io_vnbd_loader.py`
- Loads IO-VNBD smartphone CSV files (header=0, encoding=iso-8859-1)
- **Schema auto-detection:** accepts 24-col (full) OR 18-col (no mag/orientation, e.g. S-T*.csv)
- **Trailing comma fix:** drops `Unnamed: N` columns before schema detection
- **Numeric coercion:** forces all non-datetime columns to float (fixes S-A4 string dtype)
- **Column-shift repair:** detects when `time_since_start_ms` is all-NaN → reconstructs from raw column positions
- **Debug prints:** REMOVED (no more per-drive datetime spam)
- Key constants: `SMARTPHONE_COLUMNS` (24), `SMARTPHONE_COLUMNS_18` (18)

### ✅ `src/calibration/gravity_compensation.py`
- `compute_linear_acceleration(df)` → adds `lin_accel_{x,y,z}_mps2` columns
- Clean, no issues

### ✅ `src/calibration/zupt_classifier.py`
- `zupt_mask(df)` → IMU-native stationary detection
- `zupt_mask_with_fallback(df)` → GPS-speed-gated fallback (used for real data)
- Clean, no issues

### ✅ `src/calibration/bias_estimation.py`
- `estimate_constant_bias(df, mask)` → mean IMU values during stationary window
- `apply_bias_correction(df, bias)` → adds `_cal` suffix columns
- Clean, no issues

### ✅ `src/calibration/mount_alignment.py`
- `accelerating_mask()` — **FIXED:** replaced `np.gradient(speed,t)` with safe forward-diff + epsilon guard (was divide-by-zero on duplicate timestamps)
- `estimate_mount_rotation(df, mask)` → returns `{R_align, heading_deg}`
- `apply_mount_alignment(df, R_align)` → adds `lin_accel_{x,y,z}_mps2_cal_veh`

### ✅ `src/calibration/denoising.py`
- `denoise_lowpass(df, cols)` → zero-phase Butterworth LPF
- Clean, no issues

### ✅ `src/dead_reckoning/naive_drift_baseline.py`
- **FIXED:** removed `dt[0] = dt[1]` in both `naive_double_integrate()` and `gps_reference_distance()`
- Drift benchmark Day 2–5 logged in `results/metrics.md`

### ✅ `src/dead_reckoning/strapdown_ins.py` ← NEW Phase 4
- `StrapdownINS(df, zupt_mask, zupt_mode, decay_rate)` class
- `.integrate()` → 1-D (forward/x axis) integration with ZUPT
- `.integrate_3d()` → full 3-axis integration
- `evaluate_ins_against_gps()` → final error, max error, RMSE
- `compare_zupt_modes()` → runs no/hard/soft ZUPT side-by-side
- **Phase 4 benchmark on S-S1.csv:** Hard ZUPT = 47,881 m final error (97.5% better than no ZUPT)

### ✅ `src/models/dataset_prep.py`
- `build_windows(df, feature_cols, target_col, window_size=20, stride=5)` → sliding windows
- `_window_is_clean()` → rejects windows with bad dt gaps
- Clean, no issues

### ✅ `src/models/velocity_model.py`
- `FEATURE_COLS_DEFAULT` = `["lin_accel_x_mps2_cal_veh", "lin_accel_y_mps2_cal_veh", "lin_accel_z_mps2_cal_veh", "gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads"]`
- `VelocityCNN` = Conv1d(6→32) → Conv1d(32→64) → AvgPool → FC(64→32) → FC(32→1)
- `compute_normalization_stats()`, `apply_normalization()`
- Clean, no issues

### ✅ `src/models/train_velocity_model.py`
- **FIXED:** replaced fragile `@torch.no_grad() if torch is not None else (lambda f: f)` with `with ctx:` + `_null_context()` helper
- `train_velocity_model(train_windows, val_windows, ...)` → trains with early stopping
- Checkpoint format: `{model_state_dict, norm_stats, feature_cols, window_size, epoch, val_loss}`

### ✅ `src/models/multi_drive_data.py`
- `load_and_calibrate_drive(csv_path)` → full Days 1-4 pipeline for one file
- `build_windows_for_drives(csv_paths, feature_cols, ...)` → loops all drives with try/except
- `train_val_split_by_drive(windows, val_fraction=0.2)` → splits by whole drive

---

> The per-file notes above describe the pre-audit state; see `context.md` §3 "Day 8 audit" for the current function registry.

---

## Column Name Conventions (critical — must match exactly)

| Stage | Columns |
|-------|---------|
| Raw (after loader) | `accel_{x,y,z}_mps2`, `gravity_{x,y,z}_mps2`, `gyro_{yaw,pitch,roll}_rads`, `t_sec` (resets unwrapped) |
| Speed (after loader) | `gps_speed_kmh` = TRUE km/h, `gps_speed_ms` (held, m/s), `gps_speed_interp_ms` (interpolated between fixes — use this) |
| After gravity comp | `lin_accel_{x,y,z}_mps2` |
| After bias correction | `lin_accel_{x,y,z}_mps2_cal`, `gyro_{yaw,pitch,roll}_rads_cal` |
| After mount alignment | `lin_accel_{x,y,z}_mps2_cal_veh` |

**Never divide the raw CSV speed by 3.6** — 94/97 files store m/s despite the "Kmh" header; the loader handles units.

---

## Dataset Facts (IO-VNBD real data, after the Day 8 audit)

- **Total files:** 97 `S-*.csv` in `data/raw/real/`
- **GPS rate:** ~0.1 Hz in most files (1 Hz in S-A4..A8, S-Vta1a, S-Vta2), held between fixes
- **Drives loaded OK:** 75 · **Failed:** 14 (under a minute long or never moving) · **Rate-excluded:** 8 (2 Hz S-A*)
- **Windows (20 s, stride 1 s):** train 109,893 (54 drives) · val 29,925 (10) · test 32,165 (9)
- **Split is FIXED in `data/split.json`** — always reuse it
- **Val drives:** S-M, S-S2, S-T6, S-Vta26, S-Vta30, S-Vta5, S-Vw16a, S-Vw16b, S-Vw3, S-Y1
- **Test drives:** S-A8, S-T7, S-Vta10, S-Vta13, S-Vta21, S-Vta29, S-Vtb7, S-Vw4, S-Vw5

---

## Benchmark (results/metrics.md)

**Evaluation rules:** calibration only from data before each outage; entry
speed = last real GPS fix; truth = car wheel speed where a vehicle file aligns
(`src/utils/vehicle_truth.py`, files in `data/raw/vehicle/`), else GPS interpolated.

- Day 8, 57 drives (median distance error at 30 / 60 / 120 s): hold last fix 56 / 151 / 314 m; INS + ZUPT 107 / 286 / 784 m.
- Day 9, TEST drives: hold 74 / 166 / 361 m; ML 87 / 136 / 221 m; **anchored ML 74 / 113 / 206 m**.
- Cross-validated against wheel truth (39 drives): anchored ML 47 / 88 / 176 m vs hold 61 / 141 / 315 m.
- Day 10 path (end-position error at 30 / 60 / 120 s): heading only, true distance — straight line 98 / 239 / 651 m, gyro 59 / 157 / 439 m; full path on test drives — baseline 120 / 333 / 931 m, anchored ML + gyro 94 / 207 / 747 m.

---

## Trained Model

**Model:** `VelocityCNNDeep` (channels 64-128-128-256-256) — 200-sample (20 s) IMU window, 8 features → forward speed (m/s); chosen by 5-fold CV (3.42 ± 0.63 m/s vs 4.73 for the old 2 s CNN)
**Checkpoint:** `results/checkpoints/velocity_cnn_best.pt` (weights + norm stats + feature_cols + window_size)
**Result:** val MAE 2.50 m/s, **test MAE 4.53 m/s** (per-drive mean 4.71) vs 8.05 for predicting the mean; old 2 s model on the same test drives: 5.15 / 5.42
**Load with:** `train_velocity_model.load_model(path, device)` (rebuilds the right architecture)
**Details:** `results/training_summary.json`, `results/training_log.txt`, `results/plots/training_curves.png`
`velocity_model_synthetic.pt` is only the synthetic-drive demo — do not use it.

---

## Next Steps for the Agent

### Done in Day 10
- `src/dead_reckoning/heading.py`: per-phone calibrated gyro heading (axis order/signs/scale fitted on pre-outage GPS), 2-D path, benchmark `python -m src.dead_reckoning.heading`
- Phase 5 input: east/north path from anchored-ML speed + gyro heading (`simulate_path_outages` shows how to build it)

### Done in Day 9
- `src/dead_reckoning/ins_with_ml_velocity.py`: anchored ML, speed = ML(t) + (v0 - ML(0)) * exp(-t / 20 s)
- Run `python -m src.dead_reckoning.ins_with_ml_velocity` (TEST drives) and `python -m src.dead_reckoning.strapdown_ins` (Day 8, all drives)

### Phase 5 — Map Matching (Days 11–12)
Create `src/map_matching/`:
- `osm_loader.py` — download road graph from OpenStreetMap via osmnx
- `nhc_constraint.py` — Non-Holonomic Constraint (vehicle can't move sideways)
- `hmm_map_matcher.py` — Hidden Markov Model road snapping

### Phase 6 — GNSS+INS Fusion (Days 13–14)
Create `src/fusion/`:
- `kalman_filter.py` — UKF or EKF base
- `gnss_ins_fusion.py` — seamless GPS↔DR handoff

### Phase 7 — App & Deployment (Day 15)
- `src/edge_engine/` — lightweight inference API
- `mobile_app/` — UI scaffold
