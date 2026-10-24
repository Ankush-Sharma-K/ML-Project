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

**Dataset:** IO-VNBD. Real dataset uploaded and analyzed. **Folder
structure clarified this session:** both zips have a messy
per-driver "Categorised" structure AND a flat "Uncategorised" folder
of `S-*.csv` directly — use the flat ones. The Unsynchronised flat
folder (97 files) is a superset of the Synchronised flat folder's 72
(same drives, same filenames) but NOT byte-identical — Synchronised
versions are trimmed to their paired V-file's time window. **Use
Unsynchronised's 97 files for training; keep Synchronised's S+V pairs
for later evaluation against real wheel-speed ground truth.**

**Plan:**

| Phase | Days | Focus | Status |
|---|---|---|---|
| 1 — Setup & Dataset Understanding | 1–2 | Repo scaffold, IO-VNBD loader, EDA, naive drift baseline | done |
| 2 — Calibration & Preprocessing | 3–5 | Mount alignment, bias estimation, denoising, ZUPT classifier | done |
| 3 — AI Velocity Estimation | 6–8 | IMU→velocity model, robustness, on-device compression | done: model trained on 75 real drives (Day 8 audit); int8 on-device export (completion pass) |
| 4 — Core Dead Reckoning | 9–10 | Strapdown INS integration, drift benchmark evaluation | strapdown INS + outage benchmark done (Day 8 audit) |
| 5 — Map Matching | 11–12 | OSM road graph, NHC constraints, HMM map matcher | done (Days 11–12); NHC turn-speed tested, not adopted |
| 6 — GNSS+INS Fusion | 13–14 | UKF/EKF fusion, adaptive noise, seamless GNSS↔DR handoff | done (Day 13): EKF + gate + RTS |
| 7 — App & Deployment | 15 | Edge engine API, mobile app UI, packaging, final submission | done (Day 15) |

**Environment:** everything (training, benchmarks, map downloads) runs in the
user's own venv (`mlproject`, torch with CUDA). The early "no internet / no torch
sandbox" constraint no longer applies.

**Where the user's real data lives:** `data/raw/real/` (flat folder of
`S-*.csv` files, relative to the project root) — NOT an absolute path,
deliberately, so the same code works regardless of machine. See
`multi_drive_data.py`'s module docstring for the full reasoning.

---

## 2. Data Schema (locked)

Unchanged — see prior context.md content. One addition confirmed this
session: **real files need `header=0` when calling
`load_smartphone_drive()`** (they have their own header row; the
synthetic generator's output doesn't) — confirmed by testing directly,
not assumed. No special encoding handling needed: the ISO-8859-1
encoding only affects the header row's degree/mu symbols, which
`header=0` skips past.

---

## 3. Function Registry

*(Only the new entry shown in full; everything through Day 6 part 3 is
unchanged from the prior version of this file.)*

### `src/models/multi_drive_data.py` (Day 7 start — NEW)
| Name | Signature | Returns | Notes |
|---|---|---|---|
| `load_and_calibrate_drive` | `(csv_path: str \| Path) -> (pd.DataFrame, dict)` | calibrated df + `{n_samples, n_stationary, pct_stationary, heading_offset_deg, n_accelerating_samples}` | runs the full Days 1-4 pipeline on ONE real file; raises if ZUPT finds zero stationary samples or mount alignment finds no accelerating phase (caller must catch — see `build_windows_for_drives`) |
| `build_windows_for_drives` | `(csv_paths: list, feature_cols: list[str], target_col="gps_speed_ms", window_size=20, stride=5) -> dict` | `{X, y, start_idx, drive_id, drive_paths, n_rejected, per_drive_diagnostics, failed_drives}` | loops `load_and_calibrate_drive` + `dataset_prep.build_windows` over every path; **one bad drive is skipped (logged in `failed_drives`), never fatal to the whole run** — verified with a deliberately malformed test file |
| `train_val_split_by_drive` | `(windows: dict, val_fraction=0.2) -> (train: dict, val: dict)` | — | splits by WHOLE drive (last `val_fraction` of drives, by order in `drive_paths`), not by time — see docstring for why this is the correct generalization of Day 6 part 1's split for multiple files |

**⚠️ Verification status:**
- Tested end-to-end using the one real sample already in this sandbox,
  copied under 3 different filenames to exercise the multi-file LOOP
  logic (not 3 genuinely different drives — content is identical, this
  only tests the orchestration code, not whether results differ across
  real drives). Confirmed: correct concatenation (3×1355=4065 total
  windows), correct per-drive diagnostic logging, correct whole-drive
  split (exactly 1 entire drive held out, matching window counts).
- Tested the failure path with one deliberately malformed CSV: skipped
  with a clear, specific error message; the other 3 drives still
  processed correctly. The try/except guard is confirmed working, not
  just assumed safe.
- **NOT yet tested:** actual behavior across genuinely different real
  drives (only one real sample exists in this sandbox) — in
  particular, whether `load_and_calibrate_drive`'s exceptions
  (stationary-sample / accelerating-phase guards) trigger often or
  rarely across the full 97-file real dataset is unknown until the
  user runs it.
- **A finding to watch, not yet a confirmed problem:** on the one real
  sample tested, ZUPT found only 104/6813 stationary samples and
  mount alignment's heading came out as 139.65° — a visible outlier.
  Not necessarily wrong (different drives may have different real
  mounts), but worth checking the DISTRIBUTION of these numbers once
  the user runs this against all 97 real files — if most drives show
  similarly low stationary-sample counts or wildly varying headings,
  that's confirmation (not just suspicion) that Day 3's ZUPT
  thresholds need real retuning before Phase 3 training is trustworthy.

### Day 8 audit — new / changed entries (all verified on the 97 real drives)

**Column contract change:** after `load_smartphone_drive()`, `gps_speed_kmh`
is TRUE km/h (the raw files store m/s in 94 of 97 drives despite the
header). The loader also adds `gps_speed_ms` (held value, m/s) and
`gps_speed_interp_ms` (interpolated between GPS fixes, m/s — use this as
the speed reference / training target). `df.attrs` carries
`speed_unit_in_file`, `n_logger_duplicates_dropped`, `n_time_resets_repaired`.

| File | Name | Signature | Notes |
|---|---|---|---|
| `io_vnbd_loader.py` | `detect_speed_unit` | `(df) -> "mps" \| "kmh"` | compares integrated speed with haversine GPS track |
| `io_vnbd_loader.py` | `gps_track_length_m` | `(df) -> float` | haversine over fix rows |
| `io_vnbd_loader.py` | `interpolate_gps_speed` | `(df) -> np.ndarray` | linear between fix instants |
| `zupt_classifier.py` | `calibration_stationary_mask` | `(df, speed_thresh_ms=0.3, min_duration_s=3.0, window=20) -> mask` | GPS-confirmed sustained + quiet stops; used for bias/leveling |
| `zupt_classifier.py` | `zupt_mask_adaptive` | `(df, reference_mask, moving_mask=None, window=30, k_accel=1.5, k_gyro=2.0, moving_percentile=5.0, min_reference_samples=100) -> (mask, thresholds)` | IMU-only ZUPT scaled to the phone's own noise floor, capped below moving values |
| `mount_alignment.py` | `estimate_heading_offset_gps` | `(df, R_level, ...) -> {heading_rad, n_intervals, r_forward, r_lateral}` | Procrustes between IMU and GPS velocity change over fix intervals |
| `mount_alignment.py` | `estimate_mount_rotation` | adds `heading_method="auto"`; returns also `heading_method`, `leveled_on_stationary`, `heading_fit` | empty stationary mask -> level on whole-drive gravity |
| `multi_drive_data.py` | `load_and_calibrate_drive` | diagnostics add `bias_calibrated`, `heading_method`, `heading_r_forward/lateral`, `speed_unit_in_file`, `calibration_stationary_mask` | non-stopping drives no longer rejected |
| `multi_drive_data.py` | `build_windows_for_drives` | default `target_col="gps_speed_interp_ms"` | |
| `multi_drive_data.py` | `train_val_test_split_by_drive` | `(windows, val_fraction=0.15, test_fraction=0.15, seed=42) -> (train, val, test)` | seeded, by whole recording (S-S3a/b/c kept together) |
| `velocity_model.py` | `FEATURE_COLS_DEFAULT` | gyro channels now `gyro_*_rads_cal` | |
| `train_velocity_model.py` | `evaluate` | `(..., return_mae=False)` | MAE in m/s alongside Huber loss |
| `train_velocity_model.py` | `predict` | `(model, windows, norm_stats, device, batch_size) -> np.ndarray` | |
| `strapdown_ins.py` | `StrapdownINS.integrate` | `(initial_velocity=0.0, no_reverse=False)` | |
| `strapdown_ins.py` | `simulate_outages` | superseded — see "Day 8 audit part 3" below | |

**Model selection (part 2 of the audit):** `velocity_model.py` adds
`FEATURE_COLS_BASE` (the 6 original columns), `FEATURE_COLS_DEFAULT` = base +
`lin_accel_horiz_mag_mps2`, `gyro_mag_rads` (added in `load_and_calibrate_drive`),
`DEFAULT_WINDOW_SIZE = 200`, `DEFAULT_ARCH = "deep"`, `DEFAULT_ARCH_KWARGS`,
class `VelocityCNNDeep(n_features, channels)`, `build_model(arch, n_features, **kw)`.
`train_velocity_model.py` adds `train_velocity_model(..., arch, arch_kwargs, weight_decay)`,
`load_model(path, device) -> (model, ckpt)`; checkpoints now store `arch`, `arch_kwargs`.
`multi_drive_data.split_windows_by_names(windows, split)` applies the FIXED split in
`data/split.json` (always use it; do not re-derive the split).

**Checkpoints:** `results/checkpoints/velocity_cnn_best.pt` = real-data model
(from `run_training.py`). `velocity_model_synthetic.pt` = the 120 s
synthetic-drive demo model from `train_velocity_model.py`'s `__main__` —
this was the only checkpoint before the audit; it was never a real-data model.

### Day 8 audit part 3 + Day 9 — honest outage evaluation, wheel-speed truth, anchored ML

**Rule for every outage evaluation:** use only what the phone has when GPS is
lost — calibration from the rows BEFORE the outage, entry speed = last real GPS
fix (`gps_speed_ms`, held), never the interpolated speed at the start (it
interpolates towards the next fix, inside the outage). Training still uses
whole-drive calibration (offline, legitimate).

| File | Name | Signature | Notes |
|---|---|---|---|
| `multi_drive_data.py` | `load_raw_drive` | `(csv_path) -> df` | loader + gravity compensation, no calibration |
| `multi_drive_data.py` | `calibrate` | `(df, allow_leveling_only=False) -> {bias, R_align, stationary_mask, heading_offset_deg, heading_method, heading_fit}` | pass `df.iloc[:start]` for before-outage calibration; `heading_method="none"` = leveling only |
| `multi_drive_data.py` | `apply_calibration` | `(df, calib) -> df` | adds `_cal`, `_cal_veh`, `lin_accel_horiz_mag_mps2`, `gyro_mag_rads` |
| `multi_drive_data.py` | `load_and_calibrate_drive` | unchanged signature | now = `load_raw_drive` + `calibrate` + `apply_calibration`; verified bit-identical output |
| `utils/vehicle_truth.py` | `load_wheel_speed_truth` | `(phone_df, drive_name, vehicle_dir) -> (truth or None, info)` | wheel speed (m/s) aligned to phone rows; per-drive lag from speed cross-correlation (±60 s); rejects unclear alignments |
| `strapdown_ins.py` | `simulate_outages` | `(df_raw, outage_lengths_s=(30,60,120), n_starts=15, truth=None, min_start_speed_ms=3.0, seed=0, history_rows=0, extra_methods=None) -> DataFrame` | per-start calibration before the outage; methods `hold_speed`, `ins`, `ins_zupt` (+ extra) |
| `strapdown_ins.py` | `summarize_outages`, `load_truth`, `report` | see docstrings | `load_truth` -> (wheel truth or None, "wheel"/"gps") |
| `strapdown_ins.py` | `log_metric` | `(results_path, label, metrics, day="Day 8", section_title=...)` | one section per day in metrics.md; replaces same-label rows |
| `ins_with_ml_velocity.py` | `ml_speed_series` | `(model, ckpt, seg, ends, device) -> np.ndarray` | model speed for windows ending at row indices |
| `ins_with_ml_velocity.py` | `anchored_ml_speed` | `(ml, v0, tau_s, fade_s=20.0)` | `ml + (v0 - ml[0]) * exp(-tau/fade_s)` |
| `ins_with_ml_velocity.py` | `make_ml_methods` | `(model, ckpt, device, fade_s=20.0)` | `extra_methods` for `simulate_outages`: `ml`, `anchored_ml` |

**Vehicle data:** `data/raw/vehicle/V/V-*.csv` (car's own 10 Hz log: wheel speeds,
VBOX velocity, speedometer, yaw rate, ...) and `data/raw/vehicle/S_sync/S-*.csv`
(their row-for-row phone files, which carry the timestamps), from the
"Uncategorised IOVNB Dataset" folders of the Synchronised zip. 39 drives align
clearly; S-Y1 and S-S4 do not match at all.

**Run:** `python -m src.dead_reckoning.strapdown_ins` (Day 8, all drives) and
`python -m src.dead_reckoning.ins_with_ml_velocity` (Day 9, TEST drives).

### Day 10 — heading and 2-D path (`src/dead_reckoning/heading.py`)

| Name | Signature | Notes |
|---|---|---|
| `local_east_north` | `(df) -> (east, north)` | GPS lat/lon -> metres around the drive's median position |
| `gps_fix_intervals` | `(t, fix, speed, course_deg, end, min_dt=4.0, max_dt=20.0, min_speed_ms=3.0) -> (i0, i1, dcourse_rad)` | moving fix intervals before row `end` |
| `GyroTurnIntegrator` | `(df)`; `.rotation(a, b, perm, signs, bias)` | cumulative gyro x gravity products; O(1) rotation for any mapping |
| `fit_gyro_heading` | `(integrator, gyro_bias, i0, i1, dcourse, min_intervals=8, min_corr=0.5) -> GyroHeadingFit or None` | picks gyro axis order/signs + scale k from pre-outage GPS course changes |
| `gyro_heading` | `(integrator, fit, gyro_bias, start, rows, course0_rad) -> heading_rad` | clockwise from north |
| `end_position` | `(speed, heading_rad, dt) -> (dx_east, dy_north)` | |
| `simulate_path_outages` | `(df_raw, outage_lengths_s=(30,60,120), n_starts=15, truth=None, seed=0, min_history_rows=1200, ml_methods=None, window_size=200) -> (part_a, part_b)` | starts at GPS fixes; error = distance to first fix after the outage |

Gyro axis order differs between phones: the standard order was the best fit
in only ~17% of outages, so the mapping is always fitted. The phone's fused
orientation yaw is NOT used (calibratable in only ~20% of outages).
Run: `python -m src.dead_reckoning.heading`.

### Day 11 — map matching (`src/map_matching/`)

Error frame change: `simulate_path_outages` now measures end-position error in
metres around each outage's entrance fix (was the whole-drive frame).

| File | Name | Signature | Notes |
|---|---|---|---|
| `heading.py` | `simulate_path_outages` | adds `map_match=None` | callable `(s, rr, speed, heading, ends) -> {end_row: {"snap": (dx, dy), "route": (dx, dy)} or None}`; adds methods `<speed>+gyro+map_snap` / `+map_route`, column `map_ok` |
| `osm_loader.py` | `to_local_xy` | `(lat, lon, lat0, lon0) -> (x_east, y_north)` | metres around an origin (flat earth) |
| `osm_loader.py` | `clean_gps_track` | `(df, max_jump_ms=80.0) -> (lat, lon)` | GPS fixes minus zeros and isolated jumps |
| `osm_loader.py` | `drive_corridor` | `(df, buffer_m=2000.0) -> shapely polygon` | lat/lon |
| `osm_loader.py` | `load_road_graph` | `(drive_name, df=None, cache_dir="data/external_maps", buffer_m=2000.0, network_type="drive", max_region_mb=150.0) -> nx.MultiDiGraph` | cached GraphML; Geofabrik if regions small, else Overpass; `G.graph["map_source"]`, `["corridor_buffer_m"]` |
| `osm_loader.py` | `geofabrik_regions`, `region_ways`, `graph_from_ways` | see docstrings | pbf -> drivable ways (osmnx "drive" filter) -> graph truncated to corridor |
| `osm_loader.py` | `edge_segments` | `(G) -> DataFrame` | one row per straight piece: u, v, key, piece, lat0/lon0/lat1/lon1, offset_m, edge_len_m, oneway |
| `osm_loader.py` | `corridor_margin_m` | `(df, lat, lon) -> np.ndarray` | distance to nearest GPS fix |
| `hmm_map_matcher.py` | `HMMMapMatcher` | `(G, df, sample_s=5.0, sigma0_m=15.0, sigma_growth=0.08, max_radius_m=600.0, sigma_heading_deg=30.0, beta_m=30.0, beta_frac=0.2, max_candidates=10, entry_radius_m=50.0, corridor_slack_m=200.0)` | one per drive |
| `hmm_map_matcher.py` | `HMMMapMatcher.match_outage` | `(s, rr, speed, heading, ends) -> dict` | causal Viterbi; `None` per end when the leak guard fails |
| `hmm_map_matcher.py` | `from_local_xy` | inverse of `to_local_xy` | |
| `hmm_map_matcher.py` | `HMMMapMatcher` (Day 12) | adds `turn_sigma_deg=20.0` (None = off), `anchor_turn_deg=45.0`, `anchor_window_s=15.0` | `match_outage` results also carry `"turn"` (distance from last completed turn; not adopted); benchmark `__main__` logs as Day 12 |
| `heading.py` | `simulate_path_outages` (Day 12) | map methods `+map_snap`, `+map_route`, `+map_turn` | |

Maps: `data/external_maps/<drive>.graphml` (87 drives), region files in
`data/external_maps/geofabrik/` (`*-latest.osm.pbf`, `*-drive-ways.pkl` cache).
Missing: only S-Vw1/S-Vw15 (no GPS). All French drives built by the completion pass
(`--max-region-mb 500` uses Geofabrik for the large French regions).
Run: `python -m src.map_matching.osm_loader [--skip A,B | --only A,B]` and
`python -m src.map_matching.hmm_map_matcher` (Day 11 benchmark).

### Completion pass — on-device model and tests

| File | Name | Signature | Notes |
|---|---|---|---|
| `models/compress_model.py` | `DeployableVelocityModel` | `(model, norm_stats)` | raw window (batch, 200, 8) -> speed (batch,), normalization inside |
| `models/compress_model.py` | `quantize_int8` | `(model, calib_X, norm_stats, window_size, n_features, backend="qnnpack", batch_size=256) -> nn.Module` | FX static PTQ; `calib_X` = raw TRAIN windows |
| `models/compress_model.py` | `export_lite` | `(deployable, path, window_size, n_features, optimize=True) -> Path` | trace -> (optimize_for_mobile) -> `.ptl`; use `optimize=False` for int8 |
| `models/compress_model.py` | `load_lite`, `predict_raw`, `latency_ms` | see docstrings | |
| `map_matching/osm_loader.py` `__main__` | flags | `--skip A,B`, `--only A,B`, `--max-region-mb N` | |

Files: `results/checkpoints/velocity_model_int8.ptl` (0.71 MB, for the phone),
`velocity_model_fp32.ptl`; `results/compression_summary.json`.
Tests: `tests/` (conftest `synthetic_drive`), `python -m pytest tests -q`, 19 passing.

### Day 13 — GNSS + INS fusion (`src/fusion/`)

| File | Name | Signature | Notes |
|---|---|---|---|
| `dead_reckoning/heading.py` | `path_outage_starts` | `(df, outage_lengths_s, n_starts, v_true, seed, min_history_rows=1200) -> np.ndarray` | the Days 10-13 outage starts (refactored out, identical) |
| `fusion/kalman_filter.py` | `ekf_update` | `(x, P, z, hx, H, R, gate=None, angle_rows=()) -> (x, P, accepted, nis)` | Joseph form; chi-square gate |
| `fusion/kalman_filter.py` | `rts_smooth` | `(x_f, P_f, x_p, P_p, F, angle_cols=()) -> (x_s, P_s)` | F[k] = Jacobian k-1 -> k |
| `fusion/kalman_filter.py` | `CHI2_999`, `wrap_angle` | | |
| `fusion/gnss_ins_fusion.py` | `E, N, PSI, V, BV, BW` | state indices | metres around the outage entrance, heading clockwise from north |
| `fusion/gnss_ins_fusion.py` | `FusionConfig` | dataclass | tuned on VAL: `gyro_scale_sigma=0.4`, `gps_course_sigma_deg=4.0` |
| `fusion/gnss_ins_fusion.py` | `run_ekf` | `(t, omega, ml_rows, gps, x0, P0, cfg, gyro_ok) -> {x_f, P_f, x_p, P_p, F, gate}` | `gps` {row: (east, north, accuracy, speed, course_rad)}, `ml_rows` {row: speed} |
| `fusion/gnss_ins_fusion.py` | `gyro_turn_rate` | `(integrator, fit, gyro_bias, rows, t) -> rad/s` | zeros if no gyro fit |
| `fusion/gnss_ins_fusion.py` | `simulate_fusion_outages` | `(df, model, ckpt, device, outage_lengths_s=(30,60,120), n_starts=15, truth=None, seed=0, history_s=300.0, matcher=None, cfg=None, outlier_frac=0.15, outlier_m=200.0) -> (end, track, robust)` | Phase 6 benchmark per drive |

Run: `python -m src.fusion.gnss_ins_fusion`.

### Day 15 — edge engine, route verification, app (`src/edge_engine/`, `mobile_app/`)

| File | Name | Signature | Notes |
|---|---|---|---|
| `models/simple_baselines.py` | `window_features` | `(X, fs_hz=10.0, chunk=4096) -> (n, 72)` | ridge / gradient boosting baselines in `__main__` |
| `fusion/gnss_ins_fusion.py` | `VehicleEKF` | `(x0, P0, cfg, gyro_ok)`; `.predict(dt, omega) -> F`, `.update_ml(v)`, `.update_gps(e, n, acc, speed, course) -> (ok, nis)`, `.finish_step()` | `run_ekf` now loops over it (bit-identical) |
| `edge_engine/engine.py` | `IDREngine` | `(model_path, ckpt_meta, map_matcher=None, cfg=None, recalib_every_s=60, buffer_s=600, min_calib_s=120, outage_after_s=12, map_every_s=5, reorigin_m=5000, route_monitor=None, start_index=0)`; `.push(sample) -> EngineOutput` | sample = phone columns (`RAW_COLS`) + `gps_fix` |
| `edge_engine/engine.py` | `EngineOutput` | dataclass | t, lat, lon, heading_deg, speed_ms, sigma_m, status, route_status, route_dist_m, exit_realtime, track_corrected |
| `edge_engine/engine.py` | `RouteMonitor` | `(route_lat, route_lon, base_m=40, k_sigma=2, hold_s=10)`; `.update(t, lat, lon, sigma_m) -> (state, dist)`, `.distance_m(lat, lon)` | tuned on VAL |
| `edge_engine/engine.py` | `calibrated_features` | `(rows, calib, feature_cols) -> float32 (n, F)` | numpy twin of `apply_calibration` |
| `edge_engine/replay.py` | `replay`, `wrong_route` | `(df, engine, r0, r1, mask=None, monitors=None)`, `(G, lat_true, lon_true, avoid_m=40, penalty=20)` | `__main__` = Day 15 benchmark (8 workers) |
| `edge_engine/server.py` | `run_replay`, `list_drives`, `Handler` | `/api/health`, `/api/drives`, `/api/replay?drive&start&outage&model&route` | localhost unless `--host` |
| `edge_engine/make_demo.py` | `__main__` | | writes `mobile_app/assets/demo/` |
| `main.py` | CLI | `app`, `test`, `train`, `compress`, `baselines`, `maps`, `bench <day>`, `demo-bundle` | |

### Not yet built (reserve these module paths)
| Phase | Expected file(s) |
|---|---|

---

## 4. Results Log Format (locked)

Unchanged — see prior version.

---

## 5. Known Open Issues (carry forward until resolved)

Unchanged from the prior version, **plus:**

- ~~ZUPT real-data reliability~~ — resolved in the Day 8 audit (adaptive
  per-phone thresholds, `zupt_mask_adaptive`).
- **Speed from vibration depends on the phone/car.** Remaining model error
  (~3.1 m/s vs wheel truth in cross-validation) is dominated by per-drive
  bias (e.g. S-T7). Better labels (wheel speed) do not fix it (3.14 -> 3.05).
  Anchored ML removes part of it at the tunnel entrance. Per-phone
  fine-tuning helped S-T7 a lot but broke S-Vw4 when the adaptation minutes
  did not cover the later speeds -- not adopted without safeguards.
- **NHC turn speed (a_lat / yaw rate) is unusable on phone data** (Day 12: 3-17 m/s error) -- do not retry without a better lateral axis.
- **Map matching fixes direction, not distance** (Day 12): the remaining 120 s error is the speed model's distance error.
- **ML speed bias is not stable over minutes** (Day 13, VAL: corr 0.19 before vs during an outage) -- a long-term bias estimate does not transfer into the outage.
- **EKF gate rejects some genuine fixes in sharp turns** (443 in the Day 13 run); the recovery rule prevents divergence.
- **Gyro axis convention differs between phones** (matches GPS course on
  S-T2, not on S-S1/S-Vw4) — check before using gyro for direction (Phase 5).
- **Early outages**: with little driving before the outage the heading
  often cannot be resolved (leveling only; ~17% of Day 9 test outages).
- **A FIFTH occurrence of pre-existing, unverified files appearing in
  this sandbox** (`multi_drive_data.py`, plus two unexplained data
  files `REAL-S1.csv`/`REAL-S3b.csv` this time). Same standing
  practice: never reuse found content, delete and rebuild from scratch
  every time.

---

## 6. Real Dataset Analysis

Unchanged from the prior version, **plus the folder-structure
clarification** noted in Section 1 above (flat "Uncategorised" folders
are what to use; Unsynchronised's 97 files for training, Synchronised's
72 S+V pairs for later evaluation).

---

## How to use this file

1. Upload `PROGRESS.md`, `context.md`, and the specific source file(s)
   the next day's code will import from.
2. After each day, update both: narrative in `PROGRESS.md`, exact
   names/signatures in Section 3 here.
3. Never rename an existing entry without updating every caller.