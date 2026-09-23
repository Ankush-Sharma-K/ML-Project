# SUMMARY.md — Day-by-Day Concept + Coding Notes

> Companion to PROGRESS.md. This file explains **what each day's work
> means and why**, plus concrete coding tasks. Read the concept section
> before the coding section each day — the code will make more sense.

---

## Phase 1 — Setup & Dataset Understanding (Days 1–2)

### Day 1 — Environment & Repo Setup

**Concept:**
Before touching sensor fusion math, get the ground under your feet: a
reproducible environment and a mental model of the pipeline you're
building —
`raw IMU → calibrate/align → denoise → AI velocity estimate → strapdown
integration (dead reckoning) → map matching → GNSS fusion`.
Every later day builds one block of this chain.

**Coding tasks:**
- Create the folder structure from PROGRESS.md §8.
- Set up Python environment (conda/venv), core libs: `numpy`, `pandas`,
  `scipy`, `matplotlib`, `torch` (or `tensorflow`), `pymap3d` or `pyproj`
  (coordinate conversions), `filterpy` (Kalman filters as reference).
- Write a data-loading stub for IO-VNBD (`src/calibration/preprocessing.py`
  or a `data/loader.py`): load one drive's IMU stream (accel, gyro,
  timestamp) and GNSS ground truth into a pandas DataFrame.
- Sanity plot: raw accelerometer magnitude vs time for one drive.

# Day 2 — Summary (EDA + Naive Drift Baseline)

## Conceptually

Day 2's job was to answer two questions before any correction logic gets
built: **what does the raw sensor data actually look like**, and **how
bad is dead reckoning if you do nothing to fix it**. Both existed for
reasons:

- The EDA pass exists because every later phase (calibration bias
  estimation, the ZUPT classifier, the Kalman filter's noise matrices)
  needs to know the sensor's real noise floor and needs a way to detect
  when the phone is stationary — you can't calibrate against a number
  you haven't measured.
- The naive drift baseline exists because "seamless navigation via
  AI/ML dead reckoning" only means something relative to a starting
  point. Raw accelerometer double-integration is the worst-case number
  — no bias removal, no ZUPT, no fusion — and it's what every later
  phase (Day 3 calibration, Day 6 velocity model, Day 9 strapdown INS,
  Day 13 GNSS fusion) has to beat. Logging it now in
  `results/metrics.md` turns the rest of the project into a visible
  error-reduction curve instead of a set of disconnected claims.

## Logically

The actual flow of the day, in order:

1. Built the EDA and drift-baseline logic against a **schema
   reconstructed from the Day 1 writeup** (the dev sandbox resets
   between sessions, so the real repo files weren't available). Ran
   it, got clean plots and a drift number, packaged it.
2. Running it against the real repo hit
   `ModuleNotFoundError: No module named 'src'` — the scripts used
   `sys.path.insert(0, ".")`, which only works if launched from the
   project root. Fixed by resolving the project root relative to
   `__file__` instead of the caller's cwd.
3. Next error: `ImportError: cannot import name 'load_drive'` — this
   revealed the reconstructed schema didn't match the real
   `io_vnbd_loader.py` at all (different function name, different
   column names, different structure).
4. The real `io_vnbd_loader.py` was shared. Both files were rewritten
   against the actual schema (`load_smartphone_drive`, `t_sec`,
   `accel_x_mps2`, `gyro_yaw_rads`, `gps_speed_kmh`, no separate GPS-fix
   flag) and verified against a schema-matching test CSV, including
   running from a different working directory to confirm the
   path-independence fix held.

**Still unverified:** the real `synthetic_data.py` hasn't been shared,
so the drift baseline's ground truth is built by integrating
`gps_speed_kmh` directly rather than depending on a specific
`df.attrs` shape a generator might stash — this should work regardless
of what the real synthetic generator does internally, but isn't
confirmed against the actual file yet.

## Code

**`src/eda.py`**
- `channel_stats()` — mean/std/min/max for the 6 raw IMU channels
- `stationary_mask()` — flags samples where `gps_speed_kmh < 1`
- `stationary_noise_floor()` — IMU channel std restricted to those
  stationary samples (the number Day 3's ZUPT and Day 6's Kalman R/Q
  will need)
- Two plots: channel histograms, and accel/GPS-speed with stationary
  windows overlaid

**`src/dead_reckoning/naive_drift_baseline.py`**
- `naive_double_integrate()` — `accel_x_mps2` → cumsum → velocity →
  cumsum → position, zero correction
- `gps_reference_distance()` — ground truth via `gps_speed_kmh`
  integration
- `evaluate_drift()` — final error, drift rate (m/min), RMSE
- `log_metric()` — appends a row to `results/metrics.md`

## Status

Both files are logically complete and run cleanly (verified against a
schema-matching test CSV, from multiple working directories), but have
**not yet been run successfully against the user's actual data** end to
end — next step is to confirm output on the real repo and, ideally,
see `synthetic_data.py` to close the remaining gap.
