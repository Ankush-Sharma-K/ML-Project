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

---

## Phase 2 — Calibration & Preprocessing (Days 3–5)

### Day 3 — Gravity Compensation + IMU-Native ZUPT + Bias Estimation

**Concept:**
Raw `accel_x_mps2` has two error sources baked into it that naive
double integration can't tell apart: (1) **gravity leakage** — however
the phone is tilted in its mount, some slice of Earth's ~9.8 m/s²
constantly leaks onto every axis, and (2) **constant sensor bias** —
the MEMS accelerometer's own manufacturing offset, which double
integration turns into *quadratically growing* position error over
time (a tiny 0.15 m/s² bias alone caused >3000 m of error over a
120 s drive in testing). Both have to be removed before anything else
in the pipeline is worth trusting. Gravity compensation must happen
*before* bias estimation, or a constant tilt-induced gravity component
gets misattributed as sensor bias and "corrected" incorrectly.

Day 2 also left an open problem: the GPS-speed-based stationary
detector lags the true stop instant (GPS is sparse/smoothed), which
Day 3 needed a fix for before bias estimation could trust its
stationary windows — a bias estimate is only as good as the "at rest"
samples it's averaged over.

**Coding tasks / what was built:**
- `src/calibration/gravity_compensation.py` — `compute_linear_acceleration()`.
  The IO-VNBD schema conveniently provides both raw accelerometer
  (`accel_{x,y,z}_mps2`, gravity-inclusive) AND the phone's own fused
  gravity estimate (`gravity_{x,y,z}_mps2`). Subtracting one from the
  other gives linear acceleration **without needing to know the
  phone's mounting orientation** — a shortcut that made Day 4's mount
  alignment a separable, later problem instead of a blocking
  prerequisite.
- `src/calibration/zupt_classifier.py` — `zupt_mask()`: an IMU-native
  stationary detector (windowed rolling std of linear-acceleration
  magnitude AND gyro magnitude, both below threshold), replacing Day
  2's GPS-lag-prone detector. `evaluate_zupt_against_gps()` cross-checks
  agreement against GPS as an independent (imperfect) sanity check.
- `src/calibration/bias_estimation.py` — `estimate_constant_bias()`
  (mean of each linear-accel/gyro channel during ZUPT-flagged
  stationary windows — at true rest these should read exactly zero, so
  any nonzero mean *is* the bias), `apply_bias_correction()`, and an
  orchestration script that chains gravity compensation → ZUPT → bias
  estimation → correction → re-runs `naive_double_integrate()` on the
  corrected signal → logs a new `results/metrics.md` row.

**Verified:** on a test drive with a known, deliberately injected bias
(accel_x +0.15 m/s², gyro_yaw +0.02 rad/s), the estimator recovered
+0.1675 m/s² and +0.0203 rad/s — both close, small residual from noise.
Drift on that same drive: raw uncorrected double integration →
3213.93 m error; gravity-compensated + bias-corrected → 149.96 m —
roughly a **20x reduction**.

**Issue found (carried to Day 4/5):** `zupt_mask()`'s thresholds only
reached ~6% precision against GPS on the synthetic generator, because
its abrupt (linspace) accel/brake transitions inflate rolling variance
right at the true stop instant — the same GPS-lag problem as Day 2,
now visible from the other direction.

**Status:** code verified logically and numerically (bias recovery,
20x drift reduction) against a locally-built stand-in test drive — the
real `synthetic_data.py` still hasn't been shared, so these exact
numbers haven't been reproduced against the project's own data yet.

---

### Day 4 (part 1 of 2) — Mount Alignment: Leveling

**Concept:**
Days 2–3 cleaned up the *sensor* (gravity leakage, constant bias) but
the *signal* is still expressed in the phone's own coordinate frame,
not the vehicle's forward/lateral/vertical frame — "accel_x" is only
the vehicle's true forward acceleration by coincidence of how the
phone happens to be mounted (dashboard, cup holder, angled vent mount,
...). Fixing this is genuinely two separate sub-problems, because
gravity alone can only constrain 2 of the 3 rotational degrees of
freedom:

1. **Leveling (roll + pitch)** — the gravity vector measured at rest
   must point straight up in the vehicle frame; aligning it does this,
   but leaves an unknown rotation about the vertical axis.
2. **Heading (yaw)** — needs an independent signal (the vehicle's
   forward-acceleration direction during a GPS-speed-increasing phase)
   to resolve the leftover ambiguity from step 1.

Rather than ship both half-verified, Day 4 was deliberately split:
**today was leveling only.** Heading estimation, applying the full
rotation, and re-running the drift baseline are Day 4 part 2.

**Coding tasks / what was built:**
- `src/calibration/mount_alignment.py` —
  `rotation_from_vectors(a, b)`: general Rodrigues shortest-arc
  rotation between two vectors (reusable utility, also needed for
  heading in part 2). `estimate_leveling_rotation(df, stationary_mask)`:
  aligns the mean measured gravity vector (during ZUPT-flagged
  stationary windows) to `VEHICLE_UP`. `recovered_roll_pitch_deg()`:
  reports the tilt angles being corrected, for sanity-checking only.

**A real bug caught before shipping:** the gravity-sensor sign
convention was checked explicitly rather than assumed — Android's
`TYPE_GRAVITY` reports gravity using the *same* sign as the
accelerometer at rest (≈+9.81 when level), not the opposite. An
earlier draft had targeted vehicle-*down* instead of vehicle-*up*,
which would have silently flipped the leveled z-axis. Fixed and
documented in the module docstring as something to re-confirm against
real IO-VNBD data.

**Verified two ways:** (1) a purpose-built test with a known injected
tilt (roll +12.0°, pitch −7.0°) — recovered 11.995°/−7.0006°, accurate
to well within the injected sensor noise; (2) ran end-to-end against
the real Days 1–3 pipeline without error, from multiple working
directories.

**A second, separate bug — path resolution, not project logic:**
after delivery, running `mount_alignment.py` from a different location
than expected (`src/mount_alignment.py` instead of
`src/calibration/mount_alignment.py`) hit
`ModuleNotFoundError: No module named 'src'`. Root cause: every script
so far (`eda.py`, `naive_drift_baseline.py`, and all Day 3/4
calibration files) resolved the project root by counting a *fixed*
number of `.parent` levels up from `__file__` — correct only if the
file stays at exactly the depth assumed, and silently wrong otherwise.
Fixed across **all six** affected files at once with a function that
walks upward looking for a folder containing both `src/` and `data/`,
instead of counting levels — verified robust to the file being moved,
and to being run from an arbitrary working directory. (One self-caught
mistake along the way: the first batch-fix attempt had a regex bug
that left syntax garbage in `naive_drift_baseline.py`; caught by
actually re-running every file after the fix, not just assuming it
worked, and corrected before delivery.)

**Explicitly not done (Day 4 part 2, next session):** heading/yaw
estimation from accelerating-phase correlation with GPS,
`apply_mount_alignment()` to rotate `lin_accel_*_cal` into the vehicle
frame, re-running the drift baseline on the mount-aligned signal, and
logging a Day 4 row to `results/metrics.md`.

**Status:** leveling math verified correct (near-exact recovery on a
known-tilt test) and mechanically robust (works from any file location
or working directory, across all six affected scripts) — heading and
the full Day 4 drift number are still pending.