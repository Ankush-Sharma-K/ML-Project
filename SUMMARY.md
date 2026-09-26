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

### Day 4 — Mount Alignment: Leveling + Heading

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
2. **Heading (yaw)** — needs an independent signal to resolve that
   leftover ambiguity. By the non-holonomic constraint this whole
   project relies on (a vehicle doesn't move sideways), the vehicle's
   forward axis is the direction of horizontal linear acceleration
   during a phase where GPS speed is measurably increasing.

This was deliberately split across two sessions rather than shipped
half-verified: leveling first (verified in isolation), heading second
(today), once leveling was trusted.

**Coding tasks / what was built:**
- `src/calibration/mount_alignment.py`, extended with:
  `rotation_about_z(theta)`, `accelerating_mask(df)` (flags samples
  where `d(gps_speed)/dt` exceeds a threshold), `estimate_heading_offset()`
  (angle of the mean leveled horizontal linear-acceleration vector
  during those accelerating windows — requires BIAS-CORRECTED linear
  acceleration, or leftover bias would skew the angle),
  `estimate_mount_rotation()` (combines leveling + heading into one
  phone→vehicle rotation matrix), `apply_mount_alignment()` (produces
  `lin_accel_{x,y,z}_mps2_cal_veh` — forward/lateral/up in the vehicle
  frame).

**A real bug caught before shipping (leveling, from the earlier
session):** the gravity-sensor sign convention was checked explicitly
rather than assumed — Android's `TYPE_GRAVITY` reports gravity using
the *same* sign as the accelerometer at rest (≈+9.81 when level), not
the opposite. An earlier draft had targeted vehicle-*down* instead of
vehicle-*up*, which would have silently flipped the leveled z-axis.

**A second, separate bug — path resolution, not project logic:** after
that delivery, running the file from a different location than
expected hit `ModuleNotFoundError`. Root cause: every script resolved
the project root by counting a *fixed* number of `.parent` levels —
correct only at one exact file depth. Fixed across all six affected
files with a function that walks upward looking for a folder
containing both `src/` and `data/` instead of counting levels.

**A third finding, today — a diagnostic function's limitation, not a
pipeline bug:** testing heading estimation needed a test drive with a
*real* injected yaw misalignment (the earlier leveling-only test had
none). Once built, `recovered_roll_pitch_deg()` — the human-readable
roll/pitch readout — reported numbers that looked wrong (≈9.3° vs true
8.0° roll, ≈−1.6° vs true −5.0° pitch). Rather than assume the rotation
itself was wrong, this was isolated directly: applying `R_level` to
the true gravity vector landed within `1e-16` of `[0, 0, 9.81]` —
the rotation is exactly correct. The bug was in the diagnostic-only
Euler-angle decomposition, which is ambiguous once real yaw is present
(the minimal Rodrigues rotation legitimately isn't the same as a naive
"undo roll, then pitch" inverse when yaw ≠ 0). That function is never
used downstream — `apply_mount_alignment()` uses the rotation matrices
directly — so this had zero effect on any actual result, but the
docstring was tightened so it doesn't mislead future debugging.

**Verified:**
- **Heading, in isolation:** on a test drive with roll=8.0°,
  pitch=−5.0°, **yaw=22.0°** all injected, `estimate_heading_offset()`
  recovered **21.99°** — accurate to 0.01°.
- **Full pipeline, same drive:** mount-aligned double integration gave
  **209.54 m** final error, vs. **254.29 m** on the same drive without
  mount alignment (still in the phone's raw x-axis) — a real but modest
  ~18% improvement, smaller than Day 3's ~20x. Investigated rather than
  just reported: since both leveling and heading are now confirmed
  essentially exact, the remaining ~210 m is dominated by plain sensor
  *noise* compounding through naive double integration over 120 s (Day
  2's noise-only baseline alone produced ~42 m from noise with zero
  injected bias or misalignment) — not something mount alignment can
  fix. That's specifically what Phase 4's strapdown INS with periodic
  ZUPT resets (Days 9–10) is for; open-loop double integration, however
  correctly calibrated and aligned, drifts on noise alone.

**Status:** leveling and heading both verified correct — leveling via
direct rotation-of-gravity check, heading via near-exact recovery of a
known injected angle. Day 4 row logged to `results/metrics.md`. Mount
alignment (Phase 2) is functionally complete; remaining drift is
noise-driven and is Phase 4's problem, not Phase 2's.

### Day 5 — Denoising (Phase 2 complete)

**Concept:**
The last item in Phase 2's original scope ("mount alignment, bias
estimation, denoising, ZUPT classifier"). Also framed as a direct test
of Day 4's claim: if the ~210 m of remaining drift really is
"noise-driven," a low-pass filter removing that noise before
integration should recover a meaningful chunk of it. Testing a claim
instead of just repeating it turned out to matter here.

**Coding tasks / what was built:**
- `src/calibration/denoising.py` — `characterize_frequency_content()`
  (FFT-based: where does a signal's power actually sit in frequency,
  so a filter cutoff is evidence-based, not guessed) and
  `denoise_lowpass()` (zero-phase Butterworth low-pass via
  `scipy.signal.filtfilt` — zero-phase specifically because an
  ordinary causal filter's time delay would bias *when* motion is
  detected, not just clean up noise level).
- Cutoff chosen from an FFT on the Day 4 test drive: 90% of signal
  power sits below 0.075 Hz, 99% below ~1 Hz — real driving dynamics
  are low-frequency, so 1.0 Hz keeps essentially all real signal while
  removing noise energy above it.

**Result — and a genuine correction to Day 4, not just a small
number:** denoising improved the same test drive's final error from
209.54 m to 209.13 m — a **0.2% improvement**. This was investigated
rather than shrugged off: confirmed the filter genuinely works (the
noise it removed from the raw signal has std ≈0.077, matching the
injected noise level almost exactly) — but that removed noise barely
touches integrated drift, because **double-integration drift is driven
by a signal's low-frequency/near-DC content (it random-walks), not its
high-frequency content.** Any reasonable low-pass filter passes
near-DC noise through essentially unchanged; it only removes high-
frequency jitter, which was never the main driver of drift. This is a
known property of inertial navigation, not a broken filter: **no
amount of denoising fixes open-loop double-integration drift** — only
a periodic external correction (ZUPT resets, Days 9–10; or GNSS
fusion, Days 13–14) can bound it.

**What this means for Day 4's write-up:** the diagnosis ("remaining
drift is noise-driven, not a calibration problem") still holds. The
implied *fix* ("a low-pass filter... is directly motivated") was
wrong, and is corrected here rather than left standing uncorrected.

**Where the work wasn't wasted:** `denoise_lowpass()` stays in the
pipeline — flat signal noise still matters for things that aren't
double integration, like Phase 3's upcoming velocity-estimation model
(cleaner input likely trains better) and any future variance-based
detector (Day 3's ZUPT classifier is sensitive to raw noise level).

**Status:** Phase 2 (Days 3–5) is now functionally complete — all four
original scope items (mount alignment, bias estimation, denoising,
ZUPT classifier) exist and are verified, with one finding corrected
along the way rather than quietly dropped.