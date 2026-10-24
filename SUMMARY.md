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

---

## Interlude — Real Dataset Analysis

**Concept:** Everything through Day 5 was verified only against
locally-built stand-in test data, never the actual IO-VNBD dataset.
The user uploaded the real thing (both "Synchronised" and
"Unsynchronised" zips) and asked for analysis only — no files, no
fixes yet, just find out what's actually true about the real data
before building more on top of assumptions.

**What was checked, and found:**
- Schema matches exactly (24 columns, same order as `SMARTPHONE_COLUMNS`).
- Gravity sign convention (`GRAVITY Z ≈ +9.81`) confirmed on real
  data — directly validates a Day 4 assumption that had only been
  theoretically justified before.
- **A real, previously-invisible bug:** `TIME SINCE START (ms)` resets
  mid-file (found in 2 of 3 real files checked — e.g. a jump from
  2,707,520 ms down to 8 ms). The `DATE` column, by contrast, parsed
  perfectly and stayed monotonic in every file checked.
- **Real sensor noise is ~10x higher than the synthetic model
  assumed.** Isolated a confirmed 29-second stop and measured actual
  accelerometer std (~0.57 m/s²) against the synthetic assumption
  (~0.03–0.05). Traced the cause specifically: the gravity sensor
  stays clean, so it's genuine engine-idle vibration hitting the raw
  accelerometer, not a computation error.
- Real drives are much longer (86–177 min vs. 2-minute synthetic
  tests) and slower/stop-and-go (max ~20–27 km/h) — a different
  driving profile than the synthetic generator's smooth highway cycles.
- A genuinely good find: the paired `V-*.csv` vehicle files carry
  wheel-speed and indicated-vehicle-speed channels — meaningfully
  better ground truth than GPS-speed integration, once real evaluation
  starts.

**Status:** analysis only, as requested — no code or documents written
during this step. Findings carried forward into Day 6's actual design
decisions below, where they mattered immediately.

---

## Phase 3 — AI Velocity Estimation (Days 6–8)

### Day 6 (part 1 of 2) — Dataset Preparation

**Concept:** Phase 3 replaces naive double integration with a learned
model: given a short window of IMU data, predict the vehicle's true
forward velocity. Before any model or training code, the data has to
be correctly windowed and labeled — and the interlude analysis had
just surfaced two real correctness questions this step couldn't
ignore (the timestamp reset, and how different real noise looks from
synthetic). Split from model architecture/training (Day 6 part 2) for
the same reason Day 4 was split: getting the data pipeline right and
verified first, rather than building a model on top of an unverified
foundation.

**A provenance note, in the interest of not overclaiming:** this
sandbox's `src/models/` briefly contained pre-existing files (a full
`dataset_prep.py`, `velocity_model.py`, `train_velocity_model.py`)
that weren't written in this conversation and had no verifiable
origin — the same situation as an earlier day's `mount_alignment.py`
surprise. Deleted all three without reusing their content, and
independently re-verified every specific claim from scratch (including
a frequency-content number that happened to already be sitting in one
of their docstrings) rather than trusting anything just because it
was already there and looked plausible.

**Coding tasks / what was built:**
- `src/models/dataset_prep.py` — `build_windows()`,
  `train_val_split_by_time()`, `_window_is_clean()`.

**Three design decisions, each checked against evidence:**
1. **Windows built from row count, not absolute timestamps** —
   direct response to the real `TIME SINCE START` reset found in the
   interlude. Row order is still correct chronological order
   regardless of a counter resetting, so this sidesteps the bug
   entirely instead of patching around it.
2. **A `dt`-anomaly guard checks per-row gaps inside each window** —
   catches real sensor dropouts, and, as a side effect, the exact
   reset bug too (a reset produces a huge negative "gap," a dropout a
   huge positive one — one check, two failure modes).
3. **Windows are NOT denoised before being handed to the model** —
   verified this myself: ran Day 5's `characterize_frequency_content()`
   against the real `S-S3b.csv` sample directly rather than trusting
   the deleted file's claim, and got 90% of power below 3.549 Hz
   (independently computed) — confirming real accelerometer signal is
   broadband, extending almost to Nyquist, unlike the smooth synthetic
   profile. Day 5's 1.0 Hz cutoff would strip real signal on real data.

**Verified:**
- Synthetic drive: 237 clean windows, 0 rejected (no resets expected
  or found in synthetic data). Train/val split by time order (189/48)
  — random splitting would leak near-duplicate overlapping windows
  across the split, since windows overlap by design (stride < window
  size).
- **Real data, direct test:** ran the same windowing code against
  `S-S3b.csv` using its own raw, known-to-reset time column on
  purpose — built 1355 clean windows, **4 correctly rejected at
  exactly the known reset location**. Concrete confirmation the guard
  works, not just a plausible-sounding design.
- Standard robustness check (same as every prior file): correct from
  its proper location, misplaced, and from any working directory.

**Explicitly NOT done today (Day 6 part 2, next session):** the model
architecture and training script. Both existed among the untrusted
deleted files and need to be actually written and reasoned through,
not resurrected from an unverified source.

**Status:** dataset preparation verified correct and tested against
both synthetic and real data (including a direct test of the real bug
it was designed to handle) — model architecture and training are Day
6 part 2. Also surfaced, again, that Day 3's ZUPT thresholds are now
confirmed (not just suspected) unusable on real data — worth fixing
before Phase 2's calibration code is trusted on anything real.

### Day 6 part 2 — Velocity Model Architecture

**Concept:** With windowing verified, the actual model can be
designed: given a short window of calibrated IMU data, predict forward
velocity — this is the thing that eventually replaces naive double
integration. Split from the training script (part 3) at the user's
own request to split the remaining work further, which turned out to
matter: this half surfaced a real bug on its own (below) worth
catching before adding a training loop on top of it.

**A pattern repeating for the third time, named plainly rather than
downplayed:** this sandbox's file system contained a pre-existing
`velocity_model.py` again — even after the earlier `src/models/` trio
had already been deleted once this session for the exact same reason.
Deleted again without reusing any content, rebuilt from scratch.

**Coding tasks / what was built:**
- `src/models/velocity_model.py` — `compute_normalization_stats()`,
  `apply_normalization()`, `IMUWindowDataset`, `VelocityCNN`.

**Architecture choice, reasoned rather than defaulted:** a small
1D-CNN over the window's time axis, not a recurrent network — tied to
two of this phase's own later goals: Day 8's on-device compression
(CNNs quantize/export to mobile more predictably than RNNs), and the
window already being short and fixed-length (no need for carried-over
recurrent state).

**A real bug caught by actually running the file, same practice as
every prior day:** the torch-import fallback set `nn = None` when
torch is unavailable, but `class VelocityCNN(nn.Module)` evaluates
`nn.Module` at class-DEFINITION time — this crashed immediately with
`AttributeError`, before the file could even be imported, let alone
used. Not a hypothetical edge case: this is exactly the situation this
sandbox is in right now. Fixed with a separate fallback base class.

**Verification, described precisely rather than glossed as "done":**
this sandbox cannot install torch (`pip install torch` was actually
attempted this session, not assumed to be blocked — it failed
outright). So the PyTorch-specific code has NOT been executed here.
What WAS actually verified:
- **Architecture shape-correctness**, via a parallel pure-NumPy
  reimplementation of the identical layer sizes/kernels/padding,
  executed in this sandbox: every intermediate shape matched exactly
  (`(4,6,20)→(4,32,20)→(4,64,20)→(4,64)→(4,32)→(4,1)`), parameter
  count independently computed as 13,409.
- **The normalization functions themselves** — pure NumPy, no torch
  needed — were actually run against real Day 6 part 1 window output.
  Training data normalized to exactly mean≈0/std≈1 per channel; and,
  the check that actually matters, validation data normalized with
  TRAINING stats (not recomputed from its own) came out with a
  different, nonzero mean — concrete confirmation the "don't leak
  validation statistics" design is really implemented, not just
  asserted in a docstring.
- **What's still unverified:** whether the actual `Conv1d`/`Dataset`/
  `nn.Module` PyTorch calls are syntactically perfect and train
  stably. The file's own `__main__` block does the real forward-pass
  assertion (`assert out.shape == (4, 1)`) and prints the true
  parameter count — but only when run in an environment that actually
  has torch, which is the user's local/Colab setup, not here.

**Status:** model architecture built and verified as thoroughly as
this sandbox allows — every part checkable without torch was actually
checked, not assumed. What remains (does the PyTorch code itself run;
does the model train) needs the user's own environment, flagged
explicitly rather than left ambiguous. Training script is Day 6 part 3.

### Day 6 part 3 of 3 — Training Script (Day 6 complete)

**Concept:** Wire parts 1 and 2 together into an actual training loop.
The last piece before the user can run real training — everything
before this was preparation for this one step.

**A pattern hitting its fourth occurrence, named directly rather than
treated as routine:** `train_velocity_model.py` was, again, already
sitting in the sandbox before being written here — following
`mount_alignment.py`, the original `src/models/` trio, and
`velocity_model.py` reappearing after its own deletion. Deleted without
reuse, rebuilt from scratch, same as every time before.

**Coding tasks / what was built:**
- `src/models/train_velocity_model.py` — `train_one_epoch()`,
  `evaluate()`, `save_checkpoint()`, `load_checkpoint()`,
  `train_velocity_model()`.

**Three reasoned decisions:**
1. **Huber loss, not MSE** — the GPS-speed training target is noisy
   ground truth (known since Day 2), and Huber loss doesn't let a
   handful of bad target samples dominate training the way squared
   error would.
2. **Checkpoints always bundle normalization stats with the weights**
   — a checkpoint saving only model weights is a common, real
   inference-time bug (new data normalized with the wrong stats).
   Structured so that can't happen here.
3. **Saved checkpoint is always the best validation epoch, via early
   stopping** — not simply whatever the last epoch happened to be.

**Verification, same honest framing as part 2, because the same
constraint applies:** no torch here, so real training can't happen in
this sandbox. What WAS actually done:
- The file's no-torch fallback path runs cleanly — including a
  decorator pattern (`@torch.no_grad() if torch is not None else
  (lambda f: f)`) specifically designed to avoid the exact
  class-definition-time crash caught in part 2. Verified by running
  the file, not assumed safe by similarity to the earlier fix.
- **The training loop's control-flow logic was extracted and tested
  independently of torch**, against two synthetic validation-loss
  curves: one that improves then plateaus, one with a mid-training
  blip. Both cases correctly identified the true best epoch and
  triggered early stopping at the right point — real assurance that
  the *logic* is sound, even though the *model* itself is unverified.
- **The checkpoint's normalization-stats round-trip was actually
  executed** (the one part of checkpointing needing no torch) —
  confirmed exact, correct types restored.
- What remains genuinely unverified: whether the real PyTorch training
  (Adam, Huber loss, backward pass) works and whether the model learns
  anything on real data.

**A caveat surfaced now, not after a confusing training run:** only a
small real-data sample (~1355 windows) is physically present in this
sandbox — real training needs the full dataset, which exists on the
user's machine but hasn't been fully brought into any session yet.

**Status: Day 6 complete, all 3 parts.** Next real milestone: the
user's own training run, reported back with loss curves and best
epoch — which will shape Day 7's actual scope ("robustness") more
usefully than guessing at it now would.

---

## Day 7 (started) — Multi-Drive Real Data Loading

**Concept:** Every prior day's code loaded one hardcoded drive file.
Moving to real training means looping over ~97 real files — and that
loop has genuinely different failure modes than single-file code ever
exercised (a bad file, a drive with no stop, a drive with no clear
accelerating phase). This surfaced from a very practical starting
point: the user asked how to point training at the real dataset, which
turned out to need real investigation first (the two zips' folder
structures, a filename-overlap surprise, and a path-handling question)
before any code made sense to write.

**Three things resolved by actually checking, not assuming, before any
code was written:**
1. **Folder structure** — both zips have a messy per-driver
   "Categorised" layout and a flat "Uncategorised" folder of
   `S-*.csv` directly. Found this by listing the real zip contents
   with Python's `zipfile`, not guessing from file names alone.
2. **A real correction to earlier advice:** initially suggested
   copying both flat folders' files together. The user pushed back —
   correctly — that they share filenames. Checked properly: all 72
   Synchronised files share names with 72 of the Unsynchronised's 97,
   but aren't byte-identical (trimmed to match a paired V-file's
   window). Corrected advice: use only the Unsynchronised 97 (the
   untrimmed superset) for training.
3. **Path handling** — the user pasted real data into an absolute
   Windows path and asked to use it directly to avoid path errors.
   Doing that literally would have reintroduced the exact fragility
   fixed back on Day 4 (works on one machine only, breaks in Colab).
   Used the existing `_find_project_root()` pattern instead, pointed
   at a relative `data/raw/real/` — resolves to the user's exact
   pasted folder without hardcoding anything machine-specific.

**A pattern hitting its fifth occurrence:** `multi_drive_data.py` was,
again, already present in the sandbox — plus, this time, two
unexplained data files that weren't created in this conversation
either. All deleted without reuse, everything rebuilt from scratch.

**Coding tasks / what was built:**
- `src/models/multi_drive_data.py` — `load_and_calibrate_drive()`,
  `build_windows_for_drives()`, `train_val_split_by_drive()`.

**A real risk checked directly, not assumed either way:** does the
Day 3-4 calibration pipeline — whose ZUPT thresholds are known to be
synthetic-tuned — actually crash on real data? Tested against the real
sample in this sandbox: it didn't crash, but surfaced a visible
concern (only 104/6813 samples flagged stationary, a 139.65° heading
estimate). This directly shaped the design: log per-drive diagnostics
during loading so outliers are visible, and wrap each drive's
processing in try/except so one bad drive can't take down a whole
multi-drive run.

**Verified, concretely:**
- Built a 3-"drive" test (same real content under different filenames,
  since only one real sample exists in this sandbox) and confirmed
  correct concatenation, correct per-drive diagnostic logging, and a
  correct whole-drive validation split (one entire drive held out, not
  a time slice — exact window counts matched expectations).
- Added a deliberately malformed file and confirmed it was skipped
  with a clear error while the other drives still processed — the
  failure-handling guard actually works, not just assumed to.
- Standard robustness check: correct from its proper location and a
  different working directory, including a sensible message (not a
  crash) when the target folder doesn't exist.

**Status:** multi-drive loading built and verified as far as one real
sample allows. What's still unknown: how the calibration pipeline
behaves across genuinely different real drives — the user running this
against the full 97-file dataset will reveal whether the one sample's
ZUPT/heading concerns are typical (confirming real retuning is needed)
or unusual (just a hard case). That result shapes what comes next.

---

## Phase 4 — Core Dead Reckoning (Day 8) and the Day 8 audit

### Day 8 — Strapdown INS + the GPS-outage benchmark

**Concept:** A strapdown INS ("inertial navigation system") is the classic,
non-ML way to dead-reckon: take the car's forward acceleration from the
calibrated accelerometer, add it up over time to get speed, and add speed up
to get distance. ZUPT (zero-velocity update) resets speed to zero whenever
the IMU says the car is standing still, which stops errors from building up
during stops.

The first version integrated one whole 86-minute drive and reported tens of
kilometres of error. That number said nothing about a tunnel. The tunnel
problem is: GPS drops out for 30 s – 2 min, we know the speed at the
entrance, and we must keep track of distance until GPS returns. So Day 8 is
evaluated by **simulated GPS outages** at random moving points of real
drives, against the simplest competitor: **hold the last GPS speed**.

**Coding:** `src/dead_reckoning/strapdown_ins.py` — `StrapdownINS`
(`integrate(initial_velocity, no_reverse)`), `simulate_outages()`,
`log_metric()`; run as `python -m src.dead_reckoning.strapdown_ins`.

**Result:** with honest evaluation (below), the INS is clearly worse than
holding the last speed at every outage length. The phone accelerometers in
this dataset are too noisy for physics-only integration — the reason Day 9
uses the learned velocity model instead.

### The Day 8 audit — what was wrong, in short

Re-checking Days 1–8 against all 97 real drives found:
- **GPS speed is m/s, not km/h** (despite the header) in 94/97 files — every
  speed, distance and training target had been 3.6x too small.
- **GPS updates only every ~10 s**, held in between → interpolated speed
  `gps_speed_interp_ms` added; the old heading estimate (which differenced the
  held speed) was broken and was replaced by a GPS-interval rotation fit.
- **ZUPT thresholds were synthetic-tuned** → per-phone adaptive thresholds.
- Data repairs: S-A4 column shift (old repair off by one), frozen duplicate
  rows (S-T1/T4/T5/T6), time-counter resets.
- **The only "trained model" was trained on the synthetic drive** → the model
  was trained on real data for the first time.

### What calibration is

A phone in a car reports raw numbers in its own coordinate frame, distorted
by its own flaws. Calibration turns them into the car's acceleration:
1. **Remove gravity** (phone's gravity sensor; no GPS needed).
2. **Sensor bias** — average reading while GPS says the car is stopped.
3. **Mount alignment** — tilt (from gravity at stops) and heading (rotation
   that best matches the phone's motion to GPS speed changes and turns).
4. **ZUPT thresholds** — how quiet this phone is at rest vs while driving.
Steps 2–4 need GPS, so *which* GPS data they may use matters.

### Honest evaluation (audit part 3)

Two ways the benchmark used information a phone would not have in a tunnel:
- **Calibration used the whole drive's GPS**, including the outage itself.
  Fixed: each outage is calibrated only on everything recorded **before** it.
  This made the INS noticeably worse (it had been benefiting); the ML model
  was almost unaffected (4.59 vs 4.50 m/s per-window error).
- **The entry speed was the interpolated GPS speed**, which leans towards the
  next fix — inside the tunnel. Fixed: the **last real GPS fix**. This made
  "hold last speed" ~15–25% worse than it had looked.

### Better ground truth: the car's wheel speed

The Synchronised zip has, for 72 drives, a `V-*.csv` logged by the car itself
(wheel speeds, VBOX GPS velocity, speedometer at 10 Hz). The car's signals
agree with each other to ~0.1 m/s, while our phone-GPS label is ~1 m/s off.
The files are NOT always in sync with our phone files (offsets up to ±28 s),
so each drive's offset is estimated from speed and verified against gyro vs
yaw rate; 39 drives align clearly. Findings: training on wheel-speed labels
barely helps the model (3.14 → 3.05 m/s), and conclusions from GPS truth
held — but wheel speed is now the reference wherever it exists.

**Coding:** `src/utils/vehicle_truth.py` (`load_wheel_speed_truth`),
files in `data/raw/vehicle/V/` and `data/raw/vehicle/S_sync/`.

---

## Day 9 — ML velocity during outages: "anchored ML"

**Concept:** Neither competitor wins everywhere. Holding the last GPS speed
is good for a few seconds but goes stale; the ML model tracks speed changes
but has a phone/car-specific bias (e.g. a smooth car reads as "slow"). At the
tunnel entrance we can *see* that bias — the model's speed vs the last GPS
fix — so **anchored ML** corrects the model by it and lets the correction
fade out as the road and speed change:

    speed(t) = ML(t) + (last GPS speed − ML(0)) · e^(−t / 20 s)

The 20 s fade was chosen on validation drives and confirmed by a
minimum-variance mix computed from error statistics (which cannot overfit a
few drives): trust moves from the entrance correction to the model within
~10–20 s, and the derived mix did not beat the simple 20 s fade.

**Coding:** `src/dead_reckoning/ins_with_ml_velocity.py` —
`ml_speed_series`, `anchored_ml_speed`, `make_ml_methods`; run as
`python -m src.dead_reckoning.ins_with_ml_velocity` (TEST drives only).

**Result:** see PROGRESS.md / results/metrics.md (Day 9 section). In 5-fold
cross-validation against wheel truth, anchored ML beat holding the last
speed at every outage length (47/88/176 m vs 61/141/315 m at 30/60/120 s).


---

## Day 10 — Direction and the 2-D path

**Concept:** Distance alone is not a path. To draw the car's route inside the
tunnel we also need its direction. The GPS gives the direction (course) at the
tunnel entrance; after that, the gyroscope measures how fast the car turns, and
adding up those turns gives the current direction. Distance (Day 9) along that
direction gives the east/north path.

The catch, found by checking real data: phones do not report the gyroscope in
the same axis order or with the same signs as the gravity sensor, and this
differs between phones. So the "which axis is vertical rotation" question is
answered per phone, from the GPS course changes before the outage -- the same
"calibrate with GPS while you still have it" idea as Days 3-4.

**Result:** the gyro heading cuts the direction-caused position error by a
third to a half compared with assuming a straight line. Over long outages the
error still grows to hundreds of metres, because small heading errors add up
sideways -- which is exactly why Phase 5 matches the path to the road map.

**Coding:** `src/dead_reckoning/heading.py` — `GyroTurnIntegrator`,
`fit_gyro_heading`, `gyro_heading`, `simulate_path_outages`; run as
`python -m src.dead_reckoning.heading`.
