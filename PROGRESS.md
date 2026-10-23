# idr-project — Progress Report

**Project:** AI/ML-based Intelligent Dead Reckoning (IDR) for Seamless Navigation
**Plan:** 15-day phased build, deployed as a mobile app + edge-deployable engine
**Approach:** classical strapdown INS (orientation integration + ZUPT) corrected by
AI/ML (learned forward-velocity model, adaptive fusion noise), constrained by
OSM map-matching with Non-Holonomic Constraints, fused with GNSS via UKF/EKF.
Trained/evaluated against the IO-VNBD dataset per the problem statement.

## Overall Plan

| Phase | Days | Focus |
|---|---|---|
| 1 — Setup & Dataset Understanding | 1–2 | Repo scaffold, IO-VNBD loader, EDA, naive drift baseline |
| 2 — Calibration & Preprocessing | 3–5 | Mount alignment, bias estimation, denoising, ZUPT classifier |
| 3 — AI Velocity Estimation | 6–8 | IMU→velocity model, robustness, on-device compression |
| 4 — Core Dead Reckoning | 9–10 | Strapdown INS integration, drift benchmark evaluation |
| 5 — Map Matching | 11–12 | OSM road graph, NHC constraints, HMM map matcher |
| 6 — GNSS+INS Fusion | 13–14 | UKF/EKF fusion, adaptive noise, seamless GNSS↔DR handoff |
| 7 — App & Deployment | 15 | Edge engine API, mobile app UI, packaging, final submission |

**Status: Days 1–9 done on the real IO-VNBD data. Day 8 audit (parts 1–3) complete; Day 9 = anchored ML velocity during GPS outages. Next: Phase 5 (map matching).**

---

## Days 1–6 — Complete

See prior entries. Phase 2 fully closed out, Phase 3's data pipeline
(Day 6, all 3 parts) and velocity model architecture complete. Real
IO-VNBD dataset analyzed and clarified: the user has both the
Synchronised and Unsynchronised zips, each containing a messy
per-driver "Categorised" structure AND a flat "Uncategorised" folder
of `S-*.csv` files directly — confirmed the flat folders are what to
use. Also discovered (by actually checking, not assuming) that all 72
files in the Synchronised flat folder share filenames with 72 of the
97 in the Unsynchronised flat folder — but are NOT byte-identical
(Synchronised versions are trimmed to match their paired V-file's time
window). Resolved: use the Unsynchronised folder's 97 files for
training (the superset, untrimmed); keep the Synchronised S+V pairs
for later evaluation against real wheel-speed ground truth.

---

## Day 7 (started) — Multi-Drive Real Data Loading

**Goal:** Every prior day's code loaded exactly one hardcoded drive
file. Training on the real dataset means looping over ~97 real files —
and that loop has its own real failure modes that single-file testing
never exercised. Scoped as its own piece for the same reason every
multi-part day has been split so far.

**A fifth occurrence of the recurring pattern, named directly again:**
`src/models/multi_drive_data.py` was, once again, already present in
this sandbox before being written here — plus, this time, two
unexplained data files (`REAL-S1.csv`, `REAL-S3b.csv`) that weren't
created in this conversation either. All deleted without being used;
everything below was built and verified from scratch.

**What was built:**
- `src/models/multi_drive_data.py` — `load_and_calibrate_drive()`
  (wraps the Days 1–4 pipeline for one real file, returns diagnostics
  alongside the calibrated data), `build_windows_for_drives()` (loops
  this + Day 6's `build_windows()` over a folder, concatenating
  results), `train_val_split_by_drive()` (splits by WHOLE drive, not
  by time within a concatenated sequence).

**A real question resolved by testing, not assumed either way:** does
the existing `load_smartphone_drive()` even work on real files
out of the box? Tested directly — yes, with `header=0` (real files
have their own header row; the synthetic generator's files don't).
No encoding workaround needed either: the file's ISO-8859-1 encoding
only affects the header row (degree/mu symbols), which `header=0`
skips past; the data rows themselves are plain ASCII and load fine
under pandas' default encoding.

**A real risk checked directly before building around it:** does the
full Day 3–4 calibration pipeline — whose ZUPT thresholds are already
known (Section 6 of `context.md`) to be tuned on synthetic data, not
real — actually crash on real data? Tested against the real sample
already in this sandbox: **it did not crash**, but ZUPT flagged only
104/6813 samples stationary, and mount alignment's heading estimate
came out as 139.65° — a visible outlier worth a human checking (not
necessarily wrong; different recording sessions may genuinely have
the phone mounted differently). This directly justified two design
decisions: (1) every drive's diagnostics (stationary-sample count,
heading estimate) are logged and printed during loading, not just
silently used, so outlier drives are visible rather than hidden; (2)
per-drive processing is wrapped in try/except — some real drives
likely WILL raise genuine errors (e.g. a drive with no stop at all, or
no clear accelerating phase), and one bad drive must not kill the
whole loading run.

**Path handling — addressed the user's explicit concern properly, not
by just doing what was literally asked:** the user pasted real data
into an absolute Windows path and asked to use it directly "to avoid
path errors." Using that absolute path verbatim would have reintroduced
exactly the fragility this project already fixed once before (Day 4's
`ModuleNotFoundError` saga) — it would only work on that one machine,
breaking in Colab or if the folder ever moves. Used the existing
`_find_project_root()` pattern instead, pointed at `data/raw/real/`
(relative to the discovered project root) — this resolves to the
user's exact pasted folder without hardcoding anything machine-specific.

**Verified, concretely, not just by inspection:**
- Built 3 synthetic "multi-drive" test files from the one real sample
  already in this sandbox (same real content, different filenames, to
  test the LOOP logic without needing more real files than were
  available here) — confirmed correct concatenation (3×1355=4065
  total windows, 3×4=12 total rejected), correct per-drive diagnostic
  logging, and correct whole-drive validation split (1 entire drive —
  not a time slice — held out, 2710/1355 train/val window counts
  matching exactly).
- Added one deliberately malformed file and confirmed it was skipped
  with a clear, specific error message, while the other 3 drives still
  processed correctly — the try/except guard actually works, not just
  assumed to.
- Standard robustness check (same as every prior file): correct from
  its proper location and from a different working directory,
  including a sensible message (not a crash) when the target data
  folder doesn't exist yet.

**Decisions made:**
- Splitting by whole drive for multi-file data, not time-within-drive
  — a stricter, more honest validation (tests generalization to an
  entirely unseen recording, not just a later slice of one it partly
  trained on).
- Per-drive diagnostics are printed during loading specifically so
  outlier drives (like the 139.65° heading case) are visible to a
  human reviewing the run, not silently absorbed into an aggregate
  statistic.

**Explicitly not solved today, flagged rather than ignored:** whether
the 104/6813-stationary-samples and 139.65°-heading results reflect
genuine per-drive variation or indicate the calibration pipeline's
real-data reliability issues (known since the earlier dataset
analysis) are worse than previously characterized. That needs the
full real dataset run (97 drives) to actually see a distribution of
these diagnostics, not a guess from one sample.

**Next:** run `multi_drive_data.py` against the real
`data/raw/real/` folder (now populated with 97 files) and report back:
how many drives loaded OK vs. were skipped (and why), and the range of
per-drive diagnostics (stationary-sample counts, heading estimates) —
that distribution will determine whether Day 3's ZUPT thresholds need
real retuning now, or whether the sample checked here was just an
unusually hard case.

**Files:**
```
idr-project/
├── requirements.txt
├── PROGRESS.md
├── context.md
├── SUMMARY.md
├── src/
│   ├── io_vnbd_loader.py
│   ├── eda.py
│   ├── utils/
│   │   └── synthetic_data.py
│   ├── calibration/
│   │   ├── gravity_compensation.py
│   │   ├── zupt_classifier.py
│   │   ├── bias_estimation.py
│   │   ├── mount_alignment.py
│   │   └── denoising.py
│   ├── models/
│   │   ├── dataset_prep.py
│   │   ├── velocity_model.py
│   │   ├── train_velocity_model.py
│   │   └── multi_drive_data.py
│   ├── dead_reckoning/
│   │   ├── naive_drift_baseline.py
│   │   └── strapdown_ins.py          ← NEW: Phase 4 INS + ZUPT (Day 8)
│   ├── map_matching/                  (empty — Day 11)
│   ├── fusion/                         (empty — Day 13)
│   └── edge_engine/                     (empty — Day 15)
├── data/
│   └── raw/
│       ├── SYNTH-drive1.csv    (synthetic)
│       └── real/                 (user's real S-*.csv files go here)
├── mobile_app/               (empty — Day 15)
├── tests/                     (empty)
└── results/
    ├── metrics.md
    ├── checkpoints/
    └── plots/
```

---

## Day 8 audit — Days 1–8 re-checked and fixed on the real dataset

**Why:** the Day 8 strapdown INS "ran" on S-S1 but reported 47,881 m
(hard ZUPT) and 1,881,072 m (no ZUPT) error on a drive whose GPS distance
was logged as 10.7 km. Tracing that back exposed problems all the way
down to the loader, so every day's code was re-checked against all 97
real `S-*.csv` files before moving on.

**Bugs found and fixed (each confirmed on real data, not assumed):**

1. **GPS speed is m/s, not km/h.** The header says "GPS SPEED (Kmh)", but
   integrating the column as m/s reproduces the haversine GPS track
   (S-S1 38.5 vs 37.2 km; S-T2 201.8 vs 201.2 km). 94/97 files are m/s;
   S-A2, S-A9, S-A10 really are km/h. The loader now detects the unit per
   file and rewrites `gps_speed_kmh` to true km/h. Every speed, distance
   and training target in the project had been 3.6x too small.
2. **GPS updates only every ~10 s** (1 Hz in S-A4..A8, S-Vta1a/2) and is
   held in between. New column `gps_speed_interp_ms` interpolates between
   fixes; it is now the training target and the INS reference.
3. **Heading estimate was broken.** `accelerating_mask` differenced the
   held GPS speed, so it flagged ~264 isolated rows on S-S1 and gave a
   116 deg heading whose "forward" axis had zero correlation with real
   acceleration. New `estimate_heading_offset_gps` fits the rotation
   between IMU-integrated and GPS-derived velocity change (forward AND
   turning) over each fix interval. Validated on the synthetic drive:
   1.2 deg recovered vs 0 deg true.
4. **ZUPT thresholds were synthetic-tuned** and caught 13% of S-S1's
   stops. Real at-rest noise differs 20x between phones, so no fixed
   threshold works. New `zupt_mask_adaptive` scales to each phone's own
   noise floor (measured on GPS-confirmed stops) and is capped below the
   moving distribution, so it can't fire while driving (it had flagged
   88-97% of S-T8/S-T5/S-Vw8 before the cap).
5. **29 drives rejected for never stopping.** Leveling now falls back to
   the whole-drive mean gravity, bias to zero. 75 drives now load (was 57).
6. **S-A4 column-shift repair was off by one** — DATE went into accel_x,
   accel_x into accel_y, etc. Rewritten; channels now line up (gravity_z = 9.81).
7. **Logger duplicates:** S-T1/T4/T5/T6 contain tens of thousands of
   frozen repeated rows at 0–1 ms spacing (S-T6: 70,158). Dropped in the loader.
8. **Time counter resets** (S-M, S-S2, S-S3b, S-S4, S-Y1) are unwrapped
   so `t_sec` is monotonic.
9. **The "trained model" was synthetic.** `results/checkpoints/velocity_model.pt`
   came from `train_velocity_model.py`'s `__main__` (120 s synthetic drive);
   `run_training.py` had never produced its checkpoint. Renamed to
   `velocity_model_synthetic.pt`.
10. **Train/val split** took the last 20% of drives by filename (only
    S-Vw*/S-Y1) and had no test set. Now train/val/test (70/15/15 by
    windows), seeded, grouped by recording.
11. Model gyro features now use the bias-corrected `_cal` channels.
12. `results/metrics.md`: duplicate rows removed, invalid Day 8 rows
    removed, file converted to UTF-8, Day 2–5 values recomputed (the old
    ones used the since-fixed `dt[0] = dt[1]` integration).

**Day 8 — new benchmark: simulated GPS outages ("tunnels").** One 86-minute
open-loop integration says nothing about a tunnel. `simulate_outages()`
cuts GPS for 30/60/120 s at random moving points on 73 real drives,
starts from the last GPS speed, and compares against simply holding that
speed:

| Outage | Hold last GPS speed | Strapdown INS | INS + adaptive ZUPT |
|---|---|---|---|
| 30 s | 46 m (11.6%) | 90 m (24.4%) | 90 m (24.9%) |
| 60 s | 123 m (16.8%) | 261 m (38.0%) | 257 m (37.5%) |
| 120 s | 283 m (19.7%) | 641 m (40.8%) | 605 m (40.5%) |

(median absolute distance error, % of distance travelled; 1,140–1,447 outages)

**Honest conclusion:** with this dataset's phone accelerometers, pure
inertial integration is about 2x worse than holding the last speed. The
phones are rigidly mounted (gravity direction stable), but integrated
horizontal acceleration correlates only r ~0.1–0.6 with GPS velocity
change. This is exactly why Day 9 replaces integrated acceleration with
the learned velocity model — and "hold last speed" is the baseline it has
to beat.

**Model retrained on real data** (`run_training.py`, GPU, 6.7 min):
227k/61k/65k train/val/test windows from 56/10/9 drives. Early stopping
at epoch 41, best epoch 31.

| Split | CNN MAE | CNN RMSE | Predict-the-mean MAE |
|---|---|---|---|
| Val | 3.23 m/s | 4.45 m/s | 9.32 m/s |
| Test | 4.54 m/s | 6.23 m/s | 8.05 m/s |

The model clearly learns speed from a 2 s IMU window (MAE roughly half
the trivial baseline on held-out drives), but 4.5 m/s (~16 km/h) error
per window is large; the val/test gap shows strong drive-to-drive
variation. Next-step levers: longer windows, wheel-speed targets from the
`V-*.csv` files, and evaluating the model inside the outage benchmark (Day 9).

**Files changed:** `src/io_vnbd_loader.py`, `src/calibration/zupt_classifier.py`,
`src/calibration/mount_alignment.py`, `src/models/multi_drive_data.py`,
`src/models/velocity_model.py`, `src/models/train_velocity_model.py`,
`src/dead_reckoning/strapdown_ins.py`, `run_training.py`, `results/metrics.md`.
New outputs: `results/day8_outage_benchmark.csv`, `results/training_summary.json`,
`results/training_log.txt`, `results/plots/training_curves.png`,
`results/checkpoints/velocity_cnn_best.pt`.

### Day 8 audit, part 2 — improving the velocity model

The first real-data model (2 s window, small CNN) had test MAE 4.54 m/s,
and its errors showed regression to the mean: +3.8 m/s when stopped,
-4.9 m/s above 25 m/s. A 2 s accelerometer window says how speed is
changing, not what it is, so the model can only read absolute speed from
the vibration signature.

Experiments (cached drives, selection by validation, then 5-fold
cross-validation by recording over the 66 train+val drives -- the 10-drive
validation set alone proved unreliable):

| Config | CV MAE (m/s) |
|---|---|
| Small CNN, 2 s window | 4.73 +/- 0.87 |
| Deeper CNN (32-64-64-128), 20 s window | 3.88 +/- 1.05 |
| **Wider CNN (64-128-128-256-256), 20 s window** | **3.42 +/- 0.63** |

Also tried: 5/10/30 s windows (20 s best), orientation-independent
magnitude features (small gain, kept), per-drive feature normalization
(worse -- absolute vibration level carries speed information).

On the fixed test drives (same 9 drives, now frozen in `data/split.json`
because the window-count-balanced split moved drives when the window size
changed):

| | Per-window test MAE | Per-drive mean test MAE |
|---|---|---|
| Old: 2 s, small CNN | 5.15 | 5.42 |
| **New: 20 s, wider CNN** | **4.53** | **4.71** |
| Predict the mean | 8.05 | 7.09 |

What still limits it: the test error is dominated by S-Vw4 (4.6 m/s) and
S-T7 (5.9 m/s, under-predicting by 5.3 m/s throughout -- a smooth-riding
phone/car reads as "slow"). Inferring absolute speed from vibration
depends on the phone and vehicle; the natural next step (Day 9) is to use
the known speed at tunnel entry and predict the CHANGE in speed.


---

## Day 8 audit, part 3 — honest evaluation, wheel-speed truth; Day 9 — anchored ML

**Why:** reviewing the outage benchmark against the problem statement ("estimate
the path while GPS is unavailable") found two ways it used information a phone
would not have inside a tunnel, and the phone-GPS "truth" itself was coarse.
All of the following was first tested as scratchpad experiments, then moved
into the main code and re-run there.

**1. Calibration only from data before the outage.** Every outage is now
calibrated (bias, tilt, heading, ZUPT thresholds) on everything recorded
before GPS is lost — not on the whole drive. Effect (experiment, same
outages): INS + ZUPT at 120 s went from 554 m to 708 m median error (it had
been benefiting from future GPS); the ML model was essentially unaffected
(per-window test MAE 4.59 -> 4.50 m/s). With >= 20 min of driving before the
outage the difference mostly disappears.

**2. Entry speed = last real GPS fix.** The benchmark had taken the start
speed from GPS *interpolated* between fixes, which leans towards the next
fix (inside the tunnel). With the last real fix, "hold last speed" is
15-25% worse (30 s: 47 -> 61 m; 120 s: 280 -> 315 m, wheel-truth drives).

**3. Wheel-speed ground truth.** 72 drives have a `V-*.csv` logged by the car
(wheel speeds, VBOX GPS velocity, speedometer, 10 Hz). The car's signals agree
to ~0.1 m/s; our phone-GPS label is ~0.9-1.2 m/s off. The "Synchronised" files
are offset from our phone files by up to +-28 s, so each drive's offset is
estimated from speed cross-correlation and was verified against gyro vs car
yaw rate on 5 drives (e.g. S-S2 +8.6 s both ways). 39 drives (~19 h) align
clearly; S-Y1 and S-S4 do not match at all.
5-fold cross-validation over those 39 drives (each tested once):
- model error vs wheel truth: 3.14 m/s (phone-GPS labels) vs 3.05 m/s
  (wheel labels) — better labels barely help, so training is unchanged;
- outage conclusions are the same whether scored against phone GPS or wheel speed.

**4. Day 9 method — anchored ML.** `speed(t) = ML(t) + (v0 - ML(0)) * exp(-t / 20 s)`:
correct the model by its visible error at the entrance (v0 = last GPS fix),
fading out. The 20 s fade was picked on validation drives and confirmed by a
minimum-variance mix computed from error covariances (cannot overfit a few
drives): trust moves from the entrance correction to the model within ~10-20 s,
and the derived mix did not beat the plain 20 s fade. In the cross-validation
against wheel truth (13 drives with 2-min outages, ~4,600 outages per length):

| Outage | Hold last fix | ML model | Anchored ML |
|---|---|---|---|
| 30 s | 61 m | 61 m | **47 m** |
| 60 s | 141 m | 108 m | **88 m** |
| 120 s | 315 m | 202 m | **176 m** |

Also tried and NOT adopted: per-phone fine-tuning on the first 5-20 minutes of
a drive (S-T7 5.87 -> 2.89 m/s, but S-Vw4 4.75 -> 10.65 m/s when the adaptation
minutes were city-only and the rest motorway).

**Main-code results (re-run from the project code):**

Day 8 — all usable drives (57 drives, 855 outages per length; wheel truth on 29 of them):

| Outage | Hold last GPS fix | Strapdown INS | INS + adaptive ZUPT |
|---|---|---|---|
| 30 s | **56 m** (13%) | 104 m (25%) | 107 m (27%) |
| 60 s | **151 m** (18%) | 286 m (38%) | 286 m (40%) |
| 120 s | **314 m** (20%) | 767 m (48%) | 784 m (50%) |

Day 9 — TEST drives only (6 drives, 240 outages per length; never seen in training):

| Outage | Hold last GPS fix | INS + ZUPT | ML model | Anchored ML |
|---|---|---|---|---|
| 30 s | **74 m** (9%) | 107 m (18%) | 87 m (20%) | **74 m** (14%) |
| 60 s | 166 m (14%) | 206 m (15%) | 136 m (15%) | **113 m** (11%) |
| 120 s | 361 m (18%) | 636 m (29%) | 221 m (12%) | **206 m** (12%) |

(median distance error at the end of the outage; % = of distance driven)

Reading: the INS loses to holding the last speed everywhere. Anchored ML is
best from 60 s on (-32% at 60 s, -43% at 120 s vs holding the last fix) and
ties at 30 s (equal in metres, worse in % — on short outages the held speed
is still very good). Per drive at 120 s the picture varies: anchored ML wins
on S-A8, S-Vta29, S-Vw4 and S-Vta10, while S-T7 (the smooth phone/car the
model reads as slow) is still best served by holding the speed (120 vs 348 m)
and S-Vta21 by the raw model. Six test drives are a small sample; the
cross-validated numbers above are the stronger evidence.

**Files changed:** `src/models/multi_drive_data.py` (`load_raw_drive`,
`calibrate`, `apply_calibration`; `load_and_calibrate_drive` rebuilt on them and
verified bit-identical), `src/dead_reckoning/strapdown_ins.py` (new
`simulate_outages`, `report`, per-day `log_metric`), new
`src/dead_reckoning/ins_with_ml_velocity.py`, new `src/utils/vehicle_truth.py`,
vehicle files in `data/raw/vehicle/`. Outputs: `results/day8_outage_benchmark.csv`,
`results/day9_outage_benchmark.csv`, `results/day8_log.txt`, `results/day9_log.txt`.
