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

**Status: all 7 phases complete (Days 1–15) on the real IO-VNBD data. Day 15 = streaming edge engine (int8 model), route verification, HTTP API + mobile web app, packaging (`main.py`, README).**

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


---

## Day 10 — Direction and 2-D path during GPS outages

**Why:** Days 8-9 estimate how far the car travels. The problem statement also
needs the direction and the path (for the map comparison in Phase 5).

**Method (`src/dead_reckoning/heading.py`):** heading = GPS course at the
entrance + the phone's accumulated rotation about the vertical axis
(gyro . gravity direction). The gyro columns do not follow the same axis order
and signs as the gravity columns on most phones -- the standard pairing was the
best fit in only 17% of outages -- so the axis order (6 options), signs
(4 patterns) and scale are fitted on GPS course changes BEFORE each outage, like
the rest of the calibration. The fit is used when its correlation is >= 0.5
(75% of outages, median |r| 0.84); otherwise the heading falls back to the last
GPS course. The phone's fused orientation yaw was tested too but could be
calibrated in only 20% of outages, so it is not used.

**Benchmark:** outages start right after a GPS fix (so the entrance position,
speed and course are real) and are scored at the first GPS fix after 30/60/120 s:
error = metres between the estimated and the GPS end position.

Part A — heading only (true distance; 46 drives, 653 outages per length):

| Outage | Straight line (last GPS course) | Calibrated gyro heading |
|---|---|---|
| 30 s | 98 m (19%) | **59 m** (13%) |
| 60 s | 239 m (27%) | **157 m** (20%) |
| 120 s | 651 m (39%) | **439 m** (27%) |

Part B — full path on TEST drives (4 drives, 60 outages per length):

| Outage | Hold last fix + straight line (baseline) | Anchored ML + straight line | **Anchored ML + gyro (full method)** |
|---|---|---|---|
| 30 s | 120 m | 106 m | **94 m** |
| 60 s | 333 m | 231 m | **207 m** |
| 120 s | 931 m | 748 m | **747 m** |

(median distance to the GPS end position; % = of distance driven)

**Reading:** the calibrated gyro cuts the position error caused by direction by
a third to a half. The full method (anchored-ML distance + gyro heading) is the
best path estimate at every outage length (-22% / -38% / -20% vs the baseline),
but only 4 test drives qualify here, so Part A is the stronger evidence for the
heading. Even with perfect distance, a 2-minute outage still ends ~440 m off:
small heading errors grow into large sideways errors. Phone sensors alone
cannot fix this -- constraining the path to the road network (Phase 5, map
matching) is the next step.

**Files:** new `src/dead_reckoning/heading.py`; outputs
`results/day10_heading_benchmark.csv`, `results/day10_path_benchmark.csv`,
`results/day10_log.txt`; Day 10 tables in `results/metrics.md`.

---

## Day 11 — Map matching (Phase 5, part 1)

**Why:** Day 10 showed that even with perfect distance a 2-minute outage ends
~440 m off, because small heading errors grow into large sideways errors. A car
stays on roads, so the dead-reckoned path is matched to the OpenStreetMap road
network.

**Road maps (`src/map_matching/osm_loader.py`):** one road corridor per drive =
the cleaned GPS track buffered by **2 km**, cached as
`data/external_maps/<drive>.graphml`. Built from Geofabrik county extracts
(`data/external_maps/geofabrik/`, read with pyosmium using osmnx's own "drive"
filter -- checked: S-Vtb7 gives 576 nodes / 1222 edges vs 568 / 1210 from
Overpass) or, for drives whose region file is large (France), from the Overpass
API with retries. The public Overpass server rejected a bulk run of ~60 drives
("at capacity"), which is why Geofabrik is the main source. Isolated junk GPS
fixes are removed before buffering. 78 drives have a map. Not built: 7 French
train drives skipped to save time (S-T11, S-T2, S-T3, S-T4, S-T5, S-T8, S-T9);
S-T6 (val) and S-T7 (test) were still downloading when the benchmark ran;
S-Vw1 / S-Vw15 have no usable GPS.

**Leakage check:** the corridor is cut around the whole track, including the
outage. A narrow corridor would tell the matcher where the car went, so it is
wide (2 km) and, for every sample, the matcher checks that its whole search disc
lies inside the corridor with a 200 m slack; otherwise the map is not used for
that outage end and plain dead reckoning is reported. Map usable: 99% / 98% /
91% of outages at 30 / 60 / 120 s.

**Matcher (`src/map_matching/hmm_map_matcher.py`):** Hidden Markov Model
(Newson & Krumm 2009, adapted to dead reckoning). States = positions on
directed road edges (one-way streets respected); the dead-reckoned path is
sampled every 5 s; emission = distance to the road (sigma = 15 m + 8% of the
distance driven, because DR drifts) x agreement of road direction with the
gyro heading (sigma 30 deg); transition = |driving distance on the graph - DR
distance|; entrance = last GPS fix + course. Viterbi, **causal** (the position at
time t uses only samples up to t). Two outputs: *snap* = matched road point;
*route* = move the DR distance along the matched route from the entrance.
Parameters were set a priori (not tuned); *route* was chosen as the full method
on two train drives (S-A4, S-A5) before the benchmark.

**Error frame fix in `heading.py`:** end-position error is now measured in
metres around each outage's entrance fix (the old whole-drive flat-earth frame
was off by up to ~1% on 100 km drives). Day 10 baselines below are recomputed
on the same outages.

**Benchmark** (same outages as Day 10; median distance to the first GPS fix
after the outage; P90 in brackets):

Part A — true distance (38 drives, 535 outages per length):

| Outage | Straight line | Gyro heading (Day 10) | Gyro + map snap | **Gyro + map route** |
|---|---|---|---|---|
| 30 s | 95 m | 51 m | 15 m | **8 m** (218) |
| 60 s | 235 m | 141 m | 59 m | **26 m** (574) |
| 120 s | 641 m | 382 m | 278 m | **238 m** (1605) |

Part B — full pipeline, TEST drives (3 drives with a map: S-A8, S-Vta29, S-Vw4; 45 outages):

| Outage | Baseline (hold fix + course) | Anchored ML + gyro (Day 10) | **+ map route (full method)** |
|---|---|---|---|
| 30 s | 122 m | 87 m | **69 m** |
| 60 s | 353 m | 202 m | **103 m** |
| 120 s | 953 m | 726 m | **624 m** |

Part B' — TEST + VAL drives (9 drives, 135 outages): baseline 170 / 363 / 882 m,
Day 10 102 / 209 / 561 m, **full method 78 / 156 / 459 m**.

**Reading:** with the right distance, map matching puts most 30-60 s outages
within GPS noise (8 m / 26 m median, from 51 / 141 m). With the ML distance the
gain is still large at 60 s (-49% vs Day 10 on test drives) but smaller at 120 s
(-14%): the distance error from the speed model accumulates, and on long outages
the matcher sometimes follows the wrong branch at a junction -- the P90 does not
improve (wrong-road matches are far off). These are the Day 12 targets.

**Files:** new `src/map_matching/osm_loader.py`, `src/map_matching/hmm_map_matcher.py`;
`heading.py` (`map_match` hook, entrance-centred error); outputs
`results/day11_map_heading_benchmark.csv`, `results/day11_map_path_benchmark.csv`,
`results/day11_log.txt`, `results/day11_osm_log.txt`; Day 11 tables in
`results/metrics.md`. New dependencies: `osmnx`, `osmium`.
Run: `python -m src.map_matching.osm_loader` (maps), `python -m src.map_matching.hmm_map_matcher` (benchmark).

---

## Day 12 — NHC study, matcher improvements (Phase 5, part 2)

**Maps completed:** S-T6 and S-T7 built from the Overpass mirror (~13 min
each); later the 7 skipped French drives too (Geofabrik regions) -- all 87
10 Hz drives with usable GPS have a map. S-T6 has no outage that qualifies for the
benchmark, so the val set is unchanged; S-T7 joins the test set (4 drives).

**1. Non-holonomic constraint (NHC) — tested, not adopted.** The dead
reckoning already moves only along the heading (speed x heading), so the
textbook NHC (no sideways / vertical velocity) is built in. The useful extra
form is the turn relation: lateral acceleration = speed x yaw rate, which
would give an independent speed v = a_lat / omega in every turn. Checked on 15
val/test/train drives (|omega| > 0.08 rad/s, 2 s smoothing, vehicle-frame
lateral axis from the calibration): median error 3-17 m/s, correlation with true
speed near zero on most drives (best: S-T7 0.5, S-S1 0.46) -- worse than the ML
model (~3-4 m/s). The phone's lateral acceleration is too small and noisy next
to vibration and mount motion. Sign-free variants (|a_lat|/|omega|,
|a_horizontal|/|omega|) were no better.

**2. Matcher changes, tuned on TRAIN drives only** (29 drives, true speed and
speed x0.85 / x1.15 to mimic the ML model's per-drive bias):

| Change | Result on train drives | Adopted |
|---|---|---|
| Turn consistency in the transition (road direction change vs gyro heading change, sigma 20 deg) | 120 s: 194 -> 183 m (true speed); no change elsewhere | yes (default) |
| "Turn-anchored" distance (count DR distance from the last completed >= 45 deg turn instead of the entrance) | applies in only 12-23% of outages; helps with biased speed (x0.85, 120 s: 243 -> 213 m when it applies), hurts with correct speed (60 s: 52 -> 75 m) | no (computed as `map_turn`, not reported) |

With a 15% speed error every map variant ends ~430 m off at 120 s: **the map
fixes direction, not distance.**

**Benchmark** (same outages as Days 10-11; median distance to the first GPS fix
after the outage):

Part A — true distance (all 46 drives, 653 outages per length; map usable in
99% / 98% / 88% of outages):

| Outage | Straight line | Gyro heading (Day 10) | Gyro + map snap | **Gyro + map route** |
|---|---|---|---|---|
| 30 s | 98 m | 59 m | 16 m | **9 m** |
| 60 s | 239 m | 157 m | 66 m | **33 m** |
| 120 s | 651 m | 447 m | 356 m | **306 m** |

Part B — full pipeline, TEST drives (4 drives: S-A8, S-T7, S-Vta29, S-Vw4; 60 outages):

| Outage | Baseline (hold fix + course) | Anchored ML + gyro (Day 10) | + map snap | **+ map route (full method)** |
|---|---|---|---|---|
| 30 s | 120 m | 94 m | 73 m | **72 m** |
| 60 s | 332 m | 208 m | 138 m | **106 m** |
| 120 s | 929 m | 747 m | 696 m | **712 m** |

Part B' — TEST + VAL (10 drives, 150 outages): baseline 160 / 354 / 874 m,
Day 10 106 / 214 / 576 m, **full method 78 / 157 / 477 m**.

**Reading:** the map halves the 60 s error (208 -> 106 m on test drives) and
puts most outages with correct distance within GPS noise. At 120 s the gain is
small on the 4 test drives (747 -> 712 m): adding S-T7, whose speed the model
gets badly wrong (known per-drive bias, Day 9), erases most of it; on the larger
test+val set the full method is still -17% at 120 s. The remaining error is the
DISTANCE from the speed model. Phase 5 is complete; better speed (or fusion
with any absolute fix, Phase 6) is what can still move the 120 s number.

**Files:** `hmm_map_matcher.py` (turn consistency on by default, `map_turn`
output, Day 12 benchmark), `heading.py` (`map_turn` rows); outputs
`results/day12_map_heading_benchmark.csv`, `results/day12_map_path_benchmark.csv`,
`results/day12_log.txt`, `results/day11_osm_log_france.txt`; Day 12 tables in
`results/metrics.md`.

---

## Days 1–12 completion pass — nothing left open

Everything still open in Phases 1–5 was finished before Phase 6:

**1. On-device model (Phase 3) — `src/models/compress_model.py`.** The model is
exported for the PyTorch Lite interpreter (PyTorch Mobile / ExecuTorch). The
exported files take the RAW 20 s window (batch, 200, 8 features in
`ckpt["feature_cols"]` order) and return speed in m/s; the training
normalization is inside the file, so the app cannot apply the wrong statistics.

| Model | File | Val MAE | Test MAE | Latency (1 window, 1 CPU thread) |
|---|---|---|---|---|
| Original checkpoint | 2.63 MB | 2.495 m/s | 4.534 m/s | 9.8 ms |
| `velocity_model_fp32.ptl` (BatchNorm folded) | 2.64 MB | 2.495 | 4.534 | 6.4 ms |
| **`velocity_model_int8.ptl`** (static int8, calibrated on 2000 TRAIN windows) | **0.71 MB** | 2.473 | 4.580 | **0.6 ms** |

int8 vs original on the 32,165 test windows: median difference 0.18 m/s, 99%
within 1.3 m/s, max 2.9 m/s (max 8.6 m/s on one val window). `optimize_for_mobile`
stores every int8 weight twice (1.32 MB), so the int8 file is exported without
it. The int8 file is the one for the phone. Run:
`python -m src.models.compress_model` (writes `results/compression_summary.json`,
`results/compression_log.txt`).

**2. All road maps.** The 7 French train drives skipped on Day 11 were built
from Geofabrik region files (`--max-region-mb 500`): all 87 10 Hz drives with
usable GPS now have a map (S-Vw1 / S-Vw15 have no GPS). The Day 12 benchmark was
re-run: Part A now covers all 46 drives (table above); Part B is unchanged (the
French drives are train drives).

**3. Day 10 re-run with the entrance-centred error.** `results/metrics.md` Day 10
rows are now measured the same way as Days 11–12. Changes are small: gyro heading
with true distance 59 / 157 / **447** m (was 439 at 120 s); full method on test
drives 94 / **208** / 747 m (was 207 at 60 s). Old log kept as
`results/day10_log_before_frame_fix.txt`.

**4. Unit tests — `tests/` (19 tests, all passing).** Synthetic data only (no
dataset or internet needed): speed-unit detection, GPS speed interpolation,
sliding windows and the gap guard, anchored ML speed, 2-D path, gyro axis /
sign / scale fit, lat/lon round trip, the OSM "drive" filter, edge segments,
the HMM matcher on a T-junction (follows the turn despite gyro drift; refuses
an off-road entrance), normalization, the export wrapper, Lite round trip, int8
closeness. Run: `python -m pytest tests -q` (~2 min, mostly imports).

**5. Data limit, documented, not "fixed".** Only 4 of the 9 test drives have
outages that qualify for the full-path benchmark: S-Vta10 (3 min), S-Vta13
(1 min), S-Vta21 (3 min), S-Vtb7 (1 min), S-Vw5 (2 min) are too short for
2 minutes of driving before an outage plus a 2-minute outage. Changing the frozen split
to get more test outages would bias the result, so the larger test + val table
(10 drives) is reported next to it instead.

New dependency: `pytest` (requirements.txt).

---

## Day 13 — GNSS + INS fusion (Phase 6)

**Feasibility first (VAL drives only).** Idea: learn the speed model's per-drive
bias while GPS is available and carry it into the outage (the 120 s error is
distance, Day 12). Result: the model's error over the 5 minutes before an outage
barely predicts its error during it (correlation 0.19); correcting with it did
not help at 120 s (142 -> 143 m distance error) and badly hurt S-Vw16a
(75 -> 425 m). So the filter does NOT carry a long-term bias: the ML-bias state
decays with 20 s, the same as Day 9's anchored ML.

**Filter (`src/fusion/kalman_filter.py`, `src/fusion/gnss_ins_fusion.py`).** One
EKF at 10 Hz before, during and after the outage. State: east, north, heading,
speed, ML-speed bias (Gauss-Markov, 20 s), gyro bias. Inputs: calibrated gyro
turn rate (Day 10 per-phone mapping). Updates: ML speed once per second; at each
GPS fix position (noise from the phone's `gps_accuracy_m` -- adaptive), speed,
course (when moving). Position and course are chi-square gated (99.9%).
Calibration, gyro mapping and the ML model use data before the outage only;
outage starts are identical to Days 10-12 (`heading.path_outage_starts`,
refactored out of `simulate_path_outages`, checked identical).

**Bug found and fixed while building it:** with GPS only every ~9 s, the gyro
heading in a sharp turn is 20-30 deg off by the next fix; the gate then rejected
the genuine fix, the filter drifted further and never recovered (219 rejections
on S-Vw3). Fixes: heading process noise grows with the turn rate (the fitted
gyro scale is uncertain), and a fix rejected right after another rejection is
accepted with its noise inflated by NIS / threshold.

**Tuning (VAL drives only, 6 settings):** gyro-scale noise 0.2 -> 0.4 and GPS
course noise 8 -> 4 deg (120 s end error on val: 449 -> 395 m); everything else
a priori. TEST drives were used only for the final run.

**Benchmark** (same outages as Days 10-12; median metres; the Day 10 / Day 12
references are recomputed in the same run with the Phase 6 ML sampling grid,
which is why they differ slightly from the Day 12 table -- 120 s map-matched
errors are sensitive, P90 ~1.8 km):

1. Real-time end position, TEST drives (4 drives, 60 outages):

| Outage | Day 10 (anchored ML + gyro) | EKF | Day 12 (+ map) | **EKF + map** |
|---|---|---|---|---|
| 30 s | 97 | 113 | 72 | **66** |
| 60 s | 224 | 200 | 110 | **112** |
| 120 s | 751 | 565 | 813 | **484** |

TEST + VAL (10 drives, 150 outages; val drives were used for tuning): Day 12
79 / 168 / 505 m -> **EKF + map 80 / 154 / 412 m**.

2. Tunnel track after GPS returns (error at the hidden GPS fixes INSIDE the
outage, TEST drives; 578 / 1163 / 2367 fixes):

| Outage | EKF real time | Straight line entry -> exit | Day 10 DR + end error spread | **EKF + RTS smoother** |
|---|---|---|---|---|
| 30 s | 22 | 23 | 15 | **10** |
| 60 s | 45 | 52 | 41 | **21** |
| 120 s | 118 | 134 | 140 | **45** |

3. Robustness (TEST + VAL): 15% of the GPS fixes before the outage replaced by
200 m jumps with unchanged reported accuracy; error predicting the entrance
fix: last fix + extrapolation 20 m (P90 198 m), EKF without gate 26 m
(P90 207 m), **EKF with gate 20 m (P90 97 m)**. The gate rejected 1143 of 1147
injected outliers -- and 443 genuine fixes (sharp turns); the recovery rule
keeps those from causing divergence.

**Reading:** fusion helps most where Phase 5 stopped: at 120 s the EKF + map is
-40% vs Day 12 on the test drives (and -18% on test + val), mainly because the
filter's heading (GPS course + gyro, with uncertainty) enters the outage better
than the raw last course, so the matcher picks the right roads more often. At
30-60 s it is level with Day 12. After GPS returns, the RTS smoother rebuilds the
tunnel track 2-3x more accurately than any baseline (45 m vs 134+ m at 120 s) --
the "seamless" part: the logged track is corrected, not just the current
position. The gate halves the tail error under GPS outliers; its cost is
rejecting some genuine fixes in sharp turns.

**Files:** new `src/fusion/kalman_filter.py`, `src/fusion/gnss_ins_fusion.py`,
`tests/test_fusion.py` (5 tests; 24 in total); `heading.py`
(`path_outage_starts`). Outputs `results/phase6_end_benchmark.csv`,
`results/phase6_track_benchmark.csv`, `results/phase6_robust_benchmark.csv`,
`results/day13_log.txt`; Day 13 tables in `results/metrics.md`.
Run: `python -m src.fusion.gnss_ins_fusion`.

---

## Day 15 — Edge engine, route verification, app, packaging (Phase 7)

**Phase 3 gap found while re-reading the brief** ("start with simple models
before trying neural networks"): no classical model had been tried.
`src/models/simple_baselines.py` trains them on the SAME windows and split
(72 hand-made features per window: statistics and 0-1 / 1-3 / 3-5 Hz band
energy per channel):

| Model | Val MAE | Test MAE |
|---|---|---|
| Predict the training mean | 9.22 | 8.05 |
| Ridge regression | 6.21 | 6.51 |
| Gradient boosting | 3.32 | 5.18 |
| **Deep CNN** | **2.50** | **4.53** |

The CNN is 12% better than the best classical model on test.

**Second gap: Stage 6 of the brief (route verification) had not been built.**
It is part of the engine below.

**Streaming engine (`src/edge_engine/engine.py`).** `IDREngine.push(sample)`
takes one 10 Hz phone sample and returns position, heading, speed, sigma and
status (INIT / GNSS / DR / DR+MAP): gravity removal, rolling calibration and
gyro mapping (last 10 min, refreshed every 60 s while GPS is available), the
**int8 on-device model** once per second, the Phase 6 EKF (`VehicleEKF`,
refactored out of `run_ekf`; outputs checked bit-identical), HMM map matching
during the outage (every 5 s), RTS-corrected tunnel track when GPS returns,
origin re-anchoring every 5 km, and the route monitor. Speed: profiling cut a
12-minute replay from 49 s to 12.6 s (~58x real time): window features in
numpy instead of pandas (checked identical to the training pipeline),
matching every 5 s, single-thread inference.

**Route verification (`RouteMonitor`).** ON / OFF route against a planned
route polyline. Evaluated with two planned routes per outage: the route
actually driven (any alarm = false alarm) and a "wrong turn" route (shortest
road path that avoids the driven roads; only outages where the car really
ends > 100 m from it; an alarm counts only if it is not raised before the car
actually left the route, 5 s tolerance). The first rule (40 m, tighter when
map-matched, 3 s hold) gave false alarms in 60% of test outages; the rule was
then tuned on VAL drives only (120 settings): distance > max(40 m, 2 sigma)
for 10 s. A benchmark bug was found and fixed on the way (the wrong route
lacked the pre-tunnel part, so the monitor was already OFF before the tunnel).

**Benchmark (`src/edge_engine/replay.py`, same outages as Days 10-13, fresh
engine per outage starting 10 min before the tunnel):**

Real-time end error, TEST drives (median m, 60 outages):

| Method | 30 s | 60 s | 120 s |
|---|---|---|---|
| Phase 6 batch EKF + map | 66 | 112 | 484 |
| Engine, fp32 model + map | 73 | 130 | 374 |
| **Engine, int8 model + map (deployed)** | **69** | **130** | **379** |

TEST + VAL (10 drives): batch 80 / 154 / 412 m, deployed engine 80 / 156 / 376 m.
The engine is not identical to the batch run by design (rolling 10-minute
calibration vs all data before the outage, 10 vs 5 minutes filter warm-up,
ML prediction grid); int8 vs fp32 costs nothing measurable.

Tunnel track after the 120 s outage (hidden GPS fixes): real time 78 m ->
**RTS corrected 43 m** (TEST), 92 -> 52 m (TEST + VAL).

Route verification inside the tunnel, TEST: **5 / 60 false alarms (8%)**,
**35 / 54 wrong turns detected (65%)**, 7 premature alarms, median 7.8 s after
the car left the route (TEST + VAL: 26 / 150, 92 / 135). The offline tuning
estimate on test was 5% / 76%; the full replay (10 Hz updates, state carried
in from before the tunnel) is the number reported.

**API + app.** `src/edge_engine/server.py` (standard library; localhost unless
`--host`): `/api/health`, `/api/drives`, `/api/replay` (cached in
`results/replays/`), serves `mobile_app/` -- a mobile-first Leaflet app that
plays a simulated tunnel: true path, live estimate coloured by status,
uncertainty circle, speed / heading / sigma, route status, corrected track
and the error at the exit. Offline demo bundle (`make_demo.py`): first outage
of every TEST drive, correct and wrong route -- not cherry-picked (S-Vta29's
wrong turn is not detected and ends 610 m off). Checked over HTTP (all files
and endpoints, error handling) and `node --check`; not yet tried in a browser.

**Packaging.** `main.py` (app / test / train / compress / baselines / maps /
bench / demo-bundle), README rewritten (results, quick start, commands,
layout; the team's original brief kept at the end, backup in
`results/README_original_brief_backup.md`), 27 unit tests passing
(+3 engine / route-monitor tests).

**Files:** new `src/models/simple_baselines.py`, `src/edge_engine/engine.py`,
`replay.py`, `server.py`, `make_demo.py`, `mobile_app/index.html`, `app.js`,
`style.css`, `mobile_app/assets/demo/`, `tests/test_edge_engine.py`, `main.py`;
`gnss_ins_fusion.py` (`VehicleEKF`). Outputs: `results/simple_baselines.json`,
`results/phase7_engine_end.csv`, `phase7_engine_track.csv`,
`phase7_route_verification.csv`, `results/day15_log.txt`; Day 15 rows in
`results/metrics.md`.
