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

**Status: Day 6 complete. Day 7 ("robustness") started — multi-drive real data loading.**

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
│   │   └── naive_drift_baseline.py
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