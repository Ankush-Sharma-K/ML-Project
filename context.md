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
| 3 — AI Velocity Estimation | 6–8 | IMU→velocity model, robustness, on-device compression | Day 6 done; Day 7 (robustness) started |
| 4 — Core Dead Reckoning | 9–10 | Strapdown INS integration, drift benchmark evaluation | — |
| 5 — Map Matching | 11–12 | OSM road graph, NHC constraints, HMM map matcher | — |
| 6 — GNSS+INS Fusion | 13–14 | UKF/EKF fusion, adaptive noise, seamless GNSS↔DR handoff | — |
| 7 — App & Deployment | 15 | Edge engine API, mobile app UI, packaging, final submission | — |

**Constraint carried through every phase:** no internet / no torch in
this sandbox. Training happens in the user's own environment.

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

### Not yet built (reserve these module paths)
| Phase | Expected file(s) |
|---|---|
| 3 (Days 7–8) | ZUPT threshold retuning (if the full real-data run confirms it's needed), compression/export script |
| 4 (Days 9–10) | `src/dead_reckoning/strapdown_ins.py` |
| 5 (Days 11–12) | `src/map_matching/osm_graph.py`, `src/map_matching/hmm_matcher.py` |
| 6 (Days 13–14) | `src/fusion/ukf_fusion.py` |
| 7 (Day 15) | `src/edge_engine/`, `mobile_app/` |

---

## 4. Results Log Format (locked)

Unchanged — see prior version.

---

## 5. Known Open Issues (carry forward until resolved)

Unchanged from the prior version, **plus:**

- **ZUPT real-data reliability needs the FULL dataset to properly
  characterize** — one real sample showed low stationary-sample counts
  and an outlier heading estimate; is this typical or unusual? Only
  the full 97-drive run will tell.
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