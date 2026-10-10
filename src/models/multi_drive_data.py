"""
Multi-drive data loading -- loops the Days 1-4 calibration pipeline and
Day 6 part 1's windowing over an entire folder of real `S-*.csv` drives,
rather than the single hardcoded file every prior day's `__main__`
block used.

WHY A SEPARATE FILE, NOT JUST A LOOP PASTED INTO train_velocity_model.py:
real multi-drive loading has its own real failure modes (below) that
single-file testing never exercised -- worth isolating and getting
right on its own, same reasoning as every other multi-part day this
project has used.

WHY PER-DRIVE TRY/EXCEPT, NOT ASSUMED TO ALWAYS SUCCEED: tested the
full Day 3-4 calibration pipeline against the one real sample already
in this sandbox (`S-S3b_real_sample.csv`) before writing this file, not
after. It didn't crash -- but ZUPT flagged only 104/6813 samples
stationary (consistent with the already-known real-noise-floor issue:
Day 3's thresholds are tuned on synthetic data), and mount alignment's
heading estimate came out as 139.65 degrees, a visible outlier worth a
human looking at, not necessarily wrong (different recording sessions
may genuinely have the phone mounted differently each time). Some real
drives WILL raise real exceptions here too -- e.g. `bias_estimation`'s
"No stationary samples in mask" if a drive genuinely never stops, or
`mount_alignment`'s "mean horizontal acceleration ~0" if a drive has no
clear accelerating phase. A single bad drive must not take down an
entire multi-drive loading run.

WHY SPLIT BY WHOLE DRIVE, NOT BY TIME WITHIN ONE CONCATENATED SEQUENCE:
Day 6 part 1's `train_val_split_by_time` assumes ONE continuous drive
with overlapping windows; splitting a multi-drive concatenation by time
order would put early-and-late windows from the SAME drives on both
sides of the split (drives don't arrive in a single global time order
across separate recording sessions), which isn't a clean split at all.
Splitting by whole drive -- some entire drives held out for validation,
never touched during training -- is the correct generalization for
multiple files, and is also a strictly harder, more honest validation
setup (the model is tested on an ENTIRE unseen drive, not just a later
slice of a drive it partly trained on).

WHY AUTO-DETECTED SAMPLE RATE, NOT A HARDCODED 10 Hz ASSUMPTION (added
after a real run surfaced this, not anticipated in advance): three real
drives (`S-A2.csv`, `S-A3.csv`, `S-A13.csv`) were rejected entirely --
EVERY window, not just an isolated reset -- by `dataset_prep.build_windows`'s
dt guard. Investigated directly rather than just loosening the
tolerance blindly: these three drives' `TIME SINCE START` column has a
median spacing of ~0.5 s, i.e. they were genuinely recorded at ~2 Hz,
not the dataset's nominal 10 Hz -- not corrupted, just a different (and
consistent) real sample rate, likely phone hardware/OS scheduling
during that specific recording session. `detect_nominal_dt_s()` measures
each drive's own actual median sample interval instead of assuming a
fixed 10 Hz for every file. Drives within `rate_tolerance` of the
expected rate use their own measured interval (correctly absorbs minor
inter-device jitter, e.g. a phone that actually averages 9.8 Hz instead
of exactly 10). Drives further off than that are NOT silently mixed in
at the standard `window_size` -- a "20-sample window" would mean ~2 s
for a normal drive but ~10 s for one of these 2 Hz drives, a materially
different training example despite having the same shape -- they're
instead reported separately as `rate_mismatch_drives`, excluded from
this batch, left for a deliberate decision (resample them, or train a
separate model on them, or just accept the data loss) rather than
silently corrupting what "one window" means across the combined dataset.

WHERE YOUR DATA GOES: this expects `data/raw/real/` (a subfolder of the
project's existing `data/raw/`, alongside the synthetic
`SYNTH-drive1.csv`) containing your real `S-*.csv` files directly (no
further subfolders needed -- flatten them, as discussed). Deliberately
NOT an absolute path: an absolute Windows path would only work on one
machine and break the moment this runs in Colab or anywhere else --
`_find_project_root()` (same pattern as every prior file) finds the
project root regardless of machine or OS, and `data/raw/real` is just a
subfolder relative to that, so it resolves to the exact folder you
pasted files into without hardcoding anything location-specific.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def detect_nominal_dt_s(t_sec: np.ndarray) -> float:
    """Median of positive, non-huge consecutive gaps in a time column --
    a drive's actual, real sample interval, rather than assuming a
    fixed 10 Hz for every file. Excludes negative gaps (timestamp
    resets, see Day 6 part 1) and excludes gaps over 5s (treated as a
    genuine pause/dropout, not part of the regular sampling rhythm) so
    neither drags the median.
    """
    gaps = np.diff(t_sec)
    valid = gaps[(gaps > 0) & (gaps < 5.0)]
    if len(valid) == 0:
        raise ValueError(
            "No valid positive gaps found in time column -- cannot "
            "determine this drive's sample rate."
        )
    return float(np.median(valid))


def load_and_calibrate_drive(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Run one real drive through the full Days 1-4 pipeline: load ->
    gravity compensation -> ZUPT -> bias correction -> mount alignment.
    Adds `gps_speed_ms`. Returns (calibrated_df, diagnostics) so callers
    can log/inspect per-drive quality without re-deriving it.

    Real files need `header=0` (they have their own header row, unlike
    the synthetic generator's raw-no-header output) -- confirmed
    against a real sample before writing this, not assumed.
    """
    from src.io_vnbd_loader import load_smartphone_drive
    from src.calibration.gravity_compensation import compute_linear_acceleration
    from src.calibration.zupt_classifier import zupt_mask_with_fallback
    from src.calibration.bias_estimation import estimate_constant_bias, apply_bias_correction
    from src.calibration.mount_alignment import estimate_mount_rotation, apply_mount_alignment

    df = load_smartphone_drive(csv_path, header=0)
    df = compute_linear_acceleration(df)
    mask, zupt_diag = zupt_mask_with_fallback(df)
    n_stationary = int(mask.sum())
    if n_stationary == 0:
        # The GPS-speed-gated fallback already tried everything reasonable:
        # GPS speed never drops low enough anywhere in this drive for ANY
        # stationary-based calibration (bias estimation AND leveling both
        # need this mask). Not fixable by a threshold tweak, so fail with
        # a specific reason rather than the old generic message.
        raise ValueError(
            "No stationary-ish samples found even with the GPS-speed-gated "
            "fallback -- GPS speed never drops low enough anywhere in this "
            "recording for bias/leveling calibration (structural, not a "
            "threshold-tuning issue)."
        )

    bias = estimate_constant_bias(df, mask)
    df = apply_bias_correction(df, bias)

    mount = estimate_mount_rotation(df, mask)  # raises if no accelerating phase found
    df = apply_mount_alignment(df, mount["R_align"])

    df["gps_speed_ms"] = df["gps_speed_kmh"] / 3.6

    nominal_dt_s = detect_nominal_dt_s(df["t_sec"].to_numpy())

    diagnostics = {
        "n_samples": len(df),
        "n_stationary": n_stationary,
        "pct_stationary": round(100 * n_stationary / len(df), 1),
        "heading_offset_deg": round(mount["heading_offset_deg"], 2),
        "n_accelerating_samples": mount["n_accelerating_samples"],
        "nominal_dt_s": round(nominal_dt_s, 4),
        "used_zupt_fallback": zupt_diag["used_fallback"],
    }
    return df, diagnostics


def build_windows_for_drives(
    csv_paths: list[str | Path],
    feature_cols: list[str],
    target_col: str = "gps_speed_ms",
    window_size: int = 20,
    stride: int = 5,
    expected_dt_s: float = 0.1,
    rate_tolerance: float = 0.3,
) -> dict:
    """Load + calibrate + window EVERY drive in `csv_paths`, skipping
    (not crashing on) any drive that fails, and concatenate the rest.

    Each drive's ACTUAL sample rate is auto-detected (`detect_nominal_dt_s`)
    rather than assumed to be `expected_dt_s` -- a drive within
    `rate_tolerance` (fractional, default 0.3 = within 30%) of
    `expected_dt_s` uses its own detected interval for the dt-anomaly
    guard (correctly absorbs ordinary inter-device jitter). A drive
    further off than that is NOT windowed at the standard `window_size`
    (which would then span a very different real-world duration than
    every other drive's windows) -- it's reported in
    `rate_mismatch_drives` instead and excluded from X/y, for a
    deliberate decision rather than silent inconsistency.

    Returns {"X", "y", "start_idx", "drive_id" (which csv_path each
    window came from, by index into csv_paths), "n_rejected" (total,
    across all drives), "per_drive_diagnostics" (dict keyed by path,
    from load_and_calibrate_drive), "failed_drives" (dict: path ->
    error message, for drives that raised an exception),
    "rate_mismatch_drives" (dict: path -> detected_dt_s, for drives
    skipped specifically for having a too-different sample rate)}.
    """
    from src.models.dataset_prep import build_windows

    X_parts, y_parts, start_idx_parts, drive_id_parts = [], [], [], []
    per_drive_diagnostics = {}
    failed_drives = {}
    rate_mismatch_drives = {}
    total_rejected = 0

    for drive_idx, csv_path in enumerate(csv_paths):
        csv_path = Path(csv_path)
        try:
            df, diagnostics = load_and_calibrate_drive(csv_path)

            detected_dt = diagnostics["nominal_dt_s"]
            rel_diff = abs(detected_dt - expected_dt_s) / expected_dt_s
            if rel_diff > rate_tolerance:
                rate_mismatch_drives[str(csv_path)] = detected_dt
                print(f"  RATE MISMATCH {csv_path.name}: detected "
                      f"{detected_dt:.3f}s/sample ({1/detected_dt:.1f} Hz), "
                      f"expected ~{expected_dt_s}s/sample ({1/expected_dt_s:.0f} Hz) "
                      f"-- excluded from this batch, not windowed")
                continue

            windows = build_windows(
                df, feature_cols=feature_cols, target_col=target_col,
                window_size=window_size, stride=stride,
                nominal_dt_s=detected_dt,  # this drive's OWN rate, not the global default
            )
        except Exception as e:  # noqa: BLE001 -- intentionally broad: one
            # bad real drive (missing stationary phase, no accelerating
            # phase, malformed file, anything) must not kill the whole run
            failed_drives[str(csv_path)] = f"{type(e).__name__}: {e}"
            print(f"  SKIPPED {csv_path.name}: {type(e).__name__}: {e}")
            continue

        diagnostics["n_windows"] = len(windows["y"])
        diagnostics["n_rejected"] = windows["n_rejected"]
        per_drive_diagnostics[str(csv_path)] = diagnostics
        total_rejected += windows["n_rejected"]

        X_parts.append(windows["X"])
        y_parts.append(windows["y"])
        start_idx_parts.append(windows["start_idx"])
        drive_id_parts.append(np.full(len(windows["y"]), drive_idx, dtype=int))

        print(f"  OK {csv_path.name}: {len(windows['y'])} windows "
              f"({diagnostics['n_stationary']} stationary samples, "
              f"heading={diagnostics['heading_offset_deg']} deg)")

    if not X_parts:
        raise ValueError(
            f"Every drive failed or was rate-mismatched "
            f"({len(failed_drives)} failed, {len(rate_mismatch_drives)} "
            f"rate-mismatched, out of {len(csv_paths)}). See failed_drives "
            f"/ rate_mismatch_drives for reasons."
        )

    return {
        "X": np.concatenate(X_parts, axis=0),
        "y": np.concatenate(y_parts, axis=0),
        "start_idx": np.concatenate(start_idx_parts, axis=0),
        "drive_id": np.concatenate(drive_id_parts, axis=0),
        "drive_paths": [str(Path(p)) for p in csv_paths],  # index-aligned with drive_id values
        "n_rejected": total_rejected,
        "per_drive_diagnostics": per_drive_diagnostics,
        "failed_drives": failed_drives,
        "rate_mismatch_drives": rate_mismatch_drives,
    }


def train_val_split_by_drive(windows: dict, val_fraction: float = 0.2) -> tuple[dict, dict]:
    """Split by WHOLE DRIVE, not by time within a concatenated
    sequence -- see module docstring for why this is the correct
    generalization of Day 6 part 1's time-based split for multiple
    files. The last `val_fraction` of successfully-loaded drives
    (by their order in `drive_paths`, i.e. the order passed to
    `build_windows_for_drives`) become validation; the rest train.
    Pass your own pre-shuffled `csv_paths` order if you want a
    different split, rather than editing this function.
    """
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be between 0 and 1.")

    unique_drive_ids = sorted(set(windows["drive_id"].tolist()))
    n_val_drives = max(1, int(len(unique_drive_ids) * val_fraction))
    val_drive_ids = set(unique_drive_ids[-n_val_drives:])
    train_drive_ids = set(unique_drive_ids) - val_drive_ids

    val_mask = np.isin(windows["drive_id"], list(val_drive_ids))
    train_mask = ~val_mask

    def _subset(mask):
        return {
            "X": windows["X"][mask], "y": windows["y"][mask],
            "start_idx": windows["start_idx"][mask], "drive_id": windows["drive_id"][mask],
        }

    train = _subset(train_mask)
    val = _subset(val_mask)
    train["val_drive_paths"] = val["val_drive_paths"] = [
        windows["drive_paths"][i] for i in sorted(val_drive_ids)
    ]
    return train, val


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    def _find_project_root(start: "_Path") -> "_Path":
        """Walk upward from `start` until a directory containing both
        'src' and 'data' is found. Robust to this file being moved to a
        different depth under src/ -- unlike counting a fixed number of
        .parent levels, which silently breaks (ModuleNotFoundError) if
        the file isn't exactly where the count assumes."""
        for candidate in [start] + list(start.parents):
            if (candidate / "src").is_dir() and (candidate / "data").is_dir():
                return candidate
        raise RuntimeError(
            f"Could not locate the idr-project root (a folder containing "
            f"both 'src/' and 'data/') above {start}. Check that both "
            f"folders exist somewhere above this file."
        )

    PROJECT_ROOT = _find_project_root(_Path(__file__).resolve().parent)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.velocity_model import FEATURE_COLS_DEFAULT

    # Relative to the project root, NOT an absolute path -- resolves to
    # your D:\...\idr-project\data\raw\real exactly, without hardcoding
    # anything specific to your machine. See module docstring.
    real_data_dir = PROJECT_ROOT / "data" / "raw" / "real"

    if not real_data_dir.is_dir():
        print(f"No folder found at {real_data_dir}. Put your real S-*.csv "
              f"files there (flat, no subfolders needed) and re-run.")
        sys.exit(1)

    csv_paths = sorted(real_data_dir.glob("S-*.csv"))
    if not csv_paths:
        print(f"{real_data_dir} exists but contains no S-*.csv files.")
        sys.exit(1)

    print(f"Found {len(csv_paths)} real drive files in {real_data_dir}\n")

    windows = build_windows_for_drives(
        csv_paths, feature_cols=FEATURE_COLS_DEFAULT, window_size=20, stride=5,
    )

    n_ok = len(windows["per_drive_diagnostics"])
    n_failed = len(windows["failed_drives"])
    n_rate_mismatch = len(windows["rate_mismatch_drives"])
    print(f"\n{n_ok} drives loaded OK, {n_failed} failed, {n_rate_mismatch} "
          f"excluded for a different sample rate, "
          f"{len(windows['y'])} total windows, {windows['n_rejected']} rejected by the dt guard")

    if windows["failed_drives"]:
        print("\nFailed drives and why:")
        for path, reason in windows["failed_drives"].items():
            print(f"  {Path(path).name}: {reason}")

    if windows["rate_mismatch_drives"]:
        print("\nExcluded for a different sample rate (not windowed, not an error):")
        for path, dt in windows["rate_mismatch_drives"].items():
            print(f"  {Path(path).name}: {dt:.3f}s/sample ({1/dt:.1f} Hz)")

    train, val = train_val_split_by_drive(windows, val_fraction=0.2)
    print(f"\nTrain: {len(train['y'])} windows from "
          f"{n_ok - len(val['val_drive_paths'])} drives")
    print(f"Val:   {len(val['y'])} windows from {len(val['val_drive_paths'])} drives "
          f"(held out entirely, not just a later time slice)")
    print("Held-out validation drives:", [Path(p).name for p in val["val_drive_paths"]])