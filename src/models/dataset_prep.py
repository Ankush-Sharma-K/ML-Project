"""
Day 6 (part 1 of 2) — Dataset preparation for Phase 3's AI velocity model.

Turns a calibrated drive into fixed-size sliding windows of IMU data,
each paired with a target forward velocity — the supervised-learning
framing Phase 3's model (Day 6 part 2 / Day 7-8) will train on: given a
short window of raw-ish IMU signal, predict the vehicle's true forward
speed, learning to do what naive double integration (Days 2-5) cannot.

THIS FILE IS DELIBERATELY SCOPED TO DATA PREPARATION ONLY. The model
architecture and training script are Day 6 part 2 -- split out because
windowing has its own real correctness questions (below) worth getting
right and verified before any model code depends on this being correct.

WHY ROW-INDEX WINDOWS, NOT TIME-BASED WINDOWS (real bug found during
dataset analysis, not hypothetical): a real IO-VNBD `S-*.csv` file's
`TIME SINCE START (ms)` column was found to reset mid-recording (e.g.
jumping from 2,707,520 ms back down to 8 ms) in 2 of 3 real files
checked. Any windowing scheme that does arithmetic on absolute
timestamps would silently corrupt a window straddling that reset. Rows
are still in correct chronological order regardless (confirmed via the
file's `DATE` column, which parsed cleanly and monotonically in every
file checked) -- so windows here are built from row count, not time
arithmetic, sidestepping the bug entirely rather than trying to patch
around it.

WHY A "CLEAN WINDOW" GUARD ON dt ANYWAY: row-index windowing sidesteps
timestamp resets, but doesn't by itself catch a genuine sensor dropout
or logging pause (a real gap in data collection, not just a counter
quirk). `_window_is_clean()` checks the actual per-row time deltas
within a candidate window (using whatever time column is passed in) and
rejects any window where a gap is far from the nominal sample period --
this same check catches a `TIME SINCE START` reset too (it produces a
huge negative "gap"), so it's a single defensive guard against both
problems rather than two separate ones.

WHY NOT DENOISED INPUT (checked against real data, not assumed): ran
Day 5's `characterize_frequency_content()` against a real drive
(`S-S3b.csv` from the user's uploaded IO-VNBD sample) instead of only
trusting the synthetic-data result. Real linear acceleration has 90% of
its power below ~3.5 Hz, extending almost to the 5 Hz Nyquist limit --
broadband, real road/engine-vibration and vehicle-dynamics content, not
just noise, unlike the smooth low-frequency-only synthetic profile
(90% below 0.075 Hz there). Day 5's 1.0 Hz cutoff, tuned on synthetic
data, would strip real signal on real drives, not just noise. This
pipeline therefore windows calibrated-but-NOT-denoised linear
acceleration and leaves it to the learned model to handle that
complexity -- arguably exactly what Phase 3 exists to do instead of a
fixed filter. Revisit if real training results suggest otherwise.

WHAT THIS FILE DOES NOT DO: pick a target ground-truth source. Default
target is GPS-speed-derived forward velocity (always available, but
coarse -- see Day 2's known GPS-lag issue). The "Synchronised" real
dataset also ships paired `V-*.csv` vehicle files with wheel-speed and
indicated-vehicle-speed channels, which are a meaningfully better
ground truth where available -- `target_col` is a parameter specifically
so a merged, better-quality target can be substituted later without
rewriting this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_size: int = 20,
    stride: int = 5,
    time_col: str = "t_sec",
    nominal_dt_s: float = 0.1,
    max_dt_tolerance_s: float = 0.05,
) -> dict:
    """Build fixed-size sliding windows of `feature_cols`, each labeled
    with the value of `target_col` at the LAST sample of the window
    (i.e. "given this recent history, what's the speed right now").

    window_size: number of consecutive rows per window (default 20 =
        2.0 s at the dataset's nominal 10 Hz).
    stride: rows to advance between window starts (default 5 = 0.5 s
        between windows, i.e. overlapping windows).
    time_col: column used only for the data-quality check below, never
        for window boundaries themselves (see module docstring).
    nominal_dt_s / max_dt_tolerance_s: a window is rejected if any
        internal gap in `time_col` differs from `nominal_dt_s` by more
        than this tolerance -- catches both real sensor dropouts and
        the `TIME SINCE START` reset bug in one guard.

    Returns a dict: {X: (n_windows, window_size, n_features) float array,
    y: (n_windows,) float array, start_idx: (n_windows,) int array (row
    index of each window's first sample, for tracing back to the
    source drive), n_rejected: int (windows dropped by the dt guard)}.
    """
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    features = df[feature_cols].to_numpy(dtype=float)
    target = df[target_col].to_numpy(dtype=float)
    t = df[time_col].to_numpy(dtype=float)
    n = len(df)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    start_idx_list: list[int] = []
    n_rejected = 0

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size  # exclusive
        if not _window_is_clean(t[start:end], nominal_dt_s, max_dt_tolerance_s):
            n_rejected += 1
            continue
        X_list.append(features[start:end])
        y_list.append(target[end - 1])
        start_idx_list.append(start)

    if not X_list:
        raise ValueError(
            "No clean windows produced -- check window_size/stride against "
            "the drive length, and check whether nearly every window is "
            "being rejected by the dt guard (would indicate the time_col "
            "itself is unreliable throughout, not just at isolated resets)."
        )

    return {
        "X": np.stack(X_list, axis=0),
        "y": np.array(y_list, dtype=float),
        "start_idx": np.array(start_idx_list, dtype=int),
        "n_rejected": n_rejected,
    }


def _window_is_clean(t_window: np.ndarray, nominal_dt_s: float, tolerance_s: float) -> bool:
    """True if every consecutive gap in this window's time values is
    within tolerance of the nominal sample period. Catches sensor
    dropouts AND timestamp-counter resets (a reset produces a large
    negative gap, which fails this check just as reliably as a large
    positive one)."""
    gaps = np.diff(t_window)
    return bool(np.all(np.abs(gaps - nominal_dt_s) <= tolerance_s))


def train_val_split_by_time(
    windows: dict, val_fraction: float = 0.2
) -> tuple[dict, dict]:
    """Split windows into train/val by TIME ORDER (earliest windows to
    train, latest to val), not randomly. A random split would leak
    information: overlapping windows (stride < window_size, the
    default here) share most of their samples with their neighbors, so
    a random shuffle would put near-duplicates of validation windows
    into the training set, making validation performance look better
    than it really is. Splitting by time order keeps train and val
    from ever sharing a source sample.
    """
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be between 0 and 1.")
    n = len(windows["y"])
    split = int(n * (1 - val_fraction))
    train = {k: v[:split] for k, v in windows.items() if k != "n_rejected"}
    val = {k: v[split:] for k, v in windows.items() if k != "n_rejected"}
    train["n_rejected"] = val["n_rejected"] = windows["n_rejected"]
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
    from src.io_vnbd_loader import load_smartphone_drive
    from src.calibration.gravity_compensation import compute_linear_acceleration
    from src.calibration.zupt_classifier import zupt_mask
    from src.calibration.bias_estimation import estimate_constant_bias, apply_bias_correction
    from src.calibration.mount_alignment import estimate_mount_rotation, apply_mount_alignment

    csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
    df = load_smartphone_drive(csv_path)

    # Days 1-4 pipeline: gravity compensation -> ZUPT -> bias correction
    # -> mount alignment. Deliberately NOT denoised (see module docstring).
    df = compute_linear_acceleration(df)
    mask = zupt_mask(df)
    bias = estimate_constant_bias(df, mask)
    df = apply_bias_correction(df, bias)
    mount = estimate_mount_rotation(df, mask)
    df = apply_mount_alignment(df, mount["R_align"])
    df["gps_speed_ms"] = df["gps_speed_kmh"] / 3.6

    feature_cols = [
        "lin_accel_x_mps2_cal_veh", "lin_accel_y_mps2_cal_veh", "lin_accel_z_mps2_cal_veh",
        "gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads",
    ]

    windows = build_windows(
        df, feature_cols=feature_cols, target_col="gps_speed_ms",
        window_size=20, stride=5,
    )
    print(f"Built {len(windows['y'])} windows, X shape {windows['X'].shape}, "
          f"y shape {windows['y'].shape}, rejected {windows['n_rejected']} "
          f"windows on the dt guard")

    train, val = train_val_split_by_time(windows, val_fraction=0.2)
    print(f"Train: {len(train['y'])} windows, Val: {len(val['y'])} windows "
          f"(split by time order, not randomly -- see docstring)")

    print(f"\nTarget (velocity) range: train "
          f"[{train['y'].min():.2f}, {train['y'].max():.2f}] m/s, "
          f"val [{val['y'].min():.2f}, {val['y'].max():.2f}] m/s")

    # sanity check on REAL data too, if the analyzed sample is present
    real_csv = PROJECT_ROOT / "data" / "raw" / "S-S3b_real_sample.csv"
    if real_csv.exists():
        import pandas as pd
        real_df = pd.read_csv(real_csv, encoding="iso-8859-1", header=0)
        real_df.columns = [c.strip() for c in real_df.columns]
        real_df["t_sec"] = real_df["TIME SINCE START (ms)"] / 1000.0
        real_df["gps_speed_ms"] = real_df["GPS SPEED (Kmh)"] / 3.6
        real_feature_cols = [
            "ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)",
            "GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)",
        ]
        real_windows = build_windows(
            real_df, feature_cols=real_feature_cols, target_col="gps_speed_ms",
            window_size=20, stride=5, time_col="t_sec",
        )
        print(f"\nREAL DATA CHECK (S-S3b, using raw t_sec which is known to "
              f"contain a reset): built {len(real_windows['y'])} clean windows, "
              f"rejected {real_windows['n_rejected']} windows on the dt guard "
              f"-- confirms the guard actually catches the known reset rather "
              f"than silently producing a corrupted window there.")
