"""
Day 3 — IMU-native ZUPT (zero-velocity update) classifier.

Day 2's stationary detector used GPS speed thresholding, which lags the
true stop instant (GPS is sparse / smoothed) and inflated the measured
stationary noise floor (see PROGRESS.md Day 2 "Issue found"). This
version detects stationary windows directly from the IMU: a sample is
flagged stationary when BOTH the rolling variance of linear-acceleration
magnitude AND the rolling gyro magnitude are below threshold over a
short window — the standard ZUPT trigger condition used in strapdown INS
(Phase 4 will consume this same mask).

Must be run AFTER gravity_compensation.compute_linear_acceleration(), so
that vehicle vibration/motion is being measured, not a constant tilt
offset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LIN_ACCEL_COLS = ["lin_accel_x_mps2", "lin_accel_y_mps2", "lin_accel_z_mps2"]
GYRO_COLS = ["gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads"]


def _rolling_std_magnitude(df: pd.DataFrame, cols: list[str], window: int) -> np.ndarray:
    """Rolling std of the vector magnitude across `cols`, centered window."""
    mag = np.sqrt(sum(df[c] ** 2 for c in cols))
    return mag.rolling(window, center=True, min_periods=1).std().fillna(0).to_numpy()


def zupt_mask(
    df: pd.DataFrame,
    accel_var_thresh: float = 0.05,
    gyro_mag_thresh: float = 0.02,
    window: int = 5,
) -> np.ndarray:
    """Flag samples as stationary using only IMU signals (no GPS).

    df must already have lin_accel_{x,y,z}_mps2 (see gravity_compensation.py).

    Default thresholds are set from Day 2's measured stationary noise
    floor (~0.03 accel std, ~0.01 gyro std) with margin, NOT tuned
    against a specific drive -- retune against real IO-VNBD data once
    available. On the current straight-line-only synthetic generator,
    abrupt (linspace) accel/brake transitions near a stop inflate
    rolling variance right at the true stop instant, so precision
    against the GPS-based reference will look worse on synthetic data
    than it should on a real, smoother drive; check
    evaluate_zupt_against_gps() output before trusting these defaults
    on new data.
    """
    accel_var = _rolling_std_magnitude(df, LIN_ACCEL_COLS, window)
    gyro_mag = np.sqrt(sum(df[c] ** 2 for c in GYRO_COLS)).to_numpy()
    gyro_var = pd.Series(gyro_mag).rolling(window, center=True, min_periods=1).std().fillna(0).to_numpy()

    return (accel_var < accel_var_thresh) & (gyro_var < gyro_mag_thresh)

def zupt_mask_with_fallback(
    df,
    accel_var_thresh=0.05,
    gyro_mag_thresh=0.02,
    window=5,
    fallback_speed_thresh_kmh=3.0,
    fallback_percentile=10.0,
):
    """zupt_mask(), plus a GPS-speed-gated fallback for real drives whose
    noise floor exceeds the fixed threshold. The gate matters: on drives
    that never stop, the globally lowest-variance samples are just the
    smoothest MOVING stretch, and treating them as 'at rest' would feed
    real forward motion into the bias estimate. So the percentile search
    only runs inside an already-slow window. Returns (mask, diagnostics)."""
    mask = zupt_mask(df, accel_var_thresh, gyro_mag_thresh, window)
    if mask.sum() > 0:
        return mask, {"used_fallback": False, "fallback_possible": None}

    low_speed = (df["gps_speed_kmh"] < fallback_speed_thresh_kmh).to_numpy()
    if low_speed.sum() == 0:
        return mask, {"used_fallback": True, "fallback_possible": False}

    accel_var = _rolling_std_magnitude(df, LIN_ACCEL_COLS, window)
    gyro_mag = np.sqrt(sum(df[c] ** 2 for c in GYRO_COLS)).to_numpy()
    gyro_var = pd.Series(gyro_mag).rolling(window, center=True, min_periods=1).std().fillna(0).to_numpy()
    combined = accel_var + gyro_var

    idx = np.where(low_speed)[0]
    pool = combined[idx]
    cutoff = np.percentile(pool, fallback_percentile)
    fb = np.zeros(len(df), dtype=bool)
    fb[idx[pool <= cutoff]] = True
    return fb, {
        "used_fallback": True,
        "fallback_possible": True,
        "n_low_speed_samples": int(low_speed.sum()),
        "n_fallback_flagged": int(fb.sum()),
    }


def calibration_stationary_mask(
    df: pd.DataFrame,
    speed_thresh_ms: float = 0.3,
    min_duration_s: float = 3.0,
    window: int = 20,
) -> np.ndarray:
    """GPS-confirmed stops to calibrate on (bias, leveling, ZUPT noise floor).

    Calibration happens while GPS is still available, so GPS may be used
    here -- unlike the ZUPT that runs during an outage. Real IO-VNBD GPS
    only updates every ~10 s and HOLDS its last value, so a held 0 can
    persist for several seconds after the car pulls away. Two guards:
    the held speed must stay below `speed_thresh_ms` for `min_duration_s`,
    and only the quieter half (by rolling accel std) of those samples is
    kept, which drops the pulling-away tail. Empty if the drive never stops.
    """
    held_stop = (df["gps_speed_ms"] < speed_thresh_ms).to_numpy()
    dt = float(np.median(np.diff(df["t_sec"].to_numpy())))
    n = max(1, int(round(min_duration_s / dt)))
    sustained = (
        pd.Series(held_stop.astype(float)).rolling(n, center=True, min_periods=1).min().to_numpy() > 0
    )
    if not sustained.any():
        return sustained
    accel_std = _rolling_std_magnitude(df, LIN_ACCEL_COLS, window)
    return sustained & (accel_std <= np.median(accel_std[sustained]))


def zupt_mask_adaptive(
    df: pd.DataFrame,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray | None = None,
    window: int = 30,
    k_accel: float = 1.5,
    k_gyro: float = 2.0,
    moving_percentile: float = 5.0,
    min_reference_samples: int = 100,
) -> tuple[np.ndarray, dict]:
    """IMU-only ZUPT whose thresholds are scaled to THIS phone's own at-rest
    noise floor, measured on `reference_mask` (e.g. calibration_stationary_mask,
    taken while GPS was available).

    The fixed thresholds in zupt_mask() were tuned on synthetic data and do
    not transfer: across real drives the at-rest rolling accel std ranges
    from ~0.01 (S-T* phones) to ~0.2 m/s^2, and the moving value from ~0.2
    to ~1.4 -- no single threshold separates both.

    Safety caps, because a wrong ZUPT (zeroing velocity while driving) is far
    worse than a missed one: each threshold is capped below the
    `moving_percentile` of the same statistic on `moving_mask` (default:
    GPS-interpolated speed > 5 m/s -- also known at calibration time), and
    fewer than `min_reference_samples` reference samples (10 s at 10 Hz)
    disables ZUPT for the drive. Without the cap, S-T8/S-T5/S-Vw8 -- whose
    few "stop" samples were really slow rolling -- flagged 88-97% of
    samples driven at > 5 m/s. Returns (mask, thresholds).
    """
    accel_var = _rolling_std_magnitude(df, LIN_ACCEL_COLS, window)
    gyro_mag = np.sqrt(sum(df[c] ** 2 for c in GYRO_COLS))
    gyro_var = gyro_mag.rolling(window, center=True, min_periods=1).std().fillna(0).to_numpy()
    if reference_mask.sum() < min_reference_samples:
        return np.zeros(len(df), dtype=bool), {"accel_thresh": None, "gyro_thresh": None}
    accel_thresh = k_accel * float(np.median(accel_var[reference_mask]))
    gyro_thresh = k_gyro * float(np.median(gyro_var[reference_mask]))
    if moving_mask is None:
        moving_mask = (df["gps_speed_interp_ms"] > 5.0).to_numpy()
    if moving_mask.sum() > 0:
        accel_thresh = min(accel_thresh, float(np.percentile(accel_var[moving_mask], moving_percentile)))
        gyro_thresh = min(gyro_thresh, float(np.percentile(gyro_var[moving_mask], moving_percentile)))
    mask = (accel_var < accel_thresh) & (gyro_var < gyro_thresh)
    return mask, {"accel_thresh": accel_thresh, "gyro_thresh": gyro_thresh}


def evaluate_zupt_against_gps(mask: np.ndarray, df: pd.DataFrame, speed_thresh_kmh: float = 1.0) -> dict:
    """Sanity-check the IMU-only ZUPT mask against GPS speed (independent
    signal). Not a ground truth -- GPS is coarse too -- but agreement
    here is a useful cross-check, and disagreement flags where GPS lag
    was masking real stationary/moving samples.

    A flag only counts as a false positive when BOTH the held GPS speed and
    the interpolated speed (if present) say the car is moving -- with ~10 s
    real GPS fixes, either one alone mislabels the seconds around a stop."""
    gps_stationary = (df["gps_speed_kmh"] < speed_thresh_kmh).to_numpy()
    gps_moving = ~gps_stationary
    if "gps_speed_interp_ms" in df.columns:
        gps_moving &= (df["gps_speed_interp_ms"] * 3.6 >= speed_thresh_kmh).to_numpy()

    tp = int(np.sum(mask & gps_stationary))
    fp = int(np.sum(mask & gps_moving))
    fn = int(np.sum(~mask & gps_stationary))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    return {
        "n_zupt_flagged": int(mask.sum()),
        "n_gps_stationary": int(gps_stationary.sum()),
        "agreement_precision": round(precision, 3),
        "agreement_recall": round(recall, 3),
    }


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

    csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
    df = load_smartphone_drive(csv_path)
    df = compute_linear_acceleration(df)

    mask = zupt_mask(df)
    print(f"IMU-native ZUPT flagged {mask.sum()} / {len(df)} samples "
          f"({100*mask.mean():.1f}%)")
    print(evaluate_zupt_against_gps(mask, df))