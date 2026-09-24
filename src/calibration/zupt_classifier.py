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


def evaluate_zupt_against_gps(mask: np.ndarray, df: pd.DataFrame, speed_thresh_kmh: float = 1.0) -> dict:
    """Sanity-check the IMU-only ZUPT mask against GPS speed (independent
    signal). Not a ground truth -- GPS is coarse too -- but agreement
    here is a useful cross-check, and disagreement flags where GPS lag
    was masking real stationary/moving samples."""
    gps_stationary = (df["gps_speed_kmh"] < speed_thresh_kmh).to_numpy()

    tp = int(np.sum(mask & gps_stationary))
    fp = int(np.sum(mask & ~gps_stationary))
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

    PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
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
