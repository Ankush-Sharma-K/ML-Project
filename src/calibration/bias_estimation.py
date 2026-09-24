"""
Day 3 — Constant bias estimation + Day 3 calibrated drift re-run.

Estimates per-channel constant sensor bias as the mean value of each
linear-acceleration / gyro channel during ZUPT-flagged stationary
windows (at true rest, linear acceleration and angular rate should
both read exactly zero -- any nonzero mean during those windows IS the
constant bias). Applies the correction, then re-runs the Day 2 naive
double-integration baseline on the corrected signal and logs the result
as the second row of results/metrics.md, so the improvement from Day 2
is directly visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

BIAS_COLS = [
    "lin_accel_x_mps2", "lin_accel_y_mps2", "lin_accel_z_mps2",
    "gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads",
]


def estimate_constant_bias(df: pd.DataFrame, mask: np.ndarray, cols: list[str] = BIAS_COLS) -> dict:
    """Mean of each channel restricted to stationary (ZUPT) samples."""
    if mask.sum() == 0:
        raise ValueError("No stationary samples in mask -- cannot estimate bias.")
    return {c: float(df.loc[mask, c].mean()) for c in cols}


def apply_bias_correction(df: pd.DataFrame, bias: dict) -> pd.DataFrame:
    """Return a copy of df with <col>_cal columns = col - bias[col]."""
    out = df.copy()
    for col, b in bias.items():
        out[f"{col}_cal"] = df[col] - b
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.io_vnbd_loader import load_smartphone_drive
    from src.calibration.gravity_compensation import compute_linear_acceleration
    from src.calibration.zupt_classifier import zupt_mask, evaluate_zupt_against_gps
    from src.dead_reckoning.naive_drift_baseline import (
        naive_double_integrate, evaluate_drift, plot_drift, log_metric,
    )

    csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
    df = load_smartphone_drive(csv_path)

    # 1) gravity compensation
    df = compute_linear_acceleration(df)

    # 2) IMU-native ZUPT detection
    mask = zupt_mask(df)
    print(f"ZUPT: {mask.sum()} / {len(df)} samples flagged stationary "
          f"({100*mask.mean():.1f}%)")
    print("Cross-check vs GPS speed:", evaluate_zupt_against_gps(mask, df))

    # 3) constant bias estimation from those windows
    bias = estimate_constant_bias(df, mask)
    print("\nEstimated constant bias:")
    for k, v in bias.items():
        print(f"  {k}: {v:+.4f}")

    # 4) apply correction
    df = apply_bias_correction(df, bias)

    # 5) re-run the Day 2 naive double-integration baseline, now on the
    #    gravity-compensated, bias-corrected forward-axis signal
    from src.dead_reckoning.naive_drift_baseline import gps_reference_distance

    calib_integration = naive_double_integrate(df, accel_col="lin_accel_x_mps2_cal")
    true_dist = gps_reference_distance(df)
    error = calib_integration["position"] - true_dist
    duration_s = float(df["t_sec"].iloc[-1] - df["t_sec"].iloc[0])
    calibrated_result = {
        **calib_integration,
        "true_distance_m": true_dist,
        "error": error,
        "final_error_m": round(float(error[-1]), 2),
        "final_true_distance_m": round(float(true_dist[-1]), 2),
        "drift_rate_m_per_min": round(float(error[-1]) / duration_s * 60, 2),
        "rmse_position_m": round(float(np.sqrt(np.mean(error ** 2))), 2),
    }

    print(f"\nDay 3 calibrated final error: {calibrated_result['final_error_m']} m "
          f"(drift {calibrated_result['drift_rate_m_per_min']} m/min, "
          f"RMSE {calibrated_result['rmse_position_m']} m)")

    plot_path = PROJECT_ROOT / "results" / "plots" / "day3_calibrated_drift.png"
    plot_drift(
        calibrated_result, plot_path,
        title="Day 3 — Gravity-Compensated + Bias-Corrected Drift (vs. Day 2 raw)",
    )

    metrics_path = PROJECT_ROOT / "results" / "metrics.md"
    with open(metrics_path, "a") as f:
        f.write(
            f"| Day 3 | Gravity-compensated + bias-corrected double integration | "
            f"{calibrated_result['final_error_m']} | {calibrated_result['drift_rate_m_per_min']} | "
            f"{calibrated_result['rmse_position_m']} |\n"
        )
    print(f"Logged Day 3 row to {metrics_path}")
