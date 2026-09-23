"""
Day 2 — Exploratory Data Analysis pass over a loaded IO-VNBD drive.

Matches the real column schema from src/io_vnbd_loader.py
(load_smartphone_drive / drive_summary): t_sec, accel_x_mps2/y/z,
gyro_yaw_rads/pitch/roll, gps_speed_kmh, etc.

Produces:
  - results/plots/day2_eda_imu_distributions.png : histograms of raw IMU
    channels (accel_x/y/z, gyro_yaw/pitch/roll) to sanity-check noise
    scale and spot obvious bias/outliers.
  - results/plots/day2_eda_stationary_noise.png : accel/gyro noise
    characterized specifically during detected stationary (near-zero
    GPS speed) windows -- this is the number ZUPT (Phase 2) and the
    Kalman filter noise models (Phase 6) will need.
  - Printed summary stats (mean/std per channel, stationary-window
    accel/gyro std -- i.e. the noise floor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

IMU_ACCEL_COLS = ["accel_x_mps2", "accel_y_mps2", "accel_z_mps2"]
IMU_GYRO_COLS = ["gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads"]


def channel_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = IMU_ACCEL_COLS + IMU_GYRO_COLS
    stats = df[cols].agg(["mean", "std", "min", "max"]).T
    return stats


def stationary_mask(df: pd.DataFrame, speed_thresh_kmh: float = 1.0) -> np.ndarray:
    """Identify stationary samples from GPS speed near zero. This is
    exactly the window ZUPT (Phase 2) needs to detect.

    Note: unlike sparse-GPS drives, this dataset's gps_speed_kmh is
    already present at the full 10 Hz IMU rate (no separate fix flag),
    so no forward-fill is needed here.
    """
    return (df["gps_speed_kmh"] < speed_thresh_kmh).to_numpy()


def stationary_noise_floor(df: pd.DataFrame, mask: np.ndarray) -> dict:
    """Std-dev of each IMU channel restricted to stationary samples --
    this is the sensor noise floor, used later to set Kalman R/Q."""
    stat_df = df.loc[mask, IMU_ACCEL_COLS + IMU_GYRO_COLS]
    return stat_df.std().to_dict()


def plot_distributions(df: pd.DataFrame, out_path: str | Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes[0], IMU_ACCEL_COLS):
        ax.hist(df[col], bins=60, color="#3b6ea5")
        ax.set_title(col)
        ax.set_xlabel("m/s^2")
    for ax, col in zip(axes[1], IMU_GYRO_COLS):
        ax.hist(df[col], bins=60, color="#a53b3b")
        ax.set_title(col)
        ax.set_xlabel("rad/s")
    fig.suptitle("Day 2 EDA — Raw IMU Channel Distributions")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_stationary_noise(df: pd.DataFrame, mask: np.ndarray, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t = df["t_sec"].to_numpy()

    axes[0].plot(t, df["accel_x_mps2"], lw=0.6, color="#3b6ea5", label="accel_x_mps2")
    axes[0].fill_between(t, df["accel_x_mps2"].min(), df["accel_x_mps2"].max(),
                          where=mask, color="orange", alpha=0.2, label="stationary")
    axes[0].set_title("accel_x_mps2 with detected stationary windows")
    axes[0].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, df["gps_speed_kmh"], lw=1.0, color="#333")
    axes[1].axhline(1.0, color="red", ls="--", lw=0.8, label="stationary threshold")
    axes[1].set_title("GPS speed")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("km/h")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_eda(df: pd.DataFrame, plots_dir: str | Path = "results/plots") -> dict:
    stats = channel_stats(df)
    mask = stationary_mask(df)
    noise_floor = stationary_noise_floor(df, mask)

    plot_distributions(df, Path(plots_dir) / "day2_eda_imu_distributions.png")
    plot_stationary_noise(df, mask, Path(plots_dir) / "day2_eda_stationary_noise.png")

    return {
        "channel_stats": stats,
        "n_stationary_samples": int(mask.sum()),
        "pct_stationary": round(100 * mask.mean(), 1),
        "stationary_noise_floor_std": noise_floor,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    # Resolve project root relative to this file (src/eda.py -> parent
    # is src/, parent.parent is project root), not the caller's cwd --
    # matches the pattern used in src/io_vnbd_loader.py.
    PROJECT_ROOT = _Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.io_vnbd_loader import load_smartphone_drive

    csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
    df = load_smartphone_drive(csv_path)
    result = run_eda(df, plots_dir=PROJECT_ROOT / "results" / "plots")

    print(result["channel_stats"])
    print(f"\nStationary samples: {result['n_stationary_samples']} "
          f"({result['pct_stationary']}%)")
    print("Stationary noise floor (std):")
    for k, v in result["stationary_noise_floor_std"].items():
        print(f"  {k}: {v:.4f}")