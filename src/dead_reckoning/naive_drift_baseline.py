"""
Day 2 — Naive double-integration drift baseline.

Matches the real column schema from src/io_vnbd_loader.py
(load_smartphone_drive / drive_summary): t_sec, accel_x_mps2, gps_speed_kmh.

The dumbest possible dead-reckoning: take raw (uncalibrated, unbiased)
accelerometer forward-axis readings and integrate twice --
accel -> velocity -> position -- with ZERO correction: no bias removal,
no ZUPT, no map matching, no GNSS fusion.

This number is the baseline every later phase (calibration, ZUPT,
learned velocity model, strapdown INS, map matching, GNSS fusion) has
to beat. It gets logged as the first entry in results/metrics.md and
re-measured after every phase to show the error curve coming down.

Ground truth: integrated directly from gps_speed_kmh (present at the
full IMU rate in this dataset, no forward-fill needed). This works for
both the synthetic drive and real IO-VNBD "S-*.csv" files without
depending on any extra attrs the drive generator may or may not attach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def naive_double_integrate(df: pd.DataFrame, accel_col: str = "accel_x_mps2") -> dict:
    """Integrate raw accelerometer directly to velocity and position,
    with zero bias correction or ZUPT. Returns time series + final error.
    """
    t = df["t_sec"].to_numpy()
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.0

    accel = df[accel_col].to_numpy()

    # naive integration: accel -> velocity -> position, no correction at all
    velocity = np.cumsum(accel * dt)
    position = np.cumsum(velocity * dt)

    return {"t": t, "velocity": velocity, "position": position}


def gps_reference_distance(df: pd.DataFrame) -> np.ndarray:
    """Ground-truth distance built from GPS speed."""
    t = df["t_sec"].to_numpy()
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.0
    speed_ms = (df["gps_speed_kmh"] / 3.6).to_numpy()
    return np.cumsum(speed_ms * dt)


def evaluate_drift(df: pd.DataFrame, true_distance_m: np.ndarray | None = None) -> dict:
    """Run the naive baseline and compare against ground truth.
    Pass true_distance_m explicitly if you have a better reference
    (e.g. a synthetic generator's stashed true speed profile); omitted,
    falls back to the GPS-integrated track.
    """
    result = naive_double_integrate(df)

    if true_distance_m is None:
        true_distance_m = gps_reference_distance(df)

    error = result["position"] - true_distance_m
    final_error_m = float(error[-1])
    duration_s = float(df["t_sec"].iloc[-1] - df["t_sec"].iloc[0])
    drift_rate_m_per_s = final_error_m / duration_s if duration_s > 0 else float("nan")
    drift_rate_m_per_min = drift_rate_m_per_s * 60

    return {
        **result,
        "true_distance_m": true_distance_m,
        "error": error,
        "final_error_m": round(final_error_m, 2),
        "final_true_distance_m": round(float(true_distance_m[-1]), 2),
        "drift_rate_m_per_min": round(drift_rate_m_per_min, 2),
        "rmse_position_m": round(float(np.sqrt(np.mean(error ** 2))), 2),
    }


def plot_drift(eval_result: dict, out_path: str | Path, title: str = "Day 2 — Naive Double-Integration Drift Baseline (zero correction)") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(eval_result["t"], eval_result["position"], label="naive DR (integrated)", color="#c0392b")
    axes[0].plot(eval_result["t"], eval_result["true_distance_m"], label="GPS-integrated reference", color="#27ae60", ls="--")
    axes[0].set_title("Position: naive double-integration vs. GPS reference")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("distance traveled (m)")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(eval_result["t"], eval_result["error"], color="#8e44ad")
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_title(
        f"Position error (drift) — final: {eval_result['final_error_m']} m "
        f"({eval_result['drift_rate_m_per_min']} m/min)"
    )
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("error (m)")

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def log_metric(metrics_path: str | Path, eval_result: dict) -> None:
    """Append this baseline's key numbers as the first entry in
    results/metrics.md (created if it doesn't exist)."""
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    header_needed = not metrics_path.exists()
    with open(metrics_path, "a") as f:
        if header_needed:
            f.write("# results/metrics.md — drift benchmark, tracked per phase\n\n")
            f.write("| Phase / Day | Method | Final position error (m) | Drift rate (m/min) | RMSE (m) |\n")
            f.write("|---|---|---|---|---|\n")
        f.write(
            f"| Day 2 | Naive double-integration (zero correction) | "
            f"{eval_result['final_error_m']} | {eval_result['drift_rate_m_per_min']} | "
            f"{eval_result['rmse_position_m']} |\n"
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    # Resolve project root relative to this file (src/dead_reckoning/file.py
    # -> parents[1] is src/, parents[2] is project root), not the caller's
    # cwd -- matches the pattern used in src/io_vnbd_loader.py.
    PROJECT_ROOT = _Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.io_vnbd_loader import load_smartphone_drive

    csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
    df = load_smartphone_drive(csv_path)

    result = evaluate_drift(df)

    print(f"Final position error: {result['final_error_m']} m "
          f"(GPS-reference distance traveled: {result['final_true_distance_m']} m)")
    print(f"Drift rate: {result['drift_rate_m_per_min']} m/min")
    print(f"RMSE: {result['rmse_position_m']} m")

    plot_path = PROJECT_ROOT / "results" / "plots" / "day2_naive_drift_baseline.png"
    metrics_path = PROJECT_ROOT / "results" / "metrics.md"
    plot_drift(result, plot_path)
    log_metric(metrics_path, result)
    print(f"\nSaved plot to {plot_path}")
    print(f"Logged to {metrics_path}")