"""
Day 5 — Denoising.

Completes Phase 2's original scope ("Mount alignment, bias estimation,
denoising, ZUPT classifier" -- the last of the four). Directly motivated
by Day 4's finding: leveling and heading were both verified essentially
exact, yet ~210 m of drift remained on that test drive -- traced to
plain sensor NOISE compounding through naive double integration, not a
calibration problem. This file tests that diagnosis directly: if it's
right, low-pass filtering the noise out before integrating should
recover a meaningful fraction of that remaining error.

WHY A LOW-PASS FILTER, WITH THIS CUTOFF (checked, not assumed): an FFT
of `lin_accel_x_mps2_cal_veh` on the Day 4 test drive showed 90% of
signal power below 0.075 Hz and 99% below ~1 Hz -- real driving
dynamics (accelerate/cruise/brake cycles) are a low-frequency signal.
Injected sensor noise, by construction, is flat across the full
0-5 Hz Nyquist range. A cutoff of 1.0 Hz keeps effectively all of the
real signal while removing the noise energy sitting above it. This
number is specific to THIS synthetic generator's drive-cycle timescale
(~30-40 s per accelerate/cruise/brake/stop segment) -- re-run
`characterize_frequency_content()` against real IO-VNBD data or a
different synthetic profile before assuming 1.0 Hz still applies; a
much twitchier real drive (rapid stop-and-go city traffic, potholes)
could have real signal content at higher frequencies than this
test drive's smooth highway-style cycles.

WHY ZERO-PHASE (filtfilt, not a causal filter): any ordinary
(single-pass) low-pass filter introduces a time delay -- the filtered
signal lags the true signal by a phase shift that depends on frequency.
For dead reckoning that delay isn't just cosmetic: integrating a
time-shifted acceleration signal shifts WHEN motion is detected, which
biases the integrated position, not just its noise level. filtfilt
runs the filter forward then backward, canceling the phase shift
exactly (at the cost of not being usable in a true real-time/causal
system -- fine here since Days 6+ operate on whole recorded drives, but
worth flagging if this pipeline is ever adapted to stream live data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

DEFAULT_CUTOFF_HZ = 1.0
DEFAULT_ORDER = 4


def characterize_frequency_content(signal: np.ndarray, sample_rate_hz: float) -> dict:
    """FFT-based check of where a signal's power actually sits in
    frequency, so a low-pass cutoff is chosen from evidence rather than
    assumed. Returns the frequency below which 90/95/99% of power sits,
    plus the top power-contributing frequency bins (excluding DC)."""
    freqs = np.fft.rfftfreq(len(signal), d=1 / sample_rate_hz)
    power = np.abs(np.fft.rfft(signal)) ** 2
    cum = np.cumsum(power) / np.sum(power)

    def _freq_at(pct: float) -> float:
        idx = np.searchsorted(cum, pct)
        return float(freqs[min(idx, len(freqs) - 1)])

    top_idx = np.argsort(power[1:])[::-1][:5] + 1  # exclude DC (index 0)
    top_bins = [
        {"freq_hz": float(freqs[i]), "power": float(power[i])} for i in top_idx
    ]

    return {
        "f90_hz": _freq_at(0.90),
        "f95_hz": _freq_at(0.95),
        "f99_hz": _freq_at(0.99),
        "nyquist_hz": sample_rate_hz / 2,
        "top_power_bins": top_bins,
    }


def denoise_lowpass(
    df: pd.DataFrame,
    cols: list[str],
    sample_rate_hz: float = 10.0,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    order: int = DEFAULT_ORDER,
) -> pd.DataFrame:
    """Zero-phase Butterworth low-pass filter applied independently to
    each column in `cols`. Adds `<col>_dn` columns; leaves originals
    untouched. See module docstring for why zero-phase and why this
    default cutoff.
    """
    out = df.copy()
    nyquist = sample_rate_hz / 2
    normal_cutoff = cutoff_hz / nyquist
    if not (0 < normal_cutoff < 1):
        raise ValueError(
            f"cutoff_hz={cutoff_hz} is not valid for sample_rate_hz={sample_rate_hz} "
            f"(must be strictly between 0 and the Nyquist frequency {nyquist} Hz)."
        )
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    for col in cols:
        out[f"{col}_dn"] = filtfilt(b, a, df[col].to_numpy())
    return out


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
    from src.dead_reckoning.naive_drift_baseline import (
        naive_double_integrate, gps_reference_distance, plot_drift,
    )

    csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
    df = load_smartphone_drive(csv_path)

    # Days 1-4 pipeline: gravity compensation -> ZUPT -> bias correction
    # -> mount alignment (leveling + heading)
    df = compute_linear_acceleration(df)
    mask = zupt_mask(df)
    bias = estimate_constant_bias(df, mask)
    df = apply_bias_correction(df, bias)
    mount = estimate_mount_rotation(df, mask)
    df = apply_mount_alignment(df, mount["R_align"])

    # characterize frequency content BEFORE choosing to filter -- this
    # is what justified DEFAULT_CUTOFF_HZ above, re-run it against any
    # new drive before trusting the same cutoff
    freq_info = characterize_frequency_content(
        df["lin_accel_x_mps2_cal_veh"].to_numpy(), sample_rate_hz=10.0
    )
    print(f"90%/95%/99% of signal power below: "
          f"{freq_info['f90_hz']:.3f} / {freq_info['f95_hz']:.3f} / "
          f"{freq_info['f99_hz']:.3f} Hz (Nyquist {freq_info['nyquist_hz']} Hz)")

    # Day 5: denoise
    df = denoise_lowpass(df, cols=["lin_accel_x_mps2_cal_veh"])

    true_dist = gps_reference_distance(df)
    duration_s = float(df["t_sec"].iloc[-1] - df["t_sec"].iloc[0])

    def _eval(accel_col: str) -> dict:
        integ = naive_double_integrate(df, accel_col=accel_col)
        error = integ["position"] - true_dist
        final_error_m = round(float(error[-1]), 2)
        return {
            **integ, "true_distance_m": true_dist, "error": error,
            "final_error_m": final_error_m,
            "drift_rate_m_per_min": round(final_error_m / duration_s * 60, 2),
            "rmse_position_m": round(float(np.sqrt(np.mean(error ** 2))), 2),
        }

    result_d4 = _eval("lin_accel_x_mps2_cal_veh")       # Day 4, same drive, no denoising
    result_d5 = _eval("lin_accel_x_mps2_cal_veh_dn")     # Day 5, denoised

    print(f"\nDay 4 (mount-aligned, NOT denoised), this drive: "
          f"{result_d4['final_error_m']} m (RMSE {result_d4['rmse_position_m']} m)")
    print(f"Day 5 (mount-aligned + denoised): "
          f"{result_d5['final_error_m']} m (RMSE {result_d5['rmse_position_m']} m)")

    improvement_pct = round(
        100 * (1 - abs(result_d5["final_error_m"]) / abs(result_d4["final_error_m"])), 1
    )
    print(f"Improvement from denoising alone: {improvement_pct}%")

    plot_path = PROJECT_ROOT / "results" / "plots" / "day5_denoised_drift.png"
    plot_drift(result_d5, plot_path,
               title="Day 5 — Denoised Drift (low-pass filtered before integration)")

    metrics_path = PROJECT_ROOT / "results" / "metrics.md"
    with open(metrics_path, "a") as f:
        f.write(
            f"| Day 5 | Mount-aligned + denoised (1.0 Hz low-pass) double integration | "
            f"{result_d5['final_error_m']} | {result_d5['drift_rate_m_per_min']} | "
            f"{result_d5['rmse_position_m']} |\n"
        )
    print(f"Logged Day 5 row to {metrics_path}")
