"""
Day 3 — Gravity compensation.

The IO-VNBD smartphone schema provides raw accelerometer
(accel_x/y/z_mps2, gravity-inclusive, as Android's TYPE_ACCELEROMETER
does) AND the phone's own fused gravity estimate (gravity_x/y/z_mps2,
Android's TYPE_GRAVITY). Subtracting one from the other gives linear
acceleration WITHOUT needing to know the phone's mounting orientation:

    linear_accel = accel - gravity   (per axis)

This is the correct first calibration step, ahead of both bias
estimation and mount alignment (Day 4) — bias estimation on raw accel_x
would otherwise be contaminated by whatever constant tilt-induced
gravity component happens to project onto that axis.
"""

from __future__ import annotations

import pandas as pd

AXES = ["x", "y", "z"]


def compute_linear_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with added lin_accel_{x,y,z}_mps2 columns."""
    out = df.copy()
    for ax in AXES:
        out[f"lin_accel_{ax}_mps2"] = df[f"accel_{ax}_mps2"] - df[f"gravity_{ax}_mps2"]
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

    csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
    df = load_smartphone_drive(csv_path)
    df = compute_linear_acceleration(df)
    print(df[["accel_x_mps2", "gravity_x_mps2", "lin_accel_x_mps2"]].describe())