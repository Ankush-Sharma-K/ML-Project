"""
Day 4 (part 1 of 2) — Mount alignment: LEVELING only.

The phone can be mounted in the vehicle at any orientation (dashboard,
cup holder, pocket, angled vent mount, ...). Days 2-3 cleaned up the
SENSOR (gravity leakage, constant bias) but left the SIGNAL expressed in
the phone's own coordinate frame, not the vehicle's forward/lateral/
vertical frame.

Full mount alignment is two steps (gravity alone can only constrain 2 of
3 rotational degrees of freedom):

  1. LEVELING (roll + pitch) -- THIS FILE.
  2. HEADING (yaw) -- deferred to Day 4 part 2, next session. Requires
     correlating horizontal linear acceleration with GPS-derived forward
     acceleration during accelerating phases, which is more to get right
     and to test than leveling alone, so it's being split out rather
     than shipped half-verified.

LEVELING: the gravity vector measured at rest (`gravity_{x,y,z}_mps2`,
already isolated by the phone's own sensor fusion) must point straight
up in the vehicle frame when the phone is level. Aligning measured
gravity to vehicle-up gives a rotation that's correct up to an unknown
rotation about the vertical axis (that remaining yaw ambiguity is
exactly what part 2 resolves).

SIGN CONVENTION (confirmed before writing this, not assumed): Android's
TYPE_GRAVITY sensor reports gravity using the same sign convention as
the accelerometer at rest -- i.e. approximately (0, 0, +9.81) when the
phone lies flat screen-up (the reaction force opposing gravity, pointing
away from the road). This file targets VEHICLE_UP = (0, 0, +1) as the
leveling target for that reason. If a real IO-VNBD drive's
`gravity_z_mps2` comes out consistently negative at rest instead, that
means either the phone was mounted screen-down/inverted for that drive,
or this dataset's export uses the opposite sign convention -- check
`gravity_z_mps2` sign during a known-stationary window before trusting
this on real data (see "Open issues" note in PROGRESS.md).

Must run AFTER gravity_compensation + zupt_classifier + bias_estimation
(Day 3) for the stationary mask; does NOT need bias-corrected linear
acceleration itself (leveling only uses the raw `gravity_*` columns,
which are unaffected by accelerometer bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VEHICLE_UP = np.array([0.0, 0.0, 1.0])  # unit vector, vehicle frame (z = up)
GRAVITY_COLS = ["gravity_x_mps2", "gravity_y_mps2", "gravity_z_mps2"]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def rotation_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ (a / |a|) ~= (b / |b|)
    (Rodrigues' rotation formula, shortest-arc rotation)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))

    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        return _rotation_180_about_any_perpendicular(a)

    vx = _skew(v)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def _rotation_180_about_any_perpendicular(a: np.ndarray) -> np.ndarray:
    perp = np.cross(a, [1.0, 0.0, 0.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(a, [0.0, 1.0, 0.0])
    perp = perp / np.linalg.norm(perp)
    vx = _skew(perp)
    return np.eye(3) + 2 * vx @ vx


def estimate_leveling_rotation(df: pd.DataFrame, stationary_mask: np.ndarray) -> np.ndarray:
    """Rotation that levels the phone: aligns the mean measured gravity
    vector (during ZUPT-flagged stationary windows) to VEHICLE_UP.
    Correct up to an unknown yaw about the vertical axis -- see Day 4
    part 2 for resolving that."""
    if stationary_mask.sum() == 0:
        raise ValueError("No stationary samples in mask -- cannot estimate leveling.")
    g_phone = df.loc[stationary_mask, GRAVITY_COLS].mean().to_numpy()
    return rotation_from_vectors(g_phone, VEHICLE_UP)


def recovered_roll_pitch_deg(R_level: np.ndarray) -> dict:
    """Convenience: report the roll/pitch this rotation is correcting
    for, purely for sanity-checking/logging (not used downstream)."""
    # R_level maps phone-frame gravity to vehicle-up; R_level.T maps
    # vehicle-up back to phone frame, i.e. it IS (an estimate of) the
    # phone's tilt relative to the vehicle.
    R_tilt = R_level.T
    pitch = np.degrees(np.arcsin(-R_tilt[2, 0]))
    roll = np.degrees(np.arctan2(R_tilt[2, 1], R_tilt[2, 2]))
    return {"roll_deg": float(roll), "pitch_deg": float(pitch)}


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

    csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
    df = load_smartphone_drive(csv_path)
    df = compute_linear_acceleration(df)
    mask = zupt_mask(df)

    print(f"Mean gravity vector during {mask.sum()} stationary samples: "
          f"{df.loc[mask, GRAVITY_COLS].mean().to_numpy()}")

    R_level = estimate_leveling_rotation(df, mask)
    print("\nEstimated leveling rotation R_level (phone -> vehicle, roll+pitch only):")
    print(R_level)
    print("\nRecovered tilt:", recovered_roll_pitch_deg(R_level))

    # sanity check: applying R_level to the mean stationary gravity
    # vector should land close to VEHICLE_UP * |g|
    g_phone_mean = df.loc[mask, GRAVITY_COLS].mean().to_numpy()
    g_leveled = R_level @ g_phone_mean
    print(f"\nLeveled gravity vector (should be ~[0, 0, {np.linalg.norm(g_phone_mean):.2f}]): "
          f"{g_leveled}")