"""
Day 4 — Mount alignment: LEVELING (part 1) + HEADING (part 2).

The phone can be mounted in the vehicle at any orientation (dashboard,
cup holder, pocket, angled vent mount, ...). Days 2-3 cleaned up the
SENSOR (gravity leakage, constant bias) but left the SIGNAL expressed in
the phone's own coordinate frame, not the vehicle's forward/lateral/
vertical frame.

Full mount alignment is two steps (gravity alone can only constrain 2 of
3 rotational degrees of freedom):

  1. LEVELING (roll + pitch). The gravity vector measured at rest
     (`gravity_{x,y,z}_mps2`, already isolated by the phone's own sensor
     fusion) must point straight up in the vehicle frame when the phone
     is level. Aligning measured gravity to vehicle-up gives a rotation
     that's correct up to an unknown rotation about the vertical axis.

  2. HEADING (yaw) -- resolves that leftover ambiguity. During phases
     where the vehicle is measurably accelerating forward (GPS speed
     increasing), the vehicle's forward axis is -- by the
     non-holonomic constraint this whole project relies on -- the
     direction of horizontal linear acceleration. Rotating that
     direction onto the vehicle's x-axis fixes yaw.

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
(Day 3): leveling only needs the raw `gravity_*` columns (unaffected by
accelerometer bias), but heading estimation needs BIAS-CORRECTED linear
acceleration (`lin_accel_*_cal`) -- a leftover constant bias would bias
the estimated heading angle too.

Limitation carried from Day 1/3: the current synthetic generator is
straight-line only, so heading estimation can only be validated on
accelerate/brake phases, not on real cornering -- see PROGRESS.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VEHICLE_UP = np.array([0.0, 0.0, 1.0])  # unit vector, vehicle frame (z = up)
GRAVITY_COLS = ["gravity_x_mps2", "gravity_y_mps2", "gravity_z_mps2"]
DEFAULT_LIN_ACCEL_CAL_COLS = (
    "lin_accel_x_mps2_cal", "lin_accel_y_mps2_cal", "lin_accel_z_mps2_cal",
)


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
    for, purely for sanity-checking/logging (NOT used downstream --
    apply_mount_alignment() and estimate_mount_rotation() both use the
    rotation matrices directly, never these decomposed angles).

    CAVEAT (found during Day 4 part 2 testing, confirmed not a bug in
    R_level itself): this Euler-angle readout is only reliable when the
    true mount has ~zero yaw. R_level is the minimal (Rodrigues
    shortest-arc) rotation that aligns measured gravity to vehicle-up --
    correct by construction (verified: R_level @ g_measured lands within
    1e-16 of [0,0,g] in testing) -- but when the phone ALSO has real yaw
    misalignment, that minimal rotation is generally NOT the same as the
    naive "undo roll, then undo pitch" rotation this function assumes,
    so the numbers it reports can look meaningfully wrong (seen: ~9.3
    deg reported vs 8.0 true roll, ~-1.6 vs -5.0 true pitch) even though
    R_level is doing its job correctly. Don't use this function's output
    to validate leveling correctness when yaw is also present -- check
    R_level @ g_measured against vehicle-up directly instead.
    """
    # R_level maps phone-frame gravity to vehicle-up; R_level.T maps
    # vehicle-up back to phone frame, i.e. it IS (an estimate of) the
    # phone's tilt relative to the vehicle.
    R_tilt = R_level.T
    pitch = np.degrees(np.arcsin(-R_tilt[2, 0]))
    roll = np.degrees(np.arctan2(R_tilt[2, 1], R_tilt[2, 2]))
    return {"roll_deg": float(roll), "pitch_deg": float(pitch)}


def rotation_about_z(theta_rad: float) -> np.ndarray:
    """Rotation matrix for a rotation of theta_rad about the z-axis."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def accelerating_mask(df: pd.DataFrame, accel_thresh_ms2: float = 0.3) -> np.ndarray:
    """Samples where GPS speed is measurably increasing -- i.e. the
    vehicle is actively accelerating forward, so (by the non-holonomic
    constraint) horizontal linear acceleration should point along the
    vehicle's forward axis. This is the independent signal that
    resolves the yaw ambiguity leveling alone can't."""
    speed_ms = (df["gps_speed_kmh"] / 3.6).to_numpy()
    t = df["t_sec"].to_numpy()
    dv_dt = np.gradient(speed_ms, t)
    return dv_dt > accel_thresh_ms2


def estimate_heading_offset(
    df: pd.DataFrame,
    R_level: np.ndarray,
    accel_mask: np.ndarray,
    lin_accel_cal_cols: tuple = DEFAULT_LIN_ACCEL_CAL_COLS,
) -> float:
    """Angle (radians) of the mean leveled horizontal linear-acceleration
    vector during accelerating windows -- the yaw ambiguity left over
    from leveling. Rotating by -this angle puts the vehicle's forward
    axis onto the leveled frame's x-axis.

    Requires BIAS-CORRECTED linear acceleration (`_cal` columns) -- an
    uncorrected constant bias would shift the estimated angle too.
    """
    if accel_mask.sum() == 0:
        raise ValueError("No accelerating samples in mask -- cannot estimate heading.")
    accel_p = df.loc[accel_mask, list(lin_accel_cal_cols)].to_numpy()
    accel_leveled = accel_p @ R_level.T
    mean_horizontal = accel_leveled[:, :2].mean(axis=0)
    if np.linalg.norm(mean_horizontal) < 1e-6:
        raise ValueError(
            "Mean horizontal acceleration during accelerating windows is ~0 -- "
            "cannot resolve heading (check accelerating_mask threshold)."
        )
    return float(np.arctan2(mean_horizontal[1], mean_horizontal[0]))


def estimate_mount_rotation(
    df: pd.DataFrame,
    stationary_mask: np.ndarray,
    accel_mask: np.ndarray | None = None,
    lin_accel_cal_cols: tuple = DEFAULT_LIN_ACCEL_CAL_COLS,
) -> dict:
    """Full two-step mount alignment: leveling (part 1) + heading
    (part 2), combined into a single phone-frame -> vehicle-frame
    rotation. Returns intermediate estimates too, for inspection."""
    R_level = estimate_leveling_rotation(df, stationary_mask)

    if accel_mask is None:
        accel_mask = accelerating_mask(df)
    heading_offset = estimate_heading_offset(df, R_level, accel_mask, lin_accel_cal_cols)

    R_yaw = rotation_about_z(-heading_offset)
    R_align = R_yaw @ R_level

    return {
        "R_level": R_level,
        "heading_offset_rad": heading_offset,
        "heading_offset_deg": float(np.degrees(heading_offset)),
        "R_align": R_align,
        "n_accelerating_samples": int(accel_mask.sum()),
    }


def apply_mount_alignment(
    df: pd.DataFrame,
    R_align: np.ndarray,
    lin_accel_cal_cols: tuple = DEFAULT_LIN_ACCEL_CAL_COLS,
) -> pd.DataFrame:
    """Return a copy of df with vehicle-frame linear acceleration columns:
    lin_accel_{x,y,z}_mps2_cal_veh = R_align @ [phone-frame vector].
    x = forward, y = lateral, z = up, in the vehicle body frame."""
    out = df.copy()
    accel_p = df[list(lin_accel_cal_cols)].to_numpy()
    accel_v = accel_p @ R_align.T
    out["lin_accel_x_mps2_cal_veh"] = accel_v[:, 0]
    out["lin_accel_y_mps2_cal_veh"] = accel_v[:, 1]
    out["lin_accel_z_mps2_cal_veh"] = accel_v[:, 2]
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
    from src.dead_reckoning.naive_drift_baseline import (
        naive_double_integrate, gps_reference_distance, plot_drift,
    )

    csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
    df = load_smartphone_drive(csv_path)

    # Days 1-3 pipeline: gravity compensation -> ZUPT -> bias correction
    df = compute_linear_acceleration(df)
    mask = zupt_mask(df)
    bias = estimate_constant_bias(df, mask)
    df = apply_bias_correction(df, bias)

    # Day 4 part 1: leveling
    R_level = estimate_leveling_rotation(df, mask)
    print("Part 1 -- recovered tilt:", recovered_roll_pitch_deg(R_level))

    # Day 4 part 2: heading
    accel_mask = accelerating_mask(df)
    print(f"Part 2 -- {accel_mask.sum()} accelerating samples found "
          f"for heading estimation")
    mount = estimate_mount_rotation(df, mask, accel_mask=accel_mask)
    print(f"Part 2 -- estimated heading offset: {mount['heading_offset_deg']:.2f} deg")
    print("R_align (phone frame -> vehicle frame):")
    print(mount["R_align"])

    df = apply_mount_alignment(df, mount["R_align"])

    # re-run the drift baseline on the vehicle-frame forward axis
    integ = naive_double_integrate(df, accel_col="lin_accel_x_mps2_cal_veh")
    true_dist = gps_reference_distance(df)
    error = integ["position"] - true_dist
    duration_s = float(df["t_sec"].iloc[-1] - df["t_sec"].iloc[0])
    final_error_m = round(float(error[-1]), 2)
    drift_rate = round(final_error_m / duration_s * 60, 2)
    rmse = round(float(np.sqrt(np.mean(error ** 2))), 2)

    print(f"\nDay 4 (mount-aligned) final error: {final_error_m} m "
          f"(drift {drift_rate} m/min, RMSE {rmse} m)")

    # for comparison: same drive, Day 3 result (bias-corrected but NOT
    # mount-aligned -- still in the phone's own x-axis)
    integ_d3 = naive_double_integrate(df, accel_col="lin_accel_x_mps2_cal")
    error_d3 = integ_d3["position"] - true_dist
    print(f"(Day 3 same-drive comparison, phone-frame x-axis, no mount "
          f"alignment: {round(float(error_d3[-1]), 2)} m final error)")

    plot_result = {
        **integ, "true_distance_m": true_dist, "error": error,
        "final_error_m": final_error_m, "drift_rate_m_per_min": drift_rate,
    }
    plot_path = PROJECT_ROOT / "results" / "plots" / "day4_mount_aligned_drift.png"
    plot_drift(plot_result, plot_path,
               title="Day 4 — Mount-Aligned Drift (vehicle frame)")

    metrics_path = PROJECT_ROOT / "results" / "metrics.md"
    with open(metrics_path, "a") as f:
        f.write(
            f"| Day 4 | Mount-aligned (vehicle-frame) double integration | "
            f"{final_error_m} | {drift_rate} | {rmse} |\n"
        )
    print(f"Logged Day 4 row to {metrics_path}")