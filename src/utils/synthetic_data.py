"""
synthetic_data.py — Day 1 utility: generate a fake "drive" that matches
the IO-VNBD smartphone ("S-*.csv") schema from io_vnbd_loader.py.

Why this exists: the real IO-VNBD dataset needs to be downloaded from
GitHub, which this sandbox can't do (no internet). This generator lets
us build and sanity-check the whole pipeline (loader -> calibration ->
DR -> fusion) TODAY using a physically-plausible fake drive, so nothing
blocks on the download. Swap this for real data the moment you have it
— every downstream module only depends on the DataFrame shape/columns,
not on how it was produced.

The simulated vehicle: drives straight, accelerates, cruises, brakes to
a stop, sits stationary, then repeats — with realistic-ish IMU noise and
bias layered on top, so Day-2's "naive double integration drifts badly"
demo has something real to show.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

# Make sure `src/` (the parent of this file's `utils/` folder) is on
# sys.path, so `io_vnbd_loader` can be found regardless of the current
# working directory this script is launched from (VS Code "Run", a
# terminal in a different folder, double-click, etc.).
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from io_vnbd_loader import SMARTPHONE_COLUMNS, SMARTPHONE_SAMPLE_RATE_HZ


def _segment_speed_profile(duration_s: float, fs: float, cruise_speed_ms: float, seed: int):
    """Accelerate -> cruise -> brake -> stop -> repeat, in speed (m/s)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    speed = np.zeros(n)

    cycle_s = 20.0  # one accel/cruise/brake/stop cycle
    for start in np.arange(0, duration_s, cycle_s):
        mask = (t >= start) & (t < start + cycle_s)
        local_t = t[mask] - start
        s = np.zeros(local_t.shape)
        accel_end, cruise_end, brake_end = 4.0, 12.0, 16.0
        a = rng.uniform(2.0, 3.0)  # accel phase, m/s^2
        s = np.where(local_t < accel_end, a * local_t, s)
        v_cruise = a * accel_end
        s = np.where(
            (local_t >= accel_end) & (local_t < cruise_end),
            v_cruise, s
        )
        brake_dur = brake_end - cruise_end
        s = np.where(
            (local_t >= cruise_end) & (local_t < brake_end),
            v_cruise * (1 - (local_t - cruise_end) / brake_dur), s
        )
        s = np.where(local_t >= brake_end, 0.0, s)
        speed[mask] = np.clip(s, 0, cruise_speed_ms)
    return t, speed


def generate_synthetic_drive(
    duration_s: float = 120.0,
    fs: float = SMARTPHONE_SAMPLE_RATE_HZ,
    cruise_speed_kmh: float = 40.0,
    accel_noise_std: float = 0.15,   # m/s^2, consumer MEMS-ish
    accel_bias: float = 0.05,        # m/s^2 constant bias (the enemy)
    gyro_noise_std: float = 0.01,    # rad/s
    gyro_bias: float = 0.003,        # rad/s constant bias
    start_lat: float = 22.9734,      # Ludhiana-ish, arbitrary origin
    start_lon: float = 75.7873,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic straight-line drive matching SMARTPHONE_COLUMNS.
    Straight-line on purpose for Day 1-2 (no turning) — turning + curved
    roads get exercised once map matching (Phase 5) needs a real graph.
    """
    rng = np.random.default_rng(seed)
    cruise_ms = cruise_speed_kmh / 3.6
    t, true_speed = _segment_speed_profile(duration_s, fs, cruise_ms, seed)
    n = len(t)

    # True forward acceleration = derivative of true speed
    true_accel_fwd = np.gradient(true_speed, t)

    # IMU "measures" forward accel in its x-axis, with noise + bias.
    accel_x = true_accel_fwd + accel_bias + rng.normal(0, accel_noise_std, n)
    accel_y = rng.normal(0, accel_noise_std * 0.5, n)          # lateral noise only (straight line)
    accel_z = rng.normal(0, accel_noise_std * 0.3, n)          # vertical vibration/potholes
    # occasional pothole-like shock bursts
    n_potholes = int(duration_s / 25)
    for _ in range(n_potholes):
        idx = rng.integers(0, n)
        width = int(0.2 * fs)
        accel_z[idx:idx + width] += rng.uniform(3, 6)

    gravity_x = np.zeros(n)
    gravity_y = np.zeros(n)
    gravity_z = np.full(n, 9.81)

    gyro_yaw = gyro_bias + rng.normal(0, gyro_noise_std, n)     # ~0, straight line
    gyro_pitch = rng.normal(0, gyro_noise_std * 0.3, n)
    gyro_roll = rng.normal(0, gyro_noise_std * 0.3, n)

    mag_x = rng.normal(20, 2, n)
    mag_y = rng.normal(5, 2, n)
    mag_z = rng.normal(-40, 2, n)

    orientation_yaw = np.zeros(n)
    orientation_pitch = rng.normal(0, 0.5, n)
    orientation_roll = rng.normal(0, 0.5, n)

    # Ground-truth position: integrate true speed along a straight line
    # (due north) and convert to lat/lon deltas (small-distance approx).
    true_dist_m = np.concatenate([[0], np.cumsum(true_speed[:-1] * np.diff(t))])
    meters_per_deg_lat = 111_320.0
    gps_lat = start_lat + true_dist_m / meters_per_deg_lat
    gps_lon = np.full(n, start_lon)

    gps_speed_kmh = true_speed * 3.6 + rng.normal(0, 0.5, n)
    gps_speed_kmh = np.clip(gps_speed_kmh, 0, None)
    gps_alt = np.full(n, 550.0) + rng.normal(0, 1, n)
    gps_accuracy = rng.uniform(3, 8, n)
    gps_orientation = np.zeros(n)
    gps_sats = rng.integers(6, 14, n)

    time_since_start_ms = (t * 1000).astype(np.int64)
    datetime = pd.Timestamp("2026-09-22 08:00:00") + pd.to_timedelta(t, unit="s")

    df = pd.DataFrame({
        "gps_lat_deg": gps_lat,
        "gps_lon_deg": gps_lon,
        "gps_alt_m": gps_alt,
        "gps_speed_kmh": gps_speed_kmh,
        "gps_accuracy_m": gps_accuracy,
        "gps_orientation_deg": gps_orientation,
        "gps_satellites_in_range": gps_sats,
        "time_since_start_ms": time_since_start_ms,
        "datetime": datetime,
        "accel_x_mps2": accel_x,
        "accel_y_mps2": accel_y,
        "accel_z_mps2": accel_z,
        "gravity_x_mps2": gravity_x,
        "gravity_y_mps2": gravity_y,
        "gravity_z_mps2": gravity_z,
        "gyro_yaw_rads": gyro_yaw,
        "gyro_pitch_rads": gyro_pitch,
        "gyro_roll_rads": gyro_roll,
        "mag_x_ut": mag_x,
        "mag_y_ut": mag_y,
        "mag_z_ut": mag_z,
        "orientation_yaw_deg": orientation_yaw,
        "orientation_pitch_deg": orientation_pitch,
        "orientation_roll_deg": orientation_roll,
    })
    assert list(df.columns) == SMARTPHONE_COLUMNS
    df["t_sec"] = t
    df["true_speed_ms"] = true_speed          # ground truth, for eval only —
    df["true_distance_m"] = true_dist_m       # not present in real IO-VNBD,
                                                # strip before feeding real-data code paths
    return df


if __name__ == "__main__":
    df = generate_synthetic_drive()
    # _SRC_DIR is .../idr-project/src -> go one level up to the project root,
    # then into data/raw/, so this works no matter what folder you run from.
    project_root = os.path.dirname(_SRC_DIR)
    out_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "SYNTH-drive1.csv")
    df.drop(columns=["t_sec", "true_speed_ms", "true_distance_m"]).to_csv(
        out, index=False, header=False
    )
    print(f"Wrote {out}  ({len(df)} rows, {df['t_sec'].iloc[-1]:.1f}s)")