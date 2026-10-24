"""Streaming edge engine: status changes through a tunnel, corrected track on
exit, route monitor hysteresis, numpy features vs the training pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import LAT0, LON0
from src.edge_engine.engine import RAW_COLS, IDREngine, RouteMonitor, calibrated_features
from src.map_matching.hmm_map_matcher import from_local_xy


def _synthetic_phone_drive(minutes=6.0, hz=10):
    """Phone flat in the car (gravity on z), 12 m/s, gentle S-curves, GPS fix
    every second; small sensor noise; a 30 s stop at the start."""
    rng = np.random.default_rng(0)
    n = int(minutes * 60 * hz)
    t = np.arange(n) / hz
    v = np.where(t < 30, 0.0, np.minimum((t - 30) * 1.5, 12.0))
    yaw_rate = np.where(t > 60, 0.05 * np.sin(2 * np.pi * t / 60.0), 0.0)
    psi = np.cumsum(yaw_rate) / hz
    x, y = np.cumsum(v * np.sin(psi)) / hz, np.cumsum(v * np.cos(psi)) / hz
    lat, lon = from_local_xy(x, y, LAT0, LON0)
    acc_fwd = np.r_[0.0, np.diff(v) * hz]
    held = (np.arange(n) // hz) * hz
    course = (np.degrees(psi) % 360.0)
    return pd.DataFrame({
        "t_sec": t,
        "accel_x_mps2": rng.normal(0, 0.05, n),  # phone y axis points forward, x sideways
        "accel_y_mps2": acc_fwd + rng.normal(0, 0.05, n),
        "accel_z_mps2": 9.81 + rng.normal(0, 0.05, n),
        "gravity_x_mps2": np.zeros(n), "gravity_y_mps2": np.zeros(n), "gravity_z_mps2": np.full(n, 9.81),
        "gyro_yaw_rads": rng.normal(0, 0.002, n), "gyro_pitch_rads": rng.normal(0, 0.002, n),
        "gyro_roll_rads": -yaw_rate + rng.normal(0, 0.002, n),
        "gps_lat_deg": lat[held], "gps_lon_deg": lon[held], "gps_speed_ms": v[held], "gps_speed_kmh": v[held] * 3.6,
        "gps_orientation_deg": course[held], "gps_accuracy_m": np.full(n, 3.0),
    })


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from src.models.compress_model import DeployableVelocityModel, export_lite
    from src.models.velocity_model import FEATURE_COLS_DEFAULT, VelocityCNNDeep

    torch.manual_seed(0)
    m = VelocityCNNDeep(n_features=8, channels=(8, 16)).eval()
    stats = {"mean": np.zeros(8), "std": np.ones(8)}
    path = export_lite(DeployableVelocityModel(m, stats), tmp_path_factory.mktemp("m") / "tiny.ptl", 200, 8)
    return path, {"feature_cols": FEATURE_COLS_DEFAULT, "window_size": 200}


def test_numpy_features_match_training_pipeline():
    from src.models.multi_drive_data import apply_calibration
    from src.models.velocity_model import FEATURE_COLS_DEFAULT

    df = _synthetic_phone_drive(1.0)
    for a in "xyz":
        df[f"lin_accel_{a}_mps2"] = df[f"accel_{a}_mps2"] - df[f"gravity_{a}_mps2"]
    rot = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    bias = {"lin_accel_x_mps2": 0.1, "lin_accel_y_mps2": -0.2, "lin_accel_z_mps2": 0.05,
            "gyro_yaw_rads": 0.001, "gyro_pitch_rads": -0.002, "gyro_roll_rads": 0.003}
    calib = {"bias": bias, "R_align": rot}
    ref = apply_calibration(df.iloc[:200], calib)[FEATURE_COLS_DEFAULT].to_numpy(dtype=np.float32)
    got = calibrated_features(df.iloc[:200].to_dict("records"), calib, FEATURE_COLS_DEFAULT)
    assert np.allclose(ref, got, atol=1e-6)


def test_engine_goes_dead_reckoning_in_tunnel_and_corrects_track_on_exit(tiny_model):
    path, meta = tiny_model
    df = _synthetic_phone_drive()
    eng = IDREngine(path, meta, min_calib_s=60.0, recalib_every_s=30.0)
    tunnel = (int(240 * 10), int(270 * 10))  # GPS withheld from 240 s to 270 s
    statuses, exit_out = [], None
    for r in range(len(df)):
        s = {c: df[c].iloc[r] for c in RAW_COLS if c in df}
        s["gps_fix"] = (r % 10 == 0) and not (tunnel[0] < r < tunnel[1])
        o = eng.push(s)
        statuses.append(o.status)
        if o.track_corrected:
            exit_out = (r, o)
    assert statuses[0] == "INIT"
    assert statuses[int(200 * 10)] == "GNSS"
    assert statuses[int(260 * 10)] == "DR"  # no map matcher given
    assert statuses[int(280 * 10)] == "GNSS"
    assert exit_out is not None and exit_out[0] == tunnel[1]
    r_exit, o = exit_out
    assert o.exit_realtime is not None
    assert abs(len(o.track_corrected) - (tunnel[1] - tunnel[0] + 1)) <= 2
    # the corrected track ends at the filter's post-exit position, close to the exit fix
    end_lat, end_lon = o.track_corrected[-1][1:]
    from src.map_matching.osm_loader import to_local_xy
    dx, dy = to_local_xy(end_lat, end_lon, df["gps_lat_deg"].iloc[r_exit], df["gps_lon_deg"].iloc[r_exit])
    assert np.hypot(dx, dy) < 15.0


def test_route_monitor_hysteresis():
    lat, lon = from_local_xy(np.array([0.0, 2000.0]), np.array([0.0, 0.0]), LAT0, LON0)  # west-east route
    m = RouteMonitor(lat, lon, base_m=40.0, k_sigma=2.0, hold_s=3.0)
    states = []
    for k in range(100):
        t = float(k)  # 1 s steps
        off = 100.0 if 20 <= k < 60 else 5.0  # 100 m north of the route for 40 s
        sigma = 80.0 if 70 <= k < 90 else 5.0  # very uncertain later: 2 sigma = 160 m > 100 m
        if 70 <= k < 90:
            off = 100.0
        plat, plon = from_local_xy(np.array([k * 10.0]), np.array([off]), LAT0, LON0)
        states.append(m.update(t, float(plat[0]), float(plon[0]), sigma_m=sigma)[0])
    assert all(s == "ON_ROUTE" for s in states[:22])
    assert states[25] == "OFF_ROUTE"           # after the 3 s hold
    assert states[61] == "OFF_ROUTE" and states[65] == "ON_ROUTE"
    assert all(s == "ON_ROUTE" for s in states[70:90])  # 100 m off but within 2 sigma: no alarm
    assert abs(m.distance_m(float(lat[0]), float(lon[0]))) < 1e-6
