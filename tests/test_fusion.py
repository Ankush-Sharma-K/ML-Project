"""EKF update / gate / RTS smoother, and the vehicle EKF on a synthetic drive."""

import numpy as np

from src.fusion.kalman_filter import CHI2_999, ekf_update, rts_smooth
from src.fusion.gnss_ins_fusion import E, N, PSI, V, FusionConfig, run_ekf


def test_update_moves_towards_measurement_and_shrinks_covariance():
    x, P = np.array([0.0]), np.array([[4.0]])
    x2, P2, ok, _ = ekf_update(x, P, np.array([2.0]), x, np.eye(1), np.array([[4.0]]))
    assert ok and abs(x2[0] - 1.0) < 1e-12 and abs(P2[0, 0] - 2.0) < 1e-12


def test_gate_rejects_outlier():
    x, P = np.zeros(2), np.eye(2)
    _, _, ok, nis = ekf_update(x, P, np.array([50.0, 0.0]), x, np.eye(2), np.eye(2), gate=CHI2_999[2])
    assert not ok and nis > CHI2_999[2]


def test_angle_innovation_is_wrapped():
    x, P = np.array([np.radians(179.0)]), np.array([[0.1]])
    x2, _, _, _ = ekf_update(x, P, np.array([np.radians(-179.0)]), x, np.eye(1), np.array([[0.1]]), angle_rows=(0,))
    assert abs(np.degrees(x2[0]) - 180.0) < 1.5  # moved by ~1 deg, not by -358 deg


def test_rts_smoother_uses_the_final_measurement():
    """1-D constant position with process noise: measurements only at the
    start and the end; the smoother must pull the middle towards the end."""
    n = 11
    q, r = 1.0, 0.01
    x, P = np.array([0.0]), np.array([[r]])
    x_f, P_f, x_p, P_p, F = (np.zeros((n, 1)), np.zeros((n, 1, 1)), np.zeros((n, 1)), np.zeros((n, 1, 1)),
                             np.tile(np.eye(1), (n, 1, 1)))
    for k in range(n):
        if k > 0:
            P = P + q
        x_p[k], P_p[k] = x, P
        if k == n - 1:
            x, P, _, _ = ekf_update(x, P, np.array([10.0]), x, np.eye(1), np.array([[r]]))
        x_f[k], P_f[k] = x, P
    x_s, P_s = rts_smooth(x_f, P_f, x_p, P_p, F)
    assert abs(x_s[-1, 0] - x_f[-1, 0]) < 1e-12
    assert 4.0 < x_s[5, 0] < 6.0  # roughly half-way
    assert P_s[5, 0, 0] < P_f[5, 0, 0]


def test_vehicle_ekf_tracks_a_turning_car_and_coasts_through_an_outage():
    """Car at 10 m/s turning at 3 deg/s; GPS every 10 s for 120 s, then a 30 s
    outage with gyro + (biased) ML speed only."""
    hz, T = 10, 150
    t = np.arange(T * hz) / hz
    w = np.radians(3.0)
    psi = w * t
    x_true = np.cumsum(10.0 * np.sin(psi)) / hz
    y_true = np.cumsum(10.0 * np.cos(psi)) / hz
    omega = np.full(len(t), w)
    ml = {k: 10.0 + 0.5 for k in range(0, len(t), 10)}
    gps = {k: (x_true[k], y_true[k], 3.0, 10.0, psi[k]) for k in range(0, 1201, 100)}
    cfg = FusionConfig()
    x0 = np.array([x_true[0], y_true[0], 0.0, 10.0, 0.0, 0.0])
    P0 = np.diag([9.0, 9.0, 0.1, 4.0, 9.0, 1e-4])
    r = run_ekf(t, omega, ml, gps, x0, P0, cfg, gyro_ok=True)
    k_entry, k_end = 1200, len(t) - 1
    assert np.hypot(r["x_f"][k_entry, E] - x_true[k_entry], r["x_f"][k_entry, N] - y_true[k_entry]) < 5.0
    assert abs(r["x_f"][k_end, V] - 10.0) < 1.0
    err_end = np.hypot(r["x_f"][k_end, E] - x_true[k_end], r["x_f"][k_end, N] - y_true[k_end])
    assert err_end < 25.0  # 300 m driven with a 5% speed bias that the filter partly removes
    assert all(ok for ok, _ in r["gate"].values())
    d_psi = (r["x_f"][k_end, PSI] - psi[k_end] + np.pi) % (2 * np.pi) - np.pi
    assert abs(np.degrees(d_psi)) < 5.0
