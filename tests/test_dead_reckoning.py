"""Anchored ML speed, 2-D path, and the per-phone gyro heading fit."""

import numpy as np
import pandas as pd

from src.dead_reckoning.heading import (GyroTurnIntegrator, end_position, fit_gyro_heading, gyro_heading,
                                        wrap_angle)
from src.dead_reckoning.ins_with_ml_velocity import anchored_ml_speed


def test_anchored_ml_starts_at_entry_speed_and_fades_to_model():
    ml = np.full(1201, 12.0)
    tau = np.arange(1201) / 10.0
    v = anchored_ml_speed(ml, v0=20.0, tau_s=tau, fade_s=20.0)
    assert v[0] == 20.0
    assert abs(v[-1] - (12.0 + 8.0 * np.exp(-120.0 / 20.0))) < 1e-9
    assert np.all(np.diff(v) <= 0)


def test_wrap_angle_range():
    a = wrap_angle(np.linspace(-20, 20, 1001))
    assert np.all(a >= -np.pi) and np.all(a < np.pi)


def test_end_position_east_and_north():
    dt = np.full(101, 0.1)
    assert np.allclose(end_position(np.full(101, 10.0), np.full(101, np.pi / 2), dt), (100.0, 0.0), atol=1e-9)
    assert np.allclose(end_position(np.full(101, 10.0), np.zeros(101), dt), (0.0, 100.0), atol=1e-9)


def test_gyro_fit_recovers_swapped_axis_and_scale():
    """Phone lying flat (gravity on phone z), but its yaw rate is logged in the
    'gyro_roll_rads' column with a flipped sign -- the fit must find that
    mapping and the scale from GPS course changes alone."""
    rng = np.random.default_rng(0)
    n = 6000
    t = np.arange(n) / 10.0
    yaw_rate = 0.15 * np.sin(2 * np.pi * t / 40.0) + 0.05 * np.sin(2 * np.pi * t / 13.0)  # rad/s, clockwise
    df = pd.DataFrame({"t_sec": t,
                       "gravity_x_mps2": np.zeros(n), "gravity_y_mps2": np.zeros(n), "gravity_z_mps2": np.full(n, 9.81),
                       "gyro_yaw_rads": 0.01 * rng.standard_normal(n),
                       "gyro_pitch_rads": 0.01 * rng.standard_normal(n),
                       "gyro_roll_rads": -yaw_rate})
    course = np.cumsum(yaw_rate) * 0.1
    i0 = np.arange(0, n - 60, 50)
    i1 = i0 + 50
    integ = GyroTurnIntegrator(df)
    fit = fit_gyro_heading(integ, np.zeros(3), i0, i1, wrap_angle(course[i1] - course[i0]))
    assert fit is not None and abs(fit.r) > 0.99
    assert fit.perm[2] == 2  # gravity z pairs with the third gyro column
    assert abs(abs(fit.k) - 1.0) < 0.05
    rows = np.arange(1000, 2000)
    h = gyro_heading(integ, fit, np.zeros(3), 1000, rows, course[1000])
    assert np.max(np.abs(wrap_angle(h - course[rows]))) < np.radians(2.0)
