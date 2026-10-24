"""Loader unit detection, GPS speed interpolation, sliding windows."""

import numpy as np
import pandas as pd

from conftest import synthetic_drive
from src.io_vnbd_loader import detect_speed_unit, interpolate_gps_speed
from src.models.dataset_prep import build_windows


def _straight_drive(speed_ms=10.0, seconds=200):
    x = np.arange(int(seconds * 10)) * speed_ms / 10.0
    return x, np.zeros_like(x)


def test_speed_unit_mps_when_column_holds_mps():
    df = synthetic_drive(*_straight_drive(), speed_col_factor=1.0)
    assert detect_speed_unit(df) == "mps"


def test_speed_unit_kmh_when_column_holds_kmh():
    df = synthetic_drive(*_straight_drive(), speed_col_factor=3.6)
    assert detect_speed_unit(df) == "kmh"


def test_speed_unit_defaults_to_mps_without_movement():
    df = synthetic_drive(np.zeros(500), np.zeros(500))
    assert detect_speed_unit(df) == "mps"


def test_interpolate_gps_speed_is_linear_between_fixes():
    # accelerate 0 -> 20 m/s over 20 s; fixes every 5 s are held in between
    t = np.arange(201) / 10.0
    x = 0.5 * 1.0 * t ** 2
    df = synthetic_drive(x, np.zeros_like(x), fix_every_s=5.0)
    v = interpolate_gps_speed(df)
    held = df["gps_speed_ms"].to_numpy()
    mid = 75  # 7.5 s: between the fixes at 5 s and 10 s
    assert abs(v[mid] - 0.5 * (held[50] + held[100])) < 1e-6
    assert held[mid] == held[50]


def test_build_windows_labels_last_sample_and_rejects_gaps():
    n = 100
    df = pd.DataFrame({"t_sec": np.arange(n) * 0.1, "a": np.arange(n, dtype=float), "v": np.arange(n) * 2.0})
    df.loc[60:, "t_sec"] += 1.0  # a 1 s gap between rows 59 and 60
    w = build_windows(df, ["a"], "v", window_size=20, stride=10)
    for X, y, s in zip(w["X"], w["y"], w["start_idx"]):
        assert y == df["v"].iloc[s + 19]
        assert X[-1, 0] == df["a"].iloc[s + 19]
        assert not (s < 60 <= s + 19)  # no accepted window spans the gap
    assert w["n_rejected"] >= 1
