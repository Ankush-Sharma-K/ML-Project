"""Shared helpers for the unit tests. All tests use small synthetic data, so
they run in seconds without the IO-VNBD files or internet access.
Run from the project root:  python -m pytest tests -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAT0, LON0 = 52.5, -1.5


def synthetic_drive(x_m: np.ndarray, y_m: np.ndarray, hz: float = 10.0, fix_every_s: float = 1.0,
                    speed_col_factor: float = 1.0) -> pd.DataFrame:
    """Phone-log-like frame for a car at metres (x east, y north) per sample:
    t_sec, GPS lat/lon held between fixes, gps_speed_kmh (= true speed x
    `speed_col_factor`, held), gps_speed_ms."""
    from src.map_matching.hmm_map_matcher import from_local_xy

    n = len(x_m)
    t = np.arange(n) / hz
    lat, lon = from_local_xy(x_m, y_m, LAT0, LON0)
    speed = np.r_[np.hypot(np.diff(x_m), np.diff(y_m)) * hz, 0.0]
    speed[-1] = speed[-2] if n > 1 else 0.0
    step = max(int(round(fix_every_s * hz)), 1)
    held = (np.arange(n) // step) * step
    return pd.DataFrame({"t_sec": t, "gps_lat_deg": lat[held], "gps_lon_deg": lon[held],
                         "gps_speed_kmh": speed[held] * speed_col_factor, "gps_speed_ms": speed[held]})
