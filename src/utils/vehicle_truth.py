"""
Wheel-speed ground truth from the IO-VNBD vehicle ("V-*.csv") files.

The phone's GPS speed only updates every ~10 s, so interpolating it is a
blurred speed reference (median 0.9-1.2 m/s off the car's own speed). The
"Synchronised" zip ships, for 72 drives, a V-file logged by the car itself
at 10 Hz: four wheel speeds (rad/s), a VBOX GPS velocity, the speedometer,
yaw rate, ... On the 63 V-files checked, wheel speed agreed with VBOX
velocity to a median 0.09 m/s and with the speedometer to 0.07 m/s, so
wheel speed x fitted wheel radius (~0.277 m) is used as the truth.

WHERE THE FILES GO: `data/raw/vehicle/V/V-*.csv` and the matching
synchronised phone files `data/raw/vehicle/S_sync/S-*.csv` (the
"Uncategorised IOVNB Dataset/V-Dataset" and ".../S-Dataset" folders of
"Synchronised V abd S datasets.zip"). A V-file is row-for-row with its
S_sync file, which carries the timestamps.

ALIGNMENT (checked, not assumed): our phone files come from the
Unsynchronised zip, and "synchronised" does not mean in sync with them --
offsets of up to +-28 s were found. The offset is estimated per drive by
cross-correlating wheel speed with the phone's GPS speed over +-60 s; on
the five drives where the phone gyro vs car yaw rate gave a clear,
latency-free offset, both methods agreed (e.g. S-S2 +8.6 s both, S-S3a
-7.0 vs -6.8 s). Drives where the match is not clear (r < 0.9, or the best
lag barely beats lags >= 5 s away) are rejected -- S-Y1 and S-S4 fail
completely. 39 drives (~19 h) pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIN_CORRELATION = 0.9
MIN_MARGIN = 0.02
MAX_LAG_S = 60.0


def _parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(
        s.astype(str).str.strip().str.replace(r"(\d{2}:\d{2}:\d{2}):(\d{3})$", r"\1.\2", regex=True),
        format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")


def _find_case_insensitive(folder: Path, filename: str) -> Path | None:
    target = filename.lower()
    for p in folder.glob("*.csv"):
        if p.name.lower() == target:
            return p
    return None


def load_wheel_speed_truth(
    phone_df: pd.DataFrame,
    drive_name: str,
    vehicle_dir: str | Path,
) -> tuple[np.ndarray | None, dict]:
    """Wheel-speed truth (m/s) aligned row-for-row to `phone_df` (a drive
    loaded with `load_smartphone_drive`), or (None, info) if this drive has
    no V-file or the alignment is not clear. NaN where the car log does not
    cover a phone row. info = {"status", "lag_s", "r", "margin", "radius_m"}.
    """
    vehicle_dir = Path(vehicle_dir)
    key = drive_name[2:] if drive_name.upper().startswith("S-") else drive_name
    v_path = _find_case_insensitive(vehicle_dir / "V", f"V-{key}")
    s_path = _find_case_insensitive(vehicle_dir / "S_sync", f"S-{key}")
    if v_path is None or s_path is None:
        return None, {"status": "no vehicle file"}

    V = pd.read_csv(v_path, encoding="latin-1")
    V.columns = [c.strip() for c in V.columns]
    S = pd.read_csv(s_path, encoding="latin-1", header=0)
    S = S.loc[:, ~S.columns.astype(str).str.startswith("Unnamed")]
    if abs(len(V) - len(S)) > 2:
        return None, {"status": f"row count mismatch ({len(V)} vs {len(S)})"}
    n = min(len(V), len(S))
    V, S = V.iloc[:n], S.iloc[:n]

    stamps = _parse_dt(S[[c for c in S.columns if "DATE" in str(c).upper()][0]])
    wheel_rad = V[[c for c in V.columns if c.startswith("Wheel Speed")]].to_numpy(float).mean(axis=1)
    vbox = V["Velocity (km/hr)"].to_numpy(float) / 3.6
    moving = (vbox > 5) & (wheel_rad > 1)
    if moving.sum() < 100:
        return None, {"status": "too little motion"}
    radius = float(np.median(vbox[moving] / wheel_rad[moving]))
    car = pd.Series(wheel_rad * radius, index=stamps.values)
    car = car[~car.index.isna()]
    car = car[~car.index.duplicated()].sort_index()

    t0 = phone_df["datetime"].iloc[0]
    car_t = (car.index - t0).total_seconds().to_numpy()
    our_t = (phone_df["datetime"] - t0).dt.total_seconds().to_numpy()
    ok = np.isfinite(our_t)
    phone = phone_df["gps_speed_interp_ms"].to_numpy()[ok]

    lags = np.arange(-MAX_LAG_S, MAX_LAG_S + 0.01, 0.2)
    cors = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        w = np.interp(our_t[ok] + lag, car_t, car.to_numpy(), left=np.nan, right=np.nan)
        m = np.isfinite(w)
        if m.sum() > 300:
            cors[i] = np.corrcoef(w[m], phone[m])[0, 1]
    if np.all(np.isnan(cors)):
        return None, {"status": "no time overlap"}
    best = int(np.nanargmax(cors))
    lag, r = float(lags[best]), float(cors[best])
    far = np.abs(lags - lag) >= 5
    margin = float(r - np.nanmax(cors[far])) if np.any(np.isfinite(cors[far])) else 0.0
    info = {"lag_s": round(lag, 1), "r": round(r, 3), "margin": round(margin, 3), "radius_m": round(radius, 3)}
    if r < MIN_CORRELATION or margin < MIN_MARGIN:
        return None, {"status": "alignment not clear", **info}

    truth = np.full(len(phone_df), np.nan)
    truth[ok] = np.interp(our_t[ok] + lag, car_t, car.to_numpy(), left=np.nan, right=np.nan)
    return truth, {"status": "ok", **info}
