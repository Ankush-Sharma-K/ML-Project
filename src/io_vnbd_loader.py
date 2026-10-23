"""
io_vnbd_loader.py — Day 1 data-loading stub for the IO-VNBD dataset.

IO-VNBD ("Inertial and Odometry Vehicle Navigation Benchmark Dataset",
Onyekpe, Palade, Kanarachos & Szkolnik, Data in Brief 2021) is the
dataset named in the problem statement. It ships two families of CSVs:

  - "S-<name>.csv"  smartphone recordings (AndroSensor app @ 10 Hz;
                    GPS fixes only every ~10 s in most real files, held
                    constant in between) — 24 columns. THIS is the one relevant
                    to our project (phone-only IDR), so this loader
                    targets the smartphone schema.
  - "V-<name>.csv"  vehicle CAN-bus / ECU recordings (VBOX @ 10 Hz) —
                    29 columns, includes wheel speed, steering angle,
                    etc. Not our primary input, but some "V-" drives
                    have a synchronised "S-" counterpart useful for
                    cross-checking velocity ground truth.

Column layout below is taken directly from the dataset's published
specifications table (Table 5 of the Data in Brief paper) — it's
reproduced here (structure/units, not the paper's text) so the loader
is correct without guessing at a real CSV's header row.

Get the data from: https://github.com/onyekpeu/IO-VNBD

Usage:
    from io_vnbd_loader import load_smartphone_drive
    df = load_smartphone_drive("data/raw/S-S1.csv")

REAL-DATA REPAIRS (Day 8 audit, each found by checking all 97 real files):
  1. GPS SPEED UNITS. The header says "GPS SPEED (Kmh)" but in 94 of 97
     real files the values are m/s -- integrating them as m/s reproduces
     the GPS lat/lon track length (e.g. S-S1: 38.5 km vs 37.2 km track;
     as km/h it would be 10.7 km). Three 2 Hz files (S-A2, S-A9, S-A10)
     really are km/h. `detect_speed_unit()` decides per file by comparing
     against the haversine track, and the loader then rewrites
     `gps_speed_kmh` so it is TRUE km/h everywhere downstream.
  2. LOGGER DUPLICATES. S-T1/T4/T5/T6 contain long runs of rows with
     identical IMU values and 0-1 ms timestamp spacing (the logger
     re-writing one frozen sample). These are dropped.
  3. TIME COUNTER RESETS. `TIME SINCE START (ms)` jumps back to ~0
     mid-file in S-M, S-S2, S-S3b, S-S4 and S-Y1; `t_sec` is unwrapped
     across those resets so it stays monotonic.
  4. SPARSE GPS. GPS speed/position only update every ~10 s in most files
     and are held constant in between. `gps_speed_interp_ms` linearly
     interpolates speed between fix instants -- use it, not the held
     value, as the per-sample speed reference/target.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# Column order + names for the smartphone ("S-") files, per the dataset's
# specifications table. If your actual CSV already has a header row with
# different names, pass header=0 to load_smartphone_drive() and it will
# just rename positionally — check the printed column list after loading.
SMARTPHONE_COLUMNS = [
    "gps_lat_deg",
    "gps_lon_deg",
    "gps_alt_m",
    "gps_speed_kmh",
    "gps_accuracy_m",
    "gps_orientation_deg",
    "gps_satellites_in_range",
    "time_since_start_ms",
    "datetime",
    "accel_x_mps2",
    "accel_y_mps2",
    "accel_z_mps2",
    "gravity_x_mps2",
    "gravity_y_mps2",
    "gravity_z_mps2",
    "gyro_yaw_rads",
    "gyro_pitch_rads",
    "gyro_roll_rads",
    "mag_x_ut",
    "mag_y_ut",
    "mag_z_ut",
    "orientation_yaw_deg",
    "orientation_pitch_deg",
    "orientation_roll_deg",
]

# Some drives (S-T*.csv) were recorded with an older AndroSensor config
# that omits the last 6 columns (magnetometer + orientation). All IMU +
# GPS columns are identical — the loader auto-detects which schema applies.
SMARTPHONE_COLUMNS_18 = SMARTPHONE_COLUMNS[:18]

SMARTPHONE_SAMPLE_RATE_HZ = 10.0
SMARTPHONE_GPS_RATE_HZ = 0.1  # typical real fix rate; values are held between fixes

_IMU_RAW_COLS = [
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads",
]


def _drop_logger_duplicates(df: pd.DataFrame, max_gap_ms: float = 50.0) -> tuple[pd.DataFrame, int]:
    """Drop rows that repeat the previous row's IMU values within
    `max_gap_ms` -- the frozen-logger bursts seen in S-T1/T4/T5/T6."""
    imu = df[_IMU_RAW_COLS].to_numpy(dtype=float)
    ms = df["time_since_start_ms"].to_numpy(dtype=float)
    same_imu = np.r_[False, (np.diff(imu, axis=0) == 0).all(axis=1)]
    tiny_gap = np.r_[False, np.abs(np.diff(ms)) < max_gap_ms]
    dup = same_imu & tiny_gap
    return df.loc[~dup].reset_index(drop=True), int(dup.sum())


def _unwrapped_time_sec(ms: np.ndarray) -> tuple[np.ndarray, int]:
    """Seconds from drive start, with backwards counter resets removed:
    each reset is stitched on one nominal sample period after the last
    good value, so `t_sec` is monotonic. Returns (t_sec, n_resets)."""
    ms = ms.astype(float)
    d = np.diff(ms)
    pos = d[d > 0]
    nominal = float(np.median(pos)) if len(pos) else 100.0
    jumps = np.where(d < 0, -d + nominal, 0.0)
    unwrapped = ms + np.r_[0.0, np.cumsum(jumps)]
    return (unwrapped - unwrapped[0]) / 1000.0, int((d < 0).sum())


def _gps_fix_index(df: pd.DataFrame) -> np.ndarray:
    """Row indices where a new GPS fix arrives (lat, lon or speed changes)."""
    cols = ["gps_lat_deg", "gps_lon_deg", "gps_speed_kmh"]
    v = df[cols].to_numpy(dtype=float)
    changed = np.r_[True, (np.diff(v, axis=0) != 0).any(axis=1)]
    return np.where(changed)[0]


def gps_track_length_m(df: pd.DataFrame) -> float:
    """Haversine length of the GPS lat/lon track, using fix rows only."""
    idx = _gps_fix_index(df)
    lat = np.radians(df["gps_lat_deg"].to_numpy(dtype=float)[idx])
    lon = np.radians(df["gps_lon_deg"].to_numpy(dtype=float)[idx])
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat != 0)
    lat, lon = lat[ok], lon[ok]
    if len(lat) < 2:
        return 0.0
    a = (np.sin(np.diff(lat) / 2) ** 2
         + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2)
    return float(np.sum(2 * 6_371_000.0 * np.arcsin(np.sqrt(a))))


def detect_speed_unit(df: pd.DataFrame, min_track_m: float = 200.0) -> str:
    """'mps' or 'kmh' for the raw speed column, decided by which unit makes
    integrated speed match the GPS track length (ratio ~1.0 means m/s,
    ~0.28 means km/h). Too little movement to tell -> 'mps', the unit of
    the vast majority of real IO-VNBD files."""
    t = df["t_sec"].to_numpy(dtype=float)
    dt = np.clip(np.diff(t, prepend=t[0]), 0.0, 1.0)
    integrated = float(np.nansum(df["gps_speed_kmh"].to_numpy(dtype=float) * dt))
    track = gps_track_length_m(df)
    if track < min_track_m or integrated < min_track_m:
        return "mps"
    ratio = track / integrated
    return "kmh" if 0.18 < ratio < 0.45 else "mps"


def interpolate_gps_speed(df: pd.DataFrame) -> np.ndarray:
    """Per-sample speed (m/s) linearly interpolated between GPS fix
    instants, instead of the held step value. Requires true-unit
    `gps_speed_ms`."""
    idx = _gps_fix_index(df)
    t = df["t_sec"].to_numpy(dtype=float)
    v = df["gps_speed_ms"].to_numpy(dtype=float)
    return np.interp(t, t[idx], v[idx])


def load_smartphone_drive(csv_path: str | Path, header: int | None = None) -> pd.DataFrame:
    """
    Load one IO-VNBD smartphone ("S-*.csv") drive into a tidy DataFrame.

    Parameters
    ----------
    csv_path : path to the CSV file (e.g. data/raw/S-S1.csv)
    header   : pass 0 if the file already has its own header row you want
               to keep (columns will then just be renamed positionally,
               so the file's real column count must be 24). Pass None
               (default) if the file has no header row.

    Returns
    -------
    DataFrame with SMARTPHONE_COLUMNS as columns, `datetime` parsed to
    pandas Timestamp, and a derived `t_sec` column (seconds from drive
    start) for convenience in plotting/windowing later.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=header, encoding="iso-8859-1")

    # Some files (e.g. S-A4.csv) have a trailing comma in the header row which
    # causes pandas to create an extra "Unnamed: N" column full of NaN values.
    # Strip all such trailing unnamed columns before schema detection so they
    # are handled identically to their clean counterparts.
    unnamed_suffix = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_suffix:
        df = df.drop(columns=unnamed_suffix)

    # Auto-detect schema: 24-column (full) or 18-column (no mag/orientation,
    # used by some drives e.g. S-T*.csv recorded with older AndroSensor config).
    n_cols = df.shape[1]
    if n_cols == len(SMARTPHONE_COLUMNS):
        col_schema = SMARTPHONE_COLUMNS
    elif n_cols == len(SMARTPHONE_COLUMNS_18):
        col_schema = SMARTPHONE_COLUMNS_18
    else:
        raise ValueError(
            f"{csv_path.name}: expected {len(SMARTPHONE_COLUMNS)} or "
            f"{len(SMARTPHONE_COLUMNS_18)} columns (IO-VNBD smartphone schema), "
            f"found {n_cols}. Check that this is an 'S-' file and not a 'V-' "
            "(ECU) file, and that the header argument matches the file's format."
        )
    df.columns = col_schema

    # Ensure all non-datetime columns are numeric. In some files (e.g. S-A4.csv
    # which has a trailing comma artefact) pandas reads every column as object
    # dtype. Force conversion so arithmetic on t_sec and IMU cols always works.
    for c in col_schema:
        if c != "datetime":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Detect one-column-shift corruption (e.g. S-A4.csv) where
    # "GPS SATELLITES IN RANGE" ("13 / 24") landed one column late, shifting
    # EVERY later column right by one -- time is in raw column 8, date in 9,
    # accel x/y/z in 10-12, ... and orientation roll in the trailing
    # "Unnamed: 24" column that was dropped above. Signature:
    # time_since_start_ms is all-NaN after numeric coercion. Repair: take
    # raw columns 0-5 as-is, satellites -> NaN, raw columns 8-24 -> schema
    # columns 7-23. (The previous version of this repair was off by one --
    # it mapped DATE onto accel_x, accel_x onto accel_y, and so on.)
    if df["time_since_start_ms"].isna().all():
        raw = pd.read_csv(csv_path, header=header, encoding="iso-8859-1")
        if raw.shape[1] != len(SMARTPHONE_COLUMNS) + 1:
            raise ValueError(
                f"{csv_path.name}: time column is unreadable and the file does "
                f"not have the known one-column-shift layout ({raw.shape[1]} columns)."
            )
        df = raw.iloc[:, list(range(6)) + list(range(8, raw.shape[1]))].copy()
        df.insert(6, "gps_satellites_in_range", np.nan)
        df.columns = SMARTPHONE_COLUMNS
        col_schema = SMARTPHONE_COLUMNS
        for c in col_schema:
            if c != "datetime":
                df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time_since_start_ms"] + _IMU_RAW_COLS).reset_index(drop=True)
    # GPS values are held between fixes anyway; fill the odd blank cell
    # (S-Y1 speed, S-A5/A6 course) the same way.
    gps_cols = ["gps_lat_deg", "gps_lon_deg", "gps_speed_kmh", "gps_orientation_deg"]
    df[gps_cols] = df[gps_cols].ffill().bfill()
    df, n_dup = _drop_logger_duplicates(df)

    # time_since_start_ms -> seconds, monotonic from 0 (resets unwrapped)
    df["t_sec"], n_resets = _unwrapped_time_sec(df["time_since_start_ms"].to_numpy())

    # Speed column -> true units (see module docstring, repair 1)
    unit = detect_speed_unit(df)
    raw_speed = df["gps_speed_kmh"].astype(float)
    df["gps_speed_ms"] = raw_speed if unit == "mps" else raw_speed / 3.6
    df["gps_speed_kmh"] = df["gps_speed_ms"] * 3.6
    df["gps_speed_interp_ms"] = interpolate_gps_speed(df)

    df.attrs.update({
        "speed_unit_in_file": unit,
        "n_logger_duplicates_dropped": n_dup,
        "n_time_resets_repaired": n_resets,
    })

    # datetime parsing is best-effort — format varies slightly across
    # the dataset's files, so failures are coerced to NaT rather than
    # raising (t_sec is the reliable time axis regardless).
    df["datetime"] = pd.to_datetime(
        df["datetime"].astype(str).str.replace(
            r"(\d{2}:\d{2}:\d{2}):(\d{3})$", r"\1.\2", regex=True),
        format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")

    return df


def drive_summary(df: pd.DataFrame) -> dict:
    """Quick stats used as a sanity check right after loading a drive."""
    duration_s = df["t_sec"].iloc[-1] - df["t_sec"].iloc[0]
    dt = np.diff(df["t_sec"].values)
    return {
        "n_samples": len(df),
        "duration_s": round(float(duration_s), 1),
        "duration_min": round(float(duration_s) / 60, 2),
        "median_dt_s": round(float(np.median(dt)), 4),
        "implied_rate_hz": round(1.0 / float(np.median(dt)), 2) if np.median(dt) > 0 else None,
        "gps_speed_max_kmh": round(float(df["gps_speed_kmh"].max()), 1),
        "gps_speed_mean_kmh": round(float(df["gps_speed_kmh"].mean()), 1),
        **df.attrs,
    }


if __name__ == "__main__":
    import sys
    import glob

    # Resolve paths relative to the project root (parent of this file's
    # `src/` folder), not the current working directory, so this runs
    # correctly no matter where it's launched from.
    _project_root = Path(__file__).resolve().parent.parent
    _raw_dir = _project_root / "data" / "raw"

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        # No path given: auto-pick a file from data/raw/ — prefer the
        # synthetic drive from Day 1 if present, else the first S-*.csv.
        synth = _raw_dir / "SYNTH-drive1.csv"
        candidates = [synth] if synth.exists() else sorted(_raw_dir.glob("S-*.csv"))
        if not candidates:
            print(
                f"No CSV found in {_raw_dir}.\n"
                f"Run `python src/utils/synthetic_data.py` first to generate "
                f"a test drive, or download real IO-VNBD data (S-*.csv) from "
                f"https://github.com/onyekpeu/IO-VNBD into that folder.\n"
                f"You can also pass a path directly: "
                f"`python src/io_vnbd_loader.py path/to/file.csv`"
            )
            sys.exit(1)
        path = candidates[0]
        print(f"No path given — using {path}")

    df = load_smartphone_drive(path)
    print(df.head())
    print(drive_summary(df))