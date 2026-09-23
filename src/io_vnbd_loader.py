"""
io_vnbd_loader.py — Day 1 data-loading stub for the IO-VNBD dataset.

IO-VNBD ("Inertial and Odometry Vehicle Navigation Benchmark Dataset",
Onyekpe, Palade, Kanarachos & Szkolnik, Data in Brief 2021) is the
dataset named in the problem statement. It ships two families of CSVs:

  - "S-<name>.csv"  smartphone recordings (AndroSensor app @ 10 Hz,
                    GPS @ 1 Hz) — 24 columns. THIS is the one relevant
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

SMARTPHONE_SAMPLE_RATE_HZ = 10.0
SMARTPHONE_GPS_RATE_HZ = 1.0


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
    df = pd.read_csv(csv_path, header=header)

    if df.shape[1] != len(SMARTPHONE_COLUMNS):
        raise ValueError(
            f"{csv_path.name}: expected {len(SMARTPHONE_COLUMNS)} columns "
            f"(IO-VNBD smartphone schema), found {df.shape[1]}. "
            "Check that this is an 'S-' file and not a 'V-' (ECU) file, "
            "and that the header argument matches the file's format."
        )
    df.columns = SMARTPHONE_COLUMNS

    # time_since_start_ms -> seconds, monotonic from 0
    df["t_sec"] = (df["time_since_start_ms"] - df["time_since_start_ms"].iloc[0]) / 1000.0

    # datetime parsing is best-effort — format varies slightly across
    # the dataset's files, so failures are coerced to NaT rather than
    # raising (t_sec is the reliable time axis regardless).
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

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