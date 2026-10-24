"""
heading.py — Day 10: direction and 2-D path during GPS outages
==============================================================

Days 8-9 estimate HOW FAR the car travels during a GPS outage. The problem
statement also needs the DIRECTION and the PATH (for comparison with the
map route, Phase 5). This module adds the heading and turns speed + heading
into an east/north trajectory.

HEADING FROM THE GYROSCOPE. Heading = GPS course at the entrance + the
phone's accumulated rotation about the vertical axis:

    course(t) = course0 + k * integral( w_mapped . g_unit dt )

    w        = gyro minus its bias (bias from the calibration stops)
    g_unit   = direction of gravity from the phone's gravity sensor
    mapping  = which gyro column pairs with which gravity axis, with signs

WHY THE MAPPING IS FITTED (checked on the real data): the gyro columns are
not in the same axis order / sign as the gravity columns on most phones --
the standard pairing was the best fit in only 17% of outages. So, like the
rest of the calibration, the mapping (6 axis orders x 4 sign patterns) and
the scale k are fitted on the GPS course changes BEFORE the outage, over GPS
fix intervals >= 4 s apart. A fit is used only if its correlation is
>= 0.5 (75% of outages; median |r| 0.84); otherwise the heading falls back
to the last GPS course (straight line).

NOT USED: the phone's fused orientation yaw (gyro + magnetometer). It could
be calibrated (|r| >= 0.5) in only 20% of outages -- magnetometers are
unreliable in a steel car and in tunnels.

`__main__` = the Day 10 benchmark. Outages start right after a GPS fix (so
position, speed and course at the entrance are all real) and are scored at
the first GPS fix after 30 / 60 / 120 s: error = metres between the
estimated end position and that fix.
  Part A (all drives): TRUE distance + each heading -> heading error only.
  Part B (TEST drives): full path = anchored-ML distance (Day 9) + heading,
  against the baseline "hold last speed, keep last direction".
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GYRO_COLS = ["gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads"]
GRAVITY_COLS = ["gravity_x_mps2", "gravity_y_mps2", "gravity_z_mps2"]
M_PER_DEG_LAT = 111_320.0
# 6 axis orders x 4 sign patterns (the overall sign is absorbed by the scale k)
GYRO_MAPPINGS = [(perm, signs) for perm in itertools.permutations(range(3))
                 for signs in [(1, s2, s3) for s2 in (1, -1) for s3 in (1, -1)]]


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def local_east_north(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """GPS lat/lon -> metres east/north of the drive's median position
    (flat-earth approximation; fine over a few km)."""
    lat, lon = df["gps_lat_deg"].to_numpy(), df["gps_lon_deg"].to_numpy()
    lat0 = np.nanmedian(lat)
    east = (lon - np.nanmedian(lon)) * np.cos(np.radians(lat0)) * M_PER_DEG_LAT
    north = (lat - lat0) * M_PER_DEG_LAT
    return east, north


def gps_fix_intervals(t: np.ndarray, fix: np.ndarray, speed: np.ndarray, course_deg: np.ndarray,
                      end: int, min_dt: float = 4.0, max_dt: float = 20.0, min_speed_ms: float = 3.0):
    """Moving GPS fix intervals before row `end`, thinned to >= min_dt apart.
    Returns (i0, i1, course change in radians) or (None, None, None)."""
    f = fix[fix < end]
    if len(f) < 2:
        return None, None, None
    keep = [f[0]]
    for i in f[1:]:
        if t[i] - t[keep[-1]] >= min_dt:
            keep.append(i)
    keep = np.array(keep)
    if len(keep) < 2:
        return None, None, None
    i0, i1 = keep[:-1], keep[1:]
    ok = (t[i1] - t[i0] <= max_dt) & (speed[i0] > min_speed_ms) & (speed[i1] > min_speed_ms)
    i0, i1 = i0[ok], i1[ok]
    return i0, i1, wrap_angle(np.radians(course_deg[i1] - course_deg[i0]))


@dataclass
class GyroHeadingFit:
    perm: tuple
    signs: tuple
    k: float
    r: float
    n_intervals: int


class GyroTurnIntegrator:
    """Cumulative gyro x gravity products for one drive, so the rotation over
    any row range, for any axis mapping and gyro bias, is O(1)."""

    def __init__(self, df: pd.DataFrame):
        t = df["t_sec"].to_numpy(dtype=float)
        dt = np.clip(np.diff(t, prepend=t[0]), 0.0, 1.0)
        g = df[GRAVITY_COLS].to_numpy(dtype=float)
        g = g / np.linalg.norm(g, axis=1, keepdims=True)
        w = df[GYRO_COLS].to_numpy(dtype=float)
        self.cum_wg = np.cumsum(w[:, :, None] * g[:, None, :] * dt[:, None, None], axis=0)
        self.cum_g = np.cumsum(g * dt[:, None], axis=0)

    def rotation(self, a, b, perm, signs, bias: np.ndarray):
        """Integral of the mapped, bias-corrected gyro . g_unit from row a to row b (radians)."""
        return sum(sg * ((self.cum_wg[b, pi, j] - self.cum_wg[a, pi, j]) - bias[pi] * (self.cum_g[b, j] - self.cum_g[a, j]))
                   for j, (pi, sg) in enumerate(zip(perm, signs)))


def fit_gyro_heading(integrator: GyroTurnIntegrator, gyro_bias: np.ndarray, i0, i1, dcourse,
                     min_intervals: int = 8, min_corr: float = 0.5) -> GyroHeadingFit | None:
    """Choose the gyro axis mapping and scale k that best explain the GPS
    course changes over the given (pre-outage) intervals."""
    if i0 is None or len(i0) < min_intervals:
        return None
    best = None
    for perm, signs in GYRO_MAPPINGS:
        x = integrator.rotation(i0, i1, perm, signs, gyro_bias)
        if np.std(x) < 1e-9:
            continue
        k = float(np.dot(x, dcourse) / np.dot(x, x))
        r = float(np.corrcoef(x, dcourse)[0, 1])
        if best is None or abs(r) > abs(best.r):
            best = GyroHeadingFit(perm, signs, k, r, len(i0))
    return best if best is not None and abs(best.r) >= min_corr else None


def gyro_heading(integrator: GyroTurnIntegrator, fit: GyroHeadingFit, gyro_bias: np.ndarray,
                 start: int, rows: np.ndarray, course0_rad: float) -> np.ndarray:
    """Heading (radians, clockwise from north) at `rows`, starting from course0 at row `start`."""
    return course0_rad + fit.k * integrator.rotation(start, rows, fit.perm, fit.signs, gyro_bias)


def end_position(speed: np.ndarray, heading_rad: np.ndarray, dt: np.ndarray) -> tuple[float, float]:
    """East/north displacement (m) from speed and heading per row; the first
    row is the starting point and contributes no distance."""
    d = np.maximum(speed, 0.0) * dt
    d[0] = 0.0
    return float(np.sum(d * np.sin(heading_rad))), float(np.sum(d * np.cos(heading_rad)))


def path_outage_starts(df: pd.DataFrame, outage_lengths_s, n_starts: int, v_true: np.ndarray, seed: int,
                       min_history_rows: int = 1200) -> np.ndarray:
    """Outage start rows used by the Days 10-12 (and Phase 6) benchmarks: GPS
    fixes while moving (> 3 m/s), >= `min_history_rows` into the drive, with a
    fix within 15 s after the longest outage, no time gap and finite truth;
    `n_starts` drawn with `seed`."""
    from src.io_vnbd_loader import _gps_fix_index

    t = df["t_sec"].to_numpy(dtype=float)
    held = df["gps_speed_ms"].to_numpy(dtype=float)
    dt_med = float(np.median(np.diff(t)))
    fix = _gps_fix_index(df)
    fixes_t = t[fix]
    L_max = max(outage_lengths_s)
    cand = []
    for s in fix:
        if s < min_history_rows or held[s] <= 3.0:
            continue
        k = np.searchsorted(fixes_t, t[s] + L_max)
        if k >= len(fix) or fixes_t[k] - t[s] > L_max + 15:
            continue
        e = fix[k]
        if t[e] - t[s] > (e - s) * dt_med * 1.2 or np.any(~np.isfinite(v_true[s:e + 1])):
            continue
        cand.append(s)
    if not cand:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(cand, size=min(n_starts, len(cand)), replace=False))


def simulate_path_outages(df: pd.DataFrame, outage_lengths_s=(30, 60, 120), n_starts: int = 15,
                          truth: np.ndarray | None = None, seed: int = 0, min_history_rows: int = 1200,
                          ml_methods=None, window_size: int = 200, map_match=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Day 10 benchmark on one drive (loaded + gravity-compensated, not calibrated).

    `map_match` (Day 11, optional) = `HMMMapMatcher.match_outage`; for every
    speed with the gyro heading it adds methods "<speed>+gyro+map_snap" and
    "+map_route" (plain dead reckoning where the map cannot be used,
    `map_ok` = False). Errors are measured in metres around the entrance fix.

    Starts are GPS fixes while moving, with >= `min_history_rows` before them
    (2 min at 10 Hz) and a fix after the longest outage. For each start:
    calibration and gyro mapping come from rows BEFORE it only.
    Returns (part_a, part_b): part A uses the TRUE speed (`truth`, default
    GPS interpolated) with headings hold/gyro; part B (only if `ml_methods`
    is given) uses hold-speed and anchored-ML speed with each heading.
    Heading "gyro" falls back to "hold" when the gyro cannot be calibrated
    (`gyro_ok` = False)."""
    from src.io_vnbd_loader import _gps_fix_index
    from src.models.multi_drive_data import calibrate, apply_calibration

    t = df["t_sec"].to_numpy(dtype=float)
    dt = np.clip(np.diff(t, prepend=t[0]), 0.0, 1.0)
    lat, lon =df["gps_lat_deg"].to_numpy(dtype=float), df["gps_lon_deg"].to_numpy(dtype=float)
    held = df["gps_speed_ms"].to_numpy(dtype=float)
    course = df["gps_orientation_deg"].to_numpy(dtype=float)
    v_true = df["gps_speed_interp_ms"].to_numpy() if truth is None else np.asarray(truth, dtype=float)
    fix = _gps_fix_index(df)
    fixes_t = t[fix]
    integrator = GyroTurnIntegrator(df)
    L_max = max(outage_lengths_s)
    starts = path_outage_starts(df, outage_lengths_s, n_starts, v_true, seed, min_history_rows)
    if len(starts) == 0:
        return pd.DataFrame(), pd.DataFrame()

    rows_a, rows_b = [], []
    for s in starts:
        calib = calibrate(df.iloc[:s], allow_leveling_only=True)
        bias = np.array([calib["bias"][c] for c in GYRO_COLS])
        i0, i1, dcourse = gps_fix_intervals(t, fix, held, course, s)
        fit = fit_gyro_heading(integrator, bias, i0, i1, dcourse)

        e_max = fix[np.searchsorted(fixes_t, t[s] + L_max)]
        rr = np.arange(s, e_max + 1)
        c0 = np.radians(course[s])
        headings = {"hold": np.full(len(rr), c0)}
        headings["gyro"] = gyro_heading(integrator, fit, bias, s, rr, c0) if fit is not None else headings["hold"]

        speeds_b = {}
        if ml_methods is not None:
            seg = apply_calibration(df.iloc[s - (window_size - 1): e_max + 1], calib).reset_index(drop=True)
            ml = ml_methods(seg, window_size - 1, e_max - s, float(held[s]), t[rr])
            speeds_b = {"hold_speed": np.full(len(rr), held[s]), "anchored_ml": ml["anchored_ml"]}

        ends = [fix[np.searchsorted(fixes_t, t[s] + L)] for L in outage_lengths_s]
        matched_a, matched_b = {}, {}
        if map_match is not None:
            matched_a = map_match(s, rr, v_true[rr], headings["gyro"], ends)
            matched_b = {sp: map_match(s, rr, v, headings["gyro"], ends) for sp, v in speeds_b.items()}

        for L, e in zip(outage_lengths_s, ends):
            m = rr <= e
            true_dist = float(np.sum(v_true[s + 1: e + 1] * dt[s + 1: e + 1]))
            ex = (lon[e] - lon[s]) * np.cos(np.radians(lat[s])) * M_PER_DEG_LAT
            ey = (lat[e] - lat[s]) * M_PER_DEG_LAT
            base = {"start_idx": int(s), "history_min": t[s] / 60, "outage_s": L, "true_dist_m": true_dist,
                    "gyro_ok": fit is not None, "gyro_r": abs(fit.r) if fit is not None else np.nan,
                    "gyro_standard_axes": fit.perm == (0, 1, 2) if fit is not None else None}

            def _dr(v, hdg):
                return end_position(v[m], hdg[m], dt[rr[m]])

            def _row(method, dxy, map_ok=None):
                err = float(np.hypot(dxy[0] - ex, dxy[1] - ey))
                return {**base, "method": method, "abs_err_m": err, "pct_err": 100 * err / max(true_dist, 1.0),
                        "map_ok": map_ok}

            def _map_rows(prefix, matched, dr_xy):
                res = matched.get(int(e))
                for kind in ("snap", "route", "turn"):
                    yield _row(f"{prefix}+map_{kind}", res[kind] if res else dr_xy, map_ok=res is not None)

            for h, hdg in headings.items():
                rows_a.append(_row(h, _dr(v_true[rr], hdg)))
            if map_match is not None:
                rows_a.extend(_map_rows("gyro", matched_a, _dr(v_true[rr], headings["gyro"])))
            for sp, v in speeds_b.items():
                for h, hdg in headings.items():
                    rows_b.append(_row(f"{sp}+{h}", _dr(v, hdg)))
                if map_match is not None:
                    rows_b.extend(_map_rows(f"{sp}+gyro", matched_b[sp], _dr(v, headings["gyro"])))
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


DAY10_LABELS_A = {"hold": "True distance + last GPS course (straight line)",
                  "gyro": "True distance + calibrated gyro heading"}
DAY10_LABELS_B = {"hold_speed+hold": "Hold last fix + last course (baseline)",
                  "anchored_ml+hold": "Anchored ML + last course",
                  "hold_speed+gyro": "Hold last fix + gyro heading",
                  "anchored_ml+gyro": "Anchored ML + gyro heading (full method)"}
DAY10_SECTION_A = "Day 10 — heading only: end-position error with TRUE distance (all drives)"
DAY10_SECTION_B = "Day 10 — full 2-D path during simulated GPS outages (TEST drives only)"


if __name__ == "__main__":
    import json
    import warnings
    import zlib

    import torch

    root = Path(__file__).resolve()
    while not ((root / "src").is_dir() and (root / "data").is_dir()):
        root = root.parent
    sys.path.insert(0, str(root))
    from src.models.multi_drive_data import load_raw_drive
    from src.models.train_velocity_model import load_model
    from src.dead_reckoning.strapdown_ins import load_truth, log_metric
    from src.dead_reckoning.ins_with_ml_velocity import make_ml_methods

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(root / "results" / "checkpoints" / "velocity_cnn_best.pt", device)
    ml_methods = make_ml_methods(model, ckpt, device)
    test_drives = set(json.loads((root / "data" / "split.json").read_text(encoding="utf-8"))["test"])

    print(f"\n{'=' * 70}\nDay 10 — heading and 2-D path during simulated GPS outages\n"
          f"(starts at a GPS fix; calibration + gyro mapping from data before each outage;\n"
          f" error = distance to the first GPS fix after the outage)\n{'=' * 70}")
    parts_a, parts_b = [], []
    for path in sorted((root / "data" / "raw" / "real").glob("S-*.csv")):
        try:
            df = load_raw_drive(path)
        except Exception:  # noqa: BLE001
            continue
        if not 0.07 <= float(np.median(np.diff(df["t_sec"]))) <= 0.13:
            continue
        truth, source = load_truth(df, path.name, root)
        a, b = simulate_path_outages(df, truth=truth, seed=zlib.crc32(path.name.encode()),
                                     ml_methods=ml_methods if path.name in test_drives else None,
                                     window_size=ckpt["window_size"])
        if a.empty:
            continue
        a["drive"] = path.name
        parts_a.append(a)
        if not b.empty:
            b["drive"] = path.name
            parts_b.append(b)
        print(f"  {path.name:<12} {a.start_idx.nunique():>3} outages  truth={source}"
              f"{'  (+ full path, test drive)' if not b.empty else ''}")

    A = pd.concat(parts_a, ignore_index=True)
    B = pd.concat(parts_b, ignore_index=True) if parts_b else pd.DataFrame()
    A.to_csv(root / "results" / "day10_heading_benchmark.csv", index=False)
    if not B.empty:
        B.to_csv(root / "results" / "day10_path_benchmark.csv", index=False)

    starts = A.drop_duplicates(["drive", "start_idx"])
    print(f"\nGyro calibrated from pre-outage GPS (|r| >= 0.5): {100 * starts.gyro_ok.mean():.0f}% of outages, "
          f"median |r| {starts.gyro_r.median():.2f}; standard axis order was the best fit in "
          f"{100 * starts.gyro_standard_axes.dropna().astype(bool).mean():.0f}% of them")

    for res, labels, title, section in ((A, DAY10_LABELS_A, f"PART A — heading only ({A.drive.nunique()} drives)", DAY10_SECTION_A),
                                        (B, DAY10_LABELS_B, f"PART B — full path, TEST drives ({B.drive.nunique() if not B.empty else 0} drives)", DAY10_SECTION_B)):
        if res.empty:
            continue
        print(f"\n{title}")
        print(f"{'Method':<48} {'Outage':>7} {'Median err (m)':>15} {'Median %':>9} {'N':>6}")
        print("-" * 90)
        for L in sorted(res.outage_s.unique()):
            for method, label in labels.items():
                sub = res[(res.outage_s == L) & (res.method == method)]
                if sub.empty:
                    continue
                m = {"outage_s": L, "median_abs_err_m": sub.abs_err_m.median(), "median_pct_err": sub.pct_err.median(),
                     "mean_abs_err_m": sub.abs_err_m.mean(), "n_outages": len(sub), "n_drives": sub.drive.nunique()}
                print(f"{label:<48} {L:>6}s {m['median_abs_err_m']:>15.1f} {m['median_pct_err']:>8.1f}% {len(sub):>6}")
                log_metric(root / "results", f"{label}, {L}s outage", m, day="Day 10", section_title=section)
            print()
    print("Summary rows written to results/metrics.md (re-running replaces them).")
