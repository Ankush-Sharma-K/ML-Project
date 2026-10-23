"""
strapdown_ins.py — Phase 4: Strapdown Inertial Navigation System (INS)
=======================================================================

Implements a strap-down INS that integrates calibrated IMU data to
produce velocity and position estimates, with two key ZUPT (Zero-velocity
Update) correction strategies:

  1. Hard-reset ZUPT: when a stationary window is detected, set velocity
     to zero and record a position anchor. This completely eliminates
     velocity drift at every stop — the dominant error source in urban driving.

  2. Soft-reset ZUPT (optional): apply a gentle velocity decay factor during
     stationary windows instead of snapping to zero. More numerically stable
     for datasets where stationary detection is noisy.

The INS operates on **vehicle-frame** linear accelerations
(``lin_accel_{x,y,z}_mps2_cal_veh``) produced by the Phase 2 calibration
pipeline and expects ``t_sec`` to be a monotonically increasing time axis.

Coordinate convention (vehicle frame, right-hand):
  +x  →  forward  (vehicle heading direction)
  +y  →  left
  +z  →  up

Only the forward component (x) is integrated for the 1-D velocity model
used in the dead-reckoning tunnel bridge. Full 3-D integration is available
via ``StrapdownINS.integrate_3d()`` for later phases.

Day 8 deliverable
-----------------
  * ``StrapdownINS`` class: integrate, apply ZUPT, return trajectory.
  * ``evaluate_ins_against_gps()``: scalar drift metric at drive end.
  * ``simulate_outages()``: the actual benchmark (see below).
  * ``__main__`` block: outage benchmark over every usable real drive.

Why an OUTAGE benchmark, not one whole-drive integration (Day 8 audit):
the first version integrated S-S1 open-loop for all 86 minutes and
reported 47,881 m (hard ZUPT) / 1,881,072 m (no ZUPT). Those numbers
were dominated by three upstream bugs (GPS speed is m/s not km/h; the
heading estimate was wrong; synthetic-tuned ZUPT caught 13% of stops),
and even with them fixed an 86-minute open-loop run says nothing about
a tunnel. The tunnel problem is: GPS drops out for 30 s - 2 min, the
system starts from the last GPS speed, and must keep track of distance
until GPS returns. ``simulate_outages()`` does exactly that at random
moving start points and compares against the simplest possible
competitor -- holding the last GPS speed -- which every later phase
(Day 9 ML velocity, Day 13 fusion) must beat.

Honest-information rules (second Day 8 audit): each outage is calibrated
ONLY on the data recorded before it (whole-drive calibration had let the
INS benefit from GPS inside the "tunnel" -- e.g. 554 m vs 708 m median
error at 120 s), and the entry speed is the last real GPS fix, not the
interpolated speed (which peeks at the next fix). The reference speed is
the car's wheel speed where a vehicle file is available
(`utils.vehicle_truth`), otherwise GPS interpolated between fixes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project-root discovery (same pattern used across all modules)
# ---------------------------------------------------------------------------

def _find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* (or this file's dir) until we find a directory
    that contains both ``src/`` and ``data/``. Raises RuntimeError if not found."""
    candidate = (start or Path(__file__).resolve()).parent
    for _ in range(10):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
        candidate = candidate.parent
    raise RuntimeError(
        "Could not find project root (directory containing both src/ and data/). "
        f"Started search from: {start or Path(__file__).resolve().parent}"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Acceleration column after full calibration (gravity comp + bias + mount align)
ACCEL_VEH_X = "lin_accel_x_mps2_cal_veh"  # forward
ACCEL_VEH_Y = "lin_accel_y_mps2_cal_veh"  # left
ACCEL_VEH_Z = "lin_accel_z_mps2_cal_veh"  # up

TIME_COL = "t_sec"
GPS_SPEED_COL = "gps_speed_kmh"


# ---------------------------------------------------------------------------
# Core INS class
# ---------------------------------------------------------------------------

class StrapdownINS:
    """Strap-down INS that integrates vehicle-frame accelerations.

    Parameters
    ----------
    df : pd.DataFrame
        Calibrated drive DataFrame (output of ``load_and_calibrate_drive``).
        Must contain ``t_sec``, ``lin_accel_{x,y,z}_mps2_cal_veh``, and
        ``gps_speed_kmh``.
    zupt_mask : np.ndarray | None
        Boolean array (same length as df). True → stationary sample.
        If None, ZUPT corrections are disabled.
    zupt_mode : {"hard", "soft"}
        "hard" → snap velocity to zero at every stationary window start.
        "soft" → exponentially decay velocity toward zero during stationary
                 windows (decay_rate parameter controls speed).
    decay_rate : float
        Used only when zupt_mode="soft". Velocity multiplied by this factor
        each sample during stationary windows. 0.0 = immediate zero (same as
        hard), 1.0 = no decay. Recommended range: 0.8–0.95.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        zupt_mask: np.ndarray | None = None,
        zupt_mode: str = "hard",
        decay_rate: float = 0.9,
    ) -> None:
        if zupt_mode not in ("hard", "soft"):
            raise ValueError(f"zupt_mode must be 'hard' or 'soft', got {zupt_mode!r}")

        self.df = df.reset_index(drop=True)
        self.zupt_mask = zupt_mask
        self.zupt_mode = zupt_mode
        self.decay_rate = float(decay_rate)

        # Results populated by integrate()
        self.t: np.ndarray | None = None
        self.vel_x: np.ndarray | None = None   # forward velocity m/s
        self.vel_y: np.ndarray | None = None   # lateral velocity m/s
        self.vel_z: np.ndarray | None = None   # vertical velocity m/s
        self.pos_x: np.ndarray | None = None   # forward displacement m
        self.pos_y: np.ndarray | None = None
        self.pos_z: np.ndarray | None = None
        self.speed: np.ndarray | None = None   # |vel| in 3-D m/s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def integrate(self, initial_velocity: float = 0.0, no_reverse: bool = False) -> "StrapdownINS":
        """Run 1-D forward integration (x-axis = forward) with optional ZUPT.

        initial_velocity: forward speed (m/s) at the first sample -- the
            last GPS speed when bridging an outage.
        no_reverse: clamp forward velocity at >= 0 (a car on a tunnel road
            does not drive backwards; non-holonomic constraint along x).

        Populates ``vel_x``, ``pos_x``, ``t``.
        Returns self (chainable).
        """
        t, ax = self._extract_1d()
        dt = self._safe_dt(t)

        vel_x = np.zeros(len(t))
        pos_x = np.zeros(len(t))
        vel_x[0] = initial_velocity

        mask = self.zupt_mask if self.zupt_mask is not None else np.zeros(len(t), dtype=bool)

        for i in range(1, len(t)):
            # Integrate acceleration → velocity
            vel_x[i] = vel_x[i - 1] + ax[i - 1] * dt[i]

            # Apply ZUPT correction
            if mask[i]:
                if self.zupt_mode == "hard":
                    vel_x[i] = 0.0
                else:  # soft
                    vel_x[i] *= self.decay_rate
            if no_reverse and vel_x[i] < 0.0:
                vel_x[i] = 0.0

            # Integrate velocity → position
            pos_x[i] = pos_x[i - 1] + vel_x[i] * dt[i]

        self.t = t
        self.vel_x = vel_x
        self.pos_x = pos_x
        return self

    def integrate_3d(self) -> "StrapdownINS":
        """Full 3-D integration (x=fwd, y=left, z=up) with optional ZUPT
        applied to all three velocity components simultaneously.

        Populates ``vel_{x,y,z}``, ``pos_{x,y,z}``, ``speed``, ``t``.
        Returns self.
        """
        t = self.df[TIME_COL].to_numpy(dtype=float)
        dt = self._safe_dt(t)

        ax = self._col_or_zeros(ACCEL_VEH_X)
        ay = self._col_or_zeros(ACCEL_VEH_Y)
        az = self._col_or_zeros(ACCEL_VEH_Z)

        vx = np.zeros(len(t))
        vy = np.zeros(len(t))
        vz = np.zeros(len(t))
        px = np.zeros(len(t))
        py = np.zeros(len(t))
        pz = np.zeros(len(t))

        mask = self.zupt_mask if self.zupt_mask is not None else np.zeros(len(t), dtype=bool)

        for i in range(1, len(t)):
            vx[i] = vx[i - 1] + ax[i - 1] * dt[i]
            vy[i] = vy[i - 1] + ay[i - 1] * dt[i]
            vz[i] = vz[i - 1] + az[i - 1] * dt[i]

            if mask[i]:
                if self.zupt_mode == "hard":
                    vx[i] = 0.0
                    vy[i] = 0.0
                    vz[i] = 0.0
                else:
                    vx[i] *= self.decay_rate
                    vy[i] *= self.decay_rate
                    vz[i] *= self.decay_rate

            px[i] = px[i - 1] + vx[i] * dt[i]
            py[i] = py[i - 1] + vy[i] * dt[i]
            pz[i] = pz[i - 1] + vz[i] * dt[i]

        self.t = t
        self.vel_x, self.vel_y, self.vel_z = vx, vy, vz
        self.pos_x, self.pos_y, self.pos_z = px, py, pz
        self.speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        return self

    def result_df(self) -> pd.DataFrame:
        """Return a DataFrame with time, velocity and position columns."""
        if self.t is None:
            raise RuntimeError("Call integrate() or integrate_3d() first.")
        data = {"t_sec": self.t}
        for name, arr in [
            ("vel_x_mps", self.vel_x),
            ("vel_y_mps", self.vel_y),
            ("vel_z_mps", self.vel_z),
            ("pos_x_m", self.pos_x),
            ("pos_y_m", self.pos_y),
            ("pos_z_m", self.pos_z),
            ("speed_mps", self.speed),
        ]:
            if arr is not None:
                data[name] = arr
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_1d(self):
        t = self.df[TIME_COL].to_numpy(dtype=float)
        ax = self._col_or_zeros(ACCEL_VEH_X)
        return t, ax

    def _col_or_zeros(self, col: str) -> np.ndarray:
        if col in self.df.columns:
            return self.df[col].to_numpy(dtype=float)
        return np.zeros(len(self.df))

    @staticmethod
    def _safe_dt(t: np.ndarray) -> np.ndarray:
        """Forward-difference dt with zero-gap guard. dt[0] = 0 (correct)."""
        dt = np.diff(t, prepend=t[0])
        # Guard against duplicate timestamps (dt=0) or backwards resets (dt<0).
        # A zero dt contributes 0 to cumulative sum → safe. A negative dt from
        # a timestamp reset would integrate backwards → clip to 0.
        dt = np.where(dt < 0, 0.0, dt)
        return dt


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def gps_speed_ms(df: pd.DataFrame) -> np.ndarray:
    """GPS speed in m/s, interpolated between fixes when the loader
    provides it (real GPS only updates every ~10 s and is held between)."""
    if "gps_speed_interp_ms" in df.columns:
        return df["gps_speed_interp_ms"].to_numpy(dtype=float)
    return (df[GPS_SPEED_COL] / 3.6).to_numpy(dtype=float)


def gps_reference_distance(df: pd.DataFrame) -> np.ndarray:
    """Cumulative distance from GPS speed (ground-truth reference)."""
    t = df[TIME_COL].to_numpy(dtype=float)
    dt = StrapdownINS._safe_dt(t)
    return np.cumsum(gps_speed_ms(df) * dt)


def evaluate_ins_against_gps(
    ins: StrapdownINS,
    df: pd.DataFrame,
) -> dict:
    """Compute drift metrics comparing INS forward distance to GPS reference.

    Returns
    -------
    dict with keys:
      "final_ins_m"       : INS total forward distance at drive end
      "final_gps_m"       : GPS reference distance at drive end
      "final_error_m"     : |ins - gps| at drive end
      "max_abs_error_m"   : peak |ins - gps| over entire drive
      "rmse_m"            : RMS error over entire drive
    """
    if ins.pos_x is None:
        raise RuntimeError("Call ins.integrate() before evaluate_ins_against_gps().")

    gps_dist = gps_reference_distance(df)
    ins_dist = ins.pos_x

    n = min(len(gps_dist), len(ins_dist))
    err = ins_dist[:n] - gps_dist[:n]

    return {
        "final_ins_m": float(ins_dist[n - 1]),
        "final_gps_m": float(gps_dist[n - 1]),
        "final_error_m": float(abs(ins_dist[n - 1] - gps_dist[n - 1])),
        "max_abs_error_m": float(np.max(np.abs(err))),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
    }


def compare_zupt_modes(df: pd.DataFrame, zupt_mask: np.ndarray) -> dict:
    """Run three INS variants and compare their drift metrics.

    Variants: (1) no ZUPT, (2) hard ZUPT, (3) soft ZUPT.
    Returns a dict keyed by variant name, each with evaluate_ins_against_gps output.
    """
    results = {}
    for label, mask, mode, rate in [
        ("no_zupt",   None,       "hard", 0.9),
        ("hard_zupt", zupt_mask,  "hard", 0.9),
        ("soft_zupt", zupt_mask,  "soft", 0.9),
    ]:
        ins = StrapdownINS(df, zupt_mask=mask, zupt_mode=mode, decay_rate=rate)
        ins.integrate()
        results[label] = evaluate_ins_against_gps(ins, df)
    return results


def simulate_outages(
    df: pd.DataFrame,
    outage_lengths_s: tuple = (30, 60, 120),
    n_starts: int = 15,
    truth: np.ndarray | None = None,
    min_start_speed_ms: float = 3.0,
    seed: int = 0,
    history_rows: int = 0,
    extra_methods=None,
) -> pd.DataFrame:
    """Simulated GPS outages ("tunnels") at random moving start points,
    using ONLY information a phone would have at the moment GPS is lost.

    df: one drive, loaded + gravity-compensated but NOT calibrated
        (`multi_drive_data.load_raw_drive`).
    truth: per-row reference speed in m/s (e.g. wheel speed from
        `utils.vehicle_truth`); default = GPS speed interpolated between fixes.

    For every start point s:
      * calibration = `calibrate(df.iloc[:s])` -- everything recorded BEFORE
        the outage, nothing from inside or after it;
      * entry speed v0 = the last real GPS fix at or before s (`gps_speed_ms`,
        held between fixes). The interpolated speed is NOT used for v0: it
        draws a line towards the NEXT fix, which arrives inside the outage,
        and it made holding the last speed look ~15-25% better than it is;
      * ZUPT thresholds come from the stops and moving samples before s.
    Methods (speed per sample over the outage, then distance):
      hold_speed : v0, constant (no IMU)
      ins        : StrapdownINS from v0, no ZUPT, no reversing
      ins_zupt   : same + hard ZUPT (IMU-only adaptive detector)
      + whatever `extra_methods(seg, s_local, n_max, v0, t_seg)` returns as
        {name: speed array of length n_max + 1}; `seg` is the calibrated slice
        starting `history_rows` rows before the outage (Day 9 ML methods).
    The same start points are used for every outage length. Returns one row
    per (start, outage length, method) with the error at the end of the outage.
    """
    from src.models.multi_drive_data import calibrate, apply_calibration
    from src.calibration.zupt_classifier import zupt_mask_adaptive

    t = df[TIME_COL].to_numpy(dtype=float)
    held = df["gps_speed_ms"].to_numpy(dtype=float)
    ref_speed = gps_speed_ms(df) if truth is None else np.asarray(truth, dtype=float)
    dt_med = float(np.median(np.diff(t)))
    n_max = int(round(max(outage_lengths_s) / dt_med))
    first = max(history_rows, 50)
    if len(df) <= first + n_max + 1:
        return pd.DataFrame()

    cand = np.arange(first, len(df) - n_max - 1)
    cand = cand[ref_speed[cand] > min_start_speed_ms]
    cand = cand[t[cand + n_max] - t[cand] < max(outage_lengths_s) * 1.2]   # no logging dropout inside
    finite = np.r_[0, np.cumsum(~np.isfinite(ref_speed))]
    cand = cand[finite[cand + n_max + 1] - finite[cand] == 0]               # truth covers the outage
    if len(cand) == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(cand, size=min(n_starts, len(cand)), replace=False))

    rows = []
    for s in starts:
        calib = calibrate(df.iloc[:s], allow_leveling_only=True)
        lo = s - history_rows
        seg = apply_calibration(df.iloc[lo : s + n_max + 1], calib).reset_index(drop=True)
        s_local = s - lo
        out = seg.iloc[s_local:].reset_index(drop=True)
        v0 = float(held[s])

        before = np.zeros(len(df), dtype=bool)
        before[:s] = True
        ref_mask = np.zeros(len(df), dtype=bool)
        ref_mask[:s] = calib["stationary_mask"]
        zmask, _ = zupt_mask_adaptive(df, ref_mask, moving_mask=before & (held > 5.0))

        speeds = {
            "hold_speed": np.full(n_max + 1, v0),
            "ins": StrapdownINS(out).integrate(initial_velocity=v0, no_reverse=True).vel_x,
            "ins_zupt": StrapdownINS(out, zupt_mask=zmask[s : s + n_max + 1])
                        .integrate(initial_velocity=v0, no_reverse=True).vel_x,
        }
        if extra_methods is not None:
            speeds.update(extra_methods(seg, s_local, n_max, v0, t[s : s + n_max + 1]))

        dt = StrapdownINS._safe_dt(t[s : s + n_max + 1])
        for L in outage_lengths_s:
            n = int(round(L / dt_med))
            true_m = float(np.sum(ref_speed[s + 1 : s + n + 1] * dt[1 : n + 1]))
            if true_m < 1.0:
                continue
            for method, v in speeds.items():
                est = float(np.sum(np.maximum(v[1 : n + 1], 0.0) * dt[1 : n + 1]))
                rows.append({
                    "start_idx": int(s), "history_min": t[s] / 60, "outage_s": L, "method": method,
                    "heading_method": calib["heading_method"], "true_m": true_m, "est_m": est,
                    "abs_err_m": abs(est - true_m), "pct_err": 100 * abs(est - true_m) / true_m,
                })
    return pd.DataFrame(rows)


def summarize_outages(res: pd.DataFrame) -> pd.DataFrame:
    """Median / mean error per (outage length, method)."""
    return res.groupby(["outage_s", "method"]).agg(
        median_abs_err_m=("abs_err_m", "median"),
        median_pct_err=("pct_err", "median"),
        mean_abs_err_m=("abs_err_m", "mean"),
        n_outages=("abs_err_m", "size"),
        n_drives=("drive", "nunique"),
    ).reset_index()


def load_truth(df: pd.DataFrame, drive_name: str, root: Path) -> tuple[np.ndarray | None, str]:
    """Best available reference speed: wheel speed if the drive has a clearly
    aligned vehicle file, else None (= GPS interpolated). Returns (truth, source)."""
    from src.utils.vehicle_truth import load_wheel_speed_truth

    truth, info = load_wheel_speed_truth(df, drive_name, root / "data" / "raw" / "vehicle")
    return (truth, "wheel") if truth is not None else (None, "gps")


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

TABLE_HEADER = (
    "| Day | Method | Outage (s) | Median abs error (m) | Median error (% of distance) "
    "| Mean abs error (m) | Outages | Drives |\n"
)


def log_metric(results_path: Path, label: str, metrics: dict, day: str = "Day 8",
               section_title: str = "Day 8 — simulated GPS-outage benchmark (real IO-VNBD drives)") -> None:
    """Write one outage-benchmark row to results/metrics.md, REPLACING any
    existing row with the same day + label (re-running used to append
    duplicates). Each day gets its own section/table."""
    results_path.mkdir(parents=True, exist_ok=True)
    md = results_path / "metrics.md"
    prefix = f"| {day} | {label} |"
    row = (
        f"{prefix} {metrics['outage_s']:.0f} | {metrics['median_abs_err_m']:.1f} | "
        f"{metrics['median_pct_err']:.1f} | {metrics['mean_abs_err_m']:.1f} | "
        f"{int(metrics['n_outages'])} | {int(metrics['n_drives'])} |\n"
    )
    lines = md.read_text(encoding="utf-8").splitlines(keepends=True) if md.exists() else []
    lines = [ln for ln in lines if not ln.startswith(prefix)]
    heading = f"## {section_title}\n"
    if heading not in lines:
        lines += ["\n", heading, "\n", TABLE_HEADER, "|---|---|---|---|---|---|---|---|\n"]
    header_at = lines.index(TABLE_HEADER, lines.index(heading))
    insert_at = header_at + 2
    while insert_at < len(lines) and lines[insert_at].startswith(f"| {day} |"):
        insert_at += 1
    lines.insert(insert_at, row)
    md.write_text("".join(lines), encoding="utf-8")


def report(res: pd.DataFrame, labels: dict, day: str, section_title: str, root: Path,
           log: bool = True) -> None:
    """Print the outage table (overall, wheel-truth subset, by history) and
    log the overall rows to results/metrics.md."""
    def _print(sub: pd.DataFrame, title: str) -> None:
        if sub.empty:
            return
        print(f"\n{title}")
        print(f"{'Method':<40} {'Outage':>7} {'Median err (m)':>15} {'Median %':>9} {'Mean err (m)':>13} {'N':>6}")
        print("-" * 95)
        summ = summarize_outages(sub)
        for L in sorted(sub["outage_s"].unique()):
            for method, label in labels.items():
                m = summ[(summ.outage_s == L) & (summ.method == method)]
                if m.empty:
                    continue
                m = m.iloc[0]
                print(f"{label:<40} {L:>6}s {m.median_abs_err_m:>15.1f} {m.median_pct_err:>8.1f}% "
                      f"{m.mean_abs_err_m:>13.1f} {int(m.n_outages):>6}")
            print()

    _print(res, f"ALL ({res['drive'].nunique()} drives; truth = wheel speed where available, else GPS)")
    _print(res[res.truth == "wheel"], "WHEEL-SPEED TRUTH ONLY")
    _print(res[res.history_min < 20], "OUTAGES WITH < 20 MIN OF DRIVING BEFORE THEM")
    _print(res[res.history_min >= 20], "OUTAGES WITH >= 20 MIN OF DRIVING BEFORE THEM")
    print(f"Heading could not be resolved before the outage (leveling only): "
          f"{100 * (res.drop_duplicates(['drive', 'start_idx']).heading_method == 'none').mean():.1f}% of outages")

    if log:
        summ = summarize_outages(res).set_index(["outage_s", "method"])
        for L in sorted(res["outage_s"].unique()):
            for method, label in labels.items():
                if (L, method) in summ.index:
                    m = {"outage_s": L, **summ.loc[(L, method)].to_dict()}
                    log_metric(root / "results", f"{label}, {int(L)}s outage", m,
                               day=day, section_title=section_title)


# ---------------------------------------------------------------------------
# __main__ — Day 8 outage benchmark over all usable real drives
# ---------------------------------------------------------------------------

OUTAGE_LENGTHS_S = (30, 60, 120)
OUTAGES_PER_DRIVE = 15
DAY8_LABELS = {"hold_speed": "Hold last GPS fix (baseline)",
               "ins": "Strapdown INS, no ZUPT",
               "ins_zupt": "Strapdown INS + adaptive ZUPT"}
DAY8_SECTION = "Day 8 — simulated GPS-outage benchmark (real IO-VNBD drives)"

if __name__ == "__main__":
    import warnings
    import zlib

    root = _find_project_root()
    sys.path.insert(0, str(root))
    from src.models.multi_drive_data import load_raw_drive

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    drive_paths = sorted((root / "data" / "raw" / "real").glob("S-*.csv"))
    if not drive_paths:
        print(f"No real drives found in {root / 'data' / 'raw' / 'real'}")
        sys.exit(1)

    print(f"\n{'=' * 70}\nDay 8 — strapdown INS, simulated GPS-outage benchmark\n"
          f"(calibration from data BEFORE each outage; entry speed = last real GPS fix)\n{'=' * 70}")
    all_rows, skipped = [], {}
    for path in drive_paths:
        try:
            df = load_raw_drive(path)
        except Exception as e:  # noqa: BLE001 -- one bad file must not stop the benchmark
            skipped[path.name] = f"{type(e).__name__}: {e}"
            continue
        if not 0.07 <= float(np.median(np.diff(df[TIME_COL]))) <= 0.13:
            skipped[path.name] = "sample rate not ~10 Hz"
            continue
        truth, source = load_truth(df, path.name, root)
        r = simulate_outages(df, OUTAGE_LENGTHS_S, n_starts=OUTAGES_PER_DRIVE, truth=truth,
                             seed=zlib.crc32(path.name.encode()))
        if r.empty:
            skipped[path.name] = "too short / no moving start point"
            continue
        r["drive"], r["truth"] = path.name, source
        all_rows.append(r)
        print(f"  {path.name:<12} {r.start_idx.nunique():>3} outages  truth={source}")

    res = pd.concat(all_rows, ignore_index=True)
    out_csv = root / "results" / "day8_outage_benchmark.csv"
    res.to_csv(out_csv, index=False)
    report(res, DAY8_LABELS, "Day 8", DAY8_SECTION, root)

    if skipped:
        print("\nSkipped drives:")
        for name, why in skipped.items():
            print(f"  {name}: {why}")
    print(f"\nPer-outage results: {out_csv}")
    print("Summary rows written to results/metrics.md (re-running replaces them).")
