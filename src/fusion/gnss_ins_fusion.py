"""
gnss_ins_fusion.py — Phase 6 (Days 13-14): GNSS + INS fusion with an EKF
=========================================================================

One extended Kalman filter runs the whole navigation at 10 Hz, before,
during and after a GPS outage:

  state      x = [east, north, heading, speed, ml_bias, gyro_bias]
             (metres around the outage entrance, heading clockwise from
             north, m/s, m/s, rad/s)
  predict    position += speed * (sin, cos)(heading) * dt
             heading  += (gyro turn rate - gyro_bias) * dt
             speed: random walk (acceleration noise)
             ml_bias: first-order Gauss-Markov, correlation time 20 s
             gyro_bias: slow random walk
  updates    ML speed (Day 9 model, once per second):  z = speed + ml_bias
             GPS fix: position (noise from the phone's `gps_accuracy_m` --
             adaptive), speed, course (only when moving); position and course
             are chi-square gated, so junk / multipath fixes are rejected

The gyro turn rate uses the per-phone axis mapping and scale fitted on GPS
course changes before the outage (`heading.fit_gyro_heading`, Day 10); when it
cannot be calibrated the heading gets a large process noise instead.

WHY THE ML BIAS DECAYS (Day 13 study, VAL drives only): the model's speed
error over the 5 minutes before an outage barely predicts its error during
the outage (correlation 0.19); correcting the outage with that bias did not
help at 120 s (142 -> 143 m median distance error) and badly hurt some drives
(S-Vw16a 75 -> 425 m). So the filter does NOT carry a long-term bias into the
outage: the bias state decays with the same 20 s time constant as Day 9's
anchored ML -- the filter's entrance correction, fading, is the anchored ML
rule, now inside a filter that also sees GPS speed and course.

GATE RECOVERY. GPS fixes come only every ~9 s; in a sharp turn the gyro
heading can be off by 20-30 deg by the next fix, so a strict gate rejects a
genuine fix and the filter never recovers. Two remedies: heading process noise
grows with the turn rate (`gyro_scale_sigma`), and a fix rejected right after
another rejected fix is accepted with its noise inflated by NIS / threshold.

TUNING (VAL drives only, 6 configs): gyro_scale_sigma 0.2 -> 0.4 and GPS course
sigma 8 -> 4 deg gave the best real-time end error (120 s: 449 -> 395 m) with the
same robustness; the other settings were left at their a-priori values.

SEAMLESS EXIT. When GPS returns, the exit fix updates the filter and an RTS
smoother (`kalman_filter.rts_smooth`) re-estimates every state inside the
outage -- the logged tunnel track is corrected, not just the current position.

`__main__` = the Phase 6 benchmark on the Days 10-12 outages (TEST and
TEST + VAL drives, where the ML model is out of sample):
  1. real-time end error: EKF vs Day 10 (anchored ML + gyro), each with and
     without map matching (Day 12)
  2. offline tunnel track: error at the hidden GPS fixes INSIDE the outage,
     after GPS returns: real-time EKF vs RTS-smoothed EKF vs straight line
     between entry and exit fix vs dead reckoning with its end error spread
     linearly over time
  3. robustness: 15% of the GPS fixes before the outage replaced by
     multipath-like jumps (200 m, reported accuracy unchanged): position
     error at the entrance with and without the chi-square gate
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.kalman_filter import CHI2_999, ekf_update, rts_smooth, wrap_angle

E, N, PSI, V, BV, BW = range(6)


@dataclass
class FusionConfig:
    accel_sigma: float = 1.5            # m/s^2, speed random walk
    heading_rate_sigma_deg: float = 1.0  # deg/s, heading process noise with a calibrated gyro
    gyro_scale_sigma: float = 0.4        # + this fraction of |turn rate| (the fitted scale k is uncertain)
    heading_rate_sigma_nogyro_deg: float = 10.0
    ml_bias_sigma: float = 3.0           # m/s, stationary std of the ML bias
    ml_bias_tau_s: float = 20.0          # = Day 9 FADE_S
    gyro_bias_rw_deg: float = 0.02       # deg/s per sqrt(s)
    ml_sigma: float = 1.0                # m/s, white part of the ML speed error
    gps_speed_sigma: float = 0.5         # m/s
    gps_course_sigma_deg: float = 4.0
    gps_pos_floor_m: float = 3.0
    min_course_speed_ms: float = 3.0
    gate: bool = True


def gyro_turn_rate(integrator, fit, gyro_bias: np.ndarray, rows: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calibrated turn rate (rad/s, clockwise) at `rows` from the Day 10 gyro
    mapping; zeros when the gyro could not be calibrated (`fit` None)."""
    if fit is None:
        return np.zeros(len(rows))
    rot = fit.k * integrator.rotation(rows[0], rows, fit.perm, fit.signs, gyro_bias)
    dt = np.clip(np.diff(t[rows], prepend=t[rows[0]]), 1e-3, 1.0)
    return np.r_[0.0, np.diff(rot)] / dt


class VehicleEKF:
    """The Phase 6 vehicle EKF as a step-by-step object (used by the batch
    benchmark through `run_ekf` and by the streaming edge engine):
    `predict(dt, omega)`, then any of `update_ml`, `update_gps`, then
    `finish_step()` (speed >= 0)."""

    def __init__(self, x0: np.ndarray, P0: np.ndarray, cfg: FusionConfig, gyro_ok: bool):
        self.x, self.P = x0.astype(float).copy(), P0.astype(float).copy()
        self.cfg, self.gyro_ok = cfg, gyro_ok
        self.q_psi = np.radians(cfg.heading_rate_sigma_deg if gyro_ok else cfg.heading_rate_sigma_nogyro_deg) ** 2
        self.q_bw = np.radians(cfg.gyro_bias_rw_deg) ** 2
        self.H_v = np.zeros((1, 6)); self.H_v[0, V] = 1.0
        self.H_ml = self.H_v.copy(); self.H_ml[0, BV] = 1.0
        self.H_pos = np.zeros((2, 6)); self.H_pos[0, E] = self.H_pos[1, N] = 1.0
        self.H_psi = np.zeros((1, 6)); self.H_psi[0, PSI] = 1.0
        self.prev_rejected = False

    def predict(self, dt: float, omega: float) -> np.ndarray:
        """Propagate by `dt` seconds (clipped to [0, 1]) with turn rate `omega`; returns F."""
        cfg, x = self.cfg, self.x
        dt = float(np.clip(dt, 0.0, 1.0))
        s, c = np.sin(x[PSI]), np.cos(x[PSI])
        phi = np.exp(-dt / cfg.ml_bias_tau_s)
        F = np.eye(6)
        F[E, PSI], F[E, V] = x[V] * c * dt, s * dt
        F[N, PSI], F[N, V] = -x[V] * s * dt, c * dt
        F[PSI, BW] = -dt
        F[BV, BV] = phi
        x = x + np.array([x[V] * s * dt, x[V] * c * dt, (omega - x[BW]) * dt, 0.0, 0.0, 0.0])
        x[BV] *= phi
        x[PSI] = wrap_angle(x[PSI])
        q_turn = self.q_psi + (cfg.gyro_scale_sigma * omega) ** 2 if self.gyro_ok else self.q_psi
        Q = np.diag([0.01 * dt, 0.01 * dt, q_turn * dt, cfg.accel_sigma ** 2 * dt,
                     cfg.ml_bias_sigma ** 2 * (1 - phi ** 2), self.q_bw * dt])
        self.x, self.P = x, F @ self.P @ F.T + Q
        return F

    def update_ml(self, ml_speed: float) -> None:
        x = self.x
        self.x, self.P, _, _ = ekf_update(x, self.P, np.array([ml_speed]), np.array([x[V] + x[BV]]), self.H_ml,
                                          np.array([[self.cfg.ml_sigma ** 2]]))

    def update_gps(self, east: float, north: float, accuracy_m: float, speed: float, course_rad: float):
        """Position (gated, with the two-rejections recovery rule), speed,
        course (when moving, gated). Returns (position accepted, NIS)."""
        cfg = self.cfg
        sig = max(float(accuracy_m) if np.isfinite(accuracy_m) else 5.0, cfg.gps_pos_floor_m)
        x, P, ok, nis = ekf_update(self.x, self.P, np.array([east, north]), self.x[[E, N]], self.H_pos,
                                   np.eye(2) * sig ** 2, gate=CHI2_999[2] if cfg.gate else None)
        if not ok and self.prev_rejected:
            # two rejections in a row: the filter, not the GPS, has drifted -- accept, de-weighted
            x, P, _, _ = ekf_update(x, P, np.array([east, north]), x[[E, N]], self.H_pos,
                                    np.eye(2) * sig ** 2 * nis / CHI2_999[2])
        self.prev_rejected = not ok
        if np.isfinite(speed):
            x, P, _, _ = ekf_update(x, P, np.array([speed]), np.array([x[V]]), self.H_v,
                                    np.array([[cfg.gps_speed_sigma ** 2]]))
        if np.isfinite(course_rad) and speed > cfg.min_course_speed_ms:
            x, P, _, _ = ekf_update(x, P, np.array([course_rad]), np.array([x[PSI]]), self.H_psi,
                                    np.array([[np.radians(cfg.gps_course_sigma_deg) ** 2]]),
                                    gate=CHI2_999[1] if cfg.gate else None, angle_rows=(0,))
        self.x, self.P = x, P
        return ok, nis

    def finish_step(self) -> None:
        self.x[V] = max(self.x[V], 0.0)


def run_ekf(t: np.ndarray, omega: np.ndarray, ml_rows: dict, gps: dict, x0: np.ndarray, P0: np.ndarray,
            cfg: FusionConfig, gyro_ok: bool):
    """Forward EKF over consecutive rows with time stamps `t` (seconds).
    `omega[k]` turn rate (rad/s); `ml_rows` {k: ML speed}; `gps` {k: (east,
    north, accuracy_m, speed, course_rad)}. Returns dict of arrays x_f, P_f,
    x_p, P_p, F and the GPS gate decisions."""
    n = len(t)
    ekf = VehicleEKF(x0, P0, cfg, gyro_ok)
    x_f, P_f = np.zeros((n, 6)), np.zeros((n, 6, 6))
    x_p, P_p, Fs = np.zeros((n, 6)), np.zeros((n, 6, 6)), np.zeros((n, 6, 6))
    gate_log = {}
    for k in range(n):
        if k > 0:
            Fs[k] = ekf.predict(t[k] - t[k - 1], omega[k])
        x_p[k], P_p[k] = ekf.x, ekf.P
        if k in ml_rows:
            ekf.update_ml(ml_rows[k])
        if k in gps:
            gate_log[k] = ekf.update_gps(*gps[k])
        ekf.finish_step()
        x_f[k], P_f[k] = ekf.x, ekf.P
    return {"x_f": x_f, "P_f": P_f, "x_p": x_p, "P_p": P_p, "F": Fs, "gate": gate_log}


def _initial_state(gps_k, cfg: FusionConfig):
    ge, gn, acc, gv, gc = gps_k
    sig = max(float(acc) if np.isfinite(acc) else 5.0, cfg.gps_pos_floor_m)
    x0 = np.array([ge, gn, gc if np.isfinite(gc) else 0.0, gv if np.isfinite(gv) else 0.0, 0.0, 0.0])
    P0 = np.diag([sig ** 2, sig ** 2, np.radians(30.0) ** 2, 2.0 ** 2, cfg.ml_bias_sigma ** 2, np.radians(0.5) ** 2])
    return x0, P0


def simulate_fusion_outages(df: pd.DataFrame, model, ckpt: dict, device: str, outage_lengths_s=(30, 60, 120),
                            n_starts: int = 15, truth: np.ndarray | None = None, seed: int = 0,
                            history_s: float = 300.0, matcher=None, cfg: FusionConfig | None = None,
                            outlier_frac: float = 0.15, outlier_m: float = 200.0):
    """Phase 6 benchmark on one drive (loaded, gravity-compensated, NOT
    calibrated). Same outage starts as Days 10-12. Returns (end_rows,
    track_rows, robust_rows) DataFrames -- see the module docstring."""
    from src.dead_reckoning.heading import (GyroTurnIntegrator, end_position, fit_gyro_heading, gps_fix_intervals,
                                            gyro_heading, path_outage_starts, GYRO_COLS)
    from src.dead_reckoning.ins_with_ml_velocity import anchored_ml_speed, ml_speed_series
    from src.io_vnbd_loader import _gps_fix_index
    from src.map_matching.osm_loader import to_local_xy
    from src.models.multi_drive_data import apply_calibration, calibrate

    cfg = cfg or FusionConfig()
    W = ckpt["window_size"]
    t = df["t_sec"].to_numpy(dtype=float)
    lat, lon = df["gps_lat_deg"].to_numpy(dtype=float), df["gps_lon_deg"].to_numpy(dtype=float)
    held = df["gps_speed_ms"].to_numpy(dtype=float)
    course = df["gps_orientation_deg"].to_numpy(dtype=float)
    acc = df["gps_accuracy_m"].to_numpy(dtype=float) if "gps_accuracy_m" in df else np.full(len(df), 5.0)
    v_true = df["gps_speed_interp_ms"].to_numpy() if truth is None else np.asarray(truth, dtype=float)
    fix = _gps_fix_index(df)
    fixes_t = t[fix]
    integrator = GyroTurnIntegrator(df)
    starts = path_outage_starts(df, outage_lengths_s, n_starts, v_true, seed)
    rng = np.random.default_rng(seed + 1)
    end_rows, track_rows, robust_rows = [], [], []

    for s in starts:
        calib = calibrate(df.iloc[:s], allow_leveling_only=True)
        bias = np.array([calib["bias"][c] for c in GYRO_COLS])
        i0, i1, dcourse = gps_fix_intervals(t, fix, held, course, s)
        fit = fit_gyro_heading(integrator, bias, i0, i1, dcourse)
        ends = [int(fix[np.searchsorted(fixes_t, t[s] + L)]) for L in outage_lengths_s]
        e_max = max(ends)
        k0 = int(fix[np.searchsorted(fix, max(s - int(history_s * 10), W))])  # first fix of the history
        rows = np.arange(k0, e_max + 1)
        lat0, lon0 = float(lat[s]), float(lon[s])
        gx, gy = to_local_xy(lat[rows], lon[rows], lat0, lon0)

        # ML speed once per second over history + outage (pre-outage calibration only)
        seg = apply_calibration(df.iloc[k0 - (W - 1): e_max + 1], calib).reset_index(drop=True)
        ends_local = np.arange(W - 1, len(seg), 10)
        ml_pred = ml_speed_series(model, ckpt, seg, ends_local, device)
        ml_rows = {int(e - (W - 1)): float(v) for e, v in zip(ends_local, ml_pred)}
        ml_row_speed = np.interp(np.arange(len(rows)), np.array(sorted(ml_rows)), np.array([ml_rows[k] for k in sorted(ml_rows)]))

        omega = gyro_turn_rate(integrator, fit, bias, rows, t)
        fix_local = [int(f - k0) for f in fix if k0 <= f <= s]
        gps = {k: (gx[k], gy[k], acc[k0 + k], held[k0 + k], np.radians(course[k0 + k])) for k in fix_local}
        x0, P0 = _initial_state(gps[0], cfg)
        run = run_ekf(t[rows], omega, ml_rows, gps, x0, P0, cfg, fit is not None)
        x_f = run["x_f"]

        # Day 10 / 12 reference on the same rows
        so = s - k0
        rr = rows[so:]
        tau = t[rr] - t[s]
        v_anch = anchored_ml_speed(ml_row_speed[so:], float(held[s]), tau)
        c0 = np.radians(course[s])
        hdg_ref = gyro_heading(integrator, fit, bias, s, rr, c0) if fit is not None else np.full(len(rr), c0)
        dt_rr = np.clip(np.diff(t[rr], prepend=t[s]), 0.0, 1.0)
        mm_ekf = mm_ref = {}
        if matcher is not None:
            mm_ekf = matcher.match_outage(s, rr, x_f[so:, V], x_f[so:, PSI], ends)
            mm_ref = matcher.match_outage(s, rr, v_anch, hdg_ref, ends)

        for L, e in zip(outage_lengths_s, ends):
            m = rr <= e
            ke = e - k0
            ex, ey = gx[ke], gy[ke]
            true_dist = float(np.sum(v_true[s + 1: e + 1] * dt_rr[1: e - s + 1]))
            base = {"start_idx": int(s), "outage_s": L, "true_dist_m": true_dist, "gyro_ok": fit is not None}
            ref_xy = end_position(v_anch[m], hdg_ref[m], dt_rr[m])
            ekf_abs = (x_f[ke, E], x_f[ke, N])  # same frame: metres around the entrance fix
            res = {"day10_anchored_ml_gyro": ref_xy, "ekf": ekf_abs}
            if matcher is not None:
                r1, r2 = mm_ref.get(e), mm_ekf.get(e)
                res["day12_map_route"] = r1["route"] if r1 else ref_xy
                res["ekf_map_route"] = r2["route"] if r2 else ekf_abs
            for name, (px, py) in res.items():
                err = float(np.hypot(px - ex, py - ey))
                end_rows.append({**base, "method": name, "abs_err_m": err, "pct_err": 100 * err / max(true_dist, 1.0)})

            # offline track after GPS returns: hidden fixes strictly inside the outage
            hidden = [int(f - k0) for f in fix if s < f < e]
            if hidden:
                xs_e, Ps_e = x_f[ke].copy(), run["P_f"][ke].copy()
                H_pos = np.zeros((2, 6)); H_pos[0, E] = H_pos[1, N] = 1.0
                sig = max(acc[e], cfg.gps_pos_floor_m)
                xs_e, Ps_e, _, _ = ekf_update(xs_e, Ps_e, np.array([ex, ey]), xs_e[[E, N]], H_pos, np.eye(2) * sig ** 2)
                xf_seg, Pf_seg = x_f[so:ke + 1].copy(), run["P_f"][so:ke + 1].copy()
                xf_seg[-1], Pf_seg[-1] = xs_e, Ps_e
                x_s, _ = rts_smooth(xf_seg, Pf_seg, run["x_p"][so:ke + 1], run["P_p"][so:ke + 1],
                                    run["F"][so:ke + 1], angle_cols=(PSI,))
                d_ref = np.maximum(v_anch, 0) * dt_rr
                d_ref[0] = 0.0
                px_ref, py_ref = np.cumsum(d_ref * np.sin(hdg_ref)), np.cumsum(d_ref * np.cos(hdg_ref))
                end_off = (ex - px_ref[e - s], ey - py_ref[e - s])
                T = t[e] - t[s]
                for hk in hidden:
                    j = hk - so
                    hx_, hy_ = gx[hk], gy[hk]
                    frac = (t[k0 + hk] - t[s]) / T
                    cand = {"ekf_realtime": (x_f[hk, E], x_f[hk, N]),
                            "ekf_rts_smoothed": (x_s[j, E], x_s[j, N]),
                            "straight_line_entry_exit": (frac * ex, frac * ey),
                            "dr_end_error_spread": (px_ref[j] + frac * end_off[0], py_ref[j] + frac * end_off[1])}
                    for name, (px, py) in cand.items():
                        track_rows.append({**base, "method": name, "frac": frac,
                                           "abs_err_m": float(np.hypot(px - hx_, py - hy_))})

        # robustness: corrupt 15% of the history fixes (not the entrance fix), score at the entrance
        hist = [k for k in fix_local if 0 < k < so]
        if len(hist) >= 5:
            bad = set(rng.choice(hist, size=max(1, int(outlier_frac * len(hist))), replace=False).tolist())
            ang = rng.uniform(0, 2 * np.pi, size=len(gps))
            gps_bad = {k: ((v[0] + outlier_m * np.sin(a), v[1] + outlier_m * np.cos(a)) + v[2:] if k in bad else v)
                       for (k, v), a in zip(gps.items(), ang)}
            gps_eval = {k: v for k, v in gps_bad.items() if k < so}  # entrance fix withheld: predict it
            for gated in (True, False):
                c2 = FusionConfig(**{**cfg.__dict__, "gate": gated})
                r = run_ekf(t[rows[:so + 1]], omega[:so + 1], {k: v for k, v in ml_rows.items() if k <= so},
                            gps_eval, *_initial_state(gps[0], c2), c2, fit is not None)
                err = float(np.hypot(r["x_p"][so, E] - gx[so], r["x_p"][so, N] - gy[so]))
                n_rej = sum(1 for ok, _ in r["gate"].values() if not ok)
                n_bad_rej = sum(1 for k, (ok, _) in r["gate"].items() if not ok and k in bad)
                robust_rows.append({"start_idx": int(s), "method": "ekf_gated" if gated else "ekf_no_gate",
                                    "abs_err_m": err, "n_outliers": len(bad), "n_rejected": n_rej,
                                    "n_outliers_rejected": n_bad_rej})
            last = max(k for k in gps_eval)
            lx, ly = gps_bad[last][0], gps_bad[last][1]
            dtl = t[s] - t[k0 + last]
            c_last = gps_bad[last][4]
            lx, ly = lx + held[k0 + last] * dtl * np.sin(c_last), ly + held[k0 + last] * dtl * np.cos(c_last)
            robust_rows.append({"start_idx": int(s), "method": "last_fix_extrapolated",
                                "abs_err_m": float(np.hypot(lx - gx[so], ly - gy[so])), "n_outliers": len(bad),
                                "n_rejected": 0, "n_outliers_rejected": 0})
    return pd.DataFrame(end_rows), pd.DataFrame(track_rows), pd.DataFrame(robust_rows)


PHASE6_LABELS_END = {"day10_anchored_ml_gyro": "Day 10: anchored ML + gyro",
                     "ekf": "EKF fusion (real time)",
                     "day12_map_route": "Day 12: anchored ML + gyro + map (route)",
                     "ekf_map_route": "EKF fusion + map (route)"}
PHASE6_LABELS_TRACK = {"ekf_realtime": "EKF real time (no exit fix)",
                       "straight_line_entry_exit": "Straight line entry -> exit fix",
                       "dr_end_error_spread": "Day 10 DR, end error spread over time",
                       "ekf_rts_smoothed": "EKF + RTS smoother (uses exit fix)"}
PHASE6_LABELS_ROBUST = {"last_fix_extrapolated": "Last fix + speed/course extrapolation",
                        "ekf_no_gate": "EKF without gate",
                        "ekf_gated": "EKF with chi-square gate"}


if __name__ == "__main__":
    import json
    import warnings
    import zlib

    import torch

    root = Path(__file__).resolve()
    while not ((root / "src").is_dir() and (root / "data").is_dir()):
        root = root.parent
    sys.path.insert(0, str(root))
    from src.dead_reckoning.strapdown_ins import load_truth, log_metric
    from src.map_matching.hmm_map_matcher import HMMMapMatcher
    from src.map_matching.osm_loader import load_road_graph
    from src.models.multi_drive_data import load_raw_drive
    from src.models.train_velocity_model import load_model

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(root / "results" / "checkpoints" / "velocity_cnn_best.pt", device)
    split = json.loads((root / "data" / "split.json").read_text(encoding="utf-8"))
    maps = root / "data" / "external_maps"

    print(f"\n{'=' * 70}\nPhase 6 — GNSS + INS fusion (EKF) on the Days 10-12 outages (TEST + VAL drives)\n{'=' * 70}")
    parts = {"end": [], "track": [], "robust": []}
    for name in split["test"] + split["val"]:
        path = root / "data" / "raw" / "real" / name
        try:
            df = load_raw_drive(path)
        except Exception:  # noqa: BLE001
            continue
        if not 0.07 <= float(np.median(np.diff(df["t_sec"]))) <= 0.13:
            continue
        truth, source = load_truth(df, name, root)
        matcher = HMMMapMatcher(load_road_graph(name, cache_dir=maps), df) if (maps / f"{path.stem}.graphml").exists() else None
        res = simulate_fusion_outages(df, model, ckpt, device, truth=truth, seed=zlib.crc32(name.encode()), matcher=matcher)
        if res[0].empty:
            continue
        for key, r in zip(parts, res):
            r["drive"], r["split"] = name, "test" if name in split["test"] else "val"
            parts[key].append(r)
        print(f"  {name:<12} {res[0].start_idx.nunique():>3} outages  truth={source}  map={'yes' if matcher else 'no'}")

    R = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    for k, r in R.items():
        r.to_csv(root / "results" / f"phase6_{k}_benchmark.csv", index=False)

    def table(res, labels, title, section, group="outage_s", tag=""):
        print(f"\n{title}")
        print(f"{'Method':<46} {'Outage':>7} {'Median err (m)':>15} {'P90 (m)':>8} {'N':>6}")
        print("-" * 88)
        for L in sorted(res[group].unique()) if group else [None]:
            for method, label in labels.items():
                sub = res[(res.method == method) & ((res[group] == L) if group else True)]
                if sub.empty:
                    continue
                m = {"outage_s": L if L is not None else 0, "median_abs_err_m": sub.abs_err_m.median(),
                     "median_pct_err": sub.pct_err.median() if "pct_err" in sub else float("nan"),
                     "mean_abs_err_m": sub.abs_err_m.mean(), "n_outages": len(sub), "n_drives": sub.drive.nunique()}
                print(f"{label:<46} {str(L) + 's' if L is not None else '—':>7} {m['median_abs_err_m']:>15.1f} "
                      f"{sub.abs_err_m.quantile(0.9):>8.0f} {len(sub):>6}")
                log_metric(root / "results", f"{label}{tag}" + (f", {L}s outage" if L is not None else ""), m,
                           day="Day 13", section_title=section)
            print()

    for sp, sub_tag in (("test", ""), ("all", " [test+val]")):
        sel = (lambda r: r[r.split == "test"]) if sp == "test" else (lambda r: r)
        n_d = sel(R["end"]).drive.nunique()
        table(sel(R["end"]), PHASE6_LABELS_END, f"1. Real-time end position — {sp.upper()} ({n_d} drives)",
              f"Day 13 — EKF fusion, real-time end error ({'TEST' if sp == 'test' else 'TEST + VAL'} drives)", tag=sub_tag)
        table(sel(R["track"]), PHASE6_LABELS_TRACK, f"2. Tunnel track after GPS returns (hidden fixes inside the outage) — {sp.upper()}",
              f"Day 13 — EKF fusion, offline tunnel track at hidden GPS fixes ({'TEST' if sp == 'test' else 'TEST + VAL'} drives)",
              tag=" (track)" + sub_tag)
    rb = R["robust"]
    print("\n3. Robustness: 15% of pre-outage fixes replaced by 200 m jumps; error predicting the entrance fix (TEST + VAL)")
    for method, label in PHASE6_LABELS_ROBUST.items():
        sub = rb[rb.method == method]
        print(f"   {label:<46} median {sub.abs_err_m.median():7.1f} m   P90 {sub.abs_err_m.quantile(0.9):7.1f} m   N {len(sub)}")
        log_metric(root / "results", f"{label} (robustness)", {"outage_s": 0, "median_abs_err_m": sub.abs_err_m.median(),
                   "median_pct_err": float("nan"), "mean_abs_err_m": sub.abs_err_m.mean(), "n_outages": len(sub),
                   "n_drives": sub.drive.nunique()}, day="Day 13",
                   section_title="Day 13 — EKF fusion, robustness to GPS outliers (TEST + VAL drives)")
    g = rb[rb.method == "ekf_gated"]
    print(f"   gate rejected {g.n_outliers_rejected.sum()} of {g.n_outliers.sum()} injected outliers "
          f"and {int((g.n_rejected - g.n_outliers_rejected).sum())} genuine fixes")
    print("\nSummary rows written to results/metrics.md (re-running replaces them).")
