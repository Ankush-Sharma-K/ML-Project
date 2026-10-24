"""
engine.py — Phase 7 (Day 15): streaming IDR engine for the phone / edge device
==============================================================================

Everything from Phases 2-6 as ONE object that consumes sensor samples one at a
time, the way a phone delivers them, and returns a navigation solution for
every sample:

    engine = IDREngine("results/checkpoints/velocity_model_int8.ptl", ckpt_meta)
    for sample in sensor_stream:            # 10 Hz dicts, phone column names
        out = engine.push(sample)           # EngineOutput: lat, lon, heading,
                                            # speed, sigma, status, route status

PIPELINE PER SAMPLE
  - linear acceleration = accelerometer - gravity (Day 3)
  - every `recalib_every_s` while GPS is available: calibration (bias, mount
    rotation) and the per-phone gyro mapping, from the last `buffer_s` seconds
    only (Days 3-5, 10) -- a phone keeps a bounded buffer
  - turn rate from the calibrated gyro mapping (Day 10)
  - once per second: speed from the on-device model (`.ptl`, int8 by default;
    raw 20 s window in, m/s out, Phase 3) on the last 200 calibrated samples
  - EKF predict + ML update + GPS update at each new fix (Phase 6), with the
    outlier gate and the recovery rule
  - status: GNSS (fix within `outage_after_s`), DR (no fix), DR+MAP (no fix,
    position snapped to the road network by the HMM matcher, Phase 5)
  - when GPS returns after an outage: RTS smoothing of the outage segment;
    the corrected tunnel track is returned once in `out.track_corrected`
  - optional route monitor (`RouteMonitor`): ON_ROUTE / OFF_ROUTE against a
    planned route polyline (Stage 6 of the project brief)

FRAME. The EKF works in metres around an origin (first fix); the origin moves
to the current fix whenever the car is > `reorigin_m` away while GPS is
available, so the flat-earth error stays small on 100 km drives.

MAP MATCHER. `map_matcher` is any object with
`match_outage(s, rr, speed, heading, ends)` using GLOBAL sample indices (e.g.
`HMMMapMatcher(G, df)` in replay, whose leak guard checks the search area
against the drive's corridor). On a device the matcher would be built from
the device's own map, where no corridor guard is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.fusion.gnss_ins_fusion import BV, E, N, PSI, V, FusionConfig, VehicleEKF
from src.fusion.kalman_filter import rts_smooth
from src.map_matching.hmm_map_matcher import from_local_xy
from src.map_matching.osm_loader import to_local_xy

RAW_COLS = ["t_sec", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2", "gravity_x_mps2", "gravity_y_mps2",
            "gravity_z_mps2", "gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads", "gps_lat_deg", "gps_lon_deg",
            "gps_speed_ms", "gps_speed_kmh", "gps_orientation_deg", "gps_accuracy_m"]


def calibrated_features(rows: list[dict], calib: dict, feature_cols: list[str]) -> np.ndarray:
    """Model input window (len(rows), n_features) float32 from raw sample dicts:
    the same arithmetic as `multi_drive_data.apply_calibration` (bias
    subtraction, mount rotation, magnitudes) in numpy -- pandas on a 200-row
    frame every second was 25 of the 49 s of a 12-minute replay."""
    A = np.array([[r[f"accel_{a}_mps2"] - r[f"gravity_{a}_mps2"] for a in "xyz"] for r in rows], dtype=float)
    Wg = np.array([[r["gyro_yaw_rads"], r["gyro_pitch_rads"], r["gyro_roll_rads"]] for r in rows], dtype=float)
    b = calib["bias"]
    A = A - np.array([b["lin_accel_x_mps2"], b["lin_accel_y_mps2"], b["lin_accel_z_mps2"]])
    Wg = Wg - np.array([b["gyro_yaw_rads"], b["gyro_pitch_rads"], b["gyro_roll_rads"]])
    veh = A @ np.asarray(calib["R_align"]).T
    cols = {"lin_accel_x_mps2_cal_veh": veh[:, 0], "lin_accel_y_mps2_cal_veh": veh[:, 1],
            "lin_accel_z_mps2_cal_veh": veh[:, 2], "gyro_yaw_rads_cal": Wg[:, 0], "gyro_pitch_rads_cal": Wg[:, 1],
            "gyro_roll_rads_cal": Wg[:, 2], "lin_accel_horiz_mag_mps2": np.hypot(veh[:, 0], veh[:, 1]),
            "gyro_mag_rads": np.sqrt((Wg ** 2).sum(axis=1))}
    return np.stack([cols[c] for c in feature_cols], axis=1).astype(np.float32)


@dataclass
class EngineOutput:
    t: float
    lat: float
    lon: float
    heading_deg: float
    speed_ms: float
    sigma_m: float
    status: str                    # INIT / GNSS / DR / DR+MAP
    route_status: str | None = None
    route_dist_m: float | None = None
    exit_realtime: tuple | None = None       # (lat, lon) just before the exit fix was applied
    track_corrected: list | None = field(default=None, repr=False)  # [(t, lat, lon), ...] after an outage


class RouteMonitor:
    """ON_ROUTE / OFF_ROUTE against a planned route (lat/lon polyline):
    distance from the position to the route > max(base_m, k_sigma * sigma) for
    `hold_s` seconds switches the state (hysteresis both ways).

    Tuned on VAL drives only (Day 15; 120 grid points): the first rule (40 m,
    tighter when map-matched, 3 s hold) raised false alarms on the correct
    route in 72% of val outages; base 40 m, 2 sigma always, 10 s hold gave 20%
    false alarms / 80% valid detections on val, and 5% / 76% on TEST."""

    def __init__(self, route_lat, route_lon, base_m: float = 40.0, k_sigma: float = 2.0, hold_s: float = 10.0):
        self.lat = np.asarray(route_lat, dtype=float)
        self.lon = np.asarray(route_lon, dtype=float)
        self.base_m, self.k_sigma, self.hold_s = base_m, k_sigma, hold_s
        self.state, self._since, self._candidate = "ON_ROUTE", None, None

    def distance_m(self, lat: float, lon: float) -> float:
        x, y = to_local_xy(self.lat, self.lon, lat, lon)
        ax, ay, bx, by = x[:-1], y[:-1], x[1:], y[1:]
        abx, aby = bx - ax, by - ay
        u = np.clip(-(ax * abx + ay * aby) / np.maximum(abx ** 2 + aby ** 2, 1e-9), 0.0, 1.0)
        return float(np.min(np.hypot(ax + u * abx, ay + u * aby))) if len(ax) else float("inf")

    def update(self, t: float, lat: float, lon: float, sigma_m: float) -> tuple[str, float]:
        d = self.distance_m(lat, lon)
        thr = max(self.base_m, self.k_sigma * sigma_m)
        cand = "OFF_ROUTE" if d > thr else "ON_ROUTE"
        if cand == self.state:
            self._candidate, self._since = None, None
        elif cand != self._candidate:
            self._candidate, self._since = cand, t
        elif t - self._since >= self.hold_s:
            self.state, self._candidate, self._since = cand, None, None
        return self.state, d


class IDREngine:
    def __init__(self, model_path, ckpt_meta: dict, map_matcher=None, cfg: FusionConfig | None = None,
                 recalib_every_s: float = 60.0, buffer_s: float = 600.0, min_calib_s: float = 120.0,
                 outage_after_s: float = 12.0, map_every_s: float = 5.0, reorigin_m: float = 5000.0,
                 route_monitor: RouteMonitor | None = None, start_index: int = 0):
        from src.models.compress_model import load_lite

        self.model = load_lite(model_path)
        self.feature_cols, self.W = list(ckpt_meta["feature_cols"]), int(ckpt_meta["window_size"])
        self.matcher, self.cfg = map_matcher, cfg or FusionConfig()
        self.recalib_every_s, self.buffer_n, self.min_calib_s = recalib_every_s, int(buffer_s * 10), min_calib_s
        self.outage_after_s, self.map_every_s, self.reorigin_m = outage_after_s, map_every_s, reorigin_m
        self.route = route_monitor
        self.buf: list[dict] = []
        self.k = start_index - 1        # global sample index (= drive row in replay, for the map matcher)
        self.calib = self.fit = None
        self.gyro_bias = np.zeros(3)
        self.last_calib_t = -np.inf
        self.ekf: VehicleEKF | None = None
        self.origin = None              # (lat0, lon0)
        self.last_fix_k, self.last_fix_t = None, -np.inf
        self.seg = None                 # outage history for RTS / map matching
        self.map_res, self.map_k = None, None

    # ---------------------------------------------------------------- helpers
    def _df(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        for ax in "xyz":
            df[f"lin_accel_{ax}_mps2"] = df[f"accel_{ax}_mps2"] - df[f"gravity_{ax}_mps2"]
        return df

    def _recalibrate(self, t: float) -> None:
        from src.dead_reckoning.heading import GYRO_COLS, GyroTurnIntegrator, fit_gyro_heading, gps_fix_intervals
        from src.io_vnbd_loader import _gps_fix_index
        from src.models.multi_drive_data import calibrate

        df = self._df(self.buf)
        try:
            calib = calibrate(df, allow_leveling_only=True)
        except Exception:  # noqa: BLE001 -- keep the previous calibration
            return
        self.calib = calib
        self.gyro_bias = np.array([calib["bias"][c] for c in GYRO_COLS])
        tt = df["t_sec"].to_numpy(dtype=float)
        i0, i1, dc = gps_fix_intervals(tt, _gps_fix_index(df), df["gps_speed_ms"].to_numpy(dtype=float),
                                       df["gps_orientation_deg"].to_numpy(dtype=float), len(df))
        self.fit = fit_gyro_heading(GyroTurnIntegrator(df), self.gyro_bias, i0, i1, dc)
        self.last_calib_t = t

    def _turn_rate(self, s: dict) -> float:
        if self.fit is None:
            return 0.0
        from src.dead_reckoning.heading import GRAVITY_COLS, GYRO_COLS

        g = np.array([s[c] for c in GRAVITY_COLS], dtype=float)
        g = g / max(np.linalg.norm(g), 1e-9)
        w = np.array([s[c] for c in GYRO_COLS], dtype=float) - self.gyro_bias
        return float(self.fit.k * sum(sg * w[pi] * g[j] for j, (pi, sg) in enumerate(zip(self.fit.perm, self.fit.signs))))

    def _ml_speed(self) -> float:
        import torch

        X = calibrated_features(self.buf[-self.W:], self.calib, self.feature_cols)[None]
        with torch.no_grad():
            return float(self.model(torch.from_numpy(X)).numpy()[0])

    def _local(self, lat, lon):
        return to_local_xy(lat, lon, *self.origin)

    def _latlon(self, x, y):
        lat, lon = from_local_xy(x, y, *self.origin)
        return float(lat), float(lon)

    def _sigma(self) -> float:
        return float(np.sqrt(max(self.ekf.P[E, E] + self.ekf.P[N, N], 0.0) / 2.0))

    # ---------------------------------------------------------------- main
    def push(self, s: dict) -> EngineOutput:
        """One 10 Hz sample: phone column names (see RAW_COLS) plus
        `gps_fix` (True when a NEW GPS fix arrived with this sample)."""
        self.k += 1
        t = float(s["t_sec"])
        self.buf.append({c: s.get(c, np.nan) for c in RAW_COLS})
        if len(self.buf) > self.buffer_n:
            del self.buf[0]
        fix = bool(s.get("gps_fix", False)) and abs(float(s["gps_lat_deg"])) > 1.0
        have_gps = fix or (t - self.last_fix_t) <= self.outage_after_s

        if have_gps and t - self.last_calib_t >= self.recalib_every_s and len(self.buf) >= self.min_calib_s * 10:
            self._recalibrate(t)
        if self.calib is None or (self.ekf is None and not fix):
            if fix:
                self.last_fix_k, self.last_fix_t = self.k, t
            return EngineOutput(t, float(s["gps_lat_deg"]), float(s["gps_lon_deg"]),
                                float(s.get("gps_orientation_deg", np.nan)), float(s.get("gps_speed_ms", np.nan)),
                                float(s.get("gps_accuracy_m", np.nan)), "INIT")

        lat_f, lon_f = float(s["gps_lat_deg"]), float(s["gps_lon_deg"])
        if self.ekf is None:  # first fix after the first calibration
            from src.fusion.gnss_ins_fusion import _initial_state

            self.origin = (lat_f, lon_f)
            x0, P0 = _initial_state((0.0, 0.0, float(s["gps_accuracy_m"]), float(s["gps_speed_ms"]),
                                     np.radians(float(s["gps_orientation_deg"]))), self.cfg)
            self.ekf = VehicleEKF(x0, P0, self.cfg, self.fit is not None)
            self.prev_t = t
        else:
            self.ekf.gyro_ok = self.fit is not None
            F = self.ekf.predict(t - self.prev_t, self._turn_rate(s))
            self.prev_t = t
            if self.seg is not None:
                self.seg["x_p"].append(self.ekf.x.copy()); self.seg["P_p"].append(self.ekf.P.copy())
                self.seg["F"].append(F)
        if self.k % 10 == 0 and len(self.buf) >= self.W:
            self.ekf.update_ml(self._ml_speed())

        out_extra = {}
        outage = (t - self.last_fix_t) > self.outage_after_s
        if fix:
            if self.seg is not None and outage:  # exit: realtime position before the fix, then smooth
                out_extra["exit_realtime"] = self._current_position()[:2]
            ge, gn = self._local(lat_f, lon_f)
            self.ekf.update_gps(float(ge), float(gn), float(s["gps_accuracy_m"]), float(s["gps_speed_ms"]),
                                np.radians(float(s["gps_orientation_deg"])))
            self.ekf.finish_step()
            if self.seg is not None and outage:
                out_extra["track_corrected"] = self._smooth_segment()
            self.seg, self.map_res = None, None
            self.last_fix_k, self.last_fix_t = self.k, t
            if np.hypot(self.ekf.x[E], self.ekf.x[N]) > self.reorigin_m:  # move the origin to this fix
                lat_x, lon_x = self._latlon(self.ekf.x[E], self.ekf.x[N])
                self.origin = (lat_f, lon_f)
                self.ekf.x[E], self.ekf.x[N] = (float(v) for v in self._local(lat_x, lon_x))
            self._start_segment()
        else:
            self.ekf.finish_step()
            if self.seg is None:
                self._start_segment()
        self.seg["x_f"].append(self.ekf.x.copy()); self.seg["P_f"].append(self.ekf.P.copy())
        self.seg["t"].append(t)

        lat, lon, status = self._current_position()
        out = EngineOutput(t, lat, lon, float(np.degrees(self.ekf.x[PSI]) % 360.0), float(self.ekf.x[V]),
                           self._sigma(), status, **out_extra)
        if self.route is not None:
            out.route_status, out.route_dist_m = self.route.update(t, lat, lon, out.sigma_m)
        return out

    def _start_segment(self):
        """History since the last fix (the possible outage): EKF states for RTS,
        speed / heading for the map matcher."""
        self.seg = {"k0": self.k, "x_f": [], "P_f": [], "x_p": [self.ekf.x.copy()], "P_p": [self.ekf.P.copy()],
                    "F": [np.eye(6)], "t": []}

    def _current_position(self):
        x, y = self.ekf.x[E], self.ekf.x[N]
        if self.prev_t - self.last_fix_t <= self.outage_after_s or self.seg is None:
            return (*self._latlon(x, y), "GNSS")
        if self.matcher is None:
            return (*self._latlon(x, y), "DR")
        if self.map_k is None or self.map_res is None or (self.k - self.map_k) / 10.0 >= self.map_every_s:
            self._run_matcher()
        if self.map_res is None:
            return (*self._latlon(x, y), "DR")
        # map position at the last match (metres around the entrance fix) + filter displacement since then
        mx, my, x_match, y_match = self.map_res
        lat, lon = self._latlon_from_fix(mx + (x - x_match), my + (y - y_match))
        return lat, lon, "DR+MAP"

    def _latlon_from_fix(self, dx, dy):
        lat_s, lon_s = self.fix_latlon
        lat, lon = from_local_xy(dx, dy, lat_s, lon_s)
        return float(lat), float(lon)

    def _run_matcher(self):
        """Map-match the outage so far (rows since the last fix) with the
        filter's speed and heading."""
        seg = self.seg
        n = len(seg["x_f"])
        self.map_k = self.k
        if n < 2:
            self.map_res = None
            return
        s = seg["k0"]
        rr = np.arange(s, s + n)
        X = np.array(seg["x_f"])
        res = self.matcher.match_outage(s, rr, X[:, V], X[:, PSI], [int(rr[-1])]).get(int(rr[-1]))
        if res is None:
            self.map_res = None
            return
        # match_outage works in metres around the entrance fix (its own lat/lon at row s)
        self.fix_latlon = (float(self.matcher.lat[s]), float(self.matcher.lon[s]))
        self.map_res = (res["route"][0], res["route"][1], float(X[-1, E]), float(X[-1, N]))

    def _smooth_segment(self):
        seg = self.seg
        n = len(seg["x_f"])
        if n < 2:
            return None
        x_f = np.array(seg["x_f"] + [self.ekf.x.copy()])
        P_f = np.array(seg["P_f"] + [self.ekf.P.copy()])
        x_p, P_p, F = (np.array(seg[k][:n + 1]) for k in ("x_p", "P_p", "F"))
        x_s, _ = rts_smooth(x_f, P_f, x_p, P_p, F, angle_cols=(PSI,))
        ts = seg["t"] + [self.prev_t]
        return [(float(tt), *self._latlon(xs[E], xs[N])) for tt, xs in zip(ts, x_s)]
