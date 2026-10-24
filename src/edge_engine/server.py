"""
server.py — Phase 7 (Day 15): edge engine HTTP API + app host
=============================================================

Standard library only (`http.server`), so it runs on any device with Python:

  GET /                       the mobile web app (mobile_app/)
  GET /api/health             {"ok": true, "model": ...}
  GET /api/drives             drives with a road map and their outage starts
                              (TEST + VAL drives, the same outages as Days 10-15)
  GET /api/replay?drive=S-S2.csv&start=12345&outage=120&model=int8&route=correct
                              streams the drive through IDREngine with GPS
                              withheld for `outage` seconds after `start` and
                              returns the GPS truth, the live engine output
                              (1 Hz), the corrected tunnel track, the planned
                              route and the route-monitor states. Cached in
                              results/replays/.

Run:  python -m src.edge_engine.server [--port 8000] [--host 0.0.0.0]   then open http://localhost:8000
(localhost only unless --host is given)
"""

from __future__ import annotations

import json
import sys
import threading
import zlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve()
while not ((ROOT / "src").is_dir() and (ROOT / "data").is_dir()):
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
APP_DIR = ROOT / "mobile_app"
CACHE_DIR = ROOT / "results" / "replays"
OUTAGES = (30, 60, 120)
_lock = threading.Lock()
_drive_cache: dict = {}


def _load_drive(name: str):
    """(df, matcher, G, starts) for a drive, cached in memory."""
    from src.dead_reckoning.heading import path_outage_starts
    from src.dead_reckoning.strapdown_ins import load_truth
    from src.map_matching.hmm_map_matcher import HMMMapMatcher
    from src.map_matching.osm_loader import load_road_graph
    from src.models.multi_drive_data import load_raw_drive

    with _lock:
        if name not in _drive_cache:
            df = load_raw_drive(ROOT / "data" / "raw" / "real" / name)
            truth, _ = load_truth(df, name, ROOT)
            v_true = df["gps_speed_interp_ms"].to_numpy() if truth is None else truth
            starts = path_outage_starts(df, OUTAGES, 15, v_true, zlib.crc32(name.encode()))
            G = load_road_graph(name, cache_dir=ROOT / "data" / "external_maps")
            _drive_cache[name] = (df, HMMMapMatcher(G, df), G, starts)
        return _drive_cache[name]


DRIVES_INDEX = CACHE_DIR / "drives_index.json"


def list_drives() -> list[dict]:
    """TEST + VAL drives with a road map and their outage starts. Needs only
    the drive file (not the road graph); computed once and saved to
    results/replays/drives_index.json, so the app's first request is instant."""
    from src.dead_reckoning.heading import path_outage_starts
    from src.dead_reckoning.strapdown_ins import load_truth
    from src.models.multi_drive_data import load_raw_drive

    if DRIVES_INDEX.exists():
        return json.loads(DRIVES_INDEX.read_text(encoding="utf-8"))
    split = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
    out = []
    for name in split["test"] + split["val"]:
        if not (ROOT / "data" / "external_maps" / f"{Path(name).stem}.graphml").exists():
            continue
        try:
            df = load_raw_drive(ROOT / "data" / "raw" / "real" / name)
        except Exception:  # noqa: BLE001
            continue
        truth, _ = load_truth(df, name, ROOT)
        v_true = df["gps_speed_interp_ms"].to_numpy() if truth is None else truth
        starts = path_outage_starts(df, OUTAGES, 15, v_true, zlib.crc32(name.encode()))
        if len(starts):
            out.append({"drive": name, "split": "test" if name in split["test"] else "val",
                        "duration_min": round(float(df["t_sec"].iloc[-1]) / 60, 1),
                        "starts": [{"start": int(s), "t_min": round(float(df["t_sec"].iloc[s]) / 60, 1)} for s in starts]})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DRIVES_INDEX.write_text(json.dumps(out), encoding="utf-8")
    return out


def run_replay(name: str, start: int, outage_s: int, model: str = "int8", route: str = "correct") -> dict:
    """One simulated tunnel through the streaming engine, as JSON-ready dict."""
    import torch

    from src.edge_engine.engine import IDREngine, RouteMonitor
    from src.edge_engine.replay import replay, wrong_route
    from src.io_vnbd_loader import _gps_fix_index
    from src.map_matching.osm_loader import to_local_xy
    from src.models.train_velocity_model import load_checkpoint

    key = f"{Path(name).stem}_{start}_{outage_s}_{model}_{route}.json"
    cached = CACHE_DIR / key
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    df, matcher, G, starts = _load_drive(name)
    if start not in set(int(s) for s in starts):
        raise ValueError("unknown outage start for this drive")
    t = df["t_sec"].to_numpy(dtype=float)
    lat, lon = df["gps_lat_deg"].to_numpy(dtype=float), df["gps_lon_deg"].to_numpy(dtype=float)
    fix = _gps_fix_index(df)
    e = int(fix[np.searchsorted(t[fix], t[start] + outage_s)])
    r0, r_show = max(0, start - 6000), max(0, start - 1200)  # engine warm-up 10 min, show the last 2 min
    hidden = [int(f) for f in fix if start < f < e]
    tunnel_fix = [int(f) for f in fix if start <= f <= e]
    if route == "wrong":
        wr = wrong_route(G, lat[tunnel_fix], lon[tunnel_fix])
        if wr is None:
            raise ValueError("no alternative route found for this outage")
        r_lat, r_lon = wr
        r_lat = np.r_[lat[[f for f in fix if r0 <= f < start]], r_lat]
        r_lon = np.r_[lon[[f for f in fix if r0 <= f < start]], r_lon]
    else:
        drv = [int(f) for f in fix if r0 <= f <= min(len(df) - 1, e + 300)]
        r_lat, r_lon = lat[drv], lon[drv]
    monitor = RouteMonitor(r_lat, r_lon)
    ckpt = load_checkpoint(ROOT / "results" / "checkpoints" / "velocity_cnn_best.pt", "cpu")
    meta = {"feature_cols": ckpt["feature_cols"], "window_size": ckpt["window_size"]}
    model_file = "velocity_model_int8.ptl" if model == "int8" else "velocity_model_fp32.ptl"
    eng = IDREngine(ROOT / "results" / "checkpoints" / model_file, meta, map_matcher=matcher, start_index=r0,
                    route_monitor=monitor)
    outs, _ = replay(df, eng, r0, e, (start, e))
    o_exit = outs[e - r0]
    exit_rt = o_exit.exit_realtime or (outs[e - r0 - 1].lat, outs[e - r0 - 1].lon)
    end_err = float(np.hypot(*to_local_xy(exit_rt[0], exit_rt[1], lat[e], lon[e])))
    live = [{"t": round(o.t - t[start], 1), "lat": round(o.lat, 7), "lon": round(o.lon, 7), "status": o.status,
             "speed": round(o.speed_ms, 2), "heading": round(o.heading_deg, 1), "sigma": round(o.sigma_m, 1),
             "route": o.route_status, "route_dist": None if o.route_dist_m is None else round(o.route_dist_m, 1)}
            for i, o in enumerate(outs) if i + r0 >= r_show and (i % 10 == 0 or i + r0 == e)]
    res = {"drive": name, "start": start, "outage_s": outage_s, "model": model, "route_kind": route,
           "t0": float(t[start]),
           "gps_before": [[round(lat[f], 7), round(lon[f], 7)] for f in fix if r_show <= f <= start],
           "gps_hidden": [[round(float(t[f] - t[start]), 1), round(lat[f], 7), round(lon[f], 7)] for f in hidden],
           "gps_exit": [round(lat[e], 7), round(lon[e], 7)],
           "live": live,
           "exit_realtime": [round(exit_rt[0], 7), round(exit_rt[1], 7)],
           "end_error_m": round(end_err, 1),
           "track_corrected": [[round(q[0] - t[start], 1), round(q[1], 7), round(q[2], 7)]
                               for q in (o_exit.track_corrected or [])][::10],
           "planned_route": [[round(a, 7), round(b, 7)] for a, b in zip(r_lat, r_lon)],
           "route_off_detected_t": next((p["t"] for p in live if p["route"] == "OFF_ROUTE" and p["t"] > 0), None)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(res), encoding="utf-8")
    return res


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(APP_DIR), **kw)

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/health":
                return self._json({"ok": True, "model": "velocity_model_int8.ptl"})
            if u.path == "/api/drives":
                return self._json(list_drives())
            if u.path == "/api/replay":
                return self._json(run_replay(q["drive"], int(q["start"]), int(q.get("outage", 120)),
                                             q.get("model", "int8"), q.get("route", "correct")))
        except (KeyError, ValueError) as ex:
            return self._json({"error": str(ex)}, 400)
        return super().do_GET()

    def log_message(self, fmt, *args):  # quieter console: API requests and errors only
        msg = fmt % args
        if "/api/" in msg or "code 5" in msg:
            sys.stderr.write(msg + "\n")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    # localhost only by default; pass --host 0.0.0.0 to reach it from a phone on the same network
    host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "127.0.0.1"
    print(f"IDR edge engine: http://{host}:{port}  (app: {APP_DIR})")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
