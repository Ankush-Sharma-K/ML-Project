"""
replay.py — Phase 7 (Day 15): replay recorded drives through the streaming engine
=================================================================================

`replay(df, engine, r0, r1, mask)` feeds rows r0..r1 of a loaded drive into an
`IDREngine` sample by sample; GPS fixes inside `mask` (a simulated tunnel) are
withheld, exactly as a phone would see them.

`__main__` = the Phase 7 benchmark on the Days 10-13 outages (TEST + VAL
drives), with a fresh engine per outage that starts 10 minutes before the
tunnel (the phone's bounded buffer):

  1. parity: streaming engine with the float32 model vs the Phase 6 batch EKF
     (`results/phase6_end_benchmark.csv`) -- the engine must reproduce it
  2. deployment: int8 model + map matching, real-time end error at 30/60/120 s
     and the corrected tunnel track (RTS at the exit fix) for the 120 s outage
  3. route verification (Stage 6 of the brief): two planned routes per
     outage, monitored on the engine output --
       correct route = the route actually driven  -> any OFF_ROUTE is a false alarm
       wrong route   = shortest road path from the entrance to the true exit
                       that avoids the roads actually driven (the car really
                       leaves it; only outages where the driven path is > 100 m
                       from it are used)  -> should be detected, and only an
                       alarm raised no earlier than 5 s before the car actually
                       left the route counts (an earlier one is premature)
     Planned routes are navigation inputs, so building them from the recorded
     track is legitimate; the engine never sees the hidden fixes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.edge_engine.engine import RAW_COLS, IDREngine, RouteMonitor


def replay(df: pd.DataFrame, engine: IDREngine, r0: int, r1: int, mask: tuple[int, int] | None = None,
           monitors: dict | None = None):
    """Feed rows r0..r1 (inclusive). Fixes with mask[0] < row < mask[1] are
    withheld. `monitors` {name: RouteMonitor} are updated on every output.
    Returns (outputs list, {name: [(t, state, dist)]})."""
    from src.io_vnbd_loader import _gps_fix_index

    is_fix = np.zeros(len(df), dtype=bool)
    is_fix[_gps_fix_index(df)] = True
    cols = [c for c in RAW_COLS if c in df.columns]
    data = df[cols].to_numpy(dtype=float)
    outs, mon = [], {k: [] for k in (monitors or {})}
    for r in range(r0, r1 + 1):
        s = dict(zip(cols, data[r]))
        s["gps_fix"] = bool(is_fix[r]) and not (mask is not None and mask[0] < r < mask[1])
        o = engine.push(s)
        outs.append(o)
        for name, m in (monitors or {}).items():
            st, d = m.update(o.t, o.lat, o.lon, o.sigma_m)
            mon[name].append((o.t, st, d))
    return outs, mon


def wrong_route(G, lat_true, lon_true, avoid_m: float = 40.0, penalty: float = 20.0):
    """Shortest road path between the nodes nearest to the first and last true
    point, with edges near the driven path made `penalty` times longer.
    Returns (lat, lon) arrays of the route polyline, or None."""
    import networkx as nx
    from scipy.spatial import cKDTree

    from src.map_matching.osm_loader import to_local_xy

    lat0, lon0 = float(lat_true[0]), float(lon_true[0])
    nodes = list(G.nodes)
    ny = np.array([G.nodes[n]["y"] for n in nodes])
    nx_ = np.array([G.nodes[n]["x"] for n in nodes])
    px, py = to_local_xy(ny, nx_, lat0, lon0)
    tx, ty = to_local_xy(np.asarray(lat_true), np.asarray(lon_true), lat0, lon0)
    node_tree = cKDTree(np.c_[px, py])
    src = nodes[int(node_tree.query([tx[0], ty[0]])[1])]
    dst = nodes[int(node_tree.query([tx[-1], ty[-1]])[1])]
    if src == dst:
        return None
    # densify the true path to ~10 m so "near the driven path" is tested along it
    seg = np.hypot(np.diff(tx), np.diff(ty))
    dense = [np.c_[np.linspace(tx[i], tx[i + 1], max(int(seg[i] // 10), 1) + 1),
                   np.linspace(ty[i], ty[i + 1], max(int(seg[i] // 10), 1) + 1)] for i in range(len(seg))]
    true_tree = cKDTree(np.vstack(dense) if dense else np.c_[tx, ty])
    pos = dict(zip(nodes, zip(px, py)))

    def weight(u, v, d):
        best = np.inf
        for e in d.values():
            mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            near = true_tree.query([mx, my])[0] < avoid_m
            best = min(best, float(e.get("length", 1.0)) * (penalty if near else 1.0))
        return best

    try:
        path = nx.shortest_path(G, src, dst, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    lats, lons = [], []
    for u, v in zip(path[:-1], path[1:]):
        d = min(G[u][v].values(), key=lambda e: float(e.get("length", 1.0)))
        if "geometry" in d:
            xs, ys = d["geometry"].xy
            lons.extend(xs); lats.extend(ys)
        else:
            lons.extend([G.nodes[u]["x"], G.nodes[v]["x"]]); lats.extend([G.nodes[u]["y"], G.nodes[v]["y"]])
    return np.array(lats), np.array(lons)


def _run_drive(name: str, root: str):
    """Phase 7 benchmark for one drive (runs in a worker process)."""
    import json
    import warnings
    import zlib

    import torch

    from src.dead_reckoning.heading import path_outage_starts
    from src.dead_reckoning.strapdown_ins import load_truth
    from src.io_vnbd_loader import _gps_fix_index
    from src.map_matching.hmm_map_matcher import HMMMapMatcher
    from src.map_matching.osm_loader import load_road_graph, to_local_xy
    from src.models.multi_drive_data import load_raw_drive
    from src.models.train_velocity_model import load_checkpoint

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)  # phone-like, and one core per worker
    root = Path(root)
    ckdir = root / "results" / "checkpoints"
    ckpt = load_checkpoint(ckdir / "velocity_cnn_best.pt", "cpu")
    meta = {"feature_cols": ckpt["feature_cols"], "window_size": ckpt["window_size"]}
    split = json.loads((root / "data" / "split.json").read_text(encoding="utf-8"))
    L_all = (30, 60, 120)
    end_rows, track_rows, route_rows = [], [], []
    try:
        df = load_raw_drive(root / "data" / "raw" / "real" / name)
    except Exception:  # noqa: BLE001
        return end_rows, track_rows, route_rows
    if not 0.07 <= float(np.median(np.diff(df["t_sec"]))) <= 0.13:
        return end_rows, track_rows, route_rows
    truth, _ = load_truth(df, name, root)
    v_true = df["gps_speed_interp_ms"].to_numpy() if truth is None else truth
    starts = path_outage_starts(df, L_all, 15, v_true, zlib.crc32(name.encode()))
    if len(starts) == 0:
        return end_rows, track_rows, route_rows
    G = load_road_graph(name, cache_dir=root / "data" / "external_maps")
    matcher = HMMMapMatcher(G, df)
    t = df["t_sec"].to_numpy(dtype=float)
    lat, lon = df["gps_lat_deg"].to_numpy(dtype=float), df["gps_lon_deg"].to_numpy(dtype=float)
    fix = _gps_fix_index(df)
    sp = "test" if name in split["test"] else "val"
    # parity variants on TEST drives only (they are compared with the Phase 6 batch run)
    variants = [("int8_map", "velocity_model_int8.ptl", True)]
    if sp == "test":
        variants = [("fp32_nomap", "velocity_model_fp32.ptl", False), ("fp32_map", "velocity_model_fp32.ptl", True)] + variants
    for s in starts:
        ends = [int(fix[np.searchsorted(t[fix], t[s] + L)]) for L in L_all]
        e_max = max(ends)
        r0 = max(0, s - 6000)
        mask = (s, e_max)  # fixes strictly inside withheld; the exit fix at e_max arrives
        hidden = [int(f) for f in fix if s < f < e_max]
        drv = [int(f) for f in fix if r0 <= f <= e_max]
        mons = {"correct": RouteMonitor(lat[drv], lon[drv])}
        in_out_fix = [int(f) for f in fix if s <= f <= e_max]
        wr = wrong_route(G, lat[in_out_fix], lon[in_out_fix])
        wr_ok, dev_t = False, None
        if wr is not None and len(wr[0]) >= 2:
            # the plan matches what was driven up to the tunnel, then takes the other road
            pre = [int(f) for f in fix if r0 <= f < s]
            m_wr = RouteMonitor(np.r_[lat[pre], wr[0]], np.r_[lon[pre], wr[1]])
            truth_dist = [m_wr.distance_m(lat[f], lon[f]) for f in hidden + [e_max]]
            if max(truth_dist) > 100.0:
                mons["wrong"], wr_ok = m_wr, True
                dev_t = next(t[f] for f, d in zip(hidden + [e_max], truth_dist) if d > 40.0)
        for variant, model_file, use_map in variants:
            eng = IDREngine(ckdir / model_file, meta, map_matcher=matcher if use_map else None, start_index=r0)
            outs, mon = replay(df, eng, r0, e_max, mask, monitors=mons if variant == "int8_map" else None)
            base = {"drive": name, "split": sp, "start_idx": int(s), "variant": variant}
            for L, e in zip(L_all, ends):
                o = outs[e - r0]
                p_ = o.exit_realtime if (e == e_max and o.exit_realtime is not None) else (o.lat, o.lon)
                ex, ey = to_local_xy(p_[0], p_[1], lat[e], lon[e])
                end_rows.append({**base, "outage_s": L, "abs_err_m": float(np.hypot(ex, ey)),
                                 "status": outs[e - r0 - 1].status})
            if variant != "int8_map":
                continue
            trk = outs[e_max - r0].track_corrected
            if trk:
                tt = np.array([q[0] for q in trk])
                for f in hidden:
                    j = int(np.argmin(np.abs(tt - t[f])))
                    dx, dy = to_local_xy(trk[j][1], trk[j][2], lat[f], lon[f])
                    o = outs[f - r0]
                    rx, ry = to_local_xy(o.lat, o.lon, lat[f], lon[f])
                    track_rows.append({**base, "method": "realtime", "abs_err_m": float(np.hypot(rx, ry))})
                    track_rows.append({**base, "method": "rts_corrected", "abs_err_m": float(np.hypot(dx, dy))})
            in_out = [st for tt_, st, _ in mon["correct"] if t[s] < tt_ <= t[e_max]]
            route_rows.append({**base, "route": "correct", "flagged": "OFF_ROUTE" in in_out, "delay_s": np.nan})
            if wr_ok:
                det = next((tt_ for tt_, st, _ in mon["wrong"] if t[s] < tt_ <= t[e_max] and st == "OFF_ROUTE"), None)
                # valid only if the alarm is not raised before the car actually left the route (5 s tolerance)
                valid = det is not None and det >= dev_t - 5.0
                route_rows.append({**base, "route": "wrong", "flagged": valid, "premature": det is not None and not valid,
                                   "delay_s": (det - dev_t) if valid else np.nan})
    return end_rows, track_rows, route_rows


if __name__ == "__main__":
    import json
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor

    root = Path(__file__).resolve()
    while not ((root / "src").is_dir() and (root / "data").is_dir()):
        root = root.parent
    sys.path.insert(0, str(root))
    from src.dead_reckoning.strapdown_ins import log_metric

    split = json.loads((root / "data" / "split.json").read_text(encoding="utf-8"))
    names = split["test"] + split["val"]
    L_all = (30, 60, 120)
    print(f"\n{'=' * 70}\nPhase 7 — streaming edge engine replay on the Days 10-13 outages (TEST + VAL)\n{'=' * 70}")
    t_start = time.time()
    end_rows, track_rows, route_rows = [], [], []
    with ProcessPoolExecutor(max_workers=min(8, max(1, (os.cpu_count() or 2) - 2))) as pool:
        for name, (a, b, c) in zip(names, pool.map(_run_drive, names, [str(root)] * len(names))):
            end_rows += a; track_rows += b; route_rows += c
            if a:
                print(f"  {name:<12} {len({r['start_idx'] for r in a}):>3} outages  ({time.time() - t_start:.0f} s)", flush=True)

    ER, TR, RR = pd.DataFrame(end_rows), pd.DataFrame(track_rows), pd.DataFrame(route_rows)
    ER.to_csv(root / "results" / "phase7_engine_end.csv", index=False)
    TR.to_csv(root / "results" / "phase7_engine_track.csv", index=False)
    RR.to_csv(root / "results" / "phase7_route_verification.csv", index=False)
    batch = pd.read_csv(root / "results" / "phase6_end_benchmark.csv")

    def med(df_, L, **kw):
        sub = df_[df_.outage_s == L]
        for k, v in kw.items():
            sub = sub[sub[k] == v]
        return float(sub.abs_err_m.median()), float(sub.abs_err_m.mean()), len(sub)

    section = "Day 15 — streaming edge engine replay (same outages)"
    for sp, label, tag in (("test", "TEST", ""), (None, "TEST + VAL", " [test+val]")):
        er = ER if sp is None else ER[ER.split == sp]
        bt = batch if sp is None else batch[batch.split == sp]
        print(f"\n1-2. Real-time end error, {label} ({er.drive.nunique()} drives) — median m")
        print(f"{'Method':<44} {'30 s':>8} {'60 s':>8} {'120 s':>8}")
        rows = [("Phase 6 batch EKF (torch fp32)", bt, {"method": "ekf"}, False),
                ("Engine, fp32 .ptl, no map", er, {"variant": "fp32_nomap"}, True),
                ("Phase 6 batch EKF + map", bt, {"method": "ekf_map_route"}, False),
                ("Engine, fp32 .ptl + map", er, {"variant": "fp32_map"}, True),
                ("Engine, int8 .ptl + map (deployed)", er, {"variant": "int8_map"}, True)]
        for lab, d_, kw, log in rows:
            if sp is None and kw.get("variant", "").startswith("fp32"):
                continue  # parity variants ran on TEST drives only
            vals = [med(d_, L, **kw) for L in L_all]
            if vals[0][2] == 0:
                continue
            print(f"{lab:<44} " + " ".join(f"{v[0]:>8.1f}" for v in vals))
            if log:
                for L, (m_, mean_, n_) in zip(L_all, vals):
                    log_metric(root / "results", f"{lab}{tag}, {L}s outage",
                               {"outage_s": L, "median_abs_err_m": m_, "median_pct_err": float("nan"),
                                "mean_abs_err_m": mean_, "n_outages": n_, "n_drives": er.drive.nunique()},
                               day="Day 15", section_title=section)
    print("\n2. Tunnel track after the 120 s outage (hidden fixes), int8 + map — median m")
    for sp, label in (("test", "TEST"), (None, "TEST + VAL")):
        tr = TR if sp is None else TR[TR.split == sp]
        print(f"   {label:<11} real time {tr[tr.method == 'realtime'].abs_err_m.median():6.1f}   "
              f"RTS corrected {tr[tr.method == 'rts_corrected'].abs_err_m.median():6.1f}   (N {len(tr) // 2})")
    print("\n3. Route verification during the outage (int8 + map)")
    for sp, label in (("test", "TEST"), (None, "TEST + VAL")):
        rr = RR if sp is None else RR[RR.split == sp]
        c, w = rr[rr.route == "correct"], rr[rr.route == "wrong"]
        print(f"   {label:<11} false alarms on the correct route: {int(c.flagged.sum())} / {len(c)}   "
              f"wrong route detected: {int(w.flagged.sum())} / {len(w)}   "
              f"median delay after the car left it: {w.delay_s.median():.1f} s")
    print(f"\nTotal {time.time() - t_start:.0f} s. Rows in results/phase7_*.csv; metrics.md Day 15 section.")
