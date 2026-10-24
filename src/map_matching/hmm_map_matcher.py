"""
hmm_map_matcher.py — Phase 5 (Days 11-12): snap the dead-reckoned path to the road network
======================================================================================

Day 10 ends with an east/north path from anchored-ML speed + calibrated gyro
heading. Even with perfect distance a 2-minute outage still ends ~440 m off,
because small heading errors grow into large sideways errors. A car stays on
roads, so the path is matched to the OSM road graph (`osm_loader.py`) with a
Hidden Markov Model (Newson & Krumm 2009, adapted to dead reckoning):

  states       candidate positions on DIRECTED road edges near each sample
               (two-way roads are two directed edges, one-way roads one)
  samples      the dead-reckoned (DR) path every `sample_s` seconds
  emission     distance DR point -> road, Gaussian with sigma growing with
               the distance driven since the entrance (DR drifts), times
               the agreement of the road direction with the gyro heading
  transition   |route distance on the graph - DR distance driven| -- the
               car must drive along the roads between two candidates --
               plus (Day 12) turn consistency: the change of road direction
               between two candidates vs the gyro heading change (sigma 20 deg)
  entrance     the last GPS fix and course (real, before the outage)

The best sequence is found with Viterbi. Matching is CAUSAL: the position at
time t uses only samples up to t (the best final state of the forward pass at
that sample, backtracked from there).

Two outputs per outage end:
  snap   the matched road position of the last sample
  route  start from the entrance on the matched route and move the DR
         distance along it (uses the speed estimate, which is better than the
         drifting DR position along the road)

LEAK GUARD: the road corridor is cut around the whole drive (see
`osm_loader.py`). For each sample the matcher checks that its whole search
disc lies inside the corridor with margin; if not, the map is not used for
that outage end (`None`, the caller falls back to plain dead reckoning).

DAY 12 FINDINGS (tuned on TRAIN drives only, true speed and speed x0.85 / x1.15):
turn consistency helps a little at 120 s (194 -> 183 m) and nothing else, so it
is on by default. The "turn" output (DR distance counted from the last completed
turn instead of the entrance) applies in only 12-23% of outages and helps only
when the speed is biased -- computed, not adopted. With a 15% speed error every
variant ends ~430 m off at 120 s: the map fixes direction, not distance.

`__main__` = the Day 12 benchmark (same outages as Days 10-11).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.map_matching.osm_loader import (DEFAULT_BUFFER_M, M_PER_DEG_LAT, clean_gps_track, edge_segments,
                                         to_local_xy)


def wrap_angle(a):
    """Wrap radians to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def from_local_xy(x, y, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `to_local_xy`."""
    return lat0 + np.asarray(y) / M_PER_DEG_LAT, lon0 + np.asarray(x) / (np.cos(np.radians(lat0)) * M_PER_DEG_LAT)


class HMMMapMatcher:
    """Road graph + drive context for matching outages of ONE drive.

    `G` = osmnx graph from `osm_loader.load_road_graph`, `df` = the loaded
    drive (for time stamps, the GPS entrance and the corridor guard)."""

    def __init__(self, G, df: pd.DataFrame, sample_s: float = 5.0, sigma0_m: float = 15.0,
                 sigma_growth: float = 0.08, max_radius_m: float = 600.0, sigma_heading_deg: float = 30.0,
                 beta_m: float = 30.0, beta_frac: float = 0.2, max_candidates: int = 10,
                 entry_radius_m: float = 50.0, corridor_slack_m: float = 200.0,
                 turn_sigma_deg: float | None = 20.0, anchor_turn_deg: float = 45.0, anchor_window_s: float = 15.0):
        from scipy.spatial import cKDTree

        self.G = G
        self.sample_s, self.sigma0_m, self.sigma_growth = sample_s, sigma0_m, sigma_growth
        self.max_radius_m, self.sigma_heading = max_radius_m, np.radians(sigma_heading_deg)
        self.beta_m, self.beta_frac, self.max_candidates = beta_m, beta_frac, max_candidates
        self.entry_radius_m = entry_radius_m
        self.buffer_m = float(G.graph.get("corridor_buffer_m", DEFAULT_BUFFER_M))
        self.corridor_slack_m = corridor_slack_m
        self.turn_sigma = None if turn_sigma_deg is None else np.radians(turn_sigma_deg)
        self.anchor_turn, self.anchor_window_s = np.radians(anchor_turn_deg), anchor_window_s

        seg = edge_segments(G)  # pieces of one edge are consecutive rows
        new_edge = np.r_[True, (seg.u.to_numpy()[1:] != seg.u.to_numpy()[:-1])
                         | (seg.v.to_numpy()[1:] != seg.v.to_numpy()[:-1])
                         | (seg.key.to_numpy()[1:] != seg.key.to_numpy()[:-1])]
        self.seg_edge = np.cumsum(new_edge) - 1
        first = np.flatnonzero(new_edge)
        self.edge_first = first
        self.edge_count = np.diff(np.r_[first, len(seg)])
        self.edge_u = seg.u.to_numpy()[first]
        self.edge_v = seg.v.to_numpy()[first]
        self.edge_len = seg.edge_len_m.to_numpy()[first]
        self.seg_lat0, self.seg_lon0 = seg.lat0.to_numpy(), seg.lon0.to_numpy()
        self.seg_lat1, self.seg_lon1 = seg.lat1.to_numpy(), seg.lon1.to_numpy()
        self.seg_off = seg.offset_m.to_numpy()

        # drive-level frame, only for spatial queries (padded generously)
        self.lat_ref, self.lon_ref = float(np.median(self.seg_lat0)), float(np.median(self.seg_lon0))
        ax, ay = to_local_xy(self.seg_lat0, self.seg_lon0, self.lat_ref, self.lon_ref)
        bx, by = to_local_xy(self.seg_lat1, self.seg_lon1, self.lat_ref, self.lon_ref)
        self.seg_tree = cKDTree(np.c_[(ax + bx) / 2, (ay + by) / 2])
        self.max_half = float(np.max(np.hypot(bx - ax, by - ay)) / 2)

        tlat, tlon = clean_gps_track(df)
        tx, ty = to_local_xy(tlat, tlon, self.lat_ref, self.lon_ref)
        self.track_tree = cKDTree(np.c_[tx, ty])

        self.t = df["t_sec"].to_numpy(dtype=float)
        self.lat = df["gps_lat_deg"].to_numpy(dtype=float)
        self.lon = df["gps_lon_deg"].to_numpy(dtype=float)

    # ------------------------------------------------------------------ geometry
    def _local_segments(self, lat0, lon0, radius):
        """Road pieces within `radius` of (lat0, lon0), in metres around it."""
        cx, cy = to_local_xy(lat0, lon0, self.lat_ref, self.lon_ref)
        idx = np.array(self.seg_tree.query_ball_point([float(cx), float(cy)], radius * 1.02 + self.max_half), dtype=int)
        ax, ay = to_local_xy(self.seg_lat0[idx], self.seg_lon0[idx], lat0, lon0)
        bx, by = to_local_xy(self.seg_lat1[idx], self.seg_lon1[idx], lat0, lon0)
        return {"edge": self.seg_edge[idx], "off": self.seg_off[idx], "ax": ax, "ay": ay, "bx": bx, "by": by,
                "len": np.hypot(bx - ax, by - ay), "heading": np.arctan2(bx - ax, by - ay)}

    def _candidates(self, S, px, py, heading, sigma, radius, sigma_heading=None):
        """Best piece per directed edge within `radius` of (px, py): arrays
        edge, offset along edge, emission log-likelihood (top `max_candidates`)."""
        sh = self.sigma_heading if sigma_heading is None else sigma_heading
        abx, aby = S["bx"] - S["ax"], S["by"] - S["ay"]
        l2 = np.maximum(abx ** 2 + aby ** 2, 1e-9)
        u = np.clip(((px - S["ax"]) * abx + (py - S["ay"]) * aby) / l2, 0.0, 1.0)
        dist = np.hypot(px - (S["ax"] + u * abx), py - (S["ay"] + u * aby))
        ll = -0.5 * (dist / sigma) ** 2 - 0.5 * (wrap_angle(S["heading"] - heading) / sh) ** 2
        ok = np.flatnonzero(dist <= radius)
        if len(ok) == 0:
            return None
        ok = ok[np.argsort(-ll[ok])]
        _, first = np.unique(S["edge"][ok], return_index=True)
        best = ok[np.sort(first)][: self.max_candidates]
        return {"edge": S["edge"][best], "off": np.minimum(S["off"][best] + u[best] * S["len"][best],
                                                            self.edge_len[S["edge"][best]]),
                "ll": ll[best], "hdg": S["heading"][best]}

    def _point_on_edge(self, e, off):
        """lat/lon at distance `off` along edge `e`."""
        i0, n = self.edge_first[e], self.edge_count[e]
        offs = self.seg_off[i0:i0 + n]
        j = i0 + max(int(np.searchsorted(offs, off, side="right")) - 1, 0)
        ax, ay = to_local_xy(self.seg_lat0[j], self.seg_lon0[j], self.seg_lat0[j], self.seg_lon0[j])
        bx, by = to_local_xy(self.seg_lat1[j], self.seg_lon1[j], self.seg_lat0[j], self.seg_lon0[j])
        seg_len = max(float(np.hypot(bx, by)), 1e-9)
        f = float(np.clip((off - self.seg_off[j]) / seg_len, 0.0, 1.0))
        return (self.seg_lat0[j] + f * (self.seg_lat1[j] - self.seg_lat0[j]),
                self.seg_lon0[j] + f * (self.seg_lon1[j] - self.seg_lon0[j]))

    # ------------------------------------------------------------------ routing
    def _route_dist(self, ea, oa, eb, ob, cache, cutoff):
        """Driving distance from (edge ea, offset oa) to (edge eb, offset ob); inf if unreachable."""
        import networkx as nx

        if ea == eb and ob >= oa - 20.0:
            return max(ob - oa, 0.0)
        src = self.edge_v[ea]
        if src not in cache:
            cache[src] = nx.single_source_dijkstra_path_length(self.G, src, cutoff=cutoff, weight="length")
        d = cache[src].get(self.edge_u[eb])
        return np.inf if d is None else (self.edge_len[ea] - oa) + d + ob

    # ------------------------------------------------------------------ matching
    def match_outage(self, s: int, rr: np.ndarray, speed: np.ndarray, heading: np.ndarray, ends) -> dict:
        """Match one outage. `s` = entrance row (a GPS fix), `rr` = rows from s,
        `speed` (m/s) / `heading` (rad, clockwise from north) per row of `rr`,
        `ends` = rows at which a position is wanted.
        Returns {end_row: {"snap": (dx, dy), "route": (dx, dy)} or None},
        displacements in metres east/north of the entrance fix."""
        t = self.t[rr]
        dt = np.clip(np.diff(t, prepend=t[0]), 0.0, 1.0)
        d = np.maximum(speed, 0.0) * dt
        d[0] = 0.0
        x, y, dist = np.cumsum(d * np.sin(heading)), np.cumsum(d * np.cos(heading)), np.cumsum(d)
        lat0, lon0 = float(self.lat[s]), float(self.lon[s])
        out = {int(e): None for e in ends}

        ks = set(np.searchsorted(t - t[0], np.arange(0.0, t[-1] - t[0], self.sample_s)).tolist())
        ks |= {int(e) - int(rr[0]) for e in ends} | {0}
        ks = np.array(sorted(k for k in ks if k < len(rr)))
        sigma = self.sigma0_m + self.sigma_growth * dist[ks]
        radius = np.minimum(3.0 * sigma, self.max_radius_m)
        radius[0] = self.entry_radius_m

        # leak guard: search disc inside the corridor (with slack for polygon simplification)
        qlat, qlon = from_local_xy(x[ks], y[ks], lat0, lon0)
        qx, qy = to_local_xy(qlat, qlon, self.lat_ref, self.lon_ref)
        margin_ok = self.track_tree.query(np.c_[qx, qy])[0] + radius <= self.buffer_m - self.corridor_slack_m
        guard_ok = np.cumprod(margin_ok).astype(bool)

        S = self._local_segments(lat0, lon0, float(np.max(np.hypot(x[ks], y[ks]) + radius)) + 50.0)
        if len(S["edge"]) == 0:
            return out
        c0 = self._candidates(S, 0.0, 0.0, heading[0], self.sigma0_m, radius[0])
        if c0 is None or not guard_ok[0]:
            return out
        cands, deltas, backs, route_d = [c0], [c0["ll"]], [None], [None]
        end_k = {int(e) - int(rr[0]): int(e) for e in ends}

        for j in range(1, len(ks)):
            if not guard_ok[j]:
                break
            dd = dist[ks[j]] - dist[ks[j - 1]]
            c = self._candidates(S, x[ks[j]], y[ks[j]], heading[ks[j]], sigma[j], radius[j])
            if c is None:  # no road nearby: carry the previous belief over this sample
                cands.append(cands[-1]); deltas.append(deltas[-1])
                backs.append(np.arange(len(cands[-1]["edge"]))); route_d.append(np.full(len(cands[-1]["edge"]), dd))
            else:
                prev, cache = cands[-1], {}
                cutoff = 2.0 * dd + 300.0
                R = np.array([[self._route_dist(ea, oa, eb, ob, cache, cutoff)
                               for ea, oa in zip(prev["edge"], prev["off"])]
                              for eb, ob in zip(c["edge"], c["off"])])  # (n_b, n_a)
                beta = max(self.beta_m, self.beta_frac * dd)
                score = deltas[-1][None, :] - np.abs(R - dd) / beta
                if self.turn_sigma is not None:  # road turn between candidates vs gyro turn between samples
                    d_road = wrap_angle(c["hdg"][:, None] - prev["hdg"][None, :])
                    d_gyro = heading[ks[j]] - heading[ks[j - 1]]
                    score = score - 0.5 * (wrap_angle(d_road - d_gyro) / self.turn_sigma) ** 2
                if not np.any(np.isfinite(score)):  # graph break: restart from the best previous state
                    score = np.full_like(R, -np.inf)
                    score[:, int(np.argmax(deltas[-1]))] = np.max(deltas[-1])
                    R = np.where(np.isfinite(R), R, dd)
                b = np.argmax(score, axis=1)
                cands.append(c); deltas.append(score[np.arange(len(b)), b] + c["ll"])
                backs.append(b); route_d.append(R[np.arange(len(b)), b])

            if ks[j] in end_k:
                out[end_k[ks[j]]] = self._result(cands, deltas, backs, route_d, dist[ks[:j + 1]],
                                                 self._anchor_sample(heading, t - t[0], ks[:j + 1]), lat0, lon0)
        return out

    def _anchor_sample(self, heading, tau, ks) -> int:
        """Last sample index j right after a completed turn (gyro heading change
        >= anchor_turn over the preceding anchor_window_s, < 10 deg over the
        following sample), or 0 (the entrance) if there is none."""
        h = np.interp(tau[ks], tau, heading)
        h_before = np.interp(tau[ks] - self.anchor_window_s, tau, heading)
        turned = np.abs(h - h_before) >= self.anchor_turn
        settled = np.r_[np.abs(np.diff(h)) < np.radians(10.0), False]
        js = np.flatnonzero(turned & settled)
        return int(js[-1]) if len(js) else 0

    def _result(self, cands, deltas, backs, route_d, dist_k, j_anchor, lat0, lon0):
        """snap / route (DR distance from the entrance) / turn (DR distance from
        the last completed turn, whose matched position pins the along-road error)."""
        i = int(np.argmax(deltas[-1]))
        e, off = int(cands[-1]["edge"][i]), float(cands[-1]["off"][i])
        route_len, route_since_anchor = 0.0, None
        for j in range(len(cands) - 1, 0, -1):
            if j == j_anchor:
                route_since_anchor = route_len
            route_len += float(route_d[j][i])
            i = int(backs[j][i])
        if route_since_anchor is None:  # no turn: anchored at the entrance
            route_since_anchor = route_len
        res = {"snap": off,
               "route": off + dist_k[-1] - route_len,
               "turn": off + dist_k[-1] - dist_k[j_anchor] - route_since_anchor}
        out = {}
        for kind, o in res.items():
            lat_p, lon_p = self._point_on_edge(e, float(np.clip(o, 0.0, self.edge_len[e])))
            x, y = to_local_xy(lat_p, lon_p, lat0, lon0)
            out[kind] = (float(x), float(y))
        return out


DAY12_LABELS_A = {"hold": "True distance + last GPS course (straight line)",
                  "gyro": "True distance + gyro heading (Day 10)",
                  "gyro+map_snap": "True distance + gyro + map (snap)",
                  "gyro+map_route": "True distance + gyro + map (route)"}
DAY12_LABELS_B = {"hold_speed+hold": "Hold last fix + last course (baseline)",
                  "anchored_ml+gyro": "Anchored ML + gyro heading (Day 10)",
                  "anchored_ml+gyro+map_snap": "Anchored ML + gyro + map (snap)",
                  "anchored_ml+gyro+map_route": "Anchored ML + gyro + map (route, full method)"}
DAY12_SECTION_A = "Day 12 — map matching + turn consistency, heading/path only: TRUE distance (all drives with a map)"
DAY12_SECTION_B = "Day 12 — map matching + turn consistency, full pipeline during simulated GPS outages (TEST drives)"
DAY12_SECTION_BV = "Day 12 — map matching + turn consistency, full pipeline (TEST + VAL drives, more outages)"


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
    from src.dead_reckoning.heading import simulate_path_outages
    from src.map_matching.osm_loader import load_road_graph

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(root / "results" / "checkpoints" / "velocity_cnn_best.pt", device)
    ml_methods = make_ml_methods(model, ckpt, device)
    split = json.loads((root / "data" / "split.json").read_text(encoding="utf-8"))
    test_drives, val_drives = set(split["test"]), set(split["val"])
    maps = root / "data" / "external_maps"

    print(f"\n{'=' * 70}\nDay 12 — HMM map matching (+ turn consistency) during simulated GPS outages\n"
          f"(same outages as Day 10; map used only when the search area is inside the corridor;\n"
          f" error = distance to the first GPS fix after the outage)\n{'=' * 70}")
    parts_a, parts_b = [], []
    for path in sorted((root / "data" / "raw" / "real").glob("S-*.csv")):
        if not (maps / f"{path.stem}.graphml").exists():
            continue
        try:
            df = load_raw_drive(path)
        except Exception:  # noqa: BLE001
            continue
        if not 0.07 <= float(np.median(np.diff(df["t_sec"]))) <= 0.13:
            continue
        matcher = HMMMapMatcher(load_road_graph(path.name, cache_dir=maps), df)
        truth, source = load_truth(df, path.name, root)
        with_ml = path.name in test_drives or path.name in val_drives
        a, b = simulate_path_outages(df, truth=truth, seed=zlib.crc32(path.name.encode()),
                                     ml_methods=ml_methods if with_ml else None,
                                     window_size=ckpt["window_size"], map_match=matcher.match_outage)
        if a.empty:
            continue
        a["drive"] = path.name
        parts_a.append(a)
        if not b.empty:
            b["drive"] = path.name
            b["split"] = "test" if path.name in test_drives else "val"
            parts_b.append(b)
        print(f"  {path.name:<12} {a.start_idx.nunique():>3} outages  truth={source}"
              f"{'  (+ full path, ' + b.split.iloc[0] + ')' if not b.empty else ''}")

    A = pd.concat(parts_a, ignore_index=True)
    B = pd.concat(parts_b, ignore_index=True) if parts_b else pd.DataFrame()
    A.to_csv(root / "results" / "day12_map_heading_benchmark.csv", index=False)
    if not B.empty:
        B.to_csv(root / "results" / "day12_map_path_benchmark.csv", index=False)

    mrows = A[A.method == "gyro+map_route"]
    print("\nMap usable (search area inside the corridor): " + ", ".join(
        f"{L}s {100 * mrows[mrows.outage_s == L].map_ok.astype(bool).mean():.0f}%" for L in sorted(mrows.outage_s.unique())))

    tables = [(A, DAY12_LABELS_A, f"PART A — true distance ({A.drive.nunique()} drives)", DAY12_SECTION_A, "")]
    if not B.empty:
        BT = B[B.split == "test"]
        tables += [(BT, DAY12_LABELS_B, f"PART B — full pipeline, TEST drives ({BT.drive.nunique()} drives)", DAY12_SECTION_B, ""),
                   (B, DAY12_LABELS_B, f"PART B' — full pipeline, TEST + VAL drives ({B.drive.nunique()} drives)", DAY12_SECTION_BV,
                    " [test+val]")]
    for res, labels, title, section, tag in tables:  # tag keeps labels unique: log_metric replaces by label
        if res.empty:
            continue
        print(f"\n{title}")
        print(f"{'Method':<48} {'Outage':>7} {'Median err (m)':>15} {'Median %':>9} {'P90 (m)':>8} {'N':>6}")
        print("-" * 100)
        for L in sorted(res.outage_s.unique()):
            for method, label in labels.items():
                sub = res[(res.outage_s == L) & (res.method == method)]
                if sub.empty:
                    continue
                m = {"outage_s": L, "median_abs_err_m": sub.abs_err_m.median(), "median_pct_err": sub.pct_err.median(),
                     "p90_abs_err_m": sub.abs_err_m.quantile(0.9), "mean_abs_err_m": sub.abs_err_m.mean(),
                     "n_outages": len(sub), "n_drives": sub.drive.nunique()}
                print(f"{label:<48} {L:>6}s {m['median_abs_err_m']:>15.1f} {m['median_pct_err']:>8.1f}% "
                      f"{m['p90_abs_err_m']:>8.0f} {len(sub):>6}")
                log_metric(root / "results", f"{label}{tag}, {L}s outage", m, day="Day 12", section_title=section)
            print()
    print("Summary rows written to results/metrics.md (re-running replaces them).")
