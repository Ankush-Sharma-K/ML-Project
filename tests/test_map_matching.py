"""Coordinates, the OSM road filter, and the HMM matcher on a synthetic T-junction."""

import networkx as nx
import numpy as np

from conftest import LAT0, LON0, synthetic_drive
from src.map_matching.hmm_map_matcher import HMMMapMatcher, from_local_xy
from src.map_matching.osm_loader import _osmnx_tag_filter, edge_segments, to_local_xy


def test_local_xy_round_trip():
    lat, lon = np.array([52.51, 52.49]), np.array([-1.49, -1.52])
    x, y = to_local_xy(lat, lon, LAT0, LON0)
    lat2, lon2 = from_local_xy(x, y, LAT0, LON0)
    assert np.allclose(lat, lat2) and np.allclose(lon, lon2)
    assert abs(y[0] - 0.01 * 111_320.0) < 1e-6


def test_drive_filter_matches_osmnx_rules():
    ok = _osmnx_tag_filter("drive")
    assert ok({"highway": "residential"})
    assert ok({"highway": "motorway", "oneway": "yes"})
    assert not ok({"highway": "footway"})
    assert not ok({"highway": "service"})
    assert not ok({"highway": "residential", "access": "private"})
    assert not ok({"building": "yes"})


def _t_junction_graph():
    """West-east road (0..1000 m) meeting a north-south road at x = 1000 m;
    all two-way. Node positions in lon/lat like an osmnx graph."""
    pts = {1: (0, 0), 2: (1000, 0), 3: (1000, 800), 4: (1000, -800)}
    G = nx.MultiDiGraph(crs="epsg:4326", corridor_buffer_m=2000.0)
    for n, (x, y) in pts.items():
        lat, lon = from_local_xy(x, y, LAT0, LON0)
        G.add_node(n, x=float(lon), y=float(lat))
    for a, b in ((1, 2), (2, 3), (2, 4)):
        L = float(np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]))
        G.add_edge(a, b, key=0, length=L, oneway=False)
        G.add_edge(b, a, key=0, length=L, oneway=False)
    return G


def test_edge_segments_lengths():
    seg = edge_segments(_t_junction_graph())
    assert len(seg) == 6
    assert np.allclose(sorted(seg.edge_len_m.unique()), [800.0, 1000.0], rtol=1e-3)


def test_hmm_follows_the_turn_despite_heading_drift():
    """Car drives east from x = 200 m to the junction, turns NORTH, drives 400 m.
    Dead reckoning uses the true speed but a heading that drifts by 0.1 deg/s;
    the matched route must end on the northern road near the true position."""
    v = 10.0
    t_east, t_north = 80.0, 40.0
    n1, n2 = int(t_east * 10), int(t_north * 10)
    x = np.r_[200 + v * np.arange(n1) / 10, np.full(n2, 1000.0)]
    y = np.r_[np.zeros(n1), v * np.arange(n2) / 10]
    df = synthetic_drive(x, y)
    true_heading = np.r_[np.full(n1, np.pi / 2), np.zeros(n2)]
    t = df["t_sec"].to_numpy()
    heading = true_heading + np.radians(0.1) * t  # drifting gyro
    speed = np.full(len(df), v)
    rr = np.arange(len(df))
    end = int(rr[-1])

    mm = HMMMapMatcher(_t_junction_graph(), df)
    out = mm.match_outage(0, rr, speed, heading, [end])[end]
    assert out is not None
    ex, ey = to_local_xy(df["gps_lat_deg"].iloc[end], df["gps_lon_deg"].iloc[end], df["gps_lat_deg"].iloc[0],
                         df["gps_lon_deg"].iloc[0])
    rx, ry = out["route"]
    assert np.hypot(rx - ex, ry - ey) < 20.0
    assert abs((rx + 200.0) - 1000.0) < 5.0  # on the north-south road (x = 1000 m)

    # plain dead reckoning with the same drifting heading ends much further off
    d = np.r_[0.0, speed[1:] * np.diff(t)]
    dx, dy = np.sum(d * np.sin(heading)), np.sum(d * np.cos(heading))
    assert np.hypot(dx - ex, dy - ey) > 2 * np.hypot(rx - ex, ry - ey)


def test_hmm_returns_none_when_entrance_is_off_road():
    x = np.full(300, 500.0)
    y = np.linspace(400, 700, 300)  # 400+ m north of the west-east road, nowhere near a road
    df = synthetic_drive(x, y)
    mm = HMMMapMatcher(_t_junction_graph(), df)
    rr = np.arange(len(df))
    out = mm.match_outage(0, rr, np.full(len(df), 10.0), np.zeros(len(df)), [int(rr[-1])])
    assert out[int(rr[-1])] is None
