"""
osm_loader.py — Phase 5 (Day 11): OpenStreetMap road graph for each drive
=========================================================================

Map matching needs the drivable roads the car could be on. The drives span up
to ~150 km (S-T7, S-Vw4), so a bounding box would pull in far too much; this
module builds a CORRIDOR around each drive's GPS track and caches it as
GraphML in `data/external_maps/<drive>.graphml`.

SOURCES. (1) Geofabrik regional extracts (`.osm.pbf`, county level in
England), downloaded once to `data/external_maps/geofabrik/`; the drivable
ways are read with pyosmium using osmnx's own "drive" filter, so the result
matches what osmnx would fetch. (2) The Overpass API via osmnx, used only for
drives whose regions are large (France: whole regions of 230-370 MB), with
retries and back-off -- the public Overpass server rejected a bulk run of
~60 drives on Day 11 ("at capacity").

CORRIDOR WIDTH AND LEAKAGE. The corridor is cut around the whole GPS track,
including the parts that later become simulated outages. If it were narrow,
the map itself would tell the matcher where the car really went. With a wide
buffer (default 2 km) every road the matcher could reach from the dead-
reckoned path is present, as long as the dead-reckoned point stays well
inside the corridor -- `corridor_margin_m` lets the benchmark check that for
each outage and refuse to use the map when it does not hold.

GPS CLEANING. A few drives contain isolated junk fixes (e.g. lat 6.4 on a UK
drive); a fix is dropped when the implied speed to both neighbours exceeds
`max_jump_ms`, so one bad fix cannot stretch the corridor across a continent.

COORDINATES. The graph is stored in lat/lon. `to_local_xy` converts to metres
around an origin the caller picks (the outage entrance), so the flat-earth
scale error stays negligible over the few km of one outage.

`__main__` builds (or loads from cache) the corridor for every drive the
Day 10 benchmark uses (10 Hz drives) and prints its size.
"""

from __future__ import annotations

import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_320.0
DEFAULT_BUFFER_M = 2000.0
GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
OVERPASS_MIRRORS = ["https://maps.mail.ru/osm/tools/overpass/api", "https://overpass-api.de/api"]


def to_local_xy(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """lat/lon -> metres east/north of (lat0, lon0) (flat earth; fine over a few km)."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return (lon - lon0) * np.cos(np.radians(lat0)) * M_PER_DEG_LAT, (lat - lat0) * M_PER_DEG_LAT


def clean_gps_track(df: pd.DataFrame, max_jump_ms: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    """lat/lon of the GPS fixes with zero positions and isolated jumps removed."""
    from src.io_vnbd_loader import _gps_fix_index

    fix = _gps_fix_index(df)
    lat = df["gps_lat_deg"].to_numpy(dtype=float)[fix]
    lon = df["gps_lon_deg"].to_numpy(dtype=float)[fix]
    t = df["t_sec"].to_numpy(dtype=float)[fix]
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1.0) & (np.abs(lon) > 1e-3)
    lat, lon, t = lat[ok], lon[ok], t[ok]
    if len(lat) < 3:
        return lat, lon
    x, y = to_local_xy(lat, lon, float(np.median(lat)), float(np.median(lon)))
    v = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 0.1)
    fast_prev = np.r_[False, v > max_jump_ms]
    fast_next = np.r_[v > max_jump_ms, False]
    keep = ~(fast_prev & fast_next)
    return lat[keep], lon[keep]


def drive_corridor(df: pd.DataFrame, buffer_m: float = DEFAULT_BUFFER_M):
    """Polygon (lat/lon, shapely) = the cleaned GPS track buffered by `buffer_m`."""
    import geopandas as gpd
    import shapely

    lat, lon = clean_gps_track(df)
    if len(lat) < 2:
        raise ValueError("fewer than 2 usable GPS fixes")
    track = gpd.GeoSeries([shapely.LineString(np.c_[lon, lat])], crs=4326)
    utm = track.estimate_utm_crs()
    poly = track.to_crs(utm).simplify(50.0).buffer(buffer_m).simplify(100.0)
    return poly.to_crs(4326).iloc[0]


# ---------------------------------------------------------------- Geofabrik
def _geofabrik_index(gf_dir: Path):
    """Region features of the Geofabrik index (downloaded once), with shapely geometry."""
    import json
    import urllib.request

    from shapely.geometry import shape

    path = gf_dir / "index-v1.json"
    if not path.exists():
        gf_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GEOFABRIK_INDEX_URL, path)
    feats = [f for f in json.loads(path.read_text(encoding="utf-8"))["features"]
             if "pbf" in f["properties"].get("urls", {})]
    return [(f["properties"]["id"], f["properties"]["urls"]["pbf"], shape(f["geometry"])) for f in feats]


def geofabrik_regions(corridor, gf_dir: Path) -> list[tuple[str, str]]:
    """Smallest Geofabrik regions (id, pbf url) covering the corridor: regions
    intersecting it, minus any that contain a smaller one of them (drops
    'england', 'great-britain', ... when the counties are there)."""
    hits = [(rid, url, g) for rid, url, g in _geofabrik_index(gf_dir) if g.intersects(corridor)]
    keep = []
    for rid, url, g in hits:
        inner = [h for h in hits if h[2].area < g.area and g.intersection(h[2]).area > 0.9 * h[2].area]
        if not inner:
            keep.append((rid, url))
    return keep


def _remote_size_mb(url: str) -> float:
    import urllib.request

    req = urllib.request.Request(url, method="HEAD")
    return int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"]) / 1e6


def _download(url: str, path: Path) -> None:
    import urllib.request

    tmp = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, tmp)
    with open(tmp, "rb") as f:
        if b"OSMHeader" not in f.read(32):  # e.g. an HTML error or redirect page
            tmp.unlink()
            raise RuntimeError(f"{url} did not return an OSM PBF file")
    tmp.replace(path)


def _osmnx_tag_filter(network_type: str = "drive"):
    """osmnx's Overpass filter for `network_type` as a Python predicate on a tag dict."""
    import osmnx as ox

    rules = re.findall(r'\["([^"]+)"(?:(!?~)"([^"]*)")?\]', ox._overpass._get_network_filter(network_type))

    def ok(tags) -> bool:
        for key, op, pattern in rules:
            val = tags.get(key)
            if op == "":
                if val is None:
                    return False
            elif op == "!~" and val is not None and re.search(pattern, val):
                return False
            elif op == "~" and (val is None or not re.search(pattern, val)):
                return False
        return True

    return ok


def region_ways(region_id: str, url: str, gf_dir: Path, network_type: str = "drive") -> list:
    """Drivable ways of one region as (way_id, tags, node_ids, lon, lat), cached
    as a pickle next to the downloaded pbf."""
    import osmium

    cache = gf_dir / f"{region_id}-{network_type}-ways.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    pbf = gf_dir / f"{region_id}-latest.osm.pbf"
    if not pbf.exists():
        _download(url, pbf)
    ok = _osmnx_tag_filter(network_type)
    ways = []
    for w in osmium.FileProcessor(str(pbf)).with_locations().with_filter(osmium.filter.EntityFilter(osmium.osm.WAY)):
        if "highway" not in w.tags:
            continue
        tags = dict(w.tags)
        if not ok(tags):
            continue
        nodes = [(n.ref, n.lon, n.lat) for n in w.nodes if n.location.valid()]
        if len(nodes) < 2:
            continue
        ids, lon, lat = map(np.array, zip(*nodes))
        ways.append((w.id, tags, ids, lon, lat))
    cache.write_bytes(pickle.dumps(ways))
    return ways


def graph_from_ways(ways: list, corridor):
    """osmnx graph from the given ways, truncated to the corridor and simplified
    (same order of steps as osmnx.graph_from_polygon)."""
    import tempfile
    from xml.sax.saxutils import quoteattr

    import osmnx as ox
    import shapely

    w, s, e, n = corridor.bounds
    shapely.prepare(corridor)
    sel, seen = [], set()
    for way in ways:
        if way[0] in seen:
            continue
        lon, lat = way[3], way[4]
        box = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
        if box.any() and shapely.contains_xy(corridor, lon[box], lat[box]).any():
            seen.add(way[0])
            sel.append(way)
    if not sel:
        raise ValueError("no drivable roads inside the corridor")
    nodes = {}
    for way in sel:
        nodes.update(zip(way[2].tolist(), zip(way[4].tolist(), way[3].tolist())))
    with tempfile.NamedTemporaryFile("w", suffix=".osm", delete=False, encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6">\n')
        for nid, (lat, lon) in nodes.items():
            f.write(f'<node id="{nid}" lat="{lat:.7f}" lon="{lon:.7f}"/>\n')
        for wid, tags, ids, _, _ in sel:
            f.write(f'<way id="{wid}">')
            f.write("".join(f'<nd ref="{i}"/>' for i in ids.tolist()))
            f.write("".join(f"<tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items()))
            f.write("</way>\n")
        f.write("</osm>\n")
        xml = Path(f.name)
    try:
        G = ox.graph_from_xml(xml, simplify=False, retain_all=True)
    finally:
        xml.unlink(missing_ok=True)
    G = ox.truncate.truncate_graph_polygon(G, corridor)
    return ox.simplify_graph(G)


def _graph_from_overpass(corridor, network_type: str, retries: int = 5, wait_s: float = 60.0):
    """osmnx download with mirror rotation and exponential back-off."""
    import osmnx as ox

    for attempt in range(retries):
        ox.settings.overpass_url = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        try:
            return ox.graph_from_polygon(corridor, network_type=network_type, retain_all=True)
        except Exception:  # noqa: BLE001 -- server busy / timeout: wait and retry
            if attempt == retries - 1:
                raise
            time.sleep(wait_s * 2 ** attempt)


def load_road_graph(drive_name: str, df: pd.DataFrame | None = None, cache_dir: str | Path = "data/external_maps",
                    buffer_m: float = DEFAULT_BUFFER_M, network_type: str = "drive", max_region_mb: float = 150.0):
    """Drivable road graph (osmnx MultiDiGraph, lat/lon) around one drive.

    Loaded from `<cache_dir>/<drive>.graphml` when present; otherwise `df`
    (the loaded drive) is required to build the corridor. The graph is built
    from Geofabrik extracts when every region needed is <= `max_region_mb`,
    else downloaded from Overpass; then cached."""
    import osmnx as ox

    cache_dir = Path(cache_dir)
    path = cache_dir / f"{Path(drive_name).stem}.graphml"
    if path.exists():
        return ox.load_graphml(path)
    if df is None:
        raise FileNotFoundError(f"{path} not cached and no drive data given to build the corridor")
    corridor = drive_corridor(df, buffer_m)
    gf_dir = cache_dir / "geofabrik"
    regions = geofabrik_regions(corridor, gf_dir)
    local = [r for r in regions if (gf_dir / f"{r[0]}-{network_type}-ways.pkl").exists()
             or (gf_dir / f"{r[0]}-latest.osm.pbf").exists()]
    if regions and all(r in local or _remote_size_mb(r[1]) <= max_region_mb for r in regions):
        ways = [w for rid, url in regions for w in region_ways(rid, url, gf_dir, network_type)]
        G = graph_from_ways(ways, corridor)
        G.graph["map_source"] = "geofabrik:" + ",".join(r[0] for r in regions)
    else:
        G = _graph_from_overpass(corridor, network_type)
        G.graph["map_source"] = "overpass"
    G.graph["corridor_buffer_m"] = buffer_m
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)
    return G


def edge_segments(G) -> pd.DataFrame:
    """Every straight piece of every road edge, one row per piece:
    u, v, key (edge id), piece (index along the edge), lat0/lon0 -> lat1/lon1,
    offset_m (distance along the edge to the piece start), edge_len_m, oneway."""
    rows = []
    for u, v, key, d in G.edges(keys=True, data=True):
        if "geometry" in d:
            lon, lat = np.asarray(d["geometry"].xy)
        else:
            lon = np.array([G.nodes[u]["x"], G.nodes[v]["x"]])
            lat = np.array([G.nodes[u]["y"], G.nodes[v]["y"]])
        x, y = to_local_xy(lat, lon, float(lat[0]), float(lon[0]))
        seg_len = np.hypot(np.diff(x), np.diff(y))
        offset = np.r_[0.0, np.cumsum(seg_len)[:-1]]
        oneway = str(d.get("oneway", False)).lower() == "true"
        for i in range(len(seg_len)):
            rows.append((u, v, key, i, lat[i], lon[i], lat[i + 1], lon[i + 1], offset[i], float(seg_len.sum()), oneway))
    return pd.DataFrame(rows, columns=["u", "v", "key", "piece", "lat0", "lon0", "lat1", "lon1",
                                       "offset_m", "edge_len_m", "oneway"])


def corridor_margin_m(df: pd.DataFrame, lat, lon) -> np.ndarray:
    """Distance (m) from each query point to the nearest cleaned GPS fix of the
    drive. A point at distance d is inside a corridor of buffer B with margin
    B - d (up to the polygon simplification, < 150 m)."""
    from scipy.spatial import cKDTree

    tlat, tlon = clean_gps_track(df)
    lat0, lon0 = float(np.median(tlat)), float(np.median(tlon))
    tx, ty = to_local_xy(tlat, tlon, lat0, lon0)
    qx, qy = to_local_xy(lat, lon, lat0, lon0)
    d, _ = cKDTree(np.c_[tx, ty]).query(np.c_[np.atleast_1d(qx), np.atleast_1d(qy)])
    return d


if __name__ == "__main__":
    import time
    import warnings

    root = Path(__file__).resolve()
    while not ((root / "src").is_dir() and (root / "data").is_dir()):
        root = root.parent
    sys.path.insert(0, str(root))
    from src.models.multi_drive_data import load_raw_drive

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cache = root / "data" / "external_maps"
    print(f"\n{'=' * 70}\nPhase 5 — OSM road corridors ({DEFAULT_BUFFER_M:.0f} m) for the 10 Hz drives\n"
          f"cache: {cache}\n{'=' * 70}")
    # optional: --skip S-T2,S-T3 (Day 11 skipped 7 French train drives: slow Overpass)
    # or --only S-T6,S-T7 (run the slow Overpass drives in a separate process)
    skip = set(sys.argv[sys.argv.index("--skip") + 1].split(",")) if "--skip" in sys.argv else set()
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    # --max-region-mb 500: use Geofabrik also for large regions (French drives, Day 12)
    max_mb = float(sys.argv[sys.argv.index("--max-region-mb") + 1]) if "--max-region-mb" in sys.argv else 150.0
    n_ok = 0
    for path in sorted((root / "data" / "raw" / "real").glob("S-*.csv")):
        if path.stem in skip or (only is not None and path.stem not in only):
            continue
        try:
            df = load_raw_drive(path)
        except Exception:  # noqa: BLE001
            continue
        if not 0.07 <= float(np.median(np.diff(df["t_sec"]))) <= 0.13:
            continue
        cached = (cache / f"{path.stem}.graphml").exists()
        t0 = time.time()
        try:
            G = load_road_graph(path.name, df, cache, max_region_mb=max_mb)
        except Exception as e:  # noqa: BLE001
            print(f"  {path.name:<12} FAILED: {type(e).__name__}: {e}")
            continue
        n_ok += 1
        print(f"  {path.name:<12} {len(G.nodes):>7} nodes {len(G.edges):>7} edges  "
              f"{'cached' if cached else f'built in {time.time() - t0:.0f} s'}  ({G.graph.get('map_source', 'overpass')})")
    print(f"\n{n_ok} road graphs ready in {cache}")
