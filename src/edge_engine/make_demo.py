"""
make_demo.py — Phase 7: offline demo bundle for the mobile app
==============================================================

Writes real engine replays to `mobile_app/assets/demo/` so the app works as
static files without the server. Not cherry-picked: the FIRST outage start of
every TEST drive, 120 s, with the correct and the wrong planned route.

Run:  python -m src.edge_engine.make_demo
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

if __name__ == "__main__":
    import warnings

    root = Path(__file__).resolve()
    while not ((root / "src").is_dir() and (root / "data").is_dir()):
        root = root.parent
    sys.path.insert(0, str(root))
    from src.edge_engine.server import CACHE_DIR, list_drives, run_replay

    warnings.filterwarnings("ignore")
    out_dir = root / "mobile_app" / "assets" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    drives, files = [], []
    for d in list_drives():
        if d["split"] != "test":
            continue
        first = d["starts"][0]
        for route in ("correct", "wrong"):
            try:
                run_replay(d["drive"], first["start"], 120, "int8", route)
            except ValueError as ex:  # e.g. no alternative road for the wrong route
                print(f"  {d['drive']} {route}: skipped ({ex})")
                continue
            key = f"{Path(d['drive']).stem}_{first['start']}_120_int8_{route}.json"
            shutil.copy(CACHE_DIR / key, out_dir / key)
            files.append(key)
            print(f"  {key}")
        drives.append({**d, "starts": [first]})
    (out_dir / "index.json").write_text(json.dumps({"drives": drives, "files": files}, indent=1), encoding="utf-8")
    print(f"{len(files)} demo replays in {out_dir}")
