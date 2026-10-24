"""
main.py — single entry point for the IDR project
=================================================

    python main.py app            start the edge-engine server + mobile web app (http://localhost:8000)
    python main.py test           run the unit tests
    python main.py train          train the velocity model on the real drives (run_training.py)
    python main.py compress       export the on-device fp32 / int8 models
    python main.py baselines      simple models (ridge, gradient boosting) vs the CNN
    python main.py maps           build / load the OSM road corridors
    python main.py bench <name>   re-run a benchmark: day8 | day9 | day10 | day11-12 | day13 | day15
    python main.py demo-bundle    regenerate the app's offline demo replays

Every command runs the corresponding module with this project's root on the
path; see README.md for what each one produces.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = {
    "app": "src.edge_engine.server",
    "compress": "src.models.compress_model",
    "baselines": "src.models.simple_baselines",
    "maps": "src.map_matching.osm_loader",
    "demo-bundle": "src.edge_engine.make_demo",
}
BENCH = {
    "day8": "src.dead_reckoning.strapdown_ins",
    "day9": "src.dead_reckoning.ins_with_ml_velocity",
    "day10": "src.dead_reckoning.heading",
    "day11-12": "src.map_matching.hmm_map_matcher",
    "day13": "src.fusion.gnss_ins_fusion",
    "day15": "src.edge_engine.replay",
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    sys.path.insert(0, str(ROOT))
    if cmd == "test":
        return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", *rest], cwd=ROOT)
    if cmd == "train":
        sys.argv = [str(ROOT / "run_training.py"), *rest]
        runpy.run_path(str(ROOT / "run_training.py"), run_name="__main__")
        return 0
    if cmd == "bench":
        if not rest or rest[0] not in BENCH:
            print("bench needs one of:", ", ".join(BENCH))
            return 2
        module, rest = BENCH[rest[0]], rest[1:]
    elif cmd in MODULES:
        module = MODULES[cmd]
    else:
        print(f"unknown command {cmd!r}\n{__doc__}")
        return 2
    # run as a subprocess so multiprocessing workers (day15) and relative paths behave as with `python -m`
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.call([sys.executable, "-m", module, *rest], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
