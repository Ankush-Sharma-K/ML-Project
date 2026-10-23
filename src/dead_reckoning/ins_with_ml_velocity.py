"""
ins_with_ml_velocity.py — Day 9: ML velocity during GPS outages ("anchored ML")
==============================================================================

Day 8 showed that integrating the phone's accelerometer (strapdown INS)
loses to simply holding the last GPS speed. Day 9 replaces integrated
acceleration with the learned velocity model (`VelocityCNNDeep`, 20 s IMU
window -> forward speed) and combines it with the one thing that is
reliably known at the tunnel entrance: the last GPS speed.

ANCHORED ML (the method):

    speed(tau) = ml(tau) + (v0 - ml(0)) * exp(-tau / FADE_S)

    v0     = last real GPS fix before the outage
    ml(t)  = model speed from the 20 s of IMU ending at t
    tau    = seconds since GPS was lost

At the entrance the model's error is visible (ml(0) vs v0) -- e.g. a
smooth-riding phone/car that the model reads as "slow". That correction
is applied fully at first and fades out, because the bias drifts as the
road and speed change.

WHY FADE_S = 20 s (scratchpad experiments, Day 8 audit part 3):
  * chosen on validation drives from {5 ... 180 s, never}; 20 s was best;
  * cross-checked with a method that cannot overfit a handful of drives:
    on 5-fold cross-validation over the 39 drives with wheel-speed truth,
    the minimum-variance mix of [last GPS speed, ML speed, anchored ML]
    computed from validation error covariances moves trust from the
    entrance correction to the model within ~10-20 s -- and did not beat
    the plain 20 s fade (176 vs 177 m median error at 120 s).
  * in that cross-validation, anchored ML beat holding the last speed at
    every outage length (47/88/176 m vs 61/141/315 m at 30/60/120 s).

`__main__` runs the outage benchmark (strapdown_ins.simulate_outages:
calibration from data before each outage, honest entry speed, wheel-speed
truth where available) on the TEST drives of data/split.json, which the
model never saw, and logs the result to results/metrics.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FADE_S = 20.0
PREDICT_EVERY_ROWS = 10   # one model prediction per second at 10 Hz, interpolated in between


def ml_speed_series(model, ckpt: dict, seg, ends: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Model speed (m/s) for windows ENDING at row indices `ends` of the
    calibrated DataFrame `seg` (each end needs window_size - 1 rows before it)."""
    import torch
    from src.models.velocity_model import apply_normalization

    W = ckpt["window_size"]
    F = seg[ckpt["feature_cols"]].to_numpy(dtype=np.float32)
    X = np.stack([F[e - W + 1 : e + 1] for e in ends])
    X = apply_normalization(X, ckpt["norm_stats"]).transpose(0, 2, 1)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(X)).to(device)).squeeze(-1).cpu().numpy()


def anchored_ml_speed(ml: np.ndarray, v0: float, tau_s: np.ndarray, fade_s: float = FADE_S) -> np.ndarray:
    """Model speed corrected by its error at the entrance, fading out."""
    return ml + (v0 - ml[0]) * np.exp(-tau_s / fade_s)


def make_ml_methods(model, ckpt: dict, device: str = "cpu", fade_s: float = FADE_S):
    """`extra_methods` callable for strapdown_ins.simulate_outages: returns
    {"ml": ..., "anchored_ml": ...} speed arrays over the outage."""
    def _methods(seg, s_local, n_max, v0, t_seg):
        offsets = np.unique(np.r_[np.arange(0, n_max + 1, PREDICT_EVERY_ROWS), n_max])
        pred = ml_speed_series(model, ckpt, seg, s_local + offsets, device)
        ml = np.interp(np.arange(n_max + 1), offsets, pred)
        tau = t_seg - t_seg[0]
        return {"ml": ml, "anchored_ml": anchored_ml_speed(ml, v0, tau, fade_s)}
    return _methods


DAY9_LABELS = {"hold_speed": "Hold last GPS fix (baseline)",
               "ins_zupt": "Strapdown INS + adaptive ZUPT",
               "ml": "ML velocity model",
               "anchored_ml": f"Anchored ML (entry correction, {FADE_S:.0f} s fade)"}
DAY9_SECTION = "Day 9 — ML velocity during simulated GPS outages (TEST drives only)"
OUTAGE_LENGTHS_S = (30, 60, 120)
OUTAGES_PER_DRIVE = 40


if __name__ == "__main__":
    import json
    import warnings
    import zlib

    import pandas as pd
    import torch

    root = Path(__file__).resolve()
    while not ((root / "src").is_dir() and (root / "data").is_dir()):
        root = root.parent
    sys.path.insert(0, str(root))
    from src.models.multi_drive_data import load_raw_drive
    from src.models.train_velocity_model import load_model
    from src.dead_reckoning.strapdown_ins import simulate_outages, load_truth, report

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(root / "results" / "checkpoints" / "velocity_cnn_best.pt", device)
    test_drives = json.loads((root / "data" / "split.json").read_text(encoding="utf-8"))["test"]

    print(f"\n{'=' * 70}\nDay 9 — anchored ML velocity, simulated GPS outages on TEST drives\n"
          f"(model never trained on these; calibration from data before each outage;\n"
          f" entry speed = last real GPS fix; device {device})\n{'=' * 70}")
    rows = []
    for name in test_drives:
        df = load_raw_drive(root / "data" / "raw" / "real" / name)
        truth, source = load_truth(df, name, root)
        r = simulate_outages(df, OUTAGE_LENGTHS_S, n_starts=OUTAGES_PER_DRIVE, truth=truth,
                             seed=zlib.crc32(name.encode()), history_rows=ckpt["window_size"] - 1,
                             extra_methods=make_ml_methods(model, ckpt, device))
        if r.empty:
            print(f"  {name:<12} skipped (too short for a {max(OUTAGE_LENGTHS_S)} s outage + 20 s window)")
            continue
        r["drive"], r["truth"] = name, source
        rows.append(r)
        print(f"  {name:<12} {r.start_idx.nunique():>3} outages  truth={source}")

    res = pd.concat(rows, ignore_index=True)
    out_csv = root / "results" / "day9_outage_benchmark.csv"
    res.to_csv(out_csv, index=False)
    report(res, DAY9_LABELS, "Day 9", DAY9_SECTION, root)

    print("\nPer drive, 120 s outages (median abs error, m):")
    per = res[res.outage_s == 120].groupby(["drive", "method"])["abs_err_m"].median().unstack()
    print(per[[m for m in DAY9_LABELS if m in per.columns]].round(1).to_string())
    print(f"\nPer-outage results: {out_csv}")
    print("Summary rows written to results/metrics.md (re-running replaces them).")
