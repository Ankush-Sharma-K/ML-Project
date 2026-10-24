"""
simple_baselines.py — Phase 3: simple models before the neural network
======================================================================

The project brief asks to "start with simple models before trying neural
networks". This module trains classical regressors on the SAME 20 s windows,
features and fixed train / val / test split (`data/split.json`) as the CNN
(`run_training.py`), so the comparison is like for like:

  features   per channel (the 8 model inputs): mean, std, 10/50/90th
             percentile, mean |first difference| (jerkiness), and spectral
             energy in 0-1 / 1-3 / 3-5 Hz  ->  72 features per window
  models     predict the training mean (trivial), ridge regression
             (standardized features), histogram gradient boosting

`__main__` writes `results/simple_baselines.json` and prints the val / test MAE
next to the CNN's (from `results/compression_summary.json`, same windows).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BANDS_HZ = ((0.0, 1.0), (1.0, 3.0), (3.0, 5.0))


def window_features(X: np.ndarray, fs_hz: float = 10.0, chunk: int = 4096) -> np.ndarray:
    """(n, window, channels) -> (n, 9 * channels) hand-crafted features."""
    out = []
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0 / fs_hz)
    masks = [(freqs >= lo) & (freqs < hi) if lo > 0 else (freqs > 0) & (freqs < hi) for lo, hi in BANDS_HZ]
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk].astype(np.float32)
        p10, p50, p90 = np.percentile(x, [10, 50, 90], axis=1)
        spec = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)) ** 2
        bands = [spec[:, m, :].sum(axis=1) / x.shape[1] for m in masks]
        out.append(np.concatenate([x.mean(1), x.std(1), p10, p50, p90, np.abs(np.diff(x, axis=1)).mean(1)] + bands,
                                  axis=1))
    return np.concatenate(out)


if __name__ == "__main__":
    import json
    import time
    import warnings

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    root = Path(__file__).resolve()
    while not ((root / "src").is_dir() and (root / "data").is_dir()):
        root = root.parent
    sys.path.insert(0, str(root))
    from src.models.multi_drive_data import build_windows_for_drives, split_windows_by_names
    from src.models.velocity_model import DEFAULT_WINDOW_SIZE, FEATURE_COLS_DEFAULT

    warnings.filterwarnings("ignore")
    t0 = time.time()
    windows = build_windows_for_drives(sorted((root / "data" / "raw" / "real").glob("S-*.csv")),
                                       feature_cols=FEATURE_COLS_DEFAULT, window_size=DEFAULT_WINDOW_SIZE, stride=10)
    split = json.loads((root / "data" / "split.json").read_text(encoding="utf-8"))
    tr, va, te = split_windows_by_names(windows, split)
    print(f"\nwindows: train {len(tr['y'])}, val {len(va['y'])}, test {len(te['y'])} ({time.time() - t0:.0f} s)")
    F = {name: window_features(w["X"]) for name, w in (("train", tr), ("val", va), ("test", te))}
    y = {"train": tr["y"], "val": va["y"], "test": te["y"]}

    def mae(pred, name):
        return float(np.mean(np.abs(pred - y[name])))

    results = {"n_features": int(F["train"].shape[1]),
               "n_windows": {k: int(len(v)) for k, v in y.items()}}
    mean_speed = float(np.mean(y["train"]))
    results["predict_train_mean"] = {"val_mae": mae(np.full(len(y["val"]), mean_speed), "val"),
                                     "test_mae": mae(np.full(len(y["test"]), mean_speed), "test")}
    models = {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
              "gradient_boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                                 early_stopping=True, random_state=0)}
    for name, mdl in models.items():
        t1 = time.time()
        mdl.fit(F["train"], y["train"])
        results[name] = {"val_mae": mae(mdl.predict(F["val"]), "val"), "test_mae": mae(mdl.predict(F["test"]), "test"),
                         "fit_s": time.time() - t1}
    print(f"\n{'Model':<34} {'Val MAE':>8} {'Test MAE':>9}   (m/s, same windows and split)")
    print("-" * 60)
    for name, label in (("predict_train_mean", "Predict the training mean"), ("ridge", "Ridge regression (72 features)"),
                        ("gradient_boosting", "Gradient boosting (72 features)")):
        print(f"{label:<34} {results[name]['val_mae']:>8.3f} {results[name]['test_mae']:>9.3f}")
    comp = json.loads((root / "results" / "compression_summary.json").read_text(encoding="utf-8"))
    print(f"{'Deep CNN (VelocityCNNDeep)':<34} {comp['original']['val_mae']:>8.3f} {comp['original']['test_mae']:>9.3f}")
    results["cnn_deep"] = {"val_mae": comp["original"]["val_mae"], "test_mae": comp["original"]["test_mae"]}
    (root / "results" / "simple_baselines.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved results/simple_baselines.json")
