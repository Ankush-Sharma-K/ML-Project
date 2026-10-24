"""
compress_model.py — Phase 3: on-device version of the velocity model
====================================================================

The trained model (`results/checkpoints/velocity_cnn_best.pt`, VelocityCNNDeep,
~650k parameters, 2.6 MB float32) is turned into files a phone can run with
the PyTorch Lite interpreter (PyTorch Mobile / ExecuTorch-compatible export):

  velocity_model_fp32.ptl   float32, BatchNorm folded into the convolutions
  velocity_model_int8.ptl   int8 post-training static quantization of the
                            whole CNN (conv + BN + ReLU fused), calibrated on
                            TRAINING windows only, mobile (qnnpack) qconfig;
                            exported without optimize_for_mobile, which would
                            store every int8 weight twice

DEPLOYABLE INPUT. Both files take the RAW feature window, shape
(batch, 200, 8) in the column order of `ckpt["feature_cols"]`, and return the
forward speed in m/s, shape (batch,). The training normalization (mean/std
from the training windows) and the (time, channel) -> (channel, time)
transpose are inside the exported model, so the app cannot apply the wrong
statistics.

VERIFICATION (`__main__`): MAE on the val and test windows (same windows as
`run_training.py`: whole-drive calibration, stride 10) for the original model
and both exported files (reloaded from disk with the Lite interpreter), the
largest difference to the original prediction, file size, and single-thread
CPU latency for one window. Results go to `results/compression_summary.json`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DeployableVelocityModel(nn.Module):
    """Raw window (batch, window_size, n_features) -> speed (batch,), m/s."""

    def __init__(self, model: nn.Module, norm_stats: dict):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(np.asarray(norm_stats["mean"]), dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.asarray(norm_stats["std"]), dtype=torch.float32))

    def forward(self, x):
        x = ((x - self.mean) / self.std).transpose(1, 2)
        return self.model(x).squeeze(-1)


def quantize_int8(model: nn.Module, calib_X: np.ndarray, norm_stats: dict, window_size: int,
                  n_features: int, backend: str = "qnnpack", batch_size: int = 256) -> nn.Module:
    """Post-training static int8 quantization (FX graph mode) of the CNN.
    `calib_X` = raw TRAINING windows (n, window_size, n_features)."""
    import copy

    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    from src.models.velocity_model import apply_normalization

    example = torch.zeros(1, n_features, window_size)
    prepared = prepare_fx(copy.deepcopy(model).eval(), get_default_qconfig_mapping(backend), (example,))
    Xn = apply_normalization(calib_X, norm_stats).transpose(0, 2, 1)
    with torch.no_grad():
        for i in range(0, len(Xn), batch_size):
            prepared(torch.from_numpy(np.ascontiguousarray(Xn[i:i + batch_size])))
    return convert_fx(prepared)


def export_lite(deployable: nn.Module, path: Path, window_size: int, n_features: int, optimize: bool = True) -> Path:
    """TorchScript trace -> (mobile optimization) -> Lite-interpreter file."""
    from torch.utils.mobile_optimizer import optimize_for_mobile

    deployable.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(deployable, torch.zeros(1, window_size, n_features))
    if optimize:
        scripted = optimize_for_mobile(scripted)
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted._save_for_lite_interpreter(str(path))
    return path


def load_lite(path: Path):
    from torch.jit.mobile import _load_for_lite_interpreter

    return _load_for_lite_interpreter(str(path))


def predict_raw(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Speed for raw windows (n, window_size, n_features) with a deployable /
    Lite model."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size], dtype=np.float32))).numpy())
    return np.concatenate(out)


def latency_ms(model, window_size: int, n_features: int, n_runs: int = 200) -> float:
    """Median single-thread CPU time for ONE window (batch 1)."""
    import time

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.randn(1, window_size, n_features)
    times = []
    with torch.no_grad():
        for _ in range(20):
            model(x)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    torch.set_num_threads(threads)
    return 1000.0 * float(np.median(times))


if __name__ == "__main__":
    import json
    import warnings

    root = Path(__file__).resolve()
    while not ((root / "src").is_dir() and (root / "data").is_dir()):
        root = root.parent
    sys.path.insert(0, str(root))
    from src.models.multi_drive_data import build_windows_for_drives, split_windows_by_names
    from src.models.train_velocity_model import load_model

    warnings.filterwarnings("ignore")
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
    ckpt_dir = root / "results" / "checkpoints"
    model, ckpt = load_model(ckpt_dir / "velocity_cnn_best.pt", "cpu")
    W, F, cols, stats = ckpt["window_size"], len(ckpt["feature_cols"]), ckpt["feature_cols"], ckpt["norm_stats"]
    split = json.loads((root / "data" / "split.json").read_text(encoding="utf-8"))
    real = root / "data" / "raw" / "real"

    print(f"\n{'=' * 70}\nPhase 3 — on-device velocity model (quantized engine: {torch.backends.quantized.engine})\n{'=' * 70}")
    held = [real / n for n in split["val"] + split["test"]]
    windows = build_windows_for_drives(held, feature_cols=cols, window_size=W, stride=10)
    _, val_w, test_w = split_windows_by_names(windows, split)
    train_paths = sorted(p for p in real.glob("S-*.csv") if p.name not in set(split["val"]) | set(split["test"]))
    calib = build_windows_for_drives(train_paths[::4], feature_cols=cols, window_size=W, stride=50)
    rng = np.random.default_rng(0)
    calib_X = calib["X"][rng.choice(len(calib["X"]), size=min(2000, len(calib["X"])), replace=False)]
    print(f"val {len(val_w['y'])} / test {len(test_w['y'])} windows; int8 calibration: {len(calib_X)} TRAIN windows "
          f"from {len(calib['drive_paths'])} drives")

    fp32 = DeployableVelocityModel(model, stats).eval()
    int8 = DeployableVelocityModel(quantize_int8(model, calib_X, stats, W, F), stats).eval()
    files = {"fp32": export_lite(fp32, ckpt_dir / "velocity_model_fp32.ptl", W, F),
             # optimize_for_mobile stores every int8 weight twice (1.32 MB vs 0.71 MB); convert_fx already fused conv+BN+ReLU
             "int8": export_lite(int8, ckpt_dir / "velocity_model_int8.ptl", W, F, optimize=False)}

    summary = {"feature_cols": cols, "window_size": W, "input": "raw window (batch, window_size, n_features)",
               "output": "forward speed m/s (batch,)", "quantized_engine": torch.backends.quantized.engine,
               "original_checkpoint_mb": (ckpt_dir / "velocity_cnn_best.pt").stat().st_size / 1e6,
               "n_parameters": sum(p.numel() for p in model.parameters())}
    ref = {name: predict_raw(fp32, w["X"]) for name, w in (("val", val_w), ("test", test_w))}
    print(f"\n{'Model':<22} {'File (MB)':>10} {'Val MAE':>8} {'Test MAE':>9} {'Max |diff|':>11} {'Latency (ms)':>13}")
    print("-" * 78)
    row = {"file_mb": summary["original_checkpoint_mb"], "latency_ms": latency_ms(fp32, W, F)}
    for name, w in (("val", val_w), ("test", test_w)):
        row[f"{name}_mae"] = float(np.mean(np.abs(ref[name] - w["y"])))
    summary["original"] = row
    print(f"{'original (PyTorch)':<22} {row['file_mb']:>10.2f} {row['val_mae']:>8.3f} {row['test_mae']:>9.3f} {'—':>11} "
          f"{row['latency_ms']:>13.2f}")
    for kind, path in files.items():
        lite = load_lite(path)
        row = {"file": str(path.relative_to(root)), "file_mb": path.stat().st_size / 1e6,
               "latency_ms": latency_ms(lite, W, F)}
        diffs = []
        for name, w in (("val", val_w), ("test", test_w)):
            pred = predict_raw(lite, w["X"])
            row[f"{name}_mae"] = float(np.mean(np.abs(pred - w["y"])))
            diffs.append(np.abs(pred - ref[name]))
        row["max_abs_diff_ms"] = float(np.max(np.concatenate(diffs)))
        row["mean_abs_diff_ms"] = float(np.mean(np.concatenate(diffs)))
        summary[kind] = row
        print(f"{path.name:<22} {row['file_mb']:>10.2f} {row['val_mae']:>8.3f} {row['test_mae']:>9.3f} "
              f"{row['max_abs_diff_ms']:>11.3f} {row['latency_ms']:>13.2f}")
    (root / "results" / "compression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nMAE in m/s on the val/test windows; diff = |exported - original| prediction (m/s); latency = one "
          "20 s window, 1 CPU thread.\nSummary: results/compression_summary.json")
