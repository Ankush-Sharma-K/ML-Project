"""
run_training.py — One-click training script for the IDR velocity model.
=======================================================================

Run this from the idr-project root:

    python run_training.py

What it does:
  1. Loads and calibrates every usable real drive via multi_drive_data
  2. Builds sliding windows (window=200 samples = 20 s, stride=10); target
     is the GPS speed interpolated between fixes, in m/s
  3. Splits train / val / test by whole recording (seeded, see
     multi_drive_data.train_val_test_split_by_drive)
  4. Trains VelocityCNNDeep with early stopping on val loss (architecture
     and window chosen by 5-fold cross-validation -- see velocity_model.py)
  5. Saves the best checkpoint to results/checkpoints/velocity_cnn_best.pt
  6. Reports MAE (m/s) on val and test, next to the trivial baseline of
     always predicting the training-set mean speed, both per window and
     averaged per drive (so one long drive can't dominate), and writes
     results/training_summary.json + results/plots/training_curves.png

Requirements:
  - PyTorch must be installed in your venv
  - Run from: d:\\M.Tech\\1st Sem\\Machine Learning\\ML Project\\idr-project\\
"""

from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

# -- Project root setup -------------------------------------------------------
def _find_root(start: Path) -> Path:
    for c in [start] + list(start.parents):
        if (c / "src").is_dir() and (c / "data").is_dir():
            return c
    raise RuntimeError("Cannot find project root (needs src/ and data/ folders).")

ROOT = _find_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))

# -- Imports ------------------------------------------------------------------
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__} found.")
    print(f"     Device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")
except ImportError:
    print("ERROR: PyTorch is not installed.")
    print("Install it with:  pip install torch")
    sys.exit(1)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.multi_drive_data import (
    build_windows_for_drives, train_val_test_split_by_drive, split_windows_by_names,
)
from src.models.velocity_model import (
    FEATURE_COLS_DEFAULT, DEFAULT_WINDOW_SIZE, DEFAULT_ARCH, DEFAULT_ARCH_KWARGS,
)
from src.models.train_velocity_model import train_velocity_model, load_model, predict

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -- Config -------------------------------------------------------------------
WINDOW_SIZE   = DEFAULT_WINDOW_SIZE   # 200 samples = 20 s
STRIDE        = 10
EPOCHS        = 50
BATCH_SIZE    = 256
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10
SPLIT_SEED    = 42
SPLIT_FILE    = ROOT / "data" / "split.json"
CHECKPOINT    = ROOT / "results" / "checkpoints" / "velocity_cnn_best.pt"
SUMMARY_JSON  = ROOT / "results" / "training_summary.json"
CURVES_PNG    = ROOT / "results" / "plots" / "training_curves.png"

# -- Find all real drives -----------------------------------------------------
real_drives = sorted((ROOT / "data" / "raw" / "real").glob("S-*.csv"))
print(f"\n[INFO] Found {len(real_drives)} real drive files.")

# -- Build windows from all drives --------------------------------------------
print("[INFO] Loading + calibrating drives and building windows ...\n")

t0 = time.time()
all_windows = build_windows_for_drives(
    csv_paths=real_drives,
    feature_cols=FEATURE_COLS_DEFAULT,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
)
elapsed = time.time() - t0

n_ok      = len(all_windows["per_drive_diagnostics"])
n_failed  = len(all_windows["failed_drives"])
n_skipped = len(all_windows["rate_mismatch_drives"])
n_total   = len(all_windows["y"])

print(f"\n[OK] Window building done in {elapsed:.1f}s")
print(f"     Drives OK:      {n_ok}")
print(f"     Drives failed:  {n_failed}")
print(f"     Rate-excluded:  {n_skipped}")
print(f"     Total windows:  {n_total:,}")

# -- Train / val / test split by whole recording ------------------------------
# Fixed split from data/split.json, so every model is compared on the same
# drives; created (and saved) only if the file does not exist yet.
if SPLIT_FILE.exists():
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    train_w, val_w, test_w = split_windows_by_names(all_windows, split)
    print(f"     Split:          fixed, from {SPLIT_FILE.name}")
else:
    train_w, val_w, test_w = train_val_test_split_by_drive(all_windows, seed=SPLIT_SEED)
    SPLIT_FILE.write_text(json.dumps({
        "val": [Path(p).name for p in val_w["drive_paths"]],
        "test": [Path(p).name for p in test_w["drive_paths"]],
    }, indent=2), encoding="utf-8")
    print(f"     Split:          new (seed {SPLIT_SEED}), saved to {SPLIT_FILE.name}")
for name, w in (("Train", train_w), ("Val", val_w), ("Test", test_w)):
    print(f"     {name + ' windows:':<16}{len(w['y']):>9,}  from {len(w['drive_paths'])} drives")

# -- Train --------------------------------------------------------------------
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

# Auto-detect best available device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon
else:
    DEVICE = "cpu"
print(f"\n[INFO] Training device: {DEVICE.upper()}")
print(f"[INFO] Training {DEFAULT_ARCH} model {DEFAULT_ARCH_KWARGS} for up to {EPOCHS} epochs (patience={PATIENCE})...")
print(f"       Checkpoint will be saved to: {CHECKPOINT}\n")

torch.manual_seed(SPLIT_SEED)
t_train = time.time()
result = train_velocity_model(
    windows_train=train_w,
    windows_val=val_w,
    feature_cols=FEATURE_COLS_DEFAULT,
    window_size=WINDOW_SIZE,
    checkpoint_path=CHECKPOINT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    patience=PATIENCE,
    device=DEVICE,
    arch=DEFAULT_ARCH,
    arch_kwargs=DEFAULT_ARCH_KWARGS,
    weight_decay=WEIGHT_DECAY,
)
train_minutes = (time.time() - t_train) / 60

# -- Evaluate the BEST checkpoint on val and test -----------------------------
model, ckpt = load_model(CHECKPOINT, device=DEVICE)

train_mean = float(train_w["y"].mean())
scores = {}
for name, w in (("val", val_w), ("test", test_w)):
    pred = predict(model, w, ckpt["norm_stats"], device=DEVICE)
    err = pred - w["y"]
    per_drive = {}
    for d in np.unique(w["drive_id"]):
        m = w["drive_id"] == d
        per_drive[Path(all_windows["drive_paths"][d]).name] = {
            "n_windows": int(m.sum()),
            "mae_mps": float(np.mean(np.abs(err[m]))),
            "bias_mps": float(np.mean(err[m])),
            "baseline_mae_mps": float(np.mean(np.abs(train_mean - w["y"][m]))),
        }
    scores[name] = {
        "mae_mps": float(np.mean(np.abs(err))),
        "rmse_mps": float(np.sqrt(np.mean(err ** 2))),
        "baseline_mean_speed_mae_mps": float(np.mean(np.abs(train_mean - w["y"]))),
        "per_drive_mean_mae_mps": float(np.mean([v["mae_mps"] for v in per_drive.values()])),
        "per_drive_mean_baseline_mae_mps": float(np.mean([v["baseline_mae_mps"] for v in per_drive.values()])),
        "per_drive": per_drive,
        "n_windows": int(len(w["y"])),
        "drives": [Path(p).name for p in w["drive_paths"]],
    }

summary = {
    "best_epoch": result["best_epoch"] + 1,
    "epochs_run": len(result["val_loss"]),
    "best_val_huber_loss": result["best_val_loss"],
    "train_minutes": round(train_minutes, 1),
    "train_mean_speed_mps": train_mean,
    "n_train_windows": int(len(train_w["y"])),
    "train_drives": [Path(p).name for p in train_w["drive_paths"]],
    "scores": scores,
    "history": {k: result[k] for k in ("train_loss", "val_loss", "val_mae")},
    "failed_drives": {Path(p).name: why for p, why in all_windows["failed_drives"].items()},
    "rate_mismatch_drives": [Path(p).name for p in all_windows["rate_mismatch_drives"]],
    "config": {"window_size": WINDOW_SIZE, "stride": STRIDE, "batch_size": BATCH_SIZE,
               "lr": LR, "weight_decay": WEIGHT_DECAY, "patience": PATIENCE, "split_seed": SPLIT_SEED,
               "arch": DEFAULT_ARCH, "arch_kwargs": {k: list(v) for k, v in DEFAULT_ARCH_KWARGS.items()},
               "feature_cols": FEATURE_COLS_DEFAULT, "target": "gps_speed_interp_ms"},
}
SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

CURVES_PNG.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
epochs_axis = np.arange(1, len(result["val_loss"]) + 1)
axes[0].plot(epochs_axis, result["train_loss"], label="train")
axes[0].plot(epochs_axis, result["val_loss"], label="val")
axes[0].set_title("Huber loss")
axes[0].set_xlabel("epoch")
axes[0].legend()
axes[1].plot(epochs_axis, result["val_mae"], label="CNN (val)")
axes[1].axhline(scores["val"]["baseline_mean_speed_mae_mps"], color="gray", ls="--",
                label="predict train mean (val)")
axes[1].set_title("Validation MAE (m/s)")
axes[1].set_xlabel("epoch")
axes[1].legend()
fig.tight_layout()
fig.savefig(CURVES_PNG, dpi=130)
plt.close(fig)

# -- Report -------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best epoch:     {summary['best_epoch']} / {EPOCHS} (ran {summary['epochs_run']})")
print(f"Best val loss:  {result['best_val_loss']:.6f} (Huber)")
print(f"Training time:  {train_minutes:.1f} min")
for name in ("val", "test"):
    s = scores[name]
    print(f"{name.capitalize():<5} MAE:      {s['mae_mps']:.3f} m/s  (RMSE {s['rmse_mps']:.3f}; "
          f"predict-the-mean baseline {s['baseline_mean_speed_mae_mps']:.3f} m/s)")
    print(f"      per-drive mean MAE {s['per_drive_mean_mae_mps']:.3f} m/s "
          f"(baseline {s['per_drive_mean_baseline_mae_mps']:.3f})")
    for drive, v in sorted(s["per_drive"].items(), key=lambda kv: -kv[1]["n_windows"]):
        print(f"        {drive:<13} {v['n_windows']:>6} windows  MAE {v['mae_mps']:.2f}  bias {v['bias_mps']:+.2f}")
print(f"Checkpoint:     {CHECKPOINT}")
print(f"Summary:        {SUMMARY_JSON}")
print(f"Curves:         {CURVES_PNG}")
print("=" * 60)
