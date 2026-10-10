"""
Day 6 part 3 of 3 -- Training script for the velocity model.

Wires together Day 6 part 1 (`dataset_prep.build_windows`,
`train_val_split_by_time`) and part 2 (`velocity_model`'s normalization
functions, `IMUWindowDataset`, `VelocityCNN`) into an actual training
loop. This is the file the user runs in their own local/Colab
environment (torch, no internet needed beyond installing torch itself).

LOSS FUNCTION: Smooth L1 (Huber), not plain MSE. Chosen because the
training target (GPS-speed-derived velocity) is itself noisy ground
truth, not a clean label -- Day 2 already found the GPS-speed signal
lags the true stop instant, and the real-dataset analysis (see
PROGRESS.md) found GPS data can hold stale values between 1 Hz fixes.
MSE weights outlier target errors quadratically, which would let a
handful of bad GPS samples dominate training; Smooth L1 behaves like
MSE for small errors but like MAE (linear, not quadratic) for large
ones, which is the standard choice specifically for noisy-label
regression. Revisit if training curves suggest otherwise.

CHECKPOINTING -- BUNDLES NORMALIZATION STATS WITH THE MODEL, ON PURPOSE:
a checkpoint that saves only `model.state_dict()` is a common, real bug
source: at inference time (Day 9+), the raw IMU window has to be
normalized with the EXACT SAME per-channel mean/std the model was
trained on, not stats recomputed from whatever data happens to be
loaded at inference time. This script's checkpoint dict always includes
`norm_stats`, `feature_cols`, and `window_size` alongside the model
weights, and `load_checkpoint()` returns all of them together --
structured so it's not possible to load a model without also getting
back the normalization it needs.

EARLY STOPPING: tracks best validation loss, stops after `patience`
epochs without improvement, and the returned/saved checkpoint is always
the BEST epoch's weights, not simply the last epoch's -- avoids
shipping an overfit final-epoch model.

A CAVEAT ON DATASET SIZE (say this now, not discover it after a
confusing training run): the only real IO-VNBD data actually present in
this sandbox is the small `S-S3b_real_sample.csv` (~11 minutes, ~1355
windows). That's small for a from-scratch CNN. Point `--data` at the
FULL real dataset (or several concatenated drives) for actual training,
not just this sandbox's verification sample -- see the module docstring
in `dataset_prep.py` for the row-index-windowing reasoning that makes
per-drive windowing safe to concatenate across multiple drive files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:  # expected in this sandbox -- see module docstring
    torch = None
    nn = None
    DataLoader = None


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    """One pass over the training DataLoader. Returns mean loss."""
    model.train()
    total_loss, n_samples = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)
    return total_loss / n_samples


@torch.no_grad() if torch is not None else (lambda f: f)
def evaluate(model, loader, loss_fn, device) -> float:
    """One no-grad pass over a DataLoader (val or test). Returns mean loss."""
    model.eval()
    total_loss, n_samples = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)
    return total_loss / n_samples


def save_checkpoint(
    path: str | Path,
    model,
    norm_stats: dict,
    feature_cols: list[str],
    window_size: int,
    epoch: int,
    val_loss: float,
) -> None:
    """Save model weights TOGETHER WITH everything needed to correctly
    preprocess a new window at inference time -- see module docstring
    for why this bundling isn't optional."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "norm_stats": {k: v.tolist() for k, v in norm_stats.items()},
        "feature_cols": feature_cols,
        "window_size": window_size,
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    """Load a checkpoint saved by `save_checkpoint`. Returns the raw
    dict (not a constructed model) -- caller reconstructs
    `VelocityCNN(n_features=len(feature_cols))`, calls
    `model.load_state_dict(result["model_state_dict"])`, and MUST use
    `result["norm_stats"]` (converted back to np.ndarray) to normalize
    any new data before inference -- never recompute stats at
    inference time.
    """
    ckpt = torch.load(path, map_location=device)
    ckpt["norm_stats"] = {k: np.array(v) for k, v in ckpt["norm_stats"].items()}
    return ckpt


def train_velocity_model(
    windows_train: dict,
    windows_val: dict,
    feature_cols: list[str],
    window_size: int,
    checkpoint_path: str | Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
) -> dict:
    """Full training loop. Returns a history dict:
    {"train_loss": [...], "val_loss": [...], "best_epoch": int,
     "best_val_loss": float, "checkpoint_path": str}.

    Expects `windows_train`/`windows_val` as produced by
    `dataset_prep.build_windows()` + `train_val_split_by_time()` --
    i.e. NOT yet normalized. Normalization stats are computed from
    `windows_train` only, here, and applied to both splits -- callers
    should not pre-normalize before calling this.
    """
    if torch is None:
        raise ImportError(
            "torch is not installed in this environment. This function "
            "is meant to run in your local/Colab environment -- see "
            "module docstring."
        )

    from src.models.velocity_model import (
        compute_normalization_stats, IMUWindowDataset, VelocityCNN,
    )

    norm_stats = compute_normalization_stats(windows_train["X"])

    train_ds = IMUWindowDataset(windows_train, norm_stats)
    val_ds = IMUWindowDataset(windows_val, norm_stats)  # SAME stats, not recomputed
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VelocityCNN(n_features=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()  # Huber loss -- see module docstring for why

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path, model, norm_stats, feature_cols,
                window_size, epoch, val_loss,
            )
        else:
            epochs_without_improvement += 1

        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({time.time()-t0:.1f}s)"
              f"{'  <- best so far, saved' if improved else ''}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no val improvement for {patience} epochs "
                  f"(best was epoch {best_epoch+1}, val_loss={best_val_loss:.4f})")
            break

    return {
        **history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint_path),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    def _find_project_root(start: "_Path") -> "_Path":
        """Walk upward from `start` until a directory containing both
        'src' and 'data' is found. Robust to this file being moved to a
        different depth under src/ -- unlike counting a fixed number of
        .parent levels, which silently breaks (ModuleNotFoundError) if
        the file isn't exactly where the count assumes."""
        for candidate in [start] + list(start.parents):
            if (candidate / "src").is_dir() and (candidate / "data").is_dir():
                return candidate
        raise RuntimeError(
            f"Could not locate the idr-project root (a folder containing "
            f"both 'src/' and 'data/') above {start}. Check that both "
            f"folders exist somewhere above this file."
        )

    PROJECT_ROOT = _find_project_root(_Path(__file__).resolve().parent)
    sys.path.insert(0, str(PROJECT_ROOT))

    if torch is None:
        print(
            "torch is not installed in THIS sandbox (expected -- see module "
            "docstring). Run this file in your local/Colab environment "
            "instead. It will:\n"
            "  1. Load a calibrated drive (Days 1-4 pipeline, NOT denoised "
            "-- see Day 6 part 1's reasoning)\n"
            "  2. Build windows + a time-ordered train/val split\n"
            "  3. Train VelocityCNN with early stopping, checkpointing the "
            "BEST validation epoch (not the last)\n"
            "  4. Save the checkpoint to results/checkpoints/velocity_model.pt "
            "-- bundled with normalization stats, feature column names, and "
            "window size, all needed for correct inference later\n"
            "Report back: final train_loss/val_loss, which epoch was best, "
            "and whether training loss decreased steadily (a flat or "
            "increasing training loss would indicate a real problem, not "
            "just needing more epochs)."
        )
    else:
        from src.io_vnbd_loader import load_smartphone_drive
        from src.calibration.gravity_compensation import compute_linear_acceleration
        from src.calibration.zupt_classifier import zupt_mask
        from src.calibration.bias_estimation import estimate_constant_bias, apply_bias_correction
        from src.calibration.mount_alignment import estimate_mount_rotation, apply_mount_alignment
        from src.models.dataset_prep import build_windows, train_val_split_by_time
        from src.models.velocity_model import FEATURE_COLS_DEFAULT

        csv_path = PROJECT_ROOT / "data" / "raw" / "SYNTH-drive1.csv"
        df = load_smartphone_drive(csv_path)
        df = compute_linear_acceleration(df)
        mask = zupt_mask(df)
        bias = estimate_constant_bias(df, mask)
        df = apply_bias_correction(df, bias)
        mount = estimate_mount_rotation(df, mask)
        df = apply_mount_alignment(df, mount["R_align"])
        df["gps_speed_ms"] = df["gps_speed_kmh"] / 3.6

        windows = build_windows(
            df, feature_cols=FEATURE_COLS_DEFAULT, target_col="gps_speed_ms",
            window_size=20, stride=5,
        )
        train_w, val_w = train_val_split_by_time(windows, val_fraction=0.2)

        result = train_velocity_model(
            train_w, val_w, feature_cols=FEATURE_COLS_DEFAULT, window_size=20,
            checkpoint_path=PROJECT_ROOT / "results" / "checkpoints" / "velocity_model.pt",
            epochs=50, batch_size=32, lr=1e-3, patience=10,
        )
        print(f"\nBest epoch: {result['best_epoch']+1}, "
              f"best val_loss: {result['best_val_loss']:.4f}")
        print(f"Checkpoint saved to: {result['checkpoint_path']}")
