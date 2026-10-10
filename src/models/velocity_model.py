"""
Day 6 part 2 (of 3 -- see PROGRESS.md) -- Velocity model architecture.

Scope for this file, deliberately: the model + supporting Dataset/
normalization code ONLY. The training script (loss, optimizer, training
loop, checkpointing) is the next piece, split out for the same reason
every multi-part day has been split so far -- get one verified piece
right before building the next thing on top of it.

ARCHITECTURE CHOICE: a small 1D convolutional network over the window's
time axis, not a recurrent (GRU/LSTM) network. Two reasons, both tied
directly to this phase's own stated goals (see PROGRESS.md Phase 3:
"IMU->velocity model, robustness, ON-DEVICE COMPRESSION"):
  1. Day 8's compression goal favors CNNs -- they quantize and export to
     mobile runtimes (TorchScript / ONNX / TFLite) more predictably than
     recurrent architectures, whose per-step hidden state doesn't always
     survive quantization or fixed-point mobile inference cleanly.
  2. The window is short and fixed-length (20 samples = 2.0 s, per Day
     6 part 1's `build_windows()` default) -- exactly the regime where a
     CNN's fixed receptive field is a natural fit, without needing a
     recurrent state carried across windows.

INPUT SHAPE CONTRACT (must match `dataset_prep.build_windows()` exactly):
`build_windows()` returns X as (n_windows, window_size, n_features) --
e.g. (n, 20, 6) for the 3 calibrated linear-accel + 3 gyro channels used
in Day 6 part 1's own verification run. PyTorch's Conv1d expects
(batch, channels, length), i.e. the last two axes swapped -- that
transpose happens inside `IMUWindowDataset.__getitem__`, not the model
itself, so the model's `forward()` can assume its input already arrives
as (batch, n_features, window_size).

NORMALIZATION: per-channel z-score (zero mean, unit variance), computed
from TRAINING data only (see `compute_normalization_stats`) and applied
identically to validation data -- computing stats from val or test data
would leak information about their distribution into what the model
implicitly learns to expect. Necessary here specifically because the
input channels have very different natural scales (linear acceleration
in single-digit m/s^2 vs. gyro rates in fractions of a rad/s) -- without
this, gradient updates would be dominated by whichever channel happens
to have the largest raw magnitude, not whichever is most informative.

WHAT IS AND ISN'T VERIFIED IN THIS FILE (stated plainly, not left
implicit): this sandbox has no internet and cannot install torch (checked
directly this session, not assumed -- `pip install torch` was actually
attempted and failed with no matching distribution). The PyTorch code
below has therefore NOT been executed here. To still catch real
shape/architecture bugs before handing this over, every layer's output
shape was independently cross-checked against a parallel pure-NumPy
reimplementation of the identical architecture (same layer sizes, same
kernel sizes, same padding), which WAS executed in this sandbox -- see
the verification note in PROGRESS.md for the actual numbers. That
confirms the architecture is internally shape-consistent; it does NOT
confirm the PyTorch-specific code below is syntactically perfect or
trains stably -- run this in your local/Colab environment and report
back anything that doesn't match.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    _ModuleBase = nn.Module
except ImportError:  # expected in this sandbox -- see module docstring
    torch = None
    nn = None
    Dataset = object       # allows IMUWindowDataset's class body to parse/import
    _ModuleBase = object   # allows VelocityCNN's class body to parse/import


FEATURE_COLS_DEFAULT = [
    "lin_accel_x_mps2_cal_veh", "lin_accel_y_mps2_cal_veh", "lin_accel_z_mps2_cal_veh",
    "gyro_yaw_rads", "gyro_pitch_rads", "gyro_roll_rads",
]


def compute_normalization_stats(X_train: np.ndarray) -> dict:
    """Per-channel mean/std from TRAINING windows only.
    X_train shape: (n_windows, window_size, n_features).
    Returns {"mean": (n_features,), "std": (n_features,)}, both computed
    by flattening across windows and time so every channel gets one
    scalar mean/std regardless of window_size.
    """
    flat = X_train.reshape(-1, X_train.shape[-1])  # (n_windows*window_size, n_features)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1e-8  # guard against a constant/degenerate channel
    return {"mean": mean, "std": std}


def apply_normalization(X: np.ndarray, stats: dict) -> np.ndarray:
    """Apply previously-computed (training-set) mean/std to any window
    array of the same shape (train, val, or later real inference input)."""
    return (X - stats["mean"]) / stats["std"]


class IMUWindowDataset(Dataset):
    """Wraps the dict returned by `dataset_prep.build_windows()` /
    `train_val_split_by_time()` into a torch Dataset.

    Handles the (window_size, n_features) -> (n_features, window_size)
    transpose Conv1d needs, and applies normalization stats computed
    ONCE from the training split (pass the same `stats` dict to both
    the train and val Dataset instances -- never recompute stats on
    val data, see module docstring).
    """

    def __init__(self, windows: dict, norm_stats: dict):
        if torch is None:
            raise ImportError(
                "torch is not installed in this environment. This class "
                "is meant to be used in the user's local/Colab environment, "
                "not in the sandbox that wrote it -- see module docstring."
            )
        X_norm = apply_normalization(windows["X"], norm_stats)
        # (n, window_size, n_features) -> (n, n_features, window_size)
        self.X = torch.from_numpy(X_norm.transpose(0, 2, 1)).float()
        self.y = torch.from_numpy(windows["y"]).float().unsqueeze(-1)  # (n, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class VelocityCNN(_ModuleBase):
    """1D-CNN forward-velocity regressor.

    Input:  (batch, n_features, window_size)  -- e.g. (batch, 6, 20)
    Output: (batch, 1)                         -- predicted forward speed, m/s

    Architecture (see module docstring for why a CNN, not an RNN):
      Conv1d(n_features -> 32, kernel=5, padding=2) + ReLU
      Conv1d(32 -> 64, kernel=5, padding=2) + ReLU
      AdaptiveAvgPool1d(1)                          -- collapses time axis
      Linear(64 -> 32) + ReLU
      Linear(32 -> 1)

    `padding=2` with `kernel_size=5` keeps the time dimension unchanged
    after each conv (output length = input length) -- verified via the
    standard conv output-length formula AND the parallel NumPy check
    (see PROGRESS.md), so the AdaptiveAvgPool1d(1) at the end is pooling
    over the full original window_size regardless of window_size's
    exact value, making this architecture window-size-agnostic (the
    same model works if `dataset_prep.build_windows()`'s window_size is
    changed later, without editing this file).
    """

    def __init__(self, n_features: int = 6, hidden_channels: tuple = (32, 64)):
        if torch is None:
            raise ImportError(
                "torch is not installed in this environment. This class "
                "is meant to be used in the user's local/Colab environment, "
                "not in the sandbox that wrote it -- see module docstring."
            )
        super().__init__()
        c1, c2 = hidden_channels
        self.conv1 = nn.Conv1d(n_features, c1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)   # (batch, c2, 1) -> (batch, c2)
        x = self.relu(self.fc1(x))
        return self.fc2(x)             # (batch, 1)

    def count_parameters(self) -> int:
        """Total trainable parameter count -- relevant to Day 8's
        on-device compression goal; check this number against your
        target platform's model-size budget before Day 8."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    if torch is None:
        print(
            "torch is not installed in THIS sandbox (expected -- see module "
            "docstring). This file is meant to be run in your local/Colab "
            "environment, where the checks below will actually execute:\n"
            "  1. model = VelocityCNN(n_features=6)\n"
            "  2. dummy = torch.randn(4, 6, 20)  # batch=4, matches Day 6 "
            "part 1's default window_size=20 and 6 feature columns\n"
            "  3. out = model(dummy); assert out.shape == (4, 1)\n"
            "  4. print(model.count_parameters())\n"
            "Run this file there and report back the printed parameter "
            "count and confirmation the assert passed."
        )
    else:
        model = VelocityCNN(n_features=6)
        dummy = torch.randn(4, 6, 20)
        out = model(dummy)
        assert out.shape == (4, 1), f"Expected (4, 1), got {tuple(out.shape)}"
        print(f"Forward pass OK: input {tuple(dummy.shape)} -> output {tuple(out.shape)}")
        print(f"Total trainable parameters: {model.count_parameters()}")
