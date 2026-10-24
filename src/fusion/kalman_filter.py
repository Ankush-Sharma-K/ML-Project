"""
kalman_filter.py — Phase 6 (Day 13): generic extended Kalman filter pieces
==========================================================================

Small, model-agnostic building blocks used by `gnss_ins_fusion.py`:

  ekf_update   one measurement update with an optional chi-square gate on the
               normalized innovation (NIS) -- a measurement whose innovation
               is implausible given the current uncertainty is rejected
               (GPS multipath / junk fixes) instead of pulling the state away;
               angle components of the innovation are wrapped to [-pi, pi)
  rts_smooth   Rauch-Tung-Striebel backward pass over a stored forward run:
               once GPS returns after an outage, every state inside the
               outage is re-estimated with the exit fix (offline track
               reconstruction)

Covariance updates use the Joseph form, which stays symmetric and positive
definite with the large uncertainty differences between GPS (metres) and a
2-minute dead-reckoning outage (hundreds of metres).
"""

from __future__ import annotations

import numpy as np

# chi-square 99.9% quantiles for 1..3 degrees of freedom
CHI2_999 = {1: 10.83, 2: 13.82, 3: 16.27}


def wrap_angle(a):
    """Wrap radians to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def ekf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, hx: np.ndarray, H: np.ndarray, R: np.ndarray,
               gate: float | None = None, angle_rows=()) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Update state `x` / covariance `P` with measurement `z`, predicted
    measurement `hx`, Jacobian `H` (m x n), noise `R` (m x m).
    `gate` = NIS threshold (e.g. CHI2_999[m]) or None; `angle_rows` = rows of
    the innovation to wrap. Returns (x, P, accepted, nis)."""
    y = np.atleast_1d(z - hx).astype(float)
    for r in angle_rows:
        y[r] = wrap_angle(y[r])
    S = H @ P @ H.T + R
    S_inv = np.linalg.inv(S)
    nis = float(y @ S_inv @ y)
    if gate is not None and nis > gate:
        return x, P, False, nis
    K = P @ H.T @ S_inv
    I_KH = np.eye(len(x)) - K @ H
    return x + K @ y, I_KH @ P @ I_KH.T + K @ R @ K.T, True, nis


def rts_smooth(x_f: np.ndarray, P_f: np.ndarray, x_p: np.ndarray, P_p: np.ndarray, F: np.ndarray,
               angle_cols=()) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoother.

    x_f[k], P_f[k]: filtered state/covariance after the updates at step k.
    x_p[k], P_p[k]: predicted state/covariance at step k (before its updates),
    F[k]: Jacobian of the prediction from step k-1 to k (F[0] unused).
    Returns smoothed (x_s, P_s); the last step equals the filtered one."""
    n = len(x_f)
    x_s, P_s = x_f.copy(), P_f.copy()
    for k in range(n - 2, -1, -1):
        C = P_f[k] @ F[k + 1].T @ np.linalg.inv(P_p[k + 1])
        dx = x_s[k + 1] - x_p[k + 1]
        for c in angle_cols:
            dx[c] = wrap_angle(dx[c])
        x_s[k] = x_f[k] + C @ dx
        P_s[k] = P_f[k] + C @ (P_s[k + 1] - P_p[k + 1]) @ C.T
    return x_s, P_s
