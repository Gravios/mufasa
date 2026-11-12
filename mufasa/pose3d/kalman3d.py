"""
mufasa.pose3d.kalman3d
======================

Patch 122kf: 3-D Kalman smoothing of marker trajectories.

Per marker, a constant-velocity model (state ``[x, y, z, vx, vy, vz]``,
white-acceleration process noise ``sigma_a`` mm/s^1.5), a forward filter
with Mahalanobis gating (``gate`` σ; gated frames are treated as missing),
then a Rauch–Tung–Striebel backward pass (adapted from rpimocap's
``KalmanTracker3D``, which is forward-only).

Measurement noise is anisotropic when a camera is given: single-camera
3-D is uncertain mostly along the camera ray, so ``R = σ⊥² I +
(σ_ray² − σ⊥²) d dᵀ`` with ``d`` the ray direction — the smoother then
mostly evens out depth and barely moves points across the image. Without
a camera (e.g. triangulated stereo) ``R = σ⊥² I``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Kalman3DConfig:
    fps: float = 30.0
    sigma_a: float = 3000.0       # mm / s^1.5
    sigma_perp: float = 2.0       # mm, across the ray (or isotropic)
    sigma_ray: float = 25.0       # mm, along the ray
    gate: float = 6.0             # σ
    group: int = 4                # markers smoothed together (memory)


def _model(dt: float, q: float):
    I3 = np.eye(3)
    F = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
    Q = q * np.block([[dt ** 3 / 3 * I3, dt ** 2 / 2 * I3], [dt ** 2 / 2 * I3, dt * I3]])
    return F, Q


def smooth_markers(xyz: np.ndarray, cfg: Kalman3DConfig | None = None,
                   camera_center: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """``xyz`` (T, M, 3) with NaN for missing → smoothed (T, M, 3)."""
    cfg = cfg or Kalman3DConfig()
    T, M, _ = xyz.shape
    F, Q = _model(1.0 / cfg.fps, cfg.sigma_a ** 2)
    out = np.full_like(xyz, np.nan, dtype=float)
    gated = 0
    observed = 0
    for g0 in range(0, M, cfg.group):
        sl = slice(g0, min(M, g0 + cfg.group))
        Z = xyz[:, sl].astype(float)
        m = Z.shape[1]
        xf = np.zeros((T, m, 6))
        Pf = np.zeros((T, m, 6, 6), np.float32)
        Pp = np.zeros((T, m, 6, 6), np.float32)
        xp_all = np.zeros((T, m, 6))
        first = np.array([np.argmax(np.isfinite(Z[:, j]).all(1)) for j in range(m)])
        x = np.zeros((m, 6))
        for j in range(m):
            x[j, :3] = np.nan_to_num(Z[first[j], j])
        P = np.tile(np.diag([100.0] * 3 + [1e5] * 3), (m, 1, 1))
        for t in range(T):
            if t > 0:
                x = x @ F.T
                P = F @ P @ F.T + Q
            xp_all[t], Pp[t] = x, P
            z = Z[t]
            ok = np.isfinite(z).all(1)
            if ok.any():
                if camera_center is not None:
                    d = z - camera_center
                    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                    R = (cfg.sigma_perp ** 2) * np.eye(3) + (cfg.sigma_ray ** 2 - cfg.sigma_perp ** 2) * \
                        np.einsum("mi,mj->mij", np.nan_to_num(d), np.nan_to_num(d))
                else:
                    R = np.tile((cfg.sigma_perp ** 2) * np.eye(3), (m, 1, 1))
                S = P[:, :3, :3] + R
                y = np.nan_to_num(z) - x[:, :3]
                Si = np.linalg.inv(S)
                d2 = np.einsum("mi,mij,mj->m", y, Si, y)
                use = ok & (d2 <= cfg.gate ** 2) | (ok & (t <= first))
                observed += int(ok.sum())
                gated += int((ok & ~use).sum())
                if use.any():
                    Kg = np.einsum("mij,mjk->mik", P[:, :, :3], Si)
                    x = np.where(use[:, None], x + np.einsum("mij,mj->mi", Kg, y), x)
                    Pn = P - np.einsum("mij,mjk->mik", Kg, P[:, :3, :])
                    P = np.where(use[:, None, None], Pn, P)
            xf[t], Pf[t] = x, P
        xs = xf.copy()
        for t in range(T - 2, -1, -1):
            G = np.einsum("mij,kj->mik", Pf[t].astype(float), F)
            G = np.einsum("mij,mjk->mik", G, np.linalg.inv(Pp[t + 1].astype(float)))
            xs[t] = xf[t] + np.einsum("mij,mj->mi", G, xs[t + 1] - xp_all[t + 1])
        for j in range(m):                      # before the first observation: nothing to say
            xs[:first[j], j] = np.nan
        out[:, sl] = xs[:, :, :3]
    return out, {"observed": observed, "gated": gated}


def smooth_pose3d_frame(df, names: list[str], cfg: Kalman3DConfig | None = None,
                        camera_center=None):
    """Smooth ``<m>_x/_y/_z`` columns of a 3-D pose DataFrame in place
    (returns a copy) and report how far points moved."""
    X = np.stack([df[[m + "_x", m + "_y", m + "_z"]].to_numpy(float) for m in names], 1)
    S, info = smooth_markers(X, cfg, camera_center)
    out = df.copy()
    for j, m in enumerate(names):
        out[m + "_x"], out[m + "_y"], out[m + "_z"] = S[:, j].T
    moved = np.linalg.norm(S - X, axis=2)
    info["moved_mm_median"] = float(np.nanmedian(moved))
    info["moved_mm_p90"] = float(np.nanpercentile(moved, 90))
    return out, info


__all__ = ["Kalman3DConfig", "smooth_markers", "smooth_pose3d_frame"]
