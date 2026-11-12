"""
mufasa.pose3d.stereo
====================

Patch 122kf: two synchronised cameras → 3-D by triangulation (linear DLT
on undistorted pixels, adapted from rpimocap.reconstruction.triangulate),
with likelihood and reprojection gates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mufasa.pose3d.camera import Camera
from mufasa.pose3d.mono import Pose2D


def triangulate(cam0: Camera, cam1: Camera, uv0: np.ndarray, uv1: np.ndarray) -> np.ndarray:
    """Points (T, 2) in each camera (distorted pixels) → (T, 3)."""
    u0, u1 = cam0.undistort(uv0), cam1.undistort(uv1)
    P0, P1 = cam0.P, cam1.P
    T = len(u0)
    out = np.full((T, 3), np.nan)
    ok = np.isfinite(u0).all(1) & np.isfinite(u1).all(1)
    if not ok.any():
        return out
    a, b = u0[ok], u1[ok]
    A = np.stack([a[:, 0:1] * P0[2] - P0[0], a[:, 1:2] * P0[2] - P0[1],
                  b[:, 0:1] * P1[2] - P1[0], b[:, 1:2] * P1[2] - P1[1]], 1)      # (n, 4, 4)
    _, _, vt = np.linalg.svd(A)
    Xh = vt[:, -1]
    out[ok] = Xh[:, :3] / Xh[:, 3:4]
    return out


def reconstruct_stereo(pose0: Pose2D, pose1: Pose2D, cam0: Camera, cam1: Camera, *,
                       offset1: int = 0, likelihood_threshold: float = 0.6,
                       max_reproj_px: float = 15.0) -> tuple[pd.DataFrame, dict]:
    """``offset1``: frame of camera 1 that matches camera 0's frame 0."""
    names = [m for m in pose0.names if m in pose1.names]
    n0, n1 = pose0.n, pose1.n
    a0, a1 = (0, offset1) if offset1 >= 0 else (-offset1, 0)
    N = min(n0 - a0, n1 - a1)
    cols = {"frame": np.arange(a0, a0 + N)}
    rep = {"frames": N, "markers": {}}
    for m in names:
        u0, u1 = pose0.uv[m][a0:a0 + N], pose1.uv[m][a1:a1 + N]
        lk = np.minimum(pose0.lk[m][a0:a0 + N], pose1.lk[m][a1:a1 + N])
        X = triangulate(cam0, cam1, u0, u1)
        e = 0.5 * (np.linalg.norm(cam0.project(X) - u0, axis=1) + np.linalg.norm(cam1.project(X) - u1, axis=1))
        bad = (lk < likelihood_threshold) | ~(e <= max_reproj_px)
        X[bad] = np.nan
        cols[m + "_x"], cols[m + "_y"], cols[m + "_z"] = X.T
        cols[m + "_likelihood"], cols[m + "_reproj_px"] = lk, e
        rep["markers"][m] = {"triangulated": float(np.isfinite(X[:, 0]).mean()),
                             "reproj_px_median": float(np.nanmedian(np.where(bad, np.nan, e)))}
    return pd.DataFrame(cols), rep


__all__ = ["reconstruct_stereo", "triangulate"]
