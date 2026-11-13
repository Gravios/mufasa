"""
mufasa.pose3d.solver
====================

Patch 122ki: the Jacobian for the frame-by-frame single-camera fits.

The head and spine fits are long sequences in which each frame's unknowns
(6 head-pose values, or one height per spine label) meet only the next
frame's, through the smooth-motion terms. SciPy's sparse finite
differences (``jac_sparsity``) group the columns greedily and rebuild the
matrix each time; here the groups come from the structure — every other
frame is perturbed at once, ``2 × unknowns-per-frame`` residual
evaluations, exact because no residual depends on two frames that far
apart — and the sparse matrix is filled from a fixed index.

Same steps, same result as before (K7 head chunk, frames 5700–12300:
identical cost and positions), about 1.5× faster. Two faster variants were
tried and rejected because the result changed: capping LSMR at 20
iterations per step (2.3× faster; lower cost on one chunk, 10% higher on
another, heads up to 50 mm lower) and Levenberg–Marquardt with direct
banded solves (6× faster per chunk, but stuck in higher local minima,
the box-wall hinge terms being kinked).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps
from scipy.optimize import least_squares


def frame_groups(n_frames: int, per_frame: int, span: int = 2) -> np.ndarray:
    """Column groups for a sequence whose residuals involve at most
    ``span`` consecutive frames: columns of frames ``k`` and ``k + span``
    can be perturbed together."""
    k = np.repeat(np.arange(n_frames), per_frame)
    p = np.tile(np.arange(per_frame), n_frames)
    return (k % span) * per_frame + p


class GroupedJacobian:
    """Forward-difference Jacobian with a known sparsity pattern and column
    grouping (no two columns of a group share a row)."""

    def __init__(self, fun, sparsity, groups: np.ndarray) -> None:
        coo = sps.coo_matrix(sparsity)
        self.fun = fun
        self.shape = coo.shape
        self.rows, self.cols = coo.row.astype(np.int64), coo.col.astype(np.int64)
        self.groups = np.asarray(groups)
        self.n_groups = int(self.groups.max()) + 1
        self._cache = None                       # (x, f(x)) of the last evaluation

    def residual(self, x: np.ndarray) -> np.ndarray:
        f = self.fun(x)
        self._cache = (x.copy(), f)
        return f

    def __call__(self, x: np.ndarray) -> sps.csr_matrix:
        if self._cache is not None and np.array_equal(self._cache[0], x):
            f0 = self._cache[1]
        else:
            f0 = self.fun(x)
        h = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x))
        D = np.empty((self.n_groups, len(f0)))
        for g in range(self.n_groups):
            m = self.groups == g
            xg = x.copy()
            xg[m] += h[m]
            D[g] = self.fun(xg) - f0
        vals = D[self.groups[self.cols], self.rows] / h[self.cols]
        return sps.csr_matrix((vals, (self.rows, self.cols)), shape=self.shape)


def solve_sequence(fun, x0: np.ndarray, sparsity, per_frame: int, *, max_nfev: int):
    """``least_squares`` (trust region, LSMR; the settings the fits always
    used) with the grouped Jacobian, for a frame sequence with
    ``per_frame`` unknowns per frame, frame-major in ``x0``."""
    n_frames = len(x0) // per_frame
    J = GroupedJacobian(fun, sparsity, frame_groups(n_frames, per_frame))
    return least_squares(J.residual, x0, jac=J, max_nfev=max_nfev, x_scale="jac", tr_solver="lsmr")


__all__ = ["GroupedJacobian", "frame_groups", "solve_sequence"]
