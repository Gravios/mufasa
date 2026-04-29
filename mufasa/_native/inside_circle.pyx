# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
mufasa._native.inside_circle
============================

Cython port of ``FeatureExtractionMixin.is_inside_circle``.

Algorithm: per-frame Euclidean distance from body-part to circle
center; 1 if dist <= radius else 0.

Note on the numba reference: it computes ``np.sqrt(...)`` and
compares to ``roi_radius``. We do the same here, NOT the more
efficient ``dist² <= radius²`` form, because byte-equivalence
to the numba version is the goal. (The squared-distance form
would skip a sqrt per frame; for typical 50K-frame videos this
saves ~50K sqrts which is microseconds. Not worth the
behavioral divergence.)
"""
import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def is_inside_circle(
    cnp.ndarray bp,
    cnp.ndarray roi_center,
    roi_radius,
) -> cnp.ndarray:
    """Per-frame point-in-circle test.

    :param bp: (n_frames, 2) float coordinate array.
    :param roi_center: (2,) center [cx, cy].
    :param roi_radius: int radius in pixel units.
    :return: (n_frames,) int32 with 0/1. Output dtype matches
        the numba version (which uses np.full with dtype=int32).
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] bp_f64 = (
        np.ascontiguousarray(bp, dtype=np.float64)
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] center = (
        np.ascontiguousarray(roi_center, dtype=np.float64)
    )
    cdef double radius = float(roi_radius)
    cdef Py_ssize_t n_frames = bp_f64.shape[0]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] results = np.full(
        n_frames, fill_value=-1, dtype=np.int32,
    )
    cdef Py_ssize_t i
    cdef double dx, dy, dist
    cdef double cx = center[0]
    cdef double cy = center[1]

    for i in range(n_frames):
        dx = bp_f64[i, 0] - cx
        dy = bp_f64[i, 1] - cy
        dist = sqrt(dx * dx + dy * dy)
        if dist <= radius:
            results[i] = 1
        else:
            results[i] = 0
    return results


__all__ = ["is_inside_circle"]
