# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
mufasa._native.angle3pt
========================

Cython port of ``FeatureExtractionMixin.angle3pt_vectorized``.

Algorithm: per-frame 3-point angle in degrees, computed as
the difference of two atan2 results (target - reference) and
normalized to [0, 360).

Numba reference uses ``fastmath=True``. Cython port does NOT —
fastmath reorders FP ops in ways that produce non-bit-equivalent
output (notably it can swap atan2 for a polynomial approximation
or vectorize the subtraction differently). The verification
script uses ``np.allclose`` with a tolerance tuned for atan2
ULP differences instead of strict ``np.array_equal``.

Expected precision difference: <1e-9 degrees per frame. Far
below any meaningful threshold for behavioral analysis.
"""
import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport atan2, M_PI


cdef double _RAD_TO_DEG = 180.0 / M_PI


@cython.boundscheck(False)
@cython.wraparound(False)
def angle3pt_vectorized(cnp.ndarray data) -> cnp.ndarray:
    """Per-frame 3-point angle in degrees.

    :param data: (n_frames, 6) array — each row is [ax, ay, bx, by, cx, cy]
        where (a, b, c) are three points; the angle is at vertex b
        between the rays to a and c.
    :return: (n_frames,) float64 array of angles in [0, 360).
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] d = (
        np.ascontiguousarray(data, dtype=np.float64)
    )
    cdef Py_ssize_t n = d.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] results = np.zeros(
        n, dtype=np.float64,
    )
    cdef Py_ssize_t i
    cdef double ax, ay, bx, by, cx, cy, angle

    for i in range(n):
        ax = d[i, 0]
        ay = d[i, 1]
        bx = d[i, 2]
        by = d[i, 3]
        cx = d[i, 4]
        cy = d[i, 5]
        # math.degrees(atan2(c.y-b.y, c.x-b.x) - atan2(a.y-b.y, a.x-b.x))
        angle = (atan2(cy - by, cx - bx) - atan2(ay - by, ax - bx)) * _RAD_TO_DEG
        if angle < 0.0:
            angle += 360.0
        results[i] = angle
    return results


__all__ = ["angle3pt_vectorized"]
