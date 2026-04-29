# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
mufasa._native.euclidean_distance
==================================

Cython ports of the two Euclidean distance kernels:

* ``framewise_euclidean_distance`` — per-frame distance between
  two moving locations, scaled by px_per_mm. Numba reference
  uses ``@njit(parallel=True, fastmath=True)`` — the parallelism
  is what pulls in libgomp.
* ``framewise_euclidean_distance_roi`` — per-frame distance
  from a moving location to a static location, scaled by
  px_per_mm.

Honest performance expectation: numpy with AVX-512 on a modern
CPU (the 9800X3D in the user's case) is already very close to
memory bandwidth limits for these kernels. Cython will be
*comparable* to numba, not dramatically faster. The reason to
port these is libgomp removal — having every kernel non-numba
means feature extraction can use fork start method instead of
spawn, saving ~5-10s per worker startup.

Numba reference uses ``fastmath=True`` for the parallel variant.
We DO NOT use fastmath here (would diverge from byte-equivalence
goal). Output may have very slight floating-point differences
on the order of ULP which the verification script tolerates via
np.allclose with a tight tolerance.
"""
import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def framewise_euclidean_distance(
    cnp.ndarray location_1,
    cnp.ndarray location_2,
    double px_per_mm,
    bint centimeter,
) -> cnp.ndarray:
    """Per-frame Euclidean distance between two moving (n_frames, 2)
    coordinate arrays, scaled by px_per_mm. Returns float64."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] a = (
        np.ascontiguousarray(location_1, dtype=np.float64)
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=2] b = (
        np.ascontiguousarray(location_2, dtype=np.float64)
    )
    cdef Py_ssize_t n = a.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] results = np.empty(
        n, dtype=np.float64,
    )
    cdef Py_ssize_t i
    cdef double dx, dy
    # Match the numba `if centimeter and px_per_mm:` guard. If
    # px_per_mm is 0 we skip the /10 (and the divide-by-px_per_mm
    # would have produced inf, but that's the numba behavior we
    # match faithfully — caller should validate px_per_mm upstream).
    cdef double divisor = px_per_mm if not centimeter else (px_per_mm * 10.0)
    if not px_per_mm:
        divisor = px_per_mm  # preserve numba's "if centimeter and px_per_mm" gate
    for i in range(n):
        dx = a[i, 0] - b[i, 0]
        dy = a[i, 1] - b[i, 1]
        results[i] = sqrt(dx * dx + dy * dy) / divisor
    return results


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def framewise_euclidean_distance_roi(
    cnp.ndarray location_1,
    cnp.ndarray location_2,
    double px_per_mm,
    bint centimeter=False,
) -> cnp.ndarray:
    """Per-frame Euclidean distance from a moving (n_frames, 2)
    array to a static (1, 2) or (2,) location, scaled by px_per_mm.

    The numba reference accepts ``location_2`` shape (1, 2) or
    (2,) — it relies on numpy broadcasting in
    ``np.linalg.norm(location_1[i] - location_2)``. We normalize
    here by flattening to a 2-element vector.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] a = (
        np.ascontiguousarray(location_1, dtype=np.float64)
    )
    cdef cnp.ndarray loc2_arr = np.asarray(location_2, dtype=np.float64).flatten()
    cdef double cx = loc2_arr[0]
    cdef double cy = loc2_arr[1]
    cdef Py_ssize_t n = a.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] results = np.full(
        n, fill_value=np.nan, dtype=np.float64,
    )
    cdef Py_ssize_t i
    cdef double dx, dy
    for i in range(n):
        dx = a[i, 0] - cx
        dy = a[i, 1] - cy
        results[i] = sqrt(dx * dx + dy * dy) / px_per_mm
    if centimeter:
        # Match the numba version exactly: it does `results = results / 10`
        # AFTER the loop, as a separate division. Doing the same here
        # avoids any FP-ordering divergence.
        for i in range(n):
            results[i] = results[i] / 10.0
    return results


__all__ = [
    "framewise_euclidean_distance",
    "framewise_euclidean_distance_roi",
]
