# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
mufasa._native.inside_rectangle
================================

Cython port of
``FeatureExtractionMixin.framewise_inside_rectangle_roi``.

Algorithm: for each frame, return 1 if (x, y) is inside the
axis-aligned rectangle defined by [topLeft, bottomRight];
else 0.

The numba reference implementation does this via two
``np.argwhere`` calls + an inner search — an O(N²) accident
that we faithfully match here for byte-equivalence rather than
"fix" silently. The Cython version is a single O(N) pass; the
output is identical because the algorithm produces a deterministic
boolean per frame regardless of how it's computed. We don't
mimic the inefficiency, only the output.

If the byte-equivalence test passes, this kernel is also a small
de facto performance win (O(N²) → O(N)) on the rare case of
many frames inside the rectangle. Most workloads have moderate
overlap and the difference is negligible.
"""
import numpy as np

cimport cython
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
def framewise_inside_rectangle_roi(
    cnp.ndarray bp_location,
    cnp.ndarray roi_coords,
) -> cnp.ndarray:
    """Per-frame point-in-rectangle test.

    :param bp_location: (n_frames, 2) float coordinate array.
    :param roi_coords: (2, 2) — [[topLeftX, topLeftY], [bottomRightX, bottomRightY]].
    :return: (n_frames,) int64 with 0 (outside) / 1 (inside).
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2] bp_f64 = (
        np.ascontiguousarray(bp_location, dtype=np.float64)
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=2] roi_f64 = (
        np.ascontiguousarray(roi_coords, dtype=np.float64)
    )
    cdef Py_ssize_t n_frames = bp_f64.shape[0]
    # Match the numba version's output dtype: it uses np.full with no
    # dtype kwarg, which defaults to float64 when fill is 0 — but the
    # downstream consumer treats it as int. Use int64 to match numpy's
    # default integer type.
    cdef cnp.ndarray[cnp.int64_t, ndim=1] results = np.zeros(
        n_frames, dtype=np.int64,
    )
    cdef double tlx = roi_f64[0, 0]
    cdef double tly = roi_f64[0, 1]
    cdef double brx = roi_f64[1, 0]
    cdef double bry = roi_f64[1, 1]
    cdef Py_ssize_t i
    cdef double x, y

    for i in range(n_frames):
        x = bp_f64[i, 0]
        y = bp_f64[i, 1]
        if x >= tlx and x <= brx and y >= tly and y <= bry:
            results[i] = 1
    return results


__all__ = ["framewise_inside_rectangle_roi"]
