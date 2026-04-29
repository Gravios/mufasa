# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
mufasa._native.border_distances
================================

Cython port of
``FeatureExtractionSupplemental.border_distances``.

Algorithm: rolling-window mean distance from each body-part
position to the four image edges (left, right, top, bottom),
scaled by pixels_per_mm. Output is int32 (the numba reference
casts to int32 at the end via ``.astype(np.int32)``, which
truncates toward zero).

For each frame ``current_frm`` >= window_size:
  windowed = data[current_frm - window_size : current_frm]
  for each body-part position in the window:
    distance to left edge   = norm([0,         y] - [x, y]) = |x|
    distance to right edge  = norm([width,     y] - [x, y]) = |width - x|
    distance to top edge    = norm([x,         0] - [x, y]) = |y|
    distance to bottom edge = norm([x,    height] - [x, y]) = |height - y|
  results[current_frm - 1] = [mean(left), mean(right), mean(top), mean(bottom)] / px_per_mm

Frames before ``current_frm = window_size`` are filled with -1.

The numba reference uses ``np.linalg.norm`` for each axial
distance. Since one component is always 0, the norm reduces to
``abs()``. We compute that directly here for clarity and a
small speedup. Output is byte-identical because both forms
compute exactly the same value.
"""
import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport fabs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def border_distances(
    cnp.ndarray data,
    double pixels_per_mm,
    cnp.ndarray img_resolution,
    double time_window,
    int fps,
) -> cnp.ndarray:
    """Rolling-window distance to image edges (left, right, top, bottom)
    per frame, in millimeters."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2] d = (
        np.ascontiguousarray(data, dtype=np.float64)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] res_arr = (
        np.asarray(img_resolution, dtype=np.int64).flatten()
    )
    cdef double width = float(res_arr[0])
    cdef double height = float(res_arr[1])
    cdef int window_size = int(time_window * fps)
    cdef Py_ssize_t n_frames = d.shape[0]

    # Result is initialized to -1 (per the docstring: pre-window frames
    # are filled with -1). The numba version uses np.full with -1.0
    # then .astype(int32) — which casts -1.0 to int32(-1).
    cdef cnp.ndarray[cnp.float64_t, ndim=2] results = np.full(
        (n_frames, 4), fill_value=-1.0, dtype=np.float64,
    )

    cdef Py_ssize_t current_frm, j
    cdef double sum_left, sum_right, sum_top, sum_bot
    cdef double x, y
    cdef double inv_window_px = 1.0 / (window_size * pixels_per_mm)

    # Numba: ``for current_frm in prange(window_size, results.shape[0] + 1):``
    # so current_frm ranges over [window_size, n_frames] inclusive.
    # Window is data[current_frm - window_size : current_frm].
    # Result row written is current_frm - 1.
    for current_frm in range(window_size, n_frames + 1):
        sum_left = 0.0
        sum_right = 0.0
        sum_top = 0.0
        sum_bot = 0.0
        for j in range(current_frm - window_size, current_frm):
            x = d[j, 0]
            y = d[j, 1]
            # norm([0, y] - [x, y]) = sqrt(x^2 + 0^2) = |x|
            sum_left += fabs(x)
            sum_right += fabs(width - x)
            sum_top += fabs(y)
            sum_bot += fabs(height - y)
        results[current_frm - 1, 0] = sum_left * inv_window_px
        results[current_frm - 1, 1] = sum_right * inv_window_px
        results[current_frm - 1, 2] = sum_top * inv_window_px
        results[current_frm - 1, 3] = sum_bot * inv_window_px

    # Match the numba version's int32 cast at the end. The cast
    # truncates toward zero (Python/numpy convention for float→int).
    return results.astype(np.int32)


__all__ = ["border_distances"]
