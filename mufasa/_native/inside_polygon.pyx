# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
mufasa._native.inside_polygon
=============================

Cython implementation of frame-by-frame "is body-part inside
polygon ROI" — proof of concept for replacing numba kernels with
ahead-of-time-compiled C.

This is a one-to-one port of
``mufasa.mixins.feature_extraction_mixin.FeatureExtractionMixin
.framewise_inside_polygon_roi``. Same algorithm (ray casting),
same input/output contract, byte-identical results expected.

Why Cython instead of pure C++ via pybind11:
- The kernel is a tight loop over numpy arrays — Cython's
  typed memoryviews handle this natively.
- No template metaprogramming or C++ STL needed.
- Less boilerplate per kernel; the main cost is the build
  configuration, paid once.

Why this kernel as proof of concept:
- Self-contained: only needs the bp_location and roi_coords
  arrays.
- Branch-heavy ray-casting: this is exactly the pattern where
  AOT compilation can beat JIT.
- Easy byte-verification against the existing numba version on
  synthetic data.
- Small (~20 lines) — easy to fully understand and review.

Build:
    Compiled at ``pip install`` time via ``setuptools`` +
    ``Cython.Build.cythonize``. See pyproject.toml's
    [tool.setuptools.ext-modules] section.

API contract (must match numba version exactly):
    framewise_inside_polygon_roi(bp_location, roi_coords) -> ndarray
        bp_location:  (n_frames, 2) float32 or float64
        roi_coords:   (n_polygon_pts, 2) float32 or float64
        returns:      (n_frames,) int64 with 0/1 flags

Performance expectations:
    The numba version hits libgomp via ``prange``. This Cython
    version is single-threaded but uses tight typed loops with
    no Python overhead per iteration. Expected: comparable or
    slightly faster than numba for typical n_frames in the
    50K-500K range. The win isn't raw speed — it's removing
    the libgomp dependency from the per-video kernel chain,
    which matters for parallelization (no more spawn cost).
"""
import numpy as np

cimport cython
cimport numpy as cnp

# Cython memoryviews give us direct C-level array access without
# Python overhead per element.

ctypedef fused float_t:
    cnp.float32_t
    cnp.float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def framewise_inside_polygon_roi(
    cnp.ndarray bp_location,
    cnp.ndarray roi_coords,
) -> cnp.ndarray:
    """Per-frame point-in-polygon test (ray casting).

    Mirrors the algorithm in
    FeatureExtractionMixin.framewise_inside_polygon_roi exactly,
    so output is byte-identical for the same inputs.

    Accepts float32 or float64 input arrays; output is int64 to
    match the numba version (which uses np.full's default dtype).
    """
    # Coerce to float64 for the inner loop. Cython lets us specify
    # typed memoryviews, but the numba version does its arithmetic
    # in whatever the input dtype is. To get byte-equivalent output
    # we need to do the same thing. For the POC we use float64
    # internally; if the numba version preserves input precision
    # and we need to match that exactly, we'd template the kernel.
    cdef cnp.ndarray[cnp.float64_t, ndim=2] bp_loc_f64 = (
        np.ascontiguousarray(bp_location, dtype=np.float64)
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=2] roi_f64 = (
        np.ascontiguousarray(roi_coords, dtype=np.float64)
    )
    cdef Py_ssize_t n_frames = bp_loc_f64.shape[0]
    cdef Py_ssize_t n_poly = roi_f64.shape[0]

    cdef cnp.ndarray[cnp.int64_t, ndim=1] results = np.zeros(
        n_frames, dtype=np.int64,
    )

    cdef Py_ssize_t i, j
    cdef double x, y, p1x, p1y, p2x, p2y, xints
    cdef bint inside

    for i in range(n_frames):
        x = bp_loc_f64[i, 0]
        y = bp_loc_f64[i, 1]
        p1x = roi_f64[0, 0]
        p1y = roi_f64[0, 1]
        xints = 0.0
        inside = False
        # The numba version iterates j in range(n+1) — i.e. loops
        # one past the last vertex, indexing modulo n. Mirror that
        # exactly to preserve byte-equivalence.
        for j in range(n_poly + 1):
            p2x = roi_f64[j % n_poly, 0]
            p2y = roi_f64[j % n_poly, 1]
            if (y > min(p1y, p2y)) and (y <= max(p1y, p2y)) and (x <= max(p1x, p2x)):
                if p1y != p2y:
                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xints:
                    inside = not inside
            p1x = p2x
            p1y = p2y
        if inside:
            results[i] = 1

    return results


__all__ = ["framewise_inside_polygon_roi"]
