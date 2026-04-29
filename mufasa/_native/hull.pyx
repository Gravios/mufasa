# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
mufasa._native.hull
====================

Cython port of
``mufasa.feature_extractors.perimeter_jit.jitted_hull``.

Algorithm: per-frame convex hull of (n_body_parts, 2) points
via QuickHull, then either perimeter (sum of edge lengths) or
area (shoelace formula).

Numba reference uses ``fastmath=True``. Cython port does NOT.
Output may differ at the ULP level for the final perimeter/area
values (the hull *vertex set* should be identical because hull
construction uses signed-distance comparisons that are
fastmath-stable). The verification script tolerates ULP-level
differences via np.allclose.

Why this is the trickiest port:
- QuickHull is recursive; Python-level recursion is slow, so
  the recursive helper uses ``cdef`` for direct C-level calls.
- The numba version uses ``numba.np.extensions.cross2d`` for
  signed distance. Cython doesn't have that helper — we
  implement the 2D cross product inline.
- Polar-angle sort uses ``np.where(... np.arccos(...))``. Done
  in Python for clarity; the inner work is small (n_hull_vertices
  per frame, typically <10).
"""
import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport sqrt, fabs, acos, M_PI


# ---------------------------------------------------------------- #
# QuickHull recursive helper. Returns list of vertex indices from
# a to b (inclusive on `a`, exclusive on `b`) lying above the line
# (S[a], S[b]).
#
# We use a Python list because the recursion produces variable-length
# results that get concatenated. For typical n_body_parts (<20) the
# Python-list overhead is negligible vs. the geometry math.
# ---------------------------------------------------------------- #
def _process(cnp.ndarray[cnp.float32_t, ndim=2] S, list P, long a, long b):
    cdef long n_p = len(P)
    cdef double ax, ay, bx, by, dab_x, dab_y
    cdef list K
    cdef long c, i, idx
    cdef double max_sd, dx, dy, sd
    cdef list left, right

    if n_p == 0:
        return [a, b]
    # Compute signed distance for each i in P:
    #   sd = cross2d(S[i] - S[a], S[b] - S[a])
    # Find max-distance index AND filter to points with sd > 0 (and
    # not equal to a or b).
    ax = S[a, 0]
    ay = S[a, 1]
    bx = S[b, 0]
    by = S[b, 1]
    dab_x = bx - ax
    dab_y = by - ay
    K = []
    c = -1
    max_sd = -1e308  # very negative

    for i in range(n_p):
        idx = P[i]
        dx = S[idx, 0] - ax
        dy = S[idx, 1] - ay
        # cross2d((dx, dy), (dab_x, dab_y)) = dx*dab_y - dy*dab_x
        sd = dx * dab_y - dy * dab_x
        if sd > max_sd:
            max_sd = sd
            c = idx
        if sd > 0 and idx != a and idx != b:
            K.append(idx)

    if len(K) == 0:
        return [a, b]
    # Recurse: [a..c) + [c..b)
    left = _process(S, K, a, c)
    right = _process(S, K, c, b)
    return left[:-1] + right


@cython.boundscheck(False)
@cython.wraparound(False)
def jitted_hull(
    cnp.ndarray points,
    str target = "perimeter",
) -> cnp.ndarray:
    """Per-frame convex hull perimeter or area.

    :param points: (n_frames, n_body_parts, 2) float32 array.
    :param target: ``'perimeter'`` or ``'area'``.
    :return: (n_frames,) float64 array; NaN where hull fails.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=3] pts = (
        np.ascontiguousarray(points, dtype=np.float32)
    )
    cdef Py_ssize_t n_frames = pts.shape[0]
    cdef Py_ssize_t n_bp = pts.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] results = np.full(
        n_frames, fill_value=np.nan, dtype=np.float64,
    )
    cdef Py_ssize_t i, j, n_idx
    cdef long a_idx, max_idx
    cdef cnp.ndarray[cnp.float32_t, ndim=2] S
    cdef list idx_list, all_indices
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x, y, r, angles
    cdef double x0, y0
    cdef cnp.ndarray[cnp.intp_t, ndim=1] mask
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_sorted, y_sorted
    cdef list upper, lower

    for i in range(n_frames):
        S = pts[i, :, :]
        # a = leftmost point, b = rightmost point
        a_idx = int(np.argmin(S[:, 0]))
        max_idx = int(np.argmax(S[:, 0]))

        # Hull = upper chain (a → max) + lower chain (max → a)
        # Both calls drop the last index (the "to" vertex), so the
        # final list contains each hull vertex exactly once.
        all_indices = list(range(n_bp))
        upper = _process(S, all_indices, a_idx, max_idx)[:-1]
        lower = _process(S, list(range(n_bp)), max_idx, a_idx)[:-1]
        idx_list = upper + lower
        n_idx = len(idx_list)

        # Extract hull vertex coordinates.
        x = np.full(n_idx, fill_value=np.nan, dtype=np.float64)
        y = np.full(n_idx, fill_value=np.nan, dtype=np.float64)
        for j in range(n_idx):
            x[j] = S[idx_list[j], 0]
            y[j] = S[idx_list[j], 1]

        # Sort vertices by polar angle from centroid. Matches the
        # numba reference exactly.
        x0 = float(np.mean(x))
        y0 = float(np.mean(y))
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        # angles = arccos((x - x0) / r) for upper half, else 2π - arccos(...)
        angles = np.where(
            (y - y0) > 0,
            np.arccos((x - x0) / r),
            2 * np.pi - np.arccos((x - x0) / r),
        )
        mask = np.argsort(angles)
        x_sorted = x[mask]
        y_sorted = y[mask]

        if target == "perimeter":
            results[i] = _perimeter(x_sorted, y_sorted)
        elif target == "area":
            results[i] = _area(x_sorted, y_sorted)

    return results


cdef double _perimeter(
    cnp.ndarray[cnp.float64_t, ndim=1] x,
    cnp.ndarray[cnp.float64_t, ndim=1] y,
):
    """Sum of polygon edge lengths (with wrap)."""
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef double dx, dy, total = 0.0
    if n == 0:
        return 0.0
    # Wrap edge: last → first. Numba does
    # ``perimeter = norm(xy[0] - xy[-1])`` outside the loop
    # then sums norm(xy[i] - xy[i+1]) for i in 0..n-1.
    dx = x[0] - x[n - 1]
    dy = y[0] - y[n - 1]
    total = sqrt(dx * dx + dy * dy)
    for i in range(n - 1):
        dx = x[i] - x[i + 1]
        dy = y[i] - y[i + 1]
        total += sqrt(dx * dx + dy * dy)
    return total


cdef double _area(
    cnp.ndarray[cnp.float64_t, ndim=1] x,
    cnp.ndarray[cnp.float64_t, ndim=1] y,
):
    """Polygon area via the shoelace formula. Matches the numba
    reference: 0.5 * |dot(x, np.roll(y, 1)) - dot(y, np.roll(x, 1))|.

    Critical: with cdivision=True (set at module level), C's
    ``%`` operator is sign-preserving — ``(-1) % n`` returns
    ``-1``, NOT ``n - 1`` like Python's ``%``. Combined with
    ``wraparound=False`` (also set at module level), reading
    ``y[-1]`` would be undefined behavior. We update ``prev``
    explicitly each iteration instead, starting it at ``n - 1``
    so the first iteration reads the wrap element correctly.
    Pre-fix code used ``prev = (i - 1) % n`` and produced
    1e6-magnitude errors from out-of-bounds reads on i=0.
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i, prev
    cdef double sum1 = 0.0
    cdef double sum2 = 0.0
    if n == 0:
        return 0.0
    # Start prev at n-1 so iteration 0 reads y[n-1] / x[n-1] for
    # the wrap edge. Subsequent iterations advance prev to i-1.
    prev = n - 1
    for i in range(n):
        sum1 += x[i] * y[prev]
        sum2 += y[i] * x[prev]
        prev = i  # for next iteration, prev = current i
    return 0.5 * fabs(sum1 - sum2)


__all__ = ["jitted_hull"]
