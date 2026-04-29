# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
mufasa._native.hull
====================

Parallel Cython port of
``mufasa.feature_extractors.perimeter_jit.jitted_hull``.

Algorithm: per-frame convex hull of (n_body_parts, 2) points
via QuickHull, then either perimeter (sum of edge lengths) or
area (shoelace formula).

Implementation notes
--------------------

The outer per-frame loop runs in parallel via OpenMP ``prange``.
The per-frame body runs entirely in ``nogil`` mode — no numpy
calls, no Python lists, no recursive Python helpers. QuickHull
is implemented iteratively with an explicit stack (recursion in
nogil mode is awkward and the iterative form is portable).

Tradeoffs accepted:
- Parallelism reintroduces libgomp dependency for this kernel.
  Not a regression because the rest of Mufasa imports numba
  elsewhere, so libgomp is loaded by the time feature
  extraction starts. The "no libgomp" purity goal was never
  achievable without porting all of Mufasa's numba code.
- Fixed-size scratch buffers (per thread) cap supported
  body-part count at MAX_BP = 64. All known callers use ≤30.
  Exceeding 64 raises ValueError before the parallel section.
- Polar-angle sort uses insertion sort instead of np.argsort.
  For n_hull_vertices < 20 (always true given body-part limits)
  insertion sort is faster than quicksort and trivial to write
  nogil.

Output is byte-equivalent to the previous (single-threaded)
Cython version (within the existing 1e-4 tolerance for hull
operations). Each thread gets its own scratch buffer set; no
shared state.
"""
import numpy as np

cimport cython
cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport sqrt, fabs, acos, M_PI
from libc.stdlib cimport malloc, free


# Maximum body parts supported. Sized for typical use (≤30) with
# headroom. If you have a use case with > 64 body parts, increase
# this constant — scratch arrays scale linearly with it (or
# quadratically for the QuickHull pool buffer).
DEF MAX_BP = 64


# ---------------------------------------------------------------- #
# Iterative QuickHull. Pure C, nogil.
#
# Builds the convex hull of 2D points. Mirrors the recursive
# algorithm in numba's perimeter_jit.process(), but iterative
# with an explicit stack to avoid Python recursion.
#
# Returns: number of hull vertices written to hull_idx.
# ---------------------------------------------------------------- #
cdef int _quickhull(
    float* S_x, float* S_y, int n,
    int* hull_idx,
    int* stack_a, int* stack_b,
    int* stack_P_offset, int* stack_P_size,
    int* P_pool,
    int* work_buf,
) noexcept nogil:
    """Iterative QuickHull. Returns hull vertex count."""
    cdef int n_hull = 0
    cdef int i, j
    cdef int a_idx = 0
    cdef int b_idx = 0
    cdef int top = 0
    cdef int pool_top = 0
    cdef int p_off, p_size, current_a, current_b
    cdef int n_above, c_idx
    cdef double ax, ay, bx, by, dab_x, dab_y
    cdef double dx, dy, sd, max_sd
    cdef int idx
    cdef int next_p_off

    if n < 3:
        # Degenerate: 0, 1, or 2 points. Emit all as "hull" — the
        # caller's polar-angle code path produces 0 area / 0
        # perimeter for 0-2 vertices.
        for i in range(n):
            hull_idx[n_hull] = i
            n_hull += 1
        return n_hull

    # Find leftmost (a) and rightmost (b).
    a_idx = 0
    b_idx = 0
    for i in range(1, n):
        if S_x[i] < S_x[a_idx]:
            a_idx = i
        if S_x[i] > S_x[b_idx]:
            b_idx = i

    # Process upper hull (a → b). The numba reference returned
    # process(a,b)[:-1] + process(b,a)[:-1] — [:-1] drops the
    # last index from each chain so the seam isn't duplicated.
    # We emit all but the last vertex of each chain.

    # Initialize P = all indices.
    for i in range(n):
        P_pool[i] = i
    pool_top = n

    stack_a[top] = a_idx
    stack_b[top] = b_idx
    stack_P_offset[top] = 0
    stack_P_size[top] = n
    top += 1

    while top > 0:
        top -= 1
        current_a = stack_a[top]
        current_b = stack_b[top]
        p_off = stack_P_offset[top]
        p_size = stack_P_size[top]

        if p_size == 0:
            hull_idx[n_hull] = current_a
            n_hull += 1
            continue

        ax = S_x[current_a]
        ay = S_y[current_a]
        bx = S_x[current_b]
        by = S_y[current_b]
        dab_x = bx - ax
        dab_y = by - ay
        n_above = 0
        c_idx = -1
        max_sd = -1e308
        for i in range(p_size):
            idx = P_pool[p_off + i]
            dx = S_x[idx] - ax
            dy = S_y[idx] - ay
            sd = dx * dab_y - dy * dab_x
            if sd > max_sd:
                max_sd = sd
                c_idx = idx
            if sd > 0 and idx != current_a and idx != current_b:
                work_buf[n_above] = idx
                n_above += 1

        if n_above == 0:
            hull_idx[n_hull] = current_a
            n_hull += 1
            continue

        # Recursive equivalent: process(a,c) then process(c,b).
        # Push (c,b) first so (a,c) processes first (LIFO).
        # Both subproblems share the SAME filtered P (points above
        # the (a,b) line). Copy work_buf into pool at next_p_off.
        next_p_off = pool_top
        for j in range(n_above):
            P_pool[next_p_off + j] = work_buf[j]
        pool_top += n_above

        # Push (c, b) — processed second
        stack_a[top] = c_idx
        stack_b[top] = current_b
        stack_P_offset[top] = next_p_off
        stack_P_size[top] = n_above
        top += 1

        # Push (a, c) — processed first
        stack_a[top] = current_a
        stack_b[top] = c_idx
        stack_P_offset[top] = next_p_off
        stack_P_size[top] = n_above
        top += 1

    # Process lower hull (b → a). Same pattern, reset pool.
    for i in range(n):
        P_pool[i] = i
    pool_top = n

    stack_a[top] = b_idx
    stack_b[top] = a_idx
    stack_P_offset[top] = 0
    stack_P_size[top] = n
    top += 1

    while top > 0:
        top -= 1
        current_a = stack_a[top]
        current_b = stack_b[top]
        p_off = stack_P_offset[top]
        p_size = stack_P_size[top]

        if p_size == 0:
            hull_idx[n_hull] = current_a
            n_hull += 1
            continue

        ax = S_x[current_a]
        ay = S_y[current_a]
        bx = S_x[current_b]
        by = S_y[current_b]
        dab_x = bx - ax
        dab_y = by - ay
        n_above = 0
        c_idx = -1
        max_sd = -1e308
        for i in range(p_size):
            idx = P_pool[p_off + i]
            dx = S_x[idx] - ax
            dy = S_y[idx] - ay
            sd = dx * dab_y - dy * dab_x
            if sd > max_sd:
                max_sd = sd
                c_idx = idx
            if sd > 0 and idx != current_a and idx != current_b:
                work_buf[n_above] = idx
                n_above += 1

        if n_above == 0:
            hull_idx[n_hull] = current_a
            n_hull += 1
            continue

        next_p_off = pool_top
        for j in range(n_above):
            P_pool[next_p_off + j] = work_buf[j]
        pool_top += n_above

        stack_a[top] = c_idx
        stack_b[top] = current_b
        stack_P_offset[top] = next_p_off
        stack_P_size[top] = n_above
        top += 1

        stack_a[top] = current_a
        stack_b[top] = c_idx
        stack_P_offset[top] = next_p_off
        stack_P_size[top] = n_above
        top += 1

    return n_hull


# ---------------------------------------------------------------- #
# Polar-angle sort + perimeter/area. Pure C, nogil.
# ---------------------------------------------------------------- #
cdef double _compute_hull_metric(
    float* S_x, float* S_y,
    int* hull_idx, int n_hull,
    double* hull_x, double* hull_y, double* angles,
    int* sort_perm,
    int target_is_perimeter,
) noexcept nogil:
    """Given hull vertex indices, compute perimeter or area.

    Mirrors the numba reference exactly:
    1. Extract hull vertex coords (float32 → float64)
    2. Compute centroid (mean of hull coordinates)
    3. Compute polar angles via arccos / 2π - arccos
    4. Sort by polar angle (insertion sort)
    5. Perimeter: sum of edge lengths (with wrap)
       Area: shoelace formula
    """
    cdef int i, j, k
    cdef double x0 = 0.0
    cdef double y0 = 0.0
    cdef double dx, dy, r_i, total
    cdef double sum1, sum2
    cdef int prev
    cdef double tmp_angle
    cdef int tmp_perm

    if n_hull == 0:
        return 0.0

    # 1. Extract coords + accumulate centroid.
    for i in range(n_hull):
        hull_x[i] = <double>S_x[hull_idx[i]]
        hull_y[i] = <double>S_y[hull_idx[i]]
        x0 += hull_x[i]
        y0 += hull_y[i]
    x0 = x0 / n_hull
    y0 = y0 / n_hull

    # 2. Compute polar angle.
    # Numba: angles = where(y - y0 > 0, arccos((x-x0)/r), 2π - arccos((x-x0)/r))
    for i in range(n_hull):
        dx = hull_x[i] - x0
        dy = hull_y[i] - y0
        r_i = sqrt(dx * dx + dy * dy)
        if r_i == 0.0:
            angles[i] = 0.0
        elif dy > 0:
            angles[i] = acos(dx / r_i)
        else:
            angles[i] = 2.0 * M_PI - acos(dx / r_i)
        sort_perm[i] = i

    # 3. Insertion sort sort_perm by angles[].
    for i in range(1, n_hull):
        tmp_perm = sort_perm[i]
        tmp_angle = angles[tmp_perm]
        j = i - 1
        while j >= 0 and angles[sort_perm[j]] > tmp_angle:
            sort_perm[j + 1] = sort_perm[j]
            j -= 1
        sort_perm[j + 1] = tmp_perm

    # 4. Compute metric.
    if target_is_perimeter:
        # Sum of sqrt(dx² + dy²) for each consecutive pair, with wrap.
        total = 0.0
        for i in range(n_hull):
            j = sort_perm[i]
            if i == n_hull - 1:
                k = sort_perm[0]
            else:
                k = sort_perm[i + 1]
            dx = hull_x[k] - hull_x[j]
            dy = hull_y[k] - hull_y[j]
            total += sqrt(dx * dx + dy * dy)
        return total
    else:
        # Shoelace. Track prev explicitly to avoid C-modulo of -1.
        sum1 = 0.0
        sum2 = 0.0
        prev = sort_perm[n_hull - 1]
        for i in range(n_hull):
            j = sort_perm[i]
            sum1 += hull_x[j] * hull_y[prev]
            sum2 += hull_y[j] * hull_x[prev]
            prev = j
        return 0.5 * fabs(sum1 - sum2)


# ---------------------------------------------------------------- #
# Public API. Outer loop is parallel via prange.
# ---------------------------------------------------------------- #
@cython.boundscheck(False)
@cython.wraparound(False)
def jitted_hull(
    cnp.ndarray points,
    str target = "perimeter",
) -> cnp.ndarray:
    """Per-frame convex hull perimeter or area.

    :param points: (n_frames, n_body_parts, 2) float32 array.
        Currently supports n_body_parts ≤ 64.
    :param target: ``'perimeter'`` or ``'area'``.
    :return: (n_frames,) float64 array.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=3] pts_arr = (
        np.ascontiguousarray(points, dtype=np.float32)
    )
    cdef Py_ssize_t n_frames = pts_arr.shape[0]
    cdef Py_ssize_t n_bp = pts_arr.shape[1]

    if n_bp > MAX_BP:
        raise ValueError(
            f"jitted_hull: n_body_parts={n_bp} exceeds compiled "
            f"MAX_BP={MAX_BP}. Increase MAX_BP in hull.pyx and "
            f"reinstall."
        )

    # Decide perimeter vs area before the parallel section
    # (string comparison can't be done in nogil).
    cdef int target_is_perimeter = 1 if target == "perimeter" else 0
    if target != "perimeter" and target != "area":
        raise ValueError(
            f"jitted_hull: target must be 'perimeter' or 'area', "
            f"got {target!r}"
        )

    cdef cnp.ndarray[cnp.float64_t, ndim=1] results = np.full(
        n_frames, fill_value=np.nan, dtype=np.float64,
    )

    cdef float[:, :, ::1] pts_view = pts_arr
    cdef double[::1] results_view = results

    # Per-frame scratch buffer sizes.
    # max_hull = 2 * n_bp covers both upper + lower chain
    # stack_size: at most 2*n_bp frames in the iterative stack
    # pool_size: each level may store up to n_bp indices; bound
    #   by 2*n_bp*n_bp = 2*MAX_BP² to be safe with worst-case
    #   recursion depth × per-call P size
    cdef int max_hull = 2 * MAX_BP
    cdef int stack_size = 4 * MAX_BP
    cdef int pool_size = 4 * MAX_BP * MAX_BP

    cdef Py_ssize_t i, k
    cdef int n_hull
    cdef float* S_x_local
    cdef float* S_y_local
    cdef int* hull_idx_local
    cdef int* stack_a_local
    cdef int* stack_b_local
    cdef int* stack_P_off_local
    cdef int* stack_P_size_local
    cdef int* P_pool_local
    cdef int* work_buf_local
    cdef double* hull_x_local
    cdef double* hull_y_local
    cdef double* angles_local
    cdef int* sort_perm_local

    # Parallel per-frame loop. Each iteration allocates its own
    # scratch — wasteful in principle (~500K mallocs for 500K
    # frames), but each malloc is ~50ns so total overhead is
    # ~25ms vs ~1s of actual work. Negligible.
    #
    # The "right" pattern is to allocate inside `with parallel():`
    # and reuse across iterations. Doing this correctly with
    # Cython's nogil rules is more involved — saving for a later
    # tuning pass if profiling shows malloc cost matters.
    with nogil:
        for i in prange(n_frames, schedule="static"):
            S_x_local = <float*>malloc(n_bp * sizeof(float))
            S_y_local = <float*>malloc(n_bp * sizeof(float))
            hull_idx_local = <int*>malloc(max_hull * sizeof(int))
            stack_a_local = <int*>malloc(stack_size * sizeof(int))
            stack_b_local = <int*>malloc(stack_size * sizeof(int))
            stack_P_off_local = <int*>malloc(stack_size * sizeof(int))
            stack_P_size_local = <int*>malloc(stack_size * sizeof(int))
            P_pool_local = <int*>malloc(pool_size * sizeof(int))
            work_buf_local = <int*>malloc(n_bp * sizeof(int))
            hull_x_local = <double*>malloc(max_hull * sizeof(double))
            hull_y_local = <double*>malloc(max_hull * sizeof(double))
            angles_local = <double*>malloc(max_hull * sizeof(double))
            sort_perm_local = <int*>malloc(max_hull * sizeof(int))

            # Copy frame data into typed C buffers.
            for k in range(n_bp):
                S_x_local[k] = pts_view[i, k, 0]
                S_y_local[k] = pts_view[i, k, 1]

            n_hull = _quickhull(
                S_x_local, S_y_local, <int>n_bp,
                hull_idx_local,
                stack_a_local, stack_b_local,
                stack_P_off_local, stack_P_size_local,
                P_pool_local,
                work_buf_local,
            )

            results_view[i] = _compute_hull_metric(
                S_x_local, S_y_local,
                hull_idx_local, n_hull,
                hull_x_local, hull_y_local, angles_local,
                sort_perm_local,
                target_is_perimeter,
            )

            free(S_x_local)
            free(S_y_local)
            free(hull_idx_local)
            free(stack_a_local)
            free(stack_b_local)
            free(stack_P_off_local)
            free(stack_P_size_local)
            free(P_pool_local)
            free(work_buf_local)
            free(hull_x_local)
            free(hull_y_local)
            free(angles_local)
            free(sort_perm_local)

    return results


__all__ = ["jitted_hull"]
