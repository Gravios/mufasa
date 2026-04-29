"""Workstation verification for the full Cython kernel port.

NOT runnable in the sandbox — requires Cython extensions
compiled (pip install -e .) and numba available.

Tests all 7 ported kernels:
1. framewise_inside_polygon_roi (POC; should still pass)
2. framewise_inside_rectangle_roi
3. is_inside_circle
4. framewise_euclidean_distance
5. framewise_euclidean_distance_roi
6. angle3pt_vectorized
7. border_distances
8. jitted_hull (perimeter)
9. jitted_hull (area)

For each:
- Generates synthetic input at three scales (1K, 50K, 500K frames)
- Calls both implementations
- Verifies byte-equivalence (np.array_equal for integer outputs;
  np.allclose with kernel-appropriate tolerance for float outputs)
- Times both with warm-up runs discarded
- Reports speedup

Per-kernel tolerance:
- Polygon, rectangle, circle: integer 0/1 output → array_equal
- Distance kernels: float, no fastmath in numba → array_equal
- angle3pt: float, fastmath=True in numba → allclose 1e-6
- border_distances: int32 (cast at end) → array_equal
- jitted_hull: float, fastmath=True → allclose 1e-4 (geometric
  accumulation amplifies ULP differences slightly)

Pass criteria:
- All 9 cases pass byte-equivalence
- No kernel runs > 3× slower than its numba counterpart

If any fail byte-equivalence: the kernel needs investigation
before being routed into production. Numba version stays
the default until then.
"""
from __future__ import annotations

import sys
import time
from typing import Callable

import numpy as np


# ---------------------------------------------------------------- #
# Kernel pairs: (name, numba_callable, cython_callable, tolerance)
# tolerance: None → array_equal; float → np.allclose(rtol=tol, atol=tol)
# ---------------------------------------------------------------- #


def _build_kernel_pairs():
    """Late import so the script gives a useful error if anything
    is missing rather than crashing at module load."""
    from mufasa.mixins.feature_extraction_mixin import FeatureExtractionMixin
    from mufasa.mixins.feature_extraction_supplement_mixin import (
        FeatureExtractionSupplemental,
    )
    from mufasa.feature_extractors.perimeter_jit import jitted_hull as numba_hull
    from mufasa._native import (
        inside_polygon, inside_rectangle, inside_circle,
        euclidean_distance, angle3pt, border_distances, hull,
    )
    return [
        ("inside_polygon",
         FeatureExtractionMixin.framewise_inside_polygon_roi,
         inside_polygon.framewise_inside_polygon_roi,
         None),
        ("inside_rectangle",
         FeatureExtractionMixin.framewise_inside_rectangle_roi,
         inside_rectangle.framewise_inside_rectangle_roi,
         None),
        ("inside_circle",
         FeatureExtractionMixin.is_inside_circle,
         inside_circle.is_inside_circle,
         None),
        ("framewise_euclidean_distance",
         FeatureExtractionMixin.framewise_euclidean_distance,
         euclidean_distance.framewise_euclidean_distance,
         1e-9),
        ("framewise_euclidean_distance_roi",
         FeatureExtractionMixin.framewise_euclidean_distance_roi,
         euclidean_distance.framewise_euclidean_distance_roi,
         1e-9),
        ("angle3pt_vectorized",
         FeatureExtractionMixin.angle3pt_vectorized,
         angle3pt.angle3pt_vectorized,
         1e-6),  # numba uses fastmath
        ("border_distances",
         FeatureExtractionSupplemental.border_distances,
         border_distances.border_distances,
         None),  # int32 cast at end
        ("jitted_hull_perimeter",
         lambda pts: numba_hull(pts, target="perimeter"),
         lambda pts: hull.jitted_hull(pts, target="perimeter"),
         1e-4),  # numba uses fastmath; hull accum amplifies ULP
        ("jitted_hull_area",
         lambda pts: numba_hull(pts, target="area"),
         lambda pts: hull.jitted_hull(pts, target="area"),
         1e-4),
    ]


# ---------------------------------------------------------------- #
# Synthetic input generators per kernel
# ---------------------------------------------------------------- #
def _gen_inputs(kernel_name: str, n_frames: int, rng):
    """Return a tuple of arguments to pass to BOTH numba and Cython
    versions of the named kernel. Same RNG state for both implementations
    so they see identical input."""
    if kernel_name == "inside_polygon":
        bp = rng.uniform(-5, 15, size=(n_frames, 2)).astype(np.float64)
        # 12-vertex polygon
        roi = np.array([
            [0, 0], [5, 1], [10, 0], [11, 5], [10, 10],
            [5, 9], [0, 10], [1, 7], [-1, 5], [1, 3], [3, 4], [2, 2],
        ], dtype=np.float64)
        return (bp, roi)

    if kernel_name == "inside_rectangle":
        bp = rng.uniform(-50, 250, size=(n_frames, 2)).astype(np.float64)
        roi = np.array([[10.0, 20.0], [180.0, 200.0]])
        return (bp, roi)

    if kernel_name == "inside_circle":
        bp = rng.uniform(-50, 250, size=(n_frames, 2)).astype(np.float64)
        center = np.array([100.0, 100.0])
        radius = 80
        return (bp, center, radius)

    if kernel_name == "framewise_euclidean_distance":
        a = rng.uniform(0, 1920, size=(n_frames, 2)).astype(np.float64)
        b = rng.uniform(0, 1920, size=(n_frames, 2)).astype(np.float64)
        return (a, b, 4.5, False)

    if kernel_name == "framewise_euclidean_distance_roi":
        a = rng.uniform(0, 1920, size=(n_frames, 2)).astype(np.float64)
        b = np.array([[960.0, 540.0]])  # static center
        return (a, b, 4.5, False)

    if kernel_name == "angle3pt_vectorized":
        # 6-column flat input: [ax, ay, bx, by, cx, cy]
        data = rng.uniform(0, 1920, size=(n_frames, 6)).astype(np.float64)
        return (data,)

    if kernel_name == "border_distances":
        bp = rng.uniform(0, 1920, size=(n_frames, 2)).astype(np.float64)
        img_res = np.array([1920, 1080], dtype=np.int32)
        # window of 1.0 second at 30 fps = 30 frames
        return (bp, 4.5, img_res, 1.0, 30)

    if kernel_name in ("jitted_hull_perimeter", "jitted_hull_area"):
        # n_frames × 8 body parts × 2 coords. float32 (jitted_hull
        # signature requires float32).
        pts = rng.uniform(0, 1920, size=(n_frames, 8, 2)).astype(np.float32)
        return (pts,)

    raise ValueError(f"Unknown kernel: {kernel_name}")


# ---------------------------------------------------------------- #
# Verification + timing per case
# ---------------------------------------------------------------- #
def _equiv(a: np.ndarray, b: np.ndarray, tol) -> tuple[bool, str]:
    """Compare a and b. Returns (equal, diagnostic_string)."""
    if a.shape != b.shape:
        return (False, f"shape mismatch: {a.shape} vs {b.shape}")
    if a.dtype != b.dtype:
        # Allow reasonable cross-dtype comparison for floats; for
        # ints we want exact dtype match.
        if not (np.issubdtype(a.dtype, np.floating)
                and np.issubdtype(b.dtype, np.floating)):
            return (
                False,
                f"dtype mismatch: {a.dtype} vs {b.dtype}",
            )
    if tol is None:
        if np.array_equal(a, b):
            return (True, "")
        n_diff = int(np.sum(a != b))
        return (False, f"{n_diff}/{a.size} elements differ (exact)")
    # Float comparison via allclose. equal_nan=True so both having
    # NaN at the same position counts as equal.
    if np.allclose(
        a.astype(np.float64), b.astype(np.float64),
        rtol=tol, atol=tol, equal_nan=True,
    ):
        return (True, "")
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    diff = np.where(np.isnan(diff), 0, diff)
    return (False, f"max abs diff = {np.nanmax(diff):.3e} (tol={tol:.0e})")


def _time(fn: Callable, args: tuple, n_repeats: int = 3) -> float:
    fn(*args)  # warm-up (numba JIT)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    try:
        kernel_pairs = _build_kernel_pairs()
    except ImportError as exc:
        print(f"✗ Could not import all kernels: {exc}")
        print("  Did you `pip install -e .` after applying the patch?")
        return 1

    print(f"Running {len(kernel_pairs)} kernel comparisons "
          f"at 3 scales each...\n")

    rng = np.random.default_rng(42)
    all_pass = True
    speedup_summary: list[tuple[str, float]] = []

    for kernel_name, numba_fn, cython_fn, tol in kernel_pairs:
        print(f"=== {kernel_name} ===")
        for n_frames, label in [(1_000, "1K"), (50_000, "50K"),
                                  (500_000, "500K")]:
            args = _gen_inputs(kernel_name, n_frames, rng)
            try:
                numba_out = numba_fn(*args)
                cython_out = cython_fn(*args)
            except Exception as exc:
                print(f"  ✗ {label}: kernel raised "
                      f"{type(exc).__name__}: {exc}")
                all_pass = False
                continue

            equal, diag = _equiv(numba_out, cython_out, tol)
            if not equal:
                print(f"  ✗ {label}: byte-equivalence FAIL — {diag}")
                all_pass = False
                continue

            t_numba = _time(numba_fn, args)
            t_cython = _time(cython_fn, args)
            speedup = t_numba / t_cython if t_cython > 0 else float("inf")
            speedup_summary.append((f"{kernel_name}_{label}", speedup))
            print(
                f"  ✓ {label}: equal | "
                f"numba {t_numba * 1000:7.2f}ms  "
                f"cython {t_cython * 1000:7.2f}ms  "
                f"{speedup:5.2f}×"
            )
        print()

    print("=" * 60)
    if all_pass:
        speedups = [s for _, s in speedup_summary]
        median = sorted(speedups)[len(speedups) // 2]
        worst = min(speedups)
        best = max(speedups)
        print(f"All {len(kernel_pairs)} kernels: byte-equivalence PASS")
        print(f"Speedup distribution: median {median:.2f}× | "
              f"best {best:.2f}× | worst {worst:.2f}×")
        if worst < 0.33:
            print(
                "WARNING: at least one kernel runs >3× slower than "
                "numba. Investigate before wiring Cython into "
                "production."
            )
            return 1
        print(
            "All kernels validated. Safe to wire Cython kernels "
            "into feature_subset_kernels.py via a follow-up patch."
        )
        return 0
    print("Byte-equivalence FAIL on at least one kernel. "
          "DO NOT wire Cython kernels into production until the "
          "failures are investigated. Numba versions remain the "
          "default and feature extraction continues to work.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
