"""Byte-equivalence + performance comparison for the Cython
inside_polygon kernel.

NOT runnable in the sandbox — requires:
  * The Cython extension compiled (pip install -e .)
  * numba available (for the reference implementation)

Designed for the user to run on the workstation as the
proof-of-concept evaluation:

  cd <mufasa repo>
  python tests/smoke_native_inside_polygon.py

What this does:

1. Generates synthetic polygon + body-part-trajectory data at
   several scales (1K, 50K, 500K frames; simple square and
   complex 12-vertex polygons).
2. Calls both implementations (numba and Cython) on each input.
3. Verifies output is byte-identical (np.array_equal — not just
   allclose, since both produce int 0/1 results).
4. Times both implementations with multiple runs (warm cache,
   discard first run for numba JIT) and reports speedup.

Pass criteria for the POC:
  * Byte-equivalent output across all test cases.
  * Cython runtime within 2× of numba runtime (preferably faster).

If the POC passes, more kernels can be ported. If the POC fails
on byte-equivalence, the kernel logic isn't a faithful port — the
.pyx needs fixing. If the POC passes byte-equivalence but is
significantly slower, the AOT win isn't there for this kernel
shape and we'd skip the larger port.
"""
from __future__ import annotations

import sys
import time

import numpy as np


def _numba_inside_polygon(bp_location, roi_coords):
    """Reference impl — calls the existing numba version."""
    from mufasa.mixins.feature_extraction_mixin import (
        FeatureExtractionMixin,
    )
    return FeatureExtractionMixin.framewise_inside_polygon_roi(
        bp_location=bp_location, roi_coords=roi_coords,
    )


def _cython_inside_polygon(bp_location, roi_coords):
    """Cython impl from mufasa._native."""
    from mufasa._native.inside_polygon import (
        framewise_inside_polygon_roi,
    )
    return framewise_inside_polygon_roi(bp_location, roi_coords)


def _make_test_inputs():
    """Generate (label, bp_location, roi_coords) tuples covering a
    range of frame counts and polygon complexities."""
    rng = np.random.default_rng(42)

    # Simple square polygon
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    # Complex 12-vertex polygon (irregular, concave)
    complex_poly = np.array([
        [0, 0], [5, 1], [10, 0], [11, 5], [10, 10],
        [5, 9], [0, 10], [1, 7], [-1, 5], [1, 3],
        [3, 4], [2, 2],
    ], dtype=np.float64)

    cases = []
    for n_frames, label_size in [
        (1_000, "1K"),
        (50_000, "50K"),
        (500_000, "500K"),
    ]:
        # Body parts uniformly distributed in [-5, 15] (mix of inside
        # and outside the polygon).
        bp = rng.uniform(-5, 15, size=(n_frames, 2)).astype(np.float64)
        cases.append(
            (f"{label_size}_frames_simple_square", bp, square)
        )
        cases.append(
            (f"{label_size}_frames_complex_12pt", bp, complex_poly)
        )
    return cases


def _time_call(fn, *args, n_repeats: int = 3) -> float:
    """Return median elapsed seconds across n_repeats calls.

    First call is discarded for warm-up (numba JIT).
    """
    # Warmup
    fn(*args)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    # Verify both implementations are importable
    try:
        from mufasa._native.inside_polygon import (
            framewise_inside_polygon_roi,
        )
        print("✓ Cython kernel imported")
    except ImportError as exc:
        print(f"✗ Could not import Cython kernel: {exc}")
        print("  Did you run `pip install -e .` after applying "
              "the Cython POC patch?")
        return 1

    try:
        from mufasa.mixins.feature_extraction_mixin import (
            FeatureExtractionMixin,
        )
        print("✓ Numba kernel imported")
    except ImportError as exc:
        print(f"✗ Could not import Numba kernel: {exc}")
        return 1

    cases = _make_test_inputs()
    print(f"\nRunning {len(cases)} test cases...\n")

    all_equal = True
    speedups = []
    for label, bp_location, roi_coords in cases:
        # Byte-equivalence check
        result_numba = _numba_inside_polygon(bp_location, roi_coords)
        result_cython = _cython_inside_polygon(bp_location, roi_coords)
        equal = np.array_equal(result_numba, result_cython)
        if not equal:
            all_equal = False
            n_diff = int(np.sum(result_numba != result_cython))
            print(f"  ✗ [{label}] byte-equivalence FAIL: "
                  f"{n_diff}/{len(result_numba)} elements differ")
            # Show first few mismatches for debugging
            diff_idx = np.where(result_numba != result_cython)[0][:5]
            for idx in diff_idx:
                print(f"      frame {idx}: bp={bp_location[idx]} "
                      f"numba={result_numba[idx]} "
                      f"cython={result_cython[idx]}")
            continue

        # Timing comparison
        t_numba = _time_call(
            _numba_inside_polygon, bp_location, roi_coords,
        )
        t_cython = _time_call(
            _cython_inside_polygon, bp_location, roi_coords,
        )
        speedup = t_numba / t_cython if t_cython > 0 else float("inf")
        speedups.append(speedup)
        print(
            f"  ✓ [{label}] equal | "
            f"numba: {t_numba * 1000:7.2f}ms  "
            f"cython: {t_cython * 1000:7.2f}ms  "
            f"speedup: {speedup:5.2f}×"
        )

    print()
    if all_equal:
        median_speedup = sorted(speedups)[len(speedups) // 2]
        print(f"Byte-equivalence: PASS ({len(cases)} cases)")
        print(f"Median Cython/numba speedup: {median_speedup:.2f}×")
        if median_speedup >= 1.0:
            print("Cython is at least as fast as numba — POC validates.")
        elif median_speedup >= 0.5:
            print("Cython is slower but within 2× — POC marginally "
                  "validates. Worth pursuing for the libgomp-removal "
                  "benefit, not for raw speed.")
        else:
            print("Cython is significantly slower. POC does not "
                  "validate. Investigate before porting more kernels.")
        return 0
    print("Byte-equivalence: FAIL — see mismatches above.")
    print("DO NOT use the Cython kernel until the diff is "
          "investigated. The .pyx logic likely has a subtle "
          "deviation from the numba version.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
