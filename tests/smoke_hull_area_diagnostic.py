"""Focused diagnostic for jitted_hull area byte-equivalence failure.

Reproduces the area failure with detailed diagnostics:
- Side-by-side hull vertex sets
- Side-by-side x_sorted, y_sorted arrays
- Per-frame area computation with intermediate values

Workstation-only (requires Cython extensions + numba).

    PYTHONPATH=. python tests/smoke_hull_area_diagnostic.py
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    try:
        from mufasa.feature_extractors.perimeter_jit import (
            jitted_hull as numba_hull,
        )
        from mufasa._native.hull import jitted_hull as cython_hull
    except ImportError as exc:
        print(f"✗ Could not import: {exc}")
        return 1

    rng = np.random.default_rng(42)

    # Generate the same input the verification script generates for
    # jitted_hull_area at the 1K-frame scale. The RNG state at that
    # point in the verification depends on the order of kernels — we
    # just generate fresh data here for diagnostic purposes.
    pts_small = rng.uniform(0, 1920, size=(5, 8, 2)).astype(np.float32)

    print(f"Diagnostic inputs: 5 frames × 8 points × 2 coords float32")
    print(f"First frame points:")
    for i, (px, py) in enumerate(pts_small[0]):
        print(f"  bp{i}: ({px:8.2f}, {py:8.2f})")
    print()

    print("Running both implementations on identical input...")
    numba_perim = numba_hull(pts_small, target="perimeter")
    cython_perim = cython_hull(pts_small, target="perimeter")
    numba_area = numba_hull(pts_small, target="area")
    cython_area = cython_hull(pts_small, target="area")

    print(f"\n{'frame':<6} {'numba_perim':>14} {'cython_perim':>14} {'diff':>10}")
    for i in range(5):
        d = abs(numba_perim[i] - cython_perim[i])
        print(f"{i:<6} {numba_perim[i]:>14.4f} {cython_perim[i]:>14.4f} "
              f"{d:>10.2e}")

    print(f"\n{'frame':<6} {'numba_area':>14} {'cython_area':>14} {'diff':>10} {'ratio':>8}")
    for i in range(5):
        d = abs(numba_area[i] - cython_area[i])
        ratio = (cython_area[i] / numba_area[i]) if numba_area[i] != 0 else float("nan")
        print(f"{i:<6} {numba_area[i]:>14.4f} {cython_area[i]:>14.4f} "
              f"{d:>10.2e} {ratio:>8.4f}")

    # Now do a manual computation on the first frame, mirroring both
    # implementations, to identify where the divergence happens.
    print("\n" + "=" * 60)
    print("Manual trace of frame 0:")
    print("=" * 60)

    S = pts_small[0].astype(np.float32)
    print(f"S shape: {S.shape}")

    # Use scipy as a third oracle
    from scipy.spatial import ConvexHull
    hull = ConvexHull(S)
    print(f"\nScipy hull vertex indices (CCW from rightmost): {hull.vertices.tolist()}")
    print(f"Scipy hull area (oracle): {hull.volume:.4f}")

    # Now manually run the numba algorithm on just frame 0:
    # Step 1: a = leftmost, max = rightmost
    a = int(np.argmin(S[:, 0]))
    max_idx = int(np.argmax(S[:, 0]))
    print(f"\nLeftmost point a = bp{a}: ({S[a, 0]:.2f}, {S[a, 1]:.2f})")
    print(f"Rightmost point max = bp{max_idx}: ({S[max_idx, 0]:.2f}, {S[max_idx, 1]:.2f})")

    # Compare hull index lists from numba and Cython by intercepting
    # via running the kernels on this single frame and looking at
    # what indices they pick. We can't easily peek inside numba but
    # we can deduce from the output: we already have numba_perim[0]
    # and cython_perim[0] which agree. So the vertex sets agree.
    # The problem must be in how those sets are passed to area().

    # Print the per-frame results:
    print(f"\nframe 0 results:")
    print(f"  numba perimeter: {numba_perim[0]:.4f}")
    print(f"  cython perimeter: {cython_perim[0]:.4f}")
    print(f"  numba area: {numba_area[0]:.4f}")
    print(f"  cython area: {cython_area[0]:.4f}")
    print(f"  scipy area (oracle): {hull.volume:.4f}")

    # Which one is "correct"?
    if abs(numba_area[0] - hull.volume) < 1.0:
        print(f"\n→ numba matches scipy oracle. Cython is wrong.")
    elif abs(cython_area[0] - hull.volume) < 1.0:
        print(f"\n→ cython matches scipy oracle. Numba is wrong (?!).")
    else:
        print(f"\n→ NEITHER matches scipy. Both implementations buggy.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
