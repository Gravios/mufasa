"""Test for framewise_inside_rectangle_roi O(N²) → O(N) fix.

The numba reference implementation used a roundabout approach:
  1. Build within_x_idx via np.argwhere
  2. Build within_y_idx via np.argwhere
  3. For each i in within_x_idx, do np.argwhere(within_y == within_x[i])

That inner search is O(M) where M = len(within_y_idx), so the
whole thing is O(N×M). For typical animal-tracking workloads
where many frames are inside the ROI, M ≈ N/2, so it's
effectively O(N²/2).

The fix is a single-pass O(N) loop checking both x and y
inclusion per frame. Mathematically identical output: a frame
is marked 1 iff x is in range AND y is in range.

This test reimplements the pre-fix algorithm (without numba)
and verifies that the new implementation produces the same
boolean output across a range of input shapes and edge cases.
The numba-decorated kernel itself can't be imported in the
sandbox, so we test the algorithm logic.

    PYTHONPATH=. python tests/smoke_inside_rectangle_fix.py
"""
from __future__ import annotations

import sys

import numpy as np


def _legacy_inside_rectangle(bp_location, roi_coords):
    """Reimplements the pre-fix algorithm in pure numpy. Used only
    as a reference oracle for the new implementation — the actual
    pre-fix code is gone from feature_extraction_mixin.py."""
    results = np.zeros(bp_location.shape[0], dtype=np.int64)
    within_x_idx = np.argwhere(
        (bp_location[:, 0] <= roi_coords[1][0])
        & (bp_location[:, 0] >= roi_coords[0][0])
    ).flatten()
    within_y_idx = np.argwhere(
        (bp_location[:, 1] <= roi_coords[1][1])
        & (bp_location[:, 1] >= roi_coords[0][1])
    ).flatten()
    for i in range(within_x_idx.shape[0]):
        match = np.argwhere(within_y_idx == within_x_idx[i])
        if match.shape[0] > 0:
            results[within_x_idx[i]] = 1
    return results


def _new_inside_rectangle(bp_location, roi_coords):
    """The post-fix algorithm in pure numpy (no numba decorators).
    This mirrors the body of the new
    FeatureExtractionMixin.framewise_inside_rectangle_roi exactly."""
    results = np.zeros(bp_location.shape[0], dtype=np.int64)
    tlx = roi_coords[0][0]
    tly = roi_coords[0][1]
    brx = roi_coords[1][0]
    bry = roi_coords[1][1]
    for i in range(bp_location.shape[0]):
        x = bp_location[i, 0]
        y = bp_location[i, 1]
        if x >= tlx and x <= brx and y >= tly and y <= bry:
            results[i] = 1
    return results


def main() -> int:
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # Case 1: small input, all combinations of inside/outside
    # ------------------------------------------------------------------ #
    bp = np.array([
        [5.0, 5.0],     # inside
        [0.0, 5.0],     # left edge — inside (>=)
        [10.0, 5.0],    # right edge — inside (<=)
        [5.0, 0.0],     # top edge — inside
        [5.0, 10.0],    # bottom edge — inside
        [-1.0, 5.0],    # outside left
        [11.0, 5.0],    # outside right
        [5.0, -1.0],    # outside top
        [5.0, 11.0],    # outside bottom
        [-1.0, -1.0],   # corner outside
    ], dtype=np.float64)
    roi = np.array([[0.0, 0.0], [10.0, 10.0]])
    legacy = _legacy_inside_rectangle(bp, roi)
    new = _new_inside_rectangle(bp, roi)
    assert np.array_equal(legacy, new), (
        f"Mismatch on hand-crafted boundary cases:\n"
        f"  legacy: {legacy.tolist()}\n"
        f"  new:    {new.tolist()}"
    )
    expected = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert np.array_equal(new, expected), (
        f"new impl produces unexpected output: {new.tolist()}, "
        f"expected {expected.tolist()}"
    )

    # ------------------------------------------------------------------ #
    # Case 2: random inputs at three scales
    # ------------------------------------------------------------------ #
    for n_frames in [100, 10_000, 100_000]:
        bp = rng.uniform(-50, 250, size=(n_frames, 2)).astype(np.float64)
        roi = np.array([[10.0, 20.0], [180.0, 200.0]])
        legacy = _legacy_inside_rectangle(bp, roi)
        new = _new_inside_rectangle(bp, roi)
        assert np.array_equal(legacy, new), (
            f"Mismatch at n_frames={n_frames}: "
            f"{int(np.sum(legacy != new))} of {n_frames} differ"
        )

    # ------------------------------------------------------------------ #
    # Case 3: empty input
    # ------------------------------------------------------------------ #
    bp = np.zeros((0, 2), dtype=np.float64)
    roi = np.array([[0.0, 0.0], [10.0, 10.0]])
    legacy = _legacy_inside_rectangle(bp, roi)
    new = _new_inside_rectangle(bp, roi)
    assert legacy.shape == (0,)
    assert new.shape == (0,)
    assert np.array_equal(legacy, new)

    # ------------------------------------------------------------------ #
    # Case 4: all inside
    # ------------------------------------------------------------------ #
    bp = rng.uniform(1, 9, size=(1000, 2)).astype(np.float64)
    roi = np.array([[0.0, 0.0], [10.0, 10.0]])
    legacy = _legacy_inside_rectangle(bp, roi)
    new = _new_inside_rectangle(bp, roi)
    assert np.array_equal(legacy, new)
    assert int(np.sum(new)) == 1000  # all frames inside

    # ------------------------------------------------------------------ #
    # Case 5: all outside
    # ------------------------------------------------------------------ #
    bp = rng.uniform(100, 200, size=(1000, 2)).astype(np.float64)
    roi = np.array([[0.0, 0.0], [10.0, 10.0]])
    legacy = _legacy_inside_rectangle(bp, roi)
    new = _new_inside_rectangle(bp, roi)
    assert np.array_equal(legacy, new)
    assert int(np.sum(new)) == 0

    # ------------------------------------------------------------------ #
    # Case 6: NaN handling — both should produce 0 for NaN inputs
    # because all comparisons with NaN return False
    # ------------------------------------------------------------------ #
    bp = np.array([
        [5.0, 5.0],
        [np.nan, 5.0],
        [5.0, np.nan],
        [np.nan, np.nan],
    ], dtype=np.float64)
    roi = np.array([[0.0, 0.0], [10.0, 10.0]])
    legacy = _legacy_inside_rectangle(bp, roi)
    new = _new_inside_rectangle(bp, roi)
    assert np.array_equal(legacy, new)
    expected = np.array([1, 0, 0, 0])
    assert np.array_equal(new, expected), (
        f"NaN handling: new={new.tolist()}, expected={expected.tolist()}"
    )

    # ------------------------------------------------------------------ #
    # Case 7: dtype invariant — output is int64 (matches np.full
    # default on Linux)
    # ------------------------------------------------------------------ #
    bp = rng.uniform(0, 10, size=(100, 2)).astype(np.float64)
    roi = np.array([[0.0, 0.0], [10.0, 10.0]])
    new = _new_inside_rectangle(bp, roi)
    assert new.dtype == np.int64, (
        f"output dtype changed: {new.dtype}, expected int64"
    )

    # ------------------------------------------------------------------ #
    # Case 8: AST sanity check — verify the new code in
    # feature_extraction_mixin.py is the single-pass form, not
    # somehow reverted to the np.argwhere version
    # ------------------------------------------------------------------ #
    import ast
    from pathlib import Path
    src = Path(
        "mufasa/mixins/feature_extraction_mixin.py"
    ).read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "framewise_inside_rectangle_roi"):
            found = True
            body_src = ast.unparse(node)
            assert "np.argwhere" not in body_src, (
                "framewise_inside_rectangle_roi still uses "
                "np.argwhere — the O(N²) form is back. The fix "
                "was reverted; investigate."
            )
            assert "tlx" in body_src and "brx" in body_src, (
                "framewise_inside_rectangle_roi doesn't use the "
                "new variables (tlx, brx, etc.) — fix is missing"
            )
    assert found, "framewise_inside_rectangle_roi function not found"

    print("smoke_inside_rectangle_fix: 8/8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
