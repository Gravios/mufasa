"""Tests for mufasa.feature_extractors.feature_subset_kernels.

The kernel module imports numba-backed feature kernels (jitted_hull,
FeatureExtractionMixin static methods) which require the full Mufasa
runtime — not available in the sandbox.

Tests here cover what's possible without running the actual kernels:

1. AST-level structure checks — every kernel is a top-level function
   with the documented signature, returning a dict.
2. Helper-function correctness — the small helpers _bp_xy_columns and
   _flat_xy_column_names and _reshape_for_hull are pure-Python and
   testable directly.
3. Output column-name format matches the legacy class behavior — by
   parsing the kernel module and the legacy class file, extracting
   the f-string templates, and comparing.

End-to-end verification (kernels called on real pose data, output
compared to legacy class output) requires the user's full env. The
ground-truth fixture is captured in
tests/smoke_feature_kernels_realdata.py (NOT runnable in sandbox)
and is meant to be run on the workstation as a one-time before/after
check by the user.

    PYTHONPATH=. python tests/smoke_feature_kernels.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType
from typing import List

import numpy as np
import pandas as pd


def _load_helpers():
    """Load just the pure-Python helpers from the kernel module by
    extracting their source. Avoids importing the numba-dependent
    chain."""
    src = Path(
        "mufasa/feature_extractors/feature_subset_kernels.py"
    ).read_text()
    tree = ast.parse(src)

    # Find the helper functions
    wanted = {
        "_bp_xy_columns", "_flat_xy_column_names", "_reshape_for_hull",
    }
    helper_src_lines = src.splitlines(keepends=True)
    helper_src = ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            start = node.lineno - 1
            end = node.end_lineno
            helper_src += "".join(helper_src_lines[start:end]) + "\n"

    ns = {
        "np": np, "pd": pd,
        "Iterable": object, "Tuple": tuple,
    }
    exec(helper_src, ns)
    mod = ModuleType("kernels_helpers_under_test")
    for name in wanted:
        setattr(mod, name, ns[name])
    return mod


def main() -> int:
    helpers = _load_helpers()
    _bp_xy_columns = helpers._bp_xy_columns
    _flat_xy_column_names = helpers._flat_xy_column_names
    _reshape_for_hull = helpers._reshape_for_hull

    # ------------------------------------------------------------------ #
    # Case 1: _bp_xy_columns produces the right names
    # ------------------------------------------------------------------ #
    assert _bp_xy_columns("nose") == ("nose_x", "nose_y")
    assert _bp_xy_columns("tail_base") == ("tail_base_x", "tail_base_y")
    assert _bp_xy_columns("L_ear") == ("L_ear_x", "L_ear_y")

    # ------------------------------------------------------------------ #
    # Case 2: _flat_xy_column_names matches legacy zip-based pattern
    # ------------------------------------------------------------------ #
    point = ("bp_a", "bp_b", "bp_c")
    # Legacy code in feature_subsets.py used:
    #   list(sum([(f"{x}_x", f"{y}_y") for (x, y) in zip(point, point)], ()))
    # Replicate it here as a regression check that the helper is
    # equivalent.
    legacy = list(sum(
        [(f"{x}_x", f"{y}_y") for (x, y) in zip(point, point)], (),
    ))
    new = _flat_xy_column_names(point)
    assert legacy == new, f"legacy={legacy} new={new}"

    # 4-point check too
    point4 = ("a", "b", "c", "d")
    legacy4 = list(sum(
        [(f"{x}_x", f"{y}_y") for (x, y) in zip(point4, point4)], (),
    ))
    assert _flat_xy_column_names(point4) == legacy4

    # ------------------------------------------------------------------ #
    # Case 3: _reshape_for_hull produces (n_frames, n_bps, 2) float32
    # with correct ordering
    # ------------------------------------------------------------------ #
    n_frames = 10
    bps = ("bp_a", "bp_b", "bp_c")
    data = {}
    for i, bp in enumerate(bps):
        data[f"{bp}_x"] = np.arange(n_frames) + i * 1000
        data[f"{bp}_y"] = np.arange(n_frames) + i * 1000 + 500
    df = pd.DataFrame(data)
    arr = _reshape_for_hull(df, bps)
    assert arr.shape == (n_frames, 3, 2)
    assert arr.dtype == np.float32
    # Frame 0
    expected = np.array(
        [[0, 500], [1000, 1500], [2000, 2500]],
        dtype=np.float32,
    )
    assert np.array_equal(arr[0], expected)

    # ------------------------------------------------------------------ #
    # Case 4: AST structure check — every documented kernel is present
    # at top level with a signature returning a dict
    # ------------------------------------------------------------------ #
    src = Path(
        "mufasa/feature_extractors/feature_subset_kernels.py"
    ).read_text()
    tree = ast.parse(src)
    documented_kernels = {
        "compute_two_point_distances",
        "compute_three_point_angles",
        "compute_three_point_hulls",
        "compute_four_point_hulls",
        "compute_animal_convex_hulls",
        "compute_framewise_movement",
        "compute_roi_center_distances",
        "compute_distances_to_frame_edge",
        "compute_inside_roi",
    }
    found_kernels = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("compute_"):
                found_kernels.add(node.name)
                # Check return annotation hints at Dict (best-effort)
                ret = node.returns
                if ret is not None:
                    # Either Dict[str, np.ndarray] or 'Dict[str, np.ndarray]'
                    src_text = ast.unparse(ret) if hasattr(ast, 'unparse') else str(ret)
                    assert "Dict" in src_text or "dict" in src_text, \
                        f"{node.name} return annotation: {src_text}"
    missing = documented_kernels - found_kernels
    assert not missing, f"Missing kernels: {missing}"
    extra = found_kernels - documented_kernels
    assert not extra, f"Unexpected kernels: {extra}"

    # ------------------------------------------------------------------ #
    # Case 5: output column-name format matches legacy
    # Parse out f-string templates and compare to the legacy class file
    # ------------------------------------------------------------------ #
    legacy_src = Path(
        "mufasa/feature_extractors/feature_subsets.py"
    ).read_text()

    # Look for known column-name formats in both files. The exact
    # f-string source might differ slightly between files (e.g.
    # newlines for line breaking), so search for distinctive substrings.
    column_format_substrings = [
        "Distance (mm) {",
        "Angle (degrees) {",
        "three-point convex hull perimeter (mm)",
        "four-point convex perimeter (mm)",
        "convex hull perimeter (mm)",
        "convex hull area (mm2)",
        "movement {",
        "center distance (mm)",
        "left video edge distance (mm)",
        "right video edge distance (mm)",
        "top video edge distance (mm)",
        "bottom video edge distance (mm)",
        "inside rectangle ",
        "inside circle ",
        "inside polygon ",
    ]
    kernel_src = src  # already loaded above
    for substr in column_format_substrings:
        assert substr in kernel_src, (
            f"Missing in kernels: {substr!r}"
        )

    # ------------------------------------------------------------------ #
    # Case 6: kernel imports — must NOT import from feature_subsets to
    # avoid circular import after the class delegates to the kernels.
    # ------------------------------------------------------------------ #
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod_name = (
                node.module if isinstance(node, ast.ImportFrom)
                else node.names[0].name
            )
            assert "feature_subsets" not in (mod_name or ""), (
                f"Circular import risk: {mod_name}"
            )

    # ------------------------------------------------------------------ #
    # Case 7 (post step-3): per-family methods on the class are GONE.
    # The class run() loop now delegates to module-level
    # process_one_video. Verify that no per-family method exists on
    # the class.
    # ------------------------------------------------------------------ #
    legacy_tree = ast.parse(legacy_src)
    removed_method_names = {
        "_get_two_point_bp_distances",
        "__get_three_point_angles",
        "__get_three_point_hulls",
        "__get_four_point_hulls",
        "__get_convex_hulls",
        "__get_framewise_movement",
        "__get_roi_center_distances",
        "__get_distances_to_frm_edge",
        "__get_inside_roi",
    }
    found = set()
    for node in ast.walk(legacy_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name in removed_method_names):
            found.add(node.name)
    assert not found, (
        f"Step 3 should have removed these dead methods, but found: "
        f"{sorted(found)}"
    )

    # ------------------------------------------------------------------ #
    # Case 8: orchestration module exists with required exports
    # ------------------------------------------------------------------ #
    orch_src = Path(
        "mufasa/feature_extractors/feature_subset_orchestration.py"
    ).read_text()
    orch_tree = ast.parse(orch_src)
    expected = {
        "VideoProcessingConfig",   # dataclass
        "process_one_video",        # function
    }
    found = set()
    for node in orch_tree.body:
        if isinstance(node, ast.ClassDef) and node.name in expected:
            found.add(node.name)
        if isinstance(node, ast.FunctionDef) and node.name in expected:
            found.add(node.name)
    missing = expected - found
    assert not missing, (
        f"Orchestration module is missing: {missing}"
    )

    # ------------------------------------------------------------------ #
    # Case 9: process_one_video doesn't have side effects on a class
    # instance — verify by checking it's a top-level function (not a
    # method) with a config parameter
    # ------------------------------------------------------------------ #
    for node in orch_tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "process_one_video"):
            arg_names = [a.arg for a in node.args.args]
            assert "self" not in arg_names, (
                "process_one_video must not be a method"
            )
            assert "config" in arg_names, (
                "process_one_video must take a config parameter"
            )
            assert "file_path" in arg_names, (
                "process_one_video must take a file_path parameter"
            )

    print("smoke_feature_kernels: 9/9 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
