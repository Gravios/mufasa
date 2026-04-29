"""Structural tests for the full Cython kernel port.

Verifies the .pyx files exist with the right shape, the build
configuration includes all of them, and the existing numba
versions are still in place (since the port is additive — the
kernel chain still calls numba until validation passes).

Runtime byte-equivalence + benchmark is in
smoke_native_all_kernels.py (workstation-only).

    PYTHONPATH=. python tests/smoke_cython_full_port_structure.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


EXPECTED_KERNELS = {
    "inside_polygon":    "framewise_inside_polygon_roi",
    "inside_rectangle":  "framewise_inside_rectangle_roi",
    "inside_circle":     "is_inside_circle",
    "euclidean_distance": "framewise_euclidean_distance",  # also has _roi variant
    "angle3pt":          "angle3pt_vectorized",
    "border_distances":  "border_distances",
    "hull":              "jitted_hull",
}


def main() -> int:
    # ------------------------------------------------------------------ #
    # Case 1: each .pyx file exists with required compiler directives
    # ------------------------------------------------------------------ #
    required_directives = [
        "language_level=3", "boundscheck=False", "wraparound=False",
    ]
    for module_name, _ in EXPECTED_KERNELS.items():
        pyx = Path(f"mufasa/_native/{module_name}.pyx")
        assert pyx.is_file(), f"missing {pyx}"
        src = pyx.read_text()
        for d in required_directives:
            assert d in src, (
                f"{pyx.name} missing compiler directive {d!r}"
            )

    # ------------------------------------------------------------------ #
    # Case 2: each kernel function is defined in its module
    # ------------------------------------------------------------------ #
    for module_name, fn_name in EXPECTED_KERNELS.items():
        pyx = Path(f"mufasa/_native/{module_name}.pyx")
        src = pyx.read_text()
        assert f"def {fn_name}(" in src, (
            f"{pyx.name} missing function {fn_name}"
        )

    # ------------------------------------------------------------------ #
    # Case 3: euclidean_distance.pyx has both forms
    # ------------------------------------------------------------------ #
    ed = Path("mufasa/_native/euclidean_distance.pyx").read_text()
    assert "def framewise_euclidean_distance(" in ed
    assert "def framewise_euclidean_distance_roi(" in ed

    # ------------------------------------------------------------------ #
    # Case 4: setup.py declares all 7 extensions
    # ------------------------------------------------------------------ #
    setup = Path("setup.py").read_text()
    expected_ext_names = [
        "mufasa._native.inside_polygon",
        "mufasa._native.inside_rectangle",
        "mufasa._native.inside_circle",
        "mufasa._native.euclidean_distance",
        "mufasa._native.angle3pt",
        "mufasa._native.border_distances",
        "mufasa._native.hull",
    ]
    for name in expected_ext_names:
        assert name in setup, (
            f"setup.py doesn't declare extension {name!r}"
        )
    expected_sources = [
        "inside_polygon.pyx",
        "inside_rectangle.pyx",
        "inside_circle.pyx",
        "euclidean_distance.pyx",
        "angle3pt.pyx",
        "border_distances.pyx",
        "hull.pyx",
    ]
    for src_name in expected_sources:
        assert src_name in setup, (
            f"setup.py doesn't reference source file {src_name!r}"
        )

    # ------------------------------------------------------------------ #
    # Case 5: -ffast-math NOT in any extra_compile_args entry
    # ------------------------------------------------------------------ #
    args_blocks = re.findall(
        r"extra_compile_args\s*=\s*\[([^\]]*)\]",
        setup,
    )
    assert args_blocks, "setup.py extra_compile_args missing"
    for block in args_blocks:
        assert "-ffast-math" not in block, (
            "setup.py extra_compile_args must NOT include "
            "-ffast-math; it would break byte-equivalence"
        )

    # ------------------------------------------------------------------ #
    # Case 6: existing numba kernels are STILL in place. The port
    # is additive — production behavior unchanged.
    # ------------------------------------------------------------------ #
    mixin = Path(
        "mufasa/mixins/feature_extraction_mixin.py"
    ).read_text()
    for original_fn in [
        "framewise_inside_polygon_roi",
        "framewise_inside_rectangle_roi",
        "is_inside_circle",
        "framewise_euclidean_distance",
        "framewise_euclidean_distance_roi",
        "angle3pt_vectorized",
    ]:
        assert f"def {original_fn}(" in mixin, (
            f"Original numba {original_fn} removed from "
            f"FeatureExtractionMixin — port should be ADDITIVE, "
            f"not replacing the production kernel until "
            f"workstation verification completes"
        )

    supplement = Path(
        "mufasa/mixins/feature_extraction_supplement_mixin.py"
    ).read_text()
    assert "def border_distances(" in supplement, (
        "Original numba border_distances removed from "
        "FeatureExtractionSupplemental — port should be additive"
    )

    perimeter_jit = Path(
        "mufasa/feature_extractors/perimeter_jit.py"
    ).read_text()
    assert "def jitted_hull(" in perimeter_jit, (
        "Original numba jitted_hull removed from perimeter_jit.py "
        "— port should be additive"
    )

    # ------------------------------------------------------------------ #
    # Case 7: kernel chain (feature_subset_kernels.py) still uses
    # the numba imports — Cython kernels exist but aren't yet wired
    # into production.
    # ------------------------------------------------------------------ #
    kernels = Path(
        "mufasa/feature_extractors/feature_subset_kernels.py"
    ).read_text()
    assert "_native" not in kernels, (
        "feature_subset_kernels.py imports from mufasa._native — "
        "Cython kernels are not yet supposed to be wired into "
        "production. Wire them in a follow-up patch only after "
        "smoke_native_all_kernels.py validates byte-equivalence."
    )

    # ------------------------------------------------------------------ #
    # Case 8: pyproject.toml [build-system] still requires Cython
    # + numpy — these were added by the POC patch and must remain
    # ------------------------------------------------------------------ #
    pyproject = Path("pyproject.toml").read_text()
    bs = re.search(
        r"\[build-system\](.*?)(?=\n\[)",
        pyproject,
        re.DOTALL,
    ).group(1)
    assert "Cython" in bs
    assert "numpy" in bs.lower()

    # ------------------------------------------------------------------ #
    # Case 9: hull.pyx implements both perimeter and area branches.
    # The parallel rewrite uses an int flag (target_is_perimeter)
    # rather than string comparison inside the nogil section, so
    # the literal `target == "area"` no longer appears in the
    # dispatch — only `!= "area"` in input validation. Verify the
    # branch logic exists via the flag-based dispatch.
    # ------------------------------------------------------------------ #
    hull = Path("mufasa/_native/hull.pyx").read_text()
    assert 'target == "perimeter"' in hull, (
        "hull.pyx must distinguish target='perimeter'"
    )
    assert '"area"' in hull, (
        "hull.pyx must reference area mode somewhere"
    )
    assert "target_is_perimeter" in hull, (
        "hull.pyx should use a perimeter/area flag inside the "
        "nogil section (string comparison can't be done nogil)"
    )
    assert "_compute_hull_metric" in hull, (
        "hull.pyx should have a _compute_hull_metric helper"
    )
    # Has the QuickHull (formerly recursive, now iterative)
    assert "_quickhull(" in hull, (
        "hull.pyx should have a _quickhull helper"
    )

    # ------------------------------------------------------------------ #
    # Case 10: angle3pt uses libc.math (atan2) for byte-equivalence,
    # not numba's potentially fast-math'd atan2
    # ------------------------------------------------------------------ #
    angle = Path("mufasa/_native/angle3pt.pyx").read_text()
    assert "from libc.math cimport" in angle
    assert "atan2" in angle

    print("smoke_cython_full_port_structure: 10/10 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
