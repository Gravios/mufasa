"""Structural tests for the Cython kernel POC.

The Cython extension can't be imported in the sandbox (no
compilation toolchain). These tests verify the static structure
of the .pyx, the build configuration, and the wiring — the
runtime byte-equivalence + benchmark is in
smoke_native_inside_polygon.py (workstation-only).

    PYTHONPATH=. python tests/smoke_cython_poc_structure.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    # ------------------------------------------------------------------ #
    # Case 1: .pyx file exists with required header directives
    # ------------------------------------------------------------------ #
    pyx_path = Path("mufasa/_native/inside_polygon.pyx")
    assert pyx_path.is_file(), "inside_polygon.pyx missing"
    pyx_src = pyx_path.read_text()

    # Cython compiler directives that affect performance/correctness
    expected_directives = [
        "language_level=3",
        "boundscheck=False",
        "wraparound=False",
        "cdivision=True",
    ]
    for d in expected_directives:
        assert d in pyx_src, (
            f".pyx missing compiler directive {d!r} — without it, "
            f"Cython generates slower or less-correct code"
        )

    # ------------------------------------------------------------------ #
    # Case 2: .pyx defines the expected function with the right name
    # ------------------------------------------------------------------ #
    assert "def framewise_inside_polygon_roi(" in pyx_src, (
        "kernel function name must match the numba version it "
        "replaces"
    )

    # ------------------------------------------------------------------ #
    # Case 3: ray-casting algorithm structure preserved
    # ------------------------------------------------------------------ #
    # The classic ray-casting test has these structural elements:
    #   - outer loop over frames
    #   - inner loop over polygon edges using j % n_poly indexing
    #   - if/and chained conditions on min/max of p1y, p2y
    #   - inside = not inside flip on intersection
    structural_checks = [
        ("for i in range(", "outer frame loop"),
        ("for j in range(", "inner polygon-edge loop"),
        ("% n_poly", "modular indexing for the wrap-around edge"),
        ("inside = not inside", "the flip on intersection"),
        ("min(p1y, p2y)", "min comparison from ray-cast algorithm"),
        ("max(p1y, p2y)", "max comparison from ray-cast algorithm"),
    ]
    for token, description in structural_checks:
        assert token in pyx_src, (
            f"ray-casting algorithm missing {description}: "
            f"expected {token!r} in .pyx"
        )

    # ------------------------------------------------------------------ #
    # Case 4: __init__.py for the _native package exists
    # ------------------------------------------------------------------ #
    init = Path("mufasa/_native/__init__.py")
    assert init.is_file(), "mufasa/_native/__init__.py missing"

    # ------------------------------------------------------------------ #
    # Case 5: setup.py declares the extension
    # ------------------------------------------------------------------ #
    setup_path = Path("setup.py")
    assert setup_path.is_file(), "setup.py missing"
    setup_src = setup_path.read_text()
    assert "Cython.Build" in setup_src
    assert "cythonize" in setup_src
    assert "mufasa._native.inside_polygon" in setup_src
    assert "inside_polygon.pyx" in setup_src
    # numpy include path must be present — without it the
    # `cimport numpy` line in the .pyx fails to compile
    assert "np.get_include()" in setup_src or "numpy.get_include()" in setup_src, (
        "setup.py must add numpy's include path so cimport numpy "
        "compiles"
    )
    # -ffast-math must NOT be in extra_compile_args (would diverge
    # from numba's floating-point ordering and break byte-equivalence).
    # Comments mentioning -ffast-math are fine; we only care that it
    # isn't in the actual compile flags.
    args_match = re.search(
        r"extra_compile_args\s*=\s*\[([^\]]*)\]",
        setup_src,
    )
    if args_match:
        compile_flags = args_match.group(1)
        assert "-ffast-math" not in compile_flags, (
            "setup.py extra_compile_args must NOT include "
            "-ffast-math — it reorders floating-point ops in "
            "ways that diverge from the numba version, breaking "
            "byte-equivalence"
        )
    # -O3 should be on for reasonable performance
    assert "-O3" in setup_src, (
        "setup.py should enable -O3 for the Cython extension; "
        "without it auto-vectorization and inlining are skipped"
    )

    # ------------------------------------------------------------------ #
    # Case 6: pyproject.toml's [build-system] declares Cython + numpy
    # ------------------------------------------------------------------ #
    pyproject = Path("pyproject.toml").read_text()
    # Find the [build-system] block
    bs_match = re.search(
        r"\[build-system\](.*?)(?=\n\[)",
        pyproject,
        re.DOTALL,
    )
    assert bs_match, "pyproject.toml [build-system] section missing"
    bs = bs_match.group(1)
    assert "Cython" in bs, (
        "pyproject.toml [build-system] must list Cython in "
        "requires — without it, pip install fails with "
        "'No module named Cython.Build' before the build script "
        "can run"
    )
    assert "numpy" in bs.lower(), (
        "pyproject.toml [build-system] must list numpy in requires "
        "— Cython needs numpy headers at build time"
    )

    # ------------------------------------------------------------------ #
    # Case 7: existing numba kernel still in place (POC is additive,
    # not replacing)
    # ------------------------------------------------------------------ #
    mixin = Path("mufasa/mixins/feature_extraction_mixin.py").read_text()
    assert "def framewise_inside_polygon_roi" in mixin, (
        "Original numba framewise_inside_polygon_roi removed — "
        "the POC should be ADDITIVE, not replace the production "
        "kernel until verification is complete"
    )

    # ------------------------------------------------------------------ #
    # Case 8: kernel chain (feature_subset_kernels.py) still calls
    # the numba version (NOT the Cython version) — so production
    # behavior is unchanged by the POC
    # ------------------------------------------------------------------ #
    kernels = Path(
        "mufasa/feature_extractors/feature_subset_kernels.py"
    ).read_text()
    assert "framewise_inside_polygon_roi" in kernels, (
        "feature_subset_kernels should still call the polygon kernel"
    )
    assert "FeatureExtractionMixin" in kernels, (
        "feature_subset_kernels should still call via the numba "
        "FeatureExtractionMixin (not _native) until POC validates"
    )
    assert "_native" not in kernels, (
        "feature_subset_kernels must NOT yet route to the Cython "
        "version — it's a proof of concept, not a production "
        "swap. Wire it in only after the workstation benchmark "
        "shows byte-equivalence and acceptable speed."
    )

    print("smoke_cython_poc_structure: 8/8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
