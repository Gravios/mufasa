"""Structural test for Cython kernel wiring in feature_subset_kernels.

Verifies that production calls in mufasa/feature_extractors/feature_subset_kernels.py
are routed through the _native (Cython) kernels via local aliases,
and that a numba fallback is preserved for environments where
the Cython extensions aren't compiled.

Sandbox-runnable — no Cython compilation needed.

    PYTHONPATH=. python tests/smoke_cython_kernel_wiring.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


# Map of Cython kernel local-alias names → expected production
# callers. Each alias should be:
#   1. defined in the import block (try: ... from mufasa._native...)
#   2. assigned to a numba fallback in the except block
#   3. invoked at least once in the body of the file
EXPECTED_ALIASES = {
    "_kern_angle3pt":          "angle3pt_vectorized",
    "_kern_border_distances":  "border_distances",
    "_kern_euclid":            "framewise_euclidean_distance",
    "_kern_euclid_roi":        "framewise_euclidean_distance_roi",
    "_kern_hull":              "jitted_hull",
    "_kern_inside_circle":     "is_inside_circle",
    "_kern_inside_polygon":    "framewise_inside_polygon_roi",
    "_kern_inside_rectangle":  "framewise_inside_rectangle_roi",
}


def main() -> int:
    src = Path(
        "mufasa/feature_extractors/feature_subset_kernels.py"
    ).read_text()

    # ------------------------------------------------------------------ #
    # Case 1: try/except import block exists with all 8 kernels
    # ------------------------------------------------------------------ #
    tree = ast.parse(src)
    found_try = None
    for node in tree.body:
        if isinstance(node, ast.Try):
            try_src = ast.unparse(node)
            if "mufasa._native" in try_src:
                found_try = node
                break
    assert found_try is not None, (
        "feature_subset_kernels.py should have a try/except block "
        "importing from mufasa._native with a numba fallback"
    )
    try_src = ast.unparse(found_try)
    for alias in EXPECTED_ALIASES:
        assert alias in try_src, (
            f"alias {alias} must appear in the try/except wiring block"
        )

    # ------------------------------------------------------------------ #
    # Case 2: each alias resolves to the matching numba fallback in
    # the except branch. This protects users when _native is
    # unavailable (e.g. they pulled new code without reinstalling).
    # ------------------------------------------------------------------ #
    # Get the except-handler body source
    handler_srcs = []
    for handler in found_try.handlers:
        handler_srcs.append(ast.unparse(handler))
    fallback_src = "\n".join(handler_srcs)
    for alias, kernel_name in EXPECTED_ALIASES.items():
        # The fallback should contain `_kern_X = ...kernel_name...`
        # E.g. `_kern_angle3pt = FeatureExtractionMixin.angle3pt_vectorized`
        assert (
            f"{alias} = " in fallback_src
            and kernel_name in fallback_src
        ), (
            f"Fallback path missing assignment for {alias} → "
            f"a numba kernel with name {kernel_name!r}"
        )

    # ------------------------------------------------------------------ #
    # Case 3: each alias is INVOKED at least once in the production
    # call sites (not just imported — actually used)
    # ------------------------------------------------------------------ #
    # Strip the try/except block by line range. ast.unparse produces
    # a normalized form that doesn't byte-match the source, so we
    # use line numbers to slice the original source instead.
    src_lines = src.splitlines(keepends=True)
    try_start = found_try.lineno - 1
    try_end = found_try.end_lineno  # exclusive
    src_no_try = "".join(
        src_lines[:try_start] + src_lines[try_end:]
    )
    for alias in EXPECTED_ALIASES:
        # Must appear as a call: alias(
        # Allow for line breaks: alias\n( or alias (
        pattern = re.compile(
            rf"{re.escape(alias)}\s*\("
        )
        assert pattern.search(src_no_try), (
            f"{alias} is wired in imports but never called from "
            f"production code paths — wiring is incomplete"
        )

    # ------------------------------------------------------------------ #
    # Case 4: _NATIVE_AVAILABLE flag exposed for introspection
    # ------------------------------------------------------------------ #
    assert "_NATIVE_AVAILABLE" in src, (
        "feature_subset_kernels.py should expose _NATIVE_AVAILABLE "
        "so callers/tests can check whether Cython kernels are in use"
    )
    # Set to True in the try-success path
    assert "_NATIVE_AVAILABLE = True" in src, (
        "_NATIVE_AVAILABLE = True should be set when Cython imports "
        "succeed"
    )
    # Set to False in fallback
    assert "_NATIVE_AVAILABLE = False" in src, (
        "_NATIVE_AVAILABLE = False should be set in the numba "
        "fallback path"
    )

    # ------------------------------------------------------------------ #
    # Case 5: when _native is unavailable, a warning is emitted so
    # the user knows they need to reinstall
    # ------------------------------------------------------------------ #
    assert "warnings.warn" in fallback_src, (
        "Fallback path should emit warnings.warn so the user knows "
        "Cython kernels aren't being used"
    )
    assert "pip install" in fallback_src, (
        "Warning message should mention `pip install` as the fix"
    )
    # RuntimeWarning is the appropriate category — it's
    # actionable, not a deprecation
    assert "RuntimeWarning" in fallback_src

    # ------------------------------------------------------------------ #
    # Case 6: bodypart_distance and create_shifted_df remain unwired.
    # bodypart_distance has input validation we don't replicate;
    # create_shifted_df is pandas, not a numba kernel.
    # ------------------------------------------------------------------ #
    assert "FeatureExtractionMixin.bodypart_distance(" in src_no_try, (
        "bodypart_distance call should be retained — it has "
        "validation we don't replicate at the call site"
    )
    assert "create_shifted_df" in src_no_try, (
        "create_shifted_df call should be retained — pandas operation, "
        "not a numba kernel"
    )

    # ------------------------------------------------------------------ #
    # Case 7: no orphan calls. Every previously-numba kernel call in
    # the body should now use the alias. We check by ensuring
    # specific old patterns no longer appear in the production body.
    # ------------------------------------------------------------------ #
    forbidden_patterns_in_body = [
        # These were the old direct numba calls; should be gone
        # from the production body (they remain only in the fallback
        # block, which we excluded via src_no_try).
        "FeatureExtractionMixin.framewise_inside_polygon_roi",
        "FeatureExtractionMixin.framewise_inside_rectangle_roi",
        "FeatureExtractionMixin.is_inside_circle",
        "FeatureExtractionMixin.framewise_euclidean_distance(",
        "FeatureExtractionMixin.framewise_euclidean_distance_roi",
        "FeatureExtractionMixin.angle3pt_vectorized",
        "FeatureExtractionSupplemental().border_distances",
        "FeatureExtractionSupplemental.border_distances",
    ]
    for pat in forbidden_patterns_in_body:
        assert pat not in src_no_try, (
            f"Old direct numba call {pat!r} still appears in the "
            f"production body — wiring is incomplete (should use "
            f"_kern_* alias instead)"
        )

    # ------------------------------------------------------------------ #
    # Case 8: jitted_hull alias chain works — `jitted_hull` import
    # is renamed to `_numba_jitted_hull` so it doesn't clash with
    # `_kern_hull`
    # ------------------------------------------------------------------ #
    assert "jitted_hull as _numba_jitted_hull" in src, (
        "Original numba jitted_hull import should be renamed to "
        "_numba_jitted_hull to avoid colliding with the Cython "
        "kernel local alias _kern_hull"
    )

    print("smoke_cython_kernel_wiring: 8/8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
