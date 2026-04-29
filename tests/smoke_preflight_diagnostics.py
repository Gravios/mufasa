"""Structural test for preflight diagnostic console output.

The user reported that "Append to existing features_extracted CSVs"
mode didn't trigger the overwrite prompt despite the patch being
applied. Three possible failure modes:

1. Form's on_run override not running (stale __pycache__ etc.)
2. Preflight ran but probe failed silently (no output file)
3. Preflight ran and found no conflicts (column names didn't match)

The diagnostic prints distinguish these by emitting console lines
at known points. This test verifies the prints exist where they
should, so on the next run we can identify which scenario applies.

    PYTHONPATH=. python tests/smoke_preflight_diagnostics.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    backend_src = Path(
        "mufasa/feature_extractors/feature_subsets.py"
    ).read_text()
    form_src = Path("mufasa/ui_qt/forms/features.py").read_text()

    # ------------------------------------------------------------------ #
    # Case 1: Form's on_run prints a "[on_run] dispatching" line
    # before the preflight check. This is the ONLY way we can tell
    # that the override is actually being invoked vs. inheriting
    # from the parent class (which would have no preflight at all).
    # ------------------------------------------------------------------ #
    assert "[on_run] FeatureSubsetExtractorForm dispatching" in form_src, (
        "Form's on_run must print a dispatch diagnostic line so "
        "we can tell whether the override is taking effect"
    )

    # ------------------------------------------------------------------ #
    # Case 2: preflight_check prints a "[preflight] starting check"
    # line when invoked. Distinguishes "preflight didn't run" from
    # "preflight ran and silently returned nothing"
    # ------------------------------------------------------------------ #
    assert "[preflight] starting check" in backend_src, (
        "preflight_check must print a 'starting check' line on "
        "every invocation"
    )

    # ------------------------------------------------------------------ #
    # Case 3: preflight prints save_dir check result (number of
    # collisions) so we can verify the cheap check ran
    # ------------------------------------------------------------------ #
    assert "[preflight] save_dir check:" in backend_src, (
        "preflight_check must print the save_dir collision count"
    )

    # ------------------------------------------------------------------ #
    # Case 4: preflight prints number of columns produced by the
    # probe — verifies the probe ran successfully
    # ------------------------------------------------------------------ #
    assert "[preflight] probe produced" in backend_src, (
        "preflight_check must print the column count produced by "
        "the probe of the first video"
    )

    # ------------------------------------------------------------------ #
    # Case 5: preflight prints a WARNING when probe produced no
    # output file (silently failed compute), so the silent-skip
    # case is no longer silent
    # ------------------------------------------------------------------ #
    assert "[preflight] WARNING: probe of" in backend_src, (
        "preflight_check must WARN when the probe produces no "
        "output file (was silently skipping the column check)"
    )

    # ------------------------------------------------------------------ #
    # Case 6: preflight prints per-target file-check summary
    # (how many existing files matched, how many had conflicts)
    # ------------------------------------------------------------------ #
    assert "[preflight] {label}:" in backend_src or \
           "checked {files_checked}" in backend_src, (
        "preflight_check must print a per-target summary including "
        "files checked, files missing, and conflicts found"
    )

    # ------------------------------------------------------------------ #
    # Case 7: comment near the on_run diagnostic mentions
    # __pycache__ and pip install (the most likely cause of "no
    # print appears"). The comment block precedes the print line
    # since that's where readers will look when investigating.
    # ------------------------------------------------------------------ #
    on_run_diag_idx = form_src.index("[on_run]")
    # Look at the 800 chars BEFORE the marker (where the explanatory
    # comment lives) plus 200 after for the print itself
    diag_window = form_src[max(0, on_run_diag_idx - 800):on_run_diag_idx + 200]
    assert "__pycache__" in diag_window, (
        "The on_run diagnostic comment should mention __pycache__ "
        "as a likely cause of 'no print appears'"
    )
    assert "pip install -e" in diag_window, (
        "The on_run diagnostic comment should suggest "
        "`pip install -e .` to fix stale state"
    )

    # ------------------------------------------------------------------ #
    # Case 8: structural — preflight raise_warning=False on the
    # find_files call so it doesn't spam warnings when scratch is
    # legitimately empty (probe failure path)
    # ------------------------------------------------------------------ #
    assert "raise_warning=False" in backend_src, (
        "find_files_of_filetypes_in_directory should be called with "
        "raise_warning=False in the probe path so the WARNING "
        "message isn't drowned out by spurious framework warnings"
    )

    print("smoke_preflight_diagnostics: 8/8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
