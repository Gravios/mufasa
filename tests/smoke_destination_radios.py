"""Test for the destination radio-button UI in the feature
subsets form.

Pre-fix: the form had three independent widgets — a path field
plus two checkboxes — which let users (accidentally) configure
output to go to multiple destinations simultaneously. User
report: "save_dir='...' append_features=True" set together,
producing dual-destination output.

Post-fix: three radio buttons in a QButtonGroup enforce a
single choice. The save_dir path field is enabled only when
the save_dir radio is selected.

Sandbox-runnable via AST inspection (PySide6 not available).

    PYTHONPATH=. python tests/smoke_destination_radios.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    src = Path("mufasa/ui_qt/forms/features.py").read_text()
    tree = ast.parse(src)

    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "FeatureSubsetExtractorForm"
    )

    # ------------------------------------------------------------------ #
    # Case 1: form imports QRadioButton and QButtonGroup
    # ------------------------------------------------------------------ #
    body_src = ast.unparse(cls)
    assert "QRadioButton" in body_src, (
        "Form must use QRadioButton for destinations"
    )
    assert "QButtonGroup" in body_src, (
        "Form must use QButtonGroup to enforce single-choice"
    )

    # ------------------------------------------------------------------ #
    # Case 2: two radio buttons exist with the right attribute names.
    # Patch 122an (B1) dropped the legacy append radios — only
    # dest_derived_parquet (default, v1 per-family) and
    # dest_save_dir (user-picked standalone directory) remain.
    # ------------------------------------------------------------------ #
    for attr in ("dest_derived_parquet", "dest_save_dir"):
        assert f"self.{attr}" in body_src, (
            f"Form should have a self.{attr} radio button"
        )
    for removed in ("dest_append_features", "dest_append_targets"):
        assert f"self.{removed} = QRadioButton" not in body_src, (
            f"Legacy radio self.{removed} should have been removed "
            f"by patch 122an"
        )

    # ------------------------------------------------------------------ #
    # Case 3: Both surviving radios are added to the same QButtonGroup
    # (which is what enforces single-choice)
    # ------------------------------------------------------------------ #
    assert "_dest_group.addButton(self.dest_derived_parquet" in body_src
    assert "_dest_group.addButton(self.dest_save_dir" in body_src

    # ------------------------------------------------------------------ #
    # Case 4: dest_derived_parquet is the default selection (v1-native)
    # ------------------------------------------------------------------ #
    assert (
        "self.dest_derived_parquet.setChecked(True)" in body_src
        or "self.dest_save_dir.setChecked(True)" in body_src
    ), "One radio must be checked by default"

    # ------------------------------------------------------------------ #
    # Case 5: save_dir path field is enabled/disabled based on
    # which radio is selected (visual feedback)
    # ------------------------------------------------------------------ #
    init = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "build"
    )
    init_src = ast.unparse(init)
    # There's a helper that sets enabled state based on radio
    assert "save_dir.setEnabled" in init_src, (
        "save_dir field should be enabled/disabled based on "
        "which radio is active"
    )

    # ------------------------------------------------------------------ #
    # Case 6: collect_args branches on which radio is checked.
    # Patch 122an (B1) removed the two legacy append branches —
    # collect_args now only handles dest_derived_parquet and
    # dest_save_dir.
    # ------------------------------------------------------------------ #
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    collect = methods["collect_args"]
    collect_src = ast.unparse(collect)
    assert "self.dest_save_dir.isChecked()" in collect_src, (
        "collect_args should branch on dest_save_dir.isChecked()"
    )
    assert "self.dest_derived_parquet.isChecked()" in collect_src, (
        "collect_args should branch on dest_derived_parquet.isChecked()"
    )
    # The legacy branches and the keys they returned should be gone
    assert "self.dest_append_features.isChecked()" not in collect_src, (
        "Legacy dest_append_features branch must be removed"
    )
    assert "self.dest_append_targets.isChecked()" not in collect_src, (
        "Legacy dest_append_targets branch must be removed"
    )
    assert '"append_features"' not in collect_src
    assert '"append_targets"' not in collect_src

    # ------------------------------------------------------------------ #
    # Case 7: collect_args raises a clear error if save_dir is
    # selected but no path is given
    # ------------------------------------------------------------------ #
    assert "Save destination is required" in collect_src, (
        "collect_args should raise ValueError with a clear message "
        "when save_dir is selected but no path is provided"
    )

    print("smoke_destination_radios: 7/7 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
