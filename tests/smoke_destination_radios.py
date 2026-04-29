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
    # Case 2: three radio buttons exist with the right attribute names
    # ------------------------------------------------------------------ #
    for attr in ("dest_save_dir", "dest_append_features", "dest_append_targets"):
        assert f"self.{attr}" in body_src, (
            f"Form should have a self.{attr} radio button"
        )

    # ------------------------------------------------------------------ #
    # Case 3: All three radios are added to the same QButtonGroup
    # (which is what enforces single-choice)
    # ------------------------------------------------------------------ #
    assert "_dest_group.addButton(self.dest_save_dir" in body_src
    assert "_dest_group.addButton(self.dest_append_features" in body_src
    assert "_dest_group.addButton(self.dest_append_targets" in body_src

    # ------------------------------------------------------------------ #
    # Case 4: A default selection is set so the group is never
    # in "no selection" state
    # ------------------------------------------------------------------ #
    assert (
        "self.dest_save_dir.setChecked(True)" in body_src
        or "self.dest_append_features.setChecked(True)" in body_src
        or "self.dest_append_targets.setChecked(True)" in body_src
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
    # Case 6: collect_args derives kwargs from radio state, not
    # from the old checkbox state. Specifically: it should NOT
    # produce a kwargs dict where save_dir is set AND append_features
    # is True simultaneously (the old bug).
    # ------------------------------------------------------------------ #
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    collect = methods["collect_args"]
    collect_src = ast.unparse(collect)
    # New code branches on which radio is checked
    assert "self.dest_save_dir.isChecked()" in collect_src, (
        "collect_args should branch on dest_save_dir.isChecked()"
    )
    assert "self.dest_append_features.isChecked()" in collect_src
    assert "self.dest_append_targets.isChecked()" in collect_src
    # In the save_dir branch, append flags should be False
    # In the append branches, save_dir should be None
    # We verify by structure rather than literal string: the
    # save_dir branch should have lines setting append_features=False
    save_dir_branch_idx = collect_src.index("self.dest_save_dir.isChecked()")
    save_dir_branch = collect_src[save_dir_branch_idx:save_dir_branch_idx + 600]
    assert "append_features = False" in save_dir_branch, (
        "save_dir branch must set append_features=False (no dual "
        "destination)"
    )
    assert "append_targets = False" in save_dir_branch, (
        "save_dir branch must set append_targets=False"
    )

    # ------------------------------------------------------------------ #
    # Case 7: collect_args raises a clear error if save_dir is
    # selected but no path is given (the form's most user-visible
    # validation)
    # ------------------------------------------------------------------ #
    assert "Save destination is required" in collect_src, (
        "collect_args should raise ValueError with a clear message "
        "when save_dir is selected but no path is provided"
    )

    # ------------------------------------------------------------------ #
    # Case 8: backward compat — append_features and append_targets
    # attributes still exist (referenced by other tests / code)
    # ------------------------------------------------------------------ #
    assert "self.append_features = self.dest_append_features" in init_src, (
        "append_features attribute should still exist as alias for "
        "the radio (other code references it)"
    )
    assert "self.append_targets = self.dest_append_targets" in init_src

    print("smoke_destination_radios: 8/8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
