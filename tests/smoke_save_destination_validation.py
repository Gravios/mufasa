"""Tests for strict save-destination validation in feature_subsets.py.

Verifies that __init__ raises an error when the caller would
otherwise silently discard output.

Background: pre-fix behavior was that running with save_dir=None
and both append flags False would compute features into temp_dir
then delete temp_dir at the end. This silently discarded hours of
compute. Fixed by failing fast in __init__.

Tests use AST inspection — we don't import the actual class
(ConfigReader requires h5py, which isn't in the sandbox), but we
parse the file and verify the relevant validation block exists
and raises the right error type.

    PYTHONPATH=. python tests/smoke_save_destination_validation.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    src = Path(
        "mufasa/feature_extractors/feature_subsets.py"
    ).read_text()
    tree = ast.parse(src)

    # Find FeatureSubsetsCalculator.__init__
    cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FeatureSubsetsCalculator":
            cls = node
            break
    assert cls is not None, "FeatureSubsetsCalculator class missing"

    init = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init = node
            break
    assert init is not None, "__init__ missing"

    init_src = ast.unparse(init)

    # ------------------------------------------------------------------ #
    # Case 1: __init__ contains the strict validation block
    # ------------------------------------------------------------------ #
    # The block should test for: save_dir is None AND not append_to_features
    # AND not append_to_targets, then raise.
    assert "self.save_dir is None" in init_src
    assert "not self.append_to_features_extracted" in init_src
    assert "not self.append_to_targets_inserted" in init_src

    # ------------------------------------------------------------------ #
    # Case 2: the raise is an InvalidInputError (not generic Exception
    # or RuntimeError — InvalidInputError carries source info and is
    # the project's convention for parameter-validation errors)
    # ------------------------------------------------------------------ #
    raise_nodes = [
        n for n in ast.walk(init) if isinstance(n, ast.Raise)
    ]
    found_invalid_input = False
    for raise_node in raise_nodes:
        if raise_node.exc is None:
            continue
        if isinstance(raise_node.exc, ast.Call):
            func = raise_node.exc.func
            name = (
                func.id if isinstance(func, ast.Name)
                else getattr(func, "attr", None)
            )
            if name == "InvalidInputError":
                # Check the message mentions all three options
                # so users know what to do
                src_text = ast.unparse(raise_node)
                if (
                    "save_dir" in src_text
                    and "append_to_features_extracted" in src_text
                    and "append_to_targets_inserted" in src_text
                    and "discarded" in src_text.lower()
                ):
                    found_invalid_input = True
    assert found_invalid_input, (
        "__init__ must raise InvalidInputError mentioning all three "
        "save destinations and the word 'discarded' in the message"
    )

    # ------------------------------------------------------------------ #
    # Case 3: the validation runs BEFORE the data_dir/data_paths setup
    # — fail fast, before any heavy work or filesystem inspection.
    # We check ordering by line numbers within the source.
    # ------------------------------------------------------------------ #
    lines = init_src.splitlines()
    save_dir_validation_line = None
    data_dir_setup_line = None
    for i, line in enumerate(lines):
        if save_dir_validation_line is None and "self.save_dir is None" in line:
            save_dir_validation_line = i
        if data_dir_setup_line is None and "self.data_dir = " in line:
            data_dir_setup_line = i
    assert save_dir_validation_line is not None
    assert data_dir_setup_line is not None
    assert save_dir_validation_line < data_dir_setup_line, (
        f"Save destination validation should run BEFORE data_dir "
        f"setup (lines: validation={save_dir_validation_line}, "
        f"data_dir={data_dir_setup_line})"
    )

    # ------------------------------------------------------------------ #
    # Case 4: run() cleanup is conditional — temp_dir is only deleted
    # if at least one save destination succeeded
    # ------------------------------------------------------------------ #
    run_method = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_method = node
            break
    assert run_method is not None
    run_src = ast.unparse(run_method)
    assert "output_persisted" in run_src, (
        "run() should track output_persisted to know when it's safe "
        "to delete temp_dir"
    )
    assert "save_errors" in run_src, (
        "run() should collect save_errors and surface them to the user"
    )
    # The remove_a_folder call should be inside an `if output_persisted:`
    # block, not unconditional like before
    assert "if output_persisted" in run_src, (
        "remove_a_folder must be conditional on output_persisted"
    )

    # ------------------------------------------------------------------ #
    # Case 5: run() prints a recovery hint when save fails
    # ------------------------------------------------------------------ #
    assert "Intermediate output preserved in" in run_src, (
        "run() should tell the user where temp_dir is when save fails"
    )
    assert "Recover with" in run_src, (
        "run() should give a recovery hint (cp command)"
    )

    # ------------------------------------------------------------------ #
    # Case 6: _setup_run announces destinations upfront
    # ------------------------------------------------------------------ #
    setup = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_setup_run":
            setup = node
            break
    assert setup is not None
    setup_src = ast.unparse(setup)
    assert "Feature subsets will be written to" in setup_src, (
        "_setup_run should announce destinations before any compute"
    )

    # ------------------------------------------------------------------ #
    # Case 7: Qt form validates save destination in collect_args.
    # Post-radio-button refactor: collect_args branches on radio
    # state and raises if save_dir radio is selected without a path.
    # The old "save_dir_value is None and not append_features and
    # not append_targets" check was replaced with explicit radio
    # branching since the radios make multi-destination impossible.
    # ------------------------------------------------------------------ #
    qt_src = Path("mufasa/ui_qt/forms/features.py").read_text()
    qt_tree = ast.parse(qt_src)
    collect_args = None
    for node in ast.walk(qt_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "collect_args":
            collect_args = node
            break
    assert collect_args is not None, "Qt form's collect_args missing"
    collect_src = ast.unparse(collect_args)
    # Radio-button form: collect_args raises if save_dir radio is
    # picked but no path is given. The error message references
    # the radio-button label.
    assert "Save destination is required" in collect_src, (
        "Qt form's collect_args should raise a clear error when "
        "save_dir radio is selected but no path is provided"
    )
    # The form now also validates via radio-state branching, not
    # by checking all three destination fields.
    assert "self.dest_save_dir.isChecked()" in collect_src, (
        "Qt form's collect_args should branch on the radio state"
    )

    # ------------------------------------------------------------------ #
    # Case 8: Qt form's placeholder text is no longer the misleading
    # "blank = project log dir" string. Look for the placeholder= kwarg
    # specifically rather than substring-matching the whole file
    # (a comment can legitimately reference the old text for context).
    # The radio-button refactor changed the placeholder; the field
    # is now adjacent to the "Save standalone files" radio so the
    # placeholder is shorter ("…directory for the standalone
    # CSVs/parquets"). What matters is that the misleading
    # "blank = project log dir" string is gone.
    # ------------------------------------------------------------------ #
    placeholders = []
    for node in ast.walk(qt_tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "placeholder" and isinstance(kw.value, ast.Constant):
                    placeholders.append(kw.value.value)
    # No placeholder anywhere should contain the misleading old text.
    for p in placeholders:
        if p:
            assert "blank = project log dir" not in p, (
                f"Placeholder still misleading: {p!r}"
            )
    # AT LEAST ONE placeholder should be related to the save_dir
    # field (mentioning directory). This is a loose check —
    # the goal is to confirm the field still has SOME placeholder,
    # not pin down the exact wording.
    save_dir_related = [
        p for p in placeholders
        if p and ("directory" in p.lower() or "dir" in p.lower())
    ]
    assert save_dir_related, (
        f"No save-dir-related placeholder found among: {placeholders}"
    )

    print("smoke_save_destination_validation: 8/8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
