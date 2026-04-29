"""Tests for column-conflict detection + overwrite confirmation
in feature subset extraction.

Two layers tested:
1. Backend: FeatureSubsetsCalculator has preflight_check() and an
   overwrite_existing param. Verified via AST inspection (the
   actual class can't be instantiated in the sandbox without
   h5py + numba).
2. Qt form: overrides on_run to call preflight + prompt. Verified
   via AST inspection of features.py.

    PYTHONPATH=. python tests/smoke_feature_overwrite_confirm.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    backend_src = Path(
        "mufasa/feature_extractors/feature_subsets.py"
    ).read_text()
    backend_tree = ast.parse(backend_src)

    # ------------------------------------------------------------------ #
    # Case 1: __init__ accepts overwrite_existing param defaulting False
    # ------------------------------------------------------------------ #
    cls = next(
        n for n in backend_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FeatureSubsetsCalculator"
    )
    init = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    arg_names = [a.arg for a in init.args.args]
    assert "overwrite_existing" in arg_names, (
        "__init__ must accept overwrite_existing param"
    )
    # Default should be False — check the function signature's
    # defaults rather than fragile string matching
    overwrite_default = None
    for arg, default in zip(
        reversed(init.args.args),
        reversed(init.args.defaults),
    ):
        if arg.arg == "overwrite_existing":
            if isinstance(default, ast.Constant):
                overwrite_default = default.value
            break
    assert overwrite_default is False, (
        f"overwrite_existing must default to False (safer default); "
        f"got default={overwrite_default!r}"
    )
    init_src = ast.unparse(init)
    # Stored on self
    assert "self.overwrite_existing = overwrite_existing" in init_src, (
        "overwrite_existing must be stored on the instance"
    )

    # ------------------------------------------------------------------ #
    # Case 2: preflight_check method exists and returns Dict
    # ------------------------------------------------------------------ #
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "preflight_check" in methods, (
        "FeatureSubsetsCalculator should have a preflight_check method"
    )
    pf = methods["preflight_check"]
    pf_src = ast.unparse(pf)
    # Returns dict mapping filename → conflicting columns
    assert "Dict" in pf_src or "dict" in pf_src.lower(), (
        "preflight_check should return a dict"
    )
    # Mentions process_one_video — uses the orchestration to discover
    # column names rather than predicting them statically
    assert "process_one_video" in pf_src, (
        "preflight_check should run process_one_video on a probe "
        "video to discover the actual column names this run produces"
    )
    # Uses scratch dir, not self.temp_dir, to avoid polluting it
    assert "scratch" in pf_src.lower() or "preflight_scratch" in pf_src, (
        "preflight should use a separate scratch directory"
    )
    # Cleans up scratch after use
    assert "remove_a_folder" in pf_src or "rmtree" in pf_src, (
        "preflight should clean up its scratch dir"
    )
    # Returns empty dict when no append destination is set
    assert (
        "not self.append_to_features_extracted" in pf_src
        or "no append" in pf_src.lower()
    ), (
        "preflight should short-circuit return {} when neither "
        "append flag is set"
    )

    # ------------------------------------------------------------------ #
    # Case 3: run() invokes preflight_check before compute and raises
    # if conflicts found and overwrite_existing is False
    # ------------------------------------------------------------------ #
    run_method = methods["run"]
    run_src = ast.unparse(run_method)
    assert "preflight_check" in run_src, (
        "run() should call preflight_check before kicking off compute"
    )
    assert "overwrite_existing" in run_src, (
        "run() should branch on overwrite_existing"
    )
    assert "DuplicationError" in run_src, (
        "run() should raise DuplicationError on detected conflicts"
    )
    # Error message should mention how to proceed
    for hint in [
        "overwrite_existing", "different feature families",
        "save_dir",
    ]:
        assert hint in run_src, (
            f"DuplicationError message should mention {hint!r} as a "
            f"way to proceed"
        )

    # ------------------------------------------------------------------ #
    # Case 4: __check_files honors overwrite_existing
    # ------------------------------------------------------------------ #
    # __check_files is name-mangled to _FeatureSubsetsCalculator__check_files
    # internally, but in the AST it appears as __check_files literally.
    check_method = methods.get("_FeatureSubsetsCalculator__check_files") \
        or methods.get("__check_files")
    if check_method is None:
        # AST might keep the dunder-prefixed name. Find it by scanning.
        for n in cls.body:
            if isinstance(n, ast.FunctionDef) and "check_files" in n.name:
                check_method = n
                break
    assert check_method is not None, "__check_files method missing"
    check_src = ast.unparse(check_method)
    assert "overwrite_existing" in check_src, (
        "__check_files should respect self.overwrite_existing — "
        "raising DuplicationError only when overwrite isn't enabled"
    )

    # ------------------------------------------------------------------ #
    # Case 5: append helpers drop conflicting columns when
    # overwrite_existing is set
    # ------------------------------------------------------------------ #
    append_helpers = [
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and "append_to_data_in_dir" in n.name
    ]
    assert len(append_helpers) >= 1
    append_src = ast.unparse(append_helpers[0])
    assert "overwrite_existing" in append_src, (
        "__append_to_data_in_dir should branch on overwrite_existing"
    )
    assert "drop(columns=" in append_src, (
        "When overwrite_existing is set, the append helper must "
        "drop conflicting columns from the existing file before concat"
    )

    # ------------------------------------------------------------------ #
    # Case 6: Qt form overrides on_run and runs preflight before
    # dispatching the worker
    # ------------------------------------------------------------------ #
    form_src = Path("mufasa/ui_qt/forms/features.py").read_text()
    form_tree = ast.parse(form_src)
    form_cls = next(
        n for n in ast.walk(form_tree)
        if isinstance(n, ast.ClassDef) and n.name == "FeatureSubsetExtractorForm"
    )
    form_methods = {
        n.name: n for n in form_cls.body
        if isinstance(n, ast.FunctionDef)
    }
    assert "on_run" in form_methods, (
        "Form should override on_run to insert preflight + prompt"
    )
    on_run_src = ast.unparse(form_methods["on_run"])
    assert "preflight" in on_run_src.lower(), (
        "on_run should call preflight"
    )
    assert "QMessageBox" in on_run_src, (
        "on_run should use QMessageBox for the confirmation prompt"
    )
    assert "Yes" in on_run_src and "No" in on_run_src, (
        "Confirmation should offer Yes/No"
    )
    assert "WaitCursor" in on_run_src, (
        "Preflight blocks the UI; should set WaitCursor so the user "
        "knows something is happening"
    )
    # Default should be No (safer)
    # Look for QMessageBox.No appearing twice (once as button, once
    # as default) or for the default-button arg
    no_count = on_run_src.count("QMessageBox.No")
    assert no_count >= 2, (
        f"on_run should set QMessageBox.No as the default button "
        f"(safer choice when user just presses Enter)"
    )

    # ------------------------------------------------------------------ #
    # Case 7: target() in form passes overwrite_existing through to
    # the calculator
    # ------------------------------------------------------------------ #
    target_method = form_methods["target"]
    target_src = ast.unparse(target_method)
    assert "overwrite_existing" in target_src, (
        "form.target() should accept overwrite_existing and pass "
        "it to FeatureSubsetsCalculator"
    )

    # ------------------------------------------------------------------ #
    # Case 8: collect_args sets a default overwrite_existing=False
    # so on_run can flip it after user confirmation
    # ------------------------------------------------------------------ #
    collect_method = form_methods["collect_args"]
    collect_src = ast.unparse(collect_method)
    assert "overwrite_existing" in collect_src, (
        "collect_args should include overwrite_existing in returned "
        "dict so on_run has a key to update after the prompt"
    )

    # ------------------------------------------------------------------ #
    # Case 9: preflight_check ALSO catches save_dir filename
    # collisions (the common case where user picks an existing
    # directory that already has output from a prior run).
    # Pre-fix the preflight only ran for append flags.
    # ------------------------------------------------------------------ #
    pf_src_full = ast.unparse(pf)
    assert "save_dir" in pf_src_full, (
        "preflight_check must also check save_dir for filename "
        "collisions, not just append-destination column collisions"
    )
    assert "file exists" in pf_src_full, (
        "preflight_check should mark save_dir collisions with "
        "'file exists' reason string so the caller can distinguish "
        "them from column collisions"
    )

    # ------------------------------------------------------------------ #
    # Case 10: Form on_run runs preflight when ANY destination is
    # set, not just append. Pre-fix it skipped preflight for save_dir-
    # only mode, which let users silently overwrite previous output.
    # ------------------------------------------------------------------ #
    on_run_full_src = ast.unparse(form_methods["on_run"])
    # ast.unparse may normalize quotes; check both forms
    assert (
        'kwargs["save_dir"] is not None' in on_run_full_src
        or "kwargs['save_dir'] is not None" in on_run_full_src
    ), (
        "on_run must include save_dir in needs_preflight check "
        "(not just the append flags)"
    )

    print("smoke_feature_overwrite_confirm: 10/10 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
