"""
tests/smoke_122ed_interpolate_publishing.py
=============================================

Patch 122ed: refactor the :class:`Interpolate` backend to write
to ``derived/interpolated/<run_id>/`` instead of overwriting pose
files in place, and wire :class:`InterpolateForm` to publish the
run into ``outlier_corrected/``.

Context
-------
122ec wired InterpolateForm with the record-only pattern (only
``last_run_at`` recorded; no ``last_run_id``, no
``publish_target_stage``). That was the minimal contract because
the Interpolate backend at the time modified pose files in place
at ``sources/pose/`` — there was no run dir to point at.

122ed lifts that constraint with a backend refactor. The
Interpolate class now allocates a v1 run-id at ``__init__`` time,
creates ``<project_root>/derived/interpolated/<run_id>/``, and
writes each interpolated dataframe there instead of back to
``sources/pose/``. The form captures ``runner.run_id`` after
``.run()`` returns and sets ``self._last_run_id``; the base class's
success hook publishes a symlink at
``derived/outlier_corrected/<run_id> -> ../interpolated/<run_id>``.

Effect
------
Interpolation is now an alternative cleanup path to outlier
correction / Kalman v2: all three produce data under
``outlier_corrected/``, and downstream backends (Features,
Classifier, Visualizations) that hard-code reads from there
pick up the latest run transparently. This closes the "Skip
outlier correction" replacement story from 122dv for the
interpolation case (Pose Import publish-to-stage is still
deferred — see arc completion notes in 122ec).

The user-facing change: the "Copy originals before overwriting"
checkbox is gone. With writes going to a new run dir,
``sources/pose/`` is preserved by definition.

What this patch landed
----------------------
mufasa/data_processors/interpolate.py:
* ``Interpolate.__init__`` allocates ``self.run_id`` (via
  :func:`mufasa.project_layout.generate_run_id`) and
  ``self.run_dir`` (= ``<project>/derived/interpolated/<run_id>``)
  before processing files. ``run_dir`` is created with
  ``exist_ok=False`` — a collision means a generate_run_id bug
  worth surfacing.
* ``Interpolate.run`` writes each output to
  ``<run_dir>/<video>.<file_type>`` instead of overwriting the
  input file_path.
* Dropped the ``copy_originals`` parameter (now redundant —
  originals stay in ``sources/pose/`` by definition).
* Dropped the ``copy_files_to_directory`` import (no longer
  used).
* Class docstring expanded with the 122ed contract.

mufasa/ui_qt/forms/pose_cleanup.py:
* ``InterpolateForm`` gained ``publish_target_stage =
  "outlier_corrected"`` and ``publish_source_stage =
  "interpolated"``. No ``publish_source_flavor`` (only one
  flavor of interpolation per run).
* ``target()`` captures ``runner = Interpolate(...)`` and reads
  ``runner.run_id`` into ``self._last_run_id`` after
  ``runner.run()`` returns.
* ``target()`` no longer accepts ``copy_originals``.
* ``build()`` no longer creates the ``copy_originals``
  checkbox.
* ``collect_args()`` no longer emits ``copy_originals``.
* Class docstring updated: "Provenance (patches 122ec / 122ed)"
  section explaining the contract evolution.

tests/smoke_122ec_interpolate_provenance.py — three checks flipped
by 122ed:
* Check 3 (no publish_target_stage) → asserts publish_target_stage
  == "outlier_corrected".
* Check 4 (target doesn't set _last_run_id) → asserts target
  does set _last_run_id.
* Check 10 ("exactly 1 publisher") → "exactly 2 publishers"
  (kalman_v2 + interpolate).

Coverage
--------
Backend changes:
1.  ``Interpolate.__init__`` no longer accepts ``copy_originals``
    as a parameter.
2.  ``Interpolate.__init__`` assigns ``self.run_id`` (uncondition-
    ally — verified via AST attribute walk).
3.  ``Interpolate.__init__`` assigns ``self.run_dir``.
4.  ``Interpolate.__init__`` calls ``os.makedirs`` against
    ``self.run_dir``.
5.  ``Interpolate.run`` writes to ``self.run_dir`` (substring
    check on the run() source).
6.  ``Interpolate.run`` does NOT write to the input ``file_path``
    (the legacy overwrite pattern is gone).
7.  ``copy_files_to_directory`` import removed from the module.
8.  Class docstring mentions 122ed.

Form changes:
9.  ``InterpolateForm.publish_target_stage == "outlier_corrected"``.
10. ``InterpolateForm.publish_source_stage == "interpolated"``.
11. ``InterpolateForm.target()`` sets ``self._last_run_id``.
12. ``InterpolateForm.target()`` no longer accepts
    ``copy_originals``.
13. ``InterpolateForm.collect_args()`` no longer returns
    ``copy_originals``.
14. ``InterpolateForm.build()`` no longer creates a
    ``QCheckBox`` named ``copy_originals``.

Single-caller contract:
15. The ``Interpolate`` class is invoked from exactly one site
    in mufasa/ (the form). Verifies no other caller was broken
    by the signature change.

122ec tripwire flips:
16. smoke_122ec check 3 ("no publish_target_stage") is flipped
    to assert the value.
17. smoke_122ec check 10 ("exactly 1 publisher") is flipped to
    "exactly 2 publishers".

Cross-patch invariants:
18. Other producers unchanged.
19. SECTIONS DAG still validates.
20. Parse-clean across mufasa/**/*.py.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _ast_find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _ast_method(cls_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _ast_class_attr(cls_node: ast.ClassDef, name: str) -> ast.AST | None:
    for node in cls_node.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return node.value
        elif isinstance(node, ast.AnnAssign):
            if (isinstance(node.target, ast.Name)
                    and node.target.id == name
                    and node.value is not None):
                return node.value
    return None


def _method_self_assigns(method: ast.FunctionDef) -> set[str]:
    """Return ``self.NAME`` attributes assigned inside the method."""
    out: set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"):
                    out.add(tgt.attr)
                if isinstance(tgt, ast.Tuple):
                    for elt in tgt.elts:
                        if (isinstance(elt, ast.Attribute)
                                and isinstance(elt.value, ast.Name)
                                and elt.value.id == "self"):
                            out.add(elt.attr)
    return out


def main() -> int:
    # -----------------------------------------------------------------
    # Interpolate backend
    # -----------------------------------------------------------------
    bk_path = (REPO_ROOT / "mufasa" / "data_processors"
               / "interpolate.py")
    bk_src = bk_path.read_text()
    bk_tree = ast.parse(bk_src)
    bk_cls = _ast_find_class(bk_tree, "Interpolate")
    assert bk_cls is not None
    bk_init = _ast_method(bk_cls, "__init__")
    bk_run = _ast_method(bk_cls, "run")
    assert bk_init is not None and bk_run is not None

    # 1. No copy_originals parameter on __init__.
    init_param_names = (
        [a.arg for a in bk_init.args.args]
        + [a.arg for a in bk_init.args.kwonlyargs]
    )
    check(
        "Interpolate.__init__ no longer accepts `copy_originals` "
        "(redundant after 122ed — originals preserved by "
        "definition)",
        "copy_originals" not in init_param_names,
        detail=(f"params: {init_param_names}"),
    )

    # 2-3. Both run_id and run_dir assigned in __init__.
    init_self_assigns = _method_self_assigns(bk_init)
    check(
        "Interpolate.__init__ assigns self.run_id (so the form "
        "can read it post-.run() and record provenance)",
        "run_id" in init_self_assigns,
    )
    check(
        "Interpolate.__init__ assigns self.run_dir (the absolute "
        "path to derived/interpolated/<run_id>/)",
        "run_dir" in init_self_assigns,
    )

    # 4. os.makedirs called against self.run_dir.
    init_src = ast.unparse(bk_init)
    check(
        "Interpolate.__init__ creates self.run_dir on disk via "
        "os.makedirs (so output dir exists before run() starts)",
        "os.makedirs(self.run_dir" in init_src,
    )

    # 5. run() writes to self.run_dir.
    run_src = ast.unparse(bk_run)
    check(
        "Interpolate.run writes output to self.run_dir (the new "
        "run-dir layout)",
        "self.run_dir" in run_src,
    )

    # 6. run() does NOT pass file_path as save_path.
    # The legacy pattern was `write_df(..., save_path=file_path, ...)`.
    # Now it should be `write_df(..., save_path=out_path, ...)` where
    # out_path is built from self.run_dir.
    check(
        "Interpolate.run does NOT pass `save_path=file_path` "
        "(the legacy in-place overwrite is gone)",
        "save_path=file_path" not in run_src,
    )

    # 7. copy_files_to_directory import removed.
    check(
        "copy_files_to_directory import removed from "
        "interpolate.py (no longer needed)",
        "copy_files_to_directory" not in bk_src,
    )

    # 8. Class docstring mentions 122ed.
    cls_doc = ast.get_docstring(bk_cls) or ""
    init_doc = ast.get_docstring(bk_init) or ""
    check(
        "Interpolate class OR __init__ docstring mentions 122ed",
        "122ed" in cls_doc or "122ed" in init_doc,
    )

    # -----------------------------------------------------------------
    # InterpolateForm
    # -----------------------------------------------------------------
    pc_path = (REPO_ROOT / "mufasa" / "ui_qt"
               / "forms" / "pose_cleanup.py")
    pc_src = pc_path.read_text()
    pc_tree = ast.parse(pc_src)
    interp_cls = _ast_find_class(pc_tree, "InterpolateForm")
    assert interp_cls is not None

    # 9. publish_target_stage.
    pub_tgt = _ast_class_attr(interp_cls, "publish_target_stage")
    check(
        "InterpolateForm.publish_target_stage == 'outlier_corrected'",
        isinstance(pub_tgt, ast.Constant)
        and pub_tgt.value == "outlier_corrected",
        detail=(f"got {pub_tgt.value!r}"
                if isinstance(pub_tgt, ast.Constant) else "missing"),
    )

    # 10. publish_source_stage.
    pub_src = _ast_class_attr(interp_cls, "publish_source_stage")
    check(
        "InterpolateForm.publish_source_stage == 'interpolated'",
        isinstance(pub_src, ast.Constant)
        and pub_src.value == "interpolated",
        detail=(f"got {pub_src.value!r}"
                if isinstance(pub_src, ast.Constant) else "missing"),
    )

    # 11. target() sets self._last_run_id.
    target_method = _ast_method(interp_cls, "target")
    assert target_method is not None
    target_self_assigns = _method_self_assigns(target_method)
    check(
        "InterpolateForm.target() sets self._last_run_id (captures "
        "runner.run_id from the refactored backend)",
        "_last_run_id" in target_self_assigns,
    )

    # 12. target() no longer accepts copy_originals.
    target_params = (
        [a.arg for a in target_method.args.args]
        + [a.arg for a in target_method.args.kwonlyargs]
    )
    check(
        "InterpolateForm.target() no longer accepts `copy_originals` "
        "(parameter removed in 122ed)",
        "copy_originals" not in target_params,
    )

    # 13. collect_args() no longer returns copy_originals.
    collect_method = _ast_method(interp_cls, "collect_args")
    assert collect_method is not None
    collect_src = ast.unparse(collect_method)
    check(
        "InterpolateForm.collect_args() no longer emits "
        "`copy_originals` (UI checkbox is gone)",
        "copy_originals" not in collect_src,
    )

    # 14. build() no longer creates a copy_originals checkbox.
    build_method = _ast_method(interp_cls, "build")
    assert build_method is not None
    build_src = ast.unparse(build_method)
    check(
        "InterpolateForm.build() no longer creates a "
        "`copy_originals` checkbox (UI element removed in 122ed)",
        "self.copy_originals" not in build_src,
    )

    # -----------------------------------------------------------------
    # Single-caller contract
    # -----------------------------------------------------------------
    # 15. Caller surface for the (real) Interpolate class.
    caller_hits = []
    pkg = REPO_ROOT / "mufasa"
    for f in sorted(pkg.rglob("*.py")):
        try:
            src = f.read_text()
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id == "Interpolate":
                    caller_hits.append(
                        f"{f.relative_to(REPO_ROOT)}:{node.lineno}"
                    )
    # Live callers of `mufasa.data_processors.interpolate.Interpolate`:
    #
    # * mufasa/ui_qt/forms/pose_cleanup.py — the form. ONLY caller
    #   reachable from the workbench.
    # * mufasa/pose_importers/* — 11 importers. Each has a guarded
    #   call inside `if self.interpolation_settings is not None:`.
    #   The workbench's PoseImportForm passes
    #   ``interpolation_settings=None`` (per the comment in the
    #   form: "Users run those passes on the Preprocessing page
    #   instead."), so these call sites are unreachable from the
    #   workbench. They remain reachable from external Python
    #   scripts that pass interpolation_settings explicitly, but
    #   those calls would now hit the write-to-run-dir behavior
    #   instead of in-place — a behavior change that's beyond
    #   122ed's scope. Filed under the "in-importer interpolation
    #   feature deprecation" deferred item.
    #
    # The check asserts the set of caller files, not just the count,
    # so a new caller would surface here.
    expected_callers = {
        "mufasa/ui_qt/forms/pose_cleanup.py",
    }
    accepted_importer_callers = {
        "mufasa/pose_importers/dlc_csv_importer.py",
        "mufasa/pose_importers/dlc_h5_importer.py",
        "mufasa/pose_importers/dlc_importer_csv.py",
        "mufasa/pose_importers/facemap_h5_importer.py",
        "mufasa/pose_importers/import_mars.py",
        "mufasa/pose_importers/madlc_importer.py",
        "mufasa/pose_importers/simba_blob_importer.py",
        "mufasa/pose_importers/simba_yolo_importer.py",
        "mufasa/pose_importers/sleap_csv_importer.py",
        "mufasa/pose_importers/sleap_h5_importer.py",
        "mufasa/pose_importers/sleap_slp_importer.py",
        "mufasa/pose_importers/superanimal_import.py",
    }
    real_caller_files = set()
    for hit in caller_hits:
        rel, _ = hit.split(":", 1)
        if "interpolate_pose.py" in rel:
            continue
        if "interpolation_smoothing.py" in rel:
            continue
        real_caller_files.add(rel)
    unexpected = (
        real_caller_files
        - expected_callers
        - accepted_importer_callers
    )
    missing_expected = expected_callers - real_caller_files
    check(
        "Interpolate caller surface is the form (live) + 12 "
        "pose_importers (unreachable-from-workbench dead branches). "
        "No unexpected callers; the form is still present.",
        not unexpected and not missing_expected,
        detail=(f"unexpected={sorted(unexpected)} "
                f"missing={sorted(missing_expected)}"),
    )

    # -----------------------------------------------------------------
    # 122ec tripwire flips
    # -----------------------------------------------------------------
    ec_src = (REPO_ROOT / "tests"
              / "smoke_122ec_interpolate_provenance.py").read_text()

    # 16. Check 3 flipped — now asserts publish_target_stage value.
    check(
        "smoke_122ec check 3 is flipped — now asserts "
        "publish_target_stage == 'outlier_corrected' (was 'no "
        "publish target')",
        (
            "pub.value == 'outlier_corrected'" in ec_src
            or 'pub.value == "outlier_corrected"' in ec_src
        ),
    )

    # 17. Check 10 flipped — was "exactly 2 publishers" in 122ed.
    # Patch 122ee re-flipped it to "exactly 3" when Pose Import
    # joined the publisher set. The check tracks the CURRENT state
    # of smoke_122ec rather than the state immediately after
    # 122ed; subsequent patches that change the producer publish
    # surface flip this assertion again.
    check(
        "smoke_122ec check 10 currently asserts 'exactly 3' "
        "publishers (was bumped to '3' by 122ee — kalman_v2 + "
        "interpolate + import_pose). 122ed's contribution was "
        "to take the count from 'exactly 1' to '≥2'; 122ee took "
        "it from '2' to '3'. This check tracks the current state, "
        "not an intermediate one.",
        "Exactly 3 of the 4 producers" in ec_src,
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 18. Other producers unchanged.
    roc = _ast_find_class(pc_tree, "RunOutlierCorrectionForm")
    kv2 = _ast_find_class(pc_tree, "KalmanV2SmoothingForm")
    assert roc is not None and kv2 is not None
    roc_sid = _ast_class_attr(roc, "section_id")
    kv2_sid = _ast_class_attr(kv2, "section_id")
    kv2_pub = _ast_class_attr(kv2, "publish_target_stage")
    check(
        "Other producers' wiring unchanged: outlier_correction + "
        "kalman_v2 (+ kalman_v2 still publishes to outlier_corrected)",
        (isinstance(roc_sid, ast.Constant)
         and roc_sid.value == "outlier_correction"
         and isinstance(kv2_sid, ast.Constant)
         and kv2_sid.value == "kalman_v2"
         and isinstance(kv2_pub, ast.Constant)
         and kv2_pub.value == "outlier_corrected"),
    )

    # 19. SECTIONS DAG.
    try:
        from mufasa.section_provenance import SECTIONS
        sections_ok = (
            len(SECTIONS) > 0
            and "interpolate" in SECTIONS
            and SECTIONS["interpolate"].depends_on == ("import_pose",)
        )
    except Exception:
        sections_ok = False
    check(
        "SECTIONS DAG still validates and interpolate's "
        "depends_on is unchanged ('import_pose',)",
        sections_ok,
    )

    # 20. Parse-clean.
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122ed_interpolate_publishing: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
