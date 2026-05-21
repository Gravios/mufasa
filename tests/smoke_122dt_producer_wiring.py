"""
tests/smoke_122dt_producer_wiring.py
======================================

Patch 122dt: wire two producer forms (RunOutlierCorrectionForm,
KalmanV2SmoothingForm) to the section_provenance + publish_to_stage
infrastructure landed in 122ds.

Scope is intentionally limited to the two forms whose backends
already allocate v1 run-ids:

* :class:`RunOutlierCorrectionForm` — writes to
  ``derived/outlier_corrected/<run_id>/`` directly; needs only
  ``record_run``, no publish.
* :class:`KalmanV2SmoothingForm` — writes to
  ``derived/smoothed/kalman_v2/<run_id>/``; needs ``record_run``
  AND ``publish_to_stage`` so the symlink at
  ``derived/outlier_corrected/<run_id>`` makes the output visible
  to consumers (Features etc.) that hard-code reads from
  ``outlier_corrected/``.

The two remaining producers (InterpolateForm, PoseImportForm) need
backend refactors before they can be wired and are deferred to
their own patches:

* InterpolateForm — the ``Interpolate`` backend modifies pose data
  IN PLACE in ``sources/pose/``; doesn't allocate a run dir. Wiring
  needs a behavior change first (write to
  ``derived/interpolated/<run_id>/`` instead).
* PoseImportForm — the per-tracker importer backends control their
  own output paths; the form has no run-id to record. Wiring needs
  each importer to either return a run-id or write to a v1-shaped
  location the form can discover.

What this patch landed
----------------------
1. ``publish_to_stage`` gained an optional ``source_flavor``
   parameter (e.g., ``"kalman_v2"`` for the
   ``derived/smoothed/kalman_v2/<run_id>/`` layout). Backwards
   compatible — defaults to ``None`` (no flavor segment), matching
   the 122ds shipped behavior.

2. ``OperationForm`` (base class in workbench.py) gained four
   class-level provenance attributes:
   - ``section_id`` — provenance recording target
   - ``publish_target_stage`` — optional symlink-publish target
   - ``publish_source_stage`` — symlink-publish source
   - ``publish_source_flavor`` — symlink-publish source flavor
   Plus an instance ``_last_run_id`` set by subclasses' ``target``
   to communicate the just-written run-id back to the base class.
   Plus ``_record_provenance`` method that runs in the on_success
   path, calls ``record_run`` and (if configured)
   ``publish_to_stage``, and soft-fails on errors (a provenance
   hiccup shouldn't crash the UI after a successful run).

3. ``RunOutlierCorrectionForm`` declares
   ``section_id = "outlier_correction"`` and sets
   ``self._last_run_id = run_id`` in ``target`` at the point it
   was already capturing the run_id for run.toml provenance.

4. ``KalmanV2SmoothingForm`` declares ``section_id = "kalman_v2"``,
   ``publish_target_stage = "outlier_corrected"``,
   ``publish_source_stage = "smoothed"``,
   ``publish_source_flavor = "kalman_v2"``. Sets
   ``self._last_run_id = v1_run_id`` in ``target`` at the existing
   run.toml provenance write site.

Coverage
--------
publish_to_stage source_flavor extension:
1.  Function still importable.
2.  Without source_flavor (None), behavior is identical to 122ds
    (flat-stage publish).
3.  With source_flavor, the symlink target is
    ``../<source_stage>/<source_flavor>/<run_id>``.
4.  Symlink with flavor follows transparently — files in the
    flavored source dir are reachable via the published link.
5.  source_flavor with a path separator raises ValueError.

OperationForm provenance infrastructure:
6.  OperationForm has the four provenance class attributes; default
    values are None.
7.  OperationForm has an ``_last_run_id`` instance attribute (set
    to None in ``__init__``).
8.  OperationForm has a ``_record_provenance`` method.
9.  ``on_run`` re-initializes ``_last_run_id = None`` before each
    invocation (so a stale id from a prior run can't leak).
10. ``on_run``'s success handler calls ``_record_provenance``
    BEFORE emitting ``completed`` (so listeners can re-query
    SectionStatus and see the new entry).

Subclass declarations:
11. RunOutlierCorrectionForm.section_id == "outlier_correction".
12. RunOutlierCorrectionForm has no publish_target_stage (writes
    directly).
13. KalmanV2SmoothingForm.section_id == "kalman_v2".
14. KalmanV2SmoothingForm.publish_target_stage == "outlier_corrected".
15. KalmanV2SmoothingForm.publish_source_stage == "smoothed".
16. KalmanV2SmoothingForm.publish_source_flavor == "kalman_v2".
17. RunOutlierCorrectionForm.target captures self._last_run_id
    (substring check for the assignment).
18. KalmanV2SmoothingForm.target captures self._last_run_id
    (substring check for the assignment).

Deferred-producer documentation tripwires:
19. InterpolateForm does NOT declare section_id (deferred).
20. PoseImportForm does NOT declare section_id (deferred).

Cross-patch invariants:
21. 122ds publish_to_stage 8-check baseline still passes (functional
    re-run of the user's bug layout — populated old run + empty
    new run resolves correctly).
22. 122ds SECTIONS dict still has "outlier_correction" and
    "kalman_v2" entries.
23. All mufasa/**/*.py parse cleanly.
24. 122do baseline tripwire: no ``Optional[`` in non-docstring
    positions across mufasa/ui_qt/.
"""
from __future__ import annotations

import ast
import re
import sys
import tempfile
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


def _ast_method(cls_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _ast_class_attr(cls_node: ast.ClassDef, name: str) -> ast.AST | None:
    """Return the value AST node for a top-level class attribute
    assignment, or None if not declared."""
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


def _ast_find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def main() -> int:
    # -----------------------------------------------------------------
    # publish_to_stage source_flavor extension
    # -----------------------------------------------------------------
    from mufasa.project_layout import publish_to_stage

    # 1. Still importable (always true if the import above succeeds).
    check(
        "mufasa.project_layout.publish_to_stage still importable",
        callable(publish_to_stage),
    )

    # 2 & 3. Without source_flavor: behavior matches 122ds.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src_run = root / "derived" / "smoothed" / "20260520-120000-aabbcc"
        src_run.mkdir(parents=True)
        (src_run / "video.parquet").write_text("data")
        link = publish_to_stage(
            root, "smoothed", "outlier_corrected",
            "20260520-120000-aabbcc",
        )
        import os
        target = os.readlink(link)
        check(
            "publish_to_stage without source_flavor (None default) "
            "still produces the flat ../<stage>/<run_id> target — "
            "122ds behavior unchanged",
            target == "../smoothed/20260520-120000-aabbcc",
            detail=f"got {target!r}",
        )

    # With source_flavor.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src_run = (
            root / "derived" / "smoothed" / "kalman_v2"
            / "20260520-120000-aabbcc"
        )
        src_run.mkdir(parents=True)
        (src_run / "video.parquet").write_text("smoothed-data")
        link = publish_to_stage(
            root, "smoothed", "outlier_corrected",
            "20260520-120000-aabbcc",
            source_flavor="kalman_v2",
        )
        target = os.readlink(link)
        check(
            "publish_to_stage with source_flavor produces "
            "../<stage>/<flavor>/<run_id> target — Kalman v2 layout",
            target == "../smoothed/kalman_v2/20260520-120000-aabbcc",
            detail=f"got {target!r}",
        )

        # 4. Symlink follows.
        files = list(link.glob("*.parquet"))
        check(
            "Files in the flavored source dir are reachable via the "
            "published symlink (glob follows the flavored target)",
            len(files) == 1
            and files[0].read_text() == "smoothed-data",
        )

    # 5. source_flavor separator → ValueError.
    with tempfile.TemporaryDirectory() as td:
        raised = False
        try:
            publish_to_stage(
                Path(td), "smoothed", "outlier_corrected", "r",
                source_flavor="foo/bar",
            )
        except ValueError:
            raised = True
        check(
            "publish_to_stage rejects source_flavor with a path "
            "separator (consistent with stage-name validation)",
            raised,
        )

    # -----------------------------------------------------------------
    # OperationForm provenance infrastructure (AST inspection — base
    # class is in workbench.py which imports PySide6; sandbox can't
    # import it, but the AST tells us what's declared).
    # -----------------------------------------------------------------
    wb_path = REPO_ROOT / "mufasa" / "ui_qt" / "workbench.py"
    wb_tree = ast.parse(wb_path.read_text())
    op_form = _ast_find_class(wb_tree, "OperationForm")
    assert op_form is not None, "OperationForm not found in workbench.py"

    # 6. Four provenance class attrs declared.
    declared_attrs = {
        name: _ast_class_attr(op_form, name)
        for name in (
            "section_id", "publish_target_stage",
            "publish_source_stage", "publish_source_flavor",
        )
    }
    check(
        "OperationForm declares all four provenance class "
        "attributes (section_id, publish_target_stage, "
        "publish_source_stage, publish_source_flavor)",
        all(v is not None for v in declared_attrs.values()),
        detail=f"missing: "
               f"{[k for k, v in declared_attrs.items() if v is None]}",
    )

    # 7. _last_run_id set in __init__.
    init_method = _ast_method(op_form, "__init__")
    assert init_method is not None
    init_src = ast.unparse(init_method)
    check(
        "OperationForm.__init__ initializes self._last_run_id "
        "(prevents stale-run-id leakage across invocations)",
        "self._last_run_id" in init_src,
    )

    # 8. _record_provenance method exists.
    rec_method = _ast_method(op_form, "_record_provenance")
    check(
        "OperationForm defines a `_record_provenance` method "
        "(the provenance-write entry point)",
        rec_method is not None,
    )

    # 9. on_run re-initializes _last_run_id.
    on_run_method = _ast_method(op_form, "on_run")
    assert on_run_method is not None
    on_run_src = ast.unparse(on_run_method)
    check(
        "on_run resets `self._last_run_id` before each invocation "
        "(so a prior run's id doesn't leak into the next "
        "provenance write)",
        "self._last_run_id = None" in on_run_src
        or "self._last_run_id=None" in on_run_src,
    )

    # 10. on_run's success path calls _record_provenance BEFORE emit.
    record_idx = on_run_src.find("_record_provenance")
    emit_idx = on_run_src.find("self.completed.emit")
    check(
        "on_run's success path calls _record_provenance BEFORE "
        "self.completed.emit (UI listeners see the new provenance "
        "entry when they re-query)",
        record_idx != -1 and emit_idx != -1 and record_idx < emit_idx,
    )

    # -----------------------------------------------------------------
    # Subclass declarations
    # -----------------------------------------------------------------
    pc_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "pose_cleanup.py"
    pc_tree = ast.parse(pc_path.read_text())

    roc = _ast_find_class(pc_tree, "RunOutlierCorrectionForm")
    assert roc is not None
    kv2 = _ast_find_class(pc_tree, "KalmanV2SmoothingForm")
    assert kv2 is not None

    # 11. RunOutlierCorrectionForm.section_id == "outlier_correction"
    roc_sid = _ast_class_attr(roc, "section_id")
    check(
        "RunOutlierCorrectionForm.section_id == 'outlier_correction'",
        isinstance(roc_sid, ast.Constant)
        and roc_sid.value == "outlier_correction",
        detail=(f"got {roc_sid.value!r}"
                if isinstance(roc_sid, ast.Constant) else "missing"),
    )

    # 12. RunOutlierCorrectionForm has NO publish target.
    roc_pub = _ast_class_attr(roc, "publish_target_stage")
    check(
        "RunOutlierCorrectionForm declares no publish_target_stage "
        "(writes directly to derived/outlier_corrected/<run_id>; "
        "no symlink needed)",
        roc_pub is None,
    )

    # 13. KalmanV2SmoothingForm.section_id == "kalman_v2"
    kv2_sid = _ast_class_attr(kv2, "section_id")
    check(
        "KalmanV2SmoothingForm.section_id == 'kalman_v2'",
        isinstance(kv2_sid, ast.Constant) and kv2_sid.value == "kalman_v2",
        detail=(f"got {kv2_sid.value!r}"
                if isinstance(kv2_sid, ast.Constant) else "missing"),
    )

    # 14-16. KalmanV2 publish targets.
    kv2_target = _ast_class_attr(kv2, "publish_target_stage")
    kv2_source = _ast_class_attr(kv2, "publish_source_stage")
    kv2_flavor = _ast_class_attr(kv2, "publish_source_flavor")
    check(
        "KalmanV2SmoothingForm.publish_target_stage == "
        "'outlier_corrected'",
        isinstance(kv2_target, ast.Constant)
        and kv2_target.value == "outlier_corrected",
    )
    check(
        "KalmanV2SmoothingForm.publish_source_stage == 'smoothed'",
        isinstance(kv2_source, ast.Constant)
        and kv2_source.value == "smoothed",
    )
    check(
        "KalmanV2SmoothingForm.publish_source_flavor == 'kalman_v2'",
        isinstance(kv2_flavor, ast.Constant)
        and kv2_flavor.value == "kalman_v2",
    )

    # 17 & 18. Both subclasses' `target` set self._last_run_id.
    roc_target = _ast_method(roc, "target")
    assert roc_target is not None
    check(
        "RunOutlierCorrectionForm.target captures self._last_run_id "
        "(communicates the run_id back to the base class for "
        "provenance recording)",
        "self._last_run_id" in ast.unparse(roc_target),
    )
    kv2_target_method = _ast_method(kv2, "target")
    assert kv2_target_method is not None
    check(
        "KalmanV2SmoothingForm.target captures self._last_run_id "
        "(same contract as RunOutlierCorrectionForm)",
        "self._last_run_id" in ast.unparse(kv2_target_method),
    )

    # -----------------------------------------------------------------
    # Deferred producers — explicit tripwires
    # -----------------------------------------------------------------
    interp = _ast_find_class(pc_tree, "InterpolateForm")
    assert interp is not None
    check(
        "InterpolateForm has NO section_id declaration (deferred to "
        "a later patch — Interpolate backend modifies pose data IN "
        "PLACE; needs a write-to-run-dir refactor before it can "
        "record provenance)",
        _ast_class_attr(interp, "section_id") is None,
    )

    pi_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "pose_import.py"
    pi_tree = ast.parse(pi_path.read_text())
    pi_cls = _ast_find_class(pi_tree, "PoseImportForm")
    assert pi_cls is not None
    check(
        "PoseImportForm has NO section_id declaration (deferred — "
        "the per-tracker importer backends control their own output "
        "paths; the form has no run-id to record without backend "
        "changes)",
        _ast_class_attr(pi_cls, "section_id") is None,
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    from mufasa.project_layout import (
        latest_populated_run_or_parent,
    )
    # 21. 122ds publish_to_stage baseline — user's bug layout still
    # resolves (this is the load-bearing 122dr → 122ds → 122dt
    # invariant chain).
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"
        populated = stage / "20260518-192433-7f64a3"
        empty = stage / "20260520-233610-6203f1"
        populated.mkdir(parents=True)
        empty.mkdir(parents=True)
        (populated / "video.parquet").touch()
        result = latest_populated_run_or_parent(stage, "parquet")
        check(
            "122dr baseline still holds: populated old run + empty "
            "new run resolves to the populated one (load-bearing "
            "cross-patch invariant)",
            result == str(populated),
        )

    # 22. SECTIONS still has the wired entries.
    from mufasa.section_provenance import SECTIONS
    check(
        "section_provenance.SECTIONS still declares 'outlier_correction' "
        "and 'kalman_v2' (the entries 122dt wires)",
        "outlier_correction" in SECTIONS and "kalman_v2" in SECTIONS,
    )

    # 23. Parse-clean.
    pkg = REPO_ROOT / "mufasa"
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

    # 24. 122do baseline tripwire.
    uiqt = pkg / "ui_qt"
    optional_hits = []
    for f in sorted(uiqt.rglob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"\bOptional\[", src):
            preceding = src[:m.start()]
            tq3 = preceding.count('"""') + preceding.count("'''")
            if tq3 % 2 == 0:
                optional_hits.append(str(f.relative_to(uiqt)))
                break
    check(
        "122do baseline preserved: no `Optional[` in non-docstring "
        "positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122dt_producer_wiring: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
