"""
tests/smoke_122eb_pose_import_provenance.py
==============================================

Patch 122eb: wire :class:`PoseImportForm` to record provenance.
The 3rd of 4 producers in the SECTIONS DAG to get wiring; was
deferred in 122dt because the per-tracker importer backends
control their own output paths (the form has no run-id to record
without backend changes).

What this patch landed
----------------------
Minimal "record-only" wiring (no publish target). Three changes:

1. ``mufasa/ui_qt/forms/pose_import.py`` — added
   ``section_id = "import_pose"`` to ``PoseImportForm``.
   ``self._last_run_id`` is NOT set in ``target()`` (stays None);
   the form has no run-id concept to share. Result: the base
   class's ``_record_provenance`` calls
   ``record_run("import_pose", run_id=None)``, which writes
   ``[provenance.import_pose]`` with just ``last_run_at`` (no
   ``last_run_id`` field). That's the supported "settings-shaped"
   provenance entry — described in
   :func:`mufasa.section_provenance.record_run`'s docstring as
   the case for sections without a run-id concept.

2. ``mufasa/ui_qt/pages/data_import_page.py`` — renamed section
   from "Import Pose Data" (title case) to "Import pose data"
   (sentence case) so it matches the SECTIONS spec entry
   verbatim. The badge lookup at
   :func:`mufasa.section_provenance.find_section_by_title` does
   exact-string matching against ``(page, section_title)``; the
   pre-122eb title case mismatched and silently suppressed the
   badge. Sentence case is the established workbench convention
   (Run outlier correction / Frame labelling / etc.).

3. ``tests/smoke_122dt_producer_wiring.py`` — check 20 (which
   asserted PoseImportForm has NO section_id, the deferred
   status from 122dt) is flipped to assert
   ``section_id == "import_pose"``. The deferred-status tripwire
   served its purpose; the wiring-is-in-place tripwire takes
   over.

Why record-only / no publish
----------------------------
Publishing pose-import output under
``derived/outlier_corrected/<run_id>`` would let downstream
consumers (Features etc.) read pose data WITHOUT the user
having to run outlier correction or smoothing first — the
"Skip outlier correction" replacement story.

That requires the backend to either:
* allocate a v1 run-id BEFORE calling the per-tracker importer
  and pre-create ``sources/pose/<run_id>/`` (then have each
  importer write into it — needs backend changes);
* OR write to ``sources/pose/`` flatly and have the form
  post-hoc snapshot the just-written files to
  ``derived/pose/<run_id>/`` via symlinks, then publish.

Both options are bigger than 122eb. Filed as deferred follow-up
(122ec or its successor); the SECTIONS DAG's staleness story
works with just the timestamp.

Effect on the workbench
-----------------------
After this patch, re-importing pose data correctly marks the
3 dependent sections STALE:

* outlier_correction (depends_on=("import_pose",))
* kalman_v2          (depends_on=("import_pose",))
* interpolate        (depends_on=("import_pose",))

Pre-122eb: none of them could become STALE because their
parent's ``last_run_at`` was never recorded. Post-122eb: they
go orange whenever a re-import happens after they last ran.

This is the FIRST patch where the badge staleness mechanism
exercises end-to-end on a real workflow.

Coverage
--------
PoseImportForm:
1.  ``PoseImportForm`` is an ``OperationForm`` subclass.
2.  ``PoseImportForm.section_id == "import_pose"``.
3.  ``PoseImportForm`` has NO ``publish_target_stage``
    declaration (record-only, no publish — verifies the
    contract isn't accidentally extended to a publishing form
    without thinking through the backend implications).
4.  ``PoseImportForm.target()`` does NOT set
    ``self._last_run_id`` (the per-tracker backends don't
    expose run-ids; the form leaves it None so ``record_run``
    receives ``run_id=None`` and writes a ``last_run_at``-only
    entry).
5.  ``PoseImportForm.title`` is sentence-cased and matches the
    SECTIONS spec ("Import pose data" not "Import Pose Data").

data_import_page section title alignment:
6.  ``build_data_import_page`` registers the pose-import
    section with title "Import pose data" (matches SECTIONS).
7.  No remaining "Import Pose Data" (title-case) reference in
    ``mufasa/ui_qt/pages/data_import_page.py``.

SECTIONS DAG integrity:
8.  ``SECTIONS["import_pose"]`` exists and has
    ``section_title == "Import pose data"`` (no SECTIONS edit
    required for 122eb; verifies the spec already matched).
9.  ``find_section_by_title("Data Import", "Import pose data")``
    returns the import_pose SectionSpec (i.e., the title
    alignment closed the lookup gap that previously suppressed
    the badge).

122dt tripwire flipped:
10. ``tests/smoke_122dt_producer_wiring.py`` check 20 now
    asserts the section_id is "import_pose" (verifies the
    historical tripwire was updated when the wiring landed).

Cross-patch invariants:
11. The other producers' wiring is unchanged:
    - RunOutlierCorrectionForm.section_id == "outlier_correction"
    - KalmanV2SmoothingForm.section_id == "kalman_v2"
12. InterpolateForm still has no section_id (still deferred —
    the Interpolate backend modifies pose data in place, which
    is a real refactor 122eb didn't take on).
13. 122ea state preserved: ConfigReader.__init__ doesn't have
    the three legacy-shaped fallback assignments.
14. 122dz state preserved: load_machine_results_for_video has
    no legacy_fallback parameter.
15. Parse-clean across mufasa/**/*.py.
16. 122do baseline: no Optional[ in non-docstring positions
    across mufasa/ui_qt/.
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


def main() -> int:
    # -----------------------------------------------------------------
    # PoseImportForm
    # -----------------------------------------------------------------
    pi_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "pose_import.py"
    pi_src = pi_path.read_text()
    pi_tree = ast.parse(pi_src)
    pi_cls = _ast_find_class(pi_tree, "PoseImportForm")
    assert pi_cls is not None

    # 1. Inherits from OperationForm.
    base_names = [
        b.id for b in pi_cls.bases if isinstance(b, ast.Name)
    ]
    check(
        "PoseImportForm inherits from OperationForm "
        "(so the base class's provenance hook fires)",
        "OperationForm" in base_names,
    )

    # 2. section_id == "import_pose"
    sid = _ast_class_attr(pi_cls, "section_id")
    check(
        "PoseImportForm.section_id == 'import_pose' (provenance "
        "recording target)",
        isinstance(sid, ast.Constant) and sid.value == "import_pose",
        detail=(f"got {sid.value!r}"
                if isinstance(sid, ast.Constant) else "missing"),
    )

    # 3. publish_target_stage is NOW SET (flipped by 122ee — was
    # "no publish target" in 122eb's record-only contract).
    pub = _ast_class_attr(pi_cls, "publish_target_stage")
    check(
        "PoseImportForm.publish_target_stage == 'outlier_corrected' "
        "(flipped by 122ee — the 122eb record-only contract was "
        "lifted when the form-level snapshot landed)",
        isinstance(pub, ast.Constant)
        and pub.value == "outlier_corrected",
        detail=(f"got {pub.value!r}"
                if isinstance(pub, ast.Constant) else "missing"),
    )

    # 4. _last_run_id IS now set somewhere in PoseImportForm
    # (flipped by 122ee). The assignment happens in
    # _snapshot_and_set_run_id (a helper called from target);
    # walk the class body to find any method that assigns it.
    sets_run_id = False
    for member in pi_cls.body:
        if isinstance(member, ast.FunctionDef):
            for node in ast.walk(member):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if (isinstance(tgt, ast.Attribute)
                                and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "self"
                                and tgt.attr == "_last_run_id"):
                            sets_run_id = True
                            break
    check(
        "PoseImportForm sets self._last_run_id somewhere in its "
        "method body (flipped by 122ee — was 'never set' in the "
        "122eb record-only contract; now set in the "
        "_snapshot_and_set_run_id helper that target() calls "
        "post-import)",
        sets_run_id,
    )

    # 5. title is sentence-cased.
    title_node = _ast_class_attr(pi_cls, "title")
    check(
        "PoseImportForm.title == 'Import pose data' (sentence "
        "case; aligned with SECTIONS spec in 122eb)",
        isinstance(title_node, ast.Constant)
        and title_node.value == "Import pose data",
        detail=(f"got {title_node.value!r}"
                if isinstance(title_node, ast.Constant) else "missing"),
    )

    # -----------------------------------------------------------------
    # data_import_page.py section title alignment
    # -----------------------------------------------------------------
    dip_path = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
                / "data_import_page.py")
    dip_src = dip_path.read_text()

    # 6. The page registers the pose section with the sentence-case
    # title.
    check(
        "build_data_import_page registers the pose-import section "
        "with title 'Import pose data' (sentence case; matches "
        "SECTIONS)",
        '"Import pose data"' in dip_src,
    )

    # 7. The live add_section call uses sentence case. (We
    # explicitly DON'T grep the whole file because the patch
    # history block legitimately contains "Import Pose Data"
    # as a 122w-era reference.)
    has_live_title_case = bool(re.search(
        r'add_section\s*\(\s*["\']Import Pose Data["\']',
        dip_src,
    ))
    check(
        "build_data_import_page's add_section call uses 'Import "
        "pose data' (sentence case) — no live title-case "
        "registration remains (122eb fixed the badge-lookup gap)",
        not has_live_title_case,
    )

    # -----------------------------------------------------------------
    # SECTIONS DAG integrity
    # -----------------------------------------------------------------
    from mufasa.section_provenance import (
        SECTIONS,
        find_section_by_title,
    )

    # 8. SECTIONS["import_pose"] exists with matching section_title.
    spec = SECTIONS.get("import_pose")
    check(
        "SECTIONS['import_pose'] exists with section_title "
        "'Import pose data' (page 'Data Import')",
        spec is not None
        and spec.section_title == "Import pose data"
        and spec.page == "Data Import",
    )

    # 9. find_section_by_title resolves the import_pose section
    # (closes the lookup gap that suppressed the badge pre-122eb).
    looked_up = find_section_by_title("Data Import", "Import pose data")
    check(
        "find_section_by_title('Data Import', 'Import pose data') "
        "resolves to SECTIONS['import_pose'] — this is what the "
        "badge-paint code uses to find the section spec from a "
        "QToolBox item title",
        looked_up is not None
        and looked_up.section_id == "import_pose",
    )

    # -----------------------------------------------------------------
    # 122dt tripwire flipped
    # -----------------------------------------------------------------
    dt_src = (REPO_ROOT / "tests"
              / "smoke_122dt_producer_wiring.py").read_text()
    # Look for the actual assertion pattern. The flipped check
    # asserts `pi_sid.value == "import_pose"` — searching for that
    # literal is the most direct way to verify the tripwire flip.
    check(
        "smoke_122dt_producer_wiring.py check 20 now asserts the "
        "PoseImportForm.section_id value is 'import_pose' "
        "(deferred-status tripwire flipped to wiring-in-place "
        "tripwire)",
        (
            "pi_sid.value == 'import_pose'" in dt_src
            or 'pi_sid.value == "import_pose"' in dt_src
        ),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    pc_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "forms" / "pose_cleanup.py").read_text()
    pc_tree = ast.parse(pc_src)

    # 11. Other producers unchanged.
    roc = _ast_find_class(pc_tree, "RunOutlierCorrectionForm")
    kv2 = _ast_find_class(pc_tree, "KalmanV2SmoothingForm")
    assert roc is not None and kv2 is not None
    roc_sid = _ast_class_attr(roc, "section_id")
    kv2_sid = _ast_class_attr(kv2, "section_id")
    check(
        "Other producers' section_ids unchanged: "
        "outlier_correction + kalman_v2",
        (isinstance(roc_sid, ast.Constant)
         and roc_sid.value == "outlier_correction"
         and isinstance(kv2_sid, ast.Constant)
         and kv2_sid.value == "kalman_v2"),
    )

    # 12. InterpolateForm — was an unwired-tripwire here, flipped
    # by 122ec.
    interp = _ast_find_class(pc_tree, "InterpolateForm")
    assert interp is not None
    interp_sid = _ast_class_attr(interp, "section_id")
    check(
        "InterpolateForm.section_id == 'interpolate' (was an "
        "unwired-tripwire here in 122eb; flipped by 122ec which "
        "applied the same record-only pattern to Interpolate)",
        isinstance(interp_sid, ast.Constant)
        and interp_sid.value == "interpolate",
    )

    # 13. 122ea state preserved.
    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()
    check(
        "122ea state preserved: ConfigReader.__init__ no longer "
        "has the three legacy-shaped fallback assignments",
        "Paths.ANNOTATED_FRAMES_DIR" not in cr_src
        and "Paths.SINGLE_CLF_VALIDATION" not in cr_src,
    )

    # 14. 122dz state preserved.
    cio_src = (REPO_ROOT / "mufasa" / "utils"
               / "classification_io.py").read_text()
    cio_tree = ast.parse(cio_src)
    helper = None
    for node in ast.walk(cio_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "load_machine_results_for_video"):
            helper = node
            break
    assert helper is not None
    helper_params = (
        [a.arg for a in helper.args.args]
        + [a.arg for a in helper.args.kwonlyargs]
    )
    check(
        "122dz state preserved: load_machine_results_for_video "
        "still has no legacy_fallback parameter",
        "legacy_fallback" not in helper_params,
    )

    # 15. Parse-clean.
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

    # 16. 122do baseline.
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
        f"smoke_122eb_pose_import_provenance: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
