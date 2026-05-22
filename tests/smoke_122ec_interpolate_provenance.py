"""
tests/smoke_122ec_interpolate_provenance.py
=============================================

Patch 122ec: wire :class:`InterpolateForm` to record provenance.
The 4th and final producer in the SECTIONS DAG to get wiring;
was deferred in 122dt because the underlying
:class:`mufasa.data_processors.interpolate.Interpolate` backend
modifies pose data in place at ``sources/pose/``.

122ec opts for the **same record-only pattern** 122eb applied to
PoseImportForm: declare ``section_id``, don't set
``self._last_run_id`` in ``target()``, don't declare
``publish_target_stage``. The form's behavior is unchanged; the
base class records ``[provenance.interpolate]`` with just
``last_run_at`` after every successful run.

Why record-only is sufficient
-----------------------------
``interpolate`` depends on ``import_pose`` in the SECTIONS DAG.
The staleness rule is "child STALE if parent ran later." The
record-only wiring captures the timestamp; that's all the rule
needs.

The backend overwrites pose files in place, so there's no run
directory to publish a symlink for. Downstream consumers (Kalman
v2, RunOutlierCorrection) find the interpolated values
transparently because the pose files they read have already been
overwritten by Interpolate.

Future work — would the backend benefit from a write-to-run-dir
refactor? Yes: it would let users compare pre- and post-
interpolation data, and would unlock a publish-to-stage extension
(``derived/outlier_corrected/<run_id> -> ../interpolated/<run_id>``)
that closes the "Skip outlier correction" story from 122dv. But
that's a real refactor and out of scope for 122ec. Filed alongside
the Pose Import publish-to-stage work.

Section-provenance arc status after 122ec
------------------------------------------
All 4 producers wired:

* outlier_correction — record + publish=No   (122dt)
* kalman_v2          — record + publish=Yes  (122dt)
* import_pose        — record-only           (122eb)
* interpolate        — record-only           (122ec)

The deferred-producer list is empty. Further section-provenance
work would expand the DAG (label-side, classifier-side, feature-
side provenance) which isn't currently scoped.

Coverage
--------
InterpolateForm declarations:
1.  ``InterpolateForm`` is an ``OperationForm`` subclass.
2.  ``InterpolateForm.section_id == "interpolate"``.
3.  ``InterpolateForm`` has NO ``publish_target_stage``
    declaration (record-only contract; matches PoseImportForm's
    122eb pattern).
4.  ``InterpolateForm.target()`` does NOT set
    ``self._last_run_id``. The backend modifies pose files in
    place — there's no run directory to point at, so leaving the
    instance attribute None is correct.

Title alignment with SECTIONS:
5.  ``InterpolateForm.title == "Interpolate missing frames"``
    (matches SECTIONS spec; was already correct pre-122ec).
6.  ``find_section_by_title("Pose cleanup", "Interpolate
    missing frames")`` resolves to SECTIONS["interpolate"].

122dt tripwire flipped:
7.  smoke_122dt_producer_wiring's check 19 now asserts
    ``InterpolateForm.section_id == "interpolate"`` (deferred-
    status tripwire flipped).

122eb tripwire flipped:
8.  smoke_122eb_pose_import_provenance's "InterpolateForm
    still deferred" check is now also flipped to assert the
    wiring (the section-provenance arc's reciprocal-tripwire
    pattern).

Section-provenance arc completion:
9.  All four producers identified in 122dt now have section_id
    declarations (outlier_correction, kalman_v2, import_pose,
    interpolate).
10. Of those four, exactly two have publish_target_stage
    (only kalman_v2 publishes; outlier_correction writes
    directly to outlier_corrected/, the other two are
    record-only).

Cross-patch invariants:
11. 122eb state preserved: PoseImportForm.section_id ==
    "import_pose" and Data Import page uses sentence case.
12. 122ea state preserved: ConfigReader has no legacy-shaped
    fallback assignments.
13. 122dz state preserved: load_machine_results_for_video has
    no legacy_fallback parameter.
14. Parse-clean across mufasa/**/*.py.
15. 122do baseline: no Optional[ in non-docstring positions
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
    # InterpolateForm
    # -----------------------------------------------------------------
    pc_path = (REPO_ROOT / "mufasa" / "ui_qt"
               / "forms" / "pose_cleanup.py")
    pc_src = pc_path.read_text()
    pc_tree = ast.parse(pc_src)
    interp_cls = _ast_find_class(pc_tree, "InterpolateForm")
    assert interp_cls is not None

    # 1. Inherits from OperationForm.
    base_names = [
        b.id for b in interp_cls.bases if isinstance(b, ast.Name)
    ]
    check(
        "InterpolateForm inherits from OperationForm",
        "OperationForm" in base_names,
    )

    # 2. section_id == "interpolate"
    sid = _ast_class_attr(interp_cls, "section_id")
    check(
        "InterpolateForm.section_id == 'interpolate'",
        isinstance(sid, ast.Constant) and sid.value == "interpolate",
        detail=(f"got {sid.value!r}"
                if isinstance(sid, ast.Constant) else "missing"),
    )

    # 3. publish_target_stage is now SET (flipped by 122ed —
    # was "no publish target" in 122ec; backend refactor lets
    # us publish).
    pub = _ast_class_attr(interp_cls, "publish_target_stage")
    check(
        "InterpolateForm.publish_target_stage == 'outlier_corrected' "
        "(flipped by 122ed — the 122ec record-only contract was "
        "lifted when the backend refactor landed)",
        isinstance(pub, ast.Constant) and pub.value == "outlier_corrected",
        detail=(f"got {pub.value!r}"
                if isinstance(pub, ast.Constant) else "missing"),
    )

    # 4. target() NOW sets self._last_run_id (flipped by 122ed —
    # captures runner.run_id from the refactored backend).
    target_method = _ast_method(interp_cls, "target")
    assert target_method is not None
    target_src = ast.unparse(target_method)
    check(
        "InterpolateForm.target() sets self._last_run_id (flipped "
        "by 122ed — captures the backend's allocated run_id from "
        "the new run-dir layout)",
        "self._last_run_id" in target_src,
    )

    # 5. Title matches SECTIONS spec.
    title_node = _ast_class_attr(interp_cls, "title")
    check(
        "InterpolateForm.title == 'Interpolate missing frames' "
        "(matches SECTIONS spec)",
        isinstance(title_node, ast.Constant)
        and title_node.value == "Interpolate missing frames",
        detail=(f"got {title_node.value!r}"
                if isinstance(title_node, ast.Constant) else "missing"),
    )

    # -----------------------------------------------------------------
    # SECTIONS lookup
    # -----------------------------------------------------------------
    from mufasa.section_provenance import (
        SECTIONS,
        find_section_by_title,
    )

    # 6. find_section_by_title resolves the interpolate section.
    looked_up = find_section_by_title(
        "Pose cleanup", "Interpolate missing frames",
    )
    check(
        "find_section_by_title('Pose cleanup', 'Interpolate "
        "missing frames') resolves to SECTIONS['interpolate']",
        looked_up is not None
        and looked_up.section_id == "interpolate",
    )

    # -----------------------------------------------------------------
    # Reciprocal tripwire flips
    # -----------------------------------------------------------------
    # 7. 122dt's check 19 now asserts wiring is in place.
    dt_src = (REPO_ROOT / "tests"
              / "smoke_122dt_producer_wiring.py").read_text()
    check(
        "smoke_122dt_producer_wiring.py check 19 now asserts the "
        "InterpolateForm.section_id value is 'interpolate' "
        "(was: asserts it is None / deferred). 122ec flipped the "
        "tripwire when the wiring landed.",
        (
            "interp_sid.value == 'interpolate'" in dt_src
            or 'interp_sid.value == "interpolate"' in dt_src
        ),
    )

    # 8. 122eb's "InterpolateForm still deferred" check is also
    # flipped.
    eb_src = (REPO_ROOT / "tests"
              / "smoke_122eb_pose_import_provenance.py").read_text()
    check(
        "smoke_122eb_pose_import_provenance.py's InterpolateForm "
        "check is also flipped (asserts wiring, not deferral). "
        "The reciprocal-tripwire pattern means both sister tests "
        "stay in sync.",
        (
            "interp_sid.value == 'interpolate'" in eb_src
            or 'interp_sid.value == "interpolate"' in eb_src
        ),
    )

    # -----------------------------------------------------------------
    # Section-provenance arc completion
    # -----------------------------------------------------------------
    # 9. All four producers have section_id declarations.
    roc = _ast_find_class(pc_tree, "RunOutlierCorrectionForm")
    kv2 = _ast_find_class(pc_tree, "KalmanV2SmoothingForm")
    assert roc is not None and kv2 is not None
    roc_sid = _ast_class_attr(roc, "section_id")
    kv2_sid = _ast_class_attr(kv2, "section_id")

    pi_path = (REPO_ROOT / "mufasa" / "ui_qt"
               / "forms" / "pose_import.py")
    pi_tree = ast.parse(pi_path.read_text())
    pi_cls = _ast_find_class(pi_tree, "PoseImportForm")
    assert pi_cls is not None
    pi_sid = _ast_class_attr(pi_cls, "section_id")

    producer_sids = {
        "RunOutlierCorrectionForm": roc_sid,
        "KalmanV2SmoothingForm": kv2_sid,
        "InterpolateForm": sid,
        "PoseImportForm": pi_sid,
    }
    all_wired = all(
        isinstance(v, ast.Constant) and v.value is not None
        for v in producer_sids.values()
    )
    check(
        "All 4 producers identified in 122dt now declare a "
        "section_id (deferred-producer list is empty)",
        all_wired,
        detail=(", ".join(
            f"{k}={(v.value if isinstance(v, ast.Constant) else None)!r}"
            for k, v in producer_sids.items()
        )),
    )

    # 10. Of the four, exactly THREE now have publish_target_stage
    # (flipped by 122ee — added import_pose to the publisher set
    # via form-level snapshot. Was "exactly 2" after 122ed added
    # interpolate; was "exactly 1" in original 122ec).
    publish_status = {
        "RunOutlierCorrectionForm":
            _ast_class_attr(roc, "publish_target_stage"),
        "KalmanV2SmoothingForm":
            _ast_class_attr(kv2, "publish_target_stage"),
        "InterpolateForm":
            _ast_class_attr(interp_cls, "publish_target_stage"),
        "PoseImportForm":
            _ast_class_attr(pi_cls, "publish_target_stage"),
    }
    publishers = {
        k: v for k, v in publish_status.items() if v is not None
    }
    check(
        "Exactly 3 of the 4 producers have publish_target_stage "
        "(kalman_v2 + interpolate + import_pose; outlier_correction "
        "writes directly to outlier_corrected/ so doesn't need "
        "a symlink-publish — only producer without one)",
        sorted(publishers.keys()) == [
            "InterpolateForm",
            "KalmanV2SmoothingForm",
            "PoseImportForm",
        ],
        detail=(f"publishers: {sorted(publishers.keys())}"),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 11. 122eb state preserved.
    check(
        "122eb state preserved: PoseImportForm.section_id == "
        "'import_pose'",
        isinstance(pi_sid, ast.Constant)
        and pi_sid.value == "import_pose",
    )
    dip_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
               / "data_import_page.py").read_text()
    has_live_title_case = bool(re.search(
        r'add_section\s*\(\s*["\']Import Pose Data["\']',
        dip_src,
    ))
    check(
        "122eb state preserved: Data Import page's add_section "
        "still uses sentence case",
        not has_live_title_case,
    )

    # 12. 122ea state preserved.
    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()
    check(
        "122ea state preserved: ConfigReader has no legacy-shaped "
        "fallback assignments",
        "Paths.ANNOTATED_FRAMES_DIR" not in cr_src
        and "Paths.SINGLE_CLF_VALIDATION" not in cr_src,
    )

    # 13. 122dz state preserved.
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
        "has no legacy_fallback parameter",
        "legacy_fallback" not in helper_params,
    )

    # 14. Parse-clean.
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

    # 15. 122do baseline.
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
        f"smoke_122ec_interpolate_provenance: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
