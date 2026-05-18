"""
tests/smoke_122cy_stage_b_checklist.py
========================================

Patch 122cy: pre-Stage-B checklist sweep.

Doc-only patch. Audits each of the 75 popups against
`mufasa/ui_qt/forms,pages,dialogs/` to verify whether a Qt
counterpart exists. Results categorize each popup into:
- Covered (69 popups; safe to bulk-delete in Stage B)
- Hard-drop (2 popups; admin/cosmetic)
- Workflow-blocking gap (1 popup)
- Feature-decision-required (3 popups; non-blocking)

This test pins the specific findings as regression guards.
If, e.g., a future patch adds a Qt counterpart for one of the
4 gap popups, the test fails and the checklist gets refreshed.

Coverage
--------
1.  docs/stage_b_checklist.md exists.
2.  docs/simba_death_cascade.md cross-references the checklist.
3.  The 1 blocking gap popup still exists (deletion blocker).
4.  The 1 blocking gap popup's config-key dependency is still
    in the Qt AnalysisForm (the dependency surface).
5.  No Qt code-level reference to the blocking gap popup's
    settings UI (the gap is still real).
6.  The 3 non-blocking gap popups still exist.
7.  The 3 non-blocking gap popups don't have Qt form wirings
    (gap still real).
8.  The 2 hard-drop popups still exist (deletion is planned,
    not yet executed).
9.  Spot-check covered popups: 5 sentinel cases.
10. All mufasa/**/*.py files parse cleanly.
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


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # 1. Checklist doc exists
    checklist = REPO_ROOT / "docs" / "stage_b_checklist.md"
    check(
        "docs/stage_b_checklist.md exists",
        checklist.exists(),
    )

    # 2. Cascade doc cross-references checklist
    cascade = (REPO_ROOT / "docs"
               / "simba_death_cascade.md").read_text()
    check(
        "simba_death_cascade.md cross-references the checklist",
        "stage_b_checklist.md" in cascade,
    )

    # 3. Blocking gap popup still exists
    blocking = (pkg / "ui" / "pop_ups"
                / "direction_animal_to_bodypart_settings_pop_up.py")
    check(
        "Blocking gap popup still exists "
        "(direction_animal_to_bodypart_settings_pop_up.py)",
        blocking.exists(),
    )

    # 4. Qt AnalysisForm has the directing-to-bodypart route
    # (the dependency that the popup writes settings for)
    analysis_form = (pkg / "ui_qt" / "forms" / "analysis.py").read_text()
    check(
        "Qt AnalysisForm has the 'Directing toward body-part' "
        "route — the dependency requiring the gap popup's settings",
        "Directing toward body-part" in analysis_form
        and "DirectingAnimalsToBodyPartAnalyzer" in analysis_form,
    )

    # 5. No Qt code-level reference to the blocking popup's
    # settings UI (gap still real)
    qt_code = ""
    for d in [pkg / "ui_qt" / "forms",
              pkg / "ui_qt" / "pages",
              pkg / "ui_qt" / "dialogs"]:
        for f in d.glob("*.py"):
            for line in f.read_text().split("\n"):
                stripped = line.lstrip()
                # Skip docstring/comment lines for code-level check
                if stripped.startswith(("#", "*", ":class:")):
                    continue
                qt_code += line + "\n"
    bodypart_settings_in_qt = bool(re.search(
        r"DirectionAnimalToBodyPartSettings|"
        r"directing_settings.*dialog|"
        r"bodypart_direction.*dialog",
        qt_code, re.IGNORECASE))
    check(
        "No Qt code-level reference to DirectionAnimalToBodyPart"
        "Settings dialog (gap is still real; this is the "
        "regression guard)",
        not bodypart_settings_in_qt,
    )

    # 6. Non-blocking gap popups still exist
    gap_popups = [
        "simba_rois_to_yolo_pop_up.py",
        "yolo_inference_popup.py",
        "yolo_pose_train_popup.py",
    ]
    for popup in gap_popups:
        check(
            f"Non-blocking gap popup still exists ({popup})",
            (pkg / "ui" / "pop_ups" / popup).exists(),
        )

    # 7. Non-blocking gap popups don't have Qt form wirings
    # Check by looking for their class names in Qt page wirings
    pages_src = ""
    for f in (pkg / "ui_qt" / "pages").glob("*.py"):
        pages_src += f.read_text() + "\n"
    for popup_file, primary_class, qt_clue in [
        ("simba_rois_to_yolo_pop_up.py", "SimBAROIs2YOLO",
         "SimBARoisToYoloForm"),
        ("yolo_inference_popup.py", "YOLOPoseInference",
         "YOLOPoseInferenceForm"),
        ("yolo_pose_train_popup.py", "YOLOPoseTrain",
         "YOLOPoseTrainForm"),
    ]:
        check(
            f"{popup_file}: no Qt form wired in pages "
            f"({qt_clue} absent — gap is real)",
            qt_clue not in pages_src,
        )

    # 8. Hard-drop popups still exist (planned, not executed)
    hard_drops = ["about_simba_pop_up.py", "splash_popup.py"]
    for popup in hard_drops:
        check(
            f"Hard-drop popup still exists ({popup})",
            (pkg / "ui" / "pop_ups" / popup).exists(),
        )

    # 9. Spot-check covered popups — these should still resolve
    # to Qt counterparts as listed in the checklist
    covered_sentinels = [
        # (popup_name, sentinel string that should appear in Qt)
        ("delete_all_rois_pop_up.py", "ROIManageForm"),
        ("kleinberg_pop_up.py", "KleinbergForm"),
        ("smoothing_popup.py", "SmoothingForm"),
        ("distance_timebins_popup.py", "Distance by time bins"),
        ("third_party_annotator_appender_pop_up.py",
         "ThirdPartyAppenderForm"),
    ]
    for popup_name, qt_sentinel in covered_sentinels:
        # Read full Qt source (including docstrings — sentinels
        # may be labels)
        qt_all = ""
        for d in [pkg / "ui_qt" / "forms",
                  pkg / "ui_qt" / "pages",
                  pkg / "ui_qt" / "dialogs"]:
            for f in d.glob("*.py"):
                qt_all += f.read_text() + "\n"
        check(
            f"Covered popup {popup_name}: Qt counterpart sentinel "
            f"'{qt_sentinel}' present",
            qt_sentinel in qt_all,
        )

    # 10. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py files parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122cy_stage_b_checklist: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
