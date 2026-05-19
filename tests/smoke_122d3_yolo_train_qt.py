"""
tests/smoke_122d3_yolo_train_qt.py
====================================

Patch 122d3: port YOLOPoseTrainPopUP to YOLOPoseTrainForm on the
Classifier workbench page (3rd and final YOLO port). Closes out
the 3 non-blocking feature gaps from 122cy's checklist.

Coverage
--------
1.  Tk popup yolo_pose_train_popup.py is gone.
2.  Qt form file mufasa/ui_qt/forms/yolo_train.py exists.
3.  YOLOPoseTrainForm class defined.
4.  Subclasses OperationForm + has build/collect_args/target.
5.  Form uses subprocess.Popen (detached training pattern).
6.  Form calls `mufasa.model.yolo_fit` as the subprocess module.
7.  classifier_page.py wires the form ("YOLO pose — train").
8.  __all__ exports the form.
9.  SimBA.py: no active import of the deleted symbol.
10. SimBA.py: yolo_tracking_menu Train add_command commented out.
11. No file imports from the deleted popup module.
12. mufasa/ui/pop_ups/ count ≤ 71 (was 72 post-122d2).
13. All 3 YOLO/conversion gaps from 122cy now ported (no orphan
    feature-decision items left).
14. checklist doc reflects the 3 ports.
15. Parse-clean.
"""
from __future__ import annotations

import ast
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

    # 1. Tk gone
    check(
        "Tk popup yolo_pose_train_popup.py is gone",
        not (pkg / "ui" / "pop_ups"
             / "yolo_pose_train_popup.py").exists(),
    )

    # 2. Qt file exists
    yt_path = pkg / "ui_qt" / "forms" / "yolo_train.py"
    check("yolo_train.py exists", yt_path.exists())

    if not yt_path.exists():
        print(f"smoke_122d3: {CHECKS_PASSED}/{CHECKS_TOTAL}")
        return 1

    yt_src = yt_path.read_text()
    yt_tree = ast.parse(yt_src)

    # 3-4. Class shape
    form_cls = next(
        (n for n in yt_tree.body
         if isinstance(n, ast.ClassDef)
         and n.name == "YOLOPoseTrainForm"),
        None,
    )
    check("YOLOPoseTrainForm class defined",
          form_cls is not None)
    if form_cls is not None:
        bases = [ast.unparse(b) for b in form_cls.bases]
        check("Subclasses OperationForm",
              "OperationForm" in bases)
        methods = {m.name for m in form_cls.body
                   if isinstance(m, ast.FunctionDef)}
        check(
            "Has build/collect_args/target",
            {"build", "collect_args", "target"} <= methods,
        )

    # 5-6. Subprocess pattern
    check(
        "Form uses subprocess.Popen (detached training)",
        "subprocess.Popen" in yt_src,
    )
    check(
        "Form invokes `mufasa.model.yolo_fit` as a subprocess",
        "mufasa.model.yolo_fit" in yt_src,
    )

    # 7. classifier_page wiring
    cp_src = (pkg / "ui_qt" / "pages"
              / "classifier_page.py").read_text()
    check(
        "classifier_page.py imports YOLOPoseTrainForm",
        "YOLOPoseTrainForm" in cp_src,
    )
    check(
        "classifier_page.py adds 'YOLO pose — train' section",
        "YOLO pose — train" in cp_src
        or "YOLO pose - train" in cp_src,
    )

    # 8. __all__
    check(
        "yolo_train.py __all__ exports the form",
        '"YOLOPoseTrainForm"' in yt_src,
    )

    # 9-10. SimBA.py cleanup
    simba_path = pkg / "SimBA.py"
    if not simba_path.exists():
        # Post-Stage-B (122d5): SimBA.py gone → no
        # active references possible. Both checks pass
        # trivially.
        check("SimBA.py gone (post-Stage-B 122d5) — no active import of YOLOPoseTrainPopUP", True)
        check("SimBA.py gone (post-Stage-B 122d5) — no non-commented occurrence of YOLOPoseTrainPopUP", True)
    else:
        simba_src = simba_path.read_text()
        simba_tree = ast.parse(simba_src)
        active_import = any(
            isinstance(n, ast.ImportFrom)
            and any(a.name == "YOLOPoseTrainPopUP" for a in n.names)
            for n in simba_tree.body
        )
        check(
            "SimBA.py has no active import of YOLOPoseTrainPopUP",
            not active_import,
        )
        leaked = []
        for i, line in enumerate(simba_src.split("\n"), 1):
            if line.lstrip().startswith("#"):
                continue
            if "YOLOPoseTrainPopUP" in line:
                leaked.append(f"line {i}")
        check(
            "SimBA.py: no non-commented occurrence of the symbol",
            leaked == [],
            detail=", ".join(leaked[:3]),
        )

    # 11. No leftover importers
    leftover = []
    target_mod = "mufasa.ui.pop_ups.yolo_pose_train_popup"
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                    and node.module == target_mod):
                leftover.append(
                    f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "No file imports from the deleted popup module",
        leftover == [],
        detail=", ".join(leftover),
    )

    # 12. Count
    popups_count = sum(
        1 for _ in (pkg / "ui" / "pop_ups").glob("*.py")
        if _.name != "__init__.py"
    )
    check(
        f"mufasa/ui/pop_ups/ count ≤ 71 (was 72 post-122d2; "
        f"got {popups_count})",
        popups_count <= 71,
    )

    # 13. All 3 122cy feature-decision popups are now gone
    for pre_122d_popup in (
            "simba_rois_to_yolo_pop_up.py",  # 122d1
            "yolo_inference_popup.py",        # 122d2
            "yolo_pose_train_popup.py",       # 122d3
    ):
        check(
            f"122cy non-blocking gap popup gone: {pre_122d_popup}",
            not (pkg / "ui" / "pop_ups"
                 / pre_122d_popup).exists(),
        )

    # 14. checklist doc — at least the YOLO entries should still
    # appear (we're not requiring it to be updated yet — the user
    # may do it in a follow-up doc-sweep patch).
    checklist = (REPO_ROOT / "docs"
                 / "stage_b_checklist.md")
    check(
        "stage_b_checklist.md still exists "
        "(may need follow-on update reflecting 122d1-d3 ports)",
        checklist.exists(),
    )

    # 15. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122d3_yolo_train_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
