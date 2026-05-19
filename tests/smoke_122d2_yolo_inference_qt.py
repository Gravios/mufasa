"""
tests/smoke_122d2_yolo_inference_qt.py
========================================

Patch 122d2: port YOLOPoseInferencePopUP to YOLOPoseInferenceForm
on the Classifier workbench page (2nd of 3 YOLO ports).

Coverage
--------
1.  Tk popup yolo_inference_popup.py is gone.
2.  Qt form file mufasa/ui_qt/forms/yolo_inference.py exists.
3.  YOLOPoseInferenceForm class defined in that file.
4.  Subclasses OperationForm + has build/collect_args/target.
5.  Imports the two backends (YOLOPoseInference,
    YOLOPoseTrackInference) lazily in target().
6.  classifier_page.py wires the form ("YOLO pose — inference"
    section).
7.  __all__ exports the form.
8.  SimBA.py: no active import of the deleted symbol.
9.  SimBA.py: yolo_tracking_menu add_command for the popup is
    commented out.
10. No file imports from the deleted Tk popup module.
11. Form handles missing CUDA gracefully (collect_args raises
    ValueError — gentle path).
12. mufasa/ui/pop_ups/ count ≤ 72 (was 73 post-122d1).
13. Parse-clean.
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
        "Tk popup yolo_inference_popup.py is gone",
        not (pkg / "ui" / "pop_ups"
             / "yolo_inference_popup.py").exists(),
    )

    # 2. Qt file exists
    yi_path = pkg / "ui_qt" / "forms" / "yolo_inference.py"
    check("yolo_inference.py exists", yi_path.exists())

    if not yi_path.exists():
        print(f"smoke_122d2: {CHECKS_PASSED}/{CHECKS_TOTAL}")
        return 1

    yi_src = yi_path.read_text()
    yi_tree = ast.parse(yi_src)

    # 3-4. Class shape
    form_cls = next(
        (n for n in yi_tree.body
         if isinstance(n, ast.ClassDef)
         and n.name == "YOLOPoseInferenceForm"),
        None,
    )
    check("YOLOPoseInferenceForm defined", form_cls is not None)
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

    # 5. Both backends imported (lazily — text-level check)
    check(
        "Imports YOLOPoseInference backend",
        "YOLOPoseInference" in yi_src
        and "yolo_pose_inference" in yi_src,
    )
    check(
        "Imports YOLOPoseTrackInference backend",
        "YOLOPoseTrackInference" in yi_src
        and "yolo_pose_track_inference" in yi_src,
    )

    # 6. classifier_page wiring
    cp_src = (pkg / "ui_qt" / "pages"
              / "classifier_page.py").read_text()
    check(
        "classifier_page.py imports YOLOPoseInferenceForm",
        "YOLOPoseInferenceForm" in cp_src,
    )
    check(
        "classifier_page.py adds 'YOLO pose — inference' section",
        "YOLO pose — inference" in cp_src
        or "YOLO pose - inference" in cp_src,
    )

    # 7. __all__
    check(
        "yolo_inference.py __all__ exports the form",
        '"YOLOPoseInferenceForm"' in yi_src,
    )

    # 8-9. SimBA.py cleanup
    simba_path = pkg / "SimBA.py"
    if not simba_path.exists():
        # Post-Stage-B (122d5): SimBA.py gone → no
        # active references possible. Both checks pass
        # trivially.
        check("SimBA.py gone (post-Stage-B 122d5) — no active import of YOLOPoseInferencePopUP", True)
        check("SimBA.py gone (post-Stage-B 122d5) — no non-commented occurrence of YOLOPoseInferencePopUP", True)
    else:
        simba_src = simba_path.read_text()
        simba_tree = ast.parse(simba_src)
        active_import = any(
            isinstance(n, ast.ImportFrom)
            and any(a.name == "YOLOPoseInferencePopUP" for a in n.names)
            for n in simba_tree.body
        )
        check(
            "SimBA.py has no active import of YOLOPoseInferencePopUP",
            not active_import,
        )
        leaked = []
        for i, line in enumerate(simba_src.split("\n"), 1):
            if line.lstrip().startswith("#"):
                continue
            if "YOLOPoseInferencePopUP" in line:
                leaked.append(f"line {i}")
        check(
            "SimBA.py: no non-commented occurrence of the symbol",
            leaked == [],
            detail=", ".join(leaked[:3]),
        )

    # 10. No leftover importers
    leftover = []
    target_mod = "mufasa.ui.pop_ups.yolo_inference_popup"
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                    and node.module == target_mod):
                leftover.append(f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "No file imports from the deleted Tk popup module",
        leftover == [],
        detail=", ".join(leftover),
    )

    # 11. Availability check pattern present
    check(
        "Form raises ValueError when CUDA / ultralytics missing "
        "(gentle path via collect_args)",
        "No CUDA GPU detected" in yi_src
        and "ultralytics package not installed" in yi_src,
    )

    # 12. Count
    popups_count = sum(
        1 for _ in (pkg / "ui" / "pop_ups").glob("*.py")
        if _.name != "__init__.py"
    )
    check(
        f"mufasa/ui/pop_ups/ count ≤ 72 (was 73 post-122d1; "
        f"got {popups_count})",
        popups_count <= 72,
    )

    # 13. Parse-clean
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
        f"smoke_122d2_yolo_inference_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
