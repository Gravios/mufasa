"""
tests/smoke_122d1_simba_rois_to_yolo_qt.py
============================================

Patch 122d1: port SimBAROIs2YOLOPopUp to a Qt OperationForm on
the Tools workbench page.

Coverage
--------
1.  Tk popup simba_rois_to_yolo_pop_up.py is gone.
2.  Qt form SimBARoisToYoloForm exists in pose_tools.py.
3.  Form subclasses OperationForm + has build/collect_args/target.
4.  Form imports the correct backend (SimBAROI2Yolo).
5.  tools_page.py wires the form ("SimBA ROIs → YOLO conversion").
6.  pose_tools.py __all__ exports the new form.
7.  SimBA.py: deleted-symbol no longer has an active import.
8.  SimBA.py: convert-pose menu's add_command for the popup is
    commented out.
9.  No file imports from the deleted popup module.
10. mufasa/ui/pop_ups/ count ≤ 73 (was 74 post-122cz).
11. All mufasa/**/*.py files parse cleanly.
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
        "Tk popup simba_rois_to_yolo_pop_up.py is gone",
        not (pkg / "ui" / "pop_ups"
             / "simba_rois_to_yolo_pop_up.py").exists(),
    )

    # 2-4. Form shape
    pt_path = pkg / "ui_qt" / "forms" / "pose_tools.py"
    pt_src = pt_path.read_text()
    pt_tree = ast.parse(pt_src)
    form_cls = next(
        (n for n in pt_tree.body
         if isinstance(n, ast.ClassDef)
         and n.name == "SimBARoisToYoloForm"),
        None,
    )
    check("SimBARoisToYoloForm class in pose_tools.py",
          form_cls is not None)
    if form_cls is not None:
        base_names = [ast.unparse(b) for b in form_cls.bases]
        check("Subclasses OperationForm",
              "OperationForm" in base_names)
        methods = {m.name for m in form_cls.body
                   if isinstance(m, ast.FunctionDef)}
        check(
            "Has build/collect_args/target",
            {"build", "collect_args", "target"} <= methods,
        )

    check(
        "Imports SimBAROI2Yolo backend",
        "SimBAROI2Yolo" in pt_src
        and "simba_roi_to_yolo" in pt_src,
    )

    # 5. tools_page wiring
    tp_src = (pkg / "ui_qt" / "pages" / "tools_page.py").read_text()
    check(
        "tools_page.py imports SimBARoisToYoloForm",
        "SimBARoisToYoloForm" in tp_src,
    )
    check(
        "tools_page.py adds 'SimBA ROIs → YOLO conversion' section",
        "SimBA ROIs → YOLO conversion" in tp_src,
    )

    # 6. __all__
    check(
        "pose_tools.py __all__ exports SimBARoisToYoloForm",
        '"SimBARoisToYoloForm"' in pt_src,
    )

    # 7-8. SimBA.py cleanup
    simba_path = pkg / "SimBA.py"
    if not simba_path.exists():
        # Post-Stage-B (122d5): SimBA.py gone → no
        # active references possible. Both checks pass
        # trivially.
        check("SimBA.py gone (post-Stage-B 122d5) — no active import of SimBAROIs2YOLOPopUp", True)
        check("SimBA.py gone (post-Stage-B 122d5) — no non-commented occurrence of SimBAROIs2YOLOPopUp", True)
    else:
        simba_src = simba_path.read_text()
        simba_tree = ast.parse(simba_src)
        active_import = any(
            isinstance(n, ast.ImportFrom)
            and any(a.name == "SimBAROIs2YOLOPopUp" for a in n.names)
            for n in simba_tree.body
        )
        check(
            "SimBA.py has no active import of SimBAROIs2YOLOPopUp",
            not active_import,
        )
        leaked = []
        for i, line in enumerate(simba_src.split("\n"), 1):
            if line.lstrip().startswith("#"):
                continue
            if "SimBAROIs2YOLOPopUp" in line:
                leaked.append(f"line {i}")
        check(
            "SimBA.py: no non-commented occurrence of the symbol",
            leaked == [],
            detail=", ".join(leaked[:3]),
        )

    # 9. No leftover importers
    leftover = []
    target_mod = "mufasa.ui.pop_ups.simba_rois_to_yolo_pop_up"
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

    # 10. Count
    popups_count = sum(
        1 for _ in (pkg / "ui" / "pop_ups").glob("*.py")
        if _.name != "__init__.py"
    )
    check(
        f"mufasa/ui/pop_ups/ count ≤ 73 (was 74 post-122cz; "
        f"got {popups_count})",
        popups_count <= 73,
    )

    # 11. Parse-clean
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
        f"smoke_122d1_simba_rois_to_yolo_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
