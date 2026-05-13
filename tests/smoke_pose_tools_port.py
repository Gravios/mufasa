"""
tests/smoke_pose_tools_port.py
==============================

Patch 122ac: regression guard for the two pose-tool ports on
the Tools workbench page.

Coverage:

1. ``PoseReorganizerForm`` exists, extends OperationForm, has
   the expected widget surface, and dispatches to
   ``KeypointReorganizer``.
2. ``SLEAPToYoloForm`` exists, extends OperationForm, has the
   expected widget surface, and dispatches to ``Sleap2Yolo``.
3. Both legacy popup files
   (``pose_reorganizer_pop_up.py`` and
   ``sleap_csv_predictions_to_yolo_popup.py``) are deleted.
4. SimBA.py no longer imports either legacy class and no
   longer wires either menu entry.
5. Tools page imports + registers both new forms.

AST-only — Qt + the SLEAP/DLC backends bring heavy deps not
available in the sandbox.
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


def _find_class(tree: ast.Module, name: str):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        n.name: n for n in cls.body
        if isinstance(n, ast.FunctionDef)
    }


def main() -> int:
    # ==================================================================
    # 1. PoseReorganizerForm
    # ==================================================================
    pt_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "pose_tools.py")
    check("pose_tools.py exists", pt_path.is_file())
    pt_src = pt_path.read_text()
    pt_tree = ast.parse(pt_src)

    cls = _find_class(pt_tree, "PoseReorganizerForm")
    check("PoseReorganizerForm class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        bases = [
            (b.id if isinstance(b, ast.Name)
             else getattr(b, "attr", ""))
            for b in cls.bases
        ]
        check(
            "PoseReorganizerForm extends OperationForm",
            "OperationForm" in bases,
        )
        for attr in (
            "self.data_folder_edit", "self.tool_cb",
            "self.format_cb", "self.reorder_group",
        ):
            check(
                f"PoseReorganizerForm sets {attr}",
                attr in class_src,
            )
        methods = _methods(cls)
        check(
            "PoseReorganizerForm has _load_order helper "
            "(stage-1 → stage-2)",
            "_load_order" in methods,
        )
        check(
            "PoseReorganizerForm has _populate_reorder_panel "
            "helper",
            "_populate_reorder_panel" in methods,
        )
        if "_load_order" in methods:
            lo_src = ast.unparse(methods["_load_order"])
            check(
                "_load_order constructs KeypointReorganizer",
                "KeypointReorganizer(" in lo_src,
            )
        if "target" in methods:
            t_src = ast.unparse(methods["target"])
            check(
                "PoseReorganizerForm.target calls "
                "self._reorganizer.run(bp_lst=, animal_list=)",
                "self._reorganizer.run(" in t_src
                and "bp_lst=" in t_src
                and "animal_list=" in t_src,
            )

    # ==================================================================
    # 2. SLEAPToYoloForm
    # ==================================================================
    cls = _find_class(pt_tree, "SLEAPToYoloForm")
    check("SLEAPToYoloForm class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        for attr in (
            "self.sleap_dir_edit", "self.video_dir_edit",
            "self.save_dir_edit", "self.frames_per_video",
            "self.train_size", "self.threshold", "self.padding",
            "self.greyscale", "self.clahe", "self.single_id",
            "self.verbose",
        ):
            check(
                f"SLEAPToYoloForm sets {attr}",
                attr in class_src,
            )
        methods = _methods(cls)
        if "target" in methods:
            t_src = ast.unparse(methods["target"])
            check(
                "SLEAPToYoloForm.target constructs Sleap2Yolo",
                "Sleap2Yolo(" in t_src,
            )
            check(
                "SLEAPToYoloForm.target calls runner.run()",
                "runner.run()" in t_src
                or ".run()" in t_src,
            )

    # ==================================================================
    # 3. Legacy popup files deleted
    # ==================================================================
    legacy_files = [
        REPO_ROOT / "mufasa" / "ui" / "pop_ups"
        / "pose_reorganizer_pop_up.py",
        REPO_ROOT / "mufasa" / "ui" / "pop_ups"
        / "sleap_csv_predictions_to_yolo_popup.py",
    ]
    for f in legacy_files:
        check(
            f"legacy file deleted: {f.name}",
            not f.exists(),
        )

    # ==================================================================
    # 4. SimBA.py wiring scrubbed
    # ==================================================================
    simba_src = (REPO_ROOT / "mufasa" / "SimBA.py").read_text()
    for cls_name in ("PoseReorganizerPopUp", "SLEAPcsvInference2Yolo"):
        leaked = any(
            cls_name in line and not line.lstrip().startswith("#")
            for line in simba_src.splitlines()
        )
        check(
            f"SimBA.py has no code-level reference to {cls_name}",
            not leaked,
        )

    # ==================================================================
    # 5. Tools page registration
    # ==================================================================
    tools_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
                 / "tools_page.py").read_text()
    check(
        "Tools page imports PoseReorganizerForm",
        "PoseReorganizerForm" in tools_src,
    )
    check(
        "Tools page imports SLEAPToYoloForm",
        "SLEAPToYoloForm" in tools_src,
    )
    check(
        "Tools page registers 'Re-order pose keypoints' section",
        '"Re-order pose keypoints"' in tools_src,
    )
    check(
        "Tools page registers 'SLEAP → YOLO conversion' section",
        '"SLEAP → YOLO conversion"' in tools_src,
    )
    check(
        "Tools page docstring records the 122ac note",
        "122ac" in tools_src,
    )

    print(
        f"smoke_pose_tools_port: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
