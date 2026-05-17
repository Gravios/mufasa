"""
tests/smoke_122ce_drop_bodyparts_rewire.py
============================================

Patch 122ce: DropBodypartsForm rewired to KeypointRemover.

Pre-122ce the form was entirely broken — tried
`mufasa.data_processors.keypoint_dropper.KeyPointRemover(
config_path, body_parts, copy_originals)` which doesn't exist.
The 122bz audit found the real backend at
`mufasa.pose_processors.remove_keypoints.KeypointRemover(
data_folder, pose_tool, file_format)` + `.run(animal_names,
bp_to_remove_list)` — completely different shape.

Post-122ce the form:

* Drops the misleading `copy_originals` checkbox — backend always
  writes to a new `Reorganized_bp_<datetime>` subdirectory so
  originals are never overwritten regardless.
* Adds a `data_folder` QLineEdit + Browse, auto-populating
  `<project>/csv/input_csv` when that path exists.
* Adds a status label showing inferred pose_tool + file_format.
* Reads project metadata via project_metadata_from_config to
  infer pose_tool (DLC if animal_count==1, else maDLC) and
  file_format.
* Transforms the form's `[(animal, bp), ...]` selection into the
  backend's split animal_names + bp_to_remove_list lists.
* Dispatch calls KeypointRemover(data_folder, pose_tool,
  file_format).run(animal_names, bp_to_remove_list).

Coverage
--------
1. DropBodypartsForm `copy_originals` QCheckBox is gone.
2. New `data_folder_edit` QLineEdit defined.
3. New `_browse_data_folder` slot defined.
4. New `_default_data_folder` helper returns csv/input_csv path.
5. New `_refresh_defaults` slot pre-fills data_folder + status.
6. `collect_args()` does NOT return `copy_originals` key.
7. `collect_args()` returns `data_folder` key.
8. `collect_args()` validates data_folder presence and existence.
9. `target()` imports KeypointRemover from
   mufasa.pose_processors.remove_keypoints.
10. `target()` imports project_metadata_from_config.
11. `target()` does NOT reference the legacy
    `mufasa.data_processors.keypoint_dropper` path.
12. `target()` does NOT raise NotImplementedError.
13. `target()` infers pose_tool from animal_count.
14. `target()` transforms the selection: separate animal_names
    list and bp_to_remove_list list.
15. `target()` calls .run(animal_names=..., bp_to_remove_list=...).
16. qt_form_runtime_gaps.md §2d marks DropBodypartsForm FIXED.
17. backend_audit.md §4b item 7 marks DropBodyparts redesign DONE.
18. All mufasa/**/*.py files parse cleanly.
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
    src = (pkg / "ui_qt" / "forms" / "pose_cleanup.py").read_text()
    tree = ast.parse(src)

    form_cls = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "DropBodypartsForm"):
            form_cls = node
            break
    check("DropBodypartsForm class found", form_cls is not None)
    if form_cls is None:
        return 1
    cls_src = ast.unparse(form_cls)

    # ==================================================================
    # New widgets / slots / helpers
    # ==================================================================
    check(
        "copy_originals QCheckBox removed",
        "self.copy_originals" not in cls_src,
    )
    check(
        "new `data_folder_edit` QLineEdit defined",
        "self.data_folder_edit = QLineEdit" in cls_src,
    )
    check(
        "new `_browse_data_folder` slot defined",
        "def _browse_data_folder" in cls_src,
    )
    check(
        "new `_default_data_folder` helper returns csv/input_csv path",
        "def _default_data_folder" in cls_src
        and "csv" in cls_src and "input_csv" in cls_src,
    )
    check(
        "new `_refresh_defaults` slot defined",
        "def _refresh_defaults" in cls_src,
    )

    # ==================================================================
    # collect_args shape change
    # ==================================================================
    collect_args_src = ""
    target_src = ""
    for stmt in form_cls.body:
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name == "collect_args":
                collect_args_src = ast.unparse(stmt)
            elif stmt.name == "target":
                target_src = ast.unparse(stmt)

    check(
        "`collect_args` no longer returns `copy_originals` key",
        "'copy_originals'" not in collect_args_src,
    )
    check(
        "`collect_args` returns `data_folder` key",
        "'data_folder'" in collect_args_src,
    )
    check(
        "`collect_args` validates data_folder (raises ValueError; isdir)",
        "ValueError" in collect_args_src
        and "isdir" in collect_args_src,
    )

    # ==================================================================
    # target() rewired to KeypointRemover
    # ==================================================================
    check(
        "`target` imports KeypointRemover from "
        "mufasa.pose_processors.remove_keypoints",
        "from mufasa.pose_processors.remove_keypoints import" in target_src
        or "mufasa.pose_processors.remove_keypoints" in target_src,
    )
    check(
        "`target` imports project_metadata_from_config",
        "project_metadata_from_config" in target_src,
    )
    check(
        "`target` no longer references the legacy "
        "`mufasa.data_processors.keypoint_dropper` path",
        "mufasa.data_processors.keypoint_dropper" not in target_src
        and "keypoint_dropper" not in target_src,
    )
    check(
        "`target` no longer raises NotImplementedError",
        "NotImplementedError" not in target_src,
    )
    check(
        "`target` infers pose_tool from animal_count",
        "animal_count" in target_src
        and ("DLC" in target_src and "maDLC" in target_src),
    )
    check(
        "`target` transforms selection: separate animal_names list "
        "and bp_to_remove_list list",
        "animal_names = [" in target_src
        and "bp_to_remove_list = [" in target_src,
    )
    check(
        "`target` calls .run(animal_names=..., bp_to_remove_list=...)",
        ".run(animal_names=" in target_src
        and "bp_to_remove_list=" in target_src,
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    gaps = (REPO_ROOT / "docs" / "qt_form_runtime_gaps.md").read_text()
    check(
        "qt_form_runtime_gaps.md §2d marks DropBodypartsForm FIXED in 122ce",
        "FIXED in patch 122ce" in gaps,
    )
    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4b item 7 marks DropBodyparts redesign DONE in 122ce",
        "DONE in patch 122ce" in audit,
    )

    # ==================================================================
    # All files parse cleanly
    # ==================================================================
    parse_errors: list[str] = []
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
        f"smoke_122ce_drop_bodyparts_rewire: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
