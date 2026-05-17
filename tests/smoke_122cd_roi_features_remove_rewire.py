"""
tests/smoke_122cd_roi_features_remove_rewire.py
=================================================

Patch 122cd: ROIFeaturesForm Remove action redesigned + rewired.

Pre-122cd the Remove action raised NotImplementedError when it
couldn't find `mufasa.roi_tools.remove_roi_features` as a module.
The 122bz audit found the real backend at
`mufasa.mixins.config_reader.ConfigReader.remove_roi_features` —
a method on the class, not a free function — and it requires a
`data_dir` parameter the form didn't surface.

Post-122cd:

* New `data_dir_edit` QLineEdit + Browse button on the form.
* Field enabled only when the Remove action is selected
  (other actions don't need a data_dir).
* `_on_action_changed` auto-populates the default
  `<project>/csv/features_extracted` when switching to Remove
  if that directory exists.
* `_browse_data_dir` starts the QFileDialog at the project's
  csv dir if available; otherwise the project root.
* `collect_args()` raises ValueError if Remove is selected
  without a data_dir, or if the directory doesn't exist.
* `target()` instantiates `ConfigReader(config_path, ...,
  read_video_info=False, create_logger=False)` and calls
  `.remove_roi_features(data_dir=data_dir)`.

Coverage
--------
1. ROIFeaturesForm defines new `data_dir_edit` QLineEdit.
2. ROIFeaturesForm defines `_browse_data_dir` slot.
3. `_on_action_changed` toggles `data_dir_edit.setEnabled`.
4. `_on_action_changed` auto-populates default for Remove.
5. `collect_args()` validates data_dir presence and existence.
6. `target()` calls `ConfigReader.remove_roi_features(data_dir=...)`.
7. `target()` no longer raises NotImplementedError or references
   the legacy `mufasa.roi_tools.remove_roi_features` module path.
8. `target()` uses ConfigReader's lightweight init flags
   (read_video_info=False, create_logger=False) to avoid heavy
   startup work.
9. Imports include QLineEdit, QHBoxLayout, QPushButton.
10. qt_form_runtime_gaps.md §2e marks ROIFeaturesForm as FIXED.
11. backend_audit.md §4b item 5 marks Remove redesign DONE.
12. All mufasa/**/*.py files parse cleanly.
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
    src = (pkg / "ui_qt" / "forms" / "roi.py").read_text()
    tree = ast.parse(src)

    form_cls = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "ROIFeaturesForm"):
            form_cls = node
            break
    check("ROIFeaturesForm class found", form_cls is not None)
    if form_cls is None:
        return 1
    cls_src = ast.unparse(form_cls)

    # ==================================================================
    # New widgets / slots
    # ==================================================================
    check(
        "ROIFeaturesForm defines `data_dir_edit` QLineEdit",
        "self.data_dir_edit = QLineEdit" in cls_src,
    )
    check(
        "ROIFeaturesForm defines `_browse_data_dir` slot",
        "def _browse_data_dir" in cls_src,
    )

    # ==================================================================
    # _on_action_changed toggles enable + auto-populates
    # ==================================================================
    on_change_src = ""
    for stmt in form_cls.body:
        if (isinstance(stmt, ast.FunctionDef)
                and stmt.name == "_on_action_changed"):
            on_change_src = ast.unparse(stmt)
            break
    check(
        "`_on_action_changed` toggles data_dir_edit.setEnabled",
        "data_dir_edit.setEnabled" in on_change_src,
    )
    check(
        "`_on_action_changed` auto-populates features_extracted default",
        "features_extracted" in on_change_src,
    )

    # ==================================================================
    # collect_args validates
    # ==================================================================
    collect_args_src = ""
    for stmt in form_cls.body:
        if (isinstance(stmt, ast.FunctionDef)
                and stmt.name == "collect_args"):
            collect_args_src = ast.unparse(stmt)
            break
    check(
        "`collect_args` validates data_dir for the Remove action",
        "data_dir" in collect_args_src
        and "ValueError" in collect_args_src
        and "isdir" in collect_args_src,
    )

    # ==================================================================
    # target() rewired to ConfigReader
    # ==================================================================
    target_src = ""
    for stmt in form_cls.body:
        if (isinstance(stmt, ast.FunctionDef)
                and stmt.name == "target"):
            target_src = ast.unparse(stmt)
            break
    check(
        "`target` imports ConfigReader from mufasa.mixins.config_reader",
        "from mufasa.mixins.config_reader import ConfigReader" in target_src
        or "mufasa.mixins.config_reader" in target_src,
    )
    check(
        "`target` calls .remove_roi_features(data_dir=...)",
        ".remove_roi_features(data_dir=" in target_src,
    )
    check(
        "`target` no longer raises NotImplementedError for Remove",
        "NotImplementedError" not in target_src,
    )
    check(
        "`target` no longer references the legacy "
        "`mufasa.roi_tools.remove_roi_features` path",
        "mufasa.roi_tools.remove_roi_features" not in target_src
        and "from mufasa.roi_tools import remove_roi_features" not in target_src,
    )
    check(
        "`target` uses lightweight ConfigReader init flags "
        "(read_video_info=False, create_logger=False)",
        "read_video_info=False" in target_src
        and "create_logger=False" in target_src,
    )

    # ==================================================================
    # Imports updated
    # ==================================================================
    check(
        "roi.py imports include QLineEdit",
        "QLineEdit" in src,
    )
    check(
        "roi.py imports include QHBoxLayout",
        "QHBoxLayout" in src,
    )
    check(
        "roi.py imports include QPushButton",
        "QPushButton" in src,
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    gaps = (REPO_ROOT / "docs" / "qt_form_runtime_gaps.md").read_text()
    check(
        "qt_form_runtime_gaps.md §2e marks ROIFeaturesForm FIXED in 122cd",
        "FIXED in patch 122cd" in gaps,
    )
    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4b item 5 marks Remove redesign DONE in 122cd",
        "DONE in patch 122cd" in audit,
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
        f"smoke_122cd_roi_features_remove_rewire: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
