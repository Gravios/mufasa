"""
tests/smoke_roi_definitions_description.py
==========================================

Patch 122y: regression guard for the ROIManageForm description
rework. Honest about both v1 and legacy directory layouts.

Coverage:

1. Form description mentions BOTH ``sources/videos/`` (v1) and
   ``videos/`` (legacy).
2. Form description mentions
   ``logs/measures/ROI_definitions.h5`` (the persistence path
   shared by both layouts).
3. Draw-action note mentions both layouts.
4. The patch note appears in the class docstring.
5. Stale absolutes — bare ``project_folder/videos`` mentions
   that don't qualify with "(legacy)" — gone from the file.
"""
from __future__ import annotations

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
    src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
           / "roi.py").read_text()

    # Isolate the ROIManageForm class — its description is what we
    # care about, not stray strings elsewhere in the file.
    import ast
    tree = ast.parse(src)
    rmf = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.ClassDef)
                and n.name == "ROIManageForm"):
            rmf = n
            break
    check("ROIManageForm class defined", rmf is not None)

    if rmf is not None:
        class_src = ast.unparse(rmf)

        # ----- Description content -----
        check(
            "ROIManageForm description mentions "
            "'sources/videos/' (v1 layout)",
            "sources/videos/" in class_src,
        )
        check(
            "ROIManageForm description mentions "
            "'videos/' (legacy layout)",
            "videos/" in class_src,
        )
        check(
            "ROIManageForm description mentions "
            "'ROI_definitions.h5' persistence file",
            "ROI_definitions.h5" in class_src,
        )
        check(
            "ROIManageForm description mentions "
            "'logs/measures/' subtree",
            "logs/measures/" in class_src,
        )
        check(
            "ROIManageForm description distinguishes v1 vs legacy "
            "explicitly (parenthetical)",
            ("(v1)" in class_src or "v1)" in class_src)
            and ("(legacy)" in class_src or "legacy)" in class_src),
        )

        # ----- Patch note in docstring -----
        check(
            "ROIManageForm class docstring carries the 122y note",
            "122y" in class_src,
        )

        # ----- No stale absolute reference -----
        # Plain 'project_folder/videos' (no v1/legacy qualifier
        # nearby) was the misleading phrasing the patch removes.
        # The class shouldn't carry it anymore, even if the
        # 'project_folder' word appears elsewhere as context.
        check(
            "ROIManageForm no longer uses unqualified "
            "'project_folder/videos' phrasing",
            "project_folder/videos" not in class_src,
        )

    print(
        f"smoke_roi_definitions_description: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
