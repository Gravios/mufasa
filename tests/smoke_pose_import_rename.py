"""
tests/smoke_pose_import_rename.py
=================================

Patch 122w: regression guard for the
'Import pose-estimation data' → 'Import Pose Data' rename and
the updated description on :class:`PoseImportForm`.

Coverage:

1. ``PoseImportForm.title`` is the new shorter label.
2. ``PoseImportForm.description`` mentions both v1 and legacy
   destination directories explicitly.
3. The Data Import page registers a section titled
   ``Import Pose Data`` (no longer the old name).
4. The Data Import page docstring uses the new label and the
   patch note.
5. No stale references to the old label survive in the touched
   files.
"""
from __future__ import annotations

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
    pi_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
              / "pose_import.py").read_text()

    check(
        "PoseImportForm.title is 'Import Pose Data'",
        'title = "Import Pose Data"' in pi_src,
    )
    check(
        "PoseImportForm.title is NOT the old "
        "'Import pose-estimation data'",
        'title = "Import pose-estimation data"' not in pi_src,
    )
    check(
        "PoseImportForm.description mentions v1 'sources/pose/' "
        "destination",
        "sources/pose/" in pi_src,
    )
    check(
        "PoseImportForm.description mentions legacy "
        "'csv/input_csv/' destination",
        "csv/input_csv/" in pi_src,
    )
    check(
        "PoseImportForm.description mentions 'Mufasa' (v1 brand) "
        "rather than 'SimBA' (legacy brand)",
        "Mufasa's multi-index" in pi_src
        and "SimBA's multi-index" not in pi_src,
    )

    dip_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
               / "data_import_page.py").read_text()
    check(
        "data_import_page registers 'Import Pose Data' section",
        "Import Pose Data" in dip_src,
    )
    check(
        "data_import_page no longer REGISTERS the old "
        "'Import pose-estimation data' section "
        "(historical references in the patch note are fine)",
        'add_section("Import pose-estimation data"' not in dip_src,
    )
    check(
        "data_import_page docstring records the 122w patch note",
        "122w" in dip_src
        and "Import Pose Data" in dip_src,
    )

    print(
        f"smoke_pose_import_rename: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
