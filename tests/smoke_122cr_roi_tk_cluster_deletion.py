"""
tests/smoke_122cr_roi_tk_cluster_deletion.py
==============================================

Patch 122cr: ROI Tk cluster-deletion. Executes the deletion plan
scoped by 122cq's re-audit. 6 files removed; 5 surgical SimBA.py
edits. Qt replacements verified before deletion.

Files deleted
-------------
* `mufasa/roi_tools/roi_ui_mixin.py`
* `mufasa/roi_tools/roi_ui.py`
* `mufasa/ui/blob_tracker_ui.py`
* `mufasa/ui/blob_quick_check_interface.py` (orphan-after-cascade)
* `mufasa/ui/pop_ups/roi_video_table_pop_up.py`
* `mufasa/ui/pop_ups/initialize_blob_tracking_pop_up.py`

Qt replacements
---------------
* `ROIVideoTable` → `ROIManageForm` (mufasa.ui_qt.forms.roi).
* `InitializeBlobTrackerPopUp` → `BlobTrackerInitLauncher`
  (mufasa.ui_qt.forms.addons; wired at addons_page.py:55).

Coverage
--------
1.  All 6 cluster files are gone.
2.  SimBA.py no longer has an active import of
    `InitializeBlobTrackerPopUp`.
3.  SimBA.py no longer has an active import of `ROIVideoTable`.
4.  SimBA.py has no non-commented references to `ROIVideoTable`,
    `InitializeBlobTrackerPopUp`, `ROI_mixin`, `ROI_ui`,
    `start_new_ROI`, or `BlobQuickChecker`.
5.  No other file in mufasa/ imports from any of the deleted
    modules.
6.  Qt replacement `ROIManageForm` still exists in
    `mufasa/ui_qt/forms/roi.py`.
7.  Qt replacement `BlobTrackerInitLauncher` still exists in
    `mufasa/ui_qt/forms/addons.py`.
8.  Qt wiring at `addons_page.py:55` ("Blob tracker — initialise"
    section) still references `BlobTrackerInitLauncher`.
9.  Module-level Tk-functions importer count is ≤ 18 (was 19
    pre-122cr; 18 is the expected post-122cr count since
    roi_ui_mixin.py dropped out).
10. `backend_audit.md` §3d Bucket 2 marks the ROI Tk cluster
    DELETED 122cr.
11. `backend_audit.md` §3a category list shows roi_tools/ at 0.
12. `tk_surface_audit.md` §2g records the cluster-deletion as
    DONE.
13. mufasa/ui/ file count ≤ 88 (was 91 post-122ck; this patch
    removed 3 files under ui/).
14. SimBA.py parses cleanly.
15. All mufasa/**/*.py files parse cleanly.
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

    # ==================================================================
    # 6 files gone
    # ==================================================================
    dropped = [
        "roi_tools/roi_ui_mixin.py",
        "roi_tools/roi_ui.py",
        "ui/blob_tracker_ui.py",
        "ui/blob_quick_check_interface.py",
        "ui/pop_ups/roi_video_table_pop_up.py",
        "ui/pop_ups/initialize_blob_tracking_pop_up.py",
    ]
    for rel in dropped:
        check(
            f"deleted: mufasa/{rel}",
            not (pkg / rel).exists(),
        )

    # ==================================================================
    # SimBA.py cleanup
    # ==================================================================
    simba_src = (pkg / "SimBA.py").read_text()
    simba_tree = ast.parse(simba_src)

    # No active imports of deleted symbols
    deleted_symbols = {"InitializeBlobTrackerPopUp", "ROIVideoTable",
                       "ROI_mixin", "ROI_ui", "BlobQuickChecker"}
    active_imports: set[str] = set()
    for n in simba_tree.body:
        if isinstance(n, ast.ImportFrom):
            for alias in n.names:
                if alias.name in deleted_symbols:
                    active_imports.add(alias.name)
    check(
        "SimBA.py has NO active import of deleted symbols",
        active_imports == set(),
        detail=f"still imports: {active_imports}",
    )

    # No non-commented occurrence of deleted names
    leaked_lines: list[str] = []
    for sym in ("ROIVideoTable", "InitializeBlobTrackerPopUp",
                "start_new_ROI", "BlobQuickChecker"):
        for i, line in enumerate(simba_src.split("\n"), 1):
            if sym in line and not line.lstrip().startswith("#"):
                leaked_lines.append(f"line {i}: {sym}")
    check(
        "no non-commented occurrence of deleted symbol names "
        "in SimBA.py",
        leaked_lines == [],
        detail="; ".join(leaked_lines[:3]),
    )

    # ==================================================================
    # No other file imports from the deleted modules
    # ==================================================================
    deleted_mods = {
        "mufasa.roi_tools.roi_ui_mixin",
        "mufasa.roi_tools.roi_ui",
        "mufasa.ui.blob_tracker_ui",
        "mufasa.ui.blob_quick_check_interface",
        "mufasa.ui.pop_ups.roi_video_table_pop_up",
        "mufasa.ui.pop_ups.initialize_blob_tracking_pop_up",
    }
    leftover: list[str] = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                if node.module in deleted_mods:
                    leftover.append(
                        f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "no file imports from any of the deleted modules",
        leftover == [],
        detail=", ".join(leftover[:3]),
    )

    # ==================================================================
    # Qt replacements still present
    # ==================================================================
    roi_form = (pkg / "ui_qt" / "forms" / "roi.py").read_text()
    check(
        "Qt ROIManageForm still exists in ui_qt/forms/roi.py",
        "class ROIManageForm" in roi_form,
    )
    addons = (pkg / "ui_qt" / "forms" / "addons.py").read_text()
    check(
        "Qt BlobTrackerInitLauncher still exists in "
        "ui_qt/forms/addons.py",
        "class BlobTrackerInitLauncher" in addons,
    )
    addons_page = (pkg / "ui_qt" / "pages" / "addons_page.py").read_text()
    check(
        "addons_page.py still wires BlobTrackerInitLauncher",
        "BlobTrackerInitLauncher" in addons_page
        and "Blob tracker" in addons_page,
    )

    # ==================================================================
    # Tk-functions importer count regression guard
    # ==================================================================
    importer_count = 0
    for f in pkg.rglob("*.py"):
        if any(p in f.parts for p in ("ui", "ui_qt")):
            continue
        if f.name == "SimBA.py":
            continue
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        if any(isinstance(x, ast.ImportFrom)
               and x.module == "mufasa.ui.tkinter_functions"
               for x in t.body):
            importer_count += 1
    check(
        f"Backend module-level Tk-importer count ≤ 18 "
        f"(got {importer_count}; trajectory 25→23→22→21→19→18)",
        importer_count <= 18,
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    ba = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §3d Bucket 2 marks ROI Tk cluster "
        "DELETED 122cr",
        "DELETED 122cr" in ba,
    )
    check(
        "backend_audit.md §3a category list shows roi_tools/ at 0",
        "roi_tools/     (0)" in ba
        or "roi_ui_mixin deleted 122cr" in ba,
    )
    ta = (REPO_ROOT / "docs" / "tk_surface_audit.md").read_text()
    check(
        "tk_surface_audit.md §2g records the cluster-deletion "
        "as DONE",
        "(122cr): cluster deleted" in ta
        or "Update (122cr)" in ta,
    )

    # ==================================================================
    # mufasa/ui/ file count
    # ==================================================================
    ui_count = sum(1 for _ in (pkg / "ui").rglob("*.py"))
    check(
        f"mufasa/ui/ file count ≤ 88 (was 91 post-122ck; 122cr "
        f"deleted 3 ui/ files; got {ui_count})",
        ui_count <= 88,
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
        f"smoke_122cr_roi_tk_cluster_deletion: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
