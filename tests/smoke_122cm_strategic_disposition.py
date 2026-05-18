"""
tests/smoke_122cm_strategic_disposition.py
============================================

Patch 122cm: strategic disposition of remaining Tk importers +
drop two clean orphans (`boundary_menus.py`,
`batch_process_menus.py`).

The bulk of this patch is a documentation update — a new §3d in
`backend_audit.md` that classifies the remaining importers into
four buckets based on the lessons of patches 122ch through 122cl:

* Bucket 1: already-Qt-replaced or zero-consumer (delete-only)
* Bucket 2: dies with another Tier-4 work item (wait, don't decouple)
* Bucket 3: deferred — Qt code currently consumes it
* Bucket 4: lazy importer, non-blocking

The annotator_mixin.py audit item is RECLASSIFIED, not closed
(§4d item 14): considered for decoupling in 122cm, rejected on
inspection because the 5 Entry_Box instantiations are primary UI
primitives. Moved to Bucket 2 (dies with labelling Qt port).

Coverage
--------
1. mufasa/bounding_box_tools/boundary_menus.py is gone.
2. mufasa/video_processors/batch_process_menus.py is gone.
3. SimBA.py no longer has an active import of BoundaryMenus.
4. SimBA.py no longer references `anchored_roi_analysis_btn`
   outside breadcrumb comments.
5. No other file imports BoundaryMenus or BatchProcessFrame.
6. ui_qt/forms/batch_pre_process.py docstring updated to point
   at the Qt replacement (no broken `batch_process_menus`
   reference).
7. video_processors/blob_tracking_executor.py docstring updated.
8. backend_audit.md §3 count updated to 19 module-level
   importers.
9. backend_audit.md §3a category list updated post-122cm.
10. backend_audit.md §3d strategic-disposition section exists
    with the four-bucket classification.
11. backend_audit.md §3d documents the annotator_mixin
    reclassification.
12. backend_audit.md §3d documents the decision rule for future
    audits.
13. backend_audit.md §4d item 14 marked RECLASSIFIED in 122cm.
14. backend_audit.md §4f items 17 + 18 marked DONE in 122cm.
15. Module-level Tk-importer count is ≤ 19 (regression guard).
16. SimBA.py parses cleanly.
17. All mufasa/**/*.py files parse cleanly.
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


def _module_level_imports(tree: ast.Module, target: str) -> bool:
    return any(
        isinstance(n, ast.ImportFrom) and n.module == target
        for n in tree.body
    )


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # File deletions
    # ==================================================================
    check(
        "mufasa/bounding_box_tools/boundary_menus.py is gone",
        not (pkg / "bounding_box_tools" / "boundary_menus.py").exists(),
    )
    check(
        "mufasa/video_processors/batch_process_menus.py is gone",
        not (pkg / "video_processors" / "batch_process_menus.py").exists(),
    )

    # ==================================================================
    # SimBA.py cleanup
    # ==================================================================
    simba_src = (pkg / "SimBA.py").read_text()
    simba_tree = ast.parse(simba_src)

    active_import = False
    for n in simba_tree.body:
        if isinstance(n, ast.ImportFrom):
            for alias in n.names:
                if alias.name == "BoundaryMenus":
                    active_import = True
    check(
        "SimBA.py has NO active import of BoundaryMenus "
        "(commented breadcrumb may remain)",
        not active_import,
    )

    lines_with_btn = [
        i for i, line in enumerate(simba_src.split("\n"), 1)
        if "anchored_roi_analysis_btn" in line
        and not line.lstrip().startswith("#")
    ]
    check(
        "no non-commented occurrence of anchored_roi_analysis_btn",
        lines_with_btn == [],
        detail=f"hit lines: {lines_with_btn}",
    )

    # ==================================================================
    # No other importers of the dropped classes
    # ==================================================================
    dropped_classes = ["BoundaryMenus", "BatchProcessFrame"]
    leftover_importers: list[str] = []
    for f in pkg.rglob("*.py"):
        if f.name == "SimBA.py":
            continue  # breadcrumb-comments allowed
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in dropped_classes:
                        leftover_importers.append(
                            f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "no file imports BoundaryMenus or BatchProcessFrame",
        leftover_importers == [],
        detail=", ".join(leftover_importers),
    )

    # ==================================================================
    # Docstring pointers updated
    # ==================================================================
    bpp = (pkg / "ui_qt" / "forms" / "batch_pre_process.py").read_text()
    check(
        "batch_pre_process.py docstring no longer points at "
        "the deleted batch_process_menus path",
        ":class:`mufasa.video_processors.batch_process_menus" not in bpp
        and ":func:`mufasa.video_processors.batch_process_menus" not in bpp,
    )
    bte = (pkg / "video_processors" /
           "blob_tracking_executor.py").read_text()
    check(
        "blob_tracking_executor.py docstring no longer points at "
        "the deleted batch_process_menus path",
        ":func:`mufasa.video_processors.batch_process_menus" not in bte
        and ":class:`mufasa.video_processors.batch_process_menus" not in bte,
    )

    # ==================================================================
    # backend_audit.md updates
    # ==================================================================
    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §3 records 122cm's contribution to the "
        "module-level importer count trajectory",
        # 122cm dropped the count from 21 → 19. Later patches
        # may reduce further. Pin to the trajectory entry that
        # records 122cm's effect, not the live count.
        "122cm" in audit and "19" in audit,
    )
    check(
        "backend_audit.md §3a category list updated post-122cm",
        "boundary_menus deleted in 122cm" in audit
        and "batch_process_menus deleted in 122cm" in audit,
    )
    check(
        "backend_audit.md §3d strategic-disposition section exists",
        "### 3d. Strategic disposition" in audit
        or "3d. Strategic disposition" in audit,
    )
    # The four-bucket classification
    check(
        "§3d has the four-bucket classification "
        "(delete-only / dies-with / deferred / lazy)",
        "Bucket 1" in audit and "Bucket 2" in audit
        and "Bucket 3" in audit and "Bucket 4" in audit,
    )
    check(
        "§3d documents the annotator_mixin reclassification "
        "(decided not to decouple)",
        "annotator_mixin" in audit
        and ("rejected" in audit.lower() or "reclassified" in audit.lower()),
    )
    check(
        "§3d documents the decision rule for future audits",
        "Decision rule" in audit or "decision rule" in audit,
    )
    check(
        "backend_audit.md §4d item 14 (annotator_mixin) marked "
        "RECLASSIFIED in 122cm",
        "RECLASSIFIED in patch 122cm" in audit
        or "RECLASSIFIED**" in audit,
    )
    check(
        "backend_audit.md §4f items 17 + 18 marked DONE in 122cm",
        audit.count("DONE in patch 122cm") >= 2,
    )

    # ==================================================================
    # Module-level count regression guard
    # ==================================================================
    module_level_count = 0
    for f in pkg.rglob("*.py"):
        if any(p in f.parts for p in ("ui", "ui_qt")):
            continue
        if f.name == "SimBA.py":
            continue
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        if _module_level_imports(t, "mufasa.ui.tkinter_functions"):
            module_level_count += 1
    check(
        f"Backend module-level Tk-importer count is ≤ 19 "
        f"(got {module_level_count}; trajectory 25→23→22→21→19)",
        module_level_count <= 19,
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
        f"smoke_122cm_strategic_disposition: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
