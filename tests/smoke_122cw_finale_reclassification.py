"""
tests/smoke_122cw_finale_reclassification.py
==============================================

Patch 122cw: re-audit of Bucket 2 dispositions.

Doc-only patch. After the 122cv subprocess-bridge closure, the
remaining Bucket-2 entries (14 unsupervised files + 5 labelling
Tk-UI files including annotator_mixin) were re-audited. Finding:
both clusters reach only through SimBA.py / `mufasa-tk` (legacy
entry point); neither has Qt-side hooks; labelling already has
a Qt port at `ui_qt/frame_labeller.py`.

Reclassification: "dies with Tier 3b X Qt port" → "dies with
SimBA.py finale". No new work items; these are SimBA.py-death
prerequisites that have already been satisfied (or were never
required).

This re-audit becomes the 5th methodology lesson:
**Entry-point reachability matters for "needs a Qt port"
determinations.**

Coverage
--------
1.  The 13 unsupervised pop_up files exist (un-deleted; alive
    via SimBA.py:725 deferred import).
2.  unsupervised_main.py exists.
3.  SimBA.py:725 deferred-imports UnsupervisedGUI from
    unsupervised_main (sole consumer; no Qt-side import).
4.  No file under mufasa/ui_qt/ references "unsupervised" (the
    cluster is invisible to Qt users).
5.  The 4 labelling Tk-UI files exist.
6.  annotator_mixin.py exists and is only consumed by
    labelling/targeted_annotations_clips.py.
7.  Qt-side labelling exists (ui_qt/frame_labeller.py).
8.  Qt forms/annotation.py imports extract_labelled_frames +
    extract_labelling_meta (the backend utilities that DO
    survive past SimBA.py death).
9.  backend_audit.md §3d Bucket 2 reclassified — "SimBA.py
    finale" annotation present.
10. backend_audit.md §3d Bucket 2 says the labelling Tk UI
    cluster's Qt port already exists.
11. backend_audit.md notes that Tier 3b is no longer a separate
    work item (or qualified as such).
12. tk_surface_audit.md §7 has the 5th methodology lesson
    (entry-point reachability).
13. All mufasa/**/*.py files parse cleanly.
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

    # 1-2. Unsupervised cluster still present (will die later)
    unsup_popups = list((pkg / "unsupervised" / "pop_ups").glob("*.py"))
    unsup_popup_count = len([
        f for f in unsup_popups if f.name != "__init__.py"
    ])
    check(
        f"unsupervised/pop_ups/ has 13 files (got "
        f"{unsup_popup_count})",
        unsup_popup_count == 13,
    )
    check(
        "unsupervised/unsupervised_main.py exists",
        (pkg / "unsupervised" / "unsupervised_main.py").exists(),
    )

    # 3. SimBA.py:725 is the sole consumer
    simba_src = (pkg / "SimBA.py").read_text()
    check(
        "SimBA.py:725 contains the deferred UnsupervisedGUI import",
        "from mufasa.unsupervised.unsupervised_main import "
        "UnsupervisedGUI" in simba_src,
    )

    # 4. No Qt-side reach into unsupervised
    qt_unsup_hits = []
    qt_dir = pkg / "ui_qt"
    if qt_dir.exists():
        for f in qt_dir.rglob("*.py"):
            try:
                t = ast.parse(f.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(t):
                if isinstance(node, ast.ImportFrom):
                    if "unsupervised" in (node.module or ""):
                        qt_unsup_hits.append(
                            f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "No Qt-side import of unsupervised modules "
        "(confirms cluster is invisible to Qt users)",
        not qt_unsup_hits,
        detail=", ".join(qt_unsup_hits[:3]),
    )

    # 5. Labelling Tk-UI cluster still present
    tk_labelling = [
        "labelling/labelling_interface.py",
        "labelling/labelling_advanced_interface.py",
        "labelling/standard_labeller.py",
        "labelling/targeted_annotations_clips.py",
    ]
    missing_tk_labelling = [
        p for p in tk_labelling if not (pkg / p).exists()
    ]
    check(
        f"4 Tk-UI labelling files still present "
        f"(missing: {missing_tk_labelling})",
        missing_tk_labelling == [],
    )

    # 6. annotator_mixin.py: sole consumer is labelling/targeted_annotations_clips
    check(
        "mixins/annotator_mixin.py exists",
        (pkg / "mixins" / "annotator_mixin.py").exists(),
    )
    am_importers = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                if "annotator_mixin" in (node.module or ""):
                    am_importers.append(
                        str(f.relative_to(pkg)))
    check(
        f"annotator_mixin.py has exactly 1 importer = "
        f"labelling/targeted_annotations_clips.py "
        f"(got {len(am_importers)}: {am_importers})",
        len(am_importers) == 1
        and "labelling/targeted_annotations_clips.py" in
            am_importers[0].replace("\\", "/"),
    )

    # 7. Qt-side labelling exists
    check(
        "Qt-side ui_qt/frame_labeller.py exists "
        "(confirms Qt port for labelling is already shipped)",
        (pkg / "ui_qt" / "frame_labeller.py").exists(),
    )

    # 8. Qt forms/annotation.py uses the backend utilities
    annot = (pkg / "ui_qt" / "forms" / "annotation.py").read_text()
    check(
        "Qt forms/annotation.py imports extract_labelled_frames "
        "(backend utility that survives past SimBA.py)",
        "extract_labelled_frames" in annot,
    )
    check(
        "Qt forms/annotation.py imports extract_labelling_meta "
        "(backend utility)",
        "extract_labelling_meta" in annot,
    )

    # 9-11. backend_audit.md reclassification
    ba = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §3d Bucket 2 reclassified — "
        "'SimBA.py finale' annotation present",
        "SimBA.py finale" in ba
        and "reclassified in 122cw" in ba,
    )
    check(
        "backend_audit.md §3d cluster-shapes note that "
        "Qt labelling already exists",
        "ui_qt/frame_labeller.py" in ba
        or "Qt port already exists" in ba,
    )
    check(
        "backend_audit.md §3d notes Tier 3b is not a separate "
        "work item",
        "Tier 3b" in ba
        and ("not separate" in ba
             or "not a separate work item" in ba
             or "have already been satisfied" in ba),
    )

    # 12. tk_surface_audit.md §7 has the 5th lesson
    ta = (REPO_ROOT / "docs" / "tk_surface_audit.md").read_text()
    check(
        "tk_surface_audit.md §7 has the entry-point-reachability "
        "lesson (5th methodology lesson)",
        "Entry-point reachability" in ta
        and "mufasa-tk" in ta,
    )

    # 13. Parse-clean
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
        f"smoke_122cw_finale_reclassification: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
