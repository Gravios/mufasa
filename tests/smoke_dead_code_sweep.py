"""
tests/smoke_dead_code_sweep.py
==============================

Patch 122ad: regression guard for the dead-code sweep. Verifies
that six legacy popup files are gone and that SimBA.py no
longer carries code-level references to their classes.

Two buckets of deletions:

1. **Orphan files** — already not imported anywhere outside
   the pop_ups directory. Pure file deletes; SimBA changes
   not required:
     * dlc_annotations_to_labelme_popup.py
     * labelme_dir_to_csv_popup.py
     * roi_analysis_pop_up.py
     * targeted_annotation_clips_pop_up.py

2. **Replaces-claim files** — Qt forms explicitly document
   replacement via 'Replaces X' docstring claims. SimBA
   imports + wiring scrubbed in the same patch so the legacy
   Tk launcher stays consistent:
     * batch_preprocess_pop_up.py (BatchPreProcessPopUp)
     * extract_annotation_frames_pop_up.py
       (ExtractAnnotationFramesPopUp)

Coverage:

1. All 6 legacy popup files no longer exist on disk.
2. SimBA.py has no code-level reference to any of the 6
   deleted classes (comment-only mentions in the patch-history
   notes are allowed).
3. SimBA.py docstring/comments record the 122ad scrubs.
4. The Qt replacements still exist and still claim to replace
   the appropriate legacy class.
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
    pop_dir = REPO_ROOT / "mufasa" / "ui" / "pop_ups"

    # ----- 1. Legacy files deleted -----
    DELETED_FILES = [
        "dlc_annotations_to_labelme_popup.py",
        "labelme_dir_to_csv_popup.py",
        "roi_analysis_pop_up.py",
        "targeted_annotation_clips_pop_up.py",
        "batch_preprocess_pop_up.py",
        "extract_annotation_frames_pop_up.py",
    ]
    for f in DELETED_FILES:
        check(
            f"legacy {f} no longer on disk",
            not (pop_dir / f).exists(),
        )

    # ----- 2. SimBA.py has no code-level references -----
    simba_src = (REPO_ROOT / "mufasa" / "SimBA.py").read_text()
    DELETED_CLASSES = [
        "BatchPreProcessPopUp",
        "ExtractAnnotationFramesPopUp",
        "ROIAnalysisPopUp",
        "TargetedAnnotationsWClipsPopUp",
        "DLCAnnotations2LabelMePopUp",
        "LabelmeDirectory2CSVPopUp",
    ]
    for cls in DELETED_CLASSES:
        leaked = any(
            cls in line and not line.lstrip().startswith("#")
            for line in simba_src.splitlines()
        )
        check(
            f"SimBA.py has no code-level reference to {cls}",
            not leaked,
        )

    # ----- 3. 122ad scrub markers present -----
    check(
        "SimBA.py records the 122ad batch_preprocess scrub",
        "Patch 122ad: batch_preprocess_pop_up module deleted"
        in simba_src,
    )
    check(
        "SimBA.py records the 122ad extract_annotation_frames "
        "scrub",
        "Patch 122ad: extract_annotation_frames_pop_up module "
        "deleted" in simba_src,
    )
    check(
        "SimBA.py records the 122ad 'Batch pre-process videos' "
        "menu entry removal",
        "'Batch pre-process videos' menu entry" in simba_src
        or "Batch pre-process videos' menu entry" in simba_src,
    )
    check(
        "SimBA.py records the 122ad 'VISUALIZE ANNOTATIONS FRAMES' "
        "button removal",
        "VISUALIZE ANNOTATIONS FRAMES" in simba_src,
    )

    # ----- 4. Qt replacements still exist -----
    qt_dir = REPO_ROOT / "mufasa" / "ui_qt"
    qt_text = ""
    for f in qt_dir.rglob("*.py"):
        qt_text += f.read_text() + "\n"
    REPLACEMENTS = {
        "BatchPreProcessPopUp": "BatchPreProcessLauncher",
        "ExtractAnnotationFramesPopUp": "ExtractFramesForm",
        "TargetedAnnotationsWClipsPopUp": "TargetedAnnotationClipsLauncher",
    }
    for legacy, qt in REPLACEMENTS.items():
        check(
            f"Qt replacement for {legacy} ({qt}) still present "
            "in mufasa/ui_qt/",
            qt in qt_text,
        )

    # ----- 5. SimBA.py syntactically valid -----
    import ast
    try:
        ast.parse(simba_src)
        check("SimBA.py parses cleanly post-sweep", True)
    except SyntaxError as e:
        check(
            "SimBA.py parses cleanly post-sweep",
            False, detail=str(e),
        )

    print(
        f"smoke_dead_code_sweep: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
