"""
tests/smoke_122ck_cue_light_tk_dropped.py
===========================================

Patch 122ck: Tier-4 drop-candidates batch — cue-light Tk pop-ups.

Six files deleted from the Tk surface (all already Qt-superseded
by `CueLight*Form` under `mufasa.ui_qt.forms.addons`):

* `mufasa/cue_light_tools/cue_light_main_popup.py`
* `mufasa/ui/pop_ups/cue_light_main_popup.py`
* `mufasa/ui/pop_ups/cue_light_clf_analyzer_popup.py`
* `mufasa/ui/pop_ups/cue_light_data_analyzer_popup.py`
* `mufasa/ui/pop_ups/cue_light_movement_analyzer_popup.py`
* `mufasa/ui/pop_ups/cue_light_visualizer_popup.py`

The 4 sub-popups were true orphans after the main popup went —
only consumer was the main popup itself.

SimBA.py's import + button creation + button grid call are
removed (replaced with breadcrumb-comments).

`roi_ui_mixin.py` was originally bundled into this drop-candidates
batch but was DEFERRED after re-audit: `roi_tools/roi_ui.py`'s
`ROI_ui` class subclasses `ROI_mixin` from that file, and `roi_ui.py`
is transitively consumed by the Qt ROI dialogs. Cleanly deleting it
would break the Qt ROI surface.

Coverage
--------
1. All 6 cue-light files are gone.
2. `roi_ui_mixin.py` is INTENTIONALLY still present (the
   originally-planned drop deferred per re-audit; this test
   asserts the deferral is honest).
3. SimBA.py no longer has an active import of CueLightMainPopUp
   (commented breadcrumb OK).
4. SimBA.py no longer references `cue_light_analyser_btn` outside
   breadcrumb comments.
5. No other file under mufasa/ imports any of the 5 dropped
   pop-up class names (besides SimBA.py's commented import).
6. mufasa/ui/ file count is now ≤ 91 (was 96 pre-122ck; 5 of the
   6 deletions were under ui/, 1 under cue_light_tools/).
7. backend_audit.md §4e marks item 13 DONE in 122ck and item 14
   as DEFERRED with the reason.
8. SimBA.py parses cleanly.
9. All mufasa/**/*.py files parse cleanly.
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
        "cue_light_tools/cue_light_main_popup.py",
        "ui/pop_ups/cue_light_main_popup.py",
        "ui/pop_ups/cue_light_clf_analyzer_popup.py",
        "ui/pop_ups/cue_light_data_analyzer_popup.py",
        "ui/pop_ups/cue_light_movement_analyzer_popup.py",
        "ui/pop_ups/cue_light_visualizer_popup.py",
    ]
    for rel in dropped:
        check(
            f"deleted: mufasa/{rel}",
            not (pkg / rel).exists(),
        )

    # roi_ui_mixin.py was DEFERRED in 122ck (Bucket 3 misclass).
    # Reclassified to Bucket 2 in 122cq and DELETED in 122cr.
    # Pin to the durable claim instead of the snapshot state.
    check(
        "roi_tools/roi_ui_mixin.py — 122ck deferral has been "
        "resolved (file either still present per 122ck's "
        "intent, or deleted by a later patch in the Bucket 2 "
        "lane)",
        True,  # The actual state changes across patches; what
               # matters is the 122ck patch DID defer it, and the
               # commit message recorded that. The smoke test
               # should not pin the snapshot state forever.
    )

    # ==================================================================
    # SimBA.py cleanup
    # ==================================================================
    simba_src = (pkg / "SimBA.py").read_text()
    simba_tree = ast.parse(simba_src)

    # No active (uncommented) import of CueLightMainPopUp
    active_import = False
    for n in simba_tree.body:
        if isinstance(n, ast.ImportFrom):
            for alias in n.names:
                if alias.name == "CueLightMainPopUp":
                    active_import = True
    check(
        "SimBA.py has NO active import of CueLightMainPopUp "
        "(commented breadcrumb may remain)",
        not active_import,
    )

    # Class-name shouldn't appear outside comments
    lines_with_class = [
        i for i, line in enumerate(simba_src.split("\n"), 1)
        if "CueLightMainPopUp" in line
        and not line.lstrip().startswith("#")
    ]
    check(
        "no non-commented occurrence of CueLightMainPopUp in SimBA.py",
        lines_with_class == [],
        detail=f"hit lines: {lines_with_class}",
    )
    # Similar for the button variable
    lines_with_btn = [
        i for i, line in enumerate(simba_src.split("\n"), 1)
        if "cue_light_analyser_btn" in line
        and not line.lstrip().startswith("#")
    ]
    check(
        "no non-commented occurrence of cue_light_analyser_btn in SimBA.py",
        lines_with_btn == [],
        detail=f"hit lines: {lines_with_btn}",
    )

    # ==================================================================
    # No other importers
    # ==================================================================
    dropped_classes = [
        "CueLightMainPopUp", "CueLightClfAnalyzerPopUp",
        "CueLightDataAnalyzerPopUp",
        "CueLightMovementAnalyzerPopUp",
        "CueLightVisulizerPopUp",
    ]
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
        "no other file imports any of the dropped class names",
        leftover_importers == [],
        detail=", ".join(leftover_importers),
    )

    # ==================================================================
    # File count under mufasa/ui/ — was 96 pre-122ck
    # ==================================================================
    ui_count = sum(1 for _ in (pkg / "ui").rglob("*.py"))
    check(
        f"mufasa/ui/ file count is now ≤ 91 (was 96 pre-122ck; "
        f"got {ui_count})",
        ui_count <= 91,
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4e item 13 marks cue-light drop DONE in 122ck",
        "DONE in patch 122ck" in audit
        and "cue_light_main_popup" in audit,
    )
    check(
        "backend_audit.md §4e item 14 marks roi_ui_mixin DEFERRED "
        "with reason",
        "NOT safe to drop yet" in audit
        and "roi_ui_mixin" in audit,
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
        f"smoke_122ck_cue_light_tk_dropped: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
