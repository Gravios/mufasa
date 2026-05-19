"""
tests/smoke_122d5_stage_b.py
==============================

Patch 122d5: Stage B of the SimBA.py death cascade — bulk-delete
SimBA.py + all 113 files reachable only through it.

Coverage
--------

Deletion verification — all primary targets gone:
1.  mufasa/SimBA.py gone.
2.  mufasa/ui/pop_ups/ directory gone entirely (was 71 popups
    + __init__).
3.  mufasa/unsupervised/ directory gone entirely (was 30 algorithm
    files + 13 popups + 1 enums.py + 1 yaml + __init__).
4.  4 Tk-UI labelling files + annotator_mixin gone.
5.  4 ui/ Tk helpers gone (machine_model_settings_ui, utils,
    get_tree_view, video_timelaps).

Survivors confirmed:
6.  mufasa/ui/px_to_mm_ui.py still exists (Qt consumer in
    forms/video_utilities.py).
7.  mufasa/ui/tkinter_functions.py still exists (Stage C target).
8.  mufasa/mixins/pop_up_mixin.py still exists (Stage C target).
9.  mufasa/labelling/ backend files still exist (non-Tk
    utilities consumed by Qt forms).
10. mufasa/utils/confirm.py still exists (documented exception).

Integrity:
11. No surviving file in the tree imports from any deleted module.
12. All mufasa/**/*.py files parse cleanly post-deletion.
13. Total post-Stage-B .py file count is ≤ 425 (was ~534 pre-Stage-B
    per 122cx scoping; the cascade removed 109 .py files).

Doc updates:
14. cascade doc records Stage B as EXECUTED 122d5.
15. pyproject.toml's mufasa-tk entry-point removal from 122d4
    still holds (no regression).
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

    # 1. SimBA.py gone
    check("mufasa/SimBA.py is gone",
          not (pkg / "SimBA.py").exists())

    # 2. ui/pop_ups/ gone
    check("mufasa/ui/pop_ups/ directory is gone",
          not (pkg / "ui" / "pop_ups").exists())

    # 3. unsupervised/ gone
    check("mufasa/unsupervised/ directory is gone",
          not (pkg / "unsupervised").exists())

    # 4. Tk labelling cluster gone (5 files)
    for name in ["labelling_interface.py",
                 "labelling_advanced_interface.py",
                 "standard_labeller.py",
                 "targeted_annotations_clips.py"]:
        check(
            f"Tk labelling file gone: {name}",
            not (pkg / "labelling" / name).exists(),
        )
    check(
        "mixins/annotator_mixin.py gone",
        not (pkg / "mixins" / "annotator_mixin.py").exists(),
    )

    # 5. 4 ui/ Tk helpers gone
    for name in ["machine_model_settings_ui.py", "utils.py",
                 "get_tree_view.py", "video_timelaps.py"]:
        check(
            f"ui/ Tk helper gone: {name}",
            not (pkg / "ui" / name).exists(),
        )

    # 6-10. Survivors
    check(
        "ui/px_to_mm_ui.py survives (Qt consumer; documented)",
        (pkg / "ui" / "px_to_mm_ui.py").exists(),
    )
    check(
        "ui/tkinter_functions.py survives (Stage C target)",
        (pkg / "ui" / "tkinter_functions.py").exists(),
    )
    check(
        "mixins/pop_up_mixin.py survives (Stage C target)",
        (pkg / "mixins" / "pop_up_mixin.py").exists(),
    )
    # Labelling backend (non-Tk) files preserved
    labelling_backend = [
        "extract_labelled_frames.py",
        "extract_labelling_meta.py",
        "mitra_style_appender.py",
        "play_annotation_video.py",
        "single_clf_appender_excel.py",
    ]
    for name in labelling_backend:
        check(
            f"Labelling backend file survives: {name}",
            (pkg / "labelling" / name).exists(),
        )
    check(
        "utils/confirm.py survives (documented exception — "
        "lazy importer with now-broken fallback to "
        "tkinter_functions; tail in 122d7+)",
        (pkg / "utils" / "confirm.py").exists(),
    )

    # 11. No surviving file imports from any deleted module.
    # Build the deleted-module set from the deletion list, then
    # sweep every surviving .py for any ImportFrom that hits it.
    DELETED_MODS = {
        "mufasa.SimBA",
        "mufasa.ui.machine_model_settings_ui",
        "mufasa.ui.utils",
        "mufasa.ui.get_tree_view",
        "mufasa.ui.video_timelaps",
        "mufasa.labelling.labelling_interface",
        "mufasa.labelling.labelling_advanced_interface",
        "mufasa.labelling.standard_labeller",
        "mufasa.labelling.targeted_annotations_clips",
        "mufasa.mixins.annotator_mixin",
    }
    # Anything under mufasa.ui.pop_ups or mufasa.unsupervised:
    foreign_consumers = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if (mod in DELETED_MODS
                        or mod.startswith("mufasa.ui.pop_ups")
                        or mod.startswith("mufasa.unsupervised")):
                    foreign_consumers.append(
                        f"{f.relative_to(pkg)}:{node.lineno} "
                        f"← {mod}"
                    )
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if (a.name in DELETED_MODS
                            or a.name.startswith(
                                "mufasa.ui.pop_ups")
                            or a.name.startswith(
                                "mufasa.unsupervised")):
                        foreign_consumers.append(
                            f"{f.relative_to(pkg)}:{node.lineno} "
                            f"← {a.name}"
                        )
    check(
        f"No surviving file imports from any deleted module "
        f"(got {len(foreign_consumers)} foreign edges)",
        not foreign_consumers,
        detail="; ".join(foreign_consumers[:3]),
    )

    # 12. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All surviving mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    # 13. Total .py count is in expected range. Before 122d5 the
    # tree had ~530 .py files (rough); after Stage B's 109-py-file
    # cull the post count is in the low 420s. Pin a loose ceiling
    # rather than an exact value (further patches may add tests
    # or churn things slightly).
    total_py = sum(1 for _ in pkg.rglob("*.py"))
    check(
        f"Total mufasa/**/*.py count post-Stage-B ≤ 425 "
        f"(got {total_py}; Stage B removed ~109 .py files)",
        total_py <= 425,
    )

    # 14. Doc updates
    cascade = (REPO_ROOT / "docs"
               / "simba_death_cascade.md").read_text()
    check(
        "simba_death_cascade.md records Stage B EXECUTED 122d5",
        "EXECUTED 122d5" in cascade
        and "Stage B" in cascade,
    )

    # 15. 122d4 entry-point removal still holds
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    import re
    active_mufasa_tk = re.search(
        r"^\s*mufasa-tk\s*=\s*[\"']mufasa\.SimBA:main",
        pyproject, re.MULTILINE,
    )
    check(
        "122d4 mufasa-tk entry-point removal still holds "
        "(no active assignment)",
        active_mufasa_tk is None,
    )

    print(
        f"smoke_122d5_stage_b: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
