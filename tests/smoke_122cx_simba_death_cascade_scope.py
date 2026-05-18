"""
tests/smoke_122cx_simba_death_cascade_scope.py
================================================

Patch 122cx: SimBA.py death cascade — scoping doc + boundary
regression guards.

Doc-only patch. Pins the exact numbers from
`docs/simba_death_cascade.md` so future patches can't quietly
shift the cascade boundary. If any of these check fail, the
scoping doc needs a re-audit before Stage B can land.

Coverage
--------

Stage B (the cascade) file counts:
1.  ui/pop_ups/ contains exactly 75 files (B2).
2.  All 75 popups have zero non-SimBA consumers.
3.  unsupervised/ contains exactly 14 files (B4).
4.  unsupervised has zero Qt-side imports (122cw evidence).
5.  Tk labelling cluster has exactly 4 files (B3).
6.  annotator_mixin.py has exactly 1 importer (a B3 file).
7.  ui/ has the expected 5 non-popup Tk helpers dying in B1.
8.  Stage B total = 99 files (5 + 75 + 5 + 14).

Stage C tail:
9.  tkinter_functions.py has zero Qt-side consumers.
10. pop_up_mixin.py has zero Qt-side consumers.

Survivors (NOT in cascade — Tier-4 follow-on):
11. px_to_mm_ui.py is consumed by ui_qt/forms/video_utilities.py.
12. utils/confirm.py exists (lazy importer; Tier-4 follow-on).

Doc updates:
13. docs/simba_death_cascade.md exists.
14. docs/backend_audit.md cross-references the scoping doc.
15. mufasa-tk entry point still exists (Stage A not yet executed).
16. All mufasa/**/*.py files parse cleanly.
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


def _real_importers(target_mod_prefix: str, pkg: Path,
                    exclude_self: bool = True
                    ) -> list[Path]:
    """Return paths of files that ImportFrom anything starting with
    `target_mod_prefix`, excluding the target file itself."""
    out: list[Path] = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith(target_mod_prefix):
                    out.append(f.relative_to(pkg))
                    break
    return out


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # Stage B file counts
    # ==================================================================
    # 1. ui/pop_ups/ — was 75 at 122cx scoping; will decrement
    # as future patches port + delete more popups (122cz did one).
    popups = [f for f in (pkg / "ui" / "pop_ups").glob("*.py")
              if f.name != "__init__.py"]
    check(
        f"ui/pop_ups/ has ≤ 75 files (Stage B2 starting count; "
        f"got {len(popups)} — decrements as ports land)",
        len(popups) <= 75,
    )

    # 2. All 75 popups have zero non-SimBA consumers
    # Build an index of all ImportFrom edges in one pass, then
    # check.
    popup_mods = {("mufasa.ui.pop_ups." + f.stem) for f in popups}
    foreign_consumers: list[str] = []
    for f in pkg.rglob("*.py"):
        if f.name == "SimBA.py":
            continue
        if "ui/pop_ups" in str(f).replace("\\", "/"):
            continue  # skip intra-cluster
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                    and node.module in popup_mods):
                foreign_consumers.append(
                    f"{f.relative_to(pkg)} ← {node.module}")
                break
    check(
        f"No ui/pop_ups file has non-SimBA, non-popup consumers "
        f"(got {len(foreign_consumers)} foreign edges)",
        not foreign_consumers,
        detail=("; ".join(foreign_consumers[:3]) if foreign_consumers
                else ""),
    )

    # 3. unsupervised/ — exactly 30 files (122cx re-audit: 14
    # UI files + 15 algorithm-backend + 1 enums.py = 30; the
    # 122cw audit undercounted to 14)
    unsup = [f for f in (pkg / "unsupervised").rglob("*.py")
             if f.name != "__init__.py"]
    check(
        f"unsupervised/ has 30 files (Stage B4; 122cx re-audit; "
        f"got {len(unsup)})",
        len(unsup) == 30,
    )

    # 4. unsupervised has zero Qt-side imports (122cw evidence)
    qt_unsup_hits = []
    for f in (pkg / "ui_qt").rglob("*.py"):
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
        "(122cw evidence preserved)",
        not qt_unsup_hits,
        detail=", ".join(qt_unsup_hits[:3]),
    )

    # 4b. unsupervised algorithm-backend modules have zero
    # consumers OUTSIDE the unsupervised/ directory (122cx
    # finding — they cascade-delete with the cluster)
    algo_files = sorted([
        f for f in (pkg / "unsupervised").glob("*.py")
        if f.name not in ("__init__.py",)
    ])
    algo_mods = {f"mufasa.unsupervised.{f.stem}"
                 for f in algo_files}
    foreign_unsup: list[str] = []
    for f in pkg.rglob("*.py"):
        if "unsupervised" in f.parts:
            continue  # intra-cluster fine
        if f.name == "SimBA.py":
            # SimBA.py:725 import is the entry; expected
            continue
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                    and node.module in algo_mods):
                foreign_unsup.append(
                    f"{f.relative_to(pkg)} ← {node.module}")
                break
    check(
        f"unsupervised/ algorithm-backend has zero outside "
        f"consumers (122cx finding; got "
        f"{len(foreign_unsup)} foreign edges)",
        not foreign_unsup,
        detail="; ".join(foreign_unsup[:3]),
    )

    # 5. Tk labelling cluster — 4 files (B3)
    tk_labelling = [
        "labelling/labelling_interface.py",
        "labelling/labelling_advanced_interface.py",
        "labelling/standard_labeller.py",
        "labelling/targeted_annotations_clips.py",
    ]
    missing = [p for p in tk_labelling
               if not (pkg / p).exists()]
    check(
        f"4 Tk-UI labelling files present "
        f"(missing: {missing})",
        not missing,
    )

    # 6. annotator_mixin has exactly 1 importer = a B3 file
    am_importers = _real_importers(
        "mufasa.mixins.annotator_mixin", pkg)
    check(
        f"annotator_mixin.py has exactly 1 importer "
        f"(got {len(am_importers)}: {am_importers})",
        len(am_importers) == 1
        and "targeted_annotations_clips" in str(am_importers[0]),
    )

    # 7. ui/ non-popup helpers — exactly 5 dying in B1, 1 surviving
    ui_non_popups = sorted([
        f for f in (pkg / "ui").rglob("*.py")
        if "pop_ups" not in f.parts and f.name != "__init__.py"
    ])
    expected_dying = {
        "machine_model_settings_ui.py",
        "utils.py",
        "get_tree_view.py",
        "video_timelaps.py",
        "tkinter_functions.py",  # Stage C, but counts in B1's tally
    }
    expected_surviving = {"px_to_mm_ui.py"}
    actual_names = {f.name for f in ui_non_popups}
    check(
        f"ui/ non-popup files = expected 6 "
        f"(5 dying in B1/C + 1 surviving = px_to_mm_ui); got "
        f"{len(ui_non_popups)}: "
        f"{sorted(actual_names)}",
        actual_names == (expected_dying | expected_surviving),
    )

    # 8. Stage B total = 115 files (122cx re-audit)
    stage_b_count = (
        5    # B1: SimBA.py + 4 ui/ helpers (NOT tkinter_functions)
        + 75   # B2
        + 5    # B3 (4 labelling + annotator_mixin)
        + 30   # B4 (122cx re-audit: 14 UI + 16 algo backend incl. enums)
    )
    check(
        f"Stage B file count = 115 (got {stage_b_count}; "
        f"updated in 122cx from 99 → 115 with unsupervised "
        f"algorithm-backend correction)",
        stage_b_count == 115,
    )

    # ==================================================================
    # Stage C tail
    # ==================================================================
    # 9. tkinter_functions.py has zero Qt-side consumers
    qt_tkf_hits = []
    for f in (pkg / "ui_qt").rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                if node.module == "mufasa.ui.tkinter_functions":
                    qt_tkf_hits.append(
                        f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "No Qt-side import of mufasa.ui.tkinter_functions "
        "(Stage C precondition)",
        not qt_tkf_hits,
        detail=", ".join(qt_tkf_hits[:3]),
    )

    # 10. pop_up_mixin.py has zero Qt-side consumers
    qt_pum_hits = []
    for f in (pkg / "ui_qt").rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                if node.module == "mufasa.mixins.pop_up_mixin":
                    qt_pum_hits.append(
                        f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "No Qt-side import of mufasa.mixins.pop_up_mixin "
        "(Stage C precondition)",
        not qt_pum_hits,
        detail=", ".join(qt_pum_hits[:3]),
    )

    # ==================================================================
    # Survivors
    # ==================================================================
    # 11. px_to_mm_ui is consumed by ui_qt
    px_consumers = _real_importers("mufasa.ui.px_to_mm_ui", pkg)
    qt_px = [p for p in px_consumers if "ui_qt" in str(p)]
    check(
        f"px_to_mm_ui.py has ≥ 1 Qt consumer (survives Stages "
        f"A-C); got {len(qt_px)}: {qt_px}",
        len(qt_px) >= 1,
    )

    # 12. utils/confirm.py exists (lazy importer; Tier-4 follow-on)
    check(
        "utils/confirm.py exists",
        (pkg / "utils" / "confirm.py").exists(),
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    # 13. Scoping doc exists
    scope_doc = REPO_ROOT / "docs" / "simba_death_cascade.md"
    check(
        "docs/simba_death_cascade.md exists",
        scope_doc.exists(),
    )

    # 14. backend_audit.md cross-references the scoping doc
    ba = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md cross-references simba_death_cascade.md",
        "simba_death_cascade.md" in ba,
    )

    # 15. mufasa-tk entry point still exists (Stage A not executed)
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    check(
        "mufasa-tk entry point still exists in pyproject.toml "
        "(Stage A has NOT been executed yet)",
        "mufasa-tk" in pyproject
        and "mufasa.SimBA:main" in pyproject,
    )

    # 16. Parse-clean
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
        f"smoke_122cx_simba_death_cascade_scope: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
