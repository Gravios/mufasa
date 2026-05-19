"""
tests/smoke_122cz_directing_bodypart_settings_qt.py
=====================================================

Patch 122cz: port DirectionAnimalToBodyPartSettingsPopUp to Qt.

Resolves the **blocking gap** identified in 122cy's pre-Stage-B
checklist. The Qt port lives as a Form on the addons page (not a
modal dialog) — single-dropdown UI matching the backend's
single-key storage contract.

Coverage
--------
1.  Tk popup `direction_animal_to_bodypart_settings_pop_up.py`
    is gone.
2.  Qt form class `DirectingBodyPartSettingsForm` exists in
    `mufasa/ui_qt/forms/addons.py`.
3.  Form subclasses `OperationForm`.
4.  Form has `build`, `collect_args`, `target` methods.
5.  Form writes to the correct config section/key
    (`Directionality settings` / `bodypart_direction`).
6.  addons_page.py wires DirectingBodyPartSettingsForm.
7.  addons.py __all__ list exports the new form.
8.  SimBA.py no longer has an active import of the deleted
    symbol.
9.  SimBA.py has no non-commented occurrence of the deleted
    button name (`button_analyzeDirection_bp`).
10. No file imports from the deleted Tk popup module.
11. The Qt AnalysisForm's directing-to-bodypart route still
    exists (the dependency surface — unchanged).
12. checklist doc records the resolution.
13. cascade doc records the resolution.
14. mufasa/ui/pop_ups/ count ≤ 74 (was 75 pre-122cz).
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

    # 1. Tk popup gone
    check(
        "Tk popup direction_animal_to_bodypart_settings_pop_up.py "
        "is gone",
        not (pkg / "ui" / "pop_ups"
             / "direction_animal_to_bodypart_settings_pop_up.py").exists(),
    )

    # 2-5. Qt form shape
    addons_path = pkg / "ui_qt" / "forms" / "addons.py"
    addons_src = addons_path.read_text()
    addons_tree = ast.parse(addons_src)

    form_cls = None
    for n in addons_tree.body:
        if (isinstance(n, ast.ClassDef)
                and n.name == "DirectingBodyPartSettingsForm"):
            form_cls = n
            break
    check(
        "DirectingBodyPartSettingsForm class defined in "
        "mufasa/ui_qt/forms/addons.py",
        form_cls is not None,
    )

    if form_cls is not None:
        base_names = [ast.unparse(b) for b in form_cls.bases]
        check(
            "DirectingBodyPartSettingsForm subclasses OperationForm",
            "OperationForm" in base_names,
        )
        method_names = {
            m.name for m in form_cls.body
            if isinstance(m, ast.FunctionDef)
        }
        check(
            "Form has the OperationForm contract methods "
            "(build, collect_args, target)",
            {"build", "collect_args", "target"} <= method_names,
        )

    # Config write target — the form must reference the right
    # section + key constants
    check(
        "Form writes to '[Directionality settings] / "
        "bodypart_direction' (backend dependency)",
        "Directionality settings" in addons_src
        and "bodypart_direction" in addons_src,
    )

    # 6. addons_page wiring
    addons_page_src = (pkg / "ui_qt" / "pages"
                       / "addons_page.py").read_text()
    check(
        "addons_page.py imports DirectingBodyPartSettingsForm",
        "DirectingBodyPartSettingsForm" in addons_page_src,
    )
    check(
        "addons_page.py adds 'Directing — body-part settings' "
        "section",
        "Directing — body-part settings" in addons_page_src
        or "Directing - body-part settings" in addons_page_src,
    )

    # 7. __all__ exports the form
    check(
        "addons.py __all__ exports DirectingBodyPartSettingsForm",
        '"DirectingBodyPartSettingsForm"' in addons_src,
    )

    # 8-9. SimBA.py cleanup — post-Stage-B (122d5) SimBA.py is
    # gone, so the "no active reference" checks pass trivially.
    simba_path = pkg / "SimBA.py"
    if not simba_path.exists():
        check(
            "SimBA.py gone (post-Stage-B 122d5) — import + "
            "button references trivially absent",
            True,
        )
        check(
            "SimBA.py gone (post-Stage-B 122d5) — "
            "button_analyzeDirection_bp trivially absent",
            True,
        )
    else:
        simba_src = simba_path.read_text()
        simba_tree = ast.parse(simba_src)
        deleted_sym = "DirectionAnimalToBodyPartSettingsPopUp"
        active_import = False
        for n in simba_tree.body:
            if isinstance(n, ast.ImportFrom):
                for alias in n.names:
                    if alias.name == deleted_sym:
                        active_import = True
        check(
            "SimBA.py has no active import of "
            "DirectionAnimalToBodyPartSettingsPopUp",
            not active_import,
        )

        leaked = []
        for i, line in enumerate(simba_src.split("\n"), 1):
            if line.lstrip().startswith("#"):
                continue
            if ("button_analyzeDirection_bp" in line
                    or deleted_sym in line):
                leaked.append(f"line {i}")
        check(
            "SimBA.py has no non-commented occurrence of the deleted "
            "symbol / button name",
            leaked == [],
            detail=", ".join(leaked[:3]),
        )

    # 10. No leftover importers
    leftover = []
    target_mod = ("mufasa.ui.pop_ups."
                  "direction_animal_to_bodypart_settings_pop_up")
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                    and node.module == target_mod):
                leftover.append(f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "no file imports from the deleted Tk popup module",
        leftover == [],
        detail=", ".join(leftover),
    )

    # 11. AnalysisForm route still exists (dependency surface)
    analysis_src = (pkg / "ui_qt" / "forms" / "analysis.py").read_text()
    check(
        "AnalysisForm still has 'Directing toward body-part' "
        "route (the dependency that 122cz unblocks)",
        "Directing toward body-part" in analysis_src,
    )

    # 12-13. Doc updates
    checklist = (REPO_ROOT / "docs"
                 / "stage_b_checklist.md").read_text()
    check(
        "stage_b_checklist.md records the 122cz resolution",
        "RESOLVED 122cz" in checklist
        and "DirectingBodyPartSettingsForm" in checklist,
    )
    cascade = (REPO_ROOT / "docs"
               / "simba_death_cascade.md").read_text()
    check(
        "simba_death_cascade.md notes 122cz resolved the blocker",
        "RESOLVED 122cz" in cascade
        or ("122cz" in cascade and "resolved" in cascade.lower()),
    )

    # 14. Count drop
    popups_count = sum(
        1 for _ in (pkg / "ui" / "pop_ups").glob("*.py")
        if _.name != "__init__.py"
    )
    check(
        f"mufasa/ui/pop_ups/ count ≤ 74 (was 75 pre-122cz; "
        f"got {popups_count})",
        popups_count <= 74,
    )

    # 15. Parse-clean
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
        f"smoke_122cz_directing_bodypart_settings_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
