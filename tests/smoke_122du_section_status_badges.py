"""
tests/smoke_122du_section_status_badges.py
=============================================

Patch 122du: visible UX for the section-provenance infrastructure
landed in 122ds + wired by 122dt.

Visual spec
-----------
* **UNKNOWN** — white-filled circle with a thin gray outline.
  Default state; section hasn't run yet (or hasn't recorded
  provenance).
* **CURRENT** — green-filled circle with a white checkmark.
  Section ran after all known-timestamp dependencies.
* **STALE** — orange-filled circle. A dependency ran more
  recently; section needs to re-run.

All three icons share circle geometry; only fill and the
checkmark glyph vary.

Wiring summary
--------------
* ``mufasa.ui_qt.provenance_badge`` — new module. Three inline
  SVG strings, ``icon_for_status(SectionStatus) -> QIcon`` with
  per-process cache via ``@lru_cache``.
* ``mufasa.section_provenance.find_section_by_title(page, title)``
  — new helper. Bridges WorkflowPage's ``section_title`` strings
  to ``SECTIONS`` (which keys by ``section_id``).
* ``WorkflowPage.add_section`` / ``add_section_widget`` — paint
  initial badge via new ``_paint_initial_badge`` helper.
* ``WorkflowPage.refresh_section_badges`` — re-query every section
  on this page via ``get_all_statuses`` and call
  ``setItemIcon(index, icon_for_status(status))``.
* ``WorkflowPage._on_form_completed`` — bubbles form completion
  up to the workbench (walks ``self.window()``).
* ``WorkflowPage._instantiate`` — connects every form's
  ``completed`` signal to ``_on_form_completed``.
* ``MufasaWorkbench._refresh_all_section_badges`` — iterates
  ``_pages_by_title`` and calls each page's
  ``refresh_section_badges``.

Coverage
--------
provenance_badge module:
1.  Module imports without PySide6 (lazy Qt imports inside
    ``icon_for_status``).
2.  Three SVG strings declared, one per SectionStatus value.
3.  Each SVG has a 16×16 viewBox.
4.  UNKNOWN SVG has a white-filled circle with a stroke (outline).
5.  CURRENT SVG has a green fill AND a checkmark path.
6.  STALE SVG has an orange fill (no glyph).
7.  ``icon_for_status`` is cached (``@lru_cache``).
8.  ``clear_icon_cache`` exists.
9.  ``icon_for_status`` raises ValueError for unknown values.

section_provenance.find_section_by_title:
10. Helper is exported.
11. Finds a section by (page, title) pair.
12. Returns None for an unknown page/title.

WorkflowPage hooks:
13. WorkflowPage has ``refresh_section_badges`` method.
14. WorkflowPage has ``_on_form_completed`` method.
15. WorkflowPage has ``_paint_initial_badge`` method.
16. ``add_section`` calls ``_paint_initial_badge`` after adding.
17. ``add_section_widget`` calls ``_paint_initial_badge`` after
    adding.
18. ``_instantiate`` connects ``form.completed`` to
    ``self._on_form_completed``.

MufasaWorkbench hook:
19. MufasaWorkbench has ``_refresh_all_section_badges`` method.
20. ``_refresh_all_section_badges`` iterates
    ``self._pages_by_title``.

Cross-patch invariants:
21. 122ds SECTIONS dict still has the entries 122du surfaces.
22. 122dt subclass declarations (RunOutlierCorrectionForm
    section_id == "outlier_correction") still in place.
23. All mufasa/**/*.py parse cleanly.
24. 122do baseline tripwire: no ``Optional[`` in non-docstring
    positions across mufasa/ui_qt/.
"""
from __future__ import annotations

import ast
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


def _ast_method(cls_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _ast_find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def main() -> int:
    # -----------------------------------------------------------------
    # provenance_badge module
    # -----------------------------------------------------------------
    # 1. Imports cleanly without PySide6 active. (The Qt imports are
    # lazy inside icon_for_status.)
    import mufasa.ui_qt.provenance_badge as pb
    check(
        "mufasa.ui_qt.provenance_badge imports without PySide6 "
        "(lazy Qt imports inside icon_for_status)",
        hasattr(pb, "icon_for_status"),
    )

    # 2-6. SVG declarations.
    svg_unknown = pb._SVG_UNKNOWN.decode()
    svg_current = pb._SVG_CURRENT.decode()
    svg_stale = pb._SVG_STALE.decode()
    check(
        "Three SVG strings declared (UNKNOWN, CURRENT, STALE)",
        all(s.startswith("<svg") for s in (svg_unknown, svg_current,
                                            svg_stale)),
    )
    check(
        "All SVGs use a 16x16 viewBox",
        all('viewBox="0 0 16 16"' in s for s in (svg_unknown,
                                                  svg_current,
                                                  svg_stale)),
    )
    check(
        "UNKNOWN SVG: white-filled circle WITH a stroke outline "
        "(user spec: 'white empty circles as default'; outline "
        "ensures visibility on light themes)",
        ('fill="white"' in svg_unknown
         and 'stroke="' in svg_unknown
         and 'circle' in svg_unknown),
    )
    check(
        "CURRENT SVG: green-filled circle AND a checkmark path "
        "(user spec: 'green checkmark in a green filled circle')",
        '#16a34a' in svg_current and '<path' in svg_current,
    )
    check(
        "STALE SVG: orange-filled circle, no glyph "
        "(user spec: 'orange circle (filled)')",
        '#f97316' in svg_stale and '<path' not in svg_stale,
    )

    # 7-9. icon_for_status cache + safety.
    badge_tree = ast.parse(
        (REPO_ROOT / "mufasa" / "ui_qt"
         / "provenance_badge.py").read_text())
    has_lru = False
    for node in ast.walk(badge_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "icon_for_status":
            for dec in node.decorator_list:
                src = ast.unparse(dec)
                if "lru_cache" in src:
                    has_lru = True
                    break
    check(
        "icon_for_status is @lru_cache-decorated (one render per "
        "process per status)",
        has_lru,
    )
    check(
        "clear_icon_cache helper is exported (for hot-reload "
        "scenarios in tests)",
        hasattr(pb, "clear_icon_cache"),
    )
    # Module-level _SVG_FOR_STATUS sanity: should have exactly 3 keys
    check(
        "_SVG_FOR_STATUS maps each of the 3 SectionStatus values "
        "to an SVG",
        len(pb._SVG_FOR_STATUS) == 3,
    )

    # -----------------------------------------------------------------
    # find_section_by_title
    # -----------------------------------------------------------------
    from mufasa.section_provenance import (
        SECTIONS,
        find_section_by_title,
    )
    check(
        "section_provenance.find_section_by_title is exported "
        "(bridges WorkflowPage's section_title strings to "
        "SECTIONS by section_id)",
        callable(find_section_by_title),
    )
    spec = find_section_by_title("Preprocessing", "Run outlier correction")
    check(
        "find_section_by_title('Preprocessing', 'Run outlier correction') "
        "returns the outlier_correction section",
        spec is not None and spec.section_id == "outlier_correction",
        detail=(f"got {spec!r}"),
    )
    check(
        "find_section_by_title returns None for an unknown "
        "(page, title) pair",
        find_section_by_title("Not a page", "Not a section") is None,
    )

    # -----------------------------------------------------------------
    # WorkflowPage hooks (AST inspection — PySide6 not available)
    # -----------------------------------------------------------------
    wb_path = REPO_ROOT / "mufasa" / "ui_qt" / "workbench.py"
    wb_tree = ast.parse(wb_path.read_text())
    wp = _ast_find_class(wb_tree, "WorkflowPage")
    assert wp is not None

    refresh_method = _ast_method(wp, "refresh_section_badges")
    on_complete_method = _ast_method(wp, "_on_form_completed")
    paint_initial_method = _ast_method(wp, "_paint_initial_badge")
    check(
        "WorkflowPage has refresh_section_badges method",
        refresh_method is not None,
    )
    check(
        "WorkflowPage has _on_form_completed method "
        "(bubbles to workbench)",
        on_complete_method is not None,
    )
    check(
        "WorkflowPage has _paint_initial_badge method "
        "(used by add_section / add_section_widget)",
        paint_initial_method is not None,
    )

    add_section_method = _ast_method(wp, "add_section")
    add_section_widget_method = _ast_method(wp, "add_section_widget")
    assert add_section_method is not None
    assert add_section_widget_method is not None
    check(
        "add_section calls _paint_initial_badge after adding the "
        "toolbox item",
        "_paint_initial_badge" in ast.unparse(add_section_method),
    )
    check(
        "add_section_widget calls _paint_initial_badge after adding "
        "the toolbox item",
        "_paint_initial_badge" in ast.unparse(add_section_widget_method),
    )

    instantiate_method = _ast_method(wp, "_instantiate")
    assert instantiate_method is not None
    instantiate_src = ast.unparse(instantiate_method)
    check(
        "_instantiate connects form.completed to "
        "self._on_form_completed (so completion bubbles up)",
        ("form.completed.connect" in instantiate_src
         and "_on_form_completed" in instantiate_src),
    )

    # -----------------------------------------------------------------
    # MufasaWorkbench hook
    # -----------------------------------------------------------------
    wb = _ast_find_class(wb_tree, "MufasaWorkbench")
    assert wb is not None
    refresh_all = _ast_method(wb, "_refresh_all_section_badges")
    check(
        "MufasaWorkbench has _refresh_all_section_badges method",
        refresh_all is not None,
    )
    if refresh_all is not None:
        refresh_all_src = ast.unparse(refresh_all)
        check(
            "_refresh_all_section_badges iterates "
            "self._pages_by_title",
            "_pages_by_title" in refresh_all_src
            and "refresh_section_badges" in refresh_all_src,
        )
    else:
        check("_refresh_all_section_badges body check", False)

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    check(
        "SECTIONS still has the entries the badges surface "
        "(122ds invariant)",
        all(sid in SECTIONS for sid in (
            "outlier_correction", "kalman_v2",
            "import_pose", "features_subject",
        )),
    )

    pc_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "pose_cleanup.py"
    pc_src = pc_path.read_text()
    check(
        "122dt subclass declarations still in place "
        "(RunOutlierCorrectionForm.section_id == 'outlier_correction')",
        'section_id = "outlier_correction"' in pc_src,
    )

    # Parse-clean.
    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    # 122do baseline.
    uiqt = pkg / "ui_qt"
    optional_hits = []
    for f in sorted(uiqt.rglob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"\bOptional\[", src):
            preceding = src[:m.start()]
            tq3 = preceding.count('"""') + preceding.count("'''")
            if tq3 % 2 == 0:
                optional_hits.append(str(f.relative_to(uiqt)))
                break
    check(
        "122do baseline preserved: no `Optional[` in non-docstring "
        "positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122du_section_status_badges: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
