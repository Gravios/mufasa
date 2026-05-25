"""
tests/smoke_122eh_roi_coordinates_path_hotfix.py
==================================================

Patch 122eh — hotfix: align
:attr:`ConfigReader.roi_coordinates_path` with the actual ROI
save location used by
:func:`mufasa.project_layout.project_paths_from_config`. Fixes a
regression introduced by 122dy.

Real-world report (Fri May 22, 2026)
------------------------------------
User clicked "Duplicate ROIs" in the workbench's ROI page on a
v1 project where ROIs HAD been defined. Got::

    NoFilesFoundError: NO FILES FOUND ERROR: Cannot duplicate
    ROIs: no ROI definitions file found in this project.

Root cause
----------
Two helpers in the codebase compute the ROI definitions file
path and disagreed on v1 projects:

* ``mufasa.mixins.config_reader.ConfigReader._resolve_v1_paths``
  set::
    self.roi_coordinates_path = root / "logs" / "roi_definitions.h5"

* ``mufasa.project_layout.project_paths_from_config`` returned::
    "roi_definitions_path": root / "logs" / "measures"
                            / "ROI_definitions.h5"

The save-side code (``mufasa.roi_tools.roi_logic.RoiLogic``) goes
through ``project_paths_from_config`` — writes to
``<root>/logs/measures/ROI_definitions.h5``.

The reader-side code (duplicate dialog, ROI size standardizer,
``ConfigReader.read_roi_data``) goes through ConfigReader's
``roi_coordinates_path`` — read from ``<root>/logs/
roi_definitions.h5``, which never existed. NoFilesFoundError.

Why it didn't fire pre-122dy
----------------------------
Pre-122dy, ``ConfigReader.__init__`` had a line AFTER
``_apply_v1_path_overrides``::

    self.roi_coordinates_path = os.path.join(
        self.logs_path, Paths.ROI_DEFINITIONS.value,
    )

with ``Paths.ROI_DEFINITIONS = "measures/ROI_definitions.h5"``.
This OVERWROTE the v1-path-override's wrong value with the
legacy-flavored path that happened to match the save location.

122dy ripped this line as "a silent v1-path-clobbering bug."
The diagnosis was correct (the duplication shouldn't exist) but
the removal was incomplete — should have FIXED the
``_resolve_v1_paths`` value to the correct path, not just
deleted the clobber. 122eh completes the fix.

Why a hotfix and not a centralization
-------------------------------------
The root issue is ConfigReader and project_paths_from_config
both encoding the v1 layout independently. They drift; this is
the second time they've drifted (the first was the
``annotated_frm_dir`` /
``single_validation_video_save_dir`` deferred audit fixed in
122ea — those weren't covered by either helper, leading to
the legacy-shape fallback in ``__init__``).

The right long-term fix is to have ONE canonical layout helper
that both ConfigReader and ``project_paths_from_config`` call.
That's a refactor that touches both files non-trivially. For
this user-blocking crash, the hotfix is the right move; the
centralization is filed as a deferred follow-up.

What this patch landed
----------------------
mufasa/mixins/config_reader.py — single attribute change in
``_resolve_v1_paths``:

* ``self.roi_coordinates_path`` now resolves to
  ``<root>/logs/measures/ROI_definitions.h5`` (matches
  ``project_paths_from_config``'s ``roi_definitions_path``).
* Docstring path-mapping line updated.
* Method history-bullets section gained a 122eh entry
  explaining the regression and fix.
* In-source comment block at the assignment site explains
  the duplication and points at the centralization deferred-
  item.

Coverage
--------
1.  ``ConfigReader._resolve_v1_paths`` sets
    ``roi_coordinates_path`` to a path ending in
    ``logs/measures/ROI_definitions.h5`` (with capital R, O,
    I and the ``measures/`` subdirectory).
2.  The ConfigReader-side path matches
    ``project_paths_from_config``'s
    ``roi_definitions_path`` value (i.e., the two helpers are
    in agreement post-hotfix).
3.  ``_resolve_v1_paths`` docstring path-mapping line cites
    the correct path.
4.  ``_resolve_v1_paths`` history section mentions 122eh.

Cross-patch invariants:
5.  122eg state preserved: no stray
    ``project_config.ini`` references in mufasa/**/*.py
    outside legacy_layout.py.
6.  122ee state preserved: PoseImportForm publish wiring.
7.  122ef state preserved: workbench hasattr guard.
8.  Parse-clean.
9.  122do baseline.
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


def main() -> int:
    cr_path = REPO_ROOT / "mufasa" / "mixins" / "config_reader.py"
    cr_src = cr_path.read_text()
    cr_tree = ast.parse(cr_src)

    # Locate the _resolve_v1_paths method.
    cls = None
    for node in ast.walk(cr_tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "ConfigReader"):
            cls = node
            break
    assert cls is not None
    method = None
    for node in cls.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_resolve_v1_paths"):
            method = node
            break
    assert method is not None
    method_src = ast.unparse(method)

    # 1. roi_coordinates_path resolves correctly. Post-122en this
    # is centralized — ConfigReader assigns from
    # ``_paths["roi_definitions_path"]`` where ``_paths`` comes
    # from ``v1_project_paths(root)``. So we check two things:
    #   (a) the assignment in _resolve_v1_paths uses the canonical
    #       dict key (catches drift from the centralization);
    #   (b) v1_project_paths itself returns the right path
    #       (checked in check 2 below — agreement check).
    pl_path = REPO_ROOT / "mufasa" / "project_layout.py"
    pl_src = pl_path.read_text()
    routes_via_helper = (
        '_paths["roi_definitions_path"]' in method_src
        or "_paths['roi_definitions_path']" in method_src
    )
    check(
        "ConfigReader._resolve_v1_paths assigns "
        "roi_coordinates_path from "
        "_paths['roi_definitions_path'] (the canonical helper "
        "key from 122en). Pre-122en this was an inline "
        "str(root / 'logs' / 'measures' / 'ROI_definitions.h5') "
        "construction; centralization moved the literal into "
        "v1_project_paths.",
        routes_via_helper,
        detail=("substring not found in method body"),
    )

    # 2. ConfigReader-side path matches project_paths_from_config.
    # Post-122en they're guaranteed to match (both go through
    # v1_project_paths), but we still verify the canonical helper
    # produces the right literal — catches drift in
    # v1_project_paths itself.
    pl_roi_match = re.search(
        r'"roi_definitions_path":\s*str\s*\([^)]*\)',
        pl_src,
    )
    pl_has_correct = False
    if pl_roi_match:
        b = pl_roi_match.group(0)
        pl_has_correct = (
            '"logs"' in b and '"measures"' in b
            and '"ROI_definitions.h5"' in b
        )
    check(
        "project_paths_from_config's roi_definitions_path also "
        "resolves to logs/measures/ROI_definitions.h5 (verifies "
        "the two helpers are in agreement post-hotfix; if a "
        "future patch changes one helper, this check surfaces "
        "the drift)",
        pl_has_correct,
        detail=(f"project_layout block: "
                f"{pl_roi_match.group(0) if pl_roi_match else '(none)'}"),
    )

    # 3. Docstring path-mapping line.
    method_doc = ast.get_docstring(method) or ""
    check(
        "_resolve_v1_paths docstring path-mapping line cites "
        "'logs/measures/ROI_definitions.h5'",
        "logs/measures/ROI_definitions.h5" in method_doc,
    )

    # 4. History section mentions 122eh.
    check(
        "_resolve_v1_paths docstring history mentions 122eh "
        "(audit breadcrumb for the regression-and-fix)",
        "122eh" in method_doc,
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    pkg = REPO_ROOT / "mufasa"

    # 5. 122eg state preserved.
    stray = []
    for f in sorted(pkg.rglob("*.py")):
        rel = str(f.relative_to(REPO_ROOT))
        if rel == "mufasa/legacy_layout.py":
            continue
        s = f.read_text()
        if "project_config.ini" in s:
            stray.append(rel)
    check(
        "122eg state preserved: no `project_config.ini` "
        "references in mufasa/**/*.py outside legacy_layout.py",
        not stray,
        detail=("; ".join(stray[:3])),
    )

    # 6. 122ee state preserved.
    pi_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "forms" / "pose_import.py").read_text()
    check(
        "122ee state preserved: PoseImportForm has publish_target_stage",
        'publish_target_stage = "outlier_corrected"' in pi_src,
    )

    # 7. 122ef state preserved.
    wb_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "workbench.py").read_text()
    has_guard = bool(re.search(
        r'hasattr\s*\(\s*form\s*,\s*["\']completed["\']\s*\)',
        wb_src,
    ))
    check(
        "122ef-hotfix state preserved: workbench.py has the "
        "`hasattr(form, 'completed')` guard",
        has_guard,
    )

    # 8. Parse-clean.
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

    # 9. 122do baseline.
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
        f"smoke_122eh_roi_coordinates_path_hotfix: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
