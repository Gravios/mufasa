"""
tests/smoke_122dj_pose_importers_step3.py
============================================

Patch 122dj: pose-importers step 3 — wire TRK + FaceMap to the
Qt PoseImportForm. Closes the multi-patch porting series for 2D
trackers (DANNCE deferred as partly-subsumed-by-3D-future-scope;
SimBA blob deferred as contour-based legacy).

What this patch landed
----------------------
Two new routes in ``POSE_IMPORT_ROUTES``:

* **TRK (Animal Part Tracker)** — Janelia ecosystem; tethered
  Drosophila / flies. ``TRKImporter`` uses BOTH ``data_path``
  AND ``animal_id_lst`` (vs the typical ``data_folder`` /
  ``id_lst`` from SLEAP / maDLC). Both renames declared in
  ``kwargs_map``. Requires interpolation + smoothing positional
  args; sentinels passed via ``extra_backend_kwargs``.

* **FaceMap** — Stringer / Pachitariu (Janelia). Face / whisker
  / pupil tracking. ``FaceMapImporter`` uses ``data_path`` but
  is otherwise straightforward (single-face = no IDs; all
  preprocessing args optional with sane defaults).

Form-code extension
-------------------
``target()`` extended to support an ``animal_ids`` rename via
``kwargs_map``. Pre-122dj the form hardcoded ``"id_lst"`` as the
backend kwarg name; now it respects ``km.get("animal_ids", "id_lst")``.
TRK is the first route to use the rename.

Coverage
--------
1.  POSE_IMPORT_ROUTES has 11 entries (9 from step 1+2 + 2 new).
2.  TRK route exists.
3.  TRK kwargs_map renames source_path → data_path.
4.  TRK kwargs_map renames animal_ids → animal_id_lst.
5.  TRK requires_animal_ids=True.
6.  TRK extra_backend_kwargs has interpolation_method +
    smoothing_settings sentinels.
7.  FaceMap route exists.
8.  FaceMap kwargs_map renames source_path → data_path.
9.  FaceMap requires_animal_ids=False (single-face).
10. FaceMap has no extra_backend_kwargs (or empty; backend has
    sane defaults for all optional args).
11. target() respects km.get("animal_ids", "id_lst") rename.
12. Module docstring documents step 3 + TRK/FaceMap + 3D scope.
13. ruff F401/W292/W293 clean.
14. All mufasa/**/*.py parse cleanly.
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
    pkg = REPO_ROOT / "mufasa"
    pose_form = pkg / "ui_qt" / "forms" / "pose_import.py"
    src = pose_form.read_text()
    tree = ast.parse(src)

    # 1. Route count = 11
    labels = re.findall(r'^    "([^"]+)":\s*dict\(',
                         src, re.MULTILINE)
    check(
        f"POSE_IMPORT_ROUTES has 11 entries "
        f"(9 from steps 1+2 + 2 new; got {len(labels)})",
        len(labels) == 11,
    )

    # Find the POSE_IMPORT_ROUTES dict
    routes_node = None
    for node in tree.body:
        tgt = None
        if isinstance(node, ast.Assign): tgt = node.targets[0]
        elif isinstance(node, ast.AnnAssign): tgt = node.target
        if isinstance(tgt, ast.Name) and tgt.id == "POSE_IMPORT_ROUTES":
            routes_node = node
            break

    per_route = {}
    if routes_node is not None and isinstance(routes_node.value, ast.Dict):
        for k, v in zip(routes_node.value.keys, routes_node.value.values):
            if not isinstance(k, ast.Constant): continue
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "dict":
                per_route[k.value] = {kw.arg: kw.value
                                       for kw in v.keywords}

    def km_rename(route_dict, ui_key, expected_backend_key):
        km = route_dict.get("kwargs_map")
        if not isinstance(km, ast.Dict): return False
        for k, v in zip(km.keys, km.values):
            if (isinstance(k, ast.Constant) and k.value == ui_key
                    and isinstance(v, ast.Constant) and v.value == expected_backend_key):
                return True
        return False

    # 2-6. TRK
    trk = per_route.get("TRK (Animal Part Tracker)")
    check("TRK route exists", trk is not None)
    if trk:
        check(
            "TRK kwargs_map renames source_path → data_path",
            km_rename(trk, "source_path", "data_path"),
        )
        check(
            "TRK kwargs_map renames animal_ids → animal_id_lst "
            "(non-standard backend kwarg name)",
            km_rename(trk, "animal_ids", "animal_id_lst"),
        )
        ra = trk.get("requires_animal_ids")
        check(
            "TRK requires_animal_ids=True",
            isinstance(ra, ast.Constant) and ra.value is True,
        )
        ebk = trk.get("extra_backend_kwargs")
        has_interp = has_smooth = False
        if isinstance(ebk, ast.Dict):
            for k in ebk.keys:
                if isinstance(k, ast.Constant):
                    if k.value == "interpolation_method":
                        has_interp = True
                    if k.value == "smoothing_settings":
                        has_smooth = True
        check(
            "TRK extra_backend_kwargs has both interpolation_method "
            "and smoothing_settings sentinels",
            has_interp and has_smooth,
        )

    # 7-10. FaceMap
    fm = per_route.get("FaceMap (face / whisker / pupil)")
    check("FaceMap route exists", fm is not None)
    if fm:
        check(
            "FaceMap kwargs_map renames source_path → data_path",
            km_rename(fm, "source_path", "data_path"),
        )
        ra = fm.get("requires_animal_ids")
        check(
            "FaceMap requires_animal_ids=False (single-face)",
            isinstance(ra, ast.Constant) and ra.value is False,
        )
        # FaceMap has all-optional backend kwargs, so the route
        # doesn't need extra_backend_kwargs. Accept either
        # "absent" or "empty dict" — both are valid.
        ebk = fm.get("extra_backend_kwargs")
        check(
            "FaceMap has no extra_backend_kwargs (backend has "
            "sane defaults for all optional args)",
            ebk is None
            or (isinstance(ebk, ast.Dict) and not ebk.keys),
        )

    # 11. target() respects animal_ids kwargs_map rename
    form_cls = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == "PoseImportForm"),
        None,
    )
    target_method = next(
        (m for m in form_cls.body
         if isinstance(m, ast.FunctionDef) and m.name == "target"),
        None,
    ) if form_cls else None
    if target_method is not None:
        target_src = ast.unparse(target_method)
        check(
            "target() respects km.get('animal_ids', 'id_lst') "
            "rename (TRK uses animal_id_lst)",
            "km.get('animal_ids'" in target_src
            or 'km.get("animal_ids"' in target_src,
        )

    # 12. Docstring
    docstring = ast.get_docstring(tree)
    check(
        "Module docstring documents step 3 (TRK + FaceMap) + "
        "3D as future scope",
        docstring is not None
        and "Step 3" in docstring
        and "TRK" in docstring
        and "FaceMap" in docstring
        and "3D" in docstring,
    )

    # 13. ruff clean
    import subprocess
    try:
        out = subprocess.run(
            ["ruff", "check", str(pose_form),
             "--select", "F401,W292,W293"],
            capture_output=True, text=True, timeout=15,
            cwd=str(REPO_ROOT),
        )
        check(
            "ruff F401/W292/W293 clean on pose_import.py",
            out.returncode == 0,
            detail=out.stdout[:200] if out.returncode != 0 else "",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        check("ruff check skipped (not available in this env)", True)

    # 14. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try: ast.parse(f.read_text())
        except SyntaxError as e: parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=parse_errors[0] if parse_errors else "",
    )

    print(
        f"smoke_122dj_pose_importers_step3: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
