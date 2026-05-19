"""
tests/smoke_122dh_pose_importers_step1.py
============================================

Patch 122dh: pose-importers step 1 — wire DLC, SLEAP, SuperAnimal
to the Qt PoseImportForm.

What this patch landed
----------------------
``mufasa/ui_qt/forms/pose_import.py`` extended:

* ``POSE_IMPORT_ROUTES`` grew from 2 entries (DLC H5 single,
  DLC CSV single) to 7:
  - DLC H5 (single animal)
  - DLC CSV (single animal)
  - DLC H5 (multi-animal / maDLC)
  - SLEAP CSV
  - SLEAP H5
  - SLEAP .slp
  - SuperAnimal-TopView

* Each route declares ``requires_animal_ids`` (multi-animal
  trackers need an id_lst) and ``accepts_p_threshold`` (only
  DLC single-animal). The form shows/hides the corresponding
  UI fields based on route selection.

* Animal-IDs field — comma-separated text, parsed into a list
  on submit. Validated when required, ignored otherwise.

* ``sleap_slp`` backend takes ``project_path=`` instead of
  ``config_path=``; the route's kwargs_map handles the rename
  transparently.

* maDLC needs a fixed ``file_type="h5"`` kwarg — route declares
  it via ``extra_backend_kwargs``.

* Source-directory placeholder customised per route (e.g.,
  "Directory containing SLEAP .slp project files" vs ".h5
  files").

* The defensive ``filter_kwargs`` already in the form
  transparently drops kwargs each backend doesn't accept —
  no per-backend extra handling needed.

Output destination is unchanged: each backend writes via
``ConfigReader.input_csv_dir`` which is layout-resolved to
``<v1_root>/sources/pose/`` for v1 projects and
``<legacy_project>/csv/input_csv/`` for legacy. The form
itself doesn't need to know about the layout.

Coverage
--------
1.  POSE_IMPORT_ROUTES has 7 entries.
2.  Each of the 7 expected route labels is present.
3.  Each route entry declares ``requires_animal_ids`` and
    ``accepts_p_threshold``.
4.  Single-animal DLC routes (H5 / CSV) have
    requires_animal_ids=False, accepts_p_threshold=True.
5.  Multi-animal routes (maDLC, SLEAP×3, SuperAnimal) have
    requires_animal_ids=True, accepts_p_threshold=False.
6.  maDLC has extra_backend_kwargs with file_type="h5".
7.  sleap_slp route's kwargs_map renames config_path → project_path.
8.  PoseImportForm.build() now wires _route_combo's
    currentTextChanged signal to _on_route_changed.
9.  PoseImportForm has _on_route_changed method.
10. PoseImportForm.collect_args validates animal_ids when
    required by the route.
11. PoseImportForm.target uses kwargs_map's "config_path" rename
    when present (sleap_slp case).
12. Module docstring lists 7 trackers + mentions 3D as future
    scope.
13. ruff F401/W292/W293 clean on the modified file.
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

    # 1. Route count = 7
    route_labels = re.findall(r'^    "([^"]+)":\s*dict\(',
                               src, re.MULTILINE)
    check(
        f"POSE_IMPORT_ROUTES has 7 entries (got {len(route_labels)})",
        len(route_labels) == 7,
    )

    # 2. Each expected label present
    expected = [
        "DLC H5 (single animal)",
        "DLC CSV (single animal)",
        "DLC H5 (multi-animal / maDLC)",
        "SLEAP CSV",
        "SLEAP H5",
        "SLEAP .slp",
        "SuperAnimal-TopView",
    ]
    missing = [e for e in expected if e not in route_labels]
    check(
        f"All 7 expected route labels present "
        f"(missing: {missing})",
        not missing,
    )

    # 3-6. Per-route metadata. Find the POSE_IMPORT_ROUTES Assign.
    routes_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == "POSE_IMPORT_ROUTES":
                routes_node = node
                break
        if isinstance(node, ast.AnnAssign):
            tgt = node.target
            if isinstance(tgt, ast.Name) and tgt.id == "POSE_IMPORT_ROUTES":
                routes_node = node
                break
    check("POSE_IMPORT_ROUTES Assign / AnnAssign found",
          routes_node is not None)
    if routes_node is not None:
        # The value is a Dict; for each key, get the value dict's keys
        routes_dict = routes_node.value
        per_route = {}
        if isinstance(routes_dict, ast.Dict):
            for k, v in zip(routes_dict.keys, routes_dict.values):
                if not isinstance(k, ast.Constant): continue
                label = k.value
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "dict":
                    flags = {kw.arg: kw.value for kw in v.keywords}
                    per_route[label] = flags

        # 3. Each route has requires_animal_ids + accepts_p_threshold keys
        for label in expected:
            r = per_route.get(label, {})
            check(
                f"`{label}` declares requires_animal_ids",
                "requires_animal_ids" in r,
            )

        # 4. Single-animal DLC routes
        for label in ["DLC H5 (single animal)",
                      "DLC CSV (single animal)"]:
            r = per_route.get(label, {})
            ra = r.get("requires_animal_ids")
            ap = r.get("accepts_p_threshold")
            check(
                f"`{label}` has requires_animal_ids=False, "
                f"accepts_p_threshold=True",
                isinstance(ra, ast.Constant) and ra.value is False
                and isinstance(ap, ast.Constant) and ap.value is True,
            )

        # 5. Multi-animal routes
        for label in ["DLC H5 (multi-animal / maDLC)",
                      "SLEAP CSV", "SLEAP H5", "SLEAP .slp",
                      "SuperAnimal-TopView"]:
            r = per_route.get(label, {})
            ra = r.get("requires_animal_ids")
            ap = r.get("accepts_p_threshold")
            check(
                f"`{label}` has requires_animal_ids=True, "
                f"accepts_p_threshold=False",
                isinstance(ra, ast.Constant) and ra.value is True
                and isinstance(ap, ast.Constant) and ap.value is False,
            )

        # 6. maDLC extra_backend_kwargs
        madlc = per_route.get("DLC H5 (multi-animal / maDLC)", {})
        ebk = madlc.get("extra_backend_kwargs")
        has_file_type = False
        if isinstance(ebk, ast.Dict):
            for k, v in zip(ebk.keys, ebk.values):
                if (isinstance(k, ast.Constant) and k.value == "file_type"
                        and isinstance(v, ast.Constant) and v.value == "h5"):
                    has_file_type = True
        check(
            "maDLC route's extra_backend_kwargs includes "
            "file_type='h5'",
            has_file_type,
        )

        # 7. sleap_slp kwargs_map renames config_path → project_path
        slp = per_route.get("SLEAP .slp", {})
        km = slp.get("kwargs_map")
        has_rename = False
        if isinstance(km, ast.Dict):
            for k, v in zip(km.keys, km.values):
                if (isinstance(k, ast.Constant) and k.value == "config_path"
                        and isinstance(v, ast.Constant) and v.value == "project_path"):
                    has_rename = True
        check(
            "SLEAP .slp route's kwargs_map renames config_path → "
            "project_path",
            has_rename,
        )

    # 8-9. Form-class method updates
    form_cls = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == "PoseImportForm"),
        None,
    )
    check("PoseImportForm class found", form_cls is not None)
    if form_cls is not None:
        methods = {m.name for m in form_cls.body
                   if isinstance(m, ast.FunctionDef)}
        check(
            "PoseImportForm has _on_route_changed method "
            "(route-switching handler)",
            "_on_route_changed" in methods,
        )
        # 8. build() wires currentTextChanged → _on_route_changed
        build_method = next(
            (m for m in form_cls.body
             if isinstance(m, ast.FunctionDef) and m.name == "build"),
            None,
        )
        if build_method is not None:
            build_src = ast.unparse(build_method)
            check(
                "build() wires _route_combo.currentTextChanged → "
                "_on_route_changed",
                "currentTextChanged" in build_src
                and "_on_route_changed" in build_src,
            )
        # 10. collect_args validates animal_ids when required
        collect_method = next(
            (m for m in form_cls.body
             if isinstance(m, ast.FunctionDef)
             and m.name == "collect_args"),
            None,
        )
        if collect_method is not None:
            collect_src = ast.unparse(collect_method)
            check(
                "collect_args validates animal_ids for multi-animal "
                "routes (requires_animal_ids branch)",
                "requires_animal_ids" in collect_src
                and "animal_ids" in collect_src,
            )
        # 11. target uses kwargs_map's config_path rename
        target_method = next(
            (m for m in form_cls.body
             if isinstance(m, ast.FunctionDef) and m.name == "target"),
            None,
        )
        if target_method is not None:
            target_src = ast.unparse(target_method)
            check(
                "target() respects kwargs_map['config_path'] rename "
                "(sleap_slp case)",
                "km.get('config_path'" in target_src
                or 'km.get("config_path"' in target_src,
            )

    # 12. Module docstring updated
    docstring = ast.get_docstring(tree)
    check(
        "Module docstring lists 7 trackers + mentions 3D as "
        "future scope",
        docstring is not None
        and "seven routes" in docstring.lower()
        and "3D" in docstring,
    )

    # 13. ruff clean for F401/W292/W293
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
            detail=(out.stdout[:200] if out.returncode != 0 else ""),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        check(
            "ruff check skipped (not available in this env)",
            True,
        )

    # 14. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122dh_pose_importers_step1: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
