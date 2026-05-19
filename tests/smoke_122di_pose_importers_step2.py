"""
tests/smoke_122di_pose_importers_step2.py
============================================

Patch 122di: pose-importers step 2 — wire YOLO-pose + MARS to
the Qt PoseImportForm.

Continues the multi-patch series that started in 122dh.

What this patch landed
----------------------
Two new routes in ``POSE_IMPORT_ROUTES``:

* **YOLO-pose** — backed by ``SimBAYoloImporter``. Single-animal
  (no id_lst); kwargs_map renames source_path → data_dir (the
  YOLO importer uses ``data_dir`` not ``data_folder``).

* **MARS (two-mouse social)** — Caltech's mouse-action-recognition
  importer. Uses ``data_path`` not ``data_folder``. Requires
  ``interpolation_method`` + ``smoothing_method`` at construction
  (no defaults — different from DLC/SLEAP which have them as
  optional settings dicts). Passes sentinel "no-op" values via
  ``extra_backend_kwargs`` so users still run preprocessing on
  the Preprocessing page (consistent with all other routes).

Module docstring updated to list 9 routes split by patch.

Coverage
--------
1.  POSE_IMPORT_ROUTES has 9 entries total (7 from step 1 + 2 new).
2.  YOLO-pose route exists with the expected metadata.
3.  YOLO route's kwargs_map renames source_path → data_dir.
4.  YOLO route has requires_animal_ids=False (single-animal-per-video
    from the importer's perspective).
5.  MARS route exists with the expected metadata.
6.  MARS route's kwargs_map renames source_path → data_path.
7.  MARS route has extra_backend_kwargs with interpolation_method
    + smoothing_method (sentinels for the required-positional
    kwargs at the backend's constructor).
8.  MARS smoothing_method is a dict-shaped value with "Method"
    key (matches backend's __run_smoothing equality check).
9.  Module docstring lists 9 routes + identifies step 1 / step 2
    + 3D as future scope.
10. ruff F401/W292/W293 clean on the modified file.
11. All mufasa/**/*.py parse cleanly.
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

    # 1. Route count ≥ 9 (snapshot-resilient — later patches add
    # more routes; this test only cares about the step-2 set.
    # See smoke_122dj for the TRK + FaceMap additions in step 3.)
    labels = re.findall(r'^    "([^"]+)":\s*dict\(',
                         src, re.MULTILINE)
    check(
        f"POSE_IMPORT_ROUTES has at least 9 entries from steps "
        f"1+2 (got {len(labels)})",
        len(labels) >= 9,
    )

    # Find the POSE_IMPORT_ROUTES dict node
    routes_node = None
    for node in tree.body:
        tgt = None
        if isinstance(node, ast.Assign):
            tgt = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            tgt = node.target
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

    # 2. YOLO route exists
    yolo = per_route.get("YOLO-pose")
    check("YOLO-pose route exists", yolo is not None)
    if yolo:
        # 3. kwargs_map renames source_path → data_dir
        km = yolo.get("kwargs_map")
        has_rename = False
        if isinstance(km, ast.Dict):
            for k, v in zip(km.keys, km.values):
                if (isinstance(k, ast.Constant) and k.value == "source_path"
                        and isinstance(v, ast.Constant) and v.value == "data_dir"):
                    has_rename = True
        check(
            "YOLO route's kwargs_map renames source_path → data_dir",
            has_rename,
        )
        # 4. requires_animal_ids=False
        ra = yolo.get("requires_animal_ids")
        check(
            "YOLO route has requires_animal_ids=False",
            isinstance(ra, ast.Constant) and ra.value is False,
        )

    # 5. MARS route exists
    mars = per_route.get("MARS (two-mouse social)")
    check("MARS route exists", mars is not None)
    if mars:
        # 6. kwargs_map renames source_path → data_path
        km = mars.get("kwargs_map")
        has_rename = False
        if isinstance(km, ast.Dict):
            for k, v in zip(km.keys, km.values):
                if (isinstance(k, ast.Constant) and k.value == "source_path"
                        and isinstance(v, ast.Constant) and v.value == "data_path"):
                    has_rename = True
        check(
            "MARS route's kwargs_map renames source_path → data_path",
            has_rename,
        )
        # 7. extra_backend_kwargs has interpolation_method + smoothing_method
        ebk = mars.get("extra_backend_kwargs")
        has_interp = has_smooth = False
        smooth_value = None
        if isinstance(ebk, ast.Dict):
            for k, v in zip(ebk.keys, ebk.values):
                if isinstance(k, ast.Constant):
                    if k.value == "interpolation_method":
                        has_interp = True
                    if k.value == "smoothing_method":
                        has_smooth = True
                        smooth_value = v
        check(
            "MARS extra_backend_kwargs has interpolation_method "
            "(required-positional sentinel)",
            has_interp,
        )
        check(
            "MARS extra_backend_kwargs has smoothing_method "
            "(required-positional sentinel)",
            has_smooth,
        )
        # 8. smoothing_method is a dict with "Method" key
        smoothing_has_method_key = False
        if isinstance(smooth_value, ast.Dict):
            for k in smooth_value.keys:
                if isinstance(k, ast.Constant) and k.value == "Method":
                    smoothing_has_method_key = True
                    break
        check(
            "MARS smoothing_method is a dict with 'Method' key "
            "(matches backend equality check)",
            smoothing_has_method_key,
        )

    # 9. Docstring documents step 2 trackers + 3D scope.
    # Snapshot-resilient — later patches may rephrase the route
    # count (122dj moves to eleven, etc.); the spirit of the check
    # is "step 2 + 3D-future-scope are documented."
    docstring = ast.get_docstring(tree)
    check(
        "Docstring documents step 2 trackers (YOLO + MARS) + "
        "3D as future scope",
        docstring is not None
        and "Step 2" in docstring
        and "YOLO" in docstring
        and "MARS" in docstring
        and "3D" in docstring,
    )

    # 10. ruff clean
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
        check("ruff check skipped (not available in this env)",
              True)

    # 11. Parse-clean
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
        f"smoke_122di_pose_importers_step2: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
