"""
tests/smoke_122bq_simba_variable_rename.py
============================================

Patch 122bq: Phase 2D of the identifier-rename lane. Renames
the `simba_*` lowercase local variables and function parameters
that refer to mufasa application state. Keeps `simba_*` names
that legitimately describe SimBA-format data, conversion
functions, or format identifiers.

Renames (4 names, 33 references)
--------------------------------
* simba_dir       → mufasa_dir       (19 refs, 6 files)
    Local var: `os.path.dirname(mufasa.__file__)` — install dir
* simba_cw        → mufasa_cw        (5 refs, 1 file)
    Same intent as simba_dir (different name in user_defined_pose_creator.py)
* simba_pip_data  → mufasa_pip_data  (4 refs, 1 file)
    Local var: PyPI data for the mufasa-uw-tf-dev package
* simba_ini_path  → mufasa_ini_path  (5 refs, 2 files)
    Function parameter: path to a Mufasa-INI project config

Plus one docstring update in `read_write.py` to match the
renamed parameter and clarify the format (the function imports
videos into a Mufasa project; the INI is the legacy project
format, not the SimBA project format specifically).

Kept (legitimate SimBA-format references)
-----------------------------------------
* simba_to_yolo_keypoints   — SimBA → YOLO conversion function
* simba_rois_to_yolo        — SimBA ROI → YOLO conversion function
* simba_roi_to_geometries   — SimBA ROI → Shapely geometry
* simba_blob_project        — SimBA blob format project type
* simba_blob                — SimBA blob format identifier
* simba_config_path         — file-select label for SimBA config in conversion popup
* simba_legacy              — migration source marker

Out of scope (separate cleanup)
-------------------------------
* simba_dev (10 refs) — only appears in commented-out dev-time
  paths like `/Users/simon/Desktop/envs/simba_dev/...` inside
  docstring examples and stale comments. Not a rename target;
  a separate cleanup lane would strip these stale paths.

Backward compat
---------------
Renamed identifiers are LOCAL variables and one function
parameter — no module-level binding to alias. The function
parameter rename (`simba_ini_path` → `mufasa_ini_path`) does
mean external callers passing the kwarg by name would need
to update. This is a pre-1.0 utility function used only
internally; the small risk is acceptable for code legibility.

Coverage
--------
1. Renamed identifiers are gone from non-docstring code.
2. KEEP names are preserved (>= 1 each).
3. simba_dev (out of scope) is preserved as-is.
4. All mufasa/**/*.py files parse cleanly.
5. read_write.py docstring matches the renamed parameter.
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


# Helper: walk a file's non-docstring lines
def non_docstring_lines(src: str):
    in_doc = False
    doc_q = None
    for line in src.splitlines():
        for q in ('"""', "'''"):
            if line.count(q) % 2 == 1:
                if not in_doc:
                    in_doc = True
                    doc_q = q
                elif doc_q == q:
                    in_doc = False
                    doc_q = None
        if not in_doc:
            yield line


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # 1. Renamed identifiers gone from non-docstring code
    # ==================================================================
    renamed = ["simba_dir", "simba_cw", "simba_pip_data", "simba_ini_path"]
    for old in renamed:
        pat = re.compile(rf"\b{old}\b")
        offenders = []
        for f in sorted(pkg.rglob("*.py")):
            for line in non_docstring_lines(f.read_text()):
                if pat.search(line):
                    offenders.append(f)
                    break
        check(
            f"Renamed identifier '{old}' is gone from non-docstring code",
            offenders == [],
            detail=(f"{len(offenders)} files; first: "
                    f"{offenders[0]}" if offenders else ""),
        )

    # ==================================================================
    # 2. Renamed identifiers appear with the new mufasa_* name
    # ==================================================================
    # Note (patch 122bx): mufasa_cw was defined only in
    # mufasa/ui/user_defined_pose_creator.py, which was deleted as
    # one of the two UNREFERENCED Tk files. Floor reduced to 0 —
    # the rename itself was valid; the file just no longer exists.
    new_names = ["mufasa_dir", "mufasa_cw", "mufasa_pip_data",
                 "mufasa_ini_path"]
    expected_floors = {
        "mufasa_dir": 6,      # 6 files originally had simba_dir
        "mufasa_cw": 0,       # only host file deleted in 122bx
        "mufasa_pip_data": 1,
        "mufasa_ini_path": 2,  # function def + caller
    }
    for new in new_names:
        n = sum(
            1 for f in pkg.rglob("*.py")
            if re.search(rf"\b{new}\b", f.read_text())
        )
        check(
            f"New name '{new}' appears in >= {expected_floors[new]} files "
            f"(got {n})",
            n >= expected_floors[new],
        )

    # ==================================================================
    # 3. KEEP names preserved
    # ==================================================================
    kept_names = {
        "simba_to_yolo_keypoints": 1,
        "simba_rois_to_yolo": 1,
        "simba_roi_to_geometries": 1,
        "simba_blob_project": 1,
        "simba_blob": 1,
        "simba_legacy": 1,
    }
    for name, floor in kept_names.items():
        n = sum(
            1 for f in pkg.rglob("*.py")
            if re.search(rf"\b{name}\b", f.read_text())
        )
        check(
            f"KEEP name '{name}' is preserved (>= {floor} files; got {n})",
            n >= floor,
        )

    # ==================================================================
    # 4. simba_dev preserved (out of scope)
    # ==================================================================
    simba_dev_count = sum(
        1 for f in pkg.rglob("*.py")
        if re.search(r"\bsimba_dev\b", f.read_text())
    )
    check(
        f"simba_dev (out of scope) is preserved "
        f"(>= 3 files; got {simba_dev_count})",
        simba_dev_count >= 3,
    )

    # ==================================================================
    # 5. All files parse cleanly
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

    # ==================================================================
    # 6. read_write.py docstring matches renamed parameter
    # ==================================================================
    rw = (pkg / "utils" / "read_write.py").read_text()
    check(
        "read_write.py: copy_single_video_to_project signature uses "
        "mufasa_ini_path",
        re.search(
            r"def copy_single_video_to_project\(\s*"
            r"mufasa_ini_path:", rw,
        ) is not None,
    )
    check(
        "read_write.py: copy_single_video_to_project docstring references "
        "mufasa_ini_path (matching signature)",
        ":param Union[str, os.PathLike] mufasa_ini_path:" in rw,
    )
    check(
        "read_write.py: no stale simba_ini_path docstring param",
        ":param Union[str, os.PathLike] simba_ini_path:" not in rw,
    )

    print(
        f"smoke_122bq_simba_variable_rename: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
