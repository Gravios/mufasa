"""
tests/smoke_122d9_qwi1_roi_apply_all_v1_path.py
=================================================

Patch 122d9: fix QWI-1 — multiply_ROIs() hard-coded the legacy
`<project>/videos/` path; v1 projects (videos at
`<root>/sources/videos/`) crashed with NotDirectoryError.

Coverage
--------
1.  roi_utils.py: no hard-coded `os.path.join(project_path,
    "videos")` in multiply_ROIs.
2.  roi_utils.py imports + uses `project_paths_from_config`.
3.  Stale "SimBA expected" wording replaced with
    "Mufasa expected" in the error message.
4.  qt_workbench_known_issues.md marks QWI-1 Fixed 122d9.
5.  Sibling audit results recorded in the doc.
6.  Whole-codebase: no NEW `os.path.join(project_path, "videos")`
    in roi_tools/ — the pre-122d9 single offender is gone.
7.  Parse-clean.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def _code_only(src: str) -> str:
    """Return source with comment-only lines stripped, so regex
    checks for "no hard-coded join" don't fire on breadcrumb
    comments that quote the pre-fix code for archaeology."""
    return "\n".join(
        line for line in src.split("\n")
        if not line.lstrip().startswith("#")
    )


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
    roi_utils_path = pkg / "roi_tools" / "roi_utils.py"
    roi_utils_src = roi_utils_path.read_text()
    roi_utils_code = _code_only(roi_utils_src)

    # 1. No hard-coded `os.path.join(project_path, "videos")` in
    # multiply_ROIs. Use AST-unparsed body (excludes comments
    # because ast.unparse drops them).
    multiply_rois_src = ""
    tree = ast.parse(roi_utils_src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "multiply_ROIs"):
            multiply_rois_src = ast.unparse(node)
            break
    check(
        "multiply_ROIs function defined",
        bool(multiply_rois_src),
    )
    check(
        "multiply_ROIs no longer uses hard-coded "
        "`os.path.join(project_path, \"videos\")` "
        "(QWI-1 root cause)",
        not re.search(
            r"os\.path\.join\([^)]*project_path[^)]*,\s*"
            r"['\"]videos['\"]\)",
            multiply_rois_src,
        ),
    )

    # 2. Uses project_paths_from_config
    check(
        "roi_utils.py imports project_paths_from_config",
        "project_paths_from_config" in roi_utils_src,
    )
    check(
        "multiply_ROIs uses project_paths_from_config to resolve "
        "video_dir",
        "project_paths_from_config" in multiply_rois_src
        and ('"video_dir"' in multiply_rois_src
             or "'video_dir'" in multiply_rois_src),
    )

    # 3. Error message wording updated (check code only, not
    # archaeological comments)
    check(
        "Error message uses 'Mufasa expected' (not stale "
        "'SimBA expected')",
        "Mufasa expected a directory" in roi_utils_code
        and "SimBA expected a directory" not in roi_utils_code,
    )

    # 4. Known-issues doc marks QWI-1 fixed
    qwi_doc = (REPO_ROOT / "docs"
               / "qt_workbench_known_issues.md").read_text()
    check(
        "qt_workbench_known_issues.md marks QWI-1 Fixed 122d9",
        "QWI-1" in qwi_doc
        and "Fixed 122d9" in qwi_doc,
    )

    # 5. Sibling-audit results in doc
    check(
        "QWI-1 section records sibling-audit findings "
        "(5 sites surveyed; data.py legacy sites deferred)",
        ("smooth_data_savitzky_golay" in qwi_doc
         or "data.py" in qwi_doc)
        and ("config_reader.py" in qwi_doc
             or "Legacy-branch-correct" in qwi_doc),
    )

    # 6. No NEW unfixed sites in roi_tools/ (code only — skip
    # archaeological comments that quote the pre-fix code)
    bad_sites = []
    for f in (pkg / "roi_tools").rglob("*.py"):
        src_code = _code_only(f.read_text())
        for m in re.finditer(
            r"os\.path\.join\([^)]*project_path[^)]*,\s*"
            r"['\"]videos['\"]\)",
            src_code,
        ):
            # Recover original line number — approximate by
            # finding the matching text in the original.
            bad_sites.append(str(f.relative_to(pkg)))
    check(
        f"No hard-coded videos-dir join in roi_tools/ "
        f"(got {len(bad_sites)} offenders: {bad_sites})",
        not bad_sites,
    )

    # 7. Parse-clean
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
        f"smoke_122d9_qwi1_roi_apply_all_v1_path: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
