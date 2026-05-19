"""
tests/smoke_122da_hardwired_paths_audit.py
============================================

Patch 122da: comprehensive hardwired-paths audit + sibling-miss
fixes for the 122d9 (QWI-1) work.

What this patch landed
----------------------
- `docs/hardwired_paths_audit.md` — comprehensive scan results
  with disposition for every flagged site.
- `mufasa/roi_tools/roi_utils.py:474` (multiply_ROIs
  roi_coordinates_path) — sibling miss from 122d9; now routed
  through `project_paths_from_config(...)["roi_definitions_path"]`.
- `mufasa/roi_tools/roi_utils.py:561` (reset_video_ROIs
  roi_coordinates_path) — same sibling pattern in a sister
  function.

Coverage
--------
1.  Audit doc exists.
2.  Audit doc names the path-abstraction layer
    (`project_paths_from_config`).
3.  Audit doc records the disposition split (107 hits → 38
    potential bugs → 4 actual Qt-reachable).
4.  multiply_ROIs no longer hardcodes `os.path.join(project_path,
    "logs", …)` for roi_coordinates_path.
5.  reset_video_ROIs no longer hardcodes the same pattern.
6.  Both functions now use `project_paths_from_config(...)
    ["roi_definitions_path"]`.
7.  The 122d9 fix for L462 is preserved (regression guard).
8.  Audit doc enumerates the "intentional design" sites
    (clip_review, targeted_clips, etc.) so reviewers know they
    were considered, not missed.
9.  Parse-clean.
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


def _code_only(src: str) -> str:
    """Strip comment-only lines so substring/regex checks don't
    trigger on archaeological breadcrumb comments quoting pre-fix
    code. Same pattern established in 122d9."""
    return "\n".join(
        line for line in src.split("\n")
        if not line.lstrip().startswith("#")
    )


def main() -> int:
    pkg = REPO_ROOT / "mufasa"
    audit_path = REPO_ROOT / "docs" / "hardwired_paths_audit.md"

    # 1. Audit doc exists
    check("docs/hardwired_paths_audit.md exists",
          audit_path.exists())
    if not audit_path.exists():
        print(f"smoke_122da: {CHECKS_PASSED}/{CHECKS_TOTAL}")
        return 1
    audit_src = audit_path.read_text()

    # 2. Names the abstraction layer
    check(
        "Audit doc names project_paths_from_config as the "
        "abstraction layer",
        "project_paths_from_config" in audit_src
        and "abstraction layer" in audit_src.lower(),
    )

    # 3. Records the disposition split
    check(
        "Audit doc records the scan counts (107 hits / 38 "
        "potential bugs)",
        "107" in audit_src and "38" in audit_src,
    )

    # 4. multiply_ROIs no longer hardcodes the legacy join
    # (use AST-unparsed function body so comments are dropped)
    roi_utils_src = (pkg / "roi_tools" / "roi_utils.py").read_text()
    roi_utils_tree = ast.parse(roi_utils_src)

    def _function_body(name: str) -> str:
        for node in ast.walk(roi_utils_tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == name):
                return ast.unparse(node)
        return ""

    multiply_body = _function_body("multiply_ROIs")
    reset_body = _function_body("reset_video_ROIs")
    check("multiply_ROIs function defined", bool(multiply_body))
    check("reset_video_ROIs function defined", bool(reset_body))

    legacy_logs_join = re.compile(
        r"os\.path\.join\([^)]*project_path[^)]*,\s*"
        r"['\"]logs['\"]"
    )

    check(
        "multiply_ROIs no longer uses hardcoded "
        "`os.path.join(project_path, \"logs\", …)` for "
        "roi_coordinates_path (122da sibling fix)",
        not legacy_logs_join.search(multiply_body),
    )

    # 5. reset_video_ROIs same
    check(
        "reset_video_ROIs no longer uses hardcoded "
        "`os.path.join(project_path, \"logs\", …)` (122da)",
        not legacy_logs_join.search(reset_body),
    )

    # 6. Both functions use the helper
    check(
        "multiply_ROIs uses project_paths_from_config with "
        "roi_definitions_path",
        "project_paths_from_config" in multiply_body
        and ("'roi_definitions_path'" in multiply_body
             or '"roi_definitions_path"' in multiply_body),
    )
    check(
        "reset_video_ROIs uses project_paths_from_config with "
        "roi_definitions_path",
        "project_paths_from_config" in reset_body
        and ("'roi_definitions_path'" in reset_body
             or '"roi_definitions_path"' in reset_body),
    )

    # 7. 122d9 fix preserved (regression guard)
    check(
        "multiply_ROIs uses video_dir from the helper (122d9 "
        "fix preserved)",
        ("'video_dir'" in multiply_body
         or '"video_dir"' in multiply_body),
    )

    # 8. Audit doc enumerates intentional designs
    check(
        "Audit doc lists clip_review.py:313 as intentional design",
        "clip_review.py:313" in audit_src
        and "intentional" in audit_src.lower(),
    )
    check(
        "Audit doc lists targeted_clips.py:142 as intentional",
        "targeted_clips.py:142" in audit_src,
    )

    # 9. Parse-clean
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
        f"smoke_122da_hardwired_paths_audit: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
