"""
tests/smoke_122bz_backend_audit.py
====================================

Patch 122bz: backend audit. Documentation-only patch with two
AST-based audit passes:

* §1 — for each failing operation flagged by qt_form_runtime_gaps.md,
  search the codebase for the actual backend.
* §2 — inventory the 25 backend modules that import from
  mufasa.ui.tkinter_functions.

Key finding: 5 of 7 "missing" backends actually exist under
different names. Only 2 (blur, brightness/contrast) are
genuinely absent.

Coverage
--------
1. docs/backend_audit.md exists.
2. The audit names all 5 backends that turned out to be
   findable: create_average_frm, video_to_bw, KeypointRemover,
   remove_roi_features, MultiCropper.
3. The audit names the 2 genuinely-missing categories
   (Box/Gaussian blur, Brightness/contrast).
4. The audit has the four major sections (§1 Unwired-backend
   gaps, §2 Backend modules with embedded Tk UI, §3 Combined
   cleanup plan, §4 Audit methodology).
5. The audit catalogs all 4 disposition categories for the
   25 Tk-importing backend modules (Group A/B/C/D).
6. The audit has a Quick Wins subsection (§4a) listing fixes
   under 1 hour.
7. docs/README.md references the new audit.
8. The companion docs (qt_form_runtime_gaps.md,
   tk_surface_audit.md, tk_to_qt_consolidation_plan.md) are
   cross-referenced.
9. All mufasa/**/*.py files parse cleanly.

Plus a runtime check: verify the 5 "actually-exists" backends
really do exist (regression guard against future renames that
would make this audit lie).
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
    audit_path = REPO_ROOT / "docs" / "backend_audit.md"
    check("docs/backend_audit.md exists", audit_path.exists())
    if not audit_path.exists():
        return 1
    text = audit_path.read_text()

    # ==================================================================
    # 1. Names of findable backends
    # ==================================================================
    findable = ["create_average_frm", "video_to_bw",
                "KeypointRemover", "remove_roi_features",
                "MultiCropper"]
    for name in findable:
        check(
            f"audit names findable backend '{name}'",
            name in text,
        )

    # ==================================================================
    # 2. Names of genuinely-missing backends
    # ==================================================================
    for category in ["Box / Gaussian blur", "Brightness/Contrast"]:
        check(
            f"audit names genuinely-missing category '{category}'",
            category in text,
        )

    # ==================================================================
    # 3. Major sections
    # ==================================================================
    for section in [
        "Unwired-backend gaps",
        "Backend modules with embedded Tk UI",
        "Combined cleanup plan",
        "Audit methodology",
        "Quick wins",
    ]:
        check(
            f"audit contains section '{section}'",
            section in text or section.lower() in text.lower(),
        )

    # ==================================================================
    # 4. All four disposition groups
    # ==================================================================
    for group in ["Group A", "Group B", "Group C", "Group D"]:
        check(
            f"audit references disposition '{group}'",
            group in text,
        )

    # ==================================================================
    # 5. Companion doc cross-references
    # ==================================================================
    for ref in ["qt_form_runtime_gaps.md", "tk_surface_audit.md",
                "tk_to_qt_consolidation_plan.md"]:
        check(
            f"audit cross-references '{ref}'",
            ref in text,
        )

    # ==================================================================
    # 6. docs/README.md indexes the audit
    # ==================================================================
    readme = REPO_ROOT / "docs" / "README.md"
    check(
        "docs/README.md references backend_audit.md",
        "backend_audit.md" in readme.read_text(),
    )

    # ==================================================================
    # 7. Regression guard: the 5 "actually-exists" backends really exist
    # ==================================================================
    pkg = REPO_ROOT / "mufasa"
    found = {name: False for name in findable}
    for f in pkg.rglob("*.py"):
        if "ui_qt" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in found:
                    found[node.name] = True
    for name, was_found in found.items():
        check(
            f"backend '{name}' really exists in the codebase "
            f"(regression guard)",
            was_found,
        )

    # ==================================================================
    # 8. All files parse cleanly
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

    print(
        f"smoke_122bz_backend_audit: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
