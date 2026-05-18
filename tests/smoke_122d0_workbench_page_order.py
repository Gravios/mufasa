"""
tests/smoke_122d0_workbench_page_order.py
===========================================

Patch 122d0: known-issues tracking doc + QWI-4 fix
(workbench page order: Classifier before Annotation).

Doc-only-plus-trivial-fix patch. Captures 4 Qt workbench bugs
surfaced in real-world use (Stage B-orthogonal — they're in the
already-shipped Qt code, not the Tk surface being deleted), and
applies the 2-line fix for the most trivial of the four (page
ordering).

The other 3 bugs are tracked with concrete file:line pointers
and recommended fixes for future patches.

Coverage
--------
1.  docs/qt_workbench_known_issues.md exists.
2.  All 4 issues recorded in the doc with QWI-N identifiers.
3.  QWI-4 marked Fixed in the doc.
4.  workbench_app.py: Classifier page registration appears
    before Annotation page registration (line-order check).
5.  Comment block in workbench_app.py references QWI-4 / 122d0
    so the swap doesn't look accidental in future archaeology.
6.  AST: both build_classifier_page and build_annotation_page
    calls are still present in workbench_app.py's main function.
7.  All mufasa/**/*.py files parse cleanly.
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

    # 1. Doc exists
    doc = REPO_ROOT / "docs" / "qt_workbench_known_issues.md"
    check("docs/qt_workbench_known_issues.md exists", doc.exists())

    # 2. All 4 issues recorded
    doc_src = doc.read_text() if doc.exists() else ""
    for qwi_id in ("QWI-1", "QWI-2", "QWI-3", "QWI-4"):
        check(
            f"{qwi_id} recorded in known-issues doc",
            qwi_id in doc_src,
        )

    # 3. QWI-4 marked Fixed
    check(
        "QWI-4 marked Fixed 122d0",
        "Fixed 122d0" in doc_src,
    )

    # 4. Page order: classifier before annotation in workbench_app.py
    wba_src = (pkg / "ui_qt" / "workbench_app.py").read_text()
    lines = wba_src.split("\n")
    classifier_line = None
    annotation_line = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("build_classifier_page("):
            if classifier_line is None:  # first occurrence
                classifier_line = i
        if line.lstrip().startswith("build_annotation_page("):
            if annotation_line is None:
                annotation_line = i
    check(
        f"workbench_app.py: build_classifier_page line ({classifier_line}) "
        f"comes before build_annotation_page line ({annotation_line})",
        classifier_line is not None
        and annotation_line is not None
        and classifier_line < annotation_line,
    )

    # 5. Comment block references the fix
    check(
        "workbench_app.py comment block references QWI-4 / 122d0 "
        "(provenance for the swap)",
        ("122d0" in wba_src and "QWI-4" in wba_src)
        or ("Classifier moved BEFORE Annotation" in wba_src),
    )

    # 6. Both build calls still present
    wba_tree = ast.parse(wba_src)
    found_classifier_call = False
    found_annotation_call = False
    for node in ast.walk(wba_tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)):
            if node.func.id == "build_classifier_page":
                found_classifier_call = True
            elif node.func.id == "build_annotation_page":
                found_annotation_call = True
    check(
        "build_classifier_page() call still present in workbench_app.py",
        found_classifier_call,
    )
    check(
        "build_annotation_page() call still present in workbench_app.py",
        found_annotation_call,
    )

    # 7. Parse-clean
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
        f"smoke_122d0_workbench_page_order: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
