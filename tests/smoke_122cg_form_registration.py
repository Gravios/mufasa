"""
tests/smoke_122cg_form_registration.py
========================================

Patch 122cg: regression guard for OperationForm registration.
Every public ``OperationForm`` subclass under ``mufasa/ui_qt/forms/``
must appear in at least one ``mufasa/ui_qt/pages/*.py`` file
(typically referenced from an ``add_section()`` call).

This protects against the failure mode where a form is rewritten /
added but the corresponding page wiring is forgotten — the form
would exist in the codebase but be unreachable from the workbench.

The 122cc and 122ce patches each had a "registration unverified"
caveat because no test enforced this invariant; the audit run in
122cg found all 60 existing forms registered, so the caveats were
unfounded. This test ensures they STAY unfounded.

Methodology
-----------
1. Parse every ``mufasa/ui_qt/forms/*.py`` for class definitions
   subclassing OperationForm (transitively — direct base only;
   chained inheritance not traced).
2. Skip private helpers (names starting with ``_``).
3. For each public form, substring-search every
   ``mufasa/ui_qt/pages/*.py`` for the class name.
4. Fail if any form has zero page references.

Coverage
--------
1. docs/qt_form_registration_audit.md exists and references the
   no-orphans finding.
2. The forms directory contains OperationForm subclasses (sanity
   check — ≥ 50; current count is 60).
3. The pages directory has page files (≥ 10).
4. Every public OperationForm subclass is referenced from at
   least one page. THE main assertion of this test.
5. AverageFrameForm specifically is registered (carry-over from
   122cc caveat).
6. DropBodypartsForm specifically is registered (carry-over from
   122ce caveat).
7. AnalysisForm appears in BOTH analysis_page.py AND roi_page.py
   (intentional double-registration, regression guard).
8. All mufasa/**/*.py files parse cleanly.

If the main assertion (#4) fails, the test prints which forms
are orphan so the maintainer can add the missing page wiring or
add the form to the audit's exclusion list.
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
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


# Intentional exclusions: forms whose absence from a page is by
# design (e.g., backend-only forms used programmatically).
# Add a name here ONLY with a comment explaining why.
EXPECTED_ORPHANS: set[str] = {
    # (currently empty — no intentional orphans)
}


def main() -> int:
    audit_doc = REPO_ROOT / "docs" / "qt_form_registration_audit.md"
    check(
        "docs/qt_form_registration_audit.md exists",
        audit_doc.exists(),
    )
    if audit_doc.exists():
        audit_text = audit_doc.read_text()
        check(
            "audit doc references the no-orphans finding",
            "Zero orphans" in audit_text
            or "0 orphans" in audit_text
            or "60 of 60" in audit_text,
        )

    # ------------------------------------------------------------------
    # Collect public OperationForm subclasses
    # ------------------------------------------------------------------
    forms_dir = REPO_ROOT / "mufasa" / "ui_qt" / "forms"
    pages_dir = REPO_ROOT / "mufasa" / "ui_qt" / "pages"
    operation_forms: list[tuple[str, Path]] = []
    for f in sorted(forms_dir.glob("*.py")):
        if f.name == "__init__.py":
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = {ast.unparse(b) for b in node.bases}
                if any("OperationForm" in b for b in base_names):
                    if not node.name.startswith("_"):
                        operation_forms.append((node.name, f))

    check(
        f"Forms directory contains ≥ 50 OperationForm subclasses "
        f"(found {len(operation_forms)})",
        len(operation_forms) >= 50,
    )

    page_files = list(pages_dir.glob("*.py"))
    check(
        f"Pages directory has ≥ 10 page files (found {len(page_files)})",
        len(page_files) >= 10,
    )

    # ------------------------------------------------------------------
    # Substring-search every page for each form's class name
    # ------------------------------------------------------------------
    referenced: dict[str, list[str]] = defaultdict(list)
    for p in sorted(pages_dir.glob("*.py")):
        if p.name == "__init__.py":
            continue
        src = p.read_text()
        for form_name, _f in operation_forms:
            if form_name in src:
                referenced[form_name].append(p.name)

    orphans = [name for (name, _f) in operation_forms
               if name not in referenced
               and name not in EXPECTED_ORPHANS]

    check(
        "Every public OperationForm subclass is registered in at "
        "least one page (main assertion)",
        orphans == [],
        detail=(f"orphans: {orphans}" if orphans else ""),
    )

    # Specific carry-overs from 122cc + 122ce caveats
    check(
        "AverageFrameForm specifically is registered "
        "(122cc carry-over)",
        "AverageFrameForm" in referenced,
        detail=(f"AverageFrameForm pages: {referenced.get('AverageFrameForm', [])}"
                if "AverageFrameForm" in referenced else ""),
    )
    check(
        "DropBodypartsForm specifically is registered "
        "(122ce carry-over)",
        "DropBodypartsForm" in referenced,
        detail=(f"DropBodypartsForm pages: {referenced.get('DropBodypartsForm', [])}"
                if "DropBodypartsForm" in referenced else ""),
    )

    # AnalysisForm intentional double-registration
    af_pages = referenced.get("AnalysisForm", [])
    check(
        "AnalysisForm appears in BOTH analysis_page.py "
        "AND roi_page.py (intentional double-registration)",
        "analysis_page.py" in af_pages and "roi_page.py" in af_pages,
        detail=f"AnalysisForm pages: {af_pages}",
    )

    # ------------------------------------------------------------------
    # All files parse cleanly
    # ------------------------------------------------------------------
    pkg = REPO_ROOT / "mufasa"
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
        f"smoke_122cg_form_registration: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
