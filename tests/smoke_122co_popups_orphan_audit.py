"""
tests/smoke_122co_popups_orphan_audit.py
==========================================

Patch 122co: pop-ups orphan re-audit. Documentation patch with
methodology lesson: AST traversal, not regex, for cross-file
import audits.

Headline finding
----------------
Zero true orphans in `mufasa/ui/pop_ups/`. The earlier "many
likely orphans" hypothesis was wrong: every pop-up file is
referenced by at least one importer somewhere in `mufasa/` —
mostly via `SimBA.py`'s menu-callback wiring (~80 imports).

Methodology lesson
------------------
First pass used regex (`\\bimport\\b.*\\b{class}\\b`) and reported
37 false-positive orphans. The regex missed multi-line imports
(continuation-line + parenthesized forms), which `SimBA.py` uses
extensively. The corrected AST traversal (`ast.ImportFrom` nodes
+ `node.names` walk) reported the correct 0 count.

Coverage
--------
1. tk_surface_audit.md contains the new §2e section.
2. §2e announces 0 true orphans.
3. §2e explains the regex false-positive (multi-line imports).
4. §2e documents the implication (pop_ups can't be incrementally
   drained; depends on SimBA.py deletion).
5. §7 audit methodology has the "always use AST, never regex"
   directive.
6. The AST audit script reproduces the 0-orphans finding when
   run from this test (regression guard — if a future patch
   accidentally creates a pop-up file no one imports, the test
   will surface it).
7. The AST audit covers BOTH classes and top-level functions
   (was the second-pass fix in 122co's development:
   delete_all_rois_pop_up.py defines a function, not a class).
8. Demonstrates the regex false-negative pattern via a
   constructed example — pins the methodology lesson as a
   concrete regression guard, not just prose.
9. All mufasa/**/*.py files parse cleanly.
"""
from __future__ import annotations

import ast
import re
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


def _ast_orphan_audit(popups_dir: Path,
                      pkg: Path) -> tuple[int, int]:
    """Return (n_orphan, n_referenced)."""
    file_symbols: dict[Path, list[str]] = {}
    for f in sorted(popups_dir.glob("*.py")):
        if f.name == "__init__.py":
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        # Include both classes and top-level functions —
        # delete_all_rois_pop_up.py is the canonical example of
        # a function-only pop-up file
        symbols = [n.name for n in tree.body
                   if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        file_symbols[f] = symbols

    all_symbols = {sym: f for f, syms in file_symbols.items()
                   for sym in syms}

    importers: dict[str, list[Path]] = defaultdict(list)
    for f in pkg.rglob("*.py"):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.name in all_symbols
                            and f != all_symbols[alias.name]):
                        importers[alias.name].append(f)

    orphans = 0
    referenced = 0
    for f, syms in file_symbols.items():
        if not syms:
            orphans += 1
            continue
        if any(importers.get(s) for s in syms):
            referenced += 1
        else:
            orphans += 1
    return orphans, referenced


def _regex_audit_demo() -> tuple[int, int]:
    """Demonstrate the regex pattern's failure mode on a
    constructed multi-line-import string. Returns
    (single_line_hits, multi_line_hits).

    Both should be 1 if the regex were correct; reality: only
    single_line gets caught.
    """
    single_line = "from foo import BarPopUp"
    multi_line = "from foo import \\\n    BarPopUp"
    paren_multi = "from foo import (\n    BarPopUp,\n)"
    pat = re.compile(r"\bimport\b.*\bBarPopUp\b")
    # re.compile default: . does NOT match newlines
    single_hit = 1 if pat.search(single_line) else 0
    multi_hit = 1 if pat.search(multi_line) else 0
    paren_hit = 1 if pat.search(paren_multi) else 0
    return single_hit, multi_hit + paren_hit


def main() -> int:
    docs_dir = REPO_ROOT / "docs"
    audit_doc = docs_dir / "tk_surface_audit.md"
    audit_text = audit_doc.read_text() if audit_doc.exists() else ""

    # ==================================================================
    # Doc additions
    # ==================================================================
    check(
        "tk_surface_audit.md contains new §2e section",
        "### 2e. Pop-ups orphan re-audit" in audit_text,
    )
    check(
        "§2e announces 0 true orphans",
        "zero true orphans" in audit_text
        or "Finding: zero true orphans" in audit_text,
    )
    check(
        "§2e explains the regex false-positive "
        "(multi-line imports)",
        "multi-line imports" in audit_text
        and ("Continuation-line" in audit_text
             or "continuation-line" in audit_text),
    )
    check(
        "§2e documents the implication "
        "(can't incrementally drain; depends on SimBA.py deletion)",
        ("incrementally drained" in audit_text
         or "incremental drain" in audit_text)
        and "SimBA.py" in audit_text,
    )
    check(
        "§7 has the 'always use AST, never regex' directive",
        "Always use AST" in audit_text
        and "regex" in audit_text,
    )

    # ==================================================================
    # AST audit reproducibility — regression guard
    # ==================================================================
    pkg = REPO_ROOT / "mufasa"
    popups_dir = pkg / "ui" / "pop_ups"
    # Known subprocess-launched popups (kept alive at runtime by
    # `ui_qt/dialogs/roi_video_table.py:491-513`, not catchable by
    # AST). Discovered during the 122cr ROI Tk cluster-deletion;
    # documented in tk_surface_audit.md §2g + §7 (the 4th
    # methodology lesson).
    KNOWN_SUBPROCESS_POPUPS = {
        "duplicate_rois_by_source_target_popup.py",
        "import_roi_csv_popup.py",
        "min_max_draw_size_popup.py",
        "roi_size_standardizer_popup.py",
    }
    if popups_dir.exists():
        n_orphan, n_referenced = _ast_orphan_audit(popups_dir, pkg)
        # Allow exactly the known subprocess-launched popups as
        # "AST orphans" since they have a non-AST-visible consumer.
        # Anything beyond that count is a real orphan and should
        # surface.
        n_known_subprocess = sum(
            1 for f in popups_dir.iterdir()
            if f.name in KNOWN_SUBPROCESS_POPUPS
        )
        allowed_orphan_ceiling = n_known_subprocess
        check(
            f"AST audit: ≤ {allowed_orphan_ceiling} known-"
            f"subprocess-launched orphans, no others "
            f"(got orphan={n_orphan}, referenced={n_referenced})",
            n_orphan <= allowed_orphan_ceiling,
        )
        check(
            f"AST audit covers ≥ 70 referenced files "
            f"(post-122cr; was 81 pre-122cr; got "
            f"{n_referenced} referenced)",
            n_referenced >= 70,
        )

    # ==================================================================
    # Methodology lesson as a concrete regression guard
    # ==================================================================
    single_hit, multi_hit = _regex_audit_demo()
    check(
        "regex `\\bimport\\b.*\\b{class}\\b` catches single-line "
        "imports (sanity)",
        single_hit == 1,
    )
    check(
        "regex `\\bimport\\b.*\\b{class}\\b` MISSES multi-line "
        "imports (the documented failure mode)",
        # If this ever passes (multi_hit == 2 = both forms caught),
        # the lesson in §7 no longer applies. We pin the failure
        # mode as a concrete artifact, not just prose.
        multi_hit == 0,
        detail=f"multi_hit={multi_hit}",
    )

    # ==================================================================
    # All files parse cleanly
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
        f"smoke_122co_popups_orphan_audit: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
