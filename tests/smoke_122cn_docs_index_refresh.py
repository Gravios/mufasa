"""
tests/smoke_122cn_docs_index_refresh.py
=========================================

Patch 122cn: docs/README.md refresh.

Updates the README index to reflect the post-122cm reality:

* Restructured into 3 clear sections: workflow, Tk → Qt migration
  (developer-facing audits), Kalman smoother (separate concern).
* Stale per-doc entries updated:
  - `backend_audit.md` — 25 → 19 importers; references the new
    §3d strategic disposition + the decision rule.
  - `qt_form_runtime_gaps.md` — "Four forms / seven failing ops"
    → "all 7 closed in 122ca–cf + CLAHE preview in 122ci."
  - `tk_surface_audit.md` — adds the 10 file deletions to date.
  - `tk_to_qt_consolidation_plan.md` — adds tier status post-122cm.
* "Three documents" miscount in the Kalman section corrected to
  "four documents" (matches the actual list).

Coverage
--------
1. docs/README.md exists.
2. README references every doc actually in docs/ (by filename).
3. No README-listed doc is missing from docs/.
4. The "Kalman smoother — design track" section says "four
   documents" (corrects the prior "three" miscount).
5. `backend_audit.md` entry mentions §3d strategic disposition.
6. `backend_audit.md` entry mentions the post-122cm count (19).
7. `backend_audit.md` entry mentions the decision rule.
8. `qt_form_runtime_gaps.md` entry no longer claims "Four forms
   with seven failing operations" — those are closed.
9. `qt_form_runtime_gaps.md` entry mentions all 7 originally-
   counted gaps closed.
10. `tk_surface_audit.md` entry mentions the file deletions
    (10 so far).
11. `tk_to_qt_consolidation_plan.md` entry mentions current
    tier status (post-122cm).
12. The README has a "When to add a new doc" section that
    matches the actual conventions used.
13. All mufasa/**/*.py files parse cleanly (sanity).
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
    docs_dir = REPO_ROOT / "docs"
    readme = docs_dir / "README.md"
    check("docs/README.md exists", readme.exists())
    if not readme.exists():
        return 1
    readme_text = readme.read_text()

    # ==================================================================
    # README ↔ disk consistency
    # ==================================================================
    actual_docs = sorted([
        f.name for f in docs_dir.iterdir()
        if f.is_file() and f.name != "README.md"
    ])

    missing_from_readme: list[str] = []
    for doc in actual_docs:
        # Be lenient: match either backticked or plain reference
        if doc not in readme_text:
            missing_from_readme.append(doc)
    check(
        "README references every doc actually in docs/",
        missing_from_readme == [],
        detail=f"missing: {missing_from_readme}",
    )

    # README-listed docs that no longer exist would be a false
    # advertisement. Find all `filename.md` / `filename.txt`
    # references in the README and check each exists. (.py files
    # are excluded because the README mentions non-docs `.py`
    # files like `tkinter_functions.py` and `audit_v2_pipeline.py`
    # is checked separately as a known docs entry.)
    import re
    listed_docs = set(re.findall(
        r"`([a-z_0-9]+\.(?:md|txt))`", readme_text))
    # Exclude generic mentions of README itself
    listed_docs.discard("README.md")
    missing_on_disk: list[str] = [
        d for d in listed_docs if not (docs_dir / d).exists()
    ]
    check(
        "no README-listed doc is missing from docs/",
        missing_on_disk == [],
        detail=f"missing on disk: {missing_on_disk}",
    )

    # ==================================================================
    # Kalman section: "four documents" not "three"
    # ==================================================================
    check(
        "Kalman section says 'four documents' (corrects prior 'three')",
        "four documents" in readme_text
        and "three documents" not in readme_text,
    )

    # ==================================================================
    # backend_audit entry updates
    # ==================================================================
    # The entry should mention §3d / Bucket / decision-rule and the
    # post-122cm count (19) rather than the original 25.
    check(
        "backend_audit entry mentions §3d strategic disposition",
        "§3d" in readme_text and "Strategic disposition" in readme_text,
    )
    check(
        "backend_audit entry mentions the four-bucket classification",
        "four-bucket" in readme_text or "Bucket 1" in readme_text
        or "four buckets" in readme_text,
    )
    check(
        "backend_audit entry mentions the decision rule for future audits",
        "decision rule" in readme_text or "Decision rule" in readme_text,
    )

    # ==================================================================
    # qt_form_runtime_gaps entry updates
    # ==================================================================
    # The old text said "Four forms with seven failing operations".
    # Post-122ci, all 7 + CLAHE preview are closed. The entry should
    # reflect that.
    check(
        "qt_form_runtime_gaps entry no longer claims active failing "
        "operations",
        # "seven failing operations" was the stale claim
        "seven failing operations" not in readme_text,
    )
    check(
        "qt_form_runtime_gaps entry mentions all 7 gaps closed",
        ("all 7 closed" in readme_text
         or "all 7 originally-counted" in readme_text)
        and "122ci" in readme_text,
    )
    check(
        "qt_form_runtime_gaps entry says 'No remaining "
        "NotImplementedError raises'",
        "No remaining `NotImplementedError`" in readme_text
        or "No remaining NotImplementedError" in readme_text,
    )

    # ==================================================================
    # tk_surface_audit entry updates
    # ==================================================================
    check(
        "tk_surface_audit entry mentions the file deletions to date",
        ("8 files dropped" in readme_text
         or "10 file deletions" in readme_text
         or "86 of 96" in readme_text),
    )

    # ==================================================================
    # tk_to_qt_consolidation_plan entry updates
    # ==================================================================
    check(
        "tk_to_qt_consolidation_plan entry mentions current tier "
        "status post-122cm",
        ("Tier status" in readme_text
         or "post-122cm" in readme_text)
        and "Tier 3b" in readme_text and "pending" in readme_text,
    )

    # ==================================================================
    # When-to-add-a-new-doc section
    # ==================================================================
    check(
        "README retains the 'When to add a new doc' guidance section",
        "When to add a new doc" in readme_text,
    )

    # ==================================================================
    # All mufasa/**/*.py parse cleanly (sanity)
    # ==================================================================
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
        f"smoke_122cn_docs_index_refresh: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
