"""
tests/smoke_122fv_compute_features_badge_refresh.py
===================================================

Patch 122fv — two things in the Compute Features form
(FeatureSubsetExtractorForm), from one user report:

User request (Fri May 30, 2026):
> Under features : compute feature subsets (rename to "Compute
> Features") the feature "exists/computed" badge doesn't update
> after computing a feature or subset of features. I have to
> restart mufasa for the green checkmark to appear.

1. BADGE-REFRESH BUG (functional)
   The per-family ✓ "already computed" badge (122fo) is built once
   in build() from a one-time derived/features/ scan. on_run's
   on_success never re-scanned, so a family computed during the
   session stayed un-badged until the form was reopened. Fix: add
   _refresh_computed_badges() and call it from on_success.

   It reuses _make_family_item as the SINGLE source of badge
   decoration — builds a throwaway decorated item per family and
   copies text/tooltip/foreground onto the live item. This keeps
   the live selection + Qt.UserRole, correctly clears as well as
   sets badges, and avoids decoration drift (so the 122fo pins on
   _make_family_item stay valid and unduplicated).

2. RENAME (display only)
   "Compute feature subsets" -> "Compute Features" in the three
   synced display strings:
     * FeatureSubsetExtractorForm.title       (features.py)
     * features_page add_section(...)         (features_page.py)
     * SectionSpec.section_title              (section_provenance.py)
   section_id stays "features_compute_subset" (stable TOML key —
   never renamed) and the class/module names are unchanged, per
   rename-scope discipline. section_title MUST match the
   add_section title (SectionSpec docstring: "Must match exactly")
   so all three move together.

RECIPROCAL FLIPS
================
* smoke_122fj_features_badges.py: assertion fcs.section_title ==
  "Compute feature subsets" -> "Compute Features" (+ docstring).
* Stale title references in section_provenance.py comment,
  smoke_122el (docstring; its audit logic is dynamic and stays
  green since both sides renamed), smoke_122fo (docstring) updated.
* Historical user-request `>` quotes left verbatim.

NEW SMOKE: smoke_122fv_compute_features_badge_refresh.py (8 checks)
"""

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


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _method(cls: ast.ClassDef, name: str):
    for m in cls.body:
        if isinstance(m, ast.FunctionDef) and m.name == name:
            return m
    return None


def main() -> int:
    feat = _read("mufasa/ui_qt/forms/features.py")
    tree = ast.parse(feat)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FeatureSubsetExtractorForm"
    )

    # --- 1. refresh method exists ----------------------------------------
    refresh = _method(cls, "_refresh_computed_badges")
    check("_refresh_computed_badges method defined", refresh is not None)

    refresh_src = ast.unparse(refresh) if refresh else ""
    check(
        "refresh reuses _make_family_item (single decoration source)",
        "_make_family_item(" in refresh_src,
    )
    check(
        "refresh iterates both selector lists and reads Qt.UserRole",
        "self.subject_families" in refresh_src
        and "self.roi_families" in refresh_src
        and "Qt.UserRole" in refresh_src,
    )
    check(
        "refresh re-scans disk via _computed_family_slugs",
        "_computed_family_slugs()" in refresh_src,
    )

    # --- 2. hooked into on_run success path ------------------------------
    on_run = _method(cls, "on_run")
    on_run_src = ast.unparse(on_run) if on_run else ""
    check(
        "on_run calls _refresh_computed_badges (on the success path)",
        "_refresh_computed_badges()" in on_run_src,
    )

    # --- 3. _make_family_item decoration intact (122fo invariant) --------
    mfi = _method(cls, "_make_family_item")
    mfi_src = ast.unparse(mfi) if mfi else ""
    check(
        "_make_family_item still sets UserRole + ✓ + setForeground",
        "setData(Qt.UserRole, fam)" in mfi_src
        and "\u2713" in mfi_src
        and "setForeground" in mfi_src,
    )

    # --- 4. rename (display only) ----------------------------------------
    title_assign = any(
        isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "title" for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value == "Compute Features"
        for n in cls.body
    )
    page_src = _read("mufasa/ui_qt/pages/features_page.py")
    check(
        "form.title and add_section renamed to 'Compute Features'",
        title_assign and 'add_section("Compute Features"' in page_src,
    )
    check(
        "section_id stays 'features_compute_subset' (stable key)",
        'section_id="features_compute_subset"'
        in _read("mufasa/section_provenance.py"),
    )

    print(
        f"smoke_122fv_compute_features_badge_refresh: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
