"""
tests/smoke_122fo_computed_feature_badges.py
================================================

Patch 122fo — per-family "already computed" badge in the
Compute Features form.

User request (Thu May 28, 2026):

> [a badge that] shows up at the beginning of the feature name
> of the features that have already been computed and exist in
> the features folder.

Distinct from the SECTION-level badge (122fj, white/green on the
QGroupBox). This is a PER-ITEM indicator: each feature family in
the two selector lists (Subject features / ROI features) gets a
✓ prefix + green text + tooltip when its output already exists in
derived/features/<slug>/.

WHAT THIS PATCH LANDED
======================

mufasa/ui_qt/forms/features.py::FeatureSubsetExtractorForm:

* New _computed_family_slugs() -> set[str]:
  Scans derived/features/ once; a family slug counts as
  "computed" if its subdir exists and contains at least one
  non-hidden .parquet file. Soft-fails to empty set on any
  error (no project / no dir / permission) — a missing badge
  beats a crash in form construction. .parquet filter mirrors
  the 122fm stray-file audit rule.

* New _make_family_item(fam, computed_slugs) -> QListWidgetItem:
  Builds the list item. Clean family name → Qt.UserRole (read by
  collect_args). Display text gets a ✓ prefix + sea-green
  foreground + tooltip when fam's slug is in computed_slugs;
  plain otherwise.

* build() computes computed_slugs once and routes both the
  subject and ROI family loops through _make_family_item.

WHY UserRole IS UNTOUCHED

collect_args reads it.data(Qt.UserRole) for the selection, NOT
it.text(). So decorating the DISPLAY text with a ✓ prefix has
zero effect on which families get computed — the badge is purely
visual. This smoke pins that invariant.

COVERAGE
========

Detection helper (4 checks) — via an embedded reference impl
matching the patch's logic:
1.  Family with a .parquet file under its slug dir → computed.
2.  Family with an empty slug dir → NOT computed.
3.  Family whose slug dir holds only non-parquet files → NOT
    computed (stray-file safe).
4.  Hidden dotdirs under derived/features/ → ignored.

Source structure (5 checks):
5.  _computed_family_slugs method defined; filters by .parquet.
6.  _make_family_item method defined.
7.  _make_family_item preserves the clean name in Qt.UserRole
    (collect_args invariant).
8.  _make_family_item applies a ✓ prefix + setForeground when
    computed.
9.  build() routes BOTH list loops through _make_family_item
    (no plain QListWidgetItem(fam) construction left in build).

Cross-patch invariants (3 checks):
10. 122fj state preserved: FeatureSubsetExtractorForm still
    declares section_id='features_compute_subset' (the SECTION
    badge coexists with the new per-item badges).
11. 122fn state preserved: plot_clf_results_mp accepts show_bbox.
12. Parse-clean.
"""
from __future__ import annotations

import ast
import sys
import tempfile
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


def _method_src(src: str, class_name: str, method: str) -> str:
    tree = ast.parse(src)
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for m in cls.body:
                if (isinstance(m, ast.FunctionDef)
                        and m.name == method):
                    return ast.unparse(m)
    return ""


# Reference impl mirroring _computed_family_slugs.
def _ref_computed(feat_dir: Path) -> set[str]:
    if not feat_dir.is_dir():
        return set()
    out: set[str] = set()
    for sub in feat_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        has_parquet = any(
            p.is_file() and not p.name.startswith(".")
            and p.suffix.lower() == ".parquet"
            for p in sub.iterdir()
        )
        if has_parquet:
            out.add(sub.name)
    return out


def main() -> int:
    # -----------------------------------------------------------------
    # Detection helper correctness
    # -----------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        feat = Path(td) / "derived" / "features"
        (feat / "distances").mkdir(parents=True)
        (feat / "distances" / "v1.parquet").write_text("x")
        (feat / "hull").mkdir(parents=True)  # empty
        (feat / "angles").mkdir(parents=True)
        (feat / "angles" / "README.txt").write_text("notes")
        (feat / ".cache").mkdir(parents=True)
        (feat / ".cache" / "x.parquet").write_text("x")

        got = _ref_computed(feat)
        check(
            "Family with a .parquet under its slug dir is detected "
            "as computed",
            "distances" in got,
        )
        check(
            "Family with an empty slug dir is NOT computed",
            "hull" not in got,
        )
        check(
            "Family whose slug dir has only non-parquet files is "
            "NOT computed (stray-file safe — mirrors 122fm)",
            "angles" not in got,
        )
        check(
            "Hidden dotdirs under derived/features/ are ignored",
            ".cache" not in got,
            detail=(f"got {sorted(got)}"),
        )

    # -----------------------------------------------------------------
    # Source structure
    # -----------------------------------------------------------------
    feat_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                / "features.py").read_text()

    cfs_src = _method_src(
        feat_src, "FeatureSubsetExtractorForm",
        "_computed_family_slugs",
    )
    check(
        "_computed_family_slugs method defined and filters by "
        ".parquet (the computed-evidence signal)",
        cfs_src != "" and ".parquet" in cfs_src,
    )

    mfi_src = _method_src(
        feat_src, "FeatureSubsetExtractorForm",
        "_make_family_item",
    )
    check(
        "_make_family_item method defined",
        mfi_src != "",
    )
    check(
        "_make_family_item preserves the clean family name in "
        "Qt.UserRole (collect_args reads UserRole, so the badge "
        "is purely visual — selection logic unaffected)",
        "setData(Qt.UserRole, fam)" in mfi_src,
    )
    check(
        "_make_family_item applies a ✓ prefix AND setForeground "
        "when the family is computed (the visible badge)",
        (("\\u2713" in mfi_src or "\u2713" in mfi_src)
         and "setForeground" in mfi_src),
    )

    build_src = _method_src(
        feat_src, "FeatureSubsetExtractorForm", "build",
    )
    check(
        "build() routes BOTH family loops through "
        "_make_family_item and no longer constructs a plain "
        "QListWidgetItem(fam) directly (badge applied uniformly)",
        (build_src.count("_make_family_item") >= 2
         and "QListWidgetItem(fam)" not in build_src),
        detail=(
            f"_make_family_item calls: "
            f"{build_src.count('_make_family_item')}"
        ),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    check(
        "122fj state preserved: FeatureSubsetExtractorForm still "
        "declares section_id='features_compute_subset' (the "
        "section-level badge coexists with the per-item badges)",
        "section_id = 'features_compute_subset'" in feat_src
        or 'section_id = "features_compute_subset"' in feat_src,
    )

    pcr_src = (REPO_ROOT / "mufasa" / "plotting"
               / "plot_clf_results_mp.py").read_text()
    check(
        "122fn state preserved: plot_clf_results_mp accepts "
        "show_bbox",
        "show_bbox" in pcr_src,
    )

    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122fo_computed_feature_badges: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
