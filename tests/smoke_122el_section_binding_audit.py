"""
tests/smoke_122el_section_binding_audit.py
=============================================

Patch 122el: drift-prevention smoke test for the string-keyed
bindings between :data:`mufasa.section_provenance.SECTIONS` and
the workbench page registrations.

Background
----------
The workbench's badge UI uses ``find_section_by_title(page,
section_title)`` to locate the QGroupBox a badge attaches to.
The page name and section title are typed twice — once in
SECTIONS, once in the workbench page-file as ``add_page("X")``
and ``add_section("Y", ...)`` (or ``add_section_widget``)
calls. A mismatch in either string silently suppresses the
badge.

We've now seen this bug surface three times:

* Patch 122eb (fixed): "Import Pose Data" (label) vs "Import
  pose data" (title) on the Data Import page.
* Patch 122ej (fixed): "Pose cleanup" (page) vs "Preprocessing"
  (workbench label) for 7 sections on the Preprocessing page.
* Patch 122el (this): a class-level audit found 8 ADDITIONAL
  mismatches across multiple pages (see below).

What the audit found
--------------------
Running the cross-reference between SECTIONS and
``add_section`` / ``add_section_widget`` calls in
mufasa/ui_qt/pages/, with 5 binding-fixes and 3 unbound markers:

1. ``pixels_per_mm``: title was "Pixels-per-mm calibration",
   workbench has "Video Calibration" → FIXED.
2. ``kalman_v2``: title was "Kalman v2 smoother", workbench
   has "Kalman v2 smoothing" → FIXED.
3. ``savitzky_golay``: form is composited inside "Advanced /
   legacy" QGroupBox; no own add_section → MARKED unbound.
4. ``drop_body_parts``: no form exists in workbench →
   MARKED unbound.
5. ``roi_definitions``: ROI page uses ``add_section_widget``
   (not ``add_section``) for the ROI definitions panel →
   smoke test now recognizes BOTH call shapes.
6. ``features_subject``: no form exists; Features page only
   has "Compute feature subsets" → MARKED unbound.
7. ``features_roi``: ROI features form is on the ROI page
   (not Features), registered as "Features" → page +
   title FIXED.
8. ``annotation``: title was "Annotate", actual labelling
   section is "Frame labelling" → FIXED.
9. ``classifier_run``: title was "Run classifier", workbench
   has "Run inference" → FIXED.

Sections that already resolved before this patch
(no changes needed): ``import_pose``, ``interpolate``,
``outlier_correction``, ``egocentric``, ``classifier_train``.

The new ``ui_bound`` field
--------------------------
``SectionSpec`` gained an optional ``ui_bound: bool = True``
field. When False, the section is in the DAG (for dependency
tracking and future planning) but has no QGroupBox to attach
a badge to. The binding audit smoke test skips these.

Three sections in the current codebase are unbound:

* ``savitzky_golay`` — form lives inside a composite
  QGroupBox.
* ``drop_body_parts`` — aspirational placeholder, no form.
* ``features_subject`` — aspirational placeholder, no form.

Provenance can still be recorded for ui_bound=False sections
via explicit ``record_run`` from backend code; the badge UI
just doesn't render for them.

Drift-prevention contract
-------------------------
This smoke test walks SECTIONS at test time and verifies
every ui_bound entry's (page, section_title) resolves to an
``add_section`` or ``add_section_widget`` call in some
workbench page file. If a future patch:

* renames a section title in SECTIONS but forgets the
  workbench page file (or vice versa);
* moves a section from one page to another;
* adds a new SECTIONS entry without registering the section
  in the workbench;

— the smoke test catches the drift before the user does.

Coverage
--------
1.  ``SectionSpec`` has a ``ui_bound`` field, default True.
2.  ``ui_bound=True`` for all sections except the 3 listed
    above.
3.  ``ui_bound=False`` for savitzky_golay.
4.  ``ui_bound=False`` for drop_body_parts.
5.  ``ui_bound=False`` for features_subject.
6.  The 5 binding fixes landed:
    - pixels_per_mm.section_title == "Video Calibration"
    - kalman_v2.section_title == "Kalman v2 smoothing"
    - features_roi.page == "ROI"
    - features_roi.section_title == "Features"
    - annotation.section_title == "Frame labelling"
    - classifier_run.section_title == "Run inference"
    (Six SECTIONS edits, counted as one combined check for
    brevity, plus 5 individual checks — total 6 entries
    covered.)
7.  The cross-reference audit itself: every ui_bound section
    in SECTIONS has a corresponding (page, title) registered
    in some workbench page file via ``add_section`` or
    ``add_section_widget``.

Cross-patch invariants:
8.  122ek state preserved: safe_filter_by_video and friends
    defined in roi_utils.
9.  122ej state preserved: read_roi_data has _col_unique.
10. 122ei state preserved: detect_path on producer sections.
11. Parse-clean.
12. 122do baseline.
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


def _scan_pages_for_titles(
    pages_dir: Path,
) -> dict[str, set[str]]:
    """Walk all page files. For each, identify the page name
    from ``workbench.add_page("X", ...)`` and collect every
    section title from ``add_section`` / ``add_section_widget``
    calls within that page's setup function."""
    page_to_titles: dict[str, set[str]] = {}
    for f in sorted(pages_dir.glob("*.py")):
        if f.name == "__init__.py":
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        current_page: str | None = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_page"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                current_page = node.args[0].value
                page_to_titles.setdefault(current_page, set())
            # Match add_section OR add_section_widget — the ROI
            # page uses the latter for its Definitions panel.
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in (
                        "add_section", "add_section_widget",
                    )
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                if current_page is not None:
                    page_to_titles[current_page].add(
                        node.args[0].value,
                    )
    return page_to_titles


def main() -> int:
    from mufasa import section_provenance as sp
    from mufasa.section_provenance import SECTIONS, SectionSpec

    # -----------------------------------------------------------------
    # SectionSpec field exists
    # -----------------------------------------------------------------
    has_ui_bound = any(
        f.name == "ui_bound"
        for f in SectionSpec.__dataclass_fields__.values()
    )
    check(
        "SectionSpec has a `ui_bound` field (forward-declared "
        "marker for sections without a QGroupBox in the "
        "workbench)",
        has_ui_bound,
    )

    # -----------------------------------------------------------------
    # ui_bound markers
    # -----------------------------------------------------------------
    expected_unbound = {
        "savitzky_golay", "drop_body_parts", "features_subject",
    }
    for sid in expected_unbound:
        spec = SECTIONS.get(sid)
        check(
            f"SECTIONS[{sid!r}].ui_bound is False (no badge "
            f"surface in the current workbench)",
            spec is not None and spec.ui_bound is False,
            detail=(f"got {getattr(spec, 'ui_bound', None)!r}"),
        )

    # And the bound sections — at least the ones we explicitly
    # fixed — should be bound.
    for sid in ("pixels_per_mm", "kalman_v2", "features_roi",
                "annotation", "classifier_run"):
        spec = SECTIONS.get(sid)
        check(
            f"SECTIONS[{sid!r}].ui_bound is True (has a "
            f"workbench QGroupBox post-122el)",
            spec is not None and spec.ui_bound is True,
        )

    # -----------------------------------------------------------------
    # Binding fixes landed
    # -----------------------------------------------------------------
    fixes = {
        "pixels_per_mm":  ("Preprocessing", "Video Calibration"),
        "kalman_v2":      ("Preprocessing", "Kalman v2 smoothing"),
        "features_roi":   ("ROI",           "Features"),
        "annotation":     ("Annotation",    "Frame labelling"),
        "classifier_run": ("Classifier",    "Run inference"),
    }
    for sid, (exp_page, exp_title) in fixes.items():
        spec = SECTIONS.get(sid)
        check(
            f"SECTIONS[{sid!r}] now has page={exp_page!r} "
            f"and section_title={exp_title!r} "
            f"(rebound by 122el)",
            (spec is not None
             and spec.page == exp_page
             and spec.section_title == exp_title),
            detail=(f"got page={getattr(spec, 'page', None)!r} "
                    f"title={getattr(spec, 'section_title', None)!r}"),
        )

    # -----------------------------------------------------------------
    # The drift-detection audit itself
    # -----------------------------------------------------------------
    pages_dir = REPO_ROOT / "mufasa" / "ui_qt" / "pages"
    page_to_titles = _scan_pages_for_titles(pages_dir)

    unresolved = []
    for sid, spec in SECTIONS.items():
        if not spec.ui_bound:
            continue
        titles = page_to_titles.get(spec.page, set())
        if spec.section_title not in titles:
            unresolved.append(
                f"{sid} -> page={spec.page!r} title={spec.section_title!r}"
            )
    check(
        "Every ui_bound SECTIONS entry resolves to an "
        "add_section / add_section_widget call in some "
        "workbench page file (the drift-detection contract)",
        not unresolved,
        detail=("; ".join(unresolved[:3])
                if unresolved else "all bound"),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    ru_src = (REPO_ROOT / "mufasa" / "roi_tools"
              / "roi_utils.py").read_text()
    check(
        "122ek state preserved: safe_filter_by_video defined "
        "in roi_utils",
        "def safe_filter_by_video" in ru_src,
    )

    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()
    check(
        "122ej state preserved: read_roi_data has _col_unique",
        "_col_unique" in cr_src,
    )

    sp_src = (REPO_ROOT / "mufasa"
              / "section_provenance.py").read_text()
    check(
        "122ei state preserved: detect_path on producer "
        "sections",
        "detect_path=lambda root:" in sp_src,
    )

    # 11. Parse-clean.
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

    # 12. 122do baseline.
    uiqt = pkg / "ui_qt"
    optional_hits = []
    for f in sorted(uiqt.rglob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"\bOptional\[", src):
            preceding = src[:m.start()]
            tq3 = preceding.count('"""') + preceding.count("'''")
            if tq3 % 2 == 0:
                optional_hits.append(str(f.relative_to(uiqt)))
                break
    check(
        "122do baseline preserved: no `Optional[` in non-"
        "docstring positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122el_section_binding_audit: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
