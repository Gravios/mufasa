"""
tests/smoke_122fj_features_badges.py
=========================================

Patch 122fj — badge wiring for ROI:Features + Features:Compute
feature subset.

User request (Tue May 26, 2026):

> ROI : Features. should write to parquet, data needs an
> appropriate destinations. badge system: white/green,
> dependent on ROI:Definitions and Preprocessing
>
> Features: Compute feature subset, ensure that a second run
> overwrites the appropriate columns and rows. Also needs a
> badge system. white/green.

Both forms get badges via record_run on success. detect_path is
NOT used for either — both write into the shared
derived/features/ directory alongside subject features and each
other, so filesystem evidence can't distinguish "ran ROI
features" from "ran feature subsets" from "ran subject feature
extraction." record_run is the reliable signal.

WHAT THIS PATCH LANDED
======================

mufasa/section_provenance.py:

* NEW SectionSpec ``features_compute_subset``:
    page="Features", section_title="Compute feature subsets",
    depends_on=("outlier_correction",), detect_path=None.
* Updated comments on existing ``features_roi`` spec to note
  the new section_id wiring.

mufasa/ui_qt/forms/roi.py:
* ``ROIFeaturesForm.section_id = "features_roi"`` (was unset).

mufasa/ui_qt/forms/features.py:
* ``FeatureSubsetExtractorForm.section_id = "features_compute_subset"``
  (was unset).

The form is the same as before — no UI changes. Wiring
``section_id`` causes OperationForm._record_provenance to fire
on success, writing ``[provenance.features_roi]`` (or
``[provenance.features_compute_subset]``) to project.toml. The
badge transitions UNKNOWN → CURRENT on the next page render.

WHAT THIS PATCH DID NOT CHANGE
==============================

* Backend write paths in ``ROIFeatureCreator`` and
  ``FeatureSubsetsCalculator``. Both already write parquet in
  v1 projects via the project's ``file_type`` config; the user's
  "should write to parquet" requirement is already satisfied
  for v1.
* The overwrite-conflict resolution in
  ``FeatureSubsetExtractorForm.on_run``. The preflight + user-
  confirmed overwrite flow already covers the user's "ensure
  that a second run overwrites the appropriate columns and
  rows" requirement.

COVERAGE
========

SectionSpec contract (3 checks):
1.  SECTIONS["features_compute_subset"] is registered.
2.  features_compute_subset has page="Features" and
    section_title="Compute feature subsets" (matches the form's
    add_section call).
3.  features_compute_subset.depends_on includes
    "outlier_correction" (the user's "dependent on
    Preprocessing" requirement).

Form wiring (4 checks):
4.  ROIFeaturesForm declares section_id="features_roi".
5.  FeatureSubsetExtractorForm declares
    section_id="features_compute_subset".
6.  features_roi.depends_on includes "outlier_correction"
    AND "roi_definitions" (the user's "dependent on
    ROI:Definitions and Preprocessing" requirement).
7.  Neither form declares a detect_path on its SectionSpec
    (deliberate — both write to shared dir; record_run via
    section_id is the signal).

Wired-form count tripwire (1 check):
8.  Total wired forms is now 7 (was 5 pre-122fj): import_pose,
    kalman_v2, interpolate, outlier_correction, egocentric,
    features_roi, features_compute_subset.

Cross-patch invariants (3 checks):
9.  122fi state preserved: _jump_to_label_transition method
    defined.
10. 122fh state preserved: scrubber has set_playback_fps.
11. Parse-clean.
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


def _get_class_attr(
    src: str, class_name: str, attr_name: str,
):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == class_name):
            for m in node.body:
                if (isinstance(m, ast.Assign)
                        and len(m.targets) == 1
                        and isinstance(m.targets[0], ast.Name)
                        and m.targets[0].id == attr_name):
                    if isinstance(m.value, ast.Constant):
                        return m.value.value
                    return "<non-constant>"
            return None
    return None


def main() -> int:
    from mufasa.section_provenance import SECTIONS

    # -----------------------------------------------------------------
    # SectionSpec contract
    # -----------------------------------------------------------------
    fcs = SECTIONS.get("features_compute_subset")
    check(
        "SECTIONS['features_compute_subset'] is registered "
        "(new section for the Features-page Compute feature "
        "subsets form)",
        fcs is not None,
    )
    check(
        "features_compute_subset has page='Features' and "
        "section_title='Compute feature subsets' (matches the "
        "Features-page add_section call)",
        fcs is not None
        and fcs.page == "Features"
        and fcs.section_title == "Compute feature subsets",
        detail=(
            f"page={getattr(fcs, 'page', None)!r} "
            f"section_title={getattr(fcs, 'section_title', None)!r}"
        ),
    )
    check(
        "features_compute_subset.depends_on includes "
        "'outlier_correction' (the user's 'dependent on "
        "Preprocessing' requirement — outlier_correction is the "
        "input contract for feature extraction)",
        fcs is not None
        and "outlier_correction" in fcs.depends_on,
        detail=(
            f"depends_on={getattr(fcs, 'depends_on', None)!r}"
        ),
    )

    # -----------------------------------------------------------------
    # Form wiring
    # -----------------------------------------------------------------
    roi_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "roi.py").read_text()
    roi_features_sid = _get_class_attr(
        roi_src, "ROIFeaturesForm", "section_id",
    )
    check(
        "ROIFeaturesForm declares section_id='features_roi' "
        "(badge wires via record_run on successful append; "
        "detect_path stays None because ROI features are mixed "
        "into derived/features/<video>.parquet alongside subject "
        "features)",
        roi_features_sid == "features_roi",
        detail=(f"got {roi_features_sid!r}"),
    )

    feat_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                / "features.py").read_text()
    fs_sid = _get_class_attr(
        feat_src, "FeatureSubsetExtractorForm", "section_id",
    )
    check(
        "FeatureSubsetExtractorForm declares "
        "section_id='features_compute_subset' (badge wires via "
        "record_run on successful runs; detect_path stays None "
        "for the same shared-dir reason as features_roi)",
        fs_sid == "features_compute_subset",
        detail=(f"got {fs_sid!r}"),
    )

    fr = SECTIONS.get("features_roi")
    check(
        "features_roi.depends_on includes BOTH "
        "'outlier_correction' AND 'roi_definitions' (the user's "
        "'dependent on ROI:Definitions and Preprocessing' "
        "requirement)",
        fr is not None
        and "outlier_correction" in fr.depends_on
        and "roi_definitions" in fr.depends_on,
        detail=(f"depends_on={getattr(fr, 'depends_on', None)!r}"),
    )

    # Neither new wiring declares a detect_path.
    check(
        "Neither features_roi nor features_compute_subset has "
        "a detect_path (deliberate — both write to shared dir; "
        "record_run via section_id is the reliable signal, not "
        "filesystem inspection)",
        fr is not None and fr.detect_path is None
        and fcs is not None and fcs.detect_path is None,
        detail=(
            f"features_roi.detect_path="
            f"{getattr(fr, 'detect_path', None)!r}; "
            f"features_compute_subset.detect_path="
            f"{getattr(fcs, 'detect_path', None)!r}"
        ),
    )

    # -----------------------------------------------------------------
    # Wired-form count tripwire
    # -----------------------------------------------------------------
    # Collect all wired forms by scanning the codebase for
    # section_id class attributes.
    wired_ids = set()
    for f in (REPO_ROOT / "mufasa" / "ui_qt" / "forms").rglob("*.py"):
        src = f.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for m in cls.body:
                if (isinstance(m, ast.Assign)
                        and len(m.targets) == 1
                        and isinstance(m.targets[0], ast.Name)
                        and m.targets[0].id == "section_id"
                        and isinstance(m.value, ast.Constant)
                        and isinstance(m.value.value, str)):
                    wired_ids.add(m.value.value)
    expected_wired = 7
    check(
        f"Total wired forms is now {expected_wired} (was 5 "
        f"pre-122fj; features_roi + features_compute_subset "
        f"added). Pinning surfaces accidental unwiring or "
        f"over-wiring",
        len(wired_ids) == expected_wired,
        detail=(f"got {sorted(wired_ids)}"),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    fl_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "frame_labeller.py").read_text()
    check(
        "122fi state preserved: _jump_to_label_transition "
        "method defined in frame_labeller.py",
        "def _jump_to_label_transition" in fl_src,
    )

    sc_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "frame_scrubber.py").read_text()
    check(
        "122fh state preserved: scrubber has set_playback_fps",
        "def set_playback_fps" in sc_src,
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
        f"smoke_122fj_features_badges: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
