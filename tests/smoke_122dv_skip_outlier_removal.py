"""
tests/smoke_122dv_skip_outlier_removal.py
===========================================

Patch 122dv: remove ``Skip outlier correction`` form, its backend
class, and its section entry from the Pose cleanup page.

What Skip used to do
--------------------
``OutlierCorrectionSkipper`` was a no-op passthrough — read pose
from ``csv/input_csv/``, standardize headers, write to
``csv/outlier_corrected_movement_location/`` unchanged. The point
was to satisfy the downstream contract: consumers (Features,
Classifier, Visualizations) read from
``derived/outlier_corrected/``, so anyone whose pose was already
clean (MARS, hand-curated DLC, Kalman-v2-smoothed) still needed
*something* to populate that location.

Why it's gone
-------------
The 122ds → 122dt → 122du arc replaces the no-op passthrough with
producer backends that publish relative symlinks via
:func:`mufasa.project_layout.publish_to_stage`. Kalman v2 already
does this (122dt). When Interpolate and Data Import publishing
land in future patches, every legitimate "skip the correction
heuristic" workflow has a producer-driven equivalent.

Removing Skip simplifies the workbench (one fewer section on Pose
cleanup), removes a counter-intuitively-named operation ("Skip"
that nonetheless writes files), and forces the v1 architecture
through: backends produce, symlinks expose.

Coverage
--------
Backend deletion:
1.  ``mufasa/outlier_tools/skip_outlier_correction.py`` no longer
    exists on disk.
2.  ``OutlierCorrectionSkipper`` is no longer importable from
    ``mufasa.outlier_tools.skip_outlier_correction``.

Form deletion:
3.  ``mufasa.ui_qt.forms.pose_cleanup`` doesn't export
    ``SkipOutlierCorrectionForm`` (removed from ``__all__``).
4.  No ``class SkipOutlierCorrectionForm`` in
    ``mufasa/ui_qt/forms/pose_cleanup.py``.

Page deletion:
5.  ``pose_cleanup_page.py`` doesn't import
    ``SkipOutlierCorrectionForm``.
6.  ``pose_cleanup_page.py`` doesn't register a section titled
    ``"Skip outlier correction"``.
7.  The remaining sections on Pose cleanup are the 6 expected ones
    (Preprocess Videos, Video Calibration, Interpolate, Kalman v2,
    Run outlier correction, Egocentric alignment, Advanced /
    legacy — total of 7).

Error-message update:
8.  ``mufasa/ui_qt/forms/features.py`` empty-videos message no
    longer mentions ``"Skip outlier correction"`` (would dangle a
    UI option that no longer exists).
9.  The replacement message mentions the producer-driven options
    (Run outlier correction OR Kalman v2 smoothing).

Documentation sweep:
10. ``docs/workflow_audit.md`` — no live mention of
    ``SkipOutlierCorrectionForm`` outside a deprecation note.
11. ``docs/qt_workbench_known_issues.md`` — message excerpt
    updated.
12. ``docs/workflows.md`` — Skip branch removed/marked.
13. ``docs/tk_to_qt_consolidation_plan.md`` — Skip checklist
    item removed/struck through.

Cross-patch invariants:
14. 122dt subclass declarations unaffected
    (``RunOutlierCorrectionForm.section_id == "outlier_correction"``,
    Kalman v2 still publishes).
15. 122ds SECTIONS dict unchanged — provenance DAG didn't include
    a Skip entry, so no rip-out needed.
16. All ``mufasa/**/*.py`` parse cleanly.
17. 122do baseline: no ``Optional[`` in non-docstring positions
    across ``mufasa/ui_qt/``.
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


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # 1. Backend file deleted.
    backend = pkg / "outlier_tools" / "skip_outlier_correction.py"
    check(
        "mufasa/outlier_tools/skip_outlier_correction.py no longer "
        "exists",
        not backend.exists(),
    )

    # 2. Backend class no longer importable.
    import importlib
    try:
        importlib.import_module(
            "mufasa.outlier_tools.skip_outlier_correction")
        skipper_importable = True
    except ImportError:
        skipper_importable = False
    check(
        "mufasa.outlier_tools.skip_outlier_correction is no longer "
        "importable",
        not skipper_importable,
    )

    # 3 & 4. Form deletion.
    pc_path = pkg / "ui_qt" / "forms" / "pose_cleanup.py"
    pc_src = pc_path.read_text()
    pc_tree = ast.parse(pc_src)

    classes_in_pc = {
        n.name for n in pc_tree.body if isinstance(n, ast.ClassDef)
    }
    check(
        "No `class SkipOutlierCorrectionForm` in pose_cleanup.py",
        "SkipOutlierCorrectionForm" not in classes_in_pc,
    )

    # __all__ check.
    all_list_names: set[str] = set()
    for node in pc_tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__all__"
                        for t in node.targets)
                and isinstance(node.value, ast.List)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant):
                    all_list_names.add(elt.value)
    check(
        "pose_cleanup.py __all__ no longer exports "
        "SkipOutlierCorrectionForm",
        "SkipOutlierCorrectionForm" not in all_list_names,
    )

    # 5 & 6 & 7. Page-level removal.
    page_path = pkg / "ui_qt" / "pages" / "pose_cleanup_page.py"
    page_src = page_path.read_text()
    check(
        "pose_cleanup_page.py doesn't import SkipOutlierCorrectionForm",
        "SkipOutlierCorrectionForm" not in page_src,
    )
    check(
        "pose_cleanup_page.py doesn't add a section titled "
        "'Skip outlier correction' (only the deprecation comment "
        "may mention the name)",
        ('add_section("Skip outlier correction"' not in page_src
         and "add_section('Skip outlier correction'" not in page_src),
    )

    # Count add_section calls; expect 7 (PreproVideos, VideoCalib,
    # Interpolate, Kalman v2, Run outlier, Egocentric, Advanced/legacy).
    add_section_calls = re.findall(
        r'page\.add_section\(\s*["\'](.*?)["\']',
        page_src,
    )
    check(
        f"pose_cleanup_page.py has 7 sections (was 8 pre-122dv): "
        f"{add_section_calls}",
        len(add_section_calls) == 7,
        detail=f"got {len(add_section_calls)}",
    )

    # 8 & 9. Error message update.
    features_path = pkg / "ui_qt" / "forms" / "features.py"
    features_src = features_path.read_text()
    check(
        "Empty-videos message in features.py no longer mentions "
        "'Skip outlier correction'",
        "'Skip outlier correction'" not in features_src
        and '"Skip outlier correction"' not in features_src,
    )
    check(
        "Empty-videos message mentions producer alternatives "
        "(Run outlier correction / Kalman v2)",
        "Run outlier correction" in features_src
        and "Kalman v2" in features_src,
    )

    # 10-13. Documentation sweep.
    docs = REPO_ROOT / "docs"
    audit = (docs / "workflow_audit.md").read_text()
    check(
        "docs/workflow_audit.md: SkipOutlierCorrectionForm only "
        "appears in a deprecation-context paragraph (the table "
        "row was removed)",
        # The removed table row used "| `SkipOutlierCorrectionForm`" —
        # the deprecation note uses "``SkipOutlierCorrectionForm``"
        # (inside prose). The pipe-form check guards against the row
        # creeping back.
        "| `SkipOutlierCorrectionForm`" not in audit,
    )

    known = (docs / "qt_workbench_known_issues.md").read_text()
    check(
        "docs/qt_workbench_known_issues.md: the message excerpt no "
        "longer references 'Skip outlier correction' as a recovery "
        "hint",
        "Skip outlier correction" not in known
        or "removed" in known.lower(),
    )

    flows = (docs / "workflows.md").read_text()
    check(
        "docs/workflows.md: the Skip branch is removed (any "
        "remaining mention is in a deprecation note)",
        ("OutlierCorrectionSkipper.run()" not in flows)
        or ("removed" in flows.lower()),
    )

    tk_qt = (docs / "tk_to_qt_consolidation_plan.md").read_text()
    check(
        "docs/tk_to_qt_consolidation_plan.md: Skip checklist item "
        "removed or struck through",
        # The strikethrough form is `~~**Skip...**~~`. The original
        # was a plain bullet. The plain bullet should be gone.
        "- **Skip outlier correction** (existing)" not in tk_qt,
    )

    # 14. 122dt invariants.
    check(
        "122dt: RunOutlierCorrectionForm.section_id still declared "
        "(provenance wiring survives the Skip removal)",
        'section_id = "outlier_correction"' in pc_src,
    )
    check(
        "122dt: Kalman v2 still publishes to outlier_corrected",
        'publish_target_stage = "outlier_corrected"' in pc_src,
    )

    # 15. SECTIONS unchanged.
    from mufasa.section_provenance import SECTIONS
    check(
        "section_provenance.SECTIONS doesn't have a 'skip' entry "
        "(122ds never declared one; nothing to remove)",
        not any("skip" in sid.lower() for sid in SECTIONS),
    )

    # 16. Parse-clean.
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

    # 17. 122do baseline.
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
        "122do baseline preserved: no `Optional[` in non-docstring "
        "positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122dv_skip_outlier_removal: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
