"""
tests/smoke_122ew_get_all_statuses_detect_path.py
=====================================================

Patch 122ew-hotfix — ``get_all_statuses`` regression: the bulk
statuses lookup didn't consult ``detect_path`` despite
``get_status`` having done so since 122ei.

User report (Mon May 25, 2026, sixth report of the day):

> Also running the egocentric alignment incorrectly turned
> the previous badges in the table to white.

The user observed badges that were GREEN (CURRENT) before any
form completion turn WHITE (UNKNOWN) the moment any form
completed. Across all sections. Even for unrelated sections
on unrelated pages.

ROOT CAUSE
==========

Two functions in ``mufasa/section_provenance.py``:

* ``get_status(config_path, section_id)`` — single-section.
  Uses ``_resolve_run_at`` (added in 122ei), which composes:
    1. Explicit provenance entry (if present), OR
    2. Filesystem-evidence fallback via ``spec.detect_path``.

* ``get_all_statuses(config_path)`` — bulk. ORIGINAL code
  read provenance entries directly via ``_read_run_at`` and
  returned UNKNOWN whenever no entry existed. **Bypassed
  detect_path entirely.**

The badge UI uses BOTH:

* ``_paint_initial_badge`` (called on ``add_section`` /
  ``add_section_widget``, when a page first opens) →
  ``get_status`` (single) → SAW detect_path correctly.

* ``refresh_section_badges`` (called on ``_on_form_completed``,
  after ANY form runs successfully) → ``get_all_statuses``
  (bulk) → MISSED detect_path entirely.

So the badge state TRANSITIONED FROM CORRECT TO BROKEN the
moment any form succeeded — even forms unrelated to the
affected sections, even forms without a section_id (like
Egocentric Alignment in the user's report).

USER-FACING IMPACT
==================

The bug had been latent since 122ei (when ``_resolve_run_at``
was added to ``get_status`` only). The two functions diverged
silently. It surfaced now because:

* 122es added detect_path to ``pixels_per_mm`` — added a NEW
  section to the set of "shows CURRENT via detect_path only"
  sections.
* 122ev fixed egocentric so it ran for the first time on
  the user's v1 project — the first form completion that
  invoked the buggy ``refresh_section_badges`` path.

Pre-122es and pre-122ev the bug was less visible: fewer
sections used detect_path, and the form that triggered the
refresh wasn't completing on v1 projects. The session-2
fixes inadvertently brought the latent bug into view.

WHY THE 122ei SMOKE DIDN'T CATCH IT
====================================

The 122ei smoke (and 122ep, 122es) exercised the new
detect_path mechanism via ``get_status`` directly::

    s = get_status(str(cfg), "import_pose")  # ← single
    assert s == SectionStatus.CURRENT

That confirmed the SINGLE-section function worked. The BULK
function was never tested with detect_path. Test coverage gap.

The pattern: when adding a feature to one API, REWRITE every
sibling API at the same time, OR add a smoke that asserts
contract parity between siblings.

THE FIX
=======

mufasa/section_provenance.py::get_all_statuses:
* Delegates per-section to ``_resolve_run_at`` (same helper
  ``get_status`` uses). Identical composition semantics:
  explicit → implicit → UNKNOWN.
* Tolerates missing project.toml by treating prov as empty
  dict and still letting ``_resolve_run_at`` consult
  detect_path. Useful for partially-set-up projects where
  data files exist but provenance hasn't been written yet.
* Walks dependencies the same way ``get_status`` does.

Behavior is now byte-identical to a per-section ``get_status``
loop. The bulk wrapper retains its performance characteristic
(single ``read_project_toml`` call vs N) — the loop body is
the same.

Coverage
--------
Contract parity (3 checks):
1.  get_all_statuses("pixels_per_mm") returns CURRENT when
    sources/video_info.csv is present and there's no
    [provenance.pixels_per_mm] block — exercises the
    detect_path-only path through the bulk function.
2.  get_all_statuses returns the SAME status as get_status
    for every section with a detect_path (contract parity
    sweep — pin the bulk/single equivalence so future
    divergence is caught immediately).
3.  With sources/video_info.csv ABSENT, get_all_statuses
    returns UNKNOWN for pixels_per_mm (the inverse case —
    no provenance, no evidence, correctly UNKNOWN).

Staleness composition still works (2 checks):
4.  When pixels_per_mm has detect_path mtime AND outlier_correction
    has a later provenance mtime, get_all_statuses correctly
    reads outlier_correction CURRENT (depends_on doesn't
    include pixels_per_mm — sanity check).
5.  Missing project.toml — get_all_statuses doesn't crash;
    detect_path still works for partially-set-up projects.

Cross-patch invariants:
6.  122ev state preserved: egocentric_aligner has parquet
    support.
7.  122eu state preserved: get_fn_ext handles empty extensions.
8.  122es state preserved: pixels_per_mm has detect_path.
9.  Parse-clean.
10. 122do baseline.
"""
from __future__ import annotations

import ast
import re
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


def main() -> int:
    from mufasa.section_provenance import (
        SECTIONS, SectionStatus,
        get_status, get_all_statuses,
    )

    # -----------------------------------------------------------------
    # Contract parity
    # -----------------------------------------------------------------
    # 1. Bulk picks up detect_path evidence for pixels_per_mm.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "test"\n'
        )
        (root / "sources").mkdir()
        (root / "sources" / "video_info.csv").write_text(
            "Video,fps,ppm\nv1,30,12.5\n"
        )
        s = get_all_statuses(str(cfg))["pixels_per_mm"]
        check(
            "get_all_statuses returns CURRENT for pixels_per_mm "
            "when sources/video_info.csv is present (the bug "
            "the user hit — was UNKNOWN pre-122ew)",
            s == SectionStatus.CURRENT,
            detail=(f"got {s.value!r}"),
        )

    # 2. Bulk == single for every detect_path section.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "test"\n'
        )
        # Create evidence for all 9 detect_path sections.
        (root / "sources").mkdir()
        (root / "sources" / "video_info.csv").write_text("vi")
        (root / "sources" / "pose").mkdir()
        (root / "sources" / "pose" / "v1.csv").write_text("p")
        for sub in ("derived/interpolated/run1",
                    "derived/smoothed_kalman_v2/run1",
                    "derived/outlier_corrected/run1",
                    "derived/labels",
                    "derived/classifications",
                    "models",
                    "logs/measures"):
            (root / sub).mkdir(parents=True)
        # roi_definitions wants a single file, not dir
        (root / "logs" / "measures"
         / "ROI_definitions.h5").write_text("h")
        (root / "derived" / "interpolated"
         / "run1" / "v1.parquet").write_text("d")
        (root / "derived" / "smoothed_kalman_v2"
         / "run1" / "v1.parquet").write_text("d")
        (root / "derived" / "outlier_corrected"
         / "run1" / "v1.parquet").write_text("d")
        (root / "derived" / "labels"
         / "v1.parquet").write_text("l")
        (root / "derived" / "classifications"
         / "v1.parquet").write_text("c")
        (root / "models" / "clf.sav").write_text("m")

        bulk = get_all_statuses(str(cfg))
        detect_path_sids = [
            sid for sid, spec in SECTIONS.items()
            if spec.detect_path is not None and spec.ui_bound
        ]
        divergences = []
        for sid in detect_path_sids:
            single = get_status(str(cfg), sid)
            if single != bulk[sid]:
                divergences.append(
                    f"{sid}: single={single.value} bulk={bulk[sid].value}"
                )
        check(
            f"Contract parity: get_all_statuses returns the same "
            f"status as get_status for all {len(detect_path_sids)} "
            f"ui_bound sections with detect_path (the regression "
            f"contract — catches future drift)",
            not divergences,
            detail=("; ".join(divergences[:3])),
        )

    # 3. Inverse: absent evidence → UNKNOWN.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "test"\n'
        )
        # NO sources/ at all.
        s = get_all_statuses(str(cfg))["pixels_per_mm"]
        check(
            "get_all_statuses returns UNKNOWN for pixels_per_mm "
            "when sources/video_info.csv is absent (inverse case)",
            s == SectionStatus.UNKNOWN,
            detail=(f"got {s.value!r}"),
        )

    # -----------------------------------------------------------------
    # Staleness composition + edge cases
    # -----------------------------------------------------------------
    # 4. Explicit provenance + detect_path mixed.
    import datetime as dt
    import time
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"

        # Create the import_pose detect_path evidence FIRST (older).
        (root / "sources" / "pose").mkdir(parents=True)
        (root / "sources" / "pose" / "v1.csv").write_text("p")
        # Patch 122fc — bases-match validation needs matching video.
        (root / "sources" / "videos").mkdir(parents=True)
        (root / "sources" / "videos" / "v1.mp4").write_text("v")

        # Tiny sleep to ensure the explicit provenance timestamp
        # below is strictly LATER than the file mtime above.
        time.sleep(0.05)

        # Explicit provenance for outlier_correction NOW (newer
        # than the dependency's detect_path mtime).
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        cfg.write_text(
            f'project_layout_version = 1\n'
            f'[project]\nname = "test"\n'
            f'[provenance.outlier_correction]\n'
            f'last_run_at = "{now_iso}"\n'
            f'run_id = "run-test"\n'
        )

        bulk = get_all_statuses(str(cfg))
        # outlier_correction: explicit CURRENT (later than its dep)
        # import_pose: detect_path CURRENT
        check(
            "Mixed mode: explicit provenance for "
            "outlier_correction (newer) + detect_path for "
            "import_pose (older) — both read CURRENT in the bulk "
            "function (depends_on staleness math composes correctly)",
            bulk["outlier_correction"] == SectionStatus.CURRENT
            and bulk["import_pose"] == SectionStatus.CURRENT,
            detail=(
                f"outlier_correction={bulk['outlier_correction'].value}, "
                f"import_pose={bulk['import_pose'].value}"
            ),
        )

    # 5. Missing project.toml — detect_path still works.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # NO project.toml. detect_path evidence only.
        (root / "sources" / "pose").mkdir(parents=True)
        (root / "sources" / "pose" / "v1.csv").write_text("p")
        # Patch 122fc — bases-match needs matching video.
        (root / "sources" / "videos").mkdir(parents=True)
        (root / "sources" / "videos" / "v1.mp4").write_text("v")
        cfg = root / "project.toml"  # doesn't exist
        s = get_all_statuses(str(cfg))["import_pose"]
        check(
            "Missing project.toml: get_all_statuses still picks "
            "up detect_path evidence (useful for partially-set-up "
            "projects where data files exist but provenance "
            "hasn't been written yet)",
            s == SectionStatus.CURRENT,
            detail=(f"got {s.value!r}"),
        )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    ea_src = (REPO_ROOT / "mufasa" / "data_processors"
              / "egocentric_aligner.py").read_text()
    check(
        "122ev state preserved: egocentric_aligner accepts both "
        ".csv and .parquet",
        "'.parquet'" in ea_src and "'.csv'" in ea_src,
    )

    rw_src = (REPO_ROOT / "mufasa" / "utils"
              / "read_write.py").read_text()
    check(
        "122eu state preserved: get_fn_ext handles empty extensions",
        "if not file_extension:" in rw_src,
    )

    pp = SECTIONS.get("pixels_per_mm")
    check(
        "122es state preserved: pixels_per_mm has detect_path",
        pp is not None and callable(pp.detect_path),
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
        f"smoke_122ew_get_all_statuses_detect_path: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
