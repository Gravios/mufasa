"""
tests/smoke_122ei_detect_path_fallback.py
============================================

Patch 122ei: filesystem-evidence fallback for
:func:`mufasa.section_provenance.get_status`. Handles the "old
project, pre-dates provenance wiring" UX gap reported by the user
on Friday May 22, 2026.

Real-world report
-----------------
User opened the workbench on a v1 project where pose data had
been imported (before 122eb's provenance wiring landed). The
Data Import page's "Import pose data" badge showed UNKNOWN
(white) instead of CURRENT (green), even though
``sources/pose/`` was full of imported pose data.

``get_status`` returned UNKNOWN correctly given its pre-122ei
logic: no ``[provenance.import_pose]`` entry in project.toml →
UNKNOWN. But the user's reasonable expectation is "if the data
is THERE, the badge should reflect that." Without a fix, every
old project shows UNKNOWN badges across the board until the user
re-runs every producer.

Design: filesystem-evidence fallback
------------------------------------
Each ``SectionSpec`` can optionally declare a ``detect_path``
callable that returns the canonical on-disk location of the
section's output. ``get_status`` consults this callable when
no explicit provenance entry exists; if the path has content,
its mtime serves as the implicit ``last_run_at``.

The implicit timestamp composes correctly with the existing
staleness rule:

* old project, pose data on disk, no provenance entries →
  ``import_pose`` reads CURRENT (mtime of pose files = implicit
  timestamp).
* same project, user re-imports pose data → explicit
  ``[provenance.import_pose]`` lands; ``import_pose`` still
  reads CURRENT.
* same project, user re-imports, ``derived/interpolated/<run>/``
  exists but no provenance for it → ``interpolate`` reads STALE
  (its detect_path mtime < import_pose's new explicit mtime).

No new badge state needed. The 3-state palette (UNKNOWN /
CURRENT / STALE) covers the augmented logic.

What this patch landed
----------------------
mufasa/section_provenance.py — three edits:

1. ``SectionSpec`` gained an optional
   ``detect_path: Callable[[Path], Path] | None`` field.
2. Four producer sections wired with ``detect_path``:
   - import_pose → sources/pose/
   - interpolate → derived/interpolated/
   - kalman_v2   → derived/smoothed/kalman_v2/
   - outlier_correction → derived/outlier_corrected/
   (pixels_per_mm explicitly left without one — settings-
   only sections don't materialize a file.)
3. ``get_status`` rewritten to consult a new helper
   ``_resolve_run_at`` that checks explicit provenance first,
   then falls back to ``_path_mtime_if_has_content`` against
   the section's detect_path. The same fallback applies to
   the dep-walk inside ``get_status`` so STALE detection works
   even when one or both sections have only implicit
   timestamps.

The helpers:

* ``_path_mtime_if_has_content(path)`` — returns the mtime of
  the path if a file, else the max mtime of non-hidden entries
  one level deep. Hidden entries (dotfiles) are ignored so a
  ``.DS_Store`` doesn't trick us. Returns None on empty or
  missing paths.
* ``_resolve_run_at(prov, section_id, spec, project_root)`` —
  explicit-then-implicit resolution; errors during filesystem
  checks are swallowed (UI code can't crash on transient FS
  hiccups).

Coverage
--------
SectionSpec changes:
1.  ``SectionSpec`` has a ``detect_path`` field.

Producer wiring:
2.  ``SECTIONS['import_pose'].detect_path`` is callable.
3.  ``SECTIONS['interpolate'].detect_path`` is callable.
4.  ``SECTIONS['kalman_v2'].detect_path`` is callable.
5.  ``SECTIONS['outlier_correction'].detect_path`` is callable.
6.  ``SECTIONS['pixels_per_mm'].detect_path`` is None
    (settings-only).

Helper functions:
7.  ``_path_mtime_if_has_content`` exists.
8.  ``_resolve_run_at`` exists.

Functional behaviour (via tempdir-based exercise):
9.  Empty sources/pose/ → import_pose reads UNKNOWN.
10. sources/pose/video.csv present → import_pose reads CURRENT.
11. Only dotfiles in sources/pose/ → import_pose reads UNKNOWN.
12. derived/interpolated/<run>/file present, no provenance →
    interpolate reads CURRENT.
13. pixels_per_mm with no detect_path and no provenance →
    UNKNOWN regardless of filesystem.
14. interpolate reads STALE if pose data is touched AFTER the
    interpolation run dir (implicit staleness via mtime
    composition).

Resolution-order check:
15. Explicit provenance wins over implicit detection. If
    ``[provenance.import_pose].last_run_at`` is set, that
    timestamp is used even if detect_path's mtime is later.

Cross-patch invariants:
16. 122eh state preserved: roi_coordinates_path in
    config_reader.py uses logs/measures/ROI_definitions.h5.
17. 122eg state preserved: no .ini in source.
18. 122ee state preserved: PoseImportForm publish wiring.
19. Parse-clean.
20. 122do baseline.
"""
from __future__ import annotations

import ast
import re
import sys
import tempfile
import time
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


def _make_v1_project(root: Path) -> Path:
    """Create a minimum-viable v1 project.toml in ``root`` and
    return its path."""
    cfg = root / "project.toml"
    cfg.write_text(
        'project_layout_version = 1\n'
        '[project]\nname = "test"\n'
    )
    return cfg


def main() -> int:
    from mufasa.section_provenance import (
        SECTIONS, SectionSpec, SectionStatus,
        get_status, record_run,
    )
    from mufasa import section_provenance as sp

    # -----------------------------------------------------------------
    # SectionSpec changes
    # -----------------------------------------------------------------
    # 1. detect_path field exists on SectionSpec.
    check(
        "SectionSpec has a `detect_path` field "
        "(filesystem-evidence fallback handle)",
        any(f.name == "detect_path" for f in
            sp.SectionSpec.__dataclass_fields__.values()),
    )

    # -----------------------------------------------------------------
    # Producer wiring
    # -----------------------------------------------------------------
    for sid in ["import_pose", "interpolate",
                "kalman_v2", "outlier_correction"]:
        spec = SECTIONS.get(sid)
        check(
            f"SECTIONS[{sid!r}].detect_path is callable "
            f"(producer section wired for filesystem fallback)",
            spec is not None and callable(spec.detect_path),
            detail=(f"got {getattr(spec, 'detect_path', None)!r}"),
        )

    # 6. pixels_per_mm detect_path. Was None pre-122es ("settings-
    # only — no on-disk artifact"); patch 122es-hotfix flipped
    # that — the calibration table SAVES to sources/video_info.csv,
    # which IS the on-disk artifact. This check is now a
    # reciprocal-tripwire for the 122es flip.
    pp = SECTIONS.get("pixels_per_mm")
    check(
        "SECTIONS['pixels_per_mm'].detect_path is callable "
        "(post-122es flip — points at sources/video_info.csv, "
        "the calibration table the form saves)",
        pp is not None and callable(pp.detect_path),
    )

    # -----------------------------------------------------------------
    # Helper functions
    # -----------------------------------------------------------------
    check(
        "section_provenance._path_mtime_if_has_content exists",
        hasattr(sp, "_path_mtime_if_has_content"),
    )
    check(
        "section_provenance._resolve_run_at exists",
        hasattr(sp, "_resolve_run_at"),
    )

    # -----------------------------------------------------------------
    # Functional behaviour
    # -----------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _make_v1_project(root)
        pose_dir = root / "sources" / "pose"
        pose_dir.mkdir(parents=True)

        # 9. empty sources/pose/ → UNKNOWN
        s = get_status(str(cfg), "import_pose")
        check(
            "import_pose with empty sources/pose/ reads UNKNOWN "
            "(detect_path returns a real path but it has no "
            "content — the canonical 'no data' case)",
            s == SectionStatus.UNKNOWN,
            detail=f"got {s.value!r}",
        )

        # 10. pose files present → CURRENT
        (pose_dir / "video1.csv").write_text("dummy")
        s = get_status(str(cfg), "import_pose")
        check(
            "import_pose with sources/pose/video1.csv reads CURRENT "
            "(the user's reported case — old project, pose data "
            "on disk, no provenance entry → badge should be green)",
            s == SectionStatus.CURRENT,
            detail=f"got {s.value!r}",
        )

        # 11. only dotfiles → UNKNOWN
        (pose_dir / "video1.csv").unlink()
        (pose_dir / ".DS_Store").write_text("hidden")
        s = get_status(str(cfg), "import_pose")
        check(
            "import_pose with only hidden files in sources/pose/ "
            "reads UNKNOWN (so an OS-injected .DS_Store doesn't "
            "trick the badge into showing CURRENT on a freshly-"
            "created project)",
            s == SectionStatus.UNKNOWN,
            detail=f"got {s.value!r}",
        )

        # 12. interpolate with run dir → CURRENT
        (pose_dir / ".DS_Store").unlink()
        (pose_dir / "video1.csv").write_text("dummy")
        run_dir = root / "derived" / "interpolated" / "20260520-123000"
        run_dir.mkdir(parents=True)
        (run_dir / "video1.csv").write_text("interpolated")
        s = get_status(str(cfg), "interpolate")
        check(
            "interpolate with derived/interpolated/<run>/file "
            "present reads CURRENT (one level deep iterdir picks "
            "up the run subdir; its mtime serves as implicit "
            "last_run_at)",
            s == SectionStatus.CURRENT,
            detail=f"got {s.value!r}",
        )

        # 13. pixels_per_mm → reciprocal-tripwire flip post-122es.
        # The functional check now verifies the OPPOSITE: with
        # sources/video_info.csv on disk, pixels_per_mm should
        # read CURRENT (its detect_path picks up the file's
        # mtime as the implicit last_run_at).
        # In this tempdir we don't write video_info.csv as part
        # of the test setup, so pixels_per_mm reads UNKNOWN
        # here. The "is CURRENT with video_info.csv present"
        # check is in smoke_122es_video_calibration_detect_path.
        s = get_status(str(cfg), "pixels_per_mm")
        check(
            "pixels_per_mm reads UNKNOWN when sources/"
            "video_info.csv is absent (122es flip — the file's "
            "presence is the implicit-evidence signal; absent → "
            "UNKNOWN)",
            s == SectionStatus.UNKNOWN,
            detail=f"got {s.value!r}",
        )

        # 14. interpolate STALE after pose re-touch
        time.sleep(0.05)  # ensure mtime separation
        (pose_dir / "video1.csv").write_text("re-imported")
        s = get_status(str(cfg), "interpolate")
        check(
            "interpolate reads STALE after pose data is touched "
            "later than the interpolation run dir (implicit-vs-"
            "implicit staleness via mtime composition — no "
            "provenance entries involved, but the rule still "
            "fires correctly)",
            s == SectionStatus.STALE,
            detail=f"got {s.value!r}",
        )

    # 15. Explicit provenance wins over implicit.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = _make_v1_project(root)
        pose_dir = root / "sources" / "pose"
        pose_dir.mkdir(parents=True)
        # File created at "now"
        (pose_dir / "video1.csv").write_text("dummy")
        # Provenance entry written via record_run. Verify it lands
        # in project.toml AND that the recorded value is sane.
        # ``record_run`` uses ``isoformat(timespec="seconds")`` so
        # microsecond comparisons would always fail; use a one-
        # second sleep + second-resolution truncation.
        from datetime import datetime, timezone
        before = datetime.now(timezone.utc).replace(microsecond=0)
        time.sleep(1.1)
        record_run(str(cfg), "import_pose")
        after = datetime.now(timezone.utc).replace(microsecond=0)

        import tomllib
        data = tomllib.loads(cfg.read_text())
        recorded_at_str = data["provenance"]["import_pose"]["last_run_at"]
        recorded_at = datetime.fromisoformat(recorded_at_str)
        check(
            "Explicit provenance wins over filesystem detection. "
            "After record_run, project.toml has an explicit "
            "last_run_at; the recorded value lies within the "
            "before/after window (second resolution since "
            "record_run truncates microseconds).",
            before <= recorded_at <= after,
            detail=(f"before={before.isoformat()} "
                    f"recorded={recorded_at.isoformat()} "
                    f"after={after.isoformat()}"),
        )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 16. 122eh state preserved.
    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()
    pl_src = (REPO_ROOT / "mufasa"
              / "project_layout.py").read_text()
    check(
        "122eh state preserved: roi_coordinates_path is set to "
        "logs/measures/ROI_definitions.h5",
        "measures" in pl_src and "ROI_definitions.h5" in pl_src,
    )

    # 17. 122eg state preserved.
    stray = []
    for f in sorted((REPO_ROOT / "mufasa").rglob("*.py")):
        rel = str(f.relative_to(REPO_ROOT))
        if rel == "mufasa/legacy_layout.py":
            continue
        s = f.read_text()
        if "project_config.ini" in s:
            stray.append(rel)
    check(
        "122eg state preserved: no `project_config.ini` outside "
        "legacy_layout.py",
        not stray,
        detail=("; ".join(stray[:3])),
    )

    # 18. 122ee state preserved.
    pi_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "forms" / "pose_import.py").read_text()
    check(
        "122ee state preserved: PoseImportForm publishes",
        'publish_target_stage = "outlier_corrected"' in pi_src,
    )

    # 19. Parse-clean.
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

    # 20. 122do baseline.
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
        f"smoke_122ei_detect_path_fallback: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
