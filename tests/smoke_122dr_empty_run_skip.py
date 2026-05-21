"""
tests/smoke_122dr_empty_run_skip.py
=====================================

Patch 122dr: ``_latest_run_or_parent`` is now content-aware — empty
or aborted run subdirs no longer shadow earlier populated runs.

In-the-wild repro
-----------------
A user testing the Features page (via docs/testing_workflow.md) hit
"No eligible videos in this project" even though
``derived/outlier_corrected/`` contained 67 ``.parquet`` files of
outlier-corrected data. The directory listing showed two run-id
subdirs::

    derived/outlier_corrected/
      20260518-192433-7f64a3/   (link count 5 → 3 subdirs, populated)
      20260520-233610-6203f1/   (link count 2 → empty)
      ...flat .parquet files from legacy migration...

The pre-122dr ``_latest_run_or_parent`` sorted lexically and picked
``20260520-233610-6203f1`` (newer name); ``glob("*.parquet")`` against
the empty directory returned ``[]``; ``FeatureSubsetsCalculator``
saw ``data_paths == []``; the form raised the empty-state dialog.

122dr fixes this by walking newest-first and skipping runs that don't
contain any files of the project's declared ``file_type`` (recursive).

Code changes
------------
1. ``mufasa/project_layout.py`` — new module-level helper
   :func:`latest_populated_run_or_parent` that takes ``stage_parent``
   and ``file_type`` explicitly. Extracted from the nested closure
   inside ``ConfigReader._apply_v1_path_overrides`` so it's directly
   testable without spinning up a full ``ConfigReader`` instance.
2. ``mufasa/mixins/config_reader.py`` — the closure
   ``_latest_run_or_parent`` is now a thin shim that forwards to the
   module-level helper, passing ``self.file_type``. Behavior identical
   to the extracted helper.

Coverage
--------
1.  ``mufasa.project_layout`` exposes ``latest_populated_run_or_parent``.
2.  The function picks the populated run when both empty and populated
    runs are present (the user's exact bug layout).
3.  Empty runs at every position (newest, middle, only) are skipped.
4.  All runs empty → falls back to stage parent (preserves legacy
    behavior for migrated projects with flat files).
5.  No runs at all → falls back to stage parent (preserves the
    function's original fallback semantics).
6.  Stage parent doesn't exist → returns the parent path as a string
    anyway (caller's ``glob`` against a non-existent dir returns ``[]``
    cleanly; we don't want to raise here).
7.  Non-run-id subdirs (e.g., ``movement_imported_20260511/``) are
    ignored entirely — the function only considers names that pass
    ``is_run_id``.
8.  File-type mismatch: a run dir containing ``.csv`` files is treated
    as empty when the project declares ``file_type="parquet"`` (and
    vice versa). This is the right behavior — the consumer's glob
    would also see zero matches.
9.  Files in subdirectories of a run dir count toward "populated"
    (``rglob``, not ``glob``).
10. ``ConfigReader._apply_v1_path_overrides`` still references the
    extracted helper (sanity check on the replacement edit).
11. All ``mufasa/**/*.py`` parse cleanly.
12. 122do baseline tripwire: no ``Optional[`` in non-docstring
    positions across ``mufasa/ui_qt/``.
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
    # 1. Helper exported.
    from mufasa.project_layout import (
        latest_populated_run_or_parent as resolve,
    )
    check(
        "mufasa.project_layout exposes latest_populated_run_or_parent",
        callable(resolve),
    )

    # 2. The user's exact bug layout — populated old run + empty new run.
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"
        populated = stage / "20260518-192433-7f64a3"
        empty = stage / "20260520-233610-6203f1"
        populated.mkdir(parents=True)
        empty.mkdir(parents=True)
        (populated / "video1.parquet").touch()
        (populated / "video2.parquet").touch()
        result = resolve(stage, "parquet")
        check(
            "User's exact bug layout (populated old run + empty "
            "newer run): resolver picks the populated run, NOT the "
            "empty newer one",
            result == str(populated),
            detail=f"got {result!r}",
        )

    # 3a. Three runs: middle one is populated, newest is empty.
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"
        oldest = stage / "20260101-000000-aaaaaa"
        middle = stage / "20260201-000000-bbbbbb"  # populated
        newest = stage / "20260301-000000-cccccc"  # empty
        for d in (oldest, middle, newest):
            d.mkdir(parents=True)
        (middle / "video.parquet").touch()
        result = resolve(stage, "parquet")
        check(
            "Three runs (oldest empty, middle populated, newest "
            "empty): picks the middle populated run",
            result == str(middle),
            detail=f"got {result!r}",
        )

    # 3b. Three runs: newest is populated. Should pick it directly.
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"
        oldest = stage / "20260101-000000-aaaaaa"
        middle = stage / "20260201-000000-bbbbbb"
        newest = stage / "20260301-000000-cccccc"
        for d in (oldest, middle, newest):
            d.mkdir(parents=True)
        (newest / "video.parquet").touch()
        result = resolve(stage, "parquet")
        check(
            "Three runs with newest populated: picks the newest "
            "(no regression vs pre-122dr behavior)",
            result == str(newest),
            detail=f"got {result!r}",
        )

    # 4. All runs empty → falls back to stage parent.
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"
        run_a = stage / "20260101-000000-aaaaaa"
        run_b = stage / "20260201-000000-bbbbbb"
        for d in (run_a, run_b):
            d.mkdir(parents=True)
        # And put a flat file directly in the stage parent — this is
        # the legacy/migrated layout. The fallback to stage_parent
        # is exactly what makes those files discoverable.
        (stage / "legacy_video.parquet").touch()
        result = resolve(stage, "parquet")
        check(
            "All runs empty → falls back to stage parent (preserves "
            "legacy-layout backstop)",
            result == str(stage),
            detail=f"got {result!r}",
        )

    # 5. No run subdirs at all → stage parent.
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"
        stage.mkdir(parents=True)
        (stage / "legacy_video.parquet").touch()
        result = resolve(stage, "parquet")
        check(
            "No run-id subdirs at all → returns stage parent",
            result == str(stage),
            detail=f"got {result!r}",
        )

    # 6. Stage parent doesn't exist → returns parent path as string.
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"  # never created
        result = resolve(stage, "parquet")
        check(
            "Stage parent doesn't exist → returns parent path as "
            "string (caller's glob handles the missing dir cleanly)",
            result == str(stage),
            detail=f"got {result!r}",
        )

    # 7. Non-run-id subdirs ignored.
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"
        stage.mkdir(parents=True)
        # Mimics the user's project — has both run-id dirs AND
        # legacy `movement_imported_<date>` dirs. The latter must
        # be ignored.
        legacy_a = stage / "movement_imported_20260511"
        legacy_b = stage / "movement_location_imported_20260511"
        run = stage / "20260518-192433-7f64a3"
        for d in (legacy_a, legacy_b, run):
            d.mkdir(parents=True)
        (legacy_a / "legacy.parquet").touch()  # decoy — should not influence
        (run / "actual.parquet").touch()
        result = resolve(stage, "parquet")
        check(
            "Non-run-id subdirs (movement_imported_*) are ignored "
            "even when they contain matching files — only run-id dirs "
            "are considered, and the populated run wins",
            result == str(run),
            detail=f"got {result!r}",
        )

    # 8. file_type mismatch makes a run effectively empty.
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"
        wrong_type = stage / "20260301-000000-cccccc"  # newer, .csv
        right_type = stage / "20260201-000000-bbbbbb"  # older, .parquet
        for d in (wrong_type, right_type):
            d.mkdir(parents=True)
        (wrong_type / "video.csv").touch()
        (right_type / "video.parquet").touch()
        result = resolve(stage, "parquet")
        check(
            "file_type mismatch: a newer run with only .csv files is "
            "treated as empty when the project file_type is parquet "
            "(picks the older .parquet run)",
            result == str(right_type),
            detail=f"got {result!r}",
        )

    # 9. Recursive file detection.
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "outlier_corrected"
        empty = stage / "20260301-000000-cccccc"  # newer but empty
        nested = stage / "20260201-000000-bbbbbb"  # older w/ nested file
        empty.mkdir(parents=True)
        (nested / "subdir1" / "subdir2").mkdir(parents=True)
        (nested / "subdir1" / "subdir2" / "video.parquet").touch()
        result = resolve(stage, "parquet")
        check(
            "Files in subdirectories of a run dir count toward "
            "populated (rglob, not flat glob) — handles the user's "
            "actual layout where run dirs have 3 subdirs",
            result == str(nested),
            detail=f"got {result!r}",
        )

    # 10. ConfigReader still uses the helper.
    cr_src = (REPO_ROOT / "mufasa" / "mixins" /
              "config_reader.py").read_text()
    check(
        "ConfigReader._apply_v1_path_overrides forwards to the "
        "extracted helper",
        "latest_populated_run_or_parent" in cr_src,
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
            parse_errors.append(
                f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    # 12. 122do baseline tripwire.
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
        f"smoke_122dr_empty_run_skip: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
