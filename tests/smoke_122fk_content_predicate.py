"""
tests/smoke_122fk_content_predicate.py
==========================================

Patch 122fk — smarter detect_path content-predicate semantics.

Standing item from 122es. Adds a new ``content_predicate`` field
to ``SectionSpec`` that gates whether the ``detect_path``
evidence COUNTS as a valid implicit run. Cleans up the
sentinel-non-existent-path workaround introduced in 122fc.

PROBLEM (PRE-122fk)
===================

The implicit-evidence path resolved by ``_resolve_run_at`` was:

    path = spec.detect_path(project_root)
    return _path_mtime_if_has_content(path)

This worked when "has the section produced output?" mapped
cleanly to "does the canonical output location exist with
content?" But some sections need a SEMANTIC check beyond
existence:

* **Cross-directory consistency** (122fc): import_pose +
  import_video should both go green ONLY if file basenames match
  across pose/ and videos/. The workaround: ``detect_path``
  returned a sentinel non-existent path when bases didn't match,
  causing ``_path_mtime_if_has_content`` to return None. Worked
  but conflated "where to look" with "does the lookup count."

* **Future use cases** also benefit:
  - "directory contains files with the .parquet extension"
  - "file contains a specific column" (column-presence check)
  - "ROI definitions file lists at least one ROI"

THE FIX
=======

New ``content_predicate: Callable[[Path], bool] | None`` field on
SectionSpec. Signature: ``predicate(project_root) -> bool``.
Applied AFTER ``_path_mtime_if_has_content`` in
``_resolve_run_at``:

    1. If explicit provenance entry exists → return its
       last_run_at (unchanged).
    2. Else: compute mtime via detect_path + _path_mtime_if_has_
       content. If None → return None (unchanged).
    3. NEW: if content_predicate is not None, call it with the
       project root. If it returns False (or raises) → return
       None (UNKNOWN).
    4. Else return the mtime.

The predicate is called with project_root (not the detect_path
output) so it can cross-reference other locations without
walking back up from a per-section subdir.

REFACTOR
========

122fc's sentinel-path machinery is replaced:

REMOVED from mufasa/section_provenance.py:
* ``_data_import_pose_path(root) -> Path`` — returned the dir
  conditionally on bases-match.
* ``_data_import_video_path(root) -> Path`` — same for video.
* ``_BASES_MISMATCH_SENTINEL`` — module-level constant for the
  ".__bases_mismatch_sentinel__" filename.

KEPT:
* ``_data_import_bases(dir_, exts)`` — utility, used by the
  predicate.
* ``_data_import_bases_match(root)`` — the predicate itself.

UPDATED:
* SECTIONS["import_pose"]:
    detect_path = lambda root: root / "sources" / "pose"
    content_predicate = _data_import_bases_match
* SECTIONS["import_video"]:
    detect_path = lambda root: root / "sources" / "videos"
    content_predicate = _data_import_bases_match

Behaviour is bit-identical: matched bases → both green,
mismatched bases → both white, empty project → both white.
Verified end-to-end below.

COVERAGE
========

SectionSpec API (2 checks):
1.  SectionSpec dataclass declares ``content_predicate`` field
    with default None.
2.  Default is None for the SectionSpec instances that don't
    declare one.

_resolve_run_at semantics (4 checks):
3.  With content_predicate=None and a valid detect_path with
    content → returns the mtime (status quo behaviour).
4.  With content_predicate returning True → returns the mtime
    (predicate accepts).
5.  With content_predicate returning False → returns None
    (predicate gates).
6.  With content_predicate that raises → returns None
    (soft-fail, same swallow-and-treat-as-False as detect_path
    errors).

122fc refactor (3 checks):
7.  SECTIONS["import_pose"].detect_path returns
    <root>/sources/pose unconditionally (no sentinel).
8.  SECTIONS["import_pose"].content_predicate is
    _data_import_bases_match.
9.  Sentinel-path helpers and constant are removed from the
    module (_data_import_pose_path, _data_import_video_path,
    _BASES_MISMATCH_SENTINEL).

End-to-end (3 checks):
10. Project with matched bases → both import_pose AND
    import_video read CURRENT (bit-identical to 122fc).
11. Project with mismatched bases → both read UNKNOWN.
12. Empty project → both read UNKNOWN.

Cross-patch invariants (3 checks):
13. 122fj state preserved: features_compute_subset registered.
14. 122fc state preserved: _data_import_bases_match still
    exists and works.
15. Parse-clean.
"""
from __future__ import annotations

import ast
import sys
import tempfile
from datetime import datetime, timezone
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
        SECTIONS, SectionSpec, SectionStatus,
        _resolve_run_at, _data_import_bases_match,
        get_all_statuses,
    )

    # -----------------------------------------------------------------
    # SectionSpec API
    # -----------------------------------------------------------------
    # 1. Field declared.
    check(
        "SectionSpec dataclass declares the new "
        "``content_predicate`` field (122fk addition)",
        "content_predicate" in SectionSpec.__dataclass_fields__,
        detail=(
            f"fields: "
            f"{sorted(SectionSpec.__dataclass_fields__.keys())}"
        ),
    )

    # 2. Default is None for sections that don't declare one.
    interp = SECTIONS.get("interpolate")
    check(
        "SectionSpec.content_predicate defaults to None for "
        "sections that don't declare one (backwards-compatible "
        "with all existing specs)",
        interp is not None and interp.content_predicate is None,
    )

    # -----------------------------------------------------------------
    # _resolve_run_at semantics
    # -----------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Build a fake "evidence" dir with one file so
        # _path_mtime_if_has_content returns a real mtime.
        evidence = root / "evidence"
        evidence.mkdir()
        (evidence / "file.txt").write_text("data")

        # Helper to construct a SectionSpec on the fly.
        def _spec(pred):
            return SectionSpec(
                section_id="test",
                page="Test",
                section_title="Test",
                detect_path=lambda r: evidence,
                content_predicate=pred,
            )

        # 3. No predicate → mtime returned.
        out = _resolve_run_at({}, "test", _spec(None), root)
        check(
            "With content_predicate=None, _resolve_run_at "
            "returns the mtime from _path_mtime_if_has_content "
            "(status quo behaviour preserved)",
            isinstance(out, datetime),
            detail=(f"got {out!r}"),
        )

        # 4. Predicate returns True → mtime returned.
        out = _resolve_run_at(
            {}, "test", _spec(lambda r: True), root,
        )
        check(
            "With content_predicate returning True, "
            "_resolve_run_at returns the mtime (predicate "
            "accepts the evidence)",
            isinstance(out, datetime),
            detail=(f"got {out!r}"),
        )

        # 5. Predicate returns False → None.
        out = _resolve_run_at(
            {}, "test", _spec(lambda r: False), root,
        )
        check(
            "With content_predicate returning False, "
            "_resolve_run_at returns None — the predicate gates "
            "the evidence (key 122fk semantics)",
            out is None,
            detail=(f"got {out!r}"),
        )

        # 6. Predicate raises → None.
        def _angry_predicate(r):
            raise RuntimeError("oops")
        out = _resolve_run_at(
            {}, "test", _spec(_angry_predicate), root,
        )
        check(
            "With content_predicate that raises, _resolve_run_at "
            "returns None (soft-fail, same swallow-and-treat-as-"
            "False as detect_path errors)",
            out is None,
        )

    # -----------------------------------------------------------------
    # 122fc refactor — sentinel-path machinery gone
    # -----------------------------------------------------------------
    ip = SECTIONS["import_pose"]
    fake_root = Path("/tmp/test-fk-pose-path")
    check(
        "SECTIONS['import_pose'].detect_path returns "
        "<root>/sources/pose UNCONDITIONALLY (no sentinel path "
        "based on bases-match; the predicate gates evidence "
        "separately)",
        ip.detect_path(fake_root) == fake_root / "sources" / "pose",
        detail=(f"got {ip.detect_path(fake_root)!r}"),
    )

    check(
        "SECTIONS['import_pose'].content_predicate is "
        "_data_import_bases_match (the 122fk migration of 122fc's "
        "cross-directory consistency check)",
        ip.content_predicate is _data_import_bases_match,
        detail=(f"got {ip.content_predicate!r}"),
    )

    import mufasa.section_provenance as sp
    legacy_gone = (
        not hasattr(sp, "_data_import_pose_path")
        and not hasattr(sp, "_data_import_video_path")
        and not hasattr(sp, "_BASES_MISMATCH_SENTINEL")
    )
    check(
        "Sentinel-path helpers + constant removed from the "
        "module (_data_import_pose_path, _data_import_video_path, "
        "_BASES_MISMATCH_SENTINEL no longer exist)",
        legacy_gone,
    )

    # -----------------------------------------------------------------
    # End-to-end: bit-identical to 122fc behaviour
    # -----------------------------------------------------------------
    # Matched bases → both CURRENT.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n',
        )
        (root / "sources" / "pose").mkdir(parents=True)
        (root / "sources" / "videos").mkdir(parents=True)
        for base in ("v1", "v2", "v3"):
            (root / "sources" / "pose"
             / f"{base}.parquet").write_text("p")
            (root / "sources" / "videos"
             / f"{base}.mp4").write_text("v")
        bulk = get_all_statuses(str(cfg))
        check(
            "End-to-end: matched bases → BOTH import_pose AND "
            "import_video read CURRENT (bit-identical to 122fc; "
            "verifies the 122fk migration didn't regress)",
            (bulk["import_pose"] == SectionStatus.CURRENT
             and bulk["import_video"] == SectionStatus.CURRENT),
            detail=(
                f"pose={bulk['import_pose'].value!r} "
                f"video={bulk['import_video'].value!r}"
            ),
        )

    # Mismatched bases → both UNKNOWN (via predicate gating).
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n',
        )
        (root / "sources" / "pose").mkdir(parents=True)
        (root / "sources" / "videos").mkdir(parents=True)
        (root / "sources" / "pose" / "v1.parquet").write_text("p")
        (root / "sources" / "pose" / "v2.parquet").write_text("p")
        (root / "sources" / "videos" / "v1.mp4").write_text("v")
        bulk = get_all_statuses(str(cfg))
        check(
            "End-to-end: mismatched bases → BOTH UNKNOWN "
            "(content_predicate _data_import_bases_match returns "
            "False; gates the evidence even though detect_path "
            "now returns the real dir)",
            (bulk["import_pose"] == SectionStatus.UNKNOWN
             and bulk["import_video"] == SectionStatus.UNKNOWN),
            detail=(
                f"pose={bulk['import_pose'].value!r} "
                f"video={bulk['import_video'].value!r}"
            ),
        )

    # Empty project → both UNKNOWN.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n',
        )
        bulk = get_all_statuses(str(cfg))
        check(
            "End-to-end: empty project → BOTH UNKNOWN (no "
            "regression on the no-data case)",
            (bulk["import_pose"] == SectionStatus.UNKNOWN
             and bulk["import_video"] == SectionStatus.UNKNOWN),
        )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    check(
        "122fj state preserved: features_compute_subset registered",
        "features_compute_subset" in SECTIONS,
    )

    check(
        "122fc state preserved: _data_import_bases_match still "
        "defined and works (the helper survived the 122fk "
        "refactor; only the sentinel-path machinery went away)",
        callable(_data_import_bases_match),
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
        f"smoke_122fk_content_predicate: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
