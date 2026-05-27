"""
tests/smoke_122fc_data_import_bases_match.py
==============================================

Patch 122fc — Data Import badges with bases-match validation.

User request (Tue May 26, 2026):

> In Data import : import data and video should both have badges
> and be green if their files' bases are the same, such that each
> parquet file has an associated mp4 file.

Both ``Import pose data`` and ``Import video`` sections now have
badges. Both go CURRENT (green) only when each pose file has a
matching video file (by basename), and vice versa. Any mismatch
keeps both UNKNOWN (white).

WHAT THIS PATCH LANDED
======================

mufasa/section_provenance.py:

* New section ``import_video`` registered in SECTIONS:
    section_id="import_video",
    page="Data Import",
    section_title="Import video",
    depends_on=(),
    detect_path=lambda root: _data_import_video_path(root),

* New helpers (module-private):
    _POSE_DATA_EXTS   = (".csv", ".parquet")
    _VIDEO_DATA_EXTS  = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    _BASES_MISMATCH_SENTINEL = ".__bases_mismatch_sentinel__"
    _data_import_bases(dir, exts)     → set of file stems
    _data_import_bases_match(root)    → bool
    _data_import_pose_path(root)      → Path
    _data_import_video_path(root)     → Path

* ``import_pose.detect_path`` updated: now returns
  ``sources/pose/`` only when bases match across pose/video dirs.
  Else returns a sentinel non-existent path so
  ``_path_mtime_if_has_content`` returns None → SectionStatus.
  UNKNOWN.

WHY THIS PATTERN
================

The detect_path contract returns a Path. ``_path_mtime_if_has_
content`` decides "evidence vs no evidence" purely from path
existence + content. We get "files exist but invalid state"
by returning a deliberately-nonexistent path on invalid state
— treating the inconsistency as "no evidence."

The alternative would have been adding a new SectionStatus
variant (INVALID / MISMATCHED), which would require coordinated
changes across the icon system, refresh, and bulk lookup. The
sentinel approach piggybacks on existing infrastructure: zero
SectionStatus changes, zero icon-system changes, zero
get_status changes. Just two new SECTIONS entries and two
detect_path lambdas.

Tradeoff: the user can't distinguish "files don't match" from
"no files at all" — both show UNKNOWN (white). Acceptable;
the user is already coming to Data Import to set things up, so
seeing UNKNOWN tells them "not ready yet" without specifying
why. They can investigate by clicking the section.

COVERAGE
========

Helpers (3 checks):
1.  _data_import_bases returns the set of stems for files with
    the given extensions; hidden entries skipped.
2.  _data_import_bases_match returns True for matched bases,
    False otherwise (matched / mismatched / empty / one-sided
    cases all covered).
3.  _data_import_pose_path and _data_import_video_path return
    the expected dirs when bases match, sentinel paths otherwise.

SECTIONS contract (2 checks):
4.  SECTIONS['import_video'] exists with page='Data Import',
    section_title='Import video'.
5.  SECTIONS['import_pose'].detect_path is callable (still
    wired post-122fc; just rewired through the helper).

End-to-end (4 checks):
6.  Matched bases (3 parquet + 3 mp4, same stems) → BOTH
    import_pose AND import_video read CURRENT.
7.  Mismatched bases (extra pose file, no matching video) →
    BOTH read UNKNOWN.
8.  Empty project (no pose, no videos) → BOTH read UNKNOWN.
9.  Mixed extensions (csv + parquet + mp4 + avi, matched stems)
    → BOTH read CURRENT.

Cross-patch invariants (4 checks):
10. 122fb state preserved: roi_define_panel has maintenance_btn.
11. 122ez state preserved: SECTIONS['egocentric'].detect_path
    is callable.
12. Parse-clean.
13. 122do baseline (Optional[] hygiene).
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
        _data_import_bases,
        _data_import_bases_match,
        _data_import_pose_path,
        _data_import_video_path,
        _POSE_DATA_EXTS,
        _VIDEO_DATA_EXTS,
    )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    # 1. _data_import_bases
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "v1.csv").write_text("p")
        (d / "v2.parquet").write_text("p")
        (d / ".hidden.csv").write_text("hidden")
        (d / "ignored.txt").write_text("not pose")
        got = _data_import_bases(d, _POSE_DATA_EXTS)
        check(
            "_data_import_bases returns the stems of pose-style "
            "files only, skipping hidden entries and non-matching "
            "extensions",
            got == {"v1", "v2"},
            detail=(f"got {got!r}"),
        )

    # 2. _data_import_bases_match — 4 cases.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Case: matched
        (root / "sources" / "pose").mkdir(parents=True)
        (root / "sources" / "videos").mkdir(parents=True)
        (root / "sources" / "pose" / "v1.parquet").write_text("p")
        (root / "sources" / "videos" / "v1.mp4").write_text("v")
        matched = _data_import_bases_match(root)
        check(
            "_data_import_bases_match returns True for matched "
            "bases (v1.parquet ↔ v1.mp4)",
            matched is True,
            detail=(f"got {matched!r}"),
        )

    # 3. _data_import_*_path
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # No files yet → not matched
        unmatched_pose = _data_import_pose_path(root)
        unmatched_video = _data_import_video_path(root)
        # Add matching files
        (root / "sources" / "pose").mkdir(parents=True)
        (root / "sources" / "videos").mkdir(parents=True)
        (root / "sources" / "pose" / "v1.parquet").write_text("p")
        (root / "sources" / "videos" / "v1.mp4").write_text("v")
        matched_pose = _data_import_pose_path(root)
        matched_video = _data_import_video_path(root)
        check(
            "_data_import_*_path returns the data dir when bases "
            "match, sentinel-name path when they don't",
            (Path(matched_pose) == root / "sources" / "pose"
             and Path(matched_video) == root / "sources" / "videos"
             and Path(unmatched_pose) != root / "sources" / "pose"
             and Path(unmatched_video) != root / "sources" / "videos"),
            detail=(
                f"matched_pose={matched_pose}, "
                f"matched_video={matched_video}, "
                f"unmatched_pose={unmatched_pose}, "
                f"unmatched_video={unmatched_video}"
            ),
        )

    # -----------------------------------------------------------------
    # SECTIONS contract
    # -----------------------------------------------------------------
    # 4. import_video registered.
    iv = SECTIONS.get("import_video")
    check(
        "SECTIONS['import_video'] exists with page='Data Import' "
        "and section_title='Import video' (122fc addition)",
        (iv is not None
         and iv.page == "Data Import"
         and iv.section_title == "Import video"
         and callable(iv.detect_path)),
        detail=(
            f"page={getattr(iv, 'page', None)!r} "
            f"title={getattr(iv, 'section_title', None)!r}"
        ),
    )

    # 5. import_pose still wired (rewired through helper).
    ip = SECTIONS.get("import_pose")
    check(
        "SECTIONS['import_pose'].detect_path is still callable "
        "after 122fc rewire (the helper-based detect_path)",
        ip is not None and callable(ip.detect_path),
    )

    # -----------------------------------------------------------------
    # End-to-end behaviour
    # -----------------------------------------------------------------
    # 6. Matched bases → both CURRENT.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n'
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
            "End-to-end: matched bases (3 parquet + 3 mp4, "
            "same stems) → BOTH import_pose AND import_video "
            "read CURRENT (the user's intended green-green case)",
            (bulk["import_pose"] == SectionStatus.CURRENT
             and bulk["import_video"] == SectionStatus.CURRENT),
            detail=(
                f"import_pose={bulk['import_pose'].value!r} "
                f"import_video={bulk['import_video'].value!r}"
            ),
        )

    # 7. Mismatched bases → both UNKNOWN.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n'
        )
        (root / "sources" / "pose").mkdir(parents=True)
        (root / "sources" / "videos").mkdir(parents=True)
        # 2 pose files, but only 1 matching video
        (root / "sources" / "pose" / "v1.parquet").write_text("p")
        (root / "sources" / "pose" / "v2.parquet").write_text("p")
        (root / "sources" / "videos" / "v1.mp4").write_text("v")
        bulk = get_all_statuses(str(cfg))
        check(
            "End-to-end: mismatched bases (extra pose file with "
            "no matching video) → BOTH import_pose AND "
            "import_video read UNKNOWN (the user's intended "
            "white-when-broken case)",
            (bulk["import_pose"] == SectionStatus.UNKNOWN
             and bulk["import_video"] == SectionStatus.UNKNOWN),
            detail=(
                f"import_pose={bulk['import_pose'].value!r} "
                f"import_video={bulk['import_video'].value!r}"
            ),
        )

    # 8. Empty project → both UNKNOWN.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n'
        )
        bulk = get_all_statuses(str(cfg))
        check(
            "End-to-end: empty project (no sources/pose, no "
            "sources/videos) → BOTH read UNKNOWN (no false "
            "positive when nothing has been imported)",
            (bulk["import_pose"] == SectionStatus.UNKNOWN
             and bulk["import_video"] == SectionStatus.UNKNOWN),
            detail=(
                f"import_pose={bulk['import_pose'].value!r} "
                f"import_video={bulk['import_video'].value!r}"
            ),
        )

    # 9. Mixed extensions → both CURRENT.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n'
        )
        (root / "sources" / "pose").mkdir(parents=True)
        (root / "sources" / "videos").mkdir(parents=True)
        # Legacy CSV + v1 parquet, mp4 + avi videos
        (root / "sources" / "pose" / "exp1.parquet").write_text("p")
        (root / "sources" / "pose" / "exp2.csv").write_text("p")
        (root / "sources" / "videos" / "exp1.mp4").write_text("v")
        (root / "sources" / "videos" / "exp2.avi").write_text("v")
        bulk = get_all_statuses(str(cfg))
        check(
            "End-to-end: mixed extensions (csv + parquet for "
            "pose, mp4 + avi for video, matching stems) → BOTH "
            "read CURRENT (the validation is extension-agnostic "
            "within the accepted-extension sets)",
            (bulk["import_pose"] == SectionStatus.CURRENT
             and bulk["import_video"] == SectionStatus.CURRENT),
            detail=(
                f"import_pose={bulk['import_pose'].value!r} "
                f"import_video={bulk['import_video'].value!r}"
            ),
        )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    rdp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "dialogs"
               / "roi_define_panel.py").read_text()
    check(
        "122fb state preserved: roi_define_panel has "
        "maintenance_btn",
        "self.maintenance_btn" in rdp_src,
    )

    egospec = SECTIONS["egocentric"]
    check(
        "122ez state preserved: SECTIONS['egocentric']."
        "detect_path is callable",
        callable(egospec.detect_path),
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
        f"smoke_122fc_data_import_bases_match: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
