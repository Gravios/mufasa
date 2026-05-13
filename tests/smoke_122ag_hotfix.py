"""
tests/smoke_122ag_hotfix.py
===========================

Patch 122ag: hotfix on 122ab and 122x. Four v1-awareness
oversights caught by the user after applying through 122ae-1:

1. ``mufasa/ui_qt/dialogs/roi_define_panel.py`` imported the
   pre-122ab name ``_project_path_from_config`` from
   ``roi_video_table``. That function got renamed to
   ``_project_paths_lite`` (and now returns a paths dict
   instead of a bare project_path string). The
   ROIDefinePanel raised ImportError on open.

2. ``mufasa/ui_qt/forms/video_info.py:_video_info_path``
   hardcoded ``<project>/logs/video_info.csv``. Wrong for v1
   projects (canonical path is ``<root>/sources/video_info.csv``).
   The Preprocessing → Video Calibration table showed
   'no existing file' message for v1 projects.

3. ``mufasa/ui_qt/forms/video_info.py:_discover_rows``
   hardcoded ``<project>/videos/`` and
   ``<project>/csv/input_csv/``. Wrong for v1 projects
   (canonical paths are ``<root>/sources/videos/`` and
   ``<root>/sources/pose/``). The same table came up empty
   for v1 projects (no rows discoverable).

4. ``mufasa/ui_qt/clip_review.py`` read
   ``[General settings].project_path`` directly via
   :mod:`configparser`. Same bug shape that 122ab fixed for
   ``frame_labeller.py``: configparser silently parses zero
   sections out of a TOML file, then crashes on the missing
   ``[General settings]`` section. v1 users couldn't review
   classifier predictions at all.

Coverage (AST + source inspection — Qt + cv2 not importable
in sandbox):

* roi_define_panel.py imports `_project_paths_lite`, not
  the old name.
* roi_define_panel.py stores self.video_dir + self.project_path
  from the helper dict.
* roi_define_panel.py uses _list_project_videos with
  self.video_dir (the new signature).
* roi_define_panel.py resolves ROI h5 path via the layout
  helper.
* video_info.py routes both _video_info_path and
  _discover_rows through project_paths_from_config.
* clip_review.py no longer imports configparser; routes
  through project_paths_from_config + project_metadata_from_config.
* No code-level references to `_project_path_from_config`
  remain anywhere in mufasa/ (comment mentions are allowed).
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


def main() -> int:
    # ==================================================================
    # Fix 1: roi_define_panel.py
    # ==================================================================
    rdp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "dialogs"
               / "roi_define_panel.py").read_text()
    rdp_tree = ast.parse(rdp_src)
    check(
        "roi_define_panel imports _project_paths_lite "
        "(post-122ab name)",
        "_project_paths_lite" in rdp_src,
    )
    # The old name appears in the 122ag patch-note comment;
    # confirm it doesn't appear as a code-level identifier.
    code_only_old = any(
        "_project_path_from_config" in line
        and not line.lstrip().startswith("#")
        for line in rdp_src.splitlines()
    )
    check(
        "roi_define_panel has no code-level references to the "
        "pre-122ab name (historical mention in comments OK)",
        not code_only_old,
    )
    check(
        "roi_define_panel stores self.video_dir from helper",
        "self.video_dir = paths" in rdp_src
        or 'self.video_dir = paths.get("video_dir"' in rdp_src,
    )
    check(
        "roi_define_panel stores self.project_path from helper",
        "self.project_path = paths" in rdp_src,
    )
    check(
        "roi_define_panel passes self.video_dir to "
        "_list_project_videos (post-122ab signature)",
        "_list_project_videos(self.video_dir)" in rdp_src,
    )
    check(
        "roi_define_panel resolves ROI h5 via helper "
        "(no hardcoded 'logs/measures/' join)",
        'paths_lite(self.config_path).get(\n            "roi_definitions_path"'
        in rdp_src
        or "roi_definitions_path" in rdp_src,
    )
    check(
        "roi_define_panel docstring/comments record the 122ag fix",
        "122ag" in rdp_src,
    )

    # ==================================================================
    # Fix 2 + 3: video_info.py
    # ==================================================================
    vi_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
              / "video_info.py").read_text()
    check(
        "VideoInfoForm._video_info_path routes through "
        "project_paths_from_config",
        "project_paths_from_config" in vi_src,
    )
    check(
        "VideoInfoForm reads 'video_info_path' key from helper",
        '"video_info_path"' in vi_src
        or "'video_info_path'" in vi_src,
    )
    check(
        "VideoInfoForm._discover_rows reads 'video_dir' key "
        "from helper",
        '"video_dir"' in vi_src
        or "'video_dir'" in vi_src,
    )
    check(
        "VideoInfoForm._discover_rows reads 'input_pose_dir' key "
        "from helper (v1-aware pose-dir fallback)",
        '"input_pose_dir"' in vi_src
        or "'input_pose_dir'" in vi_src,
    )
    check(
        "VideoInfoForm pose-dir fallback accepts .csv / .parquet "
        "/ .h5",
        '".h5"' in vi_src
        and '".parquet"' in vi_src
        and '".csv"' in vi_src,
    )
    check(
        "video_info.py records the 122ag fix",
        "122ag" in vi_src,
    )

    # ==================================================================
    # Fix 4: clip_review.py
    # ==================================================================
    cr_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "clip_review.py").read_text()
    cr_tree = ast.parse(cr_src)
    check(
        "clip_review no longer imports configparser",
        not any(
            isinstance(n, ast.Import)
            and any(a.name == "configparser" for a in n.names)
            for n in cr_tree.body
        ),
    )
    check(
        "clip_review uses project_paths_from_config",
        "project_paths_from_config" in cr_src,
    )
    check(
        "clip_review uses project_metadata_from_config "
        "(for v1 file_type)",
        "project_metadata_from_config" in cr_src,
    )
    check(
        "clip_review no longer reads "
        "[General settings].project_path directly",
        "'General settings'" not in cr_src
        and '"General settings"' not in cr_src,
    )
    check(
        "clip_review auto-locate uses 'machine_results_dir' "
        "from helper (no hardcoded 'csv/machine_results/' join)",
        "machine_results_dir" in cr_src,
    )
    check(
        "clip_review error message mentions both v1 and legacy "
        "projects (no 'project_config.ini'-only)",
        "project_config.ini" not in cr_src
        or "project.toml" in cr_src,
    )
    check(
        "clip_review records the 122ag fix",
        "122ag" in cr_src,
    )

    # ==================================================================
    # Cross-file: no stale `_project_path_from_config` refs anywhere
    # ==================================================================
    leaked = []
    for p in (REPO_ROOT / "mufasa").rglob("*.py"):
        src = p.read_text()
        for line in src.splitlines():
            if ("_project_path_from_config" in line
                    and not line.lstrip().startswith("#")):
                leaked.append(str(p.relative_to(REPO_ROOT)))
                break
    check(
        "No code-level references to '_project_path_from_config' "
        "remain anywhere in mufasa/ (comment mentions OK)",
        not leaked,
        detail=f"leaked in {leaked}" if leaked else "",
    )

    print(
        f"smoke_122ag_hotfix: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
