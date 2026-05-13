"""Smoke-test for mufasa.ui_qt.dialogs.roi_video_table helpers.

Tests the pure-Python helpers used by ROIVideoTableDialog —
project_path lookup, video listing, and ROI-status detection — without
requiring PySide6, h5py, or numba (the helpers are designed to be
testable in isolation).

    PYTHONPATH=. python tests/smoke_roi_video_table_helpers.py
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path


def _import_helpers():
    """Load only the pure helpers from the dialog module without
    executing the PySide6-dependent class definitions."""
    src_path = Path("mufasa/ui_qt/dialogs/roi_video_table.py").resolve()
    src = src_path.read_text()

    # Find the boundary between helpers and the Qt class
    cls_start = src.find("class ROIVideoTableDialog")
    assert cls_start > 0, "ROIVideoTableDialog class not found in module"
    helpers_src = src[:cls_start]

    # The helpers import PySide6 / cv2 / numpy at module top. We need
    # to keep those imports out of the test (cv2 is OK; PySide6 isn't
    # available in the test sandbox). Strip them — including
    # multi-line imports that span several lines.
    lines = helpers_src.splitlines(keepends=True)
    keep = []
    skip_until_close = False
    for ln in lines:
        if skip_until_close:
            if ")" in ln:
                skip_until_close = False
            continue
        if ln.startswith("from PySide6") or ln.startswith("import cv2"):
            if "(" in ln and ")" not in ln:
                skip_until_close = True
            continue
        if ln.startswith("import numpy"):
            continue
        keep.append(ln)
    cleaned_src = "".join(keep)

    # Provide the imports the helpers actually use
    ns = {}
    import configparser
    import os
    from pathlib import Path as P
    ns["configparser"] = configparser
    ns["os"] = os
    ns["sys"] = sys
    ns["Path"] = P
    ns["Optional"] = None  # only used in type annotations
    exec(cleaned_src, ns)
    return ns


def main() -> int:
    helpers = _import_helpers()
    # Patch 122ab: `_project_path_from_config` was renamed to
    # `_project_paths_lite` and now returns the full
    # project_paths_from_config dict (or {} on failure) — not a
    # bare project_path string. The test cases below reflect the
    # new shape.
    _project_paths_lite = helpers["_project_paths_lite"]
    _list_project_videos = helpers["_list_project_videos"]
    _videos_with_rois = helpers["_videos_with_rois"]

    tmp = Path(tempfile.mkdtemp())
    try:
        # ------------------------------------------------------------ #
        # Case 1: paths dict for a legacy project_config.ini
        # ------------------------------------------------------------ #
        proj = tmp / "p1" / "project_folder"
        proj.mkdir(parents=True)
        cfg = proj / "project_config.ini"
        cfg.write_text(
            f"[General settings]\nproject_path = {proj}\n"
        )
        out = _project_paths_lite(str(cfg))
        assert out.get("project_root") == str(proj), (
            f"case 1: project_root mismatch — got {out!r}"
        )
        assert out.get("video_dir") == str(proj / "videos"), (
            f"case 1: video_dir mismatch — got {out!r}"
        )
        assert out.get("roi_definitions_path") == str(
            proj / "logs" / "measures" / "ROI_definitions.h5"
        ), f"case 1: roi_definitions_path mismatch — got {out!r}"

        # ------------------------------------------------------------ #
        # Case 2: malformed INI (missing project_path) returns {}
        # rather than crashing — see the docstring on
        # _project_paths_lite for the rationale (the dialog wants
        # a polite "no project" state, not an exception).
        # ------------------------------------------------------------ #
        cfg2 = tmp / "no_project_path.ini"
        cfg2.write_text("[General settings]\n")
        assert _project_paths_lite(str(cfg2)) == {}, (
            f"case 2: expected {{}} on missing project_path, got "
            f"{_project_paths_lite(str(cfg2))!r}"
        )

        # ------------------------------------------------------------ #
        # Case 3: video listing — finds .mp4 / .avi files, skips
        # dotfiles. Takes the videos directory directly now (rather
        # than the project root joined with literal 'videos/').
        # ------------------------------------------------------------ #
        vids = proj / "videos"
        vids.mkdir()
        (vids / "vid_a.mp4").write_bytes(b"")
        (vids / "vid_b.avi").write_bytes(b"")
        (vids / ".hidden.mp4").write_bytes(b"")
        (vids / "not_a_video.txt").write_bytes(b"")
        out = _list_project_videos(str(vids))
        names = sorted(Path(p).name for p in out)
        assert names == ["vid_a.mp4", "vid_b.avi"], f"case 3: {names}"

        # ------------------------------------------------------------ #
        # Case 4: video listing — empty videos dir returns []
        # ------------------------------------------------------------ #
        proj2 = tmp / "p2" / "project_folder"
        (proj2 / "videos").mkdir(parents=True)
        assert _list_project_videos(str(proj2 / "videos")) == []

        # ------------------------------------------------------------ #
        # Case 5: video listing — missing dir returns []
        # ------------------------------------------------------------ #
        proj3 = tmp / "p3" / "project_folder"
        proj3.mkdir(parents=True)
        assert _list_project_videos(str(proj3 / "videos")) == []

        # ------------------------------------------------------------ #
        # Case 6: _videos_with_rois with missing file returns empty set
        # ------------------------------------------------------------ #
        out = _videos_with_rois(str(tmp / "nonexistent.h5"))
        assert out == set(), f"case 6: {out}"

        # ------------------------------------------------------------ #
        # Case 7: _videos_with_rois with corrupt file returns empty set
        # (defensive; real H5 read failures get swallowed)
        # ------------------------------------------------------------ #
        bad_h5 = tmp / "bad.h5"
        bad_h5.write_bytes(b"not really an h5 file")
        out = _videos_with_rois(str(bad_h5))
        assert out == set(), f"case 7: {out}"

        print("smoke_roi_video_table_helpers: 7/7 cases passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
