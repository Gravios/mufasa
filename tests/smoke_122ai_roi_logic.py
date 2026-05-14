"""
tests/smoke_122ai_roi_logic.py
==============================

Patch 122ai: fix for the user-reported bug

    Could not load Cacna26_ht_w_cort_14d_post.mp4:
    MissingSectionHeaderError: File contains no section
    headers. file: '/data/testing/mufasa/test-20260427/
    project.toml', line: 1
    'project_layout_version = 1\\n'

…which fires when a v1 project user opens the ROI Definitions
panel and clicks on a video. Root cause: mufasa/roi_tools/
roi_logic.py's ROILogic.__init__ read
``[General settings].project_path`` directly via configparser,
which silently parses zero sections out of a TOML file and
then crashes on cfg.get(). Same bug shape as 122ab
(frame_labeller), 122ag (clip_review / VideoInfoForm /
roi_define_panel), and 122ah (targeted_clips) — but roi_logic.py
lives outside mufasa/ui_qt/ (it's in mufasa/roi_tools/), which
was the scope of 122ah's audit guard, so this instance got
missed.

This patch:

* Routes roi_logic.py through project_paths_from_config +
  paths['roi_definitions_path'] + paths['video_info_path'] so
  v1 projects work end-to-end.
* Drops the configparser import (no longer needed).
* Extends 122ah's audit-guard test to cover ALL active backend
  directories under mufasa/ (not just ui_qt/), with a curated
  whitelist of intentionally-legacy locations. Catches the
  next instance of this bug at format-patch time.

Coverage:

1. roi_logic.py no longer imports configparser.
2. roi_logic.py uses project_paths_from_config.
3. roi_logic.py resolves self.roi_h5_path from
   paths['roi_definitions_path'] (the helper's key from 122ab).
4. roi_logic.py resolves video_info_csv from
   paths['video_info_path'].
5. roi_logic.py no longer reads '[General settings]' at code
   level.
6. roi_logic.py records the 122ai fix in code comments.
7. The extended audit guard in smoke_122ah_targeted_clips covers
   mufasa/roi_tools, mufasa/utils, mufasa/mixins,
   mufasa/feature_extractors, mufasa/cli.
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
    # roi_logic.py — the bug site
    # ==================================================================
    rl_path = (REPO_ROOT / "mufasa" / "roi_tools" / "roi_logic.py")
    rl_src = rl_path.read_text()
    rl_tree = ast.parse(rl_src)

    check(
        "roi_logic.py no longer imports configparser at module "
        "level",
        not any(
            isinstance(n, ast.Import)
            and any(a.name == "configparser" for a in n.names)
            for n in rl_tree.body
        ),
    )
    check(
        "roi_logic.py uses project_paths_from_config",
        "project_paths_from_config" in rl_src,
    )
    check(
        "roi_logic.py resolves roi_h5_path from helper key "
        "'roi_definitions_path'",
        'paths["roi_definitions_path"]' in rl_src
        or "paths['roi_definitions_path']" in rl_src,
    )
    check(
        "roi_logic.py resolves video_info_csv from helper key "
        "'video_info_path'",
        'paths["video_info_path"]' in rl_src
        or "paths['video_info_path']" in rl_src,
    )
    leaked_general_settings = any(
        ("'General settings'" in line
         or '"General settings"' in line)
        and not line.lstrip().startswith("#")
        for line in rl_src.splitlines()
    )
    check(
        "roi_logic.py has no code-level reads of "
        "'[General settings]' (comments OK)",
        not leaked_general_settings,
    )
    check(
        "roi_logic.py records the 122ai fix in code/comments",
        "122ai" in rl_src,
    )
    check(
        "roi_logic.py preserves defensive fallback (for malformed "
        "configs)",
        "except Exception" in rl_src
        and "ROI_definitions.h5" in rl_src,
    )

    # ==================================================================
    # Extended audit guard — 122ah test now covers more directories
    # ==================================================================
    ah_path = (REPO_ROOT / "tests"
               / "smoke_122ah_targeted_clips.py")
    if ah_path.is_file():
        ah_src = ah_path.read_text()
        # Verify it scans the wider scope
        for expected_dir in (
            '"roi_tools"',
            '"utils"',
            '"mixins"',
            '"feature_extractors"',
            '"cli"',
        ):
            check(
                f"smoke_122ah audit guard covers {expected_dir}",
                expected_dir in ah_src,
            )
        check(
            "smoke_122ah audit guard whitelists "
            "config_reader.py (the v1-aware adapter)",
            "config_reader.py" in ah_src,
        )
        check(
            "smoke_122ah audit guard whitelists "
            "project_reconfigure.py (known deferred item)",
            "project_reconfigure.py" in ah_src,
        )

    # ==================================================================
    # End-to-end: ROILogic constructor should at least IMPORT
    # cleanly. We can't fully exercise it (needs cv2 + real video)
    # but verify the configparser-on-TOML branch is gone by
    # importing the module.
    # ==================================================================
    try:
        # Import as a side-effect of the test — the module's
        # other dependencies (cv2, numpy) are real but the
        # sandbox doesn't have all of them. Tolerate ImportError
        # for those.
        import importlib
        try:
            importlib.import_module("mufasa.roi_tools.roi_logic")
            check("roi_logic module imports cleanly", True)
        except ImportError as exc:
            # Acceptable if it's a missing-dep. Sandbox doesn't
            # have cv2, h5py, numpy variants, trafaret, etc.
            # Not acceptable if it's a syntax error or a missing
            # mufasa-side symbol like a renamed helper.
            msg = str(exc).lower()
            env_deps = (
                "cv2", "numpy", "h5py", "trafaret", "numba",
                "pyside6", "pyqt", "ffmpeg", "tensorflow",
                "torch", "pyarrow",
            )
            check(
                "roi_logic import failure is an env-dep, not a "
                "mufasa-side breakage",
                any(d in msg for d in env_deps),
                detail=f"got ImportError: {exc!r}",
            )
    except Exception as exc:
        check(
            "roi_logic module imports without unexpected exception",
            False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    print(
        f"smoke_122ai_roi_logic: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
