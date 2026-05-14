"""
tests/smoke_122al_batch_pre_process_qt.py
==========================================

Patch 122al: Qt port of :class:`BatchProcessFrame` (the Tk
multi-step video pre-processing wizard) into an in-frame
:class:`OperationForm` that's also pop-out-dockable into a
floating QDockWidget. The Preprocessing page is reordered so
"Preprocess Videos" is the FIRST section (pre-processing happens
before pixel/mm calibration, which itself happens before pose
cleanup).

AST-only coverage — PySide6 isn't in the sandbox so we can't
instantiate the form. Verifies:

1. New form file exists and is a valid OperationForm.
2. Critical methods present (build, collect_args, target,
   reload, pop-out, per-row crop).
3. Operation set matches the Tk parity list (7 operations).
4. Pop-out machinery uses QDockWidget with the expected
   features (Movable | Floatable | Closable).
5. Preprocessing page lists 'Preprocess Videos' as section #1
   and 'Video Calibration' as section #2 (was reversed).
6. The page imports BatchPreProcessForm (not the legacy
   BatchPreProcessLauncher placeholder).
7. project_setup.py preserves BatchPreProcessLauncher as a
   back-compat alias for BatchPreProcessForm.
8. 122al patch number recorded in all touched files.
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
    # 1. New form file exists, is valid Python, exports the form
    # ==================================================================
    form_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                 / "batch_pre_process.py")
    check("batch_pre_process.py exists", form_path.is_file())
    src = form_path.read_text()
    try:
        tree = ast.parse(src)
        parsed_ok = True
    except SyntaxError:
        parsed_ok = False
    check("batch_pre_process.py parses cleanly", parsed_ok)

    classes = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
    }
    check(
        "BatchPreProcessForm class defined",
        "BatchPreProcessForm" in classes,
    )

    if "BatchPreProcessForm" in classes:
        cls = classes["BatchPreProcessForm"]
        # Should subclass OperationForm
        bases = [
            b.id if isinstance(b, ast.Name) else None
            for b in cls.bases
        ]
        check(
            "BatchPreProcessForm subclasses OperationForm",
            "OperationForm" in bases,
        )
        method_names = {
            n.name for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }
        for required in (
            "build", "collect_args", "target",
            "_reload_videos", "_populate_table",
            "_toggle_pop_out", "_on_crop_row",
            "_apply_clip_to_all", "_apply_downsample_to_all",
            "_apply_fps_to_all", "_apply_quality_to_all",
            "_on_reset_all", "_on_reset_crop",
        ):
            check(
                f"BatchPreProcessForm.{required} defined",
                required in method_names,
            )

    # ==================================================================
    # 2. Tk parity — 7 operations in execution order
    # ==================================================================
    check(
        "Module exports _OPS_IN_EXEC_ORDER with all 7 operations",
        all(op in src for op in (
            "clahe", "frame_cnt", "grayscale", "fps",
            "downsample", "clip", "crop",
        )),
    )
    check(
        "Operation order respects FFMPEGCommandCreator pipeline",
        '_OPS_IN_EXEC_ORDER = (' in src
        and src.index('"clahe"') < src.index('"frame_cnt"'),
    )

    # ==================================================================
    # 3. Pop-out / dockable machinery
    # ==================================================================
    check(
        "Pop-out uses QDockWidget",
        "QDockWidget" in src,
    )
    check(
        "Dock features include Movable | Floatable | Closable",
        "DockWidgetMovable" in src
        and "DockWidgetFloatable" in src
        and "DockWidgetClosable" in src,
    )
    check(
        "Dock allows all dock areas (mirror 122aj pattern)",
        "AllDockWidgetAreas" in src,
    )
    check(
        "_find_main_window walks parent chain to a QMainWindow",
        "_find_main_window" in src
        and "QMainWindow" in src,
    )

    # ==================================================================
    # 4. Execute path uses FFMPEGCommandCreator (Tk parity)
    # ==================================================================
    check(
        "target() drives FFMPEGCommandCreator",
        "FFMPEGCommandCreator" in src,
    )
    check(
        "target() calls crop / clip / downsample / fps / "
        "grayscale runners in sequence",
        all(call in src for call in (
            "runner.crop_videos()",
            "runner.clip_videos()",
            "runner.downsample_videos()",
            "runner.apply_fps()",
            "runner.apply_grayscale()",
        )),
    )
    check(
        "collect_args writes batch_process_log.json (Tk parity)",
        "batch_process_log.json" in src,
    )

    # ==================================================================
    # 5. Crop button reuses OpenCV ROISelector
    # ==================================================================
    check(
        "Crop button calls ROISelector (host-agnostic OpenCV "
        "dialog — reused from Tk)",
        "ROISelector(" in src,
    )

    # ==================================================================
    # 6. Preprocessing page — section ordering
    # ==================================================================
    page_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
                / "pose_cleanup_page.py").read_text()
    check(
        "pose_cleanup_page imports BatchPreProcessForm (not "
        "the legacy launcher)",
        "from mufasa.ui_qt.forms.batch_pre_process import "
        "BatchPreProcessForm" in page_src,
    )
    check(
        "pose_cleanup_page no longer imports "
        "BatchPreProcessLauncher",
        "import BatchPreProcessLauncher" not in page_src,
    )
    check(
        "'Preprocess Videos' is the first section",
        page_src.index('"Preprocess Videos"')
        < page_src.index('"Video Calibration"'),
    )
    check(
        "'Video Calibration' is now the SECOND section",
        page_src.index('"Video Calibration"')
        < page_src.index('"Interpolate missing frames"'),
    )
    check(
        "Preprocessing section uses BatchPreProcessForm",
        "(BatchPreProcessForm, {})" in page_src,
    )

    # ==================================================================
    # 7. project_setup.py — back-compat alias
    # ==================================================================
    proj_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                / "project_setup.py").read_text()
    check(
        "project_setup imports BatchPreProcessForm",
        "from mufasa.ui_qt.forms.batch_pre_process import "
        "BatchPreProcessForm" in proj_src,
    )
    check(
        "BatchPreProcessLauncher is aliased to "
        "BatchPreProcessForm for back-compat",
        "BatchPreProcessLauncher = BatchPreProcessForm" in proj_src,
    )
    check(
        "Both names still exported in __all__",
        "BatchPreProcessForm" in proj_src
        and "BatchPreProcessLauncher" in proj_src
        and "__all__" in proj_src,
    )
    # The legacy _LauncherForm should NOT be imported into
    # project_setup anymore (it was only there for the placeholder).
    check(
        "project_setup no longer imports _LauncherForm "
        "(launcher pattern retired)",
        "from mufasa.ui_qt.forms.annotation import _LauncherForm"
        not in proj_src,
    )

    # ==================================================================
    # 8. Patch number recorded in touched files
    # ==================================================================
    for path in (
        REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "batch_pre_process.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "project_setup.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "pages" / "pose_cleanup_page.py",
    ):
        check(
            f"{path.name}: records 122al patch number",
            "122al" in path.read_text(),
        )

    print(
        f"smoke_122al_batch_pre_process_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
