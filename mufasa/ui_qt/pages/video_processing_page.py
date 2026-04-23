"""
mufasa.ui_qt.pages.video_processing_page
========================================

The Video Processing workbench page — **full consolidation**.

The legacy code had 59 popups in a single source file
(``video_processing_pop_up.py``). They're replaced here by:

* **11 consolidated forms** across 10 accordion sections (see tables below), and
* **5 Tools-menu actions** for zero-field / read-only operations.

Popup consolidation summary
---------------------------

.. list-table::
    :header-rows: 1

    * - Section
      - Form(s)
      - Legacy popups absorbed
    * - Format conversion
      - :class:`VideoFormatConverterForm`
      - 4
    * - Trim & split
      - :class:`ClipVideosForm`
      - 5
    * - Crop & mask
      - :class:`CropVideosForm`
      - 4
    * - Resize & rate
      - :class:`ResizeVideosForm`
      - 6
    * - Rotate & flip
      - :class:`RotateFlipForm`
      - 3
    * - Filters & enhancement
      - :class:`VideoFiltersForm`
      - 7
    * - Overlay / burn-in
      - :class:`VideoOverlayForm`
      - 7
    * - Frame extraction
      - :class:`ExtractFramesForm`
      - 5
    * - Join & transition
      - :class:`JoinVideosForm`
      - 4
    * - Image format conversion
      - :class:`ImageFormatConverterForm`
      - 6
    * - Metadata & audit
      - :class:`AverageFrameForm` + Tools-menu actions
      - 4

**Total: 11 forms + 5 menu actions = 16 UI surfaces replacing 55 popups.**

The remaining ≤4 popups not listed above are either:
(a) already covered by Tools-menu actions registered here (Reverse,
Crossfade, Print metadata, Check seekable), or
(b) diagnostic-only "Pose reset" / "Archive" which now live on other
workbench pages.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox

from mufasa.ui_qt.forms.video_processing import (ClipVideosForm,
                                                 VideoFormatConverterForm,
                                                 VideoOverlayForm)
from mufasa.ui_qt.forms.video_editing import (CropVideosForm, ResizeVideosForm,
                                              RotateFlipForm)
from mufasa.ui_qt.forms.video_filters import VideoFiltersForm
from mufasa.ui_qt.forms.video_frames import ExtractFramesForm
from mufasa.ui_qt.forms.video_join import JoinVideosForm
from mufasa.ui_qt.forms.image_conversion import (AverageFrameForm,
                                                 ImageFormatConverterForm)
from mufasa.ui_qt.workbench import WorkflowPage


def build_video_processing_page(workbench,
                                config_path: Optional[str] = None
                                ) -> WorkflowPage:
    """Populate and return the Video Processing workflow page."""
    page = workbench.add_page("Video Processing", icon_name="video")

    page.add_section("Format conversion",       [(VideoFormatConverterForm, {})])
    page.add_section("Trim & split",            [(ClipVideosForm, {})])
    page.add_section("Crop & mask",             [(CropVideosForm, {})])
    page.add_section("Resize & rate",           [(ResizeVideosForm, {})])
    page.add_section("Rotate & flip",           [(RotateFlipForm, {})])
    page.add_section("Filters & enhancement",   [(VideoFiltersForm, {})])
    page.add_section("Overlay / burn-in",       [(VideoOverlayForm, {})])
    page.add_section("Frame extraction",        [(ExtractFramesForm, {})])
    page.add_section("Join & transition",       [(JoinVideosForm, {})])
    page.add_section("Image format conversion", [(ImageFormatConverterForm, {})])
    page.add_section("Metadata & audit",        [(AverageFrameForm, {})])

    register_video_processing_menu_actions(workbench)
    return page


def register_video_processing_menu_actions(workbench) -> None:
    """Zero-field / read-only video operations as Tools-menu items."""

    def _pick_video(prompt: str = "Select a video") -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(
            workbench, prompt, "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm)",
        )
        return path or None

    # ---- Reverse video --------------------------------------------- #
    def _reverse_video() -> None:
        path = _pick_video("Reverse video — select a file")
        if not path:
            return
        try:
            from mufasa.video_processors import video_processing as _vp
            _vp.reverse_videos(path=path)
            QMessageBox.information(workbench, "Reverse video",
                                    f"Reversed video saved next to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(workbench, "Reverse failed", str(exc))

    # ---- Crossfade two videos -------------------------------------- #
    def _crossfade_two_videos() -> None:
        p1 = _pick_video("Crossfade — first video")
        if not p1: return
        p2 = _pick_video("Crossfade — second video")
        if not p2: return
        try:
            from mufasa.video_processors import video_processing as _vp
            _vp.crossfade_two_videos(video_path_1=p1, video_path_2=p2)
            QMessageBox.information(workbench, "Crossfade", "Crossfade complete.")
        except Exception as exc:
            QMessageBox.critical(workbench, "Crossfade failed", str(exc))

    # ---- Print video metadata (read-only inspection) --------------- #
    def _print_video_metadata() -> None:
        path = _pick_video("Print metadata — select a video")
        if not path: return
        try:
            from mufasa.utils.read_write import get_video_meta_data
            meta = get_video_meta_data(video_path=path)
        except Exception as exc:
            QMessageBox.critical(workbench, "Metadata failed", str(exc))
            return
        lines = [f"<b>{k}:</b> {v}" for k, v in meta.items()]
        QMessageBox.information(workbench, f"Metadata: {path}",
                                "<br>".join(lines))

    # ---- Check seekability (read-only inspection) ----------------- #
    def _check_seekable() -> None:
        path = _pick_video("Check seekable — select a video")
        if not path: return
        try:
            from mufasa.video_processors import video_processing as _vp
            # legacy fn name varies between forks; try two
            fn = (getattr(_vp, "check_video_seekability", None)
                  or getattr(_vp, "check_video_seekable", None))
            if fn is None:
                raise AttributeError("no check_video_seekab* in this fork")
            result = fn(video_path=path)
            ok = bool(result)
            QMessageBox.information(
                workbench, "Seekable check",
                f"{path}\n\nSeekable: {'✓ yes' if ok else '✗ no (broken keyframes)'}",
            )
        except Exception as exc:
            QMessageBox.critical(workbench, "Seekable check failed", str(exc))

    # ---- Calculate pixels-per-mm (interactive OpenCV) ------------- #
    def _pixels_per_mm() -> None:
        path = _pick_video("Pixels/mm — select a video")
        if not path: return
        try:
            from mufasa.video_processors import calculate_px_dist as _pxd
            _pxd.run(video_path=path)
        except (AttributeError, ImportError) as exc:
            QMessageBox.critical(
                workbench, "Pixels/mm",
                f"Calibration backend not available: {exc}",
            )

    workbench.add_tools_action("Reverse video…",          _reverse_video)
    workbench.add_tools_action("Crossfade two videos…",   _crossfade_two_videos)
    workbench.add_tools_action("Print video metadata…",   _print_video_metadata)
    workbench.add_tools_action("Check video seekable…",   _check_seekable)
    workbench.add_tools_action("Calibrate pixels/mm…",    _pixels_per_mm)


__all__ = [
    "build_video_processing_page",
    "register_video_processing_menu_actions",
]
