"""
mufasa.ui_qt.forms.video_editing
================================

Inline forms for geometric / timing edits:

* :class:`CropVideosForm` — rect / circle / polygon shapes, single /
  directory scope. Replaces
  :class:`CropVideoPopUp`, :class:`CropVideoCirclesPopUp`,
  :class:`CropVideoPolygonsPopUp`, :class:`MultiCropPopUp` (4 popups).
* :class:`ResizeVideosForm` — down / up-sample, change FPS. Replaces
  :class:`DownsampleSingleVideoPopUp`, :class:`DownsampleMultipleVideosPopUp`,
  :class:`DownsampleVideoPopUp`, :class:`UpsampleVideosPopUp`,
  :class:`ChangeFpsSingleVideoPopUp`, :class:`ChangeFpsMultipleVideosPopUp`
  (6 popups).
* :class:`RotateFlipForm` — rotation + horizontal/vertical flip.
  Replaces :class:`RotateVideoSetDegreesPopUp`, :class:`VideoRotatorPopUp`,
  :class:`FlipVideosPopUp` (3 popups).

**Interactive cropping** (click-and-drag rectangle / circle / polygon on
the video frame) is still needed for crop, and the OpenCV-based
interactive selector the legacy popups used is already implemented in
``mufasa.video_processors.roi_selector``. We dispatch to it from the
form — the form collects settings, the backend opens the OpenCV window
just for the crop gesture, the form hands off. That's acceptable:
OpenCV's window is live-video-rendering, Qt would be over-engineering.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox,
                               QFormLayout, QLabel, QRadioButton, QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget,
                               QCheckBox, QHBoxLayout)

from mufasa.ui_qt.forms.video_processing import _ScopePicker
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# CropVideosForm — 4 popups → 1
# --------------------------------------------------------------------------- #
class CropVideosForm(OperationForm):
    """Crop video(s) to a rectangular, circular, or polygonal region.

    Combines the rect / circle / polygon / multi-crop popups. The shape
    drop-down drives which crop backend is called; scope (single file
    vs. directory) is separate.

    **Interactive selection**: each backend opens an OpenCV window to
    capture the crop gesture on the video's first frame. Qt is
    intentionally NOT used for that — OpenCV's native video surface is
    what's already wired to the crop backends, and reproducing it in
    Qt would be scope creep with no user-visible benefit.
    """

    title = "Crop video(s)"
    description = ("Crop one or many videos to a rectangular, circular, "
                   "or polygonal region. A draw-on-frame window opens to "
                   "capture the crop geometry when you click Run.")

    SHAPES = ["Rectangle", "Circle", "Polygon"]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

        self.shape_cb = QComboBox(self)
        self.shape_cb.addItems(self.SHAPES)
        form.addRow("Crop shape:", self.shape_cb)

        self.multicrop = QCheckBox(
            "Multi-crop (one video → multiple outputs, one per drawn region)", self,
        )
        form.addRow("", self.multicrop)

        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        return {
            "path":       self.scope.path,
            "is_dir":     self.scope.is_dir,
            "shape":      self.shape_cb.currentText().lower(),
            "multi":      bool(self.multicrop.isChecked()),
        }

    def target(self, *, path: str, is_dir: bool, shape: str,
               multi: bool) -> None:
        from pathlib import Path as _P
        from mufasa.video_processors import video_processing as _vp

        def _sibling_dir(in_dir: str, tag: str) -> str:
            """Output dir next to input: ``/foo/videos`` → ``/foo/videos_cropped_{tag}``."""
            p = _P(in_dir)
            out = p.parent / f"{p.name}_cropped_{tag}"
            out.mkdir(parents=True, exist_ok=True)
            return str(out)

        if shape == "rectangle":
            if multi and not is_dir:
                raise NotImplementedError(
                    "Multi-crop from a single video: backend wiring pending.")
            if is_dir:
                # Rectangle batch uses `output_path` (not `out_dir`).
                # Upstream naming inconsistency with the circle/polygon
                # variants below — documented not fixed.
                _vp.crop_multiple_videos(directory_path=path,
                                         output_path=_sibling_dir(path, "rect"))
            else:
                _vp.crop_single_video(file_path=path)
        elif shape == "circle":
            if is_dir:
                _vp.crop_multiple_videos_circles(in_dir=path,
                                                 out_dir=_sibling_dir(path, "circle"))
            else:
                _vp.crop_single_video_circle(file_path=path)
        elif shape == "polygon":
            if is_dir:
                _vp.crop_multiple_videos_polygons(in_dir=path,
                                                  out_dir=_sibling_dir(path, "polygon"))
            else:
                _vp.crop_single_video_polygon(file_path=path)


# --------------------------------------------------------------------------- #
# ResizeVideosForm — 6 popups → 1
# --------------------------------------------------------------------------- #
class ResizeVideosForm(OperationForm):
    """Resize video(s): change resolution (down/up) or frame rate.

    Replaces DownsampleSingleVideoPopUp, DownsampleMultipleVideosPopUp,
    DownsampleVideoPopUp, UpsampleVideosPopUp,
    ChangeFpsSingleVideoPopUp, ChangeFpsMultipleVideosPopUp.
    """

    title = "Resize / re-time video(s)"
    description = ("Change resolution or frame rate. Scope (single "
                   "file / directory) and unit (px / percentage / fps) "
                   "chosen inline.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

        # Operation selector
        self.op_cb = QComboBox(self)
        self.op_cb.addItems([
            "Downsample (reduce resolution)",
            "Upsample (increase resolution)",
            "Change frame rate",
        ])
        self.op_cb.currentIndexChanged.connect(self._on_op_changed)
        form.addRow("Operation:", self.op_cb)

        # Stacked parameter panels
        self.panels = QStackedWidget(self)

        # --- Downsample panel --- #
        ds_host = QWidget()
        ds_form = QFormLayout(ds_host); ds_form.setContentsMargins(0, 0, 0, 0)
        self.ds_mode = QComboBox(ds_host)
        self.ds_mode.addItems(["Target width (px)", "Target height (px)",
                               "Scale factor (%)"])
        ds_form.addRow("Mode:", self.ds_mode)
        self.ds_value = QSpinBox(ds_host)
        self.ds_value.setRange(1, 4096); self.ds_value.setValue(640)
        ds_form.addRow("Value:", self.ds_value)
        self.panels.addWidget(ds_host)

        # --- Upsample panel --- #
        us_host = QWidget()
        us_form = QFormLayout(us_host); us_form.setContentsMargins(0, 0, 0, 0)
        self.us_factor = QDoubleSpinBox(us_host)
        self.us_factor.setRange(1.0, 8.0); self.us_factor.setValue(2.0)
        self.us_factor.setSingleStep(0.5)
        us_form.addRow("Scale factor:", self.us_factor)
        us_warn = QLabel(
            "<i>Upsampling via interpolation creates no new information "
            "and can take a while.</i>", us_host,
        )
        us_warn.setStyleSheet("color: #a86400;")
        us_form.addRow("", us_warn)
        self.panels.addWidget(us_host)

        # --- Change-FPS panel --- #
        fps_host = QWidget()
        fps_form = QFormLayout(fps_host); fps_form.setContentsMargins(0, 0, 0, 0)
        self.fps_target = QSpinBox(fps_host)
        self.fps_target.setRange(1, 240); self.fps_target.setValue(30)
        fps_form.addRow("Target fps:", self.fps_target)
        self.panels.addWidget(fps_host)

        form.addRow("Parameters:", self.panels)
        self.body_layout.addLayout(form)

    def _on_op_changed(self, index: int) -> None:
        self.panels.setCurrentIndex(index)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        op_idx = self.op_cb.currentIndex()
        op = ["downsample", "upsample", "change_fps"][op_idx]
        kwargs: dict = {"path": self.scope.path, "is_dir": self.scope.is_dir, "op": op}
        if op == "downsample":
            kwargs["mode"] = ["width", "height", "scale"][self.ds_mode.currentIndex()]
            kwargs["value"] = int(self.ds_value.value())
        elif op == "upsample":
            kwargs["factor"] = float(self.us_factor.value())
        elif op == "change_fps":
            kwargs["fps"] = int(self.fps_target.value())
        return kwargs

    def target(self, *, path: str, is_dir: bool, op: str, **params) -> None:
        from mufasa.video_processors import video_processing as _vp
        if op == "downsample":
            mode, value = params["mode"], params["value"]
            # The legacy single-video and multi-video downsample fns
            # accept the same mode args; choose by scope.
            if mode == "width":
                if is_dir:
                    _vp.resize_videos_by_width(video_paths=[path], width=value)
                else:
                    _vp.downsample_video(file_path=path, video_width=value)
            elif mode == "height":
                if is_dir:
                    _vp.resize_videos_by_height(video_paths=[path], height=value)
                else:
                    _vp.downsample_video(file_path=path, video_height=value)
            else:  # scale
                _vp.downsample_video(file_path=path, scale_factor=value/100.0)
        elif op == "upsample":
            _vp.upsample_fps(video_path=path, fps=int(30 * params["factor"]))
        elif op == "change_fps":
            _vp.change_fps_of_multiple_videos(path=path, fps=params["fps"])


# --------------------------------------------------------------------------- #
# RotateFlipForm — 3 popups → 1
# --------------------------------------------------------------------------- #
class RotateFlipForm(OperationForm):
    """Rotate and/or flip video(s). Replaces
    :class:`RotateVideoSetDegreesPopUp`, :class:`VideoRotatorPopUp`,
    :class:`FlipVideosPopUp`.
    """

    title = "Rotate / flip video(s)"
    description = ("Apply rotation and/or flip to videos. "
                   "0° + no flip is a no-op.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.scope = _ScopePicker(self)
        form.addRow("Source:", self.scope)

        # Rotation mode
        self.mode_cb = QComboBox(self)
        self.mode_cb.addItems([
            "Preset angle (90 / 180 / 270)",
            "Custom angle (degrees)",
            "Interactive (drag the frame)",
        ])
        self.mode_cb.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Rotation mode:", self.mode_cb)

        self.preset_cb = QComboBox(self)
        self.preset_cb.addItems(["0", "90", "180", "270"])
        form.addRow("Preset:", self.preset_cb)
        self.preset_row_index = form.rowCount() - 1

        self.custom_angle = QDoubleSpinBox(self)
        self.custom_angle.setRange(-360.0, 360.0); self.custom_angle.setValue(0.0)
        self.custom_angle.setSingleStep(1.0)
        form.addRow("Angle (°):", self.custom_angle)
        self.custom_row_index = form.rowCount() - 1

        # Flip toggles
        flip_row = QHBoxLayout()
        self.flip_h = QCheckBox("Flip horizontal", self)
        self.flip_v = QCheckBox("Flip vertical", self)
        flip_row.addWidget(self.flip_h)
        flip_row.addWidget(self.flip_v)
        flip_row.addStretch()
        flip_host = QWidget(); flip_host.setLayout(flip_row)
        form.addRow("Flip:", flip_host)

        self.body_layout.addLayout(form)
        self._on_mode_changed(0)

    def _on_mode_changed(self, index: int) -> None:
        # Show the right field, hide the other
        self.preset_cb.setVisible(index == 0)
        self.custom_angle.setVisible(index == 1)

    def collect_args(self) -> dict:
        if not self.scope.path:
            raise ValueError("No source selected.")
        mode_idx = self.mode_cb.currentIndex()
        if mode_idx == 0:
            angle = float(self.preset_cb.currentText())
            interactive = False
        elif mode_idx == 1:
            angle = float(self.custom_angle.value())
            interactive = False
        else:
            angle = 0.0
            interactive = True
        return {
            "path":        self.scope.path,
            "is_dir":      self.scope.is_dir,
            "angle":       angle,
            "interactive": interactive,
            "flip_h":      bool(self.flip_h.isChecked()),
            "flip_v":      bool(self.flip_v.isChecked()),
        }

    def target(self, *, path: str, is_dir: bool, angle: float,
               interactive: bool, flip_h: bool, flip_v: bool) -> None:
        from mufasa.video_processors import video_processing as _vp
        if angle != 0.0 or interactive:
            _vp.rotate_video(video_path=path, degrees=angle,
                             interactive=interactive)
        if flip_h or flip_v:
            # `flip_videos` takes a composite axis flag; translate.
            # 0 = vertical, 1 = horizontal, -1 = both.
            if flip_h and flip_v:
                flip_code = -1
            elif flip_h:
                flip_code = 1
            else:
                flip_code = 0
            _vp.flip_videos(video_path=path, flip_code=flip_code)


__all__ = ["CropVideosForm", "ResizeVideosForm", "RotateFlipForm"]
