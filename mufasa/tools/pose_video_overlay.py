"""
mufasa.tools.pose_video_overlay
================================

Qt-based video playback viewer that overlays mufasa markers
(raw + smoothed) on top of the source video. Sibling tool to
``mufasa.tools.pose_viewer``, which renders markers on a blank
canvas. This one renders them on the actual video frames.

Why this exists
---------------

After a smoother run, you want to *see* the smoothed
trajectory in the same coordinate frame as the original video.
The blank-canvas viewer (``pose_viewer``) is good for spotting
posture and motion artifacts in isolation; this one is good
for spotting alignment problems — does the smoothed nose
actually sit on the rat's nose? Does the head skeleton track
when the rat rears? Did a marker drift away from the video
geometry while still looking plausible on its own canvas?

This is especially relevant after patches that change the
observation model (119a wired fitted offsets in; 120b adds
per-marker drift state). Both can produce numerically reasonable
smoothed traces that still don't sit on the animal correctly,
and the only way to catch that is a video overlay.

Usage
-----

Smoothed only::

    python -m mufasa.tools.pose_video_overlay /path/to/video.mp4 \\
        --smoothed /path/to/smoothed.parquet

Smoothed + raw, side-by-side overlay::

    python -m mufasa.tools.pose_video_overlay /path/to/video.mp4 \\
        --smoothed /path/to/smoothed.parquet \\
        --raw /path/to/raw.csv

Options::

    --smoothed PATH        Smoothed pose file (parquet/csv)
    --raw PATH             Raw observation file to overlay
    --likelihood-threshold T  Hide raw markers below this p (default 0)
    --pose-offset N        Frame offset between video and pose (default 0).
                           Use this when pose data starts at a different
                           video frame than 0 (e.g., trimmed datasets).
    --no-skeleton          Disable skeleton edge overlay
    --no-ellipses          Disable variance ellipses on smoothed
    --raw-color R G B      Custom raw marker color
    --smoothed-color R G B Custom smoothed marker color
    --skeleton-color R G B Custom skeleton color
    --opacity F            Overlay opacity 0-1 (default 1.0)
    --start-frame N        Open at frame N (default 0)

Keyboard shortcuts
------------------

  Space          play / pause
  Left/Right     step one frame
  Shift+L/R      step 30 frames
  Home/End       jump to start/end
  S              toggle smoothed display
  R              toggle raw display
  K              toggle skeleton
  E              toggle variance ellipses
  +/=            playback faster (1x → 2x → 4x → 8x)
  -              playback slower
  0              reset playback speed to 1x
  Ctrl+= / Ctrl++  zoom in (centered on viewport)
  Ctrl+-         zoom out
  Ctrl+0         reset zoom and re-fit video to window

Mouse
-----

  Click + drag     pan the view
  Wheel scroll     zoom in / out, centered on cursor position

Speed control
-------------

The dropdown next to the Play button lets you pick a preset
multiplier (0.125× through 8×) or type a custom value (e.g.
``1.5``). Keyboard shortcuts ``+/-/0`` step through the same
range and stay in sync with the dropdown. Useful for slowing
playback to study fast articulation (running, head shakes)
or speeding it up to skim a long session.

Useful for inspecting marker drift on a specific body part:
hold the cursor over (e.g.) the head, scroll-zoom in until
ear markers fill the view, scrub through frames to watch
marker stability frame-by-frame at high magnification.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# Try imports lazily so import-time errors are clear and the
# module doesn't crash users who only want the helper functions.
try:
    import cv2  # type: ignore
except ImportError as e:
    cv2 = None
    _CV2_ERR = str(e)
else:
    _CV2_ERR = None

try:
    from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
    from PySide6.QtGui import (
        QBrush, QColor, QImage, QKeySequence, QPainter, QPen,
        QPixmap, QShortcut,
    )
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QGraphicsEllipseItem,
        QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem,
        QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
        QMainWindow, QPushButton, QSlider, QStatusBar,
        QVBoxLayout, QWidget,
    )
    _QT_ERR = None
except ImportError as e:
    _QT_ERR = str(e)


# Reuse the existing loader, marker detection, skeleton edges,
# and color palette from pose_viewer. Those are well-tested by
# the existing pose_viewer smoke tests; we don't want to drift
# them across two viewer tools.
from mufasa.tools.pose_viewer import (
    DEFAULT_SKELETON_EDGES,
    MARKER_PALETTE,
    PoseFrame,
    ZoomableGraphicsView,
    _load_pose_file,
)


# Playback speed presets (powers of 2 from 1/8 to 8×). Match the
# ratios used by the keyboard shortcuts so they land on preset
# values naturally. Exposing this as a module constant lets
# tests reference it without hard-coding.
PLAYBACK_SPEED_PRESETS: List[float] = [
    0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
]
DEFAULT_PLAYBACK_SPEED: float = 1.0
MIN_PLAYBACK_SPEED: float = 0.125
MAX_PLAYBACK_SPEED: float = 16.0


# ============================================================ #
# Video reader
# ============================================================ #


class VideoSource:
    """Random-access video reader over OpenCV.

    Reads on demand (no whole-video buffer) so memory use stays
    small even for long sessions. Seeks every frame, which is
    fine for typical 30 fps recordings on local disk; for very
    long sessions on slow disks, consider caching the
    last-N frames if scrubbing feels jerky.
    """

    def __init__(self, path):
        if cv2 is None:
            raise RuntimeError(
                f"OpenCV (cv2) is required for video display. "
                f"Install with `pip install opencv-python-headless`. "
                f"(Original import error: {_CV2_ERR})"
            )
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open the video at {path}. "
                f"Format may not be supported by your OpenCV "
                f"build, or the file may be corrupted."
            )
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Cache the most-recently-read frame so consecutive reads
        # at the same index don't re-decode.
        self._cache_idx: Optional[int] = None
        self._cache_frame: Optional[np.ndarray] = None

    def read(self, idx: int) -> Optional[np.ndarray]:
        """Return RGB frame at ``idx``, or None if past end."""
        if idx < 0 or idx >= self.n_frames:
            return None
        if idx == self._cache_idx and self._cache_frame is not None:
            return self._cache_frame
        # cv2 sequential read is much faster than seek+read, so
        # only seek when not already at the right position.
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._cache_idx = idx
        self._cache_frame = rgb
        return rgb

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ============================================================ #
# Overlay scene — video pixmap + marker overlays
# ============================================================ #


class OverlayScene(QGraphicsScene):
    """QGraphicsScene with a video pixmap as the background and
    marker / skeleton / ellipse items as overlays.

    Items are kept around between frames and updated in place
    (positions, visibility) rather than recreated. That avoids
    per-frame allocator churn during playback.
    """

    def __init__(
        self,
        video: VideoSource,
        smoothed: Optional[PoseFrame],
        raw: Optional[PoseFrame],
        marker_layout: Optional[List[str]] = None,
        likelihood_threshold: float = 0.0,
        skeleton_edges: List[Tuple[str, str]] = DEFAULT_SKELETON_EDGES,
        raw_color: Tuple[int, int, int] = (255, 80, 80),
        smoothed_color: Tuple[int, int, int] = (80, 255, 120),
        skeleton_color: Tuple[int, int, int] = (200, 220, 255),
        ellipse_color: Tuple[int, int, int] = (255, 255, 100),
        pose_offset: int = 0,
    ):
        super().__init__()
        self.video = video
        self.smoothed = smoothed
        self.raw = raw
        self.likelihood_threshold = likelihood_threshold
        self.skeleton_edges = skeleton_edges
        self.raw_color = raw_color
        self.smoothed_color = smoothed_color
        self.skeleton_color = skeleton_color
        self.ellipse_color = ellipse_color
        self.pose_offset = pose_offset

        # Decide a unified marker list. If both smoothed and raw
        # are loaded with different markers, take the union so
        # everything renders.
        markers: List[str] = []
        if marker_layout is not None:
            markers = list(marker_layout)
        else:
            seen = set()
            for src in (smoothed, raw):
                if src is None:
                    continue
                for m in src.markers:
                    if m not in seen:
                        seen.add(m)
                        markers.append(m)
        self.markers = markers

        # Background pixmap
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setZValue(0)
        self.addItem(self.pixmap_item)

        # Skeleton lines (created lazily once we know which
        # marker pairs are both present)
        self._skeleton_lines_smoothed: dict = {}
        self._skeleton_lines_raw: dict = {}
        for src_label, src in (("s", smoothed), ("r", raw)):
            if src is None:
                continue
            store = (
                self._skeleton_lines_smoothed if src_label == "s"
                else self._skeleton_lines_raw
            )
            color = (
                smoothed_color if src_label == "s" else raw_color
            )
            for a, b in self.skeleton_edges:
                if a in src.markers and b in src.markers:
                    line = QGraphicsLineItem()
                    pen = QPen(QColor(*color))
                    pen.setWidthF(1.5 if src_label == "s" else 1.0)
                    if src_label == "r":
                        # Raw skeleton dashed so it doesn't
                        # compete visually with smoothed
                        pen.setStyle(Qt.DashLine)
                    line.setPen(pen)
                    line.setZValue(1)
                    self.addItem(line)
                    store[(a, b)] = line

        # Marker dots (filled for smoothed, hollow for raw)
        self._smoothed_dots: dict = {}
        self._raw_dots: dict = {}
        radius_s = 3.5
        radius_r = 4.0
        if smoothed is not None:
            for m in smoothed.markers:
                color = self._marker_color(m, smoothed_color)
                dot = QGraphicsEllipseItem(
                    -radius_s, -radius_s, 2 * radius_s, 2 * radius_s,
                )
                dot.setBrush(QBrush(QColor(*color)))
                dot.setPen(QPen(QColor(0, 0, 0, 200), 0.5))
                dot.setZValue(3)
                self.addItem(dot)
                self._smoothed_dots[m] = dot
        if raw is not None:
            for m in raw.markers:
                color = self._marker_color(m, raw_color)
                dot = QGraphicsEllipseItem(
                    -radius_r, -radius_r, 2 * radius_r, 2 * radius_r,
                )
                # Hollow: brush transparent, pen the marker color
                dot.setBrush(QBrush(Qt.transparent))
                pen = QPen(QColor(*color))
                pen.setWidthF(1.5)
                dot.setPen(pen)
                dot.setZValue(2)
                self.addItem(dot)
                self._raw_dots[m] = dot

        # Variance ellipses (only for smoothed with variance data)
        self._variance_ellipses: dict = {}
        if smoothed is not None and smoothed.variances is not None:
            for m in smoothed.markers:
                ell = QGraphicsEllipseItem()
                pen = QPen(QColor(*ellipse_color))
                pen.setWidthF(0.8)
                pen.setStyle(Qt.DotLine)
                ell.setPen(pen)
                ell.setBrush(QBrush(Qt.transparent))
                ell.setZValue(2)
                self.addItem(ell)
                self._variance_ellipses[m] = ell

        # Visibility flags
        self.show_smoothed = smoothed is not None
        self.show_raw = raw is not None
        self.show_skeleton = True
        self.show_ellipses = smoothed is not None and smoothed.variances is not None

    def _marker_color(
        self, marker: str, fallback: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        """Per-marker color from the palette, deterministic by
        position in self.markers. Falls back to a uniform color
        when the marker isn't in the unified list (shouldn't
        happen in practice)."""
        try:
            idx = self.markers.index(marker)
        except ValueError:
            return fallback
        return MARKER_PALETTE[idx % len(MARKER_PALETTE)]

    def set_visibility(
        self,
        show_smoothed: Optional[bool] = None,
        show_raw: Optional[bool] = None,
        show_skeleton: Optional[bool] = None,
        show_ellipses: Optional[bool] = None,
    ):
        if show_smoothed is not None:
            self.show_smoothed = show_smoothed
        if show_raw is not None:
            self.show_raw = show_raw
        if show_skeleton is not None:
            self.show_skeleton = show_skeleton
        if show_ellipses is not None:
            self.show_ellipses = show_ellipses
        self._apply_visibility()

    def _apply_visibility(self):
        for d in self._smoothed_dots.values():
            d.setVisible(self.show_smoothed)
        for d in self._raw_dots.values():
            d.setVisible(self.show_raw)
        for line in self._skeleton_lines_smoothed.values():
            line.setVisible(self.show_skeleton and self.show_smoothed)
        for line in self._skeleton_lines_raw.values():
            line.setVisible(self.show_skeleton and self.show_raw)
        for e in self._variance_ellipses.values():
            e.setVisible(self.show_ellipses and self.show_smoothed)

    def update_frame(self, video_idx: int):
        """Push a new video frame + refresh marker positions for
        the corresponding pose row."""
        # 1. Video pixmap
        rgb = self.video.read(video_idx)
        if rgb is None:
            # Past end / unreadable: blank frame
            self.pixmap_item.setPixmap(QPixmap())
        else:
            h, w = rgb.shape[:2]
            qimg = QImage(
                rgb.data, w, h, 3 * w, QImage.Format_RGB888,
            ).copy()
            self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))

        # 2. Markers — pose row index = video_idx - pose_offset
        pose_idx = video_idx - self.pose_offset
        if pose_idx < 0:
            self._hide_all_markers()
            return

        if self.smoothed is not None:
            self._update_pose_layer(
                pose_idx, self.smoothed,
                self._smoothed_dots,
                self._skeleton_lines_smoothed,
                ellipses=self._variance_ellipses,
                threshold=0.0,  # smoothed always trusted
            )
        if self.raw is not None:
            self._update_pose_layer(
                pose_idx, self.raw,
                self._raw_dots,
                self._skeleton_lines_raw,
                ellipses=None,
                threshold=self.likelihood_threshold,
            )

    def _hide_all_markers(self):
        for d in self._smoothed_dots.values():
            d.setVisible(False)
        for d in self._raw_dots.values():
            d.setVisible(False)
        for line in self._skeleton_lines_smoothed.values():
            line.setVisible(False)
        for line in self._skeleton_lines_raw.values():
            line.setVisible(False)
        for e in self._variance_ellipses.values():
            e.setVisible(False)

    def _update_pose_layer(
        self, pose_idx, src, dots, skeleton_lines, ellipses,
        threshold,
    ):
        """Update one pose source's markers + skeleton + ellipses
        from row ``pose_idx`` of ``src.positions``."""
        if pose_idx >= src.n_frames:
            for d in dots.values():
                d.setVisible(False)
            for line in skeleton_lines.values():
                line.setVisible(False)
            if ellipses is not None:
                for e in ellipses.values():
                    e.setVisible(False)
            return

        positions = src.positions[pose_idx]  # (n_markers, 2)
        likelihoods = src.likelihoods[pose_idx]  # (n_markers,)
        variances = (
            src.variances[pose_idx]
            if src.variances is not None else None
        )
        is_smoothed = ellipses is not None  # convention

        marker_to_idx = {m: i for i, m in enumerate(src.markers)}

        for m, dot in dots.items():
            i = marker_to_idx[m]
            x, y = positions[i, 0], positions[i, 1]
            p = likelihoods[i]
            valid = (
                np.isfinite(x) and np.isfinite(y) and p >= threshold
            )
            if not valid:
                dot.setVisible(False)
                continue
            dot.setPos(QPointF(float(x), float(y)))
            dot.setVisible(
                self.show_smoothed if is_smoothed
                else self.show_raw
            )

        # Skeleton
        skeleton_visible = self.show_skeleton and (
            self.show_smoothed if is_smoothed else self.show_raw
        )
        for (a, b), line in skeleton_lines.items():
            ia = marker_to_idx[a]
            ib = marker_to_idx[b]
            xa, ya = positions[ia]
            xb, yb = positions[ib]
            valid = (
                np.isfinite(xa) and np.isfinite(ya)
                and np.isfinite(xb) and np.isfinite(yb)
                and likelihoods[ia] >= threshold
                and likelihoods[ib] >= threshold
            )
            if not valid:
                line.setVisible(False)
                continue
            line.setLine(
                float(xa), float(ya), float(xb), float(yb),
            )
            line.setVisible(skeleton_visible)

        # Variance ellipses
        if ellipses is not None and variances is not None:
            for m, ell in ellipses.items():
                i = marker_to_idx[m]
                x, y = positions[i, 0], positions[i, 1]
                vx, vy = variances[i, 0], variances[i, 1]
                valid = (
                    np.isfinite(x) and np.isfinite(y)
                    and np.isfinite(vx) and np.isfinite(vy)
                    and vx > 0 and vy > 0
                    and likelihoods[i] >= threshold
                )
                if not valid:
                    ell.setVisible(False)
                    continue
                # Draw 2-sigma ellipse (95% confidence approx).
                # We don't have off-diagonal cov in the saved
                # output, so axes are coordinate-aligned.
                sx = 2.0 * float(np.sqrt(vx))
                sy = 2.0 * float(np.sqrt(vy))
                ell.setRect(
                    float(x) - sx, float(y) - sy,
                    2 * sx, 2 * sy,
                )
                ell.setVisible(self.show_ellipses and self.show_smoothed)


# ============================================================ #
# Main window
# ============================================================ #


class OverlayViewer(QMainWindow):
    """Qt main window with the OverlayScene, scrubber, and
    play / pause / step controls.
    """

    def __init__(
        self,
        video: VideoSource,
        smoothed: Optional[PoseFrame],
        raw: Optional[PoseFrame],
        likelihood_threshold: float = 0.0,
        pose_offset: int = 0,
        start_frame: int = 0,
    ):
        super().__init__()
        self.setWindowTitle(
            f"mufasa pose-on-video overlay — {video.path.name}"
        )
        self.video = video
        self.scene_obj = OverlayScene(
            video=video,
            smoothed=smoothed,
            raw=raw,
            likelihood_threshold=likelihood_threshold,
            pose_offset=pose_offset,
        )
        self.scene_obj.setSceneRect(
            0, 0, video.width, video.height,
        )

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)

        # Graphics view — ZoomableGraphicsView (shared with
        # pose_viewer) gives us wheel-zoom centered on cursor
        # and reset-to-fit, on top of the click-and-drag pan
        # already provided by ScrollHandDrag.
        self.view = ZoomableGraphicsView(self.scene_obj)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        outer.addWidget(self.view, stretch=1)

        # Controls row
        controls = QHBoxLayout()
        outer.addLayout(controls)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_btn)

        # Playback speed dropdown. Editable so users can type a
        # custom multiplier (e.g. "1.5×") in addition to picking
        # from presets. Sits between Play and the scrubber so
        # it's visually grouped with the other transport
        # controls. Width is set so the longest preset ("0.125×")
        # fits without text being clipped.
        self.speed_combo = QComboBox()
        self.speed_combo.setEditable(True)
        self.speed_combo.setInsertPolicy(QComboBox.NoInsert)
        for s in PLAYBACK_SPEED_PRESETS:
            self.speed_combo.addItem(self._speed_label(s), s)
        # Initial selection at 1.0×
        idx_default = PLAYBACK_SPEED_PRESETS.index(DEFAULT_PLAYBACK_SPEED)
        self.speed_combo.setCurrentIndex(idx_default)
        self.speed_combo.setMinimumWidth(80)
        self.speed_combo.setToolTip(
            "Playback speed. Pick a preset or type a custom "
            "multiplier (e.g. 1.5). Keyboard: +/= faster, "
            "- slower, 0 reset."
        )
        # User picked a preset (activated fires only on user
        # action, not on programmatic setCurrentIndex)
        self.speed_combo.activated.connect(
            self._on_speed_combo_picked,
        )
        # User typed a custom value and pressed Enter / blurred
        self.speed_combo.lineEdit().editingFinished.connect(
            self._on_speed_combo_edited,
        )
        controls.addWidget(self.speed_combo)

        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, video.n_frames - 1)
        self.scrubber.setValue(start_frame)
        self.scrubber.valueChanged.connect(self._on_scrubber_changed)
        controls.addWidget(self.scrubber, stretch=1)

        self.frame_label = QLabel()
        self.frame_label.setMinimumWidth(140)
        controls.addWidget(self.frame_label)

        # Layer toggles
        toggles = QHBoxLayout()
        outer.addLayout(toggles)
        self.cb_smoothed = self._make_toggle(
            "Smoothed", smoothed is not None,
            lambda v: self.scene_obj.set_visibility(show_smoothed=v),
        )
        toggles.addWidget(self.cb_smoothed)
        self.cb_raw = self._make_toggle(
            "Raw", raw is not None,
            lambda v: self.scene_obj.set_visibility(show_raw=v),
        )
        toggles.addWidget(self.cb_raw)
        self.cb_skeleton = self._make_toggle(
            "Skeleton", True,
            lambda v: self.scene_obj.set_visibility(show_skeleton=v),
        )
        toggles.addWidget(self.cb_skeleton)
        self.cb_ellipses = self._make_toggle(
            "Variance",
            (smoothed is not None and smoothed.variances is not None),
            lambda v: self.scene_obj.set_visibility(show_ellipses=v),
        )
        toggles.addWidget(self.cb_ellipses)
        toggles.addStretch()

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            f"Video: {video.n_frames} frames, "
            f"{video.fps:.1f} fps, {video.width}×{video.height}"
        )

        # Playback state
        self.is_playing = False
        self.play_speed = 1.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._update_timer_interval()

        # Keyboard shortcuts
        self._wire_shortcuts()

        # Initial frame
        self._set_frame(start_frame)
        self.view.fitInView(
            self.scene_obj.sceneRect(), Qt.KeepAspectRatio,
        )

    def _make_toggle(self, label, initial, on_change):
        cb = QCheckBox(label)
        cb.setChecked(initial)
        cb.toggled.connect(on_change)
        return cb

    def _wire_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.step(+1))
        QShortcut(QKeySequence(Qt.Key_Left),  self, lambda: self.step(-1))
        QShortcut(
            QKeySequence(Qt.SHIFT | Qt.Key_Right), self,
            lambda: self.step(+30),
        )
        QShortcut(
            QKeySequence(Qt.SHIFT | Qt.Key_Left), self,
            lambda: self.step(-30),
        )
        QShortcut(QKeySequence(Qt.Key_Home), self, lambda: self._set_frame(0))
        QShortcut(
            QKeySequence(Qt.Key_End), self,
            lambda: self._set_frame(self.video.n_frames - 1),
        )
        QShortcut(QKeySequence(Qt.Key_S), self, self.cb_smoothed.toggle)
        QShortcut(QKeySequence(Qt.Key_R), self, self.cb_raw.toggle)
        QShortcut(QKeySequence(Qt.Key_K), self, self.cb_skeleton.toggle)
        QShortcut(QKeySequence(Qt.Key_E), self, self.cb_ellipses.toggle)
        QShortcut(QKeySequence(Qt.Key_Plus), self, lambda: self.set_speed(self.play_speed * 2))
        QShortcut(QKeySequence(Qt.Key_Equal), self, lambda: self.set_speed(self.play_speed * 2))
        QShortcut(QKeySequence(Qt.Key_Minus), self, lambda: self.set_speed(self.play_speed / 2))
        QShortcut(QKeySequence(Qt.Key_0), self, lambda: self.set_speed(1.0))

        # View zoom (Ctrl+modifier so as not to clash with the
        # +/-/0 playback-speed shortcuts above). Wheel zoom is
        # provided automatically by ZoomableGraphicsView.
        QShortcut(
            QKeySequence(Qt.CTRL | Qt.Key_Plus), self,
            lambda: self._keyboard_zoom(1.25),
        )
        QShortcut(
            QKeySequence(Qt.CTRL | Qt.Key_Equal), self,
            lambda: self._keyboard_zoom(1.25),
        )
        QShortcut(
            QKeySequence(Qt.CTRL | Qt.Key_Minus), self,
            lambda: self._keyboard_zoom(1 / 1.25),
        )
        QShortcut(
            QKeySequence(Qt.CTRL | Qt.Key_0), self,
            self._reset_view,
        )

    def _keyboard_zoom(self, factor: float):
        """Apply a zoom factor centered on the viewport center.

        Wheel zoom is centered on the cursor (handled by
        ZoomableGraphicsView); keyboard zoom uses the viewport
        center because there's no cursor position to anchor to.
        """
        view = self.view
        cur_scale = view.transform().m11()
        new_scale = cur_scale * factor
        if (
            new_scale < view.MIN_SCALE
            or new_scale > view.MAX_SCALE
        ):
            return
        # Temporarily anchor at viewport center for this zoom
        prev_anchor = view.transformationAnchor()
        view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        view.scale(factor, factor)
        view.setTransformationAnchor(prev_anchor)
        view._user_zoomed = True

    def _reset_view(self):
        """Reset zoom/pan to fit the video frame in the viewport."""
        self.view.reset_view()

    # ------- Playback -------
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("Pause")
            self.timer.start()
        else:
            self.play_btn.setText("Play")
            self.timer.stop()

    def step(self, delta: int):
        new_idx = max(
            0, min(self.video.n_frames - 1,
                   self.scrubber.value() + delta),
        )
        self._set_frame(new_idx)

    def set_speed(self, speed: float):
        self.play_speed = max(
            MIN_PLAYBACK_SPEED, min(MAX_PLAYBACK_SPEED, speed),
        )
        self._update_timer_interval()
        self._sync_speed_combo()
        self.statusBar().showMessage(
            f"Playback: {self.play_speed:g}× ({self.video.fps:.1f} fps source)"
        )

    @staticmethod
    def _speed_label(speed: float) -> str:
        """Formatted label for the speed combo. Avoids trailing
        zeros (e.g. '0.5×' not '0.500×')."""
        return f"{speed:g}×"

    def _sync_speed_combo(self):
        """Update the combo's display to match self.play_speed
        without re-firing the activated/editingFinished signals.
        Picks an exact preset if available, otherwise sets the
        edit text to the formatted label."""
        # Block signals throughout to avoid feedback into
        # set_speed (which calls this).
        self.speed_combo.blockSignals(True)
        try:
            idx = self.speed_combo.findData(self.play_speed)
            if idx >= 0:
                self.speed_combo.setCurrentIndex(idx)
            else:
                self.speed_combo.setEditText(
                    self._speed_label(self.play_speed)
                )
        finally:
            self.speed_combo.blockSignals(False)

    def _on_speed_combo_picked(self, idx: int):
        """User selected a preset from the dropdown."""
        speed = self.speed_combo.itemData(idx)
        if speed is not None:
            self.set_speed(float(speed))

    def _on_speed_combo_edited(self):
        """User typed a custom speed and pressed Enter / blurred.
        Tolerates trailing 'x' / '×' / whitespace; reverts the
        display on parse failure rather than mutating state."""
        text = self.speed_combo.currentText().strip()
        # Strip trailing multiplier glyphs
        for suffix in ("×", "x", "X"):
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()
                break
        try:
            speed = float(text)
        except ValueError:
            # Bad input — revert to current speed
            self._sync_speed_combo()
            return
        if speed <= 0:
            self._sync_speed_combo()
            return
        self.set_speed(speed)

    def _update_timer_interval(self):
        ms = int(round(1000.0 / (self.video.fps * self.play_speed)))
        ms = max(8, ms)  # cap at ~125 fps display rate
        self.timer.setInterval(ms)

    def _tick(self):
        next_idx = self.scrubber.value() + 1
        if next_idx >= self.video.n_frames:
            self.toggle_play()
            return
        self._set_frame(next_idx)

    def _on_scrubber_changed(self, value: int):
        self._set_frame(value, from_scrubber=True)

    def _set_frame(self, idx: int, from_scrubber: bool = False):
        if not from_scrubber:
            self.scrubber.blockSignals(True)
            self.scrubber.setValue(idx)
            self.scrubber.blockSignals(False)
        self.scene_obj.update_frame(idx)
        self._update_label(idx)

    def _update_label(self, idx: int):
        t = idx / max(1.0, self.video.fps)
        self.frame_label.setText(
            f"frame {idx}/{self.video.n_frames - 1}  "
            f"({t:6.2f}s)"
        )


# ============================================================ #
# CLI
# ============================================================ #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay mufasa pose markers on a video for visual "
            "inspection of smoothed-vs-raw alignment."
        ),
    )
    parser.add_argument(
        "video",
        help="Path to source video file (mp4, avi, etc.)",
    )
    parser.add_argument(
        "--smoothed",
        help="Smoothed pose file (parquet/csv)",
    )
    parser.add_argument(
        "--raw",
        help="Raw observation file to overlay (parquet/csv)",
    )
    parser.add_argument(
        "--likelihood-threshold", type=float, default=0.0,
        help="Hide raw markers below this likelihood (default 0)",
    )
    parser.add_argument(
        "--pose-offset", type=int, default=0,
        help=(
            "Frame offset between video and pose data (default "
            "0). Use this when pose data starts at a different "
            "video frame than 0."
        ),
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Open at frame N (default 0)",
    )
    args = parser.parse_args(argv)

    if _QT_ERR is not None:
        print(
            f"PySide6 is required for the overlay viewer. "
            f"Install with `pip install PySide6`. "
            f"(Original error: {_QT_ERR})",
            file=sys.stderr,
        )
        return 2
    if cv2 is None:
        print(
            f"OpenCV is required for video display. Install "
            f"with `pip install opencv-python-headless` (or "
            f"`opencv-python` for GUI builds). (Original "
            f"error: {_CV2_ERR})",
            file=sys.stderr,
        )
        return 2

    if not args.smoothed and not args.raw:
        print(
            "At least one of --smoothed or --raw is required.",
            file=sys.stderr,
        )
        return 2

    # Load pose sources
    smoothed_pose: Optional[PoseFrame] = None
    raw_pose: Optional[PoseFrame] = None
    if args.smoothed:
        smoothed_pose = _load_pose_file(args.smoothed)
    if args.raw:
        raw_pose = _load_pose_file(args.raw)

    # Open video
    video = VideoSource(args.video)

    # Sanity: warn if pose length doesn't match video length.
    # We don't error — sometimes you want to overlay a
    # truncated pose file or a shorter video clip.
    pose_len = max(
        smoothed_pose.n_frames if smoothed_pose else 0,
        raw_pose.n_frames if raw_pose else 0,
    )
    if pose_len and pose_len + args.pose_offset != video.n_frames:
        print(
            f"Note: pose has {pose_len} frames, video has "
            f"{video.n_frames}, offset is {args.pose_offset}. "
            f"Frames where pose data isn't available will show "
            f"the video without markers."
        )

    app = QApplication.instance() or QApplication(sys.argv)
    win = OverlayViewer(
        video=video,
        smoothed=smoothed_pose,
        raw=raw_pose,
        likelihood_threshold=args.likelihood_threshold,
        pose_offset=args.pose_offset,
        start_frame=max(0, min(video.n_frames - 1, args.start_frame)),
    )
    win.resize(1200, 800)
    win.show()
    rc = app.exec()
    video.close()
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
