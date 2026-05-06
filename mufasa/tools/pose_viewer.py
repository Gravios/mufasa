"""
mufasa.tools.pose_viewer
========================

Qt-based playback viewer for mufasa pose data. Plays back
markers as a moving point cloud / skeleton over time, with
scrub controls, variance-ellipse rendering for smoothed data,
and side-by-side overlay of smoothed and raw sources.

Why this exists
---------------

After running ``kalman_pose_smoother``, you want to *see*
whether the smoothed output looks like a sensible animal
trajectory before trusting it for downstream analysis. Numeric
diagnostics (range ratios, mean residuals) catch many failure
modes but not all — a smoother that produces too-rigid output,
implausible posture, or visible artifacts is hardest to detect
from summary statistics alone.

This viewer lets you:

  - Watch the smoothed trajectory at its native frame rate
  - Scrub forward/backward through the session
  - Compare smoothed against raw observations on the same
    canvas (raw shown as small hollow dots, smoothed as filled
    dots with variance ellipses)
  - Toggle skeleton lines for posture inspection
  - Toggle a recent-frames trail for motion visualization

Usage
-----

Single file (auto-detected as smoothed or raw)::

    python -m mufasa.tools.pose_viewer /path/to/smoothed.parquet

Overlay smoothed + raw::

    python -m mufasa.tools.pose_viewer /path/to/smoothed.parquet \\
        --raw /path/to/raw.csv

Options::

    --fps N                Playback rate (default 30)
    --raw PATH             Raw observation file to overlay
    --likelihood-threshold T  Hide raw markers below this p (default 0)
    --no-skeleton          Disable default skeleton overlay
    --no-ellipses          Disable variance ellipses on smoothed
    --trail N              Length of fade-out trail (default 0 = off)

Keyboard shortcuts
------------------

  Space          play/pause
  Left/Right     step one frame
  Shift+L/R      step 30 frames
  Home/End       jump to start/end
  S              toggle smoothed display
  R              toggle raw display
  K              toggle skeleton
  E              toggle variance ellipses
  T              toggle trail
  + / =          zoom in
  -              zoom out
  0              reset view (fit to data)

Mouse:
  Wheel          zoom in/out, centered on cursor
  Click+drag     pan
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
    from PySide6.QtGui import (
        QBrush, QColor, QKeySequence, QPainter, QPen, QShortcut,
    )
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QGraphicsEllipseItem,
        QGraphicsLineItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
        QLabel, QMainWindow, QPushButton, QSlider, QStatusBar,
        QVBoxLayout, QWidget,
    )
except ImportError:
    print(
        "PySide6 is required. Install with: pip install PySide6",
        file=sys.stderr,
    )
    raise


# Default skeleton edges. Drawn only between markers that are
# both present in the loaded data.
DEFAULT_SKELETON_EDGES: List[Tuple[str, str]] = [
    # Spine
    ("back1", "back2"),
    ("back2", "back3"),
    ("back3", "back4"),
    ("back4", "tailbase"),
    ("tailbase", "tailmid"),
    ("tailmid", "tailend"),
    # Head
    ("nose", "headmid"),
    ("ear_left", "headmid"),
    ("ear_right", "headmid"),
    ("headmid", "neck"),
    ("neck", "back1"),
    # Body sides
    ("lateral_left", "back2"),
    ("lateral_right", "back2"),
    ("lateral_left", "back3"),
    ("lateral_right", "back3"),
    # Center reference
    ("center", "back2"),
    ("center", "back3"),
]


# Color palette for markers — distinct hues, decent contrast on
# dark background. 18 colors covers any plausible mufasa marker
# layout. Cycles if exceeded.
MARKER_PALETTE: List[Tuple[int, int, int]] = [
    (255, 100, 100),  # red
    (100, 200, 255),  # cyan
    (255, 200, 100),  # orange
    (180, 255, 100),  # lime
    (255, 100, 200),  # magenta
    (100, 255, 200),  # mint
    (200, 100, 255),  # purple
    (255, 255, 100),  # yellow
    (100, 100, 255),  # blue
    (255, 150, 50),   # amber
    (50, 255, 150),   # spring green
    (200, 200, 200),  # white-ish
    (255, 50, 100),   # rose
    (50, 200, 100),   # forest
    (150, 100, 255),  # violet
    (255, 200, 200),  # pink
    (200, 255, 200),  # pale green
    (200, 200, 255),  # pale blue
]


@dataclass
class PoseFrame:
    """Per-frame data extracted from a loaded DataFrame.

    Stores positions and metadata in dense numpy form for fast
    per-frame access during playback. NaN where unobserved or
    below threshold.
    """
    positions: np.ndarray   # (T, n_markers, 2)
    likelihoods: np.ndarray  # (T, n_markers)
    variances: Optional[np.ndarray]  # (T, n_markers, 2) or None
    markers: List[str]
    n_frames: int


def _detect_markers(df: pd.DataFrame) -> List[str]:
    """Detect marker names from column suffixes.

    Looks for ``<marker>_x`` columns that have a matching
    ``<marker>_y``. Markers ending in ``_var`` are excluded
    (those are variance columns from smoothed output, not
    independent markers).
    """
    markers = set()
    for col in df.columns:
        col = str(col)
        if col.endswith("_x"):
            base = col[:-2]
            if f"{base}_y" in df.columns:
                markers.add(base)
    # Strip variance columns
    markers = {m for m in markers if not m.endswith("_var")}
    return sorted(markers)


def _load_pose_file(path) -> PoseFrame:
    """Load a parquet or CSV pose file into a PoseFrame.

    Auto-detects format from extension. Detects whether the
    file has variance columns (smoothed output) or not (raw
    input). Accepts ``str`` or ``pathlib.Path``.

    Loading strategy:
      1. Try a direct pandas read (parquet or plain CSV).
         This works for smoothed output from the kalman_pose_
         smoother (always flat schema) and any pre-flattened
         CSV.
      2. If direct read yields no marker columns, fall back to
         the diagnostic module's ``load_pose_file`` which
         handles DLC-style multi-row headers (``scorer`` /
         ``bodyparts`` / ``coords``) and SimBA-style
         IMPORTED_POSE format.

    Direct-first matters for smoothed CSVs. The diagnostic
    loader uses ``index_col=0`` which is correct for raw DLC
    files (they have an explicit frame index in column 0) but
    wrong for smoothed-output CSVs (which use ``index=False``
    on write). Trying direct first gives smoothed CSVs the
    correct schema.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    df: Optional[pd.DataFrame] = None

    # Direct read attempt
    try:
        if suffix == ".parquet":
            try:
                df_direct = pd.read_parquet(path)
            except ImportError:
                raise RuntimeError(
                    f"Could not read {path} as parquet (pyarrow / "
                    f"fastparquet not installed). Install: "
                    f"pip install pyarrow"
                )
        elif suffix in ("", ".csv", ".tsv"):
            df_direct = pd.read_csv(path, low_memory=False)
        else:
            try:
                df_direct = pd.read_parquet(path)
            except Exception:
                df_direct = pd.read_csv(path, low_memory=False)
        df_direct.columns = [str(c).lower() for c in df_direct.columns]
        # Did this give us markers?
        if _detect_markers(df_direct):
            df = df_direct
    except RuntimeError:
        raise
    except Exception:
        pass

    # Fall back to diagnostic loader for DLC-style raw CSVs
    if df is None:
        try:
            from mufasa.data_processors.kalman_diagnostic import (
                load_pose_file as _diag_load_pose_file,
            )
            df, _ = _diag_load_pose_file(str(path))
        except (ImportError, ValueError):
            df = None

    if df is None:
        raise RuntimeError(
            f"Could not load {path}. Tried direct read (no marker "
            f"columns found) and DLC multi-row header parsing."
        )

    markers = _detect_markers(df)
    if not markers:
        raise RuntimeError(
            f"No marker columns detected in {path}. Expected "
            f"columns named '<marker>_x', '<marker>_y' (with "
            f"optional '<marker>_p' and '<marker>_var_x'/_var_y "
            f"for smoothed output). Got columns: "
            f"{list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
        )

    T = len(df)
    n = len(markers)
    positions = np.full((T, n, 2), np.nan)
    likelihoods = np.zeros((T, n))
    has_var = all(
        f"{m}_var_x" in df.columns and f"{m}_var_y" in df.columns
        for m in markers
    )
    variances = np.full((T, n, 2), np.nan) if has_var else None

    for i, m in enumerate(markers):
        # Coerce to numeric defensively — DLC CSVs sometimes
        # have stray header strings even after multi-row parsing.
        positions[:, i, 0] = pd.to_numeric(
            df[f"{m}_x"], errors="coerce",
        ).to_numpy()
        positions[:, i, 1] = pd.to_numeric(
            df[f"{m}_y"], errors="coerce",
        ).to_numpy()
        if f"{m}_p" in df.columns:
            likelihoods[:, i] = pd.to_numeric(
                df[f"{m}_p"], errors="coerce",
            ).fillna(0.0).to_numpy()
        else:
            likelihoods[:, i] = 1.0
        if has_var:
            variances[:, i, 0] = pd.to_numeric(
                df[f"{m}_var_x"], errors="coerce",
            ).to_numpy()
            variances[:, i, 1] = pd.to_numeric(
                df[f"{m}_var_y"], errors="coerce",
            ).to_numpy()

    return PoseFrame(
        positions=positions,
        likelihoods=likelihoods,
        variances=variances,
        markers=markers,
        n_frames=T,
    )


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView that supports mouse-wheel zoom (centered
    on cursor), reset-to-fit, and pan via click-and-drag.

    Pan is inherited from the parent's ``ScrollHandDrag`` mode.
    Zoom is implemented here because Qt has no built-in handler.

    The cursor-centered zoom is useful for inspecting specific
    regions of a pose trajectory (a particular marker, a
    specific frame's posture). Hold the cursor over the area
    of interest and scroll; the area stays under the cursor
    as it zooms.
    """

    # Zoom limits — keep the view sane.
    MIN_SCALE = 0.05
    MAX_SCALE = 100.0

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        # Anchor zoom to mouse cursor (the scroll-wheel pivot)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._user_zoomed = False  # Track explicit zoom interactions

    def wheelEvent(self, event):
        """Zoom in/out centered on mouse cursor."""
        # Wheel delta is in 1/8 degrees; one notch = 120 typically.
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1 / 1.25
        # Compute new scale relative to current
        cur_scale = self.transform().m11()  # Uniform x-scale
        new_scale = cur_scale * factor
        if new_scale < self.MIN_SCALE or new_scale > self.MAX_SCALE:
            return
        self.scale(factor, factor)
        self._user_zoomed = True

    def reset_view(self):
        """Reset zoom to fit the scene rect in the viewport."""
        self.resetTransform()
        self.fitInView(
            self.scene().sceneRect(), Qt.KeepAspectRatio,
        )
        self._user_zoomed = False

    @property
    def user_zoomed(self) -> bool:
        """True if the user has explicitly zoomed/panned. Used
        by the parent window to skip auto-fit on resize when
        the user has chosen a custom view.
        """
        return self._user_zoomed


class PoseScene(QGraphicsScene):
    """QGraphicsScene that renders one frame of pose data.

    Holds persistent scene items (dots, lines, ellipses) and
    updates their positions on each frame change rather than
    recreating them — this keeps frame-rate flat at high fps.
    """

    def __init__(
        self,
        smoothed: Optional[PoseFrame],
        raw: Optional[PoseFrame],
        skeleton_edges: List[Tuple[str, str]],
        likelihood_threshold: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(QColor(20, 20, 25)))

        self.smoothed = smoothed
        self.raw = raw
        self.likelihood_threshold = likelihood_threshold
        self.skeleton_edges = skeleton_edges

        # Display flags
        self.show_smoothed = smoothed is not None
        self.show_raw = raw is not None
        self.show_skeleton = True
        self.show_ellipses = smoothed is not None and (
            smoothed.variances is not None
        )
        self.trail_length = 0  # set via UI

        # Build per-marker color map. If both smoothed and raw
        # share marker names, use the same color in both — easy
        # visual matching of "where did this marker move from."
        all_markers: List[str] = []
        if smoothed is not None:
            all_markers.extend(smoothed.markers)
        if raw is not None:
            for m in raw.markers:
                if m not in all_markers:
                    all_markers.append(m)
        self.marker_colors: Dict[str, QColor] = {}
        for i, m in enumerate(all_markers):
            r, g, b = MARKER_PALETTE[i % len(MARKER_PALETTE)]
            self.marker_colors[m] = QColor(r, g, b)

        # Compute scene bounds from the data
        self._compute_bounds()

        # Pre-create scene items (one per visual element).
        # Updating their positions per frame is much faster than
        # adding/removing items.
        self._smoothed_dots: Dict[str, QGraphicsEllipseItem] = {}
        self._smoothed_ellipses: Dict[str, QGraphicsEllipseItem] = {}
        self._raw_dots: Dict[str, QGraphicsEllipseItem] = {}
        self._skeleton_lines: List[
            Tuple[str, str, QGraphicsLineItem]
        ] = []
        # Trail: list of (frame_offset, dict[marker -> dot]).
        # Items recreated each frame because trail length can change.
        self._trail_items: List[QGraphicsEllipseItem] = []

        self._build_items()

    def _compute_bounds(self) -> None:
        """Compute scene rect from data extent + padding."""
        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        for src in (self.smoothed, self.raw):
            if src is None:
                continue
            x = src.positions[..., 0].ravel()
            y = src.positions[..., 1].ravel()
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if len(x):
                all_x.append(x)
            if len(y):
                all_y.append(y)
        if not all_x:
            self.setSceneRect(0, 0, 100, 100)
            return
        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        x_min, x_max = float(x_all.min()), float(x_all.max())
        y_min, y_max = float(y_all.min()), float(y_all.max())
        # Symmetric padding
        pad_x = max(20.0, 0.05 * (x_max - x_min))
        pad_y = max(20.0, 0.05 * (y_max - y_min))
        self.setSceneRect(
            x_min - pad_x, y_min - pad_y,
            (x_max - x_min) + 2 * pad_x,
            (y_max - y_min) + 2 * pad_y,
        )

    def _build_items(self) -> None:
        """Pre-create all persistent scene items (dots, lines,
        ellipses). Initial positions don't matter; they're set
        in update_frame.
        """
        # Marker dots — smoothed (filled) and raw (hollow)
        if self.smoothed is not None:
            for m in self.smoothed.markers:
                color = self.marker_colors[m]
                # Variance ellipse (drawn behind dot)
                if self.smoothed.variances is not None:
                    ell = QGraphicsEllipseItem()
                    pen_color = QColor(color)
                    pen_color.setAlpha(80)
                    ell.setPen(QPen(pen_color, 0.5))
                    fill = QColor(color)
                    fill.setAlpha(30)
                    ell.setBrush(QBrush(fill))
                    ell.setZValue(1)
                    self.addItem(ell)
                    self._smoothed_ellipses[m] = ell
                # Dot
                dot = QGraphicsEllipseItem(-3, -3, 6, 6)
                dot.setPen(QPen(QColor(0, 0, 0, 200), 0.5))
                dot.setBrush(QBrush(color))
                dot.setZValue(3)
                self.addItem(dot)
                self._smoothed_dots[m] = dot

        if self.raw is not None:
            for m in self.raw.markers:
                color = self.marker_colors[m]
                dot = QGraphicsEllipseItem(-2.5, -2.5, 5, 5)
                pen = QPen(color, 1.0)
                dot.setPen(pen)
                dot.setBrush(QBrush(Qt.NoBrush))
                dot.setZValue(2)
                self.addItem(dot)
                self._raw_dots[m] = dot

        # Skeleton lines (only if smoothed has all required markers)
        if self.smoothed is not None:
            available = set(self.smoothed.markers)
            for a, b in self.skeleton_edges:
                if a in available and b in available:
                    line = QGraphicsLineItem()
                    pen = QPen(QColor(180, 180, 180, 160), 1.5)
                    line.setPen(pen)
                    line.setZValue(0)
                    self.addItem(line)
                    self._skeleton_lines.append((a, b, line))

    def update_frame(self, frame_idx: int, trail_offsets: List[int]) -> None:
        """Update scene items to display ``frame_idx``.

        ``trail_offsets`` is a list of negative offsets (e.g.,
        [-1, -2, -3, ...]) for trail rendering; each offset's
        positions are shown with reduced opacity behind the
        current frame.
        """
        # Smoothed dots + ellipses
        if self.show_smoothed and self.smoothed is not None:
            for i, m in enumerate(self.smoothed.markers):
                dot = self._smoothed_dots[m]
                x = self.smoothed.positions[frame_idx, i, 0]
                y = self.smoothed.positions[frame_idx, i, 1]
                if np.isfinite(x) and np.isfinite(y):
                    dot.setPos(x, y)
                    dot.setVisible(True)
                else:
                    dot.setVisible(False)

                if self.show_ellipses and m in self._smoothed_ellipses:
                    ell = self._smoothed_ellipses[m]
                    if (
                        self.smoothed.variances is not None
                        and np.isfinite(x) and np.isfinite(y)
                    ):
                        vx = self.smoothed.variances[frame_idx, i, 0]
                        vy = self.smoothed.variances[frame_idx, i, 1]
                        if np.isfinite(vx) and np.isfinite(vy):
                            sx = float(np.sqrt(max(vx, 0.0)))
                            sy = float(np.sqrt(max(vy, 0.0)))
                            # Cap ellipse size to keep view sane
                            sx = min(sx, 200.0)
                            sy = min(sy, 200.0)
                            ell.setRect(
                                x - sx, y - sy, 2 * sx, 2 * sy,
                            )
                            ell.setVisible(True)
                        else:
                            ell.setVisible(False)
                    else:
                        ell.setVisible(False)
                elif m in self._smoothed_ellipses:
                    self._smoothed_ellipses[m].setVisible(False)
        else:
            for dot in self._smoothed_dots.values():
                dot.setVisible(False)
            for ell in self._smoothed_ellipses.values():
                ell.setVisible(False)

        # Raw dots
        if self.show_raw and self.raw is not None:
            for i, m in enumerate(self.raw.markers):
                dot = self._raw_dots[m]
                x = self.raw.positions[frame_idx, i, 0]
                y = self.raw.positions[frame_idx, i, 1]
                p = self.raw.likelihoods[frame_idx, i]
                if (
                    np.isfinite(x) and np.isfinite(y)
                    and p >= self.likelihood_threshold
                ):
                    dot.setPos(x, y)
                    dot.setVisible(True)
                else:
                    dot.setVisible(False)
        else:
            for dot in self._raw_dots.values():
                dot.setVisible(False)

        # Skeleton lines
        if (
            self.show_skeleton
            and self.show_smoothed
            and self.smoothed is not None
        ):
            marker_idx = {
                m: i for i, m in enumerate(self.smoothed.markers)
            }
            for a, b, line in self._skeleton_lines:
                ia, ib = marker_idx[a], marker_idx[b]
                xa = self.smoothed.positions[frame_idx, ia, 0]
                ya = self.smoothed.positions[frame_idx, ia, 1]
                xb = self.smoothed.positions[frame_idx, ib, 0]
                yb = self.smoothed.positions[frame_idx, ib, 1]
                if (
                    np.isfinite(xa) and np.isfinite(ya)
                    and np.isfinite(xb) and np.isfinite(yb)
                ):
                    line.setLine(xa, ya, xb, yb)
                    line.setVisible(True)
                else:
                    line.setVisible(False)
        else:
            for _, _, line in self._skeleton_lines:
                line.setVisible(False)

        # Trail: clear and rebuild. The trail is small (typically
        # tens of items) and only present when enabled, so
        # rebuilding per frame is fine.
        for item in self._trail_items:
            self.removeItem(item)
        self._trail_items.clear()

        if (
            self.trail_length > 0
            and self.show_smoothed
            and self.smoothed is not None
        ):
            for offset in trail_offsets:
                tf = frame_idx + offset
                if tf < 0 or tf >= self.smoothed.n_frames:
                    continue
                # Linear fade
                alpha = int(255 * (1.0 - abs(offset) / self.trail_length))
                alpha = max(20, alpha)
                for i, m in enumerate(self.smoothed.markers):
                    x = self.smoothed.positions[tf, i, 0]
                    y = self.smoothed.positions[tf, i, 1]
                    if np.isfinite(x) and np.isfinite(y):
                        d = QGraphicsEllipseItem(-1.5, -1.5, 3, 3)
                        c = QColor(self.marker_colors[m])
                        c.setAlpha(alpha)
                        d.setBrush(QBrush(c))
                        d.setPen(QPen(Qt.NoPen))
                        d.setPos(x, y)
                        d.setZValue(1.5)
                        self.addItem(d)
                        self._trail_items.append(d)


class PoseViewer(QMainWindow):
    """Main window for the pose viewer.

    Layout:
      - Central QGraphicsView showing the pose scene
      - Bottom: scrub slider + frame label
      - Below that: control panel with playback buttons,
        speed selector, display toggles
      - Status bar with frame number, time, fps
    """

    def __init__(
        self,
        smoothed: Optional[PoseFrame],
        raw: Optional[PoseFrame],
        fps: float = 30.0,
        likelihood_threshold: float = 0.0,
        skeleton_edges: Optional[List[Tuple[str, str]]] = None,
        trail_length: int = 0,
        show_skeleton: bool = True,
        show_ellipses: bool = True,
    ):
        super().__init__()

        # Determine session length from whichever source is available.
        # If both are present, they should be the same length, but
        # fall back gracefully if not.
        n_frames = 0
        if smoothed is not None:
            n_frames = smoothed.n_frames
        if raw is not None:
            n_frames = max(n_frames, raw.n_frames)
        if n_frames == 0:
            raise RuntimeError("No pose data loaded — cannot start viewer.")

        if smoothed is not None and raw is not None:
            if smoothed.n_frames != raw.n_frames:
                print(
                    f"Warning: smoothed has {smoothed.n_frames} frames, "
                    f"raw has {raw.n_frames}. Using max for slider; "
                    f"missing frames will show as hidden markers.",
                    file=sys.stderr,
                )

        self.fps = fps
        self.dt_ms = max(1, int(round(1000.0 / fps)))
        self.n_frames = n_frames
        self.frame_idx = 0
        self.playing = False
        self.speed = 1.0

        self.setWindowTitle("mufasa pose viewer")
        self.resize(1100, 800)

        # Scene + view
        edges = skeleton_edges if skeleton_edges is not None else (
            DEFAULT_SKELETON_EDGES
        )
        self.scene = PoseScene(
            smoothed=smoothed, raw=raw,
            skeleton_edges=edges,
            likelihood_threshold=likelihood_threshold,
        )
        self.scene.show_skeleton = show_skeleton
        self.scene.show_ellipses = (
            show_ellipses
            and smoothed is not None
            and smoothed.variances is not None
        )
        self.scene.trail_length = trail_length

        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setBackgroundBrush(QBrush(QColor(20, 20, 25)))
        # Y axis: pose data uses image coords (Y down), which
        # matches Qt's default — no flip needed.
        self.view.fitInView(
            self.scene.sceneRect(), Qt.KeepAspectRatio,
        )

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, self.n_frames - 1))
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(120)

        # Playback buttons
        self.play_btn = QPushButton("▶  Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.step_back_btn = QPushButton("◀")
        self.step_back_btn.clicked.connect(lambda: self._step(-1))
        self.step_fwd_btn = QPushButton("▶")
        self.step_fwd_btn.clicked.connect(lambda: self._step(1))

        # Speed selector
        self.speed_combo = QComboBox()
        for s in ["0.25x", "0.5x", "1x", "2x", "4x", "8x"]:
            self.speed_combo.addItem(s)
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed)

        # View reset button
        self.reset_view_btn = QPushButton("Reset view")
        self.reset_view_btn.setToolTip(
            "Reset zoom and pan (shortcut: 0)"
        )
        self.reset_view_btn.clicked.connect(self.view.reset_view)

        # Display toggles
        self.smoothed_chk = QCheckBox("Smoothed")
        self.smoothed_chk.setChecked(self.scene.show_smoothed)
        self.smoothed_chk.setEnabled(smoothed is not None)
        self.smoothed_chk.toggled.connect(
            lambda v: self._set_display("smoothed", v)
        )

        self.raw_chk = QCheckBox("Raw")
        self.raw_chk.setChecked(self.scene.show_raw)
        self.raw_chk.setEnabled(raw is not None)
        self.raw_chk.toggled.connect(
            lambda v: self._set_display("raw", v)
        )

        self.skel_chk = QCheckBox("Skeleton")
        self.skel_chk.setChecked(self.scene.show_skeleton)
        self.skel_chk.toggled.connect(
            lambda v: self._set_display("skeleton", v)
        )

        self.ell_chk = QCheckBox("Variance ellipses")
        self.ell_chk.setChecked(self.scene.show_ellipses)
        self.ell_chk.setEnabled(
            smoothed is not None and smoothed.variances is not None
        )
        self.ell_chk.toggled.connect(
            lambda v: self._set_display("ellipses", v)
        )

        # Trail length combo
        self.trail_combo = QComboBox()
        self.trail_combo.addItems(["off", "5", "15", "30", "60", "120"])
        self.trail_combo.setCurrentText(
            str(trail_length) if trail_length > 0 else "off"
        )
        self.trail_combo.currentTextChanged.connect(self._on_trail)

        # Layout
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.slider, stretch=1)
        slider_row.addWidget(self.frame_label)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.step_back_btn)
        controls_row.addWidget(self.play_btn)
        controls_row.addWidget(self.step_fwd_btn)
        controls_row.addSpacing(16)
        controls_row.addWidget(QLabel("Speed:"))
        controls_row.addWidget(self.speed_combo)
        controls_row.addSpacing(16)
        controls_row.addWidget(self.smoothed_chk)
        controls_row.addWidget(self.raw_chk)
        controls_row.addWidget(self.skel_chk)
        controls_row.addWidget(self.ell_chk)
        controls_row.addSpacing(16)
        controls_row.addWidget(QLabel("Trail:"))
        controls_row.addWidget(self.trail_combo)
        controls_row.addSpacing(16)
        controls_row.addWidget(self.reset_view_btn)
        controls_row.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.view, stretch=1)
        layout.addLayout(slider_row)
        layout.addLayout(controls_row)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._update_status()

        # Playback timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Initial frame draw
        self.scene.update_frame(0, self._trail_offsets())

    def showEvent(self, event):
        """Re-fit the view to the scene when the window first
        appears. ``fitInView`` before show uses the default view
        size (which is small) and zooms wrong.
        """
        super().showEvent(event)
        if not self.view.user_zoomed:
            self.view.fitInView(
                self.scene.sceneRect(), Qt.KeepAspectRatio,
            )

    def resizeEvent(self, event):
        """Re-fit on resize too — keeps the data filling the
        window when the user resizes. Skipped if the user has
        explicitly zoomed/panned (don't fight their viewport
        choice).
        """
        super().resizeEvent(event)
        if not self.view.user_zoomed:
            self.view.fitInView(
                self.scene.sceneRect(), Qt.KeepAspectRatio,
            )

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
        QShortcut(
            QKeySequence(Qt.Key_Right), self, lambda: self._step(1),
        )
        QShortcut(
            QKeySequence(Qt.Key_Left), self, lambda: self._step(-1),
        )
        QShortcut(
            QKeySequence(Qt.SHIFT | Qt.Key_Right), self,
            lambda: self._step(30),
        )
        QShortcut(
            QKeySequence(Qt.SHIFT | Qt.Key_Left), self,
            lambda: self._step(-30),
        )
        QShortcut(
            QKeySequence(Qt.Key_Home), self,
            lambda: self._jump_to(0),
        )
        QShortcut(
            QKeySequence(Qt.Key_End), self,
            lambda: self._jump_to(self.n_frames - 1),
        )
        QShortcut(
            QKeySequence(Qt.Key_S), self,
            lambda: self.smoothed_chk.toggle(),
        )
        QShortcut(
            QKeySequence(Qt.Key_R), self,
            lambda: self.raw_chk.toggle(),
        )
        QShortcut(
            QKeySequence(Qt.Key_K), self,
            lambda: self.skel_chk.toggle(),
        )
        QShortcut(
            QKeySequence(Qt.Key_E), self,
            lambda: self.ell_chk.toggle(),
        )
        QShortcut(
            QKeySequence(Qt.Key_0), self,
            lambda: self.view.reset_view(),
        )
        # Keyboard zoom (+ / - / =), with the same factor as
        # the wheel for consistency.
        QShortcut(
            QKeySequence(Qt.Key_Plus), self,
            lambda: self._zoom_keyboard(1.25),
        )
        QShortcut(
            QKeySequence(Qt.Key_Equal), self,  # Same key as +
            lambda: self._zoom_keyboard(1.25),
        )
        QShortcut(
            QKeySequence(Qt.Key_Minus), self,
            lambda: self._zoom_keyboard(1 / 1.25),
        )

    def _zoom_keyboard(self, factor: float) -> None:
        """Keyboard zoom — pivots on view center rather than
        cursor (no cursor info available in shortcut handler).
        Respects the same scale limits as wheel zoom.
        """
        cur = self.view.transform().m11()
        new = cur * factor
        if new < self.view.MIN_SCALE or new > self.view.MAX_SCALE:
            return
        # Temporarily switch anchor to center for keyboard zoom
        self.view.setTransformationAnchor(
            QGraphicsView.AnchorViewCenter,
        )
        self.view.scale(factor, factor)
        self.view.setTransformationAnchor(
            QGraphicsView.AnchorUnderMouse,
        )
        self.view._user_zoomed = True

    # ----- Event handlers -----

    def _trail_offsets(self) -> List[int]:
        if self.scene.trail_length <= 0:
            return []
        return [-i for i in range(1, self.scene.trail_length + 1)]

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("⏸  Pause")
            interval = max(1, int(round(self.dt_ms / self.speed)))
            self.timer.start(interval)
        else:
            self.play_btn.setText("▶  Play")
            self.timer.stop()

    def _on_tick(self) -> None:
        if self.frame_idx < self.n_frames - 1:
            self._jump_to(self.frame_idx + 1, from_timer=True)
        else:
            # End of session — pause
            self.playing = False
            self.play_btn.setText("▶  Play")
            self.timer.stop()

    def _step(self, delta: int) -> None:
        new_idx = max(0, min(self.n_frames - 1, self.frame_idx + delta))
        self._jump_to(new_idx)

    def _jump_to(self, idx: int, from_timer: bool = False) -> None:
        idx = max(0, min(self.n_frames - 1, idx))
        self.frame_idx = idx
        # Block slider signal to avoid feedback loop
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.scene.update_frame(idx, self._trail_offsets())
        self._update_status()

    def _on_slider(self, value: int) -> None:
        self.frame_idx = value
        self.scene.update_frame(value, self._trail_offsets())
        self._update_status()

    def _on_speed(self, text: str) -> None:
        self.speed = float(text.rstrip("x"))
        if self.playing:
            interval = max(1, int(round(self.dt_ms / self.speed)))
            self.timer.start(interval)

    def _on_trail(self, text: str) -> None:
        if text == "off":
            self.scene.trail_length = 0
        else:
            self.scene.trail_length = int(text)
        self.scene.update_frame(self.frame_idx, self._trail_offsets())

    def _set_display(self, key: str, value: bool) -> None:
        if key == "smoothed":
            self.scene.show_smoothed = value
        elif key == "raw":
            self.scene.show_raw = value
        elif key == "skeleton":
            self.scene.show_skeleton = value
        elif key == "ellipses":
            self.scene.show_ellipses = value
        self.scene.update_frame(self.frame_idx, self._trail_offsets())

    def _update_status(self) -> None:
        t = self.frame_idx / self.fps
        mins = int(t // 60)
        secs = t - mins * 60
        self.frame_label.setText(
            f"{self.frame_idx} / {self.n_frames - 1}"
        )
        self.status.showMessage(
            f"Frame {self.frame_idx}/{self.n_frames - 1}  "
            f"|  t = {mins}:{secs:05.2f}  "
            f"|  speed {self.speed}x  "
            f"|  fps {self.fps:.1f}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "mufasa pose viewer — playback a smoothed (and "
            "optionally raw) pose file."
        ),
    )
    parser.add_argument(
        "input", type=Path,
        help=(
            "Pose file: parquet or CSV. Auto-detected as smoothed "
            "(if it has _var_x columns) or raw (if not)."
        ),
    )
    parser.add_argument(
        "--raw", type=Path, default=None,
        help=(
            "Raw pose file to overlay. Use when the primary input "
            "is smoothed and you want to compare."
        ),
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Playback frame rate (default 30)",
    )
    parser.add_argument(
        "--likelihood-threshold", type=float, default=0.0,
        help="Hide raw markers with p below this value (default 0)",
    )
    parser.add_argument(
        "--no-skeleton", action="store_true",
        help="Disable default skeleton overlay",
    )
    parser.add_argument(
        "--no-ellipses", action="store_true",
        help="Disable variance ellipses on smoothed markers",
    )
    parser.add_argument(
        "--trail", type=int, default=0,
        help="Length of fade-out trail in frames (default 0 = off)",
    )
    args = parser.parse_args(argv)

    print(f"Loading {args.input}...")
    primary = _load_pose_file(args.input)
    print(
        f"  {primary.n_frames} frames × {len(primary.markers)} markers"
        f" — {'smoothed' if primary.variances is not None else 'raw'}"
    )

    raw = None
    smoothed = None
    if primary.variances is not None:
        smoothed = primary
        if args.raw is not None:
            print(f"Loading raw {args.raw}...")
            raw = _load_pose_file(args.raw)
            print(
                f"  {raw.n_frames} frames × {len(raw.markers)} markers"
            )
    else:
        # Primary is raw
        raw = primary
        if args.raw is not None:
            print(
                "Warning: --raw specified but primary input is also "
                "raw. Loading --raw as smoothed (it must have "
                "variance columns).",
                file=sys.stderr,
            )
            smoothed = _load_pose_file(args.raw)
            if smoothed.variances is None:
                print(
                    "  --raw file has no variance columns; treating "
                    "as raw and using the original input as raw too.",
                    file=sys.stderr,
                )
                smoothed = None

    app = QApplication(sys.argv)
    viewer = PoseViewer(
        smoothed=smoothed,
        raw=raw,
        fps=args.fps,
        likelihood_threshold=args.likelihood_threshold,
        trail_length=args.trail,
        show_skeleton=not args.no_skeleton,
        show_ellipses=not args.no_ellipses,
    )
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
