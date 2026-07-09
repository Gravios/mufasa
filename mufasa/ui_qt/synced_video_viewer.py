"""
mufasa.ui_qt.synced_video_viewer
================================

A separate window that shows two or more videos docked together and keeps
their frames synchronised as best as possible.

Each video is a :class:`mufasa.ui_qt.frame_scrubber.FrameScrubberWidget`
inside its own :class:`QDockWidget`, so the panes can be tiled, stacked, or
floated. Because the scrubber already carries the display rotate/flip
controls (patch 122go), each pane can be oriented independently.

Synchronisation
---------------
Any pane can drive: scrubbing one seeks the others to the matching *time*
(``frame / fps``), so videos with different frame rates or lengths stay
aligned rather than assuming a shared frame index.

Sync is relative to a per-video **reference offset** captured whenever sync
is (re)enabled. So the workflow for imperfectly-aligned recordings is:
turn sync off, nudge each video to a common event (a light flash, a door
opening), turn sync back on — that alignment becomes the zero and scrubbing
keeps the videos locked with that offset.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDockWidget, QMainWindow, QToolBar, QWidget

from mufasa.ui_qt.frame_scrubber import FrameScrubberWidget


class SyncedVideoViewer(QMainWindow):
    """Dock two or more videos together with time-synchronised scrubbing.

    :param video_paths: Paths to the videos (two or more).
    :param parent: Optional parent widget; the viewer is still a top-level
        window.
    """

    def __init__(self, video_paths, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Synced videos")
        # A parented QMainWindow still shows as its own top-level window.
        self.setWindowFlag(Qt.Window, True)

        self._scrubbers: list[FrameScrubberWidget] = []
        self._ref_frames: list[int] = []
        self._guard = False           # re-entrancy guard for sync propagation
        self._sync_enabled = True

        tb = QToolBar("Sync", self)
        self._a_sync = QAction("Sync frames", self)
        self._a_sync.setCheckable(True)
        self._a_sync.setChecked(True)
        self._a_sync.setToolTip(
            "Keep all videos aligned while scrubbing. Turn off to align "
            "them manually, then on again to lock that offset."
        )
        self._a_sync.toggled.connect(self._on_sync_toggled)
        tb.addAction(self._a_sync)
        self.addToolBar(tb)

        prev_dock: QDockWidget | None = None
        for i, path in enumerate(video_paths):
            sc = FrameScrubberWidget(self)
            try:
                sc.load(str(path))
            except Exception as exc:  # noqa: BLE001 - show, don't crash the window
                sc.setToolTip(f"Could not open {path}: {exc}")
            dock = QDockWidget(os.path.basename(str(path)), self)
            dock.setWidget(sc)
            # Movable/floatable but not closable — closing one pane would
            # silently break sync.
            dock.setFeatures(
                QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
            )
            self.addDockWidget(Qt.TopDockWidgetArea, dock)
            if prev_dock is not None:
                self.splitDockWidget(prev_dock, dock, Qt.Horizontal)
            prev_dock = dock
            sc.frame_changed.connect(lambda idx, k=i: self._on_frame(k, idx))
            self._scrubbers.append(sc)

        self._capture_reference()

    # ------------------------------------------------------------------ #
    def _capture_reference(self) -> None:
        """Snapshot the current frame of each pane as the alignment zero."""
        self._ref_frames = [sc.current_frame for sc in self._scrubbers]

    def _on_sync_toggled(self, checked: bool) -> None:
        self._sync_enabled = bool(checked)
        if checked:
            # Lock in whatever (possibly manual) alignment is on screen now.
            self._capture_reference()

    def _on_frame(self, src_idx: int, frame_idx: int) -> None:
        """When one pane moves, seek the others to the matching time,
        offset by the captured reference alignment."""
        if self._guard or not self._sync_enabled:
            return
        self._guard = True
        try:
            src = self._scrubbers[src_idx]
            dt = (frame_idx - self._ref_frames[src_idx]) / max(src.fps, 1.0)
            for j, sc in enumerate(self._scrubbers):
                if j == src_idx:
                    continue
                target = self._ref_frames[j] + round(dt * sc.fps)
                target = max(0, min(target, max(sc.total_frames - 1, 0)))
                if target != sc.current_frame:
                    sc.seek(target)
        finally:
            self._guard = False

    def closeEvent(self, ev) -> None:  # noqa: N802 - Qt override
        for sc in self._scrubbers:
            sc.close_video()
        super().closeEvent(ev)


def open_synced_video_viewer(video_paths, parent: QWidget | None = None):
    """Create, show, and return a :class:`SyncedVideoViewer`."""
    viewer = SyncedVideoViewer(list(video_paths), parent=parent)
    viewer.resize(1200, 600)
    viewer.show()
    return viewer


__all__ = ["SyncedVideoViewer", "open_synced_video_viewer"]
