"""
mufasa.ui_qt.pose3d_animation
=============================

Patch 122kp: camera animation for the 3-D viewer. You set key views and
render the camera moving smoothly between them, together with the data,
to a video file.

A **key** is a view of the 3-D view at a video frame:

* orbit: azimuth, elevation, distance (zoom), target and field of view;
* or the recording camera's view (*Camera*), with its zoom and pan.

Each key also says how many output frames the move to the next key
takes, and how it eases:

* *Ease in-out* starts and stops gently at each key;
* *Linear* moves at a constant rate;
* *Spline* is a Catmull–Rom curve through all the keys. It does not slow
  down at a key in the middle, so the motion flows through it.

Between two keys the data frame advances linearly, from the first key's
video frame to the second's. Real time is when the length equals the
frame difference. A longer length plays the data in slow motion. Two keys
on the same video frame freeze the data while the camera moves. The
azimuth takes the shorter way round, and the zoom (distance) is
interpolated on a log scale, so zooming feels even. A move between two
*Camera* keys stays in the camera view and only zooms and pans; a move
into or out of the camera view flies from or to the orbit pose of the
recording camera.

:func:`render_animation` draws every output frame offscreen at the
chosen size. It can optionally put the video frame below and writes an
.mp4 (mp4v) or .avi (MJPG) file. :class:`CameraAnimationPanel` is the
editor in the viewer. A camera path is saved as JSON
(``<pose>.camera.json``) and rendered without the GUI with::

    mufasa-pose3d render-animation PATH.camera.json --pose X_3d.parquet [--video X.mp4] --out X_anim.mp4
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufasa.pose3d.animation import (  # noqa: F401  (re-exported)
    EASINGS,
    STYLE_KEYS,
    CameraKey,
    CameraPath,
    _qimage_bgr,
    render_animation,
)


# --------------------------------------------------------------------------- #
# the editor
# --------------------------------------------------------------------------- #
class CameraAnimationPanel(QWidget):
    """Keys of the view → a smooth camera move → a video."""

    COLS = ("Video frame", "To next (frames)", "Easing", "View")

    def __init__(self, viewer, parent=None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.vol = viewer.volume
        self.path = CameraPath()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        self.buttons = {}
        for name, tip, fn in (
                ("Add key", "Add the current view at the current frame (after the selected key)",
                 self.add_key),
                ("Update key", "Set the selected key to the current view and frame", self.update_key),
                ("Delete", "Delete the selected key", self.delete_key),
                ("Go to", "Show the selected key's view and frame", self.go_to_key),
                ("Load…", "Load a camera path (.camera.json)", self.load_dialog),
                ("Save…", "Save the camera path (.camera.json)", self.save_dialog)):
            b = QPushButton(name, self)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, f=fn: f())
            row.addWidget(b)
            self.buttons[name] = b
        row.addStretch()
        lay.addLayout(row)
        self.table = QTableWidget(0, len(self.COLS), self)
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setMaximumHeight(170)
        self.table.cellDoubleClicked.connect(lambda r, _c: (self.table.selectRow(r), self.go_to_key()))
        lay.addWidget(self.table)
        trow = QHBoxLayout()
        self.play_btn = QPushButton("Preview", self)
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._play_toggled)
        trow.addWidget(self.play_btn)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.valueChanged.connect(self.show_output_frame)
        trow.addWidget(self.slider, 1)
        self.pos_label = QLabel("", self)
        trow.addWidget(self.pos_label)
        lay.addLayout(trow)
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Size", self))
        self.width_spin, self.height_spin = QSpinBox(self), QSpinBox(self)
        for s, v in ((self.width_spin, 1280), (self.height_spin, 720)):
            s.setRange(64, 7680)
            s.setSingleStep(2)
            s.setValue(v)
            rrow.addWidget(s)
        rrow.addWidget(QLabel("fps", self))
        self.fps_spin = QSpinBox(self)
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(int(round(self._video_fps())))
        rrow.addWidget(self.fps_spin)
        self.below_cb = QCheckBox("Video below", self)
        self.below_cb.setToolTip("Put the video frame under the 3-D view in the output")
        self.below_cb.setEnabled(bool(viewer.video_path))
        self.label_cb = QCheckBox("Frame label", self)
        for w in (self.below_cb, self.label_cb):
            rrow.addWidget(w)
        rrow.addStretch()
        self.render_btn = QPushButton("Render video…", self)
        self.render_btn.clicked.connect(self.render_dialog)
        rrow.addWidget(self.render_btn)
        lay.addLayout(rrow)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.fps_spin.valueChanged.connect(lambda v: self.timer.setInterval(int(1000 / max(v, 1))))
        self._refresh()

    # -- helpers --------------------------------------------------------
    def _video_fps(self) -> float:
        try:
            f = float(self.viewer.scrubber.fps) if self.viewer.video_path else 0.0
        except Exception:
            f = 0.0
        return f if f > 0 else float(self.viewer.pose.meta.get("fps", 30.0) or 30.0)

    def selected(self) -> int:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else -1

    def _current_frame(self) -> int:
        return int(self.vol.frame)

    def _show(self, frame: int, view: dict | None) -> None:
        if self.viewer.video_path:
            if self.viewer.scrubber.current_frame != frame:
                self.viewer.scrubber.seek(frame)
            else:
                self.vol.set_frame(frame)
        else:
            self.vol.set_frame(frame)
        if view is not None:
            self.vol.set_view_state(view)

    def _read_table(self) -> None:
        for r, k in enumerate(self.path.keys):
            k.frame = self.table.cellWidget(r, 0).value()
            k.length = self.table.cellWidget(r, 1).value()
            k.easing = self.table.cellWidget(r, 2).currentText()
        self._update_timeline()

    def _refresh(self, select: int | None = None) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.path.keys))
        fps = self._video_fps()
        for r, k in enumerate(self.path.keys):
            fs = QSpinBox(self.table)
            fs.setRange(0, 10 ** 8)
            fs.setValue(k.frame)
            ls = QSpinBox(self.table)
            ls.setRange(1, 10 ** 7)
            ls.setValue(k.length)
            ls.setSuffix(f"  ({k.length / fps:.1f} s)")
            ls.setEnabled(r < len(self.path.keys) - 1)
            ec = QComboBox(self.table)
            ec.addItems(EASINGS)
            ec.setCurrentText(k.easing)
            ec.setEnabled(r < len(self.path.keys) - 1)
            for wdg, sig in ((fs, fs.valueChanged), (ls, ls.valueChanged), (ec, ec.currentTextChanged)):
                sig.connect(lambda *_: self._read_table())
            ls.valueChanged.connect(lambda v, s=ls: s.setSuffix(f"  ({v / self._video_fps():.1f} s)"))
            self.table.setCellWidget(r, 0, fs)
            self.table.setCellWidget(r, 1, ls)
            self.table.setCellWidget(r, 2, ec)
            it = QTableWidgetItem(k.describe())
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 3, it)
        self.table.blockSignals(False)
        if select is not None and 0 <= select < len(self.path.keys):
            self.table.selectRow(select)
        for name in ("Update key", "Delete", "Go to", "Save…"):
            self.buttons[name].setEnabled(bool(self.path.keys))
        self.render_btn.setEnabled(len(self.path.keys) >= 1)
        self.play_btn.setEnabled(len(self.path.keys) >= 2)
        self._update_timeline()

    def _update_timeline(self) -> None:
        n = self.path.n_output()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.blockSignals(False)
        fps = max(self.fps_spin.value(), 1)
        self.pos_label.setText(f"{self.slider.value() / fps:.1f} / {max(n - 1, 0) / fps:.1f} s"
                               f"  ({n} frames)")

    # -- keys -----------------------------------------------------------
    def add_key(self) -> int:
        frame, view = self._current_frame(), self.vol.view_state()
        r = self.selected()
        at = len(self.path.keys) if r < 0 else r + 1
        fps = self._video_fps()
        key = CameraKey(frame, view, int(round(2 * fps)))
        if at > 0:
            prev = self.path.keys[at - 1]
            if frame > prev.frame:                   # real time to the new key
                prev.length = frame - prev.frame
        if at < len(self.path.keys) and self.path.keys[at].frame > frame:
            key.length = self.path.keys[at].frame - frame
        self.path.keys.insert(at, key)
        self._refresh(select=at)
        return at

    def update_key(self) -> None:
        r = self.selected()
        if r < 0:
            return
        self.path.keys[r].frame = self._current_frame()
        self.path.keys[r].view = self.vol.view_state()
        self._refresh(select=r)

    def delete_key(self) -> None:
        r = self.selected()
        if r < 0:
            return
        del self.path.keys[r]
        self._refresh(select=min(r, len(self.path.keys) - 1))

    def go_to_key(self) -> None:
        r = self.selected()
        if r < 0:
            return
        k = self.path.keys[r]
        self._show(k.frame, k.view)
        self.slider.blockSignals(True)
        self.slider.setValue(self.path.key_starts()[r])
        self.slider.blockSignals(False)
        self._update_timeline()

    # -- preview --------------------------------------------------------
    def show_output_frame(self, n: int) -> None:
        if not self.path.keys:
            return
        self._show(self.path.frame_at(n), self.path.state_at(n))
        self._update_timeline()

    def _play_toggled(self, on: bool) -> None:
        if on and len(self.path.keys) >= 2:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
            self.timer.start(int(1000 / max(self.fps_spin.value(), 1)))
        else:
            self.timer.stop()
            self.play_btn.setChecked(False)

    def _tick(self) -> None:
        if self.slider.value() >= self.slider.maximum():
            self.play_btn.setChecked(False)
            return
        self.slider.setValue(self.slider.value() + 1)

    # -- files ----------------------------------------------------------
    def settings(self) -> dict:
        return {"fps": self.fps_spin.value(), "size": [self.width_spin.value(), self.height_spin.value()],
                "video_below": self.below_cb.isChecked(), "frame_label": self.label_cb.isChecked(),
                "style": {k: getattr(self.vol, k) for k in STYLE_KEYS}}

    def default_path(self, suffix: str) -> Path:
        p = self.viewer.pose_path
        return p.with_name(p.stem.removesuffix("_3d") + suffix)

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(self.path.to_json(**self.settings()))
        return path

    def load(self, path) -> None:
        cp, st = CameraPath.from_json(Path(path).read_text())
        self.path = cp
        if "fps" in st:
            self.fps_spin.setValue(int(st["fps"]))
        if "size" in st:
            self.width_spin.setValue(int(st["size"][0]))
            self.height_spin.setValue(int(st["size"][1]))
        self.below_cb.setChecked(bool(st.get("video_below")) and self.below_cb.isEnabled())
        self.label_cb.setChecked(bool(st.get("frame_label")))
        self._refresh(select=0 if cp.keys else None)

    def save_dialog(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(self, "Save camera path", str(self.default_path(".camera.json")),
                                            "Camera path (*.camera.json *.json)")
        if fn:
            self.save(fn)

    def load_dialog(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Load camera path", str(self.viewer.pose_path.parent),
                                            "Camera path (*.camera.json *.json)")
        if fn:
            try:
                self.load(fn)
            except Exception as e:
                QMessageBox.warning(self, "Load camera path", f"{type(e).__name__}: {e}")

    def render_to(self, out_path, progress=None) -> Path:
        st = self.settings()
        return render_animation(self.viewer.pose, self.path, out_path, video_path=self.viewer.video_path,
                                size=st["size"], fps=st["fps"], style=st["style"],
                                colors=dict(self.vol.colors), video_below=st["video_below"],
                                frame_label=st["frame_label"], progress=progress)

    def render_dialog(self) -> None:
        from PySide6.QtWidgets import QApplication
        if not self.path.keys:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Render animation", str(self.default_path("_animation.mp4")),
                                            "Video (*.mp4 *.avi)")
        if not fn:
            return
        self.play_btn.setChecked(False)
        total = self.path.n_output()
        dlg = QProgressDialog("Rendering…", "Stop", 0, total, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)

        def prog(done, n):
            dlg.setValue(done)
            QApplication.processEvents()
            return not dlg.wasCanceled()
        try:
            out = self.render_to(fn, progress=prog)
        except Exception as e:
            dlg.close()
            QMessageBox.warning(self, "Render animation", f"{type(e).__name__}: {e}")
            return
        stopped = dlg.wasCanceled()
        dlg.close()
        self.save(Path(out).with_suffix(".camera.json"))        # how it was made, beside it
        print(f"camera animation: {'stopped, partial ' if stopped else ''}{out} "
              f"({total} frames at {self.fps_spin.value()} fps)")


__all__ = ["EASINGS", "CameraAnimationPanel", "CameraKey", "CameraPath", "render_animation"]
