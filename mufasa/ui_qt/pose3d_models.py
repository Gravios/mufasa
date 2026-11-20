"""
mufasa.ui_qt.pose3d_models
==========================

Patch 122kv: the 3-D viewer's *View & models* panel.

* **View** — markers (on/off, size), marker names, skeleton, floor grid,
  axes gizmo, projection (perspective / orthographic), field of view,
  background.
* **Models** — load a body-model rig (``.rig.npz``), the built-in virtual
  rodent, or a mesh (``.obj``, ``.stl``, ``.ply``). Each has a style (solid,
  transparent, wireframe, points), an opacity and a colour. A mesh also has
  its scale, rotation about the vertical and offset. Rigs follow the 3-D
  markers frame by frame; *Fit all frames* fits every frame up front (for
  smooth playback).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mufasa.pose3d.scene_models import STYLES, load_model

BACKGROUNDS = ("theme", "dark", "light", "black", "white")


class ViewModelsPanel(QWidget):
    def __init__(self, viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.vol = viewer.volume
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._view_box())
        lay.addWidget(self._models_box(), 1)
        self._selected_changed()

    # -- view ------------------------------------------------------------ #
    def _view_box(self) -> QGroupBox:
        box = QGroupBox("View", self)
        g = QGridLayout(box)
        v = self.vol
        self.markers_cb = QCheckBox("Markers", box)
        self.markers_cb.setChecked(v.show_markers)
        self.size_spin = QSpinBox(box)
        self.size_spin.setRange(20, 400)
        self.size_spin.setSuffix(" %")
        self.size_spin.setValue(int(round(100 * v.marker_scale)))
        self.size_spin.setToolTip("Marker size")
        self.names_cb = QCheckBox("Names", box)
        self.names_cb.setChecked(v.show_names)
        self.skel_cb = QCheckBox("Skeleton", box)
        self.skel_cb.setChecked(v.show_skeleton)
        self.grid_cb = QCheckBox("Grid", box)
        self.grid_cb.setChecked(v.show_grid)
        self.axes_cb = QCheckBox("Axes", box)
        self.axes_cb.setChecked(v.show_axes)
        self.proj = QComboBox(box)
        self.proj.addItems(["Perspective", "Orthographic"])
        self.proj.setCurrentIndex(1 if v.ortho else 0)
        self.fov_spin = QSpinBox(box)
        self.fov_spin.setRange(5, 120)
        self.fov_spin.setSuffix("°")
        self.fov_spin.setToolTip("Field of view (perspective)")
        self.fov_spin.setValue(int(round(v.fov * 57.2958)))
        self.bg = QComboBox(box)
        for b in BACKGROUNDS:
            self.bg.addItem(b.capitalize() if b != "theme" else "Theme", b)
        g.addWidget(self.markers_cb, 0, 0)
        g.addWidget(self.size_spin, 0, 1)
        g.addWidget(self.names_cb, 0, 2)
        g.addWidget(self.skel_cb, 1, 0)
        g.addWidget(self.grid_cb, 1, 1)
        g.addWidget(self.axes_cb, 1, 2)
        g.addWidget(QLabel("Projection", box), 2, 0)
        g.addWidget(self.proj, 2, 1)
        g.addWidget(self.fov_spin, 2, 2)
        g.addWidget(QLabel("Background", box), 3, 0)
        g.addWidget(self.bg, 3, 1, 1, 2)
        self.gpu_cb = QCheckBox("GPU", box)                   # 122ky
        self.gpu_cb.setChecked(v.gpu)
        self.gpu_cb.setToolTip("Draw the models with OpenGL (falls back to the CPU when it cannot start)")
        self.gpu_cb.toggled.connect(lambda on: (self._set("gpu", bool(on)), self._update_info()))
        g.addWidget(self.gpu_cb, 4, 0)
        for cb, attr in ((self.markers_cb, "show_markers"), (self.names_cb, "show_names"),
                         (self.skel_cb, "show_skeleton"), (self.grid_cb, "show_grid"), (self.axes_cb, "show_axes")):
            cb.toggled.connect(lambda on, a=attr: self._set(a, bool(on)))
        self.size_spin.valueChanged.connect(lambda val: self._set("marker_scale", val / 100.0))
        self.proj.currentIndexChanged.connect(lambda i: self._set("ortho", i == 1))
        self.fov_spin.valueChanged.connect(lambda val: self._set("fov", val / 57.2958))
        self.bg.currentIndexChanged.connect(lambda _i: self._set("background", self.bg.currentData()))
        return box

    def _set(self, attr, value) -> None:
        setattr(self.vol, attr, value)
        self.vol.update()

    # -- models ---------------------------------------------------------- #
    def _models_box(self) -> QGroupBox:
        box = QGroupBox("Models", self)
        outer = QHBoxLayout(box)
        left = QVBoxLayout()
        self.list = QListWidget(box)
        self.list.setMaximumHeight(110)
        left.addWidget(self.list)
        row = QHBoxLayout()
        self.load_btn = QPushButton("Load…", box)
        self.load_btn.setToolTip("A body-model rig (.rig.npz), posed to the markers; a mesh (.obj .stl .ply); "
                                 "or a gait file (_gait.parquet: paws, support, centre of mass)")
        self.rodent_btn = QPushButton("Virtual rodent", box)
        self.rodent_btn.setToolTip("The built-in rat (DeepMind's virtual rodent), posed to the markers")
        self.remove_btn = QPushButton("Remove", box)
        self.fitall_btn = QPushButton("Fit all frames", box)
        self.fitall_btn.setToolTip("Pose the selected rig to every frame, in the background, starting at the "
                                   "frame shown (batched; on the GPU with CUDA)")
        for b in (self.load_btn, self.rodent_btn, self.remove_btn, self.fitall_btn):
            row.addWidget(b)
        left.addLayout(row)
        self.bg_fit_cb = QCheckBox("Fit in the background", box)       # 122ky
        self.bg_fit_cb.setChecked(True)
        self.bg_fit_cb.setToolTip("Start posing a rig to every frame as soon as it is added, nearest the "
                                  "frame shown first, so playback runs in real time")
        left.addWidget(self.bg_fit_cb)
        self._fit_thread = None
        outer.addLayout(left, 2)
        right = QGridLayout()
        self.style = QComboBox(box)
        for s in STYLES:
            self.style.addItem(s.capitalize(), s)
        self.opacity = QSpinBox(box)
        self.opacity.setRange(5, 100)
        self.opacity.setSuffix(" %")
        self.color_btn = QPushButton("Colour…", box)
        self.scale = QDoubleSpinBox(box)
        self.scale.setRange(1e-4, 1e5)
        self.scale.setDecimals(4)
        self.scale.setToolTip("Mesh units to the 3-D data's (e.g. 1000 for a mesh in metres and data in mm)")
        self.rot = QDoubleSpinBox(box)
        self.rot.setRange(-360, 360)
        self.rot.setSuffix("°")
        self.offs = [QDoubleSpinBox(box) for _ in range(3)]
        for o in self.offs:
            o.setRange(-1e6, 1e6)
            o.setDecimals(1)
        right.addWidget(QLabel("Style", box), 0, 0)
        right.addWidget(self.style, 0, 1)
        right.addWidget(self.opacity, 0, 2)
        right.addWidget(self.color_btn, 0, 3)
        right.addWidget(QLabel("Scale", box), 1, 0)
        right.addWidget(self.scale, 1, 1)
        right.addWidget(QLabel("Turn", box), 1, 2)
        right.addWidget(self.rot, 1, 3)
        right.addWidget(QLabel("Offset x y z", box), 2, 0)
        for k, o in enumerate(self.offs):
            right.addWidget(o, 2, 1 + k)
        self.info = QLabel("", box)
        self.info.setStyleSheet("color: palette(placeholder-text);")
        self.info.setWordWrap(True)
        right.addWidget(self.info, 3, 0, 1, 4)
        outer.addLayout(right, 3)
        self.load_btn.clicked.connect(self.load_dialog)
        self.rodent_btn.clicked.connect(lambda: self.add("virtual_rodent"))
        self.remove_btn.clicked.connect(self.remove_selected)
        self.fitall_btn.clicked.connect(self.fit_all)
        self.list.currentRowChanged.connect(lambda _r: self._selected_changed())
        self.list.itemChanged.connect(self._item_changed)
        self.style.currentIndexChanged.connect(lambda _i: self._look("style", self.style.currentData()))
        self.opacity.valueChanged.connect(lambda val: self._look("opacity", val / 100.0))
        self.color_btn.clicked.connect(self._pick_color)
        self.scale.valueChanged.connect(lambda val: self._place("scale", float(val)))
        self.rot.valueChanged.connect(lambda val: self._place("rot_z_deg", float(val)))
        for k, o in enumerate(self.offs):
            o.valueChanged.connect(lambda _v, k=k: self._place("offset", tuple(x.value() for x in self.offs)))
        self.vol.on_fitted = self._update_info
        return box

    def selected(self):
        r = self.list.currentRow()
        return self.vol.models[r] if 0 <= r < len(self.vol.models) else None

    def add(self, path) -> object | None:
        try:
            m = load_model(path, self.vol.pose)
        except Exception as e:                                 # noqa: BLE001
            QMessageBox.warning(self, "Load model", f"{type(e).__name__}: {e}")
            return None
        self.vol.add_model(m)
        kind = {"rig": "posed to the markers", "mesh": "mesh", "gait": "paws and balance"}[m.kind]
        it = QListWidgetItem(f"{m.name} ({kind})")
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Checked)
        self.list.addItem(it)
        self.list.setCurrentRow(self.list.count() - 1)
        if m.kind == "rig" and self.bg_fit_cb.isChecked():
            self.fit_all()
        return m

    def load_dialog(self) -> None:
        start = str(Path(self.viewer.pose_path).parent)
        fn, _ = QFileDialog.getOpenFileName(self, "Load model", start,
                                            "Models (*.rig.npz *.obj *.stl *.ply *_gait.parquet);;All files (*)")
        if fn:
            self.add(fn)

    def remove_selected(self) -> None:
        r = self.list.currentRow()
        if 0 <= r < len(self.vol.models):
            self.vol.models.pop(r)
            self.list.takeItem(r)
            self.vol.update()
            self._selected_changed()

    def fit_all(self) -> None:
        """122ky: pose the selected rig to every frame in a background thread (batched, the GPU
        with CUDA), nearest the frame shown first; pressed again, it stops."""
        if self._fit_thread is not None and self._fit_thread.isRunning():
            self._fit_thread.requestInterruption()
            return
        m = self.selected()
        if m is None or m.kind != "rig":
            return
        frames = [int(f) for f in self.vol.pose.frames if not m.cached(int(f))]
        if not frames:
            self._update_info()
            return
        th = FitThread(m, frames, lambda: self.vol.frame, self)
        th.progress.connect(self._fit_progress)
        th.finished.connect(self._fit_done)
        self._fit_thread = th
        self.fitall_btn.setText("Stop fitting")
        self.vol.bg_fitting = True
        th.start()

    def _fit_progress(self, done: int, total: int, device: str) -> None:
        self._fit_status = f"posing {done}/{total} frames ({device})"
        self.vol.update()
        self._update_info()

    def _fit_done(self) -> None:
        self.vol.bg_fitting = False
        self.fitall_btn.setText("Fit all frames")
        self._fit_status = ""
        self._fit_thread = None
        self.vol.update()
        self._update_info()

    def stop_fitting(self) -> None:
        """Stop a background fit and wait for it (closing the viewer)."""
        th = self._fit_thread
        if th is not None:
            th.requestInterruption()
            th.wait(60000)

    def closeEvent(self, ev) -> None:  # noqa: ANN001, N802
        self.stop_fitting()
        super().closeEvent(ev)

    def _item_changed(self, it: QListWidgetItem) -> None:
        r = self.list.row(it)
        if 0 <= r < len(self.vol.models):
            self.vol.models[r].look.visible = it.checkState() == Qt.Checked
            self.vol.update()

    def _selected_changed(self) -> None:
        m = self.selected()
        on = m is not None
        for w in (self.style, self.opacity, self.color_btn, self.remove_btn):
            w.setEnabled(on)
        mesh = on and m.kind == "mesh"
        for w in (self.scale, self.rot, *self.offs):
            w.setEnabled(mesh)
        self.fitall_btn.setEnabled(on and m.kind == "rig")
        if not on:
            self.info.setText("")
            return
        for w in (self.style, self.opacity, self.scale, self.rot, *self.offs):
            w.blockSignals(True)
        self.style.setCurrentIndex(max(0, self.style.findData(m.look.style)))
        self.opacity.setValue(int(round(100 * m.look.opacity)))
        if mesh:
            self.scale.setValue(m.scale)
            self.rot.setValue(m.rot_z_deg)
            for o, v in zip(self.offs, m.offset):
                o.setValue(float(v))
        for w in (self.style, self.opacity, self.scale, self.rot, *self.offs):
            w.blockSignals(False)
        self.color_btn.setStyleSheet(f"background: {m.look.color};")
        self._update_info()

    def _update_info(self, *_a) -> None:
        m = self.selected()
        if m is not None:
            frame = self.vol.frame
            if m.kind == "rig" and not m.cached(frame):
                txt = f"{m.info(None)} · frame {frame} not posed yet"
            else:
                txt = m.info(frame)
            extra = [s for s in (getattr(self, "_fit_status", ""), self.vol.gpu_note if self.vol.gpu else "") if s]
            self.info.setText(" · ".join([txt, *extra]))

    def _look(self, attr, value) -> None:
        m = self.selected()
        if m is not None:
            setattr(m.look, attr, value)
            self.vol.update()

    def _place(self, attr, value) -> None:
        m = self.selected()
        if m is not None and m.kind == "mesh":
            setattr(m, attr, value)
            self.vol.update()

    def _pick_color(self) -> None:
        m = self.selected()
        if m is None:
            return
        c = QColorDialog.getColor(QColor(m.look.color), self, "Model colour")
        if c.isValid():
            m.look.color = c.name()
            self.color_btn.setStyleSheet(f"background: {m.look.color};")
            self.vol.update()


class FitThread(QThread):
    """122ky: a rig posed to many frames in the background, chunk by chunk, the chunk nearest
    ``current()`` (the frame shown) first; frames land in the rig's cache as they are done."""
    progress = Signal(int, int, str)

    def __init__(self, model, frames, current, parent=None, *, device: str = "auto") -> None:
        super().__init__(parent)
        from mufasa.pose3d import rig_batch
        self.model, self.frames, self.current = model, sorted(frames), current
        self.device = rig_batch.device_name(device) if rig_batch.available() else "cpu, frame by frame"
        # big batches run fastest (CPU: about 30 ms a frame at 2048, 80 at 256); the first, around
        # the frame shown, is small so that the model appears at once
        self.chunk = 2048
        self.first = 128 if self.device != "cuda" else 512

    def run(self) -> None:
        import numpy as np
        rest = list(self.frames)
        total, done = len(rest), 0
        dev = "auto" if self.device != "cpu" else "cpu"
        size = self.first
        while rest and not self.isInterruptionRequested():
            c = self.current()
            i = int(np.argmin(np.abs(np.asarray(rest) - c)))
            a = max(0, min(i - size // 4, len(rest) - size))      # mostly ahead of the frame shown
            ch, rest = rest[a:a + size], rest[:a] + rest[a + size:]
            self.model.fit_frames(ch, device=dev, stop=self.isInterruptionRequested)
            done += len(ch)
            self.progress.emit(done, total, self.device)
            size = self.chunk


__all__ = ["BACKGROUNDS", "ViewModelsPanel"]
