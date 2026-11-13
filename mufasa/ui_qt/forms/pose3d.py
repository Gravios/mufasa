"""
mufasa.ui_qt.forms.pose3d
=========================

Patch 122kf: forms of the "3-D" workbench tab.

* :class:`CameraCalibrationForm` — mark the box on a frame of the empty
  box (:class:`BoxMarkDialog`: four floor corners, points along the
  vertical wall-corner edges), enter the box size and the camera height
  (or calibrate the lens with a checkerboard video/folder, which also
  solves the height), save ``sources/calibration/camera_<name>.toml``.
* :class:`Pose3DInferenceForm` — 2-D labels (+ Kalman v2 run to fill
  gaps) → 3-D for each video with a camera, optional 3-D Kalman
  smoothing → ``derived/pose3d/<run_id>/<video>_3d.parquet``.
* :class:`Pose3DSmoothingForm` — 3-D Kalman smoothing of an existing run
  into a new run.
* :class:`Pose3DViewerLauncher` — pick a video and a 3-D file and open the
  3-D viewer (volume above, video below).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mufasa.ui_qt.workbench import OperationForm

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


# --------------------------------------------------------------------------- #
# project helpers
# --------------------------------------------------------------------------- #
def _root(config_path) -> Path | None:
    if not config_path:
        return None
    try:
        from mufasa.project_layout import resolve_v1_project_root
        return resolve_v1_project_root(str(config_path))
    except Exception:
        return None


def _videos(root: Path | None) -> list[Path]:
    d = root / "sources" / "videos" if root else None
    if not d or not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def _kalman_runs(root: Path | None) -> list[Path]:
    if root is None:
        return []
    from mufasa.project_layout import ProjectPaths, SmoothingFlavors, Stages
    return ProjectPaths(root).list_runs(Stages.SMOOTHED, flavor=SmoothingFlavors.KALMAN_V2)


def _pose3d_runs(root: Path | None) -> list[Path]:
    if root is None:
        return []
    from mufasa.project_layout import ProjectPaths
    return ProjectPaths(root).list_runs("pose3d")


def background_frame(video: str, n: int = 60) -> np.ndarray:
    """Median of ``n`` frames spread over the video (the animal vanishes)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for k in np.linspace(0, max(0, total - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if ok:
            frames.append(im)
    cap.release()
    if not frames:
        raise OSError(f"cannot read {video}")
    return np.median(np.array(frames), 0).astype(np.uint8)


def _browse_row(parent, edit: QLineEdit, *, folder: bool, caption: str, filt: str = "") -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(edit, 1)
    b = QPushButton("…", parent)
    b.setFixedWidth(32)

    def pick():
        if folder:
            p = QFileDialog.getExistingDirectory(parent, caption, edit.text())
        else:
            p, _ = QFileDialog.getOpenFileName(parent, caption, edit.text(), filt)
        if p:
            edit.setText(p)
    b.clicked.connect(pick)
    row.addWidget(b)
    return row


# --------------------------------------------------------------------------- #
# Box marking
# --------------------------------------------------------------------------- #
class _ClickImage(QWidget):
    """Shows an image; clicks are reported in image pixels."""

    def __init__(self, img_bgr: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.scale = max(1.0, min(1280 / w, 760 / h))
        self.setFixedSize(int(w * self.scale), int(h * self.scale))
        self.floor: list[tuple[float, float]] = []
        self.edges: list[list[tuple[float, float]]] = [[]]
        self.mode = "floor"
        self.on_change = None

    def mousePressEvent(self, ev) -> None:  # noqa: N802, ANN001
        p = ev.position()
        xy = (p.x() / self.scale, p.y() / self.scale)
        if ev.button() == Qt.RightButton:
            self.undo()
            return
        if self.mode == "floor":
            if len(self.floor) < 4:
                self.floor.append(xy)
        else:
            self.edges[-1].append(xy)
        self._changed()

    def undo(self) -> None:
        if self.mode == "floor" and self.floor:
            self.floor.pop()
        elif self.mode != "floor":
            if self.edges[-1]:
                self.edges[-1].pop()
            elif len(self.edges) > 1:
                self.edges.pop()
        self._changed()

    def next_edge(self) -> None:
        if self.edges[-1]:
            self.edges.append([])
        self._changed()

    def clear(self) -> None:
        self.floor, self.edges = [], [[]]
        self._changed()

    def _changed(self) -> None:
        self.update()
        if self.on_change:
            self.on_change()

    def marks(self) -> dict:
        return {"image_size": [self._img.width(), self._img.height()],
                "floor_corners": [list(p) for p in self.floor],
                "vertical_edges": [[list(p) for p in e] for e in self.edges if len(e) >= 2]}

    def set_marks(self, m: dict) -> None:
        self.floor = [tuple(p) for p in m.get("floor_corners", [])][:4]
        self.edges = [[tuple(p) for p in e] for e in m.get("vertical_edges", [])] + [[]]
        self._changed()

    def paintEvent(self, _ev) -> None:  # noqa: N802, ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.drawImage(self.rect(), self._img)
        s = self.scale
        pen = QPen(QColor("#2ecc71"), 2)
        p.setPen(pen)
        pts = [QPointF(x * s, y * s) for x, y in self.floor]
        for i in range(len(pts)):
            if i + 1 < len(pts) or len(pts) == 4:
                p.drawLine(pts[i], pts[(i + 1) % len(pts)])
        p.setBrush(QColor("#2ecc71"))
        for q in pts:
            p.drawEllipse(q, 4, 4)
        p.setPen(QPen(QColor("#e67e22"), 2))
        p.setBrush(QColor("#e67e22"))
        for e in self.edges:
            qs = [QPointF(x * s, y * s) for x, y in e]
            for a, b in zip(qs[:-1], qs[1:]):
                p.drawLine(a, b)
            for q in qs:
                p.drawEllipse(q, 3, 3)
        p.end()


class BoxMarkDialog(QDialog):
    """Click the four floor corners, then points along each vertical
    wall-corner edge (where two walls meet). Right-click undoes."""

    def __init__(self, img_bgr: np.ndarray, marks: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mark the box")
        lay = QVBoxLayout(self)
        self.help = QLabel(self)
        self.help.setWordWrap(True)
        lay.addWidget(self.help)
        self.img = _ClickImage(img_bgr, self)
        lay.addWidget(self.img, 0, Qt.AlignCenter)
        row = QHBoxLayout()
        self.floor_rb = QRadioButton("Floor corners (4)", self)
        self.edge_rb = QRadioButton("Wall-corner edges", self)
        self.floor_rb.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.floor_rb)
        grp.addButton(self.edge_rb)
        row.addWidget(self.floor_rb)
        row.addWidget(self.edge_rb)
        self.next_btn = QPushButton("Next edge", self)
        self.undo_btn = QPushButton("Undo", self)
        self.clear_btn = QPushButton("Clear", self)
        for b in (self.next_btn, self.undo_btn, self.clear_btn):
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        self.status = QLabel(self)
        lay.addWidget(self.status)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.ok_btn = bb.button(QDialogButtonBox.Ok)
        self.floor_rb.toggled.connect(self._mode)
        self.next_btn.clicked.connect(self.img.next_edge)
        self.undo_btn.clicked.connect(self.img.undo)
        self.clear_btn.clicked.connect(self.img.clear)
        self.img.on_change = self._status
        if marks:
            self.img.set_marks(marks)
        self._mode()

    def _mode(self) -> None:
        self.img.mode = "floor" if self.floor_rb.isChecked() else "edge"
        self.next_btn.setEnabled(self.img.mode == "edge")
        self.help.setText(
            "Click the four corners of the box floor (any order)." if self.img.mode == "floor" else
            "Click several points along one vertical edge where two walls meet, then 'Next edge'. "
            "Two or more edges locate the point directly below the camera.")
        self._status()

    def _status(self) -> None:
        m = self.img.marks()
        self.status.setText(f"{len(m['floor_corners'])}/4 floor corners · {len(m['vertical_edges'])} wall-corner edges")
        self.ok_btn.setEnabled(len(m["floor_corners"]) == 4)

    def marks(self) -> dict:
        return self.img.marks()


# --------------------------------------------------------------------------- #
# Camera calibration
# --------------------------------------------------------------------------- #
class CameraCalibrationForm(OperationForm):
    title = "Camera calibration"
    section_id = "pose3d_calibration"
    description = (
        "Mark the box on a frame of the empty box, enter its size and the camera height, and save "
        "a camera for 3-D inference. With a checkerboard video (or image folder) the lens is "
        "calibrated too and the camera height is solved from the marks. Cameras are saved in "
        "sources/calibration/ as camera_<name>.toml; a video uses the camera named after the first "
        "part of its name (camera_K7.toml for K7_cam_…), else camera.toml.")

    def build(self) -> None:
        self._root = _root(self.config_path)
        self._marks: dict | None = None
        f = QFormLayout()
        self.video = QComboBox(self)
        for v in _videos(self._root):
            self.video.addItem(v.name, str(v))
        self.video.setEditable(True)
        f.addRow("Video", self.video)
        self.name = QLineEdit(self)
        self.name.setPlaceholderText("e.g. K7 (blank: camera.toml)")
        f.addRow("Camera name", self.name)
        mrow = QHBoxLayout()
        self.mark_btn = QPushButton("Mark box…", self)
        self.marks_lbl = QLabel("not marked", self)
        mrow.addWidget(self.mark_btn)
        mrow.addWidget(self.marks_lbl, 1)
        f.addRow("Box marks", mrow)

        def spin(v, lo, hi, suffix, dec=0):
            s = QDoubleSpinBox(self)
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setValue(v)
            s.setSuffix(suffix)
            return s
        self.length = spin(0, 0, 5000, " mm")
        self.length.setSpecialValueText("from the marks")
        self.width = spin(300, 0, 5000, " mm")
        self.wall = spin(300, 0, 5000, " mm")
        self.height = spin(430, 0, 10000, " mm")
        f.addRow("Floor length", self.length)
        f.addRow("Floor width", self.width)
        f.addRow("Wall height", self.wall)
        f.addRow("Camera height (above floor)", self.height)
        self.trust = QComboBox(self)
        self.trust.addItems(["width", "length"])
        self.trust.setToolTip("If length and width disagree with the marked floor's shape, keep this one")
        f.addRow("If they disagree, trust", self.trust)
        self.cb_src = QLineEdit(self)
        self.cb_src.setPlaceholderText("optional: checkerboard video or image folder")
        f.addRow("Lens (checkerboard)", _browse_row(self, self.cb_src, folder=False,
                                                    caption="Checkerboard video", filt="Video (*.mp4 *.avi *.mov *.mkv)"))
        crow = QHBoxLayout()
        self.cb_cols, self.cb_rows = QSpinBox(self), QSpinBox(self)
        for s, v in ((self.cb_cols, 9), (self.cb_rows, 6)):
            s.setRange(2, 40)
            s.setValue(v)
        self.cb_sq = spin(20, 1, 500, " mm", 1)
        crow.addWidget(QLabel("inner corners", self))
        crow.addWidget(self.cb_cols)
        crow.addWidget(QLabel("×", self))
        crow.addWidget(self.cb_rows)
        crow.addWidget(QLabel("square", self))
        crow.addWidget(self.cb_sq)
        crow.addStretch()
        f.addRow("", crow)
        self.body_layout.addLayout(f)
        self.report = QLabel(self)
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body_layout.addWidget(self.report)
        self.mark_btn.clicked.connect(self.mark)
        self.video.currentIndexChanged.connect(self._video_changed)
        self.completed.connect(lambda: self.report.setText(getattr(self, "_report_text", "")))
        self._video_changed()

    def _cal_dir(self) -> Path:
        return (self._root / "sources" / "calibration") if self._root else Path.cwd()

    def _marks_path(self) -> Path:
        n = self.name.text().strip() or "camera"
        return self._cal_dir() / f"{n}.marks.json"

    def _video_changed(self) -> None:
        v = self.video.currentText()
        if v and not self.name.text().strip():
            self.name.setText(Path(v).stem.split("_")[0])
        mp = self._marks_path()
        if mp.is_file():
            try:
                self._marks = json.loads(mp.read_text())
                self._show_marks()
            except Exception:
                pass

    def _show_marks(self) -> None:
        m = self._marks or {}
        self.marks_lbl.setText(f"{len(m.get('floor_corners', []))} floor corners, "
                               f"{len(m.get('vertical_edges', []))} wall-corner edges" if m else "not marked")

    def mark(self, img: np.ndarray | None = None) -> dict | None:
        if img is None:
            path = self.video.currentData() or self.video.currentText()
            if not path:
                return None
            img = background_frame(path)
        dlg = BoxMarkDialog(img, self._marks, self)
        if dlg.exec() == QDialog.Accepted:
            self._marks = dlg.marks()
            self._show_marks()
        return self._marks

    def collect_args(self) -> dict:
        if not self._marks or len(self._marks.get("floor_corners", [])) != 4:
            raise ValueError("mark the four floor corners first (Mark box…)")
        cb = self.cb_src.text().strip()
        if not cb and not self.height.value():
            raise ValueError("enter the camera height, or give a checkerboard to calibrate the lens")
        n = self.name.text().strip()
        return dict(marks=self._marks, length=self.length.value() or None, width=self.width.value() or None,
                    wall=self.wall.value() or None, height=self.height.value() or None,
                    trust=self.trust.currentText(), checkerboard=cb or None,
                    pattern=(self.cb_cols.value(), self.cb_rows.value()), square=self.cb_sq.value(),
                    out=str(self._cal_dir() / (f"camera_{n}.toml" if n else "camera.toml")),
                    marks_out=str(self._marks_path()))

    def target(self, *, marks, length, width, wall, height, trust, checkerboard, pattern, square,
               out, marks_out) -> None:
        from mufasa.pose3d.camera import calibrate_checkerboard, calibrate_from_box_marks
        K = dist = None
        lens = ""
        if checkerboard:
            K, dist, rms, n, _ = calibrate_checkerboard(checkerboard, pattern, square, progress=print)
            lens = f"lens: {n} checkerboard views, rms {rms:.2f} px\n"
        cam, rep = calibrate_from_box_marks(
            marks["floor_corners"], floor_length_mm=length, floor_width_mm=width, wall_height_mm=wall,
            vertical_edges_px=marks.get("vertical_edges"), camera_height_mm=height,
            image_size=tuple(marks.get("image_size", (0, 0))), K=K, dist=dist, trust=trust)
        cam.notes += "\n" + lens + rep.text()
        Path(marks_out).parent.mkdir(parents=True, exist_ok=True)
        Path(marks_out).write_text(json.dumps(marks, indent=1))
        cam.save(out)
        self._report_text = f"Saved {out}\n{lens}{rep.text()}"
        print(self._report_text)


# --------------------------------------------------------------------------- #
# 3-D inference
# --------------------------------------------------------------------------- #
class Pose3DInferenceForm(OperationForm):
    title = "3-D inference"
    section_id = "pose3d"
    description = (
        "3-D marker positions from one camera (rigid head; spine, hips and pelvis tilt from Rat7M "
        "motion capture; the back kept within rat bend limits; tail draped) or from two cameras "
        "(triangulation), optionally smoothed with a 3-D Kalman filter. Low-confidence 2-D labels are "
        "filled from a Kalman v2 run (Preprocessing tab). Writes derived/pose3d/<run>/<video>_3d.parquet.")

    def build(self) -> None:
        self._root = _root(self.config_path)
        f = QFormLayout()
        self.pose_dir = QLineEdit(str(self._root / "sources" / "pose") if self._root else "", self)
        f.addRow("2-D labels", _browse_row(self, self.pose_dir, folder=True, caption="2-D pose folder"))
        frow = QHBoxLayout()
        self.use_kv2 = QCheckBox("fill from Kalman v2 run", self)
        self.kv2 = QComboBox(self)
        for r in reversed(_kalman_runs(self._root)):
            self.kv2.addItem(r.name, str(r))
        self.use_kv2.setChecked(self.kv2.count() > 0)
        frow.addWidget(self.use_kv2)
        frow.addWidget(self.kv2, 1)
        f.addRow("Gaps", frow)
        self.cal_dir = QLineEdit(str(self._root / "sources" / "calibration") if self._root else "", self)
        f.addRow("Cameras", _browse_row(self, self.cal_dir, folder=True, caption="Calibration folder"))
        self.videos = QListWidget(self)
        self.videos.setMaximumHeight(130)
        f.addRow("Videos", self.videos)
        self.mode = QComboBox(self)
        self.mode.addItems(["single camera", "two cameras"])
        trow = QHBoxLayout()
        self.tok0, self.tok1 = QLineEdit("camA", self), QLineEdit("camB", self)
        trow.addWidget(self.mode)
        trow.addWidget(QLabel("names differ by", self))
        trow.addWidget(self.tok0)
        trow.addWidget(QLabel("/", self))
        trow.addWidget(self.tok1)
        f.addRow("Mode", trow)

        def dspin(v, lo, hi, suffix, dec=1, tip=""):
            s = QDoubleSpinBox(self)
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setValue(v)
            s.setSuffix(suffix)
            s.setToolTip(tip)
            return s
        self.head_w = dspin(36.0, 5, 200, " mm", tip="distance between the head_left and head_right labels")
        self.body_scale = dspin(1.0, .3, 3, "", 2, "this rat's size relative to the Rat7M rats")
        self.wall_tol = dspin(20.0, 0, 200, " mm", 0, "labels may sit this far beyond a wall before it pushes "
                              "back; hard walls push heads up when the box size or calibration is slightly off")
        f.addRow("Ear-to-ear distance", self.head_w)
        f.addRow("Wall tolerance", self.wall_tol)
        f.addRow("Body size vs Rat7M", self.body_scale)
        orow = QHBoxLayout()
        self.prior = QCheckBox("Rat7M trunk heights", self)
        self.chain = QCheckBox("spine within bend limits", self)
        self.prior.setChecked(True)
        self.chain.setChecked(True)
        orow.addWidget(self.prior)
        orow.addWidget(self.chain)
        orow.addStretch()
        f.addRow("Body model", orow)
        krow = QHBoxLayout()
        self.kal = QCheckBox("3-D Kalman smoothing", self)
        self.kal.setChecked(True)
        self.sig_a = dspin(3000, 10, 1e6, " mm/s^1.5", 0, "process noise (larger = follows faster motion)")
        krow.addWidget(self.kal)
        krow.addWidget(self.sig_a)
        krow.addStretch()
        f.addRow("Smoothing", krow)
        self.fps = dspin(30, 1, 1000, " fps", 1)
        f.addRow("Frame rate", self.fps)
        from mufasa.pose3d.pipeline import default_workers
        self.workers = QSpinBox(self)
        self.workers.setRange(1, max(1, os.cpu_count() or 1))
        self.workers.setValue(default_workers())
        self.workers.setToolTip("Processes. Several videos run at once; a single video splits its "
                                "6000-frame chunks over them. Each needs a few hundred MB.")
        f.addRow("Workers", self.workers)
        self.body_layout.addLayout(f)
        self.cal_dir.textChanged.connect(self.refresh)
        self.pose_dir.textChanged.connect(self.refresh)
        self.mode.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        from mufasa.pose3d.pipeline import find_jobs, pose_files
        self.videos.clear()
        stereo = self.mode.currentIndex() == 1
        jobs, problems = find_jobs(self.pose_dir.text(), self.cal_dir.text(), None,
                                   stereo_tokens=(self.tok0.text(), self.tok1.text()) if stereo else None)
        ok = {j.stem for j in jobs}
        for stem in sorted(pose_files(self.pose_dir.text())):
            if stereo and self.tok0.text() not in stem:
                continue
            it = QListWidgetItem(stem if stem in ok else f"{stem}  (no camera)")
            it.setData(Qt.UserRole, stem)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if stem in ok else Qt.Unchecked)
            if stem not in ok:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
            self.videos.addItem(it)

    def collect_args(self) -> dict:
        from mufasa.pose3d.kalman3d import Kalman3DConfig
        from mufasa.pose3d.mono import MonoConfig
        from mufasa.pose3d.pipeline import RunSettings
        stems = [self.videos.item(i).data(Qt.UserRole) for i in range(self.videos.count())
                 if self.videos.item(i).checkState() == Qt.Checked]
        if not stems:
            raise ValueError("no videos selected (each needs a camera — see Camera calibration)")
        if not self.config_path:
            raise ValueError("open a project first")
        mono = MonoConfig(head_width_mm=self.head_w.value(), body_scale=self.body_scale.value(),
                          fps=self.fps.value(), use_trunk_prior=self.prior.isChecked(),
                          wall_tolerance_mm=self.wall_tol.value(),
                          spine_chain=self.chain.isChecked())
        kal = Kalman3DConfig(fps=self.fps.value(), sigma_a=self.sig_a.value()) if self.kal.isChecked() else None
        st = RunSettings(mode="stereo" if self.mode.currentIndex() == 1 else "mono", mono=mono, kalman=kal,
                         workers=self.workers.value())
        return dict(settings=st, stems=stems, pose_dir=self.pose_dir.text(),
                    fallback=self.kv2.currentData() if self.use_kv2.isChecked() else None,
                    cal_dir=self.cal_dir.text(),
                    tokens=(self.tok0.text(), self.tok1.text()) if st.mode == "stereo" else None)

    def target(self, *, settings, stems, pose_dir, fallback, cal_dir, tokens) -> None:
        from mufasa.pose3d.pipeline import run_project
        out, reps, problems = run_project(self.config_path, settings, stems=stems, pose_dir=pose_dir,
                                          fallback_dir=fallback, calibration_dir=cal_dir,
                                          use_fallback=bool(fallback), stereo_tokens=tokens, progress=print)
        self._last_run_id = Path(out).name
        for p in problems:
            print("skipped:", p)
        print(f"3-D pose: {len(reps)} video(s) → {out}")


# --------------------------------------------------------------------------- #
# 3-D Kalman smoothing of a run
# --------------------------------------------------------------------------- #
class Pose3DSmoothingForm(OperationForm):
    title = "3-D Kalman smoothing"
    section_id = "pose3d_kalman"
    description = (
        "Smooth an existing 3-D run with a constant-velocity Kalman filter and backward (RTS) pass. "
        "For single-camera runs the noise is assumed to lie mostly along each camera ray, so depth is "
        "evened out while the points stay on their 2-D labels. Writes a new derived/pose3d/<run>/.")

    def build(self) -> None:
        self._root = _root(self.config_path)
        f = QFormLayout()
        self.run = QComboBox(self)
        for r in reversed(_pose3d_runs(self._root)):
            self.run.addItem(r.name, str(r))
        f.addRow("3-D run", self.run)

        def dspin(v, lo, hi, suffix, dec=1):
            s = QDoubleSpinBox(self)
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setValue(v)
            s.setSuffix(suffix)
            return s
        self.sig_a = dspin(3000, 10, 1e6, " mm/s^1.5", 0)
        self.sig_p = dspin(2.0, .1, 500, " mm")
        self.sig_r = dspin(25.0, .1, 1000, " mm")
        self.gate = dspin(6.0, 1, 100, " σ")
        self.fps = dspin(30, 1, 1000, " fps")
        f.addRow("Process noise", self.sig_a)
        f.addRow("Noise across the ray", self.sig_p)
        f.addRow("Noise along the ray", self.sig_r)
        f.addRow("Outlier gate", self.gate)
        f.addRow("Frame rate", self.fps)
        self.body_layout.addLayout(f)

    def collect_args(self) -> dict:
        from mufasa.pose3d.kalman3d import Kalman3DConfig
        if not self.run.currentData():
            raise ValueError("no 3-D run to smooth (run 3-D inference first)")
        return dict(run=self.run.currentData(), cfg=Kalman3DConfig(
            fps=self.fps.value(), sigma_a=self.sig_a.value(), sigma_perp=self.sig_p.value(),
            sigma_ray=self.sig_r.value(), gate=self.gate.value()))

    def target(self, *, run, cfg) -> None:
        out = smooth_run(run, cfg, self._root)
        self._last_run_id = out.name
        print(f"smoothed → {out}")


def smooth_run(run_dir, cfg, root=None) -> Path:
    from dataclasses import asdict

    from mufasa.pose3d.io import marker_names, read_pose3d, write_pose3d
    from mufasa.pose3d.kalman3d import smooth_pose3d_frame
    from mufasa.project_layout import ProjectPaths, generate_run_id, write_run_toml
    run_dir = Path(run_dir)
    rid = generate_run_id()
    out = (ProjectPaths(root).stage_run_dir("pose3d", run_id=rid) if root else run_dir.parent / rid)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(run_dir.glob("*_3d.parquet"))
    for p in files:
        df, meta = read_pose3d(p)
        center = None
        P = (meta.get("camera") or {}).get("P")
        if P is not None and "two cameras" not in meta.get("method", ""):
            _, _, vt = np.linalg.svd(np.asarray(P, float))
            c = vt[-1]
            center = c[:3] / c[3]
        sm, info = smooth_pose3d_frame(df, marker_names(df), cfg, center)
        meta["method"] = meta.get("method", "") + f"; 3-D Kalman (RTS) smoothing σa={cfg.sigma_a:g}"
        meta.setdefault("report", {})["kalman3d"] = info
        write_pose3d(sm, out / p.name, meta)
        print(f"{p.name}: moved {info['moved_mm_median']:.1f} mm median")
    write_run_toml(out / "run.toml", stage="pose3d", run_id=out.name, inputs=[str(p) for p in files],
                   params={"kalman3d": asdict(cfg), "source_run": run_dir.name})
    return out


# --------------------------------------------------------------------------- #
# Viewer launcher
# --------------------------------------------------------------------------- #
class Pose3DViewerLauncher(QWidget):
    """Video + 3-D file → the 3-D viewer."""

    def __init__(self, parent=None, config_path=None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self._root = _root(config_path)
        lay = QFormLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        self.video = QComboBox(self)
        for v in _videos(self._root):
            self.video.addItem(v.name, str(v))
        self.file = QComboBox(self)
        lay.addRow("Video", self.video)
        lay.addRow("3-D file", self.file)
        row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh", self)
        self.open_btn = QPushButton("Open 3-D viewer", self)
        row.addStretch()
        row.addWidget(self.refresh_btn)
        row.addWidget(self.open_btn)
        lay.addRow(row)
        self.video.currentIndexChanged.connect(self.refresh)
        self.refresh_btn.clicked.connect(self.refresh)
        self.open_btn.clicked.connect(self.open)
        self.refresh()

    def refresh(self) -> None:
        self.file.clear()
        stem = Path(self.video.currentText()).stem
        files = []
        for r in reversed(_pose3d_runs(self._root)):
            files += [p for p in sorted(r.glob("*.parquet")) if not stem or p.name.startswith(stem)]
        for p in files:
            self.file.addItem(f"{p.parent.name}/{p.name}", str(p))
        self.open_btn.setEnabled(self.file.count() > 0)

    def open(self):
        from mufasa.ui_qt.pose3d_viewer import open_pose3d_viewer
        if not self.file.currentData():
            return None
        return open_pose3d_viewer(self, self.video.currentData(), self.file.currentData())


__all__ = ["BoxMarkDialog", "CameraCalibrationForm", "Pose3DInferenceForm", "Pose3DSmoothingForm",
           "Pose3DViewerLauncher", "background_frame", "smooth_run"]
