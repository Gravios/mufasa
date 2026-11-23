"""
mufasa.ui_qt.forms.bodymodel
============================

Patch 122kq: the "Body model" tab.

* :class:`RigImportForm` — a rigged model from a ``.blend`` (armature +
  skinned mesh) into ``sources/bodymodel/<name>.rig.npz``. Reads the file
  with Blender's Python module (``pip install bpy``) or a Blender
  executable, in a separate process. Or the built-in virtual rodent (a rat;
  patch 122ks), which needs no Blender.
* :class:`BodyFitForm` — fit the model to a video, frame by frame, from
  the silhouette and the 2-D labels, joints within per-joint limits;
  refined forwards and backwards over time; optional overlay video.
  Writes ``derived/bodymodel/<run>/<video>_fit.parquet``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
)

from mufasa.ui_qt.forms.pose3d import _browse_row, _kalman_runs, _pose3d_runs, _root, _videos
from mufasa.ui_qt.workbench import OperationForm


def rig_dir(root: Path | None) -> Path | None:
    return root / "sources" / "bodymodel" if root else None


def list_rigs(root: Path | None) -> list[Path]:
    d = rig_dir(root)
    return sorted(d.glob("*.rig.npz")) if d and d.is_dir() else []


def export_rig(blend, out, *, armature="Armature", skin="Skin", decimate=0.12, blender_exe: str | None = None,
               say=print) -> Path:
    """``.blend`` -> ``.rig.npz`` in a separate process (Blender's Python or a Blender executable)."""
    script = Path(__file__).resolve().parents[2] / "bodymodel" / "_blend_export.py"
    args = [str(blend), str(out), armature, skin, str(decimate)]
    if blender_exe:
        cmd = [blender_exe, "-b", "--python", str(script), "--", *args]
    else:
        probe = subprocess.run([sys.executable, "-c", "import bpy"], capture_output=True, text=True)
        if probe.returncode != 0:
            raise RuntimeError("reading .blend files needs Blender's Python module in this environment "
                               "(pip install bpy — it needs a matching Python version) or the path to a "
                               "Blender executable")
        cmd = [sys.executable, str(script), *args]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONNOUSERSITE": "0"})
    for line in (r.stdout or "").splitlines():
        if line.startswith("rig:"):
            say(line)
    if r.returncode != 0 or not Path(out).is_file():
        raise RuntimeError("rig export failed:\n" + (r.stderr or r.stdout)[-2000:])
    return Path(out)


# --------------------------------------------------------------------------- #
class RigImportForm(OperationForm):
    title = "Import rig (.blend)"
    section_id = "bodymodel_rig"
    description = (
        "A rigged body model: from Blender (the armature and the mesh it deforms, e.g. the HBP mouse, "
        "updated for current Blender; the fitter expects the mouse model's bone names: Spine1-17, "
        "Neck1-8, Head, Pelvis, Tail1-23, Femur.L/.R …), or the built-in virtual rodent, a rat "
        "(DeepMind, dm_control) with its own joint limits. Saved as arrays in sources/bodymodel/.")

    def build(self) -> None:
        self._root = _root(self.config_path)
        f = QFormLayout()
        self.source = QComboBox(self)
        self.source.addItem("Blender file (.blend)", "blend")
        self.source.addItem("Virtual rodent (rat, built in)", "rodent")
        f.addRow("Model", self.source)
        self.blend = QLineEdit(self)
        f.addRow(".blend file", _browse_row(self, self.blend, folder=False, caption="Blender file",
                                            filt="Blender (*.blend)"))
        self.armature = QLineEdit("Armature", self)
        self.skin = QLineEdit("Skin", self)
        f.addRow("Armature object", self.armature)
        f.addRow("Skin mesh object", self.skin)
        self.name = QLineEdit("mouse", self)
        f.addRow("Rig name", self.name)
        self.girth = QDoubleSpinBox(self)
        self.girth.setRange(0.8, 2.0)
        self.girth.setDecimals(2)
        self.girth.setSingleStep(0.05)
        self.girth.setValue(1.0)
        self.girth.setEnabled(False)
        self.girth.setToolTip("Virtual rodent: the trunk widened sideways (1: as modelled, a slim rat standing; "
                              "a heavy rat sitting or curled up covers more floor, e.g. 1.3)")
        f.addRow("Girth", self.girth)
        self.blender = QLineEdit(self)
        self.blender.setPlaceholderText("optional: Blender executable (if bpy is not installed)")
        f.addRow("Blender", _browse_row(self, self.blender, folder=False, caption="Blender executable"))
        out = rig_dir(self._root)
        f.addRow("Saved to", QLabel(str(out) if out else "sources/bodymodel/ of the open project", self))
        self.body_layout.addLayout(f)
        self._blend_rows = [self.blend, self.armature, self.skin, self.blender]
        self.source.currentIndexChanged.connect(self._source_changed)

    def _source_changed(self) -> None:
        rodent = self.source.currentData() == "rodent"
        for w in self._blend_rows:
            w.setEnabled(not rodent)
        self.girth.setEnabled(rodent)
        if rodent and self.name.text().strip() in ("", "mouse"):
            self.name.setText("rat")
        elif not rodent and self.name.text().strip() == "rat":
            self.name.setText("mouse")

    def collect_args(self) -> dict:
        if not self._root:
            raise ValueError("open a project first")
        if self.source.currentData() == "rodent":
            name = self.name.text().strip() or "rat"
            return dict(blend=None, out=str(rig_dir(self._root) / f"{name}.rig.npz"), armature=None, skin=None,
                        blender_exe=None, girth=self.girth.value())
        b = self.blend.text().strip()
        if not b or not Path(b).is_file():
            raise ValueError("choose a .blend file")
        name = self.name.text().strip() or Path(b).stem
        return dict(blend=b, out=str(rig_dir(self._root) / f"{name}.rig.npz"), armature=self.armature.text().strip(),
                    skin=self.skin.text().strip(), blender_exe=self.blender.text().strip() or None, girth=1.0)

    def target(self, *, blend, out, armature, skin, blender_exe, girth=1.0) -> None:
        if blend is None:
            from mufasa.bodymodel.rodent_rig import export_rodent_rig
            p = export_rodent_rig(out, girth=girth)
        else:
            p = export_rig(blend, out, armature=armature, skin=skin, blender_exe=blender_exe)
        print(f"rig -> {p}")


# --------------------------------------------------------------------------- #
class BodyFitForm(OperationForm):
    title = "Fit body model"
    section_id = "bodymodel_fit"
    description = (
        "Fit the rig to a top-down video: the animal's silhouette (darker than the median background) "
        "and its 2-D labels, with every joint within its limits (spine, neck, head, tail; rearing about "
        "the hind paws; nothing below the floor). Over time: forwards, each frame from the previous one, "
        "then backwards towards both neighbours. Optionally renders the model over the video.")

    def build(self) -> None:
        self._root = _root(self.config_path)
        f = QFormLayout()
        self.rig = QComboBox(self)
        for r in list_rigs(self._root):
            self.rig.addItem(r.name.removesuffix(".rig.npz"), str(r))
        f.addRow("Rig", self.rig)
        self.video = QComboBox(self)
        for v in _videos(self._root, self.config_path):
            self.video.addItem(v.name, str(v))
        f.addRow("Video", self.video)
        self.labels = QComboBox(self)
        if self._root:
            self.labels.addItem("Imported labels (sources/pose)", str(self._root / "sources" / "pose"))
        for r in reversed(_kalman_runs(self._root)):
            self.labels.addItem(f"Kalman v2 {r.name}", str(r))
        f.addRow("2-D labels", self.labels)
        row = QHBoxLayout()
        self.start, self.stop, self.step = QSpinBox(self), QSpinBox(self), QSpinBox(self)
        for s, v, tip in ((self.start, 0, "first frame"), (self.stop, 0, "last frame + 1 (0: to the end)"),
                          (self.step, 1, "fit every n-th frame (the overlay interpolates between)")):
            s.setRange(0, 10 ** 8)
            s.setValue(v)
            s.setToolTip(tip)
        self.step.setRange(1, 10000)
        for lab, s in (("from", self.start), ("to", self.stop), ("every", self.step)):
            row.addWidget(QLabel(lab, self))
            row.addWidget(s)
        row.addStretch()
        f.addRow("Frames", row)
        self.refine = QCheckBox("Forwards and backwards (use the neighbouring frames)", self)
        self.refine.setChecked(True)
        f.addRow("Over time", self.refine)
        self.temporal = QDoubleSpinBox(self)
        self.temporal.setRange(0, 10)
        self.temporal.setDecimals(2)
        self.temporal.setValue(0.5)
        self.temporal.setToolTip("How strongly each frame is pulled towards its neighbours (0: frames fitted alone)")
        f.addRow("Smoothness", self.temporal)
        self.scale = QDoubleSpinBox(self)
        self.scale.setRange(0, 1000)
        self.scale.setDecimals(3)
        self.scale.setSpecialValueText("auto")
        self.scale.setToolTip("px per model unit (mm for the mouse); auto: from the labels and a few frames, "
                              "or from a 3-D run (below)")
        f.addRow("Scale", self.scale)
        self.scale_from = QComboBox(self)
        self.scale_from.addItem("estimated from the video (labels and silhouette)", None)
        for r in reversed(_pose3d_runs(self._root)):
            self.scale_from.addItem(f"3-D run {r.name} (camera and the animal's size)", str(r))
        self.scale_from.setToolTip(
            "With 'auto' scale: take it from a single-camera 3-D run of this video — the camera's pixels per "
            "mm at the back's height, times the animal's size against the model. Width and scale then stay "
            "apart (the silhouette alone confuses a wider animal with a bigger one).")
        f.addRow("Scale from", self.scale_from)
        self.polarity = QComboBox(self)
        for lab, v in (("auto (from the labels)", "auto"), ("darker than the background", "dark"),
                       ("lighter than the background", "light")):
            self.polarity.addItem(lab, v)
        self.polarity.setToolTip("How to find the animal's silhouette: dark fur (e.g. under IR) or a white "
                                 "animal on darker ground; auto compares the pixels under the back labels "
                                 "with the background")
        f.addRow("Animal", self.polarity)
        self.workers = QSpinBox(self)
        self.workers.setRange(1, max(1, os.cpu_count() or 1))
        self.workers.setValue(max(1, min(8, (os.cpu_count() or 2) - 1)))
        f.addRow("Workers", self.workers)
        self.render = QCheckBox("Render the model over the video", self)
        self.sbs = QCheckBox("side by side with the original", self)
        r2 = QHBoxLayout()
        r2.addWidget(self.render)
        r2.addWidget(self.sbs)
        r2.addStretch()
        f.addRow("Overlay", r2)
        hint = QLabel("<i>Each fitted frame takes a few seconds of CPU; use “every” for long stretches "
                      "and more workers. Results: derived/bodymodel/&lt;run&gt;/.</i>", self)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        f.addRow(hint)
        self.body_layout.addLayout(f)

    def collect_args(self) -> dict:
        from mufasa.pose3d.pipeline import _match, pose_files
        if not self._root:
            raise ValueError("open a project first")
        if not self.rig.currentData():
            raise ValueError("no rig yet: import one from a .blend first (Import rig)")
        if not self.video.currentData():
            raise ValueError("no videos in sources/videos/")
        video = Path(self.video.currentData())
        stem = video.stem
        pf = _match(stem, pose_files(self.labels.currentData()))
        if pf is None:
            raise ValueError(f"no 2-D labels for {stem} in {self.labels.currentData()}")
        stop = self.stop.value() or None
        if stop is not None and stop <= self.start.value():
            raise ValueError("'to' must be after 'from'")
        scale_from = None
        if not self.scale.value() and self.scale_from.currentData():
            p3 = Path(self.scale_from.currentData()) / f"{stem}_3d.parquet"
            if not p3.is_file():
                raise ValueError(f"3-D run {Path(self.scale_from.currentData()).name} has no {p3.name}")
            scale_from = str(p3)
        return dict(rig=self.rig.currentData(), video=str(video), pose=str(pf), start=self.start.value(),
                    stop=stop, step=self.step.value(), refine=self.refine.isChecked(),
                    temporal=self.temporal.value() if self.refine.isChecked() else 0.0,
                    scale=self.scale.value() or None, workers=self.workers.value(),
                    polarity=self.polarity.currentData(), scale_from=scale_from,
                    render=self.render.isChecked(), side_by_side=self.sbs.isChecked())

    def target(self, **kw) -> None:
        out = run_body_fit(self._root, **kw)
        self._last_run_id = out.name


def run_body_fit(root, *, rig, video, pose, start=0, stop=None, step=1, refine=True, temporal=0.5, scale=None,
                 workers=1, render=False, side_by_side=False, evals_warm=500, evals_first=2500,
                 chunk=300, polarity="auto", scale_from=None) -> Path:
    from dataclasses import asdict

    from mufasa.bodymodel.fit import FitConfig, MaskConfig, RunConfig, fit_video
    from mufasa.project_layout import ProjectPaths, generate_run_id, write_run_toml
    rid = generate_run_id()
    out = ProjectPaths(root).stage_run_dir("bodymodel", run_id=rid)
    fcfg = FitConfig(scale=scale, temporal=temporal, refine=refine, evals_warm=evals_warm, evals_first=evals_first)
    run = RunConfig(start=start, stop=stop, step=step, workers=workers, chunk=chunk, fit=fcfg,
                    mask=MaskConfig(polarity=polarity), scale_from=scale_from)
    stem = Path(video).stem
    fit_path = fit_video(rig, video, pose, run, out / f"{stem}_fit.parquet")
    params = {"rig": str(rig), "start": start, "stop": stop, "step": step, "workers": workers,
              "polarity": run.mask.polarity, "scale_from": scale_from,
              "fit": {k: v for k, v in asdict(fcfg).items() if k != "weights"}}
    results = {"fit": str(fit_path)}
    if render:
        from mufasa.bodymodel.render import OverlayStyle, render_overlay_video
        ov = render_overlay_video(fit_path, out / f"{stem}_overlay.mp4",
                                  style=OverlayStyle(side_by_side=side_by_side))
        results["overlay"] = str(ov)
    write_run_toml(out / "run.toml", stage="bodymodel", run_id=out.name, inputs=[str(video), str(pose), str(rig)] + ([str(scale_from)] if scale_from else []),
                   params=params, results=results)
    return out


__all__ = ["BodyFitForm", "RigImportForm", "export_rig", "list_rigs", "run_body_fit"]
