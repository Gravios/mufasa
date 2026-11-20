"""
mufasa.ui_qt.forms.gait
=======================

Patch 122kw: *Gait (paws from a body model)* on the 3-D tab — paw placement,
weight distribution and gait parameters from a 3-D run, with the virtual
rodent as the physics model (:mod:`mufasa.pose3d.gait`); 122kx: bounds (CatWalk by
speed, or TactileArena's fixed ones). Writes
``derived/gait/<run>/<video>_gait.parquet`` (per frame) and
``<video>_steps.csv`` (per stance), with the summary in ``run.toml``.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QLabel

from mufasa.ui_qt.forms.pose3d import _pose3d_runs, _root
from mufasa.ui_qt.workbench import OperationForm


class GaitForm(OperationForm):
    title = "Gait (paws from a body model)"
    section_id = "pose3d_gait"
    description = (
        "Paws, weight distribution and gait from 3-D body markers. The virtual rodent's segment "
        "masses give the centre of mass; each paw stays planted while its leg can reach it and lifts "
        "when its stride clock says so (stance, swing, stride and the paws' order as uninjured rats "
        "walk on a CatWalk, by speed), the body staying balanced over the other paws for steps off "
        "the clock (centre of mass and momentum). Writes "
        "derived/gait/<run>/: per frame (paw positions, contact, load, centre of mass, ZMP) and per "
        "stance (stride, duty factor).")

    def build(self) -> None:
        self._root = _root(self.config_path)
        f = QFormLayout()
        self.run = QComboBox(self)
        for r in reversed(_pose3d_runs(self._root)):
            self.run.addItem(r.name, str(r))
        f.addRow("3-D run", self.run)
        self.mass = QDoubleSpinBox(self)
        self.mass.setRange(0, 5000)
        self.mass.setDecimals(0)
        self.mass.setSuffix(" g")
        self.mass.setSpecialValueText("from the size")
        self.mass.setToolTip("The animal's weighed mass (the loads in grams-force). 'From the size': the virtual "
                             "rodent's 338 g times the size cubed — only a rough guess")
        f.addRow("Body mass", self.mass)
        self.fps = QDoubleSpinBox(self)
        self.fps.setRange(0, 2000)
        self.fps.setDecimals(1)
        self.fps.setSpecialValueText("from the run")
        f.addRow("Frame rate", self.fps)
        self.bounds = QComboBox(self)
        self.bounds.addItem("CatWalk rats, by speed (Neckel et al. 2013)", "catwalk")
        self.bounds.addItem("TactileArena, fixed (13–25 cm/s)", "tactilearena")
        self.bounds.setToolTip("Stance, swing, stride, duty factor and base of support. CatWalk: 108 uninjured "
                               "rats, 13–71 cm/s; TactileArena: one rat's demo recording")
        f.addRow("Bounds", self.bounds)
        hint = QLabel("<i>View the result in the 3-D viewer: View &amp; models → Load… → "
                      "&lt;video&gt;_gait.parquet.</i>", self)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        f.addRow(hint)
        self.body_layout.addLayout(f)

    def collect_args(self) -> dict:
        if not self._root:
            raise ValueError("open a project first")
        if not self.run.currentData():
            raise ValueError("no 3-D run (run 3-D inference first)")
        return dict(run=self.run.currentData(), mass_g=self.mass.value() or None, fps=self.fps.value() or None,
                    bounds=self.bounds.currentData())

    def target(self, *, run, mass_g=None, fps=None, bounds="catwalk") -> None:
        out = run_gait(self._root, run, mass_g=mass_g, fps=fps, bounds=bounds)
        self._last_run_id = out.name


def run_gait(root, run, *, mass_g=None, fps=None, bounds="catwalk", say=print) -> Path:
    from mufasa.pose3d.gait import bounds_preset, gait_from_pose3d, write_gait
    from mufasa.project_layout import ProjectPaths, generate_run_id, write_run_toml
    run = Path(run)
    files = sorted(run.glob("*_3d.parquet"))
    if not files:
        raise ValueError(f"{run.name}: no *_3d.parquet files")
    out = ProjectPaths(root).stage_run_dir("gait", run_id=generate_run_id())
    results = {}
    for p in files:
        stem = p.name.removesuffix("_3d.parquet")
        res = gait_from_pose3d(p, fps=fps, mass_g=mass_g, bounds=bounds_preset(bounds))
        g, s = write_gait(res, out / f"{stem}_gait.parquet", {"pose3d": str(p)})
        sm = res.summary
        say(f"{stem}: {sm['steps']} stances, size {sm['size_vs_model']}x the model, mass {sm['mass_g']} g, "
            f"load share {sm['load_share']}")
        results[stem] = {"gait": str(g), "steps": str(s),
                         **{k: v for k, v in sm.items() if k not in ("bounds", "support", "load_share")}}
    write_run_toml(out / "run.toml", stage="gait", run_id=out.name, inputs=[str(p) for p in files],
                   params={"mass_g": mass_g, "fps": fps, "bounds": bounds, "source_run": run.name}, results=results)
    return out


__all__ = ["GaitForm", "run_gait"]
