"""Smoke test for patch 122kb — the Kalman v2 panel remembers its settings
and the last model.

User request: "can you have the kalman v2 smoothing panel save and load the
last used model and its parameters, so I don't have to set everything each
time?"

* every option of the panel (mode, run options, EM settings, model
  extensions, advanced fine tuning, input folder) is written to
  logs/measures/kalman_v2_settings.toml when a run starts, and restored when
  the panel is built; the output folder keeps its fresh run id;
* the model that run trains (or loads) becomes the "last model": it is put
  into "Load saved model" next time (and that mode is kept if it was used);
* "Load saved model" shows what the chosen model was trained with, and
  "Train a new model with these settings" copies them into the training
  options.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import tomllib
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.data_processors import kalman_pose_smoother_v2 as K  # noqa: E402
from mufasa.ui_qt.forms.pose_cleanup import KalmanV2SmoothingForm  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="kp", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
pose_in = tmp / "poses"
pose_in.mkdir()

F = KalmanV2SmoothingForm(config_path=cfg)
check("settings file lives in the project's logs/measures",
      F.settings_path().parts[-3:] == ("logs", "measures", "kalman_v2_settings.toml"))
check("fresh project: defaults, training mode, no model", F.mode_train.isChecked()
      and F.load_model_path.text() == "" and not F.settings_path().exists())
out0 = F.output_dir.text()
F.input_dir.setText(str(pose_in))
F.fps.setValue(25.0)
F.lik_thr.setValue(0.6)
F.em_iter.setValue(7)
F.em_aggregation.setCurrentText("median")
F.with_drift.setChecked(True)
F.const_accel.setText("back, head")
F.high_angular.setText("tail_3")
F.root_q_ceiling.setValue(20.0)
F.joint_prior.setChecked(True)
F.adv_toggle.setChecked(True)
F.tuning_spins["length_q_scale"].setValue(20)
F.tuning_spins["length_q_ceiling"].setValue(200)
F.tuning_spins["outlier_gate"].setValue(25)
F.device.setCurrentText("auto")
args = F.collect_args()
model_path = args["save_model_path"]
with open(F.settings_path(), "rb") as fh:
    saved = tomllib.load(fh)
check("starting a run saves the settings",
      saved["fps"] == 25.0 and saved["em_iter"] == 7 and saved["with_drift"] is True
      and saved["tuning"]["length_q_scale"] == 20 and saved["mode"] == "train")
check("…and the model it will train as the last model", saved["last_model"] == model_path)

# the run writes the model (simulate with a real model file)
SPEC = [{"name": "back", "markers": ["a", "b"]}]
lay = K.layout_from_segments(SPEC, with_drift=True)
lay = dataclasses.replace(lay, const_accel_segments=["back"],
                          tuning={"length_q_scale": 20.0, "outlier_gate": 25.0})
rng = np.random.default_rng(0)
pos = np.stack([np.c_[100 + np.arange(300) * .1, np.full(300, 100.)],
                np.c_[120 + np.arange(300) * .1, np.full(300, 100.)]], 1) + rng.normal(0, .3, (300, 2, 2))
lik = np.full((300, 2), .95)
fl = K.fit_body_lengths(pos, lik, lay, ["a", "b"], 0.5, dt=1 / 25)
params = K.fit_initial_params_v2(pos, lik, lay, ["a", "b"], fl, 25.0, 0.6)
Path(model_path).parent.mkdir(parents=True, exist_ok=True)
K.save_model_v2(model_path, lay, fl, params, 25.0, 0.6, perspective=None)

G = KalmanV2SmoothingForm(config_path=cfg)
check("next session: run options restored",
      G.fps.value() == 25.0 and G.lik_thr.value() == 0.6 and G.em_iter.value() == 7
      and G.em_aggregation.currentText() == "median" and G.device.currentText() == "auto")
check("next session: model extensions restored",
      G.with_drift.isChecked() and G.const_accel.text() == "back, head"
      and G.high_angular.text() == "tail_3" and G.root_q_ceiling.value() == 20.0
      and G.joint_prior.isChecked())
check("next session: fine tuning restored (advanced view open)",
      G.tuning() == {"length_q_scale": 20.0, "length_q_ceiling": 200.0, "outlier_gate": 25.0}
      and G.adv_toggle.isChecked())
check("input folder restored; output keeps a fresh run folder",
      G.input_dir.text() == str(pose_in) and G.output_dir.text() != ""
      and G.output_dir.text() != out0)
check("the last trained model is ready under 'Load saved model'",
      G.load_model_path.text() == model_path)
check("mode stays 'Train' when training was used last", G.mode_train.isChecked())
check("model summary shows what it was trained with",
      "25 fps" in G.model_info.text() and "per-marker drift" in G.model_info.text()
      and "const-accel: back" in G.model_info.text()
      and "length_q_scale=20" in G.model_info.text()
      and G.use_model_settings_btn.isEnabled())

# use the model: load mode is remembered
G.mode_load.setChecked(True)
a2 = G.collect_args()
H = KalmanV2SmoothingForm(config_path=cfg)
check("'Load saved model' remembered, with the model",
      a2["mode"] == "load" and H.mode_load.isChecked()
      and H.load_model_path.text() == model_path)

# copy a model's settings into training
H.with_drift.setChecked(False)
H.const_accel.setText("")
H.reset_tuning()
H.fps.setValue(30.0)
ok = H.apply_model_settings(model_path)
check("'Train a new model with these settings' copies the model's options",
      ok and H.mode_train.isChecked() and H.with_drift.isChecked()
      and H.const_accel.text() == "back" and H.fps.value() == 25.0
      and H.lik_thr.value() == 0.6
      and H.tuning() == {"length_q_scale": 20.0, "outlier_gate": 25.0})
H.load_model_path.setText(str(tmp / "missing.npz"))
check("missing model file: said so, button off",
      "not found" in H.model_info.text() and not H.use_model_settings_btn.isEnabled())

# a vanished model is not offered
os.remove(model_path)
J = KalmanV2SmoothingForm(config_path=cfg)
check("a deleted model is not put back", J.load_model_path.text() == ""
      and J.mode_train.isChecked())
F.settings_path().write_text("not = [toml")
L = KalmanV2SmoothingForm(config_path=cfg)
check("unreadable settings file: defaults, no crash",
      L.em_iter.value() == 20 and L.mode_train.isChecked() and L.tuning() == {})

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122kb_kalman_panel_remembers: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
