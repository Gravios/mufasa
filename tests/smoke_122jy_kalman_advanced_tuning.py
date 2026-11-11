"""Smoke test for patch 122jy — Kalman v2 fine tuning, wired into an
advanced view of the Kalman page.

User request: "can you wire more of the fine tuning parameters into an
advanced view for the kalman filter page?" (after asking how to loosen the
distance constraints of the Kalman filter / kinematic model).

* BodyLayout.tuning (saved with the model, reaches the workers):
  drift_radius_cap, length_q_scale, length_q_ceiling, seg_q_ceiling,
  q_floor_factor, sigma_ceiling, constraint_sigma — defaults reproduce the
  previous constants; CLI --tune KEY=VALUE;
* the Kalman v2 form gets "Show advanced fine tuning": a spin box per knob,
  unit-norm constraints, strict validation, device, reset; the M-step
  aggregation combo now offers the smoother's real choices.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from mufasa.data_processors import kalman_pose_smoother_v2 as K  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


SPEC = [
    {"name": "back", "markers": ["back_T4", "back_T8", "back_L2", "back_L6"]},
    {"name": "head", "parent": "back", "markers": ["head_mid"]},
    {"name": "tail", "parent": "back", "markers": ["tail_V6"]},
]
NAMES = ["back_T4", "back_T8", "back_L2", "back_L6", "head_mid", "tail_V6"]


def synth(T=900, seed=0, wobble=6.0):
    rng = np.random.default_rng(seed)
    th = 0.7 + np.cumsum(rng.normal(0, 0.02, T))
    cx = 300 + np.cumsum(rng.normal(0, 0.5, T))
    cy = 200 + np.cumsum(rng.normal(0, 0.5, T))
    u = np.stack([np.cos(th), np.sin(th)], 1)
    along = {"back_T4": 20, "back_T8": 8, "back_L2": -6, "back_L6": -20,
             "head_mid": 38, "tail_V6": -40}
    pos = np.zeros((T, len(NAMES), 2))
    for i, n in enumerate(NAMES):
        pos[:, i] = np.stack([cx, cy], 1) + along[n] * u
        pos[:, i] += rng.normal(0, wobble if n == "back_L6" else 0.3, (T, 2))
    return pos, np.full((T, len(NAMES)), 0.95)


# ---------------------------------------------------------------- defaults
check("defaults reproduce the previous constants",
      {k: v for k, v in K.TUNING_DEFAULTS.items()
       if k not in ("outlier_gate", "length_prior", "speed_prior")}
      == {"drift_radius_cap": 20.0, "length_q_scale": 1.0,
          "length_q_ceiling": 10.0, "seg_q_ceiling": 10.0,
          "q_floor_factor": 0.1, "sigma_ceiling": 3.0,
          "constraint_sigma": 0.05}
      and K.TUNING_DEFAULTS.get("outlier_gate", 0.0) == 0.0
      and K.TUNING_DEFAULTS["seg_q_ceiling"] == K._M_STEP_Q_CEILING_FACTOR
      and K.TUNING_DEFAULTS["sigma_ceiling"] == K._M_STEP_SIGMA_CEILING_FACTOR
      and K.TUNING_DEFAULTS["q_floor_factor"] == K._M_STEP_Q_FLOOR_FACTOR
      and K.TUNING_DEFAULTS["constraint_sigma"] == K._DEFAULT_CONSTRAINT_SIGMA)
check("every knob documented", set(K.TUNING_HELP) == set(K.TUNING_DEFAULTS))
layout = K.layout_from_segments(SPEC)
check("a layout's tuning is empty by default (= defaults)",
      layout.tuning == {} and K._tune(layout, "drift_radius_cap") == 20.0)

# ---------------------------------------------------------------- validation
check("parse_tuning: repeated and comma-separated",
      K.parse_tuning(["drift_radius_cap=40", "length_q_scale=10,sigma_ceiling=5"])
      == {"drift_radius_cap": 40.0, "length_q_scale": 10.0, "sigma_ceiling": 5.0})
for bad in (["nope=1"], ["drift_radius_cap"], ["drift_radius_cap=x"],
            ["drift_radius_cap=0"]):
    try:
        K.parse_tuning(bad)
        ok = False
    except ValueError:
        ok = True
    check(f"parse_tuning rejects {bad[0]!r}", ok)
for bad in ({"nope": 1.0}, {"sigma_ceiling": -1.0}):
    try:
        dataclasses.replace(layout, tuning=bad)
        ok = False
    except ValueError:
        ok = True
    check(f"BodyLayout rejects tuning {bad}", ok)

# ---------------------------------------------------------------- initial params
pos, lik = synth()
fl = K.fit_body_lengths(pos, lik, layout, NAMES, likelihood_threshold=0.5)
lay_d = dataclasses.replace(layout, with_drift=True)
base = K.fit_initial_params_v2(pos, lik, lay_d, NAMES, fl, 30.0, 0.5)
loose = K.fit_initial_params_v2(
    pos, lik, dataclasses.replace(lay_d, tuning={"length_q_scale": 10.0,
                                                 "constraint_sigma": 0.2}),
    NAMES, fl, 30.0, 0.5)
check("length_q_scale multiplies every segment's length noise",
      all(np.isclose(loose.q_length[s], 10 * base.q_length[s]) for s in base.q_length))
check("constraint_sigma reaches the parameters",
      base.constraint_sigma == 0.05 and loose.constraint_sigma == 0.2)
fl_w = dataclasses.replace(fl, marker_r_drift={m: 60.0 for m in NAMES})
r20 = K.fit_initial_params_v2(pos, lik, lay_d, NAMES, fl_w, 30.0, 0.5).r_drift
r50 = K.fit_initial_params_v2(
    pos, lik, dataclasses.replace(lay_d, tuning={"drift_radius_cap": 50.0}),
    NAMES, fl_w, 30.0, 0.5).r_drift
check("drift radius capped at 20 px by default, at the tuned cap otherwise",
      max(r20.values()) == 20.0 and max(r50.values()) == 50.0)

# ---------------------------------------------------------------- M-step bounds
init = K.NoiseParamsV2.default(layout, q_root_pos=200.0, q_root_ori=0.5)


def aggregate(lay, scale):
    fit = K._PerSessionFitV2.empty(lay)
    fit.q_root_pos, fit.q_root_ori = init.q_root_pos, init.q_root_ori
    fit.q_seg_ori = {k: v * scale for k, v in init.q_seg_ori.items()}
    fit.q_length = {k: v * scale for k, v in init.q_length.items()}
    fit.sigma_marker = {k: v * scale for k, v in init.sigma_marker.items()}
    fit.n_pairs = 1000
    fit.n_obs_per_marker = {m: 1000 for m in lay.marker_names}
    return K.finalize_m_step_v2_from_per_session(
        [fit], lay, prev_params=init, initial_params=init, damping=0.0)


o_def = aggregate(layout, 1000.0)
o_tun = aggregate(dataclasses.replace(layout, tuning={
    "length_q_ceiling": 100.0, "seg_q_ceiling": 50.0, "sigma_ceiling": 8.0}), 1000.0)
s = "head"
check("default: length / angle / σ clipped at 10x / 10x / 3x",
      np.isclose(o_def.q_length[s], 10 * init.q_length[s])
      and np.isclose(o_def.q_seg_ori[s], 10 * init.q_seg_ori[s])
      and np.isclose(o_def.sigma_marker["head_mid"], 3 * init.sigma_marker["head_mid"]))
check("tuned ceilings let EM go to 100x / 50x / 8x",
      np.isclose(o_tun.q_length[s], 100 * init.q_length[s])
      and np.isclose(o_tun.q_seg_ori[s], 50 * init.q_seg_ori[s])
      and np.isclose(o_tun.sigma_marker["head_mid"], 8 * init.sigma_marker["head_mid"]))
o_lo = aggregate(dataclasses.replace(layout, tuning={"q_floor_factor": 0.01}), 1e-4)
o_lo_def = aggregate(layout, 1e-4)
check("floor factor tunable (0.1x default, 0.01x tuned)",
      np.isclose(o_lo_def.q_seg_ori[s], 0.1 * init.q_seg_ori[s])
      and np.isclose(o_lo.q_seg_ori[s], 0.01 * init.q_seg_ori[s]))
lay_h = dataclasses.replace(layout, high_angular_noise_segments=["head"],
                            tuning={"seg_q_ceiling": 80.0})
check("high-angular segments keep the larger of 40x and the tuned ceiling",
      K._seg_ori_ceiling_factor(lay_h, "head") == 80.0
      and K._seg_ori_ceiling_factor(
          dataclasses.replace(lay_h, tuning={}), "head") == 40.0)

# ---------------------------------------------------------------- model file
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "m.npz")
    lay_t = dataclasses.replace(layout, tuning={"drift_radius_cap": 35.0,
                                                "length_q_scale": 4.0})
    K.save_model_v2(path, lay_t, fl, base, 30.0, 0.5, perspective=None)
    lay_back = K.load_model_v2(path)[0]
    check("tuning saved with the model and restored on load",
          lay_back.tuning == {"drift_radius_cap": 35.0, "length_q_scale": 4.0})
    K.save_model_v2(path, layout, fl, base, 30.0, 0.5, perspective=None)
    check("untuned model loads with empty tuning", K.load_model_v2(path)[0].tuning == {})

# ---------------------------------------------------------------- CLI
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check("--tune flag defined, applied to the layout, ignored with --load-model",
      '"--tune", action="append"' in src and "parse_tuning(args.tune)" in src
      and '("--tune", bool(args.tune))' in src)

# ---------------------------------------------------------------- GUI
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from mufasa.ui_qt.forms.pose_cleanup import KalmanV2SmoothingForm  # noqa: E402

F = KalmanV2SmoothingForm(config_path=None)
check("advanced view hidden behind a toggle",
      not F.adv_toggle.isChecked() and F.adv_group.isHidden())
F.adv_toggle.setChecked(True)
check("toggle shows it", not F.adv_group.isHidden())
check("a spin box per knob, at the defaults",
      set(F.tuning_spins) == set(K.TUNING_DEFAULTS) and F.tuning() == {}
      and F.tuning_summary.text() == "all defaults")
F.tuning_spins["drift_radius_cap"].setValue(45)
F.tuning_spins["length_q_scale"].setValue(10)
check("only changed knobs reported",
      F.tuning() == {"drift_radius_cap": 45.0, "length_q_scale": 10.0}
      and "drift_radius_cap=45" in F.tuning_summary.text())
F.adv_toggle.setChecked(False)
check("collapsed view says it has changes", "(changed)" in F.adv_toggle.text())
F.apply_constraints.setChecked(False)
F.strict_validation.setChecked(True)
F.device.setCurrentText("auto")
F.reset_tuning()
check("reset restores every default", F.tuning() == {} and F.apply_constraints.isChecked()
      and not F.strict_validation.isChecked() and F.device.currentText() == "cpu")
check("M-step aggregation offers the smoother's choices",
      [F.em_aggregation.itemText(i) for i in range(F.em_aggregation.count())]
      == ["pooled", "median", "trimmed_mean"])
form_src = (REPO / "mufasa/ui_qt/forms/pose_cleanup.py").read_text()
check("collect_args emits tuning / constraints / strict validation / device",
      all(k in form_src for k in ('"tuning": self.tuning()', '"apply_constraints":',
                                  '"strict_validation":', '"device":')))
check("target puts tuning on the layout",
      'replacements["tuning"] = dict(kwargs["tuning"])' in form_src)
check("both training runs pass constraints / strict validation / device",
      form_src.count('apply_constraints=kwargs.get("apply_constraints", True)') == 2
      and form_src.count('enable_strict_validation=kwargs.get("strict_validation", False)')
      == 2 and form_src.count('device=kwargs.get("device", "cpu")') == 2)

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jy_kalman_advanced_tuning: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
