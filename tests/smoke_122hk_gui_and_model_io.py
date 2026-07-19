"""Smoke test for patch 122hk — GUI wiring + model save/load correctness.

Two things:

1. Save/load correctness (the important one). Training a model with the joint
   prior (patch 122hf) then loading it to smooth other sessions must carry the
   prior through the saved file. The save call in smooth_pose_v2 omitted
   joint_prior, so save_model_v2 received its default None, wrote
   has_joint_prior=False, and the prior was silently lost — a loaded model
   would smooth new sessions WITHOUT the prior, reintroducing the covariance
   divergence (tail tip, rear back markers flying off) that the prior prevents.
   Also verifies tau_accel (patch 122hg) round-trips.

2. GUI wiring. The Kalman v2 form exposed with-drift / orientation-drift /
   const-accel but not the three newer smoother options: the joint prior
   (122hf), accel-tau (122hg), and high-angular-noise segments (122hj). This
   checks the widgets exist and are threaded through collect_args into the
   smooth_pose_v2 train calls and the layout.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys
import tempfile
import types

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402,F401

import mufasa.data_processors.kalman_pose_smoother_v2 as K  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


# ---------------------------------------------------------------- #
# 1. Model save/load carries the joint prior and tau_accel
# ---------------------------------------------------------------- #
layout = K.layout_from_segments([
    {"name": "back", "markers": ["b1", "b2"]},
    {"name": "tail_1", "parent": "back", "markers": ["t1"]},
])
params = K.NoiseParamsV2(
    sigma_marker={"b1": 3.0, "b2": 3.0, "t1": 3.0},
    q_root_pos=100.0, q_root_ori=10.0,
    q_seg_ori={"tail_1": 2.0}, q_length={"tail_1": 5.0},
    tau_accel=0.5,
)
fitted = K.FittedLengths(
    segment_lengths={"tail_1": 15.0}, segment_length_iqr={"tail_1": 2.0},
    marker_offsets={"b1": (0.0, 0.0), "b2": (10.0, 0.0), "t1": (0.0, 3.14)},
    marker_r_drift={}, marker_q_drift={},
)
jp = K.JointPriorV2(
    mu={"tail_1": 3.14}, kappa={"tail_1": 5.5}, n_obs={"tail_1": 1000},
)

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "m.npz")
    K.save_model_v2(path, layout, fitted, params, 30.0, 0.5,
                    perspective=None, joint_prior=jp)
    out = K.load_model_v2(path)
    layout2, fitted2, params2, fps2, lt2, persp2, jp2 = out

    check("load returns a 7-tuple including the joint prior", len(out) == 7)
    check("tau_accel round-trips through save/load",
          params2.tau_accel == 0.5)
    check("joint prior survives save/load (the fix)", jp2 is not None)
    check("joint prior kappa preserved",
          jp2 is not None and abs(jp2.kappa.get("tail_1", 0.0) - 5.5) < 1e-9)
    check("joint prior mu preserved",
          jp2 is not None and abs(jp2.mu.get("tail_1", 0.0) - 3.14) < 1e-6)
    check("joint prior n_obs preserved",
          jp2 is not None and jp2.n_obs.get("tail_1") == 1000)

# The bug demonstrated: the OLD save path (no joint_prior arg) loses it.
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "m2.npz")
    K.save_model_v2(path, layout, fitted, params, 30.0, 0.5, perspective=None)
    _, _, _, _, _, _, jp3 = K.load_model_v2(path)
    check("without joint_prior arg the prior is absent (bug reproduced)",
          jp3 is None)

# The save CALL in smooth_pose_v2 must pass joint_prior (not default it).
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
# Find the save_model_v2 call inside smooth_pose_v2 and check it names
# joint_prior.
tree = ast.parse(src)
save_calls_with_jp = 0
save_calls_total = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "save_model_v2":
        save_calls_total += 1
        kw = {k.arg for k in node.keywords}
        if "joint_prior" in kw:
            save_calls_with_jp += 1
check("the save_model_v2 call passes joint_prior explicitly",
      save_calls_total >= 1 and save_calls_with_jp == save_calls_total)

# The final smoothing pool must thread joint_prior (so a loaded prior is used).
check("final-smooth task args include joint_prior",
      "(sess_idx, params, perspective, joint_prior, device)" in src)
check("_pool_final_smooth applies joint_prior to the filter",
      "joint_prior=joint_prior" in src)

# ---------------------------------------------------------------- #
# 2. GUI wiring
# ---------------------------------------------------------------- #
gui = (REPO / "mufasa/ui_qt/forms/pose_cleanup.py").read_text()
gtree = ast.parse(gui)
form_src = ""
for c in ast.walk(gtree):
    if isinstance(c, ast.ClassDef) and c.name == "KalmanV2SmoothingForm":
        form_src = ast.get_source_segment(gui, c) or ""
        break
check("KalmanV2SmoothingForm exists", bool(form_src))

# widgets
check("build() adds a joint-prior checkbox",
      "self.joint_prior = QCheckBox" in form_src)
check("build() adds an accel-tau control",
      "self.accel_tau = QDoubleSpinBox" in form_src)
check("build() adds a high-angular-noise field",
      "self.high_angular = QLineEdit" in form_src)

# collect_args keys
check("collect_args emits enable_joint_prior",
      '"enable_joint_prior":' in form_src)
check("collect_args emits accel_tau",
      '"accel_tau":' in form_src)
check("collect_args emits high_angular_noise_segments",
      '"high_angular_noise_segments":' in form_src)

# target wiring
check("target adds high_angular_noise_segments to the layout",
      'replacements["high_angular_noise_segments"]' in form_src)
check("both train smooth_pose_v2 calls pass enable_joint_prior",
      form_src.count('enable_joint_prior=kwargs["enable_joint_prior"]') == 2)
check("both train smooth_pose_v2 calls pass accel_tau",
      form_src.count('accel_tau=kwargs["accel_tau"]') == 2)

# accel_tau: blank/zero maps to None (random walk), not 0.0 (invalid)
check("accel_tau of 0 maps to None (random walk), not an invalid 0.0",
      "if self.accel_tau.value() > 0.0 else None" in form_src)

# ---------------------------------------------------------------- #
n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hk_gui_and_model_io: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
