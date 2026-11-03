"""Smoke test for patch 122ig — root orientation from the trunk; root-noise ceiling.

Two findings from fitting the user's strohA-xw data:

1. The root's initial orientation and the initial estimate of its
   orientation noise (q_root_ori) were read from the legacy rig's marker
   names ``back1``/``back3``. Any other rig started at identity (up to 180 deg
   off the frame its root marker offsets are fitted in) with q_root_ori at the
   default 1.0 — and the M-step ceiling is 10x the initial value, so the
   fitted q_root_ori pinned at exactly 10. Both now come from the root
   cluster's own body axis (_root_cluster_body_axis, the same axis the root
   offsets are fitted in) via _root_axis_per_frame.

2. All whole-body motion goes through the root's constant-velocity model, so
   on bursty movement EM pins q_root_pos / q_root_ori at the 10x ceiling
   within a few iterations. The ceiling factor is now a layout setting
   (root_q_ceiling_factor, default 10 = prior behaviour) exposed as
   --root-q-ceiling and as "Root noise ceiling" in the Kalman v2 form. The
   absolute hard caps still apply.

Runs for real: the smoother's initial-state / initial-param functions on
synthetic data, its median M-step with fabricated per-session statistics, and
the Kalman v2 form offscreen.
"""
from __future__ import annotations

import os
import sys
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

import dataclasses  # noqa: E402

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


def synth(turn_sd, T=900, seed=0):
    """Trunk markers strung along an axis that rotates as a random walk."""
    rng = np.random.default_rng(seed)
    th = 0.7 + np.cumsum(rng.normal(0, turn_sd, T))
    cx = 300 + np.cumsum(rng.normal(0, 0.5, T))
    cy = 200 + np.cumsum(rng.normal(0, 0.5, T))
    u = np.stack([np.cos(th), np.sin(th)], 1)
    along = {"back_T4": 20, "back_T8": 8, "back_L2": -6, "back_L6": -20,
             "head_mid": 38, "tail_V6": -40}
    pos = np.zeros((T, len(NAMES), 2))
    for i, n in enumerate(NAMES):
        pos[:, i] = np.stack([cx, cy], 1) + along[n] * u
        pos[:, i] += rng.normal(0, 0.3, (T, 2))
    return pos, np.full((T, len(NAMES)), 0.95), th


layout = K.layout_from_segments(SPEC)
check("rig has no legacy back1/back3 names",
      "back1" not in layout.marker_names and "back3" not in layout.marker_names)

# ------------------------------------------------- 1. root orientation ---
pos, lik, th = synth(0.02)
idx = {n: i for i, n in enumerate(NAMES)}
axis = K._root_axis_per_frame(pos, lik, layout, idx, 0.5)
check("root axis is found for a renamed rig", axis is not None)
true_u = np.stack([np.cos(th), np.sin(th)], 1)
dots = np.abs(np.sum(axis * true_u, 1))
check("root axis follows the trunk direction every frame",
      np.nanmin(dots) > 0.99)

fl = K.fit_body_lengths(pos, lik, layout, NAMES, likelihood_threshold=0.5)
st = K.initial_state_from_data(pos, lik, layout, NAMES, fl, 0.5)
c, s = st[4], st[5]
check("initial root orientation is the trunk axis, not identity",
      abs(c * np.cos(th[0]) + s * np.sin(th[0])) > 0.99
      and not (abs(c - 1.0) < 1e-9 and abs(s) < 1e-9))
check("initial root orientation matches the axis root offsets are fitted in",
      abs(c * axis[0, 0] + s * axis[0, 1]) > 0.999)

p_slow = K.fit_initial_params_v2(pos, lik, layout, NAMES, fl, 30.0, 0.5)
pos_f, lik_f, _ = synth(0.08, seed=1)
fl_f = K.fit_body_lengths(pos_f, lik_f, layout, NAMES, likelihood_threshold=0.5)
p_fast = K.fit_initial_params_v2(pos_f, lik_f, layout, NAMES, fl_f, 30.0, 0.5)
check("q_root_ori is estimated from the data (not the 1.0 default)",
      abs(p_slow.q_root_ori - 1.0) > 1e-6)
check("faster turning gives a larger initial q_root_ori",
      p_fast.q_root_ori > 3 * p_slow.q_root_ori)

# ------------------------------------------------- 2. root-noise ceiling ---
check("default root ceiling is the prior 10x",
      layout.root_q_ceiling_factor == 10.0
      and K._root_q_ceiling_factor(layout) == K._M_STEP_Q_CEILING_FACTOR)
try:
    dataclasses.replace(layout, root_q_ceiling_factor=0.5)
    check("a ceiling below 1x is rejected", False)
except ValueError:
    check("a ceiling below 1x is rejected", True)

init = K.NoiseParamsV2.default(layout, q_root_pos=200.0, q_root_ori=0.5)


def aggregate(lay, raw_pos, raw_ori):
    fit = K._PerSessionFitV2.empty(lay)
    fit.q_root_pos, fit.q_root_ori = raw_pos, raw_ori
    fit.q_seg_ori = dict(init.q_seg_ori)
    fit.q_length = dict(init.q_length)
    fit.sigma_marker = dict(init.sigma_marker)
    fit.n_pairs = 1000
    fit.n_obs_per_marker = {m: 1000 for m in lay.marker_names}
    return K.finalize_m_step_v2_from_per_session(
        [fit], lay, prev_params=init, initial_params=init, damping=0.0,
    )


out10 = aggregate(layout, 200.0 * 100, 0.5 * 100)
check("default: root noise clipped at 10x its initial value",
      np.isclose(out10.q_root_pos, 2000.0)
      and np.isclose(out10.q_root_ori, 5.0))
lay50 = dataclasses.replace(layout, root_q_ceiling_factor=50.0)
out50 = aggregate(lay50, 200.0 * 100, 0.5 * 100)
check("ceiling 50x: root noise may reach 50x its initial value",
      np.isclose(out50.q_root_pos, 10000.0)
      and np.isclose(out50.q_root_ori, 25.0))
out_mid = aggregate(lay50, 200.0 * 20, 0.5 * 20)
check("below the ceiling the raw value passes through",
      np.isclose(out_mid.q_root_pos, 4000.0)
      and np.isclose(out_mid.q_root_ori, 10.0))
lay_huge = dataclasses.replace(layout, root_q_ceiling_factor=1000.0)
out_h = aggregate(lay_huge, 1e9, 1e6)
check("the absolute safety caps still apply",
      np.isclose(out_h.q_root_pos, 50000.0)
      and np.isclose(out_h.q_root_ori, 50.0))
check("the ceiling only touches the root (segment noise unchanged)",
      out50.q_seg_ori == out10.q_seg_ori
      and out50.q_length == out10.q_length)

src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check("pooled M-step uses the layout's root ceiling too",
      src.count("initial_params.q_root_pos * _root_q_ceiling_factor(layout)") == 2
      and src.count("initial_params.q_root_ori * _root_q_ceiling_factor(layout)") == 2)

# ------------------------------------------------------------- CLI / GUI ---
import contextlib  # noqa: E402
import io  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
    K.main(["--help"])
check("CLI offers --root-q-ceiling", "--root-q-ceiling" in buf.getvalue())
check("CLI applies it to the layout and reports it ignored with a model",
      "root_q_ceiling_factor=float(args.root_q_ceiling)" in src
      and '("--root-q-ceiling", args.root_q_ceiling != 10.0)' in src)

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from mufasa.ui_qt.forms.pose_cleanup import KalmanV2SmoothingForm  # noqa: E402

form = KalmanV2SmoothingForm()
check("Kalman v2 form has a Root noise ceiling control defaulting to 10x",
      hasattr(form, "root_q_ceiling") and form.root_q_ceiling.value() == 10.0)
gui = (REPO / "mufasa/ui_qt/forms/pose_cleanup.py").read_text()
check("form passes the ceiling and applies it to the layout",
      '"root_q_ceiling": float(self.root_q_ceiling.value())' in gui
      and 'replacements["root_q_ceiling_factor"]' in gui)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ig_root_axis_and_ceiling: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
