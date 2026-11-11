"""Smoke test for patch 122ka — segment-length and spatial speed priors.

User request: "We have the joint prior. Are we also doing a speed and
intermarker distance prior? Could we do it spatially weighted, by such vars
as radius from center or some other reference point?"

* fit_kinematic_priors_v2: per segment the fitted length and its spread;
  the root's typical (RMS) speed overall and on a grid over the arena
  (a spatial map: tight where the animal rests, loose where it runs);
* tuning "length_prior" / "speed_prior" (0 = off, the default; else the
  prior SD as a multiple of the observed spread / local typical speed) add
  pseudo-observations: length -> typical length, root velocity -> rest;
* stored with the joint prior in the model; in the advanced view.

Measured on K7 (frames 1500-7500, 30 head windows of 45 frames hidden on
purpose): hidden-head error median 15.0 -> 15.3 px, p90 38.8 -> 37.1, p99
93.5 -> 71.0 (with the joint prior as well: 14.6 / 35.9 / 71.5); with all
markers hidden in 30-frame windows: unchanged (6.4 / 26 / 56). The smoother
already interpolates across gaps; the priors mainly cut the worst cases.
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
    {"name": "neck", "parent": "back", "markers": ["head_back"]},
    {"name": "tail_1", "parent": "back", "markers": ["tail_V6"]},
]
layout = K.layout_from_segments(SPEC)
names = list(layout.marker_names)
rng = np.random.default_rng(0)
T = 6000
# rests in the left half (x < 300), runs in the right half
x = np.empty(T)
y = np.full(T, 200.0)
x[:3000] = 150 + np.cumsum(rng.normal(0, 0.05, 3000))          # resting
x[3000:] = 450 + 120 * np.sin(np.arange(3000) / 15.0)          # running
th = np.zeros(T)
u = np.stack([np.cos(th), np.sin(th)], 1)
c = np.stack([x, y], 1)
along = {"back_T4": 20, "back_T8": 8, "back_L2": -6, "back_L6": -20,
         "head_back": 35, "tail_V6": -38}
pos = np.stack([c + along[m] * u for m in names], 1) + rng.normal(0, 0.1, (T, len(names), 2))
lik = np.full((T, len(names)), 0.95)
fl = K.fit_body_lengths(pos, lik, layout, names, 0.5, dt=1 / 30)
kin = K.fit_kinematic_priors_v2([(pos, lik)], layout, names, fl, 30.0, 0.5, grid_cells=8)

check("length prior: every non-root segment, the fitted lengths",
      set(kin["length_mu"]) == set(layout.non_root_topo_order)
      and all(np.isclose(kin["length_mu"][s], fl.segment_lengths[s]) for s in kin["length_mu"]))
check("length spread from the IQR, floored", all(v >= 0.5 for v in kin["length_sd"].values()))
jp = K.JointPriorV2(mu={}, kappa={}, **kin)
rest, run = jp.local_speed(170, 200), jp.local_speed(470, 200)
check(f"spatial map: slow where it rests ({rest:.0f} px/s), fast where it runs "
      f"({run:.0f} px/s)", run > 5 * rest and rest >= 5.0)
check("outside the map -> overall typical speed",
      jp.local_speed(-1e4, -1e4) == jp.speed_rms and jp.local_speed(np.nan, 0) == jp.speed_rms)

# ---------------------------------------------------------------- rows
state = K.initial_state_from_data(pos, lik, layout, names, fl, 0.5)
ri = K._pack_state_layout_indices(layout)["__root__"]
lay_off = dataclasses.replace(layout, tuning={"length_prior": 0.0, "speed_prior": 0.0})
z0, _, _, _ = K._build_constraint_observations(state, lay_off, 0.05, joint_prior=jp)
check("tuning off (0): no prior rows", z0.shape[0] == 1 + len(layout.non_root_topo_order))
lay = dataclasses.replace(layout, tuning={"length_prior": 2.0, "speed_prior": 1.0})
st = state.copy()
st[ri["x"]], st[ri["y"]] = 170.0, 200.0
z, h, H, R = K._build_constraint_observations(st, lay, 0.05, joint_prior=jp)
n0 = z0.shape[0]
nl = len(jp.length_mu)
check("length rows: target = typical length, SD = 2 × spread",
      z.shape[0] == n0 + nl + 2
      and np.allclose(sorted(z[n0:n0 + nl]), sorted(jp.length_mu.values()))
      and np.allclose(sorted(R[n0:n0 + nl]),
                      sorted((2.0 * v) ** 2 for v in jp.length_sd.values())))
check("speed rows: root velocity -> 0, per-axis variance (local speed)²/2",
      np.allclose(z[-2:], 0) and np.allclose(R[-2:], rest ** 2 / 2)
      and H[-2, ri["vx"]] == 1 and H[-1, ri["vy"]] == 1)
st[ri["x"]] = 470.0
R_run = K._build_constraint_observations(st, lay, 0.05, joint_prior=jp)[3]
check("…looser where the animal runs (spatial weighting)", R_run[-1] > 20 * R[-1])
check("no fitted priors (old model): tuning on adds nothing",
      K._build_constraint_observations(st, lay, 0.05,
                                       joint_prior=K.JointPriorV2(mu={}, kappa={}))[0].shape[0]
      == n0)

# ---------------------------------------------------------------- tuning / io
check("0 allowed (= off) for the priors",
      K.parse_tuning(["length_prior=0", "speed_prior=0"]) == {"length_prior": 0.0,
                                                               "speed_prior": 0.0}
      and K.TUNING_DEFAULTS["length_prior"] == 1 and K.TUNING_DEFAULTS["speed_prior"] == 1)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "m.npz")
    params = K.fit_initial_params_v2(pos, lik, layout, names, fl, 30.0, 0.5)
    K.save_model_v2(p, lay, fl, params, 30.0, 0.5, perspective=None, joint_prior=jp)
    back = K.load_model_v2(p)
    jb = back[-1]
    check("kinematic priors saved with the model and restored",
          jb is not None and jb.length_mu == jp.length_mu
          and np.isclose(jb.speed_rms, jp.speed_rms)
          and np.allclose(jb.speed_grid, jp.speed_grid, equal_nan=True)
          and jb.local_speed(170, 200) == rest and back[0].tuning == lay.tuning)

# ---------------------------------------------------------------- end to end
import contextlib  # noqa: E402
import io  # noqa: E402

import pandas as pd  # noqa: E402

tmp = Path(tempfile.mkdtemp())
(tmp / "in").mkdir()
lik2 = lik.copy()
lik2[3500:3560, :] = 0.0                                   # all markers lost 2 s
pd.DataFrame({**{f"{m}_x": pos[:, i, 0] for i, m in enumerate(names)},
              **{f"{m}_y": pos[:, i, 1] for i, m in enumerate(names)},
              **{f"{m}_likelihood": lik2[:, i] for i, m in enumerate(names)}}
             ).to_parquet(tmp / "in" / "s.parquet")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    K.smooth_pose_v2(pose_input=str(tmp / "in"), output_dir=str(tmp / "o"), layout=lay,
                     em_max_iter=2, n_workers=1, fps=30.0, verbose=True,
                     save_model=str(tmp / "o" / "m.npz"))
out = pd.read_parquet(next((tmp / "o").glob("*.parquet")))
check("pipeline fits and reports the priors", "kinematic priors:" in buf.getvalue()
      and "spatial map" in buf.getvalue())
check("…and runs through a dropout with them",
      np.isfinite(out[[f"{m}_x" for m in names]].to_numpy()).all())
check("…priors stored in the saved model", K.load_model_v2(str(tmp / "o" / "m.npz"))[-1]
      .speed_grid is not None)

# ---------------------------------------------------------------- GUI
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from mufasa.ui_qt.forms.pose_cleanup import KalmanV2SmoothingForm  # noqa: E402

F = KalmanV2SmoothingForm(config_path=None)
check("advanced view: length and speed prior spins, 1 by default (122kd), 0 = off",
      F.tuning_spins["length_prior"].value() == 1 and F.tuning_spins["speed_prior"].value() == 1
      and F.tuning_spins["speed_prior"].minimum() == 0)
F.tuning_spins["length_prior"].setValue(0)
F.tuning_spins["speed_prior"].setValue(2)
check("set → passed as tuning", F.tuning() == {"length_prior": 0.0, "speed_prior": 2.0})

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122ka_kinematic_priors: {len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
