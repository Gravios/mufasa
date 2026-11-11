"""Smoke test for patch 122kd — confirmed settings become defaults; serial
and pooled runs apply the priors alike.

User: "It works very well" (outlier gate 25, segment-length prior 1, speed
prior 1, joint angle prior on), after asking earlier to make settings that
worked well the defaults.

* tuning defaults: outlier_gate 25, length_prior 1, speed_prior 1; the
  Kalman v2 panel's joint-prior box is ticked; the CLI's --joint-prior is on
  by default (--no-joint-prior turns it off);
* two bugs found on the way: the pooled EM E-step (n_workers > 1) did not
  pass the joint prior to the filter, and the serial final pass did not
  either — so serial and parallel runs fitted / smoothed differently
  whenever a joint (or kinematic) prior was on.
"""
from __future__ import annotations

import contextlib
import io
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
import pandas as pd  # noqa: E402

from mufasa.data_processors import kalman_pose_smoother_v2 as K  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


D = K.TUNING_DEFAULTS
check("defaults: outlier gate 25, length prior 1, speed prior 1",
      D["outlier_gate"] == 25.0 and D["length_prior"] == 1.0 and D["speed_prior"] == 1.0)
check("…and the 122kc lengths (50 / 500)",
      D["length_q_scale"] == 50.0 and D["length_q_ceiling"] == 500.0)
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check("CLI: --joint-prior on by default, --no-joint-prior to turn it off",
      '"--joint-prior", action=argparse.BooleanOptionalAction, default=True' in src)

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from mufasa.ui_qt.forms.pose_cleanup import KalmanV2SmoothingForm  # noqa: E402

F = KalmanV2SmoothingForm(config_path=None)
check("panel: joint prior ticked, prior / gate spins at the new defaults, "
      "nothing reported as changed",
      F.joint_prior.isChecked() and F.tuning_spins["outlier_gate"].value() == 25
      and F.tuning_spins["length_prior"].value() == 1
      and F.tuning_spins["speed_prior"].value() == 1 and F.tuning() == {})

# ---------------------------------------------------------------- pool parity
lay = K.standard_rat_layout()
tmp = Path(tempfile.mkdtemp())
(tmp / "in").mkdir()
for i in range(3):
    T = 150
    r = np.random.default_rng(8311 + i)
    rx = 100 + np.cumsum(r.normal(0, .5, T))
    ry = 100 + np.cumsum(r.normal(0, .5, T))
    cols = {}
    for m in lay.marker_names:
        ox, oy = r.uniform(-15, 15, 2)
        lk = np.full(T, .95)
        lk[r.random(T) < .05] = .1
        cols[m + "_x"] = rx + ox + r.normal(0, .3, T)
        cols[m + "_y"] = ry + oy + r.normal(0, .3, T)
        cols[m + "_likelihood"] = lk
    pd.DataFrame(cols).to_parquet(tmp / "in" / f"s{i}.parquet")
res = {}
for nw in (1, 2):
    with contextlib.redirect_stdout(io.StringIO()):
        res[nw] = K.smooth_pose_v2(
            pose_input=str(tmp / "in"), output_dir=str(tmp / f"o{nw}"), layout=lay,
            fps=30., likelihood_threshold=.7, em_max_iter=2, n_workers=nw,
            enable_joint_prior=True, verbose=False)
s1 = res[1]["params"].sigma_marker
s2 = res[2]["params"].sigma_marker
check("joint + kinematic priors on: serial and pooled EM fit the same model",
      all(abs(s1[m] - s2[m]) < 1e-6 for m in s1))
o1 = sorted((tmp / "o1").glob("*.parquet"))
o2 = sorted((tmp / "o2").glob("*.parquet"))
same = len(o1) == len(o2) == 3 and all(
    np.allclose(pd.read_parquet(a).filter(regex="_x$|_y$").to_numpy(),
                pd.read_parquet(b).filter(regex="_x$|_y$").to_numpy(), atol=1e-6)
    for a, b in zip(o1, o2))
check("…and smooth to the same positions", same)

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122kd_prior_defaults_and_pool_parity: {len(checks) - len(failed)}/"
      f"{len(checks)} checks passed")
sys.exit(1 if failed else 0)
