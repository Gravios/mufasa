"""Smoke test for patch 122jz — innovation gate against confident-but-wrong
detections.

User report: "When there are longer gaps the body model twists and turns too
much. Can we apply a longer look ahead to coordinate a better estimate of the
trajectories? The longer breaks often occur during long moments of
immobility." — with an example around frame 3767 of
K7_cam_20251124_1510_A10-0-051_640x360-triplet.

What the data showed: the RTS smoother already uses the whole future (a full
backward pass), but while the head is hidden FDLC still returns head
markers at p ≈ 0.5-0.7 hundreds of px away (another spot / camera tile).
They pass the likelihood threshold and yank the head (and through it the
body) around. The robust gate tests each marker's 2-D innovation against its
own predicted covariance and, beyond a χ² threshold (tuning
"outlier_gate"; 0 = off; 25 recommended), inflates that detection's noise by
d²/gate, so it barely pulls; a genuine fast move is still followed. (A hard
gate that dropped detections lost lock on the tail.) On the K7 clip
(frames 1500-7500, length ×50): head flips >60°/3 frames 7 -> 3, tail
frames >30 px off a confident detection 82 -> 31 (V32), the 3720 / 3767
head twists gone.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
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


# ---------------------------------------------------------------- unit: gate
P = np.eye(4) * 4.0
H = np.zeros((6, 4))
H[0, 0] = H[1, 1] = H[2, 0] = H[3, 1] = H[4, 2] = H[5, 3] = 1.0
z = np.array([1.0, 0.0, 0.5, 0.5, 60.0, 0.0])          # marker 3 is 60 px off
h = np.zeros(6)
R = np.ones(6)
zz, hh, HH, RR, n, nw = K._gate_marker_observations(z, h, H, R, 3, P, 25.0)
d2 = 60.0 ** 2 / 5.0
check("robust gate keeps every row but inflates the far marker's noise by d²/gate",
      n == 3 and nw == 1 and np.array_equal(zz, z) and np.allclose(RR[:4], 1.0)
      and np.allclose(RR[4:], d2 / 25.0))
K_full = P @ H.T @ np.linalg.inv(H @ P @ H.T + np.diag(RR))
pull = (K_full @ (z - h))[2]
check(f"…so it pulls the state only ~as far as a detection on the gate would "
      f"({pull:.1f} px instead of {60 * 4 / 5:.0f})", pull < 12)
out = K._gate_marker_observations(z, h, H, R, 3, P * 1e4, 25.0)
check("…and not at all when the state is very uncertain (long gap)", out[5] == 0
      and np.array_equal(out[3], R))
check("input arrays not modified", np.array_equal(R, np.ones(6)))
check("default 25 since 122kd (0 = off)",
      K.TUNING_DEFAULTS["outlier_gate"] == 25.0 and "outlier_gate" in K.TUNING_HELP)
check("0 is accepted for outlier_gate (off), not for other knobs",
      K.parse_tuning(["outlier_gate=0"]) == {"outlier_gate": 0.0})
try:
    K.parse_tuning(["sigma_ceiling=0"])
    bad = False
except ValueError:
    bad = True
check("…other knobs still must be > 0", bad)

# ---------------------------------------------------------------- synthetic
SPEC = [
    {"name": "back", "markers": ["back_T8", "back_T4", "back_L2", "back_L6",
                                 "hip_left", "hip_right"]},
    {"name": "back_rear", "parent": "back", "markers": ["back_V2"]},
    {"name": "neck", "parent": "back", "markers": ["head_back"]},
    {"name": "head", "parent": "neck",
     "markers": ["head_mid", "head_nose", "head_left", "head_right"]},
    {"name": "tail_1", "parent": "back_rear", "markers": ["tail_V6"]},
    {"name": "tail_2", "parent": "tail_1", "markers": ["tail_V18"]},
    {"name": "tail_3", "parent": "tail_2", "markers": ["tail_V32"]},
]
layout = K.layout_from_segments(SPEC)

# the 122hf generator (a moving rat with a trailing tail)
def _smooth_noise(T, scale, tau, rg):
    w = rg.standard_normal(T) * scale
    out = np.zeros(T)
    a = math.exp(-1.0 / tau)
    for t in range(1, T):
        out[t] = a * out[t - 1] + (1 - a) * w[t] * tau
    return out


def make(T=1800, seed=3, tail_swing=0.18, other_swing=None):
    rg = np.random.default_rng(seed)
    x = 300 + np.cumsum(_smooth_noise(T, 2.0, 25, rg))
    y = 300 + np.cumsum(_smooth_noise(T, 2.0, 25, rg))
    th_b = np.cumsum(_smooth_noise(T, 0.04, 20, rg))
    ang = lambda sc: _smooth_noise(T, sc, 12, rg)  # noqa: E731
    o = other_swing
    a_neck = ang(0.10 if o is None else o)
    a_head = ang(0.14 if o is None else o)
    a_rear = ang(0.08 if o is None else o)
    a1, a2, a3 = ang(tail_swing), ang(tail_swing), ang(tail_swing)
    R = lambda a: np.stack([np.stack([np.cos(a), -np.sin(a)], -1),   # noqa: E731
                            np.stack([np.sin(a), np.cos(a)], -1)], -2)
    P = np.stack([x, y], -1)
    Rb = R(th_b)
    put = lambda b, Rot, o: b + np.einsum('tij,j->ti', Rot, np.array(o))  # noqa: E731
    M = {}
    for nm, o in (("back_T4", (12, 0)), ("back_T8", (4, 0)),
                  ("back_L2", (-4, 0)), ("back_L6", (-12, 0)),
                  ("hip_left", (-14, 7)), ("hip_right", (-14, -7))):
        M[nm] = put(P, Rb, o)
    Rr = Rb @ R(a_rear)
    Pr = put(P, Rb, (-16, 0))
    M["back_V2"] = put(Pr, Rr, (-6, 0))
    Rn = Rb @ R(a_neck)
    Pn = put(P, Rb, (18, 0))
    M["head_back"] = put(Pn, Rn, (6, 0))
    Rh = Rn @ R(a_head)
    Ph = put(Pn, Rn, (6, 0))
    for nm, o in (("head_mid", (6, 0)), ("head_nose", (14, 0)),
                  ("head_left", (8, 5)), ("head_right", (8, -5))):
        M[nm] = put(Ph, Rh, o)
    R1_ = Rr @ R(a1)
    P1 = put(Pr, Rr, (-6, 0))
    M["tail_V6"] = put(P1, R1_, (-10, 0))
    R2_ = R1_ @ R(a2)
    P2 = put(P1, R1_, (-10, 0))
    M["tail_V18"] = put(P2, R2_, (-12, 0))
    R3_ = R2_ @ R(a3)
    P3 = put(P2, R2_, (-12, 0))
    M["tail_V32"] = put(P3, R3_, (-12, 0))
    names = list(layout.marker_names)
    pos = np.zeros((T, len(names), 2))
    for i, m in enumerate(names):
        pos[:, i, :] = M[m] + rg.standard_normal((T, 2)) * 1.2
    return pos, np.ones((T, len(names))), names



pos, lik, names = make()
pos = pos.copy()
lik = lik.copy() * 0.95
T = len(pos)
rng = np.random.default_rng(5)
# head hidden in bursts: FDLC still answers, confidently wrong (exercises the gate)
hi = [names.index(m) for m in ("head_back", "head_mid", "head_nose", "head_left",
                                "head_right")]
truth = pos.copy()
bad_frames = np.zeros(T, bool)
for s0 in (400, 900, 1400):
    bad_frames[s0:s0 + 90] = True
for k in hi:
    sel = bad_frames & (rng.random(T) < 0.5)
    # scattered over the frame, as in the K7 triplet video (other spots /
    # camera tiles), not a consistent second animal
    pos[sel, k] = rng.uniform([0, 0], [640, 360], (sel.sum(), 2))
    lik[sel, k] = 0.6
    lik[bad_frames & ~sel, k] = 0.1                       # the rest: honest low p
import tempfile  # noqa: E402

import pandas as pd  # noqa: E402

tmp = Path(tempfile.mkdtemp())
(tmp / "in").mkdir()
pd.DataFrame({**{f"{m}_x": pos[:, i, 0] for i, m in enumerate(names)},
              **{f"{m}_y": pos[:, i, 1] for i, m in enumerate(names)},
              **{f"{m}_likelihood": lik[:, i] for i, m in enumerate(names)}}
             ).to_parquet(tmp / "in" / "s.parquet")


def run(gate):
    """Full pipeline (EM-fitted noise) on the corrupted session."""
    lay = dataclasses.replace(layout, tuning={"outlier_gate": gate, "length_prior": 0.0,
                                              "speed_prior": 0.0})
    out = tmp / f"out_{gate:g}"
    K.smooth_pose_v2(pose_input=str(tmp / "in"), output_dir=str(out), layout=lay,
                     em_max_iter=4, n_workers=1, fps=30.0, likelihood_threshold=0.5)
    d = pd.read_parquet(next(out.glob("*.parquet")))
    return np.stack([d[[m + "_x", m + "_y"]].to_numpy() for m in names], 1)


import contextlib  # noqa: E402
import io  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    xy0 = run(0)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    xy1 = run(25.0)
log1 = buf.getvalue()
with contextlib.redirect_stdout(io.StringIO()):
    out_n = tmp / "out_plain"
    K.smooth_pose_v2(pose_input=str(tmp / "in"), output_dir=str(out_n),
                     layout=dataclasses.replace(layout, tuning={
                         "outlier_gate": 0.0, "length_prior": 0.0, "speed_prior": 0.0}),
                     em_max_iter=4, n_workers=1, fps=30.0, likelihood_threshold=0.5)
d_n = pd.read_parquet(next(out_n.glob("*.parquet")))
xy_n = np.stack([d_n[[m + "_x", m + "_y"]].to_numpy() for m in names], 1)
check("gate 0 = exactly the previous behaviour", np.allclose(xy0, xy_n, equal_nan=True))
check("with the gate the run log reports what was down-weighted",
      "outlier gate:" in log1 and "down-weighted" in log1)
check("gated run produces a full, finite result",
      xy1.shape == xy0.shape and np.isfinite(xy1).all())

# ---------------------------------------------------------------- GUI
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from mufasa.ui_qt.forms.pose_cleanup import KalmanV2SmoothingForm  # noqa: E402

F = KalmanV2SmoothingForm(config_path=None)
sp = F.tuning_spins.get("outlier_gate")
check("advanced view: Outlier gate spin, 25 by default (0 = off)",
      sp is not None and sp.value() == 25.0 and sp.minimum() == 0.0)
sp.setValue(0)
check("set → passed as tuning", F.tuning() == {"outlier_gate": 0.0})

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jz_kalman_outlier_gate: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
