"""Smoke test for patch 122he — cap iteration 0; report, don't speculate.

Two findings, from a user run that died with "validation hook triggered at
iteration 0".

1. Every q hard cap in this module was applied only inside
   ``finalize_m_step_v2``. They guarded iterations 1..N and left iteration 0
   running on whatever ``fit_initial_params_v2`` returned. The caps encode a
   numerical stability boundary for the EKF, and the EKF does not care which
   iteration it is on. Patch 121c's own comment says the absolute caps exist
   to backstop "cases where the initial q is already large" — it saw the
   initial fit going large, capped the M-step's use of it, and never capped
   the value itself.

   This is not a corner case: on ordinary synthetic mouse data the initial fit
   returns q_root_pos ≈ 78,000-86,000 against a documented boundary of 50,000.

2. The divergence message listed three candidate causes and told the reader to
   "inspect the early frames of x_smooth" — an array they have no access to.
   It now reports what can be measured: NaN fraction, first non-finite frame,
   which parameters sit on their cap, and whether const-accel is active.

NOTE ON SCOPE: the user's divergence has NOT been reproduced in-sandbox (3 GB
RAM caps synthetic sessions well below their 54k frames). These tests pin the
structural hole and the diagnostics, not a reproduction.
"""
from __future__ import annotations

import dataclasses as _dc
import pathlib
import sys
import types

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from mufasa.data_processors.kalman_pose_smoother_v2 import (  # noqa: E402
    _M_STEP_Q_JERK_ROOT_ORI_HARD_CAP,
    _M_STEP_Q_JERK_ROOT_POS_HARD_CAP,
    _M_STEP_Q_JERK_SEG_ORI_HARD_CAP,
    _M_STEP_Q_ROOT_ORI_HARD_CAP,
    _M_STEP_Q_ROOT_POS_HARD_CAP,
    NoiseParamsV2,
    SmoothResultV2,
    _cap_params_to_hard_limits,
    _validate_trajectory_v2,
    layout_from_segments,
)

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

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


plain = layout_from_segments(SPEC)
ca = _dc.replace(plain, with_drift=True, const_accel_segments=["back", "head"])

# ---------------------------------------------------------------- #
# 1. _cap_params_to_hard_limits
# ---------------------------------------------------------------- #
inside = NoiseParamsV2.default(plain)
same, clamped = _cap_params_to_hard_limits(inside, plain)
check("params already inside the caps are returned untouched",
      same is inside and clamped == {})

over = _dc.replace(
    NoiseParamsV2.default(plain),
    q_root_pos=_M_STEP_Q_ROOT_POS_HARD_CAP * 2,
    q_root_ori=_M_STEP_Q_ROOT_ORI_HARD_CAP * 2,
)
capped, clamped = _cap_params_to_hard_limits(over, plain)
check("q_root_pos over the cap is clamped",
      capped.q_root_pos == _M_STEP_Q_ROOT_POS_HARD_CAP)
check("q_root_ori over the cap is clamped",
      capped.q_root_ori == _M_STEP_Q_ROOT_ORI_HARD_CAP)
check("the clamp report names both fields",
      set(clamped) == {"q_root_pos", "q_root_ori"})
check("the clamp report carries before and after",
      clamped["q_root_pos"] == (_M_STEP_Q_ROOT_POS_HARD_CAP * 2,
                                _M_STEP_Q_ROOT_POS_HARD_CAP))
check("capping does not mutate the input",
      over.q_root_pos == _M_STEP_Q_ROOT_POS_HARD_CAP * 2)
check("capping is idempotent",
      _cap_params_to_hard_limits(capped, plain)[1] == {})

# Jerk caps apply only when const-accel is active.
jerky = _dc.replace(
    NoiseParamsV2.default(ca),
    q_jerk_root_pos=_M_STEP_Q_JERK_ROOT_POS_HARD_CAP * 3,
    q_jerk_root_ori=_M_STEP_Q_JERK_ROOT_ORI_HARD_CAP * 3,
    q_jerk_seg_ori={s: _M_STEP_Q_JERK_SEG_ORI_HARD_CAP * 3
                    for s in ca.non_root_topo_order},
)
jc, jclamped = _cap_params_to_hard_limits(jerky, ca)
check("q_jerk_root_pos is clamped when const-accel is on",
      jc.q_jerk_root_pos == _M_STEP_Q_JERK_ROOT_POS_HARD_CAP)
check("q_jerk_root_ori is clamped when const-accel is on",
      jc.q_jerk_root_ori == _M_STEP_Q_JERK_ROOT_ORI_HARD_CAP)
check("per-segment q_jerk_seg_ori is clamped",
      all(v == _M_STEP_Q_JERK_SEG_ORI_HARD_CAP
          for v in jc.q_jerk_seg_ori.values()))
check("the jerk clamp is reported",
      any(k.startswith("q_jerk_seg_ori[") for k in jclamped))

no_ca = _dc.replace(NoiseParamsV2.default(plain),
                    q_jerk_root_pos=_M_STEP_Q_JERK_ROOT_POS_HARD_CAP * 3)
nc, nclamped = _cap_params_to_hard_limits(no_ca, plain)
check("jerk caps are NOT applied when const-accel is off "
      "(the fields are unused there)",
      "q_jerk_root_pos" not in nclamped)

# ---------------------------------------------------------------- #
# 2. iteration 0 is capped
# ---------------------------------------------------------------- #
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
em = src[src.index("def fit_noise_params_em_v2("):]
em = em[:em.index("\ndef ")]


def _ordered(*markers: str, hay: str = em) -> bool:
    """True iff every marker is present and they appear in this order. Never
    raises — a missing marker is a FAIL to report, not a traceback."""
    last = -1
    for m in markers:
        if m not in hay:
            return False
        i = hay.index(m)
        if i <= last:
            return False
        last = i
    return True


check("fit_noise_params_em_v2 caps the initial params",
      "_cap_params_to_hard_limits(" in em)
check("...immediately after fitting them, before any E-step",
      _ordered("fit_initial_params_v2(", "_cap_params_to_hard_limits(",
               "fit_warm_start_sigma_v2("))

# ---------------------------------------------------------------- #
# 3. the divergence report
# ---------------------------------------------------------------- #
def diverge_msg(layout, params):
    """Drive the hook with a smoother result that has gone non-finite from
    frame 5 on — the shape of a real divergence."""
    T, K = 40, layout.n_markers
    x_smooth = np.full((T, layout.state_dim), np.nan)
    x_smooth[:5] = 0.0
    smooth = SmoothResultV2(
        x_smooth=x_smooth,
        P_smooth=np.zeros((T, layout.state_dim, layout.state_dim)),
        P_lag_one=np.zeros((T, layout.state_dim, layout.state_dim)),
    )
    pos = np.zeros((T, K, 2))
    lik = np.ones((T, K))
    try:
        _validate_trajectory_v2(smooth, pos, lik, layout, params, 0, 0.7)
    except RuntimeError as e:
        return str(e)
    except Exception as e:            # noqa: BLE001
        return f"__WRONG_EXC__ {type(e).__name__}: {e}"
    return "__NO_RAISE__"


msg = diverge_msg(ca, NoiseParamsV2.default(ca))
check("divergence still raises RuntimeError", not msg.startswith("__"))
check("it reports the NaN fraction", "NaN (" in msg or "% )" in msg or "NaN" in msg)
check("it reports the first non-finite frame", "first non-finite frame" in msg)
check("it no longer tells the user to inspect x_smooth",
      "x_smooth" not in msg)
check("it no longer offers a list of three guesses",
      "likely causes" not in msg)
check("it names const-accel when const-accel is on",
      "const_accel_segments" in msg and "--const-accel-segments" in msg)
check("...and explains the tau^5 vs tau^3 cost",
      "τ⁵" in msg and "τ³" in msg)

pinned = _dc.replace(NoiseParamsV2.default(ca),
                     q_root_pos=_M_STEP_Q_ROOT_POS_HARD_CAP)
msg_pinned = diverge_msg(ca, pinned)
check("it flags parameters pinned at their cap",
      "pinned at their stability cap" in msg_pinned
      and "q_root_pos" in msg_pinned)
check("...and says hitting the cap is a diagnostic, not success",
      "not a success" in msg_pinned)

msg_plain = diverge_msg(plain, NoiseParamsV2.default(plain))
check("const-accel advice is withheld when const-accel is off",
      "--const-accel-segments" not in msg_plain)
check("...and the actual q values are reported instead",
      "within caps" in msg_plain)

# ---------------------------------------------------------------- #
n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122he_iteration0_caps: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
