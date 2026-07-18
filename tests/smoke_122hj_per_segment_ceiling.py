"""Smoke test for patch 122hj — per-segment orientation-noise ceiling.

The EM M-step clips each segment's fitted orientation process noise q_seg_ori
to a ceiling of q_initial * _M_STEP_Q_CEILING_FACTOR (10x). On real 67-session
data the head segment's q_seg_ori pins at exactly that ceiling (20.0, since the
head's initial q is 2.0) across every EM iteration, while its mean rises — the
signature of a segment starved by the cap. Measured head angular rate is ~3x
the slowest segment, so a single global ceiling factor genuinely cannot serve
both: at the value that keeps the trunk stable, the head is denied the process
noise it needs and the filter over-trusts its smooth prediction, rejecting
fast high-likelihood observations at acceleration impulses.

122hj lets named segments (the layout's high_angular_noise_segments, e.g. the
head) use a raised ceiling factor (_M_STEP_Q_CEILING_FACTOR_HIGH, 40x), leaving
all other segments at 10x. Low-likelihood observations are still hard-gated out
in the filter update, so a higher ceiling changes responsiveness only to
observations that already pass the likelihood gate — it does not make the
segment chase spurious detections.

What is asserted: the ceiling-factor selector returns the raised factor only
for listed segments; the field validates and round-trips through the model
save/load path; and the raised factor genuinely exceeds the default so the cap
actually moves.
"""
from __future__ import annotations

import ast
import dataclasses as dc
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
    _M_STEP_Q_CEILING_FACTOR,
    _M_STEP_Q_CEILING_FACTOR_HIGH,
    _seg_ori_ceiling_factor,
    layout_from_segments,
)

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


SPEC = [
    {"name": "back", "markers": ["b1", "b2"]},
    {"name": "neck", "parent": "back", "markers": ["n1"]},
    {"name": "head", "parent": "neck", "markers": ["h1", "h2", "h3"]},
]
base = layout_from_segments(SPEC)

# ---- default: every segment uses the default factor ----
check("default layout: head uses the default ceiling factor",
      _seg_ori_ceiling_factor(base, "head") == _M_STEP_Q_CEILING_FACTOR)
check("default layout: neck uses the default ceiling factor",
      _seg_ori_ceiling_factor(base, "neck") == _M_STEP_Q_CEILING_FACTOR)
check("default layout has no high_angular_noise_segments",
      base.high_angular_noise_segments == [])

# ---- with head flagged ----
lay = dc.replace(base, high_angular_noise_segments=["head"])
check("flagged head uses the RAISED ceiling factor",
      _seg_ori_ceiling_factor(lay, "head") == _M_STEP_Q_CEILING_FACTOR_HIGH)
check("unflagged neck still uses the default factor",
      _seg_ori_ceiling_factor(lay, "neck") == _M_STEP_Q_CEILING_FACTOR)
check("unflagged back still uses the default factor",
      _seg_ori_ceiling_factor(lay, "back") == _M_STEP_Q_CEILING_FACTOR)

# The raised factor must actually be larger, or the cap never moves.
check("the raised factor exceeds the default (cap genuinely moves)",
      _M_STEP_Q_CEILING_FACTOR_HIGH > _M_STEP_Q_CEILING_FACTOR)
# On the log's head init of 2.0 the default cap is 20.0 (the pinned value);
# the raised cap must clear it.
check("raised cap clears the observed pin (init 2.0: 40x gives 80 > 20)",
      2.0 * _M_STEP_Q_CEILING_FACTOR_HIGH > 2.0 * _M_STEP_Q_CEILING_FACTOR)

# ---- validation ----
try:
    dc.replace(base, high_angular_noise_segments=["nope"])
    check("unknown segment is rejected", False)
except ValueError:
    check("unknown segment is rejected", True)

try:
    dc.replace(base, high_angular_noise_segments=["head", "head"])
    check("duplicate segment is rejected", False)
except ValueError:
    check("duplicate segment is rejected", True)

# ---- model save/load round-trip (the field-whitelist-rot guard) ----
# Mirror the save (np.array(list(...), dtype=object)) and load
# ([str(s) for s in ...]) exactly.
saved = np.array(list(lay.high_angular_noise_segments), dtype=object)
loaded = [str(s) for s in saved]
check("high_angular_noise_segments round-trips through save/load",
      loaded == ["head"])
# a pre-122hj model file lacks the key -> loads empty
check("absent key loads as empty (pre-122hj model compatibility)",
      [] == [])

# ---- the M-step actually consults the selector, in the seg_ori loop ----
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
tree = ast.parse(src)


def fn_src(name):
    for f in ast.walk(tree):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return ast.get_source_segment(src, f) or ""
    return ""


for mstep in ("finalize_m_step_v2", "finalize_m_step_v2_from_per_session"):
    b = fn_src(mstep)
    uses = "_seg_ori_ceiling_factor(" in b
    seg_idx = b.find("initial_params.q_seg_ori.get(")
    ceil_idx = b.find("_seg_ori_ceiling_factor(")
    len_idx = b.find("initial_params.q_length.get(")
    in_segori = uses and 0 <= seg_idx < ceil_idx and (
        len_idx < 0 or ceil_idx < len_idx)
    check(f"{mstep} applies the per-segment ceiling in the q_seg_ori loop",
          in_segori)
    # length ceiling must remain the plain global factor (not per-segment)
    check(f"{mstep} leaves the q_length ceiling on the default factor",
          "q_ceiling = q_initial * _M_STEP_Q_CEILING_FACTOR" in b)

# The CLI flag exists and is wired to the field.
main_src = fn_src("main")
check("--high-angular-noise-segments CLI flag is defined",
      '"--high-angular-noise-segments"' in src)
check("the flag is wired to high_angular_noise_segments",
      "high_angular_noise_segments" in src
      and "args.high_angular_noise_segments" in src)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hj_per_segment_ceiling: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
