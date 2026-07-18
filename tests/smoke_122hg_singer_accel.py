"""Smoke test for patch 122hg — Singer (OU) acceleration.

What this is for
----------------

121d's const-accel block models acceleration as a random walk: build_F_v2's
comment says so outright, "ax_new = ax (random walk)", and the accel diagonal
is left at the 1.0 that np.eye(D) put there. Unobserved, acceleration then
never decays, so across a dropout the predictor's position variance grows as
tau^5 where plain constant-velocity grows as tau^3. That is what makes
const-accel diverge on data with markers occluded for seconds at a time.

122hg sets that diagonal to phi = exp(-dt/tau_accel), making acceleration an
Ornstein-Uhlenbeck process. It stays a tracked state — so fast motion is still
captured, which was the point of const-accel — but unobserved it relaxes to
zero and the extrapolation degrades gracefully CA -> CV -> CP.

The two assertions that matter are at opposite ends: tau_accel=None must
reproduce 121d bit-for-bit (nothing changes unless asked), and a finite
tau_accel must actually bend the growth curve from t^5 to t^3.
"""
from __future__ import annotations

import dataclasses as dc
import math
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

import ast  # noqa: E402

import numpy as np  # noqa: E402

from mufasa.data_processors.kalman_pose_smoother_v2 import (  # noqa: E402
    NoiseParamsV2,
    build_F_v2,
    build_Q_v2,
    layout_from_segments,
    load_model_v2,
    save_model_v2,
)

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


SPEC = [
    {"name": "back", "markers": ["back_T8", "back_T4", "back_L2", "back_L6",
                                 "hip_left", "hip_right"]},
    {"name": "neck", "parent": "back", "markers": ["head_back"]},
    {"name": "head", "parent": "neck",
     "markers": ["head_mid", "head_nose", "head_left", "head_right"]},
]
layout = dc.replace(layout_from_segments(SPEC),
                    const_accel_segments=["back", "head"])
DT = 1.0 / 30.0

p_rw = NoiseParamsV2.default(layout)                    # tau_accel=None
p_ou = dc.replace(p_rw, tau_accel=0.5)

check("tau_accel defaults to None (the 121d random walk)", p_rw.tau_accel is None)

# ---------------------------------------------------------------- #
# 1. tau_accel=None must be EXACTLY 121d
# ---------------------------------------------------------------- #
F_rw, Q_rw = build_F_v2(layout, DT, p_rw), build_Q_v2(layout, p_rw, DT)
F_no, Q_no = build_F_v2(layout, DT), build_Q_v2(layout, p_rw, DT)
check("F(tau_accel=None) == F(no params at all)", np.array_equal(F_rw, F_no))

p_inf = dc.replace(p_rw, tau_accel=1e9)
F_inf, Q_inf = build_F_v2(layout, DT, p_inf), build_Q_v2(layout, p_inf, DT)
check("F(None) == F(tau -> inf): the OU limit is the random walk",
      np.allclose(F_rw, F_inf, atol=1e-9))
check("Q(None) == Q(tau -> inf): q*T/2*(1-e^-2dt/T) -> q*dt",
      np.allclose(Q_rw, Q_inf, rtol=1e-6))

# The accel diagonal is the whole change.
ca = layout.slice_segment_const_accel("back")
check("random walk leaves the accel diagonal at exactly 1.0",
      all(F_rw[i, i] == 1.0 for i in range(ca.start, ca.start + 4)))
phi = math.exp(-DT / 0.5)
F_ou, Q_ou = build_F_v2(layout, DT, p_ou), build_Q_v2(layout, p_ou, DT)
check(f"OU sets it to phi = exp(-dt/tau) = {phi:.4f}",
      all(math.isclose(F_ou[i, i], phi, rel_tol=1e-9)
          for i in range(ca.start, ca.start + 4)))

ca_h = layout.slice_segment_const_accel("head")
check("non-root segments get phi too (2 dims, orientation accel only)",
      all(math.isclose(F_ou[i, i], phi, rel_tol=1e-9)
          for i in range(ca_h.start, ca_h.start + 2)))

# Everything OUTSIDE the accel blocks must be untouched.
mask = np.ones(layout.state_dim, dtype=bool)
for sname in layout.const_accel_segments:
    sl = layout.slice_segment_const_accel(sname)
    mask[sl.start:sl.stop] = False
# F alone dominates the growth exponent, so the exponent checks below pass
# even if Q is left at the random walk's q*dt. Assert Q's OU value directly,
# or an F/Q mismatch — a decaying F with undecayed noise — slips through.
qp = p_ou.q_jerk_root_pos
tau_a = 0.5
expect_ou = qp * (tau_a / 2.0) * (1.0 - math.exp(-2.0 * DT / tau_a))
expect_rw = qp * DT
check("Q's accel noise uses the OU form q*T/2*(1-e^-2dt/T), not q*dt",
      math.isclose(Q_ou[ca.start, ca.start], expect_ou, rel_tol=1e-9))
check("...which is genuinely different from the random walk's q*dt",
      not math.isclose(expect_ou, expect_rw, rel_tol=1e-3))
check("the random walk's Q accel noise is still exactly q*dt",
      math.isclose(Q_rw[ca.start, ca.start], expect_rw, rel_tol=1e-9))

check("the OU touches ONLY the accel blocks (pos/vel/ori/length unchanged)",
      np.array_equal(F_rw[np.ix_(mask, mask)], F_ou[np.ix_(mask, mask)]))

check("F stays marginally stable (spectral radius 1 for the random walk)",
      math.isclose(float(np.abs(np.linalg.eigvals(F_rw)).max()), 1.0, rel_tol=1e-6))
check("the OU cannot make F unstable (spectral radius <= 1)",
      float(np.abs(np.linalg.eigvals(F_ou)).max()) <= 1.0 + 1e-9)

try:
    build_F_v2(layout, DT, dc.replace(p_rw, tau_accel=0.0))
    check("tau_accel=0 is rejected", False)
except ValueError as e:
    check("tau_accel=0 is rejected with a clear message",
          "tau_accel" in str(e) and "> 0" in str(e))
try:
    build_F_v2(layout, DT, dc.replace(p_rw, tau_accel=-1.0))
    check("negative tau_accel is rejected", False)
except ValueError:
    check("negative tau_accel is rejected", True)

# ---------------------------------------------------------------- #
# 2. the growth curve — the entire point
# ---------------------------------------------------------------- #
def growth(F, Q, secs):
    P = np.zeros((layout.state_dim, layout.state_dim))
    for _ in range(int(secs / DT)):
        P = F @ P @ F.T + Q
    return P[0, 0]


def exponent(F, Q):
    return math.log(growth(F, Q, 20.0) / growth(F, Q, 5.0)) / math.log(4.0)


k_rw, k_ou = exponent(F_rw, Q_rw), exponent(F_ou, Q_ou)
check(f"random-walk jerk grows as t^5 (measured t^{k_rw:.2f})", 4.7 < k_rw < 5.3)
check(f"the OU grows as t^3 (measured t^{k_ou:.2f})", 2.8 < k_ou < 3.5)

def sd(F, Q, s):
    return math.sqrt(growth(F, Q, s))

r1, r20 = sd(F_rw, Q_rw, 1.0) / sd(F_ou, Q_ou, 1.0), sd(F_rw, Q_rw, 20.0) / sd(F_ou, Q_ou, 20.0)
check(f"at 1s the two barely differ ({r1:.1f}x) — short-horizon capture is kept",
      r1 < 2.0)
check(f"at 20s they diverge hard ({r20:.0f}x) — that is the fix", r20 > 8.0)
check("the OU is strictly the tighter of the two at long horizons",
      sd(F_ou, Q_ou, 20.0) < sd(F_rw, Q_rw, 20.0))

# ---------------------------------------------------------------- #
# 3. tau_accel must SURVIVE the pipeline (the 122hc whitelist trap)
# ---------------------------------------------------------------- #
changes = {f.name: dict(v) for f in dc.fields(p_ou)
           if isinstance(v := getattr(p_ou, f.name), dict)}
changes["sigma_marker"] = {m: 1.0 for m in layout.marker_names}
check("tau_accel survives the warm-start rebuild",
      dc.replace(p_ou, **changes).tau_accel == 0.5)

src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
tree = ast.parse(src)


def fn_src(name):
    for f in ast.walk(tree):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return ast.get_source_segment(src, f) or ""
    return ""


for name in ("finalize_m_step_v2", "finalize_m_step_v2_from_per_session"):
    body = fn_src(name)
    check(f"{name} no longer hand-lists NoiseParamsV2 fields",
          "return NoiseParamsV2(" not in body)
    check(f"{name} carries prev_params forward wholesale",
          "_dc_ms.replace(" in body and "prev_params," in body)

check("fit_initial_params_v2 still CONSTRUCTS (it has no prev to copy)",
      "NoiseParamsV2(" in fn_src("fit_initial_params_v2"))

n_fields = len([b.target.id for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "NoiseParamsV2"
                for b in n.body if isinstance(b, ast.AnnAssign)])
check(f"NoiseParamsV2 has {n_fields} fields and no whitelist re-lists them",
      n_fields == 16)

# Every build_F_v2 call site must pass params, or phi silently reverts to 1.0
# and alpha_drift silently reverts to the hardcoded 0.05.
missing = []
for n in ast.walk(tree):
    if (isinstance(n, ast.Call)
            and getattr(n.func, "id", "") == "build_F_v2"
            and len(n.args) + len(n.keywords) < 3):
        missing.append(n.lineno)
check("every build_F_v2 call site passes params", not missing)

for name in ("rts_smooth_v2", "per_session_fit_from_stats"):
    for f in ast.walk(tree):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            check(f"{name} accepts params (its F must match the filter's)",
                  "params" in [a.arg for a in f.args.args])

# alpha_drift was inert for the same reason — verify it now lands.
lay_d = dc.replace(layout_from_segments(SPEC), with_drift=True)
p_d = NoiseParamsV2.default(lay_d)
p_d9 = dc.replace(p_d, alpha_drift={m: 0.9 for m in lay_d.marker_names})
check("alpha_drift now reaches F (it was inert while params was never passed)",
      not np.allclose(build_F_v2(lay_d, DT, p_d), build_F_v2(lay_d, DT, p_d9)))

# ---------------------------------------------------------------- #
# 4. persistence
# ---------------------------------------------------------------- #
tmp = pathlib.Path(tempfile.mkdtemp())
from mufasa.data_processors.kalman_pose_smoother_v2 import (  # noqa: E402
    FittedLengths,
)

fl = FittedLengths(
    segment_lengths={s.name: 10.0 for s in layout.segments},
    segment_length_iqr={s.name: 1.0 for s in layout.segments},
    marker_offsets={m: (5.0, 0.0) for m in layout.marker_names},
    marker_r_drift={m: 0.5 for m in layout.marker_names},
    marker_q_drift={m: 0.1 for m in layout.marker_names},
)

mp = tmp / "ou.npz"
save_model_v2(mp, layout, fl, p_ou, 30.0, 0.7)
check("tau_accel survives save/load", load_model_v2(mp)[2].tau_accel == 0.5)

mp2 = tmp / "rw.npz"
save_model_v2(mp2, layout, fl, p_rw, 30.0, 0.7)
check("tau_accel=None round-trips as None, not 0.0 or -1.0",
      load_model_v2(mp2)[2].tau_accel is None)

check("the CLI exposes --accel-tau", '"--accel-tau", type=float' in src)
check("the CLI flag is threaded through", "accel_tau=args.accel_tau," in src)
check("smooth_pose_v2 accepts accel_tau",
      "accel_tau" in [a.arg for a in next(
          f for f in ast.walk(tree)
          if isinstance(f, ast.FunctionDef) and f.name == "smooth_pose_v2").args.args])
check("EM accepts accel_tau",
      "accel_tau" in [a.arg for a in next(
          f for f in ast.walk(tree)
          if isinstance(f, ast.FunctionDef)
          and f.name == "fit_noise_params_em_v2").args.args])

# ---------------------------------------------------------------- #
n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hg_singer_accel: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
