"""Smoke test for patch 122hf — von Mises joint prior on segment angles.

Why the prior exists
--------------------

Five of the seven segments in a typical rodent tree carry exactly one marker
(``neck``, ``back_rear``, ``tail_1..3``). A segment with two or more markers
survives losing one — the others pin its pose and FK places the missing
marker. A single-marker segment has no redundancy: the moment its marker drops
below threshold, nothing observes its angle.

And nothing constrains it either. ``build_F_v2`` models segment orientation as
a random walk on (cos, sin); the only prior is ``cos²+sin²=1``, which fixes the
*radius* and says nothing about the *angle*. Within about a second the angle is
uniform and the marker is somewhere on a circle of radius L.

What is asserted here
---------------------

The interesting assertions are behavioural, on real filter runs:

  * on a tail that genuinely trails, the prior recovers the structure and
    cuts a 20 s dropout from ~115 deg RMSE to ~18 deg;
  * on a tail that is genuinely near-uniform, the prior switches ITSELF off
    and the output is bit-identical to today.

The second matters as much as the first. During a dropout nothing competes
with the prior, so even a weak one fully determines the angle — measured, a
kappa of 0.2 fitted from a near-uniform joint made the worst-case error
*worse* (151 deg -> 180 deg). The kappa floor is what makes the failure mode
"no help" rather than "wrong help".
"""
from __future__ import annotations

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

import numpy as np  # noqa: E402

from mufasa.data_processors.kalman_pose_smoother_v2 import (  # noqa: E402
    _JOINT_PRIOR_KAPPA_CAP,
    _JOINT_PRIOR_KAPPA_FLOOR,
    JointPriorV2,
    _build_constraint_observations,
    _fit_von_mises,
    fit_body_lengths,
    fit_initial_params_v2,
    fit_joint_priors_v2,
    forward_filter_v2,
    initial_state_from_data,
    layout_from_segments,
    load_model_v2,
    rts_smooth_v2,
    save_model_v2,
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


# ---------------------------------------------------------------- #
# 1. _fit_von_mises
# ---------------------------------------------------------------- #
rng = np.random.default_rng(0)

th = rng.vonmises(0.4, 20.0, 4000)
mu, kappa = _fit_von_mises(np.cos(th), np.sin(th))
check("recovers mu of a tight von Mises sample", abs(mu - 0.4) < 0.05)
check("recovers kappa of a tight von Mises sample", 15.0 < kappa < 26.0)

th_u = rng.uniform(-np.pi, np.pi, 4000)
_, kappa_u = _fit_von_mises(np.cos(th_u), np.sin(th_u))
check("a uniform sample yields kappa ~ 0", kappa_u < 0.5)

check("empty sample is safe", _fit_von_mises(np.array([]), np.array([])) == (0.0, 0.0))

# Sample far tighter than the cap, so the cap is what determines the answer.
# (Asserting `kappa <= _JOINT_PRIOR_KAPPA_CAP` would be a tautology: it holds
# for any cap, including one raised to 1e12. Assert the clamp actually bites.)
th_w = rng.vonmises(0.0, 5000.0, 4000)
_, kappa_w = _fit_von_mises(np.cos(th_w), np.sin(th_w))
check("kappa is capped, so no prior can be arbitrarily stiff",
      kappa_w == _JOINT_PRIOR_KAPPA_CAP and kappa_w < 1000.0)

# Off-circle samples must be normalized first, or the resultant length —
# and hence kappa — is biased by the radius rather than the angle.
scale = rng.uniform(0.2, 5.0, 4000)
mu_s, kappa_s = _fit_von_mises(np.cos(th) * scale, np.sin(th) * scale)
check("samples are normalized to the unit circle before fitting",
      abs(mu_s - mu) < 0.05 and abs(kappa_s - kappa) / kappa < 0.25)

# Wrap-around: a sample straddling +/-pi must not average to 0.
th_pi = rng.vonmises(np.pi, 30.0, 4000)
mu_pi, _ = _fit_von_mises(np.cos(th_pi), np.sin(th_pi))
check("handles wrap-around at +/-pi (circular mean, not arithmetic)",
      abs(abs(mu_pi) - np.pi) < 0.05)

# ---------------------------------------------------------------- #
# 2. the pseudo-observation rows
# ---------------------------------------------------------------- #
layout = layout_from_segments(SPEC)
n_seg = len(layout.non_root_topo_order)
state = np.zeros(layout.state_dim)
for s in layout.non_root_topo_order:
    sl = layout.slice_segment_orientation(s)
    state[sl] = [1.0, 0.0, 0.0, 0.0]
state[4:6] = [1.0, 0.0]

z0, h0, H0, R0 = _build_constraint_observations(state, layout, 0.05)
check("without a prior: one norm row per segment plus the root",
      z0.shape[0] == n_seg + 1)
check("R comes back per-row now", R0.shape == z0.shape)
check("norm rows all carry constraint_sigma^2",
      np.allclose(R0, 0.05 ** 2))

prior = JointPriorV2(
    mu={s: 0.3 for s in layout.non_root_topo_order},
    kappa={s: 9.0 for s in layout.non_root_topo_order},
)
z1, h1, H1, R1 = _build_constraint_observations(
    state, layout, 0.05, joint_prior=prior)
check("prior adds exactly two rows per active segment",
      z1.shape[0] == z0.shape[0] + 2 * n_seg)
check("prior rows target (cos mu, sin mu)",
      np.isclose(z1[z0.shape[0]], math.cos(0.3))
      and np.isclose(z1[z0.shape[0] + 1], math.sin(0.3)))
check("prior rows carry variance 1/kappa",
      np.allclose(R1[z0.shape[0]:], 1.0 / 9.0))
check("prior Jacobian is constant (linear in the state)",
      np.allclose(H1[z0.shape[0]:] @ np.ones(layout.state_dim),
                  H1[z0.shape[0]:].sum(axis=1)))
check("prior rows are unit selectors on the (cos,sin) pair",
      set(np.unique(H1[z0.shape[0]:])) <= {0.0, 1.0}
      and np.allclose(H1[z0.shape[0]:].sum(axis=1), 1.0))

inert = JointPriorV2(mu={s: 0.3 for s in layout.non_root_topo_order},
                     kappa={s: 0.0 for s in layout.non_root_topo_order})
z2, _, _, _ = _build_constraint_observations(
    state, layout, 0.05, joint_prior=inert)
check("kappa=0 adds no rows at all (inert, not merely weak)",
      z2.shape[0] == z0.shape[0])

# ---------------------------------------------------------------- #
# 3. real behaviour: a trailing tail
# ---------------------------------------------------------------- #
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


def evaluate(tail_swing, other_swing=None):
    pos, lik, names = make(tail_swing=tail_swing, other_swing=other_swing)
    fl = fit_body_lengths(pos, lik, layout, names, 0.7, dt=1 / 30)
    p0 = fit_initial_params_v2(pos, lik, layout, names, fl, 30.0,
                               likelihood_threshold=0.7)
    jp = fit_joint_priors_v2([(pos, lik)], layout, names, fl, p0, 30.0)

    def run(likes, pr):
        x0 = initial_state_from_data(pos, likes, layout, names, fl, 0.7)
        f = forward_filter_v2(pos, likes, layout, p0, 1 / 30.,
                              initial_state=x0, likelihood_threshold=0.7,
                              apply_constraints=True, joint_prior=pr)
        return rts_smooth_v2(f, layout, 1 / 30.).x_smooth

    ti = names.index("tail_V32")
    s0, s1 = 700, 1300                       # 600 frames = 20 s @ 30 fps
    lik_d = lik.copy()
    lik_d[s0:s1, ti] = 0.02
    sl = layout.slice_segment_orientation("tail_3")
    ref = run(lik, None)[s0:s1, sl]
    th_ref = np.arctan2(ref[:, 1], ref[:, 0])

    def rmse(pr):
        x = run(lik_d, pr)[s0:s1, sl]
        t = np.arctan2(x[:, 1], x[:, 0])
        e = np.angle(np.exp(1j * (t - th_ref)))
        return math.degrees(np.sqrt(np.mean(e ** 2)))

    return jp, rmse(None), rmse(jp)


jp_t, off_t, on_t = evaluate(tail_swing=0.18)
k3 = jp_t.kappa["tail_3"]
check("trailing tail: tail_3 fits an ACTIVE prior", k3 >= _JOINT_PRIOR_KAPPA_FLOOR)
check("trailing tail: its mu is near zero (the tail trails)",
      abs(math.degrees(jp_t.mu["tail_3"])) < 30.0)
check("trailing tail: n_obs is recorded", jp_t.n_obs["tail_3"] > 1000)
check(f"trailing tail: 20s dropout improves a lot "
      f"({off_t:.0f} deg -> {on_t:.0f} deg RMSE)",
      on_t < off_t / 3.0)
check(f"trailing tail: prior-off is genuinely bad ({off_t:.0f} deg RMSE), "
      f"i.e. the problem being fixed is real",
      off_t > 40.0)

# Widen ONLY the tail: head/neck/back_rear keep their tight swings. This is
# the sharper test — one run in which tail_3 goes inert because its angle
# carries no information, while head stays active because its does. The prior
# is per-segment and self-selecting; nothing had to be told which is which.
jp_u, off_u, on_u = evaluate(tail_swing=1.4)
check("near-uniform tail: tail_3 fits kappa below the floor -> inert",
      jp_u.kappa["tail_3"] == 0.0)
check("near-uniform tail: the tail_3 dropout is UNCHANGED (no help, no harm)",
      abs(on_u - off_u) < 2.0)
check("...while a genuinely tight joint elsewhere stays active",
      jp_u.kappa["head"] >= _JOINT_PRIOR_KAPPA_FLOOR)

# ---------------------------------------------------------------- #
# 3b. the fit must ignore dropout frames
# ---------------------------------------------------------------- #
# The crux of fit_joint_priors_v2: only frames where the segment's own
# marker was observed may contribute. Fitting from all frames would learn
# the prior from the very dropout stretches the prior exists to constrain,
# and would return "uniform" for exactly the joints that need help most.
# The tests above cannot catch a regression here because their fitting data
# has no dropouts at all, so "observed" is trivially every frame.
pos_f, lik_f, names_f = make(tail_swing=0.18)
fl_f = fit_body_lengths(pos_f, lik_f, layout, names_f, 0.7, dt=1 / 30)
p0_f = fit_initial_params_v2(pos_f, lik_f, layout, names_f, fl_f, 30.0,
                             likelihood_threshold=0.7)
jp_clean = fit_joint_priors_v2([(pos_f, lik_f)], layout, names_f, fl_f,
                               p0_f, 30.0)

lik_holes = lik_f.copy()
ti_f = names_f.index("tail_V32")
lik_holes[300:900, ti_f] = 0.02          # 20 s blind, a third of the session
jp_holes = fit_joint_priors_v2([(pos_f, lik_holes)], layout, names_f, fl_f,
                               p0_f, 30.0)

check("fitting data with a 20s dropout still yields an ACTIVE tail_3 prior",
      jp_holes.kappa["tail_3"] >= _JOINT_PRIOR_KAPPA_FLOOR)
check("...its n_obs excludes the blind frames",
      jp_holes.n_obs["tail_3"] < jp_clean.n_obs["tail_3"] - 500)
check("...and kappa is not dragged toward uniform by them",
      jp_holes.kappa["tail_3"] > 0.5 * jp_clean.kappa["tail_3"])
check("...and mu is essentially unmoved",
      abs(math.degrees(np.angle(np.exp(1j * (jp_holes.mu["tail_3"]
                                             - jp_clean.mu["tail_3"]))))) < 20.0)

# ---------------------------------------------------------------- #
# 4. save / load round-trip and backward compatibility
# ---------------------------------------------------------------- #
tmp = pathlib.Path(tempfile.mkdtemp())
pos, lik, names = make()
fl = fit_body_lengths(pos, lik, layout, names, 0.7, dt=1 / 30)
p0 = fit_initial_params_v2(pos, lik, layout, names, fl, 30.0,
                           likelihood_threshold=0.7)

mp = tmp / "with_prior.npz"
save_model_v2(mp, layout, fl, p0, 30.0, 0.7, None, jp_t)
loaded = load_model_v2(mp)
check("load_model_v2 returns a 7-tuple ending in the joint prior",
      len(loaded) == 7)
jp_r = loaded[6]
check("joint prior survives the round-trip",
      jp_r is not None
      and set(jp_r.kappa) == set(jp_t.kappa)
      and all(math.isclose(jp_r.kappa[k], jp_t.kappa[k]) for k in jp_t.kappa)
      and all(math.isclose(jp_r.mu[k], jp_t.mu[k]) for k in jp_t.mu))
check("n_obs survives too", jp_r.n_obs == jp_t.n_obs)

mp2 = tmp / "no_prior.npz"
save_model_v2(mp2, layout, fl, p0, 30.0, 0.7, None)
check("a model saved without a prior loads as None, not an error",
      load_model_v2(mp2)[6] is None)

# ---------------------------------------------------------------- #
# 5. wiring
# ---------------------------------------------------------------- #
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()


def _ordered(*markers: str, hay: str = src) -> bool:
    last = -1
    for m in markers:
        if m not in hay:
            return False
        i = hay.index(m)
        if i <= last:
            return False
        last = i
    return True


em = src[src.index("def fit_noise_params_em_v2("):]
em = em[:em.index("\ndef ")]
check("EM gained enable_joint_prior", "enable_joint_prior: bool = False" in em)
check("the prior is fitted ONCE before the iterations, after perspective",
      _ordered("fit_perspective_model_v2(", "fit_joint_priors_v2(",
               "for iteration in range(", hay=em))
check("the fit pass itself runs WITHOUT a prior (no chasing its own tail)",
      "joint_prior" not in
      src[src.index("def fit_joint_priors_v2("):src.index("def fit_joint_priors_v2(") + src[src.index("def fit_joint_priors_v2("):].index("return JointPriorV2")].split("filt = forward_filter_v2")[1].split(")")[0])
check("warm-start does NOT receive a prior (it runs before the fit)",
      "joint_prior" not in
      src[src.index("def _pool_warm_start_pass("):
          src.index("def _pool_warm_start_pass(") + 900])
check("the EM E-step worker receives it",
      "sess_idx, params, perspective, joint_prior, iteration" in src)
check("the final smoothing worker receives it",
      "sess_idx, params, perspective, joint_prior, device = args" in src)
check("smooth_pose_v2 exposes it", "enable_joint_prior: bool = False" in
      src[src.index("def smooth_pose_v2("):src.index("def smooth_pose_v2(") + 1500])
check("the CLI exposes --joint-prior", '"--joint-prior", action="store_true"' in src)
check("the CLI flag is actually passed through",
      "enable_joint_prior=args.joint_prior," in src)
check("EMResultV2 carries it back out",
      "joint_prior: JointPriorV2 | None = None" in
      src[src.index("class EMResultV2"):src.index("class EMResultV2") + 1400])

# ---------------------------------------------------------------- #
n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hf_joint_prior: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
