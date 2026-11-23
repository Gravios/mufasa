"""
mufasa.biomech.muscle
=====================

Patch 122lj: muscle forces from joint torques (stage 3 of the biomechanical model), the hindlimbs.

Each muscle is a spatial tendon in MuJoCo through its path's sites (:mod:`.osim`,
:func:`mufasa.biomech.mjmodel.build`). Its length and its moment arms on every joint come from
MuJoCo, frame by frame. Its force:

``F = F_max · (a · f_L(l̃) · f_V(ṽ) + f_P(l̃)) · cos α``

* a Hill-type muscle with a rigid tendon (Zajac 1989): the fibre is the path less the tendon's
  slack length, at constant thickness and pennation α (``l_f sin α = l_opt sin α₀``);
* f_L: active force–length, a Gaussian of width 0.45 (Thelen 2003);
* f_P: passive force–length, exponential, 1 at 60 % strain (Thelen 2003);
* f_V: force–velocity, Hill's hyperbola shortening (0 at ``v_max`` optimal lengths per second,
  curvature 0.25) and up to 1.8× lengthening.

**Static optimisation** (Crowninshield & Brand 1981; Anderson & Pandy 2001). For each frame and
each hindlimb, the activations a ∈ [0, 1] with the least Σa² that give the hip's three torques,
the knee's and the ankle's (:mod:`.dynamics`). Where the muscles cannot, a **reserve** torque
makes up the rest at a high cost, and it is reported: it says the muscles, or the motion, or the
torques, are off there. The toes have no muscle in the model (the digit flexors and extensors end
on the foot); their torque is left as is.

One frame at a time: activations do not carry over between frames here. Stage 4 optimises whole
strides, with the activations' own dynamics.
"""
from __future__ import annotations

import numpy as np

from mufasa.biomech.mjmodel import RodentMJ, mujoco

FL_WIDTH = 0.45            # active force-length: exp(-(l̃-1)² / FL_WIDTH)
PE_K, PE_STRAIN = 5.0, 0.6  # passive: (exp(PE_K (l̃-1)/PE_STRAIN) - 1) / (exp(PE_K) - 1)
FV_CURV = 0.25             # Hill's curvature (shortening)
FV_ECC = 1.8               # force at fast lengthening / isometric
RESERVE_COST = 100.0       # a reserve torque of TAU_REF costs as much as 100 muscles' full activation
TAU_REF = 0.01             # N·m
LEG_DOFS = ("hip_{S}_supinate", "hip_{S}_abduct", "hip_{S}_extend", "knee_{S}", "ankle_{S}")


def f_active(lt):
    return np.exp(-((lt - 1.0) ** 2) / FL_WIDTH)


def f_passive(lt):
    x = np.maximum(lt - 1.0, 0.0)
    return (np.exp(PE_K * x / PE_STRAIN) - 1.0) / (np.exp(PE_K) - 1.0)


def f_velocity(vt):
    """``vt``: fibre velocity in v_max (negative shortening)."""
    vt = np.asarray(vt, float)
    short = (1.0 + np.maximum(vt, -1.0)) / (1.0 - np.maximum(vt, -1.0) / FV_CURV)
    slope = (1.0 + 1.0 / FV_CURV)                          # the shortening branch's slope at 0
    c = (FV_ECC - 1.0) / slope
    long = 1.0 + (FV_ECC - 1.0) * vt / (vt + c)
    return np.where(vt < 0, short, long)


class MuscleTable:
    """The model's muscles as arrays, scaled to it (size, and its mass if it was rescaled)."""

    def __init__(self, M: RodentMJ):
        if M.muscles is None:
            raise ValueError("this model was built without muscles (build(..., muscles=...))")
        m = M.model
        mus = M.muscles.muscles
        self.names = [mu.name for mu in mus]
        self.side = np.array([mu.side for mu in mus])
        self.tid = np.array([m.tendon(f"muscle_{mu.name}").id for mu in mus])
        k = M.size
        base = (M.size ** 3) * _base_mass()
        mk = (M.mass_kg / base) ** (1.0 / 3.0) if base > 0 else 1.0   # a rescaled mass: forces ~ mass^(2/3)
        self.f_max = np.array([mu.f_max for mu in mus]) * k * k * mk * mk
        self.l_opt = np.array([mu.l_opt for mu in mus]) * k
        self.l_slack = np.array([mu.l_slack for mu in mus]) * k
        self.alpha0 = np.array([mu.pennation for mu in mus])
        self.v_max = np.array([mu.v_max for mu in mus])

    def fibre(self, L, V):
        """Normalised fibre length and velocity, and cos(pennation), from path lengths and speeds."""
        h = self.l_opt * np.sin(self.alpha0)
        x = np.maximum(L - self.l_slack, 1e-6)
        lf = np.sqrt(x ** 2 + h ** 2)
        ca = x / lf
        return lf / self.l_opt, V * ca / (self.v_max * self.l_opt), ca


_BM = None


def _base_mass() -> float:
    global _BM
    if _BM is None:
        from mufasa.biomech.mjmodel import base_model
        _BM = float(base_model().body_mass.sum())
    return _BM


def tendon_state(M: RodentMJ, q, v):
    """Every tendon's length (m), speed (m/s) and Jacobian (n_tendon, nv) at one state."""
    mj, m = mujoco(), M.model
    d = M.data()
    d.qpos[:] = q
    mj.mj_fwdPosition(m, d)
    J = np.zeros((m.ntendon, m.nv))
    for i in range(m.ntendon):
        a, nz = m.ten_J_rowadr[i], m.ten_J_rownnz[i]
        J[i, m.ten_J_colind[a:a + nz]] = d.ten_J[a:a + nz]
    return d.ten_length.copy(), J @ v, J


def static_optimisation(M: RodentMJ, Q, V, tau) -> dict:
    """Per frame: activations (T, n), forces (T, n) N, the reserve torques (T, 2, 5) N·m (L, R ×
    :data:`LEG_DOFS`), fibre lengths (T, n, in l_opt) and the torques asked (T, 2, 5)."""
    from scipy.optimize import lsq_linear
    mt = MuscleTable(M)
    T = len(Q)
    n = len(mt.names)
    act = np.full((T, n), np.nan)
    frc = np.full((T, n), np.nan)
    lfib = np.full((T, n), np.nan)
    res = np.full((T, 2, len(LEG_DOFS)), np.nan)
    ask = np.full((T, 2, len(LEG_DOFS)), np.nan)
    hidx = {h: k for k, h in enumerate(M.hinge)}
    for t in range(T):
        if not (np.isfinite(Q[t]).all() and np.isfinite(V[t]).all() and np.isfinite(tau[t]).all()):
            continue
        L, Ld, J = tendon_state(M, Q[t], V[t])
        lt, vt, ca = mt.fibre(L[mt.tid], Ld[mt.tid])
        lfib[t] = lt
        act_gain = mt.f_max * f_active(lt) * f_velocity(vt) * ca          # N per unit activation
        pas = mt.f_max * f_passive(lt) * ca
        for si, S in enumerate("LR"):
            dofs = [d.format(S=S) for d in LEG_DOFS]
            cols = [M.vadr[d] for d in dofs]
            mine = np.flatnonzero(mt.side == S)
            R = -J[mt.tid[mine]][:, cols].T                                # (5, k) moment arms, m
            A = R * act_gain[mine]
            b = np.array([tau[t, hidx[d]] for d in dofs]) - R @ pas[mine]
            ask[t, si] = [tau[t, hidx[d]] for d in dofs]
            k = len(mine)
            w_eq = 100.0 / TAU_REF
            w_r = np.sqrt(RESERVE_COST) / TAU_REF
            top = np.hstack([A, np.eye(len(dofs))]) * w_eq
            reg = np.hstack([np.eye(k), np.zeros((k, len(dofs)))])
            rr = np.hstack([np.zeros((len(dofs), k)), np.eye(len(dofs)) * w_r])
            sol = lsq_linear(np.vstack([top, reg, rr]), np.r_[b * w_eq, np.zeros(k + len(dofs))],
                             bounds=(np.r_[np.zeros(k), np.full(len(dofs), -np.inf)],
                                     np.r_[np.ones(k), np.full(len(dofs), np.inf)]))
            a = sol.x[:k]
            act[t, mine] = a
            frc[t, mine] = a * act_gain[mine] + pas[mine]
            res[t, si] = sol.x[k:]
    return {"names": mt.names, "act": act, "force": frc, "reserve": res, "asked": ask, "fibre": lfib,
            "f_max": mt.f_max}


def muscle_columns(out: dict) -> dict:
    """Columns for the dynamics table: ``<muscle>_act``, ``<muscle>_N``, ``reserve_<dof>_mNm``."""
    cols = {}
    for k, n in enumerate(out["names"]):
        cols[f"{n}_act"] = out["act"][:, k]
        cols[f"{n}_N"] = out["force"][:, k]
    for si, S in enumerate("LR"):
        for j, d in enumerate(LEG_DOFS):
            cols[f"reserve_{d.format(S=S)}_mNm"] = out["reserve"][:, si, j] * 1000.0
    return cols


def summarise_muscles(out: dict) -> dict:
    """How much of the legs' torque the muscles supply, and the busiest muscles."""
    ok = np.isfinite(out["reserve"]).all((1, 2))
    if not ok.any():
        return {}
    asked = np.abs(out["asked"][ok]).sum((1, 2))
    rsv = np.abs(out["reserve"][ok]).sum((1, 2))
    s = {"muscle_frames": int(ok.sum()),
         "reserve_vs_torque_median": round(float(np.median(rsv / np.maximum(asked, 1e-9))), 3),
         "frames_without_reserve": round(float(np.mean(rsv < 0.01 * np.maximum(asked, 1e-9))), 3)}
    med = np.nanmedian(out["act"][ok], axis=0)
    for k, i in enumerate(np.argsort(-med)[:8]):
        s[f"activation_median_{k + 1}"] = f"{out['names'][i]} {med[i]:.2f}"
    return s


__all__ = ["LEG_DOFS", "MuscleTable", "f_active", "f_passive", "f_velocity", "muscle_columns",
           "static_optimisation", "summarise_muscles", "tendon_state"]
