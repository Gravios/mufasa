"""
mufasa.biomech.dynamics
=======================

Patch 122li: joint torques and ground forces from the observed motion (stage 2 of the
biomechanical model).

1. Velocities and accelerations of every joint come from the fitted trajectory
   (:func:`mufasa.biomech.fit.fit_trajectory`): MuJoCo's position differences between frames,
   smoothed with a Savitzky–Golay filter over :data:`WINDOW_S` (positions too). Runs of frames
   are taken one at a time, broken by unfitted frames.
2. MuJoCo's inverse dynamics (``mj_inverse``) gives the generalised forces that produce them:
   the root's six (force and torque) and one per hinge.
3. Only the ground can push the root, and only through planted paws. Their forces are solved
   (``lsq_linear``) to supply the root's six: each paw's force at the middle of its sole, never
   pulling (Fz ≥ 0), within a friction cone (μ :data:`MU`, assumed; a force outside it is cut
   back to it). The torque rows are scaled by :data:`L_REF` so they count as much as the force
   rows. Four paws down give more unknowns than the six balance equations, so the split of the
   weight among the paws is taken from the gait's load sharing (its ``<paw>_load``), weighted by
   :data:`SHARE_W`. The smallest forces alone put 58 % of the weight on one hindpaw on K7, with
   spine torques to match.
4. Joint torques are the hinges' generalised forces less what the ground forces do through
   them. They are net torques, muscles and passive tissue together (the model's simulation
   springs, damping and armature are off; :mod:`.mjmodel`).
5. What the paws cannot supply is the **residual**: the force and torque at the root that the
   motion needs and nothing explains. It is the check on everything before it (kinematics,
   masses, contacts). OpenSim calls its counterpart the residual actuators.
"""
from __future__ import annotations

import numpy as np

from mufasa.biomech.mjmodel import RodentMJ, mujoco
from mufasa.pose3d.gait import PAWS

WINDOW_S = 0.15            # Savitzky–Golay window for positions, velocities, accelerations
MU = 0.8                   # paw–floor friction (assumed)
REG = 1e-6                 # a little regularisation: of the forces that balance, the smallest
L_REF = 0.05               # m: torque rows / this, to weigh like force rows
SHARE_W = 0.3              # the gait's split of the weight among planted paws, against the balance


def _sg(x, fps: float, deriv: int = 0):
    from scipy.signal import savgol_filter
    w = max(5, int(round(WINDOW_S * fps)) | 1)
    if len(x) < w:
        return np.gradient(x, 1.0 / fps, axis=0) if deriv else x
    return savgol_filter(x, w, 3 if w >= 7 else 2, deriv=deriv, delta=1.0 / fps, axis=0)


def derivatives(M: RodentMJ, Q, fps: float):
    """Smoothed qpos and their qvel, qacc (T, nv), run by run (NaN where unfitted)."""
    mj, m = mujoco(), M.model
    Q = np.array(Q, float)
    T = len(Q)
    V = np.full((T, m.nv), np.nan)
    A = np.full((T, m.nv), np.nan)
    ok = np.isfinite(Q).all(1)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    hq = np.array([M.qadr[n] for n in M.hinge])
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        if b - a < 3:
            continue
        run = Q[a:b]
        run[:, :3] = _sg(run[:, :3], fps)
        run[:, hq] = _sg(run[:, hq], fps)
        half = np.zeros((b - a - 1, m.nv))
        for k in range(b - a - 1):
            mj.mj_differentiatePos(m, half[k], 1.0 / fps, run[k], run[k + 1])
        v = np.vstack([half[:1], (half[:-1] + half[1:]) / 2, half[-1:]])
        V[a:b] = _sg(v, fps)
        A[a:b] = _sg(V[a:b], fps, deriv=1)
        Q[a:b] = run
    return Q, V, A


def _grf(Jc, f_root, share=None):
    """Paw forces (K, 3) supplying ``f_root`` (6,) through Jacobians Jc (K, 3, nv); ``share`` (K,):
    the gait's split of the vertical load among them (None: the smallest forces)."""
    from scipy.optimize import lsq_linear
    K = len(Jc)
    if K == 0:
        return np.zeros((0, 3))
    A = np.hstack([J[:, :6].T for J in Jc])                       # (6, 3K)
    sc = np.r_[np.ones(3), np.full(3, 1.0 / L_REF)]
    A, b = A * sc[:, None], f_root * sc
    prior = np.zeros(3 * K)
    w = np.full(3 * K, np.sqrt(REG))
    if share is not None and np.isfinite(share).all() and share.sum() > 0:
        prior[2::3] = max(f_root[2], 0.0) * share / share.sum()
        w[2::3] = SHARE_W
    A = np.vstack([A, np.diag(w)])
    b = np.r_[b, w * prior]
    lo = np.tile([-np.inf, -np.inf, 0.0], K)
    F = lsq_linear(A, b, bounds=(lo, np.full(3 * K, np.inf))).x.reshape(K, 3)
    t = np.linalg.norm(F[:, :2], axis=1)
    cut = t > MU * F[:, 2]
    F[cut, :2] *= (MU * F[cut, 2] / np.maximum(t[cut], 1e-12))[:, None]
    return F


def inverse(M: RodentMJ, Q, V, A, modes, shares=None) -> dict:
    """Per frame: joint torques (T, n_hinge) N·m, paw forces (T, 4, 3) N, the root's residual force
    (T, 3) N and torque (T, 3) N·m, and whether each paw bore weight (T, 4). ``shares`` (T, 4):
    the gait's load split (:data:`mufasa.pose3d.gait.PAWS` order), or None."""
    mj, m = mujoco(), M.model
    d = M.data()
    T = len(Q)
    hv = np.array([M.vadr[n] for n in M.hinge])
    tau = np.full((T, len(hv)), np.nan)
    grf = np.full((T, 4, 3), np.nan)
    res_f = np.full((T, 3), np.nan)
    res_t = np.full((T, 3), np.nan)
    down = np.zeros((T, 4), bool)
    jh, jt = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for t in range(T):
        if not (np.isfinite(Q[t]).all() and np.isfinite(A[t]).all()):
            continue
        d.qpos[:], d.qvel[:], d.qacc[:] = Q[t], V[t], A[t]
        mj.mj_inverse(m, d)
        f = d.qfrc_inverse.copy()
        mj.mj_comPos(m, d)
        paws = [p for p in PAWS if (modes[t] or {}).get(p) in ("stance", "plant")]
        Jc = []
        for p in paws:
            h, s = M.soles[p]
            mj.mj_jacSite(m, d, jh, None, h)
            mj.mj_jacSite(m, d, jt, None, s)
            Jc.append((jh + jt) / 2)
        sh = None if shares is None else np.array([shares[t][PAWS.index(p)] for p in paws], float)
        F = _grf(Jc, f[:6], sh)
        for k, p in enumerate(paws):
            f = f - Jc[k].T @ F[k]
            grf[t, PAWS.index(p)] = F[k]
            down[t, PAWS.index(p)] = True
        for p in PAWS:
            if not down[t, PAWS.index(p)]:
                grf[t, PAWS.index(p)] = 0.0
        tau[t] = f[hv]
        res_f[t], res_t[t] = f[:3], f[3:6]
    return {"tau": tau, "grf": grf, "res_f": res_f, "res_t": res_t, "down": down}


def dynamics_table(M: RodentMJ, frames, Q, rms_mm, out: dict):
    """One row per frame: the fit's error, each paw's force (N), the residual, and per hinge its
    angle (deg) and net torque (mN·m)."""
    import pandas as pd
    cols = {"frame": np.asarray(frames), "fit_rms_mm": rms_mm}
    for k, p in enumerate(PAWS):
        for c, ax in enumerate("xyz"):
            cols[f"{p}_grf_{ax}_N"] = out["grf"][:, k, c]
        cols[f"{p}_down"] = out["down"][:, k]
    for c, ax in enumerate("xyz"):
        cols[f"residual_f{ax}_N"] = out["res_f"][:, c]
    cols["residual_N"] = np.linalg.norm(out["res_f"], axis=1)
    cols["residual_Nm"] = np.linalg.norm(out["res_t"], axis=1)
    for c, ax in enumerate("xyz"):
        cols[f"root_{ax}_mm"] = Q[:, c] * 1000.0
    for c, ax in enumerate("wxyz"):
        cols[f"root_q{ax}"] = Q[:, 3 + c]
    for j, n in enumerate(M.hinge):
        cols[f"q_{n}_deg"] = np.degrees(Q[:, M.qadr[n]])
        cols[f"tau_{n}_mNm"] = out["tau"][:, j] * 1000.0
    return pd.DataFrame(cols)


def summarise(M: RodentMJ, df) -> dict:
    """Weight bearing and the residual against body weight, and the largest torques."""
    W = M.mass_kg * 9.81
    ok = df["fit_rms_mm"].notna() & df["residual_N"].notna()
    fz = sum(df[f"{p}_grf_z_N"] for p in PAWS)[ok]
    s = {"mass_g": round(M.mass_kg * 1000, 1), "weight_N": round(W, 3), "frames": int(ok.sum()),
         "fit_rms_mm": round(float(df["fit_rms_mm"][ok].median()), 1) if ok.any() else None,
         "support_vs_weight": round(float((fz / W).median()), 3) if ok.any() else None,
         "residual_vs_weight_median": round(float((df["residual_N"][ok] / W).median()), 3) if ok.any() else None,
         "residual_vs_weight_p90": round(float((df["residual_N"][ok] / W).quantile(0.9)), 3) if ok.any() else None,
         # the torque the paws cannot supply, against the weight on a lever of L_REF
         "residual_torque_vs_weight_median": (round(float((df["residual_Nm"][ok] / (W * L_REF)).median()), 3)
                                              if ok.any() else None)}
    peak = {n: float(df[f"tau_{n}_mNm"][ok].abs().quantile(0.9)) for n in M.hinge} if ok.any() else {}
    for k, (n, v) in enumerate(sorted(peak.items(), key=lambda kv: -kv[1])[:8]):
        s[f"torque_p90_{k + 1}"] = f"{n} {v:.1f} mN·m"
    return s


__all__ = ["MU", "REG", "WINDOW_S", "derivatives", "dynamics_table", "inverse", "summarise"]
