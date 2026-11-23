"""
mufasa.biomech.fit
==================

Patch 122li: every joint of the MuJoCo virtual rodent (:mod:`.mjmodel`), frame by frame, fitted to
the 3-D labels and to where the gait puts the paws. This is the same kind of fit as STAC, which
fits the virtual rodent to keypoints (Aldarondo et al. 2024).

Per frame, a Levenberg–Marquardt solve over the root (position and orientation) and the 67 hinge
joints within their ranges, with MuJoCo's exact Jacobians:

* labels: each seen label's site on its 3-D position (σ 4 mm);
* paws, as in :mod:`mufasa.pose3d.leg_ik`: stance on the floor at the gait's position (sole heights
  as in the stance, σ 1 mm; the sole's middle, σ 2 mm), swing on its arc (σ 4 mm), planted
  without a gait (σ 2 mm); a paw the gait cannot place is left free;
* a weak pull towards the tucked stance (per joint, per its whole range).

The first frame, and any frame after a gap, starts from the stance laid on the labels (Kabsch);
the others start from the frame before.

**Calibration** (:func:`calibrate`, STAC: Wu et al. 2013). Where a label sits on its body is the
animal's and the labeller's, not the model's: K7's back labels are spaced evenly, 40–45 mm apart,
where the model's named vertebrae are 21–36 mm apart, so its ``back_V2`` sits about 20 mm further
back. Poses are fitted on a sample of frames. Then each label's place on its body is re-solved,
the median over those frames of its observed position in that body's frame. This repeats a few
times. Labels stay on their bodies.
"""
from __future__ import annotations

import numpy as np

from mufasa.biomech.mjmodel import RodentMJ, mujoco
from mufasa.pose3d.gait import PAWS

SIGMA_MM = {"label": 4.0, "xy": 2.0, "sole": 1.0, "swing": 4.0, "plant": 2.0}
PRIOR = 0.3


def _kabsch(A, B):
    """R, t with R @ A + t ≈ B (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _quat(R):
    mj = mujoco()
    q = np.zeros(4)
    mj.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


class PoseFitter:
    """Fits the MuJoCo rodent's joints to labels and paw targets, a frame at a time."""

    def __init__(self, M: RodentMJ, names) -> None:
        self.M, self.m = M, M.model
        self.d = M.data()
        self.names = [n for n in names if n in M.labels]
        self.sites = np.array([M.labels[n] for n in self.names])
        self.hq = np.array([M.qadr[n] for n in M.hinge])
        self.hv = np.array([M.vadr[n] for n in M.hinge])
        self.span = M.hi - M.lo
        self.jac = np.zeros((3, M.nv))
        self.rest = self._stance_sites()

    def _stance_sites(self) -> np.ndarray:
        d = self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[self.hq] = self.M.q_stance
        mujoco().mj_kinematics(self.m, d)
        return d.site_xpos[self.sites].copy()

    def start(self, obs) -> np.ndarray | None:
        """The stance laid on the seen labels (m): qpos, or None with under 4 labels."""
        ok = np.isfinite(obs).all(1)
        if ok.sum() < 4:
            return None
        R, t = _kabsch(self.rest[ok], obs[ok])
        q = np.zeros(self.m.nq)
        q[:3] = t
        q[3:7] = _quat(R)
        q[self.hq] = self.M.q_stance
        return q

    def _site_jac(self, sid):
        mujoco().mj_jacSite(self.m, self.d, self.jac, None, int(sid))
        return self.jac.copy()

    def residual(self, q, obs, ok, paws):
        """Residuals and their Jacobian (rows) at qpos ``q``."""
        mj, m, d, M = mujoco(), self.m, self.d, self.M
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        r, J = [], []
        s = SIGMA_MM["label"] / 1000.0
        for k in np.flatnonzero(ok):
            sid = self.sites[k]
            r.append((d.site_xpos[sid] - obs[k]) / s)
            J.append(self._site_jac(sid) / s)
        for p, (mode, tgt) in paws.items():
            if mode == "free":
                continue
            h, t = M.soles[p]
            Jh, Jt = self._site_jac(h), self._site_jac(t)
            ph, pt = d.site_xpos[h], d.site_xpos[t]
            mid, Jm = (ph + pt) / 2, (Jh + Jt) / 2
            if mode in ("stance", "plant"):
                sz = SIGMA_MM["sole" if mode == "stance" else "plant"] / 1000.0
                r.append(np.array([(ph[2] - M.sole_z[p][0]) / sz, (pt[2] - M.sole_z[p][1]) / sz]))
                J.append(np.vstack([Jh[2], Jt[2]]) / sz)
            if mode == "stance":
                sx = SIGMA_MM["xy"] / 1000.0
                r.append((mid[:2] - tgt[:2]) / sx)
                J.append(Jm[:2] / sx)
            elif mode == "swing":
                sw = SIGMA_MM["swing"] / 1000.0
                r.append((mid - tgt) / sw)
                J.append(Jm / sw)
        pr = PRIOR * (q[self.hq] - M.q_stance) / self.span
        Jp = np.zeros((len(self.hq), M.nv))
        Jp[np.arange(len(self.hq)), self.hv] = PRIOR / self.span
        r.append(pr)
        J.append(Jp)
        return np.concatenate([np.ravel(x) for x in r]), np.vstack(J)

    def solve(self, q, obs, paws, iters: int = 25):
        """LM from qpos ``q``; ``obs`` (L, 3) m (NaN unseen); ``paws`` {paw: (mode, xyz m)}."""
        mj, m, M = mujoco(), self.m, self.M
        ok = np.isfinite(obs).all(1)
        q = np.array(q, float)
        r, J = self.residual(q, obs, ok, paws)
        cost = float(r @ r)
        lam = 1e-3
        for _ in range(iters):
            A = J.T @ J
            g = J.T @ r
            step = -np.linalg.solve(A + lam * (np.diag(np.diag(A)) + 1e-9 * np.eye(len(A))), g)
            qn = q.copy()
            mj.mj_integratePos(m, qn, step, 1.0)
            qn[self.hq] = np.clip(qn[self.hq], M.lo, M.hi)
            rn, Jn = self.residual(qn, obs, ok, paws)
            cn = float(rn @ rn)
            if cn < cost:
                q, r, J, cost = qn, rn, Jn, cn
                lam = max(lam * 0.3, 1e-9)
                if np.abs(step).max() < 1e-7:
                    break
            else:
                lam *= 10.0
                if lam > 1e8:
                    break
        self.d.qpos[:] = q
        mj.mj_kinematics(m, self.d)
        err = np.linalg.norm(self.d.site_xpos[self.sites[ok]] - obs[ok], axis=1)
        return q, float(np.sqrt(np.mean(err ** 2))) * 1000.0 if len(err) else np.nan


def calibrate(M: RodentMJ, names, xyz_mm, rounds: int = 4, iters: int = 25, max_shift_mm: float = 60.0) -> dict:
    """Re-place the label sites on their bodies from frames ``xyz_mm`` (S, L, 3) (a sample spread
    over a recording), STAC-style; changes ``M.model.site_pos`` in place, to the placement that
    fitted best (rounds can overshoot). Returns {"rms_mm": [the model's, then per round; median],
    "shift_mm": {label: moved}}. A label is not moved more than ``max_shift_mm`` from the model's
    place."""
    fit = PoseFitter(M, names)
    col = [list(names).index(n) for n in fit.names]
    m = M.model
    orig = m.site_pos[fit.sites].copy()
    free = {p: ("free", None) for p in PAWS}
    hist, best = [], None
    for _ in range(rounds + 1):
        fit.rest = fit._stance_sites()
        Q, E = [], []
        for x in xyz_mm:
            obs = np.asarray(x, float)[col] / 1000.0
            q0 = fit.start(obs)
            if q0 is None:
                continue
            q, e = fit.solve(q0, obs, free, iters)
            Q.append((q, obs))
            E.append(e)
        hist.append(round(float(np.median(E)), 2))
        if best is None or hist[-1] < best[0]:                # the round that fits best is kept
            best = (hist[-1], m.site_pos[fit.sites].copy())
        if len(hist) == rounds + 1:
            break
        # each label: the median of its observed place in its body's frame
        loc = [[] for _ in fit.sites]
        for q, obs in Q:
            fit.d.qpos[:] = q
            mujoco().mj_kinematics(m, fit.d)
            for k, sid in enumerate(fit.sites):
                if not np.isfinite(obs[k]).all():
                    continue
                b = m.site_bodyid[sid]
                R = fit.d.xmat[b].reshape(3, 3)
                loc[k].append(R.T @ (obs[k] - fit.d.xpos[b]))
        for k, sid in enumerate(fit.sites):
            if len(loc[k]) >= 10:
                new = np.median(np.array(loc[k]), axis=0)
                d = new - orig[k]
                n = np.linalg.norm(d) * 1000
                if n > max_shift_mm:
                    new = orig[k] + d * (max_shift_mm / n)
                m.site_pos[sid] = new
    m.site_pos[fit.sites] = best[1]
    shift = {n: round(float(np.linalg.norm(m.site_pos[s] - orig[k]) * 1000), 1) for k, (n, s) in
             enumerate(zip(fit.names, fit.sites, strict=True))}
    return {"rms_mm": hist, "shift_mm": shift}


def fit_trajectory(M: RodentMJ, names, xyz_mm, targets=None, iters: int = 25, progress=None) -> dict:
    """All joints, frame by frame: ``xyz_mm`` (T, L, 3) labels in ``names`` order; ``targets``
    per frame {paw: (mode, xyz mm)} (None: paws planted). Returns {"qpos" (T, nq), "rms_mm" (T,),
    "modes" [{paw: mode}]}; frames with under 4 labels are NaN."""
    fit = PoseFitter(M, names)
    col = [list(names).index(n) for n in fit.names]
    T = len(xyz_mm)
    Q = np.full((T, M.model.nq), np.nan)
    rms = np.full(T, np.nan)
    modes = []
    prev = None
    for t in range(T):
        obs = np.asarray(xyz_mm[t], float)[col] / 1000.0
        tg = (targets[t] if targets is not None else None) or {p: ("plant", None) for p in PAWS}
        paws = {p: (md, None if x is None else np.asarray(x, float) / 1000.0) for p, (md, x) in tg.items()}
        modes.append({p: md for p, (md, _) in tg.items()})
        q0 = prev if prev is not None else fit.start(obs)
        if q0 is None:
            prev = None
            continue
        q, e = fit.solve(q0, obs, paws, iters)
        if prev is not None and e > 25.0:              # lost: start again from the stance on the labels
            q1 = fit.start(obs)
            if q1 is not None:
                q1, e1 = fit.solve(q1, obs, paws, iters)
                if e1 < e:
                    q, e = q1, e1
        Q[t], rms[t], prev = q, e, q
        if progress and t % 200 == 0:
            progress(t, T)
    return {"qpos": Q, "rms_mm": rms, "modes": modes}


__all__ = ["PRIOR", "PoseFitter", "SIGMA_MM", "calibrate", "fit_trajectory"]
