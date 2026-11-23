"""
mufasa.biomech.optimise
=======================

Patch 122lk: whole movements optimised for tracking, balance and effort (stage 4 of the
biomechanical model).

Stages 2 and 3 take the motion as fitted, frame by frame, and ask what forces it needs. Where
the fit is off, the forces come out impossible. On K7 the gait's inferred paws leave the body
unbalanced (a residual of 41 % of the weight) and fold the knees to their stops. Here the motion
is changed too: every joint of every frame in a window, the paws' forces and (with muscles)
every hindlimb muscle's activation are solved together. The costs:

* **tracking**: the labels (σ 4 mm), as in the fit;
* **contacts**, on the gait's stance/swing timing:
  * a paw in stance stays on the floor (σ 1 mm) and does not slide (σ 1 mm a frame). Where it
    is planted is free, and it is placed where it balances;
  * a paw in swing stays above the floor and bears nothing;
* **physics**: the motion's accelerations, by inverse dynamics, must be supplied by the
  ground's forces at the planted paws, which push but do not pull and stay within friction
  (μ 0.8). The residual allowed at the root is σ 2 % of the weight;
* **effort**: each joint's net torque (τ / 50 mN·m, weighted 0.3);
* **muscles**, given a muscle set:
  * the leg torques are produced by the muscles (σ 2 mN·m);
  * their activations are kept low (Σa²);
  * they change smoothly from frame to frame (σ 0.1 a frame), so loads stay correlated over
    time;
* **posture**: a weak pull towards the stance, and every joint within its range (bounds).

The problem couples each frame to its neighbours (velocities and accelerations by central
differences; the planted paws; the activations' rates), so every Gauss–Newton step (scipy's
trust-region least squares, a sparse Jacobian) propagates information both ways through the
window. That does the work of iterated forward and backward sweeps (iLQR / DDP) over the
whole window at once. Joint Jacobians come from MuJoCo (``mj_jacSite``, ``mjd_inverseFD``).
The paws' forces and the muscles enter linearly, given the pose.

Long recordings are taken in windows (:data:`WINDOW` frames, :data:`STEP` apart), one after
another. Each holds the one before's last :data:`PIN` solved frames, so velocities, accelerations
and activations carry on across the joins.
"""
from __future__ import annotations

import numpy as np

from mufasa.biomech.mjmodel import RodentMJ, mujoco
from mufasa.pose3d.gait import PAWS

WINDOW, STEP = 40, 30          # frames per window, frames between windows
PIN = 2                        # frames each window holds from the one before
SIG = {"label_mm": 4.0, "sole_mm": 1.0, "slide_mm": 1.0, "root": 0.02, "muscle_Nm": 0.002,
       "rate": 0.1, "smooth_rad": 0.1, "smooth_root_m": 0.002, "anchor_mm": 3.0}
TAU_REF, EFFORT_W = 0.05, 0.3  # N·m; weight of the net torques' cost
PRIOR = 0.3                    # pull towards the stance, per joint, per its range (as the fit)
MU = 0.8
F_REG = 0.02                   # a little cost on the ground forces (N), so their split is defined


class _Window:
    """One window: its variables, residuals and sparse Jacobian."""

    def __init__(self, M: RodentMJ, sites, obs, Qref, modes, fps, use_muscles, physics=True,
                 label_sigma=None, smooth_rad=None, anchors=None):
        self.M, self.m = M, M.model
        self.d = M.data()
        self.sites = np.asarray(sites)
        self.obs = obs                                  # (T, L, 3) m, NaN unseen
        self.Qref = np.array(Qref, float)
        self.modes = modes
        self.T = len(Qref)
        self.dt = 1.0 / fps
        nv = self.m.nv
        self.nv = nv
        self.hq = np.array([M.qadr[n] for n in M.hinge])
        self.hv = np.array([M.vadr[n] for n in M.hinge])
        self.span = M.hi - M.lo
        self.W = M.mass_kg * 9.81
        self.down = np.array([[(modes[t] or {}).get(p) in ("stance", "plant") for p in PAWS] for t in range(self.T)])
        self.mt = None
        if use_muscles and M.muscles is not None:
            from mufasa.biomech.muscle import LEG_DOFS, MuscleTable
            self.mt = MuscleTable(M)
            self.leg = {M.vadr[d.format(S=S)] for S in "LR" for d in LEG_DOFS}
        self.nm = 0 if self.mt is None else len(self.mt.names)
        self.nb = nv + 12 + self.nm                     # variables per frame
        self.physics = physics
        self.sl = (label_sigma or SIG["label_mm"]) / 1000.0
        self.ss = smooth_rad or SIG["smooth_rad"]
        self.anchors = None if anchors is None else np.asarray(anchors, float)   # (T, 4, 2) m, NaN: none
        self.lw = np.ones(obs.shape[:2])                # per label and frame: robust weights
        self.jac = np.zeros((3, nv))

    # variables ---------------------------------------------------------------------------
    def pack(self, F, A):
        x = np.zeros((self.T, self.nb))
        x[:, self.nv:self.nv + 12] = F.reshape(self.T, 12)
        if self.nm:
            x[:, self.nv + 12:] = A
        return x.ravel()

    def unpack(self, x):
        x = x.reshape(self.T, self.nb)
        mj = mujoco()
        Q = self.Qref.copy()
        for t in range(self.T):
            mj.mj_integratePos(self.m, Q[t], x[t, :self.nv], 1.0)
        F = x[:, self.nv:self.nv + 12].reshape(self.T, 4, 3)
        A = x[:, self.nv + 12:] if self.nm else None
        return Q, F, A

    def bounds(self):
        lo = np.full((self.T, self.nb), -np.inf)
        hi = np.full((self.T, self.nb), np.inf)
        hq = self.Qref[:, self.hq]
        lo[:, self.hv] = np.minimum(self.M.lo - hq, -1e-9)
        hi[:, self.hv] = np.maximum(self.M.hi - hq, 1e-9)
        for t in range(self.T):
            for k in range(4):
                c = self.nv + 3 * k
                if self.down[t, k]:
                    lo[t, c + 2] = 0.0
                else:
                    lo[t, c:c + 3], hi[t, c:c + 3] = -1e-9, 1e-9
        if self.nm:
            lo[:, self.nv + 12:] = 0.0
            hi[:, self.nv + 12:] = 1.0
        return lo.ravel(), hi.ravel()

    # residuals ---------------------------------------------------------------------------
    def evaluate(self, x, want_jac=True):
        import scipy.sparse as sp
        mj, m, d, M = mujoco(), self.m, self.d, self.M
        Q, F, A = self.unpack(x)
        T, nv, nb, dt = self.T, self.nv, self.nb, self.dt
        rows, cols, vals, res = [], [], [], []
        nrow = [0]

        tags = []

        def add(r, blocks=(), tag=""):
            """Residuals ``r`` with their Jacobian ``blocks``: [(frame, column offset, B)], B a
            dense array or (rows, cols, values)."""
            r = np.atleast_1d(np.asarray(r, float))
            i0 = nrow[0]
            res.append(r)
            if want_jac:
                for t, c0, B in blocks:
                    if isinstance(B, tuple):
                        ii, jj, vv = B
                    else:
                        B = np.atleast_2d(B)
                        ii, jj = np.nonzero(B)
                        vv = B[ii, jj]
                    rows.append(ii + i0)
                    cols.append(jj + t * nb + c0)
                    vals.append(vv)
            nrow[0] += len(r)
            tags.append((tag, len(r)))

        diff = lambda a, b: _diff(m, a, b)            # noqa: E731
        sl = self.sl
        ss, sd = SIG["sole_mm"] / 1000.0, SIG["slide_mm"] / 1000.0
        sF, sT = SIG["root"] * self.W, SIG["root"] * self.W * 0.05
        mids, Jmid = [], []
        nh = len(self.hv)
        Jpost = (np.arange(nh), self.hv, PRIOR / self.span)
        for t in range(T):
            d.qpos[:] = Q[t]
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            # labels
            ok = np.flatnonzero(np.isfinite(self.obs[t]).all(1))
            if len(ok):
                w = np.repeat(np.sqrt(self.lw[t, ok]) / sl, 3)
                JL = np.zeros((3 * len(ok), nv))
                if want_jac:
                    for i, k in enumerate(ok):
                        mj.mj_jacSite(m, d, self.jac, None, int(self.sites[k]))
                        JL[3 * i:3 * i + 3] = self.jac
                add((d.site_xpos[self.sites[ok]] - self.obs[t, ok]).ravel() * w, [(t, 0, JL * w[:, None])], "labels")
            # posture
            pr = PRIOR * (Q[t, self.hq] - M.q_stance) / self.span
            add(pr, [(t, 0, Jpost)], "posture")
            # soles
            mt, Jt = [], []
            for k, p in enumerate(PAWS):
                h, s = M.soles[p]
                Jh, Js = np.zeros((3, nv)), np.zeros((3, nv))
                if want_jac or self.physics:                # the ground's forces act through them
                    mj.mj_jacSite(m, d, Jh, None, h)
                    mj.mj_jacSite(m, d, Js, None, s)
                ph, ps = d.site_xpos[h].copy(), d.site_xpos[s].copy()
                mt.append((ph + ps) / 2)
                Jt.append((Jh + Js) / 2)
                z0 = M.sole_z[p]
                if self.down[t, k]:
                    add([(ph[2] - z0[0]) / ss, (ps[2] - z0[1]) / ss], [(t, 0, np.vstack([Jh[2], Js[2]]) / ss)], "soles")
                else:
                    pen = np.array([min(ph[2] - z0[0], 0.0), min(ps[2] - z0[1], 0.0)]) / ss
                    Jpen = np.vstack([Jh[2] * (pen[0] < 0), Js[2] * (pen[1] < 0)]) / ss
                    add(pen, [(t, 0, Jpen)], "soles")
            mids.append(mt)
            Jmid.append(Jt)
            # 122lo: a planted paw that was seen stays where it was seen
            if self.anchors is not None:
                sa = SIG["anchor_mm"] / 1000.0
                for k in range(4):
                    if self.down[t, k] and np.isfinite(self.anchors[t, k]).all():
                        add((mt[k][:2] - self.anchors[t, k]) / sa, [(t, 0, Jt[k][:2] / sa)], "seen paws")
            # planted paws do not slide
            if t > 0:
                for k in range(4):
                    if self.down[t, k] and self.down[t - 1, k]:
                        add((mids[t][k][:2] - mids[t - 1][k][:2]) / sd,
                            [(t, 0, Jmid[t][k][:2] / sd), (t - 1, 0, -Jmid[t - 1][k][:2] / sd)], "sliding")
            # ground forces: friction, and a little cost
            for k in range(4):
                if not self.down[t, k]:
                    continue
                f = F[t, k]
                tn = np.linalg.norm(f[:2])
                viol = tn - MU * f[2]
                Jf = np.zeros((1, nb))
                if viol > 0 and tn > 0:
                    Jf[0, nv + 3 * k: nv + 3 * k + 2] = f[:2] / tn / sF
                    Jf[0, nv + 3 * k + 2] = -MU / sF
                add([max(viol, 0.0) / sF], [(t, 0, Jf)], "friction")
                add(F_REG * f, [(t, nv + 3 * k, F_REG * np.eye(3))], "forces")
            # muscles: low, and smooth
            if self.nm:
                add(A[t], [(t, nv + 12, np.eye(self.nm))], "activation")
                if t > 0:
                    sr = SIG["rate"]
                    add((A[t] - A[t - 1]) / sr, [(t, nv + 12, np.eye(self.nm) / sr),
                                                  (t - 1, nv + 12, -np.eye(self.nm) / sr)], "rate")

        # physics, at the interior frames
        DfDq, DfDv, DfDa = (np.zeros((nv, nv)) for _ in range(3))
        wr = np.r_[np.full(3, 1 / sF), np.full(3, 1 / sT)]
        self.last = {"tau": np.full((T, len(self.hv)), np.nan), "res_f": np.full((T, 3), np.nan),
                     "res_t": np.full((T, 3), np.nan), "V": np.full((T, nv), np.nan), "A": np.full((T, nv), np.nan)}
        wroot = 1.0 / np.r_[np.full(3, SIG["smooth_root_m"]), np.full(3, self.ss)]
        Jroot = (np.arange(6), np.arange(6), wroot)
        Jroot2 = (Jroot[0], Jroot[1], -2 * wroot)
        Js = (np.arange(nh), self.hv, np.full(nh, 1.0 / self.ss))
        Js2 = (Js[0], Js[1], -2 * Js[2])
        for t in range(1, T - 1):
            dm, dp = diff(Q[t - 1], Q[t]), diff(Q[t], Q[t + 1])
            v, a = (dm + dp) / (2 * dt), (dp - dm) / dt ** 2
            # joints change smoothly: their second difference (the model has redundant joints,
            # the neck's twists among them, that nothing else holds still)
            add((dp - dm)[self.hv] / self.ss, [(t - 1, 0, Js), (t, 0, Js2), (t + 1, 0, Js)], "smooth")
            if not self.physics:
                # kinematics only: the root moves smoothly too, and that is all
                add((dp - dm)[:6] * wroot, [(t - 1, 0, Jroot), (t, 0, Jroot2), (t + 1, 0, Jroot)], "smooth")
                continue
            d.qpos[:], d.qvel[:], d.qacc[:] = Q[t], v, a
            mj.mj_inverse(m, d)
            f = d.qfrc_inverse.copy()
            Jc = [Jmid[t][k] for k in range(4)]
            gen = f - sum(Jc[k].T @ F[t, k] for k in range(4))
            if want_jac:
                mj.mjd_inverseFD(m, d, 1e-6, 0, DfDq, DfDv, DfDa, None, None, None, None)
                Gq, Gv, Ga = DfDq.T, DfDv.T, DfDa.T
                Jm1 = -Gv / (2 * dt) + Ga / dt ** 2
                J0 = Gq - 2 * Ga / dt ** 2
                Jp1 = Gv / (2 * dt) + Ga / dt ** 2
                Jf = np.hstack([-Jc[k].T for k in range(4)])           # (nv, 12)
            musc = np.zeros(nv)
            Jam = None
            if self.nm:
                L, Ld = d.ten_length[self.mt.tid], None
                Jt = _tendon_J(m, d)[self.mt.tid]
                Ld = Jt @ v
                lt, vt, ca = self.mt.fibre(L, Ld)
                from mufasa.biomech.muscle import f_active, f_passive, f_velocity
                gain = self.mt.f_max * f_active(lt) * f_velocity(vt) * ca
                pas = self.mt.f_max * f_passive(lt) * ca
                musc = -Jt.T @ (A[t] * gain + pas)
                Jam = -Jt.T * gain                                      # (nv, nm)
            tau = gen - musc
            self.last["V"][t], self.last["A"][t] = v, a
            self.last["tau"][t] = gen[self.hv]
            self.last["res_f"][t], self.last["res_t"][t] = gen[:3], gen[3:6]
            # root: the ground must explain it
            blocks = []
            if want_jac:
                blocks = [(t - 1, 0, Jm1[:6] * wr[:, None]), (t, 0, J0[:6] * wr[:, None]),
                          (t + 1, 0, Jp1[:6] * wr[:, None]), (t, nv, Jf[:6] * wr[:, None])]
            add(gen[:6] * wr, blocks, "root")
            # hinges: muscles produce the legs' torques; the rest, and any shortfall, cost effort
            w = np.full(nv, EFFORT_W / TAU_REF)
            if self.nm:
                for j in self.leg:
                    w[j] = 1.0 / SIG["muscle_Nm"]
            hv = self.hv
            blocks = []
            if want_jac:
                blocks = [(t - 1, 0, Jm1[hv] * w[hv, None]), (t, 0, J0[hv] * w[hv, None]),
                          (t + 1, 0, Jp1[hv] * w[hv, None]), (t, nv, Jf[hv] * w[hv, None])]
                if self.nm:
                    blocks.append((t, nv + 12, -Jam[hv] * w[hv, None]))
            leg = np.array([j in getattr(self, "leg", ()) for j in hv]) if self.nm else np.zeros(len(hv), bool)
            order = np.r_[np.flatnonzero(~leg), np.flatnonzero(leg)]
            blocks = [(tt, c0, B[order]) for tt, c0, B in blocks]
            add((tau[hv] * w[hv])[order[:(~leg).sum()]], [(tt, c0, B[:(~leg).sum()]) for tt, c0, B in blocks], "effort")
            if leg.any():
                add((tau[hv] * w[hv])[order[(~leg).sum():]], [(tt, c0, B[(~leg).sum():]) for tt, c0, B in blocks], "muscle torque")
        r = np.concatenate(res)
        self.tags = tags
        if not want_jac:
            return r, None
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(r), T * nb))
        return r, J


def _label_err(w, Q, obs):
    """(T, L) distance of each seen label from its site, m (NaN unseen)."""
    mj, d = mujoco(), w.d
    E = np.full(obs.shape[:2], np.nan)
    for t in range(len(Q)):
        d.qpos[:] = Q[t]
        mj.mj_kinematics(w.m, d)
        ok = np.isfinite(obs[t]).all(1)
        E[t, ok] = np.linalg.norm(d.site_xpos[w.sites[ok]] - obs[t, ok], axis=1)
    return E


def _breakdown(r, tags) -> dict:
    out, i = {}, 0
    for tag, n in tags:
        out[tag] = out.get(tag, 0.0) + float(r[i:i + n] @ r[i:i + n])
        i += n
    return {k: round(v, 1) for k, v in out.items()}


def _lm(evaluate, x, lo, hi, iters: int, progress=None):
    """Projected Levenberg–Marquardt with a sparse direct solve: each step solves the whole
    window's normal equations at once. Variables at a bound and pushed outwards are held."""
    import scipy.sparse as sp
    x = np.clip(x, lo, hi)
    r, J = evaluate(x, True)
    cost = float(r @ r)
    lam = 1e-2
    n_eval = 1
    for _ in range(iters):
        g = J.T @ r
        A = (J.T @ J).tocsc()
        held = ((x <= lo + 1e-12) & (g > 0)) | ((x >= hi - 1e-12) & (g < 0))
        free = np.flatnonzero(~held)
        Af = A[free][:, free]
        dg = Af.diagonal()
        improved = False
        while lam < 1e10:
            K = Af + sp.diags(lam * (dg + 1e-6 * (dg.max() if len(dg) else 1.0)))
            step = np.zeros_like(x)
            step[free] = -_solve_sym(K, g[free])
            xn = np.clip(x + step, lo, hi)
            rn, _ = evaluate(xn, False)
            n_eval += 1
            cn = float(rn @ rn)
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-12)
                rn, Jn = evaluate(xn, True)
                x, r, J, cost = xn, rn, Jn, cn
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 4.0
        if progress:
            progress(cost)
        if not improved or rel < 1e-5:
            break
    return x, r, n_eval


def _solve_sym(K, b):
    """K x = b for the window's normal equations: symmetric, positive definite and banded
    (frames couple only to their near neighbours), so a banded Cholesky (LAPACK) unless the
    band is too wide for it to pay."""
    from scipy.linalg import LinAlgError, solveh_banded
    from scipy.sparse.linalg import spsolve
    C = K.tocoo()
    up = C.col >= C.row
    r, c, v = C.row[up], C.col[up], C.data[up]
    bw = int((c - r).max()) if len(r) else 0
    n = K.shape[0]
    if bw and bw * bw * n < 4e9:
        ab = np.zeros((bw + 1, n))
        ab[bw + r - c, c] = v
        try:
            return solveh_banded(ab, b, check_finite=False)
        except LinAlgError:
            pass
    return spsolve(K.tocsc(), b, permc_spec="NATURAL")


def _diff(m, a, b):
    v = np.zeros(m.nv)
    mujoco().mj_differentiatePos(m, v, 1.0, a, b)
    return v


def _tendon_J(m, d):
    J = np.zeros((m.ntendon, m.nv))
    for i in range(m.ntendon):
        a, nz = m.ten_J_rowadr[i], m.ten_J_rownnz[i]
        J[i, m.ten_J_colind[a:a + nz]] = d.ten_J[a:a + nz]
    return J


def optimise_window(M: RodentMJ, names, xyz_mm, Qref, modes, fps: float, *, F0=None, A0=None,
                    muscles: bool = True, iters: int = 40, pinned: int = 0, physics: bool = True,
                    label_sigma: float | None = None, robust: float | None = None,
                    smooth_rad: float | None = None, anchors=None) -> dict:
    """One window. ``xyz_mm`` (T, L, 3) labels; ``Qref`` (T, nq) the fitted poses to start from;
    ``modes`` per frame {paw: "stance" | "swing" | ...}; ``F0`` (T, 4, 3) N and ``A0`` (T, n)
    starting forces and activations; the first ``pinned`` frames' poses and activations held as
    given (the window before's, so the motion carries on smoothly). ``physics=False``: the
    kinematics alone (labels, smooth joints, posture, the floor; :mod:`.smoother`).
    ``label_sigma`` (mm): the labels' σ (default :data:`SIG`). ``robust`` (σ): labels further
    off than this are down-weighted (Huber, iterated twice) and reported in ``label_err_mm``.
    Returns poses, forces, activations, torques, the residual and the costs before and after."""
    from mufasa.biomech.fit import PoseFitter
    fitter = PoseFitter(M, names)
    col = [list(names).index(n) for n in fitter.names]
    obs = np.asarray(xyz_mm, float)[:, col] / 1000.0
    w = _Window(M, fitter.sites, obs, Qref, modes, fps, muscles and physics, physics, label_sigma, smooth_rad,
                anchors)
    T = w.T
    F0 = np.zeros((T, 4, 3)) if F0 is None else np.nan_to_num(np.asarray(F0, float))
    F0 = F0 * w.down[:, :, None]
    A0 = None if not w.nm else (np.full((T, w.nm), 0.05) if A0 is None else np.clip(np.nan_to_num(A0, nan=0.05), 0, 1))
    x0 = w.pack(F0, A0)
    lo, hi = w.bounds()
    if pinned:
        lo, hi = lo.reshape(T, w.nb), hi.reshape(T, w.nb)
        xv = x0.reshape(T, w.nb)
        lo[:pinned, :w.nv], hi[:pinned, :w.nv] = -1e-12, 1e-12
        if w.nm:
            lo[:pinned, w.nv + 12:] = xv[:pinned, w.nv + 12:] - 1e-12
            hi[:pinned, w.nv + 12:] = xv[:pinned, w.nv + 12:] + 1e-12
        lo, hi = lo.ravel(), hi.ravel()
    x0 = np.clip(x0, lo, hi)
    r0 = w.evaluate(x0, False)[0]
    parts0 = _breakdown(r0, w.tags)
    xs, rs, nfev = _lm(w.evaluate, x0, lo, hi, iters)
    if robust:
        for _ in range(2):
            err = _label_err(w, w.unpack(xs)[0], obs)
            z = err / w.sl
            w.lw = 1.0 / (1.0 + (z / robust) ** 2)          # Cauchy: a label far off hardly pulls
            w.lw[~np.isfinite(z)] = 1.0
            xs, rs, n2 = _lm(w.evaluate, xs, lo, hi, iters)
            nfev += n2
    Q, F, A = w.unpack(xs)
    parts = _breakdown(w.evaluate(xs, False)[0], w.tags)
    out = dict(w.last)
    # label error per frame
    d = w.d
    rms = np.full(T, np.nan)
    for t in range(T):
        d.qpos[:] = Q[t]
        mujoco().mj_kinematics(w.m, d)
        ok = np.isfinite(obs[t]).all(1)
        if ok.any():
            rms[t] = float(np.sqrt(np.mean(np.sum((d.site_xpos[w.sites[ok]] - obs[t, ok]) ** 2, 1)))) * 1000
    out.update({"qpos": Q, "grf": F, "act": A, "down": w.down, "rms_mm": rms,
                "label_err_mm": _label_err(w, Q, obs) * 1000.0, "label_weight": w.lw.copy(), "cost0": float(r0 @ r0),
                "cost": float(rs @ rs), "nfev": int(nfev), "muscles": w.nm > 0,
                "parts0": parts0, "parts": parts})
    return out


def optimise_trajectory(M: RodentMJ, names, xyz_mm, Qref, modes, fps: float, *, F0=None, A0=None,
                        muscles: bool = True, window: int = WINDOW, step: int = STEP, iters: int = 40,
                        progress=None, anchors=None, **kw) -> dict:
    """:func:`optimise_window` over windows of ``window`` frames, ``step`` apart, one after
    another: each starts from the one before's last solved frames, held (:data:`PIN` of them),
    and its own frames replace the one before's from there on."""
    T = len(Qref)
    starts = list(range(0, max(T - window, 0) + 1, step))
    if not starts or starts[-1] + window < T:
        starts.append(max(T - window, 0))
    keys = ("qpos", "grf", "act", "tau", "res_f", "res_t", "V", "A", "rms_mm", "label_err_mm", "label_weight")
    out = {}
    Qref = np.array(Qref, float)
    A0 = None if A0 is None else np.array(A0, float)
    for n, a in enumerate(starts):
        b = min(a + window, T)
        pin = 0
        if n > 0:
            pin = PIN
            Qref[a:a + pin] = out["qpos"][a:a + pin]
            if out.get("act") is not None and A0 is not None:
                A0[a:a + pin] = out["act"][a:a + pin]
        o = optimise_window(M, names, xyz_mm[a:b], Qref[a:b], modes[a:b], fps,
                            F0=None if F0 is None else F0[a:b], A0=None if A0 is None else A0[a:b],
                            muscles=muscles, iters=iters, pinned=pin,
                            anchors=None if anchors is None else anchors[a:b], **kw)
        if not out:
            for k in keys:
                if o.get(k) is not None:
                    out[k] = np.full((T,) + o[k].shape[1:], np.nan)
            out.update({"down": np.zeros((T, 4), bool), "cost0": [], "cost": [], "muscles": o["muscles"]})
        # the held frames' physics is this window's (it sees the frames after them)
        first = a + (pin - 1 if pin else 0)
        sl = slice(first - a, b - a)
        for k in keys:
            if o.get(k) is not None:
                out[k][first:b] = o[k][sl]
        out["down"][first:b] = o["down"][sl]
        out["cost0"].append(o["cost0"])
        out["cost"].append(o["cost"])
        out.setdefault("parts0", []).append(o["parts0"])
        out.setdefault("parts", []).append(o["parts"])
        if progress:
            progress(n + 1, len(starts))
    return out


def muscle_outputs(M: RodentMJ, Q, V, act, tau) -> dict:
    """For the optimised motion: each muscle's force, and what its leg's torques still need
    beyond the muscles (the reserve), in :func:`mufasa.biomech.muscle.static_optimisation`'s
    layout."""
    from mufasa.biomech.muscle import (
        LEG_DOFS,
        MuscleTable,
        f_active,
        f_passive,
        f_velocity,
        tendon_state,
    )
    mt = MuscleTable(M)
    T, n = len(Q), len(mt.names)
    frc = np.full((T, n), np.nan)
    lfib = np.full((T, n), np.nan)
    res = np.full((T, 2, len(LEG_DOFS)), np.nan)
    ask = np.full((T, 2, len(LEG_DOFS)), np.nan)
    hidx = {h: k for k, h in enumerate(M.hinge)}
    for t in range(T):
        if not (np.isfinite(Q[t]).all() and np.isfinite(V[t]).all() and np.isfinite(act[t]).all()
                and np.isfinite(tau[t]).all()):
            continue
        L, Ld, J = tendon_state(M, Q[t], V[t])
        lt, vt, ca = mt.fibre(L[mt.tid], Ld[mt.tid])
        F = mt.f_max * (act[t] * f_active(lt) * f_velocity(vt) + f_passive(lt)) * ca
        frc[t], lfib[t] = F, lt
        gen = -J[mt.tid].T @ F
        for si, S in enumerate("LR"):
            dofs = [d.format(S=S) for d in LEG_DOFS]
            ask[t, si] = [tau[t, hidx[d]] for d in dofs]
            res[t, si] = [tau[t, hidx[d]] - gen[M.vadr[d]] for d in dofs]
    return {"names": mt.names, "act": np.asarray(act, float), "force": frc, "reserve": res, "asked": ask,
            "fibre": lfib, "f_max": mt.f_max}


__all__ = ["SIG", "STEP", "WINDOW", "muscle_outputs", "optimise_trajectory", "optimise_window"]
