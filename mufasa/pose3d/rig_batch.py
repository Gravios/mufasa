"""
mufasa.pose3d.rig_batch
=======================

Patch 122ky: a body-model rig posed to 3-D markers for many frames at once, in PyTorch on a
GPU (CUDA) or the CPU — the batched form of :meth:`mufasa.pose3d.scene_models.PosedRig.fit_frame`.

* the same unknowns (heading, position, height, the rig's posture values within their
  limits), residuals (marker error / 4 mm, the weak posture prior, height / 30 mm) and cold
  starts (both headings, flat or rearing) as the per-frame fit;
* the rig's forward kinematics and the markers in torch: ``vmap`` over frames of ``jacfwd``
  of one frame's residuals gives every frame's Jacobian in one pass;
* bounded Levenberg–Marquardt, each frame with its own damping, steps projected into the
  bounds; every start of every frame solved together and the best kept per frame;
* keyframes (every 8th frame, and the first after a gap) from every cold start; the frames
  between from the nearest keyframe's solution, as the per-frame fit starts from a fitted
  neighbour, with the cold starts again where that is clearly worse;
* then neighbours: frames still off by more than ``good_mm`` start from the frames next to
  them and keep whichever fits best (smooth, and no heading flips between frames);
* 122le: the floor probes' residuals and the clipped marker heights of
  :mod:`mufasa.pose3d.rig_floor`, as the per-frame fit.

Float64; chunks of frames so the memory stays bounded. ``fit_rig_frames`` returns per-frame
parameters and marker RMS (mm), as the per-frame fit caches them.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    from torch.func import jacfwd, vmap
except Exception:  # noqa: BLE001
    torch = None

DEG = math.pi / 180.0


def available() -> bool:
    return torch is not None


def device_name(spec: str = "auto") -> str:
    """"auto" → "cuda" when a CUDA GPU works, else "cpu"."""
    if torch is None:
        return "none"
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu" if spec in ("cpu", "torch-cpu") else spec


class RigProblem:
    """The rig, its markers and the fit's terms as tensors (one device, float64)."""

    def __init__(self, posed, device: str = "auto") -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        self.device = device_name(device)
        dt, dev = torch.float64, self.device
        rig, mk = posed.rig, posed.mk
        self.posture = [p[0] for p in posed.posture]
        self.n = 4 + len(self.posture)
        pi = {n: 4 + k for k, n in enumerate(self.posture)}
        # each bone's local rotation: (yaw index, pitch index) into x, or none
        rot = {}
        for g in ("lum", "tho", "cer"):
            for b in rig.groups.get(g, ()):
                rot[b] = (pi.get(f"{g}_yaw"), pi.get(f"{g}_pitch"))
        rot[rig.profile["head"]] = (pi.get("head_yaw"), pi.get("head_pitch"))
        for g, yaw in (("tail_p", "tail_yaw_p"), ("tail_d", "tail_yaw_d")):
            for b in rig.groups.get(g, ()):
                rot[b] = (pi.get(yaw), pi.get("tail_pitch"))
        # only the bones the markers depend on, and their ancestors (in the rig's order)
        from mufasa.pose3d.rig_floor import floor_probes, trunk_vertices
        probes, trunk = floor_probes(rig), trunk_vertices(rig)
        need = set()
        groups = [np.flatnonzero(rig.Wd[trunk].sum(0) > 0)]        # 122le: the trunk's bones (floor)
        for n in posed.names:
            m = mk.spec[n]
            groups.append(np.flatnonzero(rig.W[list(m[1])].sum(0) > 0) if m[0] == "vert" else [m[1]])
        for bones in groups:
            for b in bones:
                b = int(b)
                while b >= 0 and b not in need:
                    need.add(b)
                    b = int(rig.parent[b])
        keep = sorted(need)
        remap = {b: k for k, b in enumerate(keep)}
        self.bone_rot = [rot.get(rig.names[b]) for b in keep]
        self.parent = [remap[int(rig.parent[b])] if rig.parent[b] >= 0 else -1 for b in keep]
        self.heads = torch.as_tensor(rig.heads[keep], dtype=dt, device=dev)
        # 122lg: how far each joint slides along the spine per unit of "stretch"
        self.soff = torch.as_tensor(getattr(rig, "stretch_offset", np.zeros((rig.B, 3)))[keep], dtype=dt, device=dev)
        self.i_stretch = pi.get("stretch")
        self.n_bones = (len(keep), rig.B)
        self.pivot = torch.as_tensor(rig.pivot, dtype=dt, device=dev)
        self.i_rear, self.i_roll = pi.get("rear"), pi.get("roll")
        # markers, in the order of posed.names: vertex markers as the mean of their skinned
        # vertices (rows of ``avg``), bone markers as a point on their bone
        vw, vv, avg, bi, bp, kind = [], [], [], [], [], []
        M = len(posed.names)
        for j, n in enumerate(posed.names):
            m = mk.spec[n]
            if m[0] == "vert":
                idx = list(m[1])
                kind.append(0)
                for v in idx:
                    vw.append(rig.W[v].astype(float)[keep])
                    vv.append(rig.V[v])
                    row = np.zeros(M)
                    row[j] = 1.0 / len(idx)
                    avg.append(row)
                bi.append(0)
                bp.append(np.zeros(3))
            else:
                kind.append(1)
                bi.append(remap[int(m[1])])
                bp.append(np.asarray(m[2], float))
        if not vw:                                   # no vertex markers: one inert row
            vw, vv, avg = [np.zeros(len(keep))], [np.zeros(3)], [np.zeros(M)]
        self.kind = torch.as_tensor(kind, dtype=dt, device=dev)
        self.vw = torch.as_tensor(np.array(vw), dtype=dt, device=dev)            # (rows, B)
        self.vv = torch.as_tensor(np.array(vv), dtype=dt, device=dev)            # (rows, 3)
        self.avg = torch.as_tensor(np.array(avg).T, dtype=dt, device=dev)        # (M, rows)
        self.bi = torch.as_tensor(bi, dtype=torch.long, device=dev)
        self.bp = torch.as_tensor(np.array(bp), dtype=dt, device=dev)
        self.size = float(posed.size)
        # 122le: floor probes (rest vertices of the drawn mesh, their widening and weights)
        self.fw = torch.as_tensor(rig.Wd[probes].astype(float)[:, keep], dtype=dt, device=dev)
        self.fv = torch.as_tensor(rig.Vd[probes], dtype=dt, device=dev)
        self.fd = torch.as_tensor(rig.Dd[probes], dtype=dt, device=dev)
        self.i_widen = pi.get("widen")
        # the drawn mesh's trunk, for the exact lift after the solve
        self.aw = torch.as_tensor(rig.Wd[trunk].astype(float)[:, keep], dtype=dt, device=dev)
        self.av = torch.as_tensor(rig.Vd[trunk], dtype=dt, device=dev)
        self.ad = torch.as_tensor(rig.Dd[trunk], dtype=dt, device=dev)
        from mufasa.pose3d.rig_floor import FLOOR_SIGMA_MM, FLOOR_Z_MM
        self.floor_z, self.floor_sigma = float(FLOOR_Z_MM), float(FLOOR_SIGMA_MM)
        lo = np.r_[-np.inf, -np.inf, -np.inf, -60.0, [p[1] for p in posed.posture]]
        hi = np.r_[np.inf, np.inf, np.inf, 60.0, [p[2] for p in posed.posture]]
        half = np.array([(p[2] - p[1]) / 2 or 1.0 for p in posed.posture])
        mid = np.array([rig.prior_centre.get(p[0], (p[1] + p[2]) / 2) for p in posed.posture])
        self.lo_np, self.hi_np = lo, hi
        self.lo = torch.as_tensor(np.where(np.isfinite(lo), lo, -1e12), dtype=dt, device=dev)
        self.hi = torch.as_tensor(np.where(np.isfinite(hi), hi, 1e12), dtype=dt, device=dev)
        self.half = torch.as_tensor(half, dtype=dt, device=dev)
        self.mid = torch.as_tensor(mid, dtype=dt, device=dev)

    # -- one frame (vmapped) ---------------------------------------------- #
    @staticmethod
    def _rz(a):
        c, s = torch.cos(a), torch.sin(a)
        o, z = torch.ones_like(a), torch.zeros_like(a)
        return torch.stack([torch.stack([c, -s, z]), torch.stack([s, c, z]), torch.stack([z, z, o])])

    @staticmethod
    def _rx(a):
        c, s = torch.cos(a), torch.sin(a)
        o, z = torch.ones_like(a), torch.zeros_like(a)
        return torch.stack([torch.stack([o, z, z]), torch.stack([z, c, -s]), torch.stack([z, s, c])])

    @staticmethod
    def _ry(a):
        c, s = torch.cos(a), torch.sin(a)
        o, z = torch.ones_like(a), torch.zeros_like(a)
        return torch.stack([torch.stack([c, z, s]), torch.stack([z, o, z]), torch.stack([-s, z, c])])

    def markers(self, x):
        """World marker positions (M, 3) for one frame's parameters ``x`` (n,)."""
        return self._world(x)[0]

    def probe_z(self, x):
        """122le: heights (mm) of the floor probes for one frame's parameters."""
        return self._world(x)[1]

    def mesh_z(self, x):
        """122le: heights (mm) of every trunk vertex of the drawn mesh for one frame's parameters."""
        R, T = self._bones(x)
        return self._z(R, T, x, self.aw, self.av, self.ad)

    def _z(self, R, T, x, W, V, D):
        """Heights of skinned vertices (weights W, rest V, widening D): the heading does not
        change heights, so only the z rows of the blend."""
        V = V + (x[self.i_widen] * D if self.i_widen is not None else 0.0)
        Rz = W @ R[:, 2, :]
        return self.size * ((Rz * V).sum(1) + W @ T[:, 2]) + x[3]

    def _bones(self, x):
        """Each kept bone's world rotation (B, 3, 3) and translation (B, 3) in model units."""
        zero = x[0] * 0
        eye3 = torch.eye(3, dtype=x.dtype, device=x.device)
        Rs, ts = [], []
        for i, r in enumerate(self.bone_rot):
            if r is None:
                R = eye3 + zero
            else:
                yaw = x[r[0]] * DEG if r[0] is not None else zero
                pitch = x[r[1]] * DEG if r[1] is not None else zero
                R = self._rz(yaw) @ self._rx(pitch)
            t = self.heads[i] - R @ self.heads[i]
            if self.i_stretch is not None:
                t = t + x[self.i_stretch] * self.soff[i]
            p = self.parent[i]
            if p >= 0:
                R, t = Rs[p] @ R, Rs[p] @ t + ts[p]
            Rs.append(R)
            ts.append(t)
        rear = x[self.i_rear] * DEG if self.i_rear is not None else zero
        roll = x[self.i_roll] * DEG if self.i_roll is not None else zero
        Rg = self._rx(-rear) @ self._ry(roll)
        tg = self.pivot - Rg @ self.pivot
        R = torch.stack(Rs)                                       # (B, 3, 3)
        T = torch.stack(ts)                                       # (B, 3)
        R = Rg @ R
        T = (Rg @ T.T).T + tg
        return R, T

    def _world(self, x):
        """(markers (M, 3), floor-probe heights (P,)) for one frame's parameters."""
        R, T = self._bones(x)
        # vertex markers: linear blend of the bones' transforms
        Rv = torch.einsum("mb,bij->mij", self.vw, R)
        Tv = self.vw @ T
        pv = self.avg @ (torch.einsum("mij,mj->mi", Rv, self.vv) + Tv)
        pb = torch.einsum("mij,mj->mi", R[self.bi], self.bp) + T[self.bi]
        k = self.kind[:, None]
        P = self.size * ((1 - k) * pv + k * pb)
        c, s = torch.cos(x[0]), torch.sin(x[0])
        Px = c * P[:, 0] - s * P[:, 1] + x[1]
        Py = s * P[:, 0] + c * P[:, 1] + x[2]
        fz = self._z(R, T, x, self.fw, self.fv, self.fd)          # the floor probes
        return torch.stack([Px, Py, P[:, 2] + x[3]], 1), fz

    def residual(self, x, Y, ok):
        P, fz = self._world(x)
        r = ((P - Y) / 4.0) * ok[:, None]
        fl = torch.clamp(self.floor_z - fz, min=0.0) / self.floor_sigma
        return torch.cat([r.reshape(-1), 0.3 * (x[4:] - self.mid) / self.half, (x[3] / 30.0)[None], fl])

    def lift(self, x, Y, ok):
        """122le: raise each row ``x`` (N, n) in place so its lowest floor probe is on the floor
        (as :func:`mufasa.pose3d.rig_floor.lift`); returns the marker RMS (mm) after."""
        dt, dev = torch.float64, self.device
        xt = torch.as_tensor(x, dtype=dt, device=dev)
        if self.fv.shape[0]:
            for a in range(0, len(xt), 1024):                       # the whole trunk, in pieces
                fz = vmap(self.mesh_z)(xt[a:a + 1024])
                xt[a:a + 1024, 3] += torch.clamp(self.floor_z - fz.min(1).values, min=0.0)
        Yt = torch.as_tensor(np.nan_to_num(Y), dtype=dt, device=dev)
        okt = torch.as_tensor(ok, dtype=dt, device=dev)
        d2 = ((vmap(self.markers)(xt) - Yt) ** 2).sum(2)
        x[:] = xt.cpu().numpy()
        return torch.sqrt((d2 * okt).sum(1) / torch.clamp(okt.sum(1), min=1)).cpu().numpy()

    # -- many frames ------------------------------------------------------ #
    def solve(self, x0, Y, ok, *, iters: int = 40, lam0: float = 1e-3):
        """Bounded Levenberg–Marquardt for every row: ``x0`` (N, n), ``Y`` (N, M, 3), ``ok`` (N, M)."""
        dt, dev = torch.float64, self.device
        x = torch.as_tensor(np.clip(x0, self.lo_np + 1e-9, self.hi_np - 1e-9), dtype=dt, device=dev)
        Y = torch.as_tensor(np.nan_to_num(Y), dtype=dt, device=dev)
        ok = torch.as_tensor(ok, dtype=dt, device=dev)
        res = vmap(self.residual)
        jac = vmap(jacfwd(self.residual, argnums=0))
        lam = torch.full((len(x),), lam0, dtype=dt, device=dev)
        r = res(x, Y, ok)
        cost = 0.5 * (r * r).sum(1)
        active = torch.ones(len(x), dtype=torch.bool, device=dev)
        eye = torch.eye(self.n, dtype=dt, device=dev)
        for _ in range(iters):
            ia = torch.nonzero(active).squeeze(1)
            if len(ia) == 0:
                break
            xa, Ya, oka = x[ia], Y[ia], ok[ia]
            J = jac(xa, Ya, oka)                                    # (Na, m, n)
            ra = r[ia]
            A = J.transpose(1, 2) @ J
            g = (J.transpose(1, 2) @ ra[:, :, None])[:, :, 0]
            D = torch.clamp(torch.diagonal(A, dim1=1, dim2=2), min=1e-9)
            la = lam[ia]
            step = torch.linalg.solve(A + la[:, None, None] * (D[:, :, None] * eye), -g[:, :, None])[:, :, 0]
            xn = torch.maximum(torch.minimum(xa + step, self.hi), self.lo)
            rn = res(xn, Ya, oka)
            cn = 0.5 * (rn * rn).sum(1)
            better = cn < cost[ia]
            rel = (cost[ia] - cn) / torch.clamp(cost[ia], min=1e-12)
            x[ia] = torch.where(better[:, None], xn, xa)
            r[ia] = torch.where(better[:, None], rn, ra)
            cost[ia] = torch.where(better, cn, cost[ia])
            lam[ia] = torch.where(better, la * 0.3, la * 10.0)
            moved = (xn - xa).abs().max(1).values
            done = (better & (rel < 1e-7)) | (moved < 1e-8) | (lam[ia] > 1e10)
            active[ia] = ~done
        P = vmap(self.markers)(x)
        d2 = ((P - Y) ** 2).sum(2)
        rms = torch.sqrt((d2 * ok).sum(1) / torch.clamp(ok.sum(1), min=1))
        return x.cpu().numpy(), cost.cpu().numpy(), rms.cpu().numpy()


def _initial(posed, Y, ok):
    """Heading and position per frame: 2-D Procrustes of the rest markers onto the data
    (as ``PosedRig._initial``), vectorised. ``Y`` (N, M, 3), ``ok`` (N, M)."""
    A = posed.size * posed._rest[None, :, :2]
    w = ok.astype(float)[:, :, None]
    n = np.maximum(w.sum(1), 1)
    a0 = (A * w).sum(1) / n
    b0 = (np.nan_to_num(Y[:, :, :2]) * w).sum(1) / n
    Ac, Bc = (A - a0[:, None]) * w, (np.nan_to_num(Y[:, :, :2]) - b0[:, None]) * w
    H = np.einsum("nmi,nmj->nij", Ac, Bc)
    phi = np.arctan2(H[:, 0, 1] - H[:, 1, 0], H[:, 0, 0] + H[:, 1, 1])
    c, s = np.cos(phi), np.sin(phi)
    t = b0 - np.c_[c * a0[:, 0] - s * a0[:, 1], s * a0[:, 0] + c * a0[:, 1]]
    x = np.zeros((len(Y), 4 + len(posed.posture)))
    x[:, 0], x[:, 1:3] = phi, t
    return x


def fit_rig_frames(posed, frames, *, device: str = "auto", chunk: int = 2048, iters: int = 40,
                   warm_iters: int = 15, key_every: int = 8, progress=None, stop=None) -> dict:
    """{frame: (parameters, marker RMS mm)} for ``frames`` of ``posed`` (a PosedRig); frames
    with fewer than three markers are left out. ``progress(done, total)``; ``stop()`` → True
    ends early (what is done is returned)."""
    from mufasa.pose3d.rig_floor import clip_targets
    prob = RigProblem(posed, device)
    zlo, zhi = posed._zlo, posed._zhi
    pose = posed.pose
    frames = [int(f) for f in frames]
    rows = [(f, pose.row(f)) for f in frames]
    rows = [(f, i) for f, i in rows if i is not None]
    out = {}
    names_ix = posed._col
    pnames = [p[0] for p in posed.posture]
    i_rear = 4 + pnames.index("rear") if "rear" in pnames else None
    total = len(rows)
    for c0 in range(0, total, chunk):
        if stop is not None and stop():
            break
        part = rows[c0:c0 + chunk]
        idx = np.array([i for _, i in part])
        Y = clip_targets(pose.xyz[idx][:, names_ix], zlo, zhi)             # 122le
        ok = np.isfinite(Y).all(2)
        if pose.likelihood is not None:
            ok &= ~(pose.likelihood[idx][:, names_ix] < 0.5)
        use = ok.sum(1) >= 3
        if not use.any():
            continue
        Y, ok = Y[use], ok[use]
        fr = [f for (f, _), u in zip(part, use) if u]
        N = len(Y)
        fa = np.array(fr)

        def cold(k):
            """Every cold start (both headings, flat or rearing) for rows ``k``; the best of each."""
            x0 = _initial(posed, Y[k], ok[k])
            starts = [x0, x0.copy(), x0.copy(), x0.copy()]
            starts[1][:, 0] += math.pi
            starts[3][:, 0] += math.pi
            if i_rear is not None:
                starts[2][:, i_rear] = 45.0
                starts[3][:, i_rear] = 45.0
            n = len(k)
            X, C, R = prob.solve(np.vstack(starts), np.tile(Y[k], (4, 1, 1)), np.tile(ok[k], (4, 1)), iters=iters)
            X, C, R = X.reshape(4, n, -1), C.reshape(4, n), R.reshape(4, n)
            b = C.argmin(0)
            return X[b, np.arange(n)], C[b, np.arange(n)], R[b, np.arange(n)]

        # keyframes: every ``key_every``-th row and the first after a gap, from cold starts
        gap = np.r_[True, np.diff(fa) > 1]
        is_key = gap | (np.arange(N) % key_every == 0)
        kk = np.flatnonzero(is_key)
        xb = np.zeros((N, prob.n))
        cb, rb = np.full(N, np.inf), np.full(N, np.inf)
        xb[kk], cb[kk], rb[kk] = cold(kk)
        # the rest: from the nearest keyframe in the same run of consecutive frames
        rest = np.flatnonzero(~is_key)
        if len(rest):
            run = np.cumsum(gap)
            near = np.array([kk[(run[kk] == run[i])][np.argmin(np.abs(fa[kk][run[kk] == run[i]] - fa[i]))]
                             for i in rest])
            Xw, Cw, Rw = prob.solve(xb[near], Y[rest], ok[rest], iters=warm_iters)
            xb[rest], cb[rest], rb[rest] = Xw, Cw, Rw
            # clearly worse than its keyframe (as the per-frame fit judges a neighbour's start)
            bad = rest[Rw > np.maximum(2 * rb[near], 25.0)]
            if len(bad):
                Xc, Cc, Rc = cold(bad)
                imp = Cc < cb[bad]
                xb[bad[imp]], cb[bad[imp]], rb[bad[imp]] = Xc[imp], Cc[imp], Rc[imp]
        # neighbours: frames still off by more than good_mm start from the frames next to them
        for sh in (1, -1):
            j = np.clip(np.arange(N) - sh, 0, N - 1)
            k = np.flatnonzero((np.abs(fa[j] - fa) == 1) & (rb > posed.good_mm) & (cb[j] < cb))
            if not len(k):
                continue
            Xn, Cn, Rn = prob.solve(xb[j[k]], Y[k], ok[k], iters=warm_iters)
            imp = Cn < cb[k] - 1e-9
            xb[k[imp]], cb[k[imp]], rb[k[imp]] = Xn[imp], Cn[imp], Rn[imp]
        rb = prob.lift(xb, Y, ok)                                  # 122le: exactly above the floor
        xb[:, 0] = (xb[:, 0] + math.pi) % (2 * math.pi) - math.pi
        for f, xx, rr in zip(fr, xb, rb):
            out[f] = (xx.copy(), float(rr))
        if progress is not None:
            progress(min(c0 + chunk, total), total)
    return out


__all__ = ["RigProblem", "available", "device_name", "fit_rig_frames"]
