"""
mufasa.pose3d.leg_ik
====================

Patch 122lh: the virtual rodent's legs bend, by its own joints, so its paws stand where they
should — stage 1 of the biomechanical model (then joint torques, muscle loads, simulation).

The rig's limb bones are the model's limb bodies (femur, tibia, foot, toe; scapula, humerus,
radius, hand, finger), held in the tucked stance (:data:`mufasa.bodymodel.rodent_rig.STANCE`).
Here each moves by the model's hinge joints with their ranges: a bone's transform relative to its
parent is ``C · J(q) · J(q_stance)⁻¹ · C⁻¹`` (``C`` the frame its joints act in, ``J`` their
rotations, in order), which does not depend on the trunk's posture, so it composes with the rig's
own skinning. Angles are stored under the model's joint names (``hip_L_extend``, ``knee_L``…),
for the MuJoCo stages that follow.

Per frame and leg, a small batched Levenberg–Marquardt solve within the joints' ranges:

* stance (a gait file loaded, the paw planted): the paw on the floor as in the stance — the
  lowest point of the foot (hand) and of the toes (fingers) at their stance heights (toes
  2.7 mm, sole 5.5 mm, fingers 2.6 mm, palm 12.8 mm at the model's size: the model stands on its
  digits) — its middle at the gait's paw position;
* swing: the paw's middle on an arc from where it lifted to where it lands
  (:data:`SWING_HEIGHT_MM` high, assumed, at the model's size);
* no gait, or a paw the gait cannot place: heel and toe on the floor where they are;
* always: a weak pull towards the stance, and nothing of the leg under the floor;
* over time: frames solved together and consecutive are solved, their angles smoothed over
  :data:`SMOOTH_FRAMES`, and solved again pulled towards the smoothed angles (twice), so contacts
  stay put while the legs do not jump from frame to frame (K7, walking: the worst joint's second
  difference was 43° in one frame of ten, frames solved alone).
"""
from __future__ import annotations

import numpy as np

from mufasa.pose3d.gait import PAWS

SWING_HEIGHT_MM = {"F": 6.0, "H": 8.0}   # paw clearance at mid-swing (assumed; × the animal's size)
PRIOR = 0.5                              # pull towards the stance: per joint, per its whole range
SIGMA = {"xy": 2.0, "sole": 1.0, "swing": 4.0, "plant": 2.0, "floor": 0.5}   # mm
SMOOTH_FRAMES = 1.5          # consecutive frames: angles smoothed over this many frames (Gaussian σ) …
SMOOTH_DEG = 3.0             # … and each frame pulled to them: 3° off costs as much as a sole 1 mm off

CHAINS = {
    "H": [("Femur", "upper_leg"), ("Tibia", "lower_leg"), ("BFoot", "foot"), ("BPhalange1", "toe")],
    "F": [("Scapula", "scapula"), ("Humerus", "upper_arm"), ("Radius", "lower_arm"), ("FHand", "hand"),
          ("FPhalange1", "finger")],
}
PARENT_BODY = {"H": "pelvis", "F": "torso"}


def _hinge(axis, pos, q) -> np.ndarray:
    """(N, 4, 4) rotations about ``axis`` through ``pos`` (body frame) by ``q`` (N,)."""
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    A = np.tile(np.eye(4), (len(q), 1, 1))
    A[:, :3, :3] = R
    A[:, :3, 3] = pos - R @ pos
    return A


class Leg:
    """One leg of the rig: its bones, the model's joints that move them, its sole and floor probes."""

    def __init__(self, paw: str, rig, model, q_stance, Tw, M) -> None:
        self.paw = paw
        side, kind = paw[0], paw[1]
        S = "L" if side == "L" else "R"
        self.bones, self.joints, self.C, self.Cinv, self.Js_inv = [], [], [], [], []
        for nm, body in CHAINS[kind]:
            b = rig.ix[f"{nm}.{S}"]
            bi = model.body_index[f"{body}_{S}"]
            js = [k for k, j in enumerate(model.joints) if j.body == bi]
            C = M @ Tw[model.parent[bi]] @ model.offset[bi]
            Js = np.eye(4)
            for k in js:
                Js = Js @ _hinge(model.joints[k].axis, model.joints[k].pos, q_stance[[k]])[0]
            self.bones.append(b)
            self.joints.append(js)
            self.C.append(C)
            self.Cinv.append(np.linalg.inv(C))
            self.Js_inv.append(np.linalg.inv(Js))
        self.parent = int(rig.parent[self.bones[0]])
        self.q_index = [k for js in self.joints for k in js]
        self.names = [model.joints[k].name for k in self.q_index]
        self.qs = q_stance[self.q_index].copy()
        self.lo = model.lo[self.q_index].copy()
        self.hi = model.hi[self.q_index].copy()
        self.qs = np.clip(self.qs, self.lo, self.hi)
        self.axes = [(model.joints[k].axis, model.joints[k].pos) for k in self.q_index]
        # the paw's two lowest points: of the foot (hand) and of the toes (fingers), with their
        # heights in the stance
        V, W = np.asarray(rig.Vd, float), np.asarray(rig.Wd)
        own = W.argmax(1)

        def lowest(bone):
            ix = np.flatnonzero(own == bone)
            return V[ix[np.argmin(V[ix, 2])]]
        self.heel = (len(self.bones) - 2, lowest(self.bones[-2]))
        self.toe = (len(self.bones) - 1, lowest(self.bones[-1]))
        self.sole_z = np.array([self.heel[1][2], self.toe[1][2]])
        # floor probes: each bone's three lowest vertices
        self.probes = []
        for k, b in enumerate(self.bones):
            ix = np.flatnonzero(own == b)
            for v in ix[np.argsort(V[ix, 2])[:3]]:
                self.probes.append((k, V[v]))

    def local(self, q) -> list:
        """Each bone's transform relative to its parent bone (N, 4, 4), for angles ``q`` (N, nq)."""
        out, c = [], 0
        for k, js in enumerate(self.joints):
            J = np.tile(np.eye(4), (len(q), 1, 1))
            for _ in js:
                ax, pos = self.axes[c]
                J = J @ _hinge(ax, pos, q[:, c])
                c += 1
            out.append(self.C[k] @ J @ self.Js_inv[k] @ self.Cinv[k])
        return out

    def chain(self, Gp, q) -> list:
        """The bones' skinning transforms (N, 4, 4) under a parent's ``Gp`` (N, 4, 4)."""
        G, out = Gp, []
        for L in self.local(q):
            G = G @ L
            out.append(G)
        return out


class Limbs:
    """The four legs of the virtual-rodent rig (:class:`mufasa.bodymodel.fit.Rig`)."""

    def __init__(self, rig) -> None:
        from mufasa.bodymodel.rodent_rig import R_FRAME, stance_qpos
        from mufasa.pose3d.rodent_model import RodentModel
        model = RodentModel()
        q0 = stance_qpos(model)
        floor = float(model.skin_points(q0)[0][:, 2].min())          # as the rig was made
        M = np.eye(4)
        M[:3, :3] = R_FRAME
        M[2, 3] = -floor
        Tw = model.fk(q0)[0]
        self.legs = {p: Leg(p, rig, model, q0[0], Tw, M) for p in PAWS}

    @staticmethod
    def for_rig(rig) -> Limbs | None:
        """Legs for a virtual-rodent rig; None for other rigs (the HBP mouse)."""
        need = [f"{n}.{s}" for k in CHAINS.values() for n, _ in k for s in "LR"]
        prof = getattr(rig, "profile", None) or {}
        if not all(n in rig.ix for n in need) or "virtual rodent" not in str(prof.get("name", "")):
            return None
        try:
            return Limbs(rig)
        except Exception:  # noqa: BLE001 - the model's files missing: legs stay tucked
            return None

    def apply(self, G, angles: dict) -> np.ndarray:
        """Skinning transforms (B, 4, 4) with each leg in ``angles`` ({paw: (nq,)}) set."""
        G = np.array(G, copy=True)
        for p, q in angles.items():
            leg = self.legs[p]
            for b, Gb in zip(leg.bones, leg.chain(G[leg.parent][None], np.asarray(q, float)[None]), strict=True):
                G[b] = Gb[0]
        return G

    def solve(self, G, place, targets: list, q0: dict | None = None, iters: int = 30,
              frames=None) -> tuple[dict, dict, dict]:
        """Leg angles for N frames at once. ``G`` (N, B, 4, 4): the rig's skinning transforms
        (trunk posed); ``place`` (phi (N,), t (N, 3), size): rig → world (heading, shift, scale);
        ``targets``: per frame {paw: (mode, xyz)}, mode "stance" (xyz on the floor), "swing" (xyz
        in the air), "plant" (as stood, where it is) or "free" (nothing to reach). Returns
        ({paw: (N, nq)}, {paw: (N,) mm off: the paw's middle, or its sole's heights},
        {paw: [mode per frame]}). ``frames``: their frame numbers; runs of consecutive ones are
        smoothed over time."""
        G = np.asarray(G, float)
        phi, t, size = place
        phi, t = np.asarray(phi, float), np.asarray(t, float)
        N = len(G)
        out, err, modes = {}, {}, {}
        for p, leg in self.legs.items():
            md = [(tg.get(p) or ("plant", None))[0] for tg in targets]
            xyz = np.array([np.zeros(3) if (tg.get(p) or ("", None))[1] is None else np.asarray(tg[p][1], float)
                            for tg in targets]).reshape(N, 3)
            q = np.tile(leg.qs, (N, 1)) if not q0 or p not in q0 else np.array(q0[p], float)
            groups = [(mode, np.array([k for k, m in enumerate(md) if m == mode])) for mode in set(md)]
            for mode, ix in groups:
                q[ix] = solve_leg(leg, G[ix, leg.parent], (phi[ix], t[ix], size), mode, xyz[ix], q[ix], iters)
            if frames is not None and N > 2:
                for _ in range(2):                                    # smoothed, solved again
                    ref = smooth_runs(q, frames, SMOOTH_FRAMES)
                    for mode, ix in groups:
                        q[ix] = solve_leg(leg, G[ix, leg.parent], (phi[ix], t[ix], size), mode, xyz[ix], ref[ix],
                                          iters, ref=ref[ix])
            e = np.zeros(N)
            for mode, ix in groups:
                e[ix] = np.abs(_res(leg, G[ix, leg.parent], (phi[ix], t[ix], size), mode, xyz[ix], q[ix],
                                    parts=True)).max(1)
            out[p], err[p], modes[p] = q, e, md
        return out, err, modes


def _points(leg: Leg, Gs, pts):
    """Rig positions (N, k, 3) of points ``pts`` [(bone k, rest xyz)] under bone transforms."""
    return np.stack([(Gs[k][:, :3, :3] @ P + Gs[k][:, :3, 3]) for k, P in pts], 1)


def _world(P, place):
    """Rig points (N, k, 3) in the world: scaled, turned by each frame's heading, shifted."""
    phi, t, size = place
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
    return np.stack([c * P[..., 0] - s * P[..., 1], s * P[..., 0] + c * P[..., 1], P[..., 2]], -1) * size \
        + t[:, None, :]


def smooth_runs(q, frames, sigma: float) -> np.ndarray:
    """Angles (N, nq) smoothed (Gaussian, ``sigma`` frames) within runs of consecutive ``frames``."""
    from scipy.ndimage import gaussian_filter1d
    q = np.asarray(q, float)
    f = np.asarray(frames, int)
    out = q.copy()
    cut = np.flatnonzero(np.diff(f) != 1) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(f)], strict=True):
        if b - a > 2:
            out[a:b] = gaussian_filter1d(q[a:b], sigma, axis=0, mode="nearest")
    return out


def _res(leg: Leg, Gp, place, mode, tgt, q, parts: bool = False, ref=None):
    """Residuals (N, m) for angles ``q`` (N, nq)."""
    Gs = leg.chain(Gp, q)
    W = _world(_points(leg, Gs, [leg.heel, leg.toe] + leg.probes), place)
    N = len(q)
    heel, toe, probes = W[:, 0], W[:, 1], W[:, 2:]
    mid = (heel + toe) / 2
    dz = np.c_[heel[:, 2], toe[:, 2]] - leg.sole_z * place[2]       # off the stance's sole heights
    if mode == "stance":
        r = np.c_[(mid[:, :2] - tgt[:, :2]) / SIGMA["xy"], dz / SIGMA["sole"]]
        main = np.c_[np.linalg.norm(mid[:, :2] - tgt[:, :2], axis=1), np.abs(dz)]
    elif mode == "swing":
        r = np.c_[(mid - tgt) / SIGMA["swing"], np.zeros(N)]
        main = np.linalg.norm(mid - tgt, axis=1)[:, None]
    elif mode == "plant":
        r = np.c_[np.zeros((N, 2)), dz / SIGMA["plant"]]
        main = np.abs(dz)
    else:
        r = np.zeros((N, 4))
        main = np.zeros((N, 1))
    if parts:
        return main
    fl = np.maximum(-probes[:, :, 2], 0.0) / SIGMA["floor"]
    pr = PRIOR * (q - leg.qs) / (leg.hi - leg.lo)
    if ref is None:
        return np.c_[r, fl, pr]
    return np.c_[r, fl, pr, (q - ref) / np.radians(SMOOTH_DEG)]


def solve_leg(leg: Leg, Gp, place, mode, tgt, q, iters: int = 30, ref=None):
    """Levenberg–Marquardt for N frames at once (Jacobian by differences), within the joints'
    ranges. ``Gp`` (N, 4, 4) the leg's parent bone; ``place`` as in :meth:`Limbs.solve`;
    ``tgt`` (N, 3); ``q`` (N, nq) the start; ``ref`` (N, nq) angles to stay near (smoothing)."""
    q = np.clip(np.array(q, float), leg.lo, leg.hi)
    N, n = q.shape
    lam = np.full(N, 1e-2)
    r = _res(leg, Gp, place, mode, tgt, q, ref=ref)
    cost = (r ** 2).sum(1)
    h = 1e-5
    for _ in range(iters):
        J = np.empty((N, r.shape[1], n))
        for k in range(n):
            dq = q.copy()
            dq[:, k] += h
            J[:, :, k] = (_res(leg, Gp, place, mode, tgt, dq, ref=ref) - r) / h
        JtJ = np.einsum("nmi,nmj->nij", J, J)
        g = np.einsum("nmi,nm->ni", J, r)
        A = JtJ + lam[:, None, None] * (np.eye(n)[None] * (np.einsum("nii->ni", JtJ)[:, :, None] + 1e-6))
        step = -np.linalg.solve(A, g[:, :, None])[:, :, 0]
        qn = np.clip(q + step, leg.lo, leg.hi)
        rn = _res(leg, Gp, place, mode, tgt, qn, ref=ref)
        cn = (rn ** 2).sum(1)
        better = cn < cost
        q[better], r[better], cost[better] = qn[better], rn[better], cn[better]
        lam = np.where(better, lam * 0.3, lam * 10.0)
        if np.all(np.abs(step).max(1) < 1e-6):
            break
    return q


class GaitTargets:
    """Where each paw should be, frame by frame, from a gait file's table (:mod:`mufasa.pose3d.gait`):
    planted at its stance position; in swing on an arc from where it lifted to where it lands."""

    def __init__(self, df, size: float = 1.0) -> None:
        self.row = {int(f): i for i, f in enumerate(df["frame"])}
        self.size = float(size)
        self.con = {p: df[f"{p}_contact"].to_numpy(int) for p in PAWS}
        self.xy = {p: df[[f"{p}_x", f"{p}_y"]].to_numpy(float) for p in PAWS}
        T = len(df)
        self.prev, self.next = {}, {}
        for p in PAWS:
            on = self.con[p] == 1
            prev = np.full(T, -1)
            nxt = np.full(T, -1)
            last = -1
            for i in range(T):
                if on[i]:
                    last = i
                prev[i] = last
            last = -1
            for i in range(T - 1, -1, -1):
                if on[i]:
                    last = i
                nxt[i] = last
            self.prev[p], self.next[p] = prev, nxt

    def at(self, frame: int) -> dict:
        """{paw: (mode, xyz or None)} for a frame (all "plant" for a frame not in the file)."""
        i = self.row.get(int(frame))
        out = {}
        for p in PAWS:
            if i is None:
                out[p] = ("plant", None)
                continue
            c = self.con[p][i]
            if c == 1 and np.isfinite(self.xy[p][i]).all():
                out[p] = ("stance", np.r_[self.xy[p][i], 0.0])
            elif c == -1:
                out[p] = ("free", None)
            else:
                a, b = self.prev[p][i], self.next[p][i]
                if a < 0 or b < 0 or b <= a:
                    out[p] = ("plant", None)
                    continue
                s = (i - a) / (b - a)
                xy = (1 - s) * self.xy[p][a] + s * self.xy[p][b]
                z = SWING_HEIGHT_MM[p[1]] * self.size * np.sin(np.pi * s)
                out[p] = ("swing", np.r_[xy, z])
        return out


__all__ = ["CHAINS", "GaitTargets", "Leg", "Limbs", "PRIOR", "SIGMA", "SMOOTH_DEG", "SMOOTH_FRAMES", "SWING_HEIGHT_MM",
           "smooth_runs", "solve_leg"]
