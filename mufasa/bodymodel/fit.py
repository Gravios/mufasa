"""
mufasa.bodymodel.fit
====================

Patch 122kq: fit a rigged body model to top-down video.

Posture (17 numbers per frame): position on the floor (tx, ty in px), heading
``phi``, scale ``s`` (px per model unit, one value per video), body pitch
about the hind paws (rearing) and roll, sideways and up/down bend of the
lumbar, thoracic and cervical spine (the same angle at each joint of a
region), head yaw and pitch, and the tail (sideways near and far half, up/down).

Joint limits (per joint; :data:`LIMITS`) come from the HBP mouse model's
joint table (Dewan et al. 2010, Shahrokni et al. 2012) where it has them;
head and body ranges are assumptions. Limbs keep their rest pose (hidden
under the body from above). Nothing may go below the floor.

A rig may carry a *profile* (patch 122ks; ``profile`` in the ``.rig.npz``,
see :data:`HBP_PROFILE`): which bones form each spine region, the head and
the tail, the hind paws and hips, and its own posture limits. Rigs without
one are taken to be the HBP mouse. The virtual rodent (a rat,
:mod:`mufasa.bodymodel.rodent_rig`) carries its own.

Per frame the loss is

* silhouette: 1 - IoU of the body (tail excluded: it is pale in IR) with the
  animal's mask (the pixels much darker than the median background), used
  when the mask is plausible (size and shape);
* keypoints: likelihood-weighted robust distance between the 2-D labels and
  the same points on the model (placed along the model's spine at the
  spacing the labels themselves show);
* floor, and a weak pull to a neutral posture.

:func:`fit_sequence` fits a run of frames: forwards, each frame starting
from the previous one and pulled towards it; then backwards, each frame
refined towards both neighbours. A frame whose loss jumps is re-searched
from scratch (both headings).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEG = math.pi / 180

# name, lo, hi (degrees; per joint for the spine/tail groups)
PARAMS = [
    ("tx", None, None), ("ty", None, None), ("phi", None, None), ("s", None, None),
    ("rear", -10, 60), ("roll", -15, 15),
    ("lum_yaw", -5, 5), ("tho_yaw", -5, 5), ("cer_yaw", -5, 5), ("head_yaw", -30, 30),
    ("lum_pitch", -6, 6), ("tho_pitch", -5, 5), ("cer_pitch", -9, 9), ("head_pitch", -30, 30),
    ("tail_yaw_p", -20, 20), ("tail_yaw_d", -20, 20), ("tail_pitch", -20, 20),
]
PN = [p[0] for p in PARAMS]
POSTURE = [p for p in PARAMS if p[1] is not None]
LIMITS = {
    "lumbar (Spine1-6)": "sideways ±5°, up/down ±6° per joint (Dewan et al. 2010; HBP mouse model)",
    "thoracic (Spine7-17)": "±5° each way per joint (HBP mouse model)",
    "cervical (Neck1-8)": "sideways ±5°, up/down ±9° per joint (Shahrokni et al. 2012; HBP mouse model)",
    "head": "yaw ±30°, pitch ±30° (assumed)",
    "tail (Tail4-23)": "±20° each way per joint (HBP mouse model); Tail1-3 fixed",
    "body": "rearing -10…60° and roll ±15° about the hind paws, which stay on the floor (assumed)",
    "floor": "no part below the floor",
    "limbs": "rest pose",
}
GROUPS = {
    "lum": [f"Spine{i}" for i in range(1, 7)],
    "tho": [f"Spine{i}" for i in range(7, 18)],
    "cer": [f"Neck{i}" for i in range(1, 9)],
    "tail_p": [f"Tail{i}" for i in range(4, 13)],
    "tail_d": [f"Tail{i}" for i in range(13, 24)],
}
# the HBP mouse (rigs without a profile)
HBP_PROFILE = {
    "name": "HBP mouse",
    "groups": GROUPS,
    "head": "Head",
    "trunk": ["Head"] + [f"Neck{i}" for i in range(8, 0, -1)] + [f"Spine{i}" for i in range(17, 0, -1)] + ["Pelvis"],
    "tail": [f"Tail{k}" for k in range(1, 24)],
    "tail_hidden": [f"Tail{i}" for i in range(4, 24)],
    "paws": [*(f"BPhalange{i}.L" for i in range(1, 6)), *(f"BPhalange{i}.R" for i in range(1, 6))],
    "hips": {"hip_left": "Femur.L", "hip_right": "Femur.R"},
    "posture": None,                     # PARAMS
    "limits": LIMITS,
}


# 2-D label -> point on the model. The chain labels are placed along the model's
# nose -> spine -> tail line at the spacing the labels show (see Markers).
CHAIN = ["head_nose", "head_mid", "head_back", "back_T4", "back_T8", "back_L2", "back_L6", "back_V2",
         "tail_V6", "tail_V18", "tail_V32"]
KP_WEIGHT = {"head_mid": 0.6, "hip_left": 0.3, "hip_right": 0.3, "tail_V18": 0.4, "tail_V32": 0.25,
             "head_nose": 0.7}


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --------------------------------------------------------------------------- #
# the rig
# --------------------------------------------------------------------------- #
class Rig:
    """Bones (parents first), rest matrices, skin (full and decimated), weights."""

    def __init__(self, path) -> None:
        z = np.load(path, allow_pickle=False)
        self.path = str(path)
        self.names = [str(n) for n in z["names"]]
        self.ix = {n: i for i, n in enumerate(self.names)}
        self.parent = z["parent"].astype(int)
        self.rest = z["rest"].astype(float)
        self.heads = self.rest[:, :3, 3]
        self.tails = z["tails"].astype(float)
        self.V, self.F, self.W = z["V"].astype(float), z["F"].astype(int), z["W"].astype(np.float32)
        self.Vd, self.Fd, self.Wd = z["Vd"].astype(float), z["Fd"].astype(int), z["Wd"].astype(np.float32)
        self.face_rgb = z["face_rgb"] if "face_rgb" in z.files and z["face_rgb"].ndim == 2 else None
        import json
        prof = dict(HBP_PROFILE)
        if "profile" in z.files:
            prof.update(json.loads(str(z["profile"])))
        self.profile = prof
        self.groups = {g: list(b) for g, b in prof["groups"].items()}
        self.posture = ([tuple(x) for x in prof["posture"]] if prof.get("posture") else list(POSTURE))
        self.limits = dict(prof.get("limits") or LIMITS)
        # the weak posture prior pulls towards these (default: the middle of each range)
        self.prior_centre = dict(prof.get("prior_centre") or {})
        tail_bones = [self.ix[b] for b in prof["tail_hidden"] if b in self.ix]
        self.body_faces_d = self.Fd[~np.isin(self.Wd.argmax(1)[self.Fd[:, 0]], tail_bones)]
        paws = [self.ix[n] for n in prof["paws"] if n in self.ix]
        hip_l = prof["hips"].get("hip_left")
        if paws:
            py = self.heads[paws][:, 1].mean()
        elif hip_l in self.ix:
            py = self.tails[self.ix[hip_l]][1]
        else:
            py = self.heads[0][1]
        self.pivot = np.array([0.0, py, 0.0])
        self.B = len(self.names)
        self._GW = self.W
        self.Gd = None

    # -- kinematics -----------------------------------------------------------
    def joint_rotations(self, p: dict) -> dict:
        R = {}
        for g in ("lum", "tho", "cer"):
            r = rot("z", p[f"{g}_yaw"] * DEG) @ rot("x", p[f"{g}_pitch"] * DEG)
            for b in self.groups.get(g, ()):
                R[b] = r
        R[self.profile["head"]] = rot("z", p["head_yaw"] * DEG) @ rot("x", p["head_pitch"] * DEG)
        for g, yaw in (("tail_p", "tail_yaw_p"), ("tail_d", "tail_yaw_d")):
            r = rot("z", p[yaw] * DEG) @ rot("x", p["tail_pitch"] * DEG)
            for b in self.groups.get(g, ()):
                R[b] = r
        return R

    def global_transform(self, p: dict) -> np.ndarray:
        """Rearing / roll about the hind paws."""
        Rg = rot("x", -p["rear"] * DEG) @ rot("y", p["roll"] * DEG)
        T = np.eye(4)
        T[:3, :3] = Rg
        T[:3, 3] = self.pivot - Rg @ self.pivot
        return T

    def skinning_matrices(self, p: dict) -> np.ndarray:
        R = self.joint_rotations(p)
        G = np.zeros((self.B, 4, 4))
        for i, n in enumerate(self.names):
            L = np.eye(4)
            if n in R:
                L[:3, :3] = R[n]
                L[:3, 3] = self.heads[i] - R[n] @ self.heads[i]
            G[i] = L if self.parent[i] < 0 else G[self.parent[i]] @ L
        return self.global_transform(p) @ G

    @staticmethod
    def lbs(G, V, W):
        T = (W @ G[:, :3, :].reshape(len(G), 12)).reshape(-1, 3, 4)
        return np.einsum("vij,vj->vi", T[:, :, :3], V) + T[:, :, 3]

    def local_quaternions(self, p: dict) -> dict:
        """Bone -> local rotation (w, x, y, z) as Blender pose bones use it."""
        out = {}
        for n, Rm in self.joint_rotations(p).items():
            if n not in self.ix:
                continue
            Rr = self.rest[self.ix[n], :3, :3]
            out[n] = _quat(Rr.T @ Rm @ Rr)
        return out


def _quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return (w, x, y, z)


def to_image(P, p) -> np.ndarray:
    """Model points -> image px (top view: x right, y up on the floor; image v down)."""
    c, s = math.cos(p["phi"]), math.sin(p["phi"])
    x = c * P[:, 0] - s * P[:, 1]
    y = s * P[:, 0] + c * P[:, 1]
    return np.c_[p["tx"] + p["s"] * x, p["ty"] - p["s"] * y]


def neutral(**kw) -> dict:
    p = {n: 0.0 for n in PN}
    p.update(kw)
    return p


# --------------------------------------------------------------------------- #
# markers: where each 2-D label sits on the model
# --------------------------------------------------------------------------- #
@dataclass
class Markers:
    """name -> ("vert", indices) or ("bone", bone index, rest point)."""
    spec: dict
    px_per_unit: float            # the label spacing's scale used to place the chain labels

    @classmethod
    def build(cls, rig: Rig, labels: dict | None = None) -> Markers:
        """``labels``: name -> (n, 2) px positions and (n,) likelihoods over the video, used to
        measure the chain spacing; without them a typical rodent spacing is assumed."""
        fixed = rig.profile.get("markers")
        if fixed:                                       # the rig says where each label sits
            spec = {}
            for n, m in fixed.items():
                if m[0] == "vert":
                    spec[n] = ("vert", [int(i) for i in m[1]])
                elif m[1] in rig.ix:
                    spec[n] = ("bone", rig.ix[m[1]], np.asarray(m[2], float))
            return cls(spec, 1.0)
        V, W = rig.V, rig.W
        hb = rig.profile["head"]
        head = W[:, rig.ix[hb]] > 0.5 if hb in rig.ix else np.zeros(len(V), bool)
        hv = np.nonzero(head)[0]
        spec = {}
        if len(hv):
            nose = hv[np.argmin(V[hv, 1])]
            spec["head_nose"] = ("vert", [int(nose)])
            half = np.abs(V[hv, 0]).max()
            hi = V[hv, 2] > np.median(V[hv, 2])
            for name, sign in (("head_left", 1), ("head_right", -1)):
                sel = hv[(sign * V[hv, 0] > 0.6 * half) & hi]
                if len(sel):
                    spec[name] = ("vert", [int(i) for i in sel])
        else:
            nose = int(np.argmin(V[:, 1]))
            spec["head_nose"] = ("vert", [nose])
        for name, bone in rig.profile["hips"].items():
            if bone in rig.ix:
                spec[name] = ("bone", rig.ix[bone], rig.tails[rig.ix[bone]].copy())
        # chain samples nose -> head -> neck -> spine -> pelvis -> tail
        samples = [(rig.ix.get(hb, 0), V[spec["head_nose"][1][0]].copy())]
        for b in rig.profile["trunk"]:                  # these point head-wards: walk them tail -> head
            if b in rig.ix:
                i = rig.ix[b]
                for t in np.linspace(0, 1, 8)[1:]:
                    samples.append((i, rig.tails[i] + t * (rig.heads[i] - rig.tails[i])))
        for b in rig.profile["tail"]:
            if b in rig.ix:
                i = rig.ix[b]
                for t in np.linspace(0, 1, 8)[1:]:
                    samples.append((i, rig.heads[i] + t * (rig.tails[i] - rig.heads[i])))
        pts = np.array([s[1] for s in samples])
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))]
        # scale anchor: nose -> head_back on the labels = nose -> the head joint on the model
        head_joint_arc = arc[min(len(arc) - 1, 7)] if hb in rig.ix else arc[-1] * 0.3
        spacing = _label_arc(labels) if labels else None
        if spacing and "head_back" in spacing and spacing["head_back"] > 0:
            k = spacing["head_back"] / head_joint_arc
        else:
            k = None
        if spacing is None or k is None:
            # typical mouse/rat spacing relative to the nose -> head-joint length
            rel = {"head_mid": 0.5, "head_back": 1.0, "back_T4": 1.39, "back_T8": 1.77, "back_L2": 2.2,
                   "back_L6": 2.58, "back_V2": 2.94, "tail_V6": 3.3, "tail_V18": 4.28, "tail_V32": 5.26}
            spacing = {n: r * head_joint_arc for n, r in rel.items()}
            k = 1.0
        for name in CHAIN[1:]:
            if name in spacing:
                j = int(np.argmin(np.abs(arc - spacing[name] / k)))
                spec[name] = ("bone", samples[j][0], pts[j].copy())
        return cls(spec, float(k))

    def positions(self, rig: Rig, G) -> dict:
        out = {}
        for n, m in self.spec.items():
            if m[0] == "vert":
                out[n] = Rig.lbs(G, rig.V[m[1]], rig.W[m[1]]).mean(0)
            else:
                out[n] = (G[m[1]] @ np.r_[m[2], 1.0])[:3]
        return out


def _label_arc(labels: dict) -> dict | None:
    """Median px distance along the chain labels (both ends confident)."""
    have = [n for n in CHAIN if n in labels]
    if "head_nose" not in have or "head_back" not in have:
        return None
    out, acc = {"head_nose": 0.0}, 0.0
    prev = "head_nose"
    for n in have[1:]:
        a, la = labels[prev]
        b, lb = labels[n]
        ok = (la > 0.7) & (lb > 0.7) & np.isfinite(a).all(1) & np.isfinite(b).all(1)
        if ok.sum() < min(20, max(3, len(la) // 4)):
            continue
        acc += float(np.median(np.linalg.norm(a[ok] - b[ok], axis=1)))
        out[n] = acc
        prev = n
    return out if len(out) > 3 else None


# --------------------------------------------------------------------------- #
# per-frame loss and search
# --------------------------------------------------------------------------- #
@dataclass
class FitConfig:
    scale: float | None = None          # px per model unit; None: estimated from the video
    sil_downsample: int = 4             # silhouette resolution 1/n (speed)
    w_sil: float = 3.0
    w_kp: float = 0.5
    kp_px: float = 20.0                 # label distance scale (px)
    min_likelihood: float = 0.5
    temporal: float = 0.5               # pull towards neighbouring frames (0: off)
    evals_first: int = 2500             # function evaluations: full search
    evals_warm: int = 500               # ... warm start from a neighbour
    refine: bool = True                 # backward pass
    seed: int = 1
    weights: dict = field(default_factory=lambda: dict(KP_WEIGHT))


def _step_scale(gap, posture=POSTURE):
    """How far each posture value may move per frame (for the temporal term)."""
    g = max(1.0, float(gap))
    sc = {"tx": 12.0 * g, "ty": 12.0 * g, "phi": 12 * DEG * g}
    for n, lo, hi in posture:
        sc[n] = 0.25 * (hi - lo) * min(1.0, 0.3 * g)
    return sc


class FrameProblem:
    """The loss for one frame."""

    def __init__(self, rig: Rig, markers: Markers, cfg: FitConfig, mask, obs: dict, refs=()) -> None:
        import cv2
        self.rig, self.mk, self.cfg = rig, markers, cfg
        self.obs = obs
        self.refs = list(refs)          # [(params, gap)] neighbours for the temporal term
        self.mask = None
        if mask is not None:
            n = cfg.sil_downsample
            h, w = mask.shape[:2]
            self.mask = cv2.resize(mask.astype(np.uint8), (w // n, h // n), interpolation=cv2.INTER_AREA) > 0
            self.area = int(self.mask.sum())

    def evaluate(self, p: dict, detail=False):
        rig, cfg = self.rig, self.cfg
        G = rig.skinning_matrices(p)
        parts = {}
        Pm = Rig.lbs(G, rig.Vd, rig.Wd)
        below = np.clip(-Pm[:, 2] - 1.0, 0, None)
        parts["floor"] = float(np.mean(below ** 2)) * 5.0
        if self.mask is not None:
            P = to_image(Pm, p) / cfg.sil_downsample
            sil = np.zeros(self.mask.shape, np.uint8)
            fill_triangles(sil, P[rig.body_faces_d])
            inter = int(np.count_nonzero(sil[self.mask]))
            union = int(sil.sum()) + self.area - inter
            parts["sil"] = 1.0 - inter / max(union, 1)
        mp = self.mk.positions(rig, G)
        names = [n for n in mp if n in self.obs and self.obs[n][1] >= cfg.min_likelihood]
        if names:
            uv = to_image(np.array([mp[n] for n in names]), p)
            e = w_sum = 0.0
            for k, n in enumerate(names):
                pos, lk = self.obs[n]
                d = float(np.linalg.norm(uv[k] - pos)) / cfg.kp_px
                hub = d * d if d < 1 else 2 * d - 1
                w = cfg.weights.get(n, 1.0) * lk
                e += w * hub
                w_sum += w
            parts["kp"] = e / max(w_sum, 1e-9)
        else:
            parts["kp"] = 0.0
        z = np.array([(p[n] - rig.prior_centre.get(n, (lo + hi) / 2)) / ((hi - lo) / 2) for n, lo, hi in rig.posture])
        parts["prior"] = 0.02 * float(np.mean(z ** 2))
        if self.refs and cfg.temporal > 0:
            t = 0.0
            for q, gap in self.refs:
                sc = _step_scale(gap, rig.posture)
                d = [(p[n] - q[n]) / sc[n] for n in ("tx", "ty")]
                d.append(_wrap(p["phi"] - q["phi"]) / sc["phi"])
                d += [(p[n] - q[n]) / sc[n] for n, _, _ in rig.posture]
                t += float(np.mean(np.square(d)))
            parts["temporal"] = cfg.temporal * t / len(self.refs)
        total = (cfg.w_sil * parts.get("sil", 0.0) + cfg.w_kp * parts["kp"] + parts["prior"] + parts["floor"]
                 + parts.get("temporal", 0.0))
        return (total, parts, G) if detail else total


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def search(problem: FrameProblem, base: dict, *, wide: bool, evals: int, seed: int = 1, flips=(0,)) -> tuple:
    """CMA-ES in a box around ``base`` (position/heading) and within the joint limits."""
    import cma
    best = None
    posture = problem.rig.posture
    span_xy = 150.0 if wide else 40.0
    span_phi = math.radians(60 if wide else 25)
    for flip in flips:
        b = dict(base)
        if flip:
            b["phi"] = base["phi"] + math.pi

        def unpack(x, b=b):
            p = dict(b)
            p["tx"] = b["tx"] + (x[0] - 0.5) * 2 * span_xy
            p["ty"] = b["ty"] + (x[1] - 0.5) * 2 * span_xy
            p["phi"] = b["phi"] + (x[2] - 0.5) * 2 * span_phi
            for v, (n, lo, hi) in zip(x[3:], posture):
                p[n] = lo + v * (hi - lo)
            return p
        x0 = [0.5, 0.5, 0.5] + [min(0.98, max(0.02, (b[n] - lo) / (hi - lo))) for n, lo, hi in posture]
        es = cma.CMAEvolutionStrategy(x0, 0.15 if wide else 0.06,
                                      {"bounds": [0, 1], "maxfevals": evals, "seed": seed + flip, "verbose": -9,
                                       "popsize": 14 if wide else 10})
        es.optimize(lambda x: problem.evaluate(unpack(x)))
        p = unpack(es.result.xbest)
        tot, parts, G = problem.evaluate(p, detail=True)
        if best is None or tot < best[0]:
            best = (tot, parts, p, G)
    return best


def similarity_start(rig: Rig, markers: Markers, obs: dict, scale: float | None, min_lk=0.6):
    """Rest-pose markers -> labels (2-D similarity): position, heading, scale."""
    mp = markers.positions(rig, rig.skinning_matrices(neutral()))
    src, dst, w = [], [], []
    for n, (pos, lk) in obs.items():
        if n in mp and lk >= min_lk and n not in ("tail_V18", "tail_V32") and np.isfinite(pos).all():
            src.append([mp[n][0], -mp[n][1]])
            dst.append(pos)
            w.append(lk)
    if len(src) < 3:
        return None
    src, dst, w = np.array(src), np.array(dst), np.array(w)
    ms, md = (w[:, None] * src).sum(0) / w.sum(), (w[:, None] * dst).sum(0) / w.sum()
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd((w[:, None] * a).T @ b)
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    s = scale or float((S * np.diag(D)).sum() / (w[:, None] * a * a).sum())
    o = md - s * Rm @ ms
    return neutral(tx=float(o[0]), ty=float(o[1]), phi=-math.atan2(Rm[1, 0], Rm[0, 0]), s=float(s))


def silhouette_area(rig: Rig, scale: float) -> float:
    """Pixel area of the rest-pose body seen from above at ``scale``."""
    P = to_image(Rig.lbs(rig.skinning_matrices(neutral()), rig.Vd, rig.Wd), neutral(tx=500, ty=500, s=scale))
    img = np.zeros((1000 + int(200 * scale), 1000 + int(200 * scale)), np.uint8)
    fill_triangles(img, P[rig.body_faces_d])
    return float(img.sum())


def fill_triangles(img, tri_px) -> None:
    """Union of the triangles (one call each: cv2.fillPoly with many overlapping polygons would
    cancel where they overlap, e.g. the top and underside of the body)."""
    import cv2
    for q in np.round(tri_px * 4).astype(np.int32):
        cv2.fillConvexPoly(img, q, 1, shift=2)


# --------------------------------------------------------------------------- #
# video: background, masks, labels
# --------------------------------------------------------------------------- #
@dataclass
class MaskConfig:
    darker_by: float = 35.0        # grey levels below the background (above it for a light animal)
    max_grey: float = 110.0        # and at most this bright (dark fur)
    # "dark" (dark fur, e.g. under IR), "light" (a white rat on darker ground) or "auto": from
    # the pixels under the confident back labels against the background (patch 122ks)
    polarity: str = "auto"
    min_grey_light: float = 120.0  # a light animal: at least this bright
    # keep only the region around the confident labels, and of its parts the one holding most of
    # them (None: for light animals, whose ground — bedding — often differs from the background)
    near_labels: bool | None = None
    blur: float = 3.0
    area_range: tuple = (0.45, 3.0)   # mask area / expected body area, for the mask to be used


def background(video, frames, n=60):
    import cv2
    cap = cv2.VideoCapture(str(video))
    imgs = []
    for f in np.linspace(min(frames), max(frames), min(n, len(frames))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if ok:
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not imgs:
        raise OSError(f"cannot read {video}")
    return np.median(np.array(imgs), 0).astype(np.float32)


def animal_mask(gray, bg, mc: MaskConfig, expected_area: float, near=None):
    """(mask or None if implausible, area). ``near``: confident label positions (k, 2); the
    region that holds most of them is the animal (otherwise the largest)."""
    import cv2
    a = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), mc.blur)
    b = cv2.GaussianBlur(bg, (0, 0), mc.blur)
    if mc.polarity == "light":
        m = (((a - b) > mc.darker_by) & (a > mc.min_grey_light)).astype(np.uint8)
    else:
        m = (((b - a) > mc.darker_by) & (a < mc.max_grey)).astype(np.uint8)
    use_near = mc.near_labels if mc.near_labels is not None else mc.polarity == "light"
    if not use_near:
        near = None
    if near is not None and len(near) >= 3:
        # only near the labels: their hull, widened by a third of their span
        q = np.asarray(near, np.float32)
        span = float(np.linalg.norm(q.max(0) - q.min(0)))
        hull = np.zeros_like(m)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.round(q).astype(np.int32)), 1)
        r = max(5, int(round(span / 3)))
        hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        m &= hull
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if k < 2:
        return None, 0
    j = 1 + int(np.argmax(st[1:, 4]))
    if near is not None and len(near):
        H_, W_ = lab.shape
        q = np.round(np.asarray(near, float)).astype(int)
        q = q[(q[:, 0] >= 0) & (q[:, 0] < W_) & (q[:, 1] >= 0) & (q[:, 1] < H_)]
        hits = np.bincount(lab[q[:, 1], q[:, 0]], minlength=k)[1:] if len(q) else np.zeros(k - 1)
        if hits.max() > 0:
            j = 1 + int(np.argmax(hits * 1e6 + st[1:, 4]))
    area = int(st[j, 4])
    mk = (lab == j).astype(np.uint8)
    ff = mk.copy()
    cv2.floodFill(ff, None, (0, 0), 1)
    mk |= (1 - ff)
    x, y, w, h = st[j, :4]
    H, Wd = gray.shape
    edge = x <= 2 or y <= 2 or x + w >= Wd - 2 or y + h >= H - 2
    lo, hi = mc.area_range
    ok = (lo * expected_area <= area <= hi * expected_area) and not edge
    return (mk.astype(bool) if ok else None), area


def _spread(frames, n):
    frames = sorted(frames)
    return [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int))]


def animal_polarity(imgs: dict, bg, labels: dict, names=("back_T4", "back_T8", "back_L2", "back_L6", "head_back"),
                    min_lk=0.9) -> str:
    """"light" when the animal (the pixels under its confident back labels) is brighter than the
    background there, else "dark"."""
    d = []
    H, Wd = bg.shape
    for f, img in imgs.items():
        for n in names:
            if n not in labels or f >= len(labels[n][1]) or labels[n][1][f] < min_lk:
                continue
            x, y = labels[n][0][f]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            xi, yi = int(np.clip(round(x), 2, Wd - 3)), int(np.clip(round(y), 2, H - 3))
            d.append(float(img[yi - 2:yi + 3, xi - 2:xi + 3].mean() - bg[yi - 2:yi + 3, xi - 2:xi + 3].mean()))
    return "light" if d and np.median(d) > 0 else "dark"


def load_labels(pose_file):
    """name -> ((n, 2) px, (n,) likelihood) from any Mufasa/FreeDLC/DLC pose file."""
    from mufasa.data_processors.kalman_diagnostic import load_pose_file
    df = load_pose_file(str(pose_file))[0]
    names = [c[:-2] for c in df.columns if c.endswith("_x") and c[:-2] + "_y" in df.columns]
    out = {}
    for m in names:
        xy = df[[m + "_x", m + "_y"]].to_numpy(float)
        lk = (df[m + "_p"] if m + "_p" in df.columns else df.get(m + "_likelihood", 1.0))
        out[m] = (xy, np.nan_to_num(np.asarray(lk, float) * np.ones(len(df))))
    return out


def observed(labels, f):
    return {n: (xy[f], float(lk[f])) for n, (xy, lk) in labels.items() if f < len(lk)}


def body_points(obs: dict, names, min_lk=0.8):
    """Confident positions of the labels ``names`` (those the model knows) on the body (not the
    tail beyond its base), for picking the animal's region in the mask."""
    return np.array([pos for n, (pos, lk) in obs.items()
                     if n in names and lk >= min_lk and np.isfinite(pos).all()
                     and n not in ("tail_V18", "tail_V32")]).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# sequence fitting
# --------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    frame: int
    params: dict
    parts: dict
    loss: float
    mask_used: bool
    mask_area: int
    kp_rms: float
    restarted: bool = False

    def row(self) -> dict:
        r = {"frame": self.frame, **{k: float(v) for k, v in self.params.items()}, "loss": self.loss,
             "iou": (1 - self.parts["sil"]) if "sil" in self.parts else np.nan, "kp_rms_px": self.kp_rms,
             "mask_used": self.mask_used, "mask_area_px": self.mask_area, "restarted": self.restarted}
        for k, v in self.parts.items():
            r[f"loss_{k}"] = float(v)
        return r


def kp_rms(rig, markers, p, obs, min_lk=0.5) -> float:
    G = rig.skinning_matrices(p)
    mp = markers.positions(rig, G)
    names = [n for n in mp if n in obs and obs[n][1] >= min_lk and n not in ("tail_V18", "tail_V32")
             and np.isfinite(obs[n][0]).all()]
    if not names:
        return float("nan")
    uv = to_image(np.array([mp[n] for n in names]), p)
    return float(np.sqrt(np.mean([np.sum((u - obs[n][0]) ** 2) for u, n in zip(uv, names)])))


class SequenceFitter:
    def __init__(self, rig: Rig, video, labels: dict, cfg: FitConfig | None = None,
                 mcfg: MaskConfig | None = None, bg=None, frames=None) -> None:
        self.rig, self.video, self.labels = rig, str(video), labels
        self.cfg, self.mcfg = cfg or FitConfig(), mcfg or MaskConfig()
        self.markers = Markers.build(rig, labels)
        self.bg = bg
        self._frames_hint = frames
        self.scale = self.cfg.scale

    # -- frames and masks -------------------------------------------------------
    def read(self, frames):
        """frame -> grey image, read in order (seeking only across gaps)."""
        import cv2
        cap = cv2.VideoCapture(self.video)
        out, pos = {}, None
        for f in sorted(frames):
            if pos is None or f < pos or f - pos > 30:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                pos = f
            while pos < f:
                cap.grab()
                pos += 1
            ok, im = cap.read()
            pos += 1
            if ok:
                out[f] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cap.release()
        return out

    def ensure_background(self, frames):
        if self.bg is None:
            self.bg = background(self.video, frames)
        if self.mcfg.polarity == "auto":
            self.mcfg.polarity = animal_polarity(self.read(_spread(frames, 12)), self.bg, self.labels)

    def estimate_scale(self, frames, n=8, say=print) -> float:
        """Similarity starts give a first scale; a free-scale fit on up to ``n`` frames with a usable
        mask gives the video's scale (median)."""
        obs0 = [observed(self.labels, f) for f in frames]
        s0 = [st["s"] for st in (similarity_start(self.rig, self.markers, o, None) for o in obs0) if st]
        if not s0:
            raise ValueError("no frame has enough confident labels to place the model")
        s_guess = float(np.median(s0))
        exp_area = silhouette_area(self.rig, s_guess)
        pick = [frames[i] for i in np.linspace(0, len(frames) - 1, min(len(frames), 4 * n)).astype(int)]
        imgs = self.read(pick)
        # the labels alone can be far off (a hunched animal shortens them, a fat one is wider than the
        # model): correct the guess by the animal's area in the masks (median, any size accepted)
        from dataclasses import replace as _replace
        anyc = _replace(self.mcfg, area_range=(0.0, float("inf")))
        areas = [a for f in pick if f in imgs
                 for m, a in [animal_mask(imgs[f], self.bg, anyc, exp_area, body_points(observed(self.labels, f), self.markers.spec))]
                 if m is not None]
        if len(areas) >= 3:
            s_area = s_guess * math.sqrt(float(np.median(areas)) / exp_area)
            if abs(math.log(s_area / s_guess)) > math.log(1.15):
                say(f"  scale from the labels {s_guess:.3f}, from the mask area {s_area:.3f} px/unit: using the latter")
                s_guess = s_area
                exp_area = silhouette_area(self.rig, s_guess)
        scales = []
        for f in pick:
            if len(scales) >= n or f not in imgs:
                continue
            o = observed(self.labels, f)
            m, _ = animal_mask(imgs[f], self.bg, self.mcfg, exp_area, body_points(o, self.markers.spec))
            st = similarity_start(self.rig, self.markers, o, None)
            if m is None or st is None:
                continue
            prob = FrameProblem(self.rig, self.markers, self.cfg, m, o)
            prob_s = _FreeScale(prob, s_guess)
            tot, parts, p, _ = prob_s.search(st)
            scales.append(p["s"])
            say(f"  scale from frame {f}: {p['s']:.3f} px/unit (IoU {1 - parts.get('sil', 1):.2f})")
        self.scale = float(np.median(scales)) if scales else s_guess
        return self.scale

    def frame_problem(self, f, img, refs):
        exp_area = silhouette_area(self.rig, self.scale) if not hasattr(self, "_exp_area") else self._exp_area
        self._exp_area = exp_area
        o = observed(self.labels, f)
        m, area = (animal_mask(img, self.bg, self.mcfg, exp_area, body_points(o, self.markers.spec))
                   if img is not None else (None, 0))
        return FrameProblem(self.rig, self.markers, self.cfg, m, o, refs), m is not None, area, o

    # -- the sequence -------------------------------------------------------------
    def fit(self, frames, progress=None, say=print) -> list[FrameResult]:
        frames = sorted(int(f) for f in frames)
        prog = progress or (lambda done, total: None)
        self.ensure_background(frames)
        if self.scale is None:
            self.estimate_scale(frames, say=say)
        imgs = self.read(frames)
        cfg = self.cfg
        total = len(frames) * (2 if cfg.refine else 1)
        done = 0
        res: dict[int, FrameResult] = {}
        prev = None
        for f in frames:                                       # forwards
            gap = (f - prev.frame) if prev else 1
            refs = [(prev.params, gap)] if prev else []
            prob, used, area, o = self.frame_problem(f, imgs.get(f), refs)
            r = self._solve(prob, f, used, area, o, prev.params if prev else None)
            res[f] = r
            prev = r
            done += 1
            prog(done, total)
        if cfg.refine and len(frames) > 1:                     # backwards, towards both neighbours
            for k in range(len(frames) - 1, -1, -1):
                f = frames[k]
                refs = []
                if k + 1 < len(frames):
                    refs.append((res[frames[k + 1]].params, frames[k + 1] - f))
                if k > 0:
                    refs.append((res[frames[k - 1]].params, f - frames[k - 1]))
                prob, used, area, o = self.frame_problem(f, imgs.get(f), refs)
                r = self._solve(prob, f, used, area, o, res[f].params, refine=True)
                if r.loss <= self._rescore(prob, res[f].params):
                    res[f] = r
                done += 1
                prog(done, total)
        return [res[f] for f in frames]

    def _rescore(self, prob, p):
        return prob.evaluate(p)

    def _solve(self, prob, f, used, area, o, warm, refine=False) -> FrameResult:
        cfg = self.cfg
        restarted = False
        if warm is None:
            st = similarity_start(self.rig, self.markers, o, self.scale)
            if st is None:
                st = neutral(tx=prob.mask.shape[1] * cfg.sil_downsample / 2 if prob.mask is not None else 0,
                             ty=0, s=self.scale)
            st["s"] = self.scale
            best = search(prob, st, wide=True, evals=cfg.evals_first, seed=cfg.seed, flips=(0, 1))
        else:
            base = dict(warm)
            base["s"] = self.scale
            best = search(prob, base, wide=False, evals=cfg.evals_warm, seed=cfg.seed)
            # lost track: much worse than a fresh start from the labels -> search widely
            if not refine and (best[1].get("sil", 0) > 0.75 or best[1]["kp"] > 4.0):
                st = similarity_start(self.rig, self.markers, o, self.scale)
                if st is not None:
                    st["s"] = self.scale
                    alt = search(prob, st, wide=True, evals=cfg.evals_first, seed=cfg.seed, flips=(0, 1))
                    if alt[0] < best[0]:
                        best, restarted = alt, True
        tot, parts, p, _ = best
        return FrameResult(int(f), {k: float(p[k]) for k in PN}, {k: float(v) for k, v in parts.items()},
                           float(tot), used, int(area), kp_rms(self.rig, self.markers, p, o), restarted)


class _FreeScale:
    """Search with the scale free (±25 %), for estimating it."""

    def __init__(self, prob, s0) -> None:
        self.prob, self.s0 = prob, s0

    def search(self, st):
        import cma
        prob, s0 = self.prob, self.s0
        posture = prob.rig.posture

        def unpack(x):
            p = dict(st)
            p["tx"] = st["tx"] + (x[0] - 0.5) * 300
            p["ty"] = st["ty"] + (x[1] - 0.5) * 300
            p["phi"] = st["phi"] + (x[2] - 0.5) * math.radians(120)
            p["s"] = s0 * (0.75 + 0.5 * x[3])
            for v, (n, lo, hi) in zip(x[4:], posture):
                p[n] = lo + v * (hi - lo)
            return p
        best = None
        for flip in (0, math.pi):
            st0 = dict(st)
            st0["phi"] += flip
            x0 = [0.5, 0.5, 0.5, 0.5] + [(0 - lo) / (hi - lo) for n, lo, hi in posture]

            def f(x, st0=st0):
                p = unpack(x)
                p["phi"] += st0["phi"] - st["phi"]
                return prob.evaluate(p)
            es = cma.CMAEvolutionStrategy(x0, 0.15, {"bounds": [0, 1], "maxfevals": 2500, "verbose": -9,
                                                     "seed": 3, "popsize": 14})
            es.optimize(f)
            p = unpack(es.result.xbest)
            p["phi"] += st0["phi"] - st["phi"]
            tot, parts, G = prob.evaluate(p, detail=True)
            if best is None or tot < best[0]:
                best = (tot, parts, p, G)
        return best


# --------------------------------------------------------------------------- #
# whole runs, in parallel chunks
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    start: int = 0
    stop: int | None = None             # exclusive; None: to the end
    step: int = 1
    workers: int = 1
    chunk: int = 300                    # frames per worker task (overlapping by `overlap`)
    overlap: int = 8
    fit: FitConfig = field(default_factory=FitConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)


def _chunk_job(args):
    rig_path, video, pose_file, frames, fit_cfg, mask_cfg, bg, scale = args
    rig = Rig(rig_path)
    labels = load_labels(pose_file)
    fit_cfg = FitConfig(**fit_cfg)
    fit_cfg.scale = scale
    sf = SequenceFitter(rig, video, labels, fit_cfg, MaskConfig(**mask_cfg), bg=bg)
    return [asdict(r) for r in sf.fit(frames, say=lambda s: None)]


def fit_video(rig_path, video, pose_file, run: RunConfig, out_path, *, progress=None, say=print):
    """Fit frames ``start:stop:step`` of a video; writes a parquet (one row per fitted frame) and
    returns it."""
    import cv2
    import pandas as pd
    rig = Rig(rig_path)
    labels = load_labels(pose_file)
    cap = cv2.VideoCapture(str(video))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()
    n_lab = len(next(iter(labels.values()))[1])
    stop = min(run.stop or n_lab, n_lab, n_video or n_lab)
    frames = list(range(run.start, stop, max(1, run.step)))
    if not frames:
        raise ValueError("no frames in the range")
    sf = SequenceFitter(rig, video, labels, run.fit, run.mask)
    # background from the whole video: over a short range a resting animal would stay in the median
    sf.ensure_background(list(range(0, n_video or stop)))
    say(f"{Path(video).name}: {len(frames)} frames ({run.start}–{stop - 1}, every {run.step})")
    scale = sf.estimate_scale(frames, say=say) if run.fit.scale is None else run.fit.scale
    say(f"scale {scale:.3f} px per model unit")
    # chunks with overlap; each frame is taken from the chunk where it is furthest from an edge
    size, ov = max(20, run.chunk), max(0, run.overlap)
    chunks, i = [], 0
    while i < len(frames):
        chunks.append(frames[max(0, i - ov): i + size + ov])
        i += size
    fit_d = asdict(run.fit)
    mask_d = asdict(run.mask)
    jobs = [(str(rig_path), str(video), str(pose_file), c, fit_d, mask_d, sf.bg, scale) for c in chunks]
    results = []
    if run.workers > 1 and len(chunks) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(min(run.workers, len(chunks)), mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_chunk_job, j): k for k, j in enumerate(jobs)}
            for n_done, fu in enumerate(as_completed(futs), 1):
                results.append((futs[fu], fu.result()))
                say(f"  chunk {n_done}/{len(chunks)} done")
                if progress:
                    progress(n_done, len(chunks))
    else:
        for k, j in enumerate(jobs):
            results.append((k, _chunk_job(j)))
            say(f"  chunk {k + 1}/{len(chunks)} done")
            if progress:
                progress(k + 1, len(chunks))
    best = {}
    for k, rows in results:
        c = chunks[k]
        for r in rows:
            pos = c.index(r["frame"])
            margin = min(pos, len(c) - 1 - pos)
            if r["frame"] not in best or margin > best[r["frame"]][0]:
                best[r["frame"]] = (margin, r)
    rows = [FrameResult(**best[f][1]).row() for f in frames if f in best]
    df = pd.DataFrame(rows)
    df.attrs["scale"] = scale
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    tab = pa.Table.from_pandas(df, preserve_index=False)
    meta = {"rig": str(rig_path), "video": str(video), "pose": str(pose_file), "fps": fps, "scale_px_per_unit": scale,
            "limits": rig.limits, "profile": rig.profile.get("name"), "config": {"start": run.start, "stop": stop, "step": run.step,
                                         "fit": fit_d, "mask": {k: list(v) if isinstance(v, tuple) else v
                                                                for k, v in mask_d.items()}}}
    tab = tab.replace_schema_metadata({**(tab.schema.metadata or {}), b"mufasa_bodymodel": json.dumps(meta).encode()})
    pq.write_table(tab, out_path)
    say(f"fit -> {out_path}: median IoU {np.nanmedian(df['iou']):.2f} ({int(df['mask_used'].sum())} frames with "
        f"a usable mask), median label error {np.nanmedian(df['kp_rms_px']):.1f} px")
    return out_path


def read_fit(path):
    import json

    import pyarrow.parquet as pq
    t = pq.read_table(path)
    meta = json.loads((t.schema.metadata or {}).get(b"mufasa_bodymodel", b"{}"))
    return t.to_pandas(), meta


def params_at(df, frame: int) -> dict | None:
    """Posture at any frame: linear between fitted frames (heading unwrapped); None outside."""
    f = df["frame"].to_numpy()
    if frame < f[0] or frame > f[-1]:
        return None
    k = int(np.searchsorted(f, frame))
    if f[k] == frame:
        return {n: float(df[n].iloc[k]) for n in PN}
    a, b = df.iloc[k - 1], df.iloc[k]
    t = (frame - a["frame"]) / (b["frame"] - a["frame"])
    p = {n: float(a[n] + t * (b[n] - a[n])) for n in PN}
    p["phi"] = float(a["phi"] + t * _wrap(b["phi"] - a["phi"]))
    return p


__all__ = ["CHAIN", "HBP_PROFILE", "LIMITS", "PARAMS", "PN", "FitConfig", "FrameProblem", "FrameResult", "MaskConfig",
           "Markers", "Rig", "RunConfig", "SequenceFitter", "animal_mask", "animal_polarity", "background", "fit_video",
           "load_labels", "neutral", "observed", "params_at", "read_fit", "search", "similarity_start",
           "to_image"]
