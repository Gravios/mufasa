"""
mufasa.pose3d.rodent_model
==========================

Patch 122kr: DeepMind's virtual rodent (dm_control ``rodent.xml`` and its
skin; Apache 2.0, see ``mufasa/assets/models/pose3d/virtual_rodent/``) as
an anatomical reference for single-camera 3-D pose.

* :class:`RodentModel` reads the MuJoCo model (bodies, hinge joints and
  their ranges, with default-class inheritance) and the skin (vertices,
  faces, per-bone weights). It computes forward kinematics for a batch of
  joint angles and skins the mesh (linear blend skinning with MuJoCo's
  bind poses). Units: mm; x forward (the nose at +x), z up.
* :data:`LABELS` places this project's 15 rat labels on the model: dorsal
  skin points above the vertebrae, the ears, the nose, the hip joints, and
  caudal vertebrae.
* :func:`build_limits` works out, once, what the body allows: for each
  label the highest and lowest it can be with every joint within range
  (highest: a hind paw on the floor; both: no skin below the floor), and
  the range of the spine's up/down bend per region (pelvis → thorax, and
  the neck), measured the way :mod:`mufasa.pose3d.mono` measures it. The
  result is stored as ``virtual_rodent_limits.json`` next to the model;
  :func:`load_limits` reads it.

Only NumPy and SciPy; no MuJoCo needed.
"""
from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "assets" / "models" / "pose3d" / "virtual_rodent"
LIMITS_PATH = MODEL_DIR / "virtual_rodent_limits.json"

# the spine chain as mono uses it (tail end → head), and the regions whose
# total bend is limited: (name, first segment, last segment) — the bend of a
# region is the elevation of its last segment minus that of its first
CHAIN = ["back_V2", "back_L6", "back_L2", "back_T8", "back_T4", "head_back"]
REGIONS = {"lumbar": (0, 3), "neck": (3, 4)}

# joint groups sampled for the spine-bend ranges and moved by the height search
SPINE_JOINTS = ("vertebra_1_extend", "vertebra_4_extend", "vertebra_2_bend", "vertebra_5_bend",
                "vertebra_3_twist", "vertebra_6_twist", "vertebra_cervical_5_extend",
                "vertebra_cervical_2_extend", "vertebra_atlant_extend", "vertebra_cervical_4_bend",
                "vertebra_cervical_1_bend", "vertebra_cervical_3_twist", "vertebra_axis_twist", "atlas")


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _axis_angle(axis: np.ndarray, th: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotations about unit ``axis`` by angles ``th`` (n,)."""
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _quat_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _euler_xyz(e) -> np.ndarray:
    """MuJoCo's default ``eulerseq="xyz"`` (rotations about the moving axes)."""
    R = np.eye(3)
    for ax, a in zip(np.eye(3), e):
        R = R @ _axis_angle(ax, np.array([a]))[0]
    return R


def _vec(s: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(v) for v in s.split()]) if s else np.array(default, float)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass
class Joint:
    name: str
    body: int
    axis: np.ndarray      # unit, body frame
    pos: np.ndarray       # mm, body frame
    lo: float             # rad
    hi: float


class RodentModel:
    """The virtual rodent: kinematic tree, joint ranges and skin."""

    def __init__(self, xml=MODEL_DIR / "rodent.xml", skin=MODEL_DIR / "rodent_walker_skin.skn") -> None:
        self._read_xml(Path(xml))
        self._read_skin(Path(skin))

    # -- MJCF ----------------------------------------------------------------- #
    def _read_xml(self, path: Path) -> None:
        root = ET.parse(path).getroot()
        comp = root.find("compiler")
        if comp is not None and comp.get("angle", "degree") != "radian":
            raise ValueError("expected <compiler angle='radian'>")
        # default classes: joint attributes, inherited down the tree
        classes: dict[str, dict] = {}

        def walk_defaults(el, inherited: dict, name: str) -> None:
            att = dict(inherited)
            j = el.find("joint")
            if j is not None:
                att.update(j.attrib)
            classes[name] = att
            for ch in el.findall("default"):
                walk_defaults(ch, att, ch.get("class"))
        top = root.find("default")
        walk_defaults(top, {}, "main")

        self.body_names: list[str] = []
        self.parent: list[int] = []
        self.offset: list[np.ndarray] = []      # (4, 4) fixed transform from the parent
        self.joints: list[Joint] = []

        def walk_body(el, parent: int) -> None:
            i = len(self.body_names)
            self.body_names.append(el.get("name"))
            self.parent.append(parent)
            T = np.eye(4)
            T[:3, 3] = _vec(el.get("pos")) * 1000.0
            if el.get("quat"):
                T[:3, :3] = _quat_mat(_vec(el.get("quat")))
            elif el.get("euler"):
                T[:3, :3] = _euler_xyz(_vec(el.get("euler")))
            self.offset.append(T)
            for j in el.findall("joint"):
                att = dict(classes[j.get("class", "main")])
                att.update(j.attrib)
                if att.get("type", "hinge") != "hinge":
                    raise ValueError(f"joint {att.get('name')}: only hinge joints are supported")
                ax = _vec(att.get("axis"), (0, 0, 1))
                lo, hi = _vec(att.get("range"), (-np.pi, np.pi))
                self.joints.append(Joint(att["name"], i, ax / np.linalg.norm(ax), _vec(att.get("pos")) * 1000.0,
                                         float(lo), float(hi)))
            for ch in el.findall("body"):
                walk_body(ch, i)
        for b in root.find("worldbody").findall("body"):
            walk_body(b, -1)
        self.body_index = {n: i for i, n in enumerate(self.body_names)}
        self.joint_index = {j.name: k for k, j in enumerate(self.joints)}
        self.lo = np.array([j.lo for j in self.joints])
        self.hi = np.array([j.hi for j in self.joints])

    # -- skin ----------------------------------------------------------------- #
    def _read_skin(self, path: Path) -> None:
        b = path.read_bytes()
        nv, nt, nf, nb = struct.unpack_from("<4i", b, 0)
        o = 16
        self.verts = np.frombuffer(b, "<f4", 3 * nv, o).reshape(nv, 3).astype(float) * 1000.0
        o += 12 * nv + 8 * nt
        self.faces = np.frombuffer(b, "<i4", 3 * nf, o).reshape(nf, 3).copy()
        o += 12 * nf
        W = np.zeros((nv, len(self.body_names)))
        self.bind = np.tile(np.eye(4), (len(self.body_names), 1, 1))
        for _ in range(nb):
            name = b[o:o + 40].split(b"\0")[0].decode()
            o += 40
            p = np.frombuffer(b, "<f4", 3, o).astype(float) * 1000.0
            q = np.frombuffer(b, "<f4", 4, o + 12).astype(float)
            o += 28
            cnt = struct.unpack_from("<i", b, o)[0]
            o += 4
            ids = np.frombuffer(b, "<i4", cnt, o)
            w = np.frombuffer(b, "<f4", cnt, o + 4 * cnt)
            o += 8 * cnt
            k = self.body_index[name]
            W[ids, k] += w
            self.bind[k, :3, :3] = _quat_mat(q)
            self.bind[k, :3, 3] = p
        if o != len(b):
            raise ValueError(f"{path.name}: {len(b) - o} trailing bytes")
        # the skin was bound to the skeleton at a larger size (bind positions
        # are the rest positions × 1/0.82, rotations equal): scale it to the
        # skeleton so the rest pose reproduces it
        used = W.sum(0) > 0
        rest = self.fk(np.zeros((1, len(self.joints))))[0]
        P, B = rest[used, :3, 3], self.bind[used, :3, 3]
        self.skin_scale = float((P * B).sum() / (B * B).sum())
        self.verts *= self.skin_scale
        self.bind[:, :3, 3] *= self.skin_scale
        self.weights = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
        self.skin_bodies = np.flatnonzero(W.sum(0) > 0)
        self.bind_inv = np.linalg.inv(self.bind)

    # -- kinematics ----------------------------------------------------------- #
    def qpos(self, n: int = 1, **angles) -> np.ndarray:
        """(n, n_joints) zeros, with named joints set (radians)."""
        q = np.zeros((n, len(self.joints)))
        for k, v in angles.items():
            q[:, self.joint_index[k]] = v
        return q

    def fk(self, q: np.ndarray) -> np.ndarray:
        """World transforms (n, n_bodies, 4, 4) for joint angles ``q`` (n, n_joints)."""
        q = np.atleast_2d(q)
        n = len(q)
        out = np.zeros((n, len(self.body_names), 4, 4))
        by_body: dict[int, list[int]] = {}
        for k, j in enumerate(self.joints):
            by_body.setdefault(j.body, []).append(k)
        for i, p in enumerate(self.parent):
            T = np.broadcast_to(self.offset[i], (n, 4, 4)).copy()
            if p >= 0:
                T = out[:, p] @ T
            for k in by_body.get(i, ()):
                j = self.joints[k]
                R = _axis_angle(j.axis, q[:, k])
                A = np.tile(np.eye(4), (n, 1, 1))
                A[:, :3, :3] = R
                A[:, :3, 3] = j.pos - R @ j.pos
                T = T @ A
            out[:, i] = T
        return out

    def skin_points(self, q: np.ndarray, idx=None, Tw: np.ndarray | None = None) -> np.ndarray:
        """Skinned vertices (n, nv, 3) (or only ``idx``)."""
        Tw = self.fk(q) if Tw is None else Tw
        V = self.verts if idx is None else self.verts[idx]
        W = self.weights if idx is None else self.weights[idx]
        bs = self.skin_bodies
        M = (Tw[:, bs] @ self.bind_inv[bs])[:, :, :3, :]          # (n, nb, 3, 4)
        Mv = np.einsum("vb,nbij->nvij", W[:, bs], M)                # blended per vertex
        return np.einsum("nvij,vj->nvi", Mv[..., :3], V) + Mv[..., 3]

    # -- labels --------------------------------------------------------------- #
    def label_points(self) -> dict:
        """Where each label sits: ("vertex", index) on the skin or
        ("body", body index, local point mm)."""
        return _place_labels(self)

    def labels(self, q: np.ndarray, Tw: np.ndarray | None = None) -> dict:
        """Label positions {name: (n, 3)} for joint angles ``q``."""
        Tw = self.fk(q) if Tw is None else Tw
        spec = self.label_points()
        vidx = [s[1] for s in spec.values() if s[0] == "vertex"]
        V = self.skin_points(q, np.array(vidx), Tw) if vidx else None
        out, k = {}, 0
        for m, s in spec.items():
            if s[0] == "vertex":
                out[m] = V[:, k]
                k += 1
            else:
                T = Tw[:, s[1]]
                out[m] = T[:, :3, :3] @ s[2] + T[:, :3, 3]
        return out


_LABEL_CACHE: dict[int, dict] = {}


def _place_labels(model: RodentModel) -> dict:
    """Labels on the rest pose: the highest midline skin point above each
    named vertebra (dorsal labels), the ear tips, the nose tip, the hip
    joints, caudal vertebrae for the tail."""
    if id(model) in _LABEL_CACHE:
        return _LABEL_CACHE[id(model)]
    q0 = model.qpos()
    Tw = model.fk(q0)
    V = model.skin_points(q0, Tw=Tw)[0]
    org = {n: Tw[0, i, :3, 3] for n, i in model.body_index.items()}
    dom = model.weights.argmax(1)
    mid = np.abs(V[:, 1]) < 2.5

    def dorsal(x):
        cand = np.flatnonzero(mid & (np.abs(V[:, 0] - x) < 6.0))
        top = cand[V[cand, 2] > V[cand, 2].max() - 4.0]           # the dorsal ones
        return ("vertex", int(top[np.argmin(np.abs(V[top, 0] - x))]))
    skull = model.body_index["skull"]
    head = np.flatnonzero(dom == skull)
    nose = int(head[np.argmax(V[head, 0])])
    # the ears (the skin has no ear flaps): the crown of the head, where it
    # is highest, and its widest points within 8 mm of that height
    crown = head[np.argmax(V[head, 2])]
    ear_x = V[crown, 0]
    top = head[(V[head, 2] > V[crown, 2] - 8.0) & (np.abs(V[head, 0] - ear_x) < 6.0)]
    ear_l = int(top[np.argmax(V[top, 1])])
    ear_r = int(top[np.argmin(V[top, 1])])
    torso_x0 = org["vertebra_1"][0]                          # the back end of the thorax
    torso_x1 = org["vertebra_cervical_5"][0]                 # its front end
    spec = {
        "head_nose": ("vertex", nose),
        "head_mid": dorsal((ear_x + V[nose, 0]) / 2),
        "head_back": dorsal(ear_x - 3.0),
        "head_left": ("vertex", ear_l),
        "head_right": ("vertex", ear_r),
        "back_T4": dorsal(torso_x0 + 0.75 * (torso_x1 - torso_x0)),
        "back_T8": dorsal(torso_x0 + 0.35 * (torso_x1 - torso_x0)),
        "back_L2": dorsal(org["vertebra_2"][0]),
        "back_L6": dorsal(org["vertebra_6"][0]),
        "back_V2": dorsal(org["pelvis"][0] - 6.0),
        "hip_left": ("body", model.body_index["upper_leg_L"], np.zeros(3)),
        "hip_right": ("body", model.body_index["upper_leg_R"], np.zeros(3)),
        "tail_V6": ("body", model.body_index["vertebra_C6"], np.zeros(3)),
        "tail_V18": ("body", model.body_index["vertebra_C18"], np.zeros(3)),
        "tail_V32": ("body", model.body_index["vertebra_C30"], np.zeros(3)),
    }
    _LABEL_CACHE[id(model)] = spec
    return spec


LABELS = tuple(["head_nose", "head_mid", "head_back", "head_left", "head_right", "back_T4", "back_T8",
                "back_L2", "back_L6", "back_V2", "hip_left", "hip_right", "tail_V6", "tail_V18", "tail_V32"])


# --------------------------------------------------------------------------- #
# limits
# --------------------------------------------------------------------------- #
def _root_pose(P: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Rotate points (n, m, 3) by ``pitch`` (n,) about +y (positive: the
    nose up)."""
    R = _axis_angle(np.array([0.0, -1.0, 0.0]), pitch)
    return np.einsum("nij,nmj->nmi", R, P)


def chain_bends(L: dict, chain=CHAIN) -> tuple[np.ndarray, np.ndarray]:
    """Up/down bends (rad, (n, len(chain) − 2)) at the inner chain nodes,
    as mono measures them: elevation of each segment along the horizontal
    tail-end → head direction, differences between consecutive segments;
    and the elevations (rad, (n, len(chain) − 1))."""
    nodes = [L[m] for m in chain]
    hd = nodes[-1][:, :2] - nodes[0][:, :2]
    hd /= np.maximum(np.linalg.norm(hd, axis=1, keepdims=True), 1e-9)
    el = [np.arctan2(b[:, 2] - a[:, 2], np.einsum("ij,ij->i", b[:, :2] - a[:, :2], hd))
          for a, b in zip(nodes[:-1], nodes[1:])]
    return np.stack([el[i + 1] - el[i] for i in range(len(el) - 1)], 1), np.stack(el, 1)


def region_bends(L: dict, chain=CHAIN, regions=REGIONS) -> dict:
    _, el = chain_bends(L, chain)
    return {r: el[:, b] - el[:, a] for r, (a, b) in regions.items()}


def _height_search(model: RodentModel, label: str, sense: int, *, rng, starts: int = 24,
                   iters: int = 400, hind_on_floor: bool = True) -> tuple[float, dict]:
    """Highest (``sense`` = +1) or lowest (−1) height of ``label`` with all
    joints within range, the body's pitch free, no skin below the floor and
    (``hind_on_floor``) a hind paw touching it. A random search refined by
    coordinate steps: robust, and the problem is small."""
    jn = [j.name for j in model.joints]
    move = [k for k, n in enumerate(jn) if not n.startswith(("vertebra_C", "mandible"))]
    lo, hi = model.lo[move], model.hi[move]
    hind = np.flatnonzero(np.isin(model.weights.argmax(1),
                                  [model.body_index[b] for b in ("foot_L", "foot_R", "toe_L", "toe_R")]))
    sub = np.arange(0, len(model.verts), 5)                     # skin points checked against the floor
    tail = np.isin(model.weights[sub].argmax(1),
                   [i for n, i in model.body_index.items() if n.startswith("vertebra_C")])
    sub = sub[~tail]                                            # the tail may hang below (it lifts)
    spec = model.label_points()

    def evaluate(x):                                            # x (n, len(move) + 1): angles, pitch
        q = np.zeros((len(x), len(jn)))
        q[:, move] = x[:, :-1]
        Tw = model.fk(q)
        S = _root_pose(model.skin_points(q, sub, Tw), x[:, -1])
        if spec[label][0] == "vertex":
            pt = model.skin_points(q, np.array([spec[label][1]]), Tw)
        else:
            T = Tw[:, spec[label][1]]
            pt = (T[:, :3, :3] @ spec[label][2] + T[:, :3, 3])[:, None]
        pt = _root_pose(pt, x[:, -1])[:, 0]
        floor = np.minimum(S[..., 2].min(1), pt[:, 2])
        z = pt[:, 2] - floor
        score = sense * z
        if hind_on_floor:
            H = _root_pose(model.skin_points(q, hind, Tw), x[:, -1])
            gap = H[..., 2].min(1) - floor                      # a hind paw is the lowest point
            score = score - 20.0 * gap
        return score, z
    n0 = 2000
    X = np.c_[rng.uniform(lo, hi, (n0, len(move))), rng.uniform(-np.pi / 2, np.pi / 2, n0)]
    X[: n0 // 4, :-1] *= 0.2                                    # some near the rest pose
    s, _ = evaluate(X)
    best = X[np.argsort(-s)[:starts]]
    lo_all, hi_all = np.r_[lo, -np.pi / 2], np.r_[hi, np.pi / 2]
    step = (hi_all - lo_all) * 0.15
    for _ in range(iters // 8):
        cand = best[:, None, :] + rng.normal(0, 1, (len(best), 8, best.shape[1])) * step
        cand = np.clip(cand, lo_all, hi_all).reshape(-1, best.shape[1])
        allx = np.r_[best, cand]
        s, _ = evaluate(allx)
        order = np.argsort(-s)
        # keep the best few distinct starts
        best = allx[order[:starts]]
        step *= 0.93
    s, z = evaluate(best[:1])
    q = np.zeros(len(jn))
    q[move] = best[0, :-1]
    return float(z[0]), {"pitch_deg": float(np.degrees(best[0, -1])),
                         **{jn[k]: round(float(np.degrees(q[k])), 1) for k in move if abs(q[k]) > 1e-3}}


def build_limits(model: RodentModel | None = None, *, samples: int = 200000, seed: int = 0,
                 out=LIMITS_PATH, say=print) -> dict:
    """Work out the label height ranges and the spine-bend ranges; write
    them as JSON (``out``)."""
    model = model or RodentModel()
    rng = np.random.default_rng(seed)
    # spine bends: the spine joints uniform within range, all else at rest
    ks = [model.joint_index[n] for n in SPINE_JOINTS]
    bends = {r: [] for r in REGIONS}
    nodes = []
    for a in range(0, samples, 20000):
        n = min(20000, samples - a)
        q = np.zeros((n, len(model.joints)))
        q[:, ks] = rng.uniform(model.lo[ks], model.hi[ks], (n, len(ks)))
        L = model.labels(q)
        for r, v in region_bends(L).items():
            bends[r].append(v)
        nodes.append(chain_bends(L)[0])
    nodes = np.degrees(np.concatenate(nodes))
    rest = model.labels(model.qpos())
    rest_bend = {r: float(np.degrees(v[0])) for r, v in region_bends(rest).items()}
    spine = {}
    for r, v in bends.items():
        v = np.degrees(np.concatenate(v))
        # 0.5 / 99.5th percentiles: the extremes come from twisted, sideways-bent
        # spines whose up/down projection degenerates
        lo_, hi_ = np.percentile(v, [0.5, 99.5])
        spine[r] = {"min_deg": round(float(lo_), 1), "max_deg": round(float(hi_), 1),
                    "rest_deg": round(rest_bend[r], 1)}
        say(f"{r} bend: {spine[r]['min_deg']} … {spine[r]['max_deg']}° (rest {spine[r]['rest_deg']}°)")
    pn = np.percentile(nodes, [0.5, 99.5], axis=0)
    node_rng = {m: {"min_deg": round(float(pn[0, i]), 1), "max_deg": round(float(pn[1, i]), 1)}
                for i, m in enumerate(CHAIN[1:-1])}
    # heights
    heights = {}
    for m in LABELS:
        if m.startswith("tail_"):
            continue
        hi_z, hi_pose = _height_search(model, m, +1, rng=rng)
        lo_z, lo_pose = _height_search(model, m, -1, rng=rng, hind_on_floor=False)
        heights[m] = {"max_mm": round(hi_z, 1), "min_mm": round(lo_z, 1),
                      "rest_mm": round(float(_rest_height(model, rest, m)), 1),
                      "max_pose": hi_pose, "min_pose": lo_pose}
        say(f"{m}: {lo_z:.0f} … {hi_z:.0f} mm")
    sizes = model_sizes(model, rest)
    res = {
        "model": "DeepMind virtual rodent (dm_control rodent.xml, Apache 2.0)",
        "units": "mm, model scale; multiply by the session's scale",
        "sizes_mm": sizes,
        "label_heights": heights,
        "spine_bend": spine,
        "spine_node_bend": node_rng,
        "chain": CHAIN,
        "regions": {r: [CHAIN[a], CHAIN[a + 1], CHAIN[b], CHAIN[b + 1]] for r, (a, b) in REGIONS.items()},
        "samples": samples, "seed": seed,
    }
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


def _rest_height(model: RodentModel, rest: dict, m: str) -> float:
    floor = model.skin_points(model.qpos())[0][:, 2].min()
    return float(rest[m][0, 2] - floor)


def model_sizes(model: RodentModel, rest: dict | None = None) -> dict:
    """Reference lengths on the rest pose (mm): the back path head_back →
    back_T4 → … → back_V2 (3-D and horizontal), and the ear-to-ear
    distance."""
    rest = rest or model.labels(model.qpos())
    path = [rest[m][0] for m in reversed(CHAIN)]
    seg = [b - a for a, b in zip(path[:-1], path[1:])]
    return {"back_path": round(float(sum(np.linalg.norm(d) for d in seg)), 2),
            "back_path_horizontal": round(float(sum(np.linalg.norm(d[:2]) for d in seg)), 2),
            "ear_width": round(float(np.linalg.norm(rest["head_left"][0] - rest["head_right"][0])), 2)}


def load_limits(path=LIMITS_PATH) -> dict:
    return json.loads(Path(path).read_text())


__all__ = ["CHAIN", "LABELS", "LIMITS_PATH", "REGIONS", "RodentModel", "build_limits", "chain_bends",
           "load_limits", "model_sizes", "region_bends"]
