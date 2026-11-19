"""
mufasa.bodymodel.rodent_rig
===========================

Patch 122ks: DeepMind's virtual rodent (a rat; dm_control, Apache 2.0) as a
rig for the Body model tab, built from the packaged model
(:mod:`mufasa.pose3d.rodent_model`) — no Blender needed.

The rig follows the conventions of the HBP mouse rig: millimetres, the
animal facing −y with its left at +x and z up, the floor at z = 0, rooted
at the pelvis, spine and neck bones pointing towards the head (each turns
about its posterior joint), tail bones pointing away from it. Bones:

* ``Pelvis`` (pelvis and the last lumbar vertebra), ``Spine1``–``Spine6``
  (the lumbar vertebrae and, last, the rigid thorax; the six lumbar
  joints), ``Neck1``–``Neck7`` (the cervical vertebrae, axis and atlas),
  ``Head`` (skull and jaw, turning at the atlas joint);
* ``Tail1``–``Tail30``;
* hind legs ``Femur``, ``Tibia``, ``BFoot``, ``BPhalange1`` (.L/.R);
  fore legs ``Scapula``, ``Humerus``, ``Radius``, ``FHand``, ``FPhalange1``.

The profile stored with the rig gives the fitter the spine regions and the
posture limits, per joint, from the model's joint ranges spread over the
bones of each region:

* lumbar (6 joints): sideways ±10°, up/down −15° (extension) … +10°
  (flexion) — the model's two bend joints ±30° and two extension joints
  −30…+45°;
* cervical (7 bones): sideways ±10°, up/down −10…+15° — three extension
  joints −23…+35°, two bend joints ±35°;
* head (atlas): up/down −45…+5°; sideways ±30° (assumed);
* tail (29 joints; the first fixed): ±8° per joint (the model allows ±15°);
* no thoracic bend (the model's thorax is rigid).

The labels sit where :mod:`mufasa.pose3d.rodent_model` puts them (dorsal
skin points above the vertebrae, the crown for the ears, the hip joints,
caudal vertebrae), stored in the profile, so the fitter does not have to
guess them from the label spacing. The skin has no texture; it is drawn
light grey.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# rodent frame (x forward, y left, z up) -> HBP rig frame (facing -y, left +x, z up)
R_FRAME = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

LUMBAR = ["vertebra_5", "vertebra_4", "vertebra_3", "vertebra_2", "vertebra_1", "torso"]   # Spine1..6
NECK = ["vertebra_cervical_5", "vertebra_cervical_4", "vertebra_cervical_3", "vertebra_cervical_2",
        "vertebra_cervical_1", "vertebra_axis", "vertebra_atlant"]                          # Neck1..7

POSTURE = [("rear", -10, 60), ("roll", -15, 15),
           ("lum_yaw", -10, 10), ("cer_yaw", -10, 10), ("head_yaw", -30, 30),
           ("lum_pitch", -15, 10), ("cer_pitch", -10, 15), ("head_pitch", -45, 5),
           ("tail_yaw_p", -8, 8), ("tail_yaw_d", -8, 8), ("tail_pitch", -8, 8)]
LIMITS = {
    "lumbar (Spine1-6)": "sideways ±10°, up/down -15° (extension) … +10° (flexion) per joint "
                         "(virtual rodent: two bend joints ±30°, two extension joints -30…45°)",
    "thoracic": "rigid (virtual rodent)",
    "cervical (Neck1-7)": "sideways ±10°, up/down -10…15° per bone (virtual rodent: three extension joints "
                          "-23…35°, two bend joints ±35°)",
    "head": "up/down -45…5° (virtual rodent atlas joint), sideways ±30° (assumed)",
    "tail (Tail2-30)": "±8° each way per joint (virtual rodent: ±15°); Tail1 fixed",
    "body": "rearing -10…60° and roll ±15° about the hind paws, which stay on the floor (assumed)",
    "floor": "no part below the floor",
    "limbs": "rest pose",
}


def _bones(model):
    """(name, rodent bodies, parent name, head point, tail point) in the rodent frame, mm."""
    q0 = model.qpos()
    Tw = model.fk(q0)[0]
    org = {n: Tw[i, :3, 3] for n, i in model.body_index.items()}

    def anchor(body):
        """The pivot of the joint(s) that attach ``body`` to its parent (its origin if none)."""
        i = model.body_index[body]
        js = [j for j in model.joints if j.body == i]
        p = js[0].pos if js else np.zeros(3)
        return Tw[i, :3, :3] @ p + Tw[i, :3, 3]
    out = []
    out.append(("Pelvis", ["pelvis", "vertebra_6"], None, org["vertebra_C1"], anchor("vertebra_6")))
    prev = "Pelvis"
    lumbar_pivots = [anchor(f"vertebra_{k}") for k in (6, 5, 4, 3, 2, 1)]        # J6 … J1
    for k, body in enumerate(LUMBAR):
        tail = lumbar_pivots[k + 1] if k + 1 < len(LUMBAR) else anchor(NECK[0])
        out.append((f"Spine{k + 1}", [body], prev, lumbar_pivots[k], tail))
        prev = f"Spine{k + 1}"
    thorax = prev
    for k, body in enumerate(NECK):
        tail = anchor(NECK[k + 1]) if k + 1 < len(NECK) else anchor("skull")
        out.append((f"Neck{k + 1}", [body], prev, anchor(body), tail))
        prev = f"Neck{k + 1}"
    V = model.skin_points(q0)[0]
    skull = np.flatnonzero(model.weights.argmax(1) == model.body_index["skull"])
    nose = V[skull[np.argmax(V[skull, 0])]]
    out.append(("Head", ["skull", "jaw"], prev, anchor("skull"), nose))
    prev = "Pelvis"
    for k in range(1, 31):
        b = f"vertebra_C{k}"
        tail = anchor(f"vertebra_C{k + 1}") if k < 30 else org[b] + (org[b] - org["vertebra_C29"])
        out.append((f"Tail{k}", [b], prev, anchor(b), tail))
        prev = f"Tail{k}"
    for s, S in (("L", "L"), ("R", "R")):
        legs = [("Femur", "upper_leg"), ("Tibia", "lower_leg"), ("BFoot", "foot"), ("BPhalange1", "toe")]
        prev = "Pelvis"
        for k, (nm, body) in enumerate(legs):
            b = f"{body}_{S}"
            tail = org[f"{legs[k + 1][1]}_{S}"] if k + 1 < len(legs) else org[b] + np.array([6.0, 0, 0])
            out.append((f"{nm}.{s}", [b], prev, org[b], tail))
            prev = f"{nm}.{s}"
        arms = [("Scapula", "scapula"), ("Humerus", "upper_arm"), ("Radius", "lower_arm"), ("FHand", "hand"),
                ("FPhalange1", "finger")]
        prev = thorax
        for k, (nm, body) in enumerate(arms):
            b = f"{body}_{S}"
            tail = org[f"{arms[k + 1][1]}_{S}"] if k + 1 < len(arms) else org[b] + np.array([4.0, 0, 0])
            out.append((f"{nm}.{s}", [b], prev, org[b], tail))
            prev = f"{nm}.{s}"
    return out


def decimate(V, F, W, cell: float):
    """Vertex clustering on a ``cell`` mm grid: vertices in a cell merge
    (mean position and weights); faces that collapse are dropped."""
    key = np.floor(V / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n).astype(float)
    Vd = np.stack([np.bincount(inv, V[:, k], n) for k in range(3)], 1) / cnt[:, None]
    Wd = np.stack([np.bincount(inv, W[:, k], n) for k in range(W.shape[1])], 1) / cnt[:, None]
    Fd = inv[F]
    ok = (Fd[:, 0] != Fd[:, 1]) & (Fd[:, 1] != Fd[:, 2]) & (Fd[:, 0] != Fd[:, 2])
    Fd = Fd[ok]
    _, first = np.unique(np.sort(Fd, axis=1), axis=0, return_index=True)
    Fd = Fd[np.sort(first)]                                            # one of each, winding kept
    return Vd, Fd, Wd


def export_rodent_rig(out, *, cell: float = 3.0, model=None, say=print) -> Path:
    """Write the virtual rodent as a ``.rig.npz`` for :class:`mufasa.bodymodel.fit.Rig`."""
    from mufasa.pose3d.rodent_model import RodentModel
    model = model or RodentModel()
    bones = _bones(model)
    names = [b[0] for b in bones]
    ix = {n: i for i, n in enumerate(names)}
    V0 = model.skin_points(model.qpos())[0]
    floor = V0[:, 2].min()

    def tf(P):
        P = np.asarray(P, float) @ R_FRAME.T
        P[..., 2] -= floor
        return P
    V = tf(V0)
    W = np.zeros((len(V), len(names)), np.float32)
    for k, (_, bodies, *_rest) in enumerate(bones):
        for b in bodies:
            W[:, k] += model.weights[:, model.body_index[b]]
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    F = model.faces.copy()
    # outward normals: most faces of a closed skin should point away from its centroid
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if np.mean(np.einsum("ij,ij->i", nrm, tri.mean(1) - V.mean(0)) > 0) < 0.5:
        F = F[:, ::-1].copy()
    Vd, Fd, Wd = decimate(V, F, W, cell)
    rest = np.tile(np.eye(4), (len(names), 1, 1))
    rest[:, :3, 3] = tf(np.array([b[3] for b in bones]))
    tails = tf(np.array([b[4] for b in bones]))
    parent = np.array([ix[b[2]] if b[2] else -1 for b in bones])
    assert all(parent[i] < i for i in range(len(names))), "bones must be parents first"
    rgb = np.full((len(F), 3), 215, np.uint8)
    # where each label sits (as in mufasa.pose3d.rodent_model): skin vertices, or points on bones
    body_bone = {b: k for k, (_, bodies, *_r) in enumerate(bones) for b in bodies}
    Tw = model.fk(model.qpos())[0]
    markers = {}
    for n, sp in model.label_points().items():
        if sp[0] == "vertex":
            markers[n] = ["vert", [int(sp[1])]]
        else:
            body = model.body_names[sp[1]]
            pt = Tw[sp[1], :3, :3] @ sp[2] + Tw[sp[1], :3, 3]
            markers[n] = ["bone", names[body_bone[body]], tf(pt).tolist()]
    profile = {
        "name": "virtual rodent (rat)",
        "groups": {"lum": [f"Spine{i}" for i in range(1, 7)], "tho": [],
                   "cer": [f"Neck{i}" for i in range(1, 8)],
                   "tail_p": [f"Tail{i}" for i in range(2, 14)], "tail_d": [f"Tail{i}" for i in range(14, 31)]},
        "head": "Head",
        "trunk": ["Head"] + [f"Neck{i}" for i in range(7, 0, -1)] + [f"Spine{i}" for i in range(6, 0, -1)] + ["Pelvis"],
        "tail": [f"Tail{i}" for i in range(1, 31)],
        "tail_hidden": [f"Tail{i}" for i in range(4, 31)],
        "paws": ["BPhalange1.L", "BPhalange1.R", "BFoot.L", "BFoot.R"],
        "hips": {"hip_left": "Femur.L", "hip_right": "Femur.R"},
        "markers": markers,
        "posture": [list(p) for p in POSTURE],
        "prior_centre": {p[0]: 0.0 for p in POSTURE},          # the rest pose (ranges are lopsided)
        "limits": LIMITS,
    }
    meta = {"source": "DeepMind virtual rodent (dm_control rodent.xml + rodent_walker_skin.skn, Apache 2.0)",
            "units": "mm", "skin_scale": model.skin_scale, "decimate_cell_mm": cell}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=np.array(names), parent=parent, rest=rest, tails=tails, V=V, F=F, W=W,
                        Vd=Vd, Fd=Fd, Wd=Wd.astype(np.float32), face_rgb=rgb, meta=json.dumps(meta),
                        profile=json.dumps(profile))
    say(f"rig: {len(names)} bones, {len(V)} vertices ({len(Vd)} for fitting) -> {out}")
    return out


__all__ = ["LIMITS", "POSTURE", "decimate", "export_rodent_rig"]
