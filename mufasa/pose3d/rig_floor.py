"""
mufasa.pose3d.rig_floor
=======================

Patch 122le: the body model fitted in the 3-D viewer stays on top of the floor, and fits to
marker heights the body can have.

* **the trunk** (the rig profile's ``trunk`` bones: pelvis, spine, neck, head) never goes below
  the floor. The fit gets one residual per *floor probe* — the lowest trunk vertex of each trunk
  bone, and the ``n_low`` lowest trunk vertices, of the (decimated, drawn) mesh in the rest pose
  — ``relu(FLOOR_Z_MM − z) / FLOOR_SIGMA_MM``: nothing while above the floor, steep below it.
  What is left after the solve is lifted out exactly (:func:`lift`, over every trunk vertex);
* **the limbs and the tail rest on the floor**: the rig's legs cannot bend (they keep the tucked
  stance), so when the spine flexes the forelimbs swing into the floor (−30 to −40 mm at 3–5°
  of lumbar flexion per joint), and K7's fits put the hind paws and the tail there. Lifting the
  whole body off its markers for a leg that cannot bend is the wrong trade, so the drawn mesh
  lays those vertices on the floor instead (:func:`rest_on_floor`), as a flexed leg or a
  dragged tail would be;
* **target heights** — marker heights are clipped to the body model's height range for that
  label (scaled to the fitted size) and the floor before the fit, so files made before the
  heights fix (``mufasa-pose3d heights``) pose sensibly too.

Both the per-frame fit (:meth:`mufasa.pose3d.scene_models.PosedRig.fit_frame`, SciPy) and the
batched one (:mod:`mufasa.pose3d.rig_batch`, PyTorch) use these, so they agree.
"""
from __future__ import annotations

import numpy as np

from mufasa.pose3d.mono import FLOOR_Z_MM

FLOOR_SIGMA_MM = 0.25         # a probe 0.25 mm under the floor costs as much as a marker 4 mm off
N_LOW = 32


def trunk_vertices(rig) -> np.ndarray:
    """Indices into the decimated mesh of the trunk's vertices (their main bone is one of the
    profile's ``trunk`` bones; every vertex when the rig names none). Cached on the rig."""
    hit = getattr(rig, "_trunk_vertices", None)
    if hit is not None:
        return hit
    owner = np.asarray(rig.Wd).argmax(1)
    trunk = (getattr(rig, "profile", None) or {}).get("trunk") or []
    tb = [rig.ix[n] for n in trunk if n in getattr(rig, "ix", {})]
    out = np.flatnonzero(np.isin(owner, tb)) if tb else np.arange(len(owner))
    rig._trunk_vertices = out
    return out


def floor_probes(rig, n_low: int = N_LOW) -> np.ndarray:
    """Indices into the rig's decimated mesh (``Vd``/``Wd``): each trunk bone's lowest vertex in
    the rest pose and the ``n_low`` lowest trunk vertices. Cached on the rig."""
    hit = getattr(rig, "_floor_probes", None)
    if hit is not None:
        return hit
    V, W = np.asarray(rig.Vd, float), np.asarray(rig.Wd)
    tv = trunk_vertices(rig)
    z = V[tv, 2]
    owner = W[tv].argmax(1)
    pick = set(tv[np.argsort(z)[:n_low]].tolist())
    for b in np.unique(owner):
        ix = np.flatnonzero(owner == b)
        pick.add(int(tv[ix[np.argmin(z[ix])]]))
    out = np.array(sorted(pick), dtype=int)
    rig._floor_probes = out
    return out


def floor_residual(z) -> np.ndarray:
    """One residual per probe height (mm): zero above the floor."""
    return np.maximum(FLOOR_Z_MM - np.asarray(z, float), 0.0) / FLOOR_SIGMA_MM


def rest_on_floor(V) -> np.ndarray:
    """The drawn mesh (V, 3) with nothing below the floor: limbs and tail that reach it lie on it."""
    V = np.array(V, float, copy=True)
    V[:, 2] = np.maximum(V[:, 2], FLOOR_Z_MM)
    return V


def lift(probe_z) -> float:
    """How far to raise the body (mm) so its lowest probe is on the floor (0 when above)."""
    z = np.asarray(probe_z, float)
    return float(max(FLOOR_Z_MM - z.min(), 0.0)) if z.size else 0.0


def target_z_range(names, size: float, *, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(lowest, highest) height per marker for a body ``size`` times the model's: the virtual
    rodent's range for the labels it knows, the floor for all."""
    from mufasa.pose3d import rodent_model
    from mufasa.pose3d.mono import Anatomy
    try:
        lim = rodent_model.load_limits()
    except Exception:  # noqa: BLE001 - no limits file: the floor only
        lim = {"label_heights": {}}
    anat = Anatomy(lim, size, {n: n for n in names}, margin=margin)
    r = np.array([anat.zrange_floor(n) for n in names], float).reshape(-1, 2)
    return r[:, 0], r[:, 1]


def clip_targets(Y, lo, hi):
    """Marker targets (..., M, 3) with heights within (lo, hi) per marker (NaN kept)."""
    Y = np.array(Y, float, copy=True)
    Y[..., 2] = np.clip(Y[..., 2], lo, hi)
    return Y


__all__ = ["FLOOR_SIGMA_MM", "N_LOW", "clip_targets", "floor_probes", "floor_residual", "lift", "rest_on_floor",
           "target_z_range", "trunk_vertices"]
