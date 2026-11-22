"""
mufasa.pose3d.gait
==================

Patch 122kw: paw placement, weight distribution and gait from 3-D body
markers, with a physics model of the body and its limbs.

The markers give the body, not the paws. The paws are inferred from what
the body can do.

**The body** (:class:`BodyPhysics`, from DeepMind's virtual rodent; see
:mod:`mufasa.pose3d.rodent_model`):

* **Mass.** Every body segment's mass and centre of mass come from its
  collision shapes (density × volume): 338 g in all, the pelvis and
  abdomen heaviest. Scaled to the animal as size³, or set to its weighed
  mass.
* **Carried by the markers.** Each segment's centre of mass, and each
  limb girdle (hip and shoulder joint), is tied to its nearest markers on
  the standing model: a convex combination of them, plus an offset in the
  back's frame. So the whole-body centre of mass follows whatever the
  markers do: crouching, rearing, the head turned, the tail swung.
* **Reach.** Each limb's reach (girdle to paw) over its joints' ranges.
* **Three sections** (patch 122lf): lower (pelvis, hind legs, tail; 212 g), mid (thorax and
  lumbar spine; 74 g) and upper (shoulders, forelegs, neck, head; 52 g). Each has its centre of
  mass, a heading from its stretch of the back seen from above (none when that stretch is
  steeper than 60°, as when rearing: held from the frames either side), and its velocity and
  momentum along and across that heading. The hips are moved by the lower section and half the
  trunk, the shoulders by the thorax (:data:`LIMB_DRIVE`); stance, swing and stride follow that
  forward speed, not speed in any direction.

**The paws** (:func:`estimate_gait`) are placed as rarely as the body
allows ("minimum placements"):

* A planted paw stays where it is until its leg can no longer reach it,
  or it has fallen a stride's half behind its neutral place.
* It may lift only after a minimum stance, and only if the body stays
  balanced on the other planted paws. Balance means the extrapolated
  centre of mass (position + velocity/ω₀, ω₀ = √(g/h); Hof et al. 2005)
  stays within their support polygon. A paw whose leg can no longer reach
  lifts anyway.
* A paw does not lift while its partner on the other side is in the air,
  unless it must.
* It swings for a physiological time and lands where its girdle's place
  will be at mid-stance, half a stance later, read from the body's actual
  path (scanning forward). When the girdle is too high for the leg to
  reach the floor (rearing forelimbs), the paw stays in the air until it
  can land. A paw's place is beside the midline between its girdle pair,
  half the base of support out across that girdle's heading.
* Scanning backward: once every stance's timing is known, each paw is
  moved to where it stays closest to its place over the whole stance,
  within the leg's reach at every frame of it.

**Weight distribution.** Each frame, the vertical load is m(g + a_z).
It is shared among the planted paws so that their centre of pressure is
at the zero-moment point,

    ZMP = CoM_xy − h · a_xy / (g + a_z),

with the most even sharing that does so (non-negative loads). When the
ZMP lies outside the support polygon, the nearest feasible sharing is
used and the stability margin is negative.

**Bounds** (:data:`BOUNDS`) come from 108 uninjured young adult female rats
crossing a CatWalk (Neckel et al. 2013, J Neurosci Methods 216:62; the
pre-injury runs of their Mendeley data, 5780 strides, 13–71 cm/s). Stride
length grows with speed (65 mm + 0.156 s × speed); stance shortens as a
power of speed while swing stays near 0.14 s, so the duty factor falls
from about 0.7 at 15 cm/s to 0.5 at 60 cm/s (:func:`gait_laws`). Left and
right alternate at half a stride; each forepaw lands about 0.39 of a
stride after its hindpaw (lateral sequence). Base of support 23 mm
(fore) and 29 mm (hind). TactileArena's demo recording (Lu et al. 2026,
13–25 cm/s) agrees within the CatWalk scatter. The bounds are adult-rat
values in mm and s; ``bounds={"scale": k}`` scales lengths by k and times
by √k (dynamic similarity) for smaller or larger animals. Mouse values
(Mendes et al. 2015, BMC Biol 13:50) are kept in :data:`MOUSE_BOUNDS`.

Units: mm, s, g. No Qt here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

G_MM = 9810.0                                          # mm/s²
PAWS = ("LF", "RF", "LH", "RH")
GIRDLE_BODY = {"LH": "upper_leg_L", "RH": "upper_leg_R", "LF": "upper_arm_L", "RF": "upper_arm_R"}
PAW_BODY = {"LH": "toe_L", "RH": "toe_R", "LF": "finger_L", "RF": "finger_R"}
PARTNER = {"LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}
TRUNK = ("back_T4", "back_T8", "back_L2", "back_L6", "back_V2")

# uninjured adult rats on a CatWalk (Neckel et al. 2013, pre-injury runs: 108 rats, 5780 strides;
# scripts/calibrate_gait_catwalk.py; mufasa/assets/gait/catwalk_rat_reference.json)
BOUNDS = {
    "stance_min_s": 0.10,          # 2nd percentile of stance
    "swing_min_s": 0.08,           # 1st–99th percentile of swing
    "swing_max_s": 0.24,
    "stance_law_fore": (13.9, -0.723),   # stance_s = a · speed^b (speed in mm/s)
    "stance_law_hind": (26.3, -0.830),
    "swing_law_fore": (0.366, -0.164),   # swing_s = a · speed^b (≈0.13–0.16 s)
    "swing_law_hind": (0.151, -0.019),
    "law_speed_mm_s": (100.0, 710.0),    # the laws are held at the ends of the speeds seen
    "stride_at_rest_mm": 65.3,     # stride = stride_at_rest + stride_per_speed × speed
    "stride_per_speed_s": 0.156,
    "stride_max_mm": 187.0,        # 99th percentile
    "paw_radius_mm": 6.0,          # the support polygon is the paws' hull widened by this
    "base_of_support_fore_mm": 23.0,   # left–right distance of the paws: 12–34 mm (5–95 %)
    "base_of_support_hind_mm": 29.0,   # 19–39 mm
    "lift_reach": 0.92,            # a planted paw lifts once its leg is this stretched
    "phase_offsets": {"LH": 0.0, "RH": 0.5, "LF": 0.394, "RF": 0.894},   # touch-down in the stride
    "duty_fore": None,             # None: from the laws at the current speed; a number holds it
    "duty_hind": None,
    "stability_tol_mm": 6.0,       # the extrapolated CoM may be this far outside the support
    "scale": 1.0,                  # lengths × scale, times × √scale (an animal unlike an adult rat)
    "source": "CatWalk, uninjured adult female rats (Neckel et al. 2013, Mendeley sg3d7dv345); "
              "agrees with TactileArena's demo (Lu et al. 2026); mouse: Mendes et al. 2015",
}
TACTILEARENA_BOUNDS = {           # the earlier, fixed bounds (TactileArena demo: 58 stances, 13–25 cm/s)
    "stance_min_s": 0.12, "swing_law_fore": (0.15, 0.0), "swing_law_hind": (0.15, 0.0),
    "stride_at_rest_mm": 70.0, "stride_per_speed_s": 0.25, "stride_max_mm": 170.0,
    "base_of_support_fore_mm": 22.0, "base_of_support_hind_mm": 26.0, "duty_fore": 0.6, "duty_hind": 0.78,
    "source": "TactileArena demo recording (Lu et al. 2026), rat, 13–25 cm/s"}
BOUND_SETS = {"catwalk": {}, "tactilearena": TACTILEARENA_BOUNDS}


def bounds_preset(name: str = "catwalk", scale: float = 1.0) -> dict:
    """Bounds by name ("catwalk": speed-dependent, the default; "tactilearena": fixed), scaled."""
    if name not in BOUND_SETS:
        raise ValueError(f"bounds {name!r}: one of {sorted(BOUND_SETS)}")
    return {**BOUNDS, **BOUND_SETS[name], "scale": float(scale)}


def gait_laws(speed_mm_s: float, limb: str, bounds: dict | None = None) -> dict:
    """Stance, swing, stride (s, s, mm) and duty factor of a rat walking at ``speed_mm_s``;
    ``limb`` "F" or "H" (or a paw name). Adult-rat units unless ``bounds["scale"]`` is set."""
    b = {**BOUNDS, **(bounds or {})}
    fh = "F" if limb[-1] == "F" else "H"
    k = float(b.get("scale", 1.0) or 1.0)
    tk = math.sqrt(k)
    v = speed_mm_s / tk if np.isfinite(speed_mm_s) else 0.0      # speed scales as √length
    lo, hi = b["law_speed_mm_s"]
    vl = min(max(v, lo), hi)
    sa, sb = b["stance_law_fore" if fh == "F" else "stance_law_hind"]
    wa, wb = b["swing_law_fore" if fh == "F" else "swing_law_hind"]
    swing = float(np.clip(wa * vl ** wb, b["swing_min_s"], b["swing_max_s"]))
    duty = b["duty_fore" if fh == "F" else "duty_hind"]
    if duty is None:
        stance = sa * vl ** sb
        duty = stance / (stance + swing)
    else:                                                    # a fixed duty factor
        duty = float(duty)
        stance = swing * duty / (1.0 - duty)
    stride = min(b["stride_max_mm"], b["stride_at_rest_mm"] + b["stride_per_speed_s"] * max(v, 0.0))
    return {"stance_s": stance * tk, "swing_s": swing * tk, "stride_mm": stride * k, "duty": duty}


MOUSE_BOUNDS = {"stance_min_s": 0.05, "step_mm": (25.0, 30.0), "duty_factor_run": 0.5,
                "walk_run_speed_mm_s": 528.0, "contralateral_phase": 0.5}


# --------------------------------------------------------------------------- #
# the body
# --------------------------------------------------------------------------- #
def body_frame(P: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(origin (3,), rotation (3, 3)) of the back from markers {name: (3,)}, or None when the back
    is missing: origin the mean of the back labels; x along the back, from its last to its first
    label (head_back when back_T4 is missing), pitched with it; y to the left, level; z up from
    the back. Rearing turns the frame with the body."""
    back = [P[m] for m in TRUNK if m in P and np.isfinite(P[m]).all()]
    if len(back) < 2:
        return None
    o = np.mean(back, axis=0)
    front = next((P[m] for m in ("back_T4", "back_T8", "head_back") if m in P and np.isfinite(P[m]).all()), None)
    rear = next((P[m] for m in ("back_V2", "back_L6", "back_L2") if m in P and np.isfinite(P[m]).all()), None)
    if front is None or rear is None:
        return None
    x = front - rear
    n = np.linalg.norm(x)
    if n < 1e-6:
        return None
    x = x / n
    y = np.cross([0.0, 0.0, 1.0], x)
    if np.linalg.norm(y) < 1e-6:
        return None
    y /= np.linalg.norm(y)
    return o, np.c_[x, y, np.cross(x, y)]                  # columns: body axes in world


ANCHORS = ("head_nose", "head_mid", "head_back", "back_T4", "back_T8", "back_L2", "back_L6", "back_V2",
           "hip_left", "hip_right", "tail_V6", "tail_V18", "tail_V32")


@dataclass
class Anchor:
    """A point carried by nearby markers: a convex combination of them plus an offset in the
    back's frame (both from the standing model)."""
    markers: tuple
    weights: np.ndarray
    offset: np.ndarray               # mm at the model's size, body frame (x forward, z up)


HEAD_SET = ("head_nose", "head_mid", "head_back")
NECK_SET = ("head_back", "back_T4", "back_T8")
TRUNK_SET = ("back_T4", "back_T8", "back_L2", "back_L6", "back_V2", "hip_left", "hip_right")
TAIL_SET = ("back_V2", "tail_V6", "tail_V18", "tail_V32")


def _anchor_set(body: str) -> tuple:
    """The markers a body may hang on: the head on the head's, the neck between head and
    shoulders, the tail on the tail's, everything else (trunk, girdles, limbs) on the back's."""
    if body in ("skull", "jaw"):
        return HEAD_SET
    if body.startswith(("vertebra_cervical", "vertebra_axis", "vertebra_atlant")):
        return NECK_SET
    if body.startswith("vertebra_C"):
        return TAIL_SET
    return TRUNK_SET


def _anchor(point, rest: dict, R, k: int = 3, allowed=ANCHORS) -> Anchor:
    from scipy.optimize import nnls
    names = [m for m in allowed if m in rest]
    d = np.array([np.linalg.norm(rest[m] - point) for m in names])
    near = [names[i] for i in np.argsort(d)[:k]]
    A = np.vstack([np.array([rest[m] for m in near]).T, 50.0 * np.ones(len(near))])
    w, _ = nnls(A, np.r_[point, 50.0])
    w = w / w.sum() if w.sum() > 0 else np.full(len(near), 1.0 / len(near))
    off = (point - np.sum([wi * rest[m] for wi, m in zip(w, near)], axis=0)) @ R
    return Anchor(tuple(near), w, off)


# 122le+1 (122lf): the body in three sections, each with its own mass, heading and momentum
SECTIONS = ("lower", "mid", "upper")
SECTION_AXIS = {                       # (rear labels, front labels): the first present of each
    "lower": (("back_V2", "back_L6"), ("back_L2", "back_T8")),        # pelvis and hindquarters
    "mid": (("back_L2", "back_L6"), ("back_T4", "back_T8")),          # lumbar and thorax
    "upper": (("back_T8", "back_T4"), ("head_back", "head_mid")),     # shoulders, neck, head
}
# what moves a girdle (K7, 3000 stances): the hips the pelvis's section and half the trunk; the
# shoulders the thorax they sit on (the head and neck, scanning, swing on their own: taking them in
# put the forepaws' left–right phase at 0.38, the whole body's momentum at 0.33; CatWalk 0.5)
LIMB_DRIVE = {"H": {"lower": 1.0, "mid": 0.5}, "F": {"mid": 1.0}}
TURN_DEG_S = 90.0              # the mid section turning faster than this: turning
SIDESTEP_RATIO = 0.5           # sideways speed above this share of forward speed: stepping sideways
LEVEL_RATIO = 0.5              # a section's axis seen from above at least this share of its length


def body_section(body: str) -> str:
    """lower (pelvis, hind legs, tail), mid (thorax, lumbar spine) or upper (shoulders, forelegs,
    neck, head) for a virtual-rodent body."""
    import re
    if body == "pelvis" or body.startswith(("upper_leg", "lower_leg", "foot", "toe")) \
            or re.fullmatch(r"vertebra_C\d+", body):
        return "lower"
    if body == "torso" or re.fullmatch(r"vertebra_\d+", body):
        return "mid"
    return "upper"


def _first(P: dict, labels, T: int) -> np.ndarray:
    """Per frame, the first of ``labels`` present (T, 3; NaN where none is)."""
    out = np.full((T, 3), np.nan)
    for m in labels:
        if m in P:
            take = ~np.isfinite(out).all(1) & np.isfinite(P[m]).all(1)
            out[take] = P[m][take]
    return out


def body_frames(names, xyz) -> np.ndarray:
    """:func:`body_frame` for every frame at once: (T, 3, 3), NaN where the back is missing."""
    names = list(names)
    xyz = np.asarray(xyz, float)
    T = len(xyz)
    P = {n: xyz[:, j] for j, n in enumerate(names)}
    n_back = sum(np.isfinite(P[m]).all(1).astype(int) for m in TRUNK if m in P) if any(m in P for m in TRUNK) \
        else np.zeros(T, int)
    front = _first(P, ("back_T4", "back_T8", "head_back"), T)
    rear = _first(P, ("back_V2", "back_L6", "back_L2"), T)
    x = front - rear
    n = np.linalg.norm(x, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = x / n[:, None]
        y = np.stack([-x[:, 1], x[:, 0], np.zeros(T)], 1)              # z × x
        ny = np.linalg.norm(y, axis=1)
        y = y / ny[:, None]
    R = np.stack([x, y, np.cross(x, y)], 2)
    bad = ~(np.isfinite(n) & (n >= 1e-6) & np.isfinite(ny) & (ny >= 1e-6) & (np.asarray(n_back) >= 2))
    R[bad] = np.nan
    return R


def section_headings(names, xyz) -> tuple[np.ndarray, np.ndarray]:
    """Each section's heading seen from above, per frame: unit vectors (T, 3, 2) in
    :data:`SECTIONS` order, and whether the section is level enough for one (T, 3): its axis
    from above at least :data:`LEVEL_RATIO` of its length (a rearing back points up, and its
    heading from above is noise). NaN where not."""
    names = list(names)
    xyz = np.asarray(xyz, float)
    T = len(xyz)
    P = {n: xyz[:, j] for j, n in enumerate(names)}
    H = np.full((T, 3, 2), np.nan)
    ok = np.zeros((T, 3), bool)
    for k, s in enumerate(SECTIONS):
        rear_l, front_l = SECTION_AXIS[s]
        a = _first(P, front_l, T) - _first(P, rear_l, T)
        h = np.linalg.norm(a[:, :2], axis=1)
        L = np.linalg.norm(a, axis=1)
        good = np.isfinite(h) & (h > 1.0) & (h >= LEVEL_RATIO * L)
        H[good, k] = a[good, :2] / h[good, None]
        ok[:, k] = good
    return H, ok


@dataclass
class BodyPhysics:
    """The virtual rodent as a physics model: segment masses carried by the markers, and the
    limbs' reach (mm, kg, at the model's size)."""
    mass_kg: float
    segments: list                   # (mass kg, Anchor) for every body with mass
    girdles: dict                    # paw -> Anchor of its girdle (hip / shoulder joint)
    reach_max: dict                  # paw -> mm
    reach_min: dict
    back_path: float                 # head_back → back_V2 seen from above, standing (mm)
    nose_to_tail: float = 171.1      # head_nose → back_V2, standing (mm)
    check_mm: dict = field(default_factory=dict)   # anchored CoM / girdle error over sampled postures
    sections: tuple = ()             # the section (SECTIONS) of each of ``segments``

    @property
    def section_mass(self) -> dict:
        """kg per section, at the model's size."""
        return {s: float(sum(mkg for (mkg, _), q in zip(self.segments, self.sections) if q == s)) for s in SECTIONS}

    @classmethod
    @lru_cache(maxsize=2)
    def virtual_rodent(cls, n: int = 1500, seed: int = 0) -> BodyPhysics:
        from mufasa.bodymodel.rodent_rig import stance_qpos
        from mufasa.pose3d.rodent_model import SPINE_JOINTS, RodentModel
        m = RodentModel()
        mass, com_local = m.body_masses()
        q0 = stance_qpos(m)
        T0 = m.fk(q0)[0]
        rest = {k: v[0] for k, v in m.labels(q0, T0[None]).items()}
        o, R = body_frame(rest)
        segs, secs = [], []
        for bi in np.flatnonzero(mass > 0):
            c = T0[bi, :3, :3] @ com_local[bi] + T0[bi, :3, 3]
            segs.append((float(mass[bi]), _anchor(c, rest, R, allowed=_anchor_set(m.body_names[bi]))))
            secs.append(body_section(m.body_names[bi]))
        girdles = {p: _anchor(T0[m.body_index[GIRDLE_BODY[p]], :3, 3], rest, R, allowed=TRUNK_SET) for p in PAWS}
        # reach: girdle → paw over the limb joints' full ranges
        rng = np.random.default_rng(seed)
        limb = [k for k, j in enumerate(m.joints) if any(s in j.name for s in ("hip", "knee", "ankle", "toe",
                                                                                  "shoulder", "elbow", "wrist",
                                                                                  "finger", "scapula"))]
        qr = np.tile(q0[0], (n, 1))
        qr[:, limb] = rng.uniform(m.lo[limb], m.hi[limb], (n, len(limb)))
        Tr = m.fk(qr)
        rmax, rmin = {}, {}
        for p in PAWS:
            d = np.linalg.norm(Tr[:, m.body_index[PAW_BODY[p]], :3, 3] - Tr[:, m.body_index[GIRDLE_BODY[p]], :3, 3],
                               axis=1)
            rmax[p], rmin[p] = float(np.percentile(d, 97)), float(np.percentile(d, 3))
        chain = ["head_back", "back_T4", "back_T8", "back_L2", "back_L6", "back_V2"]
        bp = sum(float(np.linalg.norm(rest[b][:2] - rest[a][:2])) for a, b in zip(chain[:-1], chain[1:]))
        nt = float(np.linalg.norm(rest["head_nose"] - rest["back_V2"]))
        phys = cls(float(mass.sum()), segs, girdles, rmax, rmin, bp, nt, sections=tuple(secs))
        # how well the anchored points follow the true ones over sampled spine postures
        ks = [m.joint_index[k] for k in SPINE_JOINTS]
        qs = np.tile(q0[0], (200, 1))
        qs[:, ks] = rng.uniform(0.7 * m.lo[ks], 0.7 * m.hi[ks], (200, len(ks)))
        Ts = m.fk(qs)
        Ls = m.labels(qs, Ts)
        com = np.einsum("b,nbi->ni", mass, np.einsum("nbij,bj->nbi", Ts[:, :, :3, :3], com_local)
                        + Ts[:, :, :3, 3]) / mass.sum()
        ec, eg = [], []
        for i in range(len(qs)):
            y = phys.predict({k: v[i] for k, v in Ls.items()})
            ec.append(np.linalg.norm(y[0] - com[i]))
            eg.append(np.mean([np.linalg.norm(y[1 + j] - Ts[i, m.body_index[GIRDLE_BODY[p]], :3, 3])
                               for j, p in enumerate(PAWS)]))
        phys.check_mm = {"com": round(float(np.median(ec)), 2), "girdles": round(float(np.median(eg)), 2)}
        return phys

    def _point(self, a: Anchor, P: dict, R, size: float):
        if not all(k in P and np.isfinite(P[k]).all() for k in a.markers):
            return None
        return np.sum([w * P[k] for w, k in zip(a.weights, a.markers)], axis=0) + size * (R @ a.offset)

    def _points(self, a: Anchor, P: dict, R, size: float) -> np.ndarray:
        """:meth:`_point` for every frame: (T, 3), NaN where a marker or the frame is missing."""
        if not all(k in P for k in a.markers):
            return np.full((len(R), 3), np.nan)
        return sum(w * P[k] for w, k in zip(a.weights, a.markers)) + size * np.einsum("tij,j->ti", R, a.offset)

    def predict_all(self, names, xyz, size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`predict` for every frame at once: (T, 5, 3) CoM and girdles, and the sections'
        centres of mass (T, 3, 3) in :data:`SECTIONS` order (NaN where under 60 % of its mass is
        seen)."""
        names = list(names)
        xyz = np.asarray(xyz, float)
        T = len(xyz)
        P = {n: xyz[:, j] for j, n in enumerate(names)}
        R = body_frames(names, xyz)
        tot, acc = np.zeros(T), np.zeros((T, 3))
        st, sa = np.zeros((T, 3)), np.zeros((T, 3, 3))
        secs = self.sections or ("mid",) * len(self.segments)
        for (mkg, a), s in zip(self.segments, secs):
            c = self._points(a, P, R, size)
            ok = np.isfinite(c).all(1)
            k = SECTIONS.index(s)
            tot[ok] += mkg
            acc[ok] += mkg * c[ok]
            st[ok, k] += mkg
            sa[ok, k] += mkg * c[ok]
        Y = np.full((T, 5, 3), np.nan)
        good = tot >= 0.6 * self.mass_kg
        Y[good, 0] = acc[good] / tot[good, None]
        for j, p in enumerate(PAWS):
            g = self._points(self.girdles[p], P, R, size)
            Y[good, 1 + j] = g[good]
        S = np.full((T, 3, 3), np.nan)
        sm = self.section_mass
        for k, s in enumerate(SECTIONS):
            ok = good & (st[:, k] >= 0.6 * sm[s]) & (sm[s] > 0)
            S[ok, k] = sa[ok, k] / st[ok, k, None]
        return Y, S

    def predict(self, P: dict, size: float = 1.0):
        """CoM and girdles LF RF LH RH (5, 3) in the world for one frame of markers {name: (3,)} of
        an animal ``size`` times the model, or None when the back is missing. Segments whose
        markers are missing are left out of the CoM (their mass shared by the rest)."""
        fr = body_frame(P)
        if fr is None:
            return None
        _, R = fr
        out = np.full((5, 3), np.nan)
        tot, acc = 0.0, np.zeros(3)
        for mkg, a in self.segments:
            c = self._point(a, P, R, size)
            if c is not None:
                tot += mkg
                acc += mkg * c
        if tot < 0.6 * self.mass_kg:
            return None
        out[0] = acc / tot
        for j, p in enumerate(PAWS):
            g = self._point(self.girdles[p], P, R, size)
            if g is not None:
                out[1 + j] = g
        return out


# --------------------------------------------------------------------------- #
# support, balance, loads
# --------------------------------------------------------------------------- #
def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _hull(P: np.ndarray) -> np.ndarray:
    """Convex hull (counter-clockwise) of 2-D points."""
    P = np.unique(np.round(P, 6), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(P), half(P[::-1])
    return np.array(lo[:-1] + up[:-1])


def support_margin(point: np.ndarray, feet: np.ndarray, radius: float = 0.0) -> float:
    """Signed distance (mm) of a 2-D point inside (+) / outside (−) the support polygon of the
    planted paws, widened by the paws' ``radius`` (one paw: radius minus the distance to it; two:
    to their segment; none: −inf)."""
    return _margin(point, feet) + radius if len(feet) else -np.inf


def _margin(point, feet) -> float:
    if len(feet) == 0:
        return -np.inf
    H = _hull(np.asarray(feet, float))
    if len(H) == 1:
        return -float(np.linalg.norm(point - H[0]))
    if len(H) == 2:
        a, b = H
        t = np.clip(np.dot(point - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
        return -float(np.linalg.norm(point - (a + t * (b - a))))
    d = []
    inside = True
    for a, b in zip(H, np.roll(H, -1, axis=0)):
        e = b - a
        cr = _cross2(e, point - a)
        inside &= cr >= 0
        t = np.clip(np.dot(point - a, e) / max(np.dot(e, e), 1e-12), 0, 1)
        d.append(np.linalg.norm(point - (a + t * e)))
    return float(min(d)) if inside else -float(min(d))


def share_load(feet: np.ndarray, zmp: np.ndarray) -> np.ndarray:
    """Load fractions (≥ 0, summing to 1) of the planted paws whose centre of pressure is as
    close as possible to the ZMP, most evenly shared."""
    k = len(feet)
    if k == 0:
        return np.zeros(0)
    if k == 1:
        return np.ones(1)
    from scipy.optimize import lsq_linear
    A = np.vstack([np.asarray(feet, float).T, np.ones(k)])
    b = np.r_[zmp, 1.0]
    sc = 1.0 / max(np.ptp(feet, axis=0).max(), 1.0)
    W = np.r_[np.full(2, 30.0 * sc), 30.0]
    M = np.vstack([A * W[:, None], 0.05 * np.eye(k)])
    r = lsq_linear(M, np.r_[b * W, np.full(k, 0.05 / k)], bounds=(0, 1))
    w = np.clip(r.x, 0, None)
    return w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)


# --------------------------------------------------------------------------- #
# paw placement
# --------------------------------------------------------------------------- #
@dataclass
class GaitResult:
    frames: pd.DataFrame
    steps: pd.DataFrame
    summary: dict


def _unit_xy(v) -> np.ndarray:
    """Rows scaled to unit length (NaN where zero or missing)."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-9, v / n, np.nan)


def _smooth(x: np.ndarray, fps: float, win_s: float = 0.15) -> np.ndarray:
    """NaN-tolerant smoothing (short gaps filled) with a Savitzky–Golay filter."""
    from scipy.signal import savgol_filter
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return x
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    w = max(5, int(round(win_s * fps)) // 2 * 2 + 1)
    w = min(w, (len(x) // 2) * 2 - 1)
    return savgol_filter(x, w, 2) if w >= 5 else x


def animal_size(phys: BodyPhysics, names, xyz) -> float:
    """The animal's size against the model: the straight distance from the nose to the tail's
    base (back_V2) with the head low (median), against the standing model's. Straight distances
    between the two best-defined points; the back's labels need not sit where the model's do,
    and noise lengthens a path through them."""
    names = list(names)
    for a in ("head_nose", "head_back"):
        if a in names and "back_V2" in names:
            break
    else:
        return 1.0
    ia, ib = names.index(a), names.index("back_V2")
    d = np.linalg.norm(xyz[:, ia] - xyz[:, ib], axis=1)
    hz = xyz[:, ia, 2]
    low = np.isfinite(d) & np.isfinite(hz) & (hz <= np.nanmedian(hz))
    ref = phys.nose_to_tail if a == "head_nose" else phys.back_path
    return float(np.median(d[low]) / ref) if low.sum() >= 5 and ref > 0 else 1.0


def estimate_gait(names, xyz: np.ndarray, frames, fps: float, *, size: float | None = None,
                  mass_g: float | None = None, bounds: dict | None = None,
                  phys: BodyPhysics | None = None) -> GaitResult:
    """Paws, loads and gait from 3-D markers ``xyz`` (T, M, 3; mm, z up from the floor) named
    ``names``, at video frames ``frames`` recorded at ``fps``."""
    phys = phys or BodyPhysics.virtual_rodent()
    b = {**BOUNDS, **(bounds or {})}
    names = list(names)
    xyz = np.asarray(xyz, float)
    frames = np.asarray(frames, int)
    T = len(frames)
    size = float(size) if size else animal_size(phys, names, xyz)
    mass = float(mass_g) / 1000.0 if mass_g else phys.mass_kg * size ** 3
    bk = float(b.get("scale", 1.0) or 1.0)                    # bounds: adult rat × scale
    tsc = math.sqrt(bk)
    dt = 1.0 / fps
    # body: CoM, girdles and the three sections per frame (122lf: all frames at once)
    Y, Ysec = phys.predict_all(names, xyz, size)
    com = np.stack([_smooth(Y[:, 0, k], fps) for k in range(3)], 1)
    gird = np.stack([np.stack([_smooth(Y[:, 1 + i, k], fps, 0.1) for k in range(3)], 1) for i in range(4)], 1)
    vel = np.gradient(com, dt, axis=0)
    acc = np.stack([_smooth(np.gradient(vel[:, k], dt), fps, 0.2) for k in range(3)], 1)
    valid = np.isfinite(Y[:, 0, 0])
    R = body_frames(names, xyz)
    head0 = _unit_xy(np.stack([_smooth(R[:, c, 0], fps, 0.1) for c in range(2)], 1))   # the back's (fallback)
    # 122lf: each section's centre of mass, velocity, heading (level frames only; the others filled
    # from the frames either side) and turning rate
    smass = phys.section_mass
    mscale = mass / phys.mass_kg                             # the animal's mass against the model's
    sec_m = np.array([smass[s] * mscale for s in SECTIONS])  # kg
    H_raw, level = section_headings(names, xyz)
    sec_com = np.full((T, 3, 3), np.nan)
    sec_vel = np.full((T, 3, 3), np.nan)
    sec_head = np.full((T, 3, 2), np.nan)
    for k in range(3):
        seen = np.isfinite(Ysec[:, k, 0])
        if seen.sum() >= 5:
            sec_com[:, k] = np.stack([_smooth(Ysec[:, k, c], fps) for c in range(3)], 1)
            sec_com[~valid, k] = np.nan
            sec_vel[:, k] = np.gradient(sec_com[:, k], dt, axis=0)
        h = H_raw[:, k] if level[:, k].sum() >= 5 else head0
        sec_head[:, k] = _unit_xy(np.stack([_smooth(h[:, c], fps, 0.1) for c in range(2)], 1))
    sec_head = np.where(np.isfinite(sec_head), sec_head, head0[:, None, :])
    yaw = np.stack([_smooth(np.gradient(np.unwrap(np.arctan2(sec_head[:, k, 1], sec_head[:, k, 0])), dt), fps)
                    for k in range(3)], 1)                    # rad/s

    def drive(weights):
        """Mass-weighted velocity (T, 3) and heading (T, 2) of sections ``{name: weight}``,
        leaving out a section not seen in a frame (the whole body's when none is)."""
        w = np.array([weights.get(s, 0.0) for s in SECTIONS]) * sec_m
        have = np.isfinite(sec_vel[:, :, 0]) * w[None, :]
        tot = have.sum(1)
        v = np.einsum("tk,tki->ti", have, np.nan_to_num(sec_vel)) / np.maximum(tot, 1e-12)[:, None]
        h = _unit_xy(np.einsum("k,tki->ti", w, sec_head))
        v[tot <= 0] = vel[tot <= 0]
        return v, np.where(np.isfinite(h), h, head0)
    head = drive({s: 1.0 for s in SECTIONS})[1]             # the body's heading: all three sections
    limb_v, limb_h = {}, {}
    for L in ("F", "H"):
        limb_v[L], limb_h[L] = drive(LIMB_DRIVE[L])
    lat_of = lambda h: np.stack([-h[:, 1], h[:, 0]], 1)         # noqa: E731 - to the animal's left
    v_fwd = np.einsum("ti,ti->t", vel[:, :2], head)
    v_lat = np.einsum("ti,ti->t", vel[:, :2], lat_of(head))
    limb_fwd = {L: np.abs(np.einsum("ti,ti->t", limb_v[L][:, :2], limb_h[L])) for L in limb_v}
    fwd_speed = np.abs(v_fwd)
    speed = np.linalg.norm(vel[:, :2], axis=1)
    omega0 = np.sqrt(G_MM / np.maximum(com[:, 2], 5.0))
    xcom = com[:, :2] + vel[:, :2] / omega0[:, None]
    reach = {p: phys.reach_max[p] * size for p in PAWS}
    tt = frames * dt

    def lsp(p, t):
        """The forward speed of what moves paw p's girdle (its section and half the trunk)."""
        s = limb_fwd[p[1]][t]
        return float(s) if np.isfinite(s) else 0.0

    def stride_at(t):
        """Physiological stride for the body's forward speed (CatWalk rats)."""
        return gait_laws(fwd_speed[t], "F", b)["stride_mm"]
    st_min = b["stance_min_s"] * tsc
    sw_min, sw_max = b["swing_min_s"] * tsc, b["swing_max_s"] * tsc

    def neutral_xy(p, t):
        """Mid-stance: beside the midline between the girdle pair, half the base of support out
        across the heading of what moves this girdle."""
        i, j = PAWS.index(p), PAWS.index(PARTNER[p])
        mid = (gird[t, i, :2] + gird[t, j, :2]) / 2
        h = limb_h[p[1]][t]
        lat = np.array([-h[1], h[0]])                               # to the animal's left
        side = 1.0 if p[0] == "L" else -1.0
        return mid + lat * side * bos_of(p) / 2

    def bos_of(p):
        return b["base_of_support_fore_mm" if p[1] == "F" else "base_of_support_hind_mm"] * bk

    def reachable(p, t, xy):
        g = gird[t, PAWS.index(p)]
        return np.linalg.norm(np.r_[xy, 0.0] - g) <= reach[p]

    def half_stance(p, t):
        """Half the distance the girdle travels forward while a paw is planted (a stance centred
        on the girdle lands this far ahead of it and lifts this far behind)."""
        s = lsp(p, t)
        return 0.5 * s * gait_laws(s, p, b)["stance_s"]

    def at(time_s, lo):
        return int(min(max(np.searchsorted(tt, time_s), lo), T - 1))

    def land_target(p, t, ahead_s=0.0):
        """Where paw p lands ``ahead_s`` from now: where its girdle's neutral place will be at
        mid-stance, half a stance later, read from the body's actual path (scanning forward;
        the momentum carries it on past the end of the recording), pulled back within reach of
        the girdle at touch-down."""
        k = at(tt[t] + ahead_s, t)
        half = 0.5 * gait_laws(lsp(p, k), p, b)["stance_s"]
        km = at(tt[k] + half, k)
        xy = neutral_xy(p, km)
        short = tt[k] + half - tt[km]                       # past the last frame: carry on
        if short > dt:
            xy = xy + limb_v[p[1]][km, :2] * short
        if not np.isfinite(xy).all():
            xy = neutral_xy(p, k)
        g = gird[k, PAWS.index(p)]
        for _ in range(8):                                     # pull back within reach
            if np.linalg.norm(np.r_[xy - g[:2], g[2]]) <= reach[p]:
                break
            xy = g[:2] + 0.85 * (xy - g[:2])
        return xy

    def clock_lift_ok(p, t, now, period):
        """The other side's paw planted for (duty − ½) of a stride (CatWalk rats lift a paw that
        long after the other side's landed; at most a minimum stance); no condition once the
        duty factor is below ½ (both may be in the air)."""
        duty = gait_laws(lsp(p, t), p, b)["duty"]
        if duty < 0.5:
            return True
        q = PARTNER[p]
        lag = (duty - 0.5) * period if np.isfinite(period) else st_min
        return state[q] == "stance" and now - since[q] >= min(st_min, max(lag, dt))

    state = {p: "air" for p in PAWS}
    foot = {p: np.full(2, np.nan) for p in PAWS}
    since = {p: -1e9 for p in PAWS}
    land_at = {p: None for p in PAWS}
    out_pos = np.full((T, 4, 2), np.nan)
    out_con = np.zeros((T, 4), int)              # 1 stance, 0 swing, -1 cannot reach the floor
    steps = []
    t0 = next((t for t in range(T) if valid[t] and np.isfinite(gird[t]).all()), None)
    if t0 is None:
        raise ValueError("no frame has the back markers needed to place the body")
    for p in PAWS:                                             # start: planted under the body
        xy = neutral_xy(p, t0)
        if reachable(p, t0, xy):
            state[p], foot[p], since[p] = "stance", xy, frames[t0] * dt
    stance_start = {p: (t0 if state[p] == "stance" else None) for p in PAWS}
    # the stride clock: one cycle per stride, advancing at speed / stride (still when the body is)
    offs = b["phase_offsets"]
    phi, prev = 0.0, frames[t0] * dt
    for t in range(t0, T):
        if not (valid[t] and np.isfinite(gird[t]).all()):
            continue
        now = frames[t] * dt
        if np.isfinite(fwd_speed[t]):
            phi += (now - prev) * fwd_speed[t] / stride_at(t)      # 122lf: forward progress drives it
        prev = now
        planted = {p: foot[p] for p in PAWS if state[p] == "stance"}
        # 1. lift-offs: forced first, then wanted (one at a time, keeping balance)
        want = []
        stride = stride_at(t)
        period = stride / fwd_speed[t] if fwd_speed[t] > 1e-6 else np.inf
        for p in list(planted):
            g = gird[t, PAWS.index(p)]
            d = np.linalg.norm(np.r_[foot[p], 0.0] - g)
            behind = float(np.dot(foot[p] - neutral_xy(p, t), limb_h[p[1]][t]))
            forced = d > reach[p]
            wanted = d > b["lift_reach"] * reach[p] or behind < -max(half_stance(p, t), 0.25 * stride)
            clock = False
            # the stride clock: past this paw's stance (duty) in its cycle, planted since before
            # (CatWalk rats: hind left 0, hind right ½, fore left 0.39, fore right 0.89)
            duty = gait_laws(lsp(p, t), p, b)["duty"]
            psi = (phi - offs[p]) % 1.0
            if (np.isfinite(period) and period < 4.0 and psi >= duty and behind < 0
                    and since[p] <= now - (psi - duty) * period):
                wanted = clock = True
            if forced or (wanted and now - since[p] >= st_min):
                want.append((not forced, not clock, -d / reach[p], p))
        for soft, off_clock, _, p in sorted(want):
            if p not in planted:
                continue
            if soft and not clock_lift_ok(p, t, now, period):
                continue
            if soft and off_clock:
                # dynamic balance (Hof 2008): the extrapolated CoM, when the paw lands one swing
                # from now, over the other planted paws and the paw where it will land
                sw = float(np.clip(gait_laws(lsp(p, t), p, b)["swing_s"], sw_min, sw_max))
                rest = np.array([planted[q] for q in planted if q != p] + [land_target(p, t, sw)])
                ok_bal = (support_margin(xcom[t] + vel[t, :2] * sw, rest, b["paw_radius_mm"] * size)
                          >= -b["stability_tol_mm"] * size) if len(rest) > 1 else False
                if not ok_bal:
                    continue
            planted.pop(p)
            steps.append((p, stance_start[p], t))
            state[p], since[p] = "swing", now
            # swing: until the stride clock brings this paw's touch-down round (a late paw catches
            # up), within the swings seen; a paw lifted off the clock swings its usual time
            dur = gait_laws(lsp(p, t), p, b)["swing_s"]
            psi = (phi - offs[p]) % 1.0
            if np.isfinite(period) and period < 4.0 and psi >= gait_laws(lsp(p, t), p, b)["duty"]:
                dur = (1.0 - psi) * period
            land_at[p] = now + float(np.clip(dur, sw_min, sw_max))
        # 2. touch-downs
        for p in PAWS:
            if state[p] in ("swing", "air") and (land_at[p] is None or now >= land_at[p]):
                if now - since[p] < sw_min and state[p] == "swing":
                    continue
                xy = land_target(p, t)
                if reachable(p, t, xy):
                    state[p], foot[p], since[p], land_at[p] = "stance", xy, now, None
                    stance_start[p] = t
                    planted[p] = xy
                else:
                    state[p] = "air"
        for i, p in enumerate(PAWS):
            if state[p] == "stance":
                out_pos[t, i], out_con[t, i] = foot[p], 1
            else:
                out_con[t, i] = 0 if state[p] == "swing" else -1
    for p in PAWS:
        if state[p] == "stance" and stance_start[p] is not None:
            steps.append((p, stance_start[p], T - 1))
    # 122lf: scanning backward. Each stance's timing is now known; its paw goes where it stays
    # closest to its girdle's neutral place over the whole stance, within reach throughout
    moved = []
    for p, a, z in steps:
        if a is None:
            continue
        i = PAWS.index(p)
        on = np.arange(a, z + 1)
        on = on[out_con[on, i] == 1]
        if not len(on):
            continue
        new = _place_stance(np.array([neutral_xy(p, t) for t in on]), gird[on, i], reach[p])
        if new is not None:
            moved.append(float(np.linalg.norm(new - out_pos[on[0], i])))
            out_pos[on, i] = new
    # 3. loads
    loads = np.zeros((T, 4))
    zmp = com[:, :2] - com[:, 2:3] * acc[:, :2] / np.maximum(G_MM + acc[:, 2:3], 0.2 * G_MM)
    margin = np.full(T, np.nan)
    for t in range(T):
        on = np.flatnonzero(out_con[t] == 1)
        if not valid[t] or not len(on):
            continue
        w = share_load(out_pos[t, on], zmp[t])
        loads[t, on] = w
        margin[t] = support_margin(zmp[t], out_pos[t, on], b["paw_radius_mm"] * size)
    weight_g = mass * 1000.0 * np.clip(1 + acc[:, 2] / G_MM, 0, None)
    yaw_mid = np.degrees(yaw[:, 1])
    level_mid = level[:, 1]
    turning = np.abs(yaw_mid) > TURN_DEG_S
    sidestep = (speed > 50.0 * tsc) & (np.abs(v_lat) > SIDESTEP_RATIO * np.abs(v_fwd))
    cols = {"frame": frames, "valid": valid, "com_x": com[:, 0], "com_y": com[:, 1], "com_z": com[:, 2],
            "speed_mm_s": speed, "zmp_x": zmp[:, 0], "zmp_y": zmp[:, 1], "xcom_x": xcom[:, 0], "xcom_y": xcom[:, 1],
            "zmp_margin_mm": margin, "n_support": (out_con == 1).sum(1),
            # 122lf: the body's velocity along and across its heading, how fast it turns, and when
            "v_fwd_mm_s": v_fwd, "v_lat_mm_s": v_lat, "yaw_rate_deg_s": yaw_mid,
            "heading_ok": level_mid, "turning": turning, "sidestep": sidestep}
    for k, s in enumerate(SECTIONS):
        h = sec_head[:, k]
        lt = lat_of(h)
        v = sec_vel[:, k]
        cols[f"{s}_com_x"], cols[f"{s}_com_y"], cols[f"{s}_com_z"] = sec_com[:, k].T
        cols[f"{s}_heading_deg"] = np.degrees(np.arctan2(h[:, 1], h[:, 0]))
        cols[f"{s}_yaw_rate_deg_s"] = np.degrees(yaw[:, k])
        cols[f"{s}_v_fwd_mm_s"] = np.einsum("ti,ti->t", v[:, :2], h)
        cols[f"{s}_v_lat_mm_s"] = np.einsum("ti,ti->t", v[:, :2], lt)
        cols[f"{s}_v_z_mm_s"] = v[:, 2]
        # momentum, mN·s (kg × mm/s), along the section's own heading
        for c in ("fwd", "lat", "z"):
            cols[f"{s}_p_{c}_mNs"] = sec_m[k] * cols[f"{s}_v_{c}_mm_s"]
    for i, p in enumerate(PAWS):
        cols[f"{p}_x"], cols[f"{p}_y"] = out_pos[:, i, 0], out_pos[:, i, 1]
        cols[f"{p}_contact"] = out_con[:, i]
        cols[f"{p}_load"] = loads[:, i]
        cols[f"{p}_force_g"] = loads[:, i] * weight_g
        cols[f"{p}_girdle_x"], cols[f"{p}_girdle_y"], cols[f"{p}_girdle_z"] = gird[:, i].T
    fdf = pd.DataFrame(cols)
    straight = level_mid & ~turning & ~sidestep
    sdf = _steps_table(steps, frames, out_pos, fps, straight=straight, v_fwd=v_fwd)
    summ = _summary(fdf, sdf, fps, size, mass, phys, b)
    # flat: the run's record (run.toml) takes one level of tables
    for k, s in enumerate(SECTIONS):
        summ[f"{s}_mass_g"] = round(float(sec_m[k] * 1000), 1)
        summ[f"{s}_median_v_fwd_mm_s"] = _r(np.nanmedian(fdf[f"{s}_v_fwd_mm_s"][valid])) if valid.any() else None
        summ[f"{s}_median_abs_v_lat_mm_s"] = (_r(np.nanmedian(np.abs(fdf[f"{s}_v_lat_mm_s"][valid])))
                                              if valid.any() else None)
        summ[f"{s}_level"] = round(float(level[valid, k].mean()), 3) if valid.any() else None
    summ["drive"] = "; ".join(f"{'hind' if L == 'H' else 'fore'}: " + " + ".join(
        f"{w:g} {s}" for s, w in LIMB_DRIVE[L].items()) for L in ("H", "F"))
    summ["turning"] = round(float(turning[valid].mean()), 3) if valid.any() else None
    summ["sidestep"] = round(float(sidestep[valid].mean()), 3) if valid.any() else None
    summ["replaced_mm"] = _r(np.median(moved)) if moved else None
    return GaitResult(fdf, sdf, summ)


def _r(x, n=1):
    return round(float(x), n) if np.isfinite(x) else None


def _place_stance(neutral, girdle, reach, iters: int = 30):
    """Where a planted paw should stand (2,): closest on average to its neutral places over the
    stance (their mean), within the leg's reach of the girdle at every frame of it (alternating
    projections onto the reachable discs). None when no point is within reach throughout."""
    ok = np.isfinite(neutral).all(1) & np.isfinite(girdle).all(1)
    if not ok.any():
        return None
    nb, gd = neutral[ok], girdle[ok]
    r = np.sqrt(np.maximum(reach ** 2 - gd[:, 2] ** 2, 0.0))         # the reachable disc on the floor
    xy = nb.mean(0)
    for _ in range(iters):
        d = xy - gd[:, :2]
        n = np.linalg.norm(d, axis=1)
        far = n > r
        if not far.any():
            return xy
        k = int(np.argmax(n - r))
        xy = gd[k, :2] + d[k] * (r[k] / max(n[k], 1e-9)) * 0.999
    d = np.linalg.norm(xy - gd[:, :2], axis=1)
    return xy if (d <= r + 0.5).all() else None             # none: the forward pass's place stays


def _steps_table(steps, frames, pos, fps, *, straight=None, v_fwd=None) -> pd.DataFrame:
    rows = []
    for p, a, z in steps:
        if a is None:
            continue
        i = PAWS.index(p)
        m = (a + z) // 2
        rows.append({"paw": p, "on_frame": int(frames[a]), "off_frame": int(frames[z]),
                     "on_s": frames[a] / fps, "off_s": frames[z] / fps,
                     "stance_s": (frames[z] - frames[a]) / fps, "x": pos[a, i, 0], "y": pos[a, i, 1],
                     "v_fwd_mm_s": float(v_fwd[m]) if v_fwd is not None else np.nan,
                     "straight": bool(straight[a:z + 1].mean() >= 0.5) if straight is not None else True})
    df = pd.DataFrame(rows, columns=["paw", "on_frame", "off_frame", "on_s", "off_s", "stance_s", "x", "y",
                                     "v_fwd_mm_s", "straight"])
    df = df.sort_values(["paw", "on_s"]).reset_index(drop=True)
    nxt = df.groupby("paw").shift(-1)
    df["stride_s"] = nxt["on_s"] - df["on_s"]
    df["swing_s"] = nxt["on_s"] - df["off_s"]
    df["stride_mm"] = np.hypot(nxt["x"] - df["x"], nxt["y"] - df["y"])
    df["duty_factor"] = df["stance_s"] / df["stride_s"]
    return df


WALK_MIN_MM_S = 100.0      # strides slower than this, or longer than a second, are not walking


def _walking(sdf) -> pd.DataFrame:
    """Strides of walking: at least WALK_MIN_MM_S, at most a second (the CatWalk's range), and
    (122lf) straight: the stance mostly level, not turning and not stepping sideways."""
    w = sdf.dropna(subset=["stride_s"])
    w = w[(w.stride_s > 0) & (w.stride_s <= 1.0)]
    if "straight" in w:
        w = w[w.straight.astype(bool)]
    return w[w.stride_mm / w.stride_s >= WALK_MIN_MM_S]


def _phase(sdf, a, b, walking=None) -> float:
    """Median phase of paw b's touch-downs in paw a's (walking) strides (0–1)."""
    A = (walking if walking is not None else sdf.dropna(subset=["stride_s"]))
    A = A[A.paw == a]
    B = sdf[sdf.paw == b]["on_s"].to_numpy()
    ph = []
    for _, r in A.iterrows():
        k = B[(r.on_s <= B) & (r.on_s + r.stride_s > B)]
        if len(k) and r.stride_s > 0:
            ph.append((k[0] - r.on_s) / r.stride_s)
    return float(np.median(ph)) if ph else float("nan")


def _summary(fdf, sdf, fps, size, mass, phys, bounds=None) -> dict:
    mv = fdf[fdf.valid]
    walking = _walking(sdf)
    s = {"frames": int(len(fdf)), "fps": fps, "size_vs_model": round(size, 3), "mass_g": round(mass * 1000, 1),
         "model_mass_g": round(phys.mass_kg * 1000, 1), "steps": int(len(sdf)),
         "median_speed_mm_s": round(float(np.nanmedian(mv.speed_mm_s)), 1) if len(mv) else None,
         "support": {int(k): round(float(v), 3) for k, v in (mv.n_support.value_counts(normalize=True)).items()},
         "load_share": {p: round(float(mv[f"{p}_load"].mean()), 3) for p in PAWS} if len(mv) else {},
         "zmp_outside_support": round(float((mv.zmp_margin_mm < 0).mean()), 3) if len(mv) else None,
         "bounds": bounds or BOUNDS}
    for p in PAWS:
        w = walking[walking.paw == p]
        if len(w):
            s[f"{p}_stride_mm"] = round(float(w.stride_mm.median()), 1)
            s[f"{p}_stance_s"] = round(float(w.stance_s.median()), 3)
            s[f"{p}_duty_factor"] = round(float(w.duty_factor.median()), 3)
    s["walking_strides"] = int(len(walking))
    s["phase_LF_RF"] = round(_phase(sdf, "LF", "RF", walking), 3)
    s["phase_LH_RH"] = round(_phase(sdf, "LH", "RH", walking), 3)
    s["phase_LH_LF"] = round(_phase(sdf, "LH", "LF", walking), 3)
    s["phase_LH_RF"] = round(_phase(sdf, "LH", "RF", walking), 3)
    return s


def gait_from_pose3d(path, *, fps: float | None = None, **kw) -> GaitResult:
    """:func:`estimate_gait` on a 3-D pose file (``<video>_3d.parquet``)."""
    from mufasa.pose3d.io import check_pose3d, marker_names, read_pose3d
    df, meta = read_pose3d(path)
    names = marker_names(df)
    xyz = np.stack([df[[f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(float) for n in names], 1)
    frames = df["frame"].to_numpy(int) if "frame" in df else np.arange(len(df))
    # 122kz: what the file breaks of its contract travels with the result
    warn = [w for w in check_pose3d(df, meta) if not (fps and w.startswith("no frame rate"))]
    if not (fps or meta.get("fps")):
        warn = [w.replace("times assume a default", "assumed 30 fps; give the video's frame rate")
                if w.startswith("no frame rate") else w for w in warn]
    fps = fps or meta.get("fps") or 30.0
    res = estimate_gait(names, xyz, frames, float(fps), **kw)
    res.summary["contract"] = warn
    return res


def write_gait(res: GaitResult, out, meta: dict | None = None):
    """``<video>_gait.parquet`` (per frame; summary in the metadata) and ``<video>_steps.csv``."""
    import json
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pandas(res.frames, preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}),
                                   b"mufasa_gait": json.dumps({**(meta or {}), "summary": res.summary},
                                                              default=float).encode()})
    pq.write_table(t, out)
    stem = out.name[:-len("_gait.parquet")] if out.name.endswith("_gait.parquet") else out.stem
    steps = out.with_name(stem + "_steps.csv")
    res.steps.to_csv(steps, index=False)
    return out, steps


def read_gait(path) -> tuple[pd.DataFrame, dict]:
    import json

    import pyarrow.parquet as pq
    t = pq.read_table(path)
    raw = (t.schema.metadata or {}).get(b"mufasa_gait")
    return t.to_pandas(), (json.loads(raw) if raw else {})


__all__ = ["BOUNDS", "BodyPhysics", "GaitResult", "MOUSE_BOUNDS", "PAWS", "animal_size", "body_frame",
           "estimate_gait", "gait_from_pose3d", "gait_laws", "TACTILEARENA_BOUNDS", "BOUND_SETS", "bounds_preset", "read_gait", "share_load", "support_margin", "write_gait"]
