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
* It swings for a physiological time and lands ahead of its girdle,
  where the girdle will be by then, by half a stride. When the girdle
  is too high for the leg to reach the floor (rearing forelimbs), the
  paw stays in the air until it can land. At mid-stance a paw is beside
  the body's midline, at half the base of support.

**Weight distribution.** Each frame, the vertical load is m(g + a_z).
It is shared among the planted paws so that their centre of pressure is
at the zero-moment point,

    ZMP = CoM_xy − h · a_xy / (g + a_z),

with the most even sharing that does so (non-negative loads). When the
ZMP lies outside the support polygon, the nearest feasible sharing is
used and the stability margin is negative.

**Bounds** (:data:`BOUNDS`) are those of rats walking on TactileArena's
pressure runway (Lu et al. 2026; the demo recording's 58 labelled
stances, 13–25 cm/s): stance ≥ 0.14 s, swing 0.09–0.29 s, strides
10–13.5 cm, duty factor about 0.6 (forelimbs) and 0.78 (hindlimbs), base
of support 17–33 mm (fore) and 14–30 mm (hind). The times scale with the square root of
the animal's size against the virtual rodent (dynamic similarity). Mouse
bounds (Mendes et al. 2015, BMC Biol 13:50) are kept for reference.

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

# rat walking on a pressure runway (TactileArena demo, 5 bouts, 13–25 cm/s), at the virtual rodent's size
BOUNDS = {
    "stance_min_s": 0.12,          # shortest stance seen 0.14 s
    "swing_s": 0.15,               # typical swing (0.09–0.29 s seen)
    "swing_min_s": 0.08,
    "swing_max_s": 0.30,
    "stride_mm": 120.0,            # strides 100–135 mm at 13–25 cm/s
    "stride_at_rest_mm": 70.0,     # stride = stride_at_rest + stride_per_speed × speed (fitted to the
    "stride_per_speed_s": 0.25,    # demo's bouts: 100 mm at 13 cm/s … 135 mm at 25 cm/s)
    "stride_max_mm": 170.0,
    "paw_radius_mm": 6.0,          # the support polygon is the paws' hull widened by this
    "base_of_support_fore_mm": 22.0,   # left–right distance of the paws: 17–33 mm (median 19)
    "base_of_support_hind_mm": 26.0,   # 14–30 mm (median 27)
    "lift_reach": 0.92,            # a planted paw lifts once its leg is this stretched
    "duty_fore": 0.6,              # duty factors seen (alternation: a paw lifts (duty − 0.5) of a
    "duty_hind": 0.78,             # stride after the other side's paw landed)
    "stability_tol_mm": 6.0,       # the extrapolated CoM may be this far outside the support
    "source": "TactileArena demo recording (Lu et al. 2026), rat; mouse: Mendes et al. 2015",
}
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
    check_mm: dict = field(default_factory=dict)   # anchored CoM / girdle error over sampled postures

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
        segs = []
        for bi in np.flatnonzero(mass > 0):
            c = T0[bi, :3, :3] @ com_local[bi] + T0[bi, :3, 3]
            segs.append((float(mass[bi]), _anchor(c, rest, R, allowed=_anchor_set(m.body_names[bi]))))
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
        phys = cls(float(mass.sum()), segs, girdles, rmax, rmin, bp)
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
    """The animal's size against the model: the back's length from above, head low (median)."""
    chain = [m for m in ("head_back", "back_T4", "back_T8", "back_L2", "back_L6", "back_V2") if m in names]
    if len(chain) < 6:
        return 1.0
    ix = [list(names).index(m) for m in chain]
    path = sum(np.linalg.norm(xyz[:, b, :2] - xyz[:, a, :2], axis=1) for a, b in zip(ix[:-1], ix[1:]))
    hz = xyz[:, ix[0], 2]
    low = np.isfinite(path) & np.isfinite(hz) & (hz <= np.nanmedian(hz))
    return float(np.median(path[low]) / phys.back_path) if low.sum() >= 5 else 1.0


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
    tsc = math.sqrt(size)                                     # times scale as √length
    dt = 1.0 / fps
    # body: CoM and girdles per frame
    Y = np.full((T, 5, 3), np.nan)
    for t in range(T):
        P = {n: xyz[t, j] for j, n in enumerate(names)}
        y = phys.predict(P, size)
        if y is not None:
            Y[t] = y
    com = np.stack([_smooth(Y[:, 0, k], fps) for k in range(3)], 1)
    gird = np.stack([np.stack([_smooth(Y[:, 1 + i, k], fps, 0.1) for k in range(3)], 1) for i in range(4)], 1)
    vel = np.gradient(com, dt, axis=0)
    acc = np.stack([_smooth(np.gradient(vel[:, k], dt), fps, 0.2) for k in range(3)], 1)
    valid = np.isfinite(Y[:, 0, 0])
    head = np.full((T, 2), np.nan)                           # body heading (x axis of the back frame)
    for t in range(T):
        fr = body_frame({n: xyz[t, j] for j, n in enumerate(names)})
        if fr is not None:
            head[t] = fr[1][:2, 0]
    for k in range(2):
        head[:, k] = _smooth(head[:, k], fps, 0.1)
    head /= np.maximum(np.linalg.norm(head, axis=1, keepdims=True), 1e-9)
    speed = np.linalg.norm(vel[:, :2], axis=1)
    omega0 = np.sqrt(G_MM / np.maximum(com[:, 2], 5.0))
    xcom = com[:, :2] + vel[:, :2] / omega0[:, None]
    reach = {p: phys.reach_max[p] * size for p in PAWS}

    def stride_at(t):
        """Physiological stride for the speed (dynamically similar: speed scales with √size)."""
        v = speed[t] / tsc if np.isfinite(speed[t]) else 0.0
        return size * min(b["stride_max_mm"], b["stride_at_rest_mm"] + b["stride_per_speed_s"] * v)
    st_min = b["stance_min_s"] * tsc
    sw_nom, sw_min, sw_max = b["swing_s"] * tsc, b["swing_min_s"] * tsc, b["swing_max_s"] * tsc

    def neutral_xy(p, t):
        """Mid-stance: beside the midline between the girdle pair, half the base of support out."""
        i, j = PAWS.index(p), PAWS.index(PARTNER[p])
        mid = (gird[t, i, :2] + gird[t, j, :2]) / 2
        lat = np.array([-head[t, 1], head[t, 0]])                   # to the animal's left
        side = 1.0 if p[0] == "L" else -1.0
        bos = b["base_of_support_fore_mm" if p[1] == "F" else "base_of_support_hind_mm"] * size
        return mid + lat * side * bos / 2

    def reachable(p, t, xy):
        g = gird[t, PAWS.index(p)]
        return np.linalg.norm(np.r_[xy, 0.0] - g) <= reach[p]

    def land_target(p, t):
        v = vel[t, :2]
        s = np.linalg.norm(v)
        fwd = v / s if s > 1e-6 else head[t]
        stride = stride_at(t)
        k = min(1.0, s / max(stride / (2 * sw_nom), 1e-6))     # stationary: under the girdle
        xy = neutral_xy(p, t) + fwd * 0.5 * stride * k
        g = gird[t, PAWS.index(p), :2]
        for _ in range(8):                                     # pull back within reach
            if reachable(p, t, xy):
                break
            xy = g + 0.85 * (xy - g)
        return xy

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
    for t in range(t0, T):
        if not (valid[t] and np.isfinite(gird[t]).all()):
            continue
        now = frames[t] * dt
        planted = {p: foot[p] for p in PAWS if state[p] == "stance"}
        # 1. lift-offs: forced first, then wanted (one at a time, keeping balance)
        want = []
        stride = stride_at(t)
        period = stride / speed[t] if speed[t] > 1e-6 else np.inf
        for p in list(planted):
            g = gird[t, PAWS.index(p)]
            d = np.linalg.norm(np.r_[foot[p], 0.0] - g)
            behind = float(np.dot(foot[p] - neutral_xy(p, t), head[t]))
            forced = d > reach[p]
            wanted = d > b["lift_reach"] * reach[p] or behind < -0.5 * stride
            # alternation: (duty − ½) of a stride after the other side's paw landed, if behind its place
            q = PARTNER[p]
            duty = b["duty_fore" if p[1] == "F" else "duty_hind"]
            if (np.isfinite(period) and period < 4.0 and state[q] == "stance" and behind < 0
                    and now - since[q] >= (duty - 0.5) * period and since[q] > since[p]):
                wanted = True
            if forced or (wanted and now - since[p] >= st_min):
                want.append((not forced, -d / reach[p], p))
        for soft, _, p in sorted(want):
            if p not in planted:
                continue
            if soft:
                rest = np.array([planted[q] for q in planted if q != p])
                ok_bal = (support_margin(xcom[t], rest, b["paw_radius_mm"] * size)
                          >= -b["stability_tol_mm"] * size) if len(rest) else False
                # the other side's paw: planted, and for at least a minimum stance (alternation)
                partner_ok = state[PARTNER[p]] == "stance" and now - since[PARTNER[p]] >= st_min
                if not ok_bal or not partner_ok:
                    continue
            planted.pop(p)
            steps.append((p, stance_start[p], t))
            state[p], since[p] = "swing", now
            dur = float(np.clip(sw_nom, sw_min, sw_max))
            land_at[p] = now + dur
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
    cols = {"frame": frames, "valid": valid, "com_x": com[:, 0], "com_y": com[:, 1], "com_z": com[:, 2],
            "speed_mm_s": speed, "zmp_x": zmp[:, 0], "zmp_y": zmp[:, 1], "xcom_x": xcom[:, 0], "xcom_y": xcom[:, 1],
            "zmp_margin_mm": margin, "n_support": (out_con == 1).sum(1)}
    for i, p in enumerate(PAWS):
        cols[f"{p}_x"], cols[f"{p}_y"] = out_pos[:, i, 0], out_pos[:, i, 1]
        cols[f"{p}_contact"] = out_con[:, i]
        cols[f"{p}_load"] = loads[:, i]
        cols[f"{p}_force_g"] = loads[:, i] * weight_g
        cols[f"{p}_girdle_x"], cols[f"{p}_girdle_y"], cols[f"{p}_girdle_z"] = gird[:, i].T
    fdf = pd.DataFrame(cols)
    sdf = _steps_table(steps, frames, out_pos, fps)
    summ = _summary(fdf, sdf, fps, size, mass, phys)
    return GaitResult(fdf, sdf, summ)


def _steps_table(steps, frames, pos, fps) -> pd.DataFrame:
    rows = []
    for p, a, z in steps:
        if a is None:
            continue
        i = PAWS.index(p)
        rows.append({"paw": p, "on_frame": int(frames[a]), "off_frame": int(frames[z]),
                     "on_s": frames[a] / fps, "off_s": frames[z] / fps,
                     "stance_s": (frames[z] - frames[a]) / fps, "x": pos[a, i, 0], "y": pos[a, i, 1]})
    df = pd.DataFrame(rows, columns=["paw", "on_frame", "off_frame", "on_s", "off_s", "stance_s", "x", "y"])
    df = df.sort_values(["paw", "on_s"]).reset_index(drop=True)
    nxt = df.groupby("paw").shift(-1)
    df["stride_s"] = nxt["on_s"] - df["on_s"]
    df["swing_s"] = nxt["on_s"] - df["off_s"]
    df["stride_mm"] = np.hypot(nxt["x"] - df["x"], nxt["y"] - df["y"])
    df["duty_factor"] = df["stance_s"] / df["stride_s"]
    return df


def _phase(sdf, a, b) -> float:
    """Median phase of paw b's touch-downs in paw a's strides (0–1)."""
    A = sdf[sdf.paw == a].dropna(subset=["stride_s"])
    B = sdf[sdf.paw == b]["on_s"].to_numpy()
    ph = []
    for _, r in A.iterrows():
        k = B[(r.on_s <= B) & (r.on_s + r.stride_s > B)]
        if len(k) and r.stride_s > 0:
            ph.append((k[0] - r.on_s) / r.stride_s)
    return float(np.median(ph)) if ph else float("nan")


def _summary(fdf, sdf, fps, size, mass, phys) -> dict:
    mv = fdf[fdf.valid]
    walking = sdf.dropna(subset=["stride_s"])
    s = {"frames": int(len(fdf)), "fps": fps, "size_vs_model": round(size, 3), "mass_g": round(mass * 1000, 1),
         "model_mass_g": round(phys.mass_kg * 1000, 1), "steps": int(len(sdf)),
         "median_speed_mm_s": round(float(np.nanmedian(mv.speed_mm_s)), 1) if len(mv) else None,
         "support": {int(k): round(float(v), 3) for k, v in (mv.n_support.value_counts(normalize=True)).items()},
         "load_share": {p: round(float(mv[f"{p}_load"].mean()), 3) for p in PAWS} if len(mv) else {},
         "zmp_outside_support": round(float((mv.zmp_margin_mm < 0).mean()), 3) if len(mv) else None,
         "bounds": BOUNDS}
    for p in PAWS:
        w = walking[walking.paw == p]
        if len(w):
            s[f"{p}_stride_mm"] = round(float(w.stride_mm.median()), 1)
            s[f"{p}_stance_s"] = round(float(w.stance_s.median()), 3)
            s[f"{p}_duty_factor"] = round(float(w.duty_factor.median()), 3)
    s["phase_LF_RF"] = round(_phase(sdf, "LF", "RF"), 3)
    s["phase_LH_RH"] = round(_phase(sdf, "LH", "RH"), 3)
    s["phase_LH_LF"] = round(_phase(sdf, "LH", "LF"), 3)
    s["phase_LH_RF"] = round(_phase(sdf, "LH", "RF"), 3)
    return s


def gait_from_pose3d(path, *, fps: float | None = None, **kw) -> GaitResult:
    """:func:`estimate_gait` on a 3-D pose file (``<video>_3d.parquet``)."""
    from mufasa.pose3d.io import marker_names, read_pose3d
    df, meta = read_pose3d(path)
    names = marker_names(df)
    xyz = np.stack([df[[f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(float) for n in names], 1)
    frames = df["frame"].to_numpy(int) if "frame" in df else np.arange(len(df))
    fps = fps or meta.get("fps") or 30.0
    return estimate_gait(names, xyz, frames, float(fps), **kw)


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
    steps = out.with_name(out.name.replace("_gait.parquet", "") + "_steps.csv")
    res.steps.to_csv(steps, index=False)
    return out, steps


def read_gait(path) -> tuple[pd.DataFrame, dict]:
    import json

    import pyarrow.parquet as pq
    t = pq.read_table(path)
    raw = (t.schema.metadata or {}).get(b"mufasa_gait")
    return t.to_pandas(), (json.loads(raw) if raw else {})


__all__ = ["BOUNDS", "BodyPhysics", "GaitResult", "MOUSE_BOUNDS", "PAWS", "animal_size", "body_frame",
           "estimate_gait", "gait_from_pose3d", "read_gait", "share_load", "support_margin", "write_gait"]
