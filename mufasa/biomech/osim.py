"""
mufasa.biomech.osim
===================

Patch 122lj: hindlimb muscles for the virtual rodent, taken from an OpenSim rat hindlimb model
(stage 3 of the biomechanical model).

The muscles' paths (origin, via points, insertion) and parameters come from an OpenSim model of
the rat hindlimb, such as Johnson et al. (2008)'s, as updated by Young et al. (2019) and the UVA
MAMP Lab (``rat_hindlimb_bilateral.osim``; muscle parameters of Johnson et al. 2011). Mufasa
does not ship that model: it is distributed under its own terms (SimTK project
``rat_hlimb_model``). Download it and point :func:`read_osim` (or ``mufasa-pose3d dynamics
--muscles``) at the ``.osim`` file.

The two models' bones differ in size and in how their frames sit, so each bone's attachment
points are carried over by a similarity (rotation, one scale, translation), fitted to landmarks
both models have:

* **pelvis**: the hip centre (anchor); the pelvis's long axis (OpenSim's pelvis x: ischium to
  ilium; the virtual rodent's: its os coxae capsule); outwards (the other hip away);
* **femur**: hip to knee centre; the knee's axis, pointing outwards;
* **tibia**: knee to ankle centre; the ankle's axis, pointing outwards;
* **foot**: ankle to the tip of the longest toe (the foot mesh's farthest point, if the mesh is
  beside the model; else the foot frame's x); the ankle's axis, outwards.

Each bone is scaled by its own length (pelvis and foot: the mean of femur and tibia). Muscle
lengths (optimal fibre, tendon slack) scale with the bones they span; forces with their square.

A model with only the right leg is mirrored to the left.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

BONES = ("pelvis", "femur", "tibia", "foot")
# OpenSim body (per side) -> the virtual rodent's
MJ_BODY = {"pelvis": "pelvis", "femur": "upper_leg_{S}", "tibia": "lower_leg_{S}", "foot": "foot_{S}"}
OS_BODY = {"pelvis": "pelvis_{s}", "femur": "femur_{s}", "tibia": "tibia_{s}", "foot": "foot_{s}"}
OS_JOINT = {"hip": "hip_{s}", "knee": "knee_{s}", "ankle": "ankle_{s}"}
MUSCLE_TAGS = ("Millard2012EquilibriumMuscle", "Thelen2003Muscle", "Millard2012AccelerationMuscle",
               "DeGrooteFregly2016Muscle", "Schutte1993Muscle_Deprecated", "RigidTendonMuscle")


# ---------------------------------------------------------------------------------------------
# muscles


@dataclass
class Muscle:
    """One muscle: its path on the virtual rodent's bodies (m, at size 1) and its parameters."""
    name: str                     # e.g. "VL_L"
    side: str                     # "L" / "R"
    path: list                    # [(mj body name, [x, y, z] m)], origin to insertion
    f_max: float                  # N, peak isometric force (at size 1)
    l_opt: float                  # m, optimal fibre length (at size 1)
    l_slack: float                # m, tendon slack length (at size 1)
    pennation: float = 0.0        # rad, at optimal fibre length
    v_max: float = 10.0           # optimal fibre lengths per second
    l_ref: float = 0.0            # m, its path's length in OpenSim's default pose (size 1)

    @property
    def base(self) -> str:
        return self.name.rsplit("_", 1)[0]


@dataclass
class MuscleSet:
    muscles: list
    source: str = ""
    scale: dict = field(default_factory=dict)     # bone -> scale from the OpenSim model
    knee_fit: dict = field(default_factory=dict)  # muscle -> length-curve rms (mm) before, after
    slack: dict = field(default_factory=dict)     # side -> tendon slack shifts, the matched pose

    def to_json(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps({"source": self.source, "scale": self.scale, "knee_fit": self.knee_fit, "slack": self.slack,
                                    "muscles": [asdict(m) for m in self.muscles]}, indent=1))
        return path

    @classmethod
    def from_json(cls, path) -> MuscleSet:
        d = json.loads(Path(path).read_text())
        return cls([Muscle(**{**m, "path": [(b, list(p)) for b, p in m["path"]]}) for m in d["muscles"]],
                   d.get("source", ""), d.get("scale", {}), d.get("knee_fit", {}), d.get("slack", {}))

    def names(self) -> list:
        return [m.name for m in self.muscles]


def load_muscles(path, architecture: str = "osim") -> MuscleSet:
    """A muscle set from an OpenSim model (``.osim``) or a saved set (``.json``);
    ``architecture="eng2008"``: forces, fibre lengths and pennation from Eng et al. (2008)."""
    path = Path(path)
    ms = MuscleSet.from_json(path) if path.suffix.lower() == ".json" else register(read_osim(path))
    if architecture == "eng2008":
        apply_eng2008(ms)
    elif architecture != "osim":
        raise ValueError(f"architecture: osim or eng2008, not {architecture!r}")
    return ms


# Eng, Smallwood, Rainiero, Lahey, Ward & Lieber (2008), J Exp Biol 211, 2336-2345, Table 1:
# rat hindlimb muscle architecture (N = 6 rats, 323 ± 15 g). Mass mg, fibre length cm
# (normalised to 2.4 µm sarcomeres), pennation deg, PCSA cm².
ENG_2008 = {
    "psoas": (1784.83, 2.36, 17.1, 0.67),
    "gluteus superficialis": (521.00, 1.92, 0.0, 0.26),   # −1.1° reported
    "gluteus medius": (1992.33, 2.43, 0.6, 0.79),
    "muscle X": (377.00, 3.50, 1.0, 0.10),
    "rectus femoris": (945.33, 1.16, 25.4, 0.71),
    "vastus lateralis": (1288.83, 1.96, 10.0, 0.62),
    "vastus intermedius": (170.67, 1.14, 6.9, 0.14),
    "vastus medialis": (442.50, 1.70, 24.7, 0.22),
    "gracilis cranial": (173.00, 2.57, 0.0, 0.06),
    "gracilis caudal": (276.00, 2.64, 0.0, 0.10),
    "adductor longus": (235.50, 1.07, 4.2, 0.21),
    "adductor brevis": (66.83, 1.16, 0.0, 0.06),
    "adductor magnus": (649.83, 2.21, 0.0, 0.28),
    "adductor tertius": (685.17, 1.98, 0.6, 0.33),
    "biceps femoris": (2670.83, 3.40, 3.6, 0.74),
    "semitendinosus": (789.83, 4.77, 0.0, 0.16),
    "semimembranosus": (1452.83, 3.09, 2.1, 0.45),
    "tibialis anterior": (662.17, 1.64, 12.8, 0.38),
    "extensor hallucis longus": (19.50, 1.18, 5.7, 0.02),
    "extensor digitorum longus": (131.83, 1.34, 9.0, 0.09),
    "peroneus longus": (177.50, 0.89, 20.0, 0.19),
    "peroneus brevis": (124.67, 0.82, 12.6, 0.14),
    "gastrocnemius medial": (849.17, 1.61, 14.0, 0.49),
    "gastrocnemius lateral": (1031.17, 1.56, 14.2, 0.62),
    "soleus": (134.67, 1.97, 3.9, 0.07),
    "flexor hallucis longus": (29.67, 0.89, 7.5, 0.03),
    "flexor digitorum longus": (463.83, 0.97, 19.3, 0.44),
    "tibialis posterior": (135.83, 0.60, 26.0, 0.19),
    "plantaris": (397.50, 1.37, 16.4, 0.26),
}
ENG_MASS_G = 323.0
SPECIFIC_TENSION = 22.5      # N/cm² (Powell et al. 1984), PCSA -> peak isometric force
# the model's muscles -> Eng's (a muscle Eng measured whole, the model in parts: shared out as the
# OpenSim model shares it)
ENG_MAP = {
    "BFa": ["biceps femoris"], "BFp": ["biceps femoris"], "STa": ["semitendinosus"],
    "STp": ["semitendinosus"], "SM": ["semimembranosus"], "GMa": ["gluteus superficialis"],
    "GMe": ["gluteus medius"], "GA": ["gracilis cranial"], "GP": ["gracilis caudal"],
    "AL": ["adductor longus"], "AB": ["adductor brevis"], "AM": ["adductor magnus"], "IP": ["psoas"],
    "RF": ["rectus femoris"], "VL": ["vastus lateralis"], "VI": ["vastus intermedius"],
    "VM": ["vastus medialis"], "MG": ["gastrocnemius medial"], "LG": ["gastrocnemius lateral"],
    "Pla": ["plantaris"], "Sol": ["soleus"], "TA": ["tibialis anterior"],
    "EDL": ["extensor digitorum longus"], "TP": ["tibialis posterior"], "FDL": ["flexor digitorum longus"],
    "FHL": ["flexor hallucis longus"], "Per": ["peroneus longus", "peroneus brevis"],
}


def apply_eng2008(ms: MuscleSet) -> dict:
    """Peak forces (PCSA × :data:`SPECIFIC_TENSION`), optimal fibre lengths and pennation from
    Eng et al. (2008), scaled from their 323 g rats to the virtual rodent (size 1): forces by
    mass^(2/3), lengths by mass^(1/3). The tendon slack lengths are re-solved so the fibres sit at
    the same fraction of their optimal length in OpenSim's default pose. Muscles Eng did not
    measure keep the OpenSim model's values. Returns {muscle: (F_max before, after) N}."""
    from mufasa.biomech.mjmodel import base_model
    mass = float(base_model().body_mass.sum()) * 1000.0
    kf, kl = (mass / ENG_MASS_G) ** (2 / 3), (mass / ENG_MASS_G) ** (1 / 3)
    groups = {}
    for mu in ms.muscles:
        if mu.base in ENG_MAP:
            groups.setdefault((mu.side, tuple(ENG_MAP[mu.base])), []).append(mu)
    changed = {}
    for (_, rows), mus in groups.items():
        pcsa = sum(ENG_2008[r][3] for r in rows)
        share = np.array([m.f_max for m in mus])
        share = share / share.sum() if share.sum() > 0 else np.full(len(mus), 1 / len(mus))
        lf = float(np.average([ENG_2008[r][1] for r in rows], weights=[ENG_2008[r][3] for r in rows])) / 100.0
        pen = float(np.radians(np.average([ENG_2008[r][2] for r in rows], weights=[ENG_2008[r][3] for r in rows])))
        for mu, sh in zip(mus, share, strict=True):
            old = mu.f_max
            # the fibres' stretch in the reference pose, kept
            if mu.l_ref > 0:
                x = max(mu.l_ref - mu.l_slack, 1e-6)
                lt = np.hypot(x, mu.l_opt * np.sin(mu.pennation)) / mu.l_opt
            mu.f_max = float(pcsa * SPECIFIC_TENSION * sh * kf)
            mu.l_opt = lf * kl
            mu.pennation = pen
            if mu.l_ref > 0:
                lf_new, h = lt * mu.l_opt, mu.l_opt * np.sin(pen)
                mu.l_slack = float(max(mu.l_ref - np.sqrt(max(lf_new ** 2 - h ** 2, 1e-12)), 0.0))
            changed[mu.name] = (round(old, 2), round(mu.f_max, 2))
    ms.source += " + Eng et al. 2008 architecture"
    return changed


# ---------------------------------------------------------------------------------------------
# OpenSim models


def _rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _xyz(o):
    """OpenSim's orientation: body-fixed X, Y, Z rotations."""
    return _rot([1, 0, 0], o[0]) @ _rot([0, 1, 0], o[1]) @ _rot([0, 0, 1], o[2])


def _vec(e, tag, default=None):
    x = e.find(tag)
    if x is None or x.text is None:
        return default
    return np.array([float(v) for v in x.text.split()])


class _Fn:
    """An OpenSim function of one coordinate."""

    def __init__(self, e):
        self.kind = e.tag if e is not None else "Constant"
        self.e = e
        if e is None:
            self.c = 0.0
        elif e.tag == "Constant":
            self.c = float(e.find("value").text)
        elif e.tag == "LinearFunction":
            self.ab = _vec(e, "coefficients")
        elif e.tag == "PolynomialFunction":
            self.p = _vec(e, "coefficients")
        elif e.tag in ("SimmSpline", "NaturalCubicSpline", "GCVSpline"):
            from scipy.interpolate import CubicSpline
            self.sp = CubicSpline(_vec(e, "x"), _vec(e, "y"), bc_type="natural")
        elif e.tag == "MultiplierFunction":
            inner = e.find("function")
            self.inner = _Fn(inner[0] if inner is not None and len(inner) else None)
            self.k = float(e.find("scale").text)
        else:
            raise ValueError(f"OpenSim function {e.tag} not supported")

    def __call__(self, x):
        k = self.kind
        if k == "Constant":
            return self.c
        if k == "LinearFunction":
            return self.ab[0] * x + self.ab[1]
        if k == "PolynomialFunction":
            return float(np.polyval(self.p, x))
        if k == "MultiplierFunction":
            return self.k * self.inner(x)
        return float(self.sp(x))


@dataclass
class OsimJoint:
    name: str
    parent: str
    child: str
    Tp: np.ndarray            # parent offset frame in the parent body (4x4)
    Tc: np.ndarray            # child offset frame in the child body (4x4)
    axes: list                # [(kind "r"/"t", coordinate or None, axis, function)]

    def mobilizer(self, coords: dict) -> np.ndarray:
        """The joint's transform, child offset frame in the parent offset frame."""
        R = np.eye(3)
        p = np.zeros(3)
        for kind, c, ax, fn in self.axes:
            v = fn(coords.get(c, 0.0) if c else 0.0)
            if kind == "r":
                R = R @ _rot(ax, v)
            else:
                p = p + v * np.asarray(ax, float)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T


@dataclass
class OsimModel:
    path: Path
    bodies: list
    joints: dict
    coords: dict              # name -> default value
    ranges: dict              # name -> (lo, hi)
    muscles: list             # [{name, f_max, l_opt, l_slack, pennation, v_max, path: [(body, xyz m)]}]
    meshes: dict              # body -> vertices (m) if the mesh files were found

    def fk(self, coords: dict | None = None) -> dict:
        """Body frames in ground (4x4), at ``coords`` (default: the model's defaults)."""
        c = dict(self.coords)
        c.update(coords or {})
        X = {"ground": np.eye(4)}
        pending = list(self.joints.values())
        while pending:
            rest = []
            for j in pending:
                if j.parent not in X:
                    rest.append(j)
                    continue
                X[j.child] = X[j.parent] @ j.Tp @ j.mobilizer(c) @ np.linalg.inv(j.Tc)
            if len(rest) == len(pending):
                break
            pending = rest
        return X

    def path_length(self, muscle: dict, coords: dict | None = None) -> float:
        X = self.fk(coords)
        P = [(X[b] @ np.r_[x, 1.0])[:3] for b, x in muscle["path"]]
        return float(sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def _offset(e):
    T = np.eye(4)
    T[:3, :3] = _xyz(_vec(e, "orientation", np.zeros(3)))
    T[:3, 3] = _vec(e, "translation", np.zeros(3))
    return T


def _stl(path: Path) -> np.ndarray:
    import struct
    b = path.read_bytes()
    if b[:5] == b"solid" and b"facet" in b[:400]:
        return np.array([[float(v) for v in ln.split()[1:4]] for ln in b.decode(errors="ignore").splitlines()
                         if ln.strip().startswith("vertex")])
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float)


def read_osim(path) -> OsimModel:
    """Bodies, joints, coordinates and muscles (with their path points) of an OpenSim model."""
    path = Path(path)
    root = ET.parse(path).getroot()
    model = root.find("Model")
    unit = model.find("length_units") if model is not None else None
    k = 0.001 if unit is not None and unit.text and unit.text.strip().lower() in ("mm", "millimeters") else 1.0

    coords, ranges = {}, {}
    for c in root.iter("Coordinate"):
        dv = c.find("default_value")
        coords[c.get("name")] = float(dv.text) if dv is not None and dv.text else 0.0
        r = _vec(c, "range")
        if r is not None and len(r) == 2:
            ranges[c.get("name")] = (float(r[0]), float(r[1]))

    joints = {}
    for tag in ("CustomJoint", "PinJoint", "WeldJoint", "BallJoint", "UniversalJoint"):
        for j in root.iter(tag):
            frames = {f.get("name"): f for f in (j.find("frames") or [])}
            pf, cf = frames.get(j.findtext("socket_parent_frame").strip()), frames.get(j.findtext("socket_child_frame").strip())
            parent = pf.findtext("socket_parent").strip().split("/")[-1]
            child = cf.findtext("socket_parent").strip().split("/")[-1]
            Tp, Tc = _offset(pf), _offset(cf)
            Tp[:3, 3] *= k
            Tc[:3, 3] *= k
            names = [c.get("name") for c in j.iter("Coordinate")]
            axes = []
            st = j.find("SpatialTransform")
            if st is not None:
                for ax in st:
                    kind = "r" if ax.get("name", "").startswith("rotation") else "t"
                    cn = (ax.findtext("coordinates") or "").strip() or None
                    fe = [e for e in ax if e.tag not in ("coordinates", "axis")]
                    fn = _Fn(fe[0] if fe else None)
                    if kind == "t" and k != 1.0:
                        fn = _Scaled(fn, k)
                    axes.append((kind, cn, _vec(ax, "axis"), fn))
            elif tag == "PinJoint":
                axes.append(("r", names[0], np.array([0.0, 0, 1]), _Fn(ET.fromstring(
                    "<LinearFunction><coefficients>1 0</coefficients></LinearFunction>"))))
            joints[j.get("name")] = OsimJoint(j.get("name"), parent, child, Tp, Tc, axes)

    muscles = []
    for tag in MUSCLE_TAGS:
        for m in root.iter(tag):
            pts = []
            for p in m.iter("PathPoint"):
                body = (p.findtext("socket_parent_frame") or p.findtext("body") or "").strip().split("/")[-1]
                pts.append((body, _vec(p, "location") * k))
            if len(pts) < 2:
                continue
            g = (lambda t, d=0.0: float(m.findtext(t)) if m.findtext(t) else d)
            muscles.append({"name": m.get("name"), "f_max": g("max_isometric_force"),
                            "l_opt": g("optimal_fiber_length") * k, "l_slack": g("tendon_slack_length") * k,
                            "pennation": g("pennation_angle_at_optimal"),
                            "v_max": g("max_contraction_velocity", 10.0), "path": pts})

    bodies = [b.get("name") for b in root.iter("Body")]
    meshes = {}
    for b in root.iter("Body"):
        for mesh in b.iter("Mesh"):
            fn = (mesh.findtext("mesh_file") or "").strip()
            for d in (path.parent / "Geometry", path.parent, path.parent.parent / "meshes"):
                f = d / fn
                if fn.lower().endswith(".stl") and f.exists():
                    sf = _vec(mesh, "scale_factors", np.ones(3))
                    meshes[b.get("name")] = _stl(f) * sf * k
                    break
    return OsimModel(path, bodies, joints, coords, ranges, muscles, meshes)


class _Scaled:
    def __init__(self, fn, k):
        self.fn, self.k = fn, k

    def __call__(self, x):
        return self.k * self.fn(x)


# ---------------------------------------------------------------------------------------------
# landmarks and registration


def _frame(e1, e2):
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 - e1 * (e2 @ e1)
    e2 = e2 / np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def _flex_axis(j: OsimJoint) -> np.ndarray:
    """The joint's first rotation axis (its flexion, in these models), in the parent body frame."""
    for kind, _, ax, _ in j.axes:
        if kind == "r":
            return j.Tp[:3, :3] @ (ax / np.linalg.norm(ax))
    return j.Tp[:3, :3] @ np.array([0.0, 0, 1])


def _centre_in_parent(j: OsimJoint, coords) -> np.ndarray:
    return (j.Tp @ j.mobilizer(coords))[:3, 3]


def _axis_in_child(j: OsimJoint, coords) -> np.ndarray:
    """The first rotation axis, carried into the child body frame."""
    T = j.Tp @ j.mobilizer(coords) @ np.linalg.inv(j.Tc)        # child body in parent body
    return T[:3, :3].T @ _flex_axis(j)


def osim_landmarks(om: OsimModel, s: str) -> dict:
    """Per bone (OpenSim body frames): anchor point, long axis end, second axis, length;
    plus the side's outward direction in the pelvis frame."""
    J = {k: om.joints[v.format(s=s)] for k, v in OS_JOINT.items()}
    c = om.coords
    hip_p = _centre_in_parent(J["hip"], c)                       # in the pelvis
    hip_f = J["hip"].Tc[:3, 3]                                   # in the femur
    knee_f = _centre_in_parent(J["knee"], c)
    knee_t = J["knee"].Tc[:3, 3]
    ankle_t = _centre_in_parent(J["ankle"], c)
    ankle_f = J["ankle"].Tc[:3, 3]
    # outwards: from the other hip (or the pelvis's mirror plane) to this one, in ground
    X = om.fk()
    other = "l" if s == "r" else "r"
    pel = OS_BODY["pelvis"].format(s=s)
    hip_g = (X[pel] @ np.r_[hip_p, 1])[:3]
    jo = om.joints.get(OS_JOINT["hip"].format(s=other))
    if jo is not None:
        oth = (X[jo.parent] @ np.r_[_centre_in_parent(jo, c), 1])[:3]
        out_g = hip_g - oth
    else:
        out_g = X[pel][:3, :3] @ np.array([0, 0, 1.0 if s == "r" else -1.0])
    out_g /= np.linalg.norm(out_g)

    def outward(axis_body, body):
        a = X[body][:3, :3] @ axis_body
        return axis_body if a @ out_g >= 0 else -axis_body

    fem, tib, foot = (OS_BODY[b].format(s=s) for b in ("femur", "tibia", "foot"))
    knee_ax = outward(_flex_axis(J["knee"]), fem)
    ankle_ax = outward(_axis_in_child(J["ankle"], c), foot)
    ankle_ax_t = outward(_flex_axis(J["ankle"]), tib)
    pel_long = X[pel][:3, :3].T @ (X[pel][:3, :3] @ np.array([1.0, 0, 0]))
    out_p = X[pel][:3, :3].T @ out_g
    if foot in om.meshes:
        V = om.meshes[foot]
        tip = V[np.argmax(np.linalg.norm(V - ankle_f, axis=1))]
    else:
        tip = ankle_f + np.array([0.025, 0, 0])
    return {
        "pelvis": (hip_p, hip_p + pel_long, out_p),
        "femur": (hip_f, knee_f, knee_ax),
        "tibia": (knee_t, ankle_t, ankle_ax_t),
        "foot": (ankle_f, tip, ankle_ax),
    }


def mj_landmarks(side: str) -> dict:
    """The same landmarks on the virtual rodent (m, size 1; its bodies' frames)."""
    from mufasa.biomech.mjmodel import base_model
    m = base_model()
    S = side
    out = np.array([0.0, 1.0 if S == "L" else -1.0, 0.0])        # every body frame is the pelvis's at zero
    bp = lambda n: m.body_pos[m.body(n).id].copy()               # noqa: E731
    ax = lambda n: m.jnt_axis[m.joint(n).id].copy()              # noqa: E731

    def outward(a):
        return a if a @ out >= 0 else -a

    g = m.geom(f"collision_pelvis_{S}")
    from scipy.spatial.transform import Rotation
    q = g.quat
    long = Rotation.from_quat([q[1], q[2], q[3], q[0]]).apply([0, 0, 1.0])
    if long[0] < 0:
        long = -long                                             # cranially
    hip = bp(f"upper_leg_{S}")
    from mufasa.pose3d.rodent_model import RodentModel
    rm = RodentModel()
    T0 = rm.fk(rm.qpos())[0]
    V0 = rm.skin_points(rm.qpos(), Tw=T0[None])[0] / 1000.0
    toe = rm.body_index[f"toe_{S}"]
    foot_i = rm.body_index[f"foot_{S}"]
    own = rm.weights.argmax(1) == toe
    ankle_w = T0[foot_i, :3, 3] / 1000.0
    tip_w = V0[own][np.argmax(np.linalg.norm(V0[own] - ankle_w, axis=1))]
    tip = T0[foot_i, :3, :3].T @ (tip_w - ankle_w)
    return {
        "pelvis": (hip, hip + long, out),
        "femur": (np.zeros(3), bp(f"lower_leg_{S}"), outward(ax(f"knee_{S}"))),
        "tibia": (np.zeros(3), bp(f"foot_{S}"), outward(ax(f"ankle_{S}"))),
        "foot": (np.zeros(3), tip, outward(ax(f"ankle_{S}"))),
    }


def bone_maps(om: OsimModel, s: str) -> dict:
    """bone -> (R, scale, anchor_osim, anchor_mj): x_mj = anchor_mj + scale · R (x − anchor_osim)."""
    lo, lm = osim_landmarks(om, s), mj_landmarks(s.upper())
    out = {}
    for b in BONES:
        (a_o, e_o, x_o), (a_m, e_m, x_m) = lo[b], lm[b]
        R = _frame(e_m - a_m, x_m) @ _frame(e_o - a_o, x_o).T
        out[b] = [R, np.linalg.norm(e_m - a_m) / np.linalg.norm(e_o - a_o), a_o, a_m]
    body = (out["femur"][1] + out["tibia"][1]) / 2
    for b in ("pelvis", "foot"):
        out[b][1] = body
    return out


def _inner(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


KNEE_SAMPLES = 31
NEAR_KNEE_MM = 12.0


def _knee_curves(om: OsimModel, s: str, muscles: list) -> tuple:
    """OpenSim: per sample of the knee's range, its inner angle (hip-knee-ankle) and each
    muscle's path length."""
    jk = om.joints[OS_JOINT["knee"].format(s=s)]
    cn = next(c for kind, c, _, _ in jk.axes if kind == "r" and c)
    lo, hi = om.ranges.get(cn, (-2.6, 0.0))
    fem, tib = OS_BODY["femur"].format(s=s), OS_BODY["tibia"].format(s=s)
    ank = _centre_in_parent(om.joints[OS_JOINT["ankle"].format(s=s)], om.coords)
    ang, L = [], []
    for th in np.linspace(lo, hi, KNEE_SAMPLES):
        X = om.fk({cn: th})
        hip = X[fem][:3, 3]
        knee = (X[tib] @ np.r_[jk.Tc[:3, 3], 1])[:3]
        ankle = (X[tib] @ np.r_[ank, 1])[:3]
        ang.append(_inner(hip, knee, ankle))
        L.append([sum(np.linalg.norm((X[b2] @ np.r_[x2, 1])[:3] - (X[b1] @ np.r_[x1, 1])[:3])
                      for (b1, x1), (b2, x2) in zip(m["path"][:-1], m["path"][1:], strict=True)) for m in muscles])
    o = np.argsort(ang)
    return np.array(ang)[o], np.array(L)[o]


def _mj_stance():
    import mujoco as mj

    from mufasa.biomech.mjmodel import base_model
    from mufasa.bodymodel.rodent_rig import stance_qpos
    from mufasa.pose3d.rodent_model import RodentModel
    m = base_model()
    rm = RodentModel()
    qs = stance_qpos(rm)[0]
    q0 = np.zeros(m.nq)
    for i in range(m.njnt):
        n = m.joint(i).name
        if n in rm.joint_index:
            q0[m.jnt_qposadr[i]] = qs[rm.joint_index[n]]
    return m, mj.MjData(m), q0


def _mj_frames(S: str, leg: dict) -> dict:
    """The leg bodies' world frames (R, p) in the stance with the joints in ``leg`` set (size 1)."""
    import mujoco as mj
    m, d, q = _mj_stance()
    for n, v in leg.items():
        q[m.joint(n).qposadr[0]] = v
    d.qpos[:] = q
    mj.mj_kinematics(m, d)
    return {b: (d.xmat[m.body(b).id].reshape(3, 3).copy(), d.xpos[m.body(b).id].copy())
            for b in ("pelvis", f"upper_leg_{S}", f"lower_leg_{S}", f"foot_{S}", f"toe_{S}")}


def _length(path, frames) -> float:
    W = [frames[b][0] @ np.asarray(x) + frames[b][1] for b, x in path]
    return float(sum(np.linalg.norm(np.diff(W, axis=0), axis=1)))


def _match_pose(om: OsimModel, s: str, maps: dict) -> tuple:
    """The virtual rodent's leg angles that put its bones as OpenSim's default pose puts them
    (relative to each other), and how far off that is (deg, per joint)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    m, _, _ = _mj_stance()
    S = s.upper()
    X = om.fk()
    W = {b: X[OS_BODY[b].format(s=s)][:3, :3] @ maps[b][0].T for b in BONES}
    joints = {"hip": [f"hip_{S}_supinate", f"hip_{S}_abduct", f"hip_{S}_extend"], "knee": [f"knee_{S}"],
              "ankle": [f"ankle_{S}"]}
    pairs = {"hip": ("pelvis", "femur"), "knee": ("femur", "tibia"), "ankle": ("tibia", "foot")}
    leg, err = {}, {}
    for jn, names in joints.items():
        a, b = pairs[jn]
        target = W[a].T @ W[b]
        axes = [m.jnt_axis[m.joint(n).id] for n in names]
        lo = [m.jnt_range[m.joint(n).id][0] for n in names]
        hi = [m.jnt_range[m.joint(n).id][1] for n in names]

        def rot(x, axes=axes):
            R = np.eye(3)
            for ax, v in zip(axes, x, strict=True):
                R = R @ _rot(ax, v)
            return R

        r = least_squares(lambda x, target=target, rot=rot: (rot(x) - target).ravel(),
                          np.clip(np.zeros(len(names)), lo, hi), bounds=(lo, hi))
        leg.update(dict(zip(names, r.x, strict=True)))
        err[jn] = round(float(np.degrees(Rotation.from_matrix(rot(r.x).T @ target).magnitude())), 1)
    return leg, err


def _match_slack(om: OsimModel, s: str, maps: dict, os_muscles: list, mj_muscles: list) -> dict:
    """Each muscle's tendon slack length, shifted by its path's length on the virtual rodent less
    OpenSim's (scaled), with both models' bones placed alike (OpenSim's default pose): the fibres
    keep OpenSim's lengths there. Returns the shifts (mm) and the pose's mismatch (deg)."""
    leg, err = _match_pose(om, s, maps)
    fr = _mj_frames(s.upper(), leg)
    shifts = {}
    for om_m, mu in zip(os_muscles, mj_muscles, strict=True):
        k = mu.l_opt / om_m["l_opt"] if om_m["l_opt"] else 1.0
        dl = _length(mu.path, fr) - k * om.path_length(om_m)
        mu.l_slack = max(mu.l_slack + dl, 0.0)
        mu.l_ref = _length(mu.path, fr)
        shifts[mu.name] = round(dl * 1000, 2)
    return {"slack_shift_mm": shifts, "pose_mismatch_deg": err,
            "pose_deg": {n: round(float(np.degrees(v)), 1) for n, v in leg.items()}}


def _mj_knee_frames(S: str) -> tuple:
    """The virtual rodent in its stance, the knee swept over its range: inner angles and the
    leg bodies' world frames (size 1)."""
    import mujoco as mj

    from mufasa.biomech.mjmodel import base_model
    from mufasa.bodymodel.rodent_rig import stance_qpos
    from mufasa.pose3d.rodent_model import RodentModel
    m = base_model()
    d = mj.MjData(m)
    rm = RodentModel()
    qs = stance_qpos(rm)[0]
    q0 = np.zeros(m.nq)
    for i in range(m.njnt):
        n = m.joint(i).name
        if n in rm.joint_index:
            q0[m.jnt_qposadr[i]] = qs[rm.joint_index[n]]
    kn = m.joint(f"knee_{S}")
    bodies = ["pelvis", f"upper_leg_{S}", f"lower_leg_{S}", f"foot_{S}"]
    ang, frames = [], []
    for v in np.linspace(*m.jnt_range[kn.id], KNEE_SAMPLES):
        q = q0.copy()
        q[kn.qposadr[0]] = v
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        T = {}
        for b in bodies:
            i = m.body(b).id
            T[b] = (d.xmat[i].reshape(3, 3).copy(), d.xpos[i].copy())
        ang.append(_inner(T[bodies[1]][1], T[bodies[2]][1], T[bodies[3]][1]))
        frames.append(T)
    return np.array(ang), frames


def _refine_knee(om: OsimModel, s: str, muscles: list, mj_muscles: list) -> dict:
    """OpenSim's knee slides as it bends (its translations follow the angle); the virtual
    rodent's is a hinge. For each muscle crossing the knee, its points near the knee (on the
    femur and the tibia, within :data:`NEAR_KNEE_MM`) are moved so that its length follows
    OpenSim's over the angles both knees reach (an inner angle matched to an inner angle),
    up to a constant. Returns {muscle: (rms before, after) mm}."""
    from scipy.optimize import least_squares
    S = s.upper()
    cross = [k for k, m in enumerate(muscles)
             if {b for b, _ in m["path"]} & {OS_BODY["pelvis"].format(s=s), OS_BODY["femur"].format(s=s)}
             and {b for b, _ in m["path"]} & {OS_BODY["tibia"].format(s=s), OS_BODY["foot"].format(s=s)}]
    if not cross:
        return {}
    a_os, L_os = _knee_curves(om, s, [muscles[k] for k in cross])
    a_mj, frames = _mj_knee_frames(S)
    use = (a_mj >= a_os.min()) & (a_mj <= a_os.max())
    if use.sum() < 5:
        return {}
    frames = [f for f, u in zip(frames, use, strict=True) if u]
    knee_b = f"lower_leg_{S}"
    out = {}
    for j, k in enumerate(cross):
        mu = mj_muscles[k]
        target = np.interp(a_mj[use], a_os, L_os[:, j]) * (mu.l_opt / muscles[k]["l_opt"] if muscles[k]["l_opt"] else 1.0)
        kn_w = [f[knee_b][1] for f in frames]
        P = [np.array(x, float) for _, x in mu.path]
        free = [i for i, (b, x) in enumerate(mu.path) if b in (f"upper_leg_{S}", knee_b)]
        near = []
        for i in free:
            f0 = frames[0]
            w = f0[mu.path[i][0]][0] @ P[i] + f0[mu.path[i][0]][1]
            if np.linalg.norm(w - kn_w[0]) * 1000 <= NEAR_KNEE_MM:
                near.append(i)
        if not near:
            near = [i for i in free if mu.path[i][0] == knee_b][:1]
        if not near:
            continue

        def lengths(x):
            Q = list(P)
            for n, i in enumerate(near):
                Q[i] = P[i] + x[3 * n: 3 * n + 3]
            out_ = []
            for f in frames:
                W = [f[b][0] @ q + f[b][1] for (b, _), q in zip(mu.path, Q, strict=True)]
                out_.append(sum(np.linalg.norm(np.diff(W, axis=0), axis=1)))
            return np.array(out_)

        def res(x):
            e = lengths(x[:-1]) - target - x[-1]
            return np.r_[e * 1000, 0.1 * x[:-1] * 1000]

        x0 = np.zeros(3 * len(near) + 1)
        e0 = lengths(x0[:-1]) - target
        x0[-1] = float(np.mean(e0))
        before = float(np.sqrt(np.mean((e0 - e0.mean()) ** 2)) * 1000)
        r = least_squares(res, x0)
        e1 = lengths(r.x[:-1]) - target
        after = float(np.sqrt(np.mean((e1 - e1.mean()) ** 2)) * 1000)
        if after < before:
            for n, i in enumerate(near):
                mu.path[i] = (mu.path[i][0], (P[i] + r.x[3 * n: 3 * n + 3]).tolist())
        out[mu.name] = (round(before, 2), round(min(after, before), 2))
    return out


def register(om: OsimModel, refine_knee: bool = True) -> MuscleSet:
    """The OpenSim model's muscles on the virtual rodent's bodies (m, at size 1)."""
    by_body = {}
    sides = [s for s in ("r", "l") if OS_BODY["femur"].format(s=s) in om.bodies]
    if not sides:
        raise ValueError(f"{om.path.name}: no femur_r / femur_l; not a rat hindlimb model this reads")
    maps = {s: bone_maps(om, s) for s in sides}
    for s in sides:
        for b in BONES:
            by_body[OS_BODY[b].format(s=s)] = (s, b)
    muscles = []
    for m in om.muscles:
        try:
            pts = [(by_body[b], x) for b, x in m["path"]]
        except KeyError as e:
            raise ValueError(f"{m['name']}: a path point on {e.args[0]}, not a leg bone") from None
        s = pts[0][0][0]
        S = s.upper()
        path, scales = [], []
        for (_, bone), x in pts:
            R, k, a_o, a_m = maps[s][bone]
            path.append((MJ_BODY[bone].format(S=S), (a_m + k * R @ (x - a_o)).tolist()))
            scales.append(k)
        k = float(np.mean(scales))
        base = m["name"]
        for pre in ("R_", "L_", "r_", "l_"):
            if base.startswith(pre):
                base = base[len(pre):]
        for suf in ("_r", "_l", "_R", "_L"):
            if base.endswith(suf):
                base = base[: -len(suf)]
        muscles.append(Muscle(f"{base}_{S}", S, path, m["f_max"] * k * k, m["l_opt"] * k, m["l_slack"] * k,
                              m["pennation"], m["v_max"]))
    knee, slack = {}, {}
    for s in sides:
        own = [k for k, m in enumerate(om.muscles) if muscles[k].side == s.upper()]
        if refine_knee:
            knee.update(_refine_knee(om, s, [om.muscles[k] for k in own], [muscles[k] for k in own]))
        slack[s.upper()] = _match_slack(om, s, maps[s], [om.muscles[k] for k in own], [muscles[k] for k in own])
    if sides == ["r"]:                                           # one leg: mirror it
        for mu in list(muscles):
            muscles.append(Muscle(mu.name[:-1] + "L", "L",
                                  [(b.replace("_R", "_L"), [x, -y, z]) for b, (x, y, z) in mu.path],
                                  mu.f_max, mu.l_opt, mu.l_slack, mu.pennation, mu.v_max, mu.l_ref))
    scale = {f"{b}_{s.upper()}": round(float(maps[s][b][1]), 4) for s in sides for b in BONES}
    return MuscleSet(muscles, str(om.path), scale, knee, slack)


__all__ = ["BONES", "ENG_2008", "ENG_MAP", "Muscle", "apply_eng2008", "MuscleSet", "OsimModel", "bone_maps", "load_muscles", "mj_landmarks",
           "osim_landmarks", "read_osim", "register"]
