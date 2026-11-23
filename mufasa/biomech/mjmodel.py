"""
mufasa.biomech.mjmodel
======================

Patch 122li: the virtual rodent (dm_control ``rodent.xml``, as :mod:`mufasa.pose3d.rodent_model`
reads it; Merel et al. 2020) as a MuJoCo model for dynamics:

* a free root joint on its root body (the torso), so the whole body moves and the ground must
  hold it up;
* scaled to the animal: lengths × ``size``, masses × size³, inertias × size⁵ (the same body,
  bigger; ``mass_kg`` rescales the masses to a weighed animal instead);
* a site per 3-D label (:data:`mufasa.pose3d.rodent_model.LABELS`), on the body it sits on.
  Labels that are skin vertices go on the vertex's main body, at its place in the model's zero
  pose;
* two sole sites per paw, the lowest skin points of the foot and the toes (the hand and the
  fingers) in the stance, with their heights in it (the model stands on its digits);
* without dm_control's joint springs (39 joints), damping and armature (an added inertia): they
  steady its controllers but are not the rat's, so the torques found are each joint's net torque,
  muscles and passive tissue together; and without MuJoCo's constraints (joint limits, contacts),
  which inverse dynamics would read as forces (``sim_params=True`` keeps all of these).

SI units inside MuJoCo (m, kg, s); 3-D poses are in mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mufasa.pose3d.gait import PAWS

SOLE_BODIES = {"F": ("hand", "finger"), "H": ("foot", "toe")}

# The rat's own leg ranges (122ln), in the virtual rodent's joint angles (deg). Johnson et al.
# (2008)'s rat hindlimb model gives the knee -150..-20 deg and the ankle -30..60 deg (its flexion
# coordinates); carried onto the virtual rodent through 122lj's bone registration (both models'
# bones posed alike, each end of each range), they become these. The virtual rodent's own ranges
# let the knee fold to 33 deg between thigh and shank (the rat's: 38) and the ankle bend 26 deg
# further than the rat's. The hip's ranges map to a box wider than the virtual rodent's: kept.
RAT_LIMITS_DEG = {"knee_{S}": (-50.0, 79.0), "ankle_{S}": (18.0, 80.0)}
LIMITS = ("model", "rat")


def mujoco():
    try:
        import mujoco as mj
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError('MuJoCo is needed for the biomechanical model: pip install "mufasa[biomech]" '
                          "(or pip install mujoco)") from e
    return mj


@dataclass
class RodentMJ:
    """The MuJoCo virtual rodent, scaled, with label and sole sites."""
    model: object
    size: float
    labels: dict                        # label -> site id
    soles: dict                         # paw -> (heel site id, toe site id)
    sole_z: dict                        # paw -> (2,) stance heights of the sole points (m, scaled)
    hinge: list                         # hinge joint names, in qpos order after the free root
    qadr: dict                          # joint name -> qpos address
    vadr: dict                          # joint name -> dof address
    lo: np.ndarray                      # hinge ranges (rad)
    hi: np.ndarray
    q_stance: np.ndarray                # hinge angles of the tucked stance
    mass_kg: float = 0.0
    paw_joints: dict = field(default_factory=dict)   # paw -> its leg's joint names
    muscles: object = None              # the MuscleSet its tendons were made from (size 1)
    limits: str = "model"               # "model": the virtual rodent's joint ranges; "rat": the rat's legs

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    def data(self):
        return mujoco().MjData(self.model)


_BASE = None


def base_model():
    """``rodent.xml`` as it is (size 1, no added sites); landmarks for :mod:`.osim`."""
    global _BASE
    if _BASE is None:
        from mufasa.pose3d.rodent_model import MODEL_DIR
        _BASE = mujoco().MjModel.from_xml_path(str(MODEL_DIR / "rodent.xml"))
    return _BASE


def build(size: float = 1.0, mass_kg: float | None = None, sim_params: bool = False,
          muscles=None, limits: str = "model") -> RodentMJ:
    """The virtual rodent in MuJoCo, ``size`` × the model (and weighing ``mass_kg``, if given).
    ``muscles`` (:class:`mufasa.biomech.osim.MuscleSet`): a spatial tendon per muscle through
    sites on its path (122lj). ``limits="rat"``: the knees' and ankles' ranges narrowed to the
    rat's (:data:`RAT_LIMITS_DEG`; 122ln)."""
    if limits not in LIMITS:
        raise ValueError(f"limits: one of {LIMITS}, not {limits!r}")
    mj = mujoco()
    from mufasa.bodymodel.rodent_rig import stance_qpos
    from mufasa.pose3d.leg_ik import CHAINS
    from mufasa.pose3d.rodent_model import LABELS, MODEL_DIR, RodentModel
    rm = RodentModel()
    spec = mj.MjSpec.from_file(str(MODEL_DIR / "rodent.xml"))
    root = spec.worldbody.bodies[0]
    root.add_freejoint(name="root")
    bodies = {b.name: b for b in spec.bodies}

    # label sites (model units mm → m), on the zero pose
    q0 = rm.qpos()
    T0 = rm.fk(q0)[0]
    lp = rm.label_points()
    V0 = rm.skin_points(q0, Tw=T0[None])[0]
    for name in LABELS:
        s = lp[name]
        if s[0] == "vertex":
            bi = int(rm.weights[s[1]].argmax())
            local = np.linalg.solve(T0[bi], np.r_[V0[s[1]], 1.0])[:3]
        else:
            bi, local = s[1], np.asarray(s[2], float)
        bodies[rm.body_names[bi]].add_site(name=f"label_{name}", pos=local / 1000.0, size=[0.002, 0, 0])

    # sole sites: each paw's two lowest skin points in the stance
    qs = stance_qpos(rm)
    Ts = rm.fk(qs)[0]
    Vs = rm.skin_points(qs, Tw=Ts[None])[0]
    floor = float(Vs[:, 2].min())
    own = rm.weights.argmax(1)
    sole_z = {}
    for p in PAWS:
        side = "L" if p[0] == "L" else "R"
        zs = []
        for k, body in enumerate(SOLE_BODIES[p[1]]):
            bi = rm.body_index[f"{body}_{side}"]
            ix = np.flatnonzero(own == bi)
            v = ix[np.argmin(Vs[ix, 2])]
            local = np.linalg.solve(Ts[bi], np.r_[Vs[v], 1.0])[:3]
            bodies[rm.body_names[bi]].add_site(name=f"sole_{p}_{k}", pos=local / 1000.0, size=[0.002, 0, 0])
            zs.append((Vs[v, 2] - floor) / 1000.0)
        sole_z[p] = np.array(zs) * size

    if muscles is not None:
        for mu in muscles.muscles:
            t = spec.add_tendon(name=f"muscle_{mu.name}")
            for k, (body, x) in enumerate(mu.path):
                bodies[body].add_site(name=f"muscle_{mu.name}_{k}", pos=list(x), size=[0.0005, 0, 0])
                t.wrap_site(f"muscle_{mu.name}_{k}")

    m = spec.compile()
    if size != 1.0:
        for a in (m.body_pos, m.body_ipos, m.jnt_pos, m.site_pos, m.geom_pos, m.geom_size):
            a *= size
        m.body_mass[:] *= size ** 3
        m.body_inertia[:] *= size ** 5
    if mass_kg:
        k = float(mass_kg) / float(m.body_mass.sum())
        m.body_mass[:] *= k
        m.body_inertia[:] *= k
    if not sim_params:
        m.jnt_stiffness[:] = 0.0
        m.dof_damping[:] = 0.0
        m.dof_armature[:] = 0.0
        # rigid-body inverse dynamics: joint limits are the fit's business, and the ground's
        # forces are solved for (mufasa.biomech.dynamics); MuJoCo's soft constraints would read
        # a joint resting at its limit as a constraint force
        m.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_CONSTRAINT)
    mj.mj_setConst(m, mj.MjData(m))

    if limits == "rat":
        for pat, (lo, hi) in RAT_LIMITS_DEG.items():
            for S in "LR":
                j = m.joint(pat.format(S=S)).id
                r = m.jnt_range[j]
                m.jnt_range[j] = [max(r[0], np.radians(lo)), min(r[1], np.radians(hi))]
    hinge = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
    qadr = {m.joint(i).name: int(m.jnt_qposadr[i]) for i in range(m.njnt)}
    vadr = {m.joint(i).name: int(m.jnt_dofadr[i]) for i in range(m.njnt)}
    rng = np.array([m.jnt_range[m.joint(n).id] for n in hinge])
    q_st = np.clip(np.array([qs[0, rm.joint_index[n]] for n in hinge]), rng[:, 0], rng[:, 1])
    paw_joints = {}
    for p in PAWS:
        side = "L" if p[0] == "L" else "R"
        bis = {rm.body_index[f"{b}_{side}"] for _, b in CHAINS[p[1]]}
        paw_joints[p] = [j.name for j in rm.joints if j.body in bis]
    return RodentMJ(m, float(size), {n: m.site(f"label_{n}").id for n in LABELS},
                    {p: (m.site(f"sole_{p}_0").id, m.site(f"sole_{p}_1").id) for p in PAWS}, sole_z, hinge, qadr,
                    vadr, rng[:, 0].copy(), rng[:, 1].copy(), q_st, float(m.body_mass.sum()), paw_joints,
                    muscles, limits)


__all__ = ["LIMITS", "RAT_LIMITS_DEG", "RodentMJ", "SOLE_BODIES", "base_model", "build", "mujoco"]
