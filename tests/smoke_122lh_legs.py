"""Smoke test for patch 122lh — the virtual rodent's legs bend by its own joints, to the gait's paws
(stage 1 of the biomechanical model).

User: "How do I load the gait in the 3d viewer or will that automatically apply to the virtual
rodent model" (it did not: the legs stayed tucked) → the biomechanical model, stage 1: legs that
bend so the paws stand where they should; "what temporal window is used to optimize the gait
trajectory of each limb?" (none: frames alone) → consecutive frames smoothed.

* each leg bone moves by the model's hinge joints (``C · J(q) · J(q_stance)⁻¹ · C⁻¹``), exactly as
  the model's own kinematics; the tucked stance is the rest;
* per frame and leg, within the joints' ranges: stance paws on the floor (as in the stance) at the
  gait's paw position; swing paws on an arc; without a gait, planted; nothing under the floor;
* consecutive frames are smoothed over time;
* the viewer links a loaded gait file to the rigs; *Bend the legs* switches it off;
* the gait's hips sit the model's hip width apart, whatever the hip labels' spread (K7: 55 mm);
* the virtual rodent's size is the gait's (nose to tail base).
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def walker(T=240, fps=60.0, r=260.0, v=180.0):
    """Labels of the standing model carried along an arc (turning), on the floor."""
    from mufasa.bodymodel.rodent_rig import stance_qpos
    from mufasa.pose3d import gait as G
    from mufasa.pose3d.rodent_model import LABELS, RodentModel
    m = RodentModel()
    q0 = stance_qpos(m)
    Tw = m.fk(q0)
    L = {k: v_[0] for k, v_ in m.labels(q0, Tw).items()}
    floor = min(Tw[0, m.body_index[b], 2, 3] for b in ("toe_L", "toe_R"))
    names = list(LABELS)
    X0 = np.array([L[k] for k in names]) - np.array([0, 0, floor])
    h = G.body_frame(L)[1][:2, 0]
    a = -math.atan2(h[1], h[0])
    X0 = X0 @ np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]).T
    w = v / r
    xyz = np.zeros((T, len(names), 3))
    for i in range(T):
        t = i / fps
        c, s = math.cos(w * t), math.sin(w * t)
        xyz[i] = X0 @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [r * s, r * (1 - c), 0.0]
    return names, xyz, fps


def main() -> int:
    import mufasa.bodymodel.fit as BF
    from mufasa.bodymodel.rodent_rig import R_FRAME, stance_qpos
    from mufasa.pose3d import gait as G
    from mufasa.pose3d import leg_ik as LI
    from mufasa.pose3d import scene_models as SM
    from mufasa.pose3d.rodent_model import RodentModel

    rig = BF.Rig(SM.builtin_rig_path())
    limbs = LI.Limbs.for_rig(rig)
    model = RodentModel()
    check(f"four legs from the model's joints: {[len(lg.names) for lg in limbs.legs.values()]} joints "
          f"(LF {limbs.legs['LF'].names[:3]}…)",
          sorted(limbs.legs) == sorted(G.PAWS) and all(n in model.joint_index for lg in limbs.legs.values()
                                                       for n in lg.names)
          and len(limbs.legs["LH"].names) == 6 and len(limbs.legs["LF"].names) == 8)

    G0 = rig.skinning_matrices(BF.neutral())
    same = np.abs(limbs.apply(G0, {p: lg.qs for p, lg in limbs.legs.items()}) - G0).max()

    # the legs as the model's own kinematics: toe / finger bodies under random angles
    rng = np.random.default_rng(0)
    q0 = stance_qpos(model)
    floor = float(model.skin_points(q0)[0][:, 2].min())
    M = np.eye(4)
    M[:3, :3] = R_FRAME
    M[2, 3] = -floor
    Ts = model.fk(q0)[0]
    worst = 0.0
    for _ in range(5):
        q = q0.copy()
        ang = {}
        for p, lg in limbs.legs.items():
            a = lg.lo + rng.random(len(lg.lo)) * (lg.hi - lg.lo)
            q[0, lg.q_index] = a
            ang[p] = a
        Tq = model.fk(q)[0]
        Ga = limbs.apply(G0, ang)
        for p, lg in limbs.legs.items():
            body = model.body_index[f"{LI.CHAINS[p[1]][-1][1]}_{'L' if p[0] == 'L' else 'R'}"]
            Gb = Ga[lg.bones[-1]]
            want = M @ Tq[body]
            got = Gb @ M @ Ts[body]
            worst = max(worst, float(np.abs(want - got).max()))
    check(f"the legs move as the model's kinematics (paw bodies off by {worst:.1e} mm); the stance is the rest "
          f"({same:.0e})", worst < 1e-8 and same < 1e-9)

    names, xyz, fps = walker()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rows = {"frame": np.arange(len(xyz))}
        for j, n in enumerate(names):
            rows.update({f"{n}_x": xyz[:, j, 0], f"{n}_y": xyz[:, j, 1], f"{n}_z": xyz[:, j, 2]})
        pd.DataFrame(rows).to_parquet(td / "w_3d.parquet", index=False)
        res = G.estimate_gait(names, xyz, np.arange(len(xyz)), fps)
        gp, _ = G.write_gait(res, td / "w_gait.parquet")
        from mufasa.ui_qt.pose3d_viewer import load_pose3d
        pose = load_pose3d(td / "w_3d.parquet")
        m = SM.load_model("virtual_rodent", pose)
        gsize = G.animal_size(G.BodyPhysics.virtual_rodent(), names, xyz)
        check(f"the rig's size is the gait's: {m.size:.3f} (nose to tail base {gsize:.3f})", abs(m.size - gsize) < 1e-9)

        # no gait: paws planted as stood
        frames = list(range(60, 180))
        m.fit_frames(frames)
        m.solve_legs(frames)
        modes0 = {md for f in frames for md, _ in m.leg_info(f).values()}
        off0 = max(e for f in frames for _, e in m.leg_info(f).values())
        check(f"no gait file: every paw planted ({modes0}), soles at their stance heights within {off0:.1f} mm",
              modes0 == {"plant"} and off0 < 2.0)

        # with the gait
        ov = SM.GaitOverlay(gp)
        m.set_gait(ov)
        m.solve_legs(frames)
        info = [m.leg_info(f) for f in frames]
        st = [e for i in info for md, e in i.values() if md == "stance"]
        sw = [e for i in info for md, e in i.values() if md == "swing"]
        check(f"with the gait: {len(st)} paws in stance, off {np.median(st):.1f} mm (median); {len(sw)} in swing, "
              f"off {np.median(sw):.1f} mm", len(st) > 200 and len(sw) > 40 and np.median(st) < 2.0
              and np.median(sw) < 6.0)

        lo_z, inside = np.inf, True
        for f in frames[::5]:
            q = m.legs(f)
            for p, lg in m.limbs.legs.items():
                inside &= bool(np.all(q[p] >= lg.lo - 1e-9) and np.all(q[p] <= lg.hi + 1e-9))
            x = m.fit_frame(f)[0]
            pp, phi, tx, ty, tz = m._x2p(x)
            Gq = m.limbs.apply(m.rig.skinning_matrices(pp), q)
            for lg in m.limbs.legs.values():
                P = LI._points(lg, [Gq[b][None] for b in lg.bones], lg.probes)
                lo_z = min(lo_z, float(LI._world(P, (np.array([phi]), np.array([[tx, ty, tz]]), m.size))[..., 2].min()))
        check(f"within the joints' ranges; the legs' lowest point {lo_z:+.1f} mm", inside and lo_z > -1.5)

        Q = {p: np.array([m._legs[f][0][p] for f in frames]) for p in m.limbs.legs}
        jerk = np.concatenate([np.degrees(np.abs(q[2:] - 2 * q[1:-1] + q[:-2])).max(1) for q in Q.values()])
        raw = {}
        for p, lg in m.limbs.legs.items():
            Gs = np.array([m.rig.skinning_matrices(m._x2p(m.fit_frame(f)[0])[0]) for f in frames])
            pl = [m._x2p(m.fit_frame(f)[0]) for f in frames]
            place = (np.array([v[1] for v in pl]), np.array([v[2:] for v in pl]), m.size)
            tg = [m._targets.at(f) for f in frames]
            md = [t_[p][0] for t_ in tg]
            xyz_t = np.array([np.zeros(3) if t_[p][1] is None else t_[p][1] for t_ in tg])
            q = np.tile(lg.qs, (len(frames), 1))
            for mode in set(md):
                ix = np.array([k for k, mm in enumerate(md) if mm == mode])
                q[ix] = LI.solve_leg(lg, Gs[ix, lg.parent], (place[0][ix], place[1][ix], m.size), mode, xyz_t[ix], q[ix])
            raw[p] = q
        jerk0 = np.concatenate([np.degrees(np.abs(q[2:] - 2 * q[1:-1] + q[:-2])).max(1) for q in raw.values()])
        check(f"smoothed over consecutive frames: worst joint's 2nd difference p90 {np.percentile(jerk, 90):.1f}° "
              f"(frames alone {np.percentile(jerk0, 90):.1f}°)", np.percentile(jerk, 90) < np.percentile(jerk0, 90))

        # the viewer: a gait file among the models drives the rig's legs; the switch
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from mufasa.ui_qt.pose3d_viewer import Volume3DWidget
        vol = Volume3DWidget(pose)
        r2 = SM.load_model("virtual_rodent", pose)
        vol.add_model(r2)
        vol.add_model(SM.GaitOverlay(gp))
        linked = r2.gait is vol.models[1]
        vol.models.pop(1)
        vol.link_gait()
        check("the viewer links a loaded gait file to the rig, and unlinks it when removed",
              linked and r2.gait is None)
        f = frames[30]
        bent = m.world(f)
        m.bend_legs = False
        tucked = m.world(f)
        m.bend_legs = True
        check(f"Bend the legs off: the tucked legs as before (mesh moved {np.abs(bent - tucked).max():.0f} mm); "
              f"info: {m.info(f)[-60:]}", m.legs(f) is not None and np.abs(bent - tucked).max() > 5
              and "legs:" in m.info(f))

    # the gait's hips: the model's width apart, whatever the labels' spread
    phys = G.BodyPhysics.virtual_rodent()
    names2, xyz2, _ = walker(T=4)
    wide = xyz2.copy()
    il, ir = names2.index("hip_left"), names2.index("hip_right")
    mid = (wide[:, il] + wide[:, ir]) / 2
    d = (wide[:, il] - wide[:, ir])
    wide[:, il] = mid + d * 2.4 / 2
    wide[:, ir] = mid - d * 2.4 / 2
    Y, _ = phys.predict_all(names2, wide, 1.0)
    apart = np.linalg.norm(Y[:, 3] - Y[:, 4], axis=1)
    check(f"the gait's hips {apart.mean():.1f} mm apart with hip labels {np.linalg.norm(d, axis=1).mean() * 2.4:.0f} mm "
          f"apart (the model: {2 * phys.hip_half_width:.1f})", np.allclose(apart, 2 * phys.hip_half_width))

    readme = (REPO / "README.md").read_text()
    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("README: scientific references (Eng 2008, Mak 2021, Valdez 2018, Merel 2020); docs: the legs",
          "## Scientific references" in readme and "Eng, C. M." in readme and "10.15760/etd.7690" in readme
          and "10.1186/s12938-018-0565-6" in readme and "virtual rodent" in readme and "Bend the legs" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lh_legs: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
