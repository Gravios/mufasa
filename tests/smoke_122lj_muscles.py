"""Smoke test for patch 122lj — the biomechanical model, stage 3: the hindlimbs' muscles, carried
from an OpenSim rat hindlimb model onto the virtual rodent, and their forces from the joint
torques (static optimisation).

User: "If the muscle loads can be optimized, we should be able to generate more naturalistic
movements and gaits"; hindlimbs first; Eng et al. 2008 (rat hindlimb architecture); "Continue".

The OpenSim model (Johnson et al. 2008; UVA MAMP Lab's bilateral version) is not shipped (its
own terms), so this test writes a small one of the same shape: one leg, a sliding knee, five
muscles.

* reading OpenSim: joints, coordinates, path points; its kinematics;
* each bone carried over by its landmarks (joint centres, axes); one leg mirrored to the other;
* moment arms on the virtual rodent: extensors extend, flexors flex;
* the knee: OpenSim's slides, the virtual rodent's is a hinge; the muscles crossing it follow
  OpenSim's length curve after refitting;
* Hill-type force curves; static optimisation reproduces torques the muscles can give, and a
  reserve takes what they cannot;
* Eng et al. 2008's architecture as an option;
* ``mufasa-pose3d dynamics --muscles``.
"""
from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


LIN = "<LinearFunction name=\"function\"><coefficients> 1 0</coefficients></LinearFunction>"


def const(v):
    return f"<Constant name=\"function\"><value>{v}</value></Constant>"


def spline(xs, ys):
    return (f"<SimmSpline name=\"function\"><x> {' '.join(map(str, xs))}</x>"
            f"<y> {' '.join(map(str, ys))}</y></SimmSpline>")


def axis(name, coord, ax, fn):
    return (f"<TransformAxis name=\"{name}\"><coordinates>{coord}</coordinates><axis>{ax}</axis>{fn}"
            "</TransformAxis>")


def joint(name, parent, child, pT, cT, coords, axes):
    cs = "".join(f"<Coordinate name=\"{c}\"><default_value>{d}</default_value><range>{lo} {hi}</range></Coordinate>"
                 for c, d, lo, hi in coords)
    return (f"<CustomJoint name=\"{name}\"><socket_parent_frame>{parent}_offset</socket_parent_frame>"
            f"<socket_child_frame>{child}_offset</socket_child_frame><coordinates>{cs}</coordinates><frames>"
            f"<PhysicalOffsetFrame name=\"{parent}_offset\"><socket_parent>/bodyset/{parent}</socket_parent>"
            f"<translation>{pT}</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>"
            f"<PhysicalOffsetFrame name=\"{child}_offset\"><socket_parent>/bodyset/{child}</socket_parent>"
            f"<translation>{cT}</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame>"
            f"</frames><SpatialTransform>{''.join(axes)}</SpatialTransform></CustomJoint>")


MUSCLES = {   # name: (F N, l_opt, l_slack, pennation, path)
    "R_VL": (14.0, 0.019, 0.024, 0.2, [("femur_r", "0.002 -0.010 0.003"), ("femur_r", "0.0035 -0.031 0.0"),
                                       ("tibia_r", "0.0035 0.0385 0.0")]),
    "R_SM": (10.0, 0.035, 0.006, 0.0, [("pelvis_r", "-0.014 0.0 0.0"), ("tibia_r", "-0.003 0.032 0.0")]),
    "R_MG": (11.0, 0.015, 0.033, 0.2, [("femur_r", "-0.002 -0.031 0.0"), ("tibia_r", "-0.003 0.004 0.0"),
                                       ("foot_r", "-0.005 -0.001 0.0")]),
    "R_TA": (8.0, 0.015, 0.025, 0.15, [("tibia_r", "0.004 0.030 0.0"), ("tibia_r", "0.0025 0.004 0.0"),
                                       ("foot_r", "0.006 -0.003 0.0")]),
    "R_IP": (11.0, 0.024, 0.002, 0.25, [("pelvis_r", "0.020 0.0 0.0"), ("femur_r", "0.001 -0.006 0.0")]),
}


def write_osim(path: Path) -> Path:
    kx = [-2.6, -2.0, -1.5, -1.0, -0.5, 0.0]
    bodies = "".join(f"<Body name=\"{b}\"><mass>0.02</mass></Body>"
                     for b in ("pelvis_r", "femur_r", "tibia_r", "foot_r"))
    joints = [
        joint("ground_pelvis", "ground", "pelvis_r", "0 0 0", "0 0 0", [("pelvis_ty", 0.05, 0, 0.2)],
              [axis("rotation1", "", "0 0 1", const(0)), axis("translation2", "pelvis_ty", "0 1 0", LIN)]),
        joint("hip_r", "pelvis_r", "femur_r", "0 0 0", "0 0 0",
              [("hip_r_flx", 0.7, -1.5, 1.5), ("hip_r_add", 0.0, -1, 1), ("hip_r_int", 0.0, -1, 1)],
              [axis("rotation1", "hip_r_flx", "0 0 1", LIN), axis("rotation2", "hip_r_add", "1 0 0", LIN),
               axis("rotation3", "hip_r_int", "0 1 0", LIN)]),
        joint("knee_r", "femur_r", "tibia_r", "0 0 0", "0 0.040 0", [("knee_r_flx", -2.09, -2.6, 0.0)],
              [axis("rotation1", "knee_r_flx", "0 0 1", LIN),
               axis("translation1", "knee_r_flx", "1 0 0", spline(kx, [-0.005, -0.004, -0.003, -0.002, -0.0015, -0.001])),
               axis("translation2", "knee_r_flx", "0 1 0", spline(kx, [-0.034] * 6))]),
        joint("ankle_r", "tibia_r", "foot_r", "0 0 0", "0 0 0", [("ankle_r_flx", 0.87, -0.5, 1.2)],
              [axis("rotation1", "ankle_r_flx", "0 0 1", LIN)]),
    ]
    joints[0] = joints[0].replace("<socket_parent>/bodyset/ground</socket_parent>", "<socket_parent>/ground</socket_parent>")
    mus = ""
    for n, (f, lo, ls, pen, pts) in MUSCLES.items():
        pp = "".join(f"<PathPoint name=\"{n}-P{k}\"><socket_parent_frame>/bodyset/{b}</socket_parent_frame>"
                     f"<location>{x}</location></PathPoint>" for k, (b, x) in enumerate(pts))
        mus += (f"<Millard2012EquilibriumMuscle name=\"{n}\"><GeometryPath name=\"path\"><PathPointSet><objects>{pp}"
                f"</objects></PathPointSet></GeometryPath><max_isometric_force>{f}</max_isometric_force>"
                f"<optimal_fiber_length>{lo}</optimal_fiber_length><tendon_slack_length>{ls}</tendon_slack_length>"
                f"<pennation_angle_at_optimal>{pen}</pennation_angle_at_optimal></Millard2012EquilibriumMuscle>")
    xml = (f"<?xml version=\"1.0\"?><OpenSimDocument Version=\"40000\"><Model name=\"rat\"><length_units>m"
           f"</length_units><BodySet><objects>{bodies}</objects></BodySet><JointSet><objects>{''.join(joints)}"
           f"</objects></JointSet><ForceSet><objects>{mus}</objects></ForceSet></Model></OpenSimDocument>")
    path.write_text(xml)
    return path


def moment_arms(M, q, names, dofs):
    from mufasa.biomech.muscle import tendon_state
    _, _, J = tendon_state(M, q, np.zeros(M.nv))
    tid = [M.model.tendon(f"muscle_{n}").id for n in names]
    return {n: {d: -J[i, M.vadr[d]] * 1000 for d in dofs} for n, i in zip(names, tid, strict=True)}


def main() -> int:
    try:
        import mujoco
    except ImportError:
        check("MuJoCo not installed: skipped (pip install \"mufasa[biomech]\")", True)
        return report()
    from mufasa.biomech import muscle as MU
    from mufasa.biomech import osim as O
    from mufasa.biomech.mjmodel import build

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        om = O.read_osim(write_osim(td / "rat.osim"))
        X = om.fk()
        knee_w = (X["tibia_r"] @ np.r_[0, 0.04, 0, 1])[:3]
        knee_o = O._centre_in_parent(om.joints["knee_r"], om.coords)      # the knee's centre, femur frame
        check(f"OpenSim read: {len(om.joints)} joints, {len(om.coords)} coordinates, {len(om.muscles)} muscles; "
              f"the knee {np.linalg.norm(knee_w - X['femur_r'][:3, 3]) * 1000:.1f} mm from the hip "
              f"(it slides: {knee_o[0] * 1000:.1f} mm forward at the default angle)",
              len(om.muscles) == 5 and len(om.joints) == 4 and -0.005 < knee_o[0] < -0.002
              and abs(np.linalg.norm(knee_w - X["femur_r"][:3, 3]) - np.linalg.norm(knee_o)) < 1e-9)

        ms = O.register(om)
        lm = O.mj_landmarks("R")
        mp = O.bone_maps(om, "r")
        R, k, a_o, a_m = mp["femur"]
        hit = a_m + k * R @ (knee_o - a_o)
        names = ms.names()
        check(f"bones carried over: the femur ×{k:.3f}, its knee onto the virtual rodent's "
              f"({np.linalg.norm(hit - lm['femur'][1]) * 1000:.2g} mm); {len(names)} muscles, the right leg "
              "mirrored to the left", np.linalg.norm(hit - lm["femur"][1]) < 1e-9 and len(names) == 10
              and {n[-1] for n in names} == {"L", "R"} and all(abs(np.linalg.det(v[0]) - 1) < 1e-9 for v in mp.values()))

        M = build(1.0, muscles=ms)
        q = np.zeros(M.model.nq)
        q[3] = 1.0
        for n, v in zip(M.hinge, M.q_stance, strict=True):
            q[M.qadr[n]] = v
        dofs = ["hip_L_extend", "knee_L", "ankle_L"]
        r = moment_arms(M, q, [f"{b}_L" for b in ("VL", "SM", "MG", "TA", "IP")], dofs)
        ok = (r["VL_L"]["knee_L"] > 1 and r["SM_L"]["hip_L_extend"] > 3 and r["SM_L"]["knee_L"] < -1
              and r["MG_L"]["ankle_L"] > 0.5 and r["TA_L"]["ankle_L"] < -0.5 and r["IP_L"]["hip_L_extend"] < -1)
        check("moment arms (mm, + extends): " + "; ".join(
            f"{n} " + " ".join(f"{d.split('_')[0]} {v:+.1f}" for d, v in r[n].items() if abs(v) > 0.05) for n in r), ok)
        rl = moment_arms(M, q, ["VL_R", "MG_R"], ["knee_R", "ankle_R"])
        check(f"left and right alike: VL knee {r['VL_L']['knee_L']:.2f} / {rl['VL_R']['knee_R']:.2f} mm",
              abs(r["VL_L"]["knee_L"] - rl["VL_R"]["knee_R"]) < 1e-6
              and abs(r["MG_L"]["ankle_L"] - rl["MG_R"]["ankle_R"]) < 1e-6)

        kf = ms.knee_fit
        check(f"the knee: length curves against OpenSim's sliding knee (rms mm, before → after) {kf}",
              all(a <= b for b, a in kf.values()) and "VL_R" in kf and kf["VL_R"][1] < 0.5)
        sl = ms.slack["R"]
        check(f"tendon slack lengths matched in OpenSim's pose (hip, knee, ankle off by {sl['pose_mismatch_deg']}°)",
              sl["pose_mismatch_deg"]["hip"] < 1.0 and all(m.l_ref > 0 for m in ms.muscles))

        ms2 = O.MuscleSet.from_json(ms.to_json(td / "m.json"))
        check("the muscle set saved and read back (.json)",
              ms2.names() == names and np.allclose(ms2.muscles[0].path[0][1], ms.muscles[0].path[0][1]))

        fa, fp, fv = MU.f_active, MU.f_passive, MU.f_velocity
        check(f"Hill curves: active {fa(1.0):.2f} at l_opt, {fa(1.5):.2f} at 1.5; passive {fp(1.0):.2f} / {fp(1.6):.2f}; "
              f"velocity {fv(0.0):.2f} isometric, {fv(-1.0):.2f} at v_max, {fv(-0.5):.2f} half, {fv(5.0):.2f} lengthening",
              abs(fa(1.0) - 1) < 1e-12 and fp(1.0) == 0 and abs(fp(1.6) - 1) < 1e-9 and abs(fv(0.0) - 1) < 1e-12
              and abs(fv(-1.0)) < 1e-12 and 0.1 < fv(-0.5) < 0.3 and 1.6 < fv(5.0) < 1.8)

        # static optimisation: torques the muscles can give, reproduced; more than they can, a reserve
        Q = np.tile(q, (2, 1))
        V = np.zeros((2, M.nv))
        mt = MU.MuscleTable(M)
        L, _, J = MU.tendon_state(M, q, V[0])
        lt, vt, ca = mt.fibre(L[mt.tid], np.zeros(len(mt.tid)))
        a_true = np.full(len(mt.tid), 0.3)
        F = mt.f_max * (a_true * fa(lt) * fv(vt) + fp(lt)) * ca
        gen = -J[mt.tid].T @ F
        tau = np.zeros((2, len(M.hinge)))
        for k2, h in enumerate(M.hinge):
            tau[0, k2] = gen[M.vadr[h]]
        tau[1] = tau[0]
        tau[1, M.hinge.index("knee_L")] += 5.0                         # 5 N·m: no rat's knee
        so = MU.static_optimisation(M, Q, V, tau)
        legs = [d.format(S=S) for S in "LR" for d in MU.LEG_DOFS]
        R0 = so["reserve"][0]
        got = [(-J[mt.tid].T @ so["force"][0])[M.vadr[d]] for d in legs]
        want = [tau[0, M.hinge.index(d)] for d in legs]
        check(f"static optimisation: the muscles' own torques reproduced (worst {np.abs(np.subtract(got, want)).max() * 1000:.2g} "
              f"mN·m, reserve {np.abs(R0).max() * 1000:.2g} mN·m); 5 N·m at the knee: reserve "
              f"{so['reserve'][1, 0, 3]:.2f} N·m, the muscles at {np.nanmax(so['act'][1]):.2f}",
              np.abs(np.subtract(got, want)).max() < 2e-4 and np.abs(R0).max() < 2e-4
              and so["reserve"][1, 0, 3] > 4.0 and np.nanmax(so["act"][1]) > 0.99)

        eng = O.apply_eng2008(ms2)
        vl = eng.get("VL_L")
        mass = float(O.ENG_2008["vastus lateralis"][3]) * O.SPECIFIC_TENSION
        check(f"Eng et al. 2008 architecture: {len(O.ENG_2008)} muscles in the table; VL {vl[0]} → {vl[1]} N "
              f"(PCSA {O.ENG_2008['vastus lateralis'][3]} cm² × {O.SPECIFIC_TENSION} N/cm² = {mass:.1f} N at 323 g)",
              len(O.ENG_2008) == 29 and vl is not None and 0.9 * mass < vl[1] < 1.2 * mass
              and "IP_L" in eng and "Eng" in ms2.source)

        # the command
        import pandas as pd

        from mufasa.pose3d import gait as G
        from mufasa.tools import pose3d_cli
        M2 = build(1.0)
        lab = list(M2.labels)
        d4 = M2.data()
        T = 90
        Xl = np.zeros((T, len(lab), 3))
        for t in range(T):
            qq = q.copy()
            qq[0] += 0.12 * t / 60.0
            d4.qpos[:] = qq
            mujoco.mj_kinematics(M2.model, d4)
            Xl[t] = d4.site_xpos[[M2.labels[n] for n in lab]] * 1000
        rows = {"frame": np.arange(T)}
        for j, n in enumerate(lab):
            rows.update({f"{n}_x": Xl[:, j, 0], f"{n}_y": Xl[:, j, 1], f"{n}_z": Xl[:, j, 2]})
        pd.DataFrame(rows).to_parquet(td / "s_3d.parquet", index=False)
        gp, _ = G.write_gait(G.estimate_gait(lab, Xl, np.arange(T), 60.0), td / "s_gait.parquet")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pose3d_cli.main(["dynamics", str(td / "s_3d.parquet"), "--gait", str(gp), "--fps", "60",
                                  "--calibrate", "40", "--frames", "10:79", "--muscles", str(td / "rat.osim"),
                                  "--save-muscles", str(td / "saved.json")])
        from mufasa.biomech.pipeline import read_dynamics
        tab, meta = read_dynamics(td / "s_dynamics.parquet")
        cols = ["VL_L_act", "MG_R_N", "reserve_knee_L_mNm", "reserve_hip_R_extend_mNm"]
        check(f"mufasa-pose3d dynamics --muscles: exit {rc}, {len(tab)} frames, {len(meta.get('muscles', []))} muscles; "
              f"reserve {meta['summary'].get('reserve_vs_torque_median')} × torque (median)",
              rc == 0 and len(tab) == 70 and all(c in tab for c in cols) and len(meta["muscles"]) == 10
              and (td / "saved.json").exists() and np.isfinite(tab["VL_L_act"]).mean() > 0.9)

    doc = (REPO / "docs" / "biomech.md").read_text()
    readme = (REPO / "README.md").read_text()
    check("docs/biomech.md: stage 3 (the muscles, the knee, the reserve, Eng 2008); README references",
          "Stage 3" in doc and "--muscles" in doc and "reserve" in doc and "Eng" in doc
          and "Johnson" in readme and "Crowninshield" in readme)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lj_muscles: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
