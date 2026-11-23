"""Smoke test for patch 122li — the biomechanical model, stage 2: joint torques and ground forces
from the observed motion (the virtual rodent in MuJoCo).

User: a full biological model that supplements the silhouette fitting; "If the muscle loads can
be optimized, we should be able to generate more naturalistic movements and gaits"; hindlimbs
first, MuJoCo; "Continue".

* the virtual rodent in MuJoCo: a free root, scaled to the animal, label and sole sites, the
  simulation's springs, damping, armature and constraints off; MuJoCo's kinematics = the
  repo's reader's;
* inverse dynamics: standing on four paws, the support is the weight and nothing is left over;
  falling freely, no joint needs a torque;
* every joint fitted to labels (and paws); the labels' places on their bodies calibrated
  (STAC);
* ``mufasa-pose3d dynamics``: a pose file and its gait → ``_dynamics.parquet``.

Without MuJoCo installed, the checks that need it are skipped (it is the ``biomech`` extra).
"""
from __future__ import annotations

import io
import math
import sys
import tempfile
import tomllib
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def stance(M):
    """qpos of the stance standing on the floor (soles at their stance heights), root at the origin."""
    import mujoco
    q = np.zeros(M.model.nq)
    q[3] = 1.0
    for n, v in zip(M.hinge, M.q_stance, strict=True):
        q[M.qadr[n]] = v
    d = M.data()
    d.qpos[:] = q
    mujoco.mj_kinematics(M.model, d)
    low = min(d.site_xpos[list(s)][:, 2].min() for s in M.soles.values())
    q[2] -= low
    return q


def main() -> int:
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    check(f"the biomech extra: {extras.get('biomech')}", extras.get("biomech") == ["mujoco>=3.2"]
          and "mujoco>=3.2" in extras["all"])
    try:
        import mujoco
    except ImportError:
        check("MuJoCo not installed: the rest skipped (pip install \"mufasa[biomech]\")", True)
        return report()

    from mufasa.biomech import dynamics as D
    from mufasa.biomech import fit as F
    from mufasa.biomech.mjmodel import build
    from mufasa.pose3d.gait import PAWS
    from mufasa.pose3d.rodent_model import RodentModel

    rm = RodentModel()
    M1, M2 = build(1.0), build(1.2)
    m = M2.model
    off = (m.jnt_stiffness.max() == 0 and m.dof_damping.max() == 0 and m.dof_armature.max() == 0
           and m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT))
    check(f"the model: a free root (nq {m.nq}, nv {m.nv}), {len(M2.hinge)} hinges, {len(M2.labels)} labels, "
          f"{2 * len(M2.soles)} sole sites; {M1.mass_kg * 1000:.0f} g → {M2.mass_kg * 1000:.0f} g at 1.2× (size³); "
          "simulation springs, damping, armature and constraints off",
          m.nq == 74 and m.nv == 73 and len(M2.hinge) == 67 and abs(M2.mass_kg / M1.mass_kg - 1.2 ** 3) < 1e-9
          and bool(off))

    d = M1.data()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4):
        q = rm.lo + rng.random(len(rm.lo)) * (rm.hi - rm.lo)
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        for n in M1.hinge:
            d.qpos[M1.qadr[n]] = q[rm.joint_index[n]]
        mujoco.mj_kinematics(M1.model, d)
        T = rm.fk(q[None])[0]
        tor = T[rm.body_index["torso"], :3, 3] / 1000 - d.xpos[M1.model.body("torso").id]
        for i in range(2, M1.model.nbody):
            n = M1.model.body(i).name
            worst = max(worst, float(np.abs(d.xpos[i] + tor - T[rm.body_index[n], :3, 3] / 1000).max()))
    check(f"MuJoCo's kinematics = the repo's reader's (bodies {worst * 1000:.1e} mm)", worst < 1e-9)

    # standing and falling
    q = stance(M2)
    W = M2.mass_kg * 9.81
    z = np.zeros((1, m.nv))
    st = D.inverse(M2, q[None], z, z, [{p: "stance" for p in PAWS}])
    sup = st["grf"][0, :, 2].sum()
    check(f"standing on four paws: support {sup:.4f} N = weight {W:.4f} N; residual {np.linalg.norm(st['res_f'][0]):.0e} N; "
          f"the hips hold {st['tau'][0, M2.hinge.index('hip_L_extend')] * 1000:.0f} mN·m",
          abs(sup - W) < 1e-3 * W and np.linalg.norm(st["res_f"][0]) < 1e-3 * W)
    a = z.copy()
    a[0, 2] = -9.81
    fall = D.inverse(M2, q[None], z, a, [{p: "swing" for p in PAWS}])
    check(f"falling freely: no joint torque ({np.abs(fall['tau']).max():.0e} N·m), nothing left over",
          np.abs(fall["tau"]).max() < 1e-9 and np.linalg.norm(fall["res_f"][0]) < 1e-9)

    # a walk forward at constant speed: no acceleration, the weight on the paws
    T, fps = 90, 60.0
    Q = np.tile(q, (T, 1))
    Q[:, 0] += 0.15 * np.arange(T) / fps
    Qs, V, A = D.derivatives(M2, Q, fps)
    wk = D.inverse(M2, Qs, V, A, [{p: "stance" for p in PAWS}] * T)
    check(f"gliding at 150 mm/s: velocity {np.median(V[:, 0]) * 1000:.0f} mm/s, acceleration {np.abs(A[:, :3]).max():.0e} "
          f"m/s², support {np.median(wk['grf'][:, :, 2].sum(1)) / W:.3f} × weight",
          abs(np.median(V[:, 0]) - 0.15) < 1e-6 and np.abs(A[:, :3]).max() < 1e-6
          and abs(np.median(wk["grf"][:, :, 2].sum(1)) / W - 1) < 1e-3)

    # labels from known poses, fitted back
    names = list(M2.labels)
    fitter = F.PoseFitter(M2, names)
    poses, obs = [], []
    for k in range(12):
        qq = q.copy()
        for n, dv in (("vertebra_1_extend", 0.15 * math.sin(k / 2)), ("hip_L_extend", 0.3 * math.sin(k / 3)),
                      ("atlas", 0.2 * math.cos(k / 2))):
            qq[M2.qadr[n]] += dv
        qq[0] += 0.004 * k
        d2 = M2.data()
        d2.qpos[:] = qq
        mujoco.mj_kinematics(m, d2)
        poses.append(qq)
        obs.append(d2.site_xpos[fitter.sites].copy() * 1000)
    obs = np.array(obs)
    out = F.fit_trajectory(M2, names, obs, None)
    check(f"fitted to labels from known poses: {np.nanmax(out['rms_mm']):.2f} mm (worst frame); the root within "
          f"{np.abs(out['qpos'][:, :3] - np.array(poses)[:, :3]).max() * 1000:.1f} mm",
          np.nanmax(out["rms_mm"]) < 2.0 and np.abs(out["qpos"][:, :3] - np.array(poses)[:, :3]).max() < 0.01)

    # calibration: labels moved on their bodies are found again
    M3 = build(1.2)
    moved = rng.normal(0, 0.012, (len(names), 3))
    for k, n in enumerate(names):
        M3.model.site_pos[M3.labels[n]] += moved[k]
    poses2 = []
    for _ in range(60):
        qq = q.copy()
        for n in ("vertebra_1_extend", "vertebra_2_bend", "atlas", "cervical", "hip_L_extend", "hip_R_extend"):
            if n in M3.qadr:
                qq[M3.qadr[n]] += rng.normal(0, 0.12)
        qq[0] += rng.normal(0, 0.03)
        qq[1] += rng.normal(0, 0.03)
        poses2.append(qq)
    true_obs = []
    for qq in poses2:
        d3 = M3.data()
        d3.qpos[:] = qq
        mujoco.mj_kinematics(M3.model, d3)
        true_obs.append(d3.site_xpos[[M3.labels[n] for n in names]].copy() * 1000)
    M4 = build(1.2)                                   # the model's places: wrong by ~20 mm
    before = F.fit_trajectory(M4, names, np.array(true_obs), None)["rms_mm"]
    cal = F.calibrate(M4, names, np.array(true_obs), rounds=4)
    after = F.fit_trajectory(M4, names, np.array(true_obs), None)["rms_mm"]
    check(f"calibration: labels moved ~{np.linalg.norm(moved, axis=1).mean() * 1000:.0f} mm on their bodies; fit "
          f"{np.nanmedian(before):.1f} → {np.nanmedian(after):.1f} mm (rounds {cal['rms_mm']})",
          np.nanmedian(after) < 0.5 * np.nanmedian(before))

    # the command, on a file of labels and its gait
    import pandas as pd

    from mufasa.pose3d import gait as G
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        T = 150
        d4 = M2.data()
        X = np.zeros((T, len(names), 3))
        for t in range(T):
            qq = q.copy()
            qq[0] += 0.12 * t / 60.0
            d4.qpos[:] = qq
            mujoco.mj_kinematics(m, d4)
            X[t] = d4.site_xpos[[M2.labels[n] for n in names]] * 1000
        rows = {"frame": np.arange(T)}
        for j, n in enumerate(names):
            rows.update({f"{n}_x": X[:, j, 0], f"{n}_y": X[:, j, 1], f"{n}_z": X[:, j, 2]})
        pd.DataFrame(rows).to_parquet(td / "s_3d.parquet", index=False)
        gp, _ = G.write_gait(G.estimate_gait(names, X, np.arange(T), 60.0), td / "s_gait.parquet")
        from mufasa.tools import pose3d_cli
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pose3d_cli.main(["dynamics", str(td / "s_3d.parquet"), "--gait", str(gp), "--fps", "60",
                                  "--calibrate", "60", "--frames", "20:129"])
        from mufasa.biomech.pipeline import read_dynamics
        tab, meta = read_dynamics(td / "s_dynamics.parquet")
        s = meta.get("summary", {})
        cols = ["fit_rms_mm", "LH_grf_z_N", "residual_N", "q_knee_L_deg", "tau_knee_L_mNm", "tau_hip_R_extend_mNm"]
        check(f"mufasa-pose3d dynamics: exit {rc}, {len(tab)} frames, fit {s.get('fit_rms_mm')} mm, support "
              f"{s.get('support_vs_weight')} × weight; {len(meta.get('joints', []))} joints, the calibrated sites in the file",
              rc == 0 and len(tab) == 110 and all(c in tab for c in cols) and len(meta["joints"]) == 67
              and len(meta["label_sites_m"]) == len(names) and abs(s["support_vs_weight"] - 1) < 0.05
              and "wrote" in buf.getvalue())

    doc = (REPO / "docs" / "biomech.md").read_text() if (REPO / "docs" / "biomech.md").exists() else ""
    readme = (REPO / "README.md").read_text()
    check("docs/biomech.md (the stages, the residual, K7); README: the biomech extra",
          "residual" in doc and "mufasa-pose3d dynamics" in doc and "STAC" in doc and "| `biomech` |" in readme)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122li_dynamics: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
