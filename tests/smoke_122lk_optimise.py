"""Smoke test for patch 122lk — the biomechanical model, stage 4: whole movements re-solved for
tracking, balance and effort, the muscles' activations smooth over time.

User: "Would it be better to run multiple forward and backward sweeps to generate trajectories
which maintain reasonable correlated muscle load trajectories?"; "continue".

* a stance disturbed (a leg folded, the body lifted off the floor, the neck's redundant twists
  jittering): re-solved, the paws are back on the floor, the ground carries the weight, nothing
  is left over at the root, the neck is still, the labels are tracked;
* windows stitched into one motion;
* with muscles: the legs' torques come from the muscles, and their activations change slowly;
* ``mufasa-pose3d dynamics --optimise``.
"""
from __future__ import annotations

import importlib.util
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


def main() -> int:
    try:
        import mujoco
    except ImportError:
        check("MuJoCo not installed: skipped (pip install \"mufasa[biomech]\")", True)
        return report()
    from mufasa.biomech import optimise as OP
    from mufasa.biomech.mjmodel import build
    from mufasa.pose3d.gait import PAWS

    M = build(1.0)
    m = M.model
    names = list(M.labels)
    q = np.zeros(m.nq)
    q[3] = 1.0
    for n, v in zip(M.hinge, M.q_stance, strict=True):
        q[M.qadr[n]] = v
    d = M.data()
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    q[2] -= min(d.site_xpos[list(s)][:, 2].min() for s in M.soles.values())
    T, fps = 16, 30.0
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    X = np.tile(d.site_xpos[[M.labels[n] for n in names]] * 1000, (T, 1, 1))
    soles0 = {p: d.site_xpos[list(M.soles[p])].mean(0).copy() for p in PAWS}

    Qref = np.tile(q, (T, 1))
    Qref[:, 2] += 0.003                                           # lifted 3 mm
    Qref[:, M.qadr["knee_L"]] += 0.35                             # a leg folded differently
    Qref[:, M.qadr["hip_L_extend"]] -= 0.2
    jit = 0.3 * (-1) ** np.arange(T)
    Qref[:, M.qadr["vertebra_cervical_3_twist"]] += jit           # the neck's twists, against each other
    Qref[:, M.qadr["vertebra_axis_twist"]] -= jit
    modes = [{p: "stance" for p in PAWS}] * T
    out = OP.optimise_window(M, names, X, Qref, modes, fps, muscles=False, iters=40)
    Q = out["qpos"]
    W = M.mass_kg * 9.81
    mid = slice(1, T - 1)
    sup = np.nansum(out["grf"][mid, :, 2], axis=1) / W
    res = np.linalg.norm(out["res_f"][mid], axis=1) / W
    zs, slide = [], []
    for t in range(T):
        d.qpos[:] = Q[t]
        mujoco.mj_kinematics(m, d)
        for k, p in enumerate(PAWS):
            h, s = M.soles[p]
            zs += [d.site_xpos[h][2] - M.sole_z[p][0], d.site_xpos[s][2] - M.sole_z[p][1]]
            slide.append(np.linalg.norm(d.site_xpos[[h, s]].mean(0)[:2] - soles0[p][:2]) if t == 0 else 0)
    neck = np.abs(np.diff(Q[:, M.qadr["vertebra_cervical_3_twist"]], 2)).max()
    check(f"a disturbed stance re-solved: soles on the floor (worst {np.abs(zs).max() * 1000:.2f} mm), support "
          f"{np.median(sup):.3f} × weight, residual {np.median(res):.3f} × weight (was {out['parts0'].get('root', 0):.0f} "
          f"→ {out['parts'].get('root', 0):.1f} as cost), labels {np.nanmax(out['rms_mm']):.2f} mm",
          np.abs(zs).max() < 0.001 and abs(np.median(sup) - 1) < 0.05 and np.median(res) < 0.03
          and np.nanmax(out["rms_mm"]) < 3.0)
    check(f"the neck's jitter (±{np.degrees(0.3):.0f}°, the twists against each other) stilled: second difference "
          f"{np.degrees(neck):.2f}°", np.degrees(neck) < 2.0)
    check(f"costs: {out['parts0']} → {out['parts']}", out["cost"] < 0.05 * out["cost0"])

    # windows stitched
    T2 = 26
    Q2 = np.tile(q, (T2, 1))
    Q2[:, 0] += 0.06 * np.arange(T2) / fps                        # gliding forward, 60 mm/s
    X2 = np.zeros((T2, len(names), 3))
    for t in range(T2):
        d.qpos[:] = Q2[t]
        mujoco.mj_kinematics(m, d)
        X2[t] = d.site_xpos[[M.labels[n] for n in names]] * 1000
    modes2 = [{p: "stance" for p in PAWS}] * T2
    Q2r = Q2.copy()
    Q2r[:, M.qadr["knee_R"]] += 0.2
    o2 = OP.optimise_trajectory(M, names, X2, Q2r, modes2, fps, muscles=False, window=12, step=7, iters=30)
    o1 = OP.optimise_window(M, names, X2, Q2r, modes2, fps, muscles=False, iters=30)
    steps = np.degrees(np.abs(np.diff(o2["qpos"][:, 7:], axis=0)).max(1))
    one = np.degrees(np.abs(np.diff(o1["qpos"][:, 7:], axis=0)).max(1))
    joins = [7, 8, 14, 15]
    check(f"gliding on planted paws, windows of 12 frames 7 apart, stitched over {T2} frames: every frame filled; "
          f"the largest joint step between frames {steps.max():.1f}° (one window over all: {one.max():.1f}°), at the "
          f"joins {steps[joins].max():.1f}°; residual {np.nanmedian(np.linalg.norm(o2['res_f'], axis=1)) / W:.3f} × weight",
          np.isfinite(o2["qpos"]).all() and steps.max() <= 1.25 * one.max() and steps[joins].max() <= 1.25 * one.max()
          and np.nanmedian(np.linalg.norm(o2["res_f"], axis=1)) / W < 0.05)

    # with muscles
    spec = importlib.util.spec_from_file_location("lj", REPO / "tests" / "smoke_122lj_muscles.py")
    lj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lj)
    from mufasa.biomech import muscle as MU
    from mufasa.biomech.osim import read_osim, register
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ms = register(read_osim(lj.write_osim(td / "rat.osim")))
        Mm = build(1.0, muscles=ms)
        T3 = 14
        Q3 = np.tile(q, (T3, 1))
        Q3[:, 2] += 0.004 * np.sin(np.arange(T3) / 2.0)           # bobbing 4 mm
        X3 = np.zeros((T3, len(names), 3))
        dm = Mm.data()
        for t in range(T3):
            dm.qpos[:] = Q3[t]
            mujoco.mj_kinematics(Mm.model, dm)
            X3[t] = dm.site_xpos[[Mm.labels[n] for n in names]] * 1000
        o3 = OP.optimise_window(Mm, names, X3, Q3, [{p: "stance" for p in PAWS}] * T3, fps, muscles=True, iters=40)
        mo = OP.muscle_outputs(Mm, o3["qpos"], o3["V"], o3["act"], o3["tau"])
        ok = np.isfinite(mo["reserve"]).all((1, 2))
        asked = np.abs(mo["asked"][ok]).sum((1, 2))
        rsv = np.abs(mo["reserve"][ok]).sum((1, 2))
        da = np.mean(np.abs(np.diff(o3["act"][1:-1], axis=0)))
        so = MU.static_optimisation(Mm, o3["qpos"], np.nan_to_num(o3["V"]), np.nan_to_num(o3["tau"]))
        ok2 = np.isfinite(so["reserve"]).all((1, 2)) & ok
        rsv_so = np.abs(so["reserve"][ok2]).sum((1, 2)) / np.abs(so["asked"][ok2]).sum((1, 2))
        da_so = np.nanmean(np.abs(np.diff(so["act"][1:-1], axis=0)))
        check(f"with muscles (a small model: 5 a leg, short of the hip's three torques): reserve "
              f"{np.median(rsv / asked):.2f} × torque, frame by frame {np.median(rsv_so):.2f}; activations change "
              f"{da:.4f} a frame on average, frame by frame {da_so:.4f}",
              np.median(rsv / asked) <= np.median(rsv_so) + 0.05 and da <= da_so and np.isfinite(o3["act"]).all())

        # the command
        import pandas as pd

        from mufasa.pose3d import gait as G
        from mufasa.tools import pose3d_cli
        T4 = 60
        X4 = np.zeros((T4, len(names), 3))
        for t in range(T4):
            qq = q.copy()
            qq[0] += 0.08 * t / 60.0
            d.qpos[:] = qq
            mujoco.mj_kinematics(m, d)
            X4[t] = d.site_xpos[[M.labels[n] for n in names]] * 1000
        rows = {"frame": np.arange(T4)}
        for j, n in enumerate(names):
            rows.update({f"{n}_x": X4[:, j, 0], f"{n}_y": X4[:, j, 1], f"{n}_z": X4[:, j, 2]})
        pd.DataFrame(rows).to_parquet(td / "s_3d.parquet", index=False)
        gp, _ = G.write_gait(G.estimate_gait(names, X4, np.arange(T4), 60.0), td / "s_gait.parquet")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pose3d_cli.main(["dynamics", str(td / "s_3d.parquet"), "--gait", str(gp), "--fps", "60",
                                  "--calibrate", "30", "--frames", "10:39", "--optimise", "--iters", "20"])
        from mufasa.biomech.pipeline import read_dynamics
        tab, meta = read_dynamics(td / "s_dynamics.parquet")
        s = meta["summary"]
        check(f"mufasa-pose3d dynamics --optimise: exit {rc}, {len(tab)} frames; residual "
              f"{s['before_optimising']['residual_vs_weight_median']} → {s['residual_vs_weight_median']} × weight; "
              "paw positions in the table",
              rc == 0 and len(tab) == 30 and "LH_paw_x_mm" in tab and "optimised" in meta["method"]
              and s["residual_vs_weight_median"] <= s["before_optimising"]["residual_vs_weight_median"] + 0.01
              and "before optimising" in buf.getvalue())

    doc = (REPO / "docs" / "biomech.md").read_text()
    check("docs/biomech.md: stage 4 (--optimise, the costs, the sweeps, K7)",
          "Stage 4" in doc and "--optimise" in doc and "sweeps" in doc and "smooth" in doc)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lk_optimise: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
