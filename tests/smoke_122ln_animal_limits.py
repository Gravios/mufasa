"""Smoke test for patch 122ln — the rat's leg ranges, and an animal calibrated once.

User: "Rat leg ranges", "Per-animal calibration".

* ``build(limits="rat")``: the knees' and ankles' ranges narrowed to the rat's (Johnson et al.
  2008, carried onto the virtual rodent); nothing else changes; the stance stays inside;
* ``mufasa-pose3d animal`` (``mufasa run animal``): the virtual rodent calibrated to an animal
  from several sessions, written once: its size, its labels' places (found again after they were
  moved), the sessions;
* ``dynamics --animal`` and ``smooth --model --animal``: the same body and labels, no calibration.
"""
from __future__ import annotations

import io
import json
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
    import pandas as pd

    from mufasa.biomech import animal as AN
    from mufasa.biomech.mjmodel import RAT_LIMITS_DEG, build
    from mufasa.pose3d.io import write_pose3d

    Mm, Mr = build(1.0), build(1.0, limits="rat")
    changed = [n for k, n in enumerate(Mm.hinge) if not (np.isclose(Mm.lo[k], Mr.lo[k]) and np.isclose(Mm.hi[k], Mr.hi[k]))]
    kn = Mr.hinge.index("knee_L")
    an = Mr.hinge.index("ankle_L")
    inside = np.all((Mr.q_stance >= Mr.lo - 1e-9) & (Mr.q_stance <= Mr.hi + 1e-9))
    check(f"rat ranges: {sorted(changed)} narrowed (knee {np.degrees(Mm.lo[kn]):.0f}..{np.degrees(Mm.hi[kn]):.0f} → "
          f"{np.degrees(Mr.lo[kn]):.0f}..{np.degrees(Mr.hi[kn]):.0f}°, ankle {np.degrees(Mm.lo[an]):.0f}.. → "
          f"{np.degrees(Mr.lo[an]):.0f}..°); the stance inside; the model's limits in MuJoCo too",
          sorted(changed) == ["ankle_L", "ankle_R", "knee_L", "knee_R"] and inside
          and np.isclose(np.degrees(Mr.model.jnt_range[Mr.model.joint("knee_R").id][0]), RAT_LIMITS_DEG["knee_{S}"][0])
          and Mr.limits == "rat" and Mm.limits == "model")
    try:
        build(1.0, limits="mouse")
        bad = False
    except ValueError:
        bad = True
    check("an unknown limits name is refused", bad)

    # an animal whose labels sit off the model's places, seen in two sessions
    size = 1.1
    M0 = build(size)
    names = list(M0.labels)
    rng = np.random.default_rng(1)
    moved = {n: rng.normal(0, 0.008, 3) for n in names}
    Mt = build(size)
    for n in names:
        Mt.model.site_pos[Mt.labels[n]] += moved[n]
    q = np.zeros(Mt.model.nq)
    q[3] = 1.0
    for n, v in zip(Mt.hinge, Mt.q_stance, strict=True):
        q[Mt.qadr[n]] = v
    d = Mt.data()

    def session(T, seed):
        r = np.random.default_rng(seed)
        X = np.zeros((T, len(names), 3))
        for t in range(T):
            qq = q.copy()
            qq[0] += 0.002 * t
            for n in ("vertebra_1_extend", "atlas", "vertebra_cervical_1_bend", "hip_L_extend", "hip_R_extend"):
                qq[Mt.qadr[n]] += r.normal(0, 0.12)
            d.qpos[:] = qq
            mujoco.mj_kinematics(Mt.model, d)
            X[t] = d.site_xpos[[Mt.labels[n] for n in names]] * 1000
        return X

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        files = []
        for k, seed in enumerate((2, 3)):
            X = session(60, seed)
            df = pd.DataFrame({"frame": np.arange(len(X))})
            for j, n in enumerate(names):
                for c, ax in enumerate("xyz"):
                    df[f"{n}_{ax}"] = X[:, j, c]
            files.append(write_pose3d(df, td / f"R9_s{k}_3d.parquet", {"fps": 30.0}))
        from mufasa.cli.workbench_launcher import main as mufasa
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = mufasa(["run", "animal", *map(str, files), "--out", str(td / "R9.animal.json"), "--frames", "80",
                         "--mass-g", "420", "--size", str(size)])
        a = json.loads((td / "R9.animal.json").read_text())
        Mx = AN.build_for(a)
        cr = a["calibration_rms_mm"]
        check(f"mufasa run animal: exit {rc}; {a['name']}, size {a['size']} (estimated per session: "
              f"{a['session_sizes']}), {a['frames']} frames from 2 sessions, mass {a['mass_g']} g; the labels moved "
              f"{np.median([np.linalg.norm(v) for v in moved.values()]) * 1000:.0f} mm on their bodies, the fit "
              f"{cr[0]} → {min(cr)} mm", rc == 0 and a["name"] == "R9" and len(a["sessions"]) == 2 and a["frames"] == 80
              and a["mass_g"] == 420 and min(cr) < 1.0 and cr[0] > 3 * min(cr) and abs(Mx.mass_kg - 0.42) < 1e-9
              and all(abs(s / size - 1) < 0.05 for s in a["session_sizes"]))
        sp = a["labels"]["back_T8"]
        check(f"the file: each label's body and place at size 1 ({sp['body']}), the limits ({a['limits']}), the fit",
              sp["body"] == "torso" and len(sp["pos"]) == 3 and a["limits"] == "model"
              and a["calibration_rms_mm"][0] > min(a["calibration_rms_mm"]))

        from mufasa.pose3d import gait as G
        from mufasa.tools import pose3d_cli
        X = session(80, 4)
        gp, _ = G.write_gait(G.estimate_gait(names, X, np.arange(80), 30.0), td / "R9_s2_gait.parquet")
        df = pd.DataFrame({"frame": np.arange(80)})
        for j, n in enumerate(names):
            for c, ax in enumerate("xyz"):
                df[f"{n}_{ax}"] = X[:, j, c]
        write_pose3d(df, td / "R9_s2_3d.parquet", {"fps": 30.0})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pose3d_cli.main(["dynamics", str(td / "R9_s2_3d.parquet"), "--gait", str(gp), "--animal",
                                  str(td / "R9.animal.json"), "--frames", "5:60"])
        from mufasa.biomech.pipeline import read_dynamics
        tab, meta = read_dynamics(td / "R9_s2_dynamics.parquet")
        s = meta["summary"]
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            pose3d_cli.main(["dynamics", str(td / "R9_s2_3d.parquet"), "--gait", str(gp), "--frames", "5:60",
                             "--calibrate", "80", "--out", str(td / "own.parquet")])
        own = read_dynamics(td / "own.parquet")[1]["summary"]
        sites_ok = all(np.allclose(np.asarray(meta["label_sites_m"][n]), Mx.model.site_pos[Mx.labels[n]]) for n in names)
        check(f"dynamics --animal on a third session: exit {rc}, no calibration, the animal's labels, mass "
              f"{s['mass_g']} g; fit {s['fit_rms_mm']} mm (calibrating on that session alone: {own['fit_rms_mm']} mm)",
              rc == 0 and "calibrating" not in buf.getvalue() and sites_ok and s["animal"] == "R9"
              and abs(s["mass_g"] - 420) < 0.5 and s["fit_rms_mm"] <= own["fit_rms_mm"] + 1.0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pose3d_cli.main(["smooth", str(td / "R9_s2_3d.parquet"), "--out", str(td / "R9_s2_body.parquet"),
                                  "--model", "--animal", str(td / "R9.animal.json")])
        check(f"smooth --model --animal: exit {rc}, no calibration", rc == 0 and "calibrating" not in buf.getvalue()
              and (td / "R9_s2_body.parquet").exists())

    doc = (REPO / "docs" / "biomech.md").read_text()
    check("docs/biomech.md: the rat's ranges, the animal file", "--limits rat" in doc and "animal.json" in doc)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122ln_animal_limits: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
