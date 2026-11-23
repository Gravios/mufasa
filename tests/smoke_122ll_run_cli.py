"""Smoke test for patch 122ll — ``mufasa run``: the 3-D pipeline's steps from the command line,
and ``mufasa run all``.

User: "Can we develop a CLI command group such as: mufasa run 3d-inference, mufasa run
gate-extrapolation".

* ``mufasa run`` (and ``--list``) lists the steps; ``mufasa`` alone still opens the workbench;
* ``mufasa run <step>`` is the ``mufasa-pose3d`` command with the same options;
* ``mufasa run all``, on a synthetic rat seen from above: 3d-inference → heights →
  gait-extrapolation (→ dynamics with MuJoCo), the files named from the video, the run recorded;
  run again, everything is up to date; ``--force``, ``--from``, ``--dry-run``.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def synthetic(cam, T=240, seed=0):
    """A rat walking in a loop, its 15 labels seen by ``cam`` (2-D px, with likelihoods)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 30.0
    cx, cy = 110 * np.sin(t * 0.6), 35 * np.sin(t * 0.9 + 1)
    yaw = np.unwrap(np.arctan2(np.gradient(cy), np.gradient(cx)))
    f = np.c_[np.cos(yaw), np.sin(yaw), 0 * t]
    lat = np.c_[-np.sin(yaw), np.cos(yaw), 0 * t]
    up = np.c_[0 * t, 0 * t, 1 + 0 * t]
    hips = np.c_[cx, cy, 42 + 0 * t]
    X = {"hip_left": hips + 25 * lat, "hip_right": hips - 25 * lat, "back_L6": hips + 12 * f + 18 * up,
         "back_V2": hips - 30 * f + 5 * up}
    for n, d in (("back_L2", 42), ("back_T8", 82), ("back_T4", 117)):
        X[n] = hips + d * f + (30 + 6 * np.sin(np.pi * d / 120)) * up
    neck = X["back_T4"] + 35 * f
    for n, o in {"head_back": (0, 0, 8), "head_mid": (22, 0, 12), "head_nose": (42, 0, 0),
                 "head_left": (4, 18, 6), "head_right": (4, -18, 6)}.items():
        X[n] = neck + o[0] * f + o[1] * lat + o[2] * up
    X["tail_V6"] = X["back_V2"] - 40 * f
    X["tail_V6"][:, 2] = 8
    X["tail_V18"] = X["tail_V6"] - 70 * f
    X["tail_V18"][:, 2] = 5
    X["tail_V32"] = X["tail_V18"] - 70 * f
    X["tail_V32"][:, 2] = 5
    cols = {}
    for n, P in X.items():
        uv = cam.project(P) + rng.normal(0, 0.8, (T, 2))
        cols[n + "_x"], cols[n + "_y"], cols[n + "_p"] = uv[:, 0], uv[:, 1], np.full(T, 0.97)
    return pd.DataFrame(cols)


def run(argv):
    from mufasa.cli.workbench_launcher import main
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue() + err.getvalue()


def main() -> int:
    from mufasa.cli import run as R
    rc, txt = run(["run", "--list"])
    check("mufasa run --list: the steps", rc == 0 and all(s in txt for s in
          ("3d-inference", "gait-extrapolation", "heights", "dynamics", "calibrate", "smooth", "body-fit", "all")))
    rc, txt = run(["run", "gait-extrapolation", "--help"])
    check("mufasa run gait-extrapolation --help: mufasa-pose3d gait's options",
          rc == 0 and "mufasa-pose3d gait" in txt and "--mass-g" in txt and "--bounds" in txt)
    rc, txt = run(["run", "gate-extrapolation"])
    check("an unknown step: exit 2 and the list", rc == 2 and "no step 'gate-extrapolation'" in txt)
    rc, txt = run(["run", "calibrate", "box", "--help"])
    check("mufasa run calibrate box --help: calibrate-box's options", rc == 0 and "--wall-height" in txt)
    src = (REPO / "mufasa" / "cli" / "workbench_launcher.py").read_text()
    check("mufasa without 'run' still opens the workbench", "workbench_main(argv)" in src)

    from mufasa.pose3d.camera import Camera
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cam = Camera.straight_down((330, 190), 243.0, 430.0, (640, 360), center_xy=(-5, 15),
                                   box={"x": [-340, 340], "y": [-260, 260], "z": [0, 300]})
        cam.save(td / "cam.toml")
        synthetic(cam).to_parquet(td / "R1_session.parquet")
        try:
            import mujoco  # noqa: F401
            to = []
            dyn = ["--frames", "60:99", "--dynamics-args", "--calibrate 40"]
        except ImportError:
            to, dyn = ["--to", "gait-extrapolation"], []
        out = td / "out"
        base = ["run", "all", "--pose", str(td / "R1_session.parquet"), "--camera", str(td / "cam.toml"),
                "--out-dir", str(out), *to, *dyn]
        rc, txt = run(base)
        files = {k: out / f"R1_session{k}" for k in ("_3d.parquet", "_3d_heights.parquet", "_gait.parquet",
                                                     "_steps.csv", "_dynamics.parquet", "_run.json")}
        want = [k for k in files if not (to and k == "_dynamics.parquet")]
        rec = json.loads(files["_run.json"].read_text()) if files["_run.json"].exists() else []
        check(f"mufasa run all: exit {rc}; " + ", ".join(f"{k} {'✓' if files[k].exists() else '✗'}" for k in want)
              + f"; {len(rec)} steps recorded", rc == 0 and all(files[k].exists() for k in want)
              and len(rec) == (3 if to else 4) and all(r["exit"] == 0 for r in rec)
              and rec[0]["command"].startswith("mufasa run 3d-inference"))
        rc2, txt2 = run(base)
        check("run again: every step up to date, nothing rerun",
              rc2 == 0 and txt2.count("up to date") == (3 if to else 4)
              and len(json.loads(files["_run.json"].read_text())) == len(rec))
        files["_3d_heights.parquet"].touch()
        rc3, txt3 = run(base + ["--dry-run"])
        check("a step's input newer than its output: it and the steps after would rerun (--dry-run)",
              rc3 == 0 and "[heights] up to date" in txt3 and "[gait-extrapolation] would run" in txt3)
        rc4, txt4 = run(base + ["--force", "--from", "gait-extrapolation", "--to", "gait-extrapolation"])
        check("--force --from/--to: only that step reruns",
              rc4 == 0 and "[gait-extrapolation] mufasa run gait-extrapolation" in txt4 and "[heights]" not in txt4
              and len(json.loads(files["_run.json"].read_text())) == len(rec) + 1)
        a = R._all_parser().parse_args(["--pose3d", str(td / "x" / "K9_3d.parquet")])
        p = R.plan(a)
        check("from a 3-D pose file: heights → gait-extrapolation → dynamics, named from its stem",
              [s["step"] for s in p] == ["heights", "gait-extrapolation", "dynamics"]
              and p[-1]["output"].name == "K9_dynamics.parquet")

    readme = (REPO / "README.md").read_text()
    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("README and docs/pose3d.md: mufasa run", "mufasa run all" in readme and "mufasa run" in doc)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122ll_run_cli: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
