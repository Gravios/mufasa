"""Smoke test for patch 122ki — faster single-camera 3-D pose.

User: "3d speed up" (3-D inference took about 30 min a session, 1.5 days
for 67 sessions).

* the grouped forward-difference Jacobian equals SciPy's sparse one;
* the head fit with it gives the old fit (SciPy's own sparse Jacobian) on
  a synthetic rat;
* chunks on a process pool give the same reconstruction as in-process;
* a project run with several videos runs them on parallel workers and
  writes every output; one video uses the workers for its chunks;
* CLI --workers and the 3-D tab's Workers box reach RunSettings.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def _synthetic_rat():
    """The synthetic rat of smoke_122kf (its ``make`` function)."""
    src = (REPO / "tests" / "smoke_122kf_pose3d.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "make")
    ns = {"np": np, "pd": pd, "Rot": Rot}
    exec(compile(ast.Module([fn], []), "make", "exec"), ns)
    return ns["make"]


def main() -> int:
    from scipy.optimize._numdiff import approx_derivative
    from scipy.sparse import lil_matrix

    from mufasa.pose3d.camera import Camera
    from mufasa.pose3d.mono import MonoConfig, RigidHead, load_pose2d, reconstruct_mono
    from mufasa.pose3d.solver import GroupedJacobian, frame_groups

    # ---------------------------------------------------------- Jacobian
    rng = np.random.default_rng(1)
    n, k = 40, 3
    w = rng.normal(size=(n, k))

    def fun(x):
        z = x.reshape(n, k)
        return np.r_[(np.sin(z) * w).ravel(), (np.diff(z, axis=0) ** 2).ravel(), z[:, 0] * z[:, 1]]
    sp = lil_matrix((n * k + (n - 1) * k + n, n * k), dtype=np.int8)
    for i in range(n):
        sp[i * k:(i + 1) * k, i * k:(i + 1) * k] = 1
        sp[n * k + (n - 1) * k + i, i * k:(i + 1) * k] = 1
    for i in range(n - 1):
        sp[n * k + i * k:n * k + (i + 1) * k, i * k:(i + 2) * k] = 1
    x0 = rng.normal(size=n * k)
    J = GroupedJacobian(fun, sp.tocsr(), frame_groups(n, k))
    ref = approx_derivative(fun, x0, sparsity=sp.tocsr()).toarray()
    check(f"grouped Jacobian ({J.n_groups} evaluations) equals SciPy's",
          J.n_groups == 2 * k and np.allclose(J(x0).toarray(), ref, atol=1e-6))

    # ---------------------------------------------------------- head solver
    make = _synthetic_rat()
    cam = Camera.straight_down((330, 190), 243.0, 430.0, (640, 360), center_xy=(-5, 15),
                               box={"x": [-340, 340], "y": [-260, 260], "z": [0, 300]})
    X, df2d, _ = make(900, cam=cam)
    syn_z = {"head_back": 2.0, "head_mid": 6.0, "head_nose": -6.0}
    pose = load_pose2d(df2d)
    head = ["head_back", "head_mid", "head_nose", "head_left", "head_right"]
    huv = np.stack([pose.uv[m] for m in head], 1)
    hlk = np.stack([pose.lk[m] for m in head], 1)
    rh = RigidHead(cam, head, (3, 4), (0, 2), 36.0, syn_z)
    rh.fit_template(huv[::3], hlk[::3])
    import mufasa.pose3d.mono as mono
    from scipy.optimize import least_squares

    def old_solve(fun, x0, sp, per_frame, *, max_nfev):        # before 122ki
        return least_squares(fun, x0, jac_sparsity=sp, max_nfev=max_nfev, x_scale="jac", tr_solver="lsmr")
    costs, Xs = {}, {}
    new_solve = mono.solve_sequence
    for name, solve in (("new", new_solve), ("old", old_solve)):
        mono.solve_sequence = solve
        try:
            _, Xs[name] = rh.fit_sequence(huv, hlk, cam.box)
        finally:
            mono.solve_sequence = new_solve
        costs[name] = rh.last_cost
    true = np.stack([X[m] for m in head], 1)
    err = np.median(np.linalg.norm(Xs["new"] - true, axis=2))
    dev = np.linalg.norm(Xs["new"] - Xs["old"], axis=2)
    check(f"head fit: same as before 122ki (cost {costs['new']:.2f} vs {costs['old']:.2f}; "
          f"positions within {dev.max():.2g} mm)",
          abs(costs["new"] - costs["old"]) <= 1e-3 * costs["old"] and dev.max() < 0.5)
    check(f"head fit: {err:.1f} mm from the truth", err < 5)

    # ---------------------------------------------------------- chunk pool
    cfg = dict(chunk=300, head_template_z=syn_z)
    a, _ = reconstruct_mono(pose, cam, MonoConfig(**cfg, workers=1))
    b, _ = reconstruct_mono(pose, cam, MonoConfig(**cfg, workers=2))
    zc = [c for c in a.columns if c.endswith(("_x", "_y", "_z"))]
    check("chunks on 2 worker processes = in-process (3 chunks)",
          np.allclose(a[zc].to_numpy(), b[zc].to_numpy(), equal_nan=True, atol=1e-6))

    # ---------------------------------------------------------- videos in parallel
    from mufasa.pose3d.pipeline import Job, RunSettings, default_workers, run_pose3d
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cam.save(td / "camera.toml")
        jobs = []
        for i in range(3):
            _, d, _ = make(400, seed=10 + i, cam=cam)
            d.to_parquet(td / f"rat{i}.parquet")
            jobs.append(Job(f"rat{i}", td / f"rat{i}.parquet", td / "camera.toml"))
        msgs: list[str] = []
        st = RunSettings(mono=MonoConfig(chunk=300, head_template_z=syn_z), workers=2)
        reps = run_pose3d(jobs, st, td / "out", progress=msgs.append)
        outs = sorted(p.name for p in (td / "out").glob("*_3d.parquet"))
        check("3 videos on 2 workers: every output written, reports in job order",
              outs == ["rat0_3d.parquet", "rat1_3d.parquet", "rat2_3d.parquet"]
              and list(reps) == ["rat0", "rat1", "rat2"])
        check("progress names the workers and counts videos done",
              msgs[0] == "3 videos on 2 workers" and sum("done" in m for m in msgs) == 3)
        one = run_pose3d(jobs[:1], st, td / "one", progress=msgs.append)
        same = pd.read_parquet(td / "one" / "rat0_3d.parquet")
        par = pd.read_parquet(td / "out" / "rat0_3d.parquet")
        check("a single video splits its chunks instead, same result",
              "rat0" in one and np.allclose(same[zc[:30]].to_numpy(), par[zc[:30]].to_numpy(),
                                            equal_nan=True, atol=1e-6))
    check("default workers: all cores but one", default_workers() == max(1, (os.cpu_count() or 2) - 1))

    # ---------------------------------------------------------- CLI and tab
    from mufasa.tools.pose3d_cli import build_parser, settings_from
    a_ = build_parser().parse_args(["infer", "--project", "p.toml", "--workers", "3"])
    check("CLI --workers reaches RunSettings", settings_from(a_).workers == 3)
    a_ = build_parser().parse_args(["infer", "--project", "p.toml"])
    check("CLI default: all cores but one", settings_from(a_).workers == default_workers())
    src = (REPO / "mufasa" / "ui_qt" / "forms" / "pose3d.py").read_text()
    check("3-D tab: Workers box passed to RunSettings",
          'f.addRow("Workers", self.workers)' in src and "workers=self.workers.value()" in src)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122ki_pose3d_speed: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
