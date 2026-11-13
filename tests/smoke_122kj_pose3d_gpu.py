"""Smoke test for patch 122kj — 3-D pose solver on the GPU (PyTorch).

User: "Generate the gpu version".

* the PyTorch head and spine-chain residuals equal the NumPy ones;
* the PyTorch solver (a port of SciPy's trust-region solver) gives the
  SciPy fit on a synthetic rat, for the head and for the whole
  reconstruction;
* device selection: auto → cuda / PyTorch-CPU / SciPy; a missing GPU or
  PyTorch is reported, not crashed into;
* CUDA, when this machine has it: CPU and GPU agree (skipped otherwise);
* several videos on the GPU run in fresh (spawned) processes;
* CLI --device and gpu-check, the tab's Solver box, the gpu extra.
"""
from __future__ import annotations

import ast
import os
import sys
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
    src = (REPO / "tests" / "smoke_122kf_pose3d.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "make")
    ns = {"np": np, "pd": pd, "Rot": Rot}
    exec(compile(ast.Module([fn], []), "make", "exec"), ns)
    return ns["make"]


def main() -> int:
    import torch

    import mufasa.pose3d.mono as mono
    from mufasa.pose3d import gpu
    from mufasa.pose3d.camera import Camera

    rng = np.random.default_rng(0)

    # ---------------------------------------------------------- head residuals
    make = _synthetic_rat()
    cam = Camera.straight_down((330, 190), 243.0, 430.0, (640, 360), center_xy=(-5, 15),
                               box={"x": [-340, 340], "y": [-260, 260], "z": [0, 300]})
    X, df2d, _ = make(600, cam=cam)
    syn_z = {"head_back": 2.0, "head_mid": 6.0, "head_nose": -6.0}
    pose = mono.load_pose2d(df2d)
    head = ["head_back", "head_mid", "head_nose", "head_left", "head_right"]
    huv = np.stack([pose.uv[m] for m in head], 1)
    hlk = np.stack([pose.lk[m] for m in head], 1)
    rh = mono.RigidHead(cam, head, (3, 4), (0, 2), 36.0, syn_z)
    rh.fit_template(huv[::3], hlk[::3])

    cap = {}

    def grab(fun, x0, sp, per, *, max_nfev):
        cap["fun"], cap["x0"] = fun, x0
        raise StopIteration
    real = mono.solve_sequence
    mono.solve_sequence = grab
    try:
        rh.fit_sequence(huv, hlk, cam.box)
    except StopIteration:
        pass
    finally:
        mono.solve_sequence = real
    x0 = cap["x0"].reshape(-1, 6)
    wt = np.sqrt(np.clip((np.nan_to_num(hlk) - .3) / .6, 0, 1))
    wt[~np.isfinite(huv).all(2)] = 0
    f = gpu.head_residual_fn(cam, rh.T, cam.box, sigma=2.5, roll_sd_deg=20, wall_tolerance_mm=20, device="cpu")
    xt = x0 + rng.normal(0, [3, 3, 3, .05, .05, .05], x0.shape)
    rt = torch.func.vmap(f)(torch.tensor(xt), torch.tensor(np.nan_to_num(huv)), torch.tensor(wt)).numpy()
    ct = 0.5 * (rt ** 2).sum() + 0.5 * (gpu.HEAD_SMOOTH_W * np.diff(xt, axis=0) ** 2).sum()
    rn = cap["fun"](xt.ravel())
    check(f"head residuals: PyTorch cost {ct:.6f} = NumPy {0.5 * rn @ rn:.6f}", abs(ct - 0.5 * rn @ rn) < 1e-6 * ct)

    # ---------------------------------------------------------- head fit
    _, Xs = rh.fit_sequence(huv, hlk, cam.box, device="cpu")
    c_s = rh.last_cost
    _, Xt = rh.fit_sequence(huv, hlk, cam.box, device="torch-cpu")
    c_t = rh.last_cost
    dev = np.linalg.norm(Xt - Xs, axis=2)
    true = np.stack([X[m] for m in head], 1)
    err = np.median(np.linalg.norm(Xt - true, axis=2))
    check(f"head fit: PyTorch cost {c_t:.2f} vs SciPy {c_s:.2f}; positions within {np.percentile(dev, 99):.2g} mm (p99)",
          abs(c_t - c_s) <= 1e-3 * c_s and np.percentile(dev, 99) < 0.5)
    check(f"head fit: {err:.1f} mm from the truth", err < 5)

    # ---------------------------------------------------------- chain residuals
    caps = []
    real_chunk = mono._chain_chunk

    def keep(p):
        caps.append(p)
        return real_chunk(p)
    mono._chain_chunk = keep
    try:
        cfg = dict(chunk=300, head_template_z=syn_z)
        out_s, rep_s = mono.reconstruct_mono(pose, cam, mono.MonoConfig(**cfg, device="cpu"))
    finally:
        mono._chain_chunk = real_chunk
    p = dict(caps[0], device="cpu")
    capc = {}

    def grab2(fun, x0, sp, per, *, max_nfev):
        capc["fun"] = fun
        raise StopIteration
    mono.solve_sequence = grab2
    try:
        real_chunk(p)
    except StopIteration:
        pass
    finally:
        mono.solve_sequence = real
    free, chain = p["free"], p["chain"]
    fc = gpu.chain_residual_fn(p["C"], chain, free, p["n_mid"], "cpu")
    z = p["z0"] + rng.normal(0, 5, p["z0"].shape)
    args = (z, np.stack([p["D"][m] for m in free], 1),
            np.stack([p["fixed"][m] for m in chain if m not in free], 1), p["z0"], p["hd"],
            np.stack([p["lim"][k] for k in ("M50", "M95", "F5", "F95")], 1))
    rc = torch.func.vmap(fc)(*(torch.tensor(a) for a in args)).numpy()
    cc = 0.5 * (rc ** 2).sum() + 0.5 * ((np.diff(z, axis=0) / 8) ** 2).sum()
    rn = capc["fun"](z.ravel())
    check(f"spine-chain residuals: PyTorch cost {cc:.4f} = NumPy {0.5 * rn @ rn:.4f}",
          abs(cc - 0.5 * rn @ rn) < 1e-6 * cc)

    # ---------------------------------------------------------- whole reconstruction
    out_t, rep_t = mono.reconstruct_mono(pose, cam, mono.MonoConfig(**cfg, device="torch-cpu"))
    zc = [c for c in out_s.columns if c.endswith(("_x", "_y", "_z"))]
    d = np.abs(out_t[zc].to_numpy() - out_s[zc].to_numpy())
    check(f"reconstruction: PyTorch vs SciPy median {np.nanmedian(d):.1e} mm, p99 {np.nanpercentile(d, 99):.2f} mm",
          np.nanmedian(d) < 0.01 and np.nanpercentile(d, 99) < 2)
    check("report names the solver", rep_t["solver"] == "PyTorch on the CPU" and rep_s["solver"] == "SciPy")
    out_w, _ = mono.reconstruct_mono(pose, cam, mono.MonoConfig(**cfg, device="torch-cpu", workers=2))
    check("PyTorch on the CPU: 2 workers = 1 (one thread per solve)",
          np.allclose(out_w[zc].to_numpy(), out_t[zc].to_numpy(), equal_nan=True, atol=1e-9))

    # ---------------------------------------------------------- device selection
    auto = mono.resolve_device("auto")
    check(f"auto → {auto}", auto == ("cuda" if gpu.cuda_available() else "torch-cpu"))
    check("cpu → SciPy", mono.resolve_device("cpu") == "cpu")
    saved = gpu.torch
    gpu.torch = None
    try:
        no_torch_auto = mono.resolve_device("auto")
        try:
            mono.resolve_device("torch-cpu")
            raised = False
        except RuntimeError as e:
            raised = "PyTorch" in str(e)
    finally:
        gpu.torch = saved
    check("without PyTorch: auto → SciPy, torch-cpu says PyTorch is needed", no_torch_auto == "cpu" and raised)
    if not gpu.cuda_available():
        try:
            mono.resolve_device("cuda")
            said = False
        except RuntimeError as e:
            said = "CUDA" in str(e)
        check("no GPU here: asking for cuda says CUDA is not available", said)
        check("CUDA agreement: skipped (no GPU on this machine; run `mufasa-pose3d gpu-check`)", True)
    else:
        check("GPU here: cuda selectable", mono.resolve_device("cuda") == "cuda")
        res = gpu.gpu_check(4000, verbose=lambda s: None)
        check(f"CUDA agrees with the CPU (max |difference| {res['max_difference']:.1e})", res["max_difference"] < 1e-3)

    # ---------------------------------------------------------- pipeline, CLI, tab, packaging
    src = (REPO / "mufasa" / "pose3d" / "pipeline.py").read_text()
    check("several videos on the GPU: spawned workers, at most GPU_VIDEO_WORKERS",
          'get_context("spawn")' in src and "n = min(n, GPU_VIDEO_WORKERS)" in src)
    check("no chunk pool on the GPU (one process drives it)",
          'resolve_device(device) == "cuda"' in (REPO / "mufasa" / "pose3d" / "mono.py").read_text())
    from mufasa.tools.pose3d_cli import build_parser, settings_from
    from mufasa.tools.pose3d_cli import main as cli
    a_ = build_parser().parse_args(["infer", "--project", "p.toml", "--device", "torch-cpu"])
    check("CLI --device reaches MonoConfig", settings_from(a_).mono.device == "torch-cpu")
    check("CLI default device: auto", settings_from(build_parser().parse_args(["infer", "--project", "p.toml"])).mono.device == "auto")
    check("CLI gpu-check runs", cli(["gpu-check", "--frames", "1500"]) == 0)
    fsrc = (REPO / "mufasa" / "ui_qt" / "forms" / "pose3d.py").read_text()
    check("3-D tab: Solver box (auto / GPU / PyTorch CPU / SciPy) passed to MonoConfig",
          'f.addRow("Solver", self.solver)' in fsrc and "device=self.solver.currentData()" in fsrc
          and '"cuda"' in fsrc and '"torch-cpu"' in fsrc)
    check("gpu extra lists torch", '"torch>=' in (REPO / "pyproject.toml").read_text())

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kj_pose3d_gpu: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
