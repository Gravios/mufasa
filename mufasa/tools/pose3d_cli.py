"""
``mufasa-pose3d`` — 3-D pose from the command line (patch 122kf).

    mufasa-pose3d calibrate-box MARKS.json --width 300 [--length 554] --camera-height 430 \
        [--wall-height 300] [--intrinsics lens.toml] --out sources/calibration/camera_K7.toml
    mufasa-pose3d calibrate-checkerboard VIDEO_OR_DIR --pattern 9 6 --square-mm 20 --out lens.toml
    mufasa-pose3d infer --project project.toml [--video STEM ...] [--no-kalman] [--stereo camA camB] [--workers N]
        [--device auto|cuda|torch-cpu|cpu]
    mufasa-pose3d infer --pose labels.parquet --camera camera.toml [--fallback smoothed.parquet] --out out_3d.parquet
    mufasa-pose3d smooth IN_3d.parquet --out OUT_3d.parquet [--camera camera.toml] [--sigma-a 3000]
    mufasa-pose3d gpu-check [--frames 20000]      # GPU solver: synthetic run, CPU vs CUDA

MARKS.json (pixels of one frame of the empty box)::

    {"image_size": [640, 360],
     "floor_corners": [[x, y], [x, y], [x, y], [x, y]],
     "vertical_edges": [[[x, y], [x, y], ...], ...]}       # optional, ≥2 edges

The workbench "3-D" tab writes the same file from clicks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _intrinsics_save(path, K, dist, rms, n, size) -> None:
    from mufasa.pose3d.camera import _to_toml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(_to_toml({"intrinsics": {
        "K": np.asarray(K).tolist(), "dist": np.asarray(dist).ravel().tolist(),
        "rms_px": float(rms), "views": int(n), "image_size": list(size)}}))


def intrinsics_load(path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as fh:
        d = tomllib.load(fh)
    d = d.get("intrinsics") or d.get("camera")
    return np.array(d["K"], float), np.array(d.get("dist", [0] * 5), float)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mufasa-pose3d", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("calibrate-box", help="camera from marked box edges")
    b.add_argument("marks")
    b.add_argument("--length", type=float)
    b.add_argument("--width", type=float)
    b.add_argument("--wall-height", type=float)
    b.add_argument("--camera-height", type=float)
    b.add_argument("--intrinsics", help="lens calibration (calibrate-checkerboard output)")
    b.add_argument("--trust", choices=["auto", "width", "length"], default="auto")
    b.add_argument("--out", required=True)
    c = sub.add_parser("calibrate-checkerboard", help="lens calibration")
    c.add_argument("source")
    c.add_argument("--pattern", type=int, nargs=2, required=True, metavar=("COLS", "ROWS"))
    c.add_argument("--square-mm", type=float, required=True)
    c.add_argument("--skip", type=int, default=8)
    c.add_argument("--out", required=True)
    i = sub.add_parser("infer", help="3-D pose")
    i.add_argument("--project")
    i.add_argument("--video", nargs="*", help="only these video stems")
    i.add_argument("--pose")
    i.add_argument("--fallback")
    i.add_argument("--camera")
    i.add_argument("--out")
    i.add_argument("--no-fallback", action="store_true", help="project: do not fill from Kalman v2")
    i.add_argument("--stereo", nargs=2, metavar=("TOKEN0", "TOKEN1"))
    i.add_argument("--no-kalman", action="store_true")
    i.add_argument("--sigma-a", type=float, default=3000.0)
    i.add_argument("--head-width", type=float, default=36.0)
    i.add_argument("--body-scale", type=float, default=1.0)
    i.add_argument("--wall-tolerance", type=float, default=20.0,
                   help="mm a label may sit beyond a wall before it pushes back")
    i.add_argument("--no-trunk-prior", action="store_true")
    i.add_argument("--no-spine-chain", action="store_true")
    i.add_argument("--fps", type=float, default=30.0)
    i.add_argument("--device", choices=["auto", "cuda", "torch-cpu", "cpu"], default="auto",
                   help="head/spine solver: cuda (PyTorch GPU), torch-cpu, cpu (SciPy); "
                        "auto = the first available")
    i.add_argument("--workers", type=int, default=None,
                   help="processes (default: all cores but one); several videos run at once, "
                        "a single video splits its chunks")
    gc = sub.add_parser("gpu-check", help="is the GPU solver usable here? (synthetic run, CPU vs CUDA)")
    gc.add_argument("--frames", type=int, default=20000)
    s = sub.add_parser("smooth", help="3-D Kalman smoothing of a 3-D pose file")
    s.add_argument("input")
    s.add_argument("--out", required=True)
    s.add_argument("--camera")
    s.add_argument("--sigma-a", type=float, default=3000.0)
    s.add_argument("--sigma-perp", type=float, default=2.0)
    s.add_argument("--sigma-ray", type=float, default=25.0)
    s.add_argument("--fps", type=float, default=30.0)
    return ap


def settings_from(a):
    from mufasa.pose3d.kalman3d import Kalman3DConfig
    from mufasa.pose3d.mono import MonoConfig
    from mufasa.pose3d.pipeline import RunSettings, default_workers
    mono = MonoConfig(head_width_mm=a.head_width, body_scale=a.body_scale, fps=a.fps,
                      wall_tolerance_mm=a.wall_tolerance, device=a.device,
                      use_trunk_prior=not a.no_trunk_prior, spine_chain=not a.no_spine_chain)
    kal = None if a.no_kalman else Kalman3DConfig(fps=a.fps, sigma_a=a.sigma_a)
    return RunSettings(mode="stereo" if a.stereo else "mono", mono=mono, kalman=kal,
                       workers=a.workers if a.workers else default_workers())


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "calibrate-box":
        from mufasa.pose3d.camera import calibrate_from_box_marks
        m = json.loads(Path(a.marks).read_text())
        K = dist = None
        if a.intrinsics:
            K, dist = intrinsics_load(a.intrinsics)
        cam, rep = calibrate_from_box_marks(
            m["floor_corners"], floor_length_mm=a.length, floor_width_mm=a.width,
            wall_height_mm=a.wall_height, vertical_edges_px=m.get("vertical_edges"),
            camera_height_mm=a.camera_height, image_size=tuple(m.get("image_size", (0, 0))),
            K=K, dist=dist, trust=a.trust)
        cam.notes += "\n" + rep.text()
        cam.save(a.out)
        print(rep.text())
        print(f"saved {a.out}")
        return 0
    if a.cmd == "calibrate-checkerboard":
        from mufasa.pose3d.camera import calibrate_checkerboard
        K, dist, rms, n, size = calibrate_checkerboard(a.source, tuple(a.pattern), a.square_mm,
                                                       skip=a.skip, progress=print)
        _intrinsics_save(a.out, K, dist, rms, n, size)
        print(f"lens calibrated from {n} views, rms {rms:.2f} px; saved {a.out}")
        return 0
    if a.cmd == "infer":
        from mufasa.pose3d.pipeline import Job, run_job, run_project
        st = settings_from(a)
        if a.project:
            out, reps, problems = run_project(a.project, st, stems=a.video or None,
                                              use_fallback=not a.no_fallback,
                                              stereo_tokens=tuple(a.stereo) if a.stereo else None,
                                              progress=print)
            for p in problems:
                print("skipped:", p)
            print(f"wrote {len(reps)} file(s) to {out}")
            return 0
        if not (a.pose and a.camera and a.out):
            print("give --project, or --pose, --camera and --out", file=sys.stderr)
            return 2
        job = Job(Path(a.pose).stem, Path(a.pose), Path(a.camera), Path(a.fallback) if a.fallback else None)
        out = Path(a.out)
        rep = run_job(job, st, out.parent, progress=print)
        produced = Path(rep["output"])
        if produced != out:
            produced.replace(out)
        print(f"wrote {out}")
        return 0
    if a.cmd == "gpu-check":
        from mufasa.pose3d import gpu
        if not gpu.available():
            print("PyTorch is not installed: pip install torch (a CUDA build for the GPU)")
            return 1
        import torch
        print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}"
              + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
        out = gpu.gpu_check(a.frames)
        return 0 if ("cuda" not in out or out.get("max_difference", 1) < 1e-3) else 1
    if a.cmd == "smooth":
        from mufasa.pose3d.camera import Camera
        from mufasa.pose3d.io import marker_names, read_pose3d, write_pose3d
        from mufasa.pose3d.kalman3d import Kalman3DConfig, smooth_pose3d_frame
        df, meta = read_pose3d(a.input)
        center = Camera.load(a.camera).center if a.camera else None
        cfg = Kalman3DConfig(fps=a.fps, sigma_a=a.sigma_a, sigma_perp=a.sigma_perp, sigma_ray=a.sigma_ray)
        out, info = smooth_pose3d_frame(df, marker_names(df), cfg, center)
        meta["method"] = (meta.get("method", "") + f"; 3-D Kalman (RTS) smoothing σa={a.sigma_a:g}").lstrip("; ")
        write_pose3d(out, a.out, meta)
        print(f"moved points by {info['moved_mm_median']:.1f} mm median; wrote {a.out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
