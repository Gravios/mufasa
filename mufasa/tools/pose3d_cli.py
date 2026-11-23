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
    mufasa-pose3d import-c3d TRIAL.c3d ... --project project.toml [--up y] [--rename a=b,c=d] [--edges a-b,b-c]
    mufasa-pose3d body-rig MODEL.blend OUT.rig.npz [--blender /path/to/blender]   # rig for body fitting
    mufasa-pose3d body-rig --virtual-rodent OUT.rig.npz [--girth 1.3]             # the built-in rat
    mufasa-pose3d heights X_3d.parquet --out X_3d_h.parquet [--scale 1.17] [--model none]  # heights, floor
    mufasa-pose3d dynamics X_3d.parquet --gait X_gait.parquet [--frames 30000:31000]       # joint torques
    mufasa-pose3d dynamics X_3d.parquet --gait X_gait.parquet --muscles rat_hindlimb.osim  # + muscle forces
    mufasa-pose3d dynamics X_3d.parquet --gait X_gait.parquet --muscles ... --optimise     # + re-solved motion
    mufasa-pose3d animal K7_a_3d.parquet K7_b_3d.parquet --out K7.animal.json             # calibrate once
    mufasa-pose3d gait X_3d.parquet --out X_gait.parquet [--mass-g 450] [--fps 30]  # paws, loads, gait
    mufasa-pose3d body-fit RIG.rig.npz VIDEO POSE --out FIT.parquet [--start 0 --stop N --step 1 --workers 4]
        [--no-refine] [--smoothness 0.5] [--scale PX] [--overlay OUT.mp4 [--side-by-side]]
    mufasa-pose3d render-animation PATH.camera.json --pose X_3d.parquet [--video X.mp4] --out X_anim.mp4 \
        [--size 1920 1080] [--fps 30]          # camera animation saved from the 3-D viewer

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
    b.add_argument("--wall-tilt", metavar="DEG|auto",
                   help="the walls flare outward this many degrees, or 'auto': fitted from the marked corner edges "
                        "(the box's limits then widen with height)")
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
    i.add_argument("--no-anatomy", action="store_true",
                   help="no anatomical limits (heights and spine bends from the virtual rodent)")
    i.add_argument("--anatomy-scale", type=float, help="rat size vs the virtual rodent (default: estimated)")
    i.add_argument("--fps", type=float, default=30.0)
    i.add_argument("--device", choices=["auto", "cuda", "torch-cpu", "cpu"], default="auto",
                   help="head/spine solver: cuda (PyTorch GPU), torch-cpu, cpu (SciPy); "
                        "auto = the first available")
    i.add_argument("--workers", type=int, default=None,
                   help="processes (default: all cores but one); several videos run at once, "
                        "a single video splits its chunks")
    ic = sub.add_parser("import-c3d", help="motion capture (.c3d) → 3-D pose files")
    ic.add_argument("files", nargs="+")
    ic.add_argument("--project", help="write to the project's sources/pose3d/")
    ic.add_argument("--out", help="output folder (instead of --project)")
    ic.add_argument("--up", choices=["z", "y", "x"], default="z", help="the file's up axis (becomes z)")
    ic.add_argument("--floor-z", type=float, help="the floor's height in the file (mm, z up), moved to 0")
    ic.add_argument("--rename", default="", help="mocap=project marker names, comma separated")
    ic.add_argument("--edges", default="", help="skeleton lines a-b,b-c (default: the rat skeleton "
                                                "for the markers present)")
    ic.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    help="frame range (0-based, end exclusive)")
    br = sub.add_parser("body-rig", help="rigged model (.blend, or the built-in virtual rodent) -> arrays "
                                         "for body fitting")
    br.add_argument("blend", help=".blend file (or, with --virtual-rodent, the output)")
    br.add_argument("out", nargs="?")
    br.add_argument("--virtual-rodent", action="store_true", help="the built-in rat (DeepMind's virtual rodent)")
    br.add_argument("--girth", type=float, default=1.0,
                    help="virtual rodent: widen the trunk (1: as modelled; e.g. 1.3 for a heavy rat)")
    br.add_argument("--armature", default="Armature")
    br.add_argument("--skin", default="Skin")
    br.add_argument("--blender", help="Blender executable (if bpy is not installed)")
    bf = sub.add_parser("body-fit", help="fit a rigged body model to a top-down video")
    bf.add_argument("rig")
    bf.add_argument("video")
    bf.add_argument("pose", help="2-D labels for the video")
    bf.add_argument("--out", required=True)
    bf.add_argument("--start", type=int, default=0)
    bf.add_argument("--stop", type=int)
    bf.add_argument("--step", type=int, default=1)
    bf.add_argument("--workers", type=int, default=1)
    bf.add_argument("--no-refine", action="store_true", help="forwards only (no backward pass)")
    bf.add_argument("--smoothness", type=float, default=0.5, help="pull towards neighbouring frames")
    bf.add_argument("--scale", type=float, help="px per model unit (default: estimated)")
    bf.add_argument("--overlay", help="also render the model over the video to this file")
    bf.add_argument("--side-by-side", action="store_true")
    bf.add_argument("--scale-from", metavar="VIDEO_3d.parquet",
                    help="the scale from a single-camera 3-D run of this video (camera and the animal's size)")
    bf.add_argument("--animal", choices=["auto", "dark", "light"], default="auto",
                    help="the animal darker or lighter than the background (silhouette)")
    ra = sub.add_parser("render-animation", help="render a camera path saved from the 3-D viewer")
    ra.add_argument("camera_path")
    ra.add_argument("--pose", required=True, help="3-D pose file")
    ra.add_argument("--video", help="the video (for the video in 3-D / below)")
    ra.add_argument("--out", required=True, help=".mp4 or .avi")
    ra.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), help="default: as saved")
    ra.add_argument("--fps", type=float, help="default: as saved")
    ga = sub.add_parser("gait", help="paws, weight distribution and gait from a 3-D pose file (virtual rodent physics)")
    ga.add_argument("pose3d")
    ga.add_argument("--out", required=True, help="<video>_gait.parquet (steps go to <video>_steps.csv)")
    ga.add_argument("--mass-g", type=float, help="weighed body mass (default: the model's, scaled by size³)")
    ga.add_argument("--fps", type=float, help="default: from the file, else 30")
    ga.add_argument("--size", type=float, help="the animal against the virtual rodent (default: nose to tail base)")
    ga.add_argument("--bounds", choices=["catwalk", "tactilearena"], default="catwalk",
                    help="gait bounds: CatWalk rats, speed-dependent (default), or TactileArena's fixed ones")
    ga.add_argument("--gait-scale", type=float, default=1.0,
                    help="the animal against an adult rat: lengths × this, times × its square root")
    gc = sub.add_parser("gpu-check", help="is the GPU solver usable here? (synthetic run, CPU vs CUDA)")
    gc.add_argument("--frames", type=int, default=20000)
    dp = sub.add_parser("dynamics", help="joint torques and ground forces: the virtual rodent in MuJoCo, fitted "
                                          "to the labels and the gait's paws (needs mufasa[biomech])")
    dp.add_argument("input")
    dp.add_argument("--gait", help="the video's _gait.parquet (paw targets, load split; without: paws planted)")
    dp.add_argument("--out", help="default: <video>_dynamics.parquet beside the input")
    dp.add_argument("--frames", help="first:last frame numbers (default: all)")
    dp.add_argument("--fps", type=float, help="default: the gait's, else the file's, else 30")
    dp.add_argument("--mass-g", type=float, help="the animal's weight (default: the model's, scaled to its size)")
    dp.add_argument("--calibrate", type=int, default=400, help="frames to calibrate the labels on (default 400)")
    dp.add_argument("--muscles", help="an OpenSim rat hindlimb model (.osim; e.g. rat_hindlimb_bilateral.osim) "
                                      "or a saved muscle set (.json): the hindlimb muscles' forces too")
    dp.add_argument("--save-muscles", help="save the muscles, carried onto the virtual rodent, to this .json")
    dp.add_argument("--optimise", action="store_true",
                    help="then re-solve the motion, paw forces and muscle activations together over windows "
                         "of frames, for tracking, balance and effort (stage 4; about a second a frame)")
    dp.add_argument("--iters", type=int, default=40, help="optimisation steps per window (default 40)")
    dp.add_argument("--animal", help="an animal file (mufasa-pose3d animal): its size, mass, label places; "
                                     "no calibration")
    dp.add_argument("--limits", choices=("model", "rat"), default="model",
                    help="the knees' and ankles' ranges: the virtual rodent's (default) or the rat's (narrower)")
    dp.add_argument("--architecture", choices=("osim", "eng2008"), default="osim",
                    help="muscle forces and fibre lengths: the OpenSim model's (default) or Eng et al. 2008's")
    hp = sub.add_parser("heights", help="labels within the body model's height range and above the floor "
                                         "(along the camera rays)")
    hp.add_argument("input")
    hp.add_argument("--out", required=True)
    hp.add_argument("--scale", type=float, help="the animal against the model (default: from the file's report, "
                                                "else nose to tail base)")
    hp.add_argument("--model", default="virtual_rodent", help="'virtual_rodent', or 'none' (the floor only)")
    hp.add_argument("--camera", help="camera file (default: the file's own camera)")
    hp.add_argument("--margin", type=float, default=0.1, help="slack on the model's ranges (0.1 = 10 %%)")
    s = sub.add_parser("smooth", help="3-D Kalman smoothing of a 3-D pose file (--model: with the body)")
    s.add_argument("input")
    s.add_argument("--out", required=True)
    s.add_argument("--camera")
    s.add_argument("--sigma-a", type=float, default=3000.0)
    s.add_argument("--sigma-perp", type=float, default=2.0)
    s.add_argument("--sigma-ray", type=float, default=25.0)
    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--model", action="store_true",
                   help="smooth with the body instead (the virtual rodent: bones keep their lengths, joints stay in "
                        "range, gaps filled, labels that disagree with the body flagged; needs mufasa[biomech])")
    s.add_argument("--calibrate", type=int, default=400, help="--model: frames to calibrate the labels on")
    s.add_argument("--label-sigma", type=float, default=8.0, help="--model: the labels' noise, mm (default 8)")
    s.add_argument("--joints-out", help="--model: also write the joint angles to this .parquet")
    s.add_argument("--animal", help="--model: an animal file (mufasa-pose3d animal): its body and labels, no "
                                    "calibration")
    s.add_argument("--limits", choices=("model", "rat"), default="model",
                   help="--model: the knees' and ankles' ranges, the virtual rodent's (default) or the rat's")
    an = sub.add_parser("animal", help="calibrate the virtual rodent to an animal once, from one or more of its "
                                       "3-D pose files (size, label places); dynamics and smooth --model take it")
    an.add_argument("files", nargs="+", help="the animal's 3-D pose files")
    an.add_argument("--out", required=True, help="<animal>.animal.json")
    an.add_argument("--frames", type=int, default=600, help="frames to calibrate on, shared out over the files")
    an.add_argument("--mass-g", type=float, help="the animal's weighed mass")
    an.add_argument("--name", help="the animal's name (default: from --out)")
    an.add_argument("--size", type=float, help="the animal against the virtual rodent (default: nose to tail base)")
    an.add_argument("--limits", choices=("model", "rat"), default="model",
                    help="the knees' and ankles' ranges: the virtual rodent's (default) or the rat's (narrower)")
    return ap


def settings_from(a):
    from mufasa.pose3d.kalman3d import Kalman3DConfig
    from mufasa.pose3d.mono import MonoConfig
    from mufasa.pose3d.pipeline import RunSettings, default_workers
    mono = MonoConfig(head_width_mm=a.head_width, body_scale=a.body_scale, fps=a.fps,
                      wall_tolerance_mm=a.wall_tolerance, device=a.device,
                      use_trunk_prior=not a.no_trunk_prior, spine_chain=not a.no_spine_chain,
                      anatomy=None if a.no_anatomy else "virtual_rodent", anatomy_scale=a.anatomy_scale)
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
            K=K, dist=dist, trust=a.trust,
            wall_tilt_deg=None if a.wall_tilt is None else ("auto" if a.wall_tilt == "auto" else float(a.wall_tilt)))
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
    if a.cmd == "import-c3d":
        from mufasa.pose3d.c3d import import_c3d, read_c3d
        if a.project:
            from mufasa.project_layout import resolve_v1_project_root
            root = resolve_v1_project_root(str(a.project))
            if root is None:
                print("not a v1 Mufasa project: " + a.project, file=sys.stderr)
                return 2
            out_dir = Path(root) / "sources" / "pose3d"
        elif a.out:
            out_dir = Path(a.out)
        else:
            print("give --project or --out", file=sys.stderr)
            return 2
        from mufasa.pose3d.c3d import parse_edges, parse_rename
        rename, edges = parse_rename(a.rename), parse_edges(a.edges)
        for f in a.files:
            c = read_c3d(f)
            out = import_c3d(f, out_dir, up=a.up, rename=rename, edges=edges,
                             frames=tuple(a.frames) if a.frames else None, floor_z=a.floor_z)
            print(f"{Path(f).name}: {len(c.labels)} markers, {len(c.points)} frames at {c.rate:g} Hz "
                  f"({c.units}) → {out}")
        return 0
    if a.cmd == "body-rig" and a.virtual_rodent:
        from mufasa.bodymodel.rodent_rig import export_rodent_rig
        export_rodent_rig(a.out or a.blend, girth=a.girth)
        return 0
    if a.cmd == "body-rig":
        import subprocess
        if not a.out:
            print("body-rig: give the .blend file and the output")
            return 2
        script = Path(__file__).resolve().parents[1] / "bodymodel" / "_blend_export.py"
        args = [a.blend, a.out, a.armature, a.skin]
        cmd = ([a.blender, "-b", "--python", str(script), "--", *args] if a.blender
               else [sys.executable, str(script), *args])
        return subprocess.run(cmd).returncode
    if a.cmd == "body-fit":
        from mufasa.bodymodel.fit import FitConfig, MaskConfig, RunConfig, fit_video
        cfg = FitConfig(scale=a.scale, refine=not a.no_refine, temporal=0.0 if a.no_refine else a.smoothness)
        out = fit_video(a.rig, a.video, a.pose, RunConfig(start=a.start, stop=a.stop, step=a.step,
                                                         workers=a.workers, fit=cfg,
                                                         mask=MaskConfig(polarity=a.animal),
                                                         scale_from=a.scale_from), a.out)
        if a.overlay:
            from mufasa.bodymodel.render import OverlayStyle, render_overlay_video
            render_overlay_video(out, a.overlay, style=OverlayStyle(side_by_side=a.side_by_side))
        return 0
    if a.cmd == "gait":
        from mufasa.pose3d.gait import bounds_preset, gait_from_pose3d, write_gait
        res = gait_from_pose3d(a.pose3d, fps=a.fps, mass_g=a.mass_g, size=a.size,
                               bounds=bounds_preset(a.bounds, a.gait_scale))
        g, s = write_gait(res, a.out, {"pose3d": str(a.pose3d)})
        sm = res.summary
        for w in sm.get("contract", []):
            print(f"warning: {w}", file=sys.stderr)
        print(f"{sm['steps']} stances · size {sm['size_vs_model']}x · mass {sm['mass_g']} g · "
              f"load share {sm['load_share']} -> {g}, {s}")
        return 0
    if a.cmd == "render-animation":
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication([])
        from mufasa.pose3d.animation import CameraPath, render_animation
        cp, st = CameraPath.from_json(Path(a.camera_path).read_text())
        total = cp.n_output()
        step = max(total // 20, 1)

        def prog(done, n):
            if done % step == 0 or done == n:
                print(f"  {done}/{n} frames", flush=True)
        out = render_animation(a.pose, cp, a.out, video_path=a.video,
                               size=tuple(a.size or st.get("size", (1280, 720))),
                               fps=a.fps or st.get("fps", 30.0), style=st.get("style"),
                               models=st.get("models"),
                               video_below=bool(st.get("video_below")) and bool(a.video),
                               frame_label=bool(st.get("frame_label")), progress=prog)
        print(f"wrote {out}")
        return 0
    if a.cmd == "gpu-check":
        from mufasa.pose3d import gpu
        hint = gpu.install_hint()                       # 122ld: what is missing, and how to fix it
        if not gpu.available():
            print(hint)
            return 1
        import torch
        print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}"
              + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
        if hint:
            print(hint)
        out = gpu.gpu_check(a.frames)
        return 0 if ("cuda" not in out or out.get("max_difference", 1) < 1e-3) else 1
    if a.cmd == "dynamics":
        from mufasa.biomech.pipeline import run_dynamics
        fr = None
        if a.frames:
            lo, hi = a.frames.split(":")
            fr = (int(lo), int(hi))
        mus = None
        if a.muscles:
            from mufasa.biomech.osim import load_muscles
            mus = load_muscles(a.muscles, a.architecture)
            if a.save_muscles:
                print(f"wrote {mus.to_json(a.save_muscles)}")
        out, s = run_dynamics(a.input, a.gait, a.out, frames=fr, fps=a.fps, mass_g=a.mass_g,
                              calibrate_frames=a.calibrate, muscles=mus, optimise=a.optimise, iters=a.iters,
                              limits=a.limits, animal=a.animal,
                              progress=lambda m: print(m, flush=True))
        print(f"wrote {out}")
        for k in ("mass_g", "frames", "fit_rms_mm", "support_vs_weight", "residual_vs_weight_median",
                  "torque_p90_1", "torque_p90_2", "torque_p90_3", "reserve_vs_torque_median",
                  "activation_median_1", "activation_median_2", "activation_median_3",
                  "activation_change_per_frame"):
            print(f"  {k}: {s.get(k)}")
        b = s.get("before_optimising")
        if b:
            print(f"  before optimising: fit {b.get('fit_rms_mm')} mm, residual {b.get('residual_vs_weight_median')} "
                  f"× weight, reserve {b.get('reserve_vs_torque_median')} × torque")
        return 0
    if a.cmd == "heights":
        from mufasa.pose3d.io import marker_names, read_pose3d, write_pose3d
        from mufasa.pose3d.mono import camera_center_from_meta, enforce_heights
        df, meta = read_pose3d(a.input)
        names = marker_names(df)
        if a.camera:
            from mufasa.pose3d.camera import Camera
            center = Camera.load(a.camera).center
        else:
            center = camera_center_from_meta(meta)
        anat = (meta.get("report") or {}).get("anatomy") or {}
        scale = a.scale or anat.get("scale")
        if not scale:
            from mufasa.pose3d.gait import BodyPhysics, animal_size
            xyz = np.stack([df[[f"{n}_x", f"{n}_y", f"{n}_z"]].to_numpy(float) for n in names], 1)
            scale = animal_size(BodyPhysics.virtual_rodent(), names, xyz)
        model = None if a.model == "none" else a.model
        from mufasa.pose3d.pipeline import _label_map
        fixed = enforce_heights(df, names, center=center, scale=float(scale), model=model, margin=a.margin,
                                label_map=_label_map(names, anat))
        meta.setdefault("report", {})["heights_fixed"] = fixed
        meta["method"] = (meta.get("method", "") + "; heights within the body's range, above the floor").lstrip("; ")
        write_pose3d(df, a.out, meta)
        n = max(len(df), 1)
        moved = {m: v for m, v in fixed.items() if v}
        print(f"scale {float(scale):.2f}, {'along the camera rays' if center is not None else 'straight up/down'}; "
              + (", ".join(f"{m} {v} ({100 * v / n:.1f} %)" for m, v in moved.items()) or "nothing moved")
              + f"; wrote {a.out}")
        return 0
    if a.cmd == "animal":
        from mufasa.biomech.animal import calibrate_animal
        d = calibrate_animal(a.files, a.out, frames=a.frames, mass_g=a.mass_g, name=a.name, limits=a.limits, size=a.size,
                             progress=lambda msg: print(msg, flush=True))
        print(f"{d['name']}: size {d['size']} (sessions {d['session_sizes']}), {len(d['labels'])} labels, "
              f"fit {d['calibration_rms_mm'][0]} → {min(d['calibration_rms_mm'])} mm; wrote {a.out}")
        return 0
    if a.cmd == "smooth" and a.model:
        from mufasa.biomech.smoother import smooth_pose3d
        s = smooth_pose3d(a.input, a.out, fps=a.fps, calibrate_frames=a.calibrate, joints_out=a.joints_out,
                          label_sigma=a.label_sigma, limits=a.limits, animal=a.animal,
                          progress=lambda msg: print(msg, flush=True))
        print(f"labels {s['model_mm_median']} mm from the body (median), {100 * s['flagged']:.2f} % flagged, "
              f"{s['filled']} gaps filled; wrote {a.out}" + (f" and {a.joints_out}" if a.joints_out else ""))
        return 0
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
