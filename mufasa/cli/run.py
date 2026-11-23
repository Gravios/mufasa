"""
mufasa.cli.run
==============

Patch 122ll: ``mufasa run <step>``, the 3-D pipeline from the command line, one step at a time
or all of it.

Each step is a ``mufasa-pose3d`` command under a pipeline name, with the same options
(``mufasa run <step> --help``):

=====================  ===============================================================  =========================
step                   what                                                             ``mufasa-pose3d``
=====================  ===============================================================  =========================
calibrate box          the camera from marked box edges                                 ``calibrate-box``
calibrate checkerboard the lens from a checkerboard                                     ``calibrate-checkerboard``
3d-inference           3-D pose from 2-D labels and the camera (Kalman-smoothed)        ``infer``
smooth                 3-D Kalman smoothing of a 3-D pose file                          ``smooth``
heights                labels within the body's height range, above the floor           ``heights``
gait-extrapolation     paws, steps and loads from the body's motion                     ``gait``
body-fit               a rigged body fitted to a top-down video                         ``body-fit``
dynamics               joint torques, ground and muscle forces (``--muscles``,          ``dynamics``
                       ``--optimise``)
animal                 the body model calibrated to an animal once (``--animal``)       ``animal``
=====================  ===============================================================  =========================

``mufasa run all`` chains 3d-inference → heights → gait-extrapolation → dynamics for one video
(or from a 3-D pose file onwards):

* the files are named from the video's stem, in one folder: ``<stem>_3d.parquet``,
  ``<stem>_3d_heights.parquet``, ``<stem>_gait.parquet`` (with ``<stem>_steps.csv``),
  ``<stem>_dynamics.parquet``;
* a step whose output is newer than its inputs is skipped (``--force`` reruns);
* ``--from`` / ``--to`` limit the steps;
* each step run is recorded in ``<stem>_run.json``: the command, mufasa's version, the time
  taken, the outcome.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

STEPS = {   # pipeline name -> (mufasa-pose3d command, what)
    "calibrate": (None, "the camera: `calibrate box` (marked box edges) or `calibrate checkerboard` (lens)"),
    "3d-inference": ("infer", "3-D pose from 2-D labels and the camera, Kalman-smoothed"),
    "smooth": ("smooth", "3-D Kalman smoothing of a 3-D pose file"),
    "heights": ("heights", "labels within the body's height range and above the floor"),
    "gait-extrapolation": ("gait", "paws, steps and loads from the body's motion"),
    "body-fit": ("body-fit", "a rigged body fitted to a top-down video"),
    "dynamics": ("dynamics", "joint torques, ground forces, muscle forces (--muscles), re-solved motion (--optimise)"),
    "animal": ("animal", "calibrate the body model to an animal once, from its sessions (--animal for the rest)"),
}
CALIBRATE = {"box": "calibrate-box", "checkerboard": "calibrate-checkerboard"}
CHAIN = ("3d-inference", "heights", "gait-extrapolation", "dynamics")


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("mufasa")
    except Exception:  # pragma: no cover - not installed
        return "source"


def _pose3d(argv: list[str]) -> int:
    from mufasa.tools import pose3d_cli
    try:
        return int(pose3d_cli.main(argv) or 0)
    except SystemExit as e:                     # argparse: --help, bad options
        return int(e.code or 0)


def list_steps() -> str:
    w = max(len(s) for s in STEPS) + 2
    lines = ["mufasa run <step> [options]   (mufasa run <step> --help: its options)", ""]
    for s, (cmd, what) in STEPS.items():
        lines.append(f"  {s:<{w}}{what}" + (f"   [mufasa-pose3d {cmd}]" if cmd else ""))
    lines += ["", f"  {'all':<{w}}{' → '.join(CHAIN)} for one video, skipping what is up to date"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# all


def _all_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mufasa run all", description=" → ".join(CHAIN) + " for one video.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pose", help="2-D labels of the video (starts at 3d-inference; needs --camera)")
    src.add_argument("--pose3d", help="a 3-D pose file (starts at heights)")
    ap.add_argument("--camera", help="camera file (calibrate box output)")
    ap.add_argument("--out-dir", help="folder for every file (default: beside the input)")
    ap.add_argument("--stem", help="the video's name in the files (default: from the input)")
    ap.add_argument("--from", dest="first", choices=CHAIN, help="first step")
    ap.add_argument("--to", dest="last", choices=CHAIN, help="last step")
    ap.add_argument("--force", action="store_true", help="rerun steps that are up to date")
    ap.add_argument("--dry-run", action="store_true", help="say what would run")
    ap.add_argument("--fps", type=float, help="frames per second (default: 30, or the files')")
    ap.add_argument("--mass-g", type=float, help="weighed body mass (gait, dynamics)")
    ap.add_argument("--muscles", help="dynamics: OpenSim hindlimb model (.osim) or saved muscles (.json)")
    ap.add_argument("--optimise", action="store_true", help="dynamics: re-solve the motion (stage 4)")
    ap.add_argument("--frames", help="dynamics: first:last frame numbers")
    ap.add_argument("--animal", help="dynamics: the animal file (mufasa run animal)")
    ap.add_argument("--infer-args", default="", help="more options for 3d-inference, quoted")
    ap.add_argument("--gait-args", default="", help="more options for gait-extrapolation, quoted")
    ap.add_argument("--dynamics-args", default="", help="more options for dynamics, quoted")
    return ap


def _stem(p: Path) -> str:
    s = p.stem
    for suf in ("_3d_heights", "_3d"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def plan(a) -> list[dict]:
    """The chain's steps for these arguments: name, inputs, output, the mufasa-pose3d command."""
    src = Path(a.pose or a.pose3d)
    stem = a.stem or _stem(src)
    d = Path(a.out_dir) if a.out_dir else src.parent
    f3d = Path(a.pose3d) if a.pose3d else d / f"{stem}_3d.parquet"
    fh = d / f"{stem}_3d_heights.parquet"
    fg = d / f"{stem}_gait.parquet"
    fd = d / f"{stem}_dynamics.parquet"
    fps = ["--fps", str(a.fps)] if a.fps else []
    mass = ["--mass-g", str(a.mass_g)] if a.mass_g else []
    steps = []
    if a.pose:
        if not a.camera:
            raise SystemExit("mufasa run all: --pose needs --camera")
        steps.append({"step": "3d-inference", "inputs": [Path(a.pose), Path(a.camera)], "output": f3d,
                      "argv": ["infer", "--pose", a.pose, "--camera", a.camera, "--out", str(f3d), *fps,
                               *shlex.split(a.infer_args)]})
    cam = ["--camera", a.camera] if a.camera else []
    steps.append({"step": "heights", "inputs": [f3d], "output": fh,
                  "argv": ["heights", str(f3d), "--out", str(fh), *cam]})
    steps.append({"step": "gait-extrapolation", "inputs": [fh], "output": fg,
                  "argv": ["gait", str(fh), "--out", str(fg), *fps, *mass, *shlex.split(a.gait_args)]})
    dyn = ["dynamics", str(fh), "--gait", str(fg), "--out", str(fd), *fps, *mass]
    inputs = [fh, fg]
    if a.muscles:
        dyn += ["--muscles", a.muscles]
        inputs.append(Path(a.muscles))
    if a.optimise:
        dyn.append("--optimise")
    if a.frames:
        dyn += ["--frames", a.frames]
    if a.animal:
        dyn += ["--animal", a.animal]
        inputs.append(Path(a.animal))
    steps.append({"step": "dynamics", "inputs": inputs, "output": fd, "argv": dyn + shlex.split(a.dynamics_args)})
    names = [s["step"] for s in steps]
    i0 = names.index(a.first) if a.first and a.first in names else 0
    i1 = names.index(a.last) + 1 if a.last and a.last in names else len(steps)
    for s in steps:
        s["record"] = d / f"{stem}_run.json"
    return steps[i0:i1]


def up_to_date(s: dict) -> bool:
    out = Path(s["output"])
    if not out.exists():
        return False
    t = out.stat().st_mtime
    return all(Path(p).exists() and Path(p).stat().st_mtime <= t for p in s["inputs"])


def _record(path: Path, entry: dict) -> None:
    try:
        runs = json.loads(path.read_text()) if path.exists() else []
    except Exception:
        runs = []
    runs.append(entry)
    path.write_text(json.dumps(runs, indent=1))


def run_all(argv: list[str]) -> int:
    a = _all_parser().parse_args(argv)
    steps = plan(a)
    for s in steps:
        cmd = "mufasa run " + s["step"] + " " + " ".join(shlex.quote(x) for x in s["argv"][1:])
        if not a.force and up_to_date(s):
            print(f"[{s['step']}] up to date: {s['output']}")
            continue
        missing = [str(p) for p in s["inputs"] if not Path(p).exists()]
        if a.dry_run:
            print(f"[{s['step']}] would run: {cmd}" + (f"  (waiting for {', '.join(missing)})" if missing else ""))
            continue
        if missing:
            print(f"[{s['step']}] missing input: {', '.join(missing)}", file=sys.stderr)
            return 2
        print(f"[{s['step']}] {cmd}", flush=True)
        Path(s["output"]).parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rc = _pose3d(s["argv"])
        _record(s["record"], {"step": s["step"], "command": cmd, "mufasa": _version(),
                              "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                              "seconds": round(time.time() - t0, 1), "exit": rc, "output": str(s["output"])})
        if rc != 0 or not Path(s["output"]).exists():
            print(f"[{s['step']}] failed (exit {rc})", file=sys.stderr)
            return rc or 1
    return 0


# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "--list", "list"):
        print(list_steps())
        return 0
    step, rest = argv[0], argv[1:]
    if step == "all":
        return run_all(rest)
    if step == "calibrate":
        if not rest or rest[0] not in CALIBRATE:
            print("mufasa run calibrate box|checkerboard [options]", file=sys.stderr)
            return 2
        return _pose3d([CALIBRATE[rest[0]], *rest[1:]])
    if step not in STEPS:
        print(f"mufasa run: no step {step!r}\n\n{list_steps()}", file=sys.stderr)
        return 2
    cmd = STEPS[step][0]
    if rest and rest[0] in ("-h", "--help"):
        print(f"mufasa run {step}  =  mufasa-pose3d {cmd}\n")
    return _pose3d([cmd, *rest])


__all__ = ["CHAIN", "STEPS", "list_steps", "main", "plan", "run_all", "up_to_date"]
