"""
mufasa.cli.commands
===================

Patch 122lv: ``mufasa help``, ``mufasa list …``, ``mufasa info``, ``mufasa project``,
``mufasa doctor``, ``mufasa version`` — the command line beside the workbench (no Qt needed).

=========================  ==========================================================================
command                    what
=========================  ==========================================================================
``mufasa``                 open the workbench (``--project``, ``--no-auto-discover``, ``--no-recent``)
``mufasa help [command]``  this overview, or a command's options
``mufasa list <what>``     ``models``, ``videos``, ``pose``, ``3d``, ``runs``, ``calibrations``,
                           ``steps``, ``tools``, ``projects``
``mufasa info <file>``     what a file holds: a video, a pose / 3-D / gait file, a model, a camera…
``mufasa project [path]``  a project's summary: animals, body parts, behaviours, what it holds
``mufasa doctor``          the environment: Python, packages, GPUs, FFmpeg, the display
``mufasa version``         mufasa's version
``mufasa run <step>``      the 3-D pipeline, a step at a time or all of it (:mod:`mufasa.cli.run`)
=========================  ==========================================================================

The project is ``--project`` (its ``project.toml`` or its folder), else the one the current folder
is in, else the last one opened in the workbench. ``--json`` gives the lists as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["COMMANDS", "LIST_SUBJECTS", "collect_models", "main", "resolve_project"]

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SKIP_DIRS = {"frames", ".git", "__pycache__", "logs", ".ipynb_checkpoints"}


# --------------------------------------------------------------------------- #
# the project
# --------------------------------------------------------------------------- #
def resolve_project(arg: str | None) -> tuple[Path | None, str]:
    """The project's config and how it was found: ``--project`` (a ``project.toml``, an old-layout
    ``.ini`` config, or a folder in a project), else the current folder's project, else the last
    one opened in the workbench."""
    from mufasa.project_layout import PROJECT_CONFIG_FILENAME, find_project_config
    if arg:
        p = Path(arg).expanduser()
        if p.is_file() and p.suffix in (".toml", ".ini"):
            return p.resolve(), "given"
        if p.is_dir() and (p / PROJECT_CONFIG_FILENAME).is_file():
            return (p / PROJECT_CONFIG_FILENAME).resolve(), "given"
        found = find_project_config(p) if p.exists() else None
        return (found, "given") if found else (None, f"no project at {p}")
    found = find_project_config(Path.cwd())
    if found:
        return found, "the current folder's"
    try:
        from mufasa.ui_qt.recent_project import load_recent_project
        recent = load_recent_project()
    except Exception:
        recent = None
    if recent:
        return Path(recent), "the last one opened"
    return None, "no project: pass --project, or run this inside a project's folder"


def _root(cfg: Path) -> Path:
    from mufasa.project_layout import project_paths_from_config
    try:
        return Path(project_paths_from_config(str(cfg))["project_path"])
    except Exception:
        return cfg.parent


def _rel(p: Path, root: Path | None) -> str:
    try:
        return str(Path(p).relative_to(root)) if root else str(p)
    except ValueError:
        return str(p)


def _size(n: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _when(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def _walk(root: Path, suffixes: tuple[str, ...]):
    """Files under ``root`` ending in ``suffixes``, skipping extracted frames and the like."""
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        for f in sorted(files):
            if f.lower().endswith(suffixes):
                yield Path(d) / f


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _npz_info(p: Path) -> str:
    import numpy as np
    try:
        with np.load(p, allow_pickle=False) as z:
            keys = set(z.files)
            if "version" in keys and "layout_segments" in keys:
                fps = float(z["fps"]) if "fps" in keys else None
                return f"Kalman body model {z['version']}" + (f", fitted at {fps:g} fps" if fps else "")
            if "targets" in keys and "features" in keys:
                return f"height prior: {len(z['features'])} frames" + (
                    f", {len(set(z['subject'].tolist()))} animals" if "subject" in keys else "")
            return f"{len(keys)} arrays"
    except Exception as e:
        return f"unreadable ({type(e).__name__})"


def _card(p: Path) -> dict:
    card = p.parent / "card.toml"
    if not card.is_file():
        return {}
    try:
        import tomllib
        return tomllib.loads(card.read_text())
    except Exception:
        return {}


def collect_models(cfg: Path | None, include_global: bool = True) -> list[dict]:
    """Every model the project can use: Kalman body models for smoothing (the project's and the
    global library's), behaviour classifiers, detectors, the 3-D body models (built in, rigs,
    calibrated animals, muscle sets) and the height priors."""
    out: list[dict] = []
    root = _root(cfg) if cfg else None

    def add(kind, name, path, details="", where="project"):
        path = Path(path)
        st = path.stat() if path.exists() else None
        out.append({"kind": kind, "name": name, "path": str(path), "where": where, "details": details,
                    "size": st.st_size if st and path.is_file() else None,
                    "modified": _when(st.st_mtime) if st else ""})

    if root is not None:
        from mufasa.project_layout import project_paths_from_config
        try:
            mdir = Path(project_paths_from_config(str(cfg))["models_dir"])
        except Exception:
            mdir = root / "models"
        if mdir.is_dir():
            for p in sorted(mdir.glob("*/model.npz")):
                c = _card(p)
                extra = f"; from {Path(c['source_path']).name}" if c.get("source_path") else ""
                add("smoothing", p.parent.name, p, _npz_info(p) + extra)
            for p in _walk(mdir, (".sav", ".onnx", ".pkl", ".joblib")):
                add("classifier", p.stem, p, p.suffix[1:])
            for p in _walk(mdir, (".pt", ".engine")):
                add("detector", p.stem, p, "YOLO " + p.suffix[1:])
        for p in _walk(root, (".rig.npz", ".animal.json", ".muscles.json")):
            n = p.name
            if n.endswith(".rig.npz"):
                add("body", n.removesuffix(".rig.npz"), p, "rigged body (Body model)")
            elif n.endswith(".animal.json"):
                try:
                    d = json.loads(p.read_text())
                    det = f"calibrated animal: {d.get('name', '')} {d.get('mass_g', '') or ''} g".strip()
                except Exception:
                    det = "calibrated animal"
                add("body", n.removesuffix(".animal.json"), p, det)
            else:
                try:
                    det = f"{len(json.loads(p.read_text()).get('muscles', []))} muscles"
                except Exception:
                    det = "muscle set"
                add("body", n.removesuffix(".muscles.json"), p, det)
    if include_global:
        try:
            from mufasa.project_layout import global_model_cache_dir
            for p in sorted(global_model_cache_dir().glob("*.npz")):
                add("smoothing", p.stem, p, _npz_info(p), where="global")
        except Exception:
            pass
    vr = ASSETS / "models" / "pose3d" / "virtual_rodent"
    if vr.is_dir():
        add("body", "virtual_rodent", vr / "rodent.xml", "the virtual rodent (MuJoCo), built in", where="built in")
    for p in sorted((ASSETS / "models" / "pose3d").glob("*.npz")):
        add("prior", p.stem, p, _npz_info(p), where="built in")
    return out


KIND_TITLES = {"smoothing": "Kalman body models (smoothing)", "classifier": "Behaviour classifiers",
               "detector": "Detectors (YOLO)", "body": "3-D body models", "prior": "Height priors"}


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def _table(rows: list[list[str]], head: list[str]) -> str:
    if not rows:
        return "  (none)"
    w = [max(len(str(x)) for x in col) for col in zip(head, *rows, strict=False)]
    line = lambda r: "  " + "  ".join(str(x).ljust(n) for x, n in zip(r, w, strict=False)).rstrip()  # noqa: E731
    return "\n".join([line(head), line(["-" * n for n in w])] + [line(r) for r in rows])


def _emit(args, items: list[dict], render) -> int:
    if getattr(args, "json", False):
        print(json.dumps(items, indent=2, default=str))
    else:
        print(render(items))
    return 0


def _project_line(cfg: Path | None, how: str) -> str:
    return f"Project: {_root(cfg)} ({how})" if cfg else f"Project: {how}"


# --------------------------------------------------------------------------- #
# mufasa list
# --------------------------------------------------------------------------- #
def _list_models(args, cfg, how) -> int:
    items = collect_models(cfg, include_global=not args.project_only)
    if args.kind:
        items = [m for m in items if m["kind"] == args.kind]
    root = _root(cfg) if cfg else None

    def render(items):
        parts = [_project_line(cfg, how)]
        for kind, title in KIND_TITLES.items():
            group = [m for m in items if m["kind"] == kind]
            if not group and args.kind != kind:
                continue
            parts.append(f"\n{title}:")
            parts.append(_table([[m["name"], m["where"], m["details"], m["modified"], _shown(m, root)]
                                 for m in group], ["name", "where", "what", "modified", "path"]))
        return "\n".join(parts)
    return _emit(args, items, render)


def _shown(m: dict, root: Path | None) -> str:
    """A model's path as listed: in the project relative to it, built-in ones in mufasa's
    package, the global library's from ``~``."""
    p = Path(m["path"])
    if m["where"] == "built in":
        return _rel(p, ASSETS.parents[1])
    if m["where"] == "global":
        return "~/" + _rel(p, Path.home()) if str(p).startswith(str(Path.home())) else str(p)
    return _rel(p, root)


def _need(cfg, how) -> bool:
    if cfg is None:
        print(f"mufasa: {how}", file=sys.stderr)
        return False
    return True


def _video_details(p: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(p))
    try:
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {"frames": n, "fps": round(fps, 3), "size": f"{w}x{h}",
            "duration": time.strftime("%H:%M:%S", time.gmtime(n / fps)) if fps else ""}


def _list_videos(args, cfg, how) -> int:
    if not _need(cfg, how):
        return 2
    from mufasa.pose3d.files import project_video_files
    root = _root(cfg)
    items = []
    for v in project_video_files(str(cfg), root):
        st = v.stat()
        it = {"name": v.name, "path": str(v), "bytes": st.st_size, "modified": _when(st.st_mtime)}
        if args.details:
            d = _video_details(v)
            it.update(frames=d["frames"], fps=d["fps"], px=d["size"], length=d["duration"])
        items.append(it)
    head = ["video", "size", "modified"] + (["frames", "fps", "pixels", "length"] if args.details else [])

    def row(i):
        r = [_rel(Path(i["path"]), root), _size(i["bytes"]), i["modified"]]
        return r + ([i["frames"], i["fps"], i["px"], i["length"]] if args.details else [])
    return _emit(args, items, lambda items: _project_line(cfg, how) + f"\n\n{len(items)} video(s):\n"
                 + _table([row(i) for i in items], head))


def _rows_of(p: Path) -> int | None:
    try:
        if p.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return None
    return None


def _list_pose(args, cfg, how) -> int:
    if not _need(cfg, how):
        return 2
    root = _root(cfg)
    d = root / "sources" / "pose"
    files = [p for p in _walk(d, (".csv", ".parquet", ".h5", ".slp", ".json"))] if d.is_dir() else []
    items = [{"name": p.name, "path": str(p), "rows": _rows_of(p), "modified": _when(p.stat().st_mtime)}
             for p in files if "_3d" not in p.name]
    return _emit(args, items, lambda items: _project_line(cfg, how) + f"\n\n{len(items)} 2-D pose file(s):\n"
                 + _table([[_rel(Path(i["path"]), root), i["rows"] if i["rows"] is not None else "", i["modified"]]
                           for i in items], ["file", "frames", "modified"]))


def _list_3d(args, cfg, how) -> int:
    if not _need(cfg, how):
        return 2
    from mufasa.pose3d.files import find_video, pose3d_files
    root = _root(cfg)
    items = []
    for p in pose3d_files(root):
        v = find_video(p, str(cfg))
        items.append({"name": p.name, "path": str(p), "frames": _rows_of(p), "video": str(v) if v else None,
                      "modified": _when(p.stat().st_mtime)})
    return _emit(args, items, lambda items: _project_line(cfg, how) + f"\n\n{len(items)} 3-D file(s), newest run "
                 "first:\n" + _table([[_rel(Path(i["path"]), root), i["frames"] or "",
                                       Path(i["video"]).name if i["video"] else "(none found)", i["modified"]]
                                      for i in items], ["file", "frames", "video", "modified"]))


def _list_runs(args, cfg, how) -> int:
    if not _need(cfg, how):
        return 2
    from mufasa.project_layout import ProjectPaths, is_run_id
    root = _root(cfg)
    pp = ProjectPaths(root)
    items = []
    if pp.derived_dir.is_dir():
        for stage in sorted(x for x in pp.derived_dir.iterdir() if x.is_dir()):
            if args.stage and stage.name != args.stage:
                continue
            bases = [stage] + [f for f in sorted(stage.iterdir()) if f.is_dir() and not is_run_id(f.name)]
            for base in bases:
                for r in sorted((x for x in base.iterdir() if x.is_dir() and is_run_id(x.name)), key=lambda x: x.name):
                    n = sum(1 for _ in r.glob("*.parquet")) + sum(1 for _ in r.glob("*.csv"))
                    items.append({"stage": stage.name, "flavor": base.name if base is not stage else "",
                                  "run": r.name, "files": n, "path": str(r),
                                  "has_run_toml": (r / "run.toml").is_file()})
    if not args.stage and not args.json:
        by: dict = {}
        for i in items:
            k = (i["stage"], i["flavor"])
            by.setdefault(k, []).append(i)
        rows = [[s + (f"/{f}" if f else ""), len(v), v[-1]["run"], v[-1]["files"]] for (s, f), v in sorted(by.items())]
        print(_project_line(cfg, how) + "\n\nRuns by stage (`mufasa list runs --stage S` for one stage's):\n"
              + _table(rows, ["stage", "runs", "latest", "files in latest"]))
        return 0
    return _emit(args, items, lambda items: _project_line(cfg, how) + "\n\n" + _table(
        [[i["stage"] + (f"/{i['flavor']}" if i["flavor"] else ""), i["run"], i["files"], _rel(Path(i["path"]), root)]
         for i in items], ["stage", "run", "files", "path"]))


def _camera_text(p: Path) -> str:
    """A camera file in a line: image size, focal length, height above the floor, the box."""
    from mufasa.pose3d.camera import Camera
    c = Camera.load(p)
    w, h = c.image_size
    txt = f"{w}x{h} px, focal {float(c.K[0, 0]):.0f} px, {float(c.center[2]):.0f} mm above the floor"
    if c.box:
        L = c.box["x"][1] - c.box["x"][0]
        W = c.box["y"][1] - c.box["y"][0]
        txt += f", box {L:.0f} × {W:.0f} mm" + (f", walls flaring {c.box['tilt_deg']}°" if c.box.get("tilt_deg") else "")
    return txt


def _list_calibrations(args, cfg, how) -> int:
    if not _need(cfg, how):
        return 2
    root = _root(cfg)
    items = []
    for d in (root / "sources" / "calibration",):
        for p in _walk(d, (".toml",)) if d.is_dir() else []:
            det = ""
            try:
                det = _camera_text(p)
            except Exception:
                det = "(not a camera)"
            items.append({"name": p.stem, "path": str(p), "details": det, "modified": _when(p.stat().st_mtime)})
    return _emit(args, items, lambda items: _project_line(cfg, how) + f"\n\n{len(items)} camera(s):\n" + _table(
        [[i["name"], i["details"], i["modified"], _rel(Path(i["path"]), root)] for i in items],
        ["camera", "what", "modified", "path"]))


def _list_steps(args, cfg, how) -> int:
    from mufasa.cli.run import CALIBRATE, CHAIN, STEPS
    items = [{"step": k, "command": v[0] or ", ".join(f"calibrate {c} ({x})" for c, x in CALIBRATE.items()),
              "what": v[1], "in_all": k in CHAIN} for k, v in STEPS.items()]
    return _emit(args, items, lambda items: "Pipeline steps (`mufasa run <step> --help`; `mufasa run all` chains "
                 + " → ".join(CHAIN) + "):\n" + _table([[i["step"], i["what"]] for i in items], ["step", "what"]))


def _tools() -> list[dict]:
    out = []
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="console_scripts"):
            if ep.value.startswith("mufasa."):
                out.append({"command": ep.name, "entry": ep.value})
    except Exception:
        pass
    if not out:                                       # not installed: the source tree's list
        import re
        pp = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pp.is_file():
            for m in re.finditer(r'^(mufasa[\w-]*)\s*=\s*"([^"]+)"', pp.read_text(), re.M):
                out.append({"command": m.group(1), "entry": m.group(2)})
    for t in out:
        t["what"] = TOOL_WHAT.get(t["command"], "")
    return sorted({t["command"]: t for t in out}.values(), key=lambda t: t["command"])


TOOL_WHAT = {
    "mufasa": "the workbench; `mufasa help` for the commands",
    "mufasa-workbench": "the workbench, without the environment checks",
    "mufasa-preview": "pose drawn over a video, by paths or a folder search",
    "mufasa-landscape": "joint PDFs (dipsplit) and t-SNE of pose and features",
    "mufasa-landscape-animate": "GIF/MP4 of frames moving between t-SNE and feature pairs",
    "mufasa-view3d": "the 3-D viewer: markers in the box, the video, features, JPDF",
    "mufasa-relink": "relative file links in a project (sshfs-safe)",
    "mufasa-pose3d": "3-D calibration, inference, smoothing, gait, dynamics (`mufasa run` names them)",
}


def _list_tools(args, cfg, how) -> int:
    items = _tools()
    return _emit(args, items, lambda items: "Command-line tools (each takes --help):\n"
                 + _table([[i["command"], i["what"]] for i in items], ["command", "what"]))


def _list_projects(args, cfg, how) -> int:
    paths = []
    try:
        from mufasa.ui_qt.linux_env import recent_projects_file
        f = recent_projects_file()
        if f.is_file():
            paths += [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    except Exception:
        pass
    try:
        from mufasa.ui_qt.recent_project import load_recent_project
        r = load_recent_project()
        if r:
            paths.insert(0, str(r))
    except Exception:
        pass
    seen, items = set(), []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        items.append({"path": p, "exists": Path(p).is_file()})
    return _emit(args, items, lambda items: "Recently opened projects (newest first):\n" + _table(
        [[i["path"], "" if i["exists"] else "(missing)"] for i in items], ["project", ""]))


LIST_SUBJECTS = {
    "models": (_list_models, "every model: Kalman body models, classifiers, detectors, 3-D bodies, priors"),
    "videos": (_list_videos, "the project's videos (--details: frames, fps, size)"),
    "pose": (_list_pose, "the project's 2-D pose files (sources/pose)"),
    "3d": (_list_3d, "the project's 3-D files, each with the video found for it"),
    "runs": (_list_runs, "the derived runs, by stage (--stage for one stage's)"),
    "calibrations": (_list_calibrations, "the project's cameras (sources/calibration)"),
    "steps": (_list_steps, "the pipeline steps of `mufasa run`"),
    "tools": (_list_tools, "mufasa's command-line tools"),
    "projects": (_list_projects, "the recently opened projects"),
}


def cmd_list(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="mufasa list", description="List what a project (or mufasa) holds.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(f"  {k:<13} {v[1]}" for k, v in LIST_SUBJECTS.items()))
    ap.add_argument("what", choices=list(LIST_SUBJECTS), metavar="what", help="one of: " + ", ".join(LIST_SUBJECTS))
    ap.add_argument("-p", "--project", help="project.toml or a folder in the project (default: the current one)")
    ap.add_argument("--json", action="store_true", help="as JSON")
    ap.add_argument("--kind", choices=list(KIND_TITLES), help="models: one kind only")
    ap.add_argument("--project-only", action="store_true", help="models: leave out the global library")
    ap.add_argument("--details", action="store_true", help="videos: frames, fps, size (opens each video)")
    ap.add_argument("--stage", help="runs: one stage's runs (e.g. pose3d, smoothed, features)")
    args = ap.parse_args(argv)
    cfg, how = resolve_project(args.project)
    return LIST_SUBJECTS[args.what][0](args, cfg, how)


# --------------------------------------------------------------------------- #
# mufasa info
# --------------------------------------------------------------------------- #
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")


def describe(path: Path) -> list[tuple[str, str]]:
    """What a file holds, as (field, value) pairs."""
    p = Path(path)
    st = p.stat()
    out = [("file", str(p.resolve())), ("size", _size(st.st_size)), ("modified", _when(st.st_mtime))]
    n = p.name.lower()
    if n.endswith(VIDEO_SUFFIXES):
        d = _video_details(p)
        out += [("kind", "video"), ("frames", str(d["frames"])), ("fps", str(d["fps"])),
                ("size (px)", d["size"]), ("length", d["duration"])]
    elif n.endswith(".parquet"):
        import pyarrow.parquet as pq

        from mufasa.pose3d.files import load_pose3d_meta
        pf = pq.ParquetFile(p)
        cols = pf.schema_arrow.names
        meta = load_pose3d_meta(p)
        out += [("rows", str(pf.metadata.num_rows)), ("columns", str(len(cols)))]
        if meta:
            labels = sorted({c[:-2] for c in cols if c.endswith("_x") and c[:-2] + "_z" in cols})
            out += [("kind", "3-D pose"), ("units", str(meta.get("units", ""))),
                    ("fps", str(meta.get("fps", ""))), ("labels", f"{len(labels)}: {', '.join(labels[:12])}"
                                                        + ("…" if len(labels) > 12 else ""))]
            if meta.get("box_x"):
                out.append(("box (mm)", f"x {meta['box_x']}, y {meta.get('box_y')}, z {meta.get('box_z')}"
                            + (f", walls flaring {meta['box_tilt_deg']}°" if meta.get("box_tilt_deg") else "")))
            if meta.get("camera"):
                cam = meta["camera"]
                out.append(("camera", f"{cam.get('height_mm', '?')} mm up, focal {cam.get('focal_px', '?')} px"))
            for k in ("source_pose", "video", "method"):
                if meta.get(k):
                    out.append((k.replace("_", " "), str(meta[k])))
        elif n.endswith("_gait.parquet"):
            out += [("kind", "gait"), ("columns", ", ".join(cols[:14]) + ("…" if len(cols) > 14 else ""))]
        elif n.endswith("_dynamics.parquet"):
            out += [("kind", "dynamics"), ("columns", ", ".join(cols[:14]) + ("…" if len(cols) > 14 else ""))]
        else:
            out += [("columns", ", ".join(cols[:14]) + ("…" if len(cols) > 14 else ""))]
    elif n.endswith(".npz"):
        out += [("kind", "model" if (p.parent / "card.toml").is_file() else "arrays"), ("holds", _npz_info(p))]
        c = _card(p)
        for k in ("source_path", "copied_at", "mufasa_version"):
            if c.get(k):
                out.append((k.replace("_", " "), str(c[k])))
    elif n.endswith(".json"):
        try:
            d = json.loads(p.read_text())
            kind = ("calibrated animal" if n.endswith(".animal.json") else "muscle set" if n.endswith(".muscles.json")
                    else "camera path" if n.endswith(".camera.json") else "JSON")
            out += [("kind", kind), ("keys", ", ".join(list(d)[:14]) if isinstance(d, dict) else f"list of {len(d)}")]
        except Exception as e:
            out.append(("kind", f"unreadable JSON ({type(e).__name__})"))
    elif n == "project.toml" or n.endswith(".ini"):
        out += [("kind", "project")] + _project_fields(p)
    elif n.endswith(".toml"):
        try:
            out += [("kind", "camera"), ("camera", _camera_text(p))]
        except Exception:
            out.append(("kind", "TOML"))
    elif n.endswith((".csv", ".h5")):
        out.append(("kind", "table (2-D pose, features or labels)"))
    return out


def cmd_info(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="mufasa info", description="What a file holds: a video, a pose, 3-D, gait or "
                                 "dynamics file, a model, a camera, an animal file, a project.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rc, docs = 0, []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"mufasa info: {f}: no such file", file=sys.stderr)
            rc = 1
            continue
        fields = describe(p)
        docs.append(dict(fields))
        if not args.json:
            w = max(len(k) for k, _ in fields)
            print("\n".join(f"{k:<{w}}  {v}" for k, v in fields) + ("\n" if len(args.files) > 1 else ""))
    if args.json:
        print(json.dumps(docs if len(docs) != 1 else docs[0], indent=2))
    return rc


# --------------------------------------------------------------------------- #
# mufasa project
# --------------------------------------------------------------------------- #
def _project_fields(cfg: Path) -> list[tuple[str, str]]:
    from mufasa.project_layout import project_metadata_from_config
    out = []
    try:
        m = project_metadata_from_config(str(cfg))
    except Exception as e:
        return [("metadata", f"unreadable ({e})")]
    bp = m.get("body_parts", [])
    out += [("root", str(_root(cfg))), ("layout", "project.toml" if cfg.suffix == ".toml" else "old (.ini config)"),
            ("animals", f"{m.get('animal_count', '')}" + (f" ({', '.join(m['animal_ids'])})" if m.get("animal_ids") else "")),
            ("body parts", f"{len(bp)}: {', '.join(bp[:12])}" + ("…" if len(bp) > 12 else "")),
            ("behaviours", ", ".join(m.get("classifier_targets", [])) or "(none)"),
            ("import format", str(m.get("file_type", "")))]
    return out


def cmd_project(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="mufasa project", description="A project's summary.")
    ap.add_argument("project", nargs="?", help="project.toml or a folder in it (default: the current one)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg, how = resolve_project(args.project)
    if not _need(cfg, how):
        return 2
    from mufasa.pose3d.files import pose3d_files, project_video_files
    root = _root(cfg)
    fields = [("project", f"{cfg} ({how})")] + _project_fields(cfg)
    pose = root / "sources" / "pose"
    models = [m for m in collect_models(cfg, include_global=False) if m["where"] == "project"]
    counts = {
        "videos": len(project_video_files(str(cfg), root)),
        "2-D pose files": sum(1 for p in _walk(pose, (".csv", ".parquet", ".h5")) if "_3d" not in p.name) if pose.is_dir() else 0,
        "3-D files": len(pose3d_files(root)),
        "cameras": sum(1 for _ in _walk(root / "sources" / "calibration", (".toml",))) if (root / "sources" / "calibration").is_dir() else 0,
        "models": ", ".join(f"{sum(m['kind'] == k for m in models)} {k}" for k in KIND_TITLES
                            if any(m["kind"] == k for m in models)) or "0",
    }
    from mufasa.project_layout import ProjectPaths, is_run_id
    pp = ProjectPaths(root)
    stages = {}
    if pp.derived_dir.is_dir():
        for s in sorted(x for x in pp.derived_dir.iterdir() if x.is_dir()):
            n = sum(1 for _ in s.rglob("*") if _.is_dir() and is_run_id(_.name))
            if n:
                stages[s.name] = n
    counts["runs"] = ", ".join(f"{v} {k}" for k, v in stages.items()) or "0"
    fields += list(counts.items())
    if args.json:
        print(json.dumps(dict(fields), indent=2))
    else:
        w = max(len(k) for k, _ in fields)
        print("\n".join(f"{k:<{w}}  {v}" for k, v in fields))
    return 0


# --------------------------------------------------------------------------- #
# mufasa doctor / version
# --------------------------------------------------------------------------- #
def version() -> str:
    try:
        from importlib.metadata import version as v
        return v("mufasa")
    except Exception:
        import mufasa
        return getattr(mufasa, "__version__", "source")


PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "cv2", "PySide6", "torch", "mujoco", "numba",
            "ultralytics", "deeplabcut")


def _pkg(name: str) -> tuple[str, str]:
    import importlib
    try:
        m = importlib.import_module(name)
    except Exception as e:
        return name, "not installed" if isinstance(e, ImportError) else f"broken ({type(e).__name__}: {e})"
    v = getattr(m, "__version__", "") or getattr(m, "version", "")
    if name == "torch":
        try:
            if m.cuda.is_available():
                gpus = [m.cuda.get_device_name(i) for i in range(m.cuda.device_count())]
                v = f"{v}, CUDA {m.version.cuda}: {', '.join(gpus)}"
            else:
                v = f"{v}, no CUDA"
        except Exception:
            pass
    return name, str(v)


def doctor() -> list[tuple[str, str, bool]]:
    """(check, result, ok) for the environment."""
    out = [("mufasa", version(), True),
           ("python", f"{platform.python_version()} ({sys.executable})", sys.version_info >= (3, 10)),
           ("system", f"{platform.system()} {platform.release()}", True),
           ("conda env", os.environ.get("CONDA_DEFAULT_ENV", "(none active)"), True)]
    need = {"numpy", "pandas", "pyarrow", "scipy", "cv2", "PySide6"}
    for name in PACKAGES:
        n, v = _pkg(name)
        out.append((n, v, not (name in need and (v == "not installed" or v.startswith("broken")))))
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=5)
            out.append(("GPUs", "; ".join(r.stdout.strip().splitlines()) or "none", True))
        except Exception as e:
            out.append(("GPUs", f"nvidia-smi failed ({e})", True))
    else:
        out.append(("GPUs", "no nvidia-smi (CPU only)", True))
    ff = shutil.which("ffmpeg")
    if ff:
        try:
            r = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10)
            nv = "h264_nvenc" in r.stdout
        except Exception:
            nv = False
        out.append(("ffmpeg", ff + (", NVENC" if nv else ""), True))
    else:
        out.append(("ffmpeg", "not found (videos written with OpenCV)", True))
    disp = os.environ.get("WAYLAND_DISPLAY") and "Wayland" or os.environ.get("DISPLAY") and "X11" or "none"
    try:
        from mufasa.ui_qt.linux_env import recommended_qpa_platform
        qpa = recommended_qpa_platform()
    except Exception:
        qpa = "?"
    out.append(("display", f"{disp}; Qt platform {os.environ.get('QT_QPA_PLATFORM') or qpa}", True))
    cfg, how = resolve_project(None)
    out.append(("project", f"{cfg} ({how})" if cfg else how, True))
    return out


def cmd_doctor(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="mufasa doctor", description="Check the environment mufasa runs in.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = doctor()
    if args.json:
        print(json.dumps([{"check": c, "result": r, "ok": ok} for c, r, ok in rows], indent=2))
    else:
        w = max(len(c) for c, _, _ in rows)
        for c, r, ok in rows:
            print(f"{'  ' if ok else '✗ '}{c:<{w}}  {r}")
    return 0 if all(ok for _, _, ok in rows) else 1


def cmd_version(argv: list[str]) -> int:
    print(f"mufasa {version()}")
    return 0


# --------------------------------------------------------------------------- #
# mufasa help
# --------------------------------------------------------------------------- #
def overview() -> str:
    lines = [f"mufasa {version()} — rodent pose and behaviour analysis", "",
             "Usage:",
             "  mufasa [--project P] [--no-auto-discover] [--no-recent]   open the workbench",
             "  mufasa <command> [options]                                 run a command (no window)", "",
             "Commands:"]
    w = max(len(c) for c in COMMANDS)
    lines += [f"  {c:<{w}}  {COMMANDS[c][1]}" for c in COMMANDS]
    lines += ["", "mufasa list <what>:"]
    lw = max(len(k) for k in LIST_SUBJECTS)
    lines += [f"  {k:<{lw}}  {v[1]}" for k, v in LIST_SUBJECTS.items()]
    lines += ["", "Other tools (each takes --help):"]
    tools = [t for t in _tools() if t["command"] != "mufasa"]
    tw = max((len(t["command"]) for t in tools), default=w)
    lines += [f"  {t['command']:<{tw}}  {t['what']}" for t in tools]
    lines += ["", "The project: --project (its project.toml or folder), else the current folder's, else the last",
              "one opened in the workbench. `mufasa help <command>` for a command's options; docs/cli.md."]
    return "\n".join(lines)


def cmd_help(argv: list[str]) -> int:
    if not argv:
        print(overview())
        return 0
    name = argv[0]
    if name == "list" and len(argv) > 1 and argv[1] in LIST_SUBJECTS:
        print(f"mufasa list {argv[1]}: {LIST_SUBJECTS[argv[1]][1]}\n")
        return _dispatch(["list", "--help"])
    if name in COMMANDS:
        return _dispatch([name, "--help"])
    if name in ("workbench", "gui"):
        print(overview().split("\n\nCommands:")[0])
        return 0
    tools = {t["command"]: t for t in _tools()}
    if name in tools or f"mufasa-{name}" in tools:
        t = tools.get(name) or tools[f"mufasa-{name}"]
        print(f"{t['command']}: {t['what']}\nRun `{t['command']} --help` for its options.")
        return 0
    print(f"mufasa help: no command '{name}'\n", file=sys.stderr)
    print(overview(), file=sys.stderr)
    return 2


def _run(argv: list[str]) -> int:
    from mufasa.cli.run import main as run_main
    return int(run_main(argv) or 0)


COMMANDS = {
    "help": (cmd_help, "this overview; `mufasa help <command>` for a command's options"),
    "list": (cmd_list, "list models, videos, pose, 3d, runs, calibrations, steps, tools, projects"),
    "info": (cmd_info, "what a file holds (video, pose, 3-D, gait, model, camera, animal, project)"),
    "project": (cmd_project, "a project's summary: animals, body parts, behaviours, what it holds"),
    "run": (_run, "the 3-D pipeline: `mufasa run <step>` or `mufasa run all` (`mufasa list steps`)"),
    "doctor": (cmd_doctor, "check the environment: Python, packages, GPUs, FFmpeg, the display"),
    "version": (cmd_version, "mufasa's version"),
}


def _dispatch(argv: list[str]) -> int:
    try:
        return int(COMMANDS[argv[0]][0](argv[1:]) or 0)
    except SystemExit as e:                         # argparse: --help, bad options
        return int(e.code or 0) if not isinstance(e.code, str) else 2
    except BrokenPipeError:                         # `mufasa list … | head`
        return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        return cmd_help([])
    if args[0] in ("-V", "--version"):
        return cmd_version([])
    if args[0] not in COMMANDS:
        print(f"mufasa: no command '{args[0]}'", file=sys.stderr)
        return 2
    return _dispatch(args)
