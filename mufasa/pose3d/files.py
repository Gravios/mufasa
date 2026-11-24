"""Finding a project's videos and 3-D files, and which goes with which (122lr; Qt-free since 122lv
so the command line can use it: ``mufasa list videos`` / ``list 3d``).
"""
from __future__ import annotations

import json
from pathlib import Path

META_KEY = b"mufasa_pose3d"

__all__ = ["NOT_POSE3D", "VIDEO_EXTS", "find_pose3d", "find_video", "is_pose3d_file", "load_pose3d_meta",
           "pose3d_files", "pose3d_runs", "project_root_of", "project_video_files"]


def find_pose3d(video_path: str | None, config_path: str | None = None) -> Path | None:
    """A 3-D pose file for the video: beside it (``<stem>*3d*.parquet``) or
    under the project's ``derived/pose3d/``; the newest wins."""
    if not video_path:
        return None
    v = Path(video_path)
    stem = v.stem
    hits = list(v.parent.glob(f"{stem}*3d*.parquet")) if v.parent.is_dir() else []
    if config_path:
        for d in (Path(config_path).parent / "derived" / "pose3d",
                  Path(config_path).parent / "sources" / "pose3d",     # + motion capture (122kl)
                  Path(config_path).parent / "sources" / "pose"):      # + `mufasa run`'s (122lr)
            if d.is_dir():
                hits += [h for h in d.rglob(f"{stem}*.parquet") if d.name == "pose3d" or "_3d" in h.name]
    hits = [h for h in hits if h.is_file() and is_pose3d_file(h)]
    # 122lr: not another video's whose name begins the same (K7_cam_2… for K7_cam)
    own = [h for h in hits if (f := find_video(h, config_path)) is None or f.resolve() == v.resolve()]
    hits = own or hits
    return max(hits, key=lambda h: h.stat().st_mtime) if hits else None


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")
_POSE_SUFFIXES = ("_3d_heights", "_3d_smoothed", "_3d", "_smoothed", "_gait", "_dynamics", "_joints", "_body")
NOT_POSE3D = ("_gait.parquet", "_dynamics.parquet", "_joints.parquet", "_steps.csv")   # beside a 3-D file


def is_pose3d_file(path) -> bool:
    """122lr: a 3-D pose file, not the gait / dynamics / joint numbers written beside one."""
    n = Path(path).name.lower()
    return n.endswith((".parquet", ".csv")) and not n.endswith(NOT_POSE3D)


def project_root_of(path) -> Path | None:
    """122lr: the project a file sits in (the nearest folder above it with ``project.toml``)."""
    for d in Path(path).resolve().parents:
        if (d / "project.toml").is_file():
            return d
    return None


def project_video_files(config_path=None, root=None) -> list[Path]:
    """122lr: every video of the project, in subfolders too: the project's video folder
    (``sources/videos``, or ``videos`` in an old project)."""
    dirs = []
    if config_path:
        try:
            from mufasa.project_layout import project_paths_from_config
            vd = project_paths_from_config(str(config_path)).get("video_dir")
            if vd:
                dirs.append(Path(vd))
        except Exception:
            pass
        root = root or project_root_of(config_path)
    if root:
        dirs += [Path(root) / "sources" / "videos", Path(root) / "videos"]
    out, seen = [], set()
    for d in dirs:
        if not d.is_dir():
            continue
        for v in sorted(d.rglob("*")):
            if (v.suffix.lower() in VIDEO_EXTS and not v.name.startswith(".") and v.is_file()
                    and v.resolve() not in seen):
                seen.add(v.resolve())
                out.append(v)
    return out


def _pose_stems(pose_path) -> list[str]:
    """The names a 3-D file's video may go by: the file's own (less ``_3d`` and the like) and its
    2-D labels' (from its metadata)."""
    p = Path(pose_path)
    stems = [p.name.rsplit(".", 1)[0]]
    try:
        meta = load_pose3d_meta(p)
        if meta.get("video"):
            stems.insert(0, Path(meta["video"]).stem)
        if meta.get("source_pose"):
            stems.append(Path(meta["source_pose"]).name.split(".")[0])
    except Exception:
        pass
    out = []
    for s in stems:
        for suf in _POSE_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
        out.append(s)
    return out


def find_video(pose_path, config_path=None) -> Path | None:
    """122lr: the video a 3-D file was made from: named in its metadata, or the video whose name
    the 3-D file's (or its 2-D labels') name begins with (``K7_cam_1.mp4`` for
    ``K7_cam_1DLC_resnet50…_3d.parquet``); the longest such name wins. Looked for beside the file
    and in the project's videos (subfolders too)."""
    p = Path(pose_path)
    try:
        meta = load_pose3d_meta(p)
    except Exception:
        meta = {}
    if meta.get("video") and Path(meta["video"]).is_file():
        return Path(meta["video"])
    root = project_root_of(config_path) if config_path else None
    root = root or project_root_of(p)
    cands = [v for v in sorted(p.parent.iterdir()) if v.suffix.lower() in VIDEO_EXTS and v.is_file()] \
        if p.parent.is_dir() else []
    cands += project_video_files(config_path, root)
    stems = [s.lower() for s in _pose_stems(p)]
    best, n = None, 0
    for v in cands:
        vs = v.stem.lower()
        if len(vs) > n and any(s == vs or s.startswith(vs) for s in stems):
            best, n = v, len(vs)
    return best


def load_pose3d_meta(path) -> dict:
    """The ``mufasa_pose3d`` metadata of a 3-D parquet file, without reading its columns."""
    path = Path(path)
    if path.suffix.lower() != ".parquet":
        return {}
    import pyarrow.parquet as pq
    raw = (pq.read_schema(path).metadata or {}).get(META_KEY)
    return json.loads(raw) if raw else {}


def pose3d_runs(root: Path | None) -> list[Path]:
    if root is None:
        return []
    from mufasa.project_layout import ProjectPaths
    return ProjectPaths(root).list_runs("pose3d")


def pose3d_files(root: Path | None) -> list[Path]:
    """122lr: every 3-D file of the project, newest run first: the 3-D runs, imported motion
    capture, and those ``mufasa run`` wrote beside the 2-D labels (``sources/pose``)."""
    if root is None:
        return []
    files = []
    for r in reversed(pose3d_runs(root)):
        files += sorted(r.glob("*.parquet"))
    for d in (root / "sources" / "pose3d", root / "sources" / "pose", root / "derived" / "pose3d"):
        if d.is_dir():
            files += sorted(p for p in d.glob("*.parquet") if d.name == "pose3d" or "_3d" in p.name)
    out, seen = [], set()
    for p in files:
        if p.resolve() not in seen and p.is_file() and is_pose3d_file(p):
            seen.add(p.resolve())
            out.append(p)
    return out
