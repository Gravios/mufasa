"""
mufasa.pose3d.pipeline
======================

Patch 122kf: 3-D pose for a project (or loose files).

Per video::

    2-D labels (FDLC / DLC / Mufasa)  ──┐
    Kalman v2 smoothed run (optional) ──┼─► single-camera (or stereo) 3-D
    camera file (calibration)         ──┘       └─► 3-D Kalman smoothing (optional)
                                                     └─► derived/pose3d/<run_id>/<video>_3d.parquet

Project defaults: 2-D labels from ``sources/pose/``; gaps filled from the
newest ``derived/smoothed/kalman_v2/<run>/``; cameras from
``sources/calibration/`` (``<video>.camera.toml``, ``camera_<first token
of the video name>.toml`` or ``camera.toml``); output in a new
``derived/pose3d/<run_id>/`` with ``run.toml``. The 3-D viewer
(feature viewer "3-D…") finds the files there.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mufasa.pose3d.camera import Camera, camera_for_video
from mufasa.pose3d.io import skeleton_edges, write_pose3d
from mufasa.pose3d.kalman3d import Kalman3DConfig, smooth_pose3d_frame
from mufasa.pose3d.mono import MonoConfig, load_pose2d, reconstruct_mono

POSE_EXTS = (".parquet", ".csv", ".h5")
STAGE = "pose3d"


@dataclass
class Job:
    stem: str
    pose: Path
    camera: Path
    fallback: Path | None = None
    pose1: Path | None = None          # stereo: second camera's labels
    camera1: Path | None = None
    offset1: int = 0


@dataclass
class RunSettings:
    mode: str = "mono"                 # "mono" | "stereo"
    mono: MonoConfig = field(default_factory=MonoConfig)
    kalman: Kalman3DConfig | None = field(default_factory=Kalman3DConfig)
    stereo_likelihood: float = 0.6
    stereo_max_reproj_px: float = 15.0
    # processes: several videos run at once; a single video splits its
    # head and spine chunks over them instead (122ki)
    workers: int = 1


def default_workers() -> int:
    """Leave one core for the GUI / system."""
    import os
    return max(1, (os.cpu_count() or 2) - 1)


def _stem(p: Path) -> str:
    s = p.name
    for suf in (".fdlc.parquet", ".parquet", ".csv", ".h5"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return p.stem


def pose_files(d) -> dict[str, Path]:
    d = Path(d)
    if not d.is_dir():
        return {}
    return {_stem(p): p for p in sorted(d.iterdir()) if p.is_file() and p.name.endswith(POSE_EXTS)}


def _match(stem: str, files: dict[str, Path]) -> Path | None:
    if stem in files:
        return files[stem]
    for k, v in files.items():
        if k.startswith(stem) or stem.startswith(k):
            return v
    return None


def project_defaults(config_path) -> dict:
    from mufasa.project_layout import (
        ProjectPaths,
        SmoothingFlavors,
        Stages,
        resolve_v1_project_root,
    )
    root = resolve_v1_project_root(str(config_path))
    if root is None:
        raise ValueError("3-D pose needs a v1 Mufasa project (project.toml with sources/ and derived/)")
    paths = ProjectPaths(root)
    kv = paths.latest_run(Stages.SMOOTHED, flavor=SmoothingFlavors.KALMAN_V2)
    return {"root": root, "paths": paths, "pose_dir": paths.sources_pose,
            "fallback_dir": kv, "calibration_dir": paths.sources_dir / "calibration",
            "videos_dir": paths.sources_videos}


def find_jobs(pose_dir, calibration_dir, fallback_dir=None, *, stems: list[str] | None = None,
              stereo_tokens: tuple[str, str] | None = None) -> tuple[list[Job], list[str]]:
    """Pair 2-D files with cameras (and fallback files). With
    ``stereo_tokens=("camA", "camB")`` files whose names differ only by
    the token are paired (camera 0 / camera 1). Returns (jobs, problems)."""
    poses = pose_files(pose_dir)
    fb = pose_files(fallback_dir) if fallback_dir else {}
    jobs, problems = [], []
    for stem, p in poses.items():
        if stems and stem not in stems:
            continue
        if stereo_tokens:
            t0, t1 = stereo_tokens
            if t0 not in stem:
                continue
            other = stem.replace(t0, t1)
            if other not in poses:
                problems.append(f"{stem}: no matching {t1} file")
                continue
            c0, c1 = camera_for_video(calibration_dir, stem), camera_for_video(calibration_dir, other)
            if c0 is None or c1 is None:
                problems.append(f"{stem}: camera file missing for {t0 if c0 is None else t1}")
                continue
            jobs.append(Job(stem, p, c0, pose1=poses[other], camera1=c1))
            continue
        cam = camera_for_video(calibration_dir, stem)
        if cam is None:
            problems.append(f"{stem}: no camera file in {calibration_dir}")
            continue
        jobs.append(Job(stem, p, cam, _match(stem, fb)))
    return jobs, problems


def run_job(job: Job, settings: RunSettings, out_dir, progress=None) -> dict:
    say = progress or (lambda s: None)
    cam = Camera.load(job.camera)
    t0 = time.time()
    if settings.mode == "stereo":
        from mufasa.pose3d.stereo import reconstruct_stereo
        cam1 = Camera.load(job.camera1)
        p0 = load_pose2d(job.pose, None, settings.mono.likelihood_threshold)
        p1 = load_pose2d(job.pose1, None, settings.mono.likelihood_threshold)
        say(f"{job.stem}: triangulating")
        df, rep = reconstruct_stereo(p0, p1, cam, cam1, offset1=job.offset1,
                                     likelihood_threshold=settings.stereo_likelihood,
                                     max_reproj_px=settings.stereo_max_reproj_px)
        center = None
        method = "two cameras, linear triangulation (undistorted pixels), likelihood and reprojection gates"
    else:
        pose = load_pose2d(job.pose, job.fallback, settings.mono.likelihood_threshold)
        df, rep = reconstruct_mono(pose, cam, settings.mono, progress=lambda s: say(f"{job.stem}: {s}"))
        center = cam.center
        method = (f"single camera: rigid head (ear width {settings.mono.head_width_mm:.1f} mm), "
                  "Rat7M trunk heights, pelvis pitched to the lumbar angle, spine chain within "
                  "Rat7M bend limits, tail draped; labels on their camera rays")
    names = [c[:-2] for c in df.columns if c.endswith("_z")]
    if settings.kalman is not None:
        say(f"{job.stem}: 3-D Kalman smoothing")
        df, kinfo = smooth_pose3d_frame(df, names, settings.kalman, center)
        rep["kalman3d"] = kinfo
        method += f"; 3-D Kalman (RTS) smoothing σa={settings.kalman.sigma_a:g}"
        if center is not None:
            import numpy as np
            for m in names:
                X = df[[m + "_x", m + "_y", m + "_z"]].to_numpy()
                uv = Pose2DCache.uv(job, m)
                if uv is not None:
                    df[m + "_reproj_px"] = np.linalg.norm(cam.project(X) - uv, axis=1)
    meta = {"units": "mm", "method": method, "camera": cam.viewer_meta(),
            "edges": skeleton_edges(names), "source_pose": str(job.pose),
            "fallback_pose": str(job.fallback) if job.fallback else None,
            "camera_file": str(job.camera), "report": _jsonable(rep)}
    if cam.box:
        meta.update(box_x=cam.box["x"], box_y=cam.box["y"], box_z=[0, min(cam.box["z"][1], 2000)])
    out = write_pose3d(df, Path(out_dir) / f"{job.stem}_3d.parquet", meta)
    rep["output"] = str(out)
    rep["seconds_total"] = round(time.time() - t0, 1)
    Pose2DCache.clear()
    return rep


class Pose2DCache:
    """Keeps the primary 2-D positions of the job being processed for the
    post-smoothing reprojection column."""
    _job = None
    _pose = None

    @classmethod
    def uv(cls, job: Job, marker: str):
        if cls._job is not job:
            try:
                cls._pose, cls._job = load_pose2d(job.pose, None), job
            except Exception:
                return None
        return cls._pose.uv.get(marker)

    @classmethod
    def clear(cls) -> None:
        cls._job = cls._pose = None


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _run_job_quiet(args) -> tuple[str, dict]:
    job, settings, out_dir = args
    return job.stem, run_job(job, settings, out_dir, None)


def run_pose3d(jobs: list[Job], settings: RunSettings, out_dir, *, progress=None,
               run_id: str | None = None) -> dict:
    """Run every job. With ``settings.workers`` > 1, several videos run at
    once (one process each); a single video uses the workers for its
    chunks instead."""
    from dataclasses import replace
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    w = max(1, int(settings.workers or 1))
    if w > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        per = replace(settings, mono=replace(settings.mono, workers=1))
        n = min(w, len(jobs))
        if progress:
            progress(f"{len(jobs)} videos on {n} workers")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(_run_job_quiet, (job, per, out_dir)) for job in jobs]
            for i, f in enumerate(as_completed(futs)):
                stem, rep = f.result()
                reports[stem] = rep
                if progress:
                    left = (time.time() - t0) / (i + 1) * (len(jobs) - i - 1)
                    progress(f"[{i + 1}/{len(jobs)}] {stem} done ({rep.get('seconds_total', 0):.0f} s); "
                             f"about {left / 60:.0f} min left")
        reports = {j.stem: reports[j.stem] for j in jobs}
    else:
        if w > 1 and settings.mode == "mono":
            settings = replace(settings, mono=replace(settings.mono, workers=w))
        for i, job in enumerate(jobs):
            if progress:
                progress(f"[{i + 1}/{len(jobs)}] {job.stem}")
            reports[job.stem] = run_job(job, settings, out_dir, progress)
    mono = asdict(settings.mono)
    mono["spine_map"] = [f"{k}: {a}..{b} {f:g}" for k, (a, b, f) in settings.mono.spine_map.items()]
    hz = settings.mono.head_template_z
    mono["head_template_z"] = [f"{k}: {v:g}" for k, v in hz.items()] if hz else "learned"
    params = {"mode": settings.mode, "mono": mono}
    if settings.kalman is not None:                     # TOML has no null
        params["kalman3d"] = asdict(settings.kalman)
    try:
        from mufasa.project_layout import write_run_toml
        write_run_toml(out_dir / "run.toml", stage=STAGE, run_id=run_id or out_dir.name,
                       inputs=[str(j.pose) for j in jobs], params=json.loads(json.dumps(_jsonable(params))),
                       results={"videos": len(jobs)})
    except Exception:                                   # provenance is best effort
        (out_dir / "run.json").write_text(json.dumps({"params": _jsonable(params)}, indent=1))
    (out_dir / "report.json").write_text(json.dumps(_jsonable(reports), indent=1))
    return reports


def run_project(config_path, settings: RunSettings | None = None, *, stems=None,
                pose_dir=None, fallback_dir=None, calibration_dir=None, use_fallback: bool = True,
                stereo_tokens=None, progress=None) -> tuple[Path, dict, list[str]]:
    """Run 3-D pose for a project; returns (run_dir, reports, problems)."""
    from mufasa.project_layout import generate_run_id
    settings = settings or RunSettings()
    d = project_defaults(config_path)
    pose_dir = Path(pose_dir or d["pose_dir"])
    fb = (Path(fallback_dir) if fallback_dir else d["fallback_dir"]) if use_fallback else None
    cal = Path(calibration_dir or d["calibration_dir"])
    jobs, problems = find_jobs(pose_dir, cal, fb, stems=stems, stereo_tokens=stereo_tokens)
    if not jobs:
        raise ValueError("nothing to do: " + ("; ".join(problems) or f"no 2-D pose files in {pose_dir}"))
    run_id = generate_run_id()
    out = d["paths"].stage_run_dir(STAGE, run_id=run_id)
    reports = run_pose3d(jobs, settings, out, progress=progress, run_id=run_id)
    return out, reports, problems


__all__ = ["Job", "RunSettings", "default_workers", "find_jobs", "pose_files", "project_defaults", "run_job",
           "run_pose3d", "run_project"]
