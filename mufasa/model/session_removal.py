"""
mufasa.model.session_removal
============================

Remove imported sessions from a project — the counterpart of import.

A session's imported data is its video in ``sources/videos/`` plus its pose
file(s) in ``sources/pose/`` (same stem) and its row in ``video_info.csv``.
Removal never deletes anything: the files are *moved* into
``<root>/backups/removed_sessions-<run_id>/`` (a rename on the same disk, so
instant even for multi-GB videos), the ``video_info.csv`` row is dropped after
the whole file is copied there, and a ``manifest.toml`` records where every
file came from — undoing it is moving the files back. A symlinked video moves
as the link itself; the footage it points to is never touched.

Derived outputs (smoothed pose, features, classifications under
``derived/``) are run-scoped and are left as they are.

Qt-free so the Data Import table, scripts and tests share one implementation
(patch 122if).
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

_POSE_EXTS = (".parquet", ".csv", ".h5")


def _paths(config_path) -> dict:
    from mufasa.project_layout import project_paths_from_config

    return project_paths_from_config(config_path)


def session_files(config_path, video_names) -> dict[str, dict]:
    """What removing each session would touch.

    :param video_names: file names in the project's videos folder (as the
        Data Import table lists them), e.g. ``"K7_cam_A.mp4"``.
    :returns: ``{video_name: {"video": path | None, "pose": [paths],
        "stem": str}}``. ``video`` is ``None`` when the name isn't in the
        videos folder (nothing to move for it); ``pose`` lists the imported
        pose files with the same stem.
    """
    p = _paths(config_path)
    vdir, pdir = Path(p["video_dir"]), Path(p["input_pose_dir"])
    pose_files = (
        [f for f in sorted(pdir.iterdir()) if f.is_file()]
        if pdir.is_dir() else []
    )
    out: dict[str, dict] = {}
    for name in video_names:
        name = os.path.basename(str(name))
        stem = os.path.splitext(name)[0]
        v = vdir / name
        out[name] = {
            "stem": stem,
            "video": str(v) if (v.is_file() or v.is_symlink()) else None,
            "pose": [
                str(f) for f in pose_files
                if f.suffix.lower() in _POSE_EXTS and f.stem == stem
            ],
        }
    return out


def _toml_str(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def remove_sessions(config_path, video_names, *, include_pose: bool = True) -> dict:
    """Move the selected sessions' imported data out of the project.

    :param video_names: video file names in the project's videos folder.
    :param include_pose: also move the sessions' imported pose files.
    :returns: ``{"backup_dir", "moved": [(src, dst)], "video_info_rows":
        int, "missing": [names not found]}``.
    :raises ValueError: when nothing selected exists in the project.
    """
    from mufasa.project_layout import generate_run_id

    plan = session_files(config_path, video_names)
    missing = [n for n, s in plan.items() if s["video"] is None and not s["pose"]]
    todo = {n: s for n, s in plan.items() if n not in missing}
    if not todo:
        raise ValueError(
            f"None of the selected sessions exist in this project: {missing}"
        )

    p = _paths(config_path)
    root = Path(p["project_root"])
    backup = root / "backups" / f"removed_sessions-{generate_run_id()}"
    (backup / "videos").mkdir(parents=True, exist_ok=True)
    if include_pose:
        (backup / "pose").mkdir(parents=True, exist_ok=True)

    moved: list[tuple[str, str]] = []
    for s in todo.values():
        srcs = ([("videos", s["video"])] if s["video"] else [])
        if include_pose:
            srcs += [("pose", f) for f in s["pose"]]
        for sub, src in srcs:
            dst = backup / sub / os.path.basename(src)
            # os.rename on a symlink moves the link, not its target; fall
            # back to shutil.move across filesystems.
            try:
                os.rename(src, dst)
            except OSError:
                shutil.move(src, dst)
            moved.append((src, str(dst)))

    # video_info.csv: keep a full copy in the backup, then drop the rows.
    rows_removed = 0
    vi = Path(p["video_info_path"])
    stems = {s["stem"] for s in todo.values()}
    if vi.is_file():
        import pandas as pd

        shutil.copy2(vi, backup / vi.name)
        df = pd.read_csv(vi)
        if "Video" in df.columns:
            keep = ~df["Video"].astype(str).isin(stems)
            rows_removed = int((~keep).sum())
            if rows_removed:
                df[keep].to_csv(vi, index=False)

    lines = [
        "# Sessions removed from the project via Data Import.",
        "# Undo: move each file back to its 'from' path; restore",
        "# video_info.csv from the copy in this folder.",
        f'created = "{time.strftime("%Y-%m-%dT%H:%M:%S%z")}"',
        f"sessions = [{', '.join(_toml_str(n) for n in todo)}]",
        f"video_info_rows_removed = {rows_removed}",
        "",
    ]
    for src, dst in moved:
        lines += [
            "[[files]]",
            f"from = {_toml_str(src)}",
            f"to = {_toml_str(dst)}",
            "",
        ]
    (backup / "manifest.toml").write_text("\n".join(lines), encoding="utf-8")
    return {
        "backup_dir": str(backup),
        "moved": moved,
        "video_info_rows": rows_removed,
        "missing": missing,
    }


__all__ = ["session_files", "remove_sessions"]
