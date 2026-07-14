"""
mufasa.model.marker_rename
==========================

Backend for renaming pose markers (body-parts) after a project has been
created. A rename propagates to every place a marker name lives:

* ``project.toml`` ``[pose].body_parts`` — the ordered marker list;
* ``project.toml`` ``[skeleton]`` — both ``nodes`` and the ``edges``
  (marker-connector relationships) get the new names, so connections follow
  the rename;
* the imported pose parquets under ``csv/input_csv/`` — the ``<bp>_x`` /
  ``<bp>_y`` / ``<bp>_p`` columns are renamed in place.

The pure helpers (:func:`validate_rename_map`, :func:`apply_rename_to_names`,
:func:`apply_rename_to_skeleton`, :func:`rename_pose_columns`) are
dependency-light so they can be unit-tested without a project; only
:func:`rename_markers` touches the filesystem.

NOTE: derived features embed marker names in feature names, so a rename
invalidates any previously-computed features — recompute them afterwards.
:func:`rename_markers` reports how many feature files exist so the caller can
warn.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

_SUFFIXES = ("_x", "_y", "_p", "_likelihood")


def validate_rename_map(body_parts: list[str], rename_map: dict[str, str]) -> None:
    """Validate a ``{old_name: new_name}`` map against the current markers.

    Raises ``ValueError`` if an old name is unknown, a new name is empty, a
    new name collides with a marker that is *not* being renamed away, or two
    renames target the same new name.
    """
    current = list(body_parts)
    current_set = set(current)

    unknown = [o for o in rename_map if o not in current_set]
    if unknown:
        raise ValueError(f"Cannot rename unknown marker(s): {sorted(unknown)}")

    empty = [o for o, n in rename_map.items() if not str(n).strip()]
    if empty:
        raise ValueError(f"New name is empty for marker(s): {sorted(empty)}")

    new_names = [str(n).strip() for n in rename_map.values()]
    dupes = sorted({n for n in new_names if new_names.count(n) > 1})
    if dupes:
        raise ValueError(f"Two markers renamed to the same name: {dupes}")

    # A new name may reuse a name that is itself being renamed away (a swap),
    # but not one that stays put.
    staying = current_set - set(rename_map.keys())
    collide = sorted({n for n in new_names if n in staying})
    if collide:
        raise ValueError(
            f"New name(s) collide with existing markers: {collide}"
        )


def apply_rename_to_names(names: list[str], rename_map: dict[str, str]) -> list[str]:
    """Return ``names`` with the rename map applied, order preserved."""
    return [str(rename_map.get(n, n)) for n in names]


def apply_rename_to_skeleton(
    edges: list, rename_map: dict[str, str]
) -> list[tuple[str, str]]:
    """Rename both endpoints of each skeleton edge (marker-connector
    relationships follow the rename)."""
    out: list[tuple[str, str]] = []
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            continue
        a, b = str(e[0]), str(e[1])
        out.append((rename_map.get(a, a), rename_map.get(b, b)))
    return out


def _rename_label(label: str, rename_map: dict[str, str]) -> str:
    """Rename a flat ``<bp>_<suffix>`` column label if its body-part is in
    the map; leave anything else untouched."""
    for suf in _SUFFIXES:
        if label.endswith(suf):
            bp = label[: -len(suf)]
            if bp in rename_map:
                return f"{rename_map[bp]}{suf}"
            return label
    return label


def rename_pose_columns(
    df: pd.DataFrame, rename_map: dict[str, str]
) -> pd.DataFrame:
    """Rename the ``<bp>_x`` / ``_y`` / ``_p`` columns of a pose frame.

    Handles both the flat column layout and the IMPORTED_POSE 3-level
    MultiIndex (only the innermost level carries the ``<bp>_suffix`` label).
    Non-body-part columns are left alone.
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_tuples = []
        for tup in df.columns:
            *prefix, last = tup
            new_tuples.append((*prefix, _rename_label(str(last), rename_map)))
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples(new_tuples)
    else:
        df = df.copy()
        df.columns = [_rename_label(str(c), rename_map) for c in df.columns]
    return df


def rename_markers(
    config_path: str | os.PathLike,
    rename_map: dict[str, str],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply a marker rename across project.toml and the pose parquets.

    :param config_path: Path to the project ``project.toml``.
    :param rename_map: ``{old_name: new_name}`` (only changed markers).
    :param dry_run: If True, validate and report without writing anything.
    :returns: A summary dict (renamed count, updated body_parts, pose files
        rewritten, skeleton edges updated, feature files that now need
        recompute).
    """
    from mufasa.project_layout import (
        project_paths_from_config,
        read_project_toml,
        write_project_toml,
        write_skeleton,
    )

    rename_map = {str(o): str(n).strip() for o, n in rename_map.items()
                  if str(o) != str(n).strip() and str(n).strip()}
    cp = Path(config_path)
    data = read_project_toml(cp)
    pose = data.get("pose", {}) if isinstance(data.get("pose"), dict) else {}
    body_parts = list(pose.get("body_parts", []))

    validate_rename_map(body_parts, rename_map)

    new_body_parts = apply_rename_to_names(body_parts, rename_map)

    sk = data.get("skeleton")
    new_nodes: list[str] = []
    new_edges: list[tuple[str, str]] = []
    if isinstance(sk, dict):
        new_nodes = apply_rename_to_names(
            [str(n) for n in sk.get("nodes", [])], rename_map
        )
        new_edges = apply_rename_to_skeleton(sk.get("edges", []), rename_map)

    # locate pose parquets/csvs
    file_type = str(pose.get("file_type", "csv"))
    try:
        paths = project_paths_from_config(cp)
        input_dir = paths.get("input_pose_dir", "")
    except Exception:
        input_dir = ""
    pose_files = []
    if input_dir and os.path.isdir(input_dir):
        pose_files = [
            os.path.join(input_dir, f)
            for f in sorted(os.listdir(input_dir))
            if f.lower().endswith(f".{file_type}") and not f.startswith(".")
        ]
    feature_dir = ""
    try:
        feature_dir = project_paths_from_config(cp).get("derived_features_dir", "")
    except Exception:
        feature_dir = ""
    feature_file_count = 0
    if feature_dir and os.path.isdir(feature_dir):
        feature_file_count = sum(1 for _ in _walk_files(feature_dir))

    summary = {
        "renamed": dict(rename_map),
        "n_renamed": len(rename_map),
        "body_parts": new_body_parts,
        "pose_files": len(pose_files),
        "skeleton_edges": len(new_edges),
        "feature_files_need_recompute": feature_file_count,
        "dry_run": dry_run,
    }
    if dry_run or not rename_map:
        return summary

    # 1) project.toml [pose].body_parts
    pose["body_parts"] = new_body_parts
    data["pose"] = pose
    write_project_toml(cp, data)
    # 2) project.toml [skeleton]
    if isinstance(sk, dict) and (new_nodes or new_edges):
        write_skeleton(cp, nodes=new_nodes or new_body_parts, edges=new_edges)
    # 3) pose parquets/csvs
    if pose_files:
        from mufasa.utils.read_write import read_df, write_df
        for fp in pose_files:
            df = read_df(fp, file_type, check_multiindex=True)
            df = rename_pose_columns(df, rename_map)
            write_df(df, file_type, fp,
                     multi_idx_header=isinstance(df.columns, pd.MultiIndex))

    return summary


def _walk_files(root: str):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.startswith("."):
                yield os.path.join(dirpath, f)


__all__ = [
    "validate_rename_map",
    "apply_rename_to_names",
    "apply_rename_to_skeleton",
    "rename_pose_columns",
    "rename_markers",
]
