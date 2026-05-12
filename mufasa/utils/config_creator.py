"""
mufasa.utils.config_creator
===========================

Patch 122d: rewritten to produce **only** the v1 project layout
defined in :mod:`mufasa.project_layout`. The previous legacy
SimBA tree (``project_folder/csv/{input_csv,
outlier_corrected_*,...}`` + ``project_config.ini``) is gone for
new projects; existing legacy projects still work via the
unchanged :mod:`mufasa.legacy_layout` reader and
:mod:`mufasa.cli.migrate_project` migration tool.

A fresh project lands as::

    <parent>/<project_name>/
    ├── project.toml                     # this module writes this
    ├── sources/{videos,pose,annotations}/   # ensure_skeleton() creates these
    ├── derived/                         # empty; populated by pose-cleanup
    ├── models/                          # trained models land here
    └── logs/

Consumers
---------

* :class:`mufasa.ui_qt.create_project_dialog.CreateProjectDialog`
  drives this from the workbench's File → New project flow.
* The ``__main__`` block at the bottom of this file gives a CLI
  entry point: ``python -m mufasa.utils.config_creator
  --project_path ... --project_name ...``.

Both paths set ``ProjectConfigCreator.config_path`` to the
absolute path of the freshly-written ``project.toml``; callers
hand that off to the workbench like they used to with the legacy
INI path. Any form that still parses ``config_path`` as an INI
will break under v1 — those forms are the target of the
"v1-aware form" thread (see CHANGELOG).
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path
from typing import List, Optional

import mufasa
from mufasa.project_layout import (
    PROJECT_CONFIG_FILENAME,
    PROJECT_LAYOUT_VERSION,
    ProjectPaths,
    write_project_toml,
)
from mufasa.utils.enums import Paths


def _read_preset_body_parts(
    body_part_config_idx: int,
) -> List[str]:
    """Look up the body-part list at row ``body_part_config_idx``
    of the canonical ``pose_configurations/bp_names/bp_names.csv``.

    The file is comma-separated, one preset per row, with trailing
    empty columns padded out to a fixed width. The dialog's preset
    dropdown displays the human-readable name from
    ``pose_config_names.csv`` at the same row index.
    """
    simba_dir = Path(mufasa.__file__).parent
    bp_dir_path = simba_dir / Paths.SIMBA_BP_CONFIG_PATH.value
    with open(bp_dir_path, "r", encoding="utf8") as f:
        rows = list(csv.reader(f, delimiter=","))
    if body_part_config_idx < 0 or body_part_config_idx >= len(rows):
        raise ValueError(
            f"body_part_config_idx={body_part_config_idx} is out of "
            f"range for {bp_dir_path} ({len(rows)} rows)."
        )
    chosen = [bp for bp in rows[body_part_config_idx] if bp]
    if not chosen:
        raise ValueError(
            f"Preset at row {body_part_config_idx} of {bp_dir_path} "
            f"is empty."
        )
    return chosen


class ProjectConfigCreator:
    """Create a v1-layout Mufasa project.

    :param str project_path: directory under which to create the
        project. The new project lives at ``<project_path>/<project_name>/``.
    :param str project_name: project folder name (and the
        ``project_name`` field in the resulting ``project.toml``).
    :param List[str] target_list: classifier names. May be empty —
        classifiers can be added later from the Classifier page.
    :param str pose_estimation_bp_cnt: SimBA-style body-part count
        code (``"7"``, ``"14"``, ``"16"``, ``"user_defined"``,
        ``"3D_user_defined"``, …). Kept as a project.toml field
        so downstream tooling that still expects the preset code
        can read it.
    :param int body_part_config_idx: row in
        ``pose_configurations/bp_names/bp_names.csv`` whose body-
        part names should be baked into the project. Ignored
        when ``body_parts`` is given.
    :param int animal_cnt: animal count.
    :param str file_type: pose file extension. ``"csv"`` /
        ``"parquet"`` / ``"h5"``. Persisted in
        ``project.toml`` for downstream consumers.
    :param Optional[List[str]] body_parts: explicit body-part
        list. If passed, ``body_part_config_idx`` is ignored and
        the project gets exactly these body parts (preserved in
        order). The autodetect flow in
        :class:`CreateProjectDialog` uses this.

    Attributes
    ----------
    config_path : str
        Absolute path to the written ``project.toml``. Callers
        hand this to the workbench (or to any tool that expects
        a project locator) the same way they did with the legacy
        INI path.
    project_root : Path
        ``<project_path>/<project_name>/``. The v1 layout root.
    """

    def __init__(
        self,
        project_path: str,
        project_name: str,
        target_list: List[str],
        pose_estimation_bp_cnt: str,
        body_part_config_idx: int,
        animal_cnt: int,
        file_type: str = "csv",
        body_parts: Optional[List[str]] = None,
    ) -> None:
        if not project_name or any(
            c in project_name for c in r" /\:<>|*?\""
        ):
            raise ValueError(
                f"project_name={project_name!r} contains shell-unfriendly "
                f"characters or is empty."
            )
        if animal_cnt < 1:
            raise ValueError(
                f"animal_cnt must be >= 1, got {animal_cnt}"
            )
        if file_type not in ("csv", "parquet", "h5"):
            raise ValueError(
                f"file_type must be csv, parquet, or h5; got {file_type!r}"
            )

        self.project_path = project_path
        self.project_name = project_name
        self.target_list = list(target_list)
        self.pose_estimation_bp_cnt = pose_estimation_bp_cnt
        self.body_part_config_idx = body_part_config_idx
        self.animal_cnt = animal_cnt
        self.file_type = file_type
        self.os_platform = platform.system()

        # Resolve body_parts: explicit override > preset lookup.
        if body_parts is not None:
            if not body_parts:
                raise ValueError(
                    "body_parts list, when provided, must be non-empty."
                )
            self.body_parts = list(body_parts)
        else:
            self.body_parts = _read_preset_body_parts(
                body_part_config_idx,
            )

        # Resolve project_root and refuse to clobber an existing
        # project. ProjectPaths.ensure_skeleton() is idempotent
        # on partial trees, but a complete project shouldn't be
        # overwritten — that's a user error.
        self.project_root = (
            Path(self.project_path) / self.project_name
        ).resolve()
        if (self.project_root / PROJECT_CONFIG_FILENAME).exists():
            raise FileExistsError(
                f"{self.project_root} already contains "
                f"{PROJECT_CONFIG_FILENAME}. Pick a different name "
                f"or remove the existing project first."
            )

        self._create_skeleton()
        self._write_project_toml()

    # ----------------------------------------------------------- #
    # Skeleton creation
    # ----------------------------------------------------------- #
    def _create_skeleton(self) -> None:
        """Build ``sources/``, ``derived/``, ``models/``, ``logs/``
        under ``project_root`` via the canonical ProjectPaths
        helper. Idempotent — re-running on a partial skeleton is
        safe.
        """
        self.project_root.mkdir(parents=True, exist_ok=True)
        paths = ProjectPaths(self.project_root)
        paths.ensure_skeleton()

    # ----------------------------------------------------------- #
    # project.toml writer
    # ----------------------------------------------------------- #
    def _write_project_toml(self) -> None:
        """Write the project.toml. Schema mirrors what was in the
        legacy INI but TOML-native and explicit. Sections:

        Top-level fields
            ``project_layout_version``, ``project_name``,
            ``created``, ``mufasa_version``, ``os_platform``.

        ``[pose]``
            Animal count, file type, body-parts list (in order),
            preset name / code, configured animal IDs.

        ``[classifiers]``
            ``targets`` — list of classifier names; may be empty.
            (Other classifier knobs land here as forms migrate.)

        ``[outlier_settings]``
            Movement / location criteria. Initial value
            ``"NaN"`` mirrors the legacy ``Dtypes.NONE`` sentinel
            so existing outlier-settings code paths can keep
            reading them verbatim once they're TOML-aware.
        """
        try:
            mufasa_version = getattr(mufasa, "__version__", "unknown")
        except Exception:
            mufasa_version = "unknown"

        data: dict = {
            "project_layout_version": PROJECT_LAYOUT_VERSION,
            "project_name":           self.project_name,
            "created":                time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
            ),
            "mufasa_version":         mufasa_version,
            "os_platform":            self.os_platform,
            "pose": {
                "animal_count":      self.animal_cnt,
                "file_type":         self.file_type,
                "body_parts":        self.body_parts,
                "pose_config_code":  str(self.pose_estimation_bp_cnt),
                "pose_config_idx":   int(self.body_part_config_idx),
                # animal_ids defaults to ["Animal_1", ...] —
                # exposes the same naming surface the legacy
                # multi_animal IDs section had, in case
                # downstream tooling sets explicit names later.
                "animal_ids": [
                    f"Animal_{i+1}" for i in range(self.animal_cnt)
                ],
            },
            "classifiers": {
                "targets": list(self.target_list),
            },
            "outlier_settings": {
                # "NaN" matches the legacy Dtypes.NONE.value
                # sentinel that downstream readers expect for
                # "not configured yet."
                "movement_criterion": "NaN",
                "location_criterion": "NaN",
            },
        }
        self.config_path = str(
            self.project_root / PROJECT_CONFIG_FILENAME
        )
        write_project_toml(Path(self.config_path), data)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__" and not hasattr(sys, "ps1"):
    parser = argparse.ArgumentParser(
        description="Create a Mufasa v1-layout project from the CLI.",
    )
    parser.add_argument("--project_path", type=str, required=True,
                        help="Parent directory for the new project.")
    parser.add_argument("--project_name", type=str, required=True,
                        help="Project name (becomes the subdirectory name).")
    parser.add_argument("--target_list", type=str, nargs="*", default=[],
                        help="Classifier names (may be empty).")
    parser.add_argument("--pose_estimation_bp_cnt", type=str, default="7",
                        help="Body-part count code (e.g. '7', '14', "
                             "'user_defined').")
    parser.add_argument("--body_part_config_idx", type=int, default=1,
                        help="Row index into pose_configurations/bp_names/"
                             "bp_names.csv.")
    parser.add_argument("--animal_cnt", type=int, default=1,
                        help="Number of animals.")
    parser.add_argument("--file_type", type=str, default="csv",
                        choices=("csv", "parquet", "h5"))
    parser.add_argument("--body_parts", type=str, nargs="*", default=None,
                        help="Explicit body-part list (overrides "
                             "--body_part_config_idx if given).")
    args = parser.parse_args()

    creator = ProjectConfigCreator(
        project_path=args.project_path,
        project_name=args.project_name,
        target_list=args.target_list,
        pose_estimation_bp_cnt=args.pose_estimation_bp_cnt,
        body_part_config_idx=args.body_part_config_idx,
        animal_cnt=args.animal_cnt,
        file_type=args.file_type,
        body_parts=args.body_parts,
    )
    print(f"Created project at {creator.project_root}")
    print(f"  project.toml: {creator.config_path}")
