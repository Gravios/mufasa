"""Patch 122a: read the legacy SimBA-derived project layout.

This module knows how to interpret the layout Mufasa inherited
from SimBA, so the migration tool (and any compatibility shims)
can read old projects without polluting the new v1 layout code
with backward-compatibility concerns.

Legacy layout, for reference::

    <project_path>/<project_name>/
    ├── models/
    └── project_folder/
        ├── project_config.ini
        ├── configs/
        ├── csv/
        │   ├── input_csv/
        │   ├── features_extracted/
        │   ├── machine_results/
        │   ├── outlier_corrected_movement/
        │   ├── outlier_corrected_movement_location/
        │   └── targets_inserted/
        ├── frames/{input,output}/
        ├── logs/measures/pose_configs/bp_names/
        └── videos/

The mapping from legacy paths to v1 paths is encoded in
:data:`LEGACY_TO_V1_MAPPING`. The migration tool walks this list
to move files; "merge into project.toml" entries are handled
separately by parsing the INI and writing TOML.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Maps a legacy subdirectory (relative to project_folder/) to
# the v1 destination (relative to project root) and a free-form
# label used in the migration manifest.
#
# Entries marked with destination == None denote inputs that go
# elsewhere (into project.toml, or get dropped entirely).
LEGACY_TO_V1_MAPPING: List[Tuple[str, Optional[str], str]] = [
    # source data (irreplaceable)
    ("csv/input_csv",                       "sources/pose",                   "raw tracker output"),
    ("videos",                              "sources/videos",                 "source videos"),
    # derived (regeneratable)
    ("csv/features_extracted",              "derived/features/{import_run}",  "feature tables"),
    ("csv/machine_results",                 "derived/classifications/{import_run}", "classifier output"),
    ("csv/outlier_corrected_movement",      "derived/outlier_corrected/movement_{import_run}", "outlier-corrected (movement)"),
    ("csv/outlier_corrected_movement_location", "derived/outlier_corrected/movement_location_{import_run}", "outlier-corrected (movement+location)"),
    ("csv/targets_inserted",                "derived/annotations/{import_run}", "manual target inserts"),
    # frames
    ("frames/input",                        "derived/frames/extracted/{import_run}", "extracted frames"),
    ("frames/output",                       "derived/frames/annotated/{import_run}", "annotated frames"),
    # logs (timestamped subdir so multiple migrations don't collide)
    ("logs",                                "logs/{import_run}",               "logs"),
    # config — folded into project.toml; not moved verbatim
    ("project_config.ini",                  None,                              "merged into project.toml"),
    ("configs",                             None,                              "merged into project.toml"),
]


@dataclass(frozen=True)
class LegacyProjectPaths:
    """Resolves paths within a legacy SimBA-style project.

    Constructed from the path that contains ``project_folder/``
    (i.e. the parent of project_folder, not project_folder
    itself). All paths are computed; nothing is created on
    disk.

    The class also accepts being pointed directly at the
    ``project_folder`` itself, which is a common user mistake;
    :meth:`open` handles both.
    """

    project_path: Path     # parent of project_folder/
    project_folder: Path   # the actual SimBA project_folder/
    models_folder: Path    # parent_of_project_folder/<project_name>/models/

    @classmethod
    def open(cls, path: Path) -> "LegacyProjectPaths":
        path = Path(path).resolve()
        # Case A: user pointed at the parent (project_path/<project_name>)
        # so that path/project_folder/project_config.ini exists.
        if (path / "project_folder" / "project_config.ini").is_file():
            return cls(
                project_path=path,
                project_folder=path / "project_folder",
                models_folder=path / "models",
            )
        # Case B: user pointed at project_folder itself.
        if (path / "project_config.ini").is_file():
            return cls(
                project_path=path.parent,
                project_folder=path,
                models_folder=path.parent / "models",
            )
        raise FileNotFoundError(
            f"{path}: not a legacy Mufasa/Mufasa project — no "
            f"project_folder/project_config.ini found at this "
            f"path or one level down"
        )

    @property
    def config_file(self) -> Path:
        return self.project_folder / "project_config.ini"

    def stage_path(self, relative: str) -> Path:
        """Return an absolute path under project_folder/."""
        return self.project_folder / relative


def parse_legacy_config(path: Path) -> Dict[str, Any]:
    """Parse an old-style ``project_config.ini`` into the
    dict-of-dicts shape the v1 ``project.toml`` writer expects.

    SimBA's INI sections map roughly as follows:

    * ``[General settings]`` → ``[project]`` (name, animal_no,
      pose_estimation, file_type, etc).
    * ``[SML settings]`` → ``[classification]``
    * ``[ROI settings]`` → ``[roi]``
    * ``[Outlier settings]`` → ``[outlier_correction]``
    * ``[create ensemble settings]`` →
      ``[stages.classification]``

    Anything we don't recognize is preserved verbatim in a
    ``[legacy.<section>]`` table so no information is lost
    during migration. Body-part list (``pose_estimation_body_
    parts``) is exploded into a list under
    ``[project].body_parts``.
    """
    cp = configparser.ConfigParser()
    cp.read(path)
    out: Dict[str, Any] = {}

    # ---- [General settings] → [project] ----
    proj: Dict[str, Any] = {}
    if cp.has_section("General settings"):
        gs = cp["General settings"]
        # The fields with stable semantics:
        for src, dst in [
            ("project_name",     "name"),
            ("animal_no",        "animal_count"),
            ("file_type",        "tracker_file_type"),
            ("pose_estimation",  "tracker_type"),
            ("pose_estimation_body_parts",
             "body_parts_template"),
            ("project_path",     "original_project_path"),
        ]:
            if src in gs:
                v = gs[src]
                # animal_no is numeric in practice
                if dst == "animal_count":
                    try:
                        v = int(v)
                    except ValueError:
                        pass
                proj[dst] = v
    if proj:
        out["project"] = proj

    # ---- [Outlier settings] → [stages.outlier_correction] ----
    if cp.has_section("Outlier settings"):
        stages = out.setdefault("stages", {})
        oc: Dict[str, Any] = {}
        for key in cp["Outlier settings"]:
            try:
                oc[key] = float(cp["Outlier settings"][key])
            except ValueError:
                oc[key] = cp["Outlier settings"][key]
        stages["outlier_correction"] = oc

    # ---- [SML settings] → [classification] ----
    if cp.has_section("SML settings"):
        cls_: Dict[str, Any] = {}
        for key in cp["SML settings"]:
            cls_[key] = cp["SML settings"][key]
        out["classification"] = cls_

    # ---- Preserve everything else under [legacy.<section>] ----
    known = {
        "General settings", "Outlier settings",
        "SML settings",
    }
    legacy: Dict[str, Any] = {}
    for sec in cp.sections():
        if sec in known:
            continue
        legacy[sec.replace(" ", "_").lower()] = {
            k: cp[sec][k] for k in cp[sec]
        }
    if legacy:
        out["legacy"] = legacy

    return out


def parse_legacy_body_part_names(paths: LegacyProjectPaths) -> List[str]:
    """Read the body-part-name files from
    ``logs/measures/pose_configs/bp_names/`` if present.

    SimBA stores body-part names as plain-text files (one
    name per line) in this four-levels-deep directory. The
    v1 layout pulls them into ``project.toml`` under
    ``[project].body_parts``. Returns an empty list if the
    directory doesn't exist.
    """
    bp_dir = (
        paths.project_folder
        / "logs" / "measures" / "pose_configs" / "bp_names"
    )
    if not bp_dir.is_dir():
        return []
    # SimBA writes one .csv file per configured layout. Take
    # the first one we can read — there's usually only one.
    names: List[str] = []
    for f in sorted(bp_dir.glob("*.csv")):
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            line = line.strip()
            if line and line not in names:
                names.append(line)
        if names:
            break
    return names
