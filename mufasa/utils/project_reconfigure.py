"""
mufasa.utils.project_reconfigure
================================

Reconfigure an existing Mufasa project to ``user_defined`` with a
specified body-part list. Used by the Qt "Reconfigure project from
DLC file…" flow (and can be called directly from scripts).

The function performs three edits, each backed up first:

  1. ``project_config.ini``:
     * ``[create ensemble settings] pose_estimation_body_parts = user_defined``
     * ``[General settings] animal_no = <animal_cnt>``
  2. ``project_folder/logs/measures/pose_configs/bp_names/
     project_bp_names.csv`` ← body parts written one per line.

Returns a structured summary of what was changed so the UI can show
the user exactly what happened.
"""
from __future__ import annotations

import configparser
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union


@dataclass
class ReconfigureResult:
    """What was changed during a reconfigure operation."""
    config_path: Path
    bp_names_path: Path
    config_backup: Path
    bp_backup: Path
    previous_preset: str
    previous_animal_no: str
    previous_body_parts: List[str]
    new_body_parts: List[str]
    changes: List[str] = field(default_factory=list)


class ProjectReconfigureError(Exception):
    """Raised when the project can't be reconfigured (missing files,
    malformed config, etc.)."""


def _read_existing_bps(bp_csv: Path) -> List[str]:
    """Read body parts from the project's existing bp_names.csv.

    The file is sometimes written as comma-separated on a single row
    (as the ProjectConfigCreator does for preset projects) and
    sometimes as one-per-line. Handle both.
    """
    if not bp_csv.exists():
        return []
    text = bp_csv.read_text().strip()
    if not text:
        return []
    if "," in text.splitlines()[0]:
        # Single row, comma-separated (strip trailing empty cells)
        return [x.strip() for x in text.splitlines()[0].split(",") if x.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def reconfigure_project_user_defined(
    config_path: Union[str, Path],
    body_parts: List[str],
    animal_cnt: int = 1,
) -> ReconfigureResult:
    """Switch *config_path*'s project to ``user_defined`` with the
    specified *body_parts* list (in order).

    :param config_path: Path to ``project_config.ini``.
    :param body_parts: Body-part names in the order they appear in the
        source data (x/y/likelihood triplets will be inferred).
    :param animal_cnt: Number of animals. Currently only 1 is
        well-exercised by the autodetect flow.
    :raises ProjectReconfigureError: when the project layout is
        unexpected (missing sections, missing bp_names dir, etc.).
    :returns: :class:`ReconfigureResult` describing what changed.
    """
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.is_file():
        raise ProjectReconfigureError(f"not a file: {cfg_path}")
    if not body_parts:
        raise ProjectReconfigureError("body_parts list is empty")
    if animal_cnt < 1:
        raise ProjectReconfigureError(
            f"animal_cnt must be >= 1, got {animal_cnt}"
        )

    project_folder = cfg_path.parent
    bp_csv = (
        project_folder
        / "logs" / "measures" / "pose_configs" / "bp_names"
        / "project_bp_names.csv"
    )
    if not bp_csv.parent.is_dir():
        raise ProjectReconfigureError(
            f"expected bp_names dir missing: {bp_csv.parent}. Is this "
            f"actually a Mufasa project?"
        )

    # ------------------ Backups -------------------------------------- #
    cfg_backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    shutil.copy2(cfg_path, cfg_backup)

    bp_backup = bp_csv.with_suffix(bp_csv.suffix + ".bak")
    if bp_csv.exists():
        shutil.copy2(bp_csv, bp_backup)
    else:
        # Create an empty backup stub so the return type is consistent
        bp_backup.touch()

    # ------------------ Capture prior state -------------------------- #
    previous_body_parts = _read_existing_bps(bp_csv)

    cp = configparser.ConfigParser()
    cp.read(cfg_path)

    if not cp.has_section("create ensemble settings"):
        raise ProjectReconfigureError(
            f"{cfg_path.name} is missing [create ensemble settings] "
            f"section — this doesn't look like a Mufasa project_config.ini."
        )
    if not cp.has_section("General settings"):
        raise ProjectReconfigureError(
            f"{cfg_path.name} is missing [General settings] section."
        )

    prev_preset = cp.get(
        "create ensemble settings",
        "pose_estimation_body_parts", fallback="<unset>",
    )
    prev_animal_no = cp.get(
        "General settings", "animal_no", fallback="<unset>",
    )

    changes: List[str] = []

    # ------------------ Edit project_config.ini ---------------------- #
    if prev_preset != "user_defined":
        cp.set(
            "create ensemble settings",
            "pose_estimation_body_parts", "user_defined",
        )
        changes.append(
            f"[create ensemble settings] pose_estimation_body_parts: "
            f"{prev_preset} → user_defined"
        )

    new_animal_no = str(animal_cnt)
    if prev_animal_no != new_animal_no:
        cp.set("General settings", "animal_no", new_animal_no)
        changes.append(
            f"[General settings] animal_no: "
            f"{prev_animal_no} → {new_animal_no}"
        )

    with open(cfg_path, "w") as f:
        cp.write(f)

    # ------------------ Write bp_names.csv --------------------------- #
    # Write one-per-line (the get_body_part_configurations code path
    # accepts either, and one-per-line is more readable on inspection).
    with open(bp_csv, "w") as f:
        for bp in body_parts:
            f.write(bp + "\n")

    if previous_body_parts != body_parts:
        changes.append(
            f"body parts: {len(previous_body_parts)} → "
            f"{len(body_parts)} (replaced)"
        )

    return ReconfigureResult(
        config_path=cfg_path,
        bp_names_path=bp_csv,
        config_backup=cfg_backup,
        bp_backup=bp_backup,
        previous_preset=prev_preset,
        previous_animal_no=prev_animal_no,
        previous_body_parts=previous_body_parts,
        new_body_parts=list(body_parts),
        changes=changes,
    )


__all__ = [
    "ProjectReconfigureError",
    "ReconfigureResult",
    "reconfigure_project_user_defined",
]
