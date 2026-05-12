"""
tests/smoke_config_creator_v1.py
================================

Patch 122d: behavioral smoke test for the rewritten
:class:`mufasa.utils.config_creator.ProjectConfigCreator` that
produces only v1 layout (``project.toml`` + ``sources/``,
``derived/``, ``models/``, ``logs/``).

Coverage:

* v1 skeleton lands at the expected paths
  (``ProjectPaths.ensure_skeleton`` was actually called).
* ``project.toml`` is valid TOML, has
  ``project_layout_version = 1``, populated ``[pose]`` with
  body_parts list, populated ``[classifiers].targets`` (possibly
  empty), and ``[outlier_settings]`` defaults.
* Preset path: reading from
  ``pose_configurations/bp_names/bp_names.csv`` row N gives a
  non-empty body_parts list.
* Explicit ``body_parts=`` override path: the provided list is
  used verbatim, ``body_part_config_idx`` is ignored.
* Name validation: refuses shell-unfriendly characters.
* Refuses to clobber an existing project.toml.
* Multi-animal: ``animal_ids`` length matches ``animal_cnt``.
* No legacy artifacts: ``project_folder/``, ``csv/``,
  ``project_config.ini``, ``logs/measures/`` are NOT created.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> int:
    # ----------------------------------------------------------
    # 1. Preset path: layout, skeleton, project.toml shape
    # ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        creator = ProjectConfigCreator(
            project_path=str(tmp),
            project_name="exp_preset",
            target_list=["Attack", "Sniffing"],
            pose_estimation_bp_cnt="7",
            body_part_config_idx=1,  # row 1 in bp_names.csv
            animal_cnt=1,
            file_type="csv",
        )
        root = creator.project_root
        cfg = Path(creator.config_path)

        check("config_path points to project.toml", cfg.name == "project.toml")
        check("config_path is inside project_root", cfg.parent == root)
        check("project.toml exists", cfg.is_file())

        # Skeleton
        for sub in (
            "sources",
            "sources/videos",
            "sources/pose",
            "sources/annotations",
            "derived",
            "models",
            "logs",
        ):
            check(
                f"skeleton creates {sub}/",
                (root / sub).is_dir(),
            )

        # No legacy artifacts at the root
        for legacy in (
            "project_folder",
            "csv",
            "project_config.ini",
        ):
            check(
                f"no legacy artifact: {legacy}",
                not (root / legacy).exists(),
                detail=f"unexpected: {root / legacy}",
            )
        # Legacy 'logs/measures' substructure must not exist
        check(
            "no legacy logs/measures/ substructure",
            not (root / "logs" / "measures").exists(),
        )

        data = _toml(cfg)
        check(
            "project_layout_version == 1",
            data.get("project_layout_version") == 1,
        )
        check(
            "project_name persisted",
            data.get("project_name") == "exp_preset",
        )
        check(
            "mufasa_version field present",
            "mufasa_version" in data,
        )
        check(
            "[pose].body_parts is non-empty list",
            isinstance(data.get("pose", {}).get("body_parts"), list)
            and len(data["pose"]["body_parts"]) > 0,
        )
        check(
            "[pose].animal_count == 1",
            data["pose"]["animal_count"] == 1,
        )
        check(
            "[pose].file_type == 'csv'",
            data["pose"]["file_type"] == "csv",
        )
        check(
            "[pose].pose_config_code persisted",
            data["pose"]["pose_config_code"] == "7",
        )
        check(
            "[pose].animal_ids length == animal_count",
            len(data["pose"]["animal_ids"]) == 1,
        )
        check(
            "[classifiers].targets persisted",
            data["classifiers"]["targets"] == ["Attack", "Sniffing"],
        )
        check(
            "[outlier_settings] defaults written",
            data["outlier_settings"]["movement_criterion"] == "NaN"
            and data["outlier_settings"]["location_criterion"] == "NaN",
        )

    # ----------------------------------------------------------
    # 2. Explicit body_parts override
    # ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        custom_bps = ["snout", "left_ear", "right_ear", "tail_base"]
        creator = ProjectConfigCreator(
            project_path=str(tmp),
            project_name="exp_custom",
            target_list=[],                   # empty — also allowed
            pose_estimation_bp_cnt="user_defined",
            body_part_config_idx=0,           # ignored when body_parts given
            animal_cnt=2,
            file_type="parquet",
            body_parts=custom_bps,
        )
        data = _toml(Path(creator.config_path))
        check(
            "explicit body_parts used verbatim",
            data["pose"]["body_parts"] == custom_bps,
        )
        check(
            "explicit body_parts: file_type honoured",
            data["pose"]["file_type"] == "parquet",
        )
        check(
            "empty target_list lands as []",
            data["classifiers"]["targets"] == [],
        )
        check(
            "[pose].animal_count == 2",
            data["pose"]["animal_count"] == 2,
        )
        check(
            "animal_ids has 2 entries",
            len(data["pose"]["animal_ids"]) == 2,
        )

    # ----------------------------------------------------------
    # 3. Name validation
    # ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for bad_name in ("", "has spaces", "with/slash", "weird:char"):
            raised = False
            try:
                ProjectConfigCreator(
                    project_path=str(tmp),
                    project_name=bad_name,
                    target_list=[],
                    pose_estimation_bp_cnt="7",
                    body_part_config_idx=1,
                    animal_cnt=1,
                    file_type="csv",
                )
            except ValueError:
                raised = True
            check(
                f"rejects shell-unfriendly project_name={bad_name!r}",
                raised,
            )

    # ----------------------------------------------------------
    # 4. Refuses to clobber an existing project
    # ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ProjectConfigCreator(
            project_path=str(tmp),
            project_name="dup",
            target_list=[],
            pose_estimation_bp_cnt="7",
            body_part_config_idx=1,
            animal_cnt=1,
            file_type="csv",
        )
        raised = False
        try:
            ProjectConfigCreator(
                project_path=str(tmp),
                project_name="dup",  # same name
                target_list=[],
                pose_estimation_bp_cnt="7",
                body_part_config_idx=1,
                animal_cnt=1,
                file_type="csv",
            )
        except FileExistsError:
            raised = True
        check("refuses to clobber existing project.toml", raised)

    # ----------------------------------------------------------
    # 5. Bad file_type and animal_cnt
    # ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for bad_ft in ("xls", "json", ""):
            raised = False
            try:
                ProjectConfigCreator(
                    project_path=str(tmp),
                    project_name=f"ft_{bad_ft or 'empty'}",
                    target_list=[],
                    pose_estimation_bp_cnt="7",
                    body_part_config_idx=1,
                    animal_cnt=1,
                    file_type=bad_ft,
                )
            except ValueError:
                raised = True
            check(
                f"rejects file_type={bad_ft!r}",
                raised,
            )
        raised = False
        try:
            ProjectConfigCreator(
                project_path=str(tmp),
                project_name="ac_zero",
                target_list=[],
                pose_estimation_bp_cnt="7",
                body_part_config_idx=1,
                animal_cnt=0,
                file_type="csv",
            )
        except ValueError:
            raised = True
        check("rejects animal_cnt=0", raised)

    print(
        f"smoke_config_creator_v1: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
