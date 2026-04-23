"""Smoke-test for the DLC autodetect + project reconfigure modules.

Covers the pure-Python logic that the Qt dialogs drive. Runs headless
(no PySide6 needed).

    python tests/smoke_dlc_autodetect.py
"""
from __future__ import annotations

import configparser
import csv
import shutil
import sys
import tempfile
from pathlib import Path


def build_fake_project(tmp: Path) -> Path:
    """Build a minimal project-tree layout that the reconfigure helper
    understands. Returns the path to project_config.ini."""
    project = tmp / "test_project" / "project_folder"
    (project / "logs" / "measures" / "pose_configs" / "bp_names").mkdir(
        parents=True,
    )
    cfg = project / "project_config.ini"
    cfg.write_text(
        "[General settings]\n"
        "animal_no = 1\n"
        "project_name = test\n"
        "[create ensemble settings]\n"
        "pose_estimation_body_parts = 9\n"
        "[Multi animal IDs]\n"
        "id_list = \n"
    )
    bp_names = (project / "logs/measures/pose_configs"
                / "bp_names/project_bp_names.csv")
    bp_names.write_text(
        "Mouse1_left_ear,Mouse1_right_ear,Mouse1_nose,,,,,\n"
    )
    return cfg


def build_fake_dlc_csv(tmp: Path, bodyparts: list) -> Path:
    """Write a minimally valid DLC CSV with the given body parts.
    Returns the path."""
    path = tmp / "fake_dlc.csv"
    scorer = "DLC_Resnet50_proj"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scorer"] + [scorer] * (len(bodyparts) * 3))
        w.writerow(["bodyparts"] + [bp for bp in bodyparts for _ in range(3)])
        w.writerow(["coords"] + ["x", "y", "likelihood"] * len(bodyparts))
        w.writerow(["0"] + ["0.0"] * (len(bodyparts) * 3))
    return path


def main() -> int:
    from mufasa.pose_importers.dlc_autodetect import (
        DLCAutodetectError,
        extract_bodyparts,
        extract_bodyparts_from_csv,
    )
    from mufasa.utils.project_reconfigure import (
        ProjectReconfigureError,
        reconfigure_project_user_defined,
    )

    tmp = Path(tempfile.mkdtemp())
    try:
        # ------------------------------------------------------------ #
        # Case 1: CSV autodetect preserves order
        # ------------------------------------------------------------ #
        bodyparts = [
            "nose", "headmid", "ear_left", "ear_right", "neck",
            "center", "lateral_left", "lateral_right",
            "back1", "back2", "back3", "back4",
            "tailbase", "tailmid", "tailend",
        ]
        csv_path = build_fake_dlc_csv(tmp, bodyparts)
        detected = extract_bodyparts_from_csv(csv_path)
        assert detected == bodyparts, f"case 1 order: {detected}"

        # ------------------------------------------------------------ #
        # Case 2: dispatch (.csv / .h5 / other)
        # ------------------------------------------------------------ #
        assert extract_bodyparts(csv_path) == bodyparts
        # .h5 dispatch not covered here (needs pytables)
        try:
            extract_bodyparts(tmp / "nonsense.txt")
        except DLCAutodetectError:
            pass
        else:
            raise AssertionError("case 2: expected DLCAutodetectError for .txt")

        # ------------------------------------------------------------ #
        # Case 3: CSV with too few rows
        # ------------------------------------------------------------ #
        bad = tmp / "too_short.csv"
        bad.write_text("only,one,row\n")
        try:
            extract_bodyparts_from_csv(bad)
        except DLCAutodetectError:
            pass
        else:
            raise AssertionError("case 3: expected DLCAutodetectError")

        # ------------------------------------------------------------ #
        # Case 4: end-to-end reconfigure against a fake project
        # ------------------------------------------------------------ #
        cfg = build_fake_project(tmp)
        bp_names = (cfg.parent / "logs/measures/pose_configs"
                    / "bp_names/project_bp_names.csv")

        result = reconfigure_project_user_defined(cfg, detected)
        assert result.new_body_parts == bodyparts
        assert result.previous_preset == "9"
        assert result.config_backup.exists()
        assert result.bp_backup.exists()

        cp = configparser.ConfigParser()
        cp.read(cfg)
        assert cp["create ensemble settings"]["pose_estimation_body_parts"] \
            == "user_defined"
        assert cp["General settings"]["animal_no"] == "1"

        # bp_names is now one-per-line
        lines = [ln for ln in bp_names.read_text().splitlines() if ln]
        assert lines == bodyparts, f"case 4 bp_names: {lines}"

        # Backup preserves the original comma-separated row
        assert "Mouse1_left_ear,Mouse1_right_ear" in \
            result.bp_backup.read_text()

        # ------------------------------------------------------------ #
        # Case 5: missing bp_names dir rejected clearly
        # ------------------------------------------------------------ #
        bad_project = tmp / "no_bpdir"
        bad_project.mkdir()
        bad_cfg = bad_project / "project_config.ini"
        bad_cfg.write_text(cfg.read_text())
        try:
            reconfigure_project_user_defined(bad_cfg, bodyparts)
        except ProjectReconfigureError:
            pass
        else:
            raise AssertionError(
                "case 5: expected ProjectReconfigureError"
            )

        # ------------------------------------------------------------ #
        # Case 6: empty body-parts list rejected
        # ------------------------------------------------------------ #
        try:
            reconfigure_project_user_defined(cfg, [])
        except ProjectReconfigureError:
            pass
        else:
            raise AssertionError(
                "case 6: expected ProjectReconfigureError for empty bps"
            )

        print("smoke_dlc_autodetect: 6/6 cases passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
