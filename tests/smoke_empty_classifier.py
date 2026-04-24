"""Smoke-test: empty classifier list is accepted by
ProjectConfigCreator.

Regression guard for the Qt CreateProjectDialog change that makes
classifier names optional at project creation.

Run:

    python tests/smoke_empty_classifier.py
"""
from __future__ import annotations

import configparser
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    try:
        from mufasa.utils.config_creator import ProjectConfigCreator

        # Create a project with target_list=[] — the change under test.
        creator = ProjectConfigCreator(
            project_path=str(tmp),
            project_name="empty_clf_test",
            target_list=[],                   # <-- the tested shape
            pose_estimation_bp_cnt="7",
            body_part_config_idx=1,
            animal_cnt=1,
            file_type="csv",
        )
        config_path = Path(creator.config_path)
        assert config_path.is_file(), (
            f"expected {config_path} to be created"
        )

        cp = configparser.ConfigParser()
        cp.read(config_path)

        # TARGET_CNT = "no_targets" (see mufasa.utils.enums.ConfigKey).
        # configparser lowercases keys by default.
        target_cnt = cp.get("SML settings", "no_targets", fallback=None)
        assert target_cnt == "0", (
            f"target count should be '0' for empty list, got "
            f"{target_cnt!r}"
        )
        offenders = [
            k for k in cp["SML settings"]
            if k.startswith("target_name_")
        ]
        assert not offenders, (
            f"unexpected classifier name keys with empty list: "
            f"{offenders}"
        )

        print("smoke_empty_classifier: 1/1 cases passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
