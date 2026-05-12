"""Smoke-test: empty classifier list is accepted by
ProjectConfigCreator (v1 layout, patch 122d).

Regression guard for the Qt CreateProjectDialog change that makes
classifier names optional at project creation.

Run:

    PYTHONPATH=. python tests/smoke_empty_classifier.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    try:
        from mufasa.utils.config_creator import ProjectConfigCreator

        # Create a v1 project with target_list=[] — the change
        # under test.
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
        assert config_path.name == "project.toml", (
            f"v1 layout writes project.toml; got "
            f"{config_path.name!r}"
        )

        # v1 skeleton present
        root = creator.project_root
        for sub in (
            "sources/pose", "sources/videos", "sources/annotations",
            "derived", "models", "logs",
        ):
            assert (root / sub).is_dir(), (
                f"v1 skeleton missing: {root / sub}"
            )

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Layout version + name top-level
        assert data.get("project_layout_version") == 1, (
            f"expected project_layout_version=1, got "
            f"{data.get('project_layout_version')!r}"
        )
        assert data.get("project_name") == "empty_clf_test"

        # Classifier targets empty
        targets = data.get("classifiers", {}).get("targets")
        assert targets == [], (
            f"classifier targets should be [] for empty list, got "
            f"{targets!r}"
        )

        # Pose section was populated from the preset
        bps = data.get("pose", {}).get("body_parts")
        assert isinstance(bps, list) and bps, (
            f"body_parts should be a non-empty list, got {bps!r}"
        )

        print("smoke_empty_classifier: 1/1 cases passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
