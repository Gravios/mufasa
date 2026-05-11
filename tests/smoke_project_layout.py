"""Smoke tests for mufasa.project_layout (patch 122a).

Covers:
  - PROJECT_LAYOUT_VERSION / Stages / SmoothingFlavors exist
  - generate_run_id format and is_run_id round-trip
  - ProjectPaths properties and ensure_skeleton
  - stage_run_dir / smoothed_run_dir create dirs and use run-id
  - list_runs / latest_run filter for valid run-ids only
  - project.toml write+read round-trip
  - run.toml write+read round-trip
  - detect_layout returns v1/legacy/unknown correctly
  - ProjectLayoutError fires on version-too-new

    PYTHONPATH=. python tests/smoke_project_layout.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _check_constants() -> None:
    import mufasa.project_layout as pl
    assert isinstance(pl.PROJECT_LAYOUT_VERSION, int)
    assert pl.PROJECT_LAYOUT_VERSION >= 1
    assert pl.PROJECT_CONFIG_FILENAME == "project.toml"
    assert pl.RUN_PROVENANCE_FILENAME == "run.toml"
    # Canonical stage names
    assert pl.Stages.SMOOTHED == "smoothed"
    assert pl.Stages.FEATURES == "features"
    assert pl.Stages.CLASSIFICATIONS == "classifications"
    # Smoothing flavors
    assert pl.SmoothingFlavors.KALMAN_V2 == "kalman_v2"


def _check_run_id() -> None:
    from mufasa.project_layout import generate_run_id, is_run_id
    rid = generate_run_id()
    assert is_run_id(rid), rid
    # Two consecutive ids differ (hex suffix randomized)
    rid2 = generate_run_id()
    assert rid != rid2 or rid[-6:] != rid2[-6:]
    # Format checks: 8 + 6 + 6 = 20 hex/digit/dash chars (15+3 sep)
    parts = rid.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 8
    assert len(parts[1]) == 6
    assert len(parts[2]) == 6
    # Not run-ids
    assert not is_run_id("imported_20240101")
    assert not is_run_id("scratch")
    assert not is_run_id("20240101-120000")  # missing suffix


def _check_project_paths_skeleton() -> None:
    from mufasa.project_layout import ProjectPaths
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        paths = ProjectPaths(root)
        paths.ensure_skeleton()
        assert paths.sources_dir.is_dir()
        assert paths.sources_videos.is_dir()
        assert paths.sources_pose.is_dir()
        assert paths.sources_annotations.is_dir()
        assert paths.derived_dir.is_dir()
        assert paths.models_dir.is_dir()
        assert paths.logs_dir.is_dir()
        # Idempotent — second call doesn't error
        paths.ensure_skeleton()


def _check_stage_run_dir() -> None:
    from mufasa.project_layout import (
        ProjectPaths, Stages, SmoothingFlavors, is_run_id,
    )
    with tempfile.TemporaryDirectory() as td:
        paths = ProjectPaths(Path(td))
        # Auto-generated run id
        d = paths.stage_run_dir(Stages.FEATURES)
        assert d.is_dir()
        assert is_run_id(d.name)
        assert d.parent == paths.derived_dir / Stages.FEATURES

        # Flavored
        d2 = paths.smoothed_run_dir(SmoothingFlavors.KALMAN_V2)
        assert d2.is_dir()
        assert d2.parent.name == SmoothingFlavors.KALMAN_V2
        assert (
            d2.parent.parent
            == paths.derived_dir / Stages.SMOOTHED
        )

        # Explicit run id
        d3 = paths.stage_run_dir(
            Stages.FEATURES, run_id="20240101-120000-abc123",
        )
        assert d3.name == "20240101-120000-abc123"
        assert d3.is_dir()


def _check_list_runs() -> None:
    from mufasa.project_layout import (
        ProjectPaths, Stages, SmoothingFlavors,
    )
    with tempfile.TemporaryDirectory() as td:
        paths = ProjectPaths(Path(td))
        # No runs yet
        assert paths.list_runs(Stages.SMOOTHED) == []
        assert paths.latest_run(Stages.SMOOTHED) is None

        # Make three runs with explicit run ids in non-sorted order
        for rid in [
            "20240301-120000-ccdddd",
            "20240101-120000-aaaaaa",
            "20240201-120000-bbbbbb",
        ]:
            paths.smoothed_run_dir(
                SmoothingFlavors.KALMAN_V2, run_id=rid,
            )
        # Plus a non-run-id sibling that should be filtered out
        (paths.derived_dir / Stages.SMOOTHED
         / SmoothingFlavors.KALMAN_V2 / "scratch").mkdir()

        runs = paths.list_runs(
            Stages.SMOOTHED, flavor=SmoothingFlavors.KALMAN_V2,
        )
        assert len(runs) == 3
        # Sorted lexically == chronologically
        assert runs[0].name.startswith("20240101")
        assert runs[2].name.startswith("20240301")
        # latest_run picks the most recent
        latest = paths.latest_run(
            Stages.SMOOTHED, flavor=SmoothingFlavors.KALMAN_V2,
        )
        assert latest is not None
        assert latest.name.startswith("20240301")


def _check_project_toml_roundtrip() -> None:
    from mufasa.project_layout import (
        read_project_toml, write_project_toml,
        ProjectLayoutError,
    )
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "project.toml"
        original = {
            "project_layout_version": 1,
            "project": {
                "name": "MyExperiment",
                "animal_count": 1,
                "body_parts": ["nose", "back1", "tailbase"],
                "tracker_type": "deeplabcut",
            },
            "stages": {
                "smoothed_kalman_v2": {
                    "em_max_iter": 20,
                    "likelihood_threshold": 0.5,
                    "with_drift": True,
                },
            },
        }
        write_project_toml(cfg_path, original)
        loaded = read_project_toml(cfg_path)
        assert loaded == original, (
            f"roundtrip diff:\nwrote: {original}\nread:  {loaded}"
        )

        # Version guard
        bad = dict(original)
        bad["project_layout_version"] = 9999
        write_project_toml(cfg_path, bad)
        try:
            read_project_toml(cfg_path)
        except ProjectLayoutError as e:
            assert "newer than this Mufasa supports" in str(e)
        else:
            raise AssertionError(
                "expected ProjectLayoutError on future version"
            )

        # Missing version → also raises
        bad2 = dict(original)
        del bad2["project_layout_version"]
        write_project_toml(cfg_path, bad2)
        try:
            read_project_toml(cfg_path)
        except ProjectLayoutError:
            pass
        else:
            raise AssertionError(
                "expected ProjectLayoutError on missing version"
            )


def _check_run_toml_roundtrip() -> None:
    from mufasa.project_layout import (
        ProjectPaths, Stages, SmoothingFlavors,
        write_run_toml, read_run_toml,
        RUN_PROVENANCE_FILENAME,
    )
    with tempfile.TemporaryDirectory() as td:
        paths = ProjectPaths(Path(td))
        rdir = paths.smoothed_run_dir(SmoothingFlavors.KALMAN_V2)
        write_run_toml(
            rdir / RUN_PROVENANCE_FILENAME,
            stage="smoothed.kalman_v2",
            run_id=rdir.name,
            inputs=["sources/pose/session_a.parquet"],
            params={
                "em_max_iter": 20,
                "with_drift": True,
                "orient_drift_segments": ["body", "head"],
            },
            results={
                "n_sessions": 67,
                "em_converged": False,
                "em_iterations_used": 20,
            },
            mufasa_version="1.2.3",
        )
        rt = read_run_toml(rdir / RUN_PROVENANCE_FILENAME)
        assert rt["run_id"] == rdir.name
        assert rt["stage"] == "smoothed.kalman_v2"
        assert rt["mufasa_version"] == "1.2.3"
        assert rt["params"]["em_max_iter"] == 20
        assert rt["params"]["with_drift"] is True
        assert rt["params"]["orient_drift_segments"] == [
            "body", "head",
        ]
        assert rt["results"]["n_sessions"] == 67
        assert rt["inputs"]["files"] == [
            "sources/pose/session_a.parquet",
        ]


def _check_detect_layout() -> None:
    from mufasa.project_layout import (
        detect_layout, write_project_toml,
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Empty dir
        assert detect_layout(root) == "unknown"

        # Legacy: project_folder/project_config.ini
        legacy_dir = root / "legacy_proj"
        (legacy_dir / "project_folder").mkdir(parents=True)
        (legacy_dir / "project_folder"
         / "project_config.ini").write_text(
            "[General settings]\nproject_name=Test\n",
        )
        assert detect_layout(legacy_dir) == "legacy"
        # Pointed at project_folder/ directly
        assert detect_layout(
            legacy_dir / "project_folder",
        ) == "legacy"

        # v1: project.toml
        v1_dir = root / "v1_proj"
        v1_dir.mkdir()
        write_project_toml(
            v1_dir / "project.toml",
            {"project_layout_version": 1},
        )
        assert detect_layout(v1_dir) == "v1"


def _check_projectpaths_open_validates() -> None:
    from mufasa.project_layout import (
        ProjectPaths, ProjectLayoutError,
    )
    with tempfile.TemporaryDirectory() as td:
        # ProjectPaths.open refuses non-v1 dirs
        try:
            ProjectPaths.open(Path(td))
        except ProjectLayoutError as e:
            assert "expected v1" in str(e), str(e)
        else:
            raise AssertionError(
                "expected ProjectLayoutError on unknown layout"
            )


def main() -> int:
    _check_constants()
    _check_run_id()
    _check_project_paths_skeleton()
    _check_stage_run_dir()
    _check_list_runs()
    _check_project_toml_roundtrip()
    _check_run_toml_roundtrip()
    _check_detect_layout()
    _check_projectpaths_open_validates()
    print("smoke_project_layout: 9/9 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
