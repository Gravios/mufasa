"""Smoke tests for mufasa.cli.migrate_project (patch 122a).

Builds a synthetic SimBA-layout project in a tempdir, runs the
migration, and verifies all files ended up where they should
plus that project.toml has the expected content. Then re-runs
on the migrated project to verify idempotence (returns 0,
no-op).

    PYTHONPATH=. python tests/smoke_migrate_project.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _build_synthetic_legacy_project(root: Path) -> Path:
    """Create a SimBA-layout project under ``root`` and return
    its top directory (the parent of project_folder/).
    """
    proj_top = root / "my_experiment"
    pf = proj_top / "project_folder"
    pf.mkdir(parents=True)
    (proj_top / "models").mkdir()

    # project_config.ini
    (pf / "project_config.ini").write_text(
        "[General settings]\n"
        "project_name = my_experiment\n"
        "animal_no = 1\n"
        "file_type = parquet\n"
        "pose_estimation = user_defined\n"
        "[Outlier settings]\n"
        "movement_criterion = 0.7\n"
        "location_criterion = 1.5\n"
        "[SML settings]\n"
        "no_targets = 2\n"
        "target_name_1 = grooming\n"
    )

    # csv/ subdirs with sample files
    csv_dir = pf / "csv"
    for sub, files in [
        ("input_csv",                        ["sessA.parquet", "sessB.parquet"]),
        ("outlier_corrected_movement",       ["sessA.parquet"]),
        ("outlier_corrected_movement_location", ["sessA.parquet"]),
        ("features_extracted",               ["sessA.parquet"]),
        ("machine_results",                  ["sessA.parquet"]),
        ("targets_inserted",                 ["sessA.parquet"]),
    ]:
        d = csv_dir / sub
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_bytes(b"fake-parquet")

    # videos/
    (pf / "videos").mkdir()
    (pf / "videos" / "sessA.mp4").write_bytes(b"fake-mp4")
    (pf / "videos" / "sessB.mp4").write_bytes(b"fake-mp4")

    # frames/{input,output}/
    (pf / "frames" / "input").mkdir(parents=True)
    (pf / "frames" / "input" / "sessA").mkdir()
    (pf / "frames" / "input" / "sessA" / "0001.png").write_bytes(b"png")
    (pf / "frames" / "output").mkdir(parents=True)
    (pf / "frames" / "output" / "annot_0001.png").write_bytes(b"png")

    # logs/measures/pose_configs/bp_names/
    bp_dir = pf / "logs" / "measures" / "pose_configs" / "bp_names"
    bp_dir.mkdir(parents=True)
    (bp_dir / "layout.csv").write_text(
        "nose\nback1\nback2\nback3\ntailbase\ntailmid\ntailend\n"
    )

    # configs/  (legacy place for misc INI files)
    (pf / "configs").mkdir()
    (pf / "configs" / "extra.ini").write_text("[x]\nv=1\n")

    # models/<classifier>/model.sav
    (proj_top / "models" / "grooming_classifier.sav").write_bytes(b"fake-pickle")

    return proj_top


def _check_plan_then_commit() -> None:
    """Dry-run shows operations without changing disk; --commit
    actually performs them.
    """
    from mufasa.cli.migrate_project import (
        execute_plan, plan_migration,
    )
    from mufasa.project_layout import (
        ProjectPaths, detect_layout, read_project_toml,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        proj_top = _build_synthetic_legacy_project(root)

        # Plan
        plan = plan_migration(proj_top)
        # Should detect the legacy layout
        assert plan.legacy_root == proj_top.resolve()
        assert plan.v1_root == proj_top.resolve()
        # Plan has many ops (csv subdirs, videos, frames, etc)
        assert len(plan) >= 8, f"plan too small: {len(plan)}"
        # Confirm one specific entry: input_csv → sources/pose
        moves = [
            (op.source, op.destination)
            for op in plan.ops if op.kind != "skip"
            and op.kind != "write-toml"
        ]
        sources_pose_targets = [
            d for s, d in moves
            if d is not None and str(d).endswith("sources/pose")
        ]
        assert sources_pose_targets, (
            "expected csv/input_csv → sources/pose mapping"
        )

        # Before commit, disk is unchanged: legacy still there
        assert (
            proj_top / "project_folder" / "project_config.ini"
        ).is_file()
        # And no v1 markers yet
        assert not (proj_top / "project.toml").exists()
        assert not (proj_top / "sources").exists()

        # Commit
        execute_plan(plan, verbose=False)

        # After: detect_layout returns v1
        assert detect_layout(proj_top) == "v1", (
            detect_layout(proj_top)
        )

        paths = ProjectPaths.open(proj_top)
        assert paths.config_file.is_file()

        # Source data ended up in sources/
        assert (paths.sources_pose / "sessA.parquet").is_file()
        assert (paths.sources_pose / "sessB.parquet").is_file()
        assert (paths.sources_videos / "sessA.mp4").is_file()
        assert (paths.sources_videos / "sessB.mp4").is_file()

        # Derived stages ended up in derived/<stage>/imported_<date>/
        derived = paths.derived_dir
        # features
        feat_dirs = list(
            (derived / "features").iterdir()
        )
        assert len(feat_dirs) == 1
        assert feat_dirs[0].name.startswith("imported_")
        assert (feat_dirs[0] / "sessA.parquet").is_file()
        # classifications (was machine_results)
        cls_dirs = list(
            (derived / "classifications").iterdir()
        )
        assert len(cls_dirs) == 1
        assert cls_dirs[0].name.startswith("imported_")
        # outlier_corrected: two subdirs (movement, mvmt_location)
        oc_subdirs = sorted(
            (derived / "outlier_corrected").iterdir(),
        )
        assert len(oc_subdirs) == 2
        names = [d.name for d in oc_subdirs]
        assert any("movement_imported_" in n for n in names)
        assert any(
            "movement_location_imported_" in n for n in names
        )

        # Frames
        extracted_dirs = list(
            (derived / "frames" / "extracted").iterdir(),
        )
        assert len(extracted_dirs) == 1
        annotated_dirs = list(
            (derived / "frames" / "annotated").iterdir(),
        )
        assert len(annotated_dirs) == 1

        # Logs preserved under logs/imported_<date>/
        log_subdirs = list(paths.logs_dir.iterdir())
        assert len(log_subdirs) >= 1
        assert any(
            d.name.startswith("imported_") for d in log_subdirs
        )

        # Models preserved
        assert (
            paths.models_dir / "grooming_classifier.sav"
        ).is_file()

        # project.toml has correct content
        cfg = read_project_toml(paths.config_file)
        assert cfg["project_layout_version"] == 1
        assert cfg["project"]["name"] == "my_experiment"
        assert cfg["project"]["animal_count"] == 1
        # Body parts pulled from bp_names/layout.csv
        assert "body_parts" in cfg["project"]
        assert "nose" in cfg["project"]["body_parts"]
        assert "tailbase" in cfg["project"]["body_parts"]
        # Outlier settings folded under stages
        assert "stages" in cfg
        assert "outlier_correction" in cfg["stages"]
        assert (
            cfg["stages"]["outlier_correction"][
                "movement_criterion"
            ] == 0.7
        )
        # Migration markers
        assert cfg.get("migrated_from") == "simba_legacy"

        # MIGRATION.toml present and lists moves. Read it
        # raw — it's not a project.toml, doesn't have a
        # project_layout_version field, so read_project_toml's
        # version guard would reject it.
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(paths.root / "MIGRATION.toml", "rb") as f:
            manifest = tomllib.load(f)
        assert manifest["moves"]["count"] >= 1
        assert "src" in manifest["moves"]
        assert "dst" in manifest["moves"]


def _check_idempotence() -> None:
    """Running migrate_project on an already-migrated tree is
    a no-op (returns 0, doesn't move anything).
    """
    from mufasa.cli.migrate_project import main as mig_main
    from mufasa.project_layout import detect_layout

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        proj_top = _build_synthetic_legacy_project(root)
        # First migration
        rc = mig_main([str(proj_top), "--commit", "--quiet"])
        assert rc == 0
        assert detect_layout(proj_top) == "v1"

        # Second migration: should detect v1 and bail with rc=0
        rc2 = mig_main([str(proj_top), "--commit", "--quiet"])
        assert rc2 == 0


def _check_dry_run_doesnt_touch_disk() -> None:
    """Without --commit, no files should move.
    """
    from mufasa.cli.migrate_project import main as mig_main
    from mufasa.project_layout import detect_layout

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        proj_top = _build_synthetic_legacy_project(root)

        # Default: dry-run
        rc = mig_main([str(proj_top), "--quiet"])
        assert rc == 0

        # Original layout intact
        assert detect_layout(proj_top) == "legacy"
        assert (
            proj_top / "project_folder" / "csv" / "input_csv"
            / "sessA.parquet"
        ).is_file()
        # No v1 markers created
        assert not (proj_top / "project.toml").exists()
        assert not (proj_top / "sources").exists()


def _check_unknown_path_exits_nonzero() -> None:
    """Pointed at a directory that's neither v1 nor legacy:
    exit code 2.
    """
    from mufasa.cli.migrate_project import main as mig_main
    with tempfile.TemporaryDirectory() as td:
        rc = mig_main([td, "--commit", "--quiet"])
        assert rc == 2, f"expected rc=2 on unknown, got {rc}"


def _check_legacy_pointed_at_project_folder() -> None:
    """User accidentally passes the inner project_folder/ —
    migration should still work.
    """
    from mufasa.cli.migrate_project import main as mig_main
    from mufasa.project_layout import detect_layout
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        proj_top = _build_synthetic_legacy_project(root)
        pf = proj_top / "project_folder"
        # Pass the inner dir — LegacyProjectPaths.open handles
        # both call shapes. detect_layout should also work.
        assert detect_layout(pf) == "legacy"
        rc = mig_main([str(pf), "--commit", "--quiet"])
        assert rc == 0
        # The v1 root should be proj_top (the parent of pf)
        assert detect_layout(proj_top) == "v1"


def main() -> int:
    _check_plan_then_commit()
    _check_idempotence()
    _check_dry_run_doesnt_touch_disk()
    _check_unknown_path_exits_nonzero()
    _check_legacy_pointed_at_project_folder()
    print("smoke_migrate_project: 5/5 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
