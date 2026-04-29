"""Tests for the CSV-to-parquet project migration tool.

Builds a synthetic project_folder structure, runs the migration,
and verifies:
- All pose-data CSVs are converted
- Logs/measures CSVs are NOT touched
- project_config.ini is updated
- Original CSVs survive when --delete-csv isn't passed
- Verification catches schema mismatches
- Dry run makes no changes

Most cases need pyarrow (or fastparquet) since the migration
writes parquet files. If neither is available, structural
cases still run; conversion cases are skipped.

    PYTHONPATH=. python tests/smoke_csv_to_parquet.py
"""
from __future__ import annotations

import configparser
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("mufasa/tools").resolve()))
import csv_to_parquet


def _has_parquet_engine() -> bool:
    """Probe whether pandas can write parquet (needs pyarrow or fastparquet)."""
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa
        return True
    except ImportError:
        pass
    return False


def _build_synthetic_project(root: Path) -> Path:
    """Build a Mufasa-like project tree under ``root`` with CSVs
    in pose-data dirs and unrelated CSVs in logs/. Returns the
    project_folder path."""
    pf = root / "test_project" / "project_folder"
    csv_dir = pf / "csv"
    for sub in csv_to_parquet.POSE_DATA_DIRS:
        (csv_dir / sub).mkdir(parents=True)
    (pf / "logs" / "measures").mkdir(parents=True)
    (pf / "videos").mkdir()

    # Pose data CSVs (should be converted)
    df = pd.DataFrame({
        "Nose_x": list(range(100)),
        "Nose_y": list(range(100, 200)),
        "Nose_p": [0.99] * 100,
    })
    df.to_csv(csv_dir / "input_csv" / "Video_A.csv")
    df.to_csv(csv_dir / "input_csv" / "Video_B.csv")
    df.to_csv(csv_dir / "outlier_corrected_movement_location" / "Video_A.csv")
    df.to_csv(csv_dir / "features_extracted" / "Video_A.csv")

    # Hidden file (should NOT be converted)
    (csv_dir / "input_csv" / ".DS_Store").write_text("junk")

    # Logs CSVs (should NOT be converted)
    pd.DataFrame({"Video": ["A"], "fps": [30]}).to_csv(
        pf / "logs" / "video_info.csv"
    )
    pd.DataFrame({"roi": ["left"]}).to_csv(
        pf / "logs" / "measures" / "rectangles_2026.csv"
    )

    # Project config
    cfg = configparser.ConfigParser()
    cfg["General settings"] = {
        "project_name": "test",
        "file_type": "csv",
    }
    config_path = pf / "project_config.ini"
    with open(config_path, "w") as f:
        cfg.write(f)
    return pf


def _list_files_recursive(directory: Path, suffix: str) -> list:
    return sorted(
        [str(p) for p in directory.rglob(f"*{suffix}") if p.is_file()]
    )


def main() -> int:
    # ------------------------------------------------------------------ #
    # Case 1: discover_files finds CSVs in pose-data dirs only
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        pf = _build_synthetic_project(Path(td))
        files = csv_to_parquet.discover_files(str(pf))
        assert len(files) == 4, (
            f"Expected 4 pose-data CSVs, got {len(files)}: {files}"
        )
        # No logs CSVs included
        for f in files:
            assert "/logs/" not in f, (
                f"Logs file {f} should not be in conversion list"
            )
        # No hidden files
        for f in files:
            assert not os.path.basename(f).startswith("."), (
                f"Hidden file {f} should be skipped"
            )

    # ------------------------------------------------------------------ #
    # Case 2: dry-run makes no changes
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        pf = _build_synthetic_project(Path(td))
        config_path = pf / "project_config.ini"
        before_csvs = _list_files_recursive(pf, ".csv")
        before_parquets = _list_files_recursive(pf, ".parquet")
        rc = csv_to_parquet.main([
            "--config", str(config_path), "--dry-run",
        ])
        assert rc == 0
        after_csvs = _list_files_recursive(pf, ".csv")
        after_parquets = _list_files_recursive(pf, ".parquet")
        assert before_csvs == after_csvs, "Dry run modified CSVs!"
        assert before_parquets == after_parquets == [], (
            "Dry run created parquets!"
        )
        # Config file_type unchanged
        cfg = configparser.ConfigParser()
        cfg.read(str(config_path))
        assert cfg["General settings"]["file_type"] == "csv"

    # The remaining cases all run the actual conversion. If no
    # parquet engine is available, skip them with a note (pyarrow
    # is in Mufasa's actual dependency chain — `read_df` uses it —
    # so on the user's workstation these will run).
    if not _has_parquet_engine():
        print(
            "smoke_csv_to_parquet: 2/2 structural cases passed "
            "(remaining cases need pyarrow/fastparquet — install for "
            "full coverage)"
        )
        return 0

    # ------------------------------------------------------------------ #
    # Case 3: actual run converts files, verifies, updates config,
    # keeps original CSVs by default
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        pf = _build_synthetic_project(Path(td))
        config_path = pf / "project_config.ini"
        rc = csv_to_parquet.main(["--config", str(config_path)])
        assert rc == 0
        # Should have 4 new parquets in pose-data dirs
        parquets = _list_files_recursive(pf, ".parquet")
        assert len(parquets) == 4, (
            f"Expected 4 parquets, got {len(parquets)}: {parquets}"
        )
        # Originals still there
        csvs_in_pose = [
            p for p in _list_files_recursive(pf, ".csv")
            if "/logs/" not in p and "project_config" not in p
        ]
        assert len(csvs_in_pose) == 4, (
            f"Original CSVs should be kept; got {csvs_in_pose}"
        )
        # Logs CSVs untouched (2 of them)
        logs_csvs = [
            p for p in _list_files_recursive(pf, ".csv")
            if "/logs/" in p
        ]
        assert len(logs_csvs) == 2
        # Config updated
        cfg = configparser.ConfigParser()
        cfg.read(str(config_path))
        assert cfg["General settings"]["file_type"] == "parquet"

    # ------------------------------------------------------------------ #
    # Case 4: --delete-csv removes originals after successful run
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        pf = _build_synthetic_project(Path(td))
        config_path = pf / "project_config.ini"
        rc = csv_to_parquet.main([
            "--config", str(config_path), "--delete-csv",
        ])
        assert rc == 0
        # Pose-data CSVs gone
        csvs_in_pose = [
            p for p in _list_files_recursive(pf, ".csv")
            if "/logs/" not in p and "project_config" not in p
        ]
        assert csvs_in_pose == [], (
            f"--delete-csv should remove pose CSVs; remaining: "
            f"{csvs_in_pose}"
        )
        # Logs CSVs still there
        logs_csvs = [
            p for p in _list_files_recursive(pf, ".csv")
            if "/logs/" in p
        ]
        assert len(logs_csvs) == 2

    # ------------------------------------------------------------------ #
    # Case 5: verify_parquet catches mismatched schemas
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Build matching files
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_path = str(td / "data.csv")
        pq_path = str(td / "data.parquet")
        df.to_csv(csv_path)
        df.to_parquet(pq_path)
        assert csv_to_parquet.verify_parquet(csv_path, pq_path, has_index=True)
        # Now a parquet with different columns
        df2 = pd.DataFrame({"a": [1, 2], "X": [99, 99]})
        df2.to_parquet(pq_path)
        assert not csv_to_parquet.verify_parquet(
            csv_path, pq_path, has_index=True
        )

    # ------------------------------------------------------------------ #
    # Case 6: parquet preserves the data values (round-trip check)
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        pf = _build_synthetic_project(Path(td))
        config_path = pf / "project_config.ini"
        rc = csv_to_parquet.main(["--config", str(config_path)])
        assert rc == 0
        # Pick one parquet and verify values match the CSV
        csv_path = str(pf / "csv" / "input_csv" / "Video_A.csv")
        pq_path = str(pf / "csv" / "input_csv" / "Video_A.parquet")
        csv_df = pd.read_csv(csv_path, index_col=0)
        pq_df = pd.read_parquet(pq_path)
        import numpy as np
        assert np.array_equal(csv_df.values, pq_df.values), (
            "Round-trip values should match"
        )

    # ------------------------------------------------------------------ #
    # Case 7: non-existent config returns 1 with clear error
    # ------------------------------------------------------------------ #
    rc = csv_to_parquet.main(["--config", "/nonexistent/config.ini"])
    assert rc == 1

    # ------------------------------------------------------------------ #
    # Case 8: already-parquet project is a no-op
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        pf = _build_synthetic_project(Path(td))
        config_path = pf / "project_config.ini"
        # Set file_type=parquet manually
        cfg = configparser.ConfigParser()
        cfg.read(str(config_path))
        cfg["General settings"]["file_type"] = "parquet"
        with open(config_path, "w") as f:
            cfg.write(f)

        rc = csv_to_parquet.main(["--config", str(config_path)])
        assert rc == 0
        # Should not have created any parquets
        parquets = _list_files_recursive(pf, ".parquet")
        assert parquets == [], (
            f"Already-parquet project should not create new parquets; "
            f"got {parquets}"
        )

    print("smoke_csv_to_parquet: 8/8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
