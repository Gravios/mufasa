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

    # ------------------------------------------------------------------ #
    # Case 1.5: _convert_one worker is at module top level (required
    # for ProcessPoolExecutor pickling) and accepts a tuple
    # ------------------------------------------------------------------ #
    assert hasattr(csv_to_parquet, "_convert_one"), (
        "_convert_one must exist at module level for ProcessPoolExecutor"
    )
    import inspect
    sig = inspect.signature(csv_to_parquet._convert_one)
    params = list(sig.parameters)
    assert len(params) == 1, (
        f"_convert_one should take a single tuple arg (got {params})"
    )

    # ------------------------------------------------------------------ #
    # Case 1.6: convert_csv_to_parquet uses pyarrow.csv directly
    # when available (fast path), with pandas fallback. We check
    # this structurally since pyarrow may not be installed in
    # sandbox.
    # ------------------------------------------------------------------ #
    import ast
    src_csv2pq = Path("mufasa/tools/csv_to_parquet.py").read_text()
    tree_csv2pq = ast.parse(src_csv2pq)
    convert_fn = next(
        n for n in tree_csv2pq.body
        if isinstance(n, ast.FunctionDef) and n.name == "convert_csv_to_parquet"
    )
    convert_src = ast.unparse(convert_fn)
    assert "pyarrow" in convert_src and "csv" in convert_src, (
        "convert_csv_to_parquet should import pyarrow.csv for the "
        "fast path"
    )
    assert "low_memory=False" in convert_src, (
        "Pandas fallback should use low_memory=False to avoid the "
        "DtypeWarning spam from chunked dtype inference"
    )

    # ------------------------------------------------------------------ #
    # Case 1.7: convert_csv_to_parquet detects multi-row headers
    # (input_csv/ files have 3-row IMPORTED_POSE / bodypart_coord
    # headers from Mufasa's preserved-DLC-style storage)
    # ------------------------------------------------------------------ #
    assert hasattr(csv_to_parquet, "_detect_header_rows"), (
        "_detect_header_rows helper must exist for multi-index "
        "input_csv/ files"
    )
    # Test it directly on synthetic CSVs
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        flat_path = f.name
        f.write(",Nose_x,Nose_y,Nose_p\n0,1.5,2.5,0.99\n")
    n = csv_to_parquet._detect_header_rows(flat_path)
    assert n == 1, f"Flat CSV should detect 1 header row, got {n}"
    os.unlink(flat_path)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        multi_path = f.name
        f.write(",IMPORTED_POSE,IMPORTED_POSE,IMPORTED_POSE\n")
        f.write(",IMPORTED_POSE,IMPORTED_POSE,IMPORTED_POSE\n")
        f.write(",nose_x,nose_y,nose_p\n")
        f.write("0,1.5,2.5,0.99\n")
    n = csv_to_parquet._detect_header_rows(multi_path)
    assert n == 3, (
        f"Multi-row header CSV should detect 3 header rows, got {n}"
    )
    os.unlink(multi_path)

    # The remaining cases all run the actual conversion. If no
    # parquet engine is available, skip them with a note (pyarrow
    # is in Mufasa's actual dependency chain — `read_df` uses it —
    # so on the user's workstation these will run).
    if not _has_parquet_engine():
        print(
            "smoke_csv_to_parquet: 5/5 structural cases passed "
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

    # ------------------------------------------------------------------ #
    # Case 9: REGRESSION — multi-row IMPORTED_POSE header CSV round-trips
    # through convert + verify cleanly. Prior bug: the multi-row path
    # called df.to_parquet without index=False, which caused pandas to
    # preserve the dataframe index as an "__index_level_0__" column in
    # the parquet schema. verify_parquet then compared 45 csv columns
    # against 46 parquet columns (incl. the preserved index) and
    # always returned False for input_csv/ files.
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_path = str(td / "imported_pose.csv")
        pq_path = str(td / "imported_pose.parquet")
        # Build a multi-row IMPORTED_POSE-format CSV with realistic
        # 3-row header and a small number of frames
        markers = ["nose", "ear_left", "ear_right", "center", "tail"]
        suffixes = ["x", "y", "p"]
        with open(csv_path, "w") as f:
            n_data_cols = len(markers) * len(suffixes)
            # Row 0: scorer (IMPORTED_POSE × n_data_cols)
            f.write("," + ",".join(["IMPORTED_POSE"] * n_data_cols) + "\n")
            # Row 1: bodypart-row (also IMPORTED_POSE in Mufasa's format)
            f.write("," + ",".join(["IMPORTED_POSE"] * n_data_cols) + "\n")
            # Row 2: real column names
            real_names = [f"{m}_{s}" for m in markers for s in suffixes]
            f.write("," + ",".join(real_names) + "\n")
            # Data rows
            for i in range(20):
                values = [str(i)] + [f"{(i + k) * 1.5:.4f}"
                                     for k in range(n_data_cols)]
                f.write(",".join(values) + "\n")

        # Convert
        n_rows, n_cols = csv_to_parquet.convert_csv_to_parquet(
            csv_path, pq_path, has_index=True,
        )
        assert n_rows == 20, f"Expected 20 data rows, got {n_rows}"
        assert n_cols == 15, f"Expected 15 data columns, got {n_cols}"

        # Inspect the parquet schema directly — it must NOT contain
        # the pandas-internal __index_level_0__ column.
        import pyarrow.parquet as pq_module
        pq_cols = list(pq_module.read_schema(pq_path).names)
        assert "__index_level_0__" not in pq_cols, (
            f"Parquet schema should not preserve the dataframe index "
            f"as a column; got {pq_cols}"
        )
        assert pq_cols == real_names, (
            f"Parquet columns should match the CSV's last-header-row "
            f"names exactly; got {pq_cols} vs {real_names}"
        )

        # Now the full verification cycle should succeed.
        assert csv_to_parquet.verify_parquet(csv_path, pq_path, has_index=True), (
            "verify_parquet should succeed on a clean multi-row "
            "round-trip — this was the original bug"
        )

    # ------------------------------------------------------------------ #
    # Case 10: REGRESSION (compat) — verify_parquet tolerates parquets
    # that were written by the OLD (buggy) code path with the index
    # preserved. Users may have such files on disk from a partial
    # prior run; we shouldn't fail those when they re-run.
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_path = str(td / "legacy.csv")
        pq_path = str(td / "legacy.parquet")
        # Same multi-row CSV
        markers = ["nose", "ear_left", "ear_right"]
        suffixes = ["x", "y", "p"]
        n_data_cols = len(markers) * len(suffixes)
        with open(csv_path, "w") as f:
            f.write("," + ",".join(["IMPORTED_POSE"] * n_data_cols) + "\n")
            f.write("," + ",".join(["IMPORTED_POSE"] * n_data_cols) + "\n")
            real_names = [f"{m}_{s}" for m in markers for s in suffixes]
            f.write("," + ",".join(real_names) + "\n")
            for i in range(10):
                values = [str(i)] + [f"{(i + k) * 0.5:.4f}"
                                     for k in range(n_data_cols)]
                f.write(",".join(values) + "\n")

        # Simulate the OLD (buggy) write path: pandas read with
        # multi-index, flatten, then write WITH index preservation
        df = pd.read_csv(csv_path, index_col=0, header=[0, 1, 2])
        df.columns = df.columns.get_level_values(-1)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.to_parquet(pq_path)  # NO index=False — old buggy behavior

        import pyarrow.parquet as pq_module
        pq_cols = list(pq_module.read_schema(pq_path).names)
        assert "__index_level_0__" in pq_cols, (
            "Sanity check: simulated legacy write should produce the "
            "preserved-index column"
        )

        # The hardened verify_parquet should accept this anyway
        assert csv_to_parquet.verify_parquet(csv_path, pq_path, has_index=True), (
            "verify_parquet should tolerate legacy parquets with "
            "preserved __index_level_0__ column"
        )

    print("smoke_csv_to_parquet: 10/10 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
