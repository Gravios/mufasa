"""Test for the Mufasa flat CSV → DLC multi-index CSV converter.

Round-trips through:
  flat → DLC → CSV (3-row header) → readable by DLC importer

Sandbox-runnable.

    PYTHONPATH=. python tests/smoke_csv_to_dlc.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Direct import (the tool is pure pandas, no Cython).
sys.path.insert(0, str(Path("mufasa/tools").resolve()))
import csv_to_dlc


def main() -> int:
    # ------------------------------------------------------------------ #
    # Case 1: parse_flat_columns handles the standard suffix convention
    # ------------------------------------------------------------------ #
    pairs = csv_to_dlc.parse_flat_columns([
        "Nose_x", "Nose_y", "Nose_p",
        "Tail_base_x", "Tail_base_y", "Tail_base_p",
    ])
    assert pairs == [
        ("Nose", "x"), ("Nose", "y"), ("Nose", "likelihood"),
        ("Tail_base", "x"), ("Tail_base", "y"), ("Tail_base", "likelihood"),
    ], f"Wrong pairs: {pairs}"

    # ------------------------------------------------------------------ #
    # Case 2: bodyparts with underscores in the name (e.g. Tail_base)
    # are parsed correctly using rfind on '_'
    # ------------------------------------------------------------------ #
    pairs = csv_to_dlc.parse_flat_columns([
        "Center_of_mass_x", "Center_of_mass_y", "Center_of_mass_p",
    ])
    assert pairs == [
        ("Center_of_mass", "x"),
        ("Center_of_mass", "y"),
        ("Center_of_mass", "likelihood"),
    ]

    # ------------------------------------------------------------------ #
    # Case 3: bad suffix raises ValueError
    # ------------------------------------------------------------------ #
    try:
        csv_to_dlc.parse_flat_columns(["Nose_z"])
    except ValueError as exc:
        assert "Nose_z" in str(exc) or "z" in str(exc)
    else:
        raise AssertionError("Should have raised on Nose_z suffix")

    # ------------------------------------------------------------------ #
    # Case 4: flat_to_dlc produces a 3-level MultiIndex
    # ------------------------------------------------------------------ #
    flat = pd.DataFrame({
        "Nose_x": [1.0, 2.0],
        "Nose_y": [10.0, 20.0],
        "Nose_p": [0.99, 0.98],
        "Tail_x": [3.0, 4.0],
        "Tail_y": [30.0, 40.0],
        "Tail_p": [0.97, 0.96],
    })
    dlc = csv_to_dlc.flat_to_dlc(flat, scorer="DLC_resnet50_test")
    assert isinstance(dlc.columns, pd.MultiIndex)
    assert dlc.columns.nlevels == 3, (
        f"Expected 3 levels, got {dlc.columns.nlevels}"
    )
    assert dlc.columns.names == ["scorer", "bodyparts", "coords"]
    # Top level all "DLC_resnet50_test"
    assert all(
        s == "DLC_resnet50_test" for s in dlc.columns.get_level_values(0)
    )
    # Second level: Nose, Nose, Nose, Tail, Tail, Tail
    bps = list(dlc.columns.get_level_values(1))
    assert bps == ["Nose", "Nose", "Nose", "Tail", "Tail", "Tail"], bps
    # Third level: x, y, likelihood, x, y, likelihood
    coords = list(dlc.columns.get_level_values(2))
    assert coords == ["x", "y", "likelihood", "x", "y", "likelihood"], coords

    # ------------------------------------------------------------------ #
    # Case 5: data values are unchanged
    # ------------------------------------------------------------------ #
    # Compare the raw ndarray values
    import numpy as np
    assert np.array_equal(flat.values, dlc.values), (
        "Conversion should be value-preserving"
    )

    # ------------------------------------------------------------------ #
    # Case 6: round-trip through CSV file. Write the DLC df to disk,
    # read it back the way DLC users would (with header=[0,1,2]),
    # and verify columns + values match
    # ------------------------------------------------------------------ #
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        out_path = f.name
    try:
        dlc.to_csv(out_path)
        # DLC's standard read pattern
        roundtrip = pd.read_csv(out_path, header=[0, 1, 2], index_col=0)
        assert isinstance(roundtrip.columns, pd.MultiIndex)
        assert roundtrip.columns.nlevels == 3
        assert np.array_equal(roundtrip.values, dlc.values)
    finally:
        os.unlink(out_path)

    # ------------------------------------------------------------------ #
    # Case 7: convert_file end-to-end (with index column written by
    # write_df-style output, which is the typical Mufasa case)
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        flat_path = os.path.join(td, "Video_42.csv")
        flat.to_csv(flat_path)  # writes with index column
        dlc_path = os.path.join(td, "Video_42_dlc.csv")
        csv_to_dlc.convert_file(
            in_path=flat_path,
            out_path=dlc_path,
            scorer="DLC_resnet50_test",
            has_index=True,
        )
        assert os.path.isfile(dlc_path)
        # Read it back via DLC convention
        roundtrip = pd.read_csv(dlc_path, header=[0, 1, 2], index_col=0)
        assert isinstance(roundtrip.columns, pd.MultiIndex)
        # Values should match input
        assert np.array_equal(roundtrip.values, flat.values)

    # ------------------------------------------------------------------ #
    # Case 8: convert_directory processes every CSV in a directory
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "in")
        out_dir = os.path.join(td, "out")
        os.makedirs(in_dir)
        # Three files with slightly different content
        for i, video in enumerate(["A.csv", "B.csv", "C.csv"]):
            df = flat.copy()
            df["Nose_x"] = df["Nose_x"] + i
            df.to_csv(os.path.join(in_dir, video))
        # Hidden file should be skipped
        Path(os.path.join(in_dir, ".DS_Store")).write_text("junk")

        written = csv_to_dlc.convert_directory(
            in_dir=in_dir, out_dir=out_dir,
            scorer="DLC_resnet50_test", has_index=True,
        )
        assert len(written) == 3, f"Expected 3 outputs, got {len(written)}"
        for name in ["A.csv", "B.csv", "C.csv"]:
            assert os.path.isfile(os.path.join(out_dir, name))
        # No hidden file in output
        assert not os.path.isfile(os.path.join(out_dir, ".DS_Store"))

    # ------------------------------------------------------------------ #
    # Case 9: discover_scorer falls back to default when config has
    # no scorer or doesn't exist
    # ------------------------------------------------------------------ #
    assert csv_to_dlc.discover_scorer(None) == "mufasa_export"
    assert csv_to_dlc.discover_scorer("/nonexistent/path") == "mufasa_export"
    # With a config that has no scorer key
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
        cfg_path = f.name
        f.write("[General settings]\nproject_name = test\n")
    try:
        assert csv_to_dlc.discover_scorer(cfg_path) == "mufasa_export"
    finally:
        os.unlink(cfg_path)

    # ------------------------------------------------------------------ #
    # Case 10: discover_scorer uses the config's scorer when present
    # ------------------------------------------------------------------ #
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
        cfg_path = f.name
        f.write("[create_ensemble_settings]\nscorer = DLC_my_special_model\n")
    try:
        assert csv_to_dlc.discover_scorer(cfg_path) == "DLC_my_special_model"
    finally:
        os.unlink(cfg_path)

    print("smoke_csv_to_dlc: 10/10 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
