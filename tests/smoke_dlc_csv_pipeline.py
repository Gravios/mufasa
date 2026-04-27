"""Smoke-test for DLC CSV import pipeline.

Tests the parsing + transformation logic of DLCSingleAnimalCSVImporter
against the format DLC actually emits (3-row multi-header,
empty-cell-per-low-likelihood-frame), without requiring h5py/pytables
or PySide6.

The test simulates the DataFrame transformation pipeline directly,
since instantiating the importer requires a full Mufasa project on
disk and pulls in tkinter (not available in the CI sandbox).

    PYTHONPATH=. python tests/smoke_dlc_csv_pipeline.py
"""
from __future__ import annotations

import csv as _csv
import sys
import tempfile
from pathlib import Path


def write_dlc_csv(path: Path, scorer: str, bodyparts: list,
                  rows: list) -> None:
    """Write a synthetic DLC CSV in the standard 3-header-row layout.

    rows: list of [(x, y, likelihood)] tuples per body-part per frame.
          rows[frame_idx][bp_idx] is (x, y, likelihood); use None for
          empty cell (DLC's low-confidence convention).
    """
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        # Row 0: scorer (one column per (bp, coord) plus one leading)
        scorer_cols = [scorer] + [f"{scorer}.{i}"
                                  for i in range(1, len(bodyparts) * 3)]
        w.writerow(["scorer"] + scorer_cols[:len(bodyparts) * 3])
        # Row 1: bodyparts (each repeated 3×)
        bp_row = ["bodyparts"]
        for bp in bodyparts:
            bp_row.extend([bp, bp, bp])
        w.writerow(bp_row)
        # Row 2: coords
        co_row = ["coords"]
        for _ in bodyparts:
            co_row.extend(["x", "y", "likelihood"])
        w.writerow(co_row)
        # Data rows
        for frame_idx, frame in enumerate(rows):
            cells = [str(frame_idx)]
            for triplet in frame:
                if triplet is None:
                    cells.extend(["", "", ""])
                else:
                    x, y, p = triplet
                    cells.extend([
                        "" if x is None else str(x),
                        "" if y is None else str(y),
                        str(p),
                    ])
            w.writerow(cells)


def main() -> int:
    import numpy as np
    import pandas as pd
    from mufasa.pose_importers.dlc_autodetect import extract_bodyparts
    from mufasa.pose_importers.likelihood_mask import (
        apply_likelihood_threshold,
    )

    tmp = Path(tempfile.mkdtemp())
    try:
        # ------------------------------------------------------------ #
        # Build a CSV that mimics the user-uploaded format: empty cells
        # for low-confidence frames, normal numeric for the rest.
        # ------------------------------------------------------------ #
        bps = ["nose", "headmid", "ear_left"]
        rows = [
            # frame 0: nose+headmid empty (low p), ear_left numeric
            [(None, None, 0.15), (None, None, 0.60), (580.6, 248.2, 0.84)],
            # frame 1: same shape
            [(None, None, 0.27), (None, None, 0.64), (581.2, 248.4, 0.85)],
            # frame 2: all three numeric, varied likelihood
            [(100.1, 200.2, 0.92), (110.0, 210.0, 0.88), (120.0, 220.0, 0.45)],
        ]
        csv_path = tmp / "test.csv"
        write_dlc_csv(csv_path, "DLC_HrnetW32_proj_snapshot_1760", bps, rows)

        # ------------------------------------------------------------ #
        # Case 1: autodetect picks up bodyparts in column order
        # ------------------------------------------------------------ #
        detected = extract_bodyparts(csv_path)
        assert detected == bps, f"case 1: {detected}"

        # ------------------------------------------------------------ #
        # Case 2: pd.read_csv with header=[0,1,2] handles the empties
        # ------------------------------------------------------------ #
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        assert df.shape == (3, 9), f"case 2: shape {df.shape}"
        assert df.columns.nlevels == 3
        # Empty cells become NaN
        assert df.isna().sum().sum() == 4 * 2 + 0, (
            f"case 2: expected 8 NaN (frames 0+1, x+y of 2 bps each)"
        )

        # ------------------------------------------------------------ #
        # Case 3: fillna(0) → bp_headers rename → mask works correctly
        # This simulates the importer's run() flow on the parsed frame.
        # ------------------------------------------------------------ #
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        bp_headers = []
        for bp in bps:
            bp_headers.extend([f"{bp}_x", f"{bp}_y", f"{bp}_p"])
        df.columns = bp_headers

        # Apply threshold of 0.5
        out, counts = apply_likelihood_threshold(df, threshold=0.5)
        # nose: frames 0 (p=0.15) + 1 (p=0.27) below threshold
        assert counts.get("nose") == 2, f"case 3 nose: {counts}"
        # headmid: same — both 0.60 and 0.64 are >= 0.5? No wait
        # 0.60 >= 0.5 and 0.64 >= 0.5, so headmid should NOT mask.
        # Actually wait — looking again: 0.60 and 0.64 are both > 0.5,
        # so headmid stays. But let's check.
        assert "headmid" not in counts, f"case 3 headmid: {counts}"
        # ear_left: frame 2 has p=0.45 < 0.5 → masked
        assert counts.get("ear_left") == 1, f"case 3 ear_left: {counts}"

        # ------------------------------------------------------------ #
        # Case 4: After masking, masked rows have x=y=0 (which the
        # interpolator picks up as missing).
        # ------------------------------------------------------------ #
        # nose frame 0: was empty → became 0,0 from fillna; mask
        # also leaves at 0 → still 0
        assert out.loc[0, "nose_x"] == 0.0
        # ear_left frame 2: was 120,220 with p=0.45 → masked to 0,0
        assert out.loc[2, "ear_left_x"] == 0.0
        # ear_left frames 0,1: numeric and p > 0.5 → preserved
        assert abs(out.loc[0, "ear_left_x"] - 580.6) < 0.01

        print("smoke_dlc_csv_pipeline: 4/4 cases passed")
        return 0
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
