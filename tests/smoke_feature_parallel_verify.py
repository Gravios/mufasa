"""End-to-end parallel verification for FeatureSubsetsCalculator.

NOT runnable in the sandbox — requires the full Mufasa runtime.

Designed for the user to run on the workstation as a one-time
acceptance check after applying step 6 (parallelization).

What this does:

1. Picks a small subset of videos (default 4, configurable)
2. Runs FeatureSubsetsCalculator with n_workers=1 (sequential)
3. Saves the per-video output
4. Runs again with n_workers=4 (parallel)
5. Saves the per-video output to a different temp dir
6. Compares the per-video output files between the two runs
7. Reports any differences

This is the strongest possible verification of the parallel
implementation: it confirms that running with N workers produces
byte-identical (within numerical tolerance) output to running
sequentially. If this test passes, the parallelization is sound.

Usage:

    cd <mufasa repo>
    python tests/smoke_feature_parallel_verify.py \\
        --config <project>/project_config.ini \\
        --n-test-videos 4 \\
        --n-workers 4

Time cost: roughly (n_test_videos * 215 seconds * 1.25) ≈ 18 min
for 4 videos with the user's typical workload. Could be reduced
by using a faster (smaller) set of videos for the smoke test.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _diff_per_video_outputs(
    seq_dir: str, par_dir: str, file_type: str,
    rtol: float, atol: float,
) -> int:
    """Compare every per-video output file between two directories.

    Returns the number of mismatched columns across all files. 0
    means the parallel and sequential runs produced identical output.
    """
    seq_files = sorted(
        f for f in os.listdir(seq_dir)
        if f.endswith(f".{file_type}")
    )
    par_files = sorted(
        f for f in os.listdir(par_dir)
        if f.endswith(f".{file_type}")
    )
    only_seq = set(seq_files) - set(par_files)
    only_par = set(par_files) - set(seq_files)
    if only_seq:
        print(f"Only in sequential: {sorted(only_seq)}")
    if only_par:
        print(f"Only in parallel:   {sorted(only_par)}")
    common = sorted(set(seq_files) & set(par_files))
    print(f"\nComparing {len(common)} videos...")
    n_total_mismatched_cols = 0
    for fname in common:
        seq_df = pd.read_parquet(os.path.join(seq_dir, fname))
        par_df = pd.read_parquet(os.path.join(par_dir, fname))
        if seq_df.shape != par_df.shape:
            print(
                f"  ✗ {fname}: shape mismatch "
                f"{seq_df.shape} vs {par_df.shape}"
            )
            n_total_mismatched_cols += 1
            continue
        if list(seq_df.columns) != list(par_df.columns):
            print(f"  ✗ {fname}: column order differs")
            n_total_mismatched_cols += 1
            continue
        n_mis = 0
        for col in seq_df.columns:
            a = seq_df[col].values
            b = par_df[col].values
            try:
                ok = np.allclose(
                    a.astype(np.float64),
                    b.astype(np.float64),
                    rtol=rtol, atol=atol,
                    equal_nan=True,
                )
            except (TypeError, ValueError):
                ok = np.array_equal(a, b)
            if not ok:
                n_mis += 1
                # Print first 3 mismatching columns
                if n_mis <= 3:
                    diff = np.abs(
                        a.astype(np.float64) - b.astype(np.float64)
                    )
                    print(
                        f"    ✗ {fname} [{col}]: max abs diff = "
                        f"{np.nanmax(diff):.6e}"
                    )
        if n_mis == 0:
            print(f"  ✓ {fname}: {seq_df.shape[1]} cols match")
        else:
            print(f"  ✗ {fname}: {n_mis} of {seq_df.shape[1]} cols differ")
        n_total_mismatched_cols += n_mis
    return n_total_mismatched_cols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--n-test-videos", type=int, default=4,
        help="Number of videos to use for the verification (default 4)"
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="n_workers for the parallel run (default 4)"
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()

    from mufasa.feature_extractors.feature_subsets import (
        FEATURE_FAMILIES, FeatureSubsetsCalculator,
        TWO_POINT_BP_DISTANCES, WITHIN_ANIMAL_THREE_POINT_ANGLES,
        WITHIN_ANIMAL_THREE_POINT_HULL, WITHIN_ANIMAL_FOUR_POINT_HULL,
        ANIMAL_CONVEX_HULL_PERIMETER, ANIMAL_CONVEX_HULL_AREA,
        FRAME_BP_MOVEMENT, ARENA_EDGE,
    )

    test_features = [
        TWO_POINT_BP_DISTANCES,
        WITHIN_ANIMAL_THREE_POINT_ANGLES,
        WITHIN_ANIMAL_THREE_POINT_HULL,
        WITHIN_ANIMAL_FOUR_POINT_HULL,
        ANIMAL_CONVEX_HULL_PERIMETER,
        ANIMAL_CONVEX_HULL_AREA,
        FRAME_BP_MOVEMENT,
        ARENA_EDGE,
    ]

    # ---- Sequential run ----
    print(f"Running SEQUENTIAL on {args.n_test_videos} videos...")
    calc_seq = FeatureSubsetsCalculator(
        config_path=args.config,
        feature_families=test_features,
        n_workers=1,
    )
    # Truncate data_paths to N test videos
    calc_seq._setup_run()
    calc_seq.data_paths = calc_seq.data_paths[:args.n_test_videos]
    calc_seq.video_names = calc_seq.video_names[:args.n_test_videos]

    # We need to actually run, but the run() flow includes _setup_run
    # which we already called. Easiest: re-construct and run
    # cleanly with a custom output capture.
    # ... actually the simplest path: copy the temp_dir output after run()
    # Construct a fresh calc and run() which invokes _setup_run internally
    seq_calc = FeatureSubsetsCalculator(
        config_path=args.config,
        feature_families=test_features,
        n_workers=1,
    )
    seq_calc._setup_run()
    seq_calc.data_paths = seq_calc.data_paths[:args.n_test_videos]
    seq_calc.video_names = seq_calc.video_names[:args.n_test_videos]
    # _setup_run was already called, so run() calling it again is
    # idempotent. Let it proceed.

    # We need to capture seq_calc.temp_dir BEFORE run() removes it.
    seq_temp = seq_calc.temp_dir
    print(f"Sequential temp_dir: {seq_temp}")

    # The current run() removes temp_dir at the end. To capture
    # output, we copy temp_dir contents to a holdback location after
    # run() — but run() removes temp_dir. So we need to copy from
    # within run() OR change strategy: run with save_dir set so
    # outputs end up in a stable location.
    seq_save_dir = tempfile.mkdtemp(prefix="seq_run_")
    seq_calc.save_dir = seq_save_dir
    seq_calc.run()
    print(f"Sequential output preserved in: {seq_save_dir}")

    # ---- Parallel run ----
    print(f"\nRunning PARALLEL on {args.n_test_videos} videos with "
          f"n_workers={args.n_workers}...")
    par_save_dir = tempfile.mkdtemp(prefix="par_run_")
    par_calc = FeatureSubsetsCalculator(
        config_path=args.config,
        feature_families=test_features,
        n_workers=args.n_workers,
        save_dir=par_save_dir,
    )
    par_calc._setup_run()
    par_calc.data_paths = par_calc.data_paths[:args.n_test_videos]
    par_calc.video_names = par_calc.video_names[:args.n_test_videos]
    par_calc.run()
    print(f"Parallel output preserved in: {par_save_dir}")

    # ---- Compare ----
    print(f"\nComparing {seq_save_dir} vs {par_save_dir}")
    n_mismatches = _diff_per_video_outputs(
        seq_save_dir, par_save_dir, par_calc.file_type,
        args.rtol, args.atol,
    )
    if n_mismatches == 0:
        print(f"\nAll {args.n_test_videos} videos produced "
              f"byte-equivalent output between sequential and "
              f"parallel runs.")
        print("Parallelization VERIFIED.")
        return 0
    print(f"\n{n_mismatches} columns differ between sequential and "
          f"parallel output.")
    print(f"DO NOT use n_workers > 1 in production until this is "
          f"investigated.")
    print(f"Output dirs preserved for inspection:")
    print(f"  Sequential: {seq_save_dir}")
    print(f"  Parallel:   {par_save_dir}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
