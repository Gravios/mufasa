"""Real-data byte-equivalence check for feature_subset_kernels refactor.

NOT runnable in the sandbox — requires the full Mufasa runtime
(numba, pandas, h5py, etc.). Designed for the user to run ONCE on
the workstation as a before/after acceptance check.

Pre-conditions:
    * Mufasa env active (conda activate mufasa)
    * A SimBA project with at least one outlier-corrected pose data file
    * The pre-refactor saved feature output for at least one video
      (from a previous sequential run)

What this does:

1. Locates a single outlier-corrected pose data file.
2. Reads it and runs each of the 9 feature-family kernels directly
   against the dataframe.
3. Writes the kernel output to /tmp.
4. Compares the kernel output to a previously-saved reference output
   from the legacy class run, if found.

Usage:

    cd <mufasa repo>
    python tests/smoke_feature_kernels_realdata.py \\
        --config <project_config.ini> \\
        --reference-feature-file <path/to/old/feature_output.parquet>

The reference file should be a per-video output from
.../csv/features_extracted/temp_data_<datetime>/<video_name>.parquet
or wherever your project keeps them.

If the script reports byte-equivalence (np.allclose with small
tolerance) for every column, the refactor is verified.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to SimBA project_config.ini")
    parser.add_argument(
        "--reference-feature-file",
        required=False,
        help="Optional pre-refactor saved feature output file to "
             "compare against. If omitted, only sanity checks run.",
    )
    parser.add_argument(
        "--video-name", required=False,
        help="Video name to run on. If omitted, the first video in "
             "outlier_corrected_dir is used.",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-5,
        help="np.allclose rtol (default 1e-5)",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-7,
        help="np.allclose atol (default 1e-7)",
    )
    args = parser.parse_args()

    # Construct a calculator (without running the full pipeline)
    # to get access to its discovered video names and per-video
    # configuration.
    from mufasa.feature_extractors.feature_subsets import (
        FEATURE_FAMILIES, FeatureSubsetsCalculator,
        TWO_POINT_BP_DISTANCES, WITHIN_ANIMAL_THREE_POINT_ANGLES,
        WITHIN_ANIMAL_THREE_POINT_HULL, WITHIN_ANIMAL_FOUR_POINT_HULL,
        ANIMAL_CONVEX_HULL_PERIMETER, ANIMAL_CONVEX_HULL_AREA,
        FRAME_BP_MOVEMENT, ARENA_EDGE,
    )
    from mufasa.feature_extractors.feature_subset_kernels import (
        compute_animal_convex_hulls, compute_distances_to_frame_edge,
        compute_four_point_hulls, compute_framewise_movement,
        compute_three_point_angles, compute_three_point_hulls,
        compute_two_point_distances,
    )
    from mufasa.utils.read_write import get_fn_ext, read_df

    # Construct the calculator with a non-ROI feature family list so
    # __init__ doesn't try to load ROIs (this avoids a dependency on
    # ROI data being present).
    calc = FeatureSubsetsCalculator(
        config_path=args.config,
        feature_families=[
            TWO_POINT_BP_DISTANCES,
            WITHIN_ANIMAL_THREE_POINT_ANGLES,
            WITHIN_ANIMAL_THREE_POINT_HULL,
            WITHIN_ANIMAL_FOUR_POINT_HULL,
            ANIMAL_CONVEX_HULL_PERIMETER,
            ANIMAL_CONVEX_HULL_AREA,
            FRAME_BP_MOVEMENT,
            ARENA_EDGE,
        ],
    )
    # As of step 4 of the refactor, heavy setup (temp_dir, ROI
    # filtering, body-part combinations) is deferred from __init__
    # to a _setup_run() helper that run() calls. We're using calc
    # for inspection (its two_point_combs etc.), so call it
    # explicitly here.
    calc._setup_run()

    # Pick a video
    if args.video_name:
        candidates = [
            p for p in calc.data_paths
            if get_fn_ext(filepath=p)[1] == args.video_name
        ]
        if not candidates:
            print(f"No video named {args.video_name!r} in data_paths")
            return 1
        file_path = candidates[0]
    else:
        file_path = calc.data_paths[0]

    video_name = get_fn_ext(filepath=file_path)[1]
    print(f"Running on video: {video_name}")
    print(f"  file: {file_path}")

    # Set up the per-video state
    video_info, px_per_mm, fps = calc.read_video_info(video_name=video_name)
    video_width = int(video_info["Resolution_width"].values[0])
    video_height = int(video_info["Resolution_height"].values[0])
    df = read_df(file_path=file_path, file_type=calc.file_type)
    print(f"  frames: {len(df)}  px_per_mm: {px_per_mm}  fps: {fps}")

    # Clip out-of-frame coords (matches the run() pipeline)
    for col in df.columns:
        cl = str(col).lower()
        if cl.endswith("_x"):
            df[col] = df[col].clip(lower=0, upper=video_width)
        elif cl.endswith("_y"):
            df[col] = df[col].clip(lower=0, upper=video_height)

    # Run each kernel and collect results
    results: dict[str, np.ndarray] = {}

    print("\nRunning kernels:")

    print("  compute_two_point_distances")
    results.update(compute_two_point_distances(
        df=df, two_point_combs=calc.two_point_combs,
        px_per_mm=px_per_mm,
    ))
    print("  compute_three_point_angles")
    results.update(compute_three_point_angles(
        df=df,
        within_animal_three_point_combs=calc.within_animal_three_point_combs,
    ))
    print("  compute_three_point_hulls")
    results.update(compute_three_point_hulls(
        df=df,
        within_animal_three_point_combs=calc.within_animal_three_point_combs,
        px_per_mm=px_per_mm,
    ))
    print("  compute_four_point_hulls")
    results.update(compute_four_point_hulls(
        df=df,
        within_animal_four_point_combs=calc.within_animal_four_point_combs,
        px_per_mm=px_per_mm,
    ))
    print("  compute_animal_convex_hulls (perimeter)")
    results.update(compute_animal_convex_hulls(
        df=df, animal_bps=calc.animal_bps,
        px_per_mm=px_per_mm, method="perimeter",
    ))
    print("  compute_animal_convex_hulls (area)")
    results.update(compute_animal_convex_hulls(
        df=df, animal_bps=calc.animal_bps,
        px_per_mm=px_per_mm, method="area",
    ))
    print("  compute_framewise_movement")
    results.update(compute_framewise_movement(
        df=df, animal_bps=calc.animal_bps, px_per_mm=px_per_mm,
        source=str(file_path),
    ))
    print("  compute_distances_to_frame_edge")
    results.update(compute_distances_to_frame_edge(
        df=df, animal_bps=calc.animal_bps, px_per_mm=px_per_mm,
        video_width=video_width, video_height=video_height,
        source=str(file_path),
    ))

    print(f"\nKernel run produced {len(results)} columns.")

    # Build a dataframe matching the format the legacy run() loop
    # would have written
    new_df = pd.DataFrame(results).fillna(-1)
    new_df = new_df.add_suffix("_FEATURE_SUBSET")
    new_df = new_df[sorted(new_df.columns)]

    # Save kernel output
    out_path = f"/tmp/kernel_output_{video_name}.parquet"
    new_df.to_parquet(out_path)
    print(f"Kernel output saved to: {out_path}")
    print(f"Shape: {new_df.shape}")

    if not args.reference_feature_file:
        print("\nNo --reference-feature-file passed; "
              "skipping byte-equivalence check.")
        print("To verify, re-run with the path to a pre-refactor "
              "saved feature output for this video.")
        return 0

    # Compare to reference
    print(f"\nLoading reference: {args.reference_feature_file}")
    ref_df = pd.read_parquet(args.reference_feature_file)
    print(f"Reference shape: {ref_df.shape}")

    # Sort both by column name for comparison
    common_cols = sorted(set(new_df.columns) & set(ref_df.columns))
    only_new = sorted(set(new_df.columns) - set(ref_df.columns))
    only_ref = sorted(set(ref_df.columns) - set(new_df.columns))

    if only_new:
        print(f"\nColumns ONLY in kernel output ({len(only_new)}):")
        for c in only_new[:10]:
            print(f"  {c}")
        if len(only_new) > 10:
            print(f"  ... and {len(only_new) - 10} more")
    if only_ref:
        print(f"\nColumns ONLY in reference ({len(only_ref)}):")
        for c in only_ref[:10]:
            print(f"  {c}")
        if len(only_ref) > 10:
            print(f"  ... and {len(only_ref) - 10} more")
    print(f"\nCommon columns: {len(common_cols)}")

    if not common_cols:
        print("ERROR: no common columns to compare.")
        return 2

    # Compare each common column
    n_ok = 0
    n_failed = 0
    for col in common_cols:
        a = new_df[col].values
        b = ref_df[col].values
        if len(a) != len(b):
            print(f"FAIL [{col}]: length mismatch {len(a)} vs {len(b)}")
            n_failed += 1
            continue
        try:
            ok = np.allclose(
                a.astype(np.float64),
                b.astype(np.float64),
                rtol=args.rtol, atol=args.atol,
                equal_nan=True,
            )
        except (TypeError, ValueError):
            ok = np.array_equal(a, b)
        if ok:
            n_ok += 1
        else:
            n_failed += 1
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            print(f"FAIL [{col}]: max abs diff = {np.nanmax(diff):.6e}")

    print(f"\n{n_ok}/{len(common_cols)} columns matched within tolerance.")
    if n_failed:
        print(f"{n_failed} columns differ — investigate before "
              f"trusting the refactor.")
        return 3
    print("All common columns byte-equivalent. Refactor verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
