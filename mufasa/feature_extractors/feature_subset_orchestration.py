"""
mufasa.feature_extractors.feature_subset_orchestration
======================================================

Per-video orchestration for feature subset computation.

This module extracts the body of `FeatureSubsetsCalculator.run()`'s
per-video loop into a module-level function `process_one_video`,
which:

* Takes only its inputs as parameters (no `self`)
* Has no side effects on shared state
* Writes its output to a temp file path

Why this matters
----------------

The class-method form (`run()` containing the loop body) has hidden
state coupling: the loop reads from many `self.X` attributes,
mutates `self.results`, and writes to a `self.temp_dir` shared
across iterations. This makes:

* Per-video work hard to reason about in isolation
* Per-video work hard to test without instantiating the full class
* Per-video work hard to parallelize (the class is heavyweight and
  has unpicklable parts like loggers)

Extracting `process_one_video` into a static function fixes all
three. Step 6 of the refactor plan (parallelization) wraps this
function in `concurrent.futures.ProcessPoolExecutor`.

Behavior is unchanged from the legacy class-method version. The
class's `run()` method now delegates per-video work to this
function. Output files in `temp_dir` are byte-identical to what
the previous version produced (modulo file-ordering, since iteration
order is preserved).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Mapping, Optional

import numpy as np
import pandas as pd

from mufasa.feature_extractors.feature_subset_kernels import (
    compute_animal_convex_hulls, compute_distances_to_frame_edge,
    compute_four_point_hulls, compute_framewise_movement,
    compute_inside_roi, compute_roi_center_distances,
    compute_three_point_angles, compute_three_point_hulls,
    compute_two_point_distances)
from mufasa.utils.printing import SimbaTimer
from mufasa.utils.read_write import get_fn_ext, read_df, write_df


# Feature family name constants. Duplicated from feature_subsets.py
# because importing them here would cause a circular import (the
# class module imports this orchestration module).
TWO_POINT_BP_DISTANCES = (
    'TWO-POINT BODY-PART DISTANCES (MM)'
)
WITHIN_ANIMAL_THREE_POINT_ANGLES = (
    'WITHIN-ANIMAL THREE-POINT BODY-PART ANGLES (DEGREES)'
)
WITHIN_ANIMAL_THREE_POINT_HULL = (
    'WITHIN-ANIMAL THREE-POINT CONVEX HULL PERIMETERS (MM)'
)
WITHIN_ANIMAL_FOUR_POINT_HULL = (
    'WITHIN-ANIMAL FOUR-POINT CONVEX HULL PERIMETERS (MM)'
)
ANIMAL_CONVEX_HULL_PERIMETER = (
    'ENTIRE ANIMAL CONVEX HULL PERIMETERS (MM)'
)
ANIMAL_CONVEX_HULL_AREA = (
    'ENTIRE ANIMAL CONVEX HULL AREA (MM2)'
)
FRAME_BP_MOVEMENT = (
    'FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)'
)
FRAME_BP_TO_ROI_CENTER = (
    'FRAME-BY-FRAME BODY-PART DISTANCES TO ROI CENTERS (MM)'
)
FRAME_BP_INSIDE_ROI = (
    'FRAME-BY-FRAME BODY-PARTS INSIDE ROIS (BOOLEAN)'
)
ARENA_EDGE = (
    'BODY-PART DISTANCES TO VIDEO FRAME EDGE (MM)'
)


@dataclass(frozen=True)
class VideoProcessingConfig:
    """All inputs `process_one_video` needs that are NOT specific to
    a single video. Pickleable — designed for safe transfer to a
    worker process.

    Per-video inputs (the file path, the per-video info row, the ROI
    dict for that video) are passed alongside the config rather than
    on it, so this same config object can be reused across videos.
    """

    feature_families: List[str]
    file_type: str
    temp_dir: str

    # Body-part combination data, computed once in __init__ from the
    # project's animal_bp_dict.
    two_point_combs: np.ndarray
    within_animal_three_point_combs: Mapping[str, np.ndarray]
    within_animal_four_point_combs: Mapping[str, np.ndarray]
    animal_bps: Mapping[str, list]

    # Project-level ROI data. May be None if no ROI features
    # are requested. Keyed by video name → {roi_name: roi_data}.
    roi_dict: Optional[Mapping[str, dict]] = None

    # Whole video_info dataframe. Used by process_one_video to look
    # up px_per_mm, fps, resolution per video. We pass the entire
    # frame (rather than a per-video slice) because the lookup is
    # cheap and it keeps the per-video function signature simple.
    video_info_df: Optional[pd.DataFrame] = None


def _read_video_info_for(
    video_info_df: pd.DataFrame, video_name: str,
) -> tuple[float, float, int, int]:
    """Look up per-video metadata. Mirrors the validation logic of
    ConfigReader.read_video_info but works on a dataframe directly
    so it can be called outside the class context."""
    settings = video_info_df.loc[video_info_df["Video"] == video_name]
    if len(settings) > 1:
        raise ValueError(
            f"Multiple rows in video_info named {video_name!r} — "
            "expected exactly one."
        )
    if len(settings) < 1:
        raise ValueError(
            f"No row in video_info for video {video_name!r}."
        )
    px_per_mm = float(settings["pixels/mm"].values[0])
    fps = float(settings["fps"].values[0])
    width = int(settings["Resolution_width"].values[0])
    height = int(settings["Resolution_height"].values[0])
    return px_per_mm, fps, width, height


def _clip_pose_coords_to_frame(
    df: pd.DataFrame, video_width: int, video_height: int,
) -> pd.DataFrame:
    """Clip body-part x/y coordinates to frame bounds. Same logic as
    `feature_subsets_clip_fix.patch` in the patch stack — keeping it
    here so process_one_video can be self-contained."""
    for col in df.columns:
        cl = str(col).lower()
        if cl.endswith("_x"):
            df[col] = df[col].clip(lower=0, upper=int(video_width))
        elif cl.endswith("_y"):
            df[col] = df[col].clip(lower=0, upper=int(video_height))
    return df


def process_one_video(
    file_path: str,
    config: VideoProcessingConfig,
    file_idx: int = 0,
    n_total_files: int = 1,
    print_progress: bool = True,
) -> str:
    """Process feature extraction for a single video file.

    Reads the pose data, clips coordinates to frame bounds, runs each
    requested feature family kernel, builds a results DataFrame with
    the legacy ``_FEATURE_SUBSET`` column suffix and sorted column
    order, and writes the result to ``config.temp_dir/<video>.<ext>``.

    :param file_path: path to the per-video pose data file (parquet
        or csv per `config.file_type`)
    :param config: shared per-run configuration (feature_families,
        body-part combinations, etc.)
    :param file_idx: 1-indexed position of this file in the batch
        (only used for progress prints)
    :param n_total_files: total batch size (only used for progress
        prints)
    :param print_progress: whether to emit per-family / per-video
        progress lines. Set False in parallel runs where interleaved
        output is messy.
    :return: the save path written to.
    """
    video_name = get_fn_ext(filepath=file_path)[1]
    timer = SimbaTimer(start=True)
    save_path = os.path.join(
        config.temp_dir, f"{video_name}.{config.file_type}",
    )
    if print_progress:
        print(
            f"Analyzing video {video_name}... "
            f"({file_idx + 1}/{n_total_files})"
        )

    # Per-video metadata
    if config.video_info_df is None:
        raise RuntimeError(
            "VideoProcessingConfig.video_info_df is None — required "
            "for per-video metadata lookup."
        )
    px_per_mm, fps, video_width, video_height = _read_video_info_for(
        video_info_df=config.video_info_df, video_name=video_name,
    )

    # Read pose data and clip to frame bounds
    df = read_df(file_path=file_path, file_type=config.file_type)
    df = _clip_pose_coords_to_frame(df, video_width, video_height)

    # Run requested feature family kernels
    results: dict[str, np.ndarray] = {}
    for family_idx, family in enumerate(config.feature_families):
        if print_progress:
            print(
                f"Analyzing {video_name} and {family} "
                f"(Video {file_idx + 1}/{n_total_files}, "
                f"Family {family_idx + 1}/{len(config.feature_families)})..."
            )
        if family == TWO_POINT_BP_DISTANCES:
            results.update(compute_two_point_distances(
                df=df,
                two_point_combs=config.two_point_combs,
                px_per_mm=px_per_mm,
            ))
        elif family == WITHIN_ANIMAL_THREE_POINT_ANGLES:
            results.update(compute_three_point_angles(
                df=df,
                within_animal_three_point_combs=(
                    config.within_animal_three_point_combs
                ),
            ))
        elif family == WITHIN_ANIMAL_THREE_POINT_HULL:
            results.update(compute_three_point_hulls(
                df=df,
                within_animal_three_point_combs=(
                    config.within_animal_three_point_combs
                ),
                px_per_mm=px_per_mm,
            ))
        elif family == WITHIN_ANIMAL_FOUR_POINT_HULL:
            results.update(compute_four_point_hulls(
                df=df,
                within_animal_four_point_combs=(
                    config.within_animal_four_point_combs
                ),
                px_per_mm=px_per_mm,
            ))
        elif family == ANIMAL_CONVEX_HULL_PERIMETER:
            results.update(compute_animal_convex_hulls(
                df=df, animal_bps=config.animal_bps,
                px_per_mm=px_per_mm, method="perimeter",
            ))
        elif family == ANIMAL_CONVEX_HULL_AREA:
            results.update(compute_animal_convex_hulls(
                df=df, animal_bps=config.animal_bps,
                px_per_mm=px_per_mm, method="area",
            ))
        elif family == FRAME_BP_MOVEMENT:
            results.update(compute_framewise_movement(
                df=df, animal_bps=config.animal_bps,
                px_per_mm=px_per_mm, source=str(file_path),
            ))
        elif family == FRAME_BP_TO_ROI_CENTER:
            if config.roi_dict is None or video_name not in config.roi_dict:
                raise RuntimeError(
                    f"FRAME_BP_TO_ROI_CENTER requested but no ROI "
                    f"data for {video_name}."
                )
            results.update(compute_roi_center_distances(
                df=df, animal_bps=config.animal_bps,
                px_per_mm=px_per_mm,
                video_roi_dict=config.roi_dict[video_name],
                source=str(file_path),
            ))
        elif family == FRAME_BP_INSIDE_ROI:
            if config.roi_dict is None or video_name not in config.roi_dict:
                raise RuntimeError(
                    f"FRAME_BP_INSIDE_ROI requested but no ROI "
                    f"data for {video_name}."
                )
            results.update(compute_inside_roi(
                df=df, animal_bps=config.animal_bps,
                video_roi_dict=config.roi_dict[video_name],
                source=str(file_path),
            ))
        elif family == ARENA_EDGE:
            results.update(compute_distances_to_frame_edge(
                df=df, animal_bps=config.animal_bps,
                px_per_mm=px_per_mm,
                video_width=video_width, video_height=video_height,
                source=str(file_path),
            ))

    # Build the results dataframe with the legacy column-name layout:
    # add _FEATURE_SUBSET suffix, sort columns, fill NaN with -1.
    results_df = pd.DataFrame(results)
    results_df = results_df.add_suffix("_FEATURE_SUBSET")
    results_df = results_df[sorted(results_df.columns)]
    write_df(
        df=results_df.fillna(-1),
        file_type=config.file_type,
        save_path=save_path,
    )
    timer.stop_timer()
    if print_progress:
        print(
            f"Feature subsets computed for {video_name} complete "
            f"(elapsed time {timer.elapsed_time_str}s)..."
        )
    return save_path


__all__ = [
    "VideoProcessingConfig",
    "process_one_video",
    "TWO_POINT_BP_DISTANCES",
    "WITHIN_ANIMAL_THREE_POINT_ANGLES",
    "WITHIN_ANIMAL_THREE_POINT_HULL",
    "WITHIN_ANIMAL_FOUR_POINT_HULL",
    "ANIMAL_CONVEX_HULL_PERIMETER",
    "ANIMAL_CONVEX_HULL_AREA",
    "FRAME_BP_MOVEMENT",
    "FRAME_BP_TO_ROI_CENTER",
    "FRAME_BP_INSIDE_ROI",
    "ARENA_EDGE",
]
