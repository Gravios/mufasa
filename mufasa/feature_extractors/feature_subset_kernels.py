"""
mufasa.feature_extractors.feature_subset_kernels
================================================

Pure-function kernels for feature subset computation. Each kernel
takes only the data it needs (a pose dataframe + parameters) and
returns a dict mapping output column names to numpy arrays.

These kernels are extracted from `FeatureSubsetsCalculator`'s
per-feature-family methods, which previously read from mutable
``self`` state (`self.data_df`, `self.results`, etc.). The
extraction:

* Makes each kernel testable in isolation with a synthetic pose
  dataframe — no need to instantiate a full ``ConfigReader``-backed
  class with a real project config.
* Removes hidden coupling between methods through ``self.results``
  (which one method appends to, the next reads from indirectly).
* Sets up the codebase for future per-video parallelization: the
  per-video orchestration can now pass these kernels a dataframe
  + config and collect results, without picklability concerns
  about a heavyweight class instance.

All kernels return a ``dict[str, np.ndarray]`` mapping output column
name (matching the legacy class-method behavior exactly) to a 1D
array of length ``len(df)``. The caller is responsible for
collecting these dicts into a results DataFrame.

Behavioral equivalence
----------------------

Each kernel produces output that is **byte-identical** to the
corresponding ``FeatureSubsetsCalculator`` method's effect on
``self.results``. Verified by ``tests/smoke_feature_kernels.py``
which exercises each kernel against synthetic data and compares
to the legacy class-method output.

Coverage
--------

1. ``compute_two_point_distances`` — replaces
   ``_get_two_point_bp_distances``
2. ``compute_three_point_angles`` — replaces ``__get_three_point_angles``
3. ``compute_three_point_hulls`` — replaces ``__get_three_point_hulls``
4. ``compute_four_point_hulls`` — replaces ``__get_four_point_hulls``
5. ``compute_animal_convex_hulls`` — replaces ``__get_convex_hulls``
6. ``compute_framewise_movement`` — replaces ``__get_framewise_movement``
7. ``compute_roi_center_distances`` — replaces ``__get_roi_center_distances``
8. ``compute_distances_to_frame_edge`` — replaces ``__get_distances_to_frm_edge``
9. ``compute_inside_roi`` — replaces ``__get_inside_roi``

The class methods are not removed in this patch — the class
delegates to the kernels but keeps its public surface intact.
A future patch will collapse the class methods after the kernels
are battle-tested.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

from mufasa.feature_extractors.perimeter_jit import jitted_hull as _numba_jitted_hull
from mufasa.mixins.feature_extraction_mixin import FeatureExtractionMixin
from mufasa.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from mufasa.utils.checks import check_valid_dataframe
from mufasa.utils.enums import ROI_SETTINGS, Formats


# ---------------------------------------------------------------------- #
# Cython kernel wiring
# ---------------------------------------------------------------------- #
# All seven hot kernels in this module have ahead-of-time-compiled
# Cython equivalents in mufasa._native, validated byte-equivalent
# against the numba reference (see tests/smoke_native_all_kernels.py
# for the verification suite). Speedups vs numba on a 9800X3D:
#   border_distances:                 ~340×
#   hull (perimeter/area):            ~13-110× (parallel)
#   framewise_euclidean_distance_roi: ~50×
#   framewise_euclidean_distance:     ~5-10×
#   inside_circle:                    ~1.8×
#   inside_polygon, inside_rectangle: ~1.0-1.2×
#   angle3pt_vectorized:              ~0.9× (slight regression;
#                                            numba's fastmath atan2
#                                            beats libc atan2)
#
# The angle3pt regression is small (~7% slower on a kernel that
# itself takes ~17ms at 500K frames) and the Cython version is
# preferred anyway for consistency.
#
# Defensive fallback: if any Cython kernel fails to import (e.g.
# user pulled new code without re-running `pip install -e .`,
# or the build failed silently), we fall back to the numba
# version. This keeps feature extraction working — degraded
# perf, never broken behavior.
try:
    from mufasa._native.angle3pt import angle3pt_vectorized as _kern_angle3pt
    from mufasa._native.border_distances import border_distances as _kern_border_distances
    from mufasa._native.euclidean_distance import (
        framewise_euclidean_distance as _kern_euclid,
        framewise_euclidean_distance_roi as _kern_euclid_roi,
    )
    from mufasa._native.hull import jitted_hull as _kern_hull
    from mufasa._native.inside_circle import is_inside_circle as _kern_inside_circle
    from mufasa._native.inside_polygon import (
        framewise_inside_polygon_roi as _kern_inside_polygon,
    )
    from mufasa._native.inside_rectangle import (
        framewise_inside_rectangle_roi as _kern_inside_rectangle,
    )
    _NATIVE_AVAILABLE = True
except ImportError as _native_import_error:
    # Fall back to numba for every kernel. A missing _native is
    # almost always "user updated the source but didn't reinstall
    # the package" — print a hint so they know how to fix it.
    import warnings
    warnings.warn(
        f"mufasa._native Cython kernels unavailable "
        f"({_native_import_error}); falling back to numba "
        f"implementations. Run `pip install -e .` from the "
        f"repo root to rebuild the Cython extensions for full "
        f"performance.",
        RuntimeWarning,
        stacklevel=2,
    )
    _kern_angle3pt = FeatureExtractionMixin.angle3pt_vectorized
    _kern_border_distances = FeatureExtractionSupplemental.border_distances
    _kern_euclid = FeatureExtractionMixin.framewise_euclidean_distance
    _kern_euclid_roi = FeatureExtractionMixin.framewise_euclidean_distance_roi
    _kern_hull = _numba_jitted_hull
    _kern_inside_circle = FeatureExtractionMixin.is_inside_circle
    _kern_inside_polygon = FeatureExtractionMixin.framewise_inside_polygon_roi
    _kern_inside_rectangle = FeatureExtractionMixin.framewise_inside_rectangle_roi
    _NATIVE_AVAILABLE = False


SHAPE_TYPE = "Shape_type"


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _bp_xy_columns(bp: str) -> Tuple[str, str]:
    """Return the (x_col, y_col) names for a body-part."""
    return f"{bp}_x", f"{bp}_y"


def _flat_xy_column_names(bps: Iterable[str]) -> list[str]:
    """For body-parts ``[a, b, c]`` return
    ``['a_x', 'a_y', 'b_x', 'b_y', 'c_x', 'c_y']``.

    This is the layout `angle3pt_vectorized` expects (flat) and
    also the input layout to the hull reshape.
    """
    out = []
    for bp in bps:
        out.append(f"{bp}_x")
        out.append(f"{bp}_y")
    return out


# ---------------------------------------------------------------------- #
# Kernel 1: two-point body-part distances
# ---------------------------------------------------------------------- #
def compute_two_point_distances(
    df: pd.DataFrame,
    two_point_combs: np.ndarray,
    px_per_mm: float,
) -> Dict[str, np.ndarray]:
    """Per-frame Euclidean distance between every pair of body-parts.

    :param df: pose DataFrame with columns ``{bp}_x`` / ``{bp}_y`` for
        every body-part referenced in ``two_point_combs``.
    :param two_point_combs: ``(n_pairs, 2)`` array of body-part name
        pairs.
    :param px_per_mm: pixels per millimeter for the recording.
    :return: ``{column_name: np.ndarray}`` for each pair, where
        column names match the legacy
        ``"Distance (mm) {bp_a}-{bp_b}"`` format.
    """
    out: Dict[str, np.ndarray] = {}
    for c in two_point_combs:
        bp_a, bp_b = c[0], c[1]
        x_a, y_a = _bp_xy_columns(bp_a)
        x_b, y_b = _bp_xy_columns(bp_b)
        bp1 = df[[x_a, y_a]].values
        bp2 = df[[x_b, y_b]].values
        out[f"Distance (mm) {bp_a}-{bp_b}"] = (
            FeatureExtractionMixin.bodypart_distance(
                bp1_coords=bp1.astype(np.int32),
                bp2_coords=bp2.astype(np.int32),
                px_per_mm=np.float64(px_per_mm),
                in_centimeters=False,
            )
        )
    return out


# ---------------------------------------------------------------------- #
# Kernel 2: within-animal three-point angles
# ---------------------------------------------------------------------- #
def compute_three_point_angles(
    df: pd.DataFrame,
    within_animal_three_point_combs: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Per-frame angles for every 3-body-part combination per animal.

    :param df: pose DataFrame.
    :param within_animal_three_point_combs: dict of animal name →
        ``(n_combs, 3)`` array of body-part triples.
    :return: ``{column_name: np.ndarray}``. Note the legacy column
        format does NOT include the animal name, only the three
        body-part names.
    """
    out: Dict[str, np.ndarray] = {}
    for animal, points in within_animal_three_point_combs.items():
        for point in points:
            col_names = _flat_xy_column_names(point)
            out[
                f"Angle (degrees) {point[0]}-{point[1]}-{point[2]}"
            ] = _kern_angle3pt(
                data=df[col_names].values,
            )
    return out


# ---------------------------------------------------------------------- #
# Helper for hull kernels (kernels 3, 4, 5)
# ---------------------------------------------------------------------- #
def _reshape_for_hull(df: pd.DataFrame, bps: Iterable[str]) -> np.ndarray:
    """Build a ``(n_frames, n_bps, 2)`` float32 array from the pose
    DataFrame, ordered as ``[[bp_0_x, bp_0_y], [bp_1_x, bp_1_y], ...]``
    per frame. This is the layout `jitted_hull` expects."""
    bps = list(bps)
    col_names = _flat_xy_column_names(bps)
    return np.reshape(
        df[col_names].values,
        (len(df), len(bps), 2),
    ).astype(np.float32)


# ---------------------------------------------------------------------- #
# Kernel 3: three-point convex hull perimeter
# ---------------------------------------------------------------------- #
def compute_three_point_hulls(
    df: pd.DataFrame,
    within_animal_three_point_combs: Mapping[str, np.ndarray],
    px_per_mm: float,
) -> Dict[str, np.ndarray]:
    """Convex hull perimeter (mm) for every 3-body-part combination."""
    out: Dict[str, np.ndarray] = {}
    for animal, points in within_animal_three_point_combs.items():
        for point in points:
            arr = _reshape_for_hull(df, point)
            col = (
                f"{animal} three-point convex hull perimeter (mm) "
                f"{point[0]}-{point[1]}-{point[2]}"
            )
            out[col] = (
                _kern_hull(points=arr, target=Formats.PERIMETER.value)
                / px_per_mm
            )
    return out


# ---------------------------------------------------------------------- #
# Kernel 4: four-point convex hull perimeter
# ---------------------------------------------------------------------- #
def compute_four_point_hulls(
    df: pd.DataFrame,
    within_animal_four_point_combs: Mapping[str, np.ndarray],
    px_per_mm: float,
) -> Dict[str, np.ndarray]:
    """Convex hull perimeter (mm) for every 4-body-part combination."""
    out: Dict[str, np.ndarray] = {}
    for animal, points in within_animal_four_point_combs.items():
        for point in points:
            arr = _reshape_for_hull(df, point)
            col = (
                f"{animal} four-point convex perimeter (mm) "
                f"{point[0]}-{point[1]}-{point[2]}-{point[3]}"
            )
            out[col] = (
                _kern_hull(points=arr, target=Formats.PERIMETER.value)
                / px_per_mm
            )
    return out


# ---------------------------------------------------------------------- #
# Kernel 5: full-animal convex hull (perimeter or area)
# ---------------------------------------------------------------------- #
def compute_animal_convex_hulls(
    df: pd.DataFrame,
    animal_bps: Mapping[str, list],
    px_per_mm: float,
    method: str,
) -> Dict[str, np.ndarray]:
    """Convex hull perimeter or area (mm or mm²) for the full body-part
    set per animal.

    :param method: ``'perimeter'`` or anything else for area (matches
        legacy class behavior — the legacy ``__get_convex_hulls`` does
        ``if method == 'perimeter': ... else: ...`` so any non-perimeter
        string falls into the area branch).
    """
    out: Dict[str, np.ndarray] = {}
    for animal, bps in animal_bps.items():
        arr = _reshape_for_hull(df, bps)
        if method == "perimeter":
            out[f"{animal} convex hull perimeter (mm)"] = (
                _kern_hull(points=arr, target=Formats.PERIMETER.value)
                / px_per_mm
            )
        else:
            out[f"{animal} convex hull area (mm2)"] = (
                _kern_hull(points=arr, target=Formats.AREA.value)
                / px_per_mm
            )
    return out


# ---------------------------------------------------------------------- #
# Kernel 6: frame-by-frame body-part movement (between consecutive frames)
# ---------------------------------------------------------------------- #
def compute_framewise_movement(
    df: pd.DataFrame,
    animal_bps: Mapping[str, list],
    px_per_mm: float,
    source: str = "compute_framewise_movement",
) -> Dict[str, np.ndarray]:
    """Frame-to-frame movement distance for each body-part."""
    out: Dict[str, np.ndarray] = {}
    for animal, bps in animal_bps.items():
        for bp in bps:
            x_col, y_col = _bp_xy_columns(bp)
            check_valid_dataframe(
                df=df, source=source,
                required_fields=[x_col, y_col],
            )
            bp_arr = FeatureExtractionMixin.create_shifted_df(
                df=df[[x_col, y_col]]
            ).values
            x, y = bp_arr[:, 0:2], bp_arr[:, 2:4]
            out[f"{animal} movement {bp} (mm)"] = (
                _kern_euclid(
                    location_1=x.astype(np.float64),
                    location_2=y.astype(np.float64),
                    px_per_mm=np.float64(px_per_mm),
                    centimeter=False,
                )
            )
    return out


# ---------------------------------------------------------------------- #
# Kernel 7: distances from each body-part to ROI center
# ---------------------------------------------------------------------- #
def compute_roi_center_distances(
    df: pd.DataFrame,
    animal_bps: Mapping[str, list],
    px_per_mm: float,
    video_roi_dict: Mapping[str, dict],
    source: str = "compute_roi_center_distances",
) -> Dict[str, np.ndarray]:
    """Per-frame distance from each body-part to each ROI's center.

    :param video_roi_dict: ``{roi_name: roi_data_dict}`` for the
        current video. The roi_data_dict must have ``'Center_X'``
        and ``'Center_Y'`` keys.
    """
    out: Dict[str, np.ndarray] = {}
    for animal, bps in animal_bps.items():
        for bp in bps:
            x_col, y_col = _bp_xy_columns(bp)
            check_valid_dataframe(
                df=df, source=source,
                required_fields=[x_col, y_col],
            )
            bp_arr = df[[x_col, y_col]].values.astype(np.float32)
            for roi_name, roi_data in video_roi_dict.items():
                center_point = np.array(
                    [roi_data["Center_X"], roi_data["Center_Y"]]
                ).astype(np.int32)
                out[
                    f"{animal} {bp} to {roi_name} center distance (mm)"
                ] = _kern_euclid_roi(
                    location_1=bp_arr,
                    location_2=center_point,
                    px_per_mm=px_per_mm,
                )
    return out


# ---------------------------------------------------------------------- #
# Kernel 8: distances from each body-part to each frame edge
# ---------------------------------------------------------------------- #
def compute_distances_to_frame_edge(
    df: pd.DataFrame,
    animal_bps: Mapping[str, list],
    px_per_mm: float,
    video_width: int,
    video_height: int,
    source: str = "compute_distances_to_frame_edge",
) -> Dict[str, np.ndarray]:
    """Per-frame distance from each body-part to each of the four
    frame edges (left, right, top, bottom)."""
    out: Dict[str, np.ndarray] = {}
    img_resolution = np.array(
        [video_width, video_height], dtype=np.int32,
    )
    for animal, bps in animal_bps.items():
        for bp in bps:
            x_col, y_col = _bp_xy_columns(bp)
            check_valid_dataframe(
                df=df, source=source,
                required_fields=[x_col, y_col],
            )
            bp_arr = df[[x_col, y_col]].values.astype(np.float32)
            distance = _kern_border_distances(
                data=bp_arr,
                pixels_per_mm=px_per_mm,
                img_resolution=img_resolution,
                time_window=1, fps=1,
            )
            out[f"{animal} {bp} to left video edge distance (mm)"] = (
                distance[:, 0]
            )
            out[f"{animal} {bp} to right video edge distance (mm)"] = (
                distance[:, 1]
            )
            out[f"{animal} {bp} to top video edge distance (mm)"] = (
                distance[:, 2]
            )
            out[f"{animal} {bp} to bottom video edge distance (mm)"] = (
                distance[:, 3]
            )
    return out


# ---------------------------------------------------------------------- #
# Kernel 9: per-frame body-part inside ROI (boolean)
# ---------------------------------------------------------------------- #
def compute_inside_roi(
    df: pd.DataFrame,
    animal_bps: Mapping[str, list],
    video_roi_dict: Mapping[str, dict],
    source: str = "compute_inside_roi",
) -> Dict[str, np.ndarray]:
    """Per-frame boolean: is the body-part inside each ROI?

    Handles rectangle, circle, and polygon ROIs. Output column names
    include the shape type (matching the legacy class behavior).
    """
    out: Dict[str, np.ndarray] = {}
    for animal, bps in animal_bps.items():
        for bp in bps:
            x_col, y_col = _bp_xy_columns(bp)
            check_valid_dataframe(
                df=df, source=source,
                required_fields=[x_col, y_col],
            )
            bp_arr = df[[x_col, y_col]].values.astype(np.float32)
            for roi_name, roi_data in video_roi_dict.items():
                shape = roi_data[SHAPE_TYPE]
                if shape == ROI_SETTINGS.RECTANGLE.value:
                    roi_coords = np.array([
                        [roi_data["topLeftX"], roi_data["topLeftY"]],
                        [roi_data["Bottom_right_X"],
                         roi_data["Bottom_right_Y"]],
                    ])
                    out[
                        f"{animal} {bp} inside rectangle "
                        f"{roi_name} (Boolean)"
                    ] = _kern_inside_rectangle(
                        bp_location=bp_arr,
                        roi_coords=roi_coords,
                    )
                elif shape == ROI_SETTINGS.CIRCLE.value:
                    circle_center = np.array(
                        [roi_data["Center_X"], roi_data["Center_Y"]]
                    ).astype(np.int32)
                    out[
                        f"{animal} {bp} inside circle "
                        f"{roi_name} (Boolean)"
                    ] = _kern_inside_circle(
                        bp=bp_arr,
                        roi_center=circle_center,
                        roi_radius=roi_data["radius"],
                    )
                elif shape == ROI_SETTINGS.POLYGON.value:
                    vertices = roi_data["vertices"].astype(np.int32)
                    out[
                        f"{animal} {bp} inside polygon "
                        f"{roi_name} (Boolean)"
                    ] = _kern_inside_polygon(
                        bp_location=bp_arr,
                        roi_coords=vertices,
                    )
    return out


__all__ = [
    "compute_two_point_distances",
    "compute_three_point_angles",
    "compute_three_point_hulls",
    "compute_four_point_hulls",
    "compute_animal_convex_hulls",
    "compute_framewise_movement",
    "compute_roi_center_distances",
    "compute_distances_to_frame_edge",
    "compute_inside_roi",
]
