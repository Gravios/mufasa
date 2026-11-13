from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter

__author__ = "Simon Nilsson; sronilsson@gmail.com"

import math
import os
import warnings
from copy import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mufasa.roi_tools.interactive_roi_bufferer import InteractiveROIBufferer

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon

from mufasa.utils.checks import (
    check_file_exist_and_readable,
    check_int,
    check_str,
    check_valid_array,
    check_valid_tuple,
    check_video_and_data_frm_count_align,
)
from mufasa.utils.enums import ROI_SETTINGS, Formats, Options
from mufasa.utils.errors import InvalidInputError, NoROIDataError, NotDirectoryError
from mufasa.utils.printing import stdout_success, stdout_trash
from mufasa.utils.read_write import (
    find_files_of_filetypes_in_directory,
    get_fn_ext,
    get_video_meta_data,
    read_df,
    read_roi_data,
)
from mufasa.utils.warnings import FrameRangeWarning, NoFileFoundWarning, VideoFileWarning
from mufasa.video_processors.roi_selector import ROISelector
from mufasa.video_processors.roi_selector_circle import ROISelectorCircle
from mufasa.video_processors.roi_selector_polygon import ROISelectorPolygon

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

DRAW_FRAME_NAME = "DEFINE SHAPE"


# ---------------------------------------------------------------------------
# Patch 122ek — defensive helpers for ROI consumers.
#
# Background: roi_logic.RoiLogic writes all three HDF keys (rectangles,
# circles, polygons) to the ROI definitions file even when the user only
# drew one shape type. The DataFrames for the unused shape types come back
# from ``pd.read_hdf`` as empty with NO columns at all. Naked
# ``df[df["Video"] == name]`` or ``df["Video"].unique()`` accesses raise
# KeyError 'Video' on these empty DataFrames.
#
# 122ej fixed the analogous bug in ``ConfigReader.read_roi_data`` (the dialog-
# open path). 122ek extends the same tolerance to the other ROI consumers
# audited in the same session: ``multiply_ROIs`` and ``reset_video_ROIs``
# here in roi_utils, plus the Duplicate ROIs dialog's Apply body and the ROI
# size standardizer.
# ---------------------------------------------------------------------------
def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty DataFrame with the same columns as ``df``. Used as
    the safe-filter return value when the column being filtered on is
    absent — preserves callers' downstream assumption that the result is
    still a DataFrame (just with zero rows)."""
    return df.iloc[0:0]


def safe_filter_by_video(
    df: pd.DataFrame, video_name: str,
) -> pd.DataFrame:
    """Filter ``df`` by ``Video == video_name``, tolerating DataFrames
    that don't have a ``Video`` column (the empty-from-roi_logic case
    described in the module-level note). Returns an empty DataFrame
    when the column is missing; standard pandas filter result
    otherwise.
    """
    if "Video" not in df.columns:
        return _empty_like(df)
    return df[df["Video"] == video_name]


def safe_filter_video_neq(
    df: pd.DataFrame, video_name: str,
) -> pd.DataFrame:
    """Filter ``df`` by ``Video != video_name``, tolerating missing
    ``Video`` column. Useful for "remove rows for this video"
    semantics in re-write loops."""
    if "Video" not in df.columns:
        return _empty_like(df)
    return df[df["Video"] != video_name]


def safe_videos_in(df: pd.DataFrame) -> list[str]:
    """Return list of unique videos in ``df["Video"]``, or empty list
    if the column is missing (empty-from-roi_logic case)."""
    if "Video" not in df.columns:
        return []
    return list(df["Video"].unique())


def create_rectangle_entry(rectangle_selector: ROISelector | InteractiveROIBufferer,
                           video_name: str,
                           shape_name: str,
                           clr_name: str,
                           clr_bgr: tuple[int, int, int],
                           thickness: int,
                           ear_tag_size: int,
                           px_conversion_factor: float):

    return {'Video':             video_name,
            'Shape_type':        ROI_SETTINGS.RECTANGLE.value,
            'Name':              shape_name,
            'Color name':        clr_name,
            'Color BGR':         clr_bgr,
            'Thickness':         thickness,
            'Center_X':          rectangle_selector.center[0],
            'Center_Y':          rectangle_selector.center[1],
            'topLeftX':          rectangle_selector.top_left[0],
            'topLeftY':          rectangle_selector.top_left[1],
            'Bottom_right_X':    rectangle_selector.bottom_right[0],
            'Bottom_right_Y':    rectangle_selector.bottom_right[1],
            'width':             rectangle_selector.width,
            'height':            rectangle_selector.height,
            'width_cm':          round((rectangle_selector.width / px_conversion_factor) / 10, 2),
            'height_cm':         round((rectangle_selector.height / px_conversion_factor) / 10, 2),
            'area_cm':           round((round((rectangle_selector.width / px_conversion_factor) / 10, 2) * round((rectangle_selector.height / px_conversion_factor) / 10, 2)), 2),
            "Tags":             {"Center tag": rectangle_selector.center,
                                 "Top left tag": rectangle_selector.top_left,
                                 "Bottom right tag": rectangle_selector.bottom_right,
                                 "Top right tag": rectangle_selector.top_right_tag,
                                 "Bottom left tag": rectangle_selector.bottom_left_tag,
                                 "Top tag": rectangle_selector.top_tag,
                                 "Right tag": rectangle_selector.right_tag,
                                 "Left tag": rectangle_selector.left_tag,
                                 "Bottom tag": rectangle_selector.bottom_tag},
            'Ear_tag_size':      ear_tag_size}

def create_circle_entry(circle_selector: ROISelectorCircle | InteractiveROIBufferer,
                        video_name: str,
                        shape_name: str,
                        clr_name: str,
                        clr_bgr: tuple[int, int, int],
                        thickness: int,
                        ear_tag_size: int,
                        px_conversion_factor: float):

    return {'Video':             video_name,
            'Shape_type':        ROI_SETTINGS.CIRCLE.value,
            'Name':              shape_name,
            'Color name':        clr_name,
            'Color BGR':         clr_bgr,
            'Thickness':         thickness,
            'centerX':           int(circle_selector.circle_center[0]),
            'centerY':           int(circle_selector.circle_center[1]),
            'radius':            int(circle_selector.circle_radius),
            'radius_cm':         round((circle_selector.circle_radius / px_conversion_factor) / 10, 2),
            'area_cm':           round(math.pi * (round((circle_selector.circle_radius / px_conversion_factor) / 10, 2) **2), 2),
            "Tags":             {"Center tag": circle_selector.circle_center,
                                 "Border tag": circle_selector.left_border_tag},
            'Ear_tag_size':      ear_tag_size}


def create_polygon_entry(polygon_selector: ROISelectorPolygon | InteractiveROIBufferer,
                         video_name: str,
                         shape_name: str,
                         clr_name: str,
                         clr_bgr: tuple[int, int, int],
                         thickness: int,
                         ear_tag_size: int,
                         px_conversion_factor: float) -> dict:

    return {'Video':                video_name,
            'Shape_type':           ROI_SETTINGS.POLYGON.value,
            'Name':                 shape_name,
            'Color name':           clr_name,
            'Color BGR':            clr_bgr,
            'Thickness':            thickness,
            'Center_X':             int(polygon_selector.polygon_centroid[0]),
            'Center_Y':             int(polygon_selector.polygon_centroid[1]),
            'vertices':             polygon_selector.polygon_arr,
            'center':               tuple(polygon_selector.polygon_centroid),
            'area':                 polygon_selector.polygon_area,
            'max_vertice_distance': polygon_selector.max_vertice_distance,
            "area_cm":              round(polygon_selector.polygon_area / px_conversion_factor, 2),
            'Tags':                 polygon_selector.tags,
            'Ear_tag_size':         ear_tag_size}

def create_duplicated_rectangle_entry(shape_entry: dict, jump_size: int) -> dict:
    shape_entry['Name'] = f'{shape_entry["Name"]}_duplicated'
    shape_entry['Center_X'] = shape_entry['Center_X'] + jump_size
    shape_entry['Center_Y'] = shape_entry['Center_Y'] + jump_size
    shape_entry['topLeftX'] = shape_entry['topLeftX'] + jump_size
    shape_entry['topLeftY'] = shape_entry['topLeftY'] + jump_size
    shape_entry['Bottom_right_X'] = shape_entry['Bottom_right_X'] + jump_size
    shape_entry['Bottom_right_Y'] = shape_entry['Bottom_right_Y'] + jump_size
    new_shape_tags = {}
    for tag_name, tag_value in shape_entry['Tags'].items():
        new_shape_tags[tag_name] = (int(tag_value[0] + jump_size), int(tag_value[1] + jump_size))
    shape_entry['Tags'] = new_shape_tags
    return shape_entry


def create_duplicated_circle_entry(shape_entry: dict, jump_size: int) -> dict:
    out = copy(shape_entry)
    out['Name'] = f'{out["Name"]}_duplicated'
    out['centerX'] = out['centerX'] + jump_size
    out['centerY'] = out['centerY'] + jump_size
    out['Tags'] = {'Center tag': ((out['centerX']), int(out['centerY'])),
                   'Border tag': (int(out['Tags']['Border tag'][0] + jump_size), int(out['Tags']['Border tag'][1] + jump_size))}
    return out

def create_duplicated_polygon_entry(shape_entry: dict, jump_size: int) -> dict:
    shape_entry['Name'] = f'{shape_entry["Name"]}_duplicated'
    shape_entry['Center_X'] = shape_entry['Center_X'] + jump_size
    shape_entry['Center_Y'] = shape_entry['Center_Y'] + jump_size
    shape_entry['vertices'] = shape_entry['vertices'] + jump_size
    new_shape_tags = {}
    for tag_name, tag_value in shape_entry['Tags'].items():
        new_shape_tags[tag_name] = (int(tag_value[0] + jump_size), int(tag_value[1] + jump_size))
    shape_entry['Tags'] = new_shape_tags
    return shape_entry

def get_rectangle_df_headers():
    return ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR', 'Thickness', 'Center_X', 'Center_Y', 'topLeftX', 'topLeftY', 'Bottom_right_X', 'Bottom_right_Y', 'width', 'height', 'width_cm', 'height_cm', 'area_cm', "Tags", 'Ear_tag_size']

def get_circle_df_headers():
    return ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR', 'Thickness', 'centerX', 'centerY', 'radius', 'radius_cm', 'area_cm', "Tags", 'Ear_tag_size']

def get_polygon_df_headers():
    return ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR', 'Thickness', 'Center_X', 'Center_Y', 'vertices', 'center', 'area', 'max_vertice_distance', 'area_cm', "Tags", 'Ear_tag_size']


def set_roi_metric_sizes(roi_dict: dict, px_conversion_factor: int | float) -> dict:
    """
    Helper to update the metric attributes of a shape dictionary.
    """
    out = {}
    for name, data in roi_dict.items():
        if data['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value:
            data['width_cm'] = round((data['width'] / px_conversion_factor) / 10, 2)
            data['height_cm'] = round((data['height'] / px_conversion_factor) / 10, 2)
            data['area_cm'] =  round((round((data['width'] / px_conversion_factor) / 10, 2) * round((data['height'] / px_conversion_factor) / 10, 2)), 2)
        elif data['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value:
            data['radius_cm'] = round((data['radius'] / px_conversion_factor) / 10, 2)
            data['area_cm'] = round(math.pi * (round((data['radius'] / px_conversion_factor) / 10, 2) ** 2), 2)
        elif data['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value:
            data['area'] = Polygon(data['vertices']).simplify(tolerance=20, preserve_topology=True).area
            data['area_cm'] = round(data['area'] / px_conversion_factor, 2)
            data['max_vertice_distance'] =  np.max(cdist(data['vertices'], data['vertices']).astype(np.int32))
        out[name] = data
    return out


def get_roi_df_from_dict(roi_dict: dict, video_name_nesting: bool | None = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Helper create DataFrames from a shape dictionary.

    If nesting is True, then the roi_dict has video name, and shape name keys.
    """
    rectangles_df, circles_df, polygon_df = pd.DataFrame(columns=get_rectangle_df_headers()), pd.DataFrame(columns=get_circle_df_headers()), pd.DataFrame(columns=get_polygon_df_headers())
    if not video_name_nesting:
        for shape_name, shape_data in roi_dict.items():
            if shape_data['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value:
                rectangles_df = pd.concat([rectangles_df, pd.DataFrame([shape_data])], ignore_index=True)
            elif shape_data['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value:
                circles_df = pd.concat([circles_df, pd.DataFrame([shape_data])], ignore_index=True)
            elif shape_data['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value:
                polygon_df = pd.concat([polygon_df, pd.DataFrame([shape_data])], ignore_index=True)
    else:
        for video_name, video_data in roi_dict.items():
            for shape_name, shape_data in video_data.items():
                if shape_data['Shape_type'].lower() == ROI_SETTINGS.RECTANGLE.value:
                    rectangles_df = pd.concat([rectangles_df, pd.DataFrame([shape_data])], ignore_index=True)
                elif shape_data['Shape_type'].lower() == ROI_SETTINGS.CIRCLE.value:
                    circles_df = pd.concat([circles_df, pd.DataFrame([shape_data])], ignore_index=True)
                elif shape_data['Shape_type'].lower() == ROI_SETTINGS.POLYGON.value:
                    polygon_df = pd.concat([polygon_df, pd.DataFrame([shape_data])], ignore_index=True)

    return (rectangles_df, circles_df, polygon_df)


def get_roi_dict_from_dfs(rectangle_df: pd.DataFrame,
                          circle_df: pd.DataFrame,
                          polygon_df: pd.DataFrame,
                          video_name_nesting: bool | None = False) -> dict:
    """
    Helper create dict from a shape dataframes.
    """

    out = {}
    for idx, row in rectangle_df.iterrows():
        if not video_name_nesting:
            out[row['Name']] = row.to_dict()
        else:
            if row['Video'] not in list(out.keys()):
                out[row['Video']] = {}
            out[row['Video']][row['Name']] = row.to_dict()
    for idx, row in circle_df.iterrows():
        if not video_name_nesting:
            out[row['Name']] = row.to_dict()
        else:
            if row['Video'] not in list(out.keys()):
                out[row['Video']] = {}
            out[row['Video']][row['Name']] = row.to_dict()
    for idx, row in polygon_df.iterrows():
        if not video_name_nesting:
            out[row['Name']] = row.to_dict()
        else:
            if row['Video'] not in list(out.keys()):
                out[row['Video']] = {}
            out[row['Video']][row['Name']] = row.to_dict()
    return out



def get_roi_data(roi_path: str | os.PathLike, video_name: str) -> tuple:
    """ Helper to read in ROI data from disk"""
    rectangles_df, circles_df, polygon_df = pd.DataFrame(columns=get_rectangle_df_headers()), pd.DataFrame(columns=get_circle_df_headers()), pd.DataFrame(columns=get_polygon_df_headers())
    other_roi_dict = {}
    roi_names, roi_dict = [''], {}
    other_video_names_w_rois = ['']

    if os.path.isfile(roi_path):
        in_rectangles_df, in_circles_df, in_polygon_df = read_roi_data(roi_path=roi_path)
        # Patch 122er-hotfix — same empty-DataFrame defensive
        # pattern as 122ek (multiply_ROIs / reset_video_ROIs).
        # Missed in the 122ek audit; surfaced when the user
        # exercised the ROI Definitions → Apply-to-selected
        # flow on a rectangles-only project. ``read_roi_data``
        # returns empty DataFrames (no columns) for unused
        # shape types; the naked .unique() and df['Video'] ==
        # filters then raise KeyError 'Video'.
        other_video_names_w_rois = list(set(
            list(safe_videos_in(in_rectangles_df))
            + list(safe_videos_in(in_circles_df))
            + list(safe_videos_in(in_polygon_df))
        ))
        other_video_names_w_rois = [x for x in other_video_names_w_rois if x != video_name]
        if len(other_video_names_w_rois) == 0: other_video_names_w_rois = ['']
        rectangles_df = safe_filter_by_video(in_rectangles_df, video_name).reset_index(drop=True)
        circles_df = safe_filter_by_video(in_circles_df, video_name).reset_index(drop=True)
        polygon_df = safe_filter_by_video(in_polygon_df, video_name).reset_index(drop=True)
        other_rectangles_df = safe_filter_video_neq(in_rectangles_df, video_name).reset_index(drop=True)
        other_circles_df = safe_filter_video_neq(in_circles_df, video_name).reset_index(drop=True)
        other_polygon_df = safe_filter_video_neq(in_polygon_df, video_name).reset_index(drop=True)
        other_roi_dict = get_roi_dict_from_dfs(rectangle_df=other_rectangles_df, circle_df=other_circles_df, polygon_df=other_polygon_df, video_name_nesting=True)
        if len(rectangles_df) + len(circles_df) + len(polygon_df) > 0:
            # Same defensive pattern for 'Name' column.
            def _names(df):
                return list(df['Name'].unique()) if 'Name' in df.columns else []
            roi_names = list(set(_names(rectangles_df) + _names(circles_df) + _names(polygon_df)))
            roi_dict = get_roi_dict_from_dfs(rectangle_df=rectangles_df, circle_df=circles_df, polygon_df=polygon_df, video_name_nesting=False)

    return (rectangles_df, circles_df, polygon_df, roi_dict, roi_names, other_roi_dict, other_video_names_w_rois)


def get_roi_data_for_video_name(roi_path: str | os.PathLike, video_name: str):
    in_rectangles_df, in_circles_df, in_polygon_df = read_roi_data(roi_path=roi_path)
    # Patch 122er-hotfix — same defensive pattern as 122ek.
    # The user's reported KeyError 'Video' fired here when
    # clicking Run in the "Apply ROIs to selected videos"
    # dialog on a rectangles-only project. ``read_roi_data``
    # returns empty DataFrames (no columns) for unused shape
    # types; naked df['Video'] filters raise KeyError.
    rectangles_df = safe_filter_by_video(in_rectangles_df, video_name).reset_index(drop=True)
    circles_df = safe_filter_by_video(in_circles_df, video_name).reset_index(drop=True)
    polygon_df = safe_filter_by_video(in_polygon_df, video_name).reset_index(drop=True)
    roi_dict = get_roi_dict_from_dfs(rectangle_df=rectangles_df, circle_df=circles_df, polygon_df=polygon_df, video_name_nesting=False)
    return roi_dict



def get_video_roi_data_from_dict(roi_dict: dict, video_name: str) -> dict:
    out = {}
    for video_key, data in roi_dict.items():
        if video_key == video_name:
            for shape_name, shape_data in data.items():
                out[shape_name] = shape_data
    return out


def change_roi_dict_video_name(roi_dict: dict, video_name: str) -> dict:
    out = {}
    for shape_name, shape_data in roi_dict.items():
        new_shape_data = copy(shape_data)
        new_shape_data['Video'] = video_name
        out[shape_name] = new_shape_data
    return out




def get_ear_tags_for_rectangle(center: tuple[int, int], width: int, height: int) -> dict[str, int | tuple[int, int]]:
    """
    Knowing the center, width, and height of rectangle, return its vertices.

    :param Tuple[int, int] center: The center x and y coordinates of the rectangle
    :param int width: The width of the rectangle in pixels.
    :param Tuple[int, int] width: The width of the rectangle in pixels.
    :return: Dictionary holding the name of the tag as key and coordinates as value.
    :rtype: Dict[str, Union[int, Tuple[int, int]]]
    """

    check_valid_tuple(x=center, source=get_ear_tags_for_rectangle.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='width', value=width, min_value=1)
    check_int(name='height', value=height, min_value=1)
    tags = {}
    # center is (x, y) format, so use center[0] for x and center[1] for y
    tags['Top left tag'] = (round(center[0] - (width/2)), round(center[1] - (height/2)))
    tags['Bottom right tag'] = (round(center[0] + (width/2)), round(center[1] + (height/2)))
    tags['Top right tag'] = (round(center[0] + (width/2)), round(center[1] - (height/2)))
    tags['Bottom left tag'] = (round(center[0] - (width / 2)), round(center[1] + (height / 2)))
    tags['Top tag'] = (round(center[0]), round(center[1] - (height / 2)))
    tags['Right tag'] = (round(center[0] + (width / 2)), round(center[1]))
    tags['Left tag'] = (round(center[0] - (width / 2)), round(center[1]))
    tags['Bottom tag'] = (round(center[0]), round(center[1] + (height / 2)))
    tags['Center tag'] = (round(center[0]), round(center[1]))
    return tags



def get_vertices_hexagon(center: tuple[int, int],
                         radius: int) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """
    Generates the vertices of a regular hexagon centered at a given point with a specified radius.

    :param Tuple[int, int] center:  A tuple (x, y) representing the center coordinates of the hexagon. Must contain exactly two numeric values.
    :param int radius: The radius of the hexagon, which represents the distance from the center to any of the vertices. Must be a positive integer.
    :return: A tuple containing vertices as array and dict.
    :rtype: Tuple[np.ndarray, Dict[str, Tuple[int, int]]]
    """

    check_valid_tuple(x=center, source=get_vertices_hexagon.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='radius', value=radius, min_value=1)
    vertices = []
    x_c, y_c = center
    for i in range(6):
        angle_rad = np.deg2rad(60 * i)
        x_i = x_c + radius * np.cos(angle_rad)
        y_i = y_c + radius * np.sin(angle_rad)
        vertices.append((x_i, y_i))

    vertices_dict = {"Center_tag": (center[0], center[1])}
    for tag_id, tag in enumerate(vertices):
        vertices_dict[f"Tag_{tag_id}"] = (round(tag[0]), round(tag[1]))
    return (np.round(np.array(vertices)).astype(np.int32), vertices_dict)


def get_half_circle_vertices(center: tuple[int, int],
                             radius: int,
                             direction: str,
                             n_points: int | None = 50) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:

    """
    Generates vertices for a half-circle with a given radius and direction, centered at a specific point.

    :param Tuple[int, int] center: A tuple (x, y) representing the center coordinates of the half-circle. Must contain exactly two numeric values.
    :param int radius: The radius of the half-circle. Must be a positive integer.
    :param str direction: The direction in which the half-circle is oriented.
    :param Optional[int] n_points: The number of vertices used to approximate the half-circle. Defaults to 50.
    :return: A tuple containing vertices as array and dict.
    :rtype: Tuple[np.ndarray, Dict[str, Tuple[int, int]]]
    """

    check_valid_tuple(x=center, source=get_vertices_hexagon.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='radius', value=radius, min_value=1)
    check_str(name='direction', options=['NORTH', 'SOUTH', 'WEST', 'EAST', 'NORTH-EAST', 'NORTH-WEST', 'SOUTH-EAST', 'SOUTH-WEST'], value=direction)
    x_c, y_c = center
    if direction == "WEST":
        a = np.linspace(np.pi / 2, 3 * np.pi / 2, n_points)
    elif direction == "EAST":
        a = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    elif direction == "SOUTH":
        a = np.linspace(0, np.pi, n_points)
    elif direction == 'SOUTH-WEST':
        a = np.linspace(np.pi / 4, 5 * np.pi / 4, n_points)
    elif direction == 'SOUTH-EAST':
        a = np.linspace(-np.pi / 4, 3 * np.pi / 4, n_points)
    elif direction == 'NORTH-WEST':
        a = np.linspace(3 * np.pi / 4, 7 * np.pi / 4, n_points)
    elif direction == 'NORTH-EAST':
        a = np.linspace(5 * np.pi / 4, 9 * np.pi / 4, n_points)
    else:
        a = np.linspace(np.pi, 2 * np.pi, n_points)
    x, y = x_c + radius * np.cos(a), y_c + radius * np.sin(a)
    vertices = np.round(np.column_stack((x, y))).astype(np.int32)
    shape_center = np.round(np.array(Polygon(vertices).centroid.coords)[0]).astype(np.int32)
    vertices_dict = {"Center_tag": (shape_center[0], shape_center[1])}

    for tag_id in range(vertices.shape[0]):
        vertices_dict[f"Tag_{tag_id}"] = (vertices[tag_id][0], vertices[tag_id][1])
    return (np.round(np.array(vertices)).astype(np.int32), vertices_dict)



def get_triangle_vertices(center: tuple[int, int], side_length: int, direction: int) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """
    Find equilateral triangle vertices knowing the center, direction and length side.

    :param Tuple[int, int] center: A tuple (x, y) representing the center coordinates of the hexagon. Must contain exactly two numeric values.
    :param int side_length: The length of each of the three sides in pixels.
    :param int direction: The direction to point the vertices.
    :return: 2-part tuple containing vertices as array and dict.
    :rtype: Tuple[np.ndarray, Dict[str, Tuple[int, int]]]
    """
    check_valid_tuple(x=center, source=get_vertices_hexagon.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='side_length', value=side_length, min_value=1)
    check_int(name='direction', value=direction, min_value=0, max_value=360)

    direction_radians = np.radians(direction)
    radius = side_length / np.sqrt(3)

    top_vertex = (center[0] + radius * np.cos(direction_radians), center[1] + radius * np.sin(direction_radians))
    vertex2 = (center[0] + radius * np.cos(direction_radians + np.radians(120)), center[1] + radius * np.sin(direction_radians + np.radians(120)))
    vertex3 = (center[0] + radius * np.cos(direction_radians + np.radians(-120)), center[1] + radius * np.sin(direction_radians + np.radians(-120)))
    vertices = np.round(np.array([top_vertex, vertex2, vertex3, top_vertex])).astype(np.int32)
    vertices_dict = {"Center_tag": (int(center[0]), int(center[1])), 'Tag_0':  (int(top_vertex[0]), int(top_vertex[1])), 'Tag_1': (int(vertex2[0]), int(vertex2[1])), 'Tag_2': (int(vertex3[0]), int(vertex3[1]))}
    return (vertices, vertices_dict)



def multiply_ROIs(filename: str | os.PathLike,
                  config_path: str | os.PathLike | None = None,
                  roi_coordinates_path: str | os.PathLike | None = None,
                  videos_dir: str | os.PathLike | None = None) -> None:
    """Copy one video's ROIs to every other video in the project.

    Each other video's own ROIs are replaced. ROIs are rescaled when a
    target video's frame size differs from the source's (patch 122il).

    :param filename: path of the source video.
    :param config_path: project config; resolves the ROI file and videos dir.
    """
    from mufasa.roi_tools.roi_store import load_rois, read_frame_sizes, update_rois
    if config_path is not None:
        check_file_exist_and_readable(file_path=config_path)
        from mufasa.project_layout import project_paths_from_config
        _paths = project_paths_from_config(config_path=config_path)
        videos_dir = _paths["video_dir"]
        roi_coordinates_path = _paths["roi_definitions_path"]
    check_file_exist_and_readable(file_path=filename)
    _, video_name, _ = get_fn_ext(filename)
    if not os.path.isdir(videos_dir):
        raise NotDirectoryError(msg=f'Could not find the videos directory in the Mufasa project. Mufasa expected a directory at location: {videos_dir}')
    if not load_rois(roi_coordinates_path).rois(video_name):
        raise NoROIDataError(msg=f"Cannot replicate ROIs to all other videos: {video_name} has no ROIs.", source=multiply_ROIs.__name__)
    others = [x for x in find_files_of_filetypes_in_directory(directory=videos_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value)
              if get_fn_ext(x)[1] != video_name]
    if not others:
        raise NoROIDataError(msg=f"Cannot replicate ROIs to other videos. No other videos exist in project {videos_dir} directory.", source=multiply_ROIs.__name__)
    sizes = read_frame_sizes(others + [os.fspath(filename)])
    targets = [get_fn_ext(x)[1] for x in others]
    update_rois(roi_coordinates_path, lambda st: st.copy_video(video_name, targets, sizes))
    stdout_success(msg=f"ROIs of {video_name} copied to {len(targets)} other videos.")


def copy_roi_to_all(filename: str | os.PathLike, roi_name: str,
                    config_path: str | os.PathLike) -> list[str]:
    """Copy ONE ROI of a video to every other video of the project (patch
    122jh). A same-named ROI in another video is replaced; that video's
    other ROIs are kept. Rescaled to each video's frame size. Returns the
    target video names."""
    from mufasa.project_layout import project_paths_from_config
    from mufasa.roi_tools.roi_store import read_frame_sizes, update_rois
    paths = project_paths_from_config(config_path=config_path)
    videos_dir, roi_path = paths["video_dir"], paths["roi_definitions_path"]
    _, video_name, _ = get_fn_ext(filename)
    others = [x for x in find_files_of_filetypes_in_directory(
        directory=videos_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value)
        if get_fn_ext(x)[1] != video_name]
    if not others:
        raise NoROIDataError(msg="No other videos in the project.",
                             source=copy_roi_to_all.__name__)
    sizes = read_frame_sizes(others + [os.fspath(filename)])
    targets = [get_fn_ext(x)[1] for x in others]
    update_rois(roi_path, lambda st: st.copy_roi(video_name, roi_name, targets, sizes))
    return targets


def reset_video_ROIs(config_path: str | os.PathLike,
                     filename: str | os.PathLike) -> None:
    """Delete all ROIs of one video."""
    from mufasa.project_layout import project_paths_from_config
    from mufasa.roi_tools.roi_store import load_rois, update_rois
    check_file_exist_and_readable(file_path=config_path)
    _, video_name, _ = get_fn_ext(filename)
    roi_path = project_paths_from_config(config_path=config_path)["roi_definitions_path"]
    n = len(load_rois(roi_path).rois(video_name))
    if n == 0:
        raise NoROIDataError(msg=f"Cannot delete ROIs for video {video_name}: it has none.", source=reset_video_ROIs.__name__)
    update_rois(roi_path, lambda st: st.remove_video(video_name))
    stdout_trash(msg=f"Deleted {n} ROI(s) of video {video_name}.")

def get_image_from_label(tk_lbl: tkinter.Label) -> np.ndarray:
    """ Given a tkinter label with an image, retrieve image in array format"""

    if not hasattr(tk_lbl, 'image'):
        raise InvalidInputError(msg=f'The label {tk_lbl} does not have a valid image')
    else:
        from PIL import ImageTk
        tk_img = tk_lbl.image
        pil_image = ImageTk.getimage(tk_img)
        img = np.asarray(pil_image)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_pose_for_roi_ui(pose_path: str | os.PathLike,
                        video_path: str | os.PathLike) -> None | np.ndarray:
    video_meta_data = get_video_meta_data(video_path=video_path, raise_error=False)
    if video_meta_data is None:
        VideoFileWarning(msg=f'Cannot plot pose on ROI as cannot read meta data for video {video_path}.', source=get_pose_for_roi_ui.__name__)
        return None
    file_readable = check_file_exist_and_readable(file_path=pose_path, raise_error=False)
    if not file_readable:
        NoFileFoundWarning(msg=f'Cannot plot tracking in ROI window: {pose_path} is unreadable.', source=get_pose_for_roi_ui.__name__)
        return None
    pose_df = read_df(file_path=pose_path, file_type='csv')
    data_align = check_video_and_data_frm_count_align(video=video_path, data=pose_path, raise_error=False)
    if not data_align:
        FrameRangeWarning(msg=f'Cannot plot tracking in ROI window: The data contains {len(pose_path)} frames and the video has {video_meta_data["frame_count"]} frames', source=get_pose_for_roi_ui.__name__)
        return None
    pose_df = pose_df.drop(pose_df.columns[2::3], axis=1)
    pose_data = pose_df.values.reshape(len(pose_df), round(len(pose_df.columns) / 2), 2).astype(np.int32)
    valid_pose = check_valid_array(data=pose_data, source=f'{get_pose_for_roi_ui.__name__} pose_data', accepted_ndims=(3,), accepted_axis_0_shape=[video_meta_data['frame_count']], accepted_dtypes=Formats.INTEGER_DTYPES.value, raise_error=False)
    if not valid_pose:
        FrameRangeWarning(msg=f'Cannot plot tracking in ROI window: The pose data from path {pose_path} is not a 3D numeric array with length {video_meta_data["frame_count"]}', source=get_pose_for_roi_ui.__name__)
    return pose_data


def insert_gridlines_on_roi_img(img: np.ndarray,
                                grid: list[Polygon],
                                color: tuple[int, int, int],
                                thickness: int) -> np.ndarray:

    if grid is None or len(grid) == 0:
        return img
    else:
        try:
            for polygon in grid:
                cords = np.round(np.array(polygon.exterior.coords)).astype(np.int32)
                img = cv2.polylines(img=img, pts=[cords], isClosed=True, color=color, thickness=thickness, lineType=8)
            return img
        except Exception as e:
            msg = f'Cannot draw gridlines: {e.args}'
            raise InvalidInputError(msg=msg, source=f'{insert_gridlines_on_roi_img.__name__} draw')
