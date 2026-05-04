"""
mufasa.ui_qt.pages.data_import_page
===================================

Data Import workbench page. Two sections:

* **Import pose-estimation data** — :class:`PoseImportForm`. Loads
  pose data into the currently-open project's ``csv/input_csv/``
  tree.
* **Video parameters & calibration** — :class:`VideoInfoForm`.
  Sets the per-video FPS, resolution, and pixels/mm calibration
  used by all distance-based feature kernels. Without it,
  distance features come out in pixels rather than millimeters.

The cross-format converter (DLC→YOLO, Labelme→DataFrame, etc.)
used to live here but was moved to the Tools page: conversion
doesn't need a project, is used infrequently, and clutters the
page a user visits every session. See
:mod:`mufasa.ui_qt.pages.tools_page`.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.pose_import import PoseImportForm
from mufasa.ui_qt.forms.video_info import VideoInfoForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_data_import_page(workbench,
                           config_path: Optional[str] = None
                           ) -> WorkflowPage:
    page = workbench.add_page("Data import", icon_name="pose")
    page.add_section("Import pose-estimation data",
                     [(PoseImportForm, {})])
    page.add_section("Video parameters & calibration",
                     [(VideoInfoForm, {})])
    return page


__all__ = ["build_data_import_page"]
