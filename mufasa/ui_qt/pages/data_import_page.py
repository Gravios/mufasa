"""
mufasa.ui_qt.pages.data_import_page
===================================

Data Import workbench page. Four sections:

* **Import Pose Data** — :class:`PoseImportForm`.
  Loads pose data into the currently-open project's
  ``sources/pose/`` (v1) or ``csv/input_csv/`` (legacy) tree.
* **Import video** — :class:`VideoImportForm`. Single video or
  directory of videos into ``sources/videos/`` (v1) or
  ``videos/`` (legacy). Patch 122o: replaces the legacy Tk
  ``ImportVideosFrame`` popup.
* **Video parameters & calibration** — :class:`VideoInfoForm`.
  Sets the per-video FPS, resolution, and pixels/mm calibration
  used by all distance-based feature kernels. Without it,
  distance features come out in pixels rather than millimeters.
* **Batch pre-process videos** — :class:`BatchPreProcessLauncher`.
  Multi-step video pre-processing wizard (crop → downsample →
  greyscale → flip/rotate → clip). Moved here from the
  Projects page (122i) — it's part of preparing input media,
  not configuring the project.

Section ordering follows the natural workflow: bring data in
(pose + videos), then characterise it (calibration), then
optionally pre-process it (batch).

The cross-format converter (DLC→YOLO, Labelme→DataFrame, etc.)
used to live here but was moved to the Tools page: conversion
doesn't need a project, is used infrequently, and clutters the
page a user visits every session. See
:mod:`mufasa.ui_qt.pages.tools_page`.

Patch 122o: page label capitalised from ``Data import`` to
``Data Import`` for consistency with the title-cased sidebar
entries on other pages ("Preprocessing", "Projects", "Tools",
"Visualizations").

Patch 122w: section title shortened from
``Import pose-estimation data`` to ``Import Pose Data`` to
match the shorter user-facing labels on the other Data Import
sections; the underlying :class:`PoseImportForm` title and
description updated to mention both v1 and legacy destination
directories explicitly.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.pose_import import PoseImportForm
from mufasa.ui_qt.forms.project_setup import BatchPreProcessLauncher
from mufasa.ui_qt.forms.video_import import VideoImportForm
from mufasa.ui_qt.forms.video_info import VideoInfoForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_data_import_page(workbench,
                           config_path: Optional[str] = None
                           ) -> WorkflowPage:
    page = workbench.add_page("Data Import", icon_name="pose")
    page.add_section("Import Pose Data",
                     [(PoseImportForm, {})])
    page.add_section("Import video",
                     [(VideoImportForm, {})])
    page.add_section("Video parameters & calibration",
                     [(VideoInfoForm, {})])
    page.add_section("Batch pre-process videos",
                     [(BatchPreProcessLauncher, {})])
    return page


__all__ = ["build_data_import_page"]
