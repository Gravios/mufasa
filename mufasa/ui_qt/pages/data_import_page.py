"""
mufasa.ui_qt.pages.data_import_page
===================================

Data Import workbench page. Two sections after patch 122x:

* **Import Pose Data** — :class:`PoseImportForm`.
  Loads pose data into the currently-open project's
  ``sources/pose/`` (v1) or ``csv/input_csv/`` (legacy) tree.
* **Import video** — :class:`VideoImportForm`. Single video or
  directory of videos into ``sources/videos/`` (v1) or
  ``videos/`` (legacy). Patch 122o: replaces the legacy Tk
  ``ImportVideosFrame`` popup. Patch 122v: symlink default,
  duplicate detection, already-imported table.

Section ordering: bring pose data in, then bring video data in.
Calibration and batch pre-processing — both of which act on
already-imported videos — moved to the Preprocessing page in
patch 122x where they sit naturally alongside other
upstream-of-features stages.

The cross-format converter (DLC→YOLO, Labelme→DataFrame, etc.)
used to live here but was moved to the Tools page: conversion
doesn't need a project, is used infrequently, and clutters the
page a user visits every session. See
:mod:`mufasa.ui_qt.pages.tools_page`.

Patch history
-------------
* **122o**: page label capitalised from ``Data import`` to
  ``Data Import``.
* **122w**: section title shortened from
  ``Import pose-estimation data`` to ``Import Pose Data``.
* **122x**: ``Video parameters & calibration`` (renamed to
  ``Video Calibration``) and ``Batch pre-process videos``
  (renamed to ``Preprocess Videos``) moved to the
  Preprocessing page. Both act on imported videos as upstream
  of feature extraction — they belong on the page that already
  houses Interpolate / Kalman / Outlier correction / Egocentric
  alignment.
"""
from __future__ import annotations

from mufasa.ui_qt.forms.pose_import import PoseImportForm
from mufasa.ui_qt.forms.video_import import VideoImportForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_data_import_page(workbench,
                           config_path: str | None = None
                           ) -> WorkflowPage:
    page = workbench.add_page("Data Import", icon_name="pose")
    page.add_section("Import Pose Data",
                     [(PoseImportForm, {})])
    page.add_section("Import video",
                     [(VideoImportForm, {})])
    return page


__all__ = ["build_data_import_page"]
