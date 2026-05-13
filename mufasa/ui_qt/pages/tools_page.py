"""
mufasa.ui_qt.pages.tools_page
=============================

Tools workbench page — project-independent utilities.

Houses surfaces that don't need an open Mufasa project and would
otherwise clutter project-scoped pages. Current sections:

* **Convert pose / annotation data** — cross-format bridge (DLC→YOLO,
  Labelme→DataFrame, SLEAP↔DLC, etc.). Moved here from the Data
  Import page so that page can be single-purpose (load pose data
  into the current project). Conversion doesn't need a project and
  is a utility you reach for occasionally, not every session.

* **Re-order pose keypoints** (patch 122ac) —
  :class:`PoseReorganizerForm`. Re-arrange the keypoint column
  order in a directory of DLC / maDLC pose files. Replaces
  legacy :class:`PoseReorganizerPopUp`.

* **SLEAP → YOLO conversion** (patch 122ac) —
  :class:`SLEAPToYoloForm`. Convert SLEAP CSV predictions into
  YOLOv8 keypoint annotations. Replaces legacy
  :class:`SLEAPcsvInference2Yolo`.

Add other project-less utilities here as they show up.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.data_import import ConverterForm
from mufasa.ui_qt.forms.pose_tools import (PoseReorganizerForm,
                                            SLEAPToYoloForm)
from mufasa.ui_qt.workbench import WorkflowPage


def build_tools_page(workbench,
                     config_path: Optional[str] = None
                     ) -> WorkflowPage:
    page = workbench.add_page("Tools", icon_name="convert")
    page.add_section("Convert pose / annotation data",
                     [(ConverterForm, {})])
    # Patch 122ac: pose-tools cluster on the Tools page.
    page.add_section("Re-order pose keypoints",
                     [(PoseReorganizerForm, {})])
    page.add_section("SLEAP → YOLO conversion",
                     [(SLEAPToYoloForm, {})])
    return page


__all__ = ["build_tools_page"]
