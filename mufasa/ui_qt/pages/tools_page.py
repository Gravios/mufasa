"""
mufasa.ui_qt.pages.tools_page
=============================

Tools workbench page — project-independent utilities.

Houses surfaces that don't need an open Mufasa project and would
otherwise clutter project-scoped pages. Currently:

* **Convert pose / annotation data** — cross-format bridge (DLC→YOLO,
  Labelme→DataFrame, SLEAP↔DLC, etc.). Moved here from the Data
  Import page so that page can be single-purpose (load pose data
  into the current project). Conversion doesn't need a project and
  is a utility you reach for occasionally, not every session.

Add other project-less utilities here as they show up.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.data_import import ConverterForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_tools_page(workbench,
                     config_path: Optional[str] = None
                     ) -> WorkflowPage:
    page = workbench.add_page("Tools", icon_name="convert")
    page.add_section("Convert pose / annotation data",
                     [(ConverterForm, {})])
    return page


__all__ = ["build_tools_page"]
