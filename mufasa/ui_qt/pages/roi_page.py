"""
mufasa.ui_qt.pages.roi_page
===========================

The ROI workbench page. Accordion sections:

* **Analyze**          — :class:`ROIAnalysisForm` (3 popups).
* **Features**         — :class:`ROIFeaturesForm` (3 popups).
* **Definitions**      — :class:`ROIManageForm` (2 popups).
* **Visualize**        — :class:`ROIVisualizeForm` (2 popups).
* **Draw ROIs**        — placeholder; the ROI-drawing canvas opens an
  OpenCV window and is handled by a Tools-menu action.

Draw-ROIs menu action uses the existing
:class:`mufasa.roi_tools.roi_selector.ROISelector` which already
renders via OpenCV — Qt is the wrong layer to reimplement it.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.roi import (ROIAnalysisForm, ROIFeaturesForm,
                                    ROIManageForm, ROIVisualizeForm)
from mufasa.ui_qt.workbench import WorkflowPage


def build_roi_page(workbench, config_path: Optional[str] = None
                   ) -> WorkflowPage:
    page = workbench.add_page("ROI", icon_name="roi")
    page.add_section("Analyze",        [(ROIAnalysisForm, {})])
    page.add_section("Features",       [(ROIFeaturesForm, {})])
    page.add_section("Definitions",    [(ROIManageForm, {})])
    page.add_section("Visualize",      [(ROIVisualizeForm, {})])
    return page


__all__ = ["build_roi_page"]
