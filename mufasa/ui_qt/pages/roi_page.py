"""
mufasa.ui_qt.pages.roi_page
===========================

The ROI workbench page. Sections ordered by workflow precedence:

* **Definitions**      — :class:`ROIManageForm` (2 popups). Comes
  first because nothing else on this page works without ROIs to
  reference.
* **Analyze**          — :class:`ROIAnalysisForm` (3 popups).
  Aggregates / time-bins / etc. — what the typical user does next.
* **Visualize**        — :class:`ROIVisualizeForm` (2 popups).
  Inspect the analysis output by overlaying tracking / features
  on the source video.
* **Features**         — :class:`ROIFeaturesForm` (3 popups).
  Append per-frame ROI features to an already-extracted feature CSV.
  Last because it requires both ROIs and an existing feature file.

The order matches SimBA's left-to-right ROI tab layout (Definitions
→ Analyze → Visualize → extras), with the Mufasa-specific 'Features'
section sitting in the extras position.

Draw-ROIs canvas opens an OpenCV window via
:class:`mufasa.roi_tools.roi_selector.ROISelector` — Qt is the wrong
layer to reimplement that.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.roi import (ROIAnalysisForm, ROIFeaturesForm,
                                    ROIManageForm, ROIVisualizeForm)
from mufasa.ui_qt.workbench import WorkflowPage


def build_roi_page(workbench, config_path: Optional[str] = None
                   ) -> WorkflowPage:
    page = workbench.add_page("ROI", icon_name="roi")
    page.add_section("Definitions",    [(ROIManageForm, {})])
    page.add_section("Analyze",        [(ROIAnalysisForm, {})])
    page.add_section("Visualize",      [(ROIVisualizeForm, {})])
    page.add_section("Features",       [(ROIFeaturesForm, {})])
    return page


__all__ = ["build_roi_page"]
