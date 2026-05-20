"""
mufasa.ui_qt.pages.roi_page
===========================

The ROI workbench page. Sections ordered by workflow precedence:

* **Definitions**      — :class:`ROIDefineWidget` embedded inline
  (patch 122dn). The full ROI editor lives directly in the page —
  no popup. Comes first because nothing else on this page works
  without ROIs to reference.
* **Maintenance**      — :class:`ROIManageForm` (import CSV /
  standardize). Patch 122dn renamed from "Definitions" since the
  Draw action is now handled by the embedded widget above; what's
  left in this form is the bookkeeping operations.
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
section sitting in the extras position. Patch 122dn embeds the
define panel inline (vs the legacy popup); patches 122dl + 122dm
landed subset-apply + drag-to-adjust functionality inside that
panel.
"""
from __future__ import annotations

from mufasa.ui_qt.dialogs.roi_define_panel import ROIDefineWidget
from mufasa.ui_qt.forms.roi import ROIAnalysisForm, ROIFeaturesForm, ROIManageForm, ROIVisualizeForm
from mufasa.ui_qt.workbench import WorkflowPage


def _make_define_widget(config_path: str | None) -> ROIDefineWidget:
    """Factory used by WorkflowPage.add_section_widget. Constructs a
    ROIDefineWidget with embedded-mode visibility (hides the dialog-
    only Close / Save&Close buttons since the page section provides
    its own dismiss affordance via collapse)."""
    w = ROIDefineWidget(config_path=config_path)
    w.set_embedded_mode(True)
    return w


def build_roi_page(workbench, config_path: str | None = None
                   ) -> WorkflowPage:
    page = workbench.add_page("ROI", icon_name="roi")
    # Patch 122dn — Definitions section hosts the ROI define widget
    # inline. Replaces the legacy `(ROIManageForm, {})` form entry
    # which opened the panel as a popup. ROIManageForm moved to a
    # separate "Maintenance" section since its non-Draw actions
    # (Import CSV, Standardize) are still needed.
    page.add_section_widget("Definitions",
                             _make_define_widget)
    page.add_section("Maintenance",    [(ROIManageForm, {})])
    page.add_section("Analyze",        [(ROIAnalysisForm, {})])
    page.add_section("Visualize",      [(ROIVisualizeForm, {})])
    page.add_section("Features",       [(ROIFeaturesForm, {})])
    return page


__all__ = ["build_roi_page"]
