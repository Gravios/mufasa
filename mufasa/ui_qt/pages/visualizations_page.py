"""
mufasa.ui_qt.pages.visualizations_page
======================================

Visualisations workbench page. One consolidated form covers 12+ legacy
popups via a route dropdown + declarative parameter table.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.visualizations import VisualizationForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_visualizations_page(workbench,
                              config_path: Optional[str] = None
                              ) -> WorkflowPage:
    page = workbench.add_page("Visualizations", icon_name="visualize")
    page.add_section("Create visualisation",
                     [(VisualizationForm, {})])
    return page


__all__ = ["build_visualizations_page"]
