"""
mufasa.ui_qt.pages.analysis_page
================================

Analysis workbench page. One consolidated form covers the statistics
pipeline via a route dropdown.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.analysis import AnalysisForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_analysis_page(workbench,
                        config_path: Optional[str] = None) -> WorkflowPage:
    page = workbench.add_page("Analysis", icon_name="stats")
    page.add_section("Run analysis", [(AnalysisForm, {})])
    return page


__all__ = ["build_analysis_page"]
