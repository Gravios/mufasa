"""
mufasa.ui_qt.pages.data_import_page
===================================

Data Import workbench page. Hosts one consolidated converter form
that replaces 11 legacy format-bridge popups.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.data_import import ConverterForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_data_import_page(workbench,
                           config_path: Optional[str] = None
                           ) -> WorkflowPage:
    page = workbench.add_page("Data import", icon_name="pose")
    page.add_section("Convert pose / annotation data",
                     [(ConverterForm, {})])
    # Placeholder sections for specialty importers coming later
    page.add_section("Import pose-estimation video directory", [])
    page.add_section("Create user-defined pose config",        [])
    return page


__all__ = ["build_data_import_page"]
