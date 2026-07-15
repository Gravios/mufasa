"""
mufasa.ui_qt.pages.data_inspector_page
======================================

Sidebar page for previewing the project's data files. Placed straight after
Data Import — checking that imported data looks right is the next thing you
do after importing it.
"""
from __future__ import annotations

from mufasa.ui_qt.forms.data_inspector import DataInspectorForm


def build_data_inspector_page(workbench, config_path=None):
    """Add the 'Data inspector' page."""
    page = workbench.add_page("Data inspector", icon_name="pose")
    page.add_section("Data inspector", [(DataInspectorForm, {})])
    return page
