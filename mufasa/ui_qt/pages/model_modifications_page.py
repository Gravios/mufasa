"""
mufasa.ui_qt.pages.model_modifications_page
===========================================

Sidebar page for post-creation edits to the project's pose model. Hosts the
marker-rename form; more model-modification tools can be added as sections
here later.
"""
from __future__ import annotations

from mufasa.ui_qt.forms.model_modifications import ModelModificationsForm


def build_model_modifications_page(workbench, config_path=None):
    """Add the 'Model modifications' page (rename markers section)."""
    page = workbench.add_page("Model modifications", icon_name="pose")
    page.add_section("Rename markers", [(ModelModificationsForm, {})])
    return page
