"""
mufasa.ui_qt.pages.project_setup_page
=====================================

Project setup workbench page. Hosts archive + batch-preprocess
surfaces; registers the About-Mufasa Help menu action.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.project_setup import (ArchiveFilesForm,
                                              BatchPreProcessLauncher,
                                              register_project_menu_actions)
from mufasa.ui_qt.workbench import WorkflowPage


def build_project_setup_page(workbench,
                             config_path: Optional[str] = None
                             ) -> WorkflowPage:
    page = workbench.add_page("Project setup", icon_name="settings")
    page.add_section("Archive processed files", [(ArchiveFilesForm, {})])
    page.add_section("Batch pre-process videos", [(BatchPreProcessLauncher, {})])
    # Register Help → About action (idempotent — safe if called multiple
    # times by re-entering the page builder).
    register_project_menu_actions(workbench)
    return page


__all__ = ["build_project_setup_page"]
