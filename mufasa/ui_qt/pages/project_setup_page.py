"""
mufasa.ui_qt.pages.project_setup_page
=====================================

Project setup workbench page. Hosts archive + batch-preprocess
surfaces; registers the About-Mufasa Help menu action.

When the workbench is launched without a ``config_path``, we surface
a top banner pointing the user at File → New project / File → Open
project so it's obvious how to proceed from a cold start.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from mufasa.ui_qt.forms.project_setup import (ArchiveFilesForm,
                                              BatchPreProcessLauncher,
                                              register_project_menu_actions)
from mufasa.ui_qt.workbench import WorkflowPage


def _build_no_project_banner(workbench) -> QFrame:
    """Prominent banner shown on the Project Setup page when no project
    is loaded. Provides two inline buttons — New / Open — that dispatch
    to the same slots the File menu does."""
    banner = QFrame()
    banner.setFrameShape(QFrame.StyledPanel)
    banner.setStyleSheet(
        "QFrame { background: palette(alternate-base); "
        "border: 1px solid palette(mid); border-radius: 4px; padding: 12px; }"
    )
    lay = QHBoxLayout(banner)
    msg = QLabel(
        "<b>No project loaded.</b><br>"
        "Create a new project or open an existing "
        "<code>project_config.ini</code> to begin."
    )
    msg.setTextFormat(Qt.RichText)
    msg.setWordWrap(True)
    lay.addWidget(msg, 1)

    new_btn = QPushButton("New project…")
    new_btn.clicked.connect(workbench._on_new_project)
    lay.addWidget(new_btn)

    open_btn = QPushButton("Open project…")
    open_btn.clicked.connect(workbench._on_open_project)
    lay.addWidget(open_btn)
    return banner


def build_project_setup_page(workbench,
                             config_path: Optional[str] = None
                             ) -> WorkflowPage:
    page = workbench.add_page("Project setup", icon_name="settings")
    if not config_path:
        # Use the page's top-level layout to surface the banner above
        # any form sections. WorkflowPage exposes a scrollable body;
        # inserting the banner at the top row keeps it prominent.
        page.add_banner(_build_no_project_banner(workbench))
    page.add_section("Archive processed files", [(ArchiveFilesForm, {})])
    page.add_section("Batch pre-process videos", [(BatchPreProcessLauncher, {})])
    # Register Help → About action (idempotent — safe if called multiple
    # times by re-entering the page builder).
    register_project_menu_actions(workbench)
    return page


__all__ = ["build_project_setup_page"]
