"""
mufasa.ui_qt.pages.project_setup_page
=====================================

Projects workbench page (file kept named ``project_setup_page``
for module-import compatibility; the user-facing label became
"Projects" in patch 122i).

Layout:

* When a project is loaded:

  - **Project information** — read-only summary
    (:class:`ProjectInfoForm`) showing layout, name, root,
    animals, body parts, classifiers, plus quick run-counts
    under ``derived/``.
  - **Archive processed files** — :class:`ArchiveFilesForm`.

* When no project is loaded:

  - **Create or open project** — empty-state surface
    (:class:`NewProjectForm`) with inline New / Open buttons
    and a one-click "Open most recent" shortcut when a recent
    project is on disk.

Sections moved away from this page in patch 122i:

* **Batch pre-process videos** moved to the Data Import page
  (:mod:`mufasa.ui_qt.pages.data_import_page`). Batch
  pre-processing is part of preparing input media, not
  configuring the project, and grouping it with Import pose
  data + Video parameters & calibration makes the user's
  pre-pipeline checklist live on one page.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.project_info import (NewProjectForm,
                                             ProjectInfoForm)
from mufasa.ui_qt.forms.project_setup import (ArchiveFilesForm,
                                              register_project_menu_actions)
from mufasa.ui_qt.workbench import WorkflowPage


def build_project_setup_page(workbench,
                             config_path: Optional[str] = None
                             ) -> WorkflowPage:
    """Build and return the Projects page."""
    page = workbench.add_page("Projects", icon_name="settings")
    if config_path:
        page.add_section("Project information",
                         [(ProjectInfoForm, {})])
    else:
        # NewProjectForm needs a workbench reference to wire its
        # New / Open buttons to the same slots the File menu uses.
        page.add_section("Create or open project",
                         [(NewProjectForm, {"workbench": workbench})])
    page.add_section("Archive processed files",
                     [(ArchiveFilesForm, {})])
    # Register Help → About action (idempotent — safe if called multiple
    # times by re-entering the page builder).
    register_project_menu_actions(workbench)
    return page


__all__ = ["build_project_setup_page"]
