"""
mufasa.ui_qt.pages.project_setup_page
=====================================

Projects workbench page (file kept named ``project_setup_page``
for module-import compatibility; user-facing label became
"Projects" in patch 122i).

Section layout (patch 122m):

1. **Create or open project** — :class:`NewProjectForm`.
   Always at the top. Inline buttons (New, Open, Open most
   recent if the recent path differs from the current). Lets
   users switch projects or start a new one without leaving the
   page. When no project is loaded, this is the page's only
   actionable surface.
2. **Project information** — :class:`ProjectInfoForm`. Only
   added when a project is loaded. When present, the page
   programmatically focuses (expands) this section at build
   time — that's where users continuing work on an existing
   project want to land first. The "Create or open" section
   stays collapsed but reachable.

Removed in patch 122m:

* **Archive processed files** — the legacy "shuffle outputs
  into named subfolders" model assumed the SimBA INI ``csv/``
  tree and would have crashed on v1 projects. The use-case it
  solved (clearing shared stage dirs between experiments) is
  now subsumed by v1's per-run
  ``derived/<stage>/<run_id>/`` provenance.

Removed in patch 122i:

* **Batch pre-process videos** moved to the Data Import page
  (:mod:`mufasa.ui_qt.pages.data_import_page`). Pre-processing
  is part of preparing input media, not configuring the
  project.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.project_info import (NewProjectForm,
                                             ProjectInfoForm)
from mufasa.ui_qt.forms.project_setup import register_project_menu_actions
from mufasa.ui_qt.workbench import WorkflowPage


def build_project_setup_page(workbench,
                             config_path: Optional[str] = None
                             ) -> WorkflowPage:
    """Build and return the Projects page.

    Section 1 (Create or open project) is always added at the top
    so users can switch / create projects from this page
    regardless of state. Section 2 (Project information) is
    conditional on a project being loaded; when present, the
    QToolBox is programmatically advanced to it so a user
    returning to a recent project sees their project info
    immediately on workbench launch.
    """
    page = workbench.add_page("Projects", icon_name="settings")

    # Section 1 — Create / Open / Recent. Always present.
    # NewProjectForm needs the workbench reference to wire its
    # buttons to the same slots the File menu uses. It also
    # adjusts its own messaging based on whether config_path is
    # set (see NewProjectForm.build_shell for the two modes).
    page.add_section("Create or open project",
                     [(NewProjectForm, {"workbench": workbench})])

    if config_path:
        # Section 2 — Project information.
        page.add_section("Project information",
                         [(ProjectInfoForm, {})])
        # Focus this section: a user landing on the Projects
        # page after a recent-project autoload wants to see
        # the project info first; the Create surface stays
        # reachable as the collapsed-but-visible section above.
        # setCurrentIndex emits currentChanged, which triggers
        # WorkflowPage's lazy instantiation hook — section gets
        # built immediately, no stale-form risk.
        page.toolbox.setCurrentIndex(1)

    # Register Help → About action (idempotent — safe if called
    # multiple times by re-entering the page builder).
    register_project_menu_actions(workbench)
    return page


__all__ = ["build_project_setup_page"]
