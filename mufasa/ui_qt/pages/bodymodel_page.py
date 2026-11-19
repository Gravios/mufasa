"""
mufasa.ui_qt.pages.bodymodel_page
=================================

Patch 122kq: the "Body model" tab — import a rigged model from Blender and
fit it to video (silhouette + 2-D labels, joint limits, forwards and
backwards over time), with an optional overlay video.
"""
from __future__ import annotations

from mufasa.ui_qt.forms.bodymodel import BodyFitForm, RigImportForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_bodymodel_page(workbench, config_path: str | None = None) -> WorkflowPage:
    page = workbench.add_page("Body model", icon_name="pose")
    page.add_section("Import rig (.blend)", [(RigImportForm, {})])
    page.add_section("Fit body model", [(BodyFitForm, {})])
    return page
