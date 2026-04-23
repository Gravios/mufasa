"""
mufasa.ui_qt.pages.annotation_page
==================================

Annotation workbench page. Four sections covering:

* Frame labelling (standard + pseudo) — launcher
* Targeted annotation clips — launcher
* Third-party annotation import — inline form
* Reports (extract frames + counts) — inline form
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.annotation import (AnnotationReportsForm,
                                           ClipReviewLauncher,
                                           FrameLabellingLauncher,
                                           TargetedAnnotationClipsLauncher,
                                           ThirdPartyAppenderForm)
from mufasa.ui_qt.workbench import WorkflowPage


def build_annotation_page(workbench,
                          config_path: Optional[str] = None) -> WorkflowPage:
    page = workbench.add_page("Annotation", icon_name="label")
    page.add_section("Frame labelling",          [(FrameLabellingLauncher, {})])
    page.add_section("Targeted annotation clips",
                     [(TargetedAnnotationClipsLauncher, {})])
    page.add_section("Third-party annotation import",
                     [(ThirdPartyAppenderForm, {})])
    page.add_section("Review classifier predictions",
                     [(ClipReviewLauncher, {})])
    page.add_section("Reports",                  [(AnnotationReportsForm, {})])
    return page


__all__ = ["build_annotation_page"]
