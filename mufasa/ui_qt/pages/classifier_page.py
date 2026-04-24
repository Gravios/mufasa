"""
mufasa.ui_qt.pages.classifier_page
==================================

Classifier workbench page.

Sections
--------
* **Manage** — :class:`ClassifierManageForm`: add / remove / print
  classifier names (3 legacy popups folded into one form).

Previous iterations carried empty "Train", "Run inference", and
"Validate" placeholder sections pending inline-form ports. Empty
sections in the UI look like broken features rather than unimplemented
ones, so they were removed. When the train/infer/validate backends
are wired inline they can be appended here.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.classifier import ClassifierManageForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_classifier_page(workbench,
                          config_path: Optional[str] = None
                          ) -> WorkflowPage:
    page = workbench.add_page("Classifier", icon_name="clf")
    page.add_section("Manage classifiers", [(ClassifierManageForm, {})])
    return page


__all__ = ["build_classifier_page"]
