"""
mufasa.ui_qt.pages.classifier_page
==================================

Classifier workbench page — second worked example of the workbench
pattern outside Video Processing.

Sections
--------

* **Manage** — :class:`ClassifierManageForm`: add / remove / print
  (3 legacy popups folded into one form).
* **Train**, **Run**, **Validate** — placeholder sections pending
  inline-form ports of RunMachineModels, Kleinberg, ValidationClips,
  ClassifierValidation.
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
    page.add_section("Train", [])
    page.add_section("Run inference", [])
    page.add_section("Validate", [])
    return page


__all__ = ["build_classifier_page"]
