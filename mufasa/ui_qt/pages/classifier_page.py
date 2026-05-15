"""
mufasa.ui_qt.pages.classifier_page
==================================

Classifier workbench page.

Sections
--------
* **Manage** — :class:`ClassifierManageForm`: add / remove / print
  classifier names (3 legacy popups folded into one form).
* **Run inference** — :class:`RunInferenceForm`: per-classifier
  model-path / threshold / min-bout configurator that drives
  :class:`InferenceBatch`. Patch 122ap port of
  :class:`RunMachineModelsPopUp`. In-frame with pop-out-to-dock
  support, same pattern as 122aj's frame labeller and 122al's
  batch pre-processor.

Train / Validate sections are still on the Tk side (driven by the
legacy SimBA.py root window's menus). They're more involved than
inference (multiple training algorithms with distinct parameter
shapes; validation expects specific input/output dir conventions)
and earn their own ports separately.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.classifier import ClassifierManageForm
from mufasa.ui_qt.forms.run_inference import RunInferenceForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_classifier_page(workbench,
                          config_path: Optional[str] = None
                          ) -> WorkflowPage:
    page = workbench.add_page("Classifier", icon_name="clf")
    page.add_section("Manage classifiers", [(ClassifierManageForm, {})])
    # Patch 122ap: Run inference is now a proper inline section
    # (was a Tk popup launched from SimBA.py before).
    page.add_section("Run inference", [(RunInferenceForm, {})])
    return page


__all__ = ["build_classifier_page"]
