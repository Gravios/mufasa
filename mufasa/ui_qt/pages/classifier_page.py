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
  :class:`RunMachineModelsPopUp`.
* **Train classifier** — :class:`TrainClassifierForm`: hyperparams
  + evaluation toggles + Train button that invokes
  :class:`TrainRandomForestClassifier`. Patch 122aq port of
  :class:`MachineModelSettingsPopUp` plus a Train invocation
  the Tk popup didn't have.
* **Validate classifier** — :class:`ValidateClassifierForm`:
  out-of-sample validation video runner. Picks a model + feature
  file, runs inference and renders an annotated video with pose
  tracks + classifier predictions + optional Gantt overlay.
  Patch 122ar port of :class:`ValidationVideoPopUp`.

All sections follow the 122aj+ in-frame + pop-out-to-dock pattern.
:class:`ClassifierValidationPopUp` (per-bout clip generator) is
post-inference visualization rather than validation per se — it
belongs on the Visualizations page when that gets its own ports,
not the Classifier page.
"""
from __future__ import annotations

from mufasa.ui_qt.forms.classifier import ClassifierManageForm
from mufasa.ui_qt.forms.run_inference import RunInferenceForm
from mufasa.ui_qt.forms.train_classifier import TrainClassifierForm
from mufasa.ui_qt.forms.validate_classifier import ValidateClassifierForm
from mufasa.ui_qt.forms.yolo_inference import YOLOPoseInferenceForm
from mufasa.ui_qt.forms.yolo_train import YOLOPoseTrainForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_classifier_page(workbench,
                          config_path: str | None = None
                          ) -> WorkflowPage:
    page = workbench.add_page("Classifier", icon_name="clf")
    page.add_section("Manage classifiers", [(ClassifierManageForm, {})])
    # Patch 122ap: Run inference is now a proper inline section
    # (was a Tk popup launched from SimBA.py before).
    page.add_section("Run inference", [(RunInferenceForm, {})])
    # Patch 122aq: Train classifier — port of
    # MachineModelSettingsPopUp plus a Train invocation.
    page.add_section("Train classifier", [(TrainClassifierForm, {})])
    # Patch 122ar: Validate classifier — port of
    # ValidationVideoPopUp.
    page.add_section("Validate classifier",
                     [(ValidateClassifierForm, {})])
    # Patch 122d2: YOLO pose inference — runs a trained YOLO pose
    # model on a video or directory. Sibling to RunInferenceForm
    # (which handles SimBA classifier inference) but for the YOLO
    # pose-estimation lifecycle. Requires CUDA + ultralytics; form
    # is inert without them.
    page.add_section("YOLO pose — inference",
                     [(YOLOPoseInferenceForm, {})])
    # Patch 122d3: YOLO pose training — fires off a detached
    # subprocess (`python -m mufasa.model.yolo_fit`); workbench
    # doesn't wait for completion. Same hardware/package
    # requirements as inference.
    page.add_section("YOLO pose — train",
                     [(YOLOPoseTrainForm, {})])
    return page


__all__ = ["build_classifier_page"]
