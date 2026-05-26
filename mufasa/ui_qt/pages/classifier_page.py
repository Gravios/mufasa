"""
mufasa.ui_qt.pages.classifier_page
==================================

Classifier workbench pages.

Patch 122ey — split the previously-monolithic "Classifier" page into
six separate sidebar pages, one per section. User request (May 25, 2026):

  > the Classifier tab should just have each section split into its
  > own tab.

The split also surfaces the dependency order the workflow actually
follows. ``Manage classifiers`` is a prerequisite for ``Annotation``
(you can't label frames for a classifier that doesn't exist yet); the
training / validation / inference pages are post-Annotation. YOLO pose
pages are an independent workflow (pose-model training/inference
rather than behavior-classification), grouped at the tail.

The new sidebar order (only the classifier-cluster portion shown):

   Features
   Manage classifiers          ← Classifier setup (pre-Annotation)
   Annotation                  ← existing
   Train classifier            ← train from labelled data
   Validate classifier         ← out-of-sample check
   Run inference               ← apply trained model
   YOLO pose — train           ← independent YOLO workflow
   YOLO pose — inference
   Analysis

Each new page exposes its own ``build_*_page`` function. The legacy
``build_classifier_page`` is removed — workbench_app.py was the only
caller and was updated in this same patch to call the six new
functions in workflow order.

Pages
-----
* :func:`build_manage_classifiers_page` — :class:`ClassifierManageForm`:
  add / remove / print classifier names (3 legacy popups folded into
  one form). Comes BEFORE Annotation in sidebar order.
* :func:`build_train_classifier_page` — :class:`TrainClassifierForm`:
  hyperparams + evaluation toggles + Train button that invokes
  :class:`TrainRandomForestClassifier`. Patch 122aq port of
  :class:`MachineModelSettingsPopUp`.
* :func:`build_validate_classifier_page` —
  :class:`ValidateClassifierForm`: out-of-sample validation video
  runner. Patch 122ar port.
* :func:`build_run_inference_page` — :class:`RunInferenceForm`:
  per-classifier inference batch runner. Patch 122ap port.
* :func:`build_yolo_train_page` — :class:`YOLOPoseTrainForm`:
  detached YOLO-pose subprocess trainer.
* :func:`build_yolo_inference_page` — :class:`YOLOPoseInferenceForm`:
  YOLO-pose inference on a video or directory.
"""
from __future__ import annotations

from mufasa.ui_qt.forms.classifier import ClassifierManageForm
from mufasa.ui_qt.forms.run_inference import RunInferenceForm
from mufasa.ui_qt.forms.train_classifier import TrainClassifierForm
from mufasa.ui_qt.forms.validate_classifier import ValidateClassifierForm
from mufasa.ui_qt.forms.yolo_inference import YOLOPoseInferenceForm
from mufasa.ui_qt.forms.yolo_train import YOLOPoseTrainForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_manage_classifiers_page(workbench,
                                  config_path: str | None = None
                                  ) -> WorkflowPage:
    """Standalone page hosting the classifier-setup form.

    Comes BEFORE Annotation in sidebar order — classifier identity
    is a prerequisite for labelling its frames.
    """
    page = workbench.add_page("Manage classifiers", icon_name="clf")
    page.add_section("Manage classifiers", [(ClassifierManageForm, {})])
    return page


def build_train_classifier_page(workbench,
                                config_path: str | None = None
                                ) -> WorkflowPage:
    """Standalone page for training classifiers from labelled data."""
    page = workbench.add_page("Train classifier", icon_name="clf")
    page.add_section("Train classifier", [(TrainClassifierForm, {})])
    return page


def build_validate_classifier_page(workbench,
                                   config_path: str | None = None
                                   ) -> WorkflowPage:
    """Standalone page for out-of-sample validation videos."""
    page = workbench.add_page("Validate classifier", icon_name="clf")
    page.add_section("Validate classifier",
                     [(ValidateClassifierForm, {})])
    return page


def build_run_inference_page(workbench,
                             config_path: str | None = None
                             ) -> WorkflowPage:
    """Standalone page for running trained classifiers on data."""
    page = workbench.add_page("Run inference", icon_name="clf")
    page.add_section("Run inference", [(RunInferenceForm, {})])
    return page


def build_yolo_train_page(workbench,
                          config_path: str | None = None
                          ) -> WorkflowPage:
    """Standalone page for YOLO pose-model training (independent of
    behavior-classifier workflow)."""
    page = workbench.add_page("YOLO pose — train", icon_name="clf")
    page.add_section("YOLO pose — train", [(YOLOPoseTrainForm, {})])
    return page


def build_yolo_inference_page(workbench,
                              config_path: str | None = None
                              ) -> WorkflowPage:
    """Standalone page for YOLO pose-model inference."""
    page = workbench.add_page("YOLO pose — inference", icon_name="clf")
    page.add_section("YOLO pose — inference",
                     [(YOLOPoseInferenceForm, {})])
    return page


__all__ = [
    "build_manage_classifiers_page",
    "build_train_classifier_page",
    "build_validate_classifier_page",
    "build_run_inference_page",
    "build_yolo_train_page",
    "build_yolo_inference_page",
]
