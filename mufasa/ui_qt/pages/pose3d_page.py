"""
mufasa.ui_qt.pages.pose3d_page
==============================

Patch 122kf: the "3-D" workbench tab — camera calibration, 3-D inference
(single camera with the Rat7M body prior, or two cameras), 3-D Kalman
smoothing, and the 3-D viewer. Sits after Preprocessing, whose Kalman v2
runs fill low-confidence 2-D labels.
"""
from __future__ import annotations

from mufasa.ui_qt.forms.pose3d import (
    CameraCalibrationForm,
    Pose3DInferenceForm,
    Pose3DSmoothingForm,
    Pose3DViewerLauncher,
)
from mufasa.ui_qt.workbench import WorkflowPage


def build_pose3d_page(workbench, config_path: str | None = None) -> WorkflowPage:
    page = workbench.add_page("3-D", icon_name="pose")
    page.add_section("Camera calibration", [(CameraCalibrationForm, {})])
    page.add_section("3-D inference", [(Pose3DInferenceForm, {})])
    page.add_section("3-D Kalman smoothing", [(Pose3DSmoothingForm, {})])
    page.add_section_widget("3-D viewer", Pose3DViewerLauncher)
    return page
