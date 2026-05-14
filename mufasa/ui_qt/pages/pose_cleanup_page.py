"""
mufasa.ui_qt.pages.pose_cleanup_page
====================================

Preprocessing workbench page (file name kept as
``pose_cleanup_page`` for historical compatibility; the
user-facing label is now "Preprocessing"). Runs early in the
pipeline:

    raw pose → [ interpolate → smooth → outlier-correct → align ]
             → feature extraction → classifier

Section ordering (patches 122c + 122x + 122al)
----------------------------------------------

The page reflects the actual conceptual order of operations,
with media-side prep at the top, pose-cleanup in the middle,
and legacy SimBA forms folded into an "Advanced / legacy"
section at the bottom.

Sections:

1. **Preprocess Videos** (patch 122al) — multi-step video
   pre-processing wizard (crop → downsample → greyscale →
   flip/rotate → clip → FPS → CLAHE). Ported from Tk to a
   Qt form that's both inline (renders as a normal page
   section) AND pop-out-dockable into a floating window.
   Moved to the FIRST position because raw video
   pre-processing happens before anything else in the
   pipeline — including pixel/mm calibration, since
   downsampling changes the pixel scale.
2. **Video Calibration** (patch 122x) — per-video FPS,
   resolution, and pixels/mm calibration used by all
   distance-based feature kernels. Without it, distance
   features come out in pixels rather than millimeters.
3. **Interpolate missing frames** — fill tracker dropouts before
   anything else looks at the data.
4. **Kalman v2 smoothing** — the recommended smoother. Handles
   missing frames, per-marker bias, and (with const-accel
   segments) curvature in motion. Patch 121a–e.
5. **Run outlier correction** — chains movement + location
   correction on the chosen input source. Patch 122c.
6. **Skip outlier correction** — bypass stage entirely while
   still populating the outlier-corrected output dir so
   downstream stages find data. Patch 122c.
7. **Egocentric alignment** — rotate pose (and optionally
   video) to a chosen frame of reference. Now consumes any
   prior stage's output via the picker. Patch 122b.
8. **Advanced / legacy** — three stacked legacy forms:
     - Savitzky-Golay smoother (use Kalman v2 above instead)
     - Outlier correction settings (writes thresholds /
       reference body-parts to project_config.ini; consumed
       by the "Run outlier correction" form)
     - Drop body-parts (project-setup decision; will move to
       project setup once that page exists)

Sections 5 + 6 are mutually exclusive in practice — pick one or
the other for a given run. The picker default in section 5 is
RAW input (since outlier correction usually runs before
smoothing); section 7's picker defaults to the most-processed
available output.

The legacy "Outlier correction settings" form remains in the
Advanced section because the Run/Skip forms above read the
thresholds it writes to ``project_config.ini``. Editing
thresholds without exposing the dependency would be confusing.

Page-label history
------------------
Originally labeled "Pose cleanup". Renamed to "Preprocessing"
because the page covers more than cleanup (smoothing,
egocentric alignment, etc. are upstream-of-features but not
strictly "cleanup"), and because pose-cleanup-specific
operations like smoothing were also being duplicated on the
Data import page — making it clearer that this is *the*
preprocessing surface reduces that confusion. Patch 122x
extends the page's scope by absorbing video calibration and
batch video pre-processing from the Data Import page — same
rationale.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.pose_cleanup import (DropBodypartsForm,
                                             EgocentricAlignmentForm,
                                             InterpolateForm,
                                             KalmanV2SmoothingForm,
                                             OutlierSettingsForm,
                                             RunOutlierCorrectionForm,
                                             SkipOutlierCorrectionForm,
                                             SmoothingForm)
from mufasa.ui_qt.forms.batch_pre_process import BatchPreProcessForm
from mufasa.ui_qt.forms.video_info import VideoInfoForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_pose_cleanup_page(workbench,
                            config_path: Optional[str] = None
                            ) -> WorkflowPage:
    """Build and return the Preprocessing page."""
    page = workbench.add_page("Preprocessing", icon_name="outlier")

    # Patch 122al: Preprocess Videos is now the FIRST section —
    # before any pose-side work, before video calibration. This
    # matches the natural workflow: raw videos in → preprocessed
    # videos out → calibration → pose cleanup. The Tk launcher
    # has been replaced with an inline (and pop-out-dockable) Qt
    # form, BatchPreProcessForm.
    page.add_section("Preprocess Videos",
                     [(BatchPreProcessForm, {})])

    # Patch 122x: media-side prep — now position 2.
    page.add_section("Video Calibration",
                     [(VideoInfoForm, {})])

    # Pose-side flow.
    page.add_section("Interpolate missing frames",
                     [(InterpolateForm, {})])
    page.add_section("Kalman v2 smoothing",
                     [(KalmanV2SmoothingForm, {})])
    page.add_section("Run outlier correction",
                     [(RunOutlierCorrectionForm, {})])
    page.add_section("Skip outlier correction",
                     [(SkipOutlierCorrectionForm, {})])
    page.add_section("Egocentric alignment",
                     [(EgocentricAlignmentForm, {})])
    # All three legacy forms share one section. WorkflowPage's
    # _instantiate() stacks them with bold class-title headers,
    # which is enough visual separation without adding nested
    # toolbox machinery.
    page.add_section("Advanced / legacy",
                     [(SmoothingForm, {}),
                      (OutlierSettingsForm, {}),
                      (DropBodypartsForm, {})])

    return page


__all__ = ["build_pose_cleanup_page"]
