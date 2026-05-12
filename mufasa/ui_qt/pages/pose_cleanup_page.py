"""
mufasa.ui_qt.pages.pose_cleanup_page
====================================

Preprocessing workbench page (file name kept as
``pose_cleanup_page`` for historical compatibility; the
user-facing label is now "Preprocessing"). Runs early in the
pipeline:

    raw pose → [ interpolate → smooth → outlier-correct → align ]
             → feature extraction → classifier

Section ordering (patch 122c redesign)
--------------------------------------

The page reflects the actual conceptual order of operations,
with the modern (Kalman v2 + InputSourcePicker) forms at the top
and the legacy SimBA forms folded into an "Advanced / legacy"
section at the bottom:

1. **Interpolate missing frames** — fill tracker dropouts before
   anything else looks at the data.
2. **Kalman v2 smoothing** — the recommended smoother. Handles
   missing frames, per-marker bias, and (with const-accel
   segments) curvature in motion. Patch 121a–e.
3. **Run outlier correction** — chains movement + location
   correction on the chosen input source. Patch 122c.
4. **Skip outlier correction** — bypass stage entirely while
   still populating the outlier-corrected output dir so
   downstream stages find data. Patch 122c.
5. **Egocentric alignment** — rotate pose (and optionally
   video) to a chosen frame of reference. Now consumes any
   prior stage's output via the picker. Patch 122b.
6. **Advanced / legacy** — three stacked legacy forms:
     - Savitzky-Golay smoother (use Kalman v2 above instead)
     - Outlier correction settings (writes thresholds / reference
       body-parts to project_config.ini; consumed by the
       "Run outlier correction" form)
     - Drop body-parts (project-setup decision; will move to
       project setup once that page exists)

Sections 3 + 4 are mutually exclusive in practice — pick one or
the other for a given run. The picker default in section 3 is
RAW input (since outlier correction usually runs before
smoothing); section 5's picker defaults to the most-processed
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
preprocessing surface reduces that confusion.
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
from mufasa.ui_qt.workbench import WorkflowPage


def build_pose_cleanup_page(workbench,
                            config_path: Optional[str] = None
                            ) -> WorkflowPage:
    """Build and return the Preprocessing page."""
    page = workbench.add_page("Preprocessing", icon_name="outlier")

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
