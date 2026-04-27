"""
mufasa.ui_qt.pages.pose_cleanup_page
====================================

Pose-data cleanup workbench page. Runs early in the pipeline:

    raw pose → [ smooth → interpolate → outlier correction → drop bps
                 → egocentric alignment ]
             → feature extraction → classifier

Sections
--------

* **Smooth** — :class:`SmoothingForm` (1 legacy popup).
* **Interpolate missing frames** — :class:`InterpolateForm` (1 popup).
* **Outlier correction settings** — :class:`OutlierSettingsForm`
  (1 popup).
* **Drop body-parts** — :class:`DropBodypartsForm` (1 popup).
* **Egocentric alignment** — :class:`EgocentricAlignmentForm`
  (1 popup, formerly a launcher placeholder).

**5 popups → 5 inline forms.**
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.pose_cleanup import (DropBodypartsForm,
                                             EgocentricAlignmentForm,
                                             InterpolateForm,
                                             OutlierSettingsForm,
                                             SmoothingForm)
from mufasa.ui_qt.workbench import WorkflowPage


def build_pose_cleanup_page(workbench,
                            config_path: Optional[str] = None
                            ) -> WorkflowPage:
    """Build and return the Pose Cleanup page."""
    page = workbench.add_page("Pose cleanup", icon_name="outlier")

    page.add_section("Smoothing",                  [(SmoothingForm, {})])
    page.add_section("Interpolate missing frames", [(InterpolateForm, {})])
    page.add_section("Outlier correction settings",[(OutlierSettingsForm, {})])
    page.add_section("Drop body-parts",            [(DropBodypartsForm, {})])
    page.add_section("Egocentric alignment",       [(EgocentricAlignmentForm, {})])

    return page


__all__ = ["build_pose_cleanup_page"]
