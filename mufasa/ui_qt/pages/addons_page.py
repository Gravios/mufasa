"""
mufasa.ui_qt.pages.addons_page
==============================

Add-ons workbench page. Hosts specialty workflows that don't fit in
the main pipeline.

Sections
--------

* **Cue-light** accordion (nested) with 4 sub-forms:
  data analysis, classifier statistics, movement statistics,
  visualizer.
* **Kleinberg burst smoothing**
* **Mutual exclusivity corrector**
* **Pup retrieval**
* **Spontaneous alternation**
* **Blob tracker — initialise** (launcher)

Also registers a Tools-menu "Change video playback speed…" action
alongside the existing video-processing actions.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (QFileDialog, QMessageBox)

from mufasa.ui_qt.forms.addons import (BlobTrackerInitLauncher,
                                       CueLightClfForm, CueLightDataForm,
                                       CueLightMovementForm,
                                       CueLightVisualizerForm,
                                       DirectingBodyPartSettingsForm,
                                       KleinbergForm,
                                       MutualExclusivityForm, PupRetrievalForm,
                                       SpontaneousAlternationForm)
from mufasa.ui_qt.forms.blob_quick_check import BlobQuickCheckForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_addons_page(workbench,
                      config_path: Optional[str] = None) -> WorkflowPage:
    page = workbench.add_page("Add-ons", icon_name="add_on")

    # Cue-light family: four separate QToolBox entries so each form
    # collapses independently. Consider a nested accordion if this
    # grows further.
    page.add_section("Cue-light — data analysis",        [(CueLightDataForm, {})])
    page.add_section("Cue-light — classifier statistics",[(CueLightClfForm, {})])
    page.add_section("Cue-light — movement statistics",  [(CueLightMovementForm, {})])
    page.add_section("Cue-light — visualizer",           [(CueLightVisualizerForm, {})])

    page.add_section("Kleinberg burst smoothing",        [(KleinbergForm, {})])
    page.add_section("Mutual exclusivity corrector",     [(MutualExclusivityForm, {})])
    page.add_section("Pup retrieval",                    [(PupRetrievalForm, {})])
    page.add_section("Spontaneous alternation",          [(SpontaneousAlternationForm, {})])
    # Patch 122cz: Qt port of the directing-bodypart settings popup
    # (resolves the blocking gap identified in 122cy). Writes the
    # single `bodypart_direction` key the AnalysisForm's
    # "Directing toward body-part — statistics" route depends on.
    page.add_section("Directing — body-part settings",   [(DirectingBodyPartSettingsForm, {})])
    page.add_section("Blob tracker — initialise",        [(BlobTrackerInitLauncher, {})])
    # Patch 122bw (Tier 3a): Qt port of the legacy Tk
    # BlobQuickChecker. Lets users preview the threshold-difference
    # image before committing to a full blob-tracking run.
    page.add_section("Blob quick-check",                  [(BlobQuickCheckForm, {})])

    # Tools-menu action: change playback speed — one-shot, file-picker-only
    _register_change_speed_action(workbench)

    return page


def _register_change_speed_action(workbench) -> None:
    """Register 'Change video playback speed…' as a Tools-menu action."""
    from PySide6.QtWidgets import QInputDialog

    def _change_speed() -> None:
        path, _ = QFileDialog.getOpenFileName(
            workbench, "Change playback speed — select a video",
            "", "Videos (*.mp4 *.avi *.mov *.mkv *.webm)")
        if not path:
            return
        factor, ok = QInputDialog.getDouble(
            workbench, "Change playback speed",
            "Speed factor (0.1 = 10% speed, 2.0 = 2× speed):",
            1.0, 0.1, 20.0, 2,
        )
        if not ok:
            return
        try:
            from mufasa.video_processors.video_processing import change_playback_speed
            change_playback_speed(video_path=path, speed=factor)
            QMessageBox.information(
                workbench, "Change speed",
                f"Saved re-timed video next to:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(
                workbench, "Change speed failed", str(exc)
            )

    workbench.add_tools_action("Change video playback speed…", _change_speed)


__all__ = ["build_addons_page"]
