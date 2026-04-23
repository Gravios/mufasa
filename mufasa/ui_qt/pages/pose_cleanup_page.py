"""
mufasa.ui_qt.pages.pose_cleanup_page
====================================

Pose-data cleanup workbench page. Runs early in the pipeline:

    raw pose → [ smooth → interpolate → outlier correction → drop bps ]
             → feature extraction → classifier

Sections
--------

* **Smooth** — :class:`SmoothingForm` (1 legacy popup).
* **Interpolate missing frames** — :class:`InterpolateForm` (1 popup).
* **Outlier correction settings** — :class:`OutlierSettingsForm`
  (1 popup).
* **Drop body-parts** — :class:`DropBodypartsForm` (1 popup).
* **Egocentric alignment** — launches the legacy
  :class:`EgocentricAlignPopUp` as a dialog (interactive click-on-frame;
  hard to host inline without duplicating an OpenCV render surface).

**5 popups → 4 inline forms + 1 dialog-launcher.**
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QMessageBox, QPushButton, QVBoxLayout, QWidget

from mufasa.ui_qt.forms.pose_cleanup import (DropBodypartsForm,
                                             InterpolateForm,
                                             OutlierSettingsForm,
                                             SmoothingForm)
from mufasa.ui_qt.workbench import OperationForm, WorkflowPage


class _EgocentricAlignLauncher(OperationForm):
    """Launcher 'form' that opens the interactive egocentric-alignment
    dialog. Replaces the inline section content with a one-line note +
    a big button. This is the workbench-side pattern for "this
    operation still needs its own window."
    """
    title = "Egocentric alignment"
    description = ("Rotate video + pose so the animal's body-axis aligns "
                   "with a chosen reference direction. Requires an "
                   "interactive click-on-frame step, launched in a dialog.")

    def build(self) -> None:
        launch = QPushButton("  Launch interactive alignment…", self)
        launch.setStyleSheet("padding: 8px 16px; font-size: 11pt;")
        launch.clicked.connect(self._launch)
        self.body_layout.addWidget(launch)
        # Override the base-class Run button — the dialog handles execution
        self.run_btn.setVisible(False)

    def collect_args(self) -> dict:
        # Unused — the Run button is hidden.
        return {}

    def target(self, **kwargs) -> None:  # pragma: no cover
        pass

    def _launch(self) -> None:
        try:
            # EgocentricAlignPopUp is a legacy Tk class; until it's ported
            # to MufasaDialog, we surface a clear message rather than
            # spinning up a Tk root inside a Qt process.
            QMessageBox.information(
                self.window(),
                "Egocentric alignment",
                "Egocentric alignment needs interactive click-on-frame "
                "selection and currently runs via the legacy Tk UI. "
                "Use the <code>mufasa-tk</code> entry point for now; "
                "the Qt port is on the roadmap.",
            )
        except Exception as exc:
            QMessageBox.critical(self.window(), "Launch failed", str(exc))


def build_pose_cleanup_page(workbench,
                            config_path: Optional[str] = None
                            ) -> WorkflowPage:
    """Build and return the Pose Cleanup page."""
    page = workbench.add_page("Pose cleanup", icon_name="outlier")

    page.add_section("Smoothing",                 [(SmoothingForm, {})])
    page.add_section("Interpolate missing frames",[(InterpolateForm, {})])
    page.add_section("Outlier correction settings",[(OutlierSettingsForm, {})])
    page.add_section("Drop body-parts",           [(DropBodypartsForm, {})])
    page.add_section("Egocentric alignment",      [(_EgocentricAlignLauncher, {})])

    return page


__all__ = ["build_pose_cleanup_page"]
