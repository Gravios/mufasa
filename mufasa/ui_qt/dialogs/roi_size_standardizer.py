"""
mufasa.ui_qt.dialogs.roi_size_standardizer
============================================

Qt-native dialog for normalizing ROI sizes across project videos
to a chosen baseline video's px/mm calibration.

Replaces the legacy Tk popup
``mufasa.ui.pop_ups.roi_size_standardizer_popup.ROISizeStandardizerPopUp``
which was previously bridged into the Qt workbench via the
subprocess-launch pattern in ``roi_video_table.py:_action_standardize``
(see ``docs/tk_surface_audit.md`` §2g for the broader context of
that subprocess pattern and its phase-out).

Replacement landed in patch 122cs. After that patch:
* The Tk popup file is deleted.
* ``roi_video_table.py:_action_standardize`` calls this dialog directly.

Workflow
--------

1. Open the dialog with a SimBA project config path.
2. The dialog reads the project's ROI definitions and gathers
   the list of videos with ROIs.
3. The user picks one as the baseline.
4. **OK** runs ``ROISizeStandardizer.run().save()`` and closes.
   **Cancel** closes without touching the project.

Requires ≥ 2 videos with ROIs. Fewer videos → error dialog +
auto-reject (the legacy Tk popup raised an exception instead;
the Qt port surfaces it as a message).
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QMessageBox

from mufasa.mixins.config_reader import ConfigReader
from mufasa.roi_tools.ROI_size_standardizer import ROISizeStandardizer
from mufasa.utils.checks import check_file_exist_and_readable
from mufasa.utils.errors import NoROIDataError


class ROISizeStandardizerDialog(QDialog):
    """Qt port of `ROISizeStandardizerPopUp` (122cs)."""

    def __init__(self,
                 config_path: str | os.PathLike,
                 parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.setWindowTitle("ROI Size Standardizer")
        self.setModal(True)

        # Load project metadata + list of videos with ROIs.
        # On any failure (missing ROI file, too few videos, IO),
        # surface the error and auto-reject — no half-open dialog.
        try:
            reader = ConfigReader(
                config_path=config_path, read_video_info=False)
            check_file_exist_and_readable(
                file_path=reader.roi_coordinates_path)
            reader.read_roi_data()
        except Exception as exc:
            QMessageBox.critical(
                self, "Cannot open ROI standardizer",
                f"Could not read ROI definitions: "
                f"{type(exc).__name__}: {exc}")
            # Defer the reject() to after the dialog is shown,
            # but we never show it.
            self._init_failed = True
            return

        self._init_failed = False
        video_names = list(reader.video_names_w_rois)
        if len(video_names) <= 1:
            QMessageBox.critical(
                self, "Not enough videos",
                f"Need at least 2 videos with ROIs to standardize "
                f"sizes; found {len(video_names)}.\n\n"
                f"Define ROIs on more videos and try again.")
            self._init_failed = True
            return

        self._video_names = video_names

        layout = QFormLayout(self)
        layout.addRow(QLabel(
            "Pick a baseline video. The pixels-per-millimeter of "
            "that\nvideo is used to scale all other videos' ROI "
            "sizes."))

        self._baseline_combo = QComboBox()
        self._baseline_combo.addItems(self._video_names)
        layout.addRow("Baseline video:", self._baseline_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._run)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def init_failed(self) -> bool:
        """Was the dialog unable to initialize? (Caller should
        skip exec() if True.)"""
        return getattr(self, "_init_failed", False)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        reference_video = self._baseline_combo.currentText()
        try:
            std = ROISizeStandardizer(
                config_path=self.config_path,
                reference_video=reference_video)
            std.run()
            std.save()
        except NoROIDataError as exc:
            QMessageBox.critical(
                self, "Standardize failed",
                f"No ROI data: {exc}")
            return
        except Exception as exc:
            QMessageBox.critical(
                self, "Standardize failed",
                f"{type(exc).__name__}: {exc}")
            return

        QMessageBox.information(
            self, "Done",
            f"ROI sizes standardized against '{reference_video}'.")
        self.accept()
