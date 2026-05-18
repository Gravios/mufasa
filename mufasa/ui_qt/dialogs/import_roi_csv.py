"""
mufasa.ui_qt.dialogs.import_roi_csv
=====================================

Qt-native dialog for importing SimBA-format ROI definitions
from CSV files into the project's ROI store.

Replaces the legacy Tk popup
``mufasa.ui.pop_ups.import_roi_csv_popup.ROIDefinitionsCSVImporterPopUp``
which was previously bridged into the Qt workbench via the
subprocess-launch pattern in
``roi_video_table.py:_action_import_csv``
(see ``docs/tk_surface_audit.md`` §2g for the broader context).

Replacement landed in patch 122cu. After that patch:
* The Tk popup file is deleted.
* ``roi_video_table.py:_action_import_csv`` calls this dialog
  directly.

Functional differences from the Tk original
-------------------------------------------
* **QFileDialog.getOpenFileName** browsing instead of the Tk
  ``FileSelect`` widget. Filter set to ``*.csv``; cancel-from-
  picker leaves the existing path intact.
* **Three QLineEdit + Browse rows** (rectangles, circles,
  polygons) — each shape's CSV is independent and any subset
  may be supplied. At least one path must be valid before
  the Run button enables.
* **Append toggle** as a ``QCheckBox`` (was a TRUE/FALSE
  dropdown). The checkbox is disabled if no existing
  ROI_definitions.h5 exists (nothing to append to).
* **Run button** lives inside ``QDialogButtonBox`` alongside
  Cancel, and starts disabled — gets enabled once at least
  one CSV path resolves to an existing file.
* **Result feedback** via ``QMessageBox.information`` on
  success; the Tk popup just relied on the importer's stdout.
"""
from __future__ import annotations

import os
from typing import Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QVBoxLayout)

from mufasa.mixins.config_reader import ConfigReader
from mufasa.roi_tools.import_roi_csvs import ROIDefinitionsCSVImporter
from mufasa.utils.errors import InvalidInputError


_CSV_FILTER = "CSV files (*.csv *.CSV);;All files (*)"


def _make_picker_row(grid: QGridLayout,
                     row: int,
                     label: str,
                     on_pick) -> QLineEdit:
    """Add a [label] [path display] [Browse…] row to the grid.

    Returns the QLineEdit so the caller can read the chosen path
    via ``.text()``. The path field is read-only; selection happens
    through QFileDialog only.

    ``on_pick`` is invoked after a successful pick so the parent
    dialog can refresh button-enable state.
    """
    grid.addWidget(QLabel(label), row, 0, Qt.AlignRight)
    edit = QLineEdit()
    edit.setReadOnly(True)
    edit.setPlaceholderText("(none)")
    grid.addWidget(edit, row, 1)
    btn = QPushButton("Browse…")

    def _browse() -> None:
        path, _ = QFileDialog.getOpenFileName(
            grid.parent(), f"Select {label.rstrip(':')}",
            "", _CSV_FILTER)
        if path:
            edit.setText(path)
            on_pick()

    btn.clicked.connect(_browse)
    grid.addWidget(btn, row, 2)
    return edit


class ImportRoiCsvDialog(QDialog):
    """Qt port of `ROIDefinitionsCSVImporterPopUp` (122cu)."""

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 parent: Optional[QDialog] = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.setWindowTitle("Import ROI definitions from CSV")
        self.setModal(True)

        # Load config to find the existing roi_coordinates_path
        # so we can decide whether the "append" checkbox should
        # be enabled. The whole append-vs-overwrite distinction
        # only matters when there's an existing definitions file.
        try:
            reader = ConfigReader(
                config_path=config_path,
                read_video_info=False,
                create_logger=False)
            self._has_existing = os.path.isfile(
                reader.roi_coordinates_path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Cannot open importer",
                f"Could not read project config: "
                f"{type(exc).__name__}: {exc}")
            self._init_failed = True
            return

        self._init_failed = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Import previously-exported SimBA ROI definitions in CSV\n"
            "format. At least one path must be supplied — any subset of\n"
            "rectangles, circles, and polygons is fine."))

        # File-paths group
        paths_group = QGroupBox("File paths")
        paths_grid = QGridLayout(paths_group)
        paths_grid.setColumnStretch(1, 1)
        # Wire each picker to refresh Run-button enable state.
        self._rect_edit = _make_picker_row(
            paths_grid, 0, "Rectangle CSV:",
            on_pick=lambda: self._refresh_run_enabled())
        self._circle_edit = _make_picker_row(
            paths_grid, 1, "Circle CSV:",
            on_pick=lambda: self._refresh_run_enabled())
        self._poly_edit = _make_picker_row(
            paths_grid, 2, "Polygon CSV:",
            on_pick=lambda: self._refresh_run_enabled())
        layout.addWidget(paths_group)

        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_group)
        self._append_cb = QCheckBox(
            "Append to existing ROI definitions")
        self._append_cb.setEnabled(self._has_existing)
        if not self._has_existing:
            self._append_cb.setToolTip(
                "No existing ROI_definitions.h5 in the project — "
                "import always writes fresh.")
        settings_layout.addWidget(self._append_cb)
        settings_layout.addStretch(1)
        layout.addWidget(settings_group)

        # Button box — Run starts disabled until ≥ 1 valid path.
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Cancel)
        self._run_btn = self._buttons.addButton(
            "Run", QDialogButtonBox.AcceptRole)
        self._run_btn.setEnabled(False)
        self._buttons.accepted.connect(self._run)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def init_failed(self) -> bool:
        return getattr(self, "_init_failed", False)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolved_path(self, edit: QLineEdit) -> Optional[str]:
        """Return an existing file-path from the edit, or None."""
        text = edit.text().strip()
        if text and os.path.isfile(text):
            return text
        return None

    def _refresh_run_enabled(self) -> None:
        any_valid = (
            self._resolved_path(self._rect_edit) is not None
            or self._resolved_path(self._circle_edit) is not None
            or self._resolved_path(self._poly_edit) is not None
        )
        self._run_btn.setEnabled(any_valid)

    def _run(self) -> None:
        rect = self._resolved_path(self._rect_edit)
        circle = self._resolved_path(self._circle_edit)
        poly = self._resolved_path(self._poly_edit)
        append = self._append_cb.isChecked() and self._has_existing

        if rect is None and circle is None and poly is None:
            # Belt-and-braces — Run button should be disabled
            # in this state, but guard against tabbing-to-Run.
            QMessageBox.warning(
                self, "No CSV selected",
                "Select at least one CSV file (rectangles, "
                "circles, or polygons) before running.")
            return

        try:
            importer = ROIDefinitionsCSVImporter(
                config_path=self.config_path,
                rectangles_path=rect,
                circles_path=circle,
                polygon_path=poly,
                append=append)
            importer.run()
        except InvalidInputError as exc:
            QMessageBox.critical(
                self, "Import failed", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(
                self, "Import failed",
                f"{type(exc).__name__}: {exc}")
            return

        QMessageBox.information(
            self, "Imported",
            "ROI definitions imported successfully.")
        self.accept()
