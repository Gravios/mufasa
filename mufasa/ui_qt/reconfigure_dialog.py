"""
mufasa.ui_qt.reconfigure_dialog
===============================

Qt dialog for the "Reconfigure project from DLC file…" menu action.
Pick a DLC output file (.h5 / .csv), preview the detected body parts,
confirm the reconfigure. All on-disk edits are atomic with respect to
the user's choice: nothing is written until the user clicks Apply.

The dialog uses the pure-Python helpers in
:mod:`mufasa.pose_importers.dlc_autodetect` and
:mod:`mufasa.utils.project_reconfigure` — no Qt logic lives in those
modules so they can be scripted or driven by the legacy Tk UI too.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from mufasa.pose_importers.dlc_autodetect import (DLCAutodetectError,
                                                  extract_bodyparts)
from mufasa.utils.project_reconfigure import (ProjectReconfigureError,
                                              reconfigure_project_user_defined)


class ReconfigureProjectDialog(QDialog):
    """Modal dialog for reconfiguring the currently-open project to
    ``user_defined`` with body parts auto-detected from a DLC file.

    Construct with the path to the project's ``project_config.ini``.
    After ``exec()`` returns ``Accepted``, the on-disk files have
    already been updated and the caller should rebuild the workbench
    so pages pick up the new config.
    """

    def __init__(self,
                 config_path: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.setWindowTitle("Reconfigure project from DLC file")
        self.setModal(True)
        self.resize(620, 560)

        self._detected_bps: List[str] = []

        # ------------------ Widgets ---------------------------------- #
        hint = QLabel(
            "Pick a DLC output file (<code>.h5</code> or <code>.csv</code>) "
            "from your project. Body-part names will be read from the "
            "file in DLC's column order, and this Mufasa project will "
            "be switched to <code>user_defined</code> with those body "
            "parts.<br><br>"
            "Both <code>project_config.ini</code> and "
            "<code>project_bp_names.csv</code> will be backed up "
            "before any edits."
        )
        hint.setTextFormat(Qt.RichText)
        hint.setWordWrap(True)

        self._dlc_edit = QLineEdit()
        self._dlc_edit.setReadOnly(True)
        self._dlc_edit.setPlaceholderText(
            "Select a .h5 or .csv file from your DLC output"
        )
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_file)

        file_row = QWidget()
        fr = QHBoxLayout(file_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.addWidget(self._dlc_edit, 1)
        fr.addWidget(browse)

        self._count_label = QLabel(" ")
        self._count_label.setStyleSheet("color: palette(mid);")

        self._bp_list = QListWidget()
        self._bp_list.setAlternatingRowColors(True)

        # Current project info
        cfg_display = QLabel(f"<code>{config_path}</code>")
        cfg_display.setTextFormat(Qt.RichText)
        cfg_display.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Project:", cfg_display)
        form.addRow("DLC file:", file_row)
        form.addRow("", self._count_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Cancel,
            parent=self,
        )
        self._apply_btn = self._buttons.button(QDialogButtonBox.Apply)
        self._apply_btn.setText("Apply")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        self._buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addWidget(hint)
        root.addLayout(form)
        root.addWidget(QLabel("Detected body parts (in DLC column order):"))
        root.addWidget(self._bp_list, 1)
        root.addWidget(self._buttons)

    # ------------------ Slots ---------------------------------------- #
    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a DLC output file", "",
            "DLC output (*.h5 *.csv);;All files (*)",
        )
        if not path:
            return
        self._dlc_edit.setText(path)
        self._bp_list.clear()
        self._count_label.setText("Parsing…")
        try:
            bps = extract_bodyparts(path)
        except DLCAutodetectError as exc:
            self._count_label.setText("")
            self._detected_bps = []
            self._apply_btn.setEnabled(False)
            QMessageBox.warning(
                self, "Could not read DLC file", str(exc),
            )
            return
        except Exception as exc:  # defensive: pd.read_hdf can raise anything
            self._count_label.setText("")
            self._detected_bps = []
            self._apply_btn.setEnabled(False)
            QMessageBox.critical(
                self, "Unexpected error reading DLC file",
                f"{type(exc).__name__}: {exc}",
            )
            return

        self._detected_bps = bps
        self._count_label.setText(
            f"{len(bps)} body-part(s) detected"
        )
        self._bp_list.addItems(bps)
        self._apply_btn.setEnabled(True)

    def _apply(self) -> None:
        if not self._detected_bps:
            return
        # One more sanity prompt before we write to disk.
        resp = QMessageBox.question(
            self, "Apply reconfigure?",
            f"This will:\n\n"
            f"  • back up <code>project_config.ini</code> and "
            f"<code>project_bp_names.csv</code>\n"
            f"  • set <code>pose_estimation_body_parts = user_defined</code>\n"
            f"  • set <code>animal_no = 1</code>\n"
            f"  • overwrite <code>project_bp_names.csv</code> with "
            f"the {len(self._detected_bps)} detected body parts\n\n"
            f"Previously imported pose data (if any) may no longer "
            f"match the new body-part count. Continue?".replace(
                "<code>", "").replace("</code>", ""),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        try:
            result = reconfigure_project_user_defined(
                config_path=self.config_path,
                body_parts=self._detected_bps,
                animal_cnt=1,
            )
        except ProjectReconfigureError as exc:
            QMessageBox.critical(
                self, "Reconfigure failed", str(exc),
            )
            return

        changes_html = (
            "<br>".join(f"• {c}" for c in result.changes)
            if result.changes else "(no changes needed — already configured)"
        )
        QMessageBox.information(
            self, "Reconfigure complete",
            f"<b>Changes:</b><br>{changes_html}<br><br>"
            f"<b>Backups:</b><br>"
            f"• {result.config_backup.name}<br>"
            f"• {result.bp_backup.name}<br><br>"
            f"The workbench will reload to pick up the new "
            f"configuration.",
        )
        self.accept()


__all__ = ["ReconfigureProjectDialog"]
