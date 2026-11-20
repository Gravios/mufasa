"""
mufasa.ui_qt.forms.skeleton_reimport
====================================

Patch 122km: "Re-import skeleton model" on the Data Import page — take a
project's markers and skeleton from a pose file again, e.g. after
retraining the pose model with new markers (eyes, shoulders). Pick a pose
file, read what would change, press *Apply*. See
:mod:`mufasa.model.skeleton_reimport`.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SkeletonReimportForm(QWidget):
    """Markers and skeleton from a pose file → the open project."""

    title = "Re-import skeleton model"

    def __init__(self, config_path: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.plan = None
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(
            "Take the project's markers and skeleton from a pose file again — e.g. the output of a "
            "retrained model with new markers. New markers join the Kalman segment of their nearest "
            "marker on the new skeleton. project.toml is backed up in logs/ first; pose files already "
            "imported are not changed (import the new model's output with <b>Import pose data</b>).",
            self))
        outer.itemAt(0).widget().setWordWrap(True)
        row = QHBoxLayout()
        self.file = QLineEdit(self)
        self.file.setPlaceholderText("pose file of the new model (FreeDLC / DeepLabCut parquet, .h5, .csv)")
        self.file.editingFinished.connect(self.preview)
        row.addWidget(self.file, 1)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        outer.addLayout(row)
        self.extend = QCheckBox("Add new markers to the Kalman v2 segments", self)
        self.extend.setChecked(True)
        outer.addWidget(self.extend)
        self.report = QPlainTextEdit(self)
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(170)
        self.report.setPlaceholderText("what would change appears here")
        outer.addWidget(self.report)
        brow = QHBoxLayout()
        brow.addStretch()
        self.apply_btn = QPushButton("Apply", self)
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(lambda: self.apply())   # 122la: clicked(False) skipped the confirmation
        brow.addWidget(self.apply_btn)
        outer.addLayout(brow)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pose file of the new model", "",
            "Pose files (*.parquet *.h5 *.csv);;All files (*)")
        if path:
            self.file.setText(path)
            self.preview()

    def preview(self) -> None:
        from mufasa.model.skeleton_reimport import plan_reimport
        self.plan = None
        self.apply_btn.setEnabled(False)
        path = self.file.text().strip()
        if not path:
            return
        if not self.config_path:
            self.report.setPlainText("open a project first")
            return
        if not Path(path).is_file():
            self.report.setPlainText(f"not found: {path}")
            return
        try:
            self.plan = plan_reimport(self.config_path, path)
        except Exception as e:
            self.report.setPlainText(f"could not read the skeleton model: {type(e).__name__}: {e}")
            return
        self.report.setPlainText(self.plan.text())
        self.apply_btn.setEnabled(self.plan.changed)

    def apply(self, confirm: bool = True):
        from mufasa.model.skeleton_reimport import apply_reimport
        if self.plan is None:
            return None
        if confirm and QMessageBox.question(
                self, "Re-import skeleton model",
                self.plan.text() + "\n\nWrite this to project.toml?") != QMessageBox.Yes:
            return None
        backup = apply_reimport(self.config_path, self.plan, extend_segments=self.extend.isChecked())
        msg = (f"Skeleton model updated ({len(self.plan.new_markers)} markers). Previous project.toml: "
               f"logs/{backup.name}. Reopen the project so every page sees the new markers; import the "
               "new model's pose files with Import pose data.")
        print(msg)
        self.report.setPlainText(self.plan.text() + "\n\n" + msg)
        self.apply_btn.setEnabled(False)
        if confirm:
            win = self.window()
            if hasattr(win, "_switch_to_project") and QMessageBox.question(
                    self, "Reopen project", "Reopen the project now?") == QMessageBox.Yes:
                win._switch_to_project(self.config_path)
        return backup


__all__ = ["SkeletonReimportForm"]
