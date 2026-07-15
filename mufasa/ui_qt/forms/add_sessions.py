"""
mufasa.ui_qt.forms.add_sessions
===============================

"Add sessions" — bring new recordings into an existing project and update
their derived data, without touching the sessions already there.

Pick a file or a folder, **Check** reports which files match the project's
pose, and **Add sessions** imports the accepted ones and (optionally)
re-smooths just those, reusing the project's latest model rather than
retraining.

The check is not a formality. A file whose markers don't match the project
imports "successfully" and then produces pose columns nothing downstream can
find — which shows up much later as all-NaN arrays rather than an error at
the point of the mistake. So the compatibility report is shown before
anything is written, and mismatched files are named with the exact
difference.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class AddSessionsForm(QWidget):
    """Import new sessions matching the project's pose, then refresh."""

    title = "Add sessions"

    def __init__(self, config_path: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(
            "Add recordings that use the <b>same pose</b> as this project. "
            "Files whose markers differ are listed and skipped.", self,
        ))

        row = QHBoxLayout()
        self.source = QLineEdit(self)
        self.source.setPlaceholderText("Pose file or folder…")
        row.addWidget(self.source, 1)
        b_file = QPushButton("File…", self)
        b_file.clicked.connect(self._pick_file)
        row.addWidget(b_file)
        b_dir = QPushButton("Folder…", self)
        b_dir.clicked.connect(self._pick_dir)
        row.addWidget(b_dir)
        outer.addLayout(row)

        self.smooth = QCheckBox(
            "Smooth new sessions with the project's latest model", self)
        self.smooth.setChecked(True)
        self.smooth.setToolTip(
            "Reuses the most recent saved model — it never retrains. "
            "Skipped with a note if the project has no model yet."
        )
        outer.addWidget(self.smooth)

        self.report = QTextEdit(self)
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Check to preview what would be added.")
        outer.addWidget(self.report, 1)

        btns = QHBoxLayout()
        btns.addStretch()
        self._check_btn = QPushButton("Check", self)
        self._check_btn.clicked.connect(self._on_check)
        btns.addWidget(self._check_btn)
        self._add_btn = QPushButton("Add sessions", self)
        self._add_btn.setStyleSheet("font-weight: bold; padding: 4px 16px;")
        self._add_btn.clicked.connect(self._on_add)
        btns.addWidget(self._add_btn)
        outer.addLayout(btns)

    # ------------------------------------------------------------------ #
    def _pick_file(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select a pose file", "",
            "Pose data (*.fdlc.parquet *.parquet *.csv *.h5);;All files (*)",
        )
        if p:
            self.source.setText(p)

    def _pick_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select a folder of pose files")
        if p:
            self.source.setText(p)

    def _render(self, rep: dict) -> None:
        lines = [
            f"<b>{len(rep['accepted'])}</b> of {len(rep['files'])} file(s) "
            f"match this project's pose "
            f"({len(rep['project_markers'])} markers)."
        ]
        for p in rep["accepted"]:
            lines.append(f"&nbsp;&nbsp;✓ {os.path.basename(p)}")
        for p, why in rep["rejected"].items():
            lines.append(
                f"&nbsp;&nbsp;<b>✗ {os.path.basename(p)}</b> — {why}")
        self.report.setHtml("<br>".join(lines))

    def _on_check(self) -> dict | None:
        if not self.config_path:
            QMessageBox.warning(self, self.title, "No project is loaded.")
            return None
        src = self.source.text().strip()
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, self.title, "Pick a file or folder first.")
            return None
        from mufasa.model.session_ingest import check_pose_compatibility
        try:
            rep = check_pose_compatibility(src, self.config_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, self.title, f"Could not inspect:\n{exc}")
            return None
        self._render(rep)
        return rep

    def _on_add(self) -> None:
        rep = self._on_check()
        if rep is None:
            return
        if not rep["accepted"]:
            QMessageBox.information(
                self, self.title,
                "No file matches this project's pose, so nothing was added.")
            return
        msg = (f"Import {len(rep['accepted'])} session(s)?")
        if rep["rejected"]:
            msg += f"\n\n{len(rep['rejected'])} file(s) will be skipped."
        if self.smooth.isChecked():
            msg += "\n\nThe new sessions will then be smoothed with the "\
                   "project's latest model."
        if QMessageBox.question(
            self, self.title, msg, QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        from mufasa.model.session_ingest import ingest_sessions
        self.setCursor(Qt.WaitCursor)
        try:
            res = ingest_sessions(
                self.config_path, self.source.text().strip(),
                smooth=self.smooth.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001
            self.unsetCursor()
            QMessageBox.critical(self, self.title, f"Add sessions failed:\n{exc}")
            return
        self.unsetCursor()
        out = [f"Imported {len(res['imported'])} session(s)."]
        if res.get("smoothed_dir"):
            out.append(f"Smoothed into:\n{res['smoothed_dir']}")
        out.extend(res.get("notes", []))
        QMessageBox.information(self, self.title, "\n\n".join(out))


__all__ = ["AddSessionsForm"]
