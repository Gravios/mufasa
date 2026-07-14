"""
mufasa.ui_qt.forms.model_modifications
======================================

"Model modifications" — post-creation edits to a project's pose model.

Currently: **rename markers**. The form lists the project's markers with an
editable "new name" column; pressing **Save model** validates the changes,
shows what will be affected (a dry run), and on confirmation propagates the
new names to:

* ``project.toml`` ``[pose].body_parts``;
* ``project.toml`` ``[skeleton]`` — nodes *and* edges, so the
  marker-connector relationships follow the rename;
* the imported pose parquets under ``csv/input_csv/``.

Because derived features embed marker names, the confirmation warns when
feature files exist (they must be recomputed after a rename).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ModelModificationsForm(QWidget):
    """Rename pose markers and propagate the change on save."""

    title = "Rename markers"

    def __init__(self, config_path: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self._build()
        self._reload()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(
            "Edit the <b>New name</b> column, then press "
            "<b>Save model</b> to rename markers everywhere "
            "(pose data, skeleton connections, and project.toml)."
        ))

        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Current marker", "New name"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        outer.addWidget(self.table, 1)

        row = QHBoxLayout()
        self._reload_btn = QPushButton("Reload markers", self)
        self._reload_btn.clicked.connect(self._reload)
        row.addWidget(self._reload_btn)
        row.addStretch()
        self._save_btn = QPushButton("Save model", self)
        self._save_btn.setStyleSheet("font-weight: bold; padding: 4px 16px;")
        self._save_btn.clicked.connect(self._on_save)
        row.addWidget(self._save_btn)
        outer.addLayout(row)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Refresh when the section is navigated to (markers may have
        # changed since the form was first built).
        super().showEvent(event)
        if hasattr(self, "table"):
            self._reload()

    def _current_markers(self) -> list[str]:
        if not self.config_path:
            return []
        try:
            from mufasa.project_layout import project_metadata_from_config
            return list(project_metadata_from_config(self.config_path).get(
                "body_parts", []))
        except Exception:
            return []

    def _reload(self) -> None:
        markers = self._current_markers()
        self.table.setRowCount(len(markers))
        for i, bp in enumerate(markers):
            cur = QTableWidgetItem(bp)
            cur.setFlags(cur.flags() & ~Qt.ItemIsEditable)  # read-only
            self.table.setItem(i, 0, cur)
            self.table.setItem(i, 1, QTableWidgetItem(bp))  # editable, pre-filled
        self._save_btn.setEnabled(bool(markers))

    def _collect_rename_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(self.table.rowCount()):
            old = self.table.item(i, 0).text()
            new_item = self.table.item(i, 1)
            new = new_item.text().strip() if new_item else ""
            if new and new != old:
                out[old] = new
        return out

    # ------------------------------------------------------------------ #
    def _on_save(self) -> None:
        if not self.config_path:
            QMessageBox.warning(self, self.title, "No project is loaded.")
            return
        rename_map = self._collect_rename_map()
        if not rename_map:
            QMessageBox.information(
                self, self.title, "No marker names were changed.")
            return

        from mufasa.model.marker_rename import rename_markers, validate_rename_map
        markers = self._current_markers()
        try:
            validate_rename_map(markers, rename_map)
            summary = rename_markers(self.config_path, rename_map, dry_run=True)
        except ValueError as exc:
            QMessageBox.warning(self, f"{self.title}: invalid rename", str(exc))
            return

        changes = "\n".join(f"    {o}  \u2192  {n}"
                            for o, n in rename_map.items())
        feat = summary.get("feature_files_need_recompute", 0)
        warn = (f"\n\n\u26a0 {feat} feature file(s) embed marker names and will "
                f"need to be recomputed after this rename."
                if feat else "")
        msg = (
            f"Rename {summary['n_renamed']} marker(s):\n\n{changes}\n\n"
            f"This will update project.toml, {summary['skeleton_edges']} "
            f"skeleton edge(s), and rewrite {summary['pose_files']} pose "
            f"file(s).{warn}\n\nProceed?"
        )
        if QMessageBox.question(
            self, "Save model", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        self.setCursor(Qt.WaitCursor)
        try:
            result = rename_markers(self.config_path, rename_map)
        except Exception as exc:  # noqa: BLE001
            self.unsetCursor()
            QMessageBox.critical(self, self.title, f"Rename failed:\n{exc}")
            return
        self.unsetCursor()
        QMessageBox.information(
            self, "Save model",
            f"Renamed {result['n_renamed']} marker(s). Updated project.toml, "
            f"the skeleton, and {result['pose_files']} pose file(s)."
            + (f"\n\nRemember to recompute features "
               f"({result['feature_files_need_recompute']} file(s) affected)."
               if result.get("feature_files_need_recompute") else ""),
        )
        self._reload()


__all__ = ["ModelModificationsForm"]
