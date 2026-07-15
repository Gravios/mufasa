"""
mufasa.ui_qt.forms.data_inspector
=================================

"Data inspector" — pick a file from the project and look at a sample of it,
without leaving the workbench or opening a notebook.

Sits after Data Import because that's when the question first comes up: the
import said it worked, but did the columns land where they should? The form
lists the project's data files grouped by pipeline stage (imported pose,
smoothed, outlier-corrected, features, classifications), and shows the first
N rows of whichever is selected, with the shape and the sample's NaN
fraction.

Only a head is read (see :mod:`mufasa.utils.data_sample`), so opening a
54,000-frame pose file is instant.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if f < 1024 or unit == "GiB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GiB"


class DataInspectorForm(QWidget):
    """Browse the project's data files and preview a sample of one."""

    title = ""  # the section heading already says "Data inspector"

    def __init__(self, config_path: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self._files: dict[str, list[str]] = {}
        self._build()
        self.reload()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        outer = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Source:", self))
        self.source = QComboBox(self)
        self.source.setMinimumWidth(180)
        self.source.currentTextChanged.connect(self._populate_files)
        top.addWidget(self.source)
        top.addSpacing(12)
        top.addWidget(QLabel("Rows:", self))
        self.rows = QSpinBox(self)
        self.rows.setRange(1, 1000)
        self.rows.setValue(50)
        self.rows.setToolTip("How many rows to read from the start of the file.")
        self.rows.valueChanged.connect(self._load_selected)
        top.addWidget(self.rows)
        top.addStretch()
        self._reload_btn = QPushButton("Reload", self)
        self._reload_btn.clicked.connect(self.reload)
        top.addWidget(self._reload_btn)
        outer.addLayout(top)

        split = QSplitter(Qt.Horizontal, self)

        self.file_list = QListWidget(split)
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentItemChanged.connect(lambda *_: self._load_selected())
        split.addWidget(self.file_list)

        right = QWidget(split)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.info = QLabel("", right)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info.setWordWrap(True)
        rl.addWidget(self.info)
        self.table = QTableWidget(0, 0, right)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        rl.addWidget(self.table, 1)
        split.addWidget(right)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        outer.addWidget(split, 1)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self.source.count() == 0:
            self.reload()

    # ------------------------------------------------------------------ #
    def reload(self) -> None:
        """Re-scan the project for data files."""
        if not self.config_path:
            self.info.setText("<i>No project loaded.</i>")
            return
        from mufasa.utils.data_sample import list_project_data_files
        self._files = list_project_data_files(self.config_path)
        self.source.blockSignals(True)
        current = self.source.currentText()
        self.source.clear()
        self.source.addItems(list(self._files.keys()))
        if current in self._files:
            self.source.setCurrentText(current)
        self.source.blockSignals(False)
        if not self._files:
            self.file_list.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.info.setText(
                "<i>No data files found. Import pose data first.</i>"
            )
            return
        self._populate_files()

    def _populate_files(self) -> None:
        from mufasa.utils.data_sample import describe_path
        self.file_list.clear()
        for path in self._files.get(self.source.currentText(), []):
            item = QListWidgetItem(describe_path(path))
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self.file_list.addItem(item)
        if self.file_list.count():
            self.file_list.setCurrentRow(0)

    def _load_selected(self) -> None:
        item = self.file_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.UserRole)
        from mufasa.utils.data_sample import format_columns, load_sample
        try:
            df, meta = load_sample(path, self.rows.value())
        except ValueError as exc:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.info.setText(f"<i>{exc}</i>")
            return

        total = meta["total_rows"]
        total_txt = f"{total:,}" if total is not None else "?"
        nan = meta["nan_fraction"]
        # An all-NaN sample almost always means the columns aren't what the
        # reader expected — worth making impossible to miss.
        nan_txt = (
            f"<span style='color:palette(placeholder-text)'>{nan:.1%} NaN</span>"
            if nan < 0.999
            else f"<b>{nan:.0%} NaN</b>"
        )
        self.info.setText(
            f"<b>{meta['sampled_rows']}</b> of <b>{total_txt}</b> rows &nbsp;·&nbsp; "
            f"<b>{meta['n_columns']}</b> columns &nbsp;·&nbsp; "
            f"{_human_size(meta['file_size'])} &nbsp;·&nbsp; {nan_txt}"
        )

        cols = format_columns(df)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(df))
        for r in range(len(df)):
            for c in range(len(cols)):
                v = df.iat[r, c]
                self.table.setItem(r, c, QTableWidgetItem(
                    "" if v is None else str(v)
                ))


__all__ = ["DataInspectorForm"]
