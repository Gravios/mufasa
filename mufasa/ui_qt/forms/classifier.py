"""
mufasa.ui_qt.forms.classifier
=============================

Inline forms for classifier management.

Replaces:

* :class:`AddClfPopUp`, :class:`RemoveAClassifierPopUp`,
  :class:`PrintModelInfoPopUp` → :class:`ClassifierManageForm`
  (3 popups collapsed into one form with an "action" selector).

Left as dedicated dialogs for now (interactive / long-running):

* ``KleinbergPopUp`` — burst-smoothing parameter sweep, live preview
* ``RunMachineModelsPopUp`` — inference runner with per-classifier
  threshold / min-bout configuration (enough fields to warrant its own
  panel)
"""
from __future__ import annotations

import configparser
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QLabel,
                               QLineEdit, QMessageBox, QStackedWidget,
                               QTextBrowser, QVBoxLayout, QWidget, QPushButton,
                               QHBoxLayout)

from mufasa.ui_qt.workbench import OperationForm


class _AddClfPanel(QWidget):
    """Collect the classifier name for the Add action."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self); form.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit(self)
        self.name.setPlaceholderText("e.g. Attack, Groom, Mount…")
        form.addRow("Classifier name:", self.name)


class _RemoveClfPanel(QWidget):
    """Choose which classifier to remove from a dropdown."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self); form.setContentsMargins(0, 0, 0, 0)
        self.name_cb = QComboBox(self)
        form.addRow("Classifier to remove:", self.name_cb)

    def set_options(self, names: list[str]) -> None:
        self.name_cb.clear()
        self.name_cb.addItems(names)


class _PrintClfPanel(QWidget):
    """Pick a ``.sav`` and show its metadata in a read-only pane."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.file_line = QLineEdit(self); self.file_line.setReadOnly(True)
        self.file_line.setPlaceholderText("Select a .sav model file…")
        self.browse_btn = QPushButton("Browse…", self)
        self.browse_btn.clicked.connect(self._browse)
        row.addWidget(self.file_line, 1); row.addWidget(self.browse_btn)
        outer.addLayout(row)
        self.info = QTextBrowser(self)
        self.info.setMinimumHeight(180)
        self.info.setPlaceholderText("Model metadata will appear here after Run.")
        outer.addWidget(self.info)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select classifier model", "",
            "Scikit-learn models (*.sav);;All files (*)",
        )
        if path:
            self.file_line.setText(path)


class ClassifierManageForm(OperationForm):
    """Add / remove / print classifiers — one form, three actions.

    Collapses :class:`AddClfPopUp`, :class:`RemoveAClassifierPopUp`,
    and :class:`PrintModelInfoPopUp`. The action dropdown drives which
    fields are shown.
    """

    title = "Manage classifiers"
    description = ("Add or remove a classifier name from the project "
                   "config, or inspect metadata of a trained ``.sav`` model.")

    ACTIONS = [("Add classifier",    "add"),
               ("Remove classifier", "remove"),
               ("Print model info",  "print")]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.action_cb = QComboBox(self)
        for label, _ in self.ACTIONS:
            self.action_cb.addItem(label)
        self.action_cb.currentIndexChanged.connect(self._on_action_changed)
        form.addRow("Action:", self.action_cb)

        self.panels = QStackedWidget(self)
        self.add_panel    = _AddClfPanel(self)
        self.remove_panel = _RemoveClfPanel(self)
        self.print_panel  = _PrintClfPanel(self)
        self.panels.addWidget(self.add_panel)
        self.panels.addWidget(self.remove_panel)
        self.panels.addWidget(self.print_panel)
        form.addRow("", self.panels)

        self.body_layout.addLayout(form)

        # Populate the remove dropdown from the current project config
        self._refresh_remove_options()

    def _on_action_changed(self, index: int) -> None:
        self.panels.setCurrentIndex(index)
        if index == 1:  # remove → refresh list
            self._refresh_remove_options()

    def _refresh_remove_options(self) -> None:
        names: list[str] = []
        if self.config_path:
            try:
                cfg = configparser.ConfigParser()
                cfg.read(self.config_path)
                n = cfg.getint("SML settings", "no_targets", fallback=0)
                for i in range(1, n + 1):
                    key = f"target_name_{i}"
                    if cfg.has_option("SML settings", key):
                        names.append(cfg.get("SML settings", key))
            except Exception:
                pass
        self.remove_panel.set_options(names)

    def collect_args(self) -> dict:
        action = self.ACTIONS[self.action_cb.currentIndex()][1]
        if action == "add":
            name = self.add_panel.name.text().strip()
            if not name:
                raise ValueError("Classifier name is required.")
            return {"action": "add", "name": name}
        if action == "remove":
            name = self.remove_panel.name_cb.currentText().strip()
            if not name:
                raise ValueError("No classifier selected to remove.")
            return {"action": "remove", "name": name}
        if action == "print":
            path = self.print_panel.file_line.text().strip()
            if not path:
                raise ValueError("No .sav file selected.")
            return {"action": "print", "path": path}
        raise ValueError(f"unknown action: {action}")

    def target(self, *, action: str, **params) -> None:
        if action == "add":
            self._add_classifier(params["name"])
        elif action == "remove":
            self._remove_classifier(params["name"])
        elif action == "print":
            self._print_model(params["path"])

    # ------------------------------------------------------------------ #
    # Action implementations — config-file-level, no heavy backend needed
    # ------------------------------------------------------------------ #
    def _add_classifier(self, name: str) -> None:
        if not self.config_path:
            raise RuntimeError("No project_config.ini loaded.")
        cfg = configparser.ConfigParser()
        cfg.read(self.config_path)
        n = cfg.getint("SML settings", "no_targets", fallback=0)
        # Deduplicate
        existing = [cfg.get("SML settings", f"target_name_{i}", fallback="")
                    for i in range(1, n + 1)]
        if name in existing:
            raise ValueError(f"'{name}' already exists in this project.")
        cfg.set("SML settings", "no_targets", str(n + 1))
        cfg.set("SML settings", f"target_name_{n + 1}", name)
        with open(self.config_path, "w") as f:
            cfg.write(f)
        # Refresh the dropdown for follow-up removes
        self._refresh_remove_options()

    def _remove_classifier(self, name: str) -> None:
        if not self.config_path:
            raise RuntimeError("No project_config.ini loaded.")
        cfg = configparser.ConfigParser()
        cfg.read(self.config_path)
        n = cfg.getint("SML settings", "no_targets", fallback=0)
        kept = []
        for i in range(1, n + 1):
            val = cfg.get("SML settings", f"target_name_{i}", fallback="")
            if val and val != name:
                kept.append(val)
            cfg.remove_option("SML settings", f"target_name_{i}")
        for j, v in enumerate(kept, start=1):
            cfg.set("SML settings", f"target_name_{j}", v)
        cfg.set("SML settings", "no_targets", str(len(kept)))
        with open(self.config_path, "w") as f:
            cfg.write(f)
        self._refresh_remove_options()

    def _print_model(self, path: str) -> None:
        try:
            from mufasa.utils.read_write import tabulate_clf_info
            info = tabulate_clf_info(clf_path=path)
        except Exception as exc:
            raise RuntimeError(f"Could not read model metadata: {exc}")
        # `tabulate_clf_info` prints to stdout; we capture into the
        # read-only pane for visibility. If it returns a string, show that.
        if isinstance(info, str) and info:
            self.print_panel.info.setPlainText(info)
        else:
            self.print_panel.info.setPlainText(
                f"Model loaded from:\n{path}\n\n"
                f"(details printed to console; detailed in-UI display "
                f"pending `tabulate_clf_info` return-value refactor.)"
            )


__all__ = ["ClassifierManageForm"]
