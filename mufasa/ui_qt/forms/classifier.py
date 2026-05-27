"""
mufasa.ui_qt.forms.classifier
=============================

Classifier management UI — define classifiers, assign per-classifier
keyboard hotkeys (used by Frame labelling), inspect trained models,
and remove classifiers.

Patch 122fe — full redesign from the legacy action-dropdown form.
User request (May 26, 2026):

  > Manage classifiers : Manage classifiers, should present the
  > classifiers in a table format with a delete button, and show
  > info button for each classifier. Guard against no classifiers.
  > At the bottom of the table there should be a "+" button for
  > adding a new classifier. Guard against empty or duplicate name.
  > no need for a run button. Each classifier should have an
  > associated key that is used by Frame labeling.

The form now presents one row per classifier in a QTableWidget:
columns Name, Key, Info, Delete. The Run button is hidden — all
actions happen via per-row buttons or the table-footer "+" button.

Storage in project.toml::

    [classifiers]
    targets = ["Attack", "Groom"]

    [classifiers.keys]
    Attack = "a"
    Groom = "g"

The ``targets`` list is the canonical classifier identity (one
entry per classifier). The ``keys`` table is a parallel mapping
from classifier name → keyboard hotkey character. Either may be
empty / missing. Frame labelling reads ``keys`` to bind hotkeys
during annotation (separate patch).

Legacy migration: pre-122fe projects with only ``targets`` and
no ``keys`` table show "(unset)" in the Key column. Editing
adds a key entry; deletion removes both.

Replaces (122fe):

* The ACTIONS-dropdown UI (Add / Remove / Print info) — all
  three actions now have first-class UI affordances directly
  on the table view.
* :class:`_AddClfPanel`, :class:`_RemoveClfPanel`,
  :class:`_PrintClfPanel` — removed; functionality folded into
  the new table + add-dialog.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufasa.ui_qt.workbench import OperationForm

# --------------------------------------------------------------------------- #
# Storage helpers
# --------------------------------------------------------------------------- #
# Patch 122f — these wrap the read-modify-write of classifier target
# names so the form methods don't have to branch on v1 vs legacy. v1
# reads/writes ``[classifiers].targets`` in project.toml; legacy
# reads/writes ``[SML settings] target_name_N``.

def _read_classifiers(config_path: str) -> list[str]:
    from mufasa.project_layout import project_metadata_from_config
    try:
        return list(
            project_metadata_from_config(config_path)["classifier_targets"]
        )
    except (ValueError, OSError, KeyError):
        return []


def _write_classifiers(config_path: str, targets: list[str]) -> None:
    cp = Path(config_path)
    if str(cp).lower().endswith(".toml"):
        from mufasa.project_layout import (
            read_project_toml,
            write_project_toml,
        )
        data = read_project_toml(cp)
        classifiers = dict(data.get("classifiers", {}))
        classifiers["targets"] = list(targets)
        data["classifiers"] = classifiers
        write_project_toml(cp, data)
        return
    # Legacy: rewrite the [SML settings] target_name_N keys + no_targets.
    import configparser as _cp
    cfg = _cp.ConfigParser()
    cfg.read(cp)
    if not cfg.has_section("SML settings"):
        cfg.add_section("SML settings")
    n = cfg.getint("SML settings", "no_targets", fallback=0)
    for i in range(1, n + 1):
        cfg.remove_option("SML settings", f"target_name_{i}")
    for j, v in enumerate(targets, start=1):
        cfg.set("SML settings", f"target_name_{j}", v)
    cfg.set("SML settings", "no_targets", str(len(targets)))
    with open(cp, "w") as f:
        cfg.write(f)


def _read_classifier_keys(config_path: str) -> dict[str, str]:
    """Patch 122fe — Read the per-classifier keyboard hotkey map.

    Returns a dict mapping classifier name → key character (e.g.
    {"Attack": "a", "Groom": "g"}). Missing / non-existent /
    legacy-only-format projects yield an empty dict.

    Frame labelling (separate patch) reads this map to bind hotkeys
    during annotation.
    """
    cp = Path(config_path)
    if not str(cp).lower().endswith(".toml"):
        # Legacy projects don't have a place to store keys. Return
        # empty; the user can add keys after migrating the project.
        return {}
    from mufasa.project_layout import read_project_toml
    try:
        data = read_project_toml(cp)
    except (FileNotFoundError, OSError):
        return {}
    classifiers = data.get("classifiers", {})
    keys = classifiers.get("keys", {}) if isinstance(classifiers, dict) else {}
    # Sanitise: only string→string entries.
    return {
        str(k): str(v) for k, v in keys.items()
        if isinstance(k, str) and isinstance(v, str)
    }


def _write_classifier_keys(
    config_path: str, keys: dict[str, str],
) -> None:
    """Patch 122fe — Write the per-classifier keyboard hotkey map.

    For v1 .toml projects, writes ``[classifiers.keys]``. Legacy
    projects don't support keys yet (no migration path); silently
    no-op so the legacy form can still be used for basic add/remove.
    """
    cp = Path(config_path)
    if not str(cp).lower().endswith(".toml"):
        return
    from mufasa.project_layout import (
        read_project_toml,
        write_project_toml,
    )
    data = read_project_toml(cp)
    classifiers = dict(data.get("classifiers", {}))
    # Drop empty-string values so the table stays clean.
    classifiers["keys"] = {
        str(k): str(v) for k, v in keys.items() if v
    }
    data["classifiers"] = classifiers
    write_project_toml(cp, data)


def _find_classifier_model(
    config_path: str, name: str,
) -> Path | None:
    """Patch 122fe — Best-effort lookup of a trained .sav model file
    for the named classifier.

    Looks in the conventional model output dirs (models/,
    derived/models/, csv/models/). Returns the first match or None.
    Used by the Info button to show "Trained: yes/no" status.
    """
    try:
        root = Path(config_path).parent
    except (TypeError, ValueError):
        return None
    candidates = [
        root / "models" / f"{name}.sav",
        root / "derived" / "models" / f"{name}.sav",
        root / "csv" / "models" / f"{name}.sav",
        root / "models" / f"{name}.pickle",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


# --------------------------------------------------------------------------- #
# Add-classifier dialog
# --------------------------------------------------------------------------- #
class _AddClassifierDialog(QDialog):
    """Patch 122fe — Modal dialog for adding a new classifier.

    Two fields: name (required, non-empty, unique) and key (single
    character, optional but recommended). The OK button is
    enabled only when both validations pass; otherwise an inline
    label explains why.

    On accept, the new classifier is appended to
    ``[classifiers].targets`` and (if a key was given) recorded in
    ``[classifiers.keys]``.
    """

    def __init__(self, parent: QWidget, config_path: str) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.setWindowTitle("Add classifier")
        self.setMinimumWidth(360)

        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        form = QFormLayout()
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("e.g. Attack, Groom, Mount…")
        form.addRow("Name:", self.name_edit)

        self.key_edit = QLineEdit(self)
        self.key_edit.setMaxLength(1)
        self.key_edit.setPlaceholderText("e.g. a, g, m (single char)")
        self.key_edit.setFixedWidth(60)
        form.addRow("Key:", self.key_edit)

        outer.addLayout(form)

        # Inline status label — updated on every text change so the
        # user sees WHY OK is disabled.
        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: #b34141;")
        outer.addWidget(self.status_label)

        # OK / Cancel buttons.
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self.ok_btn = buttons.button(QDialogButtonBox.Ok)
        outer.addWidget(buttons)

        # Re-validate on every text change.
        self.name_edit.textChanged.connect(self._validate)
        self.key_edit.textChanged.connect(self._validate)
        self._validate()

    def _validate(self) -> None:
        """Update the inline status label + OK button enable state."""
        name = self.name_edit.text().strip()
        key = self.key_edit.text().strip()
        if not name:
            self.status_label.setText("Name is required.")
            self.ok_btn.setEnabled(False)
            return
        existing = _read_classifiers(self.config_path)
        if name in existing:
            self.status_label.setText(
                f"'{name}' already exists — pick a different name.",
            )
            self.ok_btn.setEnabled(False)
            return
        # Check for duplicate keys (warn, don't block — user might
        # be intentionally reassigning a freed-up key).
        if key:
            existing_keys = _read_classifier_keys(self.config_path)
            collisions = [
                n for n, k in existing_keys.items() if k == key
            ]
            if collisions:
                self.status_label.setStyleSheet("color: #c87f00;")
                self.status_label.setText(
                    f"Key '{key}' is already used by "
                    f"{collisions[0]!r}. Continuing will leave that "
                    f"classifier without a hotkey."
                )
                self.ok_btn.setEnabled(True)
                return
        self.status_label.setStyleSheet("color: #2a8a2a;")
        self.status_label.setText("Ready.")
        self.ok_btn.setEnabled(True)

    def _on_accept(self) -> None:
        """Patch 122fe — Persist the new classifier on OK."""
        name = self.name_edit.text().strip()
        key = self.key_edit.text().strip()
        existing = _read_classifiers(self.config_path)
        if name in existing or not name:
            # Should have been caught by _validate; defensive guard.
            return
        _write_classifiers(self.config_path, existing + [name])
        if key:
            keys = _read_classifier_keys(self.config_path)
            # If the new key collides with an existing one, clear
            # the collision (per the warning the user saw).
            for n in list(keys):
                if keys[n] == key:
                    keys.pop(n, None)
            keys[name] = key
            _write_classifier_keys(self.config_path, keys)
        self.accept()


# --------------------------------------------------------------------------- #
# Edit-key dialog
# --------------------------------------------------------------------------- #
class _EditKeyDialog(QDialog):
    """Patch 122fe — Modal for changing an existing classifier's key.

    Lets the user reassign or clear a hotkey. Same collision-warning
    behavior as the add dialog.
    """

    def __init__(self, parent: QWidget, config_path: str,
                 name: str, current_key: str) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.name = name
        self.setWindowTitle(f"Edit key — {name}")
        self.setMinimumWidth(360)

        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        outer.addWidget(QLabel(
            f"Set the keyboard hotkey for '{name}'. Leave empty to "
            f"clear.",
        ))

        form = QFormLayout()
        self.key_edit = QLineEdit(self)
        self.key_edit.setMaxLength(1)
        self.key_edit.setText(current_key)
        self.key_edit.setFixedWidth(60)
        form.addRow("Key:", self.key_edit)
        outer.addLayout(form)

        self.status_label = QLabel("", self)
        outer.addWidget(self.status_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.key_edit.textChanged.connect(self._validate)
        self._validate()

    def _validate(self) -> None:
        key = self.key_edit.text().strip()
        if not key:
            self.status_label.setStyleSheet("color: #666;")
            self.status_label.setText(
                "(empty — clears the hotkey for this classifier)",
            )
            return
        existing_keys = _read_classifier_keys(self.config_path)
        collisions = [
            n for n, k in existing_keys.items()
            if k == key and n != self.name
        ]
        if collisions:
            self.status_label.setStyleSheet("color: #c87f00;")
            self.status_label.setText(
                f"Key '{key}' is also assigned to "
                f"{collisions[0]!r}. Continuing will clear that "
                f"classifier's hotkey."
            )
        else:
            self.status_label.setStyleSheet("color: #2a8a2a;")
            self.status_label.setText("Ready.")

    def _on_accept(self) -> None:
        key = self.key_edit.text().strip()
        keys = _read_classifier_keys(self.config_path)
        # Clear any colliding key.
        if key:
            for n in list(keys):
                if keys[n] == key and n != self.name:
                    keys.pop(n, None)
            keys[self.name] = key
        else:
            keys.pop(self.name, None)
        _write_classifier_keys(self.config_path, keys)
        self.accept()


# --------------------------------------------------------------------------- #
# Main form — table-based management
# --------------------------------------------------------------------------- #
class ClassifierManageForm(OperationForm):
    """Patch 122fe — Table-based classifier management UI.

    Per-row columns: Name | Key | Info | Delete. The "+" button at
    the bottom opens the add-classifier dialog. The Run button
    inherited from OperationForm is hidden — all actions are
    immediate (per-row buttons or the add button), no submit step.
    """

    title = "Manage classifiers"
    description = (
        "Define behavior classifiers for the project. Each classifier "
        "needs a unique name and (optionally) a single-character "
        "keyboard hotkey used by Frame labelling."
    )

    # Class attributes used by OperationForm but no-op for this form.
    section_id = None  # No provenance — config-edit only.

    def build(self) -> None:
        """Build the table UI. Called by ``OperationForm.__init__``."""
        # Hide the inherited Run button — actions are immediate.
        # Patch 122fe — user request: "no need for a run button."
        try:
            self.run_btn.setVisible(False)
        except AttributeError:
            # OperationForm may not have constructed run_btn yet on
            # some legacy paths; harmless if missing.
            pass

        outer = self.body_layout

        # Header / instructions.
        header = QLabel(
            "Classifiers defined for this project. Click the + button "
            "below the table to add a new one.",
            self,
        )
        header.setWordWrap(True)
        outer.addWidget(header)

        # Table.
        self.table = QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Key", "Info", "Delete"],
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SingleSelection,
        )
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(160)
        outer.addWidget(self.table)

        # Empty-state placeholder. Shown when there are no
        # classifiers — replaces the table view to avoid showing an
        # empty white box with no affordance.
        self.empty_label = QLabel(
            "No classifiers defined yet. Click the + button below "
            "to add the first one.",
            self,
        )
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet(
            "color: #666; padding: 24px;"
        )
        f = QFont(); f.setItalic(True)
        self.empty_label.setFont(f)
        outer.addWidget(self.empty_label)

        # "+" button row.
        add_row = QHBoxLayout()
        add_row.addStretch()
        self.add_btn = QPushButton("+ Add classifier", self)
        self.add_btn.setStyleSheet(
            "padding: 6px 16px; font-weight: bold;",
        )
        self.add_btn.clicked.connect(self._on_add_clicked)
        add_row.addWidget(self.add_btn)
        outer.addLayout(add_row)

        # Initial table population.
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Rebuild the table rows from project.toml state."""
        names = (
            _read_classifiers(self.config_path)
            if self.config_path else []
        )
        keys = (
            _read_classifier_keys(self.config_path)
            if self.config_path else {}
        )

        self.table.setRowCount(len(names))
        for i, name in enumerate(names):
            # Name (read-only).
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, name_item)

            # Key — click-to-edit via separate button cell. We use
            # a QTableWidgetItem here for display + a per-row button
            # would be redundant; instead clicking the cell opens
            # the edit-key dialog.
            key_text = keys.get(name, "")
            key_item = QTableWidgetItem(
                key_text if key_text else "(unset)",
            )
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            if not key_text:
                key_item.setForeground(Qt.gray)
            self.table.setItem(i, 1, key_item)

            # Info button.
            info_btn = QPushButton("info", self)
            info_btn.setFixedWidth(54)
            # Bind name via default-arg closure to avoid late-binding.
            info_btn.clicked.connect(
                lambda _checked=False, n=name: self._on_info_clicked(n),
            )
            self.table.setCellWidget(i, 2, info_btn)

            # Delete button.
            del_btn = QPushButton("×", self)
            del_btn.setFixedWidth(32)
            del_btn.setStyleSheet(
                "QPushButton { color: #b34141; "
                "font-weight: bold; font-size: 14pt; }"
            )
            del_btn.clicked.connect(
                lambda _checked=False, n=name: self._on_delete_clicked(n),
            )
            self.table.setCellWidget(i, 3, del_btn)

        # Empty-state toggle.
        has_any = bool(names)
        self.empty_label.setVisible(not has_any)
        self.table.setVisible(has_any)

        # Double-click on Key cell → edit-key dialog.
        try:
            self.table.cellDoubleClicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)

    # ------------------------------------------------------------------ #
    # Action handlers
    # ------------------------------------------------------------------ #
    def _on_add_clicked(self) -> None:
        """Open the add-classifier dialog and refresh on accept."""
        if not self.config_path:
            QMessageBox.warning(
                self, "No project",
                "Load a project before adding classifiers.",
            )
            return
        dlg = _AddClassifierDialog(self, self.config_path)
        if dlg.exec() == QDialog.Accepted:
            self._refresh_table()

    def _on_delete_clicked(self, name: str) -> None:
        """Confirm and remove the named classifier."""
        if not self.config_path:
            return
        if QMessageBox.question(
            self, "Delete classifier",
            f"Delete classifier '{name}'? This removes it from the "
            f"project but doesn't delete any trained model files.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        existing = _read_classifiers(self.config_path)
        _write_classifiers(
            self.config_path, [n for n in existing if n != name],
        )
        keys = _read_classifier_keys(self.config_path)
        keys.pop(name, None)
        _write_classifier_keys(self.config_path, keys)
        self._refresh_table()

    def _on_info_clicked(self, name: str) -> None:
        """Show classifier metadata: name, key, trained-model status."""
        if not self.config_path:
            return
        keys = _read_classifier_keys(self.config_path)
        key = keys.get(name, "")
        model_path = _find_classifier_model(self.config_path, name)
        lines = [
            f"<b>Name:</b> {name}",
            f"<b>Key:</b> {key if key else '<i>(unset)</i>'}",
            f"<b>Trained model:</b> "
            f"{str(model_path) if model_path else '<i>(none — train this classifier first)</i>'}",
        ]
        QMessageBox.information(
            self, f"Classifier — {name}",
            "<br>".join(lines),
        )

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Double-clicking the Key column (col=1) opens the edit
        dialog. Other columns are ignored."""
        if col != 1:
            return
        name_item = self.table.item(row, 0)
        if name_item is None:
            return
        name = name_item.text()
        keys = _read_classifier_keys(self.config_path)
        dlg = _EditKeyDialog(
            self, self.config_path, name, keys.get(name, ""),
        )
        if dlg.exec() == QDialog.Accepted:
            self._refresh_table()

    # ------------------------------------------------------------------ #
    # OperationForm contract — no-ops; actions are immediate.
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        # No Run button means collect_args is never called by the
        # OperationForm machinery. Defensive return.
        return {}

    def target(self, **params) -> None:
        # Same. Never called via the Run path.
        pass


__all__ = ["ClassifierManageForm"]
