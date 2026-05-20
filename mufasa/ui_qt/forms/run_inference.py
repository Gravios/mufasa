"""
mufasa.ui_qt.forms.run_inference
=================================

Qt port of :class:`mufasa.ui.pop_ups.run_machine_models_popup.RunMachineModelsPopUp`
— the per-classifier model-path / threshold / minimum-bout-length
configurator that drives :class:`mufasa.model.inference_batch.InferenceBatch`.

Patch 122ap (this file)
-----------------------
The Tk popup carried a hand-built grid of rows (one per classifier)
with custom ``Entry_Box`` widgets that re-implemented validation
on each keystroke. Replaced here with a declarative ``QTableWidget``:
same operation, but the per-row machinery is uniform and the
per-cell validation is done at submit-time only (much less UI churn).

In-frame + dockable
-------------------
Subclasses :class:`OperationForm`, so it lives inline on the
Classifier page like any other section. The "Pop out" button
re-parents the form into a :class:`QDockWidget` attached to the
workbench's main window — same pattern as 122aj's frame labeller
and 122al's batch pre-processor. Floating, dockable to any area,
re-dockable back into the workbench.

What this form does
-------------------
For each classifier in the project (read from
``project_metadata_from_config(config_path)['classifier_targets']``):

* Picks a ``.sav`` model file via a file dialog.
* Sets a per-classifier probability threshold (0.0 – 1.0).
* Sets a minimum-bout length (ms) — bouts shorter than this are
  filtered out as noise.

On Run:

1. Validates all three values per classifier.
2. Persists them to ``project_config.ini``'s
   ``[SML settings]`` / ``[threshold_settings]`` /
   ``[Minimum_bout_lengths]`` sections — exactly the same write
   shape the Tk popup did, so :class:`InferenceBatch` reads them
   back in the same way.
3. Invokes :class:`InferenceBatch`'s run() with the current
   config, which iterates over ``self.feature_file_paths`` and
   writes per-video classification parquet files to
   ``derived/classifications/<video>.parquet`` (post-122ax;
   the legacy ``csv/machine_results/`` write was dropped).

v1 + legacy config persistence
------------------------------
Patch 122as: settings now persist to whichever format the project
uses. Legacy ``.ini`` projects write to ``[SML settings]`` /
``[threshold_settings]`` / ``[Minimum_bout_lengths]`` (same shape
the Tk popup did). v1 ``.toml`` projects write to per-classifier
sub-tables ``[classifier_inference.<name>]`` with keys
``model_path``, ``threshold``, ``min_bout_ms``. The TOML→CP
translator in 122as injects those values into the legacy CP
section names ``InferenceBatch`` already reads from, so the
backend doesn't need to change.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from mufasa.ui_qt.workbench import OperationForm

# Column layout for the per-classifier QTableWidget
COL_CLF       = 0
COL_PATH      = 1
COL_BROWSE    = 2
COL_THRESHOLD = 3
COL_MIN_BOUT  = 4
N_COLS = 5

_COL_HEADERS = ["Classifier", "Model path (.sav)", "", "Threshold", "Min bout (ms)"]


class RunInferenceForm(OperationForm):
    """In-frame Qt port of the inference runner."""

    title = "Run inference"
    description = (
        "Configure per-classifier model paths, probability thresholds, "
        "and minimum bout lengths, then run the inference batch over "
        "all videos with features. Writes per-video predictions to "
        "<code>derived/classifications/</code>. Settings persist to "
        "<code>project.toml</code> (v1) or <code>project_config.ini</code> "
        "(legacy)."
    )

    # ----------------------------------------------------------- State
    def __init__(self,
                 parent: QWidget | None = None,
                 config_path: str | None = None) -> None:
        self._docked_widget: QDockWidget | None = None
        self._classifier_targets: list[str] = []
        super().__init__(parent=parent, config_path=config_path)

    # ----------------------------------------------------------- UI
    def build(self) -> None:
        # ---- Inputs preview header ------------------------------ #
        self.preview = QLabel(self)
        self.preview.setWordWrap(True)
        self.body_layout.addWidget(self.preview)

        # ---- Per-classifier table ------------------------------- #
        self.table = QTableWidget(0, N_COLS, self)
        self.table.setHorizontalHeaderLabels(_COL_HEADERS)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_CLF, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_PATH, QHeaderView.Stretch)
        hh.setSectionResizeMode(COL_BROWSE, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_THRESHOLD,
                                QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_MIN_BOUT,
                                QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(160)
        self.body_layout.addWidget(self.table, 1)

        # ---- Action row (Reload / Pop out) ---------------------- #
        actions = QHBoxLayout()
        self.reload_btn = QPushButton("Reload classifier list", self)
        self.reload_btn.setToolTip(
            "Re-read the classifier list from project_config / "
            "project.toml. Use after adding or removing classifiers."
        )
        self.reload_btn.clicked.connect(self._reload)
        self.pop_out_btn = QPushButton("Pop out ⇱", self)
        self.pop_out_btn.setToolTip(
            "Detach this form into a floating dockable window. "
            "Click again to re-dock into the workbench."
        )
        self.pop_out_btn.clicked.connect(self._toggle_pop_out)
        actions.addWidget(self.reload_btn)
        actions.addStretch()
        actions.addWidget(self.pop_out_btn)
        self.body_layout.addLayout(actions)

        # Re-label the inherited Run button for clarity
        self.run_btn.setText("  Run inference")

        # Initial populate
        self._reload()

    # ----------------------------------------------------------- Reload
    def _reload(self) -> None:
        """Re-read the classifier list from the project config and
        rebuild the table, preserving any in-memory edits where the
        classifier name still exists."""
        # Snapshot current rows so we don't drop user edits if the
        # classifier list shape is unchanged.
        existing: dict[str, dict] = {}
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, COL_CLF)
            if name_item is None:
                continue
            existing[name_item.text()] = {
                "path":      self._row_path(row),
                "threshold": self._row_text(row, COL_THRESHOLD),
                "min_bout":  self._row_text(row, COL_MIN_BOUT),
            }

        targets = self._read_classifier_targets()
        self._classifier_targets = targets
        if not targets:
            self.preview.setText(
                "<b>No classifiers defined.</b> Add classifiers on the "
                "Classifier page's <i>Manage classifiers</i> section "
                "first."
            )
        else:
            self.preview.setText(
                f"Running inference for <b>{len(targets)} classifier(s)</b>. "
                f"Fill in a model path, threshold (0.0–1.0), and minimum "
                f"bout length (ms) per row. Predictions write to "
                f"<code>derived/classifications/</code>."
            )

        # Seed each row from INI (if present) or from snapshot above
        prior = self._read_existing_ini_settings()
        self.table.setRowCount(0)
        for idx, clf_name in enumerate(targets):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(
                row, COL_CLF, QTableWidgetItem(clf_name),
            )
            # Path field
            path_edit = QLineEdit(self)
            path_edit.setPlaceholderText("Select a .sav model file…")
            # Pre-fill: existing in-memory edit beats INI value
            if clf_name in existing:
                path_edit.setText(existing[clf_name]["path"])
            elif idx < len(prior["paths"]):
                path_edit.setText(prior["paths"][idx])
            self.table.setCellWidget(row, COL_PATH, path_edit)
            # Browse button
            browse_btn = QPushButton("Browse…", self)
            browse_btn.clicked.connect(
                lambda _, e=path_edit: self._on_browse(e),
            )
            self.table.setCellWidget(row, COL_BROWSE, browse_btn)
            # Threshold field
            thr_text = (
                existing[clf_name]["threshold"] if clf_name in existing
                else (prior["thresholds"][idx]
                      if idx < len(prior["thresholds"]) else "")
            )
            self.table.setItem(
                row, COL_THRESHOLD, QTableWidgetItem(thr_text),
            )
            # Min bout field
            mb_text = (
                existing[clf_name]["min_bout"] if clf_name in existing
                else (prior["min_bouts"][idx]
                      if idx < len(prior["min_bouts"]) else "")
            )
            self.table.setItem(
                row, COL_MIN_BOUT, QTableWidgetItem(mb_text),
            )

    def _on_browse(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model (.sav)", edit.text(),
            "SimBA classifier (*.sav)",
        )
        if path:
            edit.setText(path)

    def _row_path(self, row: int) -> str:
        w = self.table.cellWidget(row, COL_PATH)
        return w.text().strip() if isinstance(w, QLineEdit) else ""

    def _row_text(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text().strip() if item is not None else ""

    # ----------------------------------------------------------- Config IO
    def _read_classifier_targets(self) -> list[str]:
        """Read the project's classifier-target list. Mirrors the
        helper used by ClassifierManageForm so the two forms always
        agree on the list."""
        try:
            from mufasa.project_layout import project_metadata_from_config
            return list(
                project_metadata_from_config(self.config_path)
                .get("classifier_targets", [])
            )
        except Exception:
            return []

    def _read_existing_ini_settings(self) -> dict:
        """Pre-fill the table from prior runs. Returns three parallel
        lists indexed by classifier position so the build path can
        seed each row.

        Patch 122as: this now delegates to
        :func:`mufasa.project_layout.read_classifier_inference_settings`,
        which transparently handles both v1 TOML and legacy INI
        projects. Pre-122as this method was INI-only and the form
        showed empty fields on TOML projects.
        """
        empty = {"paths": [], "thresholds": [], "min_bouts": []}
        if not self.config_path:
            return empty
        try:
            from mufasa.project_layout import (
                read_classifier_inference_settings,
            )
            settings = read_classifier_inference_settings(self.config_path)
        except Exception:
            return empty
        paths, thresholds, min_bouts = [], [], []
        for clf_name in self._classifier_targets:
            row = settings.get(clf_name, {})
            paths.append(str(row.get("model_path", "")))
            thr = row.get("threshold")
            thresholds.append(
                "" if thr is None else str(thr),
            )
            mb = row.get("min_bout_ms")
            min_bouts.append(
                "" if mb is None else str(mb),
            )
        return {"paths": paths, "thresholds": thresholds,
                "min_bouts": min_bouts}

    def _write_settings_to_ini(self,
                               filtered: list[dict]) -> None:
        """Persist per-classifier settings to whichever config format
        the project uses (v1 TOML → ``[classifier_inference.<name>]``
        sub-tables; legacy INI → canonical sections).

        Patch 122as: previously raised RuntimeError on TOML projects.
        Now delegates to
        :func:`mufasa.project_layout.write_classifier_inference_settings`
        which handles both formats.
        """
        from mufasa.project_layout import (
            write_classifier_inference_settings,
        )
        settings = {
            row["name"]: {
                "model_path":  row["path"],
                "threshold":   float(row["threshold"]),
                "min_bout_ms": int(row["min_bout"]),
            }
            for row in filtered
        }
        write_classifier_inference_settings(
            self.config_path, settings,
        )

    # ----------------------------------------------------------- Pop-out
    def _toggle_pop_out(self) -> None:
        """Re-parent the form between the inline section and a
        floating QDockWidget. Mirrors 122al's BatchPreProcessForm
        pattern."""
        if self._docked_widget is None:
            main_window = self._find_main_window()
            if main_window is None:
                QMessageBox.information(
                    self, "Pop out",
                    "No main workbench window available; the form "
                    "must stay inline.",
                )
                return
            dock = QDockWidget("Run inference", main_window)
            dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            dock.setFeatures(
                QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
                | QDockWidget.DockWidgetClosable
            )
            self.setParent(dock)
            dock.setWidget(self)
            main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.setFloating(True)
            dock.show()
            self._docked_widget = dock
            self.pop_out_btn.setText("Re-dock ⇲")
            store = getattr(main_window, "_run_inference_docks", [])
            store.append(dock)
            main_window._run_inference_docks = store
        else:
            dock = self._docked_widget
            self._docked_widget = None
            section_host = getattr(self, "_section_host", None)
            if section_host is not None:
                self.setParent(section_host)
                section_host.layout().addWidget(self)
            dock.setWidget(None)
            dock.close()
            self.pop_out_btn.setText("Pop out ⇱")

    def _find_main_window(self) -> QWidget | None:
        from PySide6.QtWidgets import QMainWindow
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QMainWindow):
                return w
            w = w.parentWidget()
        return None

    # ----------------------------------------------------------- Execute
    def collect_args(self) -> dict:
        """Validate + persist per-classifier settings, then return
        the kwargs target() needs."""
        if not self._classifier_targets:
            raise ValueError(
                "No classifiers defined. Add at least one classifier "
                "via the 'Manage classifiers' section first."
            )

        filtered: list[dict] = []
        errors: list[str] = []
        for row in range(self.table.rowCount()):
            clf_name = self._row_text(row, COL_CLF)
            path = self._row_path(row)
            thr = self._row_text(row, COL_THRESHOLD)
            mb = self._row_text(row, COL_MIN_BOUT)
            if not path:
                errors.append(f"{clf_name}: model path is required.")
                continue
            if not os.path.isfile(path):
                errors.append(
                    f"{clf_name}: model file does not exist at "
                    f"{path!r}.",
                )
                continue
            try:
                thr_val = float(thr)
                if not 0.0 <= thr_val <= 1.0:
                    raise ValueError
            except ValueError:
                errors.append(
                    f"{clf_name}: threshold must be a float in "
                    f"[0.0, 1.0]; got {thr!r}.",
                )
                continue
            try:
                mb_val = int(mb)
                if mb_val < 0:
                    raise ValueError
            except ValueError:
                errors.append(
                    f"{clf_name}: minimum bout length must be a "
                    f"non-negative integer (ms); got {mb!r}.",
                )
                continue
            filtered.append({
                "name":      clf_name,
                "path":      path,
                "threshold": thr_val,
                "min_bout":  mb_val,
            })
        if errors:
            raise ValueError(
                "Inference settings have errors:\n  - "
                + "\n  - ".join(errors)
            )
        if not filtered:
            raise ValueError("No classifiers have complete settings.")

        # Persist to INI (matches Tk popup behaviour). Raises if the
        # project is v1-TOML-only.
        self._write_settings_to_ini(filtered)

        return {
            "config_path": self.config_path,
        }

    def target(self, *, config_path: str) -> None:
        """Drive InferenceBatch. Runs in a worker thread via
        OperationForm.on_run."""
        from mufasa.model.inference_batch import InferenceBatch
        InferenceBatch(
            config_path=config_path,
            features_dir=None,
            save_dir=None,
            minimum_bout_length=None,
        ).run()


__all__ = ["RunInferenceForm"]
