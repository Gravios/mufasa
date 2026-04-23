"""
mufasa.ui_qt.forms.pose_cleanup
===============================

Inline forms for the pose-data cleanup pipeline. Sits early in every
analysis: raw pose → clean pose → feature extraction → classifier.

Replaces:

* :class:`SmoothingPopUp` → :class:`SmoothingForm` (also drops the
  dialog-style Qt port from turn 5 — one canonical surface)
* :class:`InterpolatePopUp` → :class:`InterpolateForm`
* :class:`OutlierSettingsPopUp` → :class:`OutlierSettingsForm`
* :class:`DropTrackingDataPopUp` → :class:`DropBodypartsForm`

Leaves :class:`EgocentricAlignPopUp` as a dialog — it needs interactive
click-on-frame reference selection, which is the one case a window is
still justified. The page wires a launcher button for it.

Design notes
------------

All four forms take ``config_path`` and rely on ``ConfigReader`` to
discover project state (animal names, body-parts, input directory).
The **Audit A1** fix from turn 5 means :class:`SmoothingForm` and
:class:`InterpolateForm` no longer need to compute
``multi_index_df_headers`` themselves — the backends auto-detect when
``None`` is passed. Substantially simpler forms than the Tk originals.
"""
from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGridLayout,
                               QGroupBox, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QSpinBox,
                               QDoubleSpinBox, QVBoxLayout, QWidget,
                               QPushButton, QFileDialog)

from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_animal_bps(config_path: str) -> dict[str, list[str]]:
    """Parse project_config.ini and return ``{animal_name: [bp_names]}``.

    Does this without instantiating :class:`ConfigReader` (which pulls
    numba). Reads the ``Multi animal IDs`` / ``project_bp_names.csv``
    path the same way the legacy popups do.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    project_path = cfg.get("General settings", "project_path", fallback=None)
    # Animal IDs
    animal_ids_raw = cfg.get(
        "Multi animal IDs", "id_list", fallback=""
    ).strip()
    animals = [a.strip() for a in animal_ids_raw.split(",") if a.strip()]
    if not animals:
        animals = ["Animal_1"]
    # Body parts from logs/measures/pose_configs/bp_names/project_bp_names.csv
    bp_file = None
    if project_path:
        p = Path(project_path) / "logs" / "measures" / "pose_configs" \
            / "bp_names" / "project_bp_names.csv"
        if p.is_file():
            bp_file = p
    bps_per_animal = {}
    if bp_file:
        all_bps = [ln.strip() for ln in bp_file.read_text().splitlines() if ln.strip()]
        # Heuristic: if N animals × M body-parts, split by prefix or round-robin
        # Most SimBA projects name bps as "animal1_nose", "animal1_ear", etc.
        for animal in animals:
            prefixed = [bp for bp in all_bps if bp.startswith(animal)]
            if prefixed:
                # Strip prefix for display
                bps_per_animal[animal] = [bp.removeprefix(f"{animal}_")
                                          for bp in prefixed]
            else:
                # Fall back to giving every animal the full list — the user
                # still picks which bodypart is which reference
                bps_per_animal[animal] = all_bps
    else:
        # No bp file found → empty; the form will raise clearly.
        for a in animals:
            bps_per_animal[a] = []
    return bps_per_animal


# --------------------------------------------------------------------------- #
# SmoothingForm
# --------------------------------------------------------------------------- #
class SmoothingForm(OperationForm):
    """Gaussian / Savitzky-Golay smoothing on pose time-series.

    The ``multi_index_df_headers`` flag is auto-detected by the backend
    (Audit A1 fix, turn 5), so the form has nothing to say about that.
    """

    title = "Smooth pose data"
    description = ("Temporal smoothing to remove sub-pixel tracker jitter "
                   "before feature extraction. Savitzky-Golay preserves "
                   "peaks better than Gaussian but costs more CPU.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.method_cb = QComboBox(self)
        self.method_cb.addItems(["Savitzky-Golay", "Gaussian"])
        form.addRow("Method:", self.method_cb)

        self.window_ms = QSpinBox(self)
        self.window_ms.setRange(1, 5000); self.window_ms.setValue(500)
        self.window_ms.setSuffix(" ms")
        form.addRow("Time window:", self.window_ms)

        self.copy_originals = QCheckBox("Copy originals before overwriting", self)
        self.copy_originals.setChecked(True)
        form.addRow("", self.copy_originals)

        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        return {
            "config_path":    self.config_path,
            "method":         self.method_cb.currentText(),
            "time_window":    int(self.window_ms.value()),
            "copy_originals": bool(self.copy_originals.isChecked()),
        }

    def target(self, *, config_path: str, method: str, time_window: int,
               copy_originals: bool) -> None:
        from mufasa.data_processors.smoothing import Smoothing
        cfg = configparser.ConfigParser(); cfg.read(config_path)
        input_dir = os.path.join(
            cfg.get("General settings", "project_path"),
            "csv", "input_csv",
        )
        Smoothing(
            config_path=config_path,
            data_path=input_dir,
            time_window=time_window,
            method=method,
            # multi_index_df_headers=None → auto-detect (A1 fix)
            copy_originals=copy_originals,
        ).run()


# --------------------------------------------------------------------------- #
# InterpolateForm
# --------------------------------------------------------------------------- #
class InterpolateForm(OperationForm):
    """Fill missing / low-confidence frames with interpolation.

    Same ``multi_index_df_headers`` auto-detect as SmoothingForm.
    """

    title = "Interpolate missing frames"
    description = ("Fill gaps left by tracker dropouts. Nearest is fastest "
                   "and least biased; linear/quadratic are smoother but can "
                   "invent values during long gaps.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.type_cb = QComboBox(self)
        self.type_cb.addItems(["Body-parts", "Animals"])
        form.addRow("Interpolate by:", self.type_cb)

        self.method_cb = QComboBox(self)
        self.method_cb.addItems(["Nearest", "Linear", "Quadratic"])
        form.addRow("Method:", self.method_cb)

        self.copy_originals = QCheckBox("Copy originals before overwriting", self)
        self.copy_originals.setChecked(True)
        form.addRow("", self.copy_originals)

        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        return {
            "config_path":    self.config_path,
            "type":           self.type_cb.currentText().lower().replace(" ", "-"),
            "method":         self.method_cb.currentText().lower(),
            "copy_originals": bool(self.copy_originals.isChecked()),
        }

    def target(self, *, config_path: str, type: str, method: str,
               copy_originals: bool) -> None:
        from mufasa.data_processors.interpolate import Interpolate
        cfg = configparser.ConfigParser(); cfg.read(config_path)
        input_dir = os.path.join(
            cfg.get("General settings", "project_path"),
            "csv", "input_csv",
        )
        Interpolate(
            config_path=config_path,
            data_path=input_dir,
            type=type,
            method=method,
            # multi_index_df_headers=None → auto-detect (A1 fix)
            copy_originals=copy_originals,
        ).run()


# --------------------------------------------------------------------------- #
# OutlierSettingsForm
# --------------------------------------------------------------------------- #
class OutlierSettingsForm(OperationForm):
    """Configure outlier-correction thresholds and per-animal reference
    body-parts. Writes to project_config.ini (Audit A2 fix: writes
    only to the "Outlier settings" section, no silent mutation
    elsewhere).

    Per-animal fields (two ref-body-part dropdowns each) are generated
    from the project's animal-ID list.
    """

    title = "Outlier correction — settings"
    description = ("Configure per-animal reference body-parts and "
                   "location/movement criteria for the outlier corrector. "
                   "Running correction is a separate step on the "
                   "<i>Features</i> page.")

    def build(self) -> None:
        outer = QVBoxLayout()

        # Criteria box
        crit_box = QGroupBox("Criteria (multipliers of reference distance)", self)
        crit_form = QFormLayout(crit_box)
        self.location_crit = QDoubleSpinBox(self)
        self.location_crit.setRange(0.0, 99.0); self.location_crit.setValue(1.5)
        self.location_crit.setSingleStep(0.1)
        crit_form.addRow("Location criterion:", self.location_crit)
        self.movement_crit = QDoubleSpinBox(self)
        self.movement_crit.setRange(0.0, 99.0); self.movement_crit.setValue(0.7)
        self.movement_crit.setSingleStep(0.1)
        crit_form.addRow("Movement criterion:", self.movement_crit)
        outer.addWidget(crit_box)

        # Per-animal ref body-parts
        self.refs_box = QGroupBox("Per-animal reference body-parts", self)
        self._refs_layout = QGridLayout(self.refs_box)
        self._refs_layout.addWidget(QLabel("<b>Animal</b>"), 0, 0)
        self._refs_layout.addWidget(QLabel("<b>Body-part 1</b>"), 0, 1)
        self._refs_layout.addWidget(QLabel("<b>Body-part 2</b>"), 0, 2)
        self._ref_fields: dict[str, tuple[QComboBox, QComboBox]] = {}
        self._populate_refs()
        outer.addWidget(self.refs_box)

        self.aggregation_cb = QComboBox(self)
        self.aggregation_cb.addItems(["Mean", "Median"])
        agg_row = QFormLayout()
        agg_row.addRow("Reference-distance aggregation:", self.aggregation_cb)
        outer.addLayout(agg_row)

        self.body_layout.addLayout(outer)

    def _populate_refs(self) -> None:
        if not self.config_path:
            return
        try:
            animal_bps = _load_animal_bps(self.config_path)
        except Exception:
            return
        # Pull in existing values from config if present so the form
        # isn't destructive on re-open
        cfg = configparser.ConfigParser(); cfg.read(self.config_path)
        for r, (animal, bps) in enumerate(animal_bps.items(), start=1):
            self._refs_layout.addWidget(QLabel(animal), r, 0)
            cb1 = QComboBox(self.refs_box); cb1.addItems(bps)
            cb2 = QComboBox(self.refs_box); cb2.addItems(bps)
            # Try to restore existing selection
            sect = "Outlier settings"
            key1 = f"{animal}_location_bp_1".lower().replace(" ", "_")
            key2 = f"{animal}_location_bp_2".lower().replace(" ", "_")
            if cfg.has_option(sect, key1):
                prev1 = cfg.get(sect, key1)
                if prev1 in bps:
                    cb1.setCurrentText(prev1)
            if cfg.has_option(sect, key2):
                prev2 = cfg.get(sect, key2)
                if prev2 in bps:
                    cb2.setCurrentText(prev2)
            elif len(bps) > 1:
                cb2.setCurrentIndex(1)  # default to second bp
            self._refs_layout.addWidget(cb1, r, 1)
            self._refs_layout.addWidget(cb2, r, 2)
            self._ref_fields[animal] = (cb1, cb2)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        if not self._ref_fields:
            raise RuntimeError(
                "No animals / body-parts detected in the project config. "
                "Is the pose config imported?"
            )
        refs = {}
        for animal, (cb1, cb2) in self._ref_fields.items():
            bp1, bp2 = cb1.currentText(), cb2.currentText()
            if not bp1 or not bp2:
                raise ValueError(f"Missing body-part for {animal}.")
            if bp1 == bp2:
                raise ValueError(
                    f"{animal}: body-part 1 and 2 must differ "
                    f"(both set to {bp1})."
                )
            refs[animal] = (bp1, bp2)
        return {
            "config_path":      self.config_path,
            "location_criterion": float(self.location_crit.value()),
            "movement_criterion": float(self.movement_crit.value()),
            "aggregation":      self.aggregation_cb.currentText().lower(),
            "refs":             refs,
        }

    def target(self, *, config_path: str, location_criterion: float,
               movement_criterion: float, aggregation: str,
               refs: dict[str, tuple[str, str]]) -> None:
        # Write the "Outlier settings" section only. Audit A2 fix:
        # nothing else in the config is touched.
        cfg = configparser.ConfigParser(); cfg.read(config_path)
        sect = "Outlier settings"
        if cfg.has_section(sect):
            cfg.remove_section(sect)
        cfg.add_section(sect)
        cfg.set(sect, "location_criterion", f"{location_criterion:g}")
        cfg.set(sect, "movement_criterion", f"{movement_criterion:g}")
        cfg.set(sect, "aggregation_method", aggregation)
        for animal, (bp1, bp2) in refs.items():
            key1 = f"{animal}_location_bp_1".lower().replace(" ", "_")
            key2 = f"{animal}_location_bp_2".lower().replace(" ", "_")
            cfg.set(sect, key1, bp1)
            cfg.set(sect, key2, bp2)
        with open(config_path, "w") as f:
            cfg.write(f)


# --------------------------------------------------------------------------- #
# DropBodypartsForm
# --------------------------------------------------------------------------- #
class DropBodypartsForm(OperationForm):
    """Remove selected body-parts from pose CSVs project-wide.

    Useful when a late analytic decision rules out a tracked body-part
    (e.g. tail-tip too unreliable; drop it before feature extraction).
    """

    title = "Drop body-parts from pose data"
    description = ("Remove selected body-parts from every CSV in the "
                   "project's input directory. Irreversible unless you "
                   "keep originals.")

    def build(self) -> None:
        outer = QVBoxLayout()

        self.bp_list = QListWidget(self)
        self.bp_list.setSelectionMode(QListWidget.MultiSelection)
        self.bp_list.setMinimumHeight(160)
        self._populate_bps()
        outer.addWidget(QLabel(
            "<b>Select body-parts to drop (Ctrl-click for multi-select)</b>", self,
        ))
        outer.addWidget(self.bp_list)

        self.copy_originals = QCheckBox("Copy originals before overwriting", self)
        self.copy_originals.setChecked(True)
        outer.addWidget(self.copy_originals)

        self.body_layout.addLayout(outer)

    def _populate_bps(self) -> None:
        if not self.config_path:
            return
        try:
            animal_bps = _load_animal_bps(self.config_path)
        except Exception:
            return
        self.bp_list.clear()
        for animal, bps in animal_bps.items():
            for bp in bps:
                item = QListWidgetItem(f"{animal}: {bp}")
                item.setData(Qt.UserRole, (animal, bp))
                self.bp_list.addItem(item)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        selected = [it.data(Qt.UserRole) for it in self.bp_list.selectedItems()]
        if not selected:
            raise ValueError("No body-parts selected.")
        return {
            "config_path":    self.config_path,
            "to_drop":        selected,
            "copy_originals": bool(self.copy_originals.isChecked()),
        }

    def target(self, *, config_path: str,
               to_drop: list[tuple[str, str]],
               copy_originals: bool) -> None:
        # The legacy popup used a `KeyPointRemover` / `drop_bp_cords`
        # helper. In this fork that helper lives in
        # mufasa.data_processors.keypoint_dropper or
        # mufasa.utils.read_write depending on branch; try a small set.
        try:
            from mufasa.data_processors import keypoint_dropper as _kd
            _kd.KeyPointRemover(
                config_path=config_path,
                body_parts=to_drop,
                copy_originals=copy_originals,
            ).run()
        except ImportError:
            raise NotImplementedError(
                "Drop-body-parts backend (keypoint_dropper) is not present "
                "in this fork. Install from the legacy SimBA branch or "
                "patch with the copy-out / column-drop function."
            )


__all__ = [
    "SmoothingForm",
    "InterpolateForm",
    "OutlierSettingsForm",
    "DropBodypartsForm",
]
