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
* :class:`EgocentricAlignPopUp` → :class:`EgocentricAlignmentForm`
  (despite the docstring of the original popup hinting at interactive
  click-on-frame, in practice it's a settings form with body-part
  dropdowns — the same shape as the others. No interactive surface
  was ever in the Tk original; the placeholder claim was wrong.)

Design notes
------------

All five forms take ``config_path`` and rely on ``ConfigReader`` to
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
        # Confirmation log — the form is config-write-only (settings
        # are applied later by the corrector during feature extraction),
        # so without an explicit summary the user sees only "complete"
        # in the status bar and can't tell what was saved.
        from mufasa.utils.printing import stdout_success
        ref_summary = ", ".join(
            f"{animal}=({bp1},{bp2})" for animal, (bp1, bp2) in refs.items()
        )
        print(
            f"Outlier settings saved to {config_path}:\n"
            f"  location_criterion = {location_criterion:g}\n"
            f"  movement_criterion = {movement_criterion:g}\n"
            f"  aggregation_method = {aggregation}\n"
            f"  reference body-parts: {ref_summary}\n"
            "These will be applied during outlier correction "
            "(run as part of feature extraction)."
        )
        stdout_success(
            msg="Outlier correction settings saved to project_config.ini",
        )


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


# --------------------------------------------------------------------------- #
# EgocentricAlignmentForm
# --------------------------------------------------------------------------- #
def _load_flat_bps(config_path: str) -> list[str]:
    """Return the project's body-parts as a flat list, in the order
    they appear in ``project_bp_names.csv``. Reads the project's
    config the same lightweight way :func:`_load_animal_bps` does
    (no ConfigReader / numba pull-in)."""
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    project_path = cfg.get("General settings", "project_path", fallback=None)
    if not project_path:
        return []
    bp_file = (Path(project_path) / "logs" / "measures"
               / "pose_configs" / "bp_names" / "project_bp_names.csv")
    if not bp_file.is_file():
        return []
    text = bp_file.read_text().strip()
    if not text:
        return []
    # Tolerate both one-per-line and old-style comma-separated single row.
    if "," in text.splitlines()[0]:
        return [x.strip() for x in text.splitlines()[0].split(",") if x.strip()]
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# Standard 8 named colors used by the Tk popup. Mufasa's color_dict
# has ~140 entries which is overwhelming in a dropdown; this keeps the
# common cases without making the form unwieldy.
_FILL_COLORS = {
    "Black":  (0, 0, 0),
    "White":  (255, 255, 255),
    "Red":    (0, 0, 255),       # BGR — converted to RGB by backend
    "Green":  (0, 255, 0),
    "Blue":   (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Cyan":   (255, 255, 0),
    "Magenta":(255, 0, 255),
    "Gray":   (128, 128, 128),
}


class EgocentricAlignmentForm(OperationForm):
    """Rotate pose (and optionally video) so each frame is centered on
    a chosen body-part with a chosen direction-anchor pointing toward
    a chosen heading.

    Surfaces the same options as the Tk :class:`EgocentricAlignPopUp`,
    minus the (never actually present) "click-on-frame" interaction —
    that placeholder claim was wrong; the Tk popup has only dropdowns
    too. Backend: :class:`mufasa.data_processors.egocentric_aligner.EgocentricalAligner`.

    Inputs:
      * **Center anchor** — body-part placed at the rotation origin.
        Defaults to the closest match to "center" in your project's
        bp_names.
      * **Direction anchor** — body-part whose vector from center sets
        the rotation. Defaults to the closest match to "nose".
      * **Direction angle** — target heading angle in degrees (0=right,
        90=up, etc.). Default 0.
      * **Rotate videos** — if checked, the corresponding videos in
        ``project_folder/videos`` are also rotated frame-by-frame and
        written to the save directory. Slow but matches what the Tk
        popup did. If unchecked, only pose CSVs are aligned.
      * **Fill color** — color of borders introduced when video frames
        are rotated. Ignored when not rotating videos.
      * **CPU cores** — workers for the parallel rotation kernel.
      * **GPU** — uses CUDA path if available.

    Output goes to a sibling directory under ``project_folder``
    (default: ``rotated``). The form refuses to use the data or video
    dir as the save target.
    """

    title = "Egocentric alignment"
    description = (
        "Rotate pose (and optionally video) so each frame is centered "
        "on one body-part with another pointing toward a chosen "
        "heading. Useful before computing direction-relative features."
    )

    def build(self) -> None:
        bps = _load_flat_bps(self.config_path) if self.config_path else []
        if not bps:
            note = QLabel(
                "<b>No project loaded</b>, or "
                "<code>project_bp_names.csv</code> is missing/empty. "
                "Open a project to use this form."
            )
            note.setTextFormat(Qt.RichText)
            note.setWordWrap(True)
            self.body_layout.addWidget(note)
            self.run_btn.setEnabled(False)
            return

        # Default anchor body-parts: closest name match to "center" / "nose"
        def _closest(target: str, fallback: str) -> str:
            t = target.lower()
            for bp in bps:
                if bp.lower() == t:
                    return bp
            for bp in bps:
                if t in bp.lower():
                    return bp
            return fallback
        default_center = _closest("center", bps[0])
        default_dir = _closest("nose", bps[-1])

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.center_cb = QComboBox(self)
        self.center_cb.addItems(bps)
        self.center_cb.setCurrentText(default_center)
        form.addRow("Center anchor:", self.center_cb)

        self.direction_cb = QComboBox(self)
        self.direction_cb.addItems(bps)
        self.direction_cb.setCurrentText(default_dir)
        form.addRow("Direction anchor:", self.direction_cb)

        self.angle_spin = QSpinBox(self)
        self.angle_spin.setRange(0, 360)
        self.angle_spin.setValue(0)
        self.angle_spin.setSuffix(" °")
        form.addRow("Direction angle:", self.angle_spin)

        self.rotate_video_cb = QCheckBox(
            "Also rotate videos (slow; matches the Tk popup behaviour)", self,
        )
        self.rotate_video_cb.setChecked(True)
        form.addRow("", self.rotate_video_cb)

        self.fill_clr_cb = QComboBox(self)
        self.fill_clr_cb.addItems(list(_FILL_COLORS.keys()))
        self.fill_clr_cb.setCurrentText("Black")
        form.addRow("Border fill color:", self.fill_clr_cb)

        self.core_spin = QSpinBox(self)
        self.core_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.core_spin.setValue(max(1, (os.cpu_count() or 2) // 2))
        form.addRow("CPU cores:", self.core_spin)

        self.gpu_cb = QCheckBox("Use GPU (CUDA)", self)
        self.gpu_cb.setChecked(False)
        form.addRow("", self.gpu_cb)

        self.save_dir_edit = QLineEdit(self)
        self.save_dir_edit.setReadOnly(True)
        save_default = ""
        cfg = configparser.ConfigParser()
        cfg.read(self.config_path)
        project_path = cfg.get("General settings", "project_path",
                               fallback=None)
        if project_path:
            save_default = os.path.join(project_path, "rotated")
        self.save_dir_edit.setText(save_default)
        save_btn = QPushButton("Browse…", self)
        save_btn.clicked.connect(self._pick_save_dir)
        save_row = QWidget(self)
        sl = QGridLayout(save_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.save_dir_edit, 0, 0)
        sl.addWidget(save_btn, 0, 1)
        form.addRow("Save directory:", save_row)

        self.body_layout.addLayout(form)

    def _pick_save_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pick save directory for rotated output",
            self.save_dir_edit.text() or "",
        )
        if d:
            self.save_dir_edit.setText(d)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        save_dir = self.save_dir_edit.text().strip()
        if not save_dir:
            raise RuntimeError("Pick a save directory.")
        cfg = configparser.ConfigParser()
        cfg.read(self.config_path)
        project_path = cfg.get("General settings", "project_path",
                               fallback=None)
        if not project_path:
            raise RuntimeError("project_path missing from project_config.ini")
        # Source: outlier_corrected_movement_location/ if it exists
        # and contains files; otherwise input_csv/ (covers the common
        # case where outlier correction hasn't been run yet).
        outlier_dir = os.path.join(
            project_path, "csv", "outlier_corrected_movement_location",
        )
        input_dir = os.path.join(project_path, "csv", "input_csv")
        if os.path.isdir(outlier_dir) and any(
            f for f in os.listdir(outlier_dir) if not f.startswith(".")
        ):
            data_dir = outlier_dir
        else:
            data_dir = input_dir
        if not os.path.isdir(data_dir):
            raise RuntimeError(
                f"No data directory found at {input_dir}. "
                "Import pose data first."
            )
        videos_dir = os.path.join(project_path, "videos")
        if save_dir in (data_dir, videos_dir):
            raise RuntimeError(
                "Save directory must differ from the data and video "
                "directories."
            )
        os.makedirs(save_dir, exist_ok=True)

        return {
            "data_dir":         data_dir,
            "save_dir":         save_dir,
            "anchor_1":         self.center_cb.currentText(),
            "anchor_2":         self.direction_cb.currentText(),
            "direction":        int(self.angle_spin.value()),
            "rotate_video":     bool(self.rotate_video_cb.isChecked()),
            "videos_dir":       videos_dir,
            "fill_clr":         _FILL_COLORS[self.fill_clr_cb.currentText()],
            "core_cnt":         int(self.core_spin.value()),
            "gpu":              bool(self.gpu_cb.isChecked()),
        }

    def target(self, *, data_dir: str, save_dir: str, anchor_1: str,
               anchor_2: str, direction: int, rotate_video: bool,
               videos_dir: str, fill_clr: tuple, core_cnt: int,
               gpu: bool) -> None:
        from mufasa.data_processors.egocentric_aligner import (
            EgocentricalAligner,
        )
        # The backend requires either videos_dir OR anchor_location to
        # be non-None. Two paths:
        #   * rotate_video=True  → pass videos_dir, let the aligner
        #     auto-pick anchor location from data.
        #   * rotate_video=False → pass anchor_location=(250, 250)
        #     (backend default), skip videos_dir entirely so videos
        #     don't get rotated.
        if rotate_video:
            aligner = EgocentricalAligner(
                data_dir=data_dir,
                save_dir=save_dir,
                anchor_1=anchor_1,
                anchor_2=anchor_2,
                direction=direction,
                anchor_location=None,
                core_cnt=core_cnt,
                fill_clr=fill_clr,
                verbose=True,
                gpu=gpu,
                videos_dir=videos_dir,
            )
        else:
            aligner = EgocentricalAligner(
                data_dir=data_dir,
                save_dir=save_dir,
                anchor_1=anchor_1,
                anchor_2=anchor_2,
                direction=direction,
                anchor_location=(250, 250),
                core_cnt=core_cnt,
                fill_clr=fill_clr,
                verbose=True,
                gpu=gpu,
                videos_dir=None,
            )
        aligner.run()


__all__ = [
    "SmoothingForm",
    "InterpolateForm",
    "OutlierSettingsForm",
    "DropBodypartsForm",
    "EgocentricAlignmentForm",
]
