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
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox,
                               QSpinBox, QDoubleSpinBox, QVBoxLayout, QWidget,
                               QPushButton, QFileDialog)

from mufasa.ui_qt.workbench import OperationForm
from mufasa.project_layout import (
    import_model_into_project,
    mirror_model_to_global_cache,
    resolve_v1_project_root,
)


# --------------------------------------------------------------------------- #
# Patch 121h: standard location for fitted v2 models.
# --------------------------------------------------------------------------- #
def _default_model_dir() -> str:
    """User-level home for fitted Kalman v2 models.

    Lives at ``~/.config/mufasa/models/`` (XDG-style config
    path on Linux; same path on macOS via expanduser; mostly
    fine on Windows too via the userprofile expansion).
    Created on first access. Used as the default destination
    for saved models and the starting directory for the
    'load model' browse dialog.

    Centralizing the location means a model trained once is
    discoverable from any project on the same machine, and
    deleting the dir wipes all cached models in one place.
    """
    p = Path.home() / ".config" / "mufasa" / "models"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only home, network share quirks, etc. — silently
        # degrade; the form will still work, the user just has
        # to type a path manually.
        pass
    return str(p)


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

        # Patch 122b: data-source picker replaces the previous
        # hard-coded auto-detect of csv/outlier_corrected_*/ vs
        # csv/input_csv/. Lets the user pick raw / outlier-
        # corrected / Kalman-v2-smoothed / Savitzky-Golay / custom.
        # The default-marked source is the most-processed available
        # output (see _DEFAULT_PREFER_ORDER in input_source_picker).
        from mufasa.ui_qt.input_source_picker import InputSourcePicker
        self.source_picker = InputSourcePicker(
            self, config_path=self.config_path,
        )
        form.addRow("Input data source:", self.source_picker)

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
        # Patch 122b: input directory comes from the data-source
        # picker instead of being hard-coded. The picker raises
        # ValueError for "no choice / invalid path", which we
        # convert into the form's standard RuntimeError surface.
        try:
            data_dir = str(self.source_picker.selected_path())
        except ValueError as exc:
            raise RuntimeError(str(exc))
        if not os.path.isdir(data_dir):
            raise RuntimeError(
                f"Selected input directory does not exist: {data_dir}"
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


# --------------------------------------------------------------------------- #
# KalmanV2SmoothingForm — patch 121e wiring
# --------------------------------------------------------------------------- #
class KalmanV2SmoothingForm(OperationForm):
    """Kalman v2 (forward-filter / RTS smoother) on pose data.

    Two operating modes:

    * **Train new model** — fit NoiseParamsV2 via EM on a chosen
      training set (all input files by default, or a subset),
      smooth all input files with the fitted model, optionally
      save the model for later reuse.
    * **Load saved model** — load a previously-fitted model and
      smooth input files without running EM. Layout extensions
      (per-marker drift, orientation drift, const-accel) are
      restored from the saved model and not re-configurable.

    Recent feature flags exposed in train mode:

    * **per-marker drift** (patch 120) — fit per-marker bias to
      absorb tracker offset.
    * **per-segment orientation drift** (patch 121b) — absorb
      low-frequency angular residuals (body roll/pitch projected
      onto the image plane). Recommended starting point: ``body``.
    * **constant-acceleration dynamics** (patch 121d/e) —
      predictor extrapolates with curvature instead of straight-
      line velocity. Recommended if motion looks lagged: ``body,head``.

    The form invokes :func:`smooth_pose_v2` directly via Python
    API so errors surface cleanly in the workbench's progress
    dialog.
    """

    title = "Kalman v2 smoothing"
    description = (
        "Skeletal-model EM smoother (forward-filter + RTS). "
        "Slower than Savitzky-Golay but handles missing frames, "
        "tracker dropouts, and per-marker bias. Train a new "
        "model on your data or load a previously-fitted one."
    )

    # Pose-file extensions discovered in the input dir for the
    # training subset picker. Matches what
    # discover_pose_files / smooth_pose_v2 will pick up.
    _POSE_EXTS = (".parquet", ".csv", ".tsv")

    def build(self) -> None:
        from PySide6.QtWidgets import (
            QHBoxLayout, QRadioButton, QButtonGroup,
            QAbstractItemView,
        )
        outer_form = QFormLayout()
        outer_form.setLabelAlignment(Qt.AlignRight)

        # ---- Common: Input dir ----
        in_row = QHBoxLayout()
        self.input_dir = QLineEdit(self)
        self.input_dir.setPlaceholderText(
            "Pose data directory (parquet preferred, CSV fallback)"
        )
        self.input_dir.textChanged.connect(self._refresh_file_list)
        in_row.addWidget(self.input_dir)
        in_browse = QPushButton("Browse…", self)
        in_browse.clicked.connect(self._browse_input)
        in_row.addWidget(in_browse)
        outer_form.addRow("Input dir:", in_row)

        # ---- Common: Output dir ----
        out_row = QHBoxLayout()
        self.output_dir = QLineEdit(self)
        self.output_dir.setPlaceholderText(
            "Where to write *_smoothed_v2.parquet files"
        )
        out_row.addWidget(self.output_dir)
        out_browse = QPushButton("Browse…", self)
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(out_browse)
        outer_form.addRow("Output dir:", out_row)

        # ---- Common: FPS ----
        self.fps = QDoubleSpinBox(self)
        self.fps.setRange(1.0, 1000.0)
        self.fps.setValue(30.0)
        self.fps.setSuffix(" Hz")
        outer_form.addRow("Frame rate:", self.fps)

        # ---- Common: Likelihood threshold ----
        self.lik_thr = QDoubleSpinBox(self)
        self.lik_thr.setRange(0.0, 1.0)
        self.lik_thr.setSingleStep(0.05)
        self.lik_thr.setDecimals(2)
        self.lik_thr.setValue(0.5)
        outer_form.addRow("Likelihood threshold:", self.lik_thr)

        # ---- Common: Workers ----
        try:
            import multiprocessing as _mp
            cpu_default = max(1, _mp.cpu_count() - 1)
        except Exception:
            cpu_default = 4
        self.workers = QSpinBox(self)
        self.workers.setRange(1, 64)
        self.workers.setValue(min(cpu_default, 12))
        outer_form.addRow("Worker processes:", self.workers)

        # ---- Mode selector ----
        mode_group = QGroupBox("Mode", self)
        mode_layout = QHBoxLayout(mode_group)
        self.mode_train = QRadioButton(
            "Train new model", self,
        )
        self.mode_load = QRadioButton(
            "Load saved model", self,
        )
        self.mode_train.setChecked(True)
        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.addButton(self.mode_train)
        self.mode_btn_group.addButton(self.mode_load)
        mode_layout.addWidget(self.mode_train)
        mode_layout.addWidget(self.mode_load)
        mode_layout.addStretch()
        self.mode_train.toggled.connect(self._update_mode_visibility)
        outer_form.addRow(mode_group)

        # ---- TRAIN-mode group ----
        self.train_group = QGroupBox("Training", self)
        train_form = QFormLayout(self.train_group)
        train_form.setLabelAlignment(Qt.AlignRight)

        # Training file subset
        self.train_subset = QCheckBox(
            "Train on a subset of input files (default: use all)",
            self,
        )
        self.train_subset.setChecked(False)
        self.train_subset.toggled.connect(
            self._update_subset_visibility
        )
        train_form.addRow("", self.train_subset)

        # File list (multi-select, initially hidden)
        self.train_file_list = QListWidget(self)
        self.train_file_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.train_file_list.setMaximumHeight(150)
        self.train_file_list.setVisible(False)
        train_form.addRow("Files for EM:", self.train_file_list)

        # Refresh / select-all / clear buttons row
        self.train_btn_row = QWidget(self)
        btn_layout = QHBoxLayout(self.train_btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        refresh_btn = QPushButton("Refresh", self)
        refresh_btn.clicked.connect(self._refresh_file_list)
        select_all_btn = QPushButton("Select all", self)
        select_all_btn.clicked.connect(
            self.train_file_list.selectAll
        )
        clear_btn = QPushButton("Clear", self)
        clear_btn.clicked.connect(
            self.train_file_list.clearSelection
        )
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        self.train_btn_row.setVisible(False)
        train_form.addRow("", self.train_btn_row)

        # EM parameters
        self.em_iter = QSpinBox(self)
        self.em_iter.setRange(1, 100)
        self.em_iter.setValue(20)
        train_form.addRow("EM iterations:", self.em_iter)

        self.em_tol = QDoubleSpinBox(self)
        self.em_tol.setRange(1e-6, 1.0)
        self.em_tol.setDecimals(6)
        self.em_tol.setSingleStep(0.001)
        self.em_tol.setValue(0.001)
        train_form.addRow(
            "EM tolerance (max Δ/x):", self.em_tol,
        )

        self.em_damping = QDoubleSpinBox(self)
        self.em_damping.setRange(0.0, 0.95)
        self.em_damping.setDecimals(2)
        self.em_damping.setSingleStep(0.05)
        self.em_damping.setValue(0.0)
        train_form.addRow(
            "EM damping (0 = full M-step):", self.em_damping,
        )

        self.em_aggregation = QComboBox(self)
        self.em_aggregation.addItems(["pooled", "per-session"])
        train_form.addRow(
            "M-step aggregation:", self.em_aggregation,
        )

        self.warm_start_sigma = QCheckBox(
            "Run warm-start σ pass (recommended)",
            self,
        )
        self.warm_start_sigma.setChecked(True)
        train_form.addRow("", self.warm_start_sigma)

        self.use_perspective = QCheckBox(
            "Fit perspective model",
            self,
        )
        self.use_perspective.setChecked(True)
        train_form.addRow("", self.use_perspective)

        self.use_validation = QCheckBox(
            "Run validation hook each EM iteration",
            self,
        )
        self.use_validation.setChecked(True)
        train_form.addRow("", self.use_validation)

        # Save-model checkbox + path
        self.save_model_chk = QCheckBox(
            "Save fitted model to:", self,
        )
        self.save_model_chk.setChecked(True)
        self.save_model_chk.toggled.connect(
            self._update_save_model_visibility
        )
        train_form.addRow("", self.save_model_chk)

        save_row = QHBoxLayout()
        self.save_model_path = QLineEdit(self)
        self.save_model_path.setPlaceholderText(
            "~/.config/mufasa/models/v2_model.npz"
        )
        save_row.addWidget(self.save_model_path)
        save_browse = QPushButton("Browse…", self)
        save_browse.clicked.connect(self._browse_save_model)
        save_row.addWidget(save_browse)
        self.save_model_path_label = QLabel("Path:", self)
        train_form.addRow(
            self.save_model_path_label, save_row,
        )

        # Layout extensions sub-group (training mode only)
        ext_group = QGroupBox(
            "Model extensions (defines a new model)", self,
        )
        ext_form = QFormLayout(ext_group)
        ext_form.setLabelAlignment(Qt.AlignRight)
        self.with_drift = QCheckBox(
            "Per-marker drift (patch 120 — absorbs per-marker "
            "tracker offset)",
            self,
        )
        self.with_drift.setChecked(True)
        ext_form.addRow("", self.with_drift)
        self.orient_drift = QLineEdit(self)
        self.orient_drift.setPlaceholderText(
            "comma-separated, e.g. body,head"
        )
        ext_form.addRow(
            "Orientation drift segments:", self.orient_drift,
        )
        self.const_accel = QLineEdit(self)
        self.const_accel.setPlaceholderText(
            "comma-separated, e.g. body,head"
        )
        ext_form.addRow(
            "Constant-accel segments:", self.const_accel,
        )
        train_form.addRow(ext_group)

        outer_form.addRow(self.train_group)

        # ---- LOAD-mode group ----
        self.load_group = QGroupBox("Load saved model", self)
        load_form = QFormLayout(self.load_group)
        load_form.setLabelAlignment(Qt.AlignRight)

        load_row = QHBoxLayout()
        self.load_model_path = QLineEdit(self)
        self.load_model_path.setPlaceholderText(
            "Path to v2_model.npz from a previous run"
        )
        load_row.addWidget(self.load_model_path)
        load_browse = QPushButton("Browse…", self)
        load_browse.clicked.connect(self._browse_load_model)
        load_row.addWidget(load_browse)
        load_form.addRow("Model file:", load_row)

        load_form.addRow("", QLabel(
            "Layout extensions (per-marker drift, orientation\n"
            "drift, const-accel) are restored from the saved\n"
            "model and not re-configurable here.",
            self,
        ))

        outer_form.addRow(self.load_group)

        self.body_layout.addLayout(outer_form)

        # Initial visibility
        self._update_mode_visibility(self.mode_train.isChecked())
        self._update_save_model_visibility(
            self.save_model_chk.isChecked()
        )

        # Patch 122d: v1-aware path defaults. Generate a fresh
        # run_id at build time so the displayed output_dir matches
        # exactly what the form will write on Run. Closing and
        # reopening the form gives a new run_id; rebuilding the
        # form doesn't overwrite previous runs.
        #
        # In a v1 project:
        #   input_dir       → <root>/sources/pose/
        #   output_dir      → <root>/derived/smoothed/kalman_v2/<run_id>/
        #   save_model_path → <output_dir>/model.npz   (co-located
        #                     with the smoothed pose data)
        #
        # The save_model_path default is set on save_model_path
        # itself rather than the placeholder so dual-save's
        # `import_model_into_project` lands the model under
        # `<project>/models/<run-id-named>/` in addition to the
        # in-run-dir copy.
        #
        # Legacy fallback for projects still on project_config.ini.
        self._v1_run_id: Optional[str] = None
        if self.config_path:
            v1_root = resolve_v1_project_root(self.config_path)
            if v1_root is not None:
                from mufasa.project_layout import (
                    ProjectPaths, SmoothingFlavors, generate_run_id,
                )
                paths = ProjectPaths(v1_root)
                self._v1_run_id = generate_run_id()
                default_in = paths.sources_pose
                run_dir = paths.smoothed_run_dir(
                    SmoothingFlavors.KALMAN_V2,
                    run_id=self._v1_run_id,
                )
                self.input_dir.setText(str(default_in))
                self.output_dir.setText(str(run_dir))
                self.save_model_path.setText(
                    str(run_dir / "model.npz"),
                )
            else:
                # Legacy project — fall back to the SimBA-style
                # csv/ tree. Same behaviour as before patch 122d.
                try:
                    cfg = configparser.ConfigParser()
                    cfg.read(self.config_path)
                    project_path = cfg.get(
                        "General settings", "project_path",
                        fallback=None,
                    )
                    if project_path:
                        default_in = os.path.join(
                            project_path, "csv", "input_csv",
                        )
                        if os.path.isdir(default_in):
                            self.input_dir.setText(default_in)
                        default_out = os.path.join(
                            project_path, "csv", "smoothed_v2",
                        )
                        self.output_dir.setText(default_out)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Visibility / refresh helpers
    # ------------------------------------------------------------------ #
    def _update_mode_visibility(self, train_mode: bool) -> None:
        """Show training group when 'Train new model' is
        selected; load group when 'Load saved model' is.
        """
        self.train_group.setVisible(train_mode)
        self.load_group.setVisible(not train_mode)

    def _update_subset_visibility(self, on: bool) -> None:
        """Show/hide the file-list picker based on the 'train
        on subset' checkbox.
        """
        self.train_file_list.setVisible(on)
        self.train_btn_row.setVisible(on)
        if on:
            # Trigger an initial scan when the user opts in
            self._refresh_file_list()

    def _update_save_model_visibility(self, on: bool) -> None:
        """Show/hide the save-model path field."""
        self.save_model_path.setVisible(on)
        self.save_model_path_label.setVisible(on)

    def _refresh_file_list(self) -> None:
        """Populate the training file list from the input dir.
        Called when input_dir changes or user clicks Refresh.
        """
        self.train_file_list.clear()
        in_dir = self.input_dir.text().strip()
        if not in_dir or not os.path.isdir(in_dir):
            return
        try:
            entries = sorted(os.listdir(in_dir))
        except OSError:
            return
        for name in entries:
            full = os.path.join(in_dir, name)
            if not os.path.isfile(full):
                continue
            if not name.lower().endswith(self._POSE_EXTS):
                continue
            item = QListWidgetItem(name, self.train_file_list)
            item.setData(Qt.UserRole, full)

    # ------------------------------------------------------------------ #
    # Browse callbacks
    # ------------------------------------------------------------------ #
    def _browse_input(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Choose pose data directory",
            self.input_dir.text() or os.getcwd(),
        )
        if d:
            self.input_dir.setText(d)

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Choose output directory",
            self.output_dir.text() or os.getcwd(),
        )
        if d:
            self.output_dir.setText(d)

    def _browse_save_model(self) -> None:
        # Patch 121h: default to ~/.config/mufasa/models/
        suggested = (
            self.save_model_path.text().strip()
            or os.path.join(
                _default_model_dir(), "v2_model.npz",
            )
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save fitted model as", suggested,
            "NumPy archive (*.npz);;All files (*)",
        )
        if path:
            self.save_model_path.setText(path)

    def _browse_load_model(self) -> None:
        # Patch 121h: default browse start to the model dir
        # so previously-saved models surface immediately.
        start = (
            self.load_model_path.text().strip()
            or _default_model_dir()
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose model file", start,
            "NumPy archive (*.npz);;All files (*)",
        )
        if path:
            self.load_model_path.setText(path)

    # ------------------------------------------------------------------ #
    # collect_args / target
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        in_dir = self.input_dir.text().strip()
        out_dir = self.output_dir.text().strip()
        if not in_dir:
            raise RuntimeError("Input directory is required.")
        if not os.path.isdir(in_dir):
            raise RuntimeError(
                f"Input directory does not exist: {in_dir}"
            )
        if not out_dir:
            raise RuntimeError(
                "Output directory is required (will be "
                "created if missing)."
            )

        train_mode = self.mode_train.isChecked()

        # Mode-specific validation & arg packing
        if train_mode:
            # Training file subset
            training_files: Optional[list[str]] = None
            if self.train_subset.isChecked():
                selected = self.train_file_list.selectedItems()
                if not selected:
                    raise RuntimeError(
                        "'Train on a subset' is checked but "
                        "no files are selected. Either pick "
                        "files in the list, or uncheck the "
                        "subset option to train on all files."
                    )
                training_files = [
                    item.data(Qt.UserRole) for item in selected
                ]

            # Save-model path
            save_path: Optional[str] = None
            if self.save_model_chk.isChecked():
                save_path = self.save_model_path.text().strip()
                if not save_path:
                    # Patch 121h: default to ~/.config/mufasa/models/
                    save_path = os.path.join(
                        _default_model_dir(), "v2_model.npz",
                    )

            # Layout extensions
            def _parse_csv(s: str) -> list[str]:
                return [
                    p.strip() for p in s.split(",") if p.strip()
                ]

            return {
                "mode":                 "train",
                "input_dir":            in_dir,
                "output_dir":           out_dir,
                "fps":                  float(self.fps.value()),
                "likelihood_threshold": float(self.lik_thr.value()),
                "n_workers":            int(self.workers.value()),
                "training_files":       training_files,
                "em_max_iter":          int(self.em_iter.value()),
                "em_tol":               float(self.em_tol.value()),
                "em_damping":           float(self.em_damping.value()),
                "em_aggregation":
                    self.em_aggregation.currentText(),
                "warm_start_sigma":
                    bool(self.warm_start_sigma.isChecked()),
                "use_perspective":
                    bool(self.use_perspective.isChecked()),
                "use_validation":
                    bool(self.use_validation.isChecked()),
                "save_model_path":      save_path,
                "with_drift":
                    bool(self.with_drift.isChecked()),
                "orient_drift_segments":
                    _parse_csv(self.orient_drift.text()),
                "const_accel_segments":
                    _parse_csv(self.const_accel.text()),
            }

        # Load mode
        load_path = self.load_model_path.text().strip()
        if not load_path:
            raise RuntimeError(
                "Model file is required when loading a saved "
                "model."
            )
        if not os.path.isfile(load_path):
            raise RuntimeError(
                f"Model file does not exist: {load_path}"
            )
        return {
            "mode":                 "load",
            "input_dir":            in_dir,
            "output_dir":           out_dir,
            "fps":                  float(self.fps.value()),
            "likelihood_threshold": float(self.lik_thr.value()),
            "n_workers":            int(self.workers.value()),
            "load_model_path":      load_path,
        }

    def target(self, *, mode: str, **kwargs) -> None:
        # Late import — keeps GUI startup snappy and avoids
        # forcing the entire numerics stack on workbench users
        # who never run smoothing.
        import dataclasses as _dc
        from mufasa.data_processors.kalman_pose_smoother_v2 import (
            smooth_pose_v2, standard_rat_layout,
        )

        # Patch 122b: dual-save provenance. Any model that crosses
        # an organizational boundary (filesystem ↔ project,
        # project ↔ project) gets copied so the destination has
        # its own copy. Two helpers used at the end of training
        # and at the start of loading.
        v1_root = resolve_v1_project_root(self.config_path)
        # Patch 122d: the build-time-allocated run_id is reused
        # here for run.toml provenance so the dir on disk matches
        # the toml's run_id field. None on legacy projects.
        v1_run_id = getattr(self, "_v1_run_id", None)

        def _post_train_dual_save(saved_path: Optional[str]) -> None:
            """Mirror a freshly-saved model to the global cache and
            (if a v1 project is reachable) into the project's
            ``models/`` store.

            Both copies are overwrite-on-collision for training output
            — the user just produced this model, it wins. Load-time
            imports use the default no-overwrite behavior so trained
            models can't be silently clobbered by a stale load.
            """
            if not saved_path or not os.path.isfile(saved_path):
                return
            src = Path(saved_path)
            try:
                cache_path = mirror_model_to_global_cache(src)
                if cache_path is not None and cache_path != src.resolve():
                    print(
                        f"[dual-save] mirrored model to global cache: "
                        f"{cache_path}"
                    )
            except OSError as exc:
                print(
                    f"[dual-save] WARNING: could not mirror to global "
                    f"cache ({exc}); the saved file at {src} is still "
                    f"intact."
                )
            if v1_root is not None:
                try:
                    in_proj = import_model_into_project(
                        src, v1_root, overwrite=True,
                    )
                    if in_proj.resolve() != src.resolve():
                        print(
                            f"[dual-save] imported model into project: "
                            f"{in_proj}"
                        )
                except (OSError, ValueError) as exc:
                    print(
                        f"[dual-save] WARNING: could not import to "
                        f"project models/ ({exc}); the saved file at "
                        f"{src} is still intact."
                    )

        if mode == "train":
            # Build the layout with the requested feature flags.
            layout = standard_rat_layout()
            replacements: dict = {}
            if kwargs["with_drift"]:
                replacements["with_drift"] = True
            if kwargs["orient_drift_segments"]:
                replacements["orientation_drift_segments"] = (
                    kwargs["orient_drift_segments"]
                )
            if kwargs["const_accel_segments"]:
                replacements["const_accel_segments"] = (
                    kwargs["const_accel_segments"]
                )
            if replacements:
                layout = _dc.replace(layout, **replacements)

            # Two-pass workflow when training on a subset:
            # (1) fit on training_files with save_model
            # (2) load that model, smooth all input files
            # When no subset, single call does both EM and smoothing.
            training_files = kwargs.get("training_files")
            save_path = kwargs.get("save_model_path")

            if training_files:
                # Pass 1: fit on subset, write model
                if not save_path:
                    # Patch 121h: two-pass workflow needs a path
                    # for the intermediate model. Default to the
                    # standard model dir.
                    save_path = os.path.join(
                        _default_model_dir(), "v2_model.npz",
                    )
                smooth_pose_v2(
                    pose_input=training_files,
                    output_dir=None,  # don't write smoothed output
                    layout=layout,
                    fps=kwargs["fps"],
                    likelihood_threshold=kwargs["likelihood_threshold"],
                    em_max_iter=kwargs["em_max_iter"],
                    em_tol=kwargs["em_tol"],
                    em_damping=kwargs["em_damping"],
                    em_aggregation=kwargs["em_aggregation"],
                    enable_warm_start_sigma=kwargs["warm_start_sigma"],
                    enable_perspective=kwargs["use_perspective"],
                    enable_validation=kwargs["use_validation"],
                    n_workers=kwargs["n_workers"],
                    save_model=save_path,
                    verbose=True,
                )
                # Patch 122b: mirror freshly-trained model.
                _post_train_dual_save(save_path)
                # Pass 2: load model, smooth all input
                smooth_pose_v2(
                    pose_input=kwargs["input_dir"],
                    output_dir=kwargs["output_dir"],
                    fps=kwargs["fps"],
                    likelihood_threshold=kwargs["likelihood_threshold"],
                    n_workers=kwargs["n_workers"],
                    load_model=save_path,
                    verbose=True,
                )
            else:
                # Single-pass: train + smooth on all input
                smooth_pose_v2(
                    pose_input=kwargs["input_dir"],
                    output_dir=kwargs["output_dir"],
                    layout=layout,
                    fps=kwargs["fps"],
                    likelihood_threshold=kwargs["likelihood_threshold"],
                    em_max_iter=kwargs["em_max_iter"],
                    em_tol=kwargs["em_tol"],
                    em_damping=kwargs["em_damping"],
                    em_aggregation=kwargs["em_aggregation"],
                    enable_warm_start_sigma=kwargs["warm_start_sigma"],
                    enable_perspective=kwargs["use_perspective"],
                    enable_validation=kwargs["use_validation"],
                    n_workers=kwargs["n_workers"],
                    save_model=save_path,
                    verbose=True,
                )
                # Patch 122b: mirror freshly-trained model.
                _post_train_dual_save(save_path)

        elif mode == "load":
            # Patch 122b: if a v1 project is reachable and the
            # model came from outside its models/ store, import a
            # copy so future "what model produced this run?"
            # questions are answerable from the project alone.
            # Substitute the in-project path as the actual load
            # source.
            load_path = kwargs["load_model_path"]
            if v1_root is not None:
                load_abs = Path(load_path).resolve()
                models_root = (v1_root / "models").resolve()
                try:
                    is_inside_project = (
                        models_root in load_abs.parents
                        or load_abs == models_root
                    )
                except Exception:
                    is_inside_project = False
                if not is_inside_project:
                    try:
                        in_proj = import_model_into_project(
                            Path(load_path), v1_root,
                        )
                        if in_proj.resolve() != load_abs:
                            print(
                                f"[dual-save] imported loaded model "
                                f"into project: {in_proj}"
                            )
                        load_path = str(in_proj)
                    except FileExistsError as exc:
                        # A different-content model with the same
                        # name already lives in the project. Surface
                        # loudly rather than silently using either.
                        raise RuntimeError(
                            f"Cannot import model: {exc}. "
                            "Either remove the existing model from "
                            "this project's models/ folder, or "
                            "rename the model file you are loading."
                        )
                    except (OSError, ValueError) as exc:
                        # Soft failure — we still want the smoother
                        # to run, just without the project copy.
                        print(
                            f"[dual-save] WARNING: could not import "
                            f"loaded model into project ({exc}); "
                            f"using original path {load_path}."
                        )

            # Load mode: skip EM, just smooth.
            smooth_pose_v2(
                pose_input=kwargs["input_dir"],
                output_dir=kwargs["output_dir"],
                fps=kwargs["fps"],
                likelihood_threshold=kwargs["likelihood_threshold"],
                n_workers=kwargs["n_workers"],
                load_model=load_path,
                verbose=True,
            )
        else:
            raise RuntimeError(f"Unknown mode: {mode!r}")

        # Patch 122d: write run.toml provenance for v1 projects.
        # Fires for both train and load modes after smoothing
        # finishes successfully. Soft-fails (logged but not
        # raised) so a provenance hiccup doesn't invalidate
        # results that are already on disk.
        if v1_root is not None and v1_run_id is not None:
            output_dir = kwargs.get("output_dir")
            if output_dir and os.path.isdir(output_dir):
                try:
                    from mufasa.project_layout import (
                        RUN_PROVENANCE_FILENAME, write_run_toml,
                    )
                    # Trim kwargs to JSON-friendly scalars for
                    # the params block; lists pass through, the
                    # rest stringifies.
                    safe_params: dict = {}
                    for k, v in kwargs.items():
                        if isinstance(
                            v, (str, int, float, bool, list),
                        ) or v is None:
                            safe_params[k] = v
                        else:
                            safe_params[k] = repr(v)
                    safe_params["mode"] = mode
                    write_run_toml(
                        Path(output_dir) / RUN_PROVENANCE_FILENAME,
                        stage="smoothed.kalman_v2",
                        run_id=v1_run_id,
                        params=safe_params,
                    )
                    print(
                        f"[v1] wrote run.toml: "
                        f"{Path(output_dir) / RUN_PROVENANCE_FILENAME}"
                    )
                except Exception as exc:
                    print(
                        f"[v1] WARNING: could not write run.toml "
                        f"({exc}); smoothed output at {output_dir} "
                        f"is intact."
                    )


# --------------------------------------------------------------------------- #
# RunOutlierCorrectionForm — patch 122c
# --------------------------------------------------------------------------- #
class RunOutlierCorrectionForm(OperationForm):
    """Run the SimBA outlier-correction pipeline on the chosen pose data.

    SimBA's outlier correction is two stages:

    * **Movement correction** —
      :class:`mufasa.outlier_tools.outlier_corrector_movement.OutlierCorrecterMovement`
      walks frame-to-frame distances per reference body-part and
      replaces points that jumped further than the configured
      criterion with the previous (in-bounds) value.
    * **Location correction** —
      :class:`mufasa.outlier_tools.outlier_corrector_location.OutlierCorrecterLocation`
      then walks within-frame distances between body-parts and
      replaces points whose distance to their reference body-part
      exceeded the criterion.

    The legacy SimBA workflow always runs both in sequence
    (movement → location) and writes the final result to
    ``csv/outlier_corrected_movement_location/``. The two stages
    are exposed as separate checkboxes here so users can disable
    one if their criteria don't apply to it (e.g. movement
    correction is unhelpful for high-confidence DLC + Kalman
    smoothed data, but a wide location criterion still catches
    swapped body-parts).

    The thresholds and reference body-parts themselves are
    configured in the **Outlier correction settings** form
    (which writes to ``project_config.ini``). This form just
    runs the backends; it doesn't surface the criteria.
    """

    title = "Run outlier correction"
    description = (
        "Run movement + location outlier correction on the chosen "
        "pose data. Thresholds and reference body-parts are "
        "configured in 'Outlier correction settings' (under "
        "Advanced / legacy)."
    )

    def build(self) -> None:
        from mufasa.ui_qt.input_source_picker import (
            InputSourcePicker, SourceKinds,
        )
        # Outlier correction usually runs before smoothing, so the
        # raw pose is the preferred input. Reorder the picker's
        # default-preference accordingly. (Users who explicitly
        # outlier-correct smoothed data can still pick a smoother
        # output from the dropdown.)
        prefer_raw = (
            SourceKinds.RAW,
            SourceKinds.OUTLIER_CORRECTED,
            SourceKinds.SMOOTHED_KALMAN_V2,
            SourceKinds.SMOOTHED_SAVITZKY,
        )

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.source_picker = InputSourcePicker(
            self,
            config_path=self.config_path,
            prefer_order=prefer_raw,
        )
        form.addRow("Input data source:", self.source_picker)

        # Save directory. Pre-fill with the legacy outlier-
        # corrected_movement_location dir for legacy projects; for
        # v1 projects, leave blank and let the form compute a
        # fresh derived/outlier_corrected/<run_id>/ at run time
        # (so re-running doesn't trample the prior output).
        save_row = QHBoxLayout()
        self.save_dir_edit = QLineEdit(self)
        save_default = ""
        if self.config_path:
            try:
                cfg = configparser.ConfigParser()
                cfg.read(self.config_path)
                project_path = cfg.get(
                    "General settings", "project_path", fallback=None,
                )
                if project_path:
                    save_default = os.path.join(
                        project_path, "csv",
                        "outlier_corrected_movement_location",
                    )
            except (configparser.Error, OSError):
                pass
        self.save_dir_edit.setText(save_default)
        self.save_dir_edit.setPlaceholderText(
            "Leave blank to auto-generate "
            "derived/outlier_corrected/<run_id>/ (v1 projects)"
        )
        save_row.addWidget(self.save_dir_edit)
        save_browse = QPushButton("Browse…", self)
        save_browse.clicked.connect(self._browse_save_dir)
        save_row.addWidget(save_browse)
        form.addRow("Save directory:", save_row)

        self.do_movement = QCheckBox(
            "Apply movement correction (frame-to-frame jumps)", self,
        )
        self.do_movement.setChecked(True)
        form.addRow("", self.do_movement)

        self.do_location = QCheckBox(
            "Apply location correction (within-frame distances)", self,
        )
        self.do_location.setChecked(True)
        form.addRow("", self.do_location)

        self.body_layout.addLayout(form)

    def _browse_save_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Choose output directory",
            self.save_dir_edit.text() or os.getcwd(),
        )
        if d:
            self.save_dir_edit.setText(d)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        if not (self.do_movement.isChecked()
                or self.do_location.isChecked()):
            raise RuntimeError(
                "Enable at least one of movement / location "
                "correction (or use 'Skip outlier correction' to "
                "bypass the stage entirely)."
            )
        try:
            data_dir = str(self.source_picker.selected_path())
        except ValueError as exc:
            raise RuntimeError(str(exc))
        save_dir = self.save_dir_edit.text().strip()
        # For v1 projects we allow blank save_dir — the target
        # method will compute a fresh run directory. For legacy
        # projects, the field was prefilled so it'll always be set.
        # If the user blanked it for a legacy project, fall back
        # to the prefilled default at run time.
        return {
            "config_path":  self.config_path,
            "data_dir":     data_dir,
            "save_dir":     save_dir or None,
            "do_movement":  bool(self.do_movement.isChecked()),
            "do_location":  bool(self.do_location.isChecked()),
        }

    def target(self, *, config_path: str, data_dir: str,
               save_dir: Optional[str], do_movement: bool,
               do_location: bool) -> None:
        from mufasa.outlier_tools.outlier_corrector_movement import (
            OutlierCorrecterMovement,
        )
        from mufasa.outlier_tools.outlier_corrector_location import (
            OutlierCorrecterLocation,
        )

        # Resolve save_dir. For v1 projects with no explicit
        # target, allocate a fresh derived/outlier_corrected/<run>/.
        v1_root = resolve_v1_project_root(config_path)
        run_id: Optional[str] = None
        if save_dir is None:
            if v1_root is not None:
                from mufasa.project_layout import (
                    ProjectPaths, Stages, generate_run_id,
                )
                run_id = generate_run_id()
                paths = ProjectPaths(v1_root)
                save_dir = str(paths.stage_run_dir(
                    Stages.OUTLIER_CORRECTED, run_id=run_id,
                ))
            else:
                # Legacy with no save_dir — fall back to the
                # canonical SimBA destination so downstream stages
                # find the output.
                cfg = configparser.ConfigParser()
                cfg.read(config_path)
                project_path = cfg.get(
                    "General settings", "project_path", fallback="",
                )
                if not project_path:
                    raise RuntimeError(
                        "No save directory specified and "
                        "project_path missing from "
                        "project_config.ini — cannot infer a "
                        "default output location."
                    )
                save_dir = os.path.join(
                    project_path, "csv",
                    "outlier_corrected_movement_location",
                )
        os.makedirs(save_dir, exist_ok=True)

        # Chain the two stages. When both are enabled, movement
        # writes to a sibling intermediate dir and location reads
        # from there. When only one is enabled, it writes directly
        # to save_dir.
        if do_movement and do_location:
            movement_intermediate = os.path.join(
                save_dir, "_movement_intermediate",
            )
            os.makedirs(movement_intermediate, exist_ok=True)
            print(
                f"[outlier] movement correction: {data_dir} → "
                f"{movement_intermediate}"
            )
            OutlierCorrecterMovement(
                config_path=config_path,
                data_dir=data_dir,
                save_dir=movement_intermediate,
            ).run()
            print(
                f"[outlier] location correction: "
                f"{movement_intermediate} → {save_dir}"
            )
            OutlierCorrecterLocation(
                config_path=config_path,
                data_dir=movement_intermediate,
                save_dir=save_dir,
            ).run()
        elif do_movement:
            print(
                f"[outlier] movement correction only: "
                f"{data_dir} → {save_dir}"
            )
            OutlierCorrecterMovement(
                config_path=config_path,
                data_dir=data_dir,
                save_dir=save_dir,
            ).run()
        elif do_location:
            print(
                f"[outlier] location correction only: "
                f"{data_dir} → {save_dir}"
            )
            OutlierCorrecterLocation(
                config_path=config_path,
                data_dir=data_dir,
                save_dir=save_dir,
            ).run()

        # Write run.toml provenance if we're in a v1 project.
        if v1_root is not None and run_id is not None:
            try:
                from mufasa.project_layout import (
                    RUN_PROVENANCE_FILENAME, write_run_toml,
                )
                write_run_toml(
                    Path(save_dir) / RUN_PROVENANCE_FILENAME,
                    stage="outlier_corrected",
                    run_id=run_id,
                    params={
                        "data_dir":    data_dir,
                        "do_movement": do_movement,
                        "do_location": do_location,
                    },
                )
                print(
                    f"[outlier] wrote run.toml: "
                    f"{Path(save_dir) / RUN_PROVENANCE_FILENAME}"
                )
            except Exception as exc:
                # Provenance is a nice-to-have; don't fail the run.
                print(
                    f"[outlier] WARNING: could not write run.toml "
                    f"({exc}); output at {save_dir} is intact."
                )


# --------------------------------------------------------------------------- #
# SkipOutlierCorrectionForm — patch 122c
# --------------------------------------------------------------------------- #
class SkipOutlierCorrectionForm(OperationForm):
    """Bypass outlier correction by copying the raw pose into the
    outlier-corrected destination unchanged.

    Useful when:

    * pose data is already clean (e.g. hand-curated DLC or
      Kalman-v2-smoothed output that's already handling outliers
      via likelihood weighting),
    * the SimBA outlier-correction criteria don't translate well
      to the species or arena being studied,
    * the downstream pipeline assumes the
      ``outlier_corrected_movement_location/`` directory exists
      and is non-empty.

    Wraps
    :class:`mufasa.outlier_tools.skip_outlier_correction.OutlierCorrectionSkipper`,
    which reads from ``csv/input_csv/`` and writes to
    ``csv/outlier_corrected_movement_location/`` while
    standardizing pose-data headers. The form has no fields —
    behavior is fixed by the backend.
    """

    title = "Skip outlier correction"
    description = (
        "Copy raw pose data into the outlier-corrected directory "
        "unchanged. Use when your pose is already clean (e.g. "
        "hand-curated or Kalman-v2-smoothed)."
    )

    def build(self) -> None:
        note = QLabel(
            "<i>No options — this stage just standardizes pose-data "
            "headers and copies <code>csv/input_csv/</code> into "
            "<code>csv/outlier_corrected_movement_location/</code>. "
            "Run this if downstream stages expect outlier-corrected "
            "output but you don't want SimBA's outlier criteria "
            "applied.</i>",
            self,
        )
        note.setTextFormat(Qt.RichText)
        note.setWordWrap(True)
        self.body_layout.addWidget(note)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        return {"config_path": self.config_path}

    def target(self, *, config_path: str) -> None:
        from mufasa.outlier_tools.skip_outlier_correction import (
            OutlierCorrectionSkipper,
        )
        OutlierCorrectionSkipper(config_path=config_path).run()

        # Write a run.toml stub if this is a v1 project — even a
        # skipped run is a run, and reproducing the experiment
        # later requires knowing that outlier correction was
        # deliberately skipped.
        v1_root = resolve_v1_project_root(config_path)
        if v1_root is not None:
            try:
                from mufasa.project_layout import (
                    ProjectPaths, RUN_PROVENANCE_FILENAME, Stages,
                    generate_run_id, write_run_toml,
                )
                run_id = generate_run_id()
                paths = ProjectPaths(v1_root)
                run_dir = paths.stage_run_dir(
                    Stages.OUTLIER_CORRECTED, run_id=run_id,
                )
                write_run_toml(
                    run_dir / RUN_PROVENANCE_FILENAME,
                    stage="outlier_corrected",
                    run_id=run_id,
                    params={"skipped": True},
                )
                # Drop a marker file so downstream auto-detection
                # of the latest run knows this is a skip-run (no
                # pose data lives inside this run dir — the legacy
                # csv/outlier_corrected_movement_location/ is the
                # real output until v1-aware downstream forms land).
                (run_dir / "SKIPPED").write_text(
                    "outlier correction was skipped; pose data "
                    "lives at csv/outlier_corrected_movement_location/\n"
                )
                print(
                    f"[outlier-skip] wrote run.toml: "
                    f"{run_dir / RUN_PROVENANCE_FILENAME}"
                )
            except Exception as exc:
                print(
                    f"[outlier-skip] WARNING: could not write "
                    f"run.toml ({exc}); skip operation completed."
                )


__all__ = [
    "SmoothingForm",
    "InterpolateForm",
    "OutlierSettingsForm",
    "DropBodypartsForm",
    "EgocentricAlignmentForm",
    "KalmanV2SmoothingForm",
    "RunOutlierCorrectionForm",
    "SkipOutlierCorrectionForm",
]
