"""
mufasa.ui_qt.dialogs.min_max_draw_size
========================================

Qt-native dialog for setting the min/max ROI-draw window size
ratios. The values control how big the ROI-drawing canvas appears
relative to the user's screen — too small and ROIs are hard to
place precisely; too big and the canvas overflows the screen.

Replaces the legacy Tk popup
``mufasa.ui.pop_ups.min_max_draw_size_popup.SetMinMaxDrawWindowSize``
which was previously bridged into the Qt workbench via the
subprocess-launch pattern in
``roi_video_table.py:_action_min_max_draw_size``
(see ``docs/tk_surface_audit.md`` §2g for the broader context of
that subprocess pattern and its phase-out).

Replacement landed in patch 122ct. After that patch:
* The Tk popup file is deleted.
* ``roi_video_table.py:_action_min_max_draw_size`` calls this
  dialog directly.

Functional differences from the Tk original
-------------------------------------------
* **QDoubleSpinBox** instead of dropdown lists. Same value range
  (0.0–1.0 in 0.05 steps) but native numeric-input UX — keyboard
  entry + scroll-wheel + step-arrows. The Tk dropdowns offered
  the same 21 discrete values; the spinbox covers them with
  better precision affordances.
* **Modal dialog** with OK/Cancel. The Tk popup had a SET button
  and no Cancel.
* **Validation feedback** via QMessageBox on out-of-range values.
"""
from __future__ import annotations

import os
from typing import Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QVBoxLayout, QWidget)

from mufasa.mixins.config_reader import ConfigReader
from mufasa.utils.checks import check_float
from mufasa.utils.enums import ConfigKey, Dtypes
from mufasa.utils.read_write import read_config_entry


_RATIO_MIN = 0.0
_RATIO_MAX = 1.0
_RATIO_STEP = 0.05
_RATIO_DECIMALS = 2


def _make_spinbox(value: float) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(_RATIO_MIN, _RATIO_MAX)
    sb.setSingleStep(_RATIO_STEP)
    sb.setDecimals(_RATIO_DECIMALS)
    sb.setValue(value)
    return sb


class MinMaxDrawSizeDialog(QDialog):
    """Qt port of `SetMinMaxDrawWindowSize` (122ct)."""

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 parent: Optional[QDialog] = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.setWindowTitle("Set drawing window size")
        self.setModal(True)

        # Load current ratios from config (with documented defaults
        # matching the Tk popup's defaults).
        try:
            reader = ConfigReader(
                config_path=config_path,
                read_video_info=False,
                create_logger=False)
            self._reader = reader
            self._min_h = read_config_entry(
                config=reader.config,
                section=ConfigKey.DISPLAY_SETTINGS.value,
                option=ConfigKey.MIN_ROI_DISPLAY_HEIGHT.value,
                default_value=0.60,
                data_type=Dtypes.FLOAT.value)
            self._min_w = read_config_entry(
                config=reader.config,
                section=ConfigKey.DISPLAY_SETTINGS.value,
                option=ConfigKey.MIN_ROI_DISPLAY_WIDTH.value,
                default_value=0.30,
                data_type=Dtypes.FLOAT.value)
            self._max_h = read_config_entry(
                config=reader.config,
                section=ConfigKey.DISPLAY_SETTINGS.value,
                option=ConfigKey.MAX_ROI_DISPLAY_HEIGHT.value,
                default_value=0.75,
                data_type=Dtypes.FLOAT.value)
            self._max_w = read_config_entry(
                config=reader.config,
                section=ConfigKey.DISPLAY_SETTINGS.value,
                option=ConfigKey.MAX_ROI_DISPLAY_WIDTH.value,
                default_value=0.50,
                data_type=Dtypes.FLOAT.value)
            for v in (self._min_h, self._min_w,
                      self._max_h, self._max_w):
                check_float(
                    name=f"{self.__class__.__name__} size",
                    value=v, min_value=0, max_value=1,
                    raise_error=True)
        except Exception as exc:
            QMessageBox.critical(
                self, "Cannot open size settings",
                f"Could not read display settings: "
                f"{type(exc).__name__}: {exc}")
            self._init_failed = True
            return

        self._init_failed = False

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Set the ROI-drawing window size as a ratio of the\n"
            "monitor dimensions. Min ratios floor the canvas\n"
            "(useful for small ROIs on big monitors); max ratios\n"
            "cap it (useful for ROIs on large videos)."
        )
        layout.addWidget(intro)

        # Max group
        max_group = QGroupBox("Maximum draw window")
        max_form = QFormLayout(max_group)
        self._max_w_spin = _make_spinbox(self._max_w)
        self._max_h_spin = _make_spinbox(self._max_h)
        max_form.addRow("Width ratio:", self._max_w_spin)
        max_form.addRow("Height ratio:", self._max_h_spin)
        layout.addWidget(max_group)

        # Min group
        min_group = QGroupBox("Minimum draw window")
        min_form = QFormLayout(min_group)
        self._min_w_spin = _make_spinbox(self._min_w)
        self._min_h_spin = _make_spinbox(self._min_h)
        min_form.addRow("Width ratio:", self._min_w_spin)
        min_form.addRow("Height ratio:", self._min_h_spin)
        layout.addWidget(min_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def init_failed(self) -> bool:
        return getattr(self, "_init_failed", False)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        # QDoubleSpinBox already enforces the [0,1] range; the
        # check_float() call below is belt-and-braces against
        # bypassed input.
        max_w = round(self._max_w_spin.value(), _RATIO_DECIMALS)
        max_h = round(self._max_h_spin.value(), _RATIO_DECIMALS)
        min_w = round(self._min_w_spin.value(), _RATIO_DECIMALS)
        min_h = round(self._min_h_spin.value(), _RATIO_DECIMALS)
        try:
            for v in (max_w, max_h, min_w, min_h):
                check_float(
                    name=f"{self.__class__.__name__} size",
                    value=v, min_value=0, max_value=1,
                    raise_error=True)
        except Exception as exc:
            QMessageBox.warning(
                self, "Invalid value",
                f"All ratios must be between 0 and 1.\n\n{exc}")
            return

        try:
            cfg = self._reader.config
            section = ConfigKey.DISPLAY_SETTINGS.value
            if not cfg.has_section(section):
                cfg.add_section(section)
            cfg[section][ConfigKey.MAX_ROI_DISPLAY_HEIGHT.value] = (
                str(max_h))
            cfg[section][ConfigKey.MAX_ROI_DISPLAY_WIDTH.value] = (
                str(max_w))
            cfg[section][ConfigKey.MIN_ROI_DISPLAY_WIDTH.value] = (
                str(min_w))
            cfg[section][ConfigKey.MIN_ROI_DISPLAY_HEIGHT.value] = (
                str(min_h))
            with open(self.config_path, "w") as fh:
                cfg.write(fh)
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not write config: "
                f"{type(exc).__name__}: {exc}")
            return

        QMessageBox.information(
            self, "Saved", "Display settings saved.")
        self.accept()
