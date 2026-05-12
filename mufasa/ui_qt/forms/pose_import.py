"""
mufasa.ui_qt.forms.pose_import
==============================

Project-level pose import form. Distinct from
:mod:`mufasa.ui_qt.forms.data_import` (which is for cross-format
*converters* like DLC→YOLO); this form takes pose-estimation files
produced by an external tracker and loads them into the currently-open
project's ``csv/input_csv/`` directory.

As of 6.0.0.dev4 only one route is surfaced — single-animal DLC H5 —
because that's the importer we had to add; other trackers (CSV DLC,
maDLC H5, SLEAP, FaceMap, etc.) can be wired in here incrementally
with the same declarative route pattern.

Requires an open project (``config_path``). If no project is loaded
the form disables itself with a hint pointing at File → New /
Open project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout, QWidget)

from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Lazy backend factory (mirrors the pattern in forms/data_import.py)
# --------------------------------------------------------------------------- #
def _lazy(modpath: str, classname: str) -> Callable[..., Any]:
    def _factory(**kw):
        mod = __import__(modpath, fromlist=[classname])
        return getattr(mod, classname)(**kw)
    _factory.__name__ = f"{modpath}.{classname}"
    return _factory


# --------------------------------------------------------------------------- #
# Route registry — add new tracker flavours here
# --------------------------------------------------------------------------- #
# Each entry: label → (backend_factory, kwargs_map).
# Backend is instantiated with the collected kwargs; kwargs_map renames
# the generic UI field names onto the backend's constructor parameter
# names.
POSE_IMPORT_ROUTES: dict = {
    "DLC H5 (single animal)": dict(
        backend=_lazy("mufasa.pose_importers.dlc_h5_importer",
                      "DLCSingleAnimalH5Importer"),
        kwargs_map={
            "source_path": "data_folder",
        },
    ),
    "DLC CSV (single animal)": dict(
        backend=_lazy("mufasa.pose_importers.dlc_csv_importer",
                      "DLCSingleAnimalCSVImporter"),
        kwargs_map={
            "source_path": "data_folder",
        },
    ),
}


class PoseImportForm(OperationForm):
    """Form for importing pose-estimation output into the current
    project's ``csv/input_csv/``. The ``config_path`` supplied at
    construction binds the form to the open project."""

    title = "Import pose-estimation data"
    description = (
        "Load pose tracking output into the current project. "
        "Files are normalized to SimBA's multi-index CSV/parquet "
        "layout and written to <code>project_folder/csv/input_csv/</code>."
    )

    # ------------------------------------------------------------------ #
    # Form construction
    # ------------------------------------------------------------------ #
    def build(self) -> None:
        form = QFormLayout()

        # Tracker / format picker
        self._route_combo = QComboBox()
        self._route_combo.addItems(list(POSE_IMPORT_ROUTES.keys()))
        form.addRow("Pose file format:", self._route_combo)

        # Source directory (H5 files live here)
        self._source_edit = QLineEdit()
        self._source_edit.setReadOnly(True)
        self._source_edit.setPlaceholderText(
            "Directory containing .h5 files (DLC output)"
        )
        source_row = QWidget()
        sl = QHBoxLayout(source_row)
        sl.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_source)
        sl.addWidget(self._source_edit, 1)
        sl.addWidget(browse)
        form.addRow("Source directory:", source_row)

        # Likelihood threshold (optional — default off = 0.0).
        # Points with DLC confidence below this value get their (x, y)
        # zeroed; the likelihood column itself is preserved. Combined
        # with the Preprocessing page's Interpolate form, this is the
        # primary tool for dealing with bad DLC frames. Default 0.0
        # (no mask) to keep behaviour unchanged for users who don't
        # know about this.
        self._p_threshold = QDoubleSpinBox()
        self._p_threshold.setRange(0.0, 1.0)
        self._p_threshold.setSingleStep(0.05)
        self._p_threshold.setDecimals(2)
        self._p_threshold.setValue(0.0)
        self._p_threshold.setToolTip(
            "Body-parts with DLC likelihood strictly below this value "
            "are marked as missing (x=y=0). Run Interpolate on the "
            "Preprocessing page to fill them in. 0.0 disables the "
            "filter (all points kept verbatim). Typical starting "
            "value: 0.5."
        )
        form.addRow("Likelihood threshold:", self._p_threshold)
        thresh_hint = QLabel(
            "If you set a threshold &gt; 0, run "
            "<b>Preprocessing → Interpolate missing frames</b> after "
            "import — masked points are left at (0, 0) otherwise, "
            "which corrupts movement features."
        )
        thresh_hint.setTextFormat(Qt.RichText)
        thresh_hint.setWordWrap(True)
        thresh_hint.setStyleSheet("color: palette(placeholder-text); padding: 4px;")
        form.addRow("", thresh_hint)

        # Interpolation and smoothing intentionally not exposed here:
        # the Preprocessing page (formerly "Pose cleanup") has the
        # canonical surfaces for both, with strictly more options
        # (user-controllable copy_originals, auto-detected multi-
        # index headers, picker-driven input source). Duplicating
        # stripped-down toggles at import time was redundant and
        # surfaced worse defaults — see patches 122g, 122h.

        self.body_layout.addLayout(form)

        # Disable if no project loaded
        if not self.config_path:
            hint = QLabel(
                "No project loaded. "
                "Use <b>File → New project…</b> or "
                "<b>File → Open project…</b> before importing pose data."
            )
            hint.setTextFormat(Qt.RichText)
            hint.setWordWrap(True)
            hint.setStyleSheet("color: palette(placeholder-text); padding: 6px;")
            self.body_layout.addWidget(hint)
            self.run_btn.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Validation / kwargs collection
    # ------------------------------------------------------------------ #
    def _pick_source(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select directory containing pose files", "",
        )
        if d:
            self._source_edit.setText(d)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError(
                "No project loaded. Open or create a project first."
            )
        source = self._source_edit.text().strip()
        if not source:
            raise RuntimeError("Pick a source directory.")
        if not Path(source).is_dir():
            raise RuntimeError(f"Source is not a directory: {source}")

        label = self._route_combo.currentText()
        route = POSE_IMPORT_ROUTES[label]

        return {
            "route": route,
            "config_path": self.config_path,
            "source_path": source,
            "p_threshold": float(self._p_threshold.value()),
        }

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def target(self, *, route: dict, config_path: str, source_path: str,
               p_threshold: float) -> None:
        km = route["kwargs_map"]
        kwargs = {
            "config_path": config_path,
            km.get("source_path", "source_path"): source_path,
            "p_threshold": p_threshold,
        }
        # Defensive filter — same reasoning as the other forms.
        # Backends that don't accept p_threshold (e.g. future CSV /
        # SLEAP routes) will silently drop it via the filter.
        # Importer backends still accept interpolation_settings and
        # smoothing_settings as optional parameters; we never pass
        # them, so they use the default (None). Users run those
        # passes on the Preprocessing page instead.
        from mufasa.ui_qt.forms._backend_dispatch import filter_kwargs
        kwargs = filter_kwargs(route["backend"], kwargs)
        runner = route["backend"](**kwargs)
        if runner is not None and hasattr(runner, "run"):
            runner.run()


__all__ = ["PoseImportForm", "POSE_IMPORT_ROUTES"]
