"""
mufasa.ui_qt.forms.pose_import
==============================

Project-level pose import form. Distinct from
:mod:`mufasa.ui_qt.forms.data_import` (which is for cross-format
*converters* like DLC→YOLO); this form takes pose-estimation files
produced by an external tracker and loads them into the currently-open
project's pose tree:

* v1 projects (``project.toml``): ``<root>/sources/pose/``
* Legacy projects (``project_config.ini``): ``<project>/csv/input_csv/``

The branching is delegated to
:func:`mufasa.project_layout.project_paths_from_config` via the
underlying ``mufasa.pose_importers.*`` backends; the form itself
doesn't need to know which layout is active.

As of patch 122di, nine routes are wired:

Step 1 (patch 122dh) — most-used 2D pose trackers:
* DLC H5 / CSV (single animal)
* DLC H5 (multi-animal / maDLC)
* SLEAP CSV / H5 / .slp
* SuperAnimal-TopView

Step 2 (patch 122di) — speed-prioritized + Caltech-MARS community:
* YOLO-pose
* MARS (two-mouse social)

The dormant importers (FaceMap, TRK, DANNCE, SimBA blob) can be
wired with the same declarative route pattern when their
communities call for them. **3D marker trajectory data** (Vicon /
mocap / AniPose 3D / DANNCE) is a separate concern — needs a
different ingestion path since the data is already-tracked 3D
coordinates rather than 2D-pose-from-video; see the project
roadmap.

Requires an open project (``config_path``). If no project is loaded
the form disables itself with a hint pointing at File → New /
Open project.

Patch 122w: section title renamed from
'Import pose-estimation data' to 'Import Pose Data' to match the
shorter user-facing labels used on other Data Import sections; the
description updated to mention both v1 and legacy destination
directories explicitly.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

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
# --------------------------------------------------------------------------- #
# Route registry — add new tracker flavours here
# --------------------------------------------------------------------------- #
# Each entry: label → dict with:
#   backend                — _lazy(modpath, classname) factory
#   kwargs_map             — rename UI kwargs onto backend kwargs.
#                            Default: source_path → data_folder.
#                            sleap_slp uses project_path instead of
#                            config_path; rename there.
#   requires_animal_ids    — multi-animal trackers need id_lst
#                            (comma-separated UI input).
#   accepts_p_threshold    — only DLC single-animal importers expose
#                            the likelihood-mask parameter. Other
#                            backends don't accept it; the kwargs
#                            filter drops it if passed.
#   extra_backend_kwargs   — fixed kwargs to pass to the backend
#                            (e.g., maDLCImporterH5 needs file_type="h5").
#   source_hint            — placeholder text in the source-directory
#                            field, customised per tracker.
#
# Patch 122dh: added 5 entries (maDLC H5, SLEAP CSV/H5/.slp,
# SuperAnimal-TopView) — covers the most-used 2D pose trackers
# in current rodent / general behaviour research. 3D marker data
# (Vicon, AniPose, DANNCE) is a separate future concern; needs a
# different ingestion path (already-tracked 3D coords vs 2D pose
# to be tracked).
POSE_IMPORT_ROUTES: dict = {
    "DLC H5 (single animal)": dict(
        backend=_lazy("mufasa.pose_importers.dlc_h5_importer",
                      "DLCSingleAnimalH5Importer"),
        kwargs_map={"source_path": "data_folder"},
        requires_animal_ids=False,
        accepts_p_threshold=True,
        source_hint="Directory containing DLC .h5 files",
    ),
    "DLC CSV (single animal)": dict(
        backend=_lazy("mufasa.pose_importers.dlc_csv_importer",
                      "DLCSingleAnimalCSVImporter"),
        kwargs_map={"source_path": "data_folder"},
        requires_animal_ids=False,
        accepts_p_threshold=True,
        source_hint="Directory containing DLC .csv files",
    ),
    "DLC H5 (multi-animal / maDLC)": dict(
        backend=_lazy("mufasa.pose_importers.madlc_importer",
                      "MADLCImporterH5"),
        kwargs_map={"source_path": "data_folder"},
        requires_animal_ids=True,
        accepts_p_threshold=False,
        extra_backend_kwargs={"file_type": "h5"},
        source_hint="Directory containing maDLC .h5 files",
    ),
    "SLEAP CSV": dict(
        backend=_lazy("mufasa.pose_importers.sleap_csv_importer",
                      "SLEAPImporterCSV"),
        kwargs_map={"source_path": "data_folder"},
        requires_animal_ids=True,
        accepts_p_threshold=False,
        source_hint="Directory containing SLEAP .csv files",
    ),
    "SLEAP H5": dict(
        backend=_lazy("mufasa.pose_importers.sleap_h5_importer",
                      "SLEAPImporterH5"),
        kwargs_map={"source_path": "data_folder"},
        requires_animal_ids=True,
        accepts_p_threshold=False,
        source_hint="Directory containing SLEAP .h5 files",
    ),
    "SLEAP .slp": dict(
        backend=_lazy("mufasa.pose_importers.sleap_slp_importer",
                      "SLEAPImporterSLP"),
        # sleap_slp_importer uses `project_path` instead of
        # `config_path` as the constructor parameter name.
        kwargs_map={"source_path": "data_folder",
                    "config_path": "project_path"},
        requires_animal_ids=True,
        accepts_p_threshold=False,
        source_hint="Directory containing SLEAP .slp project files",
    ),
    "SuperAnimal-TopView": dict(
        backend=_lazy("mufasa.pose_importers.superanimal_import",
                      "SuperAnimalTopViewImporter"),
        kwargs_map={"source_path": "data_folder"},
        requires_animal_ids=True,
        accepts_p_threshold=False,
        source_hint=("Directory containing SuperAnimal-TopView "
                     "inference output"),
    ),
    # Patch 122di — pose-importers step 2.
    "YOLO-pose": dict(
        backend=_lazy("mufasa.pose_importers.simba_yolo_importer",
                      "SimBAYoloImporter"),
        # YOLO importer takes `data_dir` rather than `data_folder`.
        kwargs_map={"source_path": "data_dir"},
        requires_animal_ids=False,
        accepts_p_threshold=False,
        source_hint=("Directory containing YOLO-pose inference "
                     "results"),
    ),
    "MARS (two-mouse social)": dict(
        backend=_lazy("mufasa.pose_importers.import_mars",
                      "MarsImporter"),
        # MARS uses `data_path` (accepts directory OR single .json
        # file; the form validates directory only, which is fine —
        # MARS's directory branch globs for .json inside it).
        kwargs_map={"source_path": "data_path"},
        requires_animal_ids=False,
        accepts_p_threshold=False,
        # MARS requires interpolation_method + smoothing_method as
        # positional args (no defaults — unlike DLC/SLEAP where
        # they're optional). The form's policy is to NOT expose
        # interp/smoothing at import time (run on the Preprocessing
        # page instead). Pass sentinel "no-op" values so the
        # backend instantiates cleanly. Specifically:
        #
        # - interpolation_method="None" — Interpolate.fix_missing_values
        #   treats this as a skip.
        # - smoothing_method={"Method": "None", "Parameters": {}} —
        #   MARS's __run_smoothing branches on the Method value
        #   (Gaussian / Savitzky Golay); "None" matches neither and
        #   falls through without smoothing.
        extra_backend_kwargs={
            "interpolation_method": "None",
            "smoothing_method": {"Method": "None",
                                 "Parameters": {}},
        },
        source_hint=("Directory containing MARS JSON pose-detection "
                     "output (one .json per video)"),
    ),
}


class PoseImportForm(OperationForm):
    """Form for importing pose-estimation output into the current
    project's ``csv/input_csv/``. The ``config_path`` supplied at
    construction binds the form to the open project."""

    title = "Import Pose Data"
    description = (
        "Load pose tracking output into the current project. "
        "Files are normalised to Mufasa's multi-index CSV/parquet "
        "layout and written to <code>sources/pose/</code> for v1 "
        "projects or <code>csv/input_csv/</code> for legacy "
        "projects. The destination is resolved automatically from "
        "the active project's layout."
    )

    # ------------------------------------------------------------------ #
    # Form construction
    # ------------------------------------------------------------------ #
    def build(self) -> None:
        form = QFormLayout()

        # Tracker / format picker
        self._route_combo = QComboBox()
        self._route_combo.addItems(list(POSE_IMPORT_ROUTES.keys()))
        self._route_combo.currentTextChanged.connect(
            self._on_route_changed)
        form.addRow("Pose file format:", self._route_combo)

        # Source directory (h5 / csv / .slp files live here)
        self._source_edit = QLineEdit()
        self._source_edit.setReadOnly(True)
        source_row = QWidget()
        sl = QHBoxLayout(source_row)
        sl.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_source)
        sl.addWidget(self._source_edit, 1)
        sl.addWidget(browse)
        form.addRow("Source directory:", source_row)

        # Animal IDs (multi-animal trackers only).
        # Patch 122dh: added for maDLC / SLEAP / SuperAnimal routes.
        # Comma-separated names; each name becomes one column-group
        # in the resulting multi-index dataframe. The list length
        # must match the project's animal_count (defined in the
        # project setup form / project.toml).
        self._animal_ids_row_widget = QWidget()
        ar = QHBoxLayout(self._animal_ids_row_widget)
        ar.setContentsMargins(0, 0, 0, 0)
        self._animal_ids_edit = QLineEdit()
        self._animal_ids_edit.setPlaceholderText(
            "e.g. mouse1, mouse2  (comma-separated)"
        )
        ar.addWidget(self._animal_ids_edit, 1)
        # We need to keep a handle to the form's row so it can be
        # hidden for single-animal routes. QFormLayout exposes
        # rows via labelForField; remember the label widget too.
        self._animal_ids_label = QLabel("Animal IDs:")
        form.addRow(self._animal_ids_label,
                    self._animal_ids_row_widget)

        # Likelihood threshold (DLC single-animal only).
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
        self._p_threshold_label = QLabel("Likelihood threshold:")
        form.addRow(self._p_threshold_label, self._p_threshold)
        thresh_hint = QLabel(
            "If you set a threshold &gt; 0, run "
            "<b>Preprocessing → Interpolate missing frames</b> after "
            "import — masked points are left at (0, 0) otherwise, "
            "which corrupts movement features."
        )
        thresh_hint.setTextFormat(Qt.RichText)
        thresh_hint.setWordWrap(True)
        thresh_hint.setStyleSheet(
            "color: palette(placeholder-text); padding: 4px;")
        self._p_threshold_hint = thresh_hint
        form.addRow("", thresh_hint)

        # Interpolation and smoothing intentionally not exposed here:
        # the Preprocessing page (formerly "Pose cleanup") has the
        # canonical surfaces for both, with strictly more options
        # (user-controllable copy_originals, auto-detected multi-
        # index headers, picker-driven input source). Duplicating
        # stripped-down toggles at import time was redundant and
        # surfaced worse defaults — see patches 122g, 122h.

        self.body_layout.addLayout(form)

        # Apply route-specific UI state for the initial selection
        self._on_route_changed(self._route_combo.currentText())

        # Disable if no project loaded
        if not self.config_path:
            hint = QLabel(
                "No project loaded. "
                "Use <b>File → New project…</b> or "
                "<b>File → Open project…</b> before importing pose data."
            )
            hint.setTextFormat(Qt.RichText)
            hint.setWordWrap(True)
            hint.setStyleSheet(
                "color: palette(placeholder-text); padding: 6px;")
            self.body_layout.addWidget(hint)
            self.run_btn.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Route change handler (patch 122dh) — show / hide conditional fields
    # ------------------------------------------------------------------ #
    def _on_route_changed(self, label: str) -> None:
        route = POSE_IMPORT_ROUTES.get(label, {})
        # Animal IDs row: visible only for multi-animal routes.
        needs_ids = bool(route.get("requires_animal_ids", False))
        self._animal_ids_label.setVisible(needs_ids)
        self._animal_ids_row_widget.setVisible(needs_ids)
        # Likelihood-threshold row: visible only for DLC single-
        # animal (the only backend that accepts p_threshold).
        accepts_p = bool(route.get("accepts_p_threshold", False))
        self._p_threshold_label.setVisible(accepts_p)
        self._p_threshold.setVisible(accepts_p)
        self._p_threshold_hint.setVisible(accepts_p)
        # Source-dir placeholder customised per route.
        hint = route.get(
            "source_hint",
            "Directory containing pose files",
        )
        self._source_edit.setPlaceholderText(hint)

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

        # Animal IDs: required for multi-animal routes, ignored
        # otherwise. Parse the comma-separated text into a list of
        # non-empty stripped names.
        animal_ids: list[str] = []
        if route.get("requires_animal_ids", False):
            raw = self._animal_ids_edit.text().strip()
            if not raw:
                raise RuntimeError(
                    "Enter animal IDs (comma-separated) for this "
                    "multi-animal tracker."
                )
            animal_ids = [s.strip() for s in raw.split(",")
                          if s.strip()]
            if not animal_ids:
                raise RuntimeError(
                    "Could not parse any animal IDs from the input. "
                    "Use comma-separated names, e.g. 'mouse1, mouse2'."
                )

        return {
            "route":        route,
            "config_path":  self.config_path,
            "source_path":  source,
            "p_threshold":  float(self._p_threshold.value()),
            "animal_ids":   animal_ids,
        }

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def target(self, *, route: dict, config_path: str,
               source_path: str, p_threshold: float,
               animal_ids: list[str]) -> None:
        km = route["kwargs_map"]
        # Build the canonical kwargs dict; the kwargs_map can rename
        # any of these keys onto the backend's expected name.
        # config_path → backend's project_path for sleap_slp only.
        config_kwarg = km.get("config_path", "config_path")
        source_kwarg = km.get("source_path", "data_folder")
        kwargs = {
            config_kwarg: config_path,
            source_kwarg: source_path,
            "p_threshold": p_threshold,
            "id_lst":      animal_ids,
        }
        # Tracker-specific fixed kwargs (e.g., maDLC's file_type="h5").
        kwargs.update(route.get("extra_backend_kwargs", {}))
        # Defensive filter — drops kwargs the backend doesn't accept.
        # DLC single-animal backends don't accept id_lst; SLEAP /
        # maDLC / SuperAnimal backends don't accept p_threshold;
        # the filter handles both directions transparently.
        # Backends still accept interpolation_settings + smoothing_settings
        # as optional parameters; we never pass them, so they default
        # to None. Users run those passes on the Preprocessing page
        # instead.
        from mufasa.ui_qt.forms._backend_dispatch import filter_kwargs
        kwargs = filter_kwargs(route["backend"], kwargs)
        runner = route["backend"](**kwargs)
        if runner is not None and hasattr(runner, "run"):
            runner.run()


__all__ = ["PoseImportForm", "POSE_IMPORT_ROUTES"]
