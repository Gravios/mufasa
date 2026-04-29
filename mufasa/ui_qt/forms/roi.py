"""
mufasa.ui_qt.forms.roi
======================

Inline forms for the ROI (region-of-interest) workflow. Replaces 9
legacy popups:

* :class:`ROIAnalysisPopUp`
* :class:`ROIAggregateDataAnalyzerPopUp`
* :class:`ROIAnalysisTimeBinsPopUp`
      → :class:`ROIAnalysisForm` (3 popups → 1, mode dropdown)

* :class:`AppendROIFeaturesByAnimalPopUp`
* :class:`AppendROIFeaturesByBodyPartPopUp`
* :class:`RemoveROIFeaturesPopUp`
      → :class:`ROIFeaturesForm` (3 popups → 1, action dropdown)

* :class:`ROIDefinitionsCSVImporterPopUp`
* :class:`ROISizeStandardizerPopUp`
* (Tk) :class:`ROIVideoTable` for video selection
      → :class:`ROIManageForm` (3 popups → 1, action dropdown:
        Draw / Import / Standardise)

* :class:`VisualizeROITrackingPopUp`
* :class:`VisualizeROIFeaturesPopUp`
      → :class:`ROIVisualizeForm` (2 popups → 1, target dropdown)

When the user picks the Draw action and clicks Run, the form opens
:class:`mufasa.ui_qt.dialogs.roi_video_table.ROIVideoTableDialog`
on the Qt main thread (overrides the OperationForm's default
worker-thread dispatch via ``on_run``). The dialog is a native Qt
replacement for the legacy Tk video-table picker — earlier versions
spawned the Tk popup in a subprocess but it was reported broken on
recent Tk + Wayland combinations.

The actual ROI-drawing canvas (rectangles / circles / polygons drawn
on a frame with the mouse) is still the OpenCV-based ``ROI_ui``,
which the new dialog spawns per-row in subprocesses so each drawing
window runs independently and the Qt picker stays responsive.

Design notes
------------

Forms read the project's ``no_targets`` classifier list and per-animal
body-part list from ``project_config.ini`` via
:func:`mufasa.ui_qt.forms.pose_cleanup._load_animal_bps`. That helper
is now project-setup infrastructure — keeping one canonical loader
rather than duplicating parsing logic per form.

**Audit A3 parallel:** :class:`ROIAnalysisForm` dispatches to the
single-core or multi-process backend based on a ``cores`` field —
but unlike the heatmap bug, the ROI backends share the same kwargs,
so we don't need to paper over UI-control divergence.
"""
from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGridLayout, QGroupBox, QLabel,
                               QListWidget, QListWidgetItem, QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget)

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.forms.data_import import _PathField
from mufasa.ui_qt.forms.pose_cleanup import _load_animal_bps
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Shared helper: body-part multi-picker
# --------------------------------------------------------------------------- #
def _flat_bp_list(animal_bps: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Flatten ``{animal: [bps]}`` → ordered list of ``(animal, bp)``."""
    out: list[tuple[str, str]] = []
    for animal, bps in animal_bps.items():
        for bp in bps:
            out.append((animal, bp))
    return out


class _BodyPartPicker(QWidget):
    """Multi-select list of ``animal: bodypart`` entries. Used by ROI
    analysis and feature-append forms that need the user to pick which
    tracking points count for the analysis."""

    def __init__(self, config_path: Optional[str] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget(self)
        self.list.setSelectionMode(QListWidget.MultiSelection)
        self.list.setMinimumHeight(120)
        lay.addWidget(self.list)
        if config_path:
            self._populate(config_path)

    def _populate(self, config_path: str) -> None:
        try:
            animal_bps = _load_animal_bps(config_path)
        except Exception:
            return
        self.list.clear()
        for animal, bp in _flat_bp_list(animal_bps):
            item = QListWidgetItem(f"{animal}: {bp}")
            item.setData(Qt.UserRole, (animal, bp))
            self.list.addItem(item)

    def selected(self) -> list[tuple[str, str]]:
        return [it.data(Qt.UserRole) for it in self.list.selectedItems()]


# --------------------------------------------------------------------------- #
# 1. ROIAnalysisForm — 3 popups → 1
# --------------------------------------------------------------------------- #
class ROIAnalysisForm(OperationForm):
    """Compute ROI-occupancy statistics across the project.

    Mode selector switches between:

    * **Simple** (:class:`ROIAnalyzer`): per-animal time-in-ROI,
      entries, etc. Fast and minimal options.
    * **Aggregate** (:class:`ROIAggregateStatisticsAnalyzer`): same
      metrics plus first/last-entry times, bout durations,
      outside-ROI accounting, transpose options.
    * **Time bins** (:class:`ROITimebinAnalyzer`): aggregate
      statistics bucketed into fixed-size time windows.
    """

    title = "ROI analysis"
    description = ("Compute time-in-ROI / entry / bout statistics. "
                   "Simple mode is fast; Aggregate adds first/last entry "
                   "times and bout-duration metrics; Time bins buckets "
                   "everything into fixed-length windows.")

    MODES = [("Simple",    "simple"),
             ("Aggregate", "aggregate"),
             ("Time bins", "time_bins")]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.mode_cb = QComboBox(self)
        for label, _ in self.MODES:
            self.mode_cb.addItem(label)
        self.mode_cb.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Analysis mode:", self.mode_cb)

        # Body-part picker — shared across all modes
        self.bp_picker = _BodyPartPicker(config_path=self.config_path, parent=self)
        form.addRow("Body-parts:", self.bp_picker)

        self.threshold = QDoubleSpinBox(self)
        self.threshold.setRange(0.0, 1.0); self.threshold.setValue(0.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setToolTip(
            "Pose probability threshold. Frames below this are ignored."
        )
        form.addRow("Probability threshold:", self.threshold)

        # Detailed bout data — only meaningful for aggregate/time-bins
        self.detailed_bouts = QCheckBox("Include detailed bout data", self)
        form.addRow("", self.detailed_bouts)

        self.calc_distances = QCheckBox("Compute ROI-distance metrics", self)
        form.addRow("", self.calc_distances)

        # Mode-specific: extras
        self.extras_stack = QStackedWidget(self)
        # Simple — nothing extra
        self.extras_stack.addWidget(QWidget())
        # Aggregate — toggles
        agg_host = QWidget()
        agg_form = QFormLayout(agg_host); agg_form.setContentsMargins(0, 0, 0, 0)
        self.agg_total_time = QCheckBox("Total time", self); self.agg_total_time.setChecked(True)
        self.agg_entry_counts = QCheckBox("Entry counts", self); self.agg_entry_counts.setChecked(True)
        self.agg_first_entry = QCheckBox("First entry time", self)
        self.agg_last_entry = QCheckBox("Last entry time", self)
        self.agg_mean_bout = QCheckBox("Mean bout time", self)
        self.agg_outside = QCheckBox("Outside-ROI time", self)
        for w in (self.agg_total_time, self.agg_entry_counts, self.agg_first_entry,
                  self.agg_last_entry, self.agg_mean_bout, self.agg_outside):
            agg_form.addRow("", w)
        self.extras_stack.addWidget(agg_host)
        # Time bins — bin size
        tb_host = QWidget()
        tb_form = QFormLayout(tb_host); tb_form.setContentsMargins(0, 0, 0, 0)
        self.bin_size = QDoubleSpinBox(self)
        self.bin_size.setRange(1.0, 3600.0); self.bin_size.setValue(60.0)
        self.bin_size.setSuffix(" s")
        tb_form.addRow("Bin size:", self.bin_size)
        self.extras_stack.addWidget(tb_host)

        form.addRow("Mode-specific:", self.extras_stack)

        self.body_layout.addLayout(form)
        self._on_mode_changed(0)

    def _on_mode_changed(self, idx: int) -> None:
        self.extras_stack.setCurrentIndex(idx)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        bps = self.bp_picker.selected()
        if not bps:
            raise ValueError("Select at least one body-part.")
        mode = self.MODES[self.mode_cb.currentIndex()][1]
        args = {
            "config_path":   self.config_path,
            "mode":          mode,
            "body_parts":    [bp for (_animal, bp) in bps],
            "threshold":     float(self.threshold.value()),
            "detailed_bouts": bool(self.detailed_bouts.isChecked()),
            "calc_distances": bool(self.calc_distances.isChecked()),
        }
        if mode == "aggregate":
            args["aggregates"] = {
                "total_time":       self.agg_total_time.isChecked(),
                "entry_counts":     self.agg_entry_counts.isChecked(),
                "first_entry_time": self.agg_first_entry.isChecked(),
                "last_entry_time":  self.agg_last_entry.isChecked(),
                "mean_bout_time":   self.agg_mean_bout.isChecked(),
                "outside_rois":     self.agg_outside.isChecked(),
            }
        elif mode == "time_bins":
            args["bin_size"] = float(self.bin_size.value())
        return args

    def target(self, *, config_path: str, mode: str, body_parts: list[str],
               threshold: float, detailed_bouts: bool,
               calc_distances: bool, **extras) -> None:
        # Lazy-import the three backends; pick based on mode
        if mode == "simple":
            from mufasa.roi_tools.ROI_analyzer import ROIAnalyzer
            ROIAnalyzer(
                config_path=config_path,
                data_path=None,
                detailed_bout_data=detailed_bouts,
                calculate_distances=calc_distances,
                threshold=threshold,
                body_parts=body_parts,
            ).run()
        elif mode == "aggregate":
            from mufasa.roi_tools.roi_aggregate_statistics_analyzer import (
                ROIAggregateStatisticsAnalyzer,
            )
            agg = extras["aggregates"]
            ROIAggregateStatisticsAnalyzer(
                config_path=config_path,
                data_path=None,
                threshold=threshold,
                body_parts=body_parts,
                detailed_bout_data=detailed_bouts,
                calculate_distances=calc_distances,
                total_time=agg["total_time"],
                entry_counts=agg["entry_counts"],
                first_entry_time=agg["first_entry_time"],
                last_entry_time=agg["last_entry_time"],
                mean_bout_time=agg["mean_bout_time"],
                outside_rois=agg["outside_rois"],
            ).run()
        elif mode == "time_bins":
            from mufasa.roi_tools.roi_time_bins_analyzer import ROITimebinAnalyzer
            ROITimebinAnalyzer(
                config_path=config_path,
                bin_size=extras["bin_size"],
                data_path=None,
                threshold=threshold,
                body_parts=body_parts,
                detailed_bout_data=detailed_bouts,
                calculate_distances=calc_distances,
            ).run()


# --------------------------------------------------------------------------- #
# 2. ROIFeaturesForm — 3 popups → 1
# --------------------------------------------------------------------------- #
class ROIFeaturesForm(OperationForm):
    """Append or remove ROI-derived features from project CSVs.

    Action selector:

    * **Append (by animal)**: per-whole-animal ROI occupancy features
      (time in X, distance to X, entries, etc.)
    * **Append (by body-part)**: same but per specified body-part.
      Finer grain — distinguishes "nose in zone" from "animal in zone".
    * **Remove**: strip previously-appended ROI features.
    """

    title = "ROI features (append / remove)"
    description = ("Add ROI-derived features to the feature matrix, "
                   "or strip previously-added ones. Append before "
                   "feature extraction; remove before retraining on "
                   "changed ROI definitions.")

    ACTIONS = [("Append (by animal)",    "append_animal"),
               ("Append (by body-part)", "append_bodypart"),
               ("Remove",                "remove")]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.action_cb = QComboBox(self)
        for label, _ in self.ACTIONS:
            self.action_cb.addItem(label)
        self.action_cb.currentIndexChanged.connect(self._on_action_changed)
        form.addRow("Action:", self.action_cb)

        # Body-part picker — only meaningful for "by body-part" action
        self.bp_picker = _BodyPartPicker(config_path=self.config_path, parent=self)
        form.addRow("Body-parts:", self.bp_picker)

        # Append-to-existing toggle
        self.append_existing = QCheckBox(
            "Append to existing feature files (rather than overwrite)", self,
        )
        self.append_existing.setChecked(True)
        form.addRow("", self.append_existing)

        self.body_layout.addLayout(form)
        self._on_action_changed(0)

    def _on_action_changed(self, idx: int) -> None:
        action = self.ACTIONS[idx][1]
        # BP picker only relevant for "by body-part"
        self.bp_picker.setEnabled(action == "append_bodypart")
        # "Append to existing" irrelevant for Remove
        self.append_existing.setEnabled(action != "remove")

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        action = self.ACTIONS[self.action_cb.currentIndex()][1]
        args = {"config_path": self.config_path, "action": action}
        if action == "append_bodypart":
            bps = self.bp_picker.selected()
            if not bps:
                raise ValueError("Select at least one body-part for by-body-part append.")
            args["body_parts"] = [bp for (_animal, bp) in bps]
            args["append_existing"] = bool(self.append_existing.isChecked())
        elif action == "append_animal":
            args["append_existing"] = bool(self.append_existing.isChecked())
        return args

    def target(self, *, config_path: str, action: str,
               body_parts: Optional[list[str]] = None,
               append_existing: bool = True) -> None:
        if action.startswith("append"):
            from mufasa.roi_tools.ROI_feature_analyzer import ROIFeatureCreator
            ROIFeatureCreator(
                config_path=config_path,
                body_parts=body_parts,  # None for by-animal path
                data_path=None,
                append_data=append_existing,
            ).run()
        elif action == "remove":
            # Legacy RemoveROIFeaturesPopUp had no dedicated class — it
            # iterated the features_extracted dir and dropped ROI columns.
            # Try a small set of known removers.
            try:
                from mufasa.roi_tools import remove_roi_features as _rrf
                fn = (getattr(_rrf, "remove_roi_features", None)
                      or getattr(_rrf, "RemoveROIFeatures", None))
                if fn is None:
                    raise AttributeError
                if isinstance(fn, type):
                    fn(config_path=config_path).run()
                else:
                    fn(config_path=config_path)
            except (ImportError, AttributeError):
                raise NotImplementedError(
                    "Remove-ROI-features backend not found in this fork "
                    "(looked for mufasa.roi_tools.remove_roi_features)."
                )


# --------------------------------------------------------------------------- #
# 3. ROIManageForm — 2 popups → 1
# --------------------------------------------------------------------------- #
class ROIManageForm(OperationForm):
    """Import ROI definitions from CSV, or standardize ROI sizes in
    real-world units across the project.

    Two actions — each has its own set of inputs, switched via
    :class:`QStackedWidget`.
    """

    title = "ROI definitions — draw / import / standardise"
    description = ("Draw ROIs interactively on each video, import "
                   "previously-saved ROI definitions from CSV, or "
                   "re-scale all ROIs to match a reference video's "
                   "pixels-per-mm calibration.")

    ACTIONS = [("Draw ROIs (interactive)", "draw"),
               ("Import from CSV",         "import"),
               ("Standardise sizes",       "standardize")]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.action_cb = QComboBox(self)
        for label, _ in self.ACTIONS:
            self.action_cb.addItem(label)
        self.action_cb.currentIndexChanged.connect(self._on_action_changed)
        form.addRow("Action:", self.action_cb)

        self.stack = QStackedWidget(self)

        # --- Draw panel --- #
        # The drawing canvas is an OpenCV window driven by the Tk
        # ROIVideoTable picker (project_folder/videos listing → click
        # a row → draw on the frame). Mixing Tk and Qt event loops in
        # the same process is fragile, so we launch it as a separate
        # python subprocess. The user closes the Tk window when done;
        # ROI definitions get saved to project_folder/logs/measures/
        # ROI_definitions.h5 and are immediately picked up by the rest
        # of the ROI page on next access.
        draw_host = QWidget()
        draw_form = QFormLayout(draw_host); draw_form.setContentsMargins(0, 0, 0, 0)
        draw_note = QLabel(
            "Click <b>Run</b> to open the ROI drawing window. It will "
            "list every video in <code>project_folder/videos</code>; "
            "click a row to launch the OpenCV canvas, draw rectangles, "
            "circles, or polygons with the mouse, save, and close. "
            "ROI definitions persist in <code>logs/measures/"
            "ROI_definitions.h5</code>."
        )
        draw_note.setTextFormat(Qt.RichText)
        draw_note.setWordWrap(True)
        draw_form.addRow("", draw_note)
        self.stack.addWidget(draw_host)

        # --- Import-from-CSV panel --- #
        imp_host = QWidget()
        imp_form = QFormLayout(imp_host); imp_form.setContentsMargins(0, 0, 0, 0)
        self.rect_csv = _PathField(is_file=True, file_filter="CSV (*.csv)",
                                   placeholder="Rectangles CSV (optional)…")
        self.circle_csv = _PathField(is_file=True, file_filter="CSV (*.csv)",
                                     placeholder="Circles CSV (optional)…")
        self.poly_csv = _PathField(is_file=True, file_filter="CSV (*.csv)",
                                   placeholder="Polygons CSV (optional)…")
        imp_form.addRow("Rectangles:", self.rect_csv)
        imp_form.addRow("Circles:",    self.circle_csv)
        imp_form.addRow("Polygons:",   self.poly_csv)
        self.append_existing = QCheckBox("Append to existing ROI definitions", self)
        self.append_existing.setChecked(True)
        imp_form.addRow("", self.append_existing)
        # Format hint — saves users from chasing 'missing column' errors.
        fmt_hint = QLabel(
            "<b>CSV format:</b> see column lists in "
            "<code>mufasa.roi_tools.import_roi_csvs</code> "
            "(EXPECTED_RECT_COLS / EXPECTED_CIRC_COLS / "
            "EXPECTED_POLY_COLS). Easiest way to get a correct file "
            "is: draw ROIs once on one video, then export the "
            "auto-generated CSV from "
            "<code>logs/measures/ROI_definitions.h5</code> and adapt "
            "it for other videos."
        )
        fmt_hint.setTextFormat(Qt.RichText)
        fmt_hint.setWordWrap(True)
        fmt_hint.setStyleSheet("color: palette(mid); padding: 4px;")
        imp_form.addRow("", fmt_hint)
        self.stack.addWidget(imp_host)

        # --- Standardize panel --- #
        std_host = QWidget()
        std_form = QFormLayout(std_host); std_form.setContentsMargins(0, 0, 0, 0)
        self.reference_video = _PathField(
            is_file=True,
            file_filter="Videos (*.mp4 *.avi *.mov *.mkv *.webm)",
            placeholder="Reference video (px/mm calibrated)…",
        )
        std_form.addRow("Reference video:", self.reference_video)
        self.stack.addWidget(std_host)

        form.addRow("Parameters:", self.stack)
        self.body_layout.addLayout(form)

    def _on_action_changed(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)

    # ------------------------------------------------------------------ #
    # Override on_run so the 'draw' action opens a Qt dialog on the
    # main thread instead of dispatching to a worker. Import and
    # Standardise still use the default worker-thread runner via the
    # base-class on_run.
    # ------------------------------------------------------------------ #
    def on_run(self) -> None:
        try:
            kwargs = self.collect_args()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, f"{self.title}: invalid input", str(exc))
            return
        if kwargs.get("action") == "draw":
            # Open the unified Qt ROI definition panel on the main
            # thread. We bypass run_with_progress entirely — there's
            # no work to do here, just bringing up a modal-but-not-
            # blocking window. The user does the actual ROI-drawing
            # inside the panel (which calls the OpenCV selectors
            # synchronously in-process — see roi_define_panel.py for
            # the rationale).
            #
            # Earlier versions opened a separate "ROI Definitions —
            # Project Videos" table dialog first, then a per-video
            # drawing window from there. That two-window flow was
            # awkward (user had to alt-tab between the table and the
            # canvas, the table couldn't easily move between videos).
            # The unified panel has both in one window with PgUp/PgDn
            # navigation between videos.
            from mufasa.ui_qt.dialogs.roi_define_panel import (
                ROIDefinePanel,
            )
            top = self.window()   # the MufasaWorkbench
            try:
                dlg = ROIDefinePanel(
                    config_path=kwargs["config_path"], parent=top,
                )
            except Exception as exc:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self, "Could not open ROI panel",
                    f"{type(exc).__name__}: {exc}",
                )
                return
            dlg.show()
            # Stash on the workbench so it isn't GC'd.
            top._dialog_refs = getattr(top, "_dialog_refs", [])
            top._dialog_refs.append(dlg)
            return
        # All other actions go through the standard worker-thread path.
        super().on_run()

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        action = self.ACTIONS[self.action_cb.currentIndex()][1]
        if action == "draw":
            return {"config_path": self.config_path, "action": "draw"}
        if action == "import":
            rect = self.rect_csv.path or None
            circ = self.circle_csv.path or None
            poly = self.poly_csv.path or None
            if not any([rect, circ, poly]):
                raise ValueError("Provide at least one ROI CSV (rectangles, circles, or polygons).")
            return {
                "config_path":     self.config_path,
                "action":          "import",
                "rectangles_path": rect,
                "circles_path":    circ,
                "polygon_path":    poly,
                "append":          bool(self.append_existing.isChecked()),
            }
        # standardize
        ref = self.reference_video.path
        if not ref:
            raise ValueError("Reference video is required for standardisation.")
        return {"config_path": self.config_path,
                "action": "standardize",
                "reference_video": ref}

    def target(self, *, config_path: str, action: str, **params) -> None:
        if action == "draw":
            # The 'draw' action is intercepted in on_run() before this
            # worker-thread method is invoked — it opens a Qt dialog
            # on the main thread. If we somehow reach this branch, do
            # nothing rather than crash.
            return
        if action == "import":
            from mufasa.roi_tools.import_roi_csvs import ROIDefinitionsCSVImporter
            ROIDefinitionsCSVImporter(
                config_path=config_path,
                rectangles_path=params.get("rectangles_path"),
                circles_path=params.get("circles_path"),
                polygon_path=params.get("polygon_path"),
                append=params.get("append", True),
            ).run()
        elif action == "standardize":
            from mufasa.roi_tools.ROI_size_standardizer import ROISizeStandardizer
            ROISizeStandardizer(
                config_path=config_path,
                reference_video=params["reference_video"],
            ).run()


# --------------------------------------------------------------------------- #
# 4. ROIVisualizeForm — 2 popups → 1
# --------------------------------------------------------------------------- #
class ROIVisualizeForm(OperationForm):
    """Render ROI overlays onto videos.

    Two targets:

    * **ROI tracking** — draws ROI geometry + entry/exit events frame-by-frame.
    * **ROI features** — overlays ROI-derived feature values on video.

    Both are multi-process capable; the form surfaces a ``cores`` field.
    """

    title = "Visualize ROI on video"
    description = ("Render ROI overlays onto videos. ROI tracking shows "
                   "the zone geometry + entries; ROI features overlays "
                   "the numerical values as text.")

    TARGETS = [("ROI tracking", "tracking"),
               ("ROI features", "features")]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.target_cb = QComboBox(self)
        for label, _ in self.TARGETS:
            self.target_cb.addItem(label)
        form.addRow("Overlay type:", self.target_cb)

        self.video_picker = _PathField(
            is_file=True,
            file_filter="Videos (*.mp4 *.avi *.mov *.mkv *.webm)",
            placeholder="Source video…",
        )
        form.addRow("Video:", self.video_picker)

        # Shared toggles
        self.bp_picker = _BodyPartPicker(config_path=self.config_path, parent=self)
        form.addRow("Body-parts:", self.bp_picker)

        self.show_names = QCheckBox("Show animal names", self)
        self.show_names.setChecked(True)
        form.addRow("", self.show_names)

        self.show_bp = QCheckBox("Show body-part dots", self)
        self.show_bp.setChecked(True)
        form.addRow("", self.show_bp)

        self.show_bbox = QCheckBox("Show bounding boxes", self)
        form.addRow("", self.show_bbox)

        self.threshold = QDoubleSpinBox(self)
        self.threshold.setRange(0.0, 1.0); self.threshold.setValue(0.0)
        self.threshold.setSingleStep(0.05)
        form.addRow("Probability threshold:", self.threshold)

        self.cores = QSpinBox(self)
        self.cores.setRange(1, max(1, linux_env.cpu_count()))
        self.cores.setValue(max(1, linux_env.cpu_count() // 4))
        form.addRow("Worker cores:", self.cores)

        self.gpu = QCheckBox("Use GPU (NVENC) if available", self)
        self.gpu.setChecked(linux_env.nvenc_available())
        self.gpu.setEnabled(linux_env.nvenc_available())
        form.addRow("", self.gpu)

        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        if not self.video_picker.path:
            raise ValueError("Video is required.")
        bps = self.bp_picker.selected()
        return {
            "config_path": self.config_path,
            "target":      self.TARGETS[self.target_cb.currentIndex()][1],
            "video_path":  self.video_picker.path,
            "body_parts":  [bp for (_a, bp) in bps] if bps else None,
            "show_names":  bool(self.show_names.isChecked()),
            "show_bp":     bool(self.show_bp.isChecked()),
            "show_bbox":   bool(self.show_bbox.isChecked()),
            "threshold":   float(self.threshold.value()),
            "cores":       int(self.cores.value()),
            "gpu":         bool(self.gpu.isChecked()),
        }

    def target(self, *, config_path: str, target: str, video_path: str,
               body_parts: Optional[list[str]], show_names: bool,
               show_bp: bool, show_bbox: bool, threshold: float,
               cores: int, gpu: bool) -> None:
        if target == "tracking":
            # ROIPlotter — single-core; ROIPlotMultiprocess — multi
            if cores > 1:
                from mufasa.plotting.roi_plotter_mp import ROIPlotMultiprocess
                cls = ROIPlotMultiprocess
            else:
                from mufasa.plotting.roi_plotter import ROIPlotter
                cls = ROIPlotter
            cls(
                config_path=config_path,
                video_path=video_path,
                body_parts=body_parts,
                threshold=threshold,
                show_animal_name=show_names,
                show_body_part=show_bp,
                show_bbox=show_bbox,
            ).run()
        elif target == "features":
            from mufasa.plotting.ROI_feature_visualizer_mp import (
                ROIfeatureVisualizerMultiprocess,
            )
            ROIfeatureVisualizerMultiprocess(
                config_path=config_path,
                video_path=video_path,
                body_parts=body_parts,
                show_animal_names=show_names,
                core_cnt=cores,
                gpu=gpu,
            ).run()


__all__ = [
    "ROIAnalysisForm",
    "ROIFeaturesForm",
    "ROIManageForm",
    "ROIVisualizeForm",
]
