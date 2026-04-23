"""
mufasa.ui_qt.forms.analysis
===========================

Unified analysis form. Replaces 12 legacy statistics popups:

* :class:`SeverityAnalysisPopUp` (bout-based + frame-based modes)
* :class:`ClfDescriptiveStatisticsPopUp`
* :class:`BooleanConditionalSlicerPopUp`
* :class:`DistanceAnalysisPopUp`
* :class:`FSTTCPopUp`
* :class:`MovementAnalysisPopUp`
* :class:`MovementAnalysisTimeBinsPopUp`
* :class:`AnimalDirectingOtherAnimalsPopUp`
* :class:`ClfByTimeBinsPopUp`
* :class:`ClfByROIPopUp`
* :class:`DistanceTimebinsPopUp`

Architecture mirrors :mod:`mufasa.ui_qt.forms.visualizations`:

* ``ROUTES`` declares one record per legacy popup.
* A shared form adapts its field layout per route via the
  extras-descriptor mechanism + conditional pickers
  (classifiers list, body-parts list, bin-length field).
* Backends resolved lazily so the form loads instantly.

Two analysis-specific reusable widgets live here:

* :class:`_ClassifierPicker` — multi-select list of classifier names
  read from ``SML settings`` in ``project_config.ini``.
* :class:`_BodypartPicker` — imported from :mod:`.roi` so we don't
  duplicate the parse-project-config helper.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget)

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.forms.roi import _BodyPartPicker
from mufasa.ui_qt.forms.visualizations import _ExtrasFormBuilder, _lazy_factory
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Classifier picker
# --------------------------------------------------------------------------- #
def _load_classifier_names(config_path: str) -> list[str]:
    """Read ``SML settings.target_name_N`` → list of classifier names."""
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    n = cfg.getint("SML settings", "no_targets", fallback=0)
    names = []
    for i in range(1, n + 1):
        k = f"target_name_{i}"
        if cfg.has_option("SML settings", k):
            names.append(cfg.get("SML settings", k))
    return names


class _ClassifierPicker(QWidget):
    """Multi-select classifier list, populated from project_config.ini."""

    def __init__(self, config_path: Optional[str] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget(self)
        self.list.setSelectionMode(QListWidget.MultiSelection)
        self.list.setMinimumHeight(100)
        lay.addWidget(self.list)
        if config_path:
            self._populate(config_path)

    def _populate(self, config_path: str) -> None:
        self.list.clear()
        try:
            for name in _load_classifier_names(config_path):
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, name)
                self.list.addItem(item)
        except Exception:
            pass

    def selected(self) -> list[str]:
        return [it.data(Qt.UserRole) for it in self.list.selectedItems()]


# --------------------------------------------------------------------------- #
# Route table
# --------------------------------------------------------------------------- #
@dataclass
class _AnalysisRoute:
    """Declaration of one analysis route.

    Attributes
    ----------
    label : str
        Dropdown entry.
    backend_sp : callable
        Single-process backend factory (always required — every
        analysis has at least SP).
    backend_mp : callable | None
        Multi-process alternative.
    needs : set[str]
        Input-selector requirements. Subset of
        {"body_parts", "classifiers", "bin_length", "severity_mode"}.
    extras : list[tuple]
        Field descriptors for :class:`_ExtrasFormBuilder`.
    kwargs_map : dict
        Remaps form-side kwargs to backend kwargs.
    """
    label: str
    backend_sp: Callable[..., Any]
    backend_mp: Optional[Callable[..., Any]] = None
    needs: set = field(default_factory=set)
    extras: list = field(default_factory=list)
    kwargs_map: dict = field(default_factory=dict)


ROUTES: list[_AnalysisRoute] = [
    # ---------- Severity (bout + frame modes in one route) ---------- #
    _AnalysisRoute(
        label="Severity analysis",
        backend_sp=_lazy_factory("mufasa.data_processors.severity_bout_based_calculator",
                                 "SeverityBoutCalculator"),
        needs={"severity_mode", "classifiers"},
        extras=[
            ("frames",   "bool", True),
            ("seconds",  "bool", True),
            ("clf_name", "str",  ""),
        ],
    ),
    # ---------- Descriptive classifier statistics ---------- #
    _AnalysisRoute(
        label="Classifier descriptive statistics",
        backend_sp=_lazy_factory("mufasa.data_processors.agg_clf_calculator",
                                 "AggregateClfCalculator"),
        backend_mp=_lazy_factory("mufasa.data_processors.agg_clf_counter_mp",
                                 "AggregateClfCalculatorMultiprocess"),
        needs={"classifiers"},
        extras=[
            ("first_occurrence",       "bool", True),
            ("event_count",            "bool", True),
            ("total_event_duration",   "bool", True),
            ("pct_of_session",         "bool", False),
            ("mean_event_duration",    "bool", True),
            ("median_event_duration",  "bool", False),
            ("mean_interval_duration", "bool", False),
            ("median_interval_duration","bool", False),
            ("frame_count",            "bool", False),
            ("video_length",           "bool", True),
            ("detailed_bout_data",     "bool", False),
            ("transpose",              "bool", False),
        ],
        kwargs_map={"classifiers": "classifiers"},  # 1:1 but explicit
    ),
    # ---------- Classifier by ROI ---------- #
    _AnalysisRoute(
        label="Classifier by ROI",
        backend_sp=_lazy_factory("mufasa.roi_tools.ROI_clf_calculator",
                                 "ROIClfCalculator"),
        backend_mp=_lazy_factory("mufasa.roi_tools.ROI_clf_calculator_mp",
                                 "ROIClfCalculatorMultiprocess"),
        needs={"classifiers", "body_parts"},
        extras=[
            ("total_time",     "bool", True),
            ("started_bouts",  "bool", False),
            ("ended_bouts",    "bool", False),
            ("detailed_bouts", "bool", False),
            ("transpose",      "bool", False),
        ],
        kwargs_map={"body_parts": "bp_names", "classifiers": "clf_names"},
    ),
    # ---------- Classifier by time bins ---------- #
    _AnalysisRoute(
        label="Classifier by time bins",
        backend_sp=_lazy_factory("mufasa.data_processors.timebins_clf_calculator",
                                 "TimeBinsClfCalculator"),
        needs={"classifiers", "bin_length"},
        extras=[
            ("first_occurrence",         "bool", True),
            ("event_count",              "bool", True),
            ("total_event_duration",     "bool", True),
            ("mean_event_duration",      "bool", True),
            ("median_event_duration",    "bool", False),
            ("mean_interval_duration",   "bool", False),
            ("median_interval_duration", "bool", False),
            ("include_timestamp",        "bool", False),
            ("transpose",                "bool", False),
        ],
        kwargs_map={"bin_length": "bin_length"},
    ),
    # ---------- Movement ---------- #
    _AnalysisRoute(
        label="Movement analysis",
        backend_sp=_lazy_factory("mufasa.data_processors.movement_calculator",
                                 "MovementCalculator"),
        needs={"body_parts"},
        extras=[
            ("threshold",         "float", 0.0, 0.0, 1.0, 0.05),
            ("distance",          "bool",  True),
            ("velocity",          "bool",  True),
            ("video_time_stamps", "bool",  False),
            ("transpose",         "bool",  False),
            ("frame_count",       "bool",  False),
            ("video_length",      "bool",  True),
        ],
    ),
    _AnalysisRoute(
        label="Movement analysis by time bins",
        backend_sp=_lazy_factory("mufasa.data_processors.timebins_movement_calculator",
                                 "TimeBinsMovementCalculator"),
        backend_mp=_lazy_factory("mufasa.data_processors.timebins_movement_calculator_mp",
                                 "TimeBinsMovementCalculatorMultiprocess"),
        needs={"body_parts", "bin_length"},
        extras=[
            ("threshold",         "float", 0.0, 0.0, 1.0, 0.05),
            ("distance",          "bool",  True),
            ("velocity",          "bool",  True),
            ("plots",             "bool",  False),
            ("transpose",         "bool",  False),
            ("include_timestamp", "bool",  False),
        ],
    ),
    # ---------- Distance ---------- #
    _AnalysisRoute(
        label="Distance between body-parts",
        backend_sp=_lazy_factory("mufasa.data_processors.distance_calculator",
                                 "DistanceCalculator"),
        needs={"body_parts"},
        extras=[
            ("bp_threshold",       "float", 0.0, 0.0, 1.0, 0.05),
            ("distance_threshold", "float", 0.0, 0.0, 1e6, 1.0),
            ("distance_mean",      "bool",  True),
            ("distance_median",    "bool",  False),
            ("detailed_data",      "bool",  False),
            ("transpose",          "bool",  False),
        ],
    ),
    _AnalysisRoute(
        label="Distance by time bins",
        backend_sp=_lazy_factory("mufasa.data_processors.distance_timbin_calculator",
                                 "DistanceTimeBinCalculator"),
        needs={"body_parts", "bin_length"},
        extras=[
            ("threshold",       "float", 0.0, 0.0, 1.0, 0.05),
            ("distance_mean",   "bool",  True),
            ("distance_median", "bool",  False),
            ("distance_var",    "bool",  False),
            ("transpose",       "bool",  False),
        ],
        kwargs_map={"bin_length": "time_bin"},
    ),
    # ---------- FSTTC ---------- #
    _AnalysisRoute(
        label="FSTTC (forward spike time-tiling)",
        backend_sp=_lazy_factory("mufasa.data_processors.fsttc_calculator",
                                 "FSTTCCalculator"),
        needs={"classifiers"},
        extras=[
            ("time_window",             "float", 2.0, 0.1, 60.0, 0.1),
            ("time_delta_at_onset",     "bool",  True),
            ("join_bouts_within_delta", "bool",  False),
            ("create_graphs",           "bool",  False),
        ],
        # FSTTCCalculator's backend takes `behavior_lst`, not
        # `classifiers`. Rename via the form's generic kwargs_map.
        kwargs_map={"classifiers": "behavior_lst"},
    ),
    # ---------- Boolean conditional ---------- #
    _AnalysisRoute(
        label="Boolean conditional slicing",
        backend_sp=_lazy_factory("mufasa.data_processors.boolean_conditional_calculator",
                                 "BooleanConditionalCalculator"),
        needs=set(),
        extras=[
            ("rules_csv", "str", "",
             "Path to a rules CSV (row per rule; condition expressions)"),
        ],
        kwargs_map={"rules_csv": "rules"},
    ),
    # ---------- Directing (data side) ---------- #
    _AnalysisRoute(
        label="Directing toward other animals — statistics",
        backend_sp=_lazy_factory("mufasa.data_processors.directing_other_animals_calculator",
                                 "DirectingOtherAnimalsAnalyzer"),
        needs=set(),
        extras=[
            ("bool_tables",                     "bool", True),
            ("summary_tables",                  "bool", True),
            ("append_bool_tables_to_features",  "bool", False),
            ("aggregate_statistics_tables",     "bool", True),
            ("left_ear_name",                   "str",  "Left_ear"),
            ("right_ear_name",                  "str",  "Right_ear"),
            ("nose_name",                       "str",  "Nose"),
        ],
    ),
]


# --------------------------------------------------------------------------- #
# The form
# --------------------------------------------------------------------------- #
class AnalysisForm(OperationForm):
    """Universal analysis form. Target dropdown drives the parameter
    layout; a handful of conditional selectors (classifiers, body-parts,
    bin length, severity mode) surface only where the backend needs them.
    """

    title = "Run analysis"
    description = ("Run a statistical analysis over project data. "
                   "Output goes to the project's log directory.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # --- Route selector ------------------------------------------- #
        self.target_cb = QComboBox(self)
        for r in ROUTES:
            self.target_cb.addItem(r.label)
        self.target_cb.currentIndexChanged.connect(self._on_route_changed)
        form.addRow("Analysis:", self.target_cb)

        # --- Conditional selectors ------------------------------------ #
        self.clf_picker = _ClassifierPicker(
            config_path=self.config_path, parent=self,
        )
        form.addRow("Classifiers:", self.clf_picker)

        self.bp_picker = _BodyPartPicker(
            config_path=self.config_path, parent=self,
        )
        form.addRow("Body-parts:", self.bp_picker)

        self.bin_length = QDoubleSpinBox(self)
        self.bin_length.setRange(0.1, 3600.0); self.bin_length.setValue(60.0)
        self.bin_length.setSuffix(" s")
        form.addRow("Bin length:", self.bin_length)

        # Severity mode (only for the severity route)
        self.severity_mode = QComboBox(self)
        self.severity_mode.addItems(["Bout-based", "Frame-based"])
        form.addRow("Severity mode:", self.severity_mode)

        # --- Cores (enabled iff route has MP backend) ---------------- #
        self.cores = QSpinBox(self)
        self.cores.setRange(1, max(1, linux_env.cpu_count()))
        self.cores.setValue(max(1, linux_env.cpu_count() // 2))
        form.addRow("Cores:", self.cores)

        # --- Extras ---------------------------------------------------- #
        self.extras_stack = QStackedWidget(self)
        self._extras_by_route: dict[int, _ExtrasFormBuilder] = {}
        for idx, r in enumerate(ROUTES):
            eb = _ExtrasFormBuilder(r.extras, parent=self.extras_stack)
            self.extras_stack.addWidget(eb.host)
            self._extras_by_route[idx] = eb
        form.addRow("Parameters:", self.extras_stack)

        self.body_layout.addLayout(form)
        self._on_route_changed(0)

    # ------------------------------------------------------------------ #
    # Route switching
    # ------------------------------------------------------------------ #
    def _current_route(self) -> _AnalysisRoute:
        return ROUTES[self.target_cb.currentIndex()]

    def _on_route_changed(self, idx: int) -> None:
        route = ROUTES[idx]
        # Surface or hide the conditional selectors based on `needs`
        self._set_row_visible(self.clf_picker, "classifiers" in route.needs)
        self._set_row_visible(self.bp_picker, "body_parts" in route.needs)
        self._set_row_visible(self.bin_length, "bin_length" in route.needs)
        self._set_row_visible(self.severity_mode, "severity_mode" in route.needs)
        # Cores: only meaningful when MP backend exists
        self.cores.setEnabled(route.backend_mp is not None)
        # Extras stack
        self.extras_stack.setCurrentIndex(idx)

    def _set_row_visible(self, field_widget: QWidget, visible: bool) -> None:
        """Find the QFormLayout row containing ``field_widget`` and show
        or hide both the field and its label."""
        for i in range(self.body_layout.count()):
            lo = self.body_layout.itemAt(i).layout()
            if not isinstance(lo, QFormLayout):
                continue
            for r in range(lo.rowCount()):
                item = lo.itemAt(r, QFormLayout.FieldRole)
                if item and item.widget() is field_widget:
                    field_widget.setVisible(visible)
                    lbl = lo.itemAt(r, QFormLayout.LabelRole)
                    if lbl and lbl.widget():
                        lbl.widget().setVisible(visible)
                    return

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        route = self._current_route()
        args: dict = {
            "route": route,
            "config_path": self.config_path,
            "cores": int(self.cores.value()),
            "extras": self._extras_by_route[self.target_cb.currentIndex()].to_kwargs(),
        }
        if "classifiers" in route.needs:
            sel = self.clf_picker.selected()
            if not sel:
                raise ValueError("Select at least one classifier.")
            args["classifiers"] = sel
        if "body_parts" in route.needs:
            bps = self.bp_picker.selected()
            if not bps:
                raise ValueError("Select at least one body-part.")
            args["body_parts"] = [bp for (_a, bp) in bps]
        if "bin_length" in route.needs:
            args["bin_length"] = float(self.bin_length.value())
        if "severity_mode" in route.needs:
            args["severity_mode"] = ("bout" if self.severity_mode.currentIndex() == 0
                                     else "frame")
        return args

    def target(self, *, route: _AnalysisRoute, config_path: str,
               cores: int, extras: dict, **params) -> None:
        km = route.kwargs_map
        kwargs: dict = {"config_path": config_path}

        # Severity has a special routing: bout vs frame → different backend
        if "severity_mode" in route.needs:
            mode = params.get("severity_mode", "bout")
            if mode == "bout":
                backend = _lazy_factory(
                    "mufasa.data_processors.severity_bout_based_calculator",
                    "SeverityBoutCalculator",
                )
            else:
                backend = _lazy_factory(
                    "mufasa.data_processors.severity_frame_based_calculator",
                    "SeverityFrameCalculator",
                )
            # Severity backends take a `settings` dict rather than kwargs
            settings = dict(extras)
            if "classifiers" in params:
                settings["classifiers"] = params["classifiers"]
            runner = backend(config_path=config_path, settings=settings)
            if runner is not None and hasattr(runner, "run"):
                runner.run()
            return

        # Copy through the conditional selectors, honoring kwargs_map
        for key in ("classifiers", "body_parts", "bin_length"):
            if key in params:
                kwargs[km.get(key, key)] = params[key]
        # Extras pass-through with optional rename
        for k, v in extras.items():
            kwargs[km.get(k, k)] = v

        # Backend selection
        backend = None
        if cores > 1 and route.backend_mp is not None:
            backend = route.backend_mp
            kwargs.setdefault("core_cnt", cores)
        else:
            backend = route.backend_sp
        runner = backend(**kwargs)
        if runner is not None and hasattr(runner, "run"):
            runner.run()


__all__ = ["AnalysisForm", "ROUTES"]
