"""
mufasa.ui_qt.forms.visualizations
=================================

One consolidated form for project-level visualisations. Replaces
12 legacy popups:

* :class:`HeatmapClfPopUp`
* :class:`HeatmapLocationPopup`
* :class:`PathPlotPopUp`
* :class:`GanttPlotPopUp`
* :class:`DistancePlotterPopUp`
* :class:`DataPlotterPopUp`
* :class:`SklearnVisualizationPopUp`
* :class:`VisualizeClassificationProbabilityPopUp`
* :class:`ValidationVideoPopUp`
* :class:`VisualizePoseInFolderPopUp`
* :class:`DirectingOtherAnimalsVisualizerPopUp`
* :class:`DirectingAnimalToBodyPartVisualizerPopUp`

Plus simplified launchers for:

* :class:`EzPathPlotPopUp` / :class:`QuickLineplotPopup` /
  :class:`MakePathPlotPopUp` — one form, three entry-points collapsed
  into the "Quick path plot" mode of :class:`VisualizationForm`.

Plus :class:`YoloPoseVisualizerPopUp` and :class:`BlobVisualizerPopUp`,
which don't read project config but fit the same form shape.

Design — declarative route table
--------------------------------

Following :mod:`mufasa.ui_qt.forms.data_import`, each visualisation
has a ``_VizRoute`` record declaring: target backend, which fields
to show, whether it's single-core / multi-core capable, and any
required inputs. The UI reads the route's ``fields`` list to build
its form at route-switch time — no per-viz hand-coding.

This keeps the diff cost per added visualisation at ~15 lines,
and means a backend signature change updates exactly one record.

Only a handful of common-enough fields get dedicated widgets
(``frame/video/last-frame`` toggles, ``cores``, ``gpu``). The long
tail of per-viz style knobs (font size, opacities, palettes, line
widths) is handled via a generic "extras" form generated from a
list of field descriptors — each descriptor is:
``(field_name, kind, default, [extra...])`` with ``kind`` one of
``int``, ``float``, ``bool``, ``choice``, ``color``, ``str``.

For any route that outgrows this generic mechanism, a handcrafted
:class:`QWidget` can be plugged in as the extras panel instead —
same escape hatch as the converter form's labelme-df extras.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox,
                               QDoubleSpinBox, QFormLayout, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.forms.data_import import _PathField
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Generic-extras form builder
# --------------------------------------------------------------------------- #
class _ColorButton(QPushButton):
    """Tiny colour picker — single button that opens QColorDialog."""

    def __init__(self, default: str = "#000000", parent: Optional[QWidget] = None) -> None:
        super().__init__(default, parent)
        self._color = default
        self.setStyleSheet(f"background-color: {default};")
        self.clicked.connect(self._pick)

    def _pick(self) -> None:
        from PySide6.QtGui import QColor
        start = QColor(self._color)
        col = QColorDialog.getColor(start, self)
        if col.isValid():
            self._color = col.name()
            self.setText(self._color)
            self.setStyleSheet(f"background-color: {col.name()}; color: {'white' if col.lightness() < 128 else 'black'};")

    @property
    def color(self) -> str:
        return self._color


class _ExtrasFormBuilder:
    """Builds a ``QFormLayout`` from a list of field descriptors and
    offers ``to_kwargs()`` to collect their values. Each descriptor is
    a tuple::

        (backend_kwarg_name, kind, default, *extra)

    where ``kind`` is one of:

    * ``"int"`` — QSpinBox; extras: (min, max)
    * ``"float"`` — QDoubleSpinBox; extras: (min, max, step)
    * ``"bool"`` — QCheckBox (passed to form as a row; no label)
    * ``"choice"`` — QComboBox; extras: list-of-options
    * ``"color"`` — _ColorButton
    * ``"str"`` — QLineEdit; extras: (placeholder,)

    Label text is derived from the backend kwarg name by replacing
    underscores with spaces and title-casing.
    """

    def __init__(self, descriptors: list[tuple], parent: Optional[QWidget] = None
                 ) -> None:
        self.descriptors = descriptors
        self.host = QWidget(parent)
        self._form = QFormLayout(self.host)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._widgets: dict[str, QWidget] = {}
        for desc in descriptors:
            self._add(desc)

    def _label(self, name: str) -> str:
        return name.replace("_", " ").capitalize() + ":"

    def _add(self, desc: tuple) -> None:
        name, kind, default, *extra = desc
        if kind == "int":
            lo, hi = (extra + [0, 1_000_000])[:2] if extra else (0, 1_000_000)
            w = QSpinBox(); w.setRange(int(lo), int(hi)); w.setValue(int(default))
            self._form.addRow(self._label(name), w)
        elif kind == "float":
            lo, hi, step = (extra + [0.0, 1.0, 0.05])[:3]
            w = QDoubleSpinBox(); w.setRange(float(lo), float(hi))
            w.setSingleStep(float(step)); w.setValue(float(default))
            self._form.addRow(self._label(name), w)
        elif kind == "bool":
            w = QCheckBox(self._label(name).rstrip(":"))
            w.setChecked(bool(default))
            self._form.addRow("", w)
        elif kind == "choice":
            opts = extra[0] if extra else []
            w = QComboBox()
            w.addItems([str(o) for o in opts])
            if default in opts:
                w.setCurrentIndex(opts.index(default))
            self._form.addRow(self._label(name), w)
        elif kind == "color":
            w = _ColorButton(str(default))
            self._form.addRow(self._label(name), w)
        elif kind == "str":
            placeholder = extra[0] if extra else ""
            w = QLineEdit(); w.setPlaceholderText(placeholder)
            if default:
                w.setText(str(default))
            self._form.addRow(self._label(name), w)
        else:
            raise ValueError(f"Unknown field kind: {kind!r}")
        self._widgets[name] = w

    def to_kwargs(self) -> dict:
        out: dict[str, Any] = {}
        for desc in self.descriptors:
            name, kind, *_ = desc
            w = self._widgets[name]
            if kind == "int":
                out[name] = int(w.value())
            elif kind == "float":
                out[name] = float(w.value())
            elif kind == "bool":
                out[name] = bool(w.isChecked())
            elif kind == "choice":
                out[name] = w.currentText()
            elif kind == "color":
                out[name] = w.color
            elif kind == "str":
                out[name] = w.text().strip()
        return out


# --------------------------------------------------------------------------- #
# Route table
# --------------------------------------------------------------------------- #
def _lazy_factory(modpath: str, classname: str) -> Callable[..., Any]:
    def _factory(**kw):
        mod = __import__(modpath, fromlist=[classname])
        return getattr(mod, classname)(**kw)
    _factory.__name__ = f"{modpath}.{classname}"
    return _factory


@dataclass
class _VizRoute:
    """Declaration of a single visualisation route.

    Attributes
    ----------
    label : str
        Menu entry shown in the target dropdown.
    scope_kind : str
        "project"  — reads config_path from the workbench and processes
                     all project data.
        "file"     — standalone, takes an explicit data_path.
    extras : list[tuple]
        Route-specific fields (see _ExtrasFormBuilder for format).
    backend_sp : callable | None
        Single-process backend factory (fallback when cores == 1 or MP
        backend missing).
    backend_mp : callable | None
        Multi-process backend factory.
    needs_video : bool
    needs_save_dir : bool
    common_toggles : set[str]
        Which of {frame, video, last_frame, gpu} common toggles to
        surface for this route.
    kwargs_map : dict
        Renames for generic UI field names → backend kwarg names
        (e.g. "save_path" → "save_dir"). Missing entries mean 1:1.
    default_kwargs : dict
        Backend kwargs the form supplies unconditionally when not
        overridden by user input. Used for required dict-typed kwargs
        like ``style_attr``, ``line_attr``, ``animal_attr`` — backends
        need them present but fill defaults from an empty dict.
    data_paths_source : str | None
        If set, the form auto-populates ``data_paths`` by scanning
        ``{project}/csv/{this subdir}/``. Valid: ``"machine_results"``,
        ``"outlier_corrected_movement_location"``, ``"features_extracted"``,
        ``"targets_inserted"``. Required for backends like
        HeatMapperClfSingleCore that iterate per-video CSVs.
    data_path_source : str | None
        Singular variant — passes one file as ``data_path`` (probability
        plotter, directing-to-bodypart).
    """
    label: str
    scope_kind: str = "project"
    extras: list = field(default_factory=list)
    backend_sp: Optional[Callable[..., Any]] = None
    backend_mp: Optional[Callable[..., Any]] = None
    needs_video: bool = False
    needs_save_dir: bool = False
    common_toggles: set = field(default_factory=lambda: set())
    kwargs_map: dict = field(default_factory=dict)
    default_kwargs: dict = field(default_factory=dict)
    data_paths_source: Optional[str] = None
    data_path_source: Optional[str] = None


# --------------------------------------------------------------------------- #
# Route declarations
# --------------------------------------------------------------------------- #
# Keeping these lean — adding niche fields later is a one-line append to
# ``extras``. Each entry corresponds to one legacy popup.
ROUTES: list[_VizRoute] = [
    # -------- heatmaps -------- #
    _VizRoute(
        label="Heatmap — classifier",
        scope_kind="project",
        backend_mp=_lazy_factory("mufasa.plotting.heat_mapper_clf_mp",
                                 "HeatMapperClfMultiprocess"),
        backend_sp=_lazy_factory("mufasa.plotting.heat_mapper_clf",
                                 "HeatMapperClfSingleCore"),
        common_toggles={"frame", "video"},
        extras=[
            ("bodypart",        "str",    "", "Body-part name, e.g. 'Nose'"),
            ("clf_name",        "str",    "", "Classifier name"),
            ("heatmap_opacity", "float",  0.5, 0.0, 1.0, 0.05),
            ("show_legend",     "bool",   True),
            ("show_keypoint",   "bool",   True),
            ("min_seconds",     "float",  0.0, 0.0, 3600.0, 0.5),
        ],
        # Backend reads per-video machine_results CSVs; needs data_paths
        # (plural). style_attr required but an empty dict = backend
        # fills its own defaults.
        data_paths_source="machine_results",
        default_kwargs={"style_attr": {}},
    ),
    _VizRoute(
        label="Heatmap — location",
        scope_kind="project",
        backend_mp=_lazy_factory("mufasa.plotting.heat_mapper_location_mp",
                                 "HeatMapperLocationMultiprocess"),
        backend_sp=_lazy_factory("mufasa.plotting.heat_mapper_location",
                                 "HeatmapperLocationSingleCore"),
        common_toggles={"frame", "video"},
        extras=[
            ("bodypart",        "str",   "", "Body-part name"),
            ("heatmap_opacity", "float", 0.5, 0.0, 1.0, 0.05),
            ("show_legend",     "bool",  True),
            ("show_keypoint",   "bool",  True),
            ("min_seconds",     "float", 0.0, 0.0, 3600.0, 0.5),
        ],
        data_paths_source="outlier_corrected_movement_location",
        default_kwargs={"style_attr": {}},
    ),
    # -------- paths -------- #
    _VizRoute(
        label="Path plot",
        scope_kind="project",
        backend_mp=_lazy_factory("mufasa.plotting.path_plotter_mp",
                                 "PathPlotterMulticore"),
        backend_sp=_lazy_factory("mufasa.plotting.path_plotter",
                                 "PathPlotterSingleCore"),
        common_toggles={"frame", "video"},
        extras=[
            ("print_animal_names", "bool", True),
            ("roi",                "bool", False),
        ],
        data_paths_source="outlier_corrected_movement_location",
        default_kwargs={"animal_attr": {}},
    ),
    _VizRoute(
        label="Path plot — quick (one file)",
        scope_kind="file",
        backend_sp=_lazy_factory("mufasa.plotting.ez_path_plot", "EzPathPlot"),
        needs_video=True,
        extras=[
            ("body_part",      "str",  "", "Body-part name"),
            ("bg_color",       "color","#FFFFFF"),
            ("line_color",     "color","#FF0000"),
            ("line_thickness", "int",  2, 1, 20),
            ("circle_size",    "int",  5, 1, 50),
            ("last_frm_only",  "bool", False),
            ("size",           "int",  800, 100, 4096),
            ("fps",            "int",  30,  1, 240),
        ],
        kwargs_map={"source_path": "data_path", "video_path": "video_path"},
    ),
    # -------- gantt -------- #
    _VizRoute(
        label="Gantt (behaviour timeline)",
        scope_kind="project",
        backend_mp=_lazy_factory("mufasa.plotting.gantt_creator_mp",
                                 "GanttCreatorMultiprocess"),
        backend_sp=_lazy_factory("mufasa.plotting.gantt_creator",
                                 "GanttCreatorSingleProcess"),
        common_toggles={"frame", "video", "last_frame"},
        extras=[
            ("width",        "int",   640, 100, 4096),
            ("height",       "int",   480, 100, 4096),
            ("font_size",    "int",   12,  6, 96),
            ("font_rotation","int",   45,  0, 360),
            ("bar_opacity",  "float", 1.0, 0.0, 1.0, 0.05),
            ("palette",      "str",   "Set2"),
            ("hhmmss",       "bool",  True),
        ],
    ),
    # -------- distance -------- #
    _VizRoute(
        label="Distance plot",
        scope_kind="project",
        backend_mp=_lazy_factory("mufasa.plotting.distance_plotter_mp",
                                 "DistancePlotterMultiCore"),
        backend_sp=_lazy_factory("mufasa.plotting.distance_plotter",
                                 "DistancePlotterSingleCore"),
        common_toggles={"frame", "video"},
        extras=[],
        data_paths_source="outlier_corrected_movement_location",
        # Both backends require style_attr + line_attr; MP additionally
        # requires final_img (bool) — set False (write video, not a
        # summary image).
        default_kwargs={"style_attr": {}, "line_attr": {},
                        "final_img": False},
    ),
    # -------- data plotter -------- #
    _VizRoute(
        label="Data plotter (feature overlay)",
        scope_kind="project",
        backend_sp=_lazy_factory("mufasa.plotting.data_plotter", "DataPlotter"),
        common_toggles={"frame", "video"},
        extras=[
            ("bg_clr",         "color", "#FFFFFF"),
            ("header_clr",     "color", "#000000"),
            ("font_thickness", "int",   2, 1, 10),
            ("decimals",       "int",   2, 0, 8),
        ],
        data_paths_source="features_extracted",
        # body_parts required — empty list means "detect from project
        # pose config".
        default_kwargs={"body_parts": []},
    ),
    # -------- classifier overlays -------- #
    _VizRoute(
        label="Classifier results overlay (sklearn)",
        scope_kind="project",
        backend_mp=_lazy_factory("mufasa.plotting.plot_clf_results_mp",
                                 "PlotSklearnResultsMultiProcess"),
        backend_sp=_lazy_factory("mufasa.plotting.plot_clf_results",
                                 "PlotSklearnResultsSingleCore"),
        common_toggles={"frame", "video", "gpu"},
        extras=[
            ("rotate",          "bool",  False),
            ("animal_names",    "bool",  True),
            ("show_pose",       "bool",  True),
            ("show_confidence", "bool",  True),
            ("show_gantt",      "bool",  False),
            ("font_size",       "float", 0.7, 0.1, 5.0, 0.1),
            ("circle_size",     "int",   5, 1, 50),
            ("print_timer",     "bool",  True),
            ("bbox",            "bool",  False),
        ],
    ),
    _VizRoute(
        label="Classifier probability plot",
        scope_kind="project",
        backend_mp=_lazy_factory("mufasa.plotting.probability_plot_creator_mp",
                                 "TresholdPlotCreatorMultiprocess"),
        backend_sp=_lazy_factory("mufasa.plotting.probability_plot_creator",
                                 "TresholdPlotCreatorSingleProcess"),
        common_toggles={"frame", "video"},
        extras=[
            ("clf_name",        "str",   ""),
            ("font_size",       "int",   10, 6, 96),
            ("line_width",      "int",   2,  1, 20),
            ("line_color",      "color", "#0000FF"),
            ("line_opacity",    "float", 1.0, 0.0, 1.0, 0.05),
            ("show_thresholds", "bool",  True),
        ],
        # Singular: these backends iterate over one file at a time.
        data_path_source="machine_results",
    ),
    _VizRoute(
        label="Classifier validation clips",
        scope_kind="project",
        backend_sp=_lazy_factory("mufasa.plotting.clf_validator",
                                 "ClassifierValidationClips"),
        extras=[
            ("clf_name",      "str",   ""),
            ("window",        "int",   5, 1, 60),
            ("concat_video",  "bool",  False),
            ("clips",         "bool",  True),
            ("video_speed",   "float", 1.0, 0.1, 10.0, 0.1),
        ],
        data_paths_source="machine_results",
    ),
    _VizRoute(
        label="Validation video (single model)",
        scope_kind="project",
        backend_mp=_lazy_factory("mufasa.plotting.single_run_model_validation_video_mp",
                                 "ValidateModelOneVideoMultiprocess"),
        backend_sp=_lazy_factory("mufasa.plotting.single_run_model_validation_video",
                                 "ValidateModelOneVideo"),
        common_toggles={},
        extras=[
            ("discrimination_threshold", "float", 0.5, 0.0, 1.0, 0.01),
            ("shortest_bout",            "int",   1, 0, 10000),
            ("show_pose",                "bool",  True),
            ("show_animal_names",        "bool",  True),
            ("create_gantt",             "bool",  False),
        ],
    ),
    # -------- annotated bouts -------- #
    _VizRoute(
        label="Annotated bouts overlay",
        scope_kind="project",
        backend_sp=_lazy_factory("mufasa.plotting.annotation_videos",
                                 "PlotAnnotatedBouts"),
        common_toggles={"gpu"},
        extras=[
            ("animal_names", "bool",  True),
            ("show_pose",    "bool",  True),
            ("pre_window",   "int",   2, 0, 60),
            ("post_window",  "int",   2, 0, 60),
            ("font_size",    "float", 0.7, 0.1, 5.0, 0.1),
            ("circle_size",  "int",   5, 1, 50),
            ("video_timer",  "bool",  True),
            ("bbox",         "bool",  False),
        ],
    ),
    # -------- pose-in-dir -------- #
    _VizRoute(
        label="Pose tracking (pose over videos)",
        scope_kind="file",
        backend_mp=_lazy_factory("mufasa.plotting.pose_plotter_mp",
                                 "PosePlotterMultiProcess"),
        backend_sp=_lazy_factory("mufasa.plotting.pose_plotter", "PosePlotter"),
        needs_save_dir=True,
        common_toggles={"gpu"},
        extras=[
            ("circle_size",    "int",   5, 1, 50),
            ("bbox",           "bool",  False),
            ("center_of_mass", "bool",  False),
            ("sample_time",    "int",   0, 0, 600),
        ],
        kwargs_map={"source_path": "data_path", "save_path": "out_dir"},
    ),
    # -------- YOLO / blob (non-project, external data) -------- #
    _VizRoute(
        label="YOLO pose predictions (external)",
        scope_kind="file",
        backend_sp=_lazy_factory("mufasa.plotting.yolo_pose_visualizer",
                                 "YOLOPoseVisualizer"),
        needs_video=True,
        needs_save_dir=True,
        extras=[
            ("threshold",   "float", 0.5, 0.0, 1.0, 0.01),
            ("thickness",   "int",   2, 1, 20),
            ("circle_size", "int",   5, 1, 50),
            ("bbox",        "bool",  True),
            ("skeleton",    "bool",  True),
            ("sample_n",    "int",   0, 0, 100000),
        ],
        kwargs_map={"source_path": "data_path", "video_path": "video_path",
                    "save_path": "save_dir"},
    ),
    _VizRoute(
        label="Blob tracker output (external)",
        scope_kind="file",
        backend_sp=_lazy_factory("mufasa.plotting.blob_visualizer",
                                 "BlobVisualizer"),
        needs_video=True,
        needs_save_dir=True,
        extras=[
            ("shape_opacity", "float", 0.5, 0.0, 1.0, 0.05),
            ("bg_opacity",    "float", 1.0, 0.0, 1.0, 0.05),
            ("circle_size",   "int",   5, 1, 50),
            ("hull",          "bool",  True),
        ],
        kwargs_map={"source_path": "data_path", "video_path": "video_path",
                    "save_path": "save_dir"},
    ),
    # -------- directing -------- #
    _VizRoute(
        label="Directing — toward other animals",
        scope_kind="project",
        backend_mp=_lazy_factory("mufasa.plotting.directing_animals_visualizer_mp",
                                 "DirectingOtherAnimalsVisualizerMultiprocess"),
        backend_sp=_lazy_factory("mufasa.plotting.directing_animals_visualizer",
                                 "DirectingOtherAnimalsVisualizer"),
        extras=[],
        needs_video=True,
        default_kwargs={"style_attr": {}},
    ),
    _VizRoute(
        label="Directing — toward body-part",
        scope_kind="project",
        backend_sp=_lazy_factory(
            "mufasa.plotting.directing_animals_to_bodypart_visualizer",
            "DirectingAnimalsToBodyPartVisualizer",
        ),
        extras=[],
        # Backend takes singular data_path (not plural). Point at the
        # directing-data CSV that the directing-statistics calculator
        # produces.
        data_path_source="directing_data",
        default_kwargs={"style_attr": {}},
    ),
]


# --------------------------------------------------------------------------- #
# The form
# --------------------------------------------------------------------------- #
class VisualizationForm(OperationForm):
    """Universal visualisation form. Target dropdown switches the
    entire parameter set; common toggles + extras are built on the
    fly from the route declaration."""

    title = "Create visualisation"
    description = ("Render a visualisation of project data or external "
                   "pose/annotation files. Choose a target; the form "
                   "adapts to its input requirements.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # Target selector
        self.target_cb = QComboBox(self)
        for r in ROUTES:
            self.target_cb.addItem(r.label)
        self.target_cb.currentIndexChanged.connect(self._on_route_changed)
        form.addRow("Visualisation:", self.target_cb)

        # Scope-dependent source/video/save pickers. Hidden when not
        # relevant for the active route.
        self.source_picker = _PathField(is_file=True,
                                        placeholder="Source file (data)…")
        self.source_row = ("Source:", self.source_picker)
        form.addRow(*self.source_row)

        self.video_picker = _PathField(
            is_file=True,
            file_filter="Videos (*.mp4 *.avi *.mov *.mkv *.webm)",
            placeholder="Video file…",
        )
        form.addRow("Video:", self.video_picker)

        self.save_picker = _PathField(is_file=False,
                                      placeholder="Save directory…")
        form.addRow("Save to:", self.save_picker)

        # Common toggles row
        self._common_row = QHBoxLayout()
        self.frame_cb = QCheckBox("Save frames", self)
        self.video_cb = QCheckBox("Save video", self); self.video_cb.setChecked(True)
        self.last_frame_cb = QCheckBox("Save last-frame summary", self)
        self.gpu_cb = QCheckBox("Use GPU", self)
        self.gpu_cb.setChecked(linux_env.nvenc_available())
        for w in (self.frame_cb, self.video_cb, self.last_frame_cb, self.gpu_cb):
            self._common_row.addWidget(w)
        self._common_row.addStretch()
        common_host = QWidget(self); common_host.setLayout(self._common_row)
        form.addRow("Outputs:", common_host)

        # Cores — always surfaced when MP backend exists
        self.cores = QSpinBox(self)
        self.cores.setRange(1, max(1, linux_env.cpu_count()))
        self.cores.setValue(max(1, linux_env.cpu_count() // 2))
        form.addRow("Cores:", self.cores)

        # Extras stack (built per-route on switch)
        self.extras_stack = QStackedWidget(self)
        self._extras_by_route: dict[int, _ExtrasFormBuilder] = {}
        form.addRow("Parameters:", self.extras_stack)

        # Pre-build all extras panels up-front (cheap; no heavy widgets)
        for idx, r in enumerate(ROUTES):
            eb = _ExtrasFormBuilder(r.extras, parent=self.extras_stack)
            self.extras_stack.addWidget(eb.host)
            self._extras_by_route[idx] = eb

        self.body_layout.addLayout(form)
        self._on_route_changed(0)

    # ------------------------------------------------------------------ #
    # Route switching
    # ------------------------------------------------------------------ #
    def _current_route(self) -> _VizRoute:
        return ROUTES[self.target_cb.currentIndex()]

    def _on_route_changed(self, idx: int) -> None:
        route = ROUTES[idx]
        # Scope: project routes hide the source picker (config_path
        # comes from the workbench); file routes show it.
        is_file = route.scope_kind == "file"
        self.source_picker.setVisible(is_file)
        self._set_label_visible(self.source_picker, is_file)

        self.video_picker.setVisible(route.needs_video)
        self._set_label_visible(self.video_picker, route.needs_video)

        self.save_picker.setVisible(route.needs_save_dir)
        self._set_label_visible(self.save_picker, route.needs_save_dir)

        # Common toggles: show only the ones this route uses
        self.frame_cb.setVisible("frame" in route.common_toggles)
        self.video_cb.setVisible("video" in route.common_toggles)
        self.last_frame_cb.setVisible("last_frame" in route.common_toggles)
        self.gpu_cb.setVisible("gpu" in route.common_toggles)
        has_any_toggle = bool(route.common_toggles)
        # Hide the row label if no toggles apply
        for w in (self.frame_cb, self.video_cb, self.last_frame_cb, self.gpu_cb):
            if not w.isVisible():
                continue

        # Cores: only meaningful if the route has a multi-process backend
        self.cores.setEnabled(route.backend_mp is not None)

        # Extras stack — swap to the matching panel
        self.extras_stack.setCurrentIndex(idx)

    def _set_label_visible(self, field_widget: QWidget, visible: bool) -> None:
        """Find the QFormLayout row containing ``field_widget`` and show
        or hide its label as well (Qt doesn't hide labels when you hide
        the field)."""
        for i in range(self.body_layout.count()):
            lo = self.body_layout.itemAt(i).layout()
            if not isinstance(lo, QFormLayout):
                continue
            for r in range(lo.rowCount()):
                item = lo.itemAt(r, QFormLayout.FieldRole)
                if item and item.widget() is field_widget:
                    lbl = lo.itemAt(r, QFormLayout.LabelRole)
                    if lbl and lbl.widget():
                        lbl.widget().setVisible(visible)
                    return

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        route = self._current_route()
        if route.scope_kind == "project":
            if not self.config_path:
                raise RuntimeError("No project loaded.")
            source = None
        else:
            source = self.source_picker.path
            if not source:
                raise ValueError("Source data path is required.")

        video = self.video_picker.path if route.needs_video else None
        if route.needs_video and not video:
            raise ValueError("This visualisation requires a video.")

        save = self.save_picker.path if route.needs_save_dir else None
        if route.needs_save_dir and not save:
            raise ValueError("This visualisation requires a save directory.")

        extras = self._extras_by_route[self.target_cb.currentIndex()].to_kwargs()

        common = {}
        if "frame" in route.common_toggles:
            common["frame_setting"] = bool(self.frame_cb.isChecked())
        if "video" in route.common_toggles:
            common["video_setting"] = bool(self.video_cb.isChecked())
        if "last_frame" in route.common_toggles:
            common["last_frame"] = bool(self.last_frame_cb.isChecked())
        if "gpu" in route.common_toggles:
            common["gpu"] = bool(self.gpu_cb.isChecked())

        return {
            "route": route,
            "config_path": self.config_path,
            "source": source,
            "video":  video,
            "save":   save,
            "extras": extras,
            "common": common,
            "cores":  int(self.cores.value()),
        }

    def target(self, *, route: _VizRoute, config_path: Optional[str],
               source: Optional[str], video: Optional[str],
               save: Optional[str], extras: dict, common: dict,
               cores: int) -> None:
        import configparser
        from pathlib import Path as _P

        km = route.kwargs_map
        kwargs: dict = {}

        if route.scope_kind == "project":
            kwargs["config_path"] = config_path
        else:
            kwargs[km.get("source_path", "data_path")] = source

        if route.needs_video:
            kwargs[km.get("video_path", "video_path")] = video
        if route.needs_save_dir:
            kwargs[km.get("save_path", "save_dir")] = save

        # Auto-populate data_paths / data_path from a project subdir.
        # Many plotting backends iterate per-video CSVs they can't
        # synthesise themselves; declaring data_paths_source on the
        # route lets us fill this in without a picker widget.
        if route.data_paths_source or route.data_path_source:
            if not config_path:
                raise RuntimeError(
                    f"{route.label!r} requires a loaded project "
                    "(config_path) to locate its data files."
                )
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            proj = cfg.get("General settings", "project_path", fallback=None)
            file_type = cfg.get("General settings", "workflow_file_type",
                                fallback="csv")
            if not proj:
                raise RuntimeError(
                    f"{route.label!r}: project_config.ini missing "
                    "General settings → project_path."
                )
            subdir = route.data_paths_source or route.data_path_source
            src_dir = _P(proj) / "csv" / subdir
            if not src_dir.is_dir():
                raise RuntimeError(
                    f"{route.label!r}: data source directory not found: "
                    f"{src_dir}. Run the preceding pipeline step first."
                )
            files = sorted(str(p) for p in src_dir.iterdir()
                           if p.suffix.lstrip(".") == file_type)
            if not files:
                raise RuntimeError(
                    f"{route.label!r}: no .{file_type} files in {src_dir}."
                )
            if route.data_paths_source:
                kwargs[km.get("data_paths", "data_paths")] = files
            else:
                # Singular: take the first file as the canonical input.
                kwargs[km.get("data_path", "data_path")] = files[0]

        # Apply route-level defaults *before* user extras so user fields
        # override (e.g. a user-defined style_attr wins over the
        # empty-dict fallback).
        for k, v in route.default_kwargs.items():
            kwargs.setdefault(km.get(k, k), v)

        # Common toggles + user extras (after rename pass)
        for k, v in common.items():
            kwargs[km.get(k, k)] = v
        for k, v in extras.items():
            kwargs[km.get(k, k)] = v

        # Choose single-core vs multi-core backend
        backend = None
        if cores > 1 and route.backend_mp is not None:
            backend = route.backend_mp
            kwargs.setdefault("core_cnt", cores)
        elif route.backend_sp is not None:
            backend = route.backend_sp
        elif route.backend_mp is not None:
            backend = route.backend_mp
            kwargs.setdefault("core_cnt", 1)
        if backend is None:
            raise NotImplementedError(
                f"No backend defined for {route.label!r}."
            )

        # Defensive signature filter: drop kwargs the backend doesn't
        # accept. Some form extras (sliders, toggles) exist for UX but
        # don't map cleanly onto backend params. Without this, passing
        # e.g. `heatmap_opacity` to a backend that doesn't accept it
        # would TypeError out before the runner starts.
        try:
            import inspect
            f_name = getattr(backend, "__name__", "")
            if "." in f_name:
                mod_path, _, cls_name = f_name.rpartition(".")
                mod = __import__(mod_path, fromlist=[cls_name])
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    sig = inspect.signature(cls.__init__)
                    accepts = set(sig.parameters) - {"self"}
                    has_varkw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    )
                    if not has_varkw:
                        kwargs = {k: v for k, v in kwargs.items()
                                  if k in accepts}
        except Exception:
            # Best-effort — if anything in the filter fails, pass
            # everything and let the backend raise naturally.
            pass

        runner = backend(**kwargs)
        if runner is not None and hasattr(runner, "run"):
            runner.run()


__all__ = ["VisualizationForm", "ROUTES"]
