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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QWidget,
)

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.forms.data_import import _PathField
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Generic-extras form builder
# --------------------------------------------------------------------------- #
class _ColorButton(QPushButton):
    """Tiny colour picker — single button that opens QColorDialog."""

    def __init__(self, default: str = "#000000", parent: QWidget | None = None) -> None:
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
    * ``"list"`` — QLineEdit collecting a comma-separated list of
      strings; produces ``list[str]`` at ``to_kwargs`` time.
      Default may be a list (rendered as comma-joined text) or
      a pre-joined string. Empty/whitespace tokens are dropped.
      Patch 122be: lets list-typed backend kwargs (body_parts,
      cue_light_names, arm_names, …) be declared natively
      instead of via a ``kwargs_transform`` lambda.
    * ``"dict"`` — QLineEdit collecting a JSON object as text;
      produces ``dict`` at ``to_kwargs`` time. Default may be
      a dict (rendered as compact JSON) or a pre-formatted JSON
      string. Empty input yields ``{}``. Malformed JSON also
      yields ``{}`` (silent fallback — backends typically have
      their own defaults for missing keys).
      Patch 122bg: unlocks routes whose backend needs a config
      dict (e.g. CircularFeaturePlotter's ``settings``).
    * ``"pickle"`` — _PathField (file picker) collecting a
      ``.pkl`` path; at ``to_kwargs`` time the form opens the
      file with :func:`pickle.load` and returns the
      deserialized object. Empty path or load error yields
      ``None`` (backend will raise its usual missing-arg
      error, which is clearer than the form would produce).
      extras: (file_filter, placeholder).
      Patch 122bh: unlocks routes whose backend takes a
      complex Python object that can't be JSON-serialized
      (e.g. GeometryPlotter's ``geometries`` arg, which is
      ``List[List[Shapely-object]]``). Users construct
      geometries in a Python script, pickle them, and feed
      the file to the form. SECURITY: pickle.load executes
      arbitrary code from the file — only load files you
      created or trust. The form does no sandboxing; this
      matches the trust model of a single-user desktop tool.
    * ``"file"`` — _PathField (file picker); extras: (file_filter, placeholder)

    Label text is derived from the backend kwarg name by replacing
    underscores with spaces and title-casing.
    """

    def __init__(self, descriptors: list[tuple], parent: QWidget | None = None
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
        elif kind == "list":
            # Patch 122be: comma-separated list. Default may be a
            # list (rendered as comma-joined text) or a pre-joined
            # string. Placeholder defaults to a hint about the
            # comma format if not provided.
            placeholder = extra[0] if extra else (
                "comma-separated, e.g. Item_1,Item_2"
            )
            w = QLineEdit(); w.setPlaceholderText(placeholder)
            if isinstance(default, (list, tuple)):
                if default:
                    w.setText(",".join(str(x) for x in default))
            elif default:
                w.setText(str(default))
            self._form.addRow(self._label(name), w)
        elif kind == "dict":
            # Patch 122bg: JSON-as-text input for dict-typed
            # backend kwargs. Default may be a dict (rendered as
            # compact JSON) or a pre-formatted JSON string.
            placeholder = extra[0] if extra else (
                'JSON object, e.g. {"key": "value"}'
            )
            w = QLineEdit(); w.setPlaceholderText(placeholder)
            if isinstance(default, dict):
                if default:
                    import json
                    w.setText(json.dumps(default, separators=(",", ":")))
            elif default:
                w.setText(str(default))
            self._form.addRow(self._label(name), w)
        elif kind == "pickle":
            # Patch 122bh: file picker for a pickle file. UI is
            # identical to "file"; the difference is in
            # to_kwargs, which opens and pickle.loads the
            # selected file. Default extras: (file_filter,
            # placeholder).
            file_filter = (
                extra[0] if extra else "Pickle files (*.pkl)"
            )
            placeholder = extra[1] if len(extra) > 1 else ""
            w = _PathField(is_file=True, file_filter=file_filter,
                           placeholder=placeholder)
            if default:
                w.path = str(default)
            self._form.addRow(self._label(name), w)
        elif kind == "file":
            file_filter = extra[0] if extra else ""
            placeholder  = extra[1] if len(extra) > 1 else ""
            w = _PathField(is_file=True, file_filter=file_filter,
                           placeholder=placeholder)
            if default:
                w.set_path(str(default))
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
            elif kind == "list":
                # Patch 122be: comma-split, strip, drop empties.
                raw = w.text() or ""
                out[name] = [
                    s.strip() for s in raw.split(",") if s.strip()
                ]
            elif kind == "dict":
                # Patch 122bg: parse JSON object; empty input or
                # malformed JSON yields {} (backends have their
                # own defaults for missing keys).
                raw = (w.text() or "").strip()
                if not raw:
                    out[name] = {}
                else:
                    import json
                    try:
                        parsed = json.loads(raw)
                        out[name] = (
                            parsed if isinstance(parsed, dict)
                            else {}
                        )
                    except (json.JSONDecodeError, ValueError):
                        out[name] = {}
            elif kind == "pickle":
                # Patch 122bh: open the picked file and
                # pickle.load it. Empty path or load error
                # yields None — backend will then raise its
                # usual missing-arg error (clearer than a
                # form-side message would be). pickle.load
                # executes arbitrary code from the file; the
                # user is trusted to only pick files they
                # produced themselves (single-user desktop
                # trust model — same as running any other
                # Python script).
                path = w.path or ""
                if not path:
                    out[name] = None
                else:
                    import pickle
                    try:
                        with open(path, "rb") as fh:
                            out[name] = pickle.load(fh)
                    except (OSError, pickle.UnpicklingError,
                            EOFError, AttributeError,
                            ImportError, ValueError):
                        out[name] = None
            elif kind == "file":
                out[name] = w.path
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


# =========================================================================== #
# Layout-aware viz source-dir resolution (patch 122dc)
# =========================================================================== #
# Each visualisation route declares `data_paths_source` (plural) or
# `data_path_source` (singular) as a string naming the pipeline
# stage that produces its input files. Pre-122dc, this string was
# joined unconditionally as `<project_root>/csv/<name>/`, which is
# the legacy SimBA layout. v1 projects don't have a `csv/` tree;
# their per-stage data lives under `derived/<stage>/` (typically
# with run-id subdirs).
#
# `_resolve_viz_source_dir` maps each route source name to the
# right v1 directory under `derived/`. The run-id subdir resolution
# (latest-run-or-parent) matches ConfigReader's behaviour in
# `_apply_v1_path_overrides`: if the parent has run-id subdirs,
# return the newest one; otherwise return the parent itself.
#
# Adding a new route source: extend `_VIZ_SOURCE_V1_MAP` below.
# If a route source name doesn't appear in the map, the helper
# falls back to `derived/<name>/` — a best-effort guess that may
# or may not match where the v1 stage actually writes. The caller
# `target()` raises a clear "not a directory" error in that case,
# so the symptom is loud rather than silent-wrong.

_VIZ_SOURCE_V1_MAP: dict = {
    # Route source name → v1 subpath under <root>/.
    # Matches ConfigReader._apply_v1_path_overrides assignments
    # so visualisations see the same dirs as the rest of the code.
    "machine_results": ("derived", "classifications"),
    "outlier_corrected_movement_location":
        ("derived", "outlier_corrected"),
    "outlier_corrected_movement":
        ("derived", "outlier_corrected"),  # alias used by some routes
    "features_extracted": ("derived", "features"),
    "directing_data": ("derived", "directionality"),
    # input_csv routes (raw pose) — v1 has these at sources/pose
    "input_csv": ("sources", "pose"),
    # targets_inserted (labels) — v1 puts at derived/labels
    "targets_inserted": ("derived", "labels"),
}


def _resolve_viz_source_dir(*, config_path: str,
                            project_root, source_name: str):
    """Layout-aware resolver for a viz route's data source dir.

    Parameters
    ----------
    config_path : str
        Project config path; used to detect layout. Anything
        ending `.toml` (case-insensitive) is v1; else legacy.
    project_root : Path
        Project root for the active layout. From
        `project_paths_from_config(config_path)["project_root"]`.
    source_name : str
        Value of the route's `data_paths_source` or
        `data_path_source` attribute (e.g. "machine_results").

    Returns
    -------
    pathlib.Path
        Resolved source directory. For v1 projects with run-id
        subdirs, returns the latest run's directory; for stages
        without run subdirs, returns the parent stage directory.
        For legacy projects, returns `<root>/csv/<source_name>/`.
    """
    from pathlib import Path as _P

    from mufasa.project_layout import is_run_id

    is_v1 = str(config_path).lower().endswith(".toml")
    project_root = _P(project_root)

    if not is_v1:
        return project_root / "csv" / source_name

    # v1 mapping
    if source_name in _VIZ_SOURCE_V1_MAP:
        parts = _VIZ_SOURCE_V1_MAP[source_name]
    else:
        # Best-effort fallback: derived/<name>/. If the caller
        # passes a source name not in the map AND that's wrong,
        # the downstream "data source directory not found" error
        # will be loud and recoverable.
        parts = ("derived", source_name)

    stage_parent = project_root.joinpath(*parts)
    # Latest-run-or-parent (matches ConfigReader's logic). If the
    # stage_parent has run-id subdirs, return the newest; otherwise
    # return stage_parent itself (some v1 outputs are written flat,
    # e.g. derived/classifications/<video>.parquet without a run
    # subdir per 122ax's flat-classifier-output decision).
    if stage_parent.is_dir():
        try:
            runs = sorted(
                d for d in stage_parent.iterdir()
                if d.is_dir() and is_run_id(d.name)
            )
        except OSError:
            runs = []
        if runs:
            return runs[-1]
    return stage_parent


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
    kwargs_transform : Callable[[dict], dict] | None
        Patch 122az hook for routes whose backend signature can't
        be expressed as plain rename/default. Called after
        ``default_kwargs`` + extras merge and before
        ``filter_kwargs``. Use for type coercions like
        ``body_part: str -> body_parts: [body_part]`` that backends
        need but the form only collects as scalars (no list-type
        extras builder yet).
    """
    label: str
    scope_kind: str = "project"
    extras: list = field(default_factory=list)
    backend_sp: Callable[..., Any] | None = None
    backend_mp: Callable[..., Any] | None = None
    needs_video: bool = False
    needs_save_dir: bool = False
    common_toggles: set = field(default_factory=lambda: set())
    kwargs_map: dict = field(default_factory=dict)
    default_kwargs: dict = field(default_factory=dict)
    data_paths_source: str | None = None
    data_path_source: str | None = None
    kwargs_transform: Callable[[dict], dict] | None = None


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
        # sp and mp backends have slightly different signatures:
        #   sp: show_bbox (bool), print_timers (plural); no gpu
        #   mp: bbox (Literal['axis-aligned','animal-aligned']),
        #       print_timer (singular), has gpu
        # Form names default to sp-compatible (they're more common);
        # the runtime signature filter silently drops them when mp runs.
        # Users who need gpu/axis-aligned bboxes should pick mp explicitly.
        common_toggles={"frame", "video", "gpu"},
        extras=[
            ("rotate",          "bool",  False),
            ("animal_names",    "bool",  True),
            ("show_pose",       "bool",  True),
            ("show_confidence", "bool",  True),
            ("show_gantt",      "bool",  False),
            ("font_size",       "float", 0.7, 0.1, 5.0, 0.1),
            ("circle_size",     "int",   5, 1, 50),
            ("print_timers",    "bool",  True),
            ("show_bbox",       "bool",  False),
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
        # Patch 122bt: mp variant exists and is interface-compatible;
        # adding here so users with multiple cores get parallel runs.
        backend_mp=_lazy_factory("mufasa.plotting.clf_validator_mp",
                                 "ClassifierValidationClipsMultiprocess"),
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
        # This route differs from the other viz routes: instead of iterating
        # per-video project CSVs (data_paths_source), it takes one
        # features_extracted CSV and one classifier .sav as required inputs.
        # We surface them as file pickers rather than auto-scanning because
        # the user needs to pick *which* model to validate.
        extras=[
            ("feature_path",             "file",  "",
             "CSV files (*.csv);;Parquet files (*.parquet);;All files (*)",
             "Select a features_extracted CSV from your project"),
            ("model_path",               "file",  "",
             "Model files (*.sav);;All files (*)",
             "Select a classifier .sav from models/generated_models/"),
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
    # -------- Patch 122az: ROI / skeleton fill-ins ---------- #
    # Three routes that closed gaps in the viz form: ROI overlay,
    # ROI feature overlay, and pose-skeleton video. The ROI ones
    # take body_parts as a list[str]; originally that was
    # expressed as singular body_part + a kwargs_transform
    # lambda wrapping it into [body_part]. Patch 122be moved to
    # the native "list" extras kind, dropping the transforms.
    _VizRoute(
        label="ROI overlay (per video)",
        scope_kind="file",
        backend_sp=_lazy_factory("mufasa.plotting.roi_plotter",
                                 "ROIPlotter"),
        # Patch 122bt: mp variant exists and is interface-compatible.
        backend_mp=_lazy_factory("mufasa.plotting.roi_plotter_mp",
                                 "ROIPlotMultiprocess"),
        needs_video=True,
        extras=[
            # Patch 122be: native "list" kind — was a singular
            # body_part: str + kwargs_transform wrap. The user
            # can now type "Nose" or "Nose,Tail" and the backend
            # receives a list directly.
            ("body_parts",        "list",  ["Nose"],
             "Body-part name(s) to overlay (comma-separated)"),
            ("outside_roi",       "bool",  False),
            ("threshold",         "float", 0.0, 0.0, 1.0, 0.05),
            ("show_animal_name",  "bool",  True),
            ("show_body_part",    "bool",  True),
            ("show_bbox",         "bool",  False),
        ],
        kwargs_map={"video_path": "video_path"},
    ),
    _VizRoute(
        label="ROI feature overlay (per video)",
        scope_kind="file",
        backend_sp=_lazy_factory(
            "mufasa.plotting.ROI_feature_visualizer",
            "ROIfeatureVisualizer",
        ),
        # Patch 122bt: mp variant exists and is interface-compatible.
        backend_mp=_lazy_factory(
            "mufasa.plotting.ROI_feature_visualizer_mp",
            "ROIfeatureVisualizerMultiprocess",
        ),
        needs_video=True,
        extras=[
            # Patch 122be: native "list" kind — same migration
            # as ROIPlotter above.
            ("body_parts", "list", ["Nose"],
             "Body-part name(s) to overlay (comma-separated)"),
        ],
        kwargs_map={"video_path": "video_path"},
        # Backend takes style_attr dict; empty dict triggers
        # backend's defaults.
        default_kwargs={"style_attr": {}},
    ),
    _VizRoute(
        label="Skeleton video (project pose)",
        scope_kind="project",
        backend_sp=_lazy_factory(
            "mufasa.plotting.skeleton_video_creator",
            "SkeletonVideoCreator",
        ),
        extras=[
            ("circle_size",     "int",   5, 1, 50),
            ("line_thickness",  "int",   2, 1, 20),
            ("bp_threshold",    "float", 0.0, 0.0, 1.0, 0.05),
            ("ego_direction",   "bool",  False),
        ],
        # SkeletonVideoCreator iterates over project pose data
        # at outlier_corrected_movement; uses save_dir for output.
        data_paths_source="outlier_corrected_movement_location",
        needs_save_dir=True,
        kwargs_map={"data_paths": "data_path", "save_path": "save_dir"},
    ),
    # -------- Patch 122ba: niche viz plotter fill-ins ---------- #
    # 7 backends that exist in mufasa/plotting/ but weren't yet
    # surfaced through the form. Patch 122bi migrated the two
    # str → list[str] coercions (cue_light_names, arm_names)
    # from kwargs_transform lambdas to the native "list" kind.
    _VizRoute(
        label="Cue light visualizer",
        scope_kind="project",
        backend_sp=_lazy_factory(
            "mufasa.plotting.cue_light_visualizer",
            "CueLightVisualizer",
        ),
        needs_video=True,
        common_toggles={"frame", "video"},
        extras=[
            ("data_path",       "file", "",
             "CSV files (*.csv);;Parquet files (*.parquet);;"
             "All files (*)",
             "Cue light data file (CSV / parquet)"),
            # Patch 122bi: native "list" kind — was str +
            # kwargs_transform comma-split.
            ("cue_light_names", "list", [],
             "Cue light names (comma-separated, "
             "e.g. CueLight_1,CueLight_2)"),
            ("show_pose",       "bool", True),
            ("core_cnt",        "int",  1, 1, 64),
            ("verbose",         "bool", True),
        ],
    ),
    _VizRoute(
        label="Interactive probability inspector",
        scope_kind="project",
        backend_sp=_lazy_factory(
            "mufasa.plotting.interactive_probability_grapher",
            "InteractiveProbabilityGrapher",
        ),
        extras=[
            ("file_path",   "file", "",
             "CSV files (*.csv);;Parquet files (*.parquet);;"
             "All files (*)",
             "machine_results file to inspect"),
            ("model_path",  "file", "",
             "Model files (*.sav);;All files (*)",
             "Classifier .sav to load thresholds from"),
            ("lbl_font_size",          "int",  10, 1, 60),
            ("show_thresholds",        "bool", True),
            ("show_statistics_legend", "bool", True),
        ],
    ),
    _VizRoute(
        label="Spontaneous alternation plot",
        scope_kind="project",
        backend_sp=_lazy_factory(
            "mufasa.plotting.spontaneous_alternation_plotter",
            "SpontaneousAlternationsPlotter",
        ),
        extras=[
            # Patch 122bi: native "list" kind — was str +
            # kwargs_transform comma-split.
            ("arm_names",   "list",  [],
             "Arm ROI names (comma-separated, "
             "e.g. Arm_1,Arm_2,Arm_3)"),
            ("center_name", "str",   "Center",
             "Center ROI name"),
            ("animal_area", "float", 100.0, 1.0, 100000.0, 1.0),
            ("threshold",   "float", 0.0, 0.0, 1.0, 0.05),
            ("buffer",      "int",   0, 0, 1000),
            ("core_cnt",    "int",   1, 1, 64),
            ("verbose",     "bool",  True),
        ],
    ),
    _VizRoute(
        label="YOLO pose+track predictions (external)",
        scope_kind="file",
        backend_sp=_lazy_factory(
            "mufasa.plotting.yolo_pose_track_visualizer",
            "YOLOPoseTrackVisualizer",
        ),
        needs_video=True,
        needs_save_dir=True,
        extras=[
            ("threshold",   "float", 0.5, 0.0, 1.0, 0.01),
            ("thickness",   "int",   2, 1, 20),
            ("circle_size", "int",   5, 1, 50),
            ("bbox",        "bool",  True),
            ("overwrite",   "bool",  False),
            ("core_cnt",    "int",   1, 1, 64),
        ],
        kwargs_map={"source_path": "data_path",
                    "video_path": "video_path",
                    "save_path": "save_dir"},
    ),
    _VizRoute(
        label="YOLO segmentation predictions (external)",
        scope_kind="file",
        backend_sp=_lazy_factory(
            "mufasa.plotting.yolo_seg_visualizer",
            "YOLOSegmentationVisualizer",
        ),
        needs_video=True,
        needs_save_dir=True,
        extras=[
            ("threshold",     "float", 0.5, 0.0, 1.0, 0.01),
            ("shape_opacity", "float", 0.5, 0.0, 1.0, 0.05),
            ("core_cnt",      "int",   1, 1, 64),
        ],
        kwargs_map={"source_path": "data_path",
                    "video_path": "video_path",
                    "save_path": "save_dir"},
    ),
    _VizRoute(
        label="YOLO annotation visualizer (training data)",
        scope_kind="file",
        backend_sp=_lazy_factory(
            "mufasa.plotting.yolo_annotation_visualizer",
            "YOLOAnnotationVisualizer",
        ),
        needs_save_dir=True,
        extras=[
            ("split",        "choice", "train",
             ["train", "val", "test"]),
            ("n",            "int",    10, 1, 100000),
            ("circle_size",  "int",    5, 1, 50),
            ("thickness",    "int",    2, 1, 20),
            ("seg_opacity",  "float",  0.5, 0.0, 1.0, 0.05),
            ("img_format",   "choice", "png",
             ["png", "jpg", "jpeg", "webp"]),
            ("show_names",   "bool",   True),
            ("show_outline", "bool",   True),
            ("verbose",      "bool",   True),
        ],
        kwargs_map={"source_path": "map_yaml_path",
                    "save_path": "save_dir"},
    ),
    _VizRoute(
        label="Blob plotter (simple)",
        scope_kind="file",
        backend_sp=_lazy_factory(
            "mufasa.plotting.blob_plotter",
            "BlobPlotter",
        ),
        needs_save_dir=True,
        common_toggles={"gpu"},
        extras=[
            ("circle_size", "int", 5, 1, 50),
            ("smoothing",   "int", 0, 0, 100),
            ("batch_size",  "int", 100, 1, 100000),
            ("core_cnt",    "int", 1, 1, 64),
            ("verbose",     "bool", True),
        ],
        kwargs_map={"source_path": "data_path",
                    "save_path": "save_dir"},
    ),
    # -------- Patch 122bg: dict-kind route fill-in ---------- #
    # CircularFeaturePlotter takes a settings dict whose values
    # are JSON-serializable (e.g. {'center': {'Animal_1':
    # 'SwimBladder'}, 'text_settings': False, 'palette': 'bwr'}).
    # The new "dict" extras kind lets users paste a JSON object
    # directly. GeometryPlotter was NOT surfaced in 122bg — its
    # `geometries` arg is List[List[Shapely-objects]] which
    # isn't JSON-serializable. Patch 122bh unlocks it via the
    # new "pickle" extras kind (see route below).
    _VizRoute(
        label="Circular feature overlay (per video)",
        scope_kind="file",
        backend_sp=_lazy_factory(
            "mufasa.plotting.circular_feature_overlay_plotter",
            "CircularFeaturePlotter",
        ),
        extras=[
            # settings: JSON dict. Default is a minimal example
            # the user can edit; backends typically inspect
            # specific keys, so empty {} works too.
            ("settings", "dict",
             {"text_settings": False, "palette": "bwr"},
             'JSON, e.g. {"center": {"Animal_1": "SwimBladder"}, '
             '"text_settings": false, "palette": "bwr"}'),
        ],
        # File scope: user picks the circular-features data CSV.
        kwargs_map={"source_path": "data_path"},
    ),
    # -------- Patch 122bh: pickle-kind route fill-in ---------- #
    # GeometryPlotter takes a List[List[Shapely-object]] for
    # geometries — not JSON-serializable. The new "pickle"
    # extras kind lets users construct geometries in a Python
    # script, pickle them, and point the form at the .pkl file.
    # At dispatch the form pickle.loads the file and passes the
    # deserialized list-of-lists through to the backend.
    # SECURITY: pickle.load runs arbitrary code from the file;
    # the user trust model is "only load files you produced
    # yourself" (same as running any Python script).
    _VizRoute(
        label="Geometry overlay (Shapely objects)",
        scope_kind="project",
        backend_sp=_lazy_factory(
            "mufasa.plotting.geometry_plotter",
            "GeometryPlotter",
        ),
        needs_save_dir=True,
        extras=[
            # geometries: pickle file with List[List[Shapely]].
            ("geometries", "pickle", "",
             "Pickle files (*.pkl);;All files (*)",
             "Pickle file with List[List[Shapely]] geometries"),
            ("video_name",       "str",   "",
             "Video name (filename, not path)"),
            ("thickness",        "int",   2, 1, 20),
            ("circle_size",      "int",   5, 1, 50),
            ("bg_opacity",       "float", 0.5, 0.0, 1.0, 0.05),
            ("shape_opacity",    "float", 0.5, 0.0, 1.0, 0.05),
            ("intersection_clr", "color", "(255, 255, 255)"),
            ("outline_clr",      "color", "(0, 0, 0)"),
            ("palette",          "str",   "Set1",
             "Matplotlib palette name (e.g. Set1, tab10)"),
            ("core_cnt",         "int",   1, 1, 64),
            ("verbose",          "bool",  True),
        ],
        kwargs_map={"save_path": "save_dir"},
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

    def target(self, *, route: _VizRoute, config_path: str | None,
               source: str | None, video: str | None,
               save: str | None, extras: dict, common: dict,
               cores: int) -> None:
        from pathlib import Path as _P

        from mufasa.project_layout import (
            project_metadata_from_config,
            project_paths_from_config,
        )

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
        #
        # Patch 122f: layout-agnostic project resolution.
        # Patch 122dc: layout-agnostic SOURCE DIR resolution too.
        # Pre-122dc, the source dir was unconditionally
        # `<root>/csv/<subdir>` — works for legacy SimBA, but v1
        # projects have no top-level `csv/` so EVERY route that
        # declared `data_paths_source` raised a "data source not
        # found" error. _resolve_viz_source_dir maps the route's
        # source name to the right v1 directory under
        # `derived/` (with latest-run selection matching
        # ConfigReader's _latest_run_or_parent behaviour).
        if route.data_paths_source or route.data_path_source:
            if not config_path:
                raise RuntimeError(
                    f"{route.label!r} requires a loaded project "
                    "(config_path) to locate its data files."
                )
            try:
                paths = project_paths_from_config(config_path)
                meta = project_metadata_from_config(config_path)
            except (ValueError, OSError) as exc:
                raise RuntimeError(
                    f"{route.label!r}: cannot parse project config: "
                    f"{exc}"
                )
            proj = paths["project_root"]
            file_type = meta["file_type"]
            subdir = route.data_paths_source or route.data_path_source
            src_dir = _resolve_viz_source_dir(
                config_path=config_path,
                project_root=_P(proj),
                source_name=subdir,
            )
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

        # Patch 122az: route-level kwargs_transform — runs after
        # merge/rename, before the defensive signature filter.
        # Used for type coercions the declarative form can't
        # express (e.g. wrapping a single body_part into a list
        # for backends that take body_parts: list[str]).
        if route.kwargs_transform is not None:
            kwargs = route.kwargs_transform(kwargs)

        # Defensive signature filter — delegated to the shared helper
        # so data_import and analysis forms all get the same behaviour.
        # See _backend_dispatch.py for the "why".
        from mufasa.ui_qt.forms._backend_dispatch import filter_kwargs
        kwargs = filter_kwargs(backend, kwargs)

        runner = backend(**kwargs)
        if runner is not None and hasattr(runner, "run"):
            runner.run()


__all__ = ["VisualizationForm", "ROUTES"]
