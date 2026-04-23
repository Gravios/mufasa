"""
mufasa.ui_qt.forms.data_import
==============================

One consolidated form for all pose-data / annotation-format bridges.
Replaces 11 legacy popups:

* :class:`DLC2LabelmePopUp`
* :class:`DLCAnnotations2LabelMePopUp`
* :class:`DLCH5Inference2YoloPopUp`
* :class:`DLCYoloKeypointsPopUp`
* :class:`SLEAPAnnotations2YoloPopUp`
* :class:`SLEAPH5Inference2YoloPopUp`
* :class:`SimBA2YoloKeypointsPopUp`
* :class:`COCOKeypoints2YOLOkeypointsPopUp`
* :class:`Labelme2DataFramePopUp`
* :class:`Labelme2ImgsPopUp`
* :class:`LabelmeBbox2YoloBboxPopUp`
* :class:`LabelmeDirectory2CSVPopUp`
* :class:`MergeCOCOKeypointFilesPopUp`

Pattern
-------

Every converter fits the shape::

    source (file/dir) + save_dir + [optional video_dir]
    + common flags (greyscale, clahe, verbose)
    + route-specific extras (train_size, padding, image size, ...)

One :class:`ConverterForm` does the UI; the **route table** below
declares, for each (source_fmt, target_fmt) pair, which backend class
to instantiate, which extra fields to show, and what their kwargs are.

Adding a new converter later is a one-record append to ``ROUTES`` —
not a new window, not a new file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget)

from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Reusable path picker (file or directory)
# --------------------------------------------------------------------------- #
class _PathField(QWidget):
    """Read-only line edit + Browse… button. Picks a file or a folder
    depending on ``is_file`` / ``file_filter``."""

    def __init__(self, *, is_file: bool = False,
                 file_filter: str = "",
                 placeholder: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.is_file = is_file
        self.file_filter = file_filter
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.line = QLineEdit(self); self.line.setReadOnly(True)
        if placeholder:
            self.line.setPlaceholderText(placeholder)
        self.btn = QPushButton("Browse…", self)
        self.btn.clicked.connect(self._browse)
        lay.addWidget(self.line, 1); lay.addWidget(self.btn)

    def _browse(self) -> None:
        if self.is_file:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select file", "", self.file_filter or "All files (*)",
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "Select directory", "")
        if path:
            self.line.setText(path)

    @property
    def path(self) -> str:
        return self.line.text().strip()

    def set_path(self, p: str) -> None:
        self.line.setText(p)


# --------------------------------------------------------------------------- #
# Extra-fields panels (one per distinct "shape" of converter extras)
# --------------------------------------------------------------------------- #
class _NoExtras(QWidget):
    """Converters that only need source + save (+ common flags)."""
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        QVBoxLayout(self).setContentsMargins(0, 0, 0, 0)
    def to_kwargs(self) -> dict:
        return {}


class _YoloExtras(QWidget):
    """Shared shape for the six ``*-to-YOLO`` converters: train split,
    padding, sample size."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self); form.setContentsMargins(0, 0, 0, 0)

        self.train_size = QSpinBox(self)
        self.train_size.setRange(10, 100); self.train_size.setValue(70)
        self.train_size.setSuffix(" %")
        form.addRow("Train split:", self.train_size)

        self.sample_size = QSpinBox(self)
        self.sample_size.setRange(0, 10_000); self.sample_size.setValue(0)
        self.sample_size.setSpecialValueText("(all)")
        self.sample_size.setToolTip("Maximum number of frames to sample. 0 = all.")
        form.addRow("Sample size:", self.sample_size)

        self.padding = QDoubleSpinBox(self)
        self.padding.setRange(0.0, 10.0); self.padding.setSingleStep(0.01)
        self.padding.setValue(0.05)
        self.padding.setToolTip("Fractional bbox padding around pose.")
        form.addRow("Padding:", self.padding)

    def to_kwargs(self) -> dict:
        return {
            "train_size": int(self.train_size.value()) / 100.0,
            "sample_size": int(self.sample_size.value()) or None,
            "padding": float(self.padding.value()),
        }


class _LabelmeDFExtras(QWidget):
    """Extras for Labelme→DataFrame: pad, normalize, image-size."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self); form.setContentsMargins(0, 0, 0, 0)

        self.pad = QCheckBox("Pad to square", self)
        form.addRow("", self.pad)

        self.normalize = QCheckBox("Normalise coordinates [0, 1]", self)
        form.addRow("", self.normalize)

        self.size = QSpinBox(self)
        self.size.setRange(0, 4096); self.size.setValue(0)
        self.size.setSpecialValueText("(no resize)")
        self.size.setSuffix(" px")
        form.addRow("Target size:", self.size)

    def to_kwargs(self) -> dict:
        return {
            "pad": bool(self.pad.isChecked()),
            "normalize": bool(self.normalize.isChecked()),
            "size": int(self.size.value()) or None,
        }


class _Labelme2ImgsExtras(QWidget):
    """Labelme→Images: image format only."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self); form.setContentsMargins(0, 0, 0, 0)
        self.img_fmt = QComboBox(self)
        self.img_fmt.addItems(["png", "jpg", "bmp", "webp"])
        form.addRow("Output image format:", self.img_fmt)

    def to_kwargs(self) -> dict:
        return {"img_format": self.img_fmt.currentText()}


# --------------------------------------------------------------------------- #
# Route table — the declarative heart
# --------------------------------------------------------------------------- #
@dataclass
class _Route:
    """A single (source → target) converter route.

    Attributes
    ----------
    source_label : str         e.g. "DLC"
    target_label : str         e.g. "YOLO keypoints"
    source_kind  : str         "file" or "dir"
    source_filter: str         QFileDialog filter when source_kind == "file"
    needs_video  : bool        requires a video directory
    extras_key   : str         which extras widget to use
                               ("none" | "yolo" | "labelme_df" | "labelme_imgs")
    backend      : callable    constructor / function to invoke with kwargs
    kwargs_map   : dict        maps UI field name → backend kwarg name
                               (source_path, save_path, video_path, and the
                               extras panel's to_kwargs() contents are all
                               remapped via this dict)
    common_flags : set[str]    which of the common (greyscale, clahe, verbose)
                               flags the backend accepts.
    """
    source_label: str
    target_label: str
    source_kind: str = "dir"
    source_filter: str = ""
    needs_video: bool = False
    extras_key: str = "none"
    backend: Optional[Callable[..., Any]] = None
    kwargs_map: dict = field(default_factory=dict)
    common_flags: set = field(default_factory=lambda: {"verbose"})


# Backend classes are imported lazily inside ``target()`` so the form
# loads instantly even if a backend module is broken or missing.
def _lazy(modpath: str, classname: str) -> Callable[..., Any]:
    def _factory(**kw):
        mod = __import__(modpath, fromlist=[classname])
        return getattr(mod, classname)(**kw)
    _factory.__name__ = f"{modpath}.{classname}"
    return _factory


ROUTES: list[_Route] = [
    # ---------------- DLC sources ---------------- #
    _Route(
        "DLC", "Labelme",
        source_kind="dir",
        extras_key="none",
        backend=_lazy("mufasa.third_party_label_appenders.transform.dlc_to_labelme", "DLC2Labelme"),
        kwargs_map={"source_path": "dlc_dir", "save_path": "save_dir"},
        common_flags={"greyscale", "clahe", "verbose"},
    ),
    _Route(
        "DLC", "YOLO keypoints",
        source_kind="dir",
        extras_key="yolo",
        backend=_lazy("mufasa.third_party_label_appenders.transform.dlc_to_yolo", "DLC2Yolo"),
        kwargs_map={"source_path": "dlc_dir", "save_path": "save_dir"},
        common_flags={"greyscale", "clahe", "verbose"},
    ),
    _Route(
        "DLC (multi-animal H5)", "YOLO keypoints",
        source_kind="dir",
        needs_video=True,
        extras_key="yolo",
        backend=_lazy("mufasa.third_party_label_appenders.transform.dlc_ma_h5_to_yolo", "MADLCH52Yolo"),
        kwargs_map={"source_path": "data_dir", "save_path": "save_dir",
                    "video_path": "video_dir"},
        common_flags={"greyscale", "clahe", "verbose"},
    ),
    # ---------------- SLEAP sources ---------------- #
    _Route(
        "SLEAP (.slp annotations)", "YOLO keypoints",
        source_kind="dir",
        needs_video=True,
        extras_key="yolo",
        backend=_lazy("mufasa.third_party_label_appenders.transform.sleap_to_yolo", "SleapAnnotations2Yolo"),
        kwargs_map={"source_path": "sleap_dir", "save_path": "save_dir",
                    "video_path": "video_dir"},
        common_flags={"greyscale", "clahe", "verbose"},
    ),
    _Route(
        "SLEAP (H5 inference)", "YOLO keypoints",
        source_kind="dir",
        needs_video=True,
        extras_key="yolo",
        backend=_lazy("mufasa.third_party_label_appenders.transform.sleap_h5_to_yolo", "SleapH52Yolo"),
        kwargs_map={"source_path": "data_dir", "save_path": "save_dir",
                    "video_path": "video_dir"},
        common_flags={"greyscale", "clahe", "verbose"},
    ),
    # ---------------- SimBA source ---------------- #
    _Route(
        "SimBA / Mufasa project", "YOLO keypoints",
        source_kind="file",
        source_filter="INI files (*.ini)",
        extras_key="yolo",
        backend=_lazy("mufasa.third_party_label_appenders.transform.simba_to_yolo", "SimBA2Yolo"),
        kwargs_map={"source_path": "config_path", "save_path": "save_dir"},
        common_flags={"greyscale", "clahe", "verbose"},
    ),
    # ---------------- COCO source ---------------- #
    _Route(
        "COCO keypoints", "YOLO keypoints",
        source_kind="file",
        source_filter="JSON files (*.json)",
        extras_key="coco_yolo",
        backend=_lazy("mufasa.third_party_label_appenders.transform.coco_keypoints_to_yolo", "COCOKeypoints2Yolo"),
        kwargs_map={"source_path": "coco_path", "save_path": "save_dir"},
        common_flags={"greyscale", "clahe", "verbose"},
    ),
    # ---------------- Labelme sources ---------------- #
    _Route(
        "Labelme (keypoints)", "DataFrame (CSV)",
        source_kind="dir",
        extras_key="labelme_df",
        backend=_lazy("mufasa.third_party_label_appenders.transform.labelme_to_df", "LabelMe2DataFrame"),
        # Note: save_path is a FILE for this converter, not a dir. The
        # form always gives a directory; backend coerces. We set the
        # save-field label accordingly at runtime.
        kwargs_map={"source_path": "labelme_dir", "save_path": "save_path"},
        common_flags={"greyscale", "clahe", "verbose"},
    ),
    _Route(
        "Labelme (bbox)", "YOLO bounding boxes",
        source_kind="dir",
        extras_key="yolo",
        backend=_lazy("mufasa.third_party_label_appenders.transform.labelme_to_yolo",
                      "LabelmeBoundingBoxes2YoloBoundingBoxes"),
        kwargs_map={"source_path": "labelme_dir", "save_path": "save_dir"},
        common_flags={"greyscale", "clahe", "verbose"},
    ),
    _Route(
        "Labelme (annotations)", "Images",
        source_kind="dir",
        extras_key="labelme_imgs",
        # Functional backend, not a class
        backend=lambda **kw: (
            __import__("mufasa.third_party_label_appenders.converters",
                       fromlist=["labelme_to_img_dir"]).labelme_to_img_dir(**kw)
        ),
        kwargs_map={"source_path": "labelme_dir", "save_path": "img_dir"},
        common_flags={"greyscale"},
    ),
]


class _CocoYoloExtras(QWidget):
    """COCO keypoints → YOLO: unique among the YOLO converters because
    the COCO JSON doesn't embed pixel data, so the backend needs a
    separate ``img_dir``. Backend kwargs: ``img_dir``, ``bbox_pad``
    (not ``padding``), ``train_size``. No ``sample_size``.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self); form.setContentsMargins(0, 0, 0, 0)

        self.img_dir = _PathField(is_file=False,
                                  placeholder="Directory of COCO images…")
        form.addRow("Images directory:", self.img_dir)

        self.train_size = QSpinBox(self)
        self.train_size.setRange(10, 100); self.train_size.setValue(70)
        self.train_size.setSuffix(" %")
        form.addRow("Train split:", self.train_size)

        self.bbox_pad = QDoubleSpinBox(self)
        self.bbox_pad.setRange(0.0, 10.0); self.bbox_pad.setSingleStep(0.01)
        self.bbox_pad.setValue(0.05)
        self.bbox_pad.setToolTip("Fractional bbox padding around pose.")
        form.addRow("BBox padding:", self.bbox_pad)

    def to_kwargs(self) -> dict:
        # Return keys even when img_dir is empty so the canary can
        # statically discover the schema. Form-level validation happens
        # in ConverterForm.collect_args via validate().
        return {
            "img_dir": self.img_dir.path or "",
            "train_size": int(self.train_size.value()) / 100.0,
            "bbox_pad": float(self.bbox_pad.value()),
        }

    def validate(self) -> None:
        if not self.img_dir.path:
            raise ValueError("Images directory is required for COCO→YOLO.")


# Map extras_key → widget class
_EXTRAS_WIDGETS = {
    "none":         _NoExtras,
    "yolo":         _YoloExtras,
    "coco_yolo":    _CocoYoloExtras,
    "labelme_df":   _LabelmeDFExtras,
    "labelme_imgs": _Labelme2ImgsExtras,
}


# --------------------------------------------------------------------------- #
# The form
# --------------------------------------------------------------------------- #
class ConverterForm(OperationForm):
    """Universal source → target converter. One ``ROUTES`` row per
    source/target pair; the form picks the right fields based on the
    selection."""

    title = "Convert pose / annotation data"
    description = ("Convert pose-estimation data and annotations between "
                   "formats (DLC, SLEAP, Labelme, COCO, SimBA/Mufasa, YOLO). "
                   "Pick source and target; extra fields appear as needed.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # ---- Source dropdown ------------------------------------------ #
        source_labels = []
        for r in ROUTES:
            if r.source_label not in source_labels:
                source_labels.append(r.source_label)
        self.source_cb = QComboBox(self)
        self.source_cb.addItems(source_labels)
        self.source_cb.currentTextChanged.connect(self._on_source_changed)
        form.addRow("Source format:", self.source_cb)

        # ---- Target dropdown (populated from source) ------------------ #
        self.target_cb = QComboBox(self)
        self.target_cb.currentTextChanged.connect(self._on_route_changed)
        form.addRow("Target format:", self.target_cb)

        # ---- Source path --------------------------------------------- #
        self.source_path = _PathField(placeholder="Source path…")
        form.addRow("Source:", self.source_path)
        # Track which form-row the source is on so we can relabel it
        self._source_row_label = form.itemAt(form.rowCount() - 1, QFormLayout.LabelRole)

        # ---- Save path ----------------------------------------------- #
        self.save_path = _PathField(placeholder="Save directory…")
        form.addRow("Save to:", self.save_path)

        # ---- Video dir (conditional) --------------------------------- #
        self.video_path = _PathField(placeholder="Video directory (required for this route)…")
        form.addRow("Videos:", self.video_path)
        # Remember the LABEL widget too, so we can hide it along with the field
        self._video_row_index = form.rowCount() - 1

        # ---- Common flags -------------------------------------------- #
        flags_row = QHBoxLayout()
        self.flag_greyscale = QCheckBox("Greyscale", self)
        self.flag_clahe = QCheckBox("CLAHE", self)
        self.flag_verbose = QCheckBox("Verbose", self)
        self.flag_verbose.setChecked(True)
        flags_row.addWidget(self.flag_greyscale)
        flags_row.addWidget(self.flag_clahe)
        flags_row.addWidget(self.flag_verbose)
        flags_row.addStretch()
        flags_host = QWidget(self); flags_host.setLayout(flags_row)
        form.addRow("Options:", flags_host)

        # ---- Extras stack -------------------------------------------- #
        self.extras_stack = QStackedWidget(self)
        self._extras_indices: dict[str, int] = {}
        for key, cls in _EXTRAS_WIDGETS.items():
            widget = cls(self)
            self._extras_indices[key] = self.extras_stack.addWidget(widget)
        form.addRow("Parameters:", self.extras_stack)

        self.body_layout.addLayout(form)

        # Kick off with the first source selected
        self._on_source_changed(self.source_cb.currentText())

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #
    def _current_route(self) -> Optional[_Route]:
        src = self.source_cb.currentText()
        tgt = self.target_cb.currentText()
        for r in ROUTES:
            if r.source_label == src and r.target_label == tgt:
                return r
        return None

    def _on_source_changed(self, src: str) -> None:
        # Repopulate targets for this source
        self.target_cb.blockSignals(True)
        self.target_cb.clear()
        for r in ROUTES:
            if r.source_label == src:
                self.target_cb.addItem(r.target_label)
        self.target_cb.blockSignals(False)
        self._on_route_changed(self.target_cb.currentText())

    def _on_route_changed(self, _: str) -> None:
        route = self._current_route()
        if route is None:
            return
        # Source: file vs directory
        self.source_path.is_file = (route.source_kind == "file")
        self.source_path.file_filter = route.source_filter
        # Video row: show/hide
        self.video_path.setVisible(route.needs_video)
        # Also hide the label next to it — QFormLayout label lookup:
        lbl = self._form_label_at(self._video_row_index)
        if lbl is not None:
            lbl.setVisible(route.needs_video)
        # Extras panel
        self.extras_stack.setCurrentIndex(self._extras_indices[route.extras_key])
        # Common-flags enable state
        self.flag_greyscale.setEnabled("greyscale" in route.common_flags)
        self.flag_clahe.setEnabled("clahe" in route.common_flags)
        self.flag_verbose.setEnabled("verbose" in route.common_flags)

    def _form_label_at(self, row: int) -> Optional[QWidget]:
        """Fetch the label widget of the Nth form row (or None)."""
        form: Optional[QFormLayout] = None
        for i in range(self.body_layout.count()):
            item = self.body_layout.itemAt(i)
            lo = item.layout()
            if isinstance(lo, QFormLayout):
                form = lo
                break
        if form is None or row >= form.rowCount():
            return None
        item = form.itemAt(row, QFormLayout.LabelRole)
        return item.widget() if item else None

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        route = self._current_route()
        if route is None:
            raise ValueError("No valid route selected.")
        src = self.source_path.path
        if not src:
            raise ValueError("Source path is empty.")
        save = self.save_path.path
        if not save:
            raise ValueError("Save path is empty.")
        video = self.video_path.path if route.needs_video else ""
        if route.needs_video and not video:
            raise ValueError(
                f"Route '{route.source_label} → {route.target_label}' "
                f"needs a video directory."
            )
        # Pull extras
        panel_idx = self._extras_indices[route.extras_key]
        extras_panel = self.extras_stack.widget(panel_idx)
        # Route-specific validation hook — panels can raise here to
        # surface clear errors on missing required extras fields.
        if hasattr(extras_panel, "validate"):
            extras_panel.validate()
        extras = extras_panel.to_kwargs()
        # Common flags (only ones the route accepts)
        flags = {}
        if "greyscale" in route.common_flags:
            flags["greyscale"] = bool(self.flag_greyscale.isChecked())
        if "clahe" in route.common_flags:
            flags["clahe"] = bool(self.flag_clahe.isChecked())
        if "verbose" in route.common_flags:
            flags["verbose"] = bool(self.flag_verbose.isChecked())
        return {
            "route": route,
            "source": src,
            "save":   save,
            "video":  video or None,
            "extras": extras,
            "flags":  flags,
        }

    def target(self, *, route: _Route, source: str, save: str,
               video: Optional[str], extras: dict, flags: dict) -> None:
        # Build backend kwargs using the route's kwargs_map
        km = route.kwargs_map
        kwargs: dict = {}
        kwargs[km["source_path"]] = source
        kwargs[km["save_path"]]   = save
        if route.needs_video and "video_path" in km:
            kwargs[km["video_path"]] = video
        # Merge extras (these already use backend kwarg names — e.g.
        # "train_size", "padding"). If a route needs different names it
        # can preprocess here in future; for now one-to-one works.
        kwargs.update(extras)
        kwargs.update(flags)
        runner = route.backend(**kwargs)
        # Class-based runners expose .run(); function-based just return.
        if runner is not None and hasattr(runner, "run"):
            runner.run()


__all__ = ["ConverterForm", "ROUTES"]
