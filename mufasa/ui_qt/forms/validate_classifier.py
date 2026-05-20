"""
mufasa.ui_qt.forms.validate_classifier
=======================================

Qt port of
:class:`mufasa.ui.pop_ups.validation_plot_pop_up.ValidationVideoPopUp`
— the per-video out-of-sample validation runner. Drives
:class:`ValidateModelOneVideo` (single-core) or
:class:`ValidateModelOneVideoMultiprocess` (multi-core) to
render an annotated validation video showing pose tracks +
classifier predictions overlaid.

Patch 122ar (this file)
-----------------------
The Tk popup took ``config_path / feature_path / model_path /
discrimination_threshold / shortest_bout`` in its constructor —
meaning it relied on whoever launched it to have already
picked those values. In the Qt port the form is the entry
point, so we add file pickers for model + feature files and
QLineEdit fields for threshold + min-bout up top.

Otherwise the field shape mirrors the Tk popup faithfully:

* **Style group** — font size, text spacing, circle size, text
  opacity, text thickness, body-part palette.
* **Tracking group** — show pose, show animal names, show
  classifier confidence, show bounding box, CPU count.
* **Gantt group** — gantt overlay type (None / final frame /
  video).

In-frame + dockable
-------------------
Subclasses :class:`OperationForm` so it lives inline on the
Classifier page like any other section. Pop-out button
re-parents into a :class:`QDockWidget` attached to the
workbench main window — same pattern as 122aj, 122al, 122ap,
122aq.

Single-core vs multi-core dispatch
----------------------------------
Mirrors the Tk popup: ``core_cnt == 1`` uses
:class:`ValidateModelOneVideo` (no bbox / bp_palette /
core_cnt kwargs); ``core_cnt > 1`` uses
:class:`ValidateModelOneVideoMultiprocess` (full kwarg set).
The single-core path produces identical video output but
without the multiprocess fan-out — useful when debugging
since stack traces are clean.

No INI persistence
------------------
The Tk popup never persisted these settings — they're
per-run only. The Qt port matches that behaviour. Re-opening
the form starts from defaults each time.

What's deferred from Tk parity
------------------------------
* No deferred items — every Tk field is ported.
"""
from __future__ import annotations

import math
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mufasa.ui_qt.workbench import OperationForm

AUTO = "AUTO"
GANTT_NONE  = "None"
GANTT_FRAME = "Gantt chart: final frame only (slightly faster)"
GANTT_VIDEO = "Gantt chart: video"
GANTT_OPTIONS = [GANTT_NONE, GANTT_FRAME, GANTT_VIDEO]

BBOX_FALSE         = "FALSE"
BBOX_AXIS_ALIGNED  = "axis-aligned"
BBOX_ANIMAL_ALIGN  = "animal-aligned"
BBOX_OPTIONS = [BBOX_FALSE, BBOX_AXIS_ALIGNED, BBOX_ANIMAL_ALIGN]

# AUTO + 1-55 for font_size / text_thickness
FONT_SIZE_OPTIONS = [AUTO] + [str(i) for i in range(1, 56)]

# AUTO + odd integers 1-105 for text_spacing / circle_size
TEXT_SPACE_OPTIONS = [AUTO] + [str(i) for i in range(1, 106, 2)]

# 0.1 .. 1.0 in 0.1 steps
OPACITY_OPTIONS = [f"{round(0.1 * i, 1)}" for i in range(1, 11)]

# Fallback palette list if Options.PALETTE_OPTIONS_* aren't importable
# (e.g. in test scaffolding); the live form pulls from Options.
DEFAULT_PALETTE_OPTIONS = [
    "Set1", "Set2", "Set3", "tab10", "Pastel1", "Pastel2",
    "viridis", "plasma", "magma", "inferno", "cividis",
    "Spectral", "coolwarm", "RdYlBu", "RdBu",
]


def _build_palette_options() -> list[str]:
    """Resolve the palette options from Options.* if possible."""
    try:
        from mufasa.utils.enums import Options
        return list(Options.PALETTE_OPTIONS_CATEGORICAL.value) + list(
            Options.PALETTE_OPTIONS.value,
        )
    except Exception:
        return list(DEFAULT_PALETTE_OPTIONS)


def _find_core_count() -> int:
    """Get the system CPU count; fall back to 4 on platforms where
    the helper isn't available in the sandbox."""
    try:
        from mufasa.utils.read_write import find_core_cnt
        return int(find_core_cnt()[0])
    except Exception:
        return os.cpu_count() or 4


class ValidateClassifierForm(OperationForm):
    """In-frame Qt port of the out-of-sample validation video
    runner."""

    title = "Validate classifier"
    description = (
        "Render an annotated validation video for a trained "
        "classifier on a single feature file. Shows pose tracks, "
        "classifier predictions, and an optional Gantt overlay. "
        "Use this to sanity-check a model before running batch "
        "inference across the whole project."
    )

    # ----------------------------------------------------------- State
    def __init__(self,
                 parent: QWidget | None = None,
                 config_path: str | None = None) -> None:
        self._docked_widget: QDockWidget | None = None
        self._cpu_cnt = _find_core_count()
        super().__init__(parent=parent, config_path=config_path)

    # ----------------------------------------------------------- UI
    def build(self) -> None:
        # ---- Inputs (top) ---------------------------------------- #
        self.body_layout.addWidget(
            self._build_inputs_group(), 0,
        )

        # ---- Three setting groups in a row ----------------------- #
        groups = QHBoxLayout()
        groups.addWidget(self._build_style_group(), 1)
        groups.addWidget(self._build_tracking_group(), 1)
        groups.addWidget(self._build_gantt_group(), 1)
        self.body_layout.addLayout(groups, 1)

        # ---- Action row (Pop out) -------------------------------- #
        actions = QHBoxLayout()
        self.pop_out_btn = QPushButton("Pop out ⇱", self)
        self.pop_out_btn.setToolTip(
            "Detach this form into a floating dockable window. "
            "Click again to re-dock into the workbench."
        )
        self.pop_out_btn.clicked.connect(self._toggle_pop_out)
        actions.addStretch()
        actions.addWidget(self.pop_out_btn)
        self.body_layout.addLayout(actions)

        # Re-label the inherited Run button
        self.run_btn.setText("  Create validation video")

    # ----------------------------------------------------------- Inputs group
    def _build_inputs_group(self) -> QGroupBox:
        box = QGroupBox("Inputs", self)
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        # Model path picker
        model_row = QHBoxLayout()
        self.model_path = QLineEdit(self)
        self.model_path.setPlaceholderText("Select a .sav model file…")
        model_browse = QPushButton("Browse…", self)
        model_browse.clicked.connect(self._on_browse_model)
        model_row.addWidget(self.model_path, 1)
        model_row.addWidget(model_browse)
        form.addRow("Model path (.sav):", model_row)

        # Feature file picker
        feat_row = QHBoxLayout()
        self.feature_path = QLineEdit(self)
        self.feature_path.setPlaceholderText(
            "Select a feature file (.csv / .parquet)…",
        )
        feat_browse = QPushButton("Browse…", self)
        feat_browse.clicked.connect(self._on_browse_feature)
        feat_row.addWidget(self.feature_path, 1)
        feat_row.addWidget(feat_browse)
        form.addRow("Feature file:", feat_row)

        # Threshold
        self.threshold = QLineEdit("0.5", self)
        self.threshold.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
        form.addRow("Discrimination threshold:", self.threshold)

        # Min bout length (ms)
        self.min_bout_ms = QLineEdit("0", self)
        self.min_bout_ms.setValidator(QIntValidator(0, 10_000_000, self))
        form.addRow("Minimum bout length (ms):", self.min_bout_ms)

        return box

    def _on_browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model (.sav)", self.model_path.text(),
            "SimBA classifier (*.sav)",
        )
        if path:
            self.model_path.setText(path)

    def _on_browse_feature(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select feature file",
            self.feature_path.text(),
            "Feature files (*.csv *.parquet);;All files (*)",
        )
        if path:
            self.feature_path.setText(path)

    # ----------------------------------------------------------- Style group
    def _build_style_group(self) -> QGroupBox:
        box = QGroupBox("Style", self)
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self.font_size = QComboBox(self)
        self.font_size.addItems(FONT_SIZE_OPTIONS)
        self.font_size.setCurrentText(AUTO)
        form.addRow("Font size:", self.font_size)

        self.text_space = QComboBox(self)
        self.text_space.addItems(TEXT_SPACE_OPTIONS)
        self.text_space.setCurrentText(AUTO)
        form.addRow("Text spacing:", self.text_space)

        self.circle_size = QComboBox(self)
        self.circle_size.addItems(TEXT_SPACE_OPTIONS)
        self.circle_size.setCurrentText(AUTO)
        form.addRow("Circle size:", self.circle_size)

        self.text_opacity = QComboBox(self)
        self.text_opacity.addItems(OPACITY_OPTIONS)
        self.text_opacity.setCurrentText("0.8")
        form.addRow("Text opacity:", self.text_opacity)

        self.text_thickness = QComboBox(self)
        self.text_thickness.addItems(FONT_SIZE_OPTIONS)
        self.text_thickness.setCurrentText("2")
        form.addRow("Text thickness:", self.text_thickness)

        self.bp_palette = QComboBox(self)
        palette_options = _build_palette_options()
        self.bp_palette.addItems(palette_options)
        if palette_options:
            self.bp_palette.setCurrentText(palette_options[0])
        form.addRow("Body-part palette:", self.bp_palette)

        return box

    # ----------------------------------------------------------- Tracking group
    def _build_tracking_group(self) -> QGroupBox:
        box = QGroupBox("Tracking", self)
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self.show_pose = QComboBox(self)
        self.show_pose.addItems(["TRUE", "FALSE"])
        self.show_pose.setCurrentText("TRUE")
        form.addRow("Show pose:", self.show_pose)

        self.show_animal_names = QComboBox(self)
        self.show_animal_names.addItems(["TRUE", "FALSE"])
        self.show_animal_names.setCurrentText("FALSE")
        form.addRow("Show animal names:", self.show_animal_names)

        self.show_clf_conf = QComboBox(self)
        self.show_clf_conf.addItems(["TRUE", "FALSE"])
        self.show_clf_conf.setCurrentText("TRUE")
        form.addRow("Show classifier confidence:", self.show_clf_conf)

        self.show_bbox = QComboBox(self)
        self.show_bbox.addItems(BBOX_OPTIONS)
        self.show_bbox.setCurrentText(BBOX_FALSE)
        form.addRow("Show bounding box:", self.show_bbox)

        # CPU count: 1 (single-core dispatch) ... cpu_cnt
        self.core_cnt = QComboBox(self)
        for i in range(1, self._cpu_cnt + 1):
            self.core_cnt.addItem(str(i))
        default_cores = max(1, int(math.ceil(self._cpu_cnt / 3)))
        self.core_cnt.setCurrentText(str(default_cores))
        form.addRow("CPU count:", self.core_cnt)

        return box

    # ----------------------------------------------------------- Gantt group
    def _build_gantt_group(self) -> QGroupBox:
        box = QGroupBox("Gantt overlay", self)
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel(
            "Optional Gantt chart overlay showing classifier "
            "states across time. Frame mode is faster; Video mode "
            "is more informative.", self,
        ))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.gantt_type = QComboBox(self)
        self.gantt_type.addItems(GANTT_OPTIONS)
        self.gantt_type.setCurrentText(GANTT_NONE)
        form.addRow("Gantt type:", self.gantt_type)
        layout.addLayout(form)
        layout.addStretch()
        return box

    # ----------------------------------------------------------- Pop-out
    def _toggle_pop_out(self) -> None:
        """Re-parent the form between the inline section and a
        floating QDockWidget. Same pattern as 122aj / 122al /
        122ap / 122aq."""
        if self._docked_widget is None:
            main_window = self._find_main_window()
            if main_window is None:
                QMessageBox.information(
                    self, "Pop out",
                    "No main workbench window available; the form "
                    "must stay inline.",
                )
                return
            dock = QDockWidget("Validate classifier", main_window)
            dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            dock.setFeatures(
                QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
                | QDockWidget.DockWidgetClosable
            )
            self.setParent(dock)
            dock.setWidget(self)
            main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.setFloating(True)
            dock.show()
            self._docked_widget = dock
            self.pop_out_btn.setText("Re-dock ⇲")
            store = getattr(main_window, "_validate_clf_docks", [])
            store.append(dock)
            main_window._validate_clf_docks = store
        else:
            dock = self._docked_widget
            self._docked_widget = None
            section_host = getattr(self, "_section_host", None)
            if section_host is not None:
                self.setParent(section_host)
                section_host.layout().addWidget(self)
            dock.setWidget(None)
            dock.close()
            self.pop_out_btn.setText("Pop out ⇱")

    def _find_main_window(self) -> QWidget | None:
        from PySide6.QtWidgets import QMainWindow
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QMainWindow):
                return w
            w = w.parentWidget()
        return None

    # ----------------------------------------------------------- Helpers
    @staticmethod
    def _gantt_value(text: str) -> int | None:
        """Translate the dropdown text into the int the validator
        expects: 1 = final-frame, 2 = video, None = no Gantt."""
        if text == GANTT_FRAME:
            return 1
        if text == GANTT_VIDEO:
            return 2
        return None

    @staticmethod
    def _bbox_value(text: str) -> str | None:
        """Translate the bbox dropdown into kwarg expected by
        ValidateModelOneVideoMultiprocess (None when FALSE)."""
        return None if text == BBOX_FALSE else text

    @staticmethod
    def _auto_or_int(text: str) -> int | None:
        return None if text == AUTO else int(text)

    # ----------------------------------------------------------- Execute
    def collect_args(self) -> dict:
        # Required inputs
        model_path = self.model_path.text().strip()
        feature_path = self.feature_path.text().strip()
        if not model_path:
            raise ValueError("Model path is required.")
        if not os.path.isfile(model_path):
            raise ValueError(
                f"Model file does not exist at {model_path!r}.",
            )
        if not feature_path:
            raise ValueError("Feature file path is required.")
        if not os.path.isfile(feature_path):
            raise ValueError(
                f"Feature file does not exist at {feature_path!r}.",
            )
        try:
            threshold = float(self.threshold.text().strip())
        except ValueError:
            raise ValueError(
                f"Discrimination threshold must be a float; got "
                f"{self.threshold.text()!r}",
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Discrimination threshold must be in [0.0, 1.0]; "
                f"got {threshold}",
            )
        try:
            min_bout = int(self.min_bout_ms.text().strip())
        except ValueError:
            raise ValueError(
                f"Minimum bout length must be a non-negative "
                f"integer (ms); got "
                f"{self.min_bout_ms.text()!r}",
            )
        if min_bout < 0:
            raise ValueError(
                f"Minimum bout length must be non-negative; "
                f"got {min_bout}",
            )

        # Style + tracking + gantt
        return {
            "config_path":              self.config_path,
            "model_path":               model_path,
            "feature_path":             feature_path,
            "discrimination_threshold": threshold,
            "shortest_bout":            min_bout,
            "font_size":                self._auto_or_int(
                self.font_size.currentText(),
            ),
            "text_spacing":             self._auto_or_int(
                self.text_space.currentText(),
            ),
            "circle_size":              self._auto_or_int(
                self.circle_size.currentText(),
            ),
            "text_opacity":             float(
                self.text_opacity.currentText(),
            ),
            "text_thickness":           int(
                self.text_thickness.currentText(),
            ) if self.text_thickness.currentText() != AUTO else 2,
            "bp_palette":               self.bp_palette.currentText(),
            "show_pose":                self.show_pose.currentText()
                                        == "TRUE",
            "show_animal_names":        self.show_animal_names.currentText()
                                        == "TRUE",
            "show_clf_confidence":      self.show_clf_conf.currentText()
                                        == "TRUE",
            "bbox":                     self._bbox_value(
                self.show_bbox.currentText(),
            ),
            "core_cnt":                 int(self.core_cnt.currentText()),
            "create_gantt":             self._gantt_value(
                self.gantt_type.currentText(),
            ),
        }

    def target(self, *, config_path: str, model_path: str,
               feature_path: str, discrimination_threshold: float,
               shortest_bout: int, font_size: int | None,
               text_spacing: int | None,
               circle_size: int | None, text_opacity: float,
               text_thickness: int, bp_palette: str,
               show_pose: bool, show_animal_names: bool,
               show_clf_confidence: bool, bbox: str | None,
               core_cnt: int, create_gantt: int | None) -> None:
        """Dispatch single-core vs multi-core based on core_cnt.
        Mirrors the Tk popup's __run logic exactly: single-core
        ignores bbox / bp_palette / core_cnt; multi-core takes
        the full kwarg set."""
        if core_cnt == 1:
            from mufasa.plotting.single_run_model_validation_video import (
                ValidateModelOneVideo,
            )
            ValidateModelOneVideo(
                config_path=config_path,
                feature_path=feature_path,
                model_path=model_path,
                discrimination_threshold=discrimination_threshold,
                shortest_bout=shortest_bout,
                font_size=font_size,
                create_gantt=create_gantt,
                circle_size=circle_size,
                text_spacing=text_spacing,
                show_pose=show_pose,
                text_thickness=text_thickness,
                text_opacity=text_opacity,
                show_animal_names=show_animal_names,
                show_clf_confidence=show_clf_confidence,
            ).run()
        else:
            from mufasa.plotting.single_run_model_validation_video_mp import (
                ValidateModelOneVideoMultiprocess,
            )
            ValidateModelOneVideoMultiprocess(
                config_path=config_path,
                feature_path=feature_path,
                model_path=model_path,
                discrimination_threshold=discrimination_threshold,
                shortest_bout=shortest_bout,
                font_size=font_size,
                create_gantt=create_gantt,
                bbox=bbox,
                circle_size=circle_size,
                text_spacing=text_spacing,
                show_pose=show_pose,
                text_thickness=text_thickness,
                text_opacity=text_opacity,
                bp_palette=bp_palette,
                show_animal_names=show_animal_names,
                core_cnt=core_cnt,
                show_clf_confidence=show_clf_confidence,
            ).run()


__all__ = ["ValidateClassifierForm"]
