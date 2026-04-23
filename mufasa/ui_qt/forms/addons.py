"""
mufasa.ui_qt.forms.addons
=========================

Specialty / add-on workflows. Replaces 10 legacy popups:

* Cue-light family (5): analyzer + clf-stats + movement-stats +
  visualizer + a main "launcher" popup. Consolidated into 4 forms
  sharing a cue-light-names field.
* :class:`KleinbergPopUp` — burst-detection smoothing of classifier
  outputs.
* :class:`MutualExclusivityPopUp` — pairwise conflict resolution
  across simultaneously-active classifiers.
* :class:`PupRetrievalPopUp` — pup-retrieval behavioural scoring.
* :class:`SpontaneousAlternationPopUp` — Y-maze / T-maze spontaneous
  alternation scoring.
* :class:`InitializeBlobTrackingPopUp` — launcher (OpenCV blob
  tracker parameter tuner).
* :class:`ChangeSpeedPopUp` — one-off video playback speed change;
  moves to the Tools menu alongside Reverse / Crossfade.
"""
from __future__ import annotations

import configparser
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QSpinBox,
                               QVBoxLayout, QWidget)

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.forms.analysis import _ClassifierPicker
from mufasa.ui_qt.forms.annotation import _LauncherForm
from mufasa.ui_qt.forms.data_import import _PathField
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Shared helper — cue-light names field
# --------------------------------------------------------------------------- #
def _load_cue_light_names(config_path: str) -> list[str]:
    """Read ``ROI settings.cue_light_N`` → list of cue-light ROI names.

    SimBA stores cue-light ROI names as numbered keys. If the config
    doesn't have them yet, returns empty list; the form surfaces a
    note asking the user to define cue-lights via the ROI page first.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    names: list[str] = []
    for section in ("ROI settings", "Cue light settings"):
        if not cfg.has_section(section):
            continue
        for k, v in cfg.items(section):
            if "cue_light" in k.lower() and v.strip():
                names.append(v.strip())
    return names


class _CueLightPicker(QWidget):
    """Multi-select list of cue-light ROI names, or a free-text line
    edit fallback when the config hasn't been populated yet."""

    def __init__(self, config_path: Optional[str] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self._known_names: list[str] = []
        if config_path:
            try:
                self._known_names = _load_cue_light_names(config_path)
            except Exception:
                pass
        self.list = QListWidget(self)
        self.list.setSelectionMode(QListWidget.MultiSelection)
        self.list.setMinimumHeight(80)
        for n in self._known_names:
            item = QListWidgetItem(n); item.setData(Qt.UserRole, n)
            self.list.addItem(item)
        lay.addWidget(self.list)
        # Fallback: free text when nothing in the config
        self.manual = QLineEdit(self)
        self.manual.setPlaceholderText(
            "Or type comma-separated names manually…"
        )
        lay.addWidget(self.manual)

    def selected(self) -> list[str]:
        picked = [it.data(Qt.UserRole) for it in self.list.selectedItems()]
        manual = [n.strip() for n in self.manual.text().split(",") if n.strip()]
        # dedupe-preserving-order
        out, seen = [], set()
        for n in picked + manual:
            if n not in seen:
                out.append(n); seen.add(n)
        return out


# --------------------------------------------------------------------------- #
# Cue-light forms — 4 forms (Main popup was a landing-page only,
# folded into the others)
# --------------------------------------------------------------------------- #
class CueLightDataForm(OperationForm):
    """Compute cue-light activation statistics (events, durations)."""
    title = "Cue-light — data analysis"
    description = ("Compute activation events, durations, and "
                   "inter-activation intervals for each cue-light ROI.")

    def build(self) -> None:
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight)
        self.cue_lights = _CueLightPicker(config_path=self.config_path, parent=self)
        form.addRow("Cue-lights:", self.cue_lights)
        self.detailed = QCheckBox("Emit detailed per-frame table", self)
        form.addRow("", self.detailed)
        self.cores = QSpinBox(self)
        self.cores.setRange(1, max(1, linux_env.cpu_count()))
        self.cores.setValue(max(1, linux_env.cpu_count() // 2))
        form.addRow("Cores:", self.cores)
        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        cl = self.cue_lights.selected()
        if not cl:
            raise ValueError("Select or enter at least one cue-light name.")
        return {
            "config_path":    self.config_path,
            "cue_light_names": cl,
            "detailed_data":  bool(self.detailed.isChecked()),
            "cores":          int(self.cores.value()),
        }

    def target(self, *, config_path: str, cue_light_names: list[str],
               detailed_data: bool, cores: int) -> None:
        from mufasa.data_processors.cue_light_analyzer import CueLightAnalyzer
        CueLightAnalyzer(
            config_path=config_path,
            data_dir=None,
            cue_light_names=cue_light_names,
            save_dir=None,
            core_cnt=cores,
            detailed_data=detailed_data,
            verbose=True,
        ).run()


class CueLightClfForm(OperationForm):
    """Classifier statistics relative to cue-light activations."""
    title = "Cue-light — classifier statistics"
    description = ("Compute classifier behaviour counts in pre/post "
                   "windows around each cue-light activation.")

    def build(self) -> None:
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight)
        self.cue_lights = _CueLightPicker(config_path=self.config_path, parent=self)
        form.addRow("Cue-lights:", self.cue_lights)
        self.clfs = _ClassifierPicker(config_path=self.config_path, parent=self)
        form.addRow("Classifiers:", self.clfs)
        self.pre_window = QDoubleSpinBox(self)
        self.pre_window.setRange(0.0, 3600.0); self.pre_window.setValue(5.0)
        self.pre_window.setSuffix(" s")
        form.addRow("Pre-activation window:", self.pre_window)
        self.post_window = QDoubleSpinBox(self)
        self.post_window.setRange(0.0, 3600.0); self.post_window.setValue(5.0)
        self.post_window.setSuffix(" s")
        form.addRow("Post-activation window:", self.post_window)
        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        cl = self.cue_lights.selected()
        if not cl:
            raise ValueError("Select or enter at least one cue-light name.")
        clfs = self.clfs.selected()
        if not clfs:
            raise ValueError("Select at least one classifier.")
        return {
            "config_path":    self.config_path,
            "cue_light_names": cl,
            "clf_names":      clfs,
            "pre_window":     float(self.pre_window.value()),
            "post_window":    float(self.post_window.value()),
        }

    def target(self, *, config_path: str, cue_light_names: list[str],
               clf_names: list[str], pre_window: float,
               post_window: float) -> None:
        from mufasa.data_processors.cue_light_clf_statistics import (
            CueLightClfAnalyzer,
        )
        CueLightClfAnalyzer(
            config_path=config_path,
            cue_light_names=cue_light_names,
            clf_names=clf_names,
            data_dir=None,
            pre_window=pre_window,
            post_window=post_window,
        ).run()


class CueLightMovementForm(OperationForm):
    """Movement statistics around cue-light activations."""
    title = "Cue-light — movement statistics"
    description = ("Compute per-body-part movement around each "
                   "cue-light activation window.")

    def build(self) -> None:
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight)
        self.cue_lights = _CueLightPicker(config_path=self.config_path, parent=self)
        form.addRow("Cue-lights:", self.cue_lights)
        self.bp_name = QLineEdit(self)
        self.bp_name.setPlaceholderText("Body-part name (e.g. 'Nose')")
        form.addRow("Body-part:", self.bp_name)
        self.pre_window = QDoubleSpinBox(self)
        self.pre_window.setRange(0.0, 3600.0); self.pre_window.setValue(5.0)
        self.pre_window.setSuffix(" s")
        form.addRow("Pre-activation window:", self.pre_window)
        self.post_window = QDoubleSpinBox(self)
        self.post_window.setRange(0.0, 3600.0); self.post_window.setValue(5.0)
        self.post_window.setSuffix(" s")
        form.addRow("Post-activation window:", self.post_window)
        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        cl = self.cue_lights.selected()
        if not cl:
            raise ValueError("Select or enter at least one cue-light name.")
        bp = self.bp_name.text().strip()
        if not bp:
            raise ValueError("Body-part name is required.")
        return {
            "config_path":    self.config_path,
            "cue_light_names": cl,
            "bp_name":        bp,
            "pre_window":     float(self.pre_window.value()),
            "post_window":    float(self.post_window.value()),
        }

    def target(self, *, config_path: str, cue_light_names: list[str],
               bp_name: str, pre_window: float, post_window: float) -> None:
        from mufasa.data_processors.cue_light_movement_statistics import (
            CueLightMovementAnalyzer,
        )
        CueLightMovementAnalyzer(
            config_path=config_path,
            cue_light_names=cue_light_names,
            bp_name=bp_name,
            data_dir=None,
            pre_window=pre_window,
            post_window=post_window,
            verbose=True,
        ).run()


class CueLightVisualizerForm(OperationForm):
    """Render cue-light activation overlays onto videos."""
    title = "Cue-light — visualizer"
    description = ("Overlay cue-light activation windows (as tinted "
                   "rectangles) onto videos, with optional pose overlay.")

    def build(self) -> None:
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight)
        self.cue_lights = _CueLightPicker(config_path=self.config_path, parent=self)
        form.addRow("Cue-lights:", self.cue_lights)
        self.video = _PathField(
            is_file=True,
            file_filter="Videos (*.mp4 *.avi *.mov *.mkv *.webm)",
            placeholder="Source video…",
        )
        form.addRow("Video:", self.video)
        self.show_pose = QCheckBox("Overlay pose estimates", self)
        self.show_pose.setChecked(True)
        form.addRow("", self.show_pose)
        self.frame = QCheckBox("Save frames", self)
        self.video_out = QCheckBox("Save video", self)
        self.video_out.setChecked(True)
        form.addRow("Outputs:", self.frame)
        form.addRow("", self.video_out)
        self.cores = QSpinBox(self)
        self.cores.setRange(1, max(1, linux_env.cpu_count()))
        self.cores.setValue(max(1, linux_env.cpu_count() // 2))
        form.addRow("Cores:", self.cores)
        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        cl = self.cue_lights.selected()
        if not cl:
            raise ValueError("Select or enter at least one cue-light name.")
        if not self.video.path:
            raise ValueError("Video is required.")
        return {
            "config_path":    self.config_path,
            "cue_light_names": cl,
            "video_path":     self.video.path,
            "show_pose":      bool(self.show_pose.isChecked()),
            "frame_setting":  bool(self.frame.isChecked()),
            "video_setting":  bool(self.video_out.isChecked()),
            "cores":          int(self.cores.value()),
        }

    def target(self, *, config_path: str, cue_light_names: list[str],
               video_path: str, show_pose: bool, frame_setting: bool,
               video_setting: bool, cores: int) -> None:
        from mufasa.plotting.cue_light_visualizer import CueLightVisualizer
        CueLightVisualizer(
            config_path=config_path,
            cue_light_names=cue_light_names,
            video_path=video_path,
            data_path=None,
            frame_setting=frame_setting,
            video_setting=video_setting,
            core_cnt=cores,
            show_pose=show_pose,
            verbose=True,
        ).run()


# --------------------------------------------------------------------------- #
# Kleinberg — burst-detection smoothing
# --------------------------------------------------------------------------- #
class KleinbergForm(OperationForm):
    """Apply Kleinberg burst-detection smoothing to classifier outputs.

    Reduces spurious single-frame classifier flips by modelling each
    classifier's on/off stream as a two-state automaton with configurable
    promotion cost (sigma) and decay (gamma)."""

    title = "Kleinberg burst smoothing"
    description = ("Kleinberg (2003) two-state burst smoothing on "
                   "classifier streams. Higher <b>sigma</b> = more "
                   "conservative state changes; higher <b>gamma</b> = "
                   "faster decay back to the baseline state.")

    def build(self) -> None:
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight)
        self.clfs = _ClassifierPicker(config_path=self.config_path, parent=self)
        form.addRow("Classifiers:", self.clfs)
        self.sigma = QDoubleSpinBox(self)
        self.sigma.setRange(1.0, 20.0); self.sigma.setValue(2.0); self.sigma.setSingleStep(0.5)
        form.addRow("Sigma (promotion cost):", self.sigma)
        self.gamma = QDoubleSpinBox(self)
        self.gamma.setRange(0.0, 10.0); self.gamma.setValue(0.3); self.gamma.setSingleStep(0.05)
        form.addRow("Gamma (decay):", self.gamma)
        self.hierarchy = QSpinBox(self)
        self.hierarchy.setRange(1, 20); self.hierarchy.setValue(1)
        form.addRow("Hierarchy level:", self.hierarchy)
        self.save_originals = QCheckBox("Keep original classifier columns", self)
        self.save_originals.setChecked(True)
        form.addRow("", self.save_originals)
        self.hierarchical_search = QCheckBox(
            "Search best hierarchy level automatically", self,
        )
        form.addRow("", self.hierarchical_search)
        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        clfs = self.clfs.selected()
        if not clfs:
            raise ValueError("Select at least one classifier.")
        return {
            "config_path":        self.config_path,
            "classifier_names":   clfs,
            "sigma":              float(self.sigma.value()),
            "gamma":              float(self.gamma.value()),
            "hierarchy":          int(self.hierarchy.value()),
            "save_originals":     bool(self.save_originals.isChecked()),
            "hierarchical_search": bool(self.hierarchical_search.isChecked()),
        }

    def target(self, *, config_path: str, classifier_names: list[str],
               sigma: float, gamma: float, hierarchy: int,
               save_originals: bool, hierarchical_search: bool) -> None:
        from mufasa.data_processors.kleinberg_calculator import KleinbergCalculator
        KleinbergCalculator(
            config_path=config_path,
            classifier_names=classifier_names,
            sigma=sigma,
            gamma=gamma,
            hierarchy=hierarchy,
            verbose=True,
            save_originals=save_originals,
            hierarchical_search=hierarchical_search,
            input_dir=None,
            output_dir=None,
        ).run()


# --------------------------------------------------------------------------- #
# Mutual exclusivity
# --------------------------------------------------------------------------- #
class MutualExclusivityForm(OperationForm):
    """Resolve pairwise conflicts between simultaneously-active
    classifiers using a rules CSV."""

    title = "Mutual exclusivity corrector"
    description = ("Resolve frames where two or more classifiers are "
                   "simultaneously active using a rules file "
                   "(one rule per row: conflicting pair + resolution).")

    def build(self) -> None:
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight)
        self.rules = _PathField(
            is_file=True, file_filter="CSV (*.csv)",
            placeholder="Path to rules CSV…",
        )
        form.addRow("Rules CSV:", self.rules)
        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        if not self.rules.path:
            raise ValueError("Rules CSV is required.")
        return {"config_path": self.config_path, "rules": self.rules.path}

    def target(self, *, config_path: str, rules: str) -> None:
        from mufasa.data_processors.mutual_exclusivity_corrector import (
            MutualExclusivityCorrector,
        )
        MutualExclusivityCorrector(rules=rules, config_path=config_path).run()


# --------------------------------------------------------------------------- #
# Pup retrieval
# --------------------------------------------------------------------------- #
class PupRetrievalForm(OperationForm):
    """Score pup-retrieval behaviour.

    The legacy popup collected ~15 settings (pup body-parts, dam
    body-parts, threshold, smoothing window, zone names, etc.) and
    passed them as a ``settings`` dict to
    :class:`PupRetrieverCalculator`. The form mirrors that contract.
    """

    title = "Pup retrieval analysis"
    description = ("Score when a dam retrieves pups into a nest zone. "
                   "Requires the ROI named below plus correctly-named "
                   "body-parts for dam and pups in the pose config.")

    def build(self) -> None:
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight)
        self.dam_bp = QLineEdit(self)
        self.dam_bp.setPlaceholderText("e.g. 'Dam_nose'")
        form.addRow("Dam body-part:", self.dam_bp)
        self.pup_bp = QLineEdit(self)
        self.pup_bp.setPlaceholderText("e.g. 'Pup_center'")
        form.addRow("Pup body-part:", self.pup_bp)
        self.nest_roi = QLineEdit(self)
        self.nest_roi.setPlaceholderText("Nest ROI name")
        form.addRow("Nest ROI:", self.nest_roi)
        self.approach_dist = QDoubleSpinBox(self)
        self.approach_dist.setRange(0.0, 1e5); self.approach_dist.setValue(50.0)
        self.approach_dist.setSuffix(" mm")
        form.addRow("Approach distance:", self.approach_dist)
        self.threshold = QDoubleSpinBox(self)
        self.threshold.setRange(0.0, 1.0); self.threshold.setValue(0.5)
        self.threshold.setSingleStep(0.05)
        form.addRow("Pose probability threshold:", self.threshold)
        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        for label, widget in (("dam body-part", self.dam_bp),
                              ("pup body-part", self.pup_bp),
                              ("nest ROI",      self.nest_roi)):
            if not widget.text().strip():
                raise ValueError(f"{label.capitalize()} is required.")
        settings = {
            "dam_bp":           self.dam_bp.text().strip(),
            "pup_bp":           self.pup_bp.text().strip(),
            "nest_roi":         self.nest_roi.text().strip(),
            "approach_distance": float(self.approach_dist.value()),
            "threshold":        float(self.threshold.value()),
        }
        return {"config_path": self.config_path, "settings": settings}

    def target(self, *, config_path: str, settings: dict) -> None:
        from mufasa.data_processors.pup_retrieval_calculator import (
            PupRetrieverCalculator,
        )
        PupRetrieverCalculator(config_path=config_path, settings=settings).run()


# --------------------------------------------------------------------------- #
# Spontaneous alternation
# --------------------------------------------------------------------------- #
class SpontaneousAlternationForm(OperationForm):
    """Y-maze / T-maze / radial-maze spontaneous alternation scoring."""

    title = "Spontaneous alternation"
    description = ("Score arm-entry patterns for Y-maze, T-maze or "
                   "radial-arm maze paradigms. Requires ROI definitions "
                   "for each arm plus a centre zone.")

    def build(self) -> None:
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignRight)
        self.arm_names = QLineEdit(self)
        self.arm_names.setPlaceholderText("Comma-separated arm ROI names")
        form.addRow("Arm ROIs:", self.arm_names)
        self.center_name = QLineEdit(self)
        self.center_name.setPlaceholderText("Centre ROI name")
        form.addRow("Centre ROI:", self.center_name)
        self.animal_area = QSpinBox(self)
        self.animal_area.setRange(1, 99); self.animal_area.setValue(80)
        self.animal_area.setSuffix(" % of bps in zone")
        form.addRow("Entry threshold:", self.animal_area)
        self.threshold = QDoubleSpinBox(self)
        self.threshold.setRange(0.0, 1.0); self.threshold.setValue(0.5)
        self.threshold.setSingleStep(0.05)
        form.addRow("Pose probability threshold:", self.threshold)
        self.buffer = QSpinBox(self)
        self.buffer.setRange(0, 1000); self.buffer.setValue(2)
        self.buffer.setSuffix(" frames")
        form.addRow("Dwell buffer:", self.buffer)
        self.detailed = QCheckBox("Emit detailed data table", self)
        form.addRow("", self.detailed)
        self.make_plot = QCheckBox("Also render visualization video", self)
        form.addRow("", self.make_plot)
        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        arms = [a.strip() for a in self.arm_names.text().split(",") if a.strip()]
        if len(arms) < 2:
            raise ValueError("Enter at least 2 arm ROI names (comma-separated).")
        centre = self.center_name.text().strip()
        if not centre:
            raise ValueError("Centre ROI name is required.")
        return {
            "config_path":  self.config_path,
            "arm_names":    arms,
            "center_name":  centre,
            "animal_area":  int(self.animal_area.value()),
            "threshold":    float(self.threshold.value()),
            "buffer":       int(self.buffer.value()),
            "detailed":     bool(self.detailed.isChecked()),
            "make_plot":    bool(self.make_plot.isChecked()),
        }

    def target(self, *, config_path: str, arm_names: list[str],
               center_name: str, animal_area: int, threshold: float,
               buffer: int, detailed: bool, make_plot: bool) -> None:
        from mufasa.data_processors.spontaneous_alternation_calculator import (
            SpontaneousAlternationCalculator,
        )
        SpontaneousAlternationCalculator(
            config_path=config_path,
            arm_names=arm_names,
            center_name=center_name,
            animal_area=animal_area,
            threshold=threshold,
            buffer=buffer,
            verbose=True,
            detailed_data=detailed,
            data_path=None,
        ).run()
        if make_plot:
            from mufasa.plotting.spontaneous_alternation_plotter import (
                SpontaneousAlternationsPlotter,
            )
            SpontaneousAlternationsPlotter(
                config_path=config_path,
                arm_names=arm_names,
                center_name=center_name,
                animal_area=animal_area,
                threshold=threshold,
                buffer=buffer,
                core_cnt=max(1, linux_env.cpu_count() // 2),
                verbose=True,
                data_path=None,
            ).run()


# --------------------------------------------------------------------------- #
# Blob-tracker initialisation — OpenCV interactive tuner, launcher only
# --------------------------------------------------------------------------- #
class BlobTrackerInitLauncher(_LauncherForm):
    """Launcher for :class:`BlobTrackingUI`. The UI is a multi-trackbar
    OpenCV window for tuning blob-detection parameters (threshold,
    erode/dilate kernels, min/max area); Qt can't preview OpenCV frames
    without re-implementing the video surface, so this stays a
    launcher until that piece lands."""
    title = "Initialise blob tracker"
    description = ("Interactive OpenCV tuner for blob-tracker "
                   "parameters. Produces a saved settings JSON "
                   "consumed by downstream blob-tracker runs.")
    launch_button_text = "Open blob-tracker tuner (legacy UI)…"
    launch_title = "Blob tracker"
    launch_message = (
        "The blob-tracker initialisation UI uses OpenCV trackbars "
        "plus a live preview. That's still easier to maintain as a "
        "direct OpenCV window than to reimplement in Qt. Use "
        "<code>mufasa-tk</code> to launch the tuner; the resulting "
        "JSON settings file then feeds the Visualizations → 'Blob "
        "tracker output (external)' form."
    )


__all__ = [
    "CueLightDataForm",
    "CueLightClfForm",
    "CueLightMovementForm",
    "CueLightVisualizerForm",
    "KleinbergForm",
    "MutualExclusivityForm",
    "PupRetrievalForm",
    "SpontaneousAlternationForm",
    "BlobTrackerInitLauncher",
]
