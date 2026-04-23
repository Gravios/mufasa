"""
mufasa.ui_qt.forms.annotation
=============================

Inline forms for the annotation workflow:

* :class:`FrameLabellingLauncher` — wraps :class:`LabellingInterface`
  launches (both the standard and pseudo-labelling variants) as a
  dialog. The labelling UI itself is heavily Tk/OpenCV-specific
  (frame-scrubbing video player + per-class checkbox grid with
  keystroke bindings), so it stays as a launcher for now.
* :class:`TargetedAnnotationClipsLauncher` — same reasoning:
  :class:`multi_split_video`'s front-end is an interactive Tk
  table editor. Launcher keeps this accessible from the workbench
  without a full Qt port.
* :class:`ThirdPartyAppenderForm` — inline form wrapping
  :class:`ThirdPartyLabelAppender`. Straightforward scope picker +
  format dropdown.
* :class:`AnnotationReportsForm` — action dropdown combining
  :class:`AnnotationFrameExtractor` (export labelled frames as
  images) and :class:`AnnotationMetaDataExtractor` (compute per-clf
  annotation counts).

**4 popups → 4 inline surfaces (2 forms + 2 dialog launchers).**
"""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)
from mufasa.ui_qt.forms.analysis import _ClassifierPicker
from mufasa.ui_qt.forms.data_import import _PathField
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# Shared helper: dialog-launcher pattern
# --------------------------------------------------------------------------- #
class _LauncherForm(OperationForm):
    """Base class for forms that are just a 'Launch …' button + message
    for operations still backed by legacy Tk UIs. Hides the default Run
    button and substitutes a single launch action."""

    launch_button_text: str = "Launch…"
    launch_message: str = ("This operation uses the legacy interactive "
                           "(Tk) UI. The Qt port is on the roadmap.")
    launch_title: str = "Not yet ported"

    def build(self) -> None:
        msg = QLabel(f"<i>{self.launch_message}</i>", self)
        msg.setWordWrap(True)
        self.body_layout.addWidget(msg)
        row = QHBoxLayout()
        self.launch_btn = QPushButton(self.launch_button_text, self)
        self.launch_btn.clicked.connect(self._do_launch)
        row.addWidget(self.launch_btn)
        row.addStretch()
        self.body_layout.addLayout(row)
        self.run_btn.setVisible(False)

    # Subclass hook
    def _do_launch(self) -> None:  # pragma: no cover
        QMessageBox.information(self.window(), self.launch_title,
                                self.launch_message)

    # OperationForm interface — not used, launcher handles everything
    def collect_args(self) -> dict: return {}
    def target(self, **kwargs) -> None: pass  # pragma: no cover


# --------------------------------------------------------------------------- #
# FrameLabellingLauncher — launches the Qt FrameLabellerDialog
# --------------------------------------------------------------------------- #
class FrameLabellingLauncher(OperationForm):
    """Launch the Qt frame-labeller for behavioural annotation.

    Replaces :class:`SelectVideoForLabellingPopUp` and
    :class:`SelectVideoForPseudoLabellingPopUp`. The mode dropdown
    covers all three pathways the legacy UI supported: fresh
    annotation, continue previous annotation, or pseudo-labelling
    from an existing machine_results file.
    """

    title = "Frame labelling"
    description = ("Frame-by-frame behavioural annotation. Opens a "
                   "scrubber + per-classifier checkboxes dialog. "
                   "Use <b>Continue</b> to resume a prior session, or "
                   "<b>Pseudo</b> to seed labels from an existing "
                   "classifier inference file.")

    MODES = [("New labelling",        "new"),
             ("Continue labelling",   "continue"),
             ("Pseudo-labelling (seed from machine_results)", "pseudo")]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.mode_cb = QComboBox(self)
        for label, _ in self.MODES:
            self.mode_cb.addItem(label)
        form.addRow("Mode:", self.mode_cb)
        self.body_layout.addLayout(form)

        launch = QPushButton("  Select video and launch labeller…", self)
        launch.setStyleSheet("padding: 8px 16px; font-size: 11pt;")
        launch.clicked.connect(self._launch)
        self.body_layout.addWidget(launch)

        # Labellers are long-running interactive sessions; the base
        # class's Run button doesn't fit this flow.
        self.run_btn.setVisible(False)

    def _launch(self) -> None:
        if not self.config_path:
            QMessageBox.warning(
                self.window(), "No project",
                "Load a project (project_config.ini) before labelling.",
            )
            return
        mode = self.MODES[self.mode_cb.currentIndex()][1]
        # Import here to keep module-load light — cv2 comes with a cost
        from mufasa.ui_qt.frame_labeller import launch_frame_labeller
        launch_frame_labeller(self.window(),
                              config_path=self.config_path, mode=mode)

    # OperationForm interface — unused, launcher handles everything
    def collect_args(self) -> dict:
        return {}

    def target(self, **kwargs) -> None:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# TargetedAnnotationClipsLauncher
# --------------------------------------------------------------------------- #
class TargetedAnnotationClipsLauncher(OperationForm):
    """Launch the Qt targeted-clips editor. Replaces
    :class:`TargetedAnnotationsWClipsPopUp`.

    The legacy Tk UI collected a clip count up front and built a fixed
    table of ``HH:MM:SS`` entry boxes; the Qt version is driven by the
    scrubber — mark start at the current frame, mark end, repeat. Rows
    remain editable for exact-value entry.
    """

    title = "Targeted annotation clips"
    description = ("Define multiple clip ranges within a video for "
                   "targeted annotation. Each range is extracted as a "
                   "video clip + a slice of the machine_results CSV.")

    def build(self) -> None:
        hint = QLabel(
            "<i>Scrub to a start frame, click <b>Mark start</b>; scrub "
            "to the end, click <b>Mark end</b>. Repeat for each clip. "
            "Extracted clips + data slices go to "
            "<code>frames/input/advanced_clip_annotator/{video}/</code>.</i>",
            self,
        )
        hint.setWordWrap(True)
        self.body_layout.addWidget(hint)

        launch = QPushButton("  Select video and launch clip editor…", self)
        launch.setStyleSheet("padding: 8px 16px; font-size: 11pt;")
        launch.clicked.connect(self._launch)
        self.body_layout.addWidget(launch)

        self.run_btn.setVisible(False)

    def _launch(self) -> None:
        if not self.config_path:
            QMessageBox.warning(
                self.window(), "No project",
                "Load a project before defining clips.",
            )
            return
        from mufasa.ui_qt.targeted_clips import launch_targeted_clips
        launch_targeted_clips(self.window(), config_path=self.config_path)

    def collect_args(self) -> dict:
        return {}

    def target(self, **kwargs) -> None:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# ThirdPartyAppenderForm — inline
# --------------------------------------------------------------------------- #
class ThirdPartyAppenderForm(OperationForm):
    """Append annotations produced by BORIS / BENTO / ETHOVISION /
    SOLOMON / DeepEthogram into a Mufasa project.

    Format dropdown drives the backend's ``app`` + ``file_format``
    args. The list of supported third-party formats is discovered at
    runtime from the lookup table so new backends propagate
    automatically.
    """

    title = "Append third-party annotations"
    description = ("Merge externally-annotated behaviour files "
                   "(BORIS, BENTO, Ethovision, Solomon Coder, "
                   "DeepEthogram) into the project.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.source = _PathField(is_file=False,
                                 placeholder="Directory of annotation files…")
        form.addRow("Source directory:", self.source)

        self.app_cb = QComboBox(self)
        apps = self._discover_apps()
        if apps:
            for app in apps.keys():
                self.app_cb.addItem(app)
        else:
            self.app_cb.addItems(["BORIS", "BENTO", "ETHOVISION",
                                  "SOLOMON", "DEEPETHOGRAM"])
        self.app_cb.currentTextChanged.connect(self._on_app_changed)
        form.addRow("Third-party app:", self.app_cb)

        self.fmt_cb = QComboBox(self)
        form.addRow("File format:", self.fmt_cb)

        self.error_mode = QComboBox(self)
        self.error_mode.addItems(["WARNING", "ERROR"])
        form.addRow("On format mismatch:", self.error_mode)

        self.log = QCheckBox("Write merge log", self)
        self.log.setChecked(True)
        form.addRow("", self.log)

        self.body_layout.addLayout(form)
        self._on_app_changed(self.app_cb.currentText())

    def _discover_apps(self) -> dict[str, list[str]]:
        """Query the backend lookup for {app_name: [file_formats]}."""
        try:
            from mufasa.utils.lookups import get_third_party_appender_file_formats
            return get_third_party_appender_file_formats()
        except Exception:
            return {}

    def _on_app_changed(self, app: str) -> None:
        apps = self._discover_apps()
        self.fmt_cb.clear()
        if app in apps:
            self.fmt_cb.addItems(apps[app])
        else:
            # Fallback defaults
            fallback = {
                "BORIS": ["csv"], "BENTO": ["annot"], "ETHOVISION": ["xlsx"],
                "SOLOMON": ["csv"], "DEEPETHOGRAM": ["csv"],
            }.get(app, ["csv"])
            self.fmt_cb.addItems(fallback)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        src = self.source.path
        if not src:
            raise ValueError("Source directory is required.")
        return {
            "config_path":  self.config_path,
            "data_dir":     src,
            "app":          self.app_cb.currentText(),
            "file_format":  self.fmt_cb.currentText(),
            "error_settings": self.error_mode.currentText(),
            "log":          bool(self.log.isChecked()),
        }

    def target(self, *, config_path: str, data_dir: str, app: str,
               file_format: str, error_settings: str, log: bool) -> None:
        from mufasa.third_party_label_appenders.third_party_appender import (
            ThirdPartyLabelAppender,
        )
        ThirdPartyLabelAppender(
            config_path=config_path,
            data_dir=data_dir,
            app=app,
            file_format=file_format,
            error_settings=error_settings,
            log=log,
        ).run()


# --------------------------------------------------------------------------- #
# AnnotationReportsForm — 2 popups → 1 form
# --------------------------------------------------------------------------- #
class AnnotationReportsForm(OperationForm):
    """Produce reports / exports from existing annotations:

    * **Extract labelled frames as images** — one image per frame per
      classifier, filterable by downsampling factor.
    * **Annotation counts** — per-classifier statistics (events,
      durations).

    Replaces :class:`ExtractAnnotationFramesPopUp` and
    :class:`ClfAnnotationCountsPopUp`.
    """

    title = "Annotation reports"
    description = ("Export labelled frames as images, or compute "
                   "per-classifier annotation statistics.")

    ACTIONS = [("Extract labelled frames", "extract_frames"),
               ("Annotation counts",        "counts")]

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.action_cb = QComboBox(self)
        for label, _ in self.ACTIONS:
            self.action_cb.addItem(label)
        self.action_cb.currentIndexChanged.connect(self._on_action_changed)
        form.addRow("Action:", self.action_cb)

        self.clf_picker = _ClassifierPicker(
            config_path=self.config_path, parent=self,
        )
        form.addRow("Classifiers:", self.clf_picker)

        self.panels = QStackedWidget(self)

        # --- Extract-frames panel --- #
        ef_host = QWidget()
        ef_form = QFormLayout(ef_host); ef_form.setContentsMargins(0, 0, 0, 0)
        self.downsample = QSpinBox(self)
        self.downsample.setRange(1, 20); self.downsample.setValue(1)
        ef_form.addRow("Downsample factor:", self.downsample)
        self.img_fmt = QComboBox(self)
        self.img_fmt.addItems(["png", "jpg", "bmp", "webp"])
        ef_form.addRow("Image format:", self.img_fmt)
        self.greyscale = QCheckBox("Greyscale", self)
        ef_form.addRow("", self.greyscale)
        self.panels.addWidget(ef_host)

        # --- Counts panel --- #
        cn_host = QWidget()
        cn_form = QFormLayout(cn_host); cn_form.setContentsMargins(0, 0, 0, 0)
        self.split_by_video = QCheckBox("Split counts per video", self)
        self.split_by_video.setChecked(True)
        cn_form.addRow("", self.split_by_video)
        self.annotated_bouts = QCheckBox("Include annotated bouts", self)
        cn_form.addRow("", self.annotated_bouts)
        self.panels.addWidget(cn_host)

        form.addRow("Parameters:", self.panels)
        self.body_layout.addLayout(form)

    def _on_action_changed(self, idx: int) -> None:
        self.panels.setCurrentIndex(idx)
        # Classifiers required only for extract_frames
        self.clf_picker.setEnabled(idx == 0)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        action = self.ACTIONS[self.action_cb.currentIndex()][1]
        args: dict = {"config_path": self.config_path, "action": action}
        if action == "extract_frames":
            clfs = self.clf_picker.selected()
            if not clfs:
                raise ValueError("Select at least one classifier to extract.")
            args["clfs"] = clfs
            args["img_downsample_factor"] = int(self.downsample.value())
            args["img_format"] = self.img_fmt.currentText()
            args["img_greyscale"] = bool(self.greyscale.isChecked())
        else:
            args["split_by_video"] = bool(self.split_by_video.isChecked())
            args["annotated_bouts"] = bool(self.annotated_bouts.isChecked())
        return args

    def target(self, *, config_path: str, action: str, **params) -> None:
        if action == "extract_frames":
            from mufasa.labelling.extract_labelled_frames import (
                AnnotationFrameExtractor,
            )
            AnnotationFrameExtractor(
                config_path=config_path,
                data_paths=None,
                clfs=params["clfs"],
                img_downsample_factor=params["img_downsample_factor"],
                img_format=params["img_format"],
                img_greyscale=params["img_greyscale"],
            ).run()
        elif action == "counts":
            from mufasa.labelling.extract_labelling_meta import (
                AnnotationMetaDataExtractor,
            )
            AnnotationMetaDataExtractor(
                config_path=config_path,
                split_by_video=params["split_by_video"],
                annotated_bouts=params["annotated_bouts"],
            ).run()


__all__ = [
    "FrameLabellingLauncher",
    "TargetedAnnotationClipsLauncher",
    "ThirdPartyAppenderForm",
    "AnnotationReportsForm",
    "ClipReviewLauncher",
]


# --------------------------------------------------------------------------- #
# ClipReviewLauncher — interactive bout-by-bout review
# --------------------------------------------------------------------------- #
class ClipReviewLauncher(OperationForm):
    """Launch the interactive clip-review dialog for bout-by-bout
    validation of classifier predictions.

    Complements (doesn't replace) the batch-render
    :class:`ClassifierValidationClips` backend that's still available
    via the Visualizations page. Interactive review is faster when
    the user just wants to walk through bouts and mark them valid /
    invalid; batch rendering is better when the output needs to
    leave the workstation.
    """

    title = "Review classifier predictions (interactive)"
    description = ("Step through each classifier-predicted bout and "
                   "mark it valid / invalid / unsure. Ratings are "
                   "saved to <code>validation_results/{video}.csv</code>.")

    def build(self) -> None:
        hint = QLabel(
            "<i>Select a video + its machine_results CSV. The dialog "
            "shows each bout on a timeline, with a scrubber to verify "
            "each detection.</i>", self,
        )
        hint.setWordWrap(True)
        self.body_layout.addWidget(hint)

        launch = QPushButton("  Select video and launch reviewer…", self)
        launch.setStyleSheet("padding: 8px 16px; font-size: 11pt;")
        launch.clicked.connect(self._launch)
        self.body_layout.addWidget(launch)

        self.run_btn.setVisible(False)

    def _launch(self) -> None:
        if not self.config_path:
            QMessageBox.warning(
                self.window(), "No project",
                "Load a project (project_config.ini) before reviewing.",
            )
            return
        from mufasa.ui_qt.clip_review import launch_clip_review
        launch_clip_review(self.window(), config_path=self.config_path)

    def collect_args(self) -> dict:
        return {}

    def target(self, **kwargs) -> None:  # pragma: no cover
        pass
