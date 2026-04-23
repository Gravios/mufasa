"""
mufasa.ui_qt.dialogs.frame_labeller
===================================

Qt-native replacement for :class:`mufasa.labelling.standard_labeller.LabellingInterface`.

The legacy Tk interface is a single Toplevel window combining a video
frame, per-classifier checkboxes, keystroke shortcuts, and a jog bar.
This Qt port provides the same workflow:

* Open a video.
* For every classifier in the project, a checkbox toggles the current
  frame's "is this behaviour present?" label.
* Jog through frames with keys, slider, or spinbox.
* Save to the project's ``targets_inserted/{video_name}.{file_type}``
  CSV on demand.
* Pseudo-labelling mode seeds initial labels from an existing
  ``machine_results`` file.
* Continue-labelling mode loads existing ``targets_inserted`` labels.

Design
------

* Random-access via :class:`FrameScrubberWidget` — no playback thread.
* The label matrix is an in-memory dict ``{classifier_name: np.ndarray}``
  of ``total_frames`` booleans, flushed to CSV on Save. No per-frame
  disk writes — annotation is fast even over ThinLinc.
* Keystroke bindings: 1-9 toggle classifiers 0-8; ``←``/``→`` jog
  one frame; ``Shift+←/→`` ten frames; ``Ctrl+S`` save; ``Space``
  advances one frame (fast rapid-fire labelling).

Saved format exactly matches what the legacy labeller writes, so
downstream feature-extraction / model-training pipelines keep
working without modification.
"""
from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from mufasa.ui_qt.forms.analysis import _load_classifier_names
from mufasa.ui_qt.frame_scrubber import FrameScrubberWidget


class FrameLabellerDialog(QDialog):
    """Modeless dialog hosting the frame scrubber plus per-classifier
    toggles. Replaces the Tk ``LabellingInterface``."""

    def __init__(self,
                 config_path: str,
                 video_path: str,
                 *,
                 mode: str = "new",
                 parent: Optional[QWidget] = None) -> None:
        """
        Parameters
        ----------
        config_path : str
            Path to the Mufasa project_config.ini.
        video_path : str
            Path to the video being annotated.
        mode : {"new", "continue", "pseudo"}
            * "new" — start from all-zero labels.
            * "continue" — load labels from
              ``targets_inserted/{video_name}.{file_type}``.
            * "pseudo" — load initial labels from
              ``machine_results/{video_name}.{file_type}``.
        """
        super().__init__(parent)
        self.config_path = config_path
        self.video_path = video_path
        self.mode = mode
        self.video_name = Path(video_path).stem

        self._classifier_names: list[str] = []
        self._clf_cbs: dict[str, QCheckBox] = {}
        self._labels: dict[str, np.ndarray] = {}
        self._dirty: bool = False

        self.setWindowTitle(f"Label frames — {self.video_name}")
        self.resize(1100, 720)
        self._load_project_metadata()
        self._build_ui()
        self._setup_shortcuts()
        self.scrubber.load(video_path)
        self._initialize_labels()
        self.scrubber.frame_changed.connect(self._on_frame_changed)
        # Prime the UI with frame 0's label state
        self._on_frame_changed(0)

    # ------------------------------------------------------------------ #
    # Project metadata loading
    # ------------------------------------------------------------------ #
    def _load_project_metadata(self) -> None:
        """Discover classifier list + project paths from config."""
        self._classifier_names = _load_classifier_names(self.config_path)
        if not self._classifier_names:
            raise RuntimeError(
                "No classifiers defined in project_config.ini. Add at "
                "least one via the Classifier → Manage page before "
                "labelling."
            )
        cfg = configparser.ConfigParser()
        cfg.read(self.config_path)
        project_path = cfg.get("General settings", "project_path")
        self.file_type = cfg.get("General settings", "workflow_file_type",
                                 fallback="csv")
        self.features_dir = os.path.join(project_path, "csv",
                                         "features_extracted")
        self.targets_dir = os.path.join(project_path, "csv",
                                        "targets_inserted")
        self.machine_results_dir = os.path.join(project_path, "csv",
                                                "machine_results")
        os.makedirs(self.targets_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        # Scrubber
        self.scrubber = FrameScrubberWidget(self)
        outer.addWidget(self.scrubber, 1)

        # Per-classifier checkbox grid + keystroke hint
        clf_bar = QHBoxLayout()
        clf_bar.addWidget(QLabel("<b>Behaviours at this frame:</b>", self))
        for i, name in enumerate(self._classifier_names[:9]):
            cb = QCheckBox(f"{i + 1}. {name}", self)
            cb.toggled.connect(lambda checked, n=name:
                               self._on_clf_toggled(n, checked))
            self._clf_cbs[name] = cb
            clf_bar.addWidget(cb)
        # Overflow (≥ 10 classifiers) — no keystroke, but still in UI
        for name in self._classifier_names[9:]:
            cb = QCheckBox(name, self)
            cb.toggled.connect(lambda checked, n=name:
                               self._on_clf_toggled(n, checked))
            self._clf_cbs[name] = cb
            clf_bar.addWidget(cb)
        clf_bar.addStretch()
        outer.addLayout(clf_bar)

        # Hint + status
        hint = QLabel(
            "<i>Keys: 1–9 toggle behaviours · ← / → jog 1 frame · "
            "Shift+← / Shift+→ jog 10 · Space = next frame · Ctrl+S save</i>",
            self,
        )
        hint.setStyleSheet("color: #555;")
        outer.addWidget(hint)

        self.status = QLabel("Ready.", self)
        outer.addWidget(self.status)

        # Save / close buttons
        btns = QDialogButtonBox(self)
        self.save_btn = btns.addButton("Save", QDialogButtonBox.ApplyRole)
        self.save_btn.clicked.connect(self._save)
        close_btn = btns.addButton("Close", QDialogButtonBox.RejectRole)
        close_btn.clicked.connect(self.reject)
        outer.addWidget(btns)

    def _setup_shortcuts(self) -> None:
        # 1-9 → toggle classifier i-1
        for i, name in enumerate(self._classifier_names[:9]):
            sc = QShortcut(QKeySequence(str(i + 1)), self)
            sc.activated.connect(lambda n=name: self._toggle_clf(n))
        # Jog keys
        for key, delta in [(Qt.Key_Left, -1), (Qt.Key_Right, 1)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda d=delta: self.scrubber.seek(
                self.scrubber.current_frame + d))
        for key, delta in [(QKeySequence(Qt.ShiftModifier | Qt.Key_Left), -10),
                           (QKeySequence(Qt.ShiftModifier | Qt.Key_Right), 10)]:
            sc = QShortcut(key, self)
            sc.activated.connect(lambda d=delta: self.scrubber.seek(
                self.scrubber.current_frame + d))
        # Space → next frame (rapid-fire labelling)
        sp = QShortcut(QKeySequence(Qt.Key_Space), self)
        sp.activated.connect(lambda: self.scrubber.seek(
            self.scrubber.current_frame + 1))
        # Ctrl+S → save
        ss = QShortcut(QKeySequence.Save, self)
        ss.activated.connect(self._save)

    # ------------------------------------------------------------------ #
    # Label-state management
    # ------------------------------------------------------------------ #
    def _initialize_labels(self) -> None:
        """Allocate label arrays and, if mode requires, load starting values."""
        n = self.scrubber.total_frames
        self._labels = {name: np.zeros(n, dtype=np.uint8)
                        for name in self._classifier_names}

        if self.mode == "continue":
            self._load_existing_labels(self.targets_dir, "continue")
        elif self.mode == "pseudo":
            self._load_existing_labels(self.machine_results_dir, "pseudo")
        # mode == "new" → keep zeros

    def _load_existing_labels(self, source_dir: str, label: str) -> None:
        """Read classifier columns from an existing CSV, if present.

        NOTE: mufasa.utils.read_write.read_df has an upstream quirk for
        CSV input — it unconditionally strips the first column
        (``iloc[:, 1:]``) regardless of the ``has_index`` parameter.
        That would eat a real classifier column. Read CSV directly via
        pandas; fall back to read_df for .parquet / .pickle only.
        """
        src = os.path.join(source_dir, f"{self.video_name}.{self.file_type}")
        if not os.path.isfile(src):
            self.status.setText(
                f"{label.capitalize()} source not found; starting from zeros."
            )
            return
        df = None
        try:
            import pandas as pd
            if self.file_type.lower() == "csv":
                df = pd.read_csv(src)
            else:
                from mufasa.utils.read_write import read_df
                df = read_df(src, self.file_type)
        except Exception as exc:
            self.status.setText(f"Could not read {src}: {exc}")
            return
        loaded = []
        n = self.scrubber.total_frames
        for name in self._classifier_names:
            if name in df.columns:
                col = df[name].to_numpy()
                # Clamp to video length if CSV disagrees (rare but possible)
                if col.shape[0] != n:
                    col = col[:n] if col.shape[0] > n else np.pad(
                        col, (0, n - col.shape[0]), constant_values=0)
                self._labels[name] = col.astype(np.uint8)
                loaded.append(name)
        self.status.setText(
            f"Loaded {len(loaded)} classifier column(s) from {os.path.basename(src)}."
        )
        loaded = []
        n = self.scrubber.total_frames
        for name in self._classifier_names:
            if name in df.columns:
                col = df[name].to_numpy()
                # Clamp to video length if CSV disagrees (rare but possible)
                if col.shape[0] != n:
                    col = col[:n] if col.shape[0] > n else np.pad(
                        col, (0, n - col.shape[0]), constant_values=0)
                self._labels[name] = col.astype(np.uint8)
                loaded.append(name)
        self.status.setText(
            f"Loaded {len(loaded)} classifier column(s) from {os.path.basename(src)}."
        )

    def _on_frame_changed(self, frame_idx: int) -> None:
        """Refresh checkbox states to reflect the new frame's labels."""
        for name, cb in self._clf_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(bool(self._labels[name][frame_idx]))
            cb.blockSignals(False)

    def _toggle_clf(self, name: str) -> None:
        """Flip a classifier's checkbox (via keystroke)."""
        cb = self._clf_cbs.get(name)
        if cb is not None:
            cb.setChecked(not cb.isChecked())

    def _on_clf_toggled(self, name: str, checked: bool) -> None:
        """Checkbox changed → update label array for the current frame."""
        idx = self.scrubber.current_frame
        new_val = 1 if checked else 0
        if self._labels[name][idx] != new_val:
            self._labels[name][idx] = new_val
            self._dirty = True
            self.setWindowTitle(f"Label frames — {self.video_name} *")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        """Write the label matrix to targets_inserted/{video_name}.{file_type}.

        Preserves any existing features in features_extracted/ as the
        preceding columns, matching the legacy labeller's output.

        Falls back to raw pandas if ``mufasa.utils.read_write`` is
        unavailable (e.g. environments without h5py for pickle-file
        support). The fallback covers CSV, which is the common case;
        .parquet/.pickle paths raise a clear error instead of silently
        producing the wrong format.
        """
        try:
            import pandas as pd
            target_path = os.path.join(
                self.targets_dir, f"{self.video_name}.{self.file_type}",
            )
            feat_path = os.path.join(
                self.features_dir, f"{self.video_name}.{self.file_type}",
            )
            # Try to load features via the project reader; if that
            # fails for any reason, fall through to an empty frame.
            df = None
            if os.path.isfile(feat_path):
                df = self._read_df_best_effort(feat_path)
            if df is None:
                df = pd.DataFrame(index=range(self.scrubber.total_frames))
            else:
                # Trim/pad to video length so the join doesn't mismatch
                n = self.scrubber.total_frames
                if len(df) != n:
                    df = df.iloc[:n].reset_index(drop=True)
                    while len(df) < n:
                        df.loc[len(df)] = 0
            for name in self._classifier_names:
                df[name] = self._labels[name]
            self._write_df_best_effort(df, target_path)
            self._dirty = False
            self.setWindowTitle(f"Label frames — {self.video_name}")
            self.status.setText(f"Saved → {target_path}")
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not save labels: {exc}",
            )

    def _read_df_best_effort(self, path: str):
        """Read a project data file. For CSV, use pandas directly to
        dodge ``mufasa.utils.read_write.read_df``'s unconditional
        first-column strip. For other file types (parquet/pickle) use
        read_df, which handles those formats correctly."""
        if self.file_type.lower() == "csv":
            import pandas as pd
            return pd.read_csv(path)
        from mufasa.utils.read_write import read_df
        return read_df(path, self.file_type)

    def _write_df_best_effort(self, df, path: str) -> None:
        """Try project's write_df first, fall back to pandas for csv."""
        try:
            from mufasa.utils.read_write import write_df
            write_df(df, self.file_type, path)
            return
        except Exception:
            if self.file_type.lower() == "csv":
                df.to_csv(path, index=False)
                return
            raise

    # ------------------------------------------------------------------ #
    # Close handling
    # ------------------------------------------------------------------ #
    def reject(self) -> None:
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved annotations. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Save:
                self._save()
        self.scrubber.close_video()
        super().reject()


# --------------------------------------------------------------------------- #
# Launcher helper — called from the Annotation page's launcher form
# --------------------------------------------------------------------------- #
def launch_frame_labeller(parent: QWidget,
                          config_path: str,
                          mode: str = "new") -> None:
    """Prompt for a video file, then open the labeller dialog.

    ``mode`` is ``"new"``, ``"continue"``, or ``"pseudo"``.
    """
    if not config_path:
        QMessageBox.warning(
            parent, "No project",
            "Load a project (project_config.ini) before labelling.",
        )
        return
    path, _ = QFileDialog.getOpenFileName(
        parent, "Select video to annotate", "",
        "Videos (*.mp4 *.avi *.mov *.mkv *.webm)",
    )
    if not path:
        return
    try:
        dlg = FrameLabellerDialog(
            config_path=config_path,
            video_path=path,
            mode=mode,
            parent=parent,
        )
    except Exception as exc:
        QMessageBox.critical(parent, "Could not open labeller", str(exc))
        return
    # Keep a reference on the parent so shiboken doesn't collect it
    if not hasattr(parent, "_active_labeller_dialogs"):
        parent._active_labeller_dialogs = []
    parent._active_labeller_dialogs.append(dlg)
    dlg.show()


__all__ = ["FrameLabellerDialog", "launch_frame_labeller"]
