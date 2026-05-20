"""
mufasa.ui_qt.frame_labeller
===========================

Qt-native replacement for the legacy Tk :class:`LabellingInterface`.
The labeling UI lives in a reusable :class:`FrameLabellerWidget`
that can be hosted three ways:

* **Dock** (preferred) — :func:`open_frame_labeller_dock` wraps the
  widget in a :class:`QDockWidget` attached to the workbench's main
  window. Floats by default; the user can drag-dock it into any of
  the four dock areas, then drag it back out.
* **Dialog** — :class:`FrameLabellerDialog` wraps the widget as a
  modeless QDialog. Preserved for callers that explicitly want a
  separate top-level window with a close-button bar.
* **Anywhere** — the widget is just a QWidget; embed it in any
  layout.

The legacy entry point :func:`launch_frame_labeller` was used by
the Annotation page's launcher form. It now prompts for a video
then opens the dock when a QMainWindow ancestor is available,
falling back to the dialog otherwise.

Patch 122aj — what changed vs the pre-refactor module
-----------------------------------------------------

* Splits the monolithic ``FrameLabellerDialog`` into a content
  widget + thin dialog wrapper so the same UI can live in either
  a dialog or a dock without duplication.
* Adds :func:`open_frame_labeller_dock` and a private
  ``_find_main_window`` walk that locates the workbench so the
  dock is attached correctly.
* Continue-mode label loading now routes through
  :func:`mufasa.utils.label_io.load_labels_for_video`. That helper
  was added in 122ae-3.5 specifically to return ``just the
  behavior label collection`` (Int64-nullable per-classifier
  columns), reading first from ``derived/labels/<video>.parquet``
  (the 122ae-5c sidecar) and falling back to legacy
  ``csv/targets_inserted/`` automatically.
* Removes the pre-existing copy-paste bug in
  ``_load_existing_labels`` where the per-classifier loop ran
  twice (same result both times, just wasted CPU).
* Save-side dual-write — the legacy
  ``targets_inserted/<video>.<ext>`` write stays (primary
  contract; classifier training reads from there), plus a new
  sidecar via :func:`save_labels_for_video` to keep parity with
  the 122ae-5c label-writes story.
* In dock mode, the redundant Close button is hidden — the dock
  has its own close X in its title bar, so the dialog-style
  button bar is just UX noise.
* Pseudo-mode kept on the legacy path (reads ``machine_results``,
  which doesn't have a v1 location yet — separate scope).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from mufasa.ui_qt.forms.analysis import _load_classifier_names
from mufasa.ui_qt.frame_scrubber import FrameScrubberWidget


# --------------------------------------------------------------------------- #
# Reusable content widget — the labeling UI itself
# --------------------------------------------------------------------------- #
class FrameLabellerWidget(QWidget):
    """The labeling UI — scrubber + per-classifier toggles + save.

    Embeddable in any container (dock, dialog, plain layout). Owns
    the in-memory label matrix and the unsaved-changes flag;
    callers should consult :meth:`is_dirty` and
    :meth:`confirm_discard_changes` before destroying it.
    """

    def __init__(self,
                 config_path: str,
                 video_path: str,
                 *,
                 mode: str = "new",
                 parent: QWidget | None = None) -> None:
        """
        Parameters
        ----------
        config_path : str
            Path to the project config (``project.toml`` for v1 or
            ``project_config.ini`` for legacy).
        video_path : str
            Absolute path to the video being annotated.
        mode : {"new", "continue", "pseudo"}
            * "new" — start from all-zero labels.
            * "continue" — load existing labels via
              :func:`load_labels_for_video` (the behaviour label
              collection only, not the features).
            * "pseudo" — load initial labels from
              ``machine_results/<video>.<ext>``.
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
        # Set by host containers (dialog/dock) — when True, hide UI
        # elements that the host renders itself (close button,
        # window title management).
        self._in_dock: bool = False

        self._load_project_metadata()
        self._build_ui()
        self._setup_shortcuts()
        self.scrubber.load(video_path)
        self._initialize_labels()
        self.scrubber.frame_changed.connect(self._on_frame_changed)
        self._on_frame_changed(0)

    # ------------------------------------------------------------------ #
    # Public API for host containers
    # ------------------------------------------------------------------ #
    def is_dirty(self) -> bool:
        return self._dirty

    def confirm_discard_changes(self) -> bool:
        """Prompt the user about unsaved annotations. Returns True
        if the host should proceed with closing (saved or
        discarded), False to keep the widget open. Safe to call
        from a host's closeEvent / reject."""
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self, "Unsaved changes",
            "You have unsaved annotations. Save before closing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self._save()
        return True

    def cleanup(self) -> None:
        """Release video resources. Hosts should call this on close
        so cv2 capture handles don't linger."""
        self.scrubber.close_video()

    def set_in_dock(self, value: bool) -> None:
        """Hosts call this to tell the widget it's running inside a
        QDockWidget. The widget hides redundant UX (close button)
        accordingly."""
        self._in_dock = value
        # The dialog's button-bar Close becomes redundant when the
        # dock provides its own close X.
        if hasattr(self, "_close_btn") and self._close_btn is not None:
            self._close_btn.setVisible(not value)

    # ------------------------------------------------------------------ #
    # Project metadata
    # ------------------------------------------------------------------ #
    def _load_project_metadata(self) -> None:
        """Discover classifier list + project paths from config."""
        self._classifier_names = _load_classifier_names(self.config_path)
        if not self._classifier_names:
            raise RuntimeError(
                "No classifiers defined in the project. Add at "
                "least one via the Classifier → Manage page before "
                "labelling."
            )
        from mufasa.project_layout import project_metadata_from_config, project_paths_from_config
        paths = project_paths_from_config(self.config_path)
        meta = project_metadata_from_config(self.config_path)
        self.file_type = meta.get("file_type", "csv")
        # Patch 122ao: features_extracted_dir and targets_inserted_dir
        # keys were dropped from project_paths_from_config. After
        # 122ak the labeller doesn't write to those locations
        # anyway — _save uses save_labels_for_video which writes
        # to derived/labels/<video>.parquet via the layout
        # helper's derived_labels_dir.
        # Patch 122ax: machine_results_dir is legacy-only post-122ax.
        # v1 projects don't define this key; consumers must handle
        # None. The frame labeller uses it only to build the
        # legacy_fallback path for pseudo-label seeding —
        # _load_pseudo_labels handles the None case.
        self.machine_results_dir = paths.get("machine_results_dir")
        # derived/labels/ is auto-created by save_labels_for_video
        # on first write.

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # Scrubber takes the lion's share of vertical space
        self.scrubber = FrameScrubberWidget(self)
        outer.addWidget(self.scrubber, 1)

        # Per-classifier checkbox bar
        clf_bar = QHBoxLayout()
        clf_bar.addWidget(QLabel("<b>Behaviours at this frame:</b>", self))
        for i, name in enumerate(self._classifier_names[:9]):
            cb = QCheckBox(f"{i + 1}. {name}", self)
            cb.toggled.connect(lambda checked, n=name:
                               self._on_clf_toggled(n, checked))
            self._clf_cbs[name] = cb
            clf_bar.addWidget(cb)
        # Overflow (10+ classifiers) — no keystroke binding
        for name in self._classifier_names[9:]:
            cb = QCheckBox(name, self)
            cb.toggled.connect(lambda checked, n=name:
                               self._on_clf_toggled(n, checked))
            self._clf_cbs[name] = cb
            clf_bar.addWidget(cb)
        clf_bar.addStretch()
        outer.addLayout(clf_bar)

        # Keystroke hint + status line
        hint = QLabel(
            "<i>Keys: 1–9 toggle behaviours · ← / → jog 1 frame · "
            "Shift+← / Shift+→ jog 10 · Space = next frame · "
            "Ctrl+S save</i>",
            self,
        )
        hint.setStyleSheet("color: palette(placeholder-text);")
        outer.addWidget(hint)

        self.status = QLabel("Ready.", self)
        outer.addWidget(self.status)

        # Save button (always visible) + Close button (hidden in
        # dock mode by set_in_dock)
        btns = QDialogButtonBox(self)
        self.save_btn = btns.addButton("Save", QDialogButtonBox.ApplyRole)
        self.save_btn.clicked.connect(self._save)
        self._close_btn = btns.addButton(
            "Close", QDialogButtonBox.RejectRole,
        )
        self._close_btn.clicked.connect(self._request_close)
        outer.addWidget(btns)

    def _setup_shortcuts(self) -> None:
        for i, name in enumerate(self._classifier_names[:9]):
            sc = QShortcut(QKeySequence(str(i + 1)), self)
            sc.activated.connect(lambda n=name: self._toggle_clf(n))
        for key, delta in [(Qt.Key_Left, -1), (Qt.Key_Right, 1)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(
                lambda d=delta: self.scrubber.seek(
                    self.scrubber.current_frame + d
                )
            )
        for combo, delta in [
            (QKeySequence(Qt.ShiftModifier | Qt.Key_Left), -10),
            (QKeySequence(Qt.ShiftModifier | Qt.Key_Right), 10),
        ]:
            sc = QShortcut(combo, self)
            sc.activated.connect(
                lambda d=delta: self.scrubber.seek(
                    self.scrubber.current_frame + d
                )
            )
        sp = QShortcut(QKeySequence(Qt.Key_Space), self)
        sp.activated.connect(lambda: self.scrubber.seek(
            self.scrubber.current_frame + 1))
        ss = QShortcut(QKeySequence.Save, self)
        ss.activated.connect(self._save)

    # ------------------------------------------------------------------ #
    # Label-state management
    # ------------------------------------------------------------------ #
    def _initialize_labels(self) -> None:
        n = self.scrubber.total_frames
        self._labels = {name: np.zeros(n, dtype=np.uint8)
                        for name in self._classifier_names}
        if self.mode == "continue":
            # Patch 122aj: continue-mode just loads the behaviour
            # label collection — labels only, not features.
            self._load_continue_labels()
        elif self.mode == "pseudo":
            # Pseudo-mode reads machine_results which doesn't have
            # a v1 location yet — keep the legacy CSV reader.
            self._load_pseudo_labels()

    def _load_continue_labels(self) -> None:
        """Continue-mode loader. Routes through
        :func:`mufasa.utils.label_io.load_labels_for_video` so labels
        in either derived/labels/ (v1) or csv/targets_inserted/
        (legacy) both resolve."""
        try:
            from mufasa.utils.label_io import load_labels_for_video
            df = load_labels_for_video(self.video_name, self.config_path)
        except FileNotFoundError:
            self.status.setText(
                "No existing labels found; starting from zeros."
            )
            return
        except Exception as exc:
            self.status.setText(f"Could not load labels: {exc}")
            return
        n = self.scrubber.total_frames
        loaded = []
        for name in self._classifier_names:
            if name not in df.columns:
                continue
            # The helper returns Int64 nullable; fill any NA with 0
            # before casting to uint8 for the in-memory label array.
            col = df[name].fillna(0).astype(np.uint8).to_numpy()
            if col.shape[0] != n:
                col = (col[:n] if col.shape[0] > n
                       else np.pad(col, (0, n - col.shape[0]),
                                   constant_values=0))
            self._labels[name] = col
            loaded.append(name)
        self.status.setText(
            f"Loaded {len(loaded)} classifier column(s) from the "
            f"project's label store."
        )

    def _load_pseudo_labels(self) -> None:
        """Pseudo-labelling: seed labels from machine_results
        predictions.

        Patch 122aw: dual-read via classification_io helper.
        Tries v1 (derived/classifications/<video>.parquet) first,
        falls back to the legacy machine_results CSV. The
        docstring's old caveat 'machine_results doesn't have a v1
        derived/ location yet' is no longer true post-122at.
        """
        src = (
            os.path.join(
                self.machine_results_dir,
                f"{self.video_name}.{self.file_type}",
            )
            if self.machine_results_dir is not None else None
        )
        try:
            from mufasa.utils.classification_io import (
                load_machine_results_for_video,
            )
            df = load_machine_results_for_video(
                video_name=self.video_name,
                config_path=self.config_path,
                legacy_fallback=(
                    src
                    if src is not None and os.path.isfile(src)
                    else None
                ),
            )
        except FileNotFoundError:
            self.status.setText(
                "Pseudo source not found; starting from zeros."
            )
            return
        except Exception as exc:
            self.status.setText(f"Could not read {src}: {exc}")
            return
        n = self.scrubber.total_frames
        loaded = []
        for name in self._classifier_names:
            if name not in df.columns:
                continue
            col = df[name].to_numpy()
            if col.shape[0] != n:
                col = (col[:n] if col.shape[0] > n
                       else np.pad(col, (0, n - col.shape[0]),
                                   constant_values=0))
            self._labels[name] = col.astype(np.uint8)
            loaded.append(name)
        self.status.setText(
            f"Loaded {len(loaded)} pseudo column(s) from "
            f"{os.path.basename(src)}."
        )

    def _on_frame_changed(self, frame_idx: int) -> None:
        for name, cb in self._clf_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(bool(self._labels[name][frame_idx]))
            cb.blockSignals(False)

    def _toggle_clf(self, name: str) -> None:
        cb = self._clf_cbs.get(name)
        if cb is not None:
            cb.setChecked(not cb.isChecked())

    def _on_clf_toggled(self, name: str, checked: bool) -> None:
        idx = self.scrubber.current_frame
        new_val = 1 if checked else 0
        if self._labels[name][idx] != new_val:
            self._labels[name][idx] = new_val
            self._dirty = True

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        """Write the label matrix to derived/labels/<video>.parquet
        via save_labels_for_video. Labels-only — features stay in
        derived/features/ where the bulk extractor put them, and
        classifier training reads them via load_features_for_video
        when it needs them. No combined features+labels file is
        written; the legacy csv/targets_inserted/ layout is gone.
        """
        try:
            import pandas as pd

            from mufasa.utils.label_io import save_labels_for_video
            labels_df = pd.DataFrame({
                name: self._labels[name].astype(np.int64)
                for name in self._classifier_names
            })
            out_path = save_labels_for_video(
                video_name=self.video_name,
                config_path=self.config_path,
                labels=labels_df,
            )
            self._dirty = False
            self.status.setText(f"Saved → {out_path}")
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not save labels: {exc}",
            )

    def _request_close(self) -> None:
        """Close button clicked. Walk up to the host container and
        close it via its normal close path so the unsaved-changes
        check runs once."""
        host = self.parent()
        while host is not None:
            if isinstance(host, (QDialog, QDockWidget)):
                host.close()
                return
            host = host.parent() if hasattr(host, "parent") else None
        # No container — close ourselves
        if self.confirm_discard_changes():
            self.cleanup()
            self.close()


# --------------------------------------------------------------------------- #
# Dialog wrapper — preserves the existing API
# --------------------------------------------------------------------------- #
class FrameLabellerDialog(QDialog):
    """Modeless dialog hosting :class:`FrameLabellerWidget`.

    Kept for back-compat with callers that explicitly want a
    separate top-level window. The dock helper is the preferred
    surface for the workbench-integrated flow.
    """

    def __init__(self,
                 config_path: str,
                 video_path: str,
                 *,
                 mode: str = "new",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.widget = FrameLabellerWidget(
            config_path=config_path,
            video_path=video_path,
            mode=mode,
            parent=self,
        )
        self.video_name = self.widget.video_name
        self.setWindowTitle(f"Label frames — {self.video_name}")
        self.resize(1100, 720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.widget)

    def reject(self) -> None:
        if not self.widget.confirm_discard_changes():
            return
        self.widget.cleanup()
        super().reject()


# --------------------------------------------------------------------------- #
# Dock helper — preferred surface
# --------------------------------------------------------------------------- #
def _find_main_window(start: QWidget | None) -> QMainWindow | None:
    """Walk up the parent chain looking for a QMainWindow. The
    workbench is one; arbitrary dialogs and forms are not.
    Returns None if no main window found — caller should fall
    back to dialog hosting."""
    node = start
    while node is not None:
        if isinstance(node, QMainWindow):
            return node
        node = node.parent() if hasattr(node, "parent") else None
    return None


def open_frame_labeller_dock(parent: QWidget,
                             config_path: str,
                             video_path: str,
                             *,
                             mode: str = "new",
                             ) -> QWidget | None:
    """Open the labeler in a :class:`QDockWidget` attached to the
    workbench's main window. Floats by default; the user can
    drag-dock it into the workbench layout or float it back out.

    Falls back to :class:`FrameLabellerDialog` if no QMainWindow
    ancestor exists (e.g., when called from a standalone test
    harness).

    Returns the dock (or dialog, in fallback) so the caller can
    keep a reference, focus it, or close it programmatically.
    """
    main = _find_main_window(parent)
    if main is None:
        dlg = FrameLabellerDialog(
            config_path=config_path,
            video_path=video_path,
            mode=mode,
            parent=parent,
        )
        dlg.show()
        return dlg

    widget = FrameLabellerWidget(
        config_path=config_path,
        video_path=video_path,
        mode=mode,
    )
    widget.set_in_dock(True)

    dock = QDockWidget(
        f"Label frames — {widget.video_name}", main,
    )
    dock.setWidget(widget)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setFeatures(
        QDockWidget.DockWidgetMovable
        | QDockWidget.DockWidgetFloatable
        | QDockWidget.DockWidgetClosable
    )
    dock.setFloating(True)
    dock.resize(1100, 720)
    main.addDockWidget(Qt.RightDockWidgetArea, dock)

    # Intercept the dock's close so unsaved-changes get prompted.
    original_close_event = dock.closeEvent

    def _on_close(event):  # type: ignore[no-untyped-def]
        if not widget.confirm_discard_changes():
            event.ignore()
            return
        widget.cleanup()
        original_close_event(event)

    dock.closeEvent = _on_close  # type: ignore[assignment]

    # Keep a reference on the main window so Python doesn't
    # garbage-collect the dock the moment open_frame_labeller_dock
    # returns. addDockWidget transfers ownership C++-side, but the
    # Python wrapper still needs a live reference.
    if not hasattr(main, "_active_labeller_docks"):
        main._active_labeller_docks = []
    main._active_labeller_docks.append(dock)

    dock.show()
    dock.raise_()
    return dock


# --------------------------------------------------------------------------- #
# Launcher entry point — prompts for video, then opens the dock
# --------------------------------------------------------------------------- #
def launch_frame_labeller(parent: QWidget,
                          config_path: str,
                          mode: str = "new") -> None:
    """Prompt for a video file, then open the labeler.

    Patch 122aj: opens via :func:`open_frame_labeller_dock` so the
    labeler lives inside the workbench when one is available, with
    a dialog fallback for headless / detached parents.

    ``mode`` is ``"new"``, ``"continue"``, or ``"pseudo"``.
    """
    if not config_path:
        QMessageBox.warning(
            parent, "No project",
            "Load a project (project.toml or project_config.ini) "
            "before labelling.",
        )
        return
    path, _ = QFileDialog.getOpenFileName(
        parent, "Select video to annotate", "",
        "Videos (*.mp4 *.avi *.mov *.mkv *.webm)",
    )
    if not path:
        return
    try:
        open_frame_labeller_dock(
            parent=parent,
            config_path=config_path,
            video_path=path,
            mode=mode,
        )
    except Exception as exc:
        QMessageBox.critical(
            parent, "Could not open labeller", str(exc),
        )


__all__ = [
    "FrameLabellerDialog",
    "FrameLabellerWidget",
    "launch_frame_labeller",
    "open_frame_labeller_dock",
]
