"""
mufasa.ui_qt.forms.project_setup
================================

Inline forms for project-level actions.

* :class:`ArchiveFilesForm` — wraps :func:`archive_processed_files`.
  Replaces :class:`ArchiveFilesPopUp`.
* :class:`BatchPreProcessLauncher` — launcher for
  :class:`BatchProcessFrame`. Replaces :class:`BatchPreProcessPopUp`.

The About dialog isn't a form at all — it becomes a Help-menu action
(registered by :func:`register_project_menu_actions`).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFormLayout, QLineEdit, QMessageBox)

from mufasa.ui_qt.forms.annotation import _LauncherForm
from mufasa.ui_qt.workbench import OperationForm


# --------------------------------------------------------------------------- #
# ArchiveFilesForm
# --------------------------------------------------------------------------- #
class ArchiveFilesForm(OperationForm):
    """Archive processed project files under a named folder.

    Wraps :func:`mufasa.utils.read_write.archive_processed_files`. One
    field; one button. Kept as a form (not a menu action) because the
    archive name is user-specified and benefits from inline
    validation/feedback.
    """

    title = "Archive processed files"
    description = ("Move processed project files (features_extracted, "
                   "targets_inserted, machine_results, logs) into a "
                   "named archive folder to clean up the working "
                   "directory before a new experiment.")

    def build(self) -> None:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.archive_name = QLineEdit(self)
        self.archive_name.setPlaceholderText(
            "Archive folder name, e.g. 'experiment_1_2025'"
        )
        form.addRow("Archive name:", self.archive_name)
        self.body_layout.addLayout(form)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        name = self.archive_name.text().strip()
        if not name:
            raise ValueError("Archive name is required.")
        # Sanity-check the name — no slashes, no whitespace-only, no
        # filesystem-nasty characters. Matches the legacy popup's
        # validation.
        if any(c in name for c in "/\\:<>|*?\""):
            raise ValueError(
                f"Archive name '{name}' contains filesystem-reserved "
                f"characters. Use letters, numbers, '_' or '-' only."
            )
        return {"config_path": self.config_path, "archive_name": name}

    def target(self, *, config_path: str, archive_name: str) -> None:
        from mufasa.utils.read_write import archive_processed_files
        archive_processed_files(config_path=config_path,
                                archive_name=archive_name)


# --------------------------------------------------------------------------- #
# BatchPreProcessLauncher — Tk wizard, kept as launcher
# --------------------------------------------------------------------------- #
class BatchPreProcessLauncher(_LauncherForm):
    """Launcher for :class:`BatchProcessFrame`. Wraps a legacy multi-step
    video pre-processing wizard (crop → downsample → greyscale → etc.
    across multiple videos), which is implemented as a Tk Frame with
    custom row widgets. Porting that to Qt is a meaningful amount of
    work (several QTableWidget columns of per-video toggles + preview);
    scheduled as a separate item.
    """
    title = "Batch pre-process videos"
    description = ("Multi-step video pre-processing wizard for whole "
                   "directories (crop → downsample → greyscale → "
                   "flip/rotate → clip).")
    launch_button_text = "Open batch pre-process wizard (legacy UI)…"
    launch_title = "Batch pre-process"
    launch_message = (
        "Batch pre-processing is a multi-column wizard (one row per "
        "video, one column per transform) that hasn't been ported to "
        "Qt yet. Use <code>mufasa-tk</code> for this step, or drive "
        "the video-processing forms on the <i>Video Processing</i> "
        "page one operation at a time."
    )


# --------------------------------------------------------------------------- #
# Menu actions
# --------------------------------------------------------------------------- #
def register_project_menu_actions(workbench) -> None:
    """Register Help-menu entries for project-level actions.

    :class:`MufasaWorkbench` already installs an "About Mufasa" action
    at construction time, so this function is a placeholder kept for
    API symmetry with the other ``register_*_menu_actions`` helpers
    (video-processing registers reverse/crossfade/etc.). Adding further
    project-level Help entries would happen here.
    """
    # Safely poke the workbench's help menu to confirm it exists; any
    # future additions go here.
    _ = getattr(workbench, "_help_menu", None)
    return


__all__ = [
    "ArchiveFilesForm",
    "BatchPreProcessLauncher",
    "register_project_menu_actions",
]
