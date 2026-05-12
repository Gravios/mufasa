"""
mufasa.ui_qt.forms.project_setup
================================

Inline forms for project-level actions.

* :class:`BatchPreProcessLauncher` — launcher for
  :class:`BatchProcessFrame`. Replaces :class:`BatchPreProcessPopUp`.

Patch 122m: removed :class:`ArchiveFilesForm` (and its backing
``archive_processed_files`` helper). The legacy "shuffle outputs
into named subfolders" model assumes the SimBA INI layout's
``csv/<stage>/`` tree which doesn't exist in v1; the function
would have crashed on v1 projects on a missing ``csv/`` dir,
and the use-case it solved (clearing shared stage dirs between
experiments) is now subsumed by v1's per-run
``derived/<stage>/<run_id>/`` provenance.

The About dialog isn't a form at all — it becomes a Help-menu action
(registered by :func:`register_project_menu_actions`).
"""
from __future__ import annotations

from mufasa.ui_qt.forms.annotation import _LauncherForm


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
    "BatchPreProcessLauncher",
    "register_project_menu_actions",
]
