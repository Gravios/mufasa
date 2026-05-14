"""
mufasa.ui_qt.forms.project_setup
================================

Inline forms for project-level actions.

Patch 122al: :class:`BatchPreProcessLauncher` is now an alias
for :class:`BatchPreProcessForm` (the Qt port of the legacy Tk
wizard, ported in 122al). The launcher placeholder is kept as
an import alias so any external code that referenced
``BatchPreProcessLauncher`` keeps working; new code should
import :class:`BatchPreProcessForm` directly.

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

from mufasa.ui_qt.forms.batch_pre_process import BatchPreProcessForm


# --------------------------------------------------------------------------- #
# BatchPreProcessLauncher — alias for back-compat (patch 122al)
# --------------------------------------------------------------------------- #
# The Tk launcher pattern (a "Launch…" button opening the legacy
# wizard in a separate Tk window) is gone. BatchPreProcessForm
# is a real Qt OperationForm rendered inline on the Preprocessing
# page, and supports pop-out into a floating dockable window.
# The alias here preserves the old import name for any external
# consumer that might still reference it.
BatchPreProcessLauncher = BatchPreProcessForm


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
    "BatchPreProcessForm",
    "BatchPreProcessLauncher",  # legacy alias
    "register_project_menu_actions",
]
