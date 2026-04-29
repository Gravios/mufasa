"""
mufasa.ui_qt.workbench_app
==========================

Alternative entry point that launches the consolidated, sidebar-
navigated :class:`MufasaWorkbench` in place of the legacy tabs-and-
popups main window (:mod:`mufasa.ui_qt.app`).

The two entry points coexist for the duration of the migration: the
legacy ``mufasa`` command (see ``[project.scripts]`` in
``pyproject.toml``) still launches :mod:`app`; a parallel
``mufasa-workbench`` command launches this module. Once all the pages
are ported, ``mufasa`` will be repointed here and the legacy tab UI
retired.

CLI:

    python -m mufasa.ui_qt.workbench_app
    python -m mufasa.ui_qt.workbench_app --project /path/to/project_config.ini
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.workbench import MufasaWorkbench
from mufasa.ui_qt.pages.video_processing_page import build_video_processing_page


def build_workbench(project_config_path: Optional[str] = None
                    ) -> MufasaWorkbench:
    """Assemble a workbench with all currently-ported pages.

    Pages are added via ``build_<stage>_page(workbench, config_path)``
    helpers from :mod:`mufasa.ui_qt.pages`. Order below follows the
    canonical pipeline top-to-bottom: setup → data in → clean → extract
    → label → classify → analyze → visualize. Video Processing is a
    utility used opportunistically (re-encoding, trimming), so it sits
    in the utility-adjacent zone at the bottom alongside Add-ons and
    Tools rather than in the pipeline's step-2 slot. As more pages are
    ported, add them in the position the pipeline demands.
    """
    wb = MufasaWorkbench(project_config_path=project_config_path)

    # --- Pages, in sidebar order (pipeline progression) --------------- #
    from mufasa.ui_qt.pages.project_setup_page import build_project_setup_page
    build_project_setup_page(wb, config_path=project_config_path)

    from mufasa.ui_qt.pages.data_import_page import build_data_import_page
    build_data_import_page(wb, config_path=project_config_path)

    from mufasa.ui_qt.pages.pose_cleanup_page import build_pose_cleanup_page
    build_pose_cleanup_page(wb, config_path=project_config_path)

    from mufasa.ui_qt.pages.roi_page import build_roi_page
    build_roi_page(wb, config_path=project_config_path)

    from mufasa.ui_qt.pages.features_page import build_features_page
    build_features_page(wb, config_path=project_config_path)

    from mufasa.ui_qt.pages.annotation_page import build_annotation_page
    build_annotation_page(wb, config_path=project_config_path)

    from mufasa.ui_qt.pages.classifier_page import build_classifier_page
    build_classifier_page(wb, config_path=project_config_path)

    from mufasa.ui_qt.pages.analysis_page import build_analysis_page
    build_analysis_page(wb, config_path=project_config_path)

    from mufasa.ui_qt.pages.visualizations_page import build_visualizations_page
    build_visualizations_page(wb, config_path=project_config_path)

    # --- Utility-adjacent tail --------------------------------------- #
    # Video Processing: re-encode / trim / crop / rotate. Not part of
    # the pipeline, used ad-hoc when a video needs munging before or
    # after import.
    build_video_processing_page(wb, config_path=project_config_path)

    from mufasa.ui_qt.pages.addons_page import build_addons_page
    build_addons_page(wb, config_path=project_config_path)

    # Tools: project-independent utilities (converters, etc.). Bottom
    # of the sidebar because nothing above depends on it, and it's
    # reached occasionally rather than every session.
    from mufasa.ui_qt.pages.tools_page import build_tools_page
    build_tools_page(wb, config_path=project_config_path)

    return wb


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mufasa-workbench")
    p.add_argument("--project", type=str, default=None,
                   help="auto-load a project_config.ini on launch")
    p.add_argument("--no-auto-discover", action="store_true",
                   help="disable auto-discovery of project_config.ini "
                        "in the current directory or its ancestors")
    return p.parse_args(argv)


# Number of parent directories to walk when auto-discovering a
# project config. 3 levels covers the common cases: user cd'd into
# project_folder/, project_folder/csv/, or project_folder/csv/<stage>/.
# Going higher risks picking up unrelated projects from sibling trees.
_AUTO_DISCOVER_MAX_DEPTH = 3
_PROJECT_CONFIG_FILENAME = "project_config.ini"
# Mufasa creates projects with a hardcoded "project_folder" subdirectory
# (DirNames.PROJECT.value). The config file lives inside there, NOT at
# the project root. So when walking up from CWD, we also probe each
# level's "project_folder/" subdirectory — that catches the common case
# where the user is one level above project_folder/ (e.g. cd'd into the
# named project directory itself, not its project_folder subdir).
_PROJECT_FOLDER_NAME = "project_folder"


def _auto_discover_project(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for a ``project_config.ini``.

    At each level checks two locations:
      1. ``<level>/project_config.ini`` — user is inside project_folder/
         or one of its descendants.
      2. ``<level>/project_folder/project_config.ini`` — user is in
         the parent directory (the named project root) and hasn't
         entered project_folder/ yet.

    Returns the closest match (deepest path) or None if nothing found
    within the depth cap. The depth cap exists because walking all the
    way to ``/`` could pick up unrelated projects in sibling trees —
    typical Mufasa users cd into the project_folder, the project root,
    or one of project_folder's immediate subdirectories, so 3 levels
    covers everything.
    """
    current = start.resolve()
    for _ in range(_AUTO_DISCOVER_MAX_DEPTH + 1):
        # Direct hit: config in the current level
        candidate = current / _PROJECT_CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        # Sideways hit: user is one above project_folder/
        sibling = current / _PROJECT_FOLDER_NAME / _PROJECT_CONFIG_FILENAME
        if sibling.is_file():
            return sibling
        if current.parent == current:
            # Reached filesystem root.
            break
        current = current.parent
    return None


def main(argv: Optional[list[str]] = None) -> int:
    """Workbench entry — invoked by the ``mufasa-workbench`` console script."""
    args = _parse_args(argv)
    linux_env.setup_multiprocessing()
    os.environ.setdefault("QT_QPA_PLATFORM", linux_env.recommended_qpa_platform())

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Mufasa")
    app.setOrganizationName("Mufasa")

    project_config_path: Optional[str] = None
    if args.project:
        # User specified an explicit path — respect it. Don't auto-search.
        if not Path(args.project).is_file():
            print(f"project not found: {args.project}", file=sys.stderr)
            return 1
        project_config_path = args.project
    elif not args.no_auto_discover:
        # Walk up from CWD looking for a project_config.ini.
        # Tell the user what we found so the auto-load is never
        # silent (avoids surprise "wait, why did it open *that*
        # project?" confusion when working across multiple
        # projects on one machine).
        discovered = _auto_discover_project(Path.cwd())
        if discovered is not None:
            print(
                f"Auto-loading project config: {discovered}",
                file=sys.stderr,
            )
            project_config_path = str(discovered)

    wb = build_workbench(project_config_path=project_config_path)
    # Track the workbench on the QApplication so File → New/Open project
    # (which builds a fresh window and closes the old one) doesn't leak
    # or GC the replacement before it's shown.
    refs = app.property("_mufasa_workbenches") or []
    refs.append(wb)
    app.setProperty("_mufasa_workbenches", refs)
    wb.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
