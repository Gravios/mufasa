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
    helpers from :mod:`mufasa.ui_qt.pages`. As more pages are ported,
    add them here.
    """
    wb = MufasaWorkbench(project_config_path=project_config_path)

    # --- Pages, in sidebar order -------------------------------------- #
    from mufasa.ui_qt.pages.project_setup_page import build_project_setup_page
    build_project_setup_page(wb, config_path=project_config_path)

    build_video_processing_page(wb, config_path=project_config_path)

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
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Workbench entry — invoked by the ``mufasa-workbench`` console script."""
    args = _parse_args(argv)
    linux_env.setup_multiprocessing()
    os.environ.setdefault("QT_QPA_PLATFORM", linux_env.recommended_qpa_platform())

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Mufasa")
    app.setOrganizationName("Mufasa")

    if args.project and not Path(args.project).is_file():
        print(f"project not found: {args.project}", file=sys.stderr)
        return 1

    wb = build_workbench(project_config_path=args.project)
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
