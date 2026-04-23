"""
mufasa.ui_qt.app
================

Main Qt 6.8 application entry point. Replaces
:func:`mufasa.SimBA.main` (the legacy Tk launcher) with a Linux-native
PySide6 main window.

The launcher presents two actions:

1. **Load Project** — pick an existing ``project_config.ini``.
2. **Create Project** — create a fresh one (delegates to the existing
   ``ProjectCreatorPopUp`` once ported; for now, tagged as TODO).

After loading, the main window displays a :class:`QTabWidget` mirroring
the legacy 10-tab layout (Further imports / Video parameters / Outlier
correction / ROI / Extract features / Label behavior / Train / Run /
Visualizations / Add-ons). Each tab is lazy — popup modules are
imported on first click, not at app startup, so cold start stays fast.

CLI:

    python -m mufasa.ui_qt.app                # interactive
    python -m mufasa.ui_qt.app --project PATH # auto-load project
    python -m mufasa.ui_qt.app --batch PATH   # headless batch mode (stub)

Environment:

* Qt platform plugin is auto-selected via
  :func:`mufasa.ui_qt.linux_env.recommended_qpa_platform`.
* ``multiprocessing`` start method is forced to ``fork`` before any
  backend import (see :func:`linux_env.setup_multiprocessing`).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPushButton,
                               QStatusBar, QTabWidget, QVBoxLayout, QWidget)

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.icon_cache import icon


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _read_recent_projects() -> list[str]:
    """Read MRU project list from XDG data dir."""
    f = linux_env.recent_projects_file()
    if not f.is_file():
        return []
    try:
        lines = [
            ln.strip()
            for ln in f.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except OSError:
        return []
    # Drop entries whose file no longer exists.
    return [ln for ln in lines if Path(ln).is_file()]


def _write_recent_project(path: str, keep: int = 10) -> None:
    """Prepend ``path`` to the MRU list, dedup, cap length."""
    current = _read_recent_projects()
    if path in current:
        current.remove(path)
    current.insert(0, path)
    f = linux_env.recent_projects_file()
    f.write_text("\n".join(current[:keep]), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Launcher (shown when no project is loaded yet)
# --------------------------------------------------------------------------- #
class LauncherWindow(QMainWindow):
    """Initial window — load or create a project."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mufasa")
        self.setWindowIcon(icon("SimBA_logo_3_small"))  # logo asset retained
        self.resize(640, 360)

        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        title = QLabel("Mufasa — behavioural analysis", central)
        title.setStyleSheet("font-size: 20pt; font-weight: bold;")
        subtitle = QLabel(
            f"Qt {__import__('PySide6').__version__}  ·  "
            f"CPU {linux_env.cpu_count()}  ·  "
            f"{'CUDA ✓' if linux_env.cuda_available() else 'CUDA ✗'}  ·  "
            f"{'NVENC ✓' if linux_env.nvenc_available() else 'NVENC ✗'}  ·  "
            f"{linux_env.detect_display_server()}",
            central,
        )
        subtitle.setStyleSheet("color: #888;")
        lay.addWidget(title)
        lay.addWidget(subtitle)

        # Action buttons
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton(icon("file"), "  Load project…", central)
        self.load_btn.clicked.connect(self._load_project_dialog)
        self.new_btn = QPushButton(icon("plus"), "  Create project…", central)
        self.new_btn.clicked.connect(self._new_project_stub)
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.new_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # Recent projects
        recents = _read_recent_projects()
        if recents:
            lay.addWidget(QLabel("<b>Recent projects</b>", central))
            for p in recents[:6]:
                b = QPushButton("  " + p, central)
                b.setStyleSheet("text-align: left;")
                b.clicked.connect(lambda _=False, path=p: self._load_project(path))
                lay.addWidget(b)
        lay.addStretch()
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

    def _load_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load project_config.ini",
            str(Path.home()),
            "Mufasa / SimBA project (*.ini)",
        )
        if path:
            self._load_project(path)

    def _load_project(self, path: str) -> None:
        try:
            win = ProjectWindow(project_config_path=path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load", str(exc))
            return
        _write_recent_project(path)
        win.show()
        self.close()

    def _new_project_stub(self) -> None:
        # Placeholder until ProjectCreatorPopUp is ported to Qt.
        QMessageBox.information(
            self,
            "Not yet ported",
            "Project creation has not yet been ported to the Qt UI. "
            "For now, create a project with the legacy Tk UI\n"
            "(mufasa-tk) and then load the resulting project_config.ini here.",
        )


# --------------------------------------------------------------------------- #
# Project window — the main 10-tab surface
# --------------------------------------------------------------------------- #
#
# Each tab is described by a small metadata record: the tab label, its
# icon name, and a list of (button-label, popup-class-path) pairs. The
# popup class is only imported when the user clicks — ``cold start of
# the app is NOT proportional to number of popups``, unlike the Tk
# ``SimBA.py`` which does ~120 imports at the top of the file.
# --------------------------------------------------------------------------- #
TAB_SPEC: list[dict] = [
    {
        "label": "Further imports",
        "icon":  "pose",
        "actions": [
            ("Add classifier", "mufasa.ui_qt.popups.clf_add_remove_print:AddClfPopUp"),
            ("Remove classifier", "mufasa.ui_qt.popups.clf_add_remove_print:RemoveAClassifierPopUp"),
            ("Smooth pose", "mufasa.ui_qt.popups.smoothing:SmoothingPopUp"),
            # TODO: Interpolate, Egocentric Align, Archive, Reverse identities
        ],
    },
    {"label": "Video parameters",   "icon": "calipher", "actions": []},
    {"label": "Outlier correction", "icon": "outlier",  "actions": []},
    {"label": "ROI",                "icon": "roi",      "actions": []},
    {"label": "Extract features",   "icon": "features", "actions": []},
    {"label": "Label behavior",     "icon": "label",    "actions": []},
    {"label": "Train",              "icon": "clf",      "actions": []},
    {"label": "Run",                "icon": "clf_2",    "actions": []},
    {"label": "Visualizations",     "icon": "visualize",
     "actions": [
         ("Heatmap (classifier)", "mufasa.ui_qt.popups.heatmap_clf:HeatmapClfPopUp"),
     ]},
    {"label": "Add-ons",            "icon": "add_on",   "actions": []},
]


class ProjectWindow(QMainWindow):
    """Main window shown after loading a project_config.ini."""

    def __init__(self, project_config_path: str) -> None:
        super().__init__()
        self.project_config_path = project_config_path
        # Read project name from the config without triggering the full
        # ConfigReader (which pulls numba). Keeps launcher fast.
        self.setWindowTitle(f"Mufasa — {Path(project_config_path).parent.name}")
        self.setWindowIcon(icon("SimBA_logo_3_small"))
        self.resize(1280, 800)

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        for spec in TAB_SPEC:
            tab = self._build_tab(spec)
            icn = icon(spec["icon"])
            if icn.isNull():
                self.tabs.addTab(tab, spec["label"])
            else:
                self.tabs.addTab(tab, icn, spec["label"])
        self.setCentralWidget(self.tabs)

        # Menu bar
        self._build_menus()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage(
            f"Loaded {project_config_path}  ·  "
            f"CPU {linux_env.cpu_count()}  ·  "
            f"{'CUDA ✓' if linux_env.cuda_available() else 'CUDA ✗'}"
        )

    def _build_tab(self, spec: dict) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)
        if not spec["actions"]:
            stub = QLabel(
                f"<i>Tab '{spec['label']}' has no actions ported yet.</i>", w
            )
            stub.setStyleSheet("color: #888;")
            lay.addWidget(stub)
            lay.addStretch()
            return w
        for label, dotted in spec["actions"]:
            btn = QPushButton(f"  {label}", w)
            btn.setStyleSheet("text-align: left; padding: 8px;")
            btn.clicked.connect(
                lambda _=False, d=dotted, lbl=label: self._open_popup(d, lbl)
            )
            lay.addWidget(btn)
        lay.addStretch()
        return w

    def _open_popup(self, dotted: str, label: str) -> None:
        """Lazy-import the popup class and instantiate it modelessly."""
        module_path, class_name = dotted.split(":", 1)
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
        except (ImportError, AttributeError) as exc:
            QMessageBox.critical(self, "Popup not available", str(exc))
            return
        try:
            # Two calling conventions: some popups take config_path
            # (project-aware), some take none (PrintModelInfoPopUp).
            if "config_path" in cls.__init__.__code__.co_varnames:
                dlg = cls(config_path=self.project_config_path)
            else:
                dlg = cls()
        except Exception as exc:
            QMessageBox.critical(self, f"Failed to open '{label}'", str(exc))
            return
        dlg.show()
        # Hold a reference so the dialog isn't GC'd immediately.
        self._keep_alive = getattr(self, "_keep_alive", [])
        self._keep_alive.append(dlg)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        a_load = QAction("Load project…", self)
        a_load.setShortcut("Ctrl+O")
        a_load.triggered.connect(self._switch_project)
        m_file.addAction(a_load)
        m_file.addSeparator()
        a_quit = QAction("Quit", self)
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_help = mb.addMenu("&Help")
        a_about = QAction("About Mufasa", self)
        a_about.triggered.connect(self._about)
        m_help.addAction(a_about)

    def _switch_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load project_config.ini", str(Path.home()),
            "Mufasa project (*.ini)",
        )
        if path:
            _write_recent_project(path)
            win = ProjectWindow(project_config_path=path)
            win.show()
            self.close()

    def _about(self) -> None:
        import PySide6
        QMessageBox.about(
            self,
            "About Mufasa",
            f"<b>Mufasa</b><br>"
            f"Behavioural analysis toolkit (fork of SimBA)<br><br>"
            f"PySide6 {PySide6.__version__}  ·  "
            f"Python {sys.version.split()[0]}<br>"
            f"Display: {linux_env.detect_display_server()}<br>"
            f"CPU: {linux_env.cpu_count()} threads<br>"
            f"CUDA: {'available' if linux_env.cuda_available() else 'not detected'}<br>"
            f"NVENC: {'available' if linux_env.nvenc_available() else 'not detected'}",
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mufasa")
    p.add_argument("--project", type=str, default=None,
                   help="auto-load a project_config.ini on launch")
    p.add_argument("--batch", type=str, default=None,
                   help="headless batch mode (stub — not yet implemented)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry — invoked by the ``mufasa`` console script."""
    args = _parse_args(argv)

    # Linux-native defaults, applied before any backend import that might
    # fork workers.
    linux_env.setup_multiprocessing()

    # Let Qt pick its platform plugin per our detector, unless the user
    # has already set one explicitly.
    os.environ.setdefault("QT_QPA_PLATFORM", linux_env.recommended_qpa_platform())

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Mufasa")
    app.setOrganizationName("Mufasa")
    app.setWindowIcon(icon("SimBA_logo_3_small"))

    if args.batch:
        # Reserved for a future non-GUI pipeline runner. For now, just fail
        # gracefully — the scaffolding is here so scripts don't have to
        # change when the batch runner lands.
        print("--batch is not yet implemented", file=sys.stderr)
        return 2

    if args.project:
        if not Path(args.project).is_file():
            print(f"project not found: {args.project}", file=sys.stderr)
            return 1
        win: QMainWindow = ProjectWindow(project_config_path=args.project)
    else:
        win = LauncherWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
