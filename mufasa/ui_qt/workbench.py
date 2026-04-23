"""
mufasa.ui_qt.workbench
======================

A sidebar-navigated main window that replaces the 152-popup-windows
pattern with a small number of workflow pages, each hosting sub-sections
with inline forms. Popup classes that existed purely to collect 3-5
inputs and click "run" become inline forms here; no new ``Toplevel`` /
``QDialog`` is spawned.

Architecture
------------

::

    MufasaWorkbench (QMainWindow)
    ├─ MenuBar            — one-shot actions (File/Edit/Tools/Help)
    ├─ Left sidebar       — QListWidget of workflow stages
    ├─ Central stack      — QStackedWidget; one WorkflowPage per sidebar entry
    │   └─ WorkflowPage   — QToolBox (accordion) with sub-sections
    │       └─ OperationForm(s)  — inline forms (no window)
    └─ Status bar         — capability line + current project

Design notes
------------

* :class:`OperationForm` is a ``QWidget`` that exposes ``title``,
  ``help_url`` and two slots: ``validate()`` (raise on bad input) and
  ``run()`` (kick off work). Subclasses add their own fields. A single
  "Run" button and progress routing are handled by the base class.

* :class:`WorkflowPage` hosts an ordered list of ``OperationForm`` by
  section name; sections are lazy-instantiated on first expand to keep
  tab switching cheap.

* Popups that legitimately need their own window (interactive
  previews, wizards, annotation clip players) are still
  :class:`~mufasa.ui_qt.dialog.MufasaDialog` — the workbench has a
  ``launch_dialog(cls)`` helper for that case. They're the minority.
"""
from __future__ import annotations

from typing import Callable, Optional, Type

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox,
                               QPushButton, QScrollArea, QSplitter,
                               QStackedWidget, QStatusBar, QToolBox,
                               QVBoxLayout, QWidget)

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.icon_cache import icon
from mufasa.ui_qt.progress import run_with_progress


# --------------------------------------------------------------------------- #
# OperationForm — the inline replacement for "click button → new window"
# --------------------------------------------------------------------------- #
class OperationForm(QWidget):
    """Base class for inline operation forms.

    Subclasses build the form body in :meth:`build` and implement
    :meth:`collect_args` (collect + validate inputs) and
    :meth:`target` (the actual work function). The base class wires the
    Run button and routes execution through
    :func:`mufasa.ui_qt.progress.run_with_progress`, so cancellation
    and progress reporting come for free.

    Subclass responsibilities:
    * set ``title`` attribute (section header)
    * implement :meth:`build` to populate ``self.body_layout``
    * implement :meth:`collect_args` to return a dict (raise on error)
    * implement :meth:`target` (or override :meth:`on_run`)

    The form never opens a new window by itself; it lives inline in a
    :class:`WorkflowPage`.
    """

    title: str = ""
    description: str = ""
    help_url: Optional[str] = None

    completed = Signal()

    def __init__(self, parent: Optional[QWidget] = None,
                 config_path: Optional[str] = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self._build_shell()
        self.build()

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def build(self) -> None:
        """Populate ``self.body_layout`` with form fields."""
        raise NotImplementedError

    def collect_args(self) -> dict:
        """Return validated kwargs for :meth:`target`. Raise on error."""
        return {}

    def target(self, **kwargs) -> None:
        """Actual work — runs in a worker thread. Override this."""
        raise NotImplementedError("OperationForm subclass must implement target()")

    # ------------------------------------------------------------------ #
    # Frame
    # ------------------------------------------------------------------ #
    def _build_shell(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(6)

        if self.description:
            desc = QLabel(self.description, self)
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #666; font-size: 10pt;")
            outer.addWidget(desc)

        # Body — subclass fills this
        body_host = QWidget(self)
        self.body_layout = QVBoxLayout(body_host)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(6)
        outer.addWidget(body_host)

        # Run button row
        row = QHBoxLayout()
        row.addStretch()
        self.run_btn = QPushButton(icon("rocket"), "  Run", self)
        self.run_btn.setMinimumWidth(140)
        self.run_btn.clicked.connect(self.on_run)
        row.addWidget(self.run_btn)
        outer.addLayout(row)

    # ------------------------------------------------------------------ #
    # Run dispatch
    # ------------------------------------------------------------------ #
    def on_run(self) -> None:
        try:
            kwargs = self.collect_args()
        except Exception as exc:
            QMessageBox.warning(self, f"{self.title}: invalid input", str(exc))
            return

        # Discover work function. Some forms override ``on_run`` directly
        # and never hit this path; others rely on ``target``.
        def _work() -> None:
            self.target(**kwargs)

        run_with_progress(
            parent=self.window(),
            title=f"{self.title}…",
            target=_work,
            on_success=lambda: (self.completed.emit(),
                                QMessageBox.information(self, self.title, "Done.")),
        )


# --------------------------------------------------------------------------- #
# WorkflowPage — a QToolBox with sub-sections hosting forms
# --------------------------------------------------------------------------- #
class WorkflowPage(QWidget):
    """A page in the central stack. Hosts sub-sections, each a set of forms.

    Structure::

        WorkflowPage
        └─ QScrollArea
            └─ QToolBox
                ├─ Section 1 (collapsed by default)
                │   ├─ form1
                │   └─ form2
                └─ Section 2
                    └─ form3

    Forms are lazy — a section's forms are instantiated on first expand.
    Cold page-switching cost stays flat regardless of how many forms
    the page declares.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None,
                 config_path: Optional[str] = None) -> None:
        super().__init__(parent)
        self.title = title
        self.config_path = config_path
        # declared: {section_title: [(form_cls, kwargs)]}
        self._declared: dict[str, list[tuple[Type[OperationForm], dict]]] = {}
        # instantiated forms by section index
        self._instantiated: set[int] = set()
        self._section_hosts: dict[int, QWidget] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        self.toolbox = QToolBox(scroll)
        scroll.setWidget(self.toolbox)
        outer.addWidget(scroll)
        self.toolbox.currentChanged.connect(self._on_section_changed)

    def add_section(self, section_title: str,
                    forms: list[tuple[Type[OperationForm], dict]]) -> None:
        """Add a named section with a list of (form_class, kwargs) pairs.
        Forms are built lazily on first expand.
        """
        host = QWidget(self.toolbox)
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(8, 8, 8, 8)
        host_layout.setSpacing(10)
        index = self.toolbox.addItem(host, section_title)
        self._declared[section_title] = forms
        self._section_hosts[index] = host
        # If this is the first section, force-instantiate so the page
        # isn't empty on first show.
        if index == 0:
            self._instantiate(index)

    def _on_section_changed(self, index: int) -> None:
        if index not in self._instantiated and index >= 0:
            self._instantiate(index)

    def _instantiate(self, index: int) -> None:
        host = self._section_hosts.get(index)
        if host is None:
            return
        title = self.toolbox.itemText(index)
        forms = self._declared.get(title, [])
        layout = host.layout()
        for form_cls, kwargs in forms:
            kwargs = dict(kwargs)
            kwargs.setdefault("config_path", self.config_path)
            kwargs.setdefault("parent", host)
            form = form_cls(**kwargs)
            if form.title:
                hdr = QLabel(f"<b>{form.title}</b>", host)
                hdr.setStyleSheet("font-size: 11pt; padding-top: 4px;")
                layout.addWidget(hdr)
            layout.addWidget(form)
        layout.addStretch()
        self._instantiated.add(index)


# --------------------------------------------------------------------------- #
# MufasaWorkbench — the main window
# --------------------------------------------------------------------------- #
class MufasaWorkbench(QMainWindow):
    """Main workbench window — sidebar nav + workflow pages."""

    def __init__(self, project_config_path: Optional[str] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.project_config_path = project_config_path
        self.setWindowTitle(
            f"Mufasa — {project_config_path}" if project_config_path
            else "Mufasa"
        )
        self.setWindowIcon(icon("SimBA_logo_3_small"))
        self.resize(1440, 900)

        # Central split: sidebar | content stack
        split = QSplitter(Qt.Horizontal, self)
        self.sidebar = QListWidget(split)
        self.sidebar.setFixedWidth(220)
        # Structural padding only — leave selection colours to the
        # system theme so the sidebar matches the rest of the OS UI.
        self.sidebar.setStyleSheet(
            "QListWidget::item { padding: 10px 12px; }"
        )
        self.stack = QStackedWidget(split)
        split.addWidget(self.sidebar)
        split.addWidget(self.stack)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)

        self._pages_by_title: dict[str, WorkflowPage] = {}

        self._build_menus()
        self._build_statusbar()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_page(self, title: str, icon_name: Optional[str] = None
                 ) -> WorkflowPage:
        """Add a new sidebar entry + workflow page; return the page."""
        page = WorkflowPage(title, parent=self.stack,
                            config_path=self.project_config_path)
        self.stack.addWidget(page)
        item = QListWidgetItem(title, self.sidebar)
        if icon_name:
            icn = icon(icon_name)
            if not icn.isNull():
                item.setIcon(icn)
        self._pages_by_title[title] = page
        if self.sidebar.count() == 1:
            self.sidebar.setCurrentRow(0)
        return page

    def launch_dialog(self, dialog_cls: Type[QWidget], **kwargs) -> None:
        """Escape hatch for popups that legitimately need their own window.

        Subclasses of :class:`MufasaDialog` take ``config_path`` and render
        full-feature forms. The workbench stashes a reference so the
        dialog isn't GC'd immediately.
        """
        kwargs.setdefault("config_path", self.project_config_path)
        dlg = dialog_cls(**kwargs)
        dlg.show()
        self._dialog_refs = getattr(self, "_dialog_refs", [])
        self._dialog_refs.append(dlg)

    # ------------------------------------------------------------------ #
    # Chrome
    # ------------------------------------------------------------------ #
    def _build_menus(self) -> None:
        mb = self.menuBar()
        self._file_menu = mb.addMenu("&File")
        a_quit = QAction("Quit", self)
        a_quit.setShortcut(QKeySequence.Quit)
        a_quit.triggered.connect(self.close)
        self._file_menu.addAction(a_quit)

        self._tools_menu = mb.addMenu("&Tools")

        self._help_menu = mb.addMenu("&Help")
        a_about = QAction("About Mufasa", self)
        a_about.triggered.connect(self._about)
        self._help_menu.addAction(a_about)

    def add_tools_action(self, label: str,
                         callback: Callable[[], None],
                         shortcut: Optional[str] = None) -> None:
        """Register a menu action (for zero-field operations that used
        to be popups: reverse, flip, archive, etc.).
        """
        act = QAction(label, self)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(callback)
        self._tools_menu.addAction(act)

    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        msg = (f"CPU {linux_env.cpu_count()}  ·  "
               f"{'CUDA ✓' if linux_env.cuda_available() else 'CUDA ✗'}  ·  "
               f"{'NVENC ✓' if linux_env.nvenc_available() else 'NVENC ✗'}  ·  "
               f"{linux_env.detect_display_server()}")
        if self.project_config_path:
            msg = f"{self.project_config_path}  —  {msg}"
        sb.showMessage(msg)

    def _about(self) -> None:
        import sys, PySide6
        QMessageBox.about(
            self, "About Mufasa",
            f"<b>Mufasa</b><br>"
            f"Behavioural analysis toolkit (fork of SimBA)<br><br>"
            f"PySide6 {PySide6.__version__}  ·  "
            f"Python {sys.version.split()[0]}<br>"
            f"Display: {linux_env.detect_display_server()}"
        )


__all__ = ["OperationForm", "WorkflowPage", "MufasaWorkbench"]
