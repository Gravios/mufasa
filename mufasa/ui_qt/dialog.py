"""
mufasa.ui_qt.dialog
==================

``QDialog`` subclass replacing :class:`mufasa.mixins.pop_up_mixin.PopUpMixin`.

The Tk :class:`PopUpMixin` is a base class that creates a
:class:`Toplevel`, wraps it in a scrollable :class:`Canvas`, preloads ~200
icons, and exposes helper methods for building common pieces (checkbox
groups, dropdown groups, a "RUN" button bar).

This Qt6 version preserves the **method names and return types** so
ported popups keep compiling:

    >>> class MyPopup(MufasaDialog, ConfigReader):
    ...     def __init__(self, config_path):
    ...         ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
    ...         MufasaDialog.__init__(self, title="MY POPUP", icon="cog")
    ...         cb_dict = self.create_cb_frame(cb_titles=["Attack", "Sniff"])
    ...         self.create_run_frm(run_function=self.run)
    ...         self.exec()  # modal; use .show() for modeless

Key differences from the Tk original:

* No scrollable Canvas wrapper around a Frame — :class:`QScrollArea` is
  the standard idiom and handles resize + HiDPI correctly.
* ``create_cb_frame`` returns ``Dict[str, _MufasaCheckbox]``. The checkbox
  objects expose ``.get()``/``.set()`` so Tk-style
  ``cb_dict["Attack"].get()`` still works.
* Icons come from :mod:`mufasa.ui_qt.icon_cache` (one decode, shared).
* The dialog is parented to the currently-active window if any, matching
  the Tk convention of popups transient-to-root.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication, QDialog, QGridLayout, QGroupBox,
                               QScrollArea, QVBoxLayout, QWidget)

from mufasa.ui_qt.icon_cache import icon as _icon
from mufasa.ui_qt.widgets import (NW, Formats, MufasaButton, MufasaDropDown,
                                 MufasaCheckbox, _MufasaCheckbox)


class MufasaDialog(QDialog):
    """Drop-in replacement for :class:`PopUpMixin`.

    Subclasses typically inherit from both this class and ``ConfigReader``
    (legacy MI pattern preserved from the Tk codebase).
    """

    def __init__(
        self,
        title: str,
        config_path: Optional[str] = None,
        main_scrollbar: bool = True,
        size: tuple[int, int] = (960, 720),
        icon: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        # Auto-parent to the active top-level window so popups behave
        # transient-to-root (matching the Tk default).
        if parent is None:
            app = QApplication.instance()
            parent = app.activeWindow() if app is not None else None
        # Explicit base init. Under MI with non-Qt bases, Shiboken's
        # cooperative super() walks the MRO past Qt classes and re-invokes
        # the non-Qt __init__ with stray positional args — a well-known
        # PySide6 trap. The Qt port resolves this by **composition, not
        # MI**: ``config_path`` here constructs a ConfigReader which is
        # attached as ``self._cfg``, and its attributes are exposed via
        # :meth:`__getattr__` so subclasses see ``self.config_path``,
        # ``self.clf_names``, ``self.project_path`` etc. as if they'd
        # inherited from ConfigReader.
        QDialog.__init__(self, parent)
        self.setWindowTitle(title)
        self.setMinimumSize(*size)
        if icon is not None:
            icn = _icon(icon)
            if not icn.isNull():
                self.setWindowIcon(icn)

        # Outer layout → scroll area → inner content widget with grid.
        # All helper methods append into ``self.main_frm``'s grid layout.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        if main_scrollbar:
            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            self.main_frm = QWidget(scroll)
            scroll.setWidget(self.main_frm)
            outer.addWidget(scroll)
        else:
            self.main_frm = QWidget(self)
            outer.addWidget(self.main_frm)
        self._main_layout = QGridLayout(self.main_frm)
        self._main_layout.setContentsMargins(6, 6, 6, 6)
        self._main_layout.setSpacing(4)
        # Track next auto-append row for children_cnt_main() parity.
        self._next_row = 0

        # NB: deliberately **not** setting placeholder ``self.palette_options
        # = []`` etc. here. The attribute proxy (:meth:`__getattr__`)
        # forwards to the attached ConfigReader, which sets these to real
        # values. Instance-level empty defaults would shadow the proxy.

        # Legacy PopUpMixin also exposes ``self.root`` as the Toplevel. In
        # Qt there's no separate root; the dialog itself is the window.
        self.root = self

        # Composition with ConfigReader, if a config_path was supplied.
        # Lazy-imported so ``mufasa.ui_qt.dialog`` stays importable without
        # the heavy mufasa backend (numba, cv2, pandas) being initialised
        # until a popup actually wants to read a project.
        self._cfg = None
        if config_path is not None:
            from mufasa.mixins.config_reader import ConfigReader  # noqa
            self._cfg = ConfigReader(
                config_path=config_path, read_video_info=False
            )

    # ----- ConfigReader attribute proxy ---------------------------------- #
    def __getattr__(self, name: str) -> Any:
        """Fall-through attribute lookup onto the attached ConfigReader.

        Only triggered when normal lookup (instance dict, class, Qt
        properties/signals) fails. Guards:

        * Dunder and private names never proxy (``_cfg``, ``_next_row``,
          ``__class__``, ``__signal__`` etc.) — avoids interfering with
          Qt's meta-object lookup.
        * If no ConfigReader is attached, raises ``AttributeError`` as
          Python requires.
        """
        if name.startswith("_") or name.startswith("__"):
            raise AttributeError(name)
        # Use object.__getattribute__ to bypass our own hook on recursion.
        cfg = object.__getattribute__(self, "__dict__").get("_cfg")
        if cfg is None:
            raise AttributeError(name)
        try:
            return getattr(cfg, name)
        except AttributeError:
            raise AttributeError(name) from None

    # ====================================================================== #
    # Helpers
    # ====================================================================== #
    def children_cnt_main(self) -> int:
        """Number of widgets currently placed in the main grid.

        Matches ``PopUpMixin.children_cnt_main()`` semantics — used by
        callers wanting to append a new frame at the bottom.
        """
        return self._main_layout.count()

    # ---- create_cb_frame: checkbox group --------------------------------- #
    def create_cb_frame(
        self,
        cb_titles: List[str],
        main_frm: Optional[QWidget] = None,
        frm_title: str = "",
        idx_row: int = -1,
        command: Optional[Callable[[str], Any]] = None,
    ) -> Dict[str, _MufasaCheckbox]:
        """Port of :meth:`PopUpMixin.create_cb_frame`.

        Returns a dict mapping title → checkbox widget. Call-sites that
        did ``d["Attack"].get()`` continue to work because the Qt checkbox
        exposes ``.get()``/``.set()``.
        """
        if not cb_titles:
            raise ValueError("cb_titles must contain at least one entry")
        parent = main_frm if main_frm is not None else self.main_frm
        parent_layout = parent.layout()
        if not isinstance(parent_layout, QGridLayout):
            raise TypeError("create_cb_frame requires a QGridLayout on parent")
        cb_frm = QGroupBox(frm_title, parent)
        inner = QGridLayout(cb_frm)
        inner.setContentsMargins(6, 6, 6, 6)
        cb_dict: Dict[str, _MufasaCheckbox] = {}
        for cnt, title in enumerate(cb_titles):
            cb, _ = MufasaCheckbox(
                cb_frm,
                txt=title,
                val=False,
                cmd=(lambda k=title: command(k)) if command else None,
            )
            cb.grid(row=cnt, column=0, sticky=NW)
            cb_dict[title] = cb
        row = self._next_row if idx_row == -1 else idx_row
        parent_layout.addWidget(cb_frm, row, 0, 1, 1, Qt.AlignTop | Qt.AlignLeft)
        if idx_row == -1:
            self._next_row += 1
        return cb_dict

    # ---- create_dropdown_frame: dropdown group --------------------------- #
    def create_dropdown_frame(
        self,
        drop_down_titles: List[str],
        drop_down_options: List[str],
        frm_title: str = "",
        idx_row: int = -1,
        main_frm: Optional[QWidget] = None,
    ) -> Dict[str, MufasaDropDown]:
        """Port of :meth:`PopUpMixin.create_dropdown_frame`."""
        if not drop_down_titles:
            raise ValueError("drop_down_titles must contain at least one entry")
        if len(drop_down_options) < 2:
            raise ValueError("drop_down_options must contain at least 2 options")
        parent = main_frm if main_frm is not None else self.main_frm
        parent_layout = parent.layout()
        dd_frm = QGroupBox(frm_title, parent)
        inner = QGridLayout(dd_frm)
        inner.setContentsMargins(6, 6, 6, 6)
        dd_dict: Dict[str, MufasaDropDown] = {}
        for cnt, title in enumerate(drop_down_titles):
            dd = MufasaDropDown(
                dd_frm, drop_down_options, label=title, label_width=35,
                value=drop_down_options[0],
            )
            dd.grid(row=cnt, column=0, sticky=NW)
            dd_dict[title] = dd
        row = self._next_row if idx_row == -1 else idx_row
        parent_layout.addWidget(dd_frm, row, 0, 1, 1, Qt.AlignTop | Qt.AlignLeft)
        if idx_row == -1:
            self._next_row += 1
        return dd_dict

    # ---- create_run_frm: single RUN button ------------------------------- #
    def create_run_frm(
        self,
        run_function: Callable,
        title: str = "RUN",
        btn_txt_clr: str = "black",
        idx: Optional[int] = None,
        btn_icon_name: Optional[str] = "rocket",
        padx: int = 0,
        pady: int = 0,
    ) -> None:
        """Port of :meth:`PopUpMixin.create_run_frm`.

        Replaces any existing run frame (same semantics as Tk version).
        """
        if hasattr(self, "run_frm") and self.run_frm is not None:
            self.run_frm.setParent(None)
            self.run_frm.deleteLater()
        self.run_frm = QGroupBox(title, self.main_frm)
        # No colour override — the QGroupBox title and its MufasaButton
        # inherit from the system theme now.
        run_layout = QGridLayout(self.run_frm)
        run_layout.setContentsMargins(6, 6, 6, 6)
        self.run_btn = MufasaButton(
            self.run_frm,
            txt=title,
            img=btn_icon_name,
            cmd=run_function,
        )
        self.run_btn.grid(row=0, column=0, sticky=NW)
        row = self._next_row + 1 if idx is None else idx
        self._main_layout.addWidget(
            self.run_frm, row, 0, 1, 1, Qt.AlignTop | Qt.AlignLeft
        )
        if idx is None:
            self._next_row = row + 1

    # ---- place_window_at_corner / place_frm_at_top_right ---------------- #
    def place_frm_at_top_right(self, frm: Optional[QWidget] = None) -> None:
        """Move the dialog to the top-right of its screen. Tk-parity method."""
        target = frm if frm is not None else self
        screen = target.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        target.move(geo.right() - target.width(), geo.top())


__all__ = ["MufasaDialog"]
