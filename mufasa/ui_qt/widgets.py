"""
mufasa.ui_qt.widgets
===================

PySide6 (Qt 6.8) replacements for the ~8 Tkinter widget factories that
account for the bulk of widget instantiation in SimBA. The public
surface deliberately mirrors the signatures in
:mod:`mufasa.ui.tkinter_functions`, so most popup call-sites port
mechanically.

**Design notes**

* Every widget exposes a geometry-shim method :meth:`grid` that forwards
  to the parent's :class:`QGridLayout` (creating one lazily). This lets
  call-sites keep their ``w.grid(row=0, column=1, sticky="w")`` lines.
  ``sticky`` is translated to a :class:`Qt.AlignmentFlag`.
* ``cmd`` callbacks are connected to ``clicked`` / ``toggled`` /
  ``textChanged`` signals as appropriate. Lazy kwargs in the legacy
  ``cmd_kwargs`` dict are evaluated at click-time (matching Tk
  behaviour).
* ``thread=True`` on :class:`MufasaButton` dispatches to a shared
  :class:`QThreadPool` via :class:`_Runnable`. **Do not touch widgets
  inside the callback** — cross back via signals (the legacy code
  already respects this, since Tk widgets aren't thread-safe either).
* Icons go through :mod:`mufasa.ui_qt.icon_cache` — one decode, process-
  wide reuse.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from PySide6.QtCore import Qt, QRunnable, QThreadPool, QTimer, Signal
from PySide6.QtGui import QFont, QIcon, QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

from mufasa.ui_qt.icon_cache import icon as _icon, tooltip as _tooltip


# --------------------------------------------------------------------------- #
# Font / colour constants — inlined to keep ``mufasa.ui_qt`` free of the
# ``mufasa.utils.enums`` import chain (which triggers numpy/cv2 at package
# load). These are kept in lockstep with ``mufasa.utils.enums.Formats`` and
# exported so ported popups can still ``from mufasa.ui_qt.widgets import
# Formats`` if they prefer.
# --------------------------------------------------------------------------- #
class Formats:  # simple namespace, no Enum machinery needed
    class _V:
        def __init__(self, v: Any) -> None: self.value = v
    FONT_REGULAR         = _V(("Poppins Regular", 10))
    FONT_REGULAR_ITALICS = _V(("Poppins Regular", 10, "italic"))
    FONT_REGULAR_BOLD    = _V(("Poppins Bold", 10))
    FONT_HEADER          = _V(("Poppins Regular", 10, "bold"))
    LABELFRAME_GREY      = _V("#DCDCDC")
    BTN_HOVER_CLR        = _V("#d1e0e0")
    BUTTON_WIDTH_XS      = _V(105)
    BUTTON_WIDTH_S       = _V(135)
    BUTTON_WIDTH_L       = _V(310)
    BUTTON_WIDTH_XL      = _V(370)
    BUTTON_WIDTH_XXL     = _V(380)

# --------------------------------------------------------------------------- #
# Drop-in Tk-constant aliases. Legacy code does things like
# ``anchor=W, sticky=NW, state=NORMAL`` — the shim accepts strings and
# translates them. Kept as string constants so ``from mufasa.ui_qt.widgets
# import NW`` still works inside ported popups.
# --------------------------------------------------------------------------- #
N, S, E, W = "n", "s", "e", "w"
NW, NE, SW, SE = "nw", "ne", "sw", "se"
NSEW = "nsew"
CENTER = "center"
NORMAL, DISABLED, READONLY = "normal", "disabled", "readonly"
LEFT, RIGHT, TOP, BOTTOM = "left", "right", "top", "bottom"
BOTH, X, Y = "both", "x", "y"
TRUE, FALSE = True, False

_STICKY_TO_ALIGN = {
    "":   Qt.AlignmentFlag(0),
    "n":  Qt.AlignTop | Qt.AlignHCenter,
    "s":  Qt.AlignBottom | Qt.AlignHCenter,
    "e":  Qt.AlignRight | Qt.AlignVCenter,
    "w":  Qt.AlignLeft | Qt.AlignVCenter,
    "nw": Qt.AlignTop | Qt.AlignLeft,
    "ne": Qt.AlignTop | Qt.AlignRight,
    "sw": Qt.AlignBottom | Qt.AlignLeft,
    "se": Qt.AlignBottom | Qt.AlignRight,
    "center": Qt.AlignCenter,
    "nsew": Qt.AlignmentFlag(0),   # no alignment = widget fills cell
}


def _qfont_from_tk(tk_font: tuple) -> QFont:
    """Translate SimBA's ``Formats.FONT_REGULAR.value`` tuples
    (family, size[, style]) into a :class:`QFont`."""
    family = tk_font[0] if len(tk_font) >= 1 else "Sans"
    size = tk_font[1] if len(tk_font) >= 2 else 10
    style = tk_font[2] if len(tk_font) >= 3 else ""
    f = QFont(family, int(size))
    if "bold" in style:
        f.setBold(True)
    if "italic" in style:
        f.setItalic(True)
    return f


# --------------------------------------------------------------------------- #
# Geometry shim — lets Qt widgets accept ``.grid(row=, column=, sticky=)``
# --------------------------------------------------------------------------- #
class _GeometryShim:
    """Mixin providing a Tk-compatible ``grid()`` method.

    Assumes ``self`` is a :class:`QWidget` whose parent's layout is (or
    will become) a :class:`QGridLayout`. Lazily installs one on first
    call if the parent has none.
    """

    # These are Python ints (px). Tk callers pass ``padx=5`` or ``padx=(3,0)``.
    def grid(
        self,
        row: int = 0,
        column: int = 0,
        sticky: str = "",
        padx: int | tuple = 0,
        pady: int | tuple = 0,
        columnspan: int = 1,
        rowspan: int = 1,
        **_ignored,
    ) -> None:
        parent: QWidget = self.parentWidget()  # type: ignore[attr-defined]
        if parent is None:
            raise RuntimeError(
                f"{type(self).__name__}.grid() called before the widget "
                f"was parented. Pass ``parent=`` to the constructor."
            )
        layout = parent.layout()
        if layout is None:
            layout = QGridLayout(parent)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)
        if not isinstance(layout, QGridLayout):
            # Caller previously used pack() on this parent. Fall back to
            # appending at the end of whatever layout exists.
            layout.addWidget(self)  # type: ignore[attr-defined]
            return
        align = _STICKY_TO_ALIGN.get(sticky.lower(), Qt.AlignmentFlag(0))
        layout.addWidget(self, row, column, rowspan, columnspan, align)
        # Honour padx/pady by nudging the widget's contentsMargins. Tk
        # semantics: external padding around the cell.
        def _pad(v):
            if isinstance(v, tuple):
                return int(v[0]), int(v[1])
            return int(v), int(v)
        lx, rx = _pad(padx)
        ty, by = _pad(pady)
        self.setContentsMargins(lx, ty, rx, by)  # type: ignore[attr-defined]

    def pack(
        self, side: str = "top", fill: str = "", expand: bool = False, **_ignored
    ) -> None:
        parent: QWidget = self.parentWidget()  # type: ignore[attr-defined]
        if parent is None:
            raise RuntimeError("pack() before parenting")
        layout = parent.layout()
        if layout is None:
            if side in ("left", "right"):
                layout = QHBoxLayout(parent)
            else:
                layout = QVBoxLayout(parent)
            layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self)  # type: ignore[attr-defined]
        if expand:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Thread-pool runner for ``thread=True`` buttons
# --------------------------------------------------------------------------- #
class _Runnable(QRunnable):
    def __init__(self, fn: Callable[[], None]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:  # noqa: D401  (Qt signature)
        try:
            self._fn()
        except Exception as exc:  # pragma: no cover — surfaced via logs
            import traceback
            traceback.print_exc()


_POOL = QThreadPool.globalInstance()


# --------------------------------------------------------------------------- #
# MufasaButton — QPushButton with hover styling + cmd_kwargs lazy eval
# --------------------------------------------------------------------------- #
class _MufasaButton(_GeometryShim, QPushButton):
    """QPushButton subclass reproducing the Tk MufasaButton hover semantics."""

    def __init__(
        self,
        parent: QWidget,
        txt: str,
        *,
        txt_clr: Optional[str] = None,      # deprecated; ignored
        bg_clr: Optional[str] = None,       # deprecated; ignored
        hover_bg_clr: Optional[str] = None,  # deprecated; ignored
        font: tuple = Formats.FONT_REGULAR.value,
        hover_font: tuple = Formats.FONT_REGULAR_BOLD.value,
        width: Optional[int] = None,
        height: Optional[int] = None,
        img: Optional[str | QIcon] = None,
        cmd: Optional[Callable] = None,
        cmd_kwargs: Optional[dict] = None,
        enabled: bool = True,
        thread: bool = False,
        tooltip_txt: Optional[str] = None,
        tooltip_key: Optional[str] = None,
        **_ignored,
    ) -> None:
        super().__init__(txt, parent)
        self._cmd = cmd
        self._cmd_kwargs = cmd_kwargs or {}
        self._thread = thread
        self._base_font = _qfont_from_tk(font)
        self._hover_font = _qfont_from_tk(hover_font)
        self.setFont(self._base_font)
        # Intentionally no setStyleSheet() here — let the system Qt
        # theme drive button appearance. Legacy Tk-era colour overrides
        # (bg_clr, hover_bg_clr, forced black text) were removed; those
        # kwargs are accepted for backward compatibility but ignored.
        if img is not None:
            if isinstance(img, str):
                self.setIcon(_icon(img))
            elif isinstance(img, QIcon):
                self.setIcon(img)
        if width is not None:
            # Tk widths are char-count-ish; use as pixel hint (SimBA already
            # passes pixel-ish values like 310, 370 via BUTTON_WIDTH_L/XL).
            self.setFixedWidth(int(width))
        if height is not None:
            self.setFixedHeight(int(height))
        self.setEnabled(enabled)
        if tooltip_txt:
            self.setToolTip(tooltip_txt)
        elif tooltip_key:
            tip = _tooltip(tooltip_key)
            if tip is not None:
                self.setToolTip(tip)
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self) -> None:
        if self._cmd is None:
            return
        # Lazy kwargs: callables in cmd_kwargs are evaluated at click time
        kwargs = {
            k: (v() if callable(v) else v) for k, v in self._cmd_kwargs.items()
        }
        fn = self._cmd
        if self._thread:
            _POOL.start(_Runnable(lambda: fn(**kwargs)))
        else:
            fn(**kwargs)


def MufasaButton(
    parent: QWidget,
    txt: str,
    **kwargs,
) -> _MufasaButton:
    """Factory matching :func:`mufasa.ui.tkinter_functions.MufasaButton`."""
    return _MufasaButton(parent, txt, **kwargs)


# --------------------------------------------------------------------------- #
# MufasaLabel — QLabel with optional icon + hyperlink behaviour
# --------------------------------------------------------------------------- #
class _MufasaLabel(_GeometryShim, QLabel):
    def __init__(
        self,
        parent: QWidget,
        txt: str = "",
        *,
        txt_clr: Optional[str] = None,
        bg_clr: Optional[str] = None,
        font: tuple = Formats.FONT_REGULAR.value,
        img: Optional[str] = None,
        link: Optional[str] = None,
        cursor: Optional[str] = None,
        width: Optional[int] = None,
        anchor: str = "w",
        compound: Optional[str] = "left",
        **_ignored,
    ) -> None:
        super().__init__(txt, parent)
        self.setFont(_qfont_from_tk(font))
        # Only emit a stylesheet when the caller actually specified a
        # colour. Default labels inherit from the system palette.
        css: list[str] = []
        if txt_clr is not None:
            css.append(f"color: {txt_clr};")
        if bg_clr is not None:
            css.append(f"background-color: {bg_clr};")
        if css:
            self.setStyleSheet("QLabel {" + " ".join(css) + "}")
        if width is not None:
            self.setMinimumWidth(int(width))
        self.setAlignment(_STICKY_TO_ALIGN.get(anchor.lower(), Qt.AlignLeft))
        if img is not None:
            self.setPixmap(_icon(img).pixmap(18, 18))
        self._link = link
        if link is not None:
            self.setCursor(Qt.PointingHandCursor)
            # QLabel native rich-text linking for free
            self.setText(f'<a href="{link}">{txt}</a>')
            self.setOpenExternalLinks(True)
        elif cursor == "hand2":
            self.setCursor(Qt.PointingHandCursor)


def MufasaLabel(parent: QWidget, txt: str = "", **kwargs) -> _MufasaLabel:
    return _MufasaLabel(parent, txt, **kwargs)


# --------------------------------------------------------------------------- #
# MufasaCheckbox — returns (widget, value-getter) matching the Tk API
# --------------------------------------------------------------------------- #
class _MufasaCheckbox(_GeometryShim, QCheckBox):
    def __init__(
        self,
        parent: QWidget,
        txt: str,
        *,
        txt_clr: Optional[str] = None,   # deprecated; ignored
        font: tuple = Formats.FONT_REGULAR.value,
        val: bool = False,
        state: str = NORMAL,
        cmd: Optional[Callable] = None,
        **_ignored,
    ) -> None:
        super().__init__(txt, parent)
        self.setFont(_qfont_from_tk(font))
        # No setStyleSheet — system theme drives checkbox appearance.
        self.setChecked(bool(val))
        self.setEnabled(state != DISABLED)
        if cmd is not None:
            self.toggled.connect(lambda _=None: cmd())

    # Tk-style facade so old code calling ``var.get()`` keeps working if
    # callers stash the checkbox as the "var".
    def get(self) -> bool:
        return self.isChecked()

    def set(self, value: bool) -> None:
        self.setChecked(bool(value))


def MufasaCheckbox(parent: QWidget, txt: str, **kwargs) -> tuple[_MufasaCheckbox, _MufasaCheckbox]:
    """Returns ``(widget, value_proxy)`` for API parity with the Tk factory,
    which returned ``(Checkbutton, BooleanVar)``. Here both are the same
    object (the checkbox is its own ``BooleanVar``)."""
    cb = _MufasaCheckbox(parent, txt, **kwargs)
    return cb, cb


# --------------------------------------------------------------------------- #
# Entry_Box — QLineEdit with optional numeric validator + label prefix
# --------------------------------------------------------------------------- #
class Entry_Box(_GeometryShim, QWidget):
    """API-compatible port of :class:`mufasa.ui.tkinter_functions.Entry_Box`."""

    textChangedSig = Signal(str)

    def __init__(
        self,
        parent: QWidget,
        fileDescription: str = "",
        *,
        labelwidth: Optional[int] = None,
        status: str = NORMAL,
        validation: Optional[str] = None,
        entry_box_width: Optional[int] = None,
        value: Optional[Any] = None,
        label_font: tuple = Formats.FONT_REGULAR.value,
        entry_font: tuple = Formats.FONT_REGULAR.value,
        justify: str = "left",
        cmd: Optional[Callable] = None,
        allow_blank: bool = False,
        **_ignored,
    ) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.labelname = fileDescription
        self.allow_blank = allow_blank
        self._lbl = QLabel(fileDescription, self)
        self._lbl.setFont(_qfont_from_tk(label_font))
        if labelwidth is not None:
            self._lbl.setMinimumWidth(int(labelwidth) * 7)  # ~7px per char
        self._entry = QLineEdit(self)
        self._entry.setFont(_qfont_from_tk(entry_font))
        if entry_box_width is not None:
            self._entry.setFixedWidth(int(entry_box_width) * 7)
        align_map = {"left": Qt.AlignLeft, "right": Qt.AlignRight, "center": Qt.AlignCenter}
        self._entry.setAlignment(align_map.get(justify, Qt.AlignLeft))
        if validation == "numeric":
            # ``QIntValidator`` won't reject intermediate blank input, which
            # matches the Tk helper's behaviour of allowing editing.
            self._entry.setValidator(QIntValidator(self._entry))
        self._entry.setEnabled(status != DISABLED)
        self._entry.setReadOnly(status == READONLY)
        if value is not None:
            self._entry.setText(str(value))
        if cmd is not None:
            self._entry.textChanged.connect(lambda s: cmd(s.strip()))
        self._entry.textChanged.connect(self.textChangedSig)
        lay.addWidget(self._lbl)
        lay.addWidget(self._entry, 1)

    # ----- API parity with the Tk version ---------------------------- #
    @property
    def entry_get(self) -> str:
        return self._entry.text()

    def entry_set(self, val: Any) -> None:
        self._entry.setText(str(val))

    def set_state(self, setstatus: str) -> None:
        self._entry.setEnabled(setstatus != DISABLED)
        self._entry.setReadOnly(setstatus == READONLY)


# --------------------------------------------------------------------------- #
# FileSelect / FolderSelect — QLineEdit + QPushButton + QFileDialog
# --------------------------------------------------------------------------- #
class _PathSelect(_GeometryShim, QWidget):
    """Shared implementation for FileSelect / FolderSelect."""

    def __init__(
        self,
        parent: QWidget,
        fileDescription: str = "",
        *,
        title: Optional[str] = None,
        lblwidth: Optional[int] = None,
        file_types: Optional[list[tuple[str, str]]] = None,
        initialdir: Optional[str | Path] = None,
        initial_path: Optional[str | Path] = None,
        _mode: str = "file",
        **_ignored,
    ) -> None:
        super().__init__(parent)
        self._mode = _mode
        self._title = title or ("Select file" if _mode == "file" else "Select folder")
        self._file_types = file_types
        self._initialdir = str(initialdir) if initialdir else None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._lbl = QLabel(fileDescription, self)
        if lblwidth is not None:
            self._lbl.setMinimumWidth(int(lblwidth) * 7)
        self._path_display = QLineEdit(self)
        self._path_display.setReadOnly(True)
        self._path_display.setText(str(initial_path) if initial_path else "No file/folder selected")
        self._btn = QPushButton("Browse", self)
        self._btn.setIcon(_icon("browse"))
        self._btn.clicked.connect(self._browse)
        lay.addWidget(self._lbl)
        lay.addWidget(self._path_display, 1)
        lay.addWidget(self._btn)

    def _browse(self) -> None:
        if self._mode == "folder":
            path = QFileDialog.getExistingDirectory(
                self, self._title, self._initialdir or ""
            )
        else:
            filter_str = ""
            if self._file_types:
                filter_str = ";;".join(
                    f"{desc} ({pat})" for desc, pat in self._file_types
                )
            path, _ = QFileDialog.getOpenFileName(
                self, self._title, self._initialdir or "", filter_str
            )
        if path:
            self._path_display.setText(path)

    # ----- API parity ------------------------------------------------- #
    @property
    def file_path(self) -> str:
        txt = self._path_display.text()
        return "" if txt.startswith("No file") else txt

    @property
    def folder_path(self) -> str:
        return self.file_path

    def set_state(self, setstatus: str) -> None:
        self._btn.setEnabled(setstatus != DISABLED)


def FileSelect(parent: QWidget, fileDescription: str = "", **kwargs) -> _PathSelect:
    kwargs.setdefault("_mode", "file")
    return _PathSelect(parent, fileDescription, **kwargs)


def FolderSelect(parent: QWidget, folderDescription: str = "", **kwargs) -> _PathSelect:
    kwargs.setdefault("_mode", "folder")
    return _PathSelect(parent, folderDescription, **kwargs)


# --------------------------------------------------------------------------- #
# MufasaDropDown — QComboBox with label
# --------------------------------------------------------------------------- #
class MufasaDropDown(_GeometryShim, QWidget):
    """API-compatible port of :class:`mufasa.ui.tkinter_functions.MufasaDropDown`."""

    currentChanged = Signal(str)

    def __init__(
        self,
        parent: QWidget,
        dropdown_options: Iterable[Any],
        *,
        label: Optional[str] = None,
        label_width: Optional[int] = None,
        label_font: tuple = Formats.FONT_REGULAR.value,
        dropdown_width: Optional[int] = None,
        command: Optional[Callable] = None,
        value: Optional[Any] = None,
        state: Optional[str] = None,
        searchable: bool = False,
        **_ignored,
    ) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        if label is not None:
            self._lbl = QLabel(label, self)
            self._lbl.setFont(_qfont_from_tk(label_font))
            if label_width is not None:
                self._lbl.setMinimumWidth(int(label_width) * 7)
            lay.addWidget(self._lbl)
        else:
            self._lbl = None
        self._cb = QComboBox(self)
        self._cb.addItems([str(o) for o in dropdown_options])
        if searchable:
            self._cb.setEditable(True)
            # QCompleter is on-by-default when editable; nothing more needed
        if dropdown_width is not None:
            self._cb.setMinimumWidth(int(dropdown_width) * 7)
        if value is not None:
            idx = self._cb.findText(str(value))
            if idx >= 0:
                self._cb.setCurrentIndex(idx)
        if state == DISABLED:
            self._cb.setEnabled(False)
        lay.addWidget(self._cb, 1)
        if command is not None:
            self._cb.currentTextChanged.connect(lambda s: command(s))
        self._cb.currentTextChanged.connect(self.currentChanged)

    # ----- Tk parity -------------------------------------------------- #
    def get_value(self) -> str:
        return self._cb.currentText()

    def getChoices(self) -> str:
        return self.get_value()

    def setChoices(self, val: Any) -> None:
        idx = self._cb.findText(str(val))
        if idx >= 0:
            self._cb.setCurrentIndex(idx)
        elif self._cb.isEditable():
            self._cb.setEditText(str(val))


# --------------------------------------------------------------------------- #
# CreateLabelFrameWithIcon — QGroupBox with a titlebar icon
# --------------------------------------------------------------------------- #
class _LabelFrame(_GeometryShim, QGroupBox):
    """Returned by :func:`CreateLabelFrameWithIcon`.

    Carries an internal :class:`QGridLayout` so children using
    ``.grid(row, column)`` just work.
    """

    def __init__(self, parent: QWidget, header: str = "", **_ignored) -> None:
        super().__init__(header, parent)
        self.setFont(_qfont_from_tk(Formats.FONT_HEADER.value))
        inner = QGridLayout(self)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(4)


def CreateLabelFrameWithIcon(
    parent: QWidget,
    header: str = "",
    *,
    icon_name: Optional[str] = None,
    icon_link: Optional[str] = None,
    **kwargs,
) -> _LabelFrame:
    """Factory matching the Tk version's keyword surface.

    The ``icon_link`` click-through to docs is preserved via a small
    helper button in the groupbox header (Qt has no built-in titlebar
    icon support on ``QGroupBox``).
    """
    gb = _LabelFrame(parent, header=header, **kwargs)
    if icon_link is not None:
        # Add a tiny "(?)" button at top-right linking to docs
        help_btn = QPushButton("?", gb)
        help_btn.setFlat(True)
        help_btn.setFixedSize(18, 18)
        help_btn.setToolTip(icon_link)
        help_btn.clicked.connect(
            lambda: __import__("webbrowser").open(icon_link)
        )
        # Attach top-right via absolute positioning on resize
        def _place() -> None:
            help_btn.move(gb.width() - 24, 2)
        QTimer.singleShot(0, _place)
        gb.resizeEvent = lambda e: (_place(), QGroupBox.resizeEvent(gb, e))  # type: ignore
    return gb


__all__ = [
    "MufasaButton", "MufasaLabel", "MufasaCheckbox", "Entry_Box", "FileSelect",
    "FolderSelect", "MufasaDropDown", "CreateLabelFrameWithIcon",
    "NW", "NE", "SW", "SE", "N", "S", "E", "W", "NSEW", "CENTER",
    "NORMAL", "DISABLED", "READONLY", "LEFT", "RIGHT", "TOP", "BOTTOM",
]
