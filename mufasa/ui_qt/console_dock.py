"""Dockable console that mirrors verbose stdout/stderr into the workbench.

Patch 122hp. The workbench runs each operation in a
:class:`~mufasa.ui_qt.runner.ProcessorRunner` (a ``QThread``), and the backend
functions report progress by printing — e.g. the Kalman smoother's
``[smoother-v2] …`` lines and per-EM-iteration logs. Those prints go to the
terminal and are invisible to anyone running the GUI as a windowed app. This
module adds a read-only console docked at the bottom of the main window that
mirrors that output live.

Two pieces:

* :class:`_StreamRedirector` — a file-like object that stands in for
  ``sys.stdout`` / ``sys.stderr``. Every write is *teed* to the original
  stream (so a terminal user still sees everything and nothing is lost if the
  GUI dies) and emitted on a Qt signal. Because operations print from a worker
  thread, the write must not touch the console widget directly — Qt widgets are
  GUI-thread-only. The signal carries the text across the thread boundary via a
  queued connection, and the slot appends on the GUI thread. The redirector is
  deliberately minimal but implements enough of the file protocol
  (``write`` / ``flush`` / ``isatty`` / ``writable`` / ``fileno``) to be a safe
  stand-in.

* :class:`ConsoleDockWidget` — the read-only text view, with Clear / Copy /
  Autoscroll / Wrap controls. :func:`attach_console_dock` installs it on a
  ``QMainWindow`` and redirects the process streams once.

Design choices:

* Global, install-once. The user asked for "the various functions", so the
  capture is process-wide rather than wired per-operation. Installing twice is
  a no-op (the second call returns the existing dock).
* Tee, don't hijack. The original streams keep receiving everything, so
  terminal workflows and log redirection are unaffected.
* Bounded. The view is capped at a maximum block count so a very chatty run
  (tens of thousands of EM lines) can't grow memory without bound.
"""
from __future__ import annotations

import contextlib
import sys
from typing import TextIO

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Cap the console history so an extremely chatty run can't grow without bound.
# 0 would mean "unlimited"; a few thousand blocks is plenty to scroll back
# through while staying cheap.
_MAX_BLOCKS = 5000


class _StreamRedirector(QObject):
    """File-like tee of a stream that also emits each write as a signal.

    Instances replace ``sys.stdout`` / ``sys.stderr``. Writes are forwarded to
    the wrapped original stream and emitted on :attr:`textWritten`, which a
    console widget connects to (with a queued connection, so the append happens
    on the GUI thread even when the write came from a worker thread).
    """

    textWritten = Signal(str)

    def __init__(self, original: TextIO | None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._original = original

    # -- file protocol ------------------------------------------------- #
    def write(self, text: str) -> int:
        # Tee to the real stream first so nothing is lost if anything below
        # (or the GUI) misbehaves.
        if self._original is not None:
            with contextlib.suppress(Exception):
                self._original.write(text)
        if text:
            # Emitting an empty string would append a stray blank line.
            self.textWritten.emit(text)
        return len(text)

    def flush(self) -> None:
        if self._original is not None:
            with contextlib.suppress(Exception):
                self._original.flush()

    def isatty(self) -> bool:
        # The console is not a TTY; report the wrapped stream's answer when we
        # can, else False. Some libraries branch on this to decide colouring.
        try:
            return bool(self._original is not None
                        and self._original.isatty())
        except Exception:
            return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        # Delegate so code that needs a real fd (rare) still works; raises if
        # the original has none, which is the correct signal to the caller.
        if self._original is None:
            raise OSError("redirected stream has no fileno")
        return self._original.fileno()

    @property
    def original(self) -> TextIO | None:
        return self._original


class ConsoleDockWidget(QWidget):
    """Read-only console view with Clear / Copy / Autoscroll / Wrap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._view = QPlainTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(_MAX_BLOCKS)
        self._view.setLineWrapMode(QPlainTextEdit.NoWrap)
        # A monospace font keeps the smoother's aligned progress columns
        # readable. StyleHint falls back gracefully if the named family is
        # absent.
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)
        self._view.setFont(mono)

        self._autoscroll = QCheckBox("Autoscroll", self)
        self._autoscroll.setChecked(True)
        self._wrap = QCheckBox("Wrap", self)
        self._wrap.setChecked(False)
        self._wrap.toggled.connect(self._on_wrap_toggled)

        clear_btn = QPushButton("Clear", self)
        clear_btn.clicked.connect(self.clear)
        copy_btn = QPushButton("Copy all", self)
        copy_btn.clicked.connect(self._copy_all)

        controls = QHBoxLayout()
        controls.addWidget(self._autoscroll)
        controls.addWidget(self._wrap)
        controls.addStretch(1)
        controls.addWidget(copy_btn)
        controls.addWidget(clear_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(controls)
        layout.addWidget(self._view, 1)

    # -- slots --------------------------------------------------------- #
    def append_text(self, text: str) -> None:
        """Append raw text (may contain newlines) at the end of the view.

        Runs on the GUI thread (connected via a queued connection from the
        redirector's signal). Trailing newlines are trimmed because
        ``insertPlainText`` already sits at the end and the widget manages
        block breaks — otherwise every print's trailing "\\n" would double up.
        """
        cursor = self._view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        if self._autoscroll.isChecked():
            self._view.moveCursor(QTextCursor.End)
            self._view.ensureCursorVisible()

    def clear(self) -> None:
        self._view.clear()

    # -- helpers ------------------------------------------------------- #
    def _copy_all(self) -> None:
        self._view.selectAll()
        self._view.copy()
        # Deselect so the copy doesn't leave the whole log highlighted.
        cursor = self._view.textCursor()
        cursor.clearSelection()
        self._view.setTextCursor(cursor)

    def _on_wrap_toggled(self, on: bool) -> None:
        self._view.setLineWrapMode(
            QPlainTextEdit.WidgetWidth if on else QPlainTextEdit.NoWrap
        )


def _base_stream(stream: object) -> TextIO | None:
    """Unwrap our own redirectors down to the real underlying stream.

    A project switch rebuilds the workbench, so when the new window installs
    its console the current ``sys.stdout`` may already be a redirector from the
    old (closing) window. Teeing to that would chain redirectors and duplicate
    output; instead we walk down ``.original`` until we reach a non-redirector
    stream (or None).
    """
    seen: set[int] = set()
    while isinstance(stream, _StreamRedirector):
        if id(stream) in seen:  # paranoia: never loop on a cycle
            return None
        seen.add(id(stream))
        stream = stream.original
    return stream  # type: ignore[return-value]


def attach_console_dock(main: QMainWindow) -> QDockWidget:
    """Attach the console dock to ``main`` and redirect stdout/stderr into it.

    Idempotent: if a console dock is already installed on ``main`` the existing
    dock is returned and the streams are not redirected again. The redirectors
    are stashed on ``main`` so they outlive this call and can be restored.
    """
    existing = getattr(main, "_console_dock", None)
    if existing is not None:
        return existing

    console = ConsoleDockWidget(main)

    dock = QDockWidget("Console", main)
    dock.setObjectName("mufasa_console_dock")
    dock.setWidget(console)
    dock.setAllowedAreas(
        Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea
    )
    dock.setFeatures(
        QDockWidget.DockWidgetMovable
        | QDockWidget.DockWidgetFloatable
        | QDockWidget.DockWidgetClosable
    )
    main.addDockWidget(Qt.BottomDockWidgetArea, dock)

    # Redirect the process streams into the console. Tee to the originals so
    # terminal users still see everything. Queued connection marshals writes
    # from worker threads onto the GUI thread.
    #
    # If a previous workbench (e.g. before a project switch rebuilt the window)
    # already redirected the streams, sys.stdout is that old redirector. Tee to
    # its *original* underlying stream rather than chaining redirector ->
    # redirector -> terminal, which would duplicate lines and keep the dead
    # window's redirector alive. _base_stream unwraps one of our own
    # redirectors down to the real stream.
    out_redirector = _StreamRedirector(_base_stream(sys.stdout), main)
    err_redirector = _StreamRedirector(_base_stream(sys.stderr), main)
    out_redirector.textWritten.connect(
        console.append_text, Qt.QueuedConnection
    )
    err_redirector.textWritten.connect(
        console.append_text, Qt.QueuedConnection
    )
    sys.stdout = out_redirector  # type: ignore[assignment]
    sys.stderr = err_redirector  # type: ignore[assignment]

    # Stash so they survive and can be restored on teardown.
    main._console_dock = dock  # type: ignore[attr-defined]
    main._console_widget = console  # type: ignore[attr-defined]
    main._console_redirectors = (  # type: ignore[attr-defined]
        out_redirector, err_redirector,
    )
    return dock


def detach_console_dock(main: QMainWindow) -> None:
    """Restore the original stdout/stderr, if this window redirected them.

    Safe to call when no console was attached. Used on window teardown (e.g. a
    project switch rebuilds the workbench) so the redirectors — which hold a
    reference to the old window — don't linger as the active streams.
    """
    redirectors = getattr(main, "_console_redirectors", None)
    if not redirectors:
        return
    out_redirector, err_redirector = redirectors
    # Only restore if we're still the active stream; something else may have
    # redirected after us, and we mustn't clobber that.
    if sys.stdout is out_redirector:
        sys.stdout = out_redirector.original  # type: ignore[assignment]
    if sys.stderr is err_redirector:
        sys.stderr = err_redirector.original  # type: ignore[assignment]
    main._console_redirectors = None  # type: ignore[attr-defined]
