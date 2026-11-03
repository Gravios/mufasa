"""
mufasa.ui_qt.progress
=====================

Thin wrapper gluing :class:`~mufasa.ui_qt.runner.ProcessorRunner` to a
:class:`QProgressDialog` so every port gets a progress bar + cancel
button without re-implementing the signal wiring.

Usage::

    from mufasa.ui_qt.progress import run_with_progress

    run_with_progress(
        parent=self,
        title="Smoothing…",
        target=my_op.run,
        inject_runner_kw="runner",  # op can check runner.cancel_event
        on_success=lambda: QMessageBox.information(self, "Done", "OK"),
    )

The returned :class:`ProcessorRunner` can be stashed on the parent if
the caller needs to interact with it (e.g. cancel from a menu). In
most cases the helper manages everything.

Design notes
------------

* No dialog (patch 122ih). Progress shows inline in the main window's
  status bar — a label, a bar and a Cancel button — so nothing blocks the
  window: the console dock can be scrolled, other pages browsed, while
  the operation runs. (The old window-modal QProgressDialog froze the
  whole workbench, including the console showing the run's own output.)
  Parents not inside a QMainWindow fall back to a NON-modal
  QProgressDialog.
* Because nothing is modal any more, :class:`OperationForm` disables its
  own Run button while its operation is in flight, so the same form
  can't be started twice.
* We *don't* use indeterminate mode for backends that don't report
  progress until there's no progress at all: the bar starts busy and
  switches to 0..100 at the first progress report.
* Cancel is cooperative. Clicking Cancel calls
  ``runner.request_cancel()`` which sets an :class:`threading.Event`.
  Backends that check the event will stop at the next checkpoint.
  Backends that don't will complete normally; the indicator says
  "Cancelling…" until the worker returns.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QWidget,
)

from mufasa.ui_qt.runner import ProcessorRunner


class StatusProgress(QWidget):
    """Inline progress indicator for a QMainWindow status bar."""

    def __init__(self, title: str, label: str, cancellable: bool,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(6)
        self.title = title.rstrip("…. ")
        self.label = QLabel(self)
        self.label.setMaximumWidth(420)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 0)            # busy until the first report
        self.bar.setFixedWidth(140)
        self.bar.setMaximumHeight(14)
        self.bar.setTextVisible(False)
        self.cancel_btn = QPushButton("Cancel", self) if cancellable else None
        lay.addWidget(self.label)
        lay.addWidget(self.bar)
        if self.cancel_btn is not None:
            self.cancel_btn.setMaximumHeight(20)
            lay.addWidget(self.cancel_btn)
        self.setLabelText(label)

    def setLabelText(self, text: str) -> None:  # noqa: N802 (Qt-style API)
        text = " ".join(str(text).split())
        full = f"{self.title}: {text}" if text else self.title
        self.label.setText(full if len(full) <= 80 else full[:77] + "…")
        self.label.setToolTip(full)

    def setValue(self, v: int) -> None:  # noqa: N802
        if self.bar.maximum() == 0:
            self.bar.setRange(0, 100)
        self.bar.setValue(int(v))


def _main_window(parent: QWidget | None) -> QMainWindow | None:
    w = parent.window() if parent is not None else None
    return w if isinstance(w, QMainWindow) else None


def run_with_progress(
    parent: QWidget,
    *,
    title: str,
    target: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    inject_runner_kw: str | None = None,
    on_success: Callable[[], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    cancellable: bool = True,
    label_template: str = "Running…",
    on_finished: Callable[[], None] | None = None,
) -> ProcessorRunner:
    """Run ``target`` in a worker thread with a non-blocking progress indicator.

    :param parent: any :class:`QWidget`. Inside a QMainWindow the progress
        shows in its status bar; otherwise in a non-modal dialog.
    :param title: name of the operation.
    :param target, args, kwargs: forwarded to
        :class:`ProcessorRunner`. The target may call
        ``runner.report_progress(int)`` and ``runner.report_status(str)``
        to update the indicator — if ``inject_runner_kw`` is provided.
    :param inject_runner_kw: if set (e.g. ``"runner"``), the runner is
        injected as that keyword into ``target``.
    :param on_success: called on the GUI thread after the target
        completes normally. Receives no args.
    :param on_error: called on the GUI thread with the :class:`Exception`
        instance on failure. If ``None``, a :class:`QMessageBox.critical`
        with the exception message is shown instead.
    :param cancellable: whether to show a Cancel button.
    :param label_template: initial label text. Status updates from the
        worker override this.
    :param on_finished: called on the GUI thread when the run ends either
        way, before ``on_success`` / ``on_error`` (e.g. to re-enable a Run
        button).
    """
    runner = ProcessorRunner(
        target=target, args=args, kwargs=kwargs or {},
        inject_runner_kw=inject_runner_kw,
    )

    main = _main_window(parent)
    if main is not None:
        ind = StatusProgress(title, label_template, cancellable,
                             main.statusBar())
        main.statusBar().addPermanentWidget(ind)
        ind.show()
        cancel_signal = (ind.cancel_btn.clicked
                         if ind.cancel_btn is not None else None)

        def _close() -> None:
            main.statusBar().removeWidget(ind)
            ind.deleteLater()
    else:
        ind = QProgressDialog(label_template,
                              "Cancel" if cancellable else "", 0, 100, parent)
        ind.setWindowTitle(title)
        ind.setWindowModality(Qt.NonModal)
        ind.setMinimumDuration(150)   # don't flash for trivially quick ops
        ind.setAutoClose(False)
        ind.setAutoReset(False)
        if not cancellable:
            # Passing "" as cancel-button text still draws a button in Qt 6.
            ind.setCancelButton(None)  # type: ignore[arg-type]
        cancel_signal = ind.canceled if cancellable else None
        _close = ind.close

    # Keep the runner alive for as long as its indicator exists.
    ind._runner = runner  # type: ignore[attr-defined]

    # ---- signal wiring ------------------------------------------------- #
    runner.progress.connect(ind.setValue)
    runner.status.connect(ind.setLabelText)

    def _finish() -> None:
        _close()
        if on_finished is not None:
            on_finished()

    def _on_completed() -> None:
        _finish()
        if on_success is not None:
            on_success()

    def _on_errored(exc: Exception) -> None:
        _finish()
        if on_error is not None:
            on_error(exc)
        else:
            QMessageBox.critical(parent, title, str(exc))

    runner.completed.connect(_on_completed)
    runner.errored.connect(_on_errored)

    if cancel_signal is not None:
        # Cooperative: a backend that never checks cancel_event runs to
        # completion; the completed/errored handler closes the indicator.
        def _on_cancel() -> None:
            runner.request_cancel()
            ind.setLabelText("Cancelling… (waiting for current step)")
            if isinstance(ind, StatusProgress) and ind.cancel_btn is not None:
                ind.cancel_btn.setEnabled(False)
            else:
                ind.show()

        cancel_signal.connect(_on_cancel)

    runner.start()
    if not isinstance(ind, StatusProgress):
        ind.show()
    return runner


__all__ = ["run_with_progress", "StatusProgress"]
