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

* The progress dialog is **modeless** by default (``Qt.WindowModal``
  instead of ``Qt.ApplicationModal``). The GUI stays responsive for
  other popups — only the parent window is blocked.
* We *don't* use ``QProgressDialog.setValue(-1)`` indeterminate mode
  for backends that don't report progress — it ends up busy-animating
  forever. Instead, we leave it at 0 and update the status label.
* Cancel is cooperative. Clicking Cancel calls
  ``runner.request_cancel()`` which sets an :class:`threading.Event`.
  Backends that check the event will stop at the next checkpoint.
  Backends that don't will complete normally; the dialog reflects this
  as "Finishing…" until the worker returns.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from mufasa.ui_qt.runner import ProcessorRunner


def run_with_progress(
    parent: QWidget,
    *,
    title: str,
    target: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    inject_runner_kw: Optional[str] = None,
    on_success: Optional[Callable[[], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    cancellable: bool = True,
    label_template: str = "Running…",
) -> ProcessorRunner:
    """Run ``target`` in a worker thread with a Qt progress dialog.

    :param parent: any :class:`QWidget`; owns the progress dialog.
    :param title: title of the progress dialog window.
    :param target, args, kwargs: forwarded to
        :class:`ProcessorRunner`. The target may call
        ``runner.report_progress(int)`` and ``runner.report_status(str)``
        to update the dialog — if ``inject_runner_kw`` is provided.
    :param inject_runner_kw: if set (e.g. ``"runner"``), the runner is
        injected as that keyword into ``target``.
    :param on_success: called on the GUI thread after the target
        completes normally. Receives no args.
    :param on_error: called on the GUI thread with the :class:`Exception`
        instance on failure. If ``None``, a :class:`QMessageBox.critical`
        with the exception message is shown instead.
    :param cancellable: whether to show a Cancel button. Disabling is
        appropriate for operations that can't be safely interrupted
        (e.g. atomic file rewrites).
    :param label_template: initial label text. Status updates from the
        worker override this.
    """
    runner = ProcessorRunner(
        target=target, args=args, kwargs=kwargs or {},
        inject_runner_kw=inject_runner_kw,
    )

    dlg = QProgressDialog(label_template, "Cancel" if cancellable else "", 0, 100, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(150)   # don't flash for trivially quick ops
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    if not cancellable:
        # Passing "" as cancel-button text still draws a button in Qt 6.
        # Explicitly disable it.
        dlg.setCancelButton(None)  # type: ignore[arg-type]

    # ---- signal wiring ------------------------------------------------- #
    runner.progress.connect(dlg.setValue)
    runner.status.connect(dlg.setLabelText)

    def _on_completed() -> None:
        dlg.setValue(100)
        dlg.close()
        if on_success is not None:
            on_success()

    def _on_errored(exc: Exception) -> None:
        dlg.close()
        if on_error is not None:
            on_error(exc)
        else:
            QMessageBox.critical(parent, title, str(exc))

    runner.completed.connect(_on_completed)
    runner.errored.connect(_on_errored)

    if cancellable:
        # Wire the dialog's Cancel button. Note: request_cancel is
        # cooperative — if the backend doesn't check cancel_event, it
        # will run to completion. Surface that to the user by updating
        # the dialog label.
        def _on_cancel() -> None:
            runner.request_cancel()
            dlg.setLabelText("Cancelling… (waiting for current step)")
            # Prevent the dialog from closing until the worker actually
            # returns — the completed/errored handler will close it.
            dlg.show()

        dlg.canceled.connect(_on_cancel)

    runner.start()
    dlg.show()
    return runner


__all__ = ["run_with_progress"]
