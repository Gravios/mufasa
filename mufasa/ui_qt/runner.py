"""
mufasa.ui_qt.runner
==================

QThread-backed wrapper for running SimBA's backend processors off the
GUI thread. Every popup that takes more than ~50 ms to finish should
route its work through :class:`ProcessorRunner`.

**Why this exists**

The Tk codebase has three distinct — and all incorrect — patterns for
long-running work:

1. ``threading.Thread(target=fn).start()`` scattered in
   :func:`MufasaButton`. No error propagation, no progress, no cancel,
   no join — the button silently "finishes" while work continues in
   the background.
2. ``multiprocessing.Process(target=fn).start()`` where the fork
   semantics re-import the world (because ``spawn`` has been forced
   module-side in many files — wasted on Linux).
3. Worst: :file:`heatmap_clf_pop_up.py:192-193` has
   ``multiprocessing.Process(heatmapper_clf.run())`` — **with parens**
   — so the work executes in the GUI thread and a zombie worker with
   ``target=None`` is spawned. Freezes the UI for the duration of
   generation, which can be minutes.

This module replaces all three with one right-sized pattern.

**Design**

* :class:`ProcessorRunner` is a :class:`QThread`. The backend work runs
  in the Qt worker thread, **not** a separate process. SimBA's backend
  classes (``HeatMapperClfMultiprocess``, etc.) already spawn their
  own worker pools internally — wrapping them in *another* process
  layer is double-taxation on Linux. On Linux-only, a single
  QThread-in-GUI-process + N worker-pool-processes (fork'd on demand
  by the backend) is the right shape.
* Cancel is cooperative via :class:`threading.Event`. The runner sets
  it; the backend checks ``runner.cancel_event.is_set()`` at loop
  heads. Backends that don't support cancel will simply finish and
  ignore it.
* Progress is reported via Qt signals (cross-thread-safe by design).
  Progress percentage is optional — backends that can't estimate just
  don't call ``report_progress``.
* Errors are **always** captured and re-emitted on the GUI thread via
  the ``errored`` signal. Nothing raises silently into a thread where
  it's swallowed.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal


class ProcessorRunner(QThread):
    """Run a callable in a Qt worker thread, signalling progress.

    Typical usage::

        runner = ProcessorRunner(target=my_processor.run)
        runner.progress.connect(progress_bar.setValue)
        runner.finished.connect(lambda: status.setText("Done"))
        runner.errored.connect(lambda e: QMessageBox.critical(None, "Error", str(e)))
        runner.start()
        # Later: user clicks "Cancel" →
        runner.request_cancel()
    """

    # --- Qt signals ------------------------------------------------------ #
    #: Emitted as ``(percent,)`` where percent is 0..100. Backends that
    #: don't track progress simply don't call ``report_progress``.
    progress = Signal(int)
    #: Emitted as ``(message,)`` for status updates ("Rendering frame 42/300…").
    status = Signal(str)
    #: Emitted once the callable returns normally. No payload.
    completed = Signal()
    #: Emitted on exception. Payload is the ``Exception`` instance.
    errored = Signal(Exception)

    def __init__(
        self,
        target: Callable[..., Any],
        args: tuple = (),
        kwargs: Optional[dict] = None,
        *,
        inject_runner_kw: Optional[str] = None,
    ) -> None:
        """
        :param target: the callable (usually ``some_processor.run``).
        :param args, kwargs: passed to the target.
        :param inject_runner_kw: if set (e.g. ``"runner"``), the runner
            itself is passed as that keyword — the target can then call
            ``runner.report_progress(...)`` and check
            ``runner.cancel_event.is_set()`` mid-loop.
        """
        super().__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self._inject_kw = inject_runner_kw
        self.cancel_event = threading.Event()
        # Populated post-run for tests / debugging.
        self.result: Any = None
        self.exception: Optional[BaseException] = None
        self.wall_time_s: Optional[float] = None

    # ----- control ------------------------------------------------------ #
    def request_cancel(self) -> None:
        """Cooperative cancel. Backends that don't honour the event will
        still finish to completion; the signal is a request, not a kill."""
        self.cancel_event.set()
        self.status.emit("Cancelling…")

    # ----- progress API for targets to call ----------------------------- #
    def report_progress(self, percent: int) -> None:
        self.progress.emit(max(0, min(100, int(percent))))

    def report_status(self, msg: str) -> None:
        self.status.emit(str(msg))

    # ----- Qt entry point ---------------------------------------------- #
    def run(self) -> None:  # noqa: D401 (Qt signature)
        started = time.monotonic()
        kwargs = dict(self._kwargs)
        if self._inject_kw:
            kwargs[self._inject_kw] = self
        try:
            self.result = self._target(*self._args, **kwargs)
            self.wall_time_s = time.monotonic() - started
            self.completed.emit()
        except BaseException as exc:  # catch everything — we re-emit
            self.wall_time_s = time.monotonic() - started
            self.exception = exc
            # Log the traceback before emitting the signal — the signal
            # handler may re-raise or suppress but the stack trace
            # belongs in the log regardless.
            traceback.print_exc()
            if isinstance(exc, Exception):
                self.errored.emit(exc)
            else:
                # KeyboardInterrupt / SystemExit — wrap so the signal
                # type-checks, but preserve the original
                self.errored.emit(RuntimeError(f"{type(exc).__name__}: {exc}"))


__all__ = ["ProcessorRunner"]
