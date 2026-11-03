"""Smoke test for patch 122ih — no blocking progress dialog.

Running an operation used to open a window-modal QProgressDialog that froze
the whole workbench, including the console dock showing that run's own output
(Kalman v2 EM iterations), so it couldn't be scrolled until the run ended.
Progress now shows inline in the main window's status bar (label, bar,
Cancel); nothing is modal. Because of that the form's Run button is disabled
while its operation is in flight. A parent outside a QMainWindow gets a
NON-modal dialog.

Runs for real: an OperationForm inside a QMainWindow with Qt offscreen, a
worker thread that blocks until released, the real run_with_progress.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QWidget,
)

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])
seen: list[str] = []
QMessageBox.information = staticmethod(lambda *a, **k: seen.append("info"))
QMessageBox.critical = staticmethod(lambda *a, **k: seen.append("critical"))

from mufasa.ui_qt.progress import StatusProgress, run_with_progress  # noqa: E402
from mufasa.ui_qt.workbench import OperationForm  # noqa: E402

release = threading.Event()
started = threading.Event()


class Slow(OperationForm):
    title = "Slow op"

    def build(self) -> None:
        pass

    def target(self) -> None:
        started.set()
        release.wait(10)


def pump(cond, timeout=10.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


win = QMainWindow()
form = Slow()
win.setCentralWidget(form)
win.show()
win.statusBar()

form.on_run()
pump(started.is_set)
inds = win.statusBar().findChildren(StatusProgress)
check("running shows an inline indicator in the status bar", len(inds) == 1)
check("no progress dialog is opened",
      not [w for w in app.topLevelWidgets()
           if isinstance(w, QProgressDialog) and w.isVisible()])
check("nothing is modal while it runs (console etc. stay usable)",
      app.activeModalWidget() is None)
check("the window itself still accepts input", win.isEnabled() and form.isEnabled())
check("Run is disabled while the operation is in flight",
      not form.run_btn.isEnabled())
ind = inds[0] if inds else None
check("indicator names the operation",
      ind is not None and ind.label.text().startswith("Slow op"))

release.set()
pump(lambda: form.run_btn.isEnabled())
check("Run is re-enabled when it finishes", form.run_btn.isEnabled()
      and "Run" in form.run_btn.text())
pump(lambda: not win.statusBar().findChildren(StatusProgress))
check("the indicator is removed when it finishes",
      not [w for w in win.statusBar().findChildren(StatusProgress)
           if not w.isHidden()])
check("success is still reported", "info" in seen)

# Progress + cancel through the indicator.
gate = threading.Event()


def _work(runner):
    runner.report_status("iteration 3/20")
    runner.report_progress(40)
    gate.wait(10)
    if runner.cancel_event.is_set():
        raise RuntimeError("cancelled")


errs: list[Exception] = []
r = run_with_progress(win, title="EM…", target=_work,
                      inject_runner_kw="runner", on_error=errs.append)
pump(lambda: win.statusBar().findChildren(StatusProgress)
     and win.statusBar().findChildren(StatusProgress)[-1].bar.value() == 40)
ind2 = win.statusBar().findChildren(StatusProgress)[-1]
check("progress reports drive the bar", ind2.bar.value() == 40)
check("status reports drive the label",
      "iteration 3/20" in ind2.label.text())
ind2.cancel_btn.click()
check("Cancel requests a cooperative cancel", r.cancel_event.is_set()
      and not ind2.cancel_btn.isEnabled())
gate.set()
pump(lambda: bool(errs))
check("a cancelled run reaches on_error", len(errs) == 1)
r.wait(2000)

# Fallback: parent outside a QMainWindow -> non-modal dialog.
lone = QWidget()
lone.show()
done = threading.Event()
hold = threading.Event()
r2 = run_with_progress(lone, title="Lone…", target=lambda: hold.wait(10),
                       on_success=done.set)
dlgs = [w for w in lone.findChildren(QProgressDialog)]
check("outside a main window a dialog is used, and it is NON-modal",
      len(dlgs) == 1 and not dlgs[0].isModal()
      and app.activeModalWidget() is None)
hold.set()
pump(done.is_set)
r2.wait(2000)
check("the fallback still completes", done.is_set())

src = (REPO / "mufasa/ui_qt/progress.py").read_text()
check("no window-modal progress anywhere",
      "Qt.WindowModal" not in src and "ApplicationModal" not in src)
for rel in ("mufasa/ui_qt/forms/features.py",
            "mufasa/ui_qt/forms/video_filters.py"):
    check(f"{Path(rel).name} locks its Run button too",
          "on_finished=self._lock_run_button()" in (REPO / rel).read_text())

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ih_inline_progress: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
