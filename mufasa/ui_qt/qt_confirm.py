"""
mufasa.ui_qt.qt_confirm
========================

Qt implementation of the UI-agnostic confirm helper introduced in
patch 122ch (:mod:`mufasa.utils.confirm`). Patch 122cj wires this
implementation in at workbench startup so backend confirmations
(EXTRACT ALL FRAMES, META CONFIG FILE ERROR, etc.) render as
native :class:`QMessageBox` dialogs instead of the lazy-imported
Tk fallback.

Install pattern
---------------
Called from :func:`mufasa.ui_qt.workbench_app.main` after the
``QApplication`` instance exists. The install is a one-line
attribute reassignment on the public binding::

    mufasa.utils.confirm.confirm_two_option = qt_confirm_two_option

After the install, every backend call to
``confirm_two_option(...)`` routes through Qt. The Tk fallback in
:mod:`mufasa.utils.confirm._default_confirm` is never invoked
unless the override is explicitly removed (tests sometimes do
this).

Compatibility with the abstraction
----------------------------------
The Qt implementation honours the same signature the Tk default
exposes: ``(question, option_one="YES", option_two="NO",
title=None) -> str`` returning the chosen option label. Callers
don't need to know which UI surface is active.

We deliberately use ``addButton(label, AcceptRole/RejectRole)``
rather than the predefined ``QMessageBox.Yes/No`` constants —
this preserves the caller's option text (e.g., "SKIP" /
"TERMINATE" for the training-meta-config error path) rather than
forcing "Yes" / "No" everywhere.
"""
from __future__ import annotations


def qt_confirm_two_option(question: str,
                          option_one: str = "YES",
                          option_two: str = "NO",
                          title: str | None = None) -> str:
    """Qt-native confirmation dialog. Returns the chosen option's
    label (``option_one`` or ``option_two``).

    Requires a live QApplication. Safe to call from the Qt main
    thread; calling from a worker thread is undefined behaviour
    (QMessageBox needs the GUI thread).
    """
    # Lazy-import PySide6 so the module is parseable without it.
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox()
    box.setWindowTitle(title or "Confirm")
    box.setText(question)
    box.setIcon(QMessageBox.Question)

    # addButton(text, role) returns the QPushButton — used below to
    # identify which one was clicked. AcceptRole / RejectRole give
    # the dialog the right behaviour for default + Esc-cancel.
    btn_one = box.addButton(option_one, QMessageBox.AcceptRole)
    box.addButton(option_two, QMessageBox.RejectRole)
    box.setDefaultButton(btn_one)
    box.exec()
    return (option_one if box.clickedButton() is btn_one
            else option_two)


def install_qt_confirm_override() -> None:
    """Replace ``mufasa.utils.confirm.confirm_two_option`` with
    the Qt implementation. Idempotent — calling twice is safe.

    Call this once at workbench startup, after the
    ``QApplication`` is constructed.
    """
    import mufasa.utils.confirm as _cf
    _cf.confirm_two_option = qt_confirm_two_option


__all__ = ["qt_confirm_two_option", "install_qt_confirm_override"]
