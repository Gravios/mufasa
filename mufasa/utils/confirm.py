"""
mufasa.utils.confirm
====================

UI-agnostic confirmation helper. Backend modules call
:func:`confirm_two_option` to ask the user a binary question
during a long-running operation; the answer is returned as a
string matching one of the two option labels.

Background
----------
Several backend files (``video_processors/video_processing.py``,
``mixins/train_model_mixin.py``) used to import
``TwoOptionQuestionPopUp`` from the Tk surface directly. That
made those files Tk-coupled even though they're fundamental
backend code with no reason to know about the UI.

Patch 122ch introduces this helper to break that coupling. The
backend files now import :func:`confirm_two_option` from this
module instead of pulling in Tk. The Tk popup is still the
default implementation (lazy-imported only when needed), but Qt
code (or tests, or headless callers) can swap in a different
implementation by reassigning the module-level attribute.

Override patterns
-----------------
**Qt workbench startup** (recommended; applies project-wide):
::

    import mufasa.utils.confirm as _cf

    def _qt_confirm(question, option_one="YES", option_two="NO",
                    title=None):
        from PySide6.QtWidgets import QMessageBox
        btn = QMessageBox.question(
            None, title or "Confirm", question,
            QMessageBox.Yes | QMessageBox.No,
        )
        return option_one if btn == QMessageBox.Yes else option_two

    _cf.confirm_two_option = _qt_confirm

**Tests / scripts** (auto-confirm without prompting):
::

    import mufasa.utils.confirm as _cf
    _cf.confirm_two_option = lambda **_: "YES"

The Qt-side override is **not** installed by this module. Patch
122ch is backend-only; the Qt workbench falls back to the Tk
default until/unless a separate patch wires the Qt override at
workbench startup.

Default behaviour
-----------------
1. Try to lazy-import the Tk popup. If available, open it and
   return the user's choice.
2. If Tk isn't installed (headless / minimal environment), fall
   back to a stdin prompt. Useful for CLI / CI contexts.
3. If stdin is unavailable (no controlling terminal), default to
   ``option_one`` — the typical "YES / SKIP / CONTINUE" choice.
"""
from __future__ import annotations

import sys
from typing import Optional


def _default_confirm(question: str,
                     option_one: str = "YES",
                     option_two: str = "NO",
                     title: Optional[str] = None) -> str:
    """Default Tk-backed confirmation. Falls back to stdin /
    auto-yes if Tk isn't usable.
    """
    # Lazy import: callers don't pay the Tk dependency cost at
    # module load. The import happens only when this default is
    # actually invoked AND a Qt override hasn't been installed.
    try:
        from mufasa.ui.tkinter_functions import (
            TwoOptionQuestionPopUp,
        )
    except ImportError:
        return _stdin_confirm(question, option_one, option_two,
                              title)
    try:
        popup = TwoOptionQuestionPopUp(
            question=question,
            option_one=option_one,
            option_two=option_two,
            title=title or "Confirm",
        )
        choice = getattr(popup, "selected_option", None)
        # Tk popup might fail to render (e.g., no display);
        # fall back to stdin in that case.
        if choice not in (option_one, option_two):
            return _stdin_confirm(question, option_one,
                                  option_two, title)
        return choice
    except Exception:
        return _stdin_confirm(question, option_one,
                              option_two, title)


def _stdin_confirm(question: str,
                   option_one: str,
                   option_two: str,
                   title: Optional[str]) -> str:
    """Stdin fallback. Returns ``option_one`` if no input or
    if input is unavailable.
    """
    msg = f"\n{title or 'CONFIRM'}: {question}\n"
    msg += f"  [1] {option_one}\n  [2] {option_two}\n"
    msg += f"Choose [1/2] (default {option_one}): "
    sys.stdout.write(msg)
    sys.stdout.flush()
    try:
        ans = input().strip()
    except (EOFError, KeyboardInterrupt, OSError):
        ans = ""
    if ans == "2":
        return option_two
    # "1", empty, or anything else → default to option_one
    return option_one


# Public binding. Qt code overrides this attribute at workbench
# startup; tests can patch it to a constant lambda.
confirm_two_option = _default_confirm
