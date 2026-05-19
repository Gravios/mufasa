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

Patch 122ch introduced this helper to break that coupling. The
backend files import :func:`confirm_two_option` from this
module instead of pulling in Tk. Qt code (or tests, or headless
callers) installs an override by reassigning the module-level
attribute.

The Tk popup that was originally the default implementation is
gone (deleted in 122d6 Stage C). The default now does:

1. Try a stdin prompt — works for CLI / CI / interactive shells.
2. If stdin is unavailable (no controlling terminal), default to
   ``option_one`` — the typical "YES / SKIP / CONTINUE" choice.

Patch 122dd updates this docstring to match the post-Stage-C
behaviour and removes the now-stale "Tk default" framing. The
old Tk-import-with-ImportError-fallback code path still exists
in ``_default_confirm`` as a no-op safety net (the ImportError
branch always fires post-Stage-C, since the module is gone) —
keeping it for diff-stability rather than rewriting the function
body. The behaviour is identical either way.

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

The Qt override **IS now installed at workbench startup** (patch
122dd) — see ``mufasa/ui_qt/qt_confirm_override.py``. When a
backend function calls ``confirm_two_option`` from a workbench
session, the user sees a QMessageBox dialog. CLI / headless
callers still fall through to the stdin path.

**Tests / scripts** (auto-confirm without prompting):
::

    import mufasa.utils.confirm as _cf
    _cf.confirm_two_option = lambda **_: "YES"

Default behaviour (post-Stage-C)
--------------------------------
1. If a Qt override has been installed, the override handles it.
2. Otherwise, prompt via stdin. Useful for CLI / CI contexts.
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
    """Default confirmation. Stdin if available; auto-yes if not.

    Post-Stage-C (122d6), the lazy `from mufasa.ui.tkinter_functions
    import …` always raises ImportError (the Tk module is gone), so
    this function effectively always routes to `_stdin_confirm`.
    The try/except block is kept for diff-stability — the behaviour
    is identical with or without it.
    """
    # Lazy import retained for diff-stability. The except ImportError
    # branch is now the de-facto only path post-Stage-C (the Tk
    # tkinter_functions module was deleted in 122d6). Behaviour is
    # unchanged from the user's perspective.
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
