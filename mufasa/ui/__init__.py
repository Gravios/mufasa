"""
mufasa.ui — legacy Tkinter UI surface (deprecated)

This package contains the legacy Tk-based launcher widgets,
popups, and forms. The active UI is the Qt workbench under
``mufasa.ui_qt``. New work should target Qt, not Tk.

See ``docs/tk_surface_audit.md`` for the per-file removal plan.
"""
import warnings as _warnings

_warnings.warn(
    "mufasa.ui (the Tkinter UI surface) is deprecated. Use "
    "mufasa.ui_qt instead. See docs/tk_surface_audit.md for the "
    "removal plan.",
    DeprecationWarning,
    stacklevel=2,
)
