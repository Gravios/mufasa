"""
mufasa.ui_qt.icon_cache
======================

Module-level :class:`QIcon` / :class:`QPixmap` registry.

The legacy Tkinter path re-opens the same ~200 PNG assets (via PIL +
``ImageTk.PhotoImage``) inside **every** ``PopUpMixin.__init__``. For 105
popup classes this is both slow (disk + decode) and leaky (PhotoImages
are retained on module-level dicts).

This module replaces that pattern with a single lazy, process-wide cache:
each icon is decoded once, on first request, and the :class:`QIcon`
object is shared. ``QIcon`` is reference-counted in Qt and cheap to copy
by value, so handing the same instance to 100 buttons is fine.

**Why we don't reuse** ``mufasa.utils.lookups.get_icons_paths``:
``lookups.py`` imports ``tkinter`` at module level, which would pull Tk
into the Qt code path. We keep ``mufasa.ui_qt`` strictly tkinter-free by
re-implementing the tiny directory scan here.
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QPixmap

# Base: .../mufasa/ui_qt/icon_cache.py -> .../mufasa/
_SIMBA_ROOT = Path(__file__).resolve().parent.parent
_ICON_DIR = _SIMBA_ROOT / "assets" / "icons"
_TOOLTIPS_JSON = _SIMBA_ROOT / "assets" / "lookups" / "tooltips.json"


@lru_cache(maxsize=1)
def _icon_map() -> dict[str, Path]:
    """Scan ``mufasa/assets/icons/*.png`` once; return name→path dict."""
    if not _ICON_DIR.is_dir():
        return {}
    return {Path(p).stem: Path(p) for p in glob.glob(str(_ICON_DIR / "*.png"))}


@lru_cache(maxsize=1)
def _tooltips() -> dict[str, str]:
    """Load tooltip strings from the same JSON the Tk path uses."""
    if not _TOOLTIPS_JSON.is_file():
        return {}
    import json
    try:
        return json.loads(_TOOLTIPS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=None)
def icon(name: str) -> QIcon:
    """Return a cached :class:`QIcon` for a logical name (e.g. ``"rocket"``).

    If the name is unknown *or* the file cannot be read, an empty ``QIcon``
    is returned. Callers never have to ``None``-check — they can always
    call ``QPushButton(icon(name), ...)``.
    """
    path = _icon_map().get(name)
    if path is None or not path.is_file():
        return QIcon()
    return QIcon(str(path))


@lru_cache(maxsize=None)
def pixmap(name: str, size: Optional[tuple[int, int]] = None) -> QPixmap:
    """Return a cached :class:`QPixmap` for a logical name."""
    path = _icon_map().get(name)
    if path is None or not path.is_file():
        return QPixmap()
    pm = QPixmap(str(path))
    if size is not None and not pm.isNull():
        pm = pm.scaled(QSize(*size))
    return pm


def tooltip(key: str) -> Optional[str]:
    """Look up a tooltip string by key, or ``None`` if missing."""
    return _tooltips().get(key)


def icon_from_path(path: str | os.PathLike) -> QIcon:
    """Build a ``QIcon`` from an absolute path. Not cached — callers
    needing caching should use :func:`icon` with a logical name."""
    p = Path(path)
    if not p.is_file():
        return QIcon()
    return QIcon(str(p))
