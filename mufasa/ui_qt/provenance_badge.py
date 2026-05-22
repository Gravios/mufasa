"""mufasa/ui_qt/provenance_badge.py — section-status badge icons.

Patch 122du. Renders 16×16 ``QIcon`` instances for the three
:class:`SectionStatus` values, ready to drop onto a ``QToolBox``
section header via ``setItemIcon(index, icon)``.

Visual spec (locked in by user during 122du planning):

* **UNKNOWN** — white-filled circle with a thin gray outline.
  Indicates the section hasn't run yet (or hasn't recorded
  provenance). The outline ensures visibility on light themes
  where pure white would disappear into the toolbox header.
* **CURRENT** — green-filled circle with a white checkmark.
  Indicates the section ran AFTER every declared dependency
  that has a known timestamp.
* **STALE** — orange-filled circle (no glyph). Indicates the
  section ran BEFORE at least one dependency. The user
  should re-run it.

All three icons share the same circle geometry (radius and
center) so the rendered badges look consistent in a column of
section headers. The only differences across states are fill
color and the optional checkmark.

Colors are Tailwind-derived for accessible contrast:

* Outline gray: ``#94a3b8`` (slate-400)
* Green:        ``#16a34a`` (green-600)
* Orange:       ``#f97316`` (orange-500)
* Checkmark:    ``white`` (max contrast on green)

Implementation
==============
Icons are rendered from inline SVG strings via
:class:`PySide6.QtSvg.QSvgRenderer` onto a :class:`QPixmap`, then
wrapped in a :class:`QIcon`. The result is cached per status
value — the SVG parse + raster happens once per Python process.

Module-level Qt imports are intentionally avoided so this module
can be imported in headless contexts (tests, docs builds) without
PySide6 present. Qt names are imported lazily inside
:func:`icon_for_status`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtGui import QIcon

# Patch 122du -- import these lazily so this module can be parsed
# (e.g., by smoke tests) in environments without PySide6.

from mufasa.section_provenance import SectionStatus

# -----------------------------------------------------------------------
# SVG sources
# -----------------------------------------------------------------------
# Same 16×16 viewBox across all three so consumers can drop them
# interchangeably without re-sizing concerns. Radius = 6.5 leaves room
# for the 1-px outline on the UNKNOWN variant without pushing into the
# bounding box.

_SVG_UNKNOWN = b"""<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
  <circle cx="8" cy="8" r="6.5" fill="white" stroke="#94a3b8" stroke-width="1"/>
</svg>"""

_SVG_CURRENT = b"""<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
  <circle cx="8" cy="8" r="7" fill="#16a34a"/>
  <path d="M4.5 8.5 L7 11 L11.5 5.5" stroke="white" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>"""

_SVG_STALE = b"""<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
  <circle cx="8" cy="8" r="7" fill="#f97316"/>
</svg>"""

_SVG_FOR_STATUS = {
    SectionStatus.UNKNOWN: _SVG_UNKNOWN,
    SectionStatus.CURRENT: _SVG_CURRENT,
    SectionStatus.STALE:   _SVG_STALE,
}


@lru_cache(maxsize=None)
def icon_for_status(status: SectionStatus) -> "QIcon":
    """Return the :class:`QIcon` for ``status``.

    Renders the corresponding SVG to a 16×16 pixmap on first call;
    subsequent calls return the cached :class:`QIcon`. ``QIcon`` is
    reference-counted in Qt so handing the same instance to many
    toolbox items is fine.

    Raises ``RuntimeError`` if a :class:`QApplication` isn't running
    — QIcon construction needs the Qt event loop primed. In normal
    workbench usage the app is up before any section header builds,
    so this isn't user-visible.
    """
    # Lazy imports — keeps the module importable in PySide6-free
    # environments (tests, AST inspection).
    from PySide6.QtCore import QByteArray, QSize, Qt
    from PySide6.QtGui import QGuiApplication, QIcon, QPainter, QPixmap
    from PySide6.QtSvg import QSvgRenderer

    if QGuiApplication.instance() is None:
        raise RuntimeError(
            "icon_for_status requires a QApplication / QGuiApplication "
            "to be running"
        )

    svg = _SVG_FOR_STATUS.get(status)
    if svg is None:
        raise ValueError(f"unknown SectionStatus: {status!r}")

    renderer = QSvgRenderer(QByteArray(svg))
    pixmap = QPixmap(QSize(16, 16))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    return QIcon(pixmap)


def clear_icon_cache() -> None:
    """Clear the :func:`icon_for_status` cache.

    Useful only in tests / hot-reload scenarios where the
    :class:`QApplication` is restarted; production workflows never
    need this.
    """
    icon_for_status.cache_clear()


__all__ = ["icon_for_status", "clear_icon_cache"]
