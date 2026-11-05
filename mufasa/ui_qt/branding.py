"""
mufasa.ui_qt.branding
=====================

Mufasa's app icon and startup splash (patch 122it).

Assets live in ``mufasa/assets/branding/``:

* ``mufasa_icon_<size>.png`` (16–512 px), ``mufasa_icon.svg``,
  ``mufasa_icon.ico`` — the lion-head symbol, used as the application /
  window icon;
* ``mufasa_splash.png`` / ``mufasa_wordmark.svg`` — the symbol with the
  MUFASA wordmark, shown on the startup splash.

The splash is drawn on an opaque dark card rather than relying on window
translucency, which needs a compositor that remote X11 sessions often lack.
Set ``MUFASA_NO_SPLASH=1`` to skip it; it is also skipped on the offscreen /
minimal Qt platforms.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

BRANDING_DIR = Path(__file__).resolve().parent.parent / "assets" / "branding"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
SPLASH_IMAGE = BRANDING_DIR / "mufasa_splash.png"
_CARD = QColor(29, 31, 36)
_TEXT = QColor(205, 170, 120)


def app_icon() -> QIcon:
    """The lion icon with every rendered size, so each context (title bar,
    taskbar, alt-tab, dock) gets a crisp image."""
    ic = QIcon()
    for s in ICON_SIZES:
        p = BRANDING_DIR / f"mufasa_icon_{s}.png"
        if p.is_file():
            ic.addFile(str(p))
    return ic


def splash_pixmap(width: int = 420) -> QPixmap:
    """The splash card: dark background, the gold logo with wordmark, room
    for a one-line status message at the bottom. ``width`` is in device-
    independent pixels; the pixmap is rendered at the screen's pixel ratio."""
    dpr = 1.0
    screen = QGuiApplication.primaryScreen() if QGuiApplication.instance() else None
    if screen is not None:
        dpr = max(1.0, float(screen.devicePixelRatio()))
    logo = QPixmap(str(SPLASH_IMAGE))
    margin, footer = 36, 44
    logo_w = width - 2 * margin
    logo_h = round(logo_w * logo.height() / logo.width()) if not logo.isNull() else 0
    height = margin + logo_h + footer
    pm = QPixmap(round(width * dpr), round(height * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(_CARD)
    if not logo.isNull():
        p = QPainter(pm)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.drawPixmap(QRectF(margin, margin, logo_w, logo_h), logo,
                     QRectF(logo.rect()))
        p.end()
    return pm


class MufasaSplash(QSplashScreen):
    """Frameless splash that stays on top while the workbench is built."""

    def __init__(self) -> None:
        super().__init__(splash_pixmap(), Qt.WindowStaysOnTopHint)
        f = QFont(self.font())
        f.setPointSizeF(max(8.0, f.pointSizeF() - 0.5))
        self.setFont(f)

    def message(self, text: str) -> None:
        self.showMessage(text, Qt.AlignHCenter | Qt.AlignBottom, _TEXT)
        QApplication.processEvents()


def show_splash(app: QApplication) -> MufasaSplash | None:
    """Show the splash (unless disabled or on a headless platform)."""
    if os.environ.get("MUFASA_NO_SPLASH", "").strip() not in ("", "0"):
        return None
    if app.platformName() in ("offscreen", "minimal"):
        return None
    splash = MufasaSplash()
    splash.show()
    splash.message("Starting…")
    return splash


__all__ = ["BRANDING_DIR", "ICON_SIZES", "SPLASH_IMAGE", "MufasaSplash",
           "app_icon", "show_splash", "splash_pixmap"]
