"""Smoke test for patch 122it — Mufasa app icon and startup splash.

The lion logo (mufasa/assets/branding/) becomes the application / window
icon (replacing the SimBA logo) and the wordmark version is shown on a
splash while the workbench is built.

Runs for real: the branding module offscreen, a real workbench window, and
the workbench entry point's wiring.
"""
from __future__ import annotations

import os
import sys
import tomllib
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

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])
from mufasa.ui_qt import branding as B  # noqa: E402

files = {p.name for p in B.BRANDING_DIR.iterdir()}
check("branding assets are in the package",
      {f"mufasa_icon_{s}.png" for s in B.ICON_SIZES} | {
          "mufasa_icon.svg", "mufasa_icon.ico", "mufasa_splash.png",
          "mufasa_wordmark.svg"} <= files)
img = QImage(str(B.BRANDING_DIR / "mufasa_icon_256.png"))
check("the icon PNGs have a transparent background",
      img.hasAlphaChannel() and img.pixelColor(0, 0).alpha() == 0
      and img.pixelColor(128, 128).alpha() == 255)
pkg = tomllib.loads((REPO / "pyproject.toml").read_text())
pd_ = pkg["tool"]["setuptools"]["package-data"]
check("assets/** (incl. branding) is shipped as package data",
      any("assets/**/*" in v for v in pd_.values()))

ic = B.app_icon()
check("app_icon carries every rendered size",
      not ic.isNull()
      and {(s.width(), s.height()) for s in ic.availableSizes()}
      == {(s, s) for s in B.ICON_SIZES})

pm = B.splash_pixmap()
check("splash pixmap renders (logo + wordmark on an opaque card)",
      not pm.isNull() and pm.width() >= 420
      and pm.toImage().pixelColor(2, 2).alpha() == 255)
os.environ.pop("MUFASA_NO_SPLASH", None)
check("no splash on the offscreen platform (headless/tests)",
      B.show_splash(app) is None)
s = B.MufasaSplash()
s.show()
s.message("Opening project…")
check("the splash widget shows and takes status messages",
      s.isVisible() and s.message is not None)
s.close()

from mufasa.ui_qt.workbench_app import build_workbench  # noqa: E402

wb = build_workbench(project_config_path=None)
check("the workbench window uses the lion icon (not the SimBA logo)",
      {(z.width(), z.height()) for z in wb.windowIcon().availableSizes()}
      == {(s, s) for s in B.ICON_SIZES})
wb.close()

src = (REPO / "mufasa/ui_qt/workbench_app.py").read_text()
main_src = src[src.index("def main("):]
check("main() sets the app icon and shows / finishes the splash",
      "app.setWindowIcon(app_icon())" in main_src
      and main_src.index("show_splash(app)") < main_src.index("build_workbench(")
      and "splash.finish(wb)" in main_src)
from mufasa.utils.enums import Paths  # noqa: E402

check("legacy icon paths point at the new icon files",
      (REPO / "mufasa" / Paths.LOGO_ICON_WINDOWS_PATH.value).is_file()
      and (REPO / "mufasa" / Paths.LOGO_ICON_DARWIN_PATH.value).is_file()
      and "branding" in str(Paths.LOGO_ICON_WINDOWS_PATH.value))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122it_branding: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
