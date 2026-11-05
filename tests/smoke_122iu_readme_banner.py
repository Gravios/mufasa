"""Smoke test for patch 122iu — README / docs banner.

The GitHub README and the docs index open with the Mufasa banner: a wide
version (2560x800) and a stacked version for narrow screens (1200x820), each
in dark and light, picked by ``<picture>`` media queries. Checks the files
exist with those sizes, every referenced path resolves, and the narrow-screen
sources come before the wide ones (the first matching ``<source>`` wins).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
IMG = REPO / "docs" / "images"
checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


sizes = {"mufasa_banner_dark": (2560, 800), "mufasa_banner_light": (2560, 800),
         "mufasa_banner_mobile_dark": (1200, 820),
         "mufasa_banner_mobile_light": (1200, 820),
         "mufasa_social_preview": (1280, 640)}
for stem, wh in sizes.items():
    png, svg = IMG / f"{stem}.png", IMG / f"{stem}.svg"
    check(f"{stem}: png {wh[0]}x{wh[1]} and svg source present",
          png.is_file() and svg.is_file() and Image.open(png).size == wh)

for doc, base in [(REPO / "README.md", REPO), (REPO / "docs" / "README.md", REPO / "docs")]:
    text = doc.read_text()
    pic = text[: text.find("</picture>")]
    refs = re.findall(r'(?:srcset|src)="([^"]+)"', pic)
    check(f"{doc.relative_to(REPO)} opens with the banner <picture>",
          text.lstrip().startswith("<p align=\"center\">") and len(refs) == 4)
    check(f"{doc.relative_to(REPO)}: every banner path resolves",
          all((base / r).is_file() for r in refs))
    check(f"{doc.relative_to(REPO)}: mobile sources precede the wide ones",
          [("mobile" in r) for r in refs] == [True, True, False, False])

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122iu_readme_banner: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
