"""
tests/smoke_text_color_convention.py
====================================

Patch 122j: regression guard for the muted-text colour convention.

The wrong patterns:

* ``color: palette(mid)`` — ``mid`` is a structural UI role
  (grooves, borders), not a text role. On Ubuntu (Yaru / Adwaita
  / Breeze) and most other themes it reads as illegible light-grey
  on light-grey or dark-grey on dark-grey.
* ``color: #666`` / ``#888`` / ``#555`` and similar hardcoded
  greys — fixed values that don't adapt to dark mode. The text
  becomes either way too low-contrast (dark grey on dark bg) or
  the rest of the surface clashes (light grey on dark bg, while
  body text is light).

The right pattern for "secondary text, still readable":

* ``color: palette(placeholder-text)`` — theme-aware. Standard
  Qt palette role since 5.12; recognised by Yaru / Adwaita /
  Breeze / Windows / macOS native themes.
* Or just don't set a colour at all — inherits ``palette(text)``
  which is the highest-contrast text colour. Pair with a size /
  weight cue for hierarchy.

Semantic colours (saturated red / orange / green for errors,
warnings, success) are intentional and unaffected by this rule;
they have adequate contrast against both light and dark
backgrounds.

This guard fails if any file under ``mufasa/ui_qt/`` reintroduces
the bad pattern outside the documented semantic-colour set.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


# Allowed semantic colours — saturated, readable on both light
# and dark backgrounds. Anything else hardcoded is suspect.
SEMANTIC_COLORS = {
    "#c44",     # error red
    "#a86400",  # warning orange
    "#5a8f5a",  # success green
}

# Matches "color: <something>" inside a Python string literal
# (this is what setStyleSheet calls look like in source).
# Negative lookbehind for `-` prevents accidental matches on
# background-color, border-color, etc. — we only care about
# foreground text colour.
COLOR_RE = re.compile(
    r"""(?<!-)\bcolor\s*:\s*([^\s;\"']+)""",
    re.IGNORECASE,
)


def main() -> int:
    ui_qt_root = REPO_ROOT / "mufasa" / "ui_qt"
    bad_mid: list[tuple[Path, int, str]] = []
    bad_hardcoded: list[tuple[Path, int, str]] = []

    for path in ui_qt_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            # Skip comments. We comment about palette(mid) in
            # project_info.py's docstring deliberately.
            if stripped.startswith("#"):
                continue
            for m in COLOR_RE.finditer(line):
                color_value = m.group(1).rstrip(",;").strip()
                # palette(mid) used as a text colour — wrong
                if color_value.lower() == "palette(mid)":
                    bad_mid.append((path, lineno, line.strip()))
                    continue
                # Hardcoded hex
                if color_value.startswith("#"):
                    hexv = color_value.lower()
                    # 3-digit and 6-digit hex; case-insensitive
                    # match against semantic set
                    if hexv in {c.lower() for c in SEMANTIC_COLORS}:
                        continue
                    # Pure-grey hex codes (rr=gg=bb in 3 or 6
                    # digit form) used as text colours are the
                    # main offender — flag them
                    body = hexv.lstrip("#")
                    if len(body) == 3 and len(set(body)) == 1:
                        bad_hardcoded.append(
                            (path, lineno, line.strip()),
                        )
                    elif len(body) == 6 and body[0:2] == body[2:4] == body[4:6]:
                        bad_hardcoded.append(
                            (path, lineno, line.strip()),
                        )

    check(
        "no palette(mid) used as a text colour under mufasa/ui_qt/",
        len(bad_mid) == 0,
        detail=(
            f"{len(bad_mid)} occurrence(s); first: "
            f"{bad_mid[0] if bad_mid else '-'}"
        ),
    )
    check(
        "no hardcoded grey hex codes used as a text colour "
        "under mufasa/ui_qt/ (semantic warning / error / ok "
        "colours excepted)",
        len(bad_hardcoded) == 0,
        detail=(
            f"{len(bad_hardcoded)} occurrence(s); first: "
            f"{bad_hardcoded[0] if bad_hardcoded else '-'}"
        ),
    )

    # Positive assertion: the convention is in use somewhere.
    # Without this the test would silently pass on a codebase that
    # simply has no styled text at all.
    placeholder_count = 0
    for path in ui_qt_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            placeholder_count += path.read_text().count(
                "palette(placeholder-text)",
            )
        except (OSError, UnicodeDecodeError):
            continue
    check(
        "palette(placeholder-text) is used (the convention is "
        "in force, not silently absent)",
        placeholder_count > 0,
        detail=f"got {placeholder_count} occurrences",
    )

    # Spot-check: forms / pages the user has recently surfaced
    # should use the convention.
    must_use = (
        "mufasa/ui_qt/forms/project_info.py",
        "mufasa/ui_qt/forms/pose_import.py",
        "mufasa/ui_qt/workbench.py",
    )
    for rel in must_use:
        path = REPO_ROOT / rel
        try:
            text = path.read_text()
        except OSError:
            check(
                f"{rel} is readable", False,
                detail="file missing",
            )
            continue
        check(
            f"{rel} uses palette(placeholder-text)",
            "palette(placeholder-text)" in text,
        )

    print(
        f"smoke_text_color_convention: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
