"""
tests/smoke_122dd_cosmetic_polish.py
======================================

Patch 122dd: bundle of cosmetic / documentation cleanup items
following the substantial work in 122d7→122dc.

What this patch covers
----------------------
1. `mufasa/utils/confirm.py` module docstring updated to reflect
   post-Stage-C reality (Tk module is gone; Qt override is
   installed at workbench startup per 122cj). `_default_confirm`
   docstring + inline comment updated to match.
2. `mufasa/mixins/network_mixin.py` annotated with a module-status
   note: zero internal callers, pre-existing dead code BEFORE
   the cascade, kept as a library-API building block.
3. `docs/simba_death_cascade.md` corrected:
   - §1 (px_to_mm_ui) no longer claims this is a Tk file (it's
     cv2-based; pure OpenCV, no Tk imports).
   - §2 (confirm.py) no longer claims the fallback is broken
     (it's intentional headless behaviour).
4. `README.md` enhanced with a small "Running Mufasa" + "Project
   layout" section in the Mufasa preamble before the historical
   SimBA README.

Coverage
--------
1.  confirm.py docstring no longer says "Tk popup is still the
    default implementation" (now post-Stage-C accurate).
2.  confirm.py docstring acknowledges 122cj's Qt override install.
3.  confirm.py `_default_confirm` docstring says "Stdin if
    available; auto-yes if not" (not the stale "Tk-backed").
4.  network_mixin.py has a module-status note mentioning 122dd.
5.  network_mixin.py status note classifies the file as "library-
    API building block" (not "dead code").
6.  Cascade doc no longer claims `px_to_mm_ui.py` is Tk.
7.  Cascade doc explicitly notes the cv2-based reality.
8.  Cascade doc §2 (confirm.py) corrected — calls out 122cj's
    Qt override installation.
9.  README has a Mufasa preamble section about running.
10. README mentions that `mufasa-tk` was removed.
11. README documents both project layouts (v1 + legacy).
12. README points readers at `project_paths_from_config`.
13. All mufasa/**/*.py parse cleanly.
"""
from __future__ import annotations

import ast
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


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # --- confirm.py ---
    cf_src = (pkg / "utils" / "confirm.py").read_text()
    check(
        "confirm.py docstring no longer claims Tk is the default "
        "(stale post-Stage-C)",
        "Tk popup is still the default implementation"
        not in cf_src,
    )
    check(
        "confirm.py docstring acknowledges 122cj Qt-override "
        "installation",
        "Qt override **IS now installed at workbench startup**"
        in cf_src or "patch 122cj" in cf_src.lower()
        or "patch\n122dd" in cf_src.lower(),
    )
    check(
        "confirm.py _default_confirm docstring updated to "
        "'Stdin if available; auto-yes if not'",
        "Stdin if available; auto-yes if not" in cf_src,
    )

    # --- network_mixin.py ---
    nm_src = (pkg / "mixins" / "network_mixin.py").read_text()
    check(
        "network_mixin.py has 122dd module-status block",
        "122dd" in nm_src,
    )
    check(
        "network_mixin.py classified as library-API building block",
        "library-API building block" in nm_src
        or "library API building block" in nm_src,
    )

    # --- cascade doc ---
    cascade = (REPO_ROOT / "docs"
               / "simba_death_cascade.md").read_text()
    check(
        "Cascade doc no longer calls px_to_mm_ui a Tk file in §1",
        "cv2-based" in cascade and "px_to_mm_ui" in cascade,
    )
    check(
        "Cascade doc §1 explicitly mentions OpenCV / cv2 APIs",
        "cv2.namedWindow" in cascade
        or "cv2.imshow" in cascade,
    )
    check(
        "Cascade doc §2 (confirm.py) corrected — mentions 122cj "
        "Qt override",
        "patch 122cj" in cascade
        and "qt_confirm.py" in cascade.lower(),
    )

    # --- README ---
    readme = (REPO_ROOT / "README.md").read_text()
    check(
        "README has a Running Mufasa section in the preamble",
        "Running Mufasa" in readme
        or "## Running" in readme,
    )
    check(
        "README mentions mufasa-tk removal",
        "mufasa-tk" in readme
        and "Removed in patch 122d4" in readme,
    )
    check(
        "README documents both project layouts (v1 + legacy)",
        ("Legacy SimBA layout" in readme
         or "legacy SimBA" in readme.lower())
        and "v1 layout" in readme,
    )
    check(
        "README points at project_paths_from_config",
        "project_paths_from_config" in readme,
    )

    # --- Parse-clean ---
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122dd_cosmetic_polish: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
