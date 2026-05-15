"""
tests/smoke_preprocessing_tab_moves.py
======================================

Patch 122x: regression guard for the move of two sections
from the Data Import page to the Preprocessing page, with
rename:

* ``Video parameters & calibration`` → ``Video Calibration``
* ``Batch pre-process videos`` → ``Preprocess Videos``

Coverage:

1. Data Import page no longer registers either renamed section
   (matches the exact ``add_section('<old or new>', …)`` call
   form so historical references in docstrings are tolerated).
2. Data Import page no longer imports
   :class:`BatchPreProcessLauncher` or :class:`VideoInfoForm`.
3. Preprocessing page registers ``Video Calibration`` and
   ``Preprocess Videos`` (the new names).
4. Preprocessing page imports both required forms.
5. Both renames appear in the Preprocessing page docstring.
"""
from __future__ import annotations

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
    dip_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
               / "data_import_page.py").read_text()
    ppp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
               / "pose_cleanup_page.py").read_text()

    # ---------- Data Import page should no longer register them ----------
    for old in ("Video parameters & calibration",
                "Batch pre-process videos"):
        check(
            f"Data Import page no longer registers "
            f"'{old}' section",
            f'add_section("{old}"' not in dip_src,
        )
    for new in ("Video Calibration", "Preprocess Videos"):
        check(
            f"Data Import page does NOT register '{new}' "
            "(should be on Preprocessing page)",
            f'add_section("{new}"' not in dip_src,
        )

    check(
        "Data Import page no longer imports BatchPreProcessLauncher",
        "BatchPreProcessLauncher" not in dip_src,
    )
    check(
        "Data Import page no longer imports VideoInfoForm",
        "VideoInfoForm" not in dip_src,
    )

    # ---------- Preprocessing page should now register them ----------
    check(
        "Preprocessing page registers 'Video Calibration' section",
        'add_section("Video Calibration"' in ppp_src,
    )
    check(
        "Preprocessing page registers 'Preprocess Videos' section",
        'add_section("Preprocess Videos"' in ppp_src,
    )
    check(
        "Preprocessing page imports VideoInfoForm",
        "VideoInfoForm" in ppp_src,
    )
    check(
        "Preprocessing page imports BatchPreProcessForm "
        "(was BatchPreProcessLauncher; renamed during the "
        "Qt port — class is now BatchPreProcessForm)",
        "BatchPreProcessForm" in ppp_src,
    )

    # ---------- Docstring updates ----------
    check(
        "Preprocessing page docstring records the 122x move",
        "122x" in ppp_src,
    )
    check(
        "Preprocessing page docstring mentions 'Video Calibration'",
        "Video Calibration" in ppp_src,
    )
    check(
        "Preprocessing page docstring mentions 'Preprocess Videos'",
        "Preprocess Videos" in ppp_src,
    )
    check(
        "Data Import page docstring records the 122x move",
        "122x" in dip_src,
    )

    print(
        f"smoke_preprocessing_tab_moves: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
