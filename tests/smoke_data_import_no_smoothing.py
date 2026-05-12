"""
tests/smoke_data_import_no_smoothing.py
=======================================

Patch 122g: regression guard.

* PoseImportForm no longer exposes smoothing widgets — the
  Preprocessing page (formerly "Pose cleanup") is the canonical
  smoothing surface and the duplicate import-time toggle was
  redundant.
* The Preprocessing page registers itself under the label
  "Preprocessing", not the historical "Pose cleanup".
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
    # ------------------------------------------------------------------
    # 1. PoseImportForm: no smoothing widgets, no smoothing param
    # ------------------------------------------------------------------
    pose_import_src = (
        REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "pose_import.py"
    ).read_text()
    pose_import_tree = ast.parse(pose_import_src)

    # No `self._smooth_*` widget attributes assigned anywhere
    forbidden_attrs = (
        "self._smooth_enable",
        "self._smooth_method",
        "self._smooth_window",
    )
    for attr in forbidden_attrs:
        check(
            f"PoseImportForm no longer creates {attr}",
            attr not in pose_import_src,
            detail="found a still-attached widget assignment",
        )

    # No "Enable smoothing" / "Smoothing method:" / "Smoothing window"
    # UI labels — these are the user-visible strings that would tell
    # us the duplicate surface is back
    for ui_label in (
        '"Enable smoothing"',
        '"Smoothing method:"',
        '"  Smoothing method:"',
        '"Smoothing window (ms):"',
        '"  Smoothing window (ms):"',
    ):
        check(
            f"PoseImportForm UI: {ui_label} not present",
            ui_label not in pose_import_src,
        )

    # PoseImportForm class found
    pif_cls = None
    for node in ast.walk(pose_import_tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "PoseImportForm"
        ):
            pif_cls = node
            break
    check("PoseImportForm class exists", pif_cls is not None)

    if pif_cls is not None:
        methods = {
            n.name: n for n in pif_cls.body
            if isinstance(n, ast.FunctionDef)
        }
        # target() no longer takes smoothing_settings
        if "target" in methods:
            args = methods["target"].args
            arg_names = (
                [a.arg for a in args.args]
                + [a.arg for a in args.kwonlyargs]
            )
            check(
                "PoseImportForm.target signature has no "
                "smoothing_settings parameter",
                "smoothing_settings" not in arg_names,
                detail=f"got args: {arg_names}",
            )
            # collect_args returns no smoothing_settings key
        if "collect_args" in methods:
            body_src = ast.unparse(methods["collect_args"])
            check(
                "PoseImportForm.collect_args does not return "
                "'smoothing_settings'",
                "'smoothing_settings'" not in body_src
                and '"smoothing_settings"' not in body_src,
            )

    # ------------------------------------------------------------------
    # 2. Preprocessing page label
    # ------------------------------------------------------------------
    page_src = (
        REPO_ROOT / "mufasa" / "ui_qt" / "pages" / "pose_cleanup_page.py"
    ).read_text()
    check(
        "Preprocessing page uses the 'Preprocessing' label",
        '"Preprocessing"' in page_src
        and 'add_page("Preprocessing"' in page_src,
    )
    check(
        "Old 'Pose cleanup' label no longer present in add_page(...)",
        'add_page("Pose cleanup"' not in page_src,
    )

    print(
        f"smoke_data_import_no_smoothing: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
