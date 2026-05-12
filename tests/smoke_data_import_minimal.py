"""
tests/smoke_data_import_minimal.py
==================================

Patches 122g + 122h: regression guard.

PoseImportForm now exposes only:

* Route picker (DLC H5, DLC CSV, etc.)
* Source directory + Browse
* Likelihood threshold (p_threshold)

Smoothing widgets (122g) and interpolation widgets (122h) were
removed — the Preprocessing page (formerly "Pose cleanup") is
the canonical surface for both, with strictly more options
(user-controllable copy_originals, auto-detected multi-index
headers, picker-driven input source).

Also asserts the Preprocessing page registers under the
"Preprocessing" label rather than the historical "Pose cleanup".
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
    # 1. PoseImportForm — no smoothing widgets, no interpolation
    #    widgets, no smoothing_settings / interpolation_settings in
    #    the form's collect_args or target signature.
    # ------------------------------------------------------------------
    pose_import_src = (
        REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "pose_import.py"
    ).read_text()
    pose_import_tree = ast.parse(pose_import_src)

    # No widget attributes for either feature
    forbidden_attrs = (
        # 122g
        "self._smooth_enable", "self._smooth_method", "self._smooth_window",
        # 122h
        "self._interp_enable", "self._interp_type", "self._interp_method",
    )
    for attr in forbidden_attrs:
        check(
            f"PoseImportForm no longer creates {attr}",
            attr not in pose_import_src,
            detail="found a still-attached widget assignment",
        )

    # No user-visible labels for either feature
    forbidden_labels = (
        # 122g
        '"Enable smoothing"',
        '"Smoothing method:"',
        '"  Smoothing method:"',
        '"Smoothing window (ms):"',
        '"  Smoothing window (ms):"',
        # 122h
        '"Enable interpolation"',
        '"Interpolation type:"',
        '"  Interpolation type:"',
        '"Interpolation method:"',
        '"  Interpolation method:"',
    )
    for ui_label in forbidden_labels:
        check(
            f"PoseImportForm UI: {ui_label} not present",
            ui_label not in pose_import_src,
        )

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
        # target() signature minimal
        if "target" in methods:
            args = methods["target"].args
            arg_names = (
                [a.arg for a in args.args]
                + [a.arg for a in args.kwonlyargs]
            )
            for forbidden in (
                "smoothing_settings",
                "interpolation_settings",
            ):
                check(
                    f"PoseImportForm.target signature has no "
                    f"{forbidden!r} parameter",
                    forbidden not in arg_names,
                    detail=f"got args: {arg_names}",
                )
            # The remaining kwonly args should be exactly the
            # minimal four: route, config_path, source_path,
            # p_threshold. (self is in args.args, not kwonlyargs.)
            kwonly = [a.arg for a in args.kwonlyargs]
            check(
                "PoseImportForm.target kwonly args = "
                "{route, config_path, source_path, p_threshold}",
                set(kwonly) == {
                    "route", "config_path", "source_path", "p_threshold",
                },
                detail=f"got: {kwonly}",
            )
        # collect_args returns no smoothing_settings /
        # interpolation_settings keys
        if "collect_args" in methods:
            body_src = ast.unparse(methods["collect_args"])
            for forbidden in (
                "smoothing_settings",
                "interpolation_settings",
            ):
                check(
                    f"PoseImportForm.collect_args does not return "
                    f"{forbidden!r}",
                    f"'{forbidden}'" not in body_src
                    and f'"{forbidden}"' not in body_src,
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
        f"smoke_data_import_minimal: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
