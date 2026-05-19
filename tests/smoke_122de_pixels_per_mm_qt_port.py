"""
tests/smoke_122de_pixels_per_mm_qt_port.py
============================================

Patch 122de: port the pixel-calibration UI from the legacy
cv2-based ``mufasa.ui.px_to_mm_ui.GetPixelsPerMillimeterInterface``
to the existing Qt-native
``mufasa.ui_qt.dialogs.pixel_calibration.PixelCalibrationDialog``.

What this patch landed
----------------------
- Deleted ``mufasa/ui/px_to_mm_ui.py`` (154 lines; pure cv2
  standalone-window UI).
- Deleted ``mufasa/ui/__init__.py`` (zero importers; the package
  was a deprecation-warning shim for a no-longer-present
  surface). With this gone, ``mufasa/ui/`` directory is removed
  entirely.
- Edited ``mufasa/ui_qt/forms/video_utilities.py``:
  - ``PixelsPerMMForm`` now uses ``PixelCalibrationDialog``
    (the existing Qt dialog used elsewhere in the workbench)
    instead of the cv2-based ``GetPixelsPerMillimeterInterface``.
  - Override of ``on_run`` so the calibration dialog opens on
    the GUI thread (was implicitly fine when cv2 spawned its
    own native window; needs explicit handling for QDialog).
  - Module + class docstrings updated to mention the Qt dialog.
  - Added ``QDialog`` import (needed for the ``exec()`` ==
    ``QDialog.Accepted`` check).

Coverage
--------
1.  mufasa/ui/ directory is gone entirely.
2.  mufasa/ui/px_to_mm_ui.py is gone.
3.  mufasa/ui/__init__.py is gone.
4.  No surviving file imports from mufasa.ui.* (package or
    submodules).
5.  video_utilities.py imports QDialog (needed for exec result).
6.  PixelsPerMMForm uses PixelCalibrationDialog (the existing
    Qt dialog, not the deleted cv2 class).
7.  PixelsPerMMForm overrides on_run (not delegated to a worker
    thread; QDialog must run on GUI thread).
8.  PixelsPerMMForm.target still exists (OperationForm contract;
    re-routes to _launch_calibration for tests).
9.  image_mixin.py docstring updated to point at the Qt dialog
    (was referring to the deleted cv2 class).
10. No surviving file imports the deleted
    GetPixelsPerMillimeterInterface class.
11. Total mufasa/**/*.py count is 416 (was 418 post-122dd;
    -2 from deleting px_to_mm_ui.py + ui/__init__.py).
12. All mufasa/**/*.py parse cleanly.
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


def _code_only(src: str) -> str:
    return "\n".join(
        line for line in src.split("\n")
        if not line.lstrip().startswith("#")
    )


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # 1-3. Deletions
    check(
        "mufasa/ui/ directory is gone entirely",
        not (pkg / "ui").exists(),
    )
    check(
        "mufasa/ui/px_to_mm_ui.py is gone",
        not (pkg / "ui" / "px_to_mm_ui.py").exists(),
    )
    check(
        "mufasa/ui/__init__.py is gone",
        not (pkg / "ui" / "__init__.py").exists(),
    )

    # 4. No surviving file imports from mufasa.ui (package or submodules)
    foreign = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "mufasa.ui" or mod.startswith(
                        "mufasa.ui."):
                    foreign.append(
                        f"{f.relative_to(pkg)}:{node.lineno}"
                    )
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "mufasa.ui" or a.name.startswith(
                            "mufasa.ui."):
                        foreign.append(
                            f"{f.relative_to(pkg)}:{node.lineno}"
                        )
    check(
        f"No surviving file imports mufasa.ui or mufasa.ui.*"
        f"(got {len(foreign)})",
        not foreign,
        detail=", ".join(foreign[:3]),
    )

    # 5. video_utilities.py has QDialog import
    vu_src = (pkg / "ui_qt" / "forms"
              / "video_utilities.py").read_text()
    vu_code = _code_only(vu_src)
    check(
        "video_utilities.py imports QDialog",
        "QDialog" in vu_code and "from PySide6.QtWidgets" in vu_code,
    )

    # 6. PixelsPerMMForm uses PixelCalibrationDialog
    check(
        "PixelsPerMMForm uses PixelCalibrationDialog "
        "(the existing Qt dialog)",
        "PixelCalibrationDialog" in vu_code
        and "mufasa.ui_qt.dialogs.pixel_calibration" in vu_code,
    )

    # 7-8. PixelsPerMMForm has on_run + target; no executable
    # reference to the deleted class (docstring breadcrumb OK)
    vu_tree = ast.parse(vu_src)
    form_cls = next(
        (n for n in vu_tree.body
         if isinstance(n, ast.ClassDef)
         and n.name == "PixelsPerMMForm"),
        None,
    )
    check("PixelsPerMMForm class defined", form_cls is not None)
    if form_cls is not None:
        methods = {m.name for m in form_cls.body
                   if isinstance(m, ast.FunctionDef)}
        check(
            "PixelsPerMMForm overrides on_run "
            "(GUI-thread dialog launch)",
            "on_run" in methods,
        )
        check(
            "PixelsPerMMForm.target still exists "
            "(OperationForm contract)",
            "target" in methods,
        )
        # Check: no executable reference to the deleted class
        # in any method body. Class docstring is allowed to
        # mention it for archaeology. ast.unparse() of each
        # method drops comments naturally.
        method_bodies = "\n".join(
            ast.unparse(m) for m in form_cls.body
            if isinstance(m, ast.FunctionDef)
        )
        check(
            "PixelsPerMMForm has no executable reference to "
            "GetPixelsPerMillimeterInterface in any method body "
            "(class docstring breadcrumb is allowed)",
            "GetPixelsPerMillimeterInterface" not in method_bodies,
        )

    # 9. image_mixin.py docstring updated
    im_src = (pkg / "mixins" / "image_mixin.py").read_text()
    check(
        "image_mixin.py docstring points at PixelCalibrationDialog "
        "(no longer at the deleted cv2 class)",
        "PixelCalibrationDialog" in im_src,
    )
    check(
        "image_mixin.py docstring no longer references "
        "mufasa.ui.px_to_mm_ui (the deleted module)",
        "mufasa.ui.px_to_mm_ui" not in im_src,
    )

    # 10. No surviving file imports GetPixelsPerMillimeterInterface
    gpm_hits = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom):
                if any(a.name == "GetPixelsPerMillimeterInterface"
                       for a in node.names):
                    gpm_hits.append(
                        f"{f.relative_to(pkg)}:{node.lineno}"
                    )
    check(
        "No surviving file imports "
        "GetPixelsPerMillimeterInterface",
        not gpm_hits,
        detail=", ".join(gpm_hits[:3]),
    )

    # 11. Total .py count = 416
    total_py = sum(1 for _ in pkg.rglob("*.py"))
    check(
        f"Total mufasa/**/*.py count is 416 (was 418 post-122dd; "
        f"got {total_py}; -2 from px_to_mm_ui.py + ui/__init__.py)",
        total_py == 416,
    )

    # 12. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({total_py} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122de_pixels_per_mm_qt_port: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
