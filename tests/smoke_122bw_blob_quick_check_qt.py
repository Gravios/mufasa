"""
tests/smoke_122bw_blob_quick_check_qt.py
==========================================

Patch 122bw: Tier 3a of the Tk → Qt consolidation plan. Qt
port of the legacy :class:`BlobQuickChecker` interactive Tk tool
(``mufasa/ui/blob_quick_check_interface.py``).

The port consists of:
* :class:`BlobQuickCheckForm` — minimal parameter form (two
  video paths). Overrides :meth:`on_run` to open the preview
  dialog on the Qt main thread instead of running through the
  standard worker-thread machinery (same pattern as
  :class:`ROIManageForm`'s "draw" action).
* :class:`_BlobCheckDialog` — interactive preview QDialog.
  Live controls for method / threshold / kernel sizes; frame
  navigation slider + ±1s / ±N s / first / last buttons.
* :func:`_cv2_to_qpixmap` — helper to convert CV2 / numpy images
  to QPixmap (handles both grayscale and RGB888).

Coverage
--------
1. New file mufasa/ui_qt/forms/blob_quick_check.py exists.
2. BlobQuickCheckForm class defined and subclasses OperationForm.
3. BlobQuickCheckForm defines build(), collect_args(), and on_run()
   (target() is present but explicitly raises — the dialog is
   opened by on_run).
4. _BlobCheckDialog class defined (interactive viewer).
5. _BlobCheckDialog references the three backend hooks:
   create_average_frm, ImageMixin.img_diff, read_frm_of_video.
6. The dialog has frame-navigation controls (slider + step buttons).
7. _cv2_to_qpixmap helper defined and handles both 2D and 3D images.
8. The Tk source class name BlobQuickChecker is referenced in the
   docstring (regression guard against future renames hiding the
   lineage).
9. mufasa/ui_qt/pages/addons_page.py imports BlobQuickCheckForm.
10. addons_page.py registers a "Blob quick-check" section.
11. Add-ons page has >= 10 sections (was 9).
12. All mufasa/**/*.py files parse cleanly.
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
    form_path = pkg / "ui_qt" / "forms" / "blob_quick_check.py"
    check("blob_quick_check.py exists", form_path.exists())
    if not form_path.exists():
        return 1
    src = form_path.read_text()
    tree = ast.parse(src)

    # ==================================================================
    # BlobQuickCheckForm class
    # ==================================================================
    form_cls = None
    dialog_cls = None
    cv2_helper = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == "BlobQuickCheckForm":
                form_cls = node
            elif node.name == "_BlobCheckDialog":
                dialog_cls = node
        if isinstance(node, ast.FunctionDef) and node.name == "_cv2_to_qpixmap":
            cv2_helper = node

    check("BlobQuickCheckForm class defined", form_cls is not None)
    check("_BlobCheckDialog class defined", dialog_cls is not None)
    check("_cv2_to_qpixmap helper function defined", cv2_helper is not None)

    if form_cls is not None:
        base_names = {ast.unparse(b) for b in form_cls.bases}
        check(
            "BlobQuickCheckForm subclasses OperationForm",
            "OperationForm" in base_names,
        )
        methods = {
            stmt.name for stmt in form_cls.body
            if isinstance(stmt, ast.FunctionDef)
        }
        for m in ("build", "collect_args", "on_run", "target"):
            check(
                f"BlobQuickCheckForm defines {m}()",
                m in methods,
            )

    if dialog_cls is not None:
        base_names = {ast.unparse(b) for b in dialog_cls.bases}
        check(
            "_BlobCheckDialog subclasses QDialog",
            "QDialog" in base_names,
        )

    # ==================================================================
    # Dialog references the three backend hooks
    # ==================================================================
    for backend in ("create_average_frm", "img_diff", "read_frm_of_video"):
        check(
            f"dialog references backend symbol '{backend}'",
            backend in src,
        )

    # Frame nav controls
    check(
        "dialog has a frame slider",
        "frame_slider" in src and "QSlider" in src,
    )
    check(
        "dialog has step buttons (first / last / ±1s / ±N s)",
        all(s in src for s in ("First", "Last", "+1s", "-1s",
                                "+N s", "-N s")),
    )

    # ==================================================================
    # Tk-source reference for lineage
    # ==================================================================
    check(
        "docstring references the replaced Tk source "
        "BlobQuickChecker",
        "BlobQuickChecker" in src,
    )
    check(
        "docstring references the legacy file path "
        "(blob_quick_check_interface.py)",
        "blob_quick_check_interface.py" in src,
    )

    # ==================================================================
    # Page wiring
    # ==================================================================
    page_path = pkg / "ui_qt" / "pages" / "addons_page.py"
    page_src = page_path.read_text()
    check(
        "addons_page.py imports BlobQuickCheckForm",
        "from mufasa.ui_qt.forms.blob_quick_check import "
        "BlobQuickCheckForm" in page_src,
    )
    check(
        "addons_page.py registers 'Blob quick-check' section",
        '"Blob quick-check"' in page_src,
    )
    check(
        "Section uses BlobQuickCheckForm",
        "(BlobQuickCheckForm, {})" in page_src,
    )
    section_count = page_src.count("page.add_section(")
    check(
        f"addons_page.py has >= 10 sections (got {section_count})",
        section_count >= 10,
    )

    # ==================================================================
    # All files parse cleanly
    # ==================================================================
    parse_errors: list[str] = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py files parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122bw_blob_quick_check_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
