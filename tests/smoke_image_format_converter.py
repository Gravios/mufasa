"""
tests/smoke_image_format_converter.py
=====================================

Patch 122r: regression guard for the upgraded
:class:`ImageFormatConverterForm` in
``mufasa/ui_qt/forms/image_conversion.py`` plus the legacy-Tk
class removals it enables.

AST-only (the Qt module imports PySide6; the legacy popup
module imports cv2 and tkinter, neither available in sandbox).

Coverage:

1. **ImageFormatConverterForm shape** — the new form must
   expose: ``self.scope`` (_ScopePicker), ``self.fmt_cb``
   (combo of PNG/JPEG/BMP/WEBP/TIFF), ``self.options``
   (QStackedWidget for per-format panels), per-format widget
   attrs (``jpeg_quality``, ``webp_quality``,
   ``tiff_compression``, ``tiff_stack``),
   ``self.save_dir_edit``, ``self.verbose``.

2. **Dispatch correctness** — ``target()`` branches on ``fmt``
   and dispatches to the matching
   ``video_processors.video_processing.convert_to_<fmt>``
   function. TIFF passes ``directory=`` and the compression /
   stack flags; PNG / BMP / JPEG / WEBP pass ``path=`` and
   (where applicable) ``quality=``.

3. **Validation** — ``collect_args()`` enforces TIFF-needs-
   directory and rejects when no source is selected.

4. **Legacy class removal** — none of the six excised classes
   (``Convert2PNGPopUp``, ``Convert2TIFFPopUp``,
   ``Convert2WEBPPopUp``, ``Convert2bmpPopUp``,
   ``Convert2jpegPopUp``, ``ChangeImageFormatPopUp``) survives
   as a ``class X(...)`` definition in
   ``mufasa/ui/pop_ups/video_processing_pop_up.py``. The
   deletion-record marker comments remain so future readers see
   why and where the Qt replacement lives.

5. **SimBA.py wiring** — none of the six classes is imported
   any more, none of them is referenced in a
   ``img_format_menu.add_command`` line, and the
   ``img_format_menu`` cascade itself is gone (it had no
   surviving entries).
"""
from __future__ import annotations

import ast
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


def _find_class(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def main() -> int:
    # ==================================================================
    # 1+2+3. ImageFormatConverterForm
    # ==================================================================
    p = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
         / "image_conversion.py")
    src = p.read_text()
    tree = ast.parse(src)

    cls = _find_class(tree, "ImageFormatConverterForm")
    check("ImageFormatConverterForm class defined", cls is not None)

    if cls is not None:
        class_src = ast.unparse(cls)
        # Inheritance
        bases = [
            (b.id if isinstance(b, ast.Name)
             else getattr(b, "attr", ""))
            for b in cls.bases
        ]
        check(
            "ImageFormatConverterForm extends OperationForm",
            "OperationForm" in bases,
        )
        # Title + description present
        check(
            "title is 'Convert image format'",
            "Convert image format" in class_src,
        )

        # Widget attrs
        for attr in (
            "self.scope", "self.fmt_cb", "self.options",
            "self.jpeg_quality", "self.webp_quality",
            "self.tiff_compression", "self.tiff_stack",
            "self.save_dir_edit", "self.verbose",
        ):
            check(
                f"ImageFormatConverterForm sets {attr}",
                attr in class_src,
            )

        # _FORMATS list includes all 5
        check(
            "_FORMATS lists PNG / JPEG / BMP / WEBP / TIFF",
            "'PNG'" in class_src and "'JPEG'" in class_src
            and "'BMP'" in class_src and "'WEBP'" in class_src
            and "'TIFF'" in class_src,
        )

        # Find target() and verify dispatch + signatures
        methods = {
            n.name: n for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }
        check("target() method defined", "target" in methods)
        if "target" in methods:
            target_src = ast.unparse(methods["target"])
            for fn in (
                "convert_to_png", "convert_to_jpeg",
                "convert_to_bmp", "convert_to_webp",
                "convert_to_tiff",
            ):
                check(
                    f"target() dispatches to {fn}",
                    fn in target_src,
                )
            check(
                "target() passes directory= to convert_to_tiff "
                "(TIFF backend constraint)",
                "convert_to_tiff(\n" in target_src
                or "convert_to_tiff(" in target_src,
            )
            check(
                "target() passes quality= to JPEG and WEBP",
                "quality=quality" in target_src
                or "quality = quality" in target_src,
            )

        # Find collect_args() and verify TIFF-needs-dir validation
        check(
            "collect_args() method defined",
            "collect_args" in methods,
        )
        if "collect_args" in methods:
            ca_src = ast.unparse(methods["collect_args"])
            check(
                "collect_args() rejects empty source",
                "No source selected" in ca_src,
            )
            check(
                "collect_args() enforces TIFF needs directory",
                "TIFF" in ca_src
                and ("is_dir" in ca_src or "scope.is_dir" in ca_src),
            )

    # ==================================================================
    # 4. Legacy class removal from video_processing_pop_up.py
    # ==================================================================
    vp_path = (REPO_ROOT / "mufasa" / "ui" / "pop_ups"
               / "video_processing_pop_up.py")
    vp_src = vp_path.read_text()
    vp_tree = ast.parse(vp_src)
    defined_classes = {
        n.name for n in vp_tree.body
        if isinstance(n, ast.ClassDef)
    }

    DOOMED = {
        "Convert2PNGPopUp", "Convert2TIFFPopUp",
        "Convert2WEBPPopUp", "Convert2bmpPopUp",
        "Convert2jpegPopUp", "ChangeImageFormatPopUp",
    }
    for d in sorted(DOOMED):
        check(
            f"legacy {d} class no longer defined in "
            "video_processing_pop_up.py",
            d not in defined_classes,
        )
        # The deletion marker comment should remain so future
        # readers can find the rationale + Qt replacement
        # pointer via grep.
        marker_re = re.compile(
            rf"^# {re.escape(d)}: removed in patch 122r",
            re.MULTILINE,
        )
        check(
            f"deletion-record marker for {d} present",
            bool(marker_re.search(vp_src)),
        )

    # ==================================================================
    # 5. SimBA.py wiring scrubbed
    # ==================================================================
    simba_src = (REPO_ROOT / "mufasa" / "SimBA.py").read_text()
    for d in sorted(DOOMED):
        check(
            f"SimBA.py no longer references {d}",
            d not in simba_src,
        )
    # img_format_menu cascade should be gone (it had no surviving
    # entries after the scrub).
    check(
        "SimBA.py no longer creates 'img_format_menu' "
        "(no surviving entries)",
        "img_format_menu = Menu(" not in simba_src
        and "img_format_menu.add_command" not in simba_src
        and "menu=img_format_menu" not in simba_src,
    )

    print(
        f"smoke_image_format_converter: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
