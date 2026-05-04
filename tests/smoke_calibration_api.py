"""Test for the calibration backend API surface.

This test is a regression guard for the px-dist-widget bug: I
previously hardcoded the wrong class name (GetPixelsPerMillimeter-
Interface, the legacy SimBA name) and the wrong constructor
parameter (known_metric_mm instead of known_mm_distance). The
form failed at runtime with ImportError when the user clicked
Calibrate.

This test verifies, via AST inspection of the actual backend
module, that:
- The class name we call exists in calculate_px_dist.py
- The constructor parameter we pass is what the class expects
- The result attribute (.ppm) is set in the constructor body

If any of these change in mufasa upstream, this test breaks
loudly instead of letting the form crash silently at runtime.

    PYTHONPATH=. python tests/smoke_calibration_api.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    src = Path("mufasa/video_processors/calculate_px_dist.py").read_text()
    tree = ast.parse(src)

    # ------------------------------------------------------------------ #
    # Case 1: CalculatePixelDistanceTool class is defined and exported
    # ------------------------------------------------------------------ #
    cls = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef)
         and n.name == "CalculatePixelDistanceTool"),
        None,
    )
    assert cls is not None, (
        "calculate_px_dist.py should define class "
        "CalculatePixelDistanceTool — the form depends on this name"
    )

    # ------------------------------------------------------------------ #
    # Case 2: __init__ accepts (video_path, known_mm_distance)
    # parameters by name. This is what the form passes; if the
    # parameter is renamed upstream, the form's keyword call will
    # fail with TypeError.
    # ------------------------------------------------------------------ #
    init = next(
        (n for n in cls.body
         if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    assert init is not None, "Class must have __init__"
    arg_names = [arg.arg for arg in init.args.args]
    assert "video_path" in arg_names, (
        f"Constructor should take video_path; got {arg_names}"
    )
    assert "known_mm_distance" in arg_names, (
        f"Constructor should take known_mm_distance; got {arg_names}. "
        f"If this is now 'known_metric_mm' or similar, update the "
        f"call site in mufasa/ui_qt/forms/video_info.py."
    )

    # ------------------------------------------------------------------ #
    # Case 3: __init__ sets self.ppm (the form reads tool.ppm after
    # construction). Search assignments in the function body and any
    # other methods called from __init__.
    # ------------------------------------------------------------------ #
    full_src = ast.unparse(cls)
    assert "self.ppm" in full_src, (
        "Class must set self.ppm somewhere — the form reads it "
        "after construction"
    )

    # ------------------------------------------------------------------ #
    # Case 4: __init__ runs the OpenCV widget directly (not via a
    # separate .run() method). The form does NOT call .run() on the
    # instance, so if the class structure changed to require an
    # explicit .run(), this would catch it.
    # ------------------------------------------------------------------ #
    init_src = ast.unparse(init)
    # The constructor calls choose_first_coordinate, choose_second,
    # and manipulate_choices — all the actual UI work happens here.
    assert (
        "choose_first_coordinate" in init_src
        or "manipulate_choices" in init_src
    ), (
        "Constructor should run the OpenCV workflow inline. If "
        "this changed (e.g. the workflow moved to a .run() method), "
        "update the form's call pattern in video_info.py."
    )

    # ------------------------------------------------------------------ #
    # Case 5: form is calling the new Qt dialog now (replaces the
    # OpenCV-based CalculatePixelDistanceTool — see commentary in
    # mufasa/ui_qt/dialogs/pixel_calibration.py for why). The form
    # must NOT reference the legacy GetPixelsPerMillimeterInterface
    # name OR the OpenCV CalculatePixelDistanceTool.
    # ------------------------------------------------------------------ #
    form_src = Path("mufasa/ui_qt/forms/video_info.py").read_text()
    assert "PixelCalibrationDialog" in form_src, (
        "Form must import PixelCalibrationDialog from "
        "mufasa.ui_qt.dialogs.pixel_calibration"
    )
    assert "GetPixelsPerMillimeterInterface" not in form_src, (
        "Form must not reference the legacy "
        "GetPixelsPerMillimeterInterface name"
    )
    # No actual import/call of CalculatePixelDistanceTool — but a
    # comment mentioning it for context is fine. Check via AST.
    form_tree = ast.parse(form_src)
    bad_imports = []
    for node in ast.walk(form_tree):
        if isinstance(node, ast.ImportFrom):
            for alias in (node.names or []):
                if alias.name == "CalculatePixelDistanceTool":
                    bad_imports.append(ast.unparse(node))
        elif isinstance(node, ast.Name) and node.id == "CalculatePixelDistanceTool":
            # Reference to the name in code (not in a string/comment)
            bad_imports.append(node.id)
    assert not bad_imports, (
        f"Form should use the Qt PixelCalibrationDialog instead "
        f"of the OpenCV CalculatePixelDistanceTool. Found "
        f"references in code: {bad_imports}"
    )
    # known_mm_distance keyword preserved (Qt dialog uses the
    # same name for API consistency)
    assert "known_mm_distance" in form_src

    # ------------------------------------------------------------------ #
    # Case 6: the Tools-menu helper in video_processing_page.py also
    # uses the Qt dialog (was previously using CalculatePixelDistanceTool)
    # ------------------------------------------------------------------ #
    page_src = Path(
        "mufasa/ui_qt/pages/video_processing_page.py"
    ).read_text()
    assert "PixelCalibrationDialog" in page_src, (
        "Tools-menu calibration helper must also use "
        "PixelCalibrationDialog"
    )
    # No bare _pxd.run() *call* anywhere (was the original bug).
    # A comment mentioning the historical bug is fine; we check
    # for actual function calls via AST.
    page_tree = ast.parse(page_src)
    bad_calls = []
    for node in ast.walk(page_tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "run"
                and isinstance(f.value, ast.Name)
                and f.value.id == "_pxd"
            ):
                bad_calls.append(ast.unparse(node))
    assert not bad_calls, (
        f"_pxd.run() doesn't exist as a module-level function in "
        f"calculate_px_dist.py. Found: {bad_calls}"
    )

    print("smoke_calibration_api: 6/6 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
