"""Tests for the Qt-native PixelCalibrationDialog.

Replaces the OpenCV-based CalculatePixelDistanceTool used by the
video_info form. The OpenCV widget had:
  - truncated instruction text on some displays
  - no native confirm/cancel buttons
  - ESC-to-confirm semantic that was unclear when text clipped

The new Qt dialog has:
  - QDialogButtonBox with explicit OK/Cancel
  - inline pixel-distance + px/mm readouts
  - Reset Points button
  - QDoubleSpinBox for the known mm distance
  - canvas widget that translates widget coords → frame coords
    (so click points are aspect-ratio-stable)

Sandbox-runnable via AST inspection (PySide6 not available).

    PYTHONPATH=. python tests/smoke_pixel_calibration_dialog.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    src = Path("mufasa/ui_qt/dialogs/pixel_calibration.py").read_text()
    tree = ast.parse(src)

    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    # ------------------------------------------------------------------ #
    # Case 1: Both classes exist
    # ------------------------------------------------------------------ #
    assert "PixelCalibrationDialog" in classes, (
        "PixelCalibrationDialog class must be defined"
    )
    assert "_CalibrationCanvas" in classes, (
        "Internal canvas widget _CalibrationCanvas must be defined"
    )
    assert "_ImageMapping" in classes, (
        "Coord-mapping helper _ImageMapping must be defined"
    )

    # ------------------------------------------------------------------ #
    # Case 2: Dialog inherits from QDialog
    # ------------------------------------------------------------------ #
    dlg = classes["PixelCalibrationDialog"]
    base_names = [
        b.id if isinstance(b, ast.Name) else None
        for b in dlg.bases
    ]
    assert "QDialog" in base_names, (
        f"PixelCalibrationDialog must inherit QDialog; got {base_names}"
    )

    # ------------------------------------------------------------------ #
    # Case 3: __init__ accepts video_path and known_mm_distance
    # (matches the call from video_info form)
    # ------------------------------------------------------------------ #
    init = next(
        (n for n in dlg.body
         if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    assert init is not None, "Dialog must have __init__"
    arg_names = [arg.arg for arg in init.args.args]
    assert "video_path" in arg_names, (
        f"__init__ should accept video_path; got {arg_names}"
    )
    assert "known_mm_distance" in arg_names, (
        f"__init__ should accept known_mm_distance; got {arg_names}"
    )
    assert "parent" in arg_names, (
        "__init__ should accept parent (Qt widget parent)"
    )

    # ------------------------------------------------------------------ #
    # Case 4: __init__ raises RuntimeError if first frame can't be read
    # (so the form can show a clear error before the dialog appears)
    # ------------------------------------------------------------------ #
    init_src = ast.unparse(init)
    assert "raise RuntimeError" in init_src, (
        "Dialog should raise RuntimeError on frame-read failure "
        "rather than showing a broken dialog"
    )

    # ------------------------------------------------------------------ #
    # Case 5: Dialog uses QDialogButtonBox with Ok | Cancel
    # ------------------------------------------------------------------ #
    assert "QDialogButtonBox" in init_src, (
        "Dialog should use QDialogButtonBox for OK/Cancel"
    )
    # OK starts disabled until both points are placed (defensive UX)
    methods = {
        n.name: n for n in dlg.body
        if isinstance(n, ast.FunctionDef)
    }
    assert "_update_readouts" in methods, (
        "Dialog should have a method to update readouts as click "
        "points / known-distance change"
    )
    update_src = ast.unparse(methods["_update_readouts"])
    assert "setEnabled(False)" in update_src, (
        "OK button should be disabled while either both points "
        "aren't placed OR the known-distance value is invalid"
    )
    assert "setEnabled(True)" in update_src, (
        "OK button should be enabled when state is valid"
    )

    # ------------------------------------------------------------------ #
    # Case 6: ppm attribute set on accept (the form reads dialog.ppm)
    # ------------------------------------------------------------------ #
    assert "_on_accept" in methods, (
        "Dialog should have an _on_accept handler"
    )
    accept_src = ast.unparse(methods["_on_accept"])
    assert "self.ppm" in accept_src, (
        "_on_accept should set self.ppm so the form can read it "
        "after exec() returns Accepted"
    )
    assert "self.accept()" in accept_src, (
        "_on_accept should call self.accept() to close with Accepted"
    )

    # ------------------------------------------------------------------ #
    # Case 7: Canvas widget has the correct coordinate mapping
    # (widget pixels → frame pixels) so click points are stored
    # in frame coords and don't drift on resize
    # ------------------------------------------------------------------ #
    canvas = classes["_CalibrationCanvas"]
    canvas_methods = {
        n.name: n for n in canvas.body
        if isinstance(n, ast.FunctionDef)
    }
    assert "mousePressEvent" in canvas_methods, (
        "Canvas must override mousePressEvent to capture clicks"
    )
    mouse_src = ast.unparse(canvas_methods["mousePressEvent"])
    # Widget→frame translation must happen before storing points;
    # otherwise a window resize would silently shift the points.
    assert "widget_to_frame" in mouse_src, (
        "Canvas must translate widget coords → frame coords on "
        "click, otherwise resize shifts the points silently"
    )
    # Two-click cycle
    assert "_point_a" in mouse_src and "_point_b" in mouse_src

    # ------------------------------------------------------------------ #
    # Case 8: Canvas exposes pixel_distance() and reset_points()
    # for the dialog to call
    # ------------------------------------------------------------------ #
    assert "pixel_distance" in canvas_methods
    assert "reset_points" in canvas_methods
    pd_src = ast.unparse(canvas_methods["pixel_distance"])
    assert "hypot" in pd_src or "sqrt" in pd_src, (
        "pixel_distance must compute Euclidean distance"
    )

    # ------------------------------------------------------------------ #
    # Case 9: First-frame loader uses cv2.VideoCapture with cleanup
    # ------------------------------------------------------------------ #
    loader = methods.get("_load_first_frame")
    assert loader is not None, "Need a _load_first_frame helper"
    loader_src = ast.unparse(loader)
    assert "cv2.VideoCapture" in loader_src
    assert "cap.release()" in loader_src, (
        "VideoCapture must be released to avoid leaking handles"
    )
    assert "finally:" in loader_src, "Cleanup should be in finally"

    # ------------------------------------------------------------------ #
    # Case 10: Out-of-frame clicks (in aspect-ratio bars) are ignored
    # ------------------------------------------------------------------ #
    # widget_to_frame returns None when clicks land outside the
    # frame; mousePressEvent must check for that None.
    assert "if frame_pt is None" in mouse_src or "frame_pt is None" in mouse_src, (
        "mousePressEvent must guard against widget_to_frame "
        "returning None (click in aspect-ratio bars)"
    )

    # ------------------------------------------------------------------ #
    # Case 11: Dialog persists known_mm_distance on accept
    # (regression guard for a real bug — only ppm was being
    # exposed previously, so the form's Distance cell drifted
    # out of sync when the user edited the dialog's spinbox)
    # ------------------------------------------------------------------ #
    on_accept_src = ast.unparse(methods["_on_accept"])
    assert "self.known_mm_distance" in on_accept_src, (
        "Dialog must persist self.known_mm_distance on accept so "
        "the form can sync the Distance (mm) cell with whatever "
        "value the user ended at in the dialog spinbox"
    )

    # Init must declare the attribute so getattr() works on cancel
    init_src_full = ast.unparse(init)
    assert "self.known_mm_distance" in init_src_full, (
        "Dialog __init__ must initialize self.known_mm_distance "
        "(to None) so callers can distinguish accepted-with-value "
        "from cancelled even when the attribute is read defensively"
    )

    print("smoke_pixel_calibration_dialog: 11/11 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
