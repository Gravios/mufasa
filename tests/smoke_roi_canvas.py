"""Smoke-test for mufasa.ui_qt.dialogs.roi_canvas helpers.

Focuses on the coordinate transform (_ImageMapping) and draw-state
reset semantics — both testable without PySide6 / a display.

The canvas widget itself can't be tested headlessly. Visual
verification on the user's workstation is the actual integration
test.

    PYTHONPATH=. python tests/smoke_roi_canvas.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def _load_helpers():
    """Extract _ImageMapping and _DrawState from roi_canvas.py without
    importing the PySide6-dependent rest of the module."""
    src = Path("mufasa/ui_qt/dialogs/roi_canvas.py").read_text()

    # Extract just the dataclasses we want to test
    # _ImageMapping is between its decorator and the next class boundary
    lines = src.splitlines(keepends=True)

    # Find the lines that define _ImageMapping and _DrawState
    out_lines = []
    in_target = False
    for i, ln in enumerate(lines):
        # Capture @dataclass + class _ImageMapping ... end of class
        if ln.startswith("@dataclass"):
            # Look ahead — is the next class one we want?
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if "_ImageMapping" in nxt or "_DrawState" in nxt:
                    in_target = True
                    out_lines.append(ln)
                    continue
        if in_target:
            out_lines.append(ln)
            # End of class: blank line followed by non-indented line
            if (i + 1 < len(lines)
                    and not lines[i + 1].startswith(" ")
                    and not lines[i + 1].startswith("\t")
                    and lines[i + 1].strip() != ""):
                in_target = False

    helpers_src = "".join(out_lines)

    # Set up module namespace with the imports the dataclasses need
    from dataclasses import dataclass, field
    from typing import List, Optional, Tuple
    ns = {
        "dataclass": dataclass, "field": field,
        "List": List, "Optional": Optional, "Tuple": Tuple,
    }
    # Register a fake module so dataclass works (dataclasses references
    # cls.__module__ via sys.modules)
    mod = ModuleType("mufasa.ui_qt.dialogs.roi_canvas")
    sys.modules["mufasa.ui_qt.dialogs.roi_canvas"] = mod
    ns["__name__"] = "mufasa.ui_qt.dialogs.roi_canvas"
    mod.__dict__.update(ns)
    exec(helpers_src, mod.__dict__)
    return mod


def main() -> int:
    mod = _load_helpers()
    _ImageMapping = mod._ImageMapping
    _DrawState = mod._DrawState

    # ------------------------------------------------------------------ #
    # Case 1: 1:1 mapping (no scaling, no offset)
    # ------------------------------------------------------------------ #
    m = _ImageMapping(scale=1.0, offset_x=0, offset_y=0,
                      frame_w=100, frame_h=100)
    assert m.widget_to_frame(50, 50) == (50, 50)
    assert m.widget_to_frame(0, 0) == (0, 0)
    assert m.widget_to_frame(100, 100) == (100, 100)
    # Outside frame
    assert m.widget_to_frame(-1, 50) is None
    assert m.widget_to_frame(50, 101) is None
    # Round-trip
    assert m.frame_to_widget(50, 50) == (50, 50)

    # ------------------------------------------------------------------ #
    # Case 2: aspect-preserving fit, frame centered with vertical bars
    # 600x338 frame in a 1000x1000 widget → scale = 1000/600 = 1.667
    # New frame size: 1000 x 563.3, centered vertically with offsets
    # of (1000 - 563.3) / 2 ≈ 218 on top/bottom
    # ------------------------------------------------------------------ #
    scale = 1000 / 600
    offset_y = (1000 - int(round(338 * scale))) // 2
    m = _ImageMapping(scale=scale, offset_x=0, offset_y=offset_y,
                      frame_w=600, frame_h=338)

    # Click at widget (500, 500) — middle of widget
    # Widget x=500 → frame x=500/scale=300 (frame center)
    # Widget y=500 → frame y=(500-offset_y)/scale
    fp = m.widget_to_frame(500, 500)
    assert fp is not None
    assert fp[0] == 300, f"x: want 300 got {fp[0]}"
    # y: depends on offset
    expected_y = int(round((500 - offset_y) / scale))
    assert fp[1] == expected_y

    # Click in the top black bar — y < offset_y → outside frame
    fp = m.widget_to_frame(500, offset_y // 2)
    assert fp is None

    # ------------------------------------------------------------------ #
    # Case 3: round-trip frame→widget→frame is stable
    # ------------------------------------------------------------------ #
    m = _ImageMapping(scale=2.0, offset_x=10, offset_y=20,
                      frame_w=100, frame_h=80)
    for fx, fy in [(0, 0), (50, 40), (100, 80), (25, 25)]:
        wx, wy = m.frame_to_widget(fx, fy)
        rt = m.widget_to_frame(wx, wy)
        assert rt == (fx, fy), f"round-trip ({fx},{fy}) → ({wx},{wy}) → {rt}"

    # ------------------------------------------------------------------ #
    # Case 4: scale=0 returns None defensively
    # ------------------------------------------------------------------ #
    m = _ImageMapping(scale=0.0, offset_x=0, offset_y=0,
                      frame_w=100, frame_h=100)
    assert m.widget_to_frame(50, 50) is None

    # ------------------------------------------------------------------ #
    # Case 5: _DrawState.reset clears everything
    # ------------------------------------------------------------------ #
    s = _DrawState()
    s.rect_start = (10, 20)
    s.rect_end = (30, 40)
    s.circle_center = (50, 50)
    s.circle_edge = (60, 50)
    s.poly_vertices = [(0, 0), (10, 10), (20, 0)]
    s.poly_rubber_end = (15, 15)
    s.reset()
    assert s.rect_start is None
    assert s.rect_end is None
    assert s.circle_center is None
    assert s.circle_edge is None
    assert s.poly_vertices == []
    assert s.poly_rubber_end is None

    # ------------------------------------------------------------------ #
    # Case 6: _DrawState fresh has correct initial values
    # ------------------------------------------------------------------ #
    s = _DrawState()
    assert s.rect_start is None
    assert s.rect_end is None
    assert s.circle_center is None
    assert s.circle_edge is None
    assert s.poly_vertices == []
    assert s.poly_rubber_end is None

    # ------------------------------------------------------------------ #
    # Case 7: clicks just inside frame boundary register
    # ------------------------------------------------------------------ #
    m = _ImageMapping(scale=1.0, offset_x=0, offset_y=0,
                      frame_w=100, frame_h=100)
    assert m.widget_to_frame(0, 0) == (0, 0)            # exact corner
    assert m.widget_to_frame(100, 100) == (100, 100)    # exact corner
    assert m.widget_to_frame(99, 99) == (99, 99)         # inside

    # ------------------------------------------------------------------ #
    # Case 8: very-tall frame in a square widget — horizontal bars
    # 100x300 frame in a 200x200 widget → scale = 200/300 = 0.667
    # New frame: 67x200 centered horizontally with ~67px bars on sides
    # ------------------------------------------------------------------ #
    scale = 200 / 300
    new_w = int(round(100 * scale))
    offset_x = (200 - new_w) // 2
    m = _ImageMapping(scale=scale, offset_x=offset_x, offset_y=0,
                      frame_w=100, frame_h=300)

    # Click in the LEFT black bar (widget_x < offset_x) → outside
    assert m.widget_to_frame(offset_x // 2, 100) is None
    # Click in the frame center
    fp = m.widget_to_frame(100, 100)
    assert fp is not None
    # x: (100 - offset_x) / scale ≈ frame center
    assert abs(fp[0] - 50) <= 1   # 1-pixel tolerance for rounding
    assert abs(fp[1] - 150) <= 1

    print("smoke_roi_canvas: 8/8 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
