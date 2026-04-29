"""Structural tests for FrameScrubberWidget play/pause buttons.

The widget itself can't be instantiated in the sandbox (PySide6
isn't available), so we verify the structure via AST inspection.
The behavior tests would need a Qt event loop and a real video
file — those happen on the workstation when the user actually
clicks the buttons.

Verifies:
- Play-forward and play-backward buttons exist
- Play buttons are positioned BETWEEN the single-step buttons
  (not at the edges)
- A QTimer drives the playback
- Wrap-around is implemented (next at end → 0, prev at 0 → end)
- close_video and load() stop in-progress playback
- Same-button-twice toggles pause

    PYTHONPATH=. python tests/smoke_scrubber_play_buttons.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    src = Path("mufasa/ui_qt/frame_scrubber.py").read_text()
    tree = ast.parse(src)

    # ------------------------------------------------------------------ #
    # Case 1: QTimer imported
    # ------------------------------------------------------------------ #
    qtimer_imported = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "PySide6.QtCore":
            for alias in node.names:
                if alias.name == "QTimer":
                    qtimer_imported = True
    assert qtimer_imported, "QTimer must be imported from PySide6.QtCore"

    # ------------------------------------------------------------------ #
    # Case 2: FrameScrubberWidget defines play state and timer in __init__
    # ------------------------------------------------------------------ #
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FrameScrubberWidget"
    )
    init = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    init_src = ast.unparse(init)
    assert "_play_direction" in init_src, (
        "__init__ must initialize _play_direction state"
    )
    assert "_play_timer" in init_src, (
        "__init__ must create a QTimer for playback"
    )
    assert "QTimer(self)" in init_src, (
        "Timer should be a QTimer with self as parent (auto-cleanup)"
    )
    assert "_on_play_tick" in init_src, (
        "Timer should connect to _on_play_tick"
    )

    # ------------------------------------------------------------------ #
    # Case 3: build_ui creates play-forward and play-backward buttons
    # ------------------------------------------------------------------ #
    build_ui = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_build_ui"
    )
    build_src = ast.unparse(build_ui)
    assert "_b_play_back" in build_src, (
        "build_ui must create _b_play_back button"
    )
    assert "_b_play_fwd" in build_src, (
        "build_ui must create _b_play_fwd button"
    )

    # ------------------------------------------------------------------ #
    # Case 4: Play buttons are positioned BETWEEN the single-step
    # buttons (◀ at index N-1, ⏪ at N, ⏩ at N+1, ▶ at N+2 in the
    # ctrl layout). Verify by checking the order of addWidget calls.
    # ------------------------------------------------------------------ #
    # Find the order of ctrl.addWidget(...) calls
    addwidget_seq = []
    for node in ast.walk(build_ui):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "addWidget"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ctrl"
            ):
                if node.args and isinstance(node.args[0], ast.Attribute):
                    addwidget_seq.append(node.args[0].attr)
    # Expected order: prev100, prev10, prev, play_back, play_fwd, next, next10, next100
    expected_order = [
        "_b_prev100", "_b_prev10", "_b_prev",
        "_b_play_back", "_b_play_fwd",
        "_b_next", "_b_next10", "_b_next100",
    ]
    actual_order = [w for w in addwidget_seq if w in expected_order]
    assert actual_order == expected_order, (
        f"Play buttons must be between the single-step buttons. "
        f"Expected order:\n  {expected_order}\n"
        f"Got:\n  {actual_order}"
    )

    # ------------------------------------------------------------------ #
    # Case 5: _on_play_tick wraps around at boundaries
    # ------------------------------------------------------------------ #
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "_on_play_tick" in methods, "_on_play_tick method missing"
    tick_src = ast.unparse(methods["_on_play_tick"])
    assert "_total_frames" in tick_src, (
        "_on_play_tick should reference _total_frames for wrap"
    )
    # End → 0 (forward wrap)
    assert "next_idx = 0" in tick_src or "next_idx=0" in tick_src.replace(" ", ""), (
        "Forward wrap: next_idx >= total_frames → 0"
    )
    # 0 → end (backward wrap)
    assert "_total_frames - 1" in tick_src, (
        "Backward wrap: next_idx < 0 → total_frames - 1"
    )
    # Should call seek (so book-keeping happens consistently)
    assert "self.seek(" in tick_src, (
        "_on_play_tick should call seek() for the actual frame change"
    )

    # ------------------------------------------------------------------ #
    # Case 6: _toggle_play handles the three transitions
    # ------------------------------------------------------------------ #
    assert "_toggle_play" in methods, "_toggle_play method missing"
    toggle_src = ast.unparse(methods["_toggle_play"])
    # Same direction → pause
    assert "self._play_direction == direction" in toggle_src, (
        "Same-direction click should be detected for pause"
    )
    # Direction can be set
    assert "self._play_direction = direction" in toggle_src, (
        "Toggle should set the direction"
    )
    # Timer started conditionally (not double-started)
    assert "isActive" in toggle_src, (
        "Should check isActive before starting timer to avoid "
        "double-start when switching direction"
    )

    # ------------------------------------------------------------------ #
    # Case 7: _stop_playback resets state and stops timer
    # ------------------------------------------------------------------ #
    assert "_stop_playback" in methods
    stop_src = ast.unparse(methods["_stop_playback"])
    assert "_play_timer.stop()" in stop_src, "Should stop the timer"
    assert "_play_direction = 0" in stop_src, (
        "Should reset _play_direction to 0 (idle)"
    )

    # ------------------------------------------------------------------ #
    # Case 8: load() and close_video() stop in-progress playback
    # ------------------------------------------------------------------ #
    load = methods["load"]
    load_src = ast.unparse(load)
    assert "_stop_playback" in load_src, (
        "load() should stop playback before loading a new video"
    )
    close = methods["close_video"]
    close_src = ast.unparse(close)
    assert "_stop_playback" in close_src, (
        "close_video() should stop playback before releasing the cap"
    )

    # ------------------------------------------------------------------ #
    # Case 9: Timer interval set from FPS in load()
    # ------------------------------------------------------------------ #
    assert "setInterval" in load_src, (
        "load() should set the timer interval based on video FPS so "
        "playback runs at recorded speed"
    )
    assert "1000" in load_src, (
        "Interval should be 1000/fps ms"
    )

    # ------------------------------------------------------------------ #
    # Case 10: Refresh helper updates button glyphs
    # ------------------------------------------------------------------ #
    assert "_refresh_play_button_glyphs" in methods, (
        "Should have a helper to update play button labels (pause "
        "glyph when active, play glyph when idle)"
    )
    refresh_src = ast.unparse(methods["_refresh_play_button_glyphs"])
    # Pause glyph used when active
    assert "\\u23F8" in refresh_src or "⏸" in refresh_src, (
        "Should set pause glyph (⏸ U+23F8) on the active button"
    )

    print("smoke_scrubber_play_buttons: 10/10 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
