"""Smoke-test for mufasa.utils.lookups.get_monitor_info.

Verifies the new screeninfo-based implementation handles both the
happy path (monitors present) and the failure path (screeninfo
missing or returns nothing) gracefully — by returning a stub
1920x1080 monitor.

This test extracts the function source rather than importing the
full lookups module (which pulls in numba, pyglet, etc. — none
available in the headless test sandbox).

    PYTHONPATH=. python tests/smoke_get_monitor_info.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def _extract_function_src() -> str:
    """Pull just the get_monitor_info function out of lookups.py."""
    src = Path("mufasa/utils/lookups.py").read_text()
    needle = "def get_monitor_info("
    start = src.find(needle)
    assert start >= 0, "get_monitor_info not found in lookups.py"
    # Walk forward; function ends at next unindented non-blank line
    lines = src[start:].splitlines(keepends=True)
    func_lines = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() == "" or ln[0] in " \t":
            func_lines.append(ln)
        else:
            break
    return "".join(func_lines)


def _run(monitors_to_return, expect_w, expect_h, expect_count):
    """Execute get_monitor_info() with a fake screeninfo module that
    returns ``monitors_to_return`` (or raises if it's an Exception
    instance), and assert the result."""
    func_src = _extract_function_src()

    # Build fake screeninfo module
    fake = ModuleType("screeninfo")
    if isinstance(monitors_to_return, Exception):
        def _raise():
            raise monitors_to_return
        fake.get_monitors = _raise
    else:
        fake.get_monitors = lambda: monitors_to_return
    sys.modules["screeninfo"] = fake

    # Provide other names referenced by the function (signature uses
    # Tuple, Dict, Union — all from typing)
    from typing import Dict, Tuple, Union
    ns = {"Dict": Dict, "Tuple": Tuple, "Union": Union}
    exec(func_src, ns)
    fn = ns["get_monitor_info"]
    results, (w, h) = fn()
    assert w == expect_w, f"want w={expect_w}, got {w}"
    assert h == expect_h, f"want h={expect_h}, got {h}"
    assert len(results) == expect_count, (
        f"want {expect_count} monitors, got {len(results)}: {results}"
    )
    # Cleanup
    del sys.modules["screeninfo"]
    return results


class _FakeMonitor:
    def __init__(self, x, y, width, height, is_primary=None):
        self.x = x; self.y = y
        self.width = width; self.height = height
        if is_primary is not None:
            self.is_primary = is_primary


def main() -> int:
    # Case 1: single monitor at (0, 0)
    res = _run(
        [_FakeMonitor(0, 0, 2560, 1440, is_primary=True)],
        expect_w=2560, expect_h=1440, expect_count=1,
    )
    assert res[0]["primary"] is True

    # Case 2: dual-monitor setup, second is primary (e.g. on screeninfo
    # platforms that expose is_primary)
    res = _run(
        [_FakeMonitor(0, 0, 1920, 1080, is_primary=False),
         _FakeMonitor(1920, 0, 2560, 1440, is_primary=True)],
        expect_w=2560, expect_h=1440, expect_count=2,
    )
    assert res[0]["primary"] is False
    assert res[1]["primary"] is True

    # Case 3: dual-monitor where is_primary attr is absent (older
    # screeninfo); falls back to (0,0) heuristic
    res = _run(
        [_FakeMonitor(0, 0, 1920, 1080),
         _FakeMonitor(1920, 0, 2560, 1440)],
        expect_w=1920, expect_h=1080, expect_count=2,
    )
    assert res[0]["primary"] is True
    assert res[1]["primary"] is False

    # Case 4: screeninfo raises (e.g. no DISPLAY)
    res = _run(
        Exception("no DISPLAY"),
        expect_w=1920, expect_h=1080, expect_count=1,
    )
    assert res[0]["primary"] is True

    # Case 5: screeninfo returns empty list
    res = _run(
        [],
        expect_w=1920, expect_h=1080, expect_count=1,
    )

    # Case 6: no monitor flagged primary; falls back to first monitor
    res = _run(
        [_FakeMonitor(100, 0, 1280, 720, is_primary=False),
         _FakeMonitor(1380, 0, 1920, 1080, is_primary=False)],
        expect_w=1280, expect_h=720, expect_count=2,
    )

    print("smoke_get_monitor_info: 6/6 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
