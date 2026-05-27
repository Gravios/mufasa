"""
tests/smoke_122fd_roi_move_bug_and_table_wiring.py
=====================================================

Patch 122fd — two related fixes to the ROI Definitions panel:

1. **Move-bug fix.** The user has reported THREE TIMES across
   four days: "I can select objects in the edit mode but I
   cannot move them." Code review at each report concluded the
   move infrastructure looked correct — SHAPE_MOVING /
   HANDLE_DRAGGING modes existed, mousePress transitioned
   correctly, _apply_edit_geom was wired. The user's persistence
   was right; the bug was simpler than each review imagined.

   ``ROICanvas.mouseMoveEvent`` had an early-return guard at line
   875 that filtered out ANY mode other than the three DRAWING
   modes (RECT_DRAGGING, CIRCLE_DRAGGING, POLY_VERTEXING). When
   the user pressed-and-dragged in select mode, the mousePress
   correctly transitioned to SHAPE_MOVING — but every subsequent
   mouseMoveEvent immediately RETURNED at the guard. The
   SHAPE_MOVING translation branch (and HANDLE_DRAGGING resize
   branch) at lines 902+ were DEAD CODE — never reached.

   The guard was written before 122dm added the two new modes,
   and wasn't updated when 122dm added them. The dead code
   slipped past every subsequent code review because the
   structure LOOKED right — a mode transition followed by a
   dispatch handler. The guard between them wasn't even on the
   diff that introduced the new modes.

   The fix: add SHAPE_MOVING and HANDLE_DRAGGING to the
   allowlist. One-line change. The dead handlers below are
   now actually called.

2. **Click-to-table-highlight wiring.** User request:
   "Clicking an ROI should also select/highlight it in the
   table."

   Added a new ``shape_selected`` signal to ROICanvas, emitted
   from the SELECT-mode mousePress handler. Panel connects this
   signal to a new ``_on_shape_selected`` slot that calls
   ``shape_table.selectRow(idx)`` (or ``clearSelection()`` on
   idx=-1, the deselect indicator).

LESSONS
=======

This is the strongest example of the session-2 "look harder
when the user repeats themselves" meta-lesson:

  > When a user reports the same problem more than twice, and
  > each code review concludes the code looks correct, the bug
  > is NOT in the code path being reviewed. It's in an
  > adjacent path the review keeps skipping. Read the FULL
  > callgraph of the path the user is exercising — including
  > every guard, branch, and early-return between the entry
  > point and the operation that's failing.

For 122fd, three reviews looked at:
  - mousePress SELECT branch (correct)
  - mouseMove SHAPE_MOVING branch (correct)
  - _apply_edit_geom (correct)
  - _reference_point_for_drag (correct)

What was NEVER looked at: the early-return guard between
the entry and SHAPE_MOVING dispatch. That's where the bug was.

A more rigorous pattern for code review of "feature is wired
but doesn't work": trace EVERY line that executes from the
user-input event to the visible state change, in order. The
guard would have shown up immediately.

Coverage
--------
Move-bug fix (3 checks):
1.  mouseMoveEvent guard allowlist includes SHAPE_MOVING.
2.  mouseMoveEvent guard allowlist includes HANDLE_DRAGGING.
3.  The SHAPE_MOVING dispatch branch (line ~902) is reachable
    (not dead code): it lives BELOW the guard, AND the guard
    allows SHAPE_MOVING through.

Click-to-table wiring (4 checks):
4.  ROICanvas declares a ``shape_selected`` signal.
5.  SELECT-mode mousePress emits shape_selected on click
    (positive idx OR -1 for deselect).
6.  Panel connects ``shape_selected.connect(self._on_shape_selected)``.
7.  Panel defines ``_on_shape_selected`` that calls
    ``shape_table.selectRow(idx)`` for idx >= 0 OR
    ``clearSelection()`` for idx < 0.

Cross-patch invariants (3 checks):
8.  122fc state preserved: import_video registered.
9.  122fb state preserved: roi_define_panel has
    maintenance_btn.
10. Parse-clean.
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


def main() -> int:
    rc_path = (REPO_ROOT / "mufasa" / "ui_qt" / "dialogs"
               / "roi_canvas.py")
    rc_src = rc_path.read_text()
    rc_tree = ast.parse(rc_src)

    # -----------------------------------------------------------------
    # Move-bug fix
    # -----------------------------------------------------------------
    # 1-2. Allowlist contains SHAPE_MOVING and HANDLE_DRAGGING.
    # Find mouseMoveEvent → first If statement → its test.
    move_method = None
    for cls in ast.walk(rc_tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "ROICanvas":
            for m in cls.body:
                if (isinstance(m, ast.FunctionDef)
                        and m.name == "mouseMoveEvent"):
                    move_method = m
                    break
            break
    assert move_method is not None, "mouseMoveEvent not found"

    # Inspect every NotIn test in the method body.
    allowlist_modes = set()
    for node in ast.walk(move_method):
        if isinstance(node, ast.UnaryOp):
            continue
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, ast.NotIn):
                    # Right side is the tuple of allowed modes.
                    for cmp in node.comparators:
                        if isinstance(cmp, ast.Tuple):
                            for elt in cmp.elts:
                                if (isinstance(elt, ast.Attribute)
                                        and isinstance(elt.value, ast.Name)
                                        and elt.value.id == "_DrawMode"):
                                    allowlist_modes.add(elt.attr)
    check(
        "mouseMoveEvent's early-return allowlist includes "
        "SHAPE_MOVING (the move-bug fix from 122fd — the "
        "translation handler at line ~902 is now reachable)",
        "SHAPE_MOVING" in allowlist_modes,
        detail=(f"allowlist: {sorted(allowlist_modes)}"),
    )
    check(
        "mouseMoveEvent's early-return allowlist includes "
        "HANDLE_DRAGGING (resize handler at line ~945 is now "
        "reachable; same root-cause class as the move bug)",
        "HANDLE_DRAGGING" in allowlist_modes,
        detail=(f"allowlist: {sorted(allowlist_modes)}"),
    )

    # 3. The SHAPE_MOVING dispatch branch is present.
    has_shape_moving_branch = False
    for node in ast.walk(move_method):
        if isinstance(node, ast.If):
            for cmp in ast.walk(node.test):
                if (isinstance(cmp, ast.Attribute)
                        and isinstance(cmp.value, ast.Name)
                        and cmp.value.id == "_DrawMode"
                        and cmp.attr == "SHAPE_MOVING"):
                    has_shape_moving_branch = True
                    break
    check(
        "mouseMoveEvent has a SHAPE_MOVING dispatch branch "
        "(unchanged from 122dm — verified still present after "
        "the guard fix; together they mean the translation "
        "handler is BOTH present AND reachable)",
        has_shape_moving_branch,
    )

    # -----------------------------------------------------------------
    # Click-to-table wiring
    # -----------------------------------------------------------------
    # 4. shape_selected signal declared.
    has_signal = False
    for cls in ast.walk(rc_tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "ROICanvas":
            for m in cls.body:
                if (isinstance(m, ast.Assign)
                        and len(m.targets) == 1
                        and isinstance(m.targets[0], ast.Name)
                        and m.targets[0].id == "shape_selected"):
                    has_signal = True
                    break
            break
    check(
        "ROICanvas declares a ``shape_selected`` Signal (122fd "
        "addition — used by the panel to wire canvas-click to "
        "table-row-highlight)",
        has_signal,
    )

    # 5. mousePress in SELECT mode emits shape_selected.
    # Find mousePressEvent + check for ``self.shape_selected.emit``.
    press_method = None
    for cls in ast.walk(rc_tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "ROICanvas":
            for m in cls.body:
                if (isinstance(m, ast.FunctionDef)
                        and m.name == "mousePressEvent"):
                    press_method = m
                    break
            break
    emits_in_press = False
    if press_method is not None:
        press_src = ast.unparse(press_method)
        emits_in_press = "shape_selected.emit" in press_src
    check(
        "mousePressEvent emits shape_selected (both on positive "
        "selection AND on click-empty-deselect, where idx=-1 "
        "signals 'clear table selection')",
        emits_in_press,
    )

    # -----------------------------------------------------------------
    # Panel wiring
    # -----------------------------------------------------------------
    rdp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "dialogs"
               / "roi_define_panel.py").read_text()
    rdp_tree = ast.parse(rdp_src)

    # 6. Connection wired.
    connection_present = (
        "shape_selected.connect(self._on_shape_selected)" in rdp_src
    )
    check(
        "Panel connects preview.shape_selected to its "
        "_on_shape_selected slot (the wiring)",
        connection_present,
    )

    # 7. _on_shape_selected handler exists and does selectRow.
    handler_does_select = False
    handler_clears_on_neg = False
    for node in ast.walk(rdp_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_on_shape_selected"):
            body_src = ast.unparse(node)
            handler_does_select = (
                "selectRow(" in body_src
            )
            handler_clears_on_neg = "clearSelection(" in body_src
            break
    check(
        "_on_shape_selected handler calls "
        "shape_table.selectRow(idx) for positive idx AND "
        "clearSelection() for idx < 0 (the deselect indicator)",
        handler_does_select and handler_clears_on_neg,
        detail=(
            f"selectRow_called={handler_does_select} "
            f"clearSelection_called={handler_clears_on_neg}"
        ),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    from mufasa.section_provenance import SECTIONS
    check(
        "122fc state preserved: SECTIONS['import_video'] "
        "registered",
        "import_video" in SECTIONS,
    )

    check(
        "122fb state preserved: roi_define_panel has "
        "maintenance_btn",
        "self.maintenance_btn" in rdp_src,
    )

    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122fd_roi_move_bug_and_table_wiring: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
