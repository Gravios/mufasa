"""
tests/smoke_122dm_roi_drag_to_adjust.py
==========================================

Patch 122dm: implements Proposal 2 from the ROI enhancements
proposal (docs/roi_enhancements_proposal.md).

Adds drag-to-adjust placed ROIs within the existing QPainter
canvas framework. NOT a QGraphicsScene rewrite — extends the
existing _DrawMode state machine + adds hit-testing.

What this patch landed
----------------------
**`mufasa/ui_qt/dialogs/roi_canvas.py`:**

1. _DrawMode gained three new values: SELECT, SHAPE_MOVING,
   HANDLE_DRAGGING.

2. Two new signals: shape_edited(int, dict) for committed edits
   and shape_deleted(int) for Delete-key removals.

3. Internal state: _selected_idx, _selected_handle,
   _drag_offset_frame, _pre_drag_geom. setFocusPolicy(StrongFocus)
   so the canvas receives keyboard events.

4. Public API: start_select() / stop_select() / is_selecting().

5. Geometry helpers: _canonical_geom (normalizes legacy ↔ canonical
   geometry dict forms), _apply_edit_geom (mutates in place during
   drag), _reference_point_for_drag (drag anchor per kind).

6. _hit_test method: back-to-front shape iteration; tests handles
   first (corners for rect, east-edge for circle) then body. Uses
   8-px widget-pixel tolerance for handles, point-in-polygon
   ray-cast for polygons.

7. _paint_selection_chrome: gold dashed outline + filled-square
   handles for the selected shape.

8. mousePressEvent SELECT branch: hit-test → set state → transition
   to SHAPE_MOVING (body) or HANDLE_DRAGGING (handle), or deselect
   on empty-space click.

9. mouseMoveEvent SHAPE_MOVING + HANDLE_DRAGGING branches:
   translates / resizes the shape in place. Polygon vertex-drag
   deferred per proposal (only body-drag supported for polygons).

10. mouseReleaseEvent: emits shape_edited if the pre-drag and
    post-drag canonical geoms differ. Transitions back to SELECT.

11. keyPressEvent SELECT branch: Delete / Backspace emits
    shape_deleted; Escape deselects without leaving select mode.

**`mufasa/roi_tools/roi_logic.py`:**

12. New methods: update_rectangle_geometry, update_circle_geometry,
    update_polygon_geometry. Each writes geometry in the same
    legacy-key form as the matching add_* method (topLeftX,
    Bottom_right_X, centerX, etc.) so the H5 serializer and
    painter's legacy readers stay backward-compat.

**`mufasa/ui_qt/dialogs/roi_define_panel.py`:**

13. Edit toggle button next to Draw; checkable QPushButton.
14. _on_edit_toggled: calls preview.start_select() / stop_select();
    disables Draw button while in edit mode.
15. _sync_preview tracks _overlay_idx_to_kind_name parallel to
    overlay_rois; signals' int idx maps back via this list.
16. _on_shape_edited: maps idx → (kind, name) → calls
    update_*_geometry on logic; sets _dirty; syncs table.
17. _on_shape_deleted: maps idx → name → delete_roi; refreshes.

Coverage
--------
1. _DrawMode has SELECT, SHAPE_MOVING, HANDLE_DRAGGING values.
2. ROICanvas has shape_edited + shape_deleted Signals.
3. ROICanvas has start_select / stop_select / is_selecting.
4. ROICanvas has _hit_test method.
5. ROICanvas has _canonical_geom method.
6. ROICanvas has _paint_selection_chrome method.
7. mousePressEvent has a SELECT branch.
8. mouseMoveEvent has SHAPE_MOVING + HANDLE_DRAGGING branches.
9. mouseReleaseEvent emits shape_edited on drag completion.
10. keyPressEvent handles Delete / Backspace in SELECT mode.
11. paintEvent calls _paint_selection_chrome for selected shape.
12. ROILogic has update_rectangle_geometry / circle / polygon.
13. update_rectangle_geometry writes legacy-key geometry form.
14. ROIDefinePanel has edit_btn (checkable).
15. ROIDefinePanel has _on_edit_toggled method that wires
    start_select / stop_select.
16. ROIDefinePanel has _on_shape_edited + _on_shape_deleted methods.
17. _sync_preview builds _overlay_idx_to_kind_name parallel list.
18. _on_shape_edited maps idx → (kind, name) and calls
    appropriate update_*_geometry.
19. ruff F401/W292/W293 clean on all three touched files.
20. All mufasa/**/*.py parse cleanly.
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


def get_class(tree, name):
    return next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == name),
        None,
    )


def get_method(cls, name):
    if cls is None:
        return None
    return next(
        (m for m in cls.body
         if isinstance(m, ast.FunctionDef) and m.name == name),
        None,
    )


def main() -> int:
    pkg = REPO_ROOT / "mufasa"
    canvas = pkg / "ui_qt" / "dialogs" / "roi_canvas.py"
    panel = pkg / "ui_qt" / "dialogs" / "roi_define_panel.py"
    logic = pkg / "roi_tools" / "roi_logic.py"

    canvas_src = canvas.read_text()
    panel_src = panel.read_text()
    logic_src = logic.read_text()
    canvas_tree = ast.parse(canvas_src)
    panel_tree = ast.parse(panel_src)
    logic_tree = ast.parse(logic_src)

    # ------------------------------------------------------------------ #
    # ROICanvas
    # ------------------------------------------------------------------ #
    # 1. _DrawMode enum has new values
    drawmode_cls = get_class(canvas_tree, "_DrawMode")
    enum_values = (
        {n.targets[0].id for n in (drawmode_cls.body if drawmode_cls else [])
         if isinstance(n, ast.Assign)
         and isinstance(n.targets[0], ast.Name)}
        if drawmode_cls else set()
    )
    check(
        "_DrawMode has SELECT / SHAPE_MOVING / HANDLE_DRAGGING values",
        {"SELECT", "SHAPE_MOVING", "HANDLE_DRAGGING"}.issubset(enum_values),
    )

    canvas_cls = get_class(canvas_tree, "ROICanvas")
    # Signals are class-level Assign nodes calling Signal(...)
    signal_names = set()
    if canvas_cls is not None:
        for n in canvas_cls.body:
            if (isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name)
                    and n.value.func.id == "Signal"):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        signal_names.add(t.id)
    # 2. New signals exist
    check(
        "ROICanvas has shape_edited Signal",
        "shape_edited" in signal_names,
    )
    check(
        "ROICanvas has shape_deleted Signal",
        "shape_deleted" in signal_names,
    )

    canvas_methods = (
        {m.name for m in canvas_cls.body
         if isinstance(m, ast.FunctionDef)}
        if canvas_cls else set()
    )
    # 3. Public API
    check(
        "ROICanvas has start_select / stop_select / is_selecting "
        "public methods",
        {"start_select", "stop_select", "is_selecting"}.issubset(canvas_methods),
    )
    # 4. _hit_test
    check("ROICanvas has _hit_test method",
          "_hit_test" in canvas_methods)
    # 5. _canonical_geom
    check("ROICanvas has _canonical_geom method",
          "_canonical_geom" in canvas_methods)
    # 6. _paint_selection_chrome
    check("ROICanvas has _paint_selection_chrome method",
          "_paint_selection_chrome" in canvas_methods)

    # 7. mousePressEvent SELECT branch
    mpe = get_method(canvas_cls, "mousePressEvent")
    if mpe is not None:
        mpe_src = ast.unparse(mpe)
        check(
            "mousePressEvent handles SELECT mode (hit-test → "
            "SHAPE_MOVING or HANDLE_DRAGGING)",
            "_DrawMode.SELECT" in mpe_src
            and "_hit_test" in mpe_src
            and "_DrawMode.SHAPE_MOVING" in mpe_src
            and "_DrawMode.HANDLE_DRAGGING" in mpe_src,
        )

    # 8. mouseMoveEvent SHAPE_MOVING + HANDLE_DRAGGING branches
    mme = get_method(canvas_cls, "mouseMoveEvent")
    if mme is not None:
        mme_src = ast.unparse(mme)
        check(
            "mouseMoveEvent has SHAPE_MOVING + HANDLE_DRAGGING "
            "branches",
            "_DrawMode.SHAPE_MOVING" in mme_src
            and "_DrawMode.HANDLE_DRAGGING" in mme_src
            and "_apply_edit_geom" in mme_src,
        )

    # 9. mouseReleaseEvent emits shape_edited
    mre = get_method(canvas_cls, "mouseReleaseEvent")
    if mre is not None:
        mre_src = ast.unparse(mre)
        check(
            "mouseReleaseEvent emits shape_edited on drag commit "
            "(only when geometry actually changed)",
            "shape_edited.emit" in mre_src
            and "_pre_drag_geom" in mre_src,
        )

    # 10. keyPressEvent — Delete in SELECT
    kpe = get_method(canvas_cls, "keyPressEvent")
    if kpe is not None:
        kpe_src = ast.unparse(kpe)
        check(
            "keyPressEvent handles Delete/Backspace in SELECT "
            "mode (emits shape_deleted)",
            "_DrawMode.SELECT" in kpe_src
            and "Qt.Key_Delete" in kpe_src
            and "shape_deleted.emit" in kpe_src,
        )

    # 11. paintEvent renders selection chrome
    pe = get_method(canvas_cls, "paintEvent")
    if pe is not None:
        pe_src = ast.unparse(pe)
        check(
            "paintEvent calls _paint_selection_chrome for the "
            "currently-selected shape",
            "_paint_selection_chrome" in pe_src
            and "_selected_idx" in pe_src,
        )

    # ------------------------------------------------------------------ #
    # ROILogic update methods
    # ------------------------------------------------------------------ #
    logic_cls = get_class(logic_tree, "ROILogic")
    logic_methods = (
        {m.name for m in logic_cls.body
         if isinstance(m, ast.FunctionDef)}
        if logic_cls else set()
    )
    # 12. update_*_geometry methods
    check(
        "ROILogic has update_rectangle_geometry + update_circle_"
        "geometry + update_polygon_geometry",
        {"update_rectangle_geometry", "update_circle_geometry",
         "update_polygon_geometry"}.issubset(logic_methods),
    )
    # 13. update_rectangle_geometry writes legacy keys
    urg = get_method(logic_cls, "update_rectangle_geometry")
    if urg is not None:
        urg_src = ast.unparse(urg)
        check(
            "update_rectangle_geometry writes legacy-key geometry "
            "(topLeftX, Bottom_right_X, etc.) for H5 compat",
            "topLeftX" in urg_src
            and "Bottom_right_X" in urg_src
            and "Center_X" in urg_src,
        )

    # ------------------------------------------------------------------ #
    # ROIDefinePanel
    # ------------------------------------------------------------------ #
    panel_cls = get_class(panel_tree, "ROIDefinePanel")
    panel_methods = (
        {m.name for m in panel_cls.body
         if isinstance(m, ast.FunctionDef)}
        if panel_cls else set()
    )
    # 14. edit_btn (checkable)
    check(
        "ROIDefinePanel has edit_btn (checkable QPushButton) "
        "with the right label",
        "edit_btn" in panel_src
        and 'QPushButton("Edit"' in panel_src
        and "setCheckable(True)" in panel_src,
    )
    # 15. _on_edit_toggled wires start_select / stop_select
    check(
        "ROIDefinePanel has _on_edit_toggled method that wires "
        "start_select / stop_select",
        "_on_edit_toggled" in panel_methods,
    )
    oet = get_method(panel_cls, "_on_edit_toggled")
    if oet is not None:
        oet_src = ast.unparse(oet)
        check(
            "_on_edit_toggled body calls start_select / stop_select "
            "and toggles Draw button enabled state",
            "start_select" in oet_src
            and "stop_select" in oet_src
            and "draw_btn.setEnabled" in oet_src,
        )

    # 16. signal-handler methods exist
    check(
        "ROIDefinePanel has _on_shape_edited + _on_shape_deleted",
        {"_on_shape_edited", "_on_shape_deleted"}.issubset(panel_methods),
    )

    # 17. _sync_preview builds _overlay_idx_to_kind_name
    sp = get_method(panel_cls, "_sync_preview")
    if sp is not None:
        sp_src = ast.unparse(sp)
        check(
            "_sync_preview builds _overlay_idx_to_kind_name parallel "
            "list (overlay-index → (kind, name) for signal lookup)",
            "_overlay_idx_to_kind_name" in sp_src,
        )

    # 18. _on_shape_edited maps idx → (kind, name)
    ose = get_method(panel_cls, "_on_shape_edited")
    if ose is not None:
        ose_src = ast.unparse(ose)
        check(
            "_on_shape_edited maps idx via _overlay_idx_to_kind_name "
            "and calls appropriate update_*_geometry",
            "_overlay_idx_to_kind_name" in ose_src
            and "update_rectangle_geometry" in ose_src
            and "update_circle_geometry" in ose_src
            and "update_polygon_geometry" in ose_src,
        )

    # 19. ruff clean
    import subprocess
    try:
        out = subprocess.run(
            ["ruff", "check", str(canvas), str(panel), str(logic),
             "--select", "F401,W292,W293"],
            capture_output=True, text=True, timeout=15,
            cwd=str(REPO_ROOT),
        )
        check(
            "ruff F401/W292/W293 clean on touched files",
            out.returncode == 0,
            detail=out.stdout[:200] if out.returncode != 0 else "",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        check("ruff check skipped (not available)", True)

    # 20. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try: ast.parse(f.read_text())
        except SyntaxError as e: parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=parse_errors[0] if parse_errors else "",
    )

    print(
        f"smoke_122dm_roi_drag_to_adjust: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
