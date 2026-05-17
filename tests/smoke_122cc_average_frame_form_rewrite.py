"""
tests/smoke_122cc_average_frame_form_rewrite.py
=================================================

Patch 122cc: `AverageFrameForm` rewrite against the actual
`create_average_frm` backend signature.

Pre-122cc the form was entirely broken — looked for
`create_average_frame` (with `e`) which doesn't exist; even if
corrected, passed `method` and `stride` kwargs the backend
doesn't accept. The form's UI promised Mean/Median + sampling
stride; the backend supports neither.

Post-122cc the form:

* Drops the `method` (Mean/Median) and `stride` fields entirely.
  Backend is mean-only over all frames in the requested window.
* Adds a `WINDOW_MODES` selector: Whole video / Frame range /
  Time range (HH:MM:SS), driving a QStackedWidget of
  mode-specific panels.
* Adds an optional save-path field with file picker. If empty,
  defaults to `<source>_avgframe_<timestamp>.png` alongside
  source.
* Dispatch calls `create_average_frm(video_path, start_frm,
  end_frm, start_time, end_time, save_path, verbose=False)`
  with kwargs assembled from the selected window mode.

Coverage
--------
1. AverageFrameForm class defined.
2. Subclasses OperationForm.
3. Has WINDOW_MODES class attribute with 3 entries.
4. build() creates the QStackedWidget for window-mode panels.
5. build() creates the save-path QLineEdit + Browse button.
6. build() no longer creates `stat_cb` (Mean/Median dropdown).
7. build() no longer creates `stride` QSpinBox.
8. collect_args() returns video_path key.
9. collect_args() does NOT return 'method' or 'stride' keys.
10. collect_args() validates start ≤ end for both frame and time.
11. target() imports and calls `create_average_frm` (with no `e`).
12. target() does NOT call `create_average_frame` (with `e`).
13. target() defaults save_path to `_avgframe_<timestamp>.png`
    when not provided.
14. qt_form_runtime_gaps.md §2a marks AverageFrameForm as FIXED.
15. backend_audit.md §4b item 8 marked DONE in 122cc.
16. All mufasa/**/*.py files parse cleanly.
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
    form_path = pkg / "ui_qt" / "forms" / "image_conversion.py"
    src = form_path.read_text()
    tree = ast.parse(src)

    # ==================================================================
    # AverageFrameForm class
    # ==================================================================
    form_cls = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "AverageFrameForm"):
            form_cls = node
            break
    check("AverageFrameForm class defined", form_cls is not None)
    if form_cls is None:
        print("smoke_122cc_average_frame_form_rewrite: cannot continue")
        return 1

    base_names = {ast.unparse(b) for b in form_cls.bases}
    check(
        "AverageFrameForm subclasses OperationForm",
        "OperationForm" in base_names,
    )

    # WINDOW_MODES class attribute
    has_window_modes = False
    for stmt in form_cls.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == "WINDOW_MODES":
                    has_window_modes = True
                    break
    check(
        "AverageFrameForm has WINDOW_MODES class attribute",
        has_window_modes,
    )

    # Body src for substring checks
    cls_src = ast.unparse(form_cls)

    check(
        "WINDOW_MODES has 3 entries (Whole / Frame / Time)",
        "Whole video" in cls_src and "Frame range" in cls_src
        and "Time range" in cls_src,
    )
    check(
        "build() creates a QStackedWidget for window-mode panels",
        "QStackedWidget" in cls_src,
    )
    check(
        "build() creates the save-path QLineEdit + Browse",
        "QLineEdit" in cls_src and "Browse" in cls_src,
    )
    check(
        "build() no longer creates the legacy stat_cb (Mean/Median)",
        # The literal words 'Mean'/'Median' still appear in the
        # class docstring explaining what was dropped — only check
        # that the actual control identifier is gone.
        "self.stat_cb" not in cls_src,
    )
    check(
        "build() no longer creates the legacy stride QSpinBox",
        "self.stride" not in cls_src,
    )

    # collect_args returns video_path / no method or stride
    collect_args_src = ""
    target_src = ""
    for stmt in form_cls.body:
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name == "collect_args":
                collect_args_src = ast.unparse(stmt)
            elif stmt.name == "target":
                target_src = ast.unparse(stmt)

    check(
        "collect_args() returns 'video_path' key",
        "'video_path'" in collect_args_src,
    )
    check(
        "collect_args() does NOT return 'method' key",
        "'method'" not in collect_args_src,
    )
    check(
        "collect_args() does NOT return 'stride' key",
        "'stride'" not in collect_args_src,
    )
    check(
        "collect_args() validates start ≤ end (frame mode)",
        "Start frame" in collect_args_src
        and "end frame" in collect_args_src,
    )
    check(
        "collect_args() validates start < end (time mode)",
        "Start time" in collect_args_src
        and "end time" in collect_args_src,
    )

    # target() calls create_average_frm (no 'e')
    check(
        "target() imports/calls 'create_average_frm' (no 'e')",
        "create_average_frm" in target_src,
    )
    check(
        "target() does NOT call legacy 'create_average_frame' (with 'e')",
        "create_average_frame" not in target_src,
    )
    check(
        "target() defaults save_path to '_avgframe_<timestamp>.png' when empty",
        "_avgframe_" in target_src and ".png" in target_src,
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    gaps = (REPO_ROOT / "docs" / "qt_form_runtime_gaps.md").read_text()
    check(
        "qt_form_runtime_gaps.md §2a marks AverageFrameForm FIXED in 122cc",
        "FIXED in patch 122cc" in gaps,
    )
    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4b item 8 marks AverageFrameForm DONE in 122cc",
        "DONE in patch 122cc" in audit,
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
        f"smoke_122cc_average_frame_form_rewrite: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
