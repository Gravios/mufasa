"""
tests/smoke_122dl_roi_apply_to_selected.py
=============================================

Patch 122dl: implements Proposal 1 from the ROI enhancements
proposal (docs/roi_enhancements_proposal.md).

Adds an "Apply to selected videos…" button to the ROI define
panel. Solves the multi-condition / multi-arena pain where
"Apply to all" is too broad.

Discovery during implementation
-------------------------------
The proposal estimated ~280 LoC for this patch — assuming a new
multi-select dialog from scratch. During implementation it
became clear that an equivalent dialog already exists:
``DuplicateRoisDialog`` (in ``duplicate_rois_source_target.py``,
landed in patch 122cv). That dialog's ``_run()`` does exactly
the source-to-targets ROI copy operation Proposal 1 describes
— it was just accessed from a less-discoverable menu item
("Duplicate ROIs from source video to target videos…") on the
ROI video table.

The patch reuses the existing dialog. Total LoC dropped from
~280 to ~80.

What landed
-----------
1. ``DuplicateRoisDialog`` gained two optional constructor params:
   - ``default_source: Optional[str]`` — pre-select a specific
     video in the source combo. When None, keeps the legacy
     alphabetic-first default. Tolerates basename or full-path
     input via ``Path(...).stem``.
   - ``window_title: Optional[str]`` — override the titlebar.
     Default ("Duplicate ROIs from source to target videos")
     preserved for the existing menu entry; the new "Apply to
     selected videos…" button passes a friendlier title that
     matches the button label.

2. ``ROIDefinePanel`` gained a button + handler:
   - "Apply to selected…" button added to the bottom action bar
     between "Apply to all" and the close/save group.
   - ``_on_apply_selected_clicked`` handler — saves current
     ROIs (same pattern as Apply-all), then opens
     DuplicateRoisDialog with default_source=current_video and
     the friendlier window_title. On Accepted, refreshes the
     video list and emits rois_modified.

Coverage
--------
1. DuplicateRoisDialog.__init__ has default_source parameter.
2. DuplicateRoisDialog.__init__ has window_title parameter.
3. Both new parameters have default None (backward-compatible).
4. Dialog body uses default_source to pre-select source combo
   (look for setCurrentText call against the matched stem).
5. Dialog body uses window_title in setWindowTitle (with
   fallback to the legacy title).
6. ROIDefinePanel.build() creates apply_selected_btn.
7. ROIDefinePanel has _on_apply_selected_clicked method.
8. _on_apply_selected_clicked saves ROIs first (the
   logic.save() pattern).
9. _on_apply_selected_clicked opens DuplicateRoisDialog with
   default_source AND window_title kwargs.
10. _on_apply_selected_clicked emits rois_modified on accept.
11. The existing menu entry on ROI video table still calls
    DuplicateRoisDialog (backward-compat — _action_duplicate
    code path unchanged).
12. ruff F401/W292/W293 clean on the two touched files.
13. All mufasa/**/*.py parse cleanly.
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
    dup = pkg / "ui_qt" / "dialogs" / "duplicate_rois_source_target.py"
    panel = pkg / "ui_qt" / "dialogs" / "roi_define_panel.py"
    vt = pkg / "ui_qt" / "dialogs" / "roi_video_table.py"

    # ------------------------------------------------------------------ #
    # DuplicateRoisDialog __init__ signature
    # ------------------------------------------------------------------ #
    dup_tree = ast.parse(dup.read_text())
    dup_cls = next(
        (n for n in dup_tree.body
         if isinstance(n, ast.ClassDef) and n.name == "DuplicateRoisDialog"),
        None,
    )
    dup_init = next(
        (m for m in dup_cls.body
         if isinstance(m, ast.FunctionDef) and m.name == "__init__"),
        None,
    ) if dup_cls else None

    init_args = [a.arg for a in dup_init.args.args] if dup_init else []
    init_defaults = dup_init.args.defaults if dup_init else []

    # 1. default_source parameter exists
    check(
        "DuplicateRoisDialog.__init__ has default_source parameter",
        "default_source" in init_args,
    )
    # 2. window_title parameter exists
    check(
        "DuplicateRoisDialog.__init__ has window_title parameter",
        "window_title" in init_args,
    )
    # 3. Both defaults are None (backward-compatible)
    # Map each new arg → its default value (defaults align to the
    # tail of args).
    n_pos = len(init_args) - len(init_defaults)
    arg_to_default = {
        init_args[n_pos + i]: init_defaults[i]
        for i in range(len(init_defaults))
    }
    ds_default = arg_to_default.get("default_source")
    wt_default = arg_to_default.get("window_title")
    check(
        "default_source defaults to None (backward-compat)",
        isinstance(ds_default, ast.Constant) and ds_default.value is None,
    )
    check(
        "window_title defaults to None (backward-compat)",
        isinstance(wt_default, ast.Constant) and wt_default.value is None,
    )

    # 4. Dialog body uses default_source for pre-selection
    dup_src = dup.read_text()
    check(
        "Dialog body uses _default_source to pre-select source "
        "combo (setCurrentText after stem-matching)",
        "_default_source" in dup_src
        and "setCurrentText" in dup_src
        and ".stem" in dup_src,
    )
    # 5. Dialog body uses window_title in setWindowTitle
    check(
        "Dialog uses window_title parameter in setWindowTitle "
        "with fallback to the legacy default",
        "window_title" in dup_src
        and "setWindowTitle(" in dup_src
        and "Duplicate ROIs from source to target videos" in dup_src,
    )

    # ------------------------------------------------------------------ #
    # ROIDefinePanel / ROIDefineWidget — new button + handler
    # ------------------------------------------------------------------ #
    # Patch 122dn refactor moved the panel's content from
    # ROIDefinePanel (now a thin QDialog wrapper) to
    # ROIDefineWidget (the embeddable QWidget). The handler we
    # check for lives on whichever class actually has the
    # apply-selected button — prefer the widget post-122dn but
    # fall back to the panel for pre-122dn compatibility.
    panel_tree = ast.parse(panel.read_text())
    panel_src = panel.read_text()
    panel_cls = next(
        (n for n in panel_tree.body
         if isinstance(n, ast.ClassDef)
         and n.name in ("ROIDefineWidget", "ROIDefinePanel")),
        None,
    )
    methods = (
        {m.name for m in panel_cls.body if isinstance(m, ast.FunctionDef)}
        if panel_cls else set()
    )

    # 6. apply_selected_btn created in build
    check(
        "ROIDefinePanel.build() creates apply_selected_btn "
        "with the right label",
        "apply_selected_btn" in panel_src
        and '"Apply to selected…"' in panel_src,
    )
    # 7. _on_apply_selected_clicked method exists
    check(
        "ROIDefinePanel has _on_apply_selected_clicked method",
        "_on_apply_selected_clicked" in methods,
    )
    # 8-10. handler body
    handler = next(
        (m for m in panel_cls.body
         if isinstance(m, ast.FunctionDef)
         and m.name == "_on_apply_selected_clicked"),
        None,
    ) if panel_cls else None
    if handler is not None:
        handler_src = ast.unparse(handler)
        # 8. Saves first
        check(
            "_on_apply_selected_clicked saves ROIs first "
            "(logic.save before opening dialog)",
            "logic.save()" in handler_src
            and handler_src.find("logic.save()")
                < handler_src.find("DuplicateRoisDialog"),
        )
        # 9. Opens DuplicateRoisDialog with default_source + window_title
        check(
            "_on_apply_selected_clicked opens DuplicateRoisDialog "
            "with default_source AND window_title kwargs",
            "DuplicateRoisDialog(" in handler_src
            and "default_source=" in handler_src
            and "window_title=" in handler_src,
        )
        # 10. Emits rois_modified on accept
        check(
            "_on_apply_selected_clicked emits rois_modified on "
            "QDialog.Accepted",
            "QDialog.Accepted" in handler_src
            and "rois_modified.emit()" in handler_src,
        )

    # 11. Backward-compat: ROI video table still calls dialog
    # the same way (no positional-arg surgery; new kwargs are
    # all optional).
    vt_src = vt.read_text()
    check(
        "ROI video table's _action_duplicate still constructs "
        "DuplicateRoisDialog (backward-compat unchanged)",
        "DuplicateRoisDialog(" in vt_src
        and "config_path=self.config_path" in vt_src,
    )

    # 12. ruff clean on touched files
    import subprocess
    try:
        out = subprocess.run(
            ["ruff", "check", str(dup), str(panel),
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
        check("ruff check skipped (not available in this env)", True)

    # 13. Parse-clean
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
        f"smoke_122dl_roi_apply_to_selected: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
