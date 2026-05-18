"""
tests/smoke_122ci_clahe_preview_dialog.py
===========================================

Patch 122ci: CLAHE interactive-preview Qt dialog.

The legacy `_ClahePanel` had an "interactive" checkbox that
raised `NotImplementedError` when checked — the dialog was
documented as needing a Qt port. This patch ships the port:

* `_ClahePreviewDialog(QDialog)` — interactive live-preview
  with `clip_limit` + `tile_size` + frame-nav slider, modal,
  returns Accepted with the final params or Rejected.
* `_cv2_to_qpixmap` — local helper (same shape as the one in
  blob_quick_check.py; kept local to avoid a cross-module
  dependency).
* `VideoFiltersForm.on_run` override — intercepts the
  interactive-CLAHE case, opens the dialog modally, mutates
  the collected kwargs with the dialog's final values, and
  proceeds to the standard worker-thread dispatch.
* `VideoFiltersForm.target` — defensive change: pops the
  `interactive` flag if it's still True (logic bug in
  `on_run`) rather than raising. The previous raise is gone.

Coverage
--------
1. `_ClahePreviewDialog` class defined.
2. `_ClahePreviewDialog` subclasses QDialog.
3. `_cv2_to_qpixmap` helper defined at module scope.
4. Dialog references the CLAHE backend primitive (cv2.createCLAHE).
5. Dialog references read_frm_of_video + get_video_meta_data
   backend hooks.
6. Dialog has clip_limit, tile_size, frame_slider widgets.
7. Dialog has Apply / Cancel buttons (QDialogButtonBox).
8. Dialog exposes clip_limit + tile_size as properties for the
   caller to read post-Accept.
9. VideoFiltersForm defines on_run().
10. on_run() opens the _ClahePreviewDialog when op==clahe and
    interactive is True.
11. on_run() falls through to run_with_progress for the standard
    worker dispatch (covers both interactive and non-interactive
    paths).
12. target()'s CLAHE branch no longer raises NotImplementedError
    for interactive=True (defensive pop now).
13. target()'s CLAHE branch no longer references "Interactive
    preview launches a dialog; not yet wired" message.
14. qt_form_runtime_gaps.md §1 summary marks CLAHE preview FIXED
    in 122ci.
15. qt_form_runtime_gaps.md §2b marks CLAHE row as FIXED with the
    122ci reference.
16. qt_form_runtime_gaps.md §2b severity line: "Severity (post-
    122ci): none" or similar.
17. All mufasa/**/*.py files parse cleanly.
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
    form_path = pkg / "ui_qt" / "forms" / "video_filters.py"
    src = form_path.read_text()
    tree = ast.parse(src)

    # ==================================================================
    # Dialog + helper
    # ==================================================================
    dialog_cls = None
    form_cls = None
    helper_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == "_ClahePreviewDialog":
                dialog_cls = node
            elif node.name == "VideoFiltersForm":
                form_cls = node
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_cv2_to_qpixmap"):
            helper_fn = node

    check("_ClahePreviewDialog class defined", dialog_cls is not None)
    check("VideoFiltersForm class defined", form_cls is not None)
    check("_cv2_to_qpixmap helper defined at module scope",
          helper_fn is not None)

    if dialog_cls is not None:
        base_names = {ast.unparse(b) for b in dialog_cls.bases}
        check(
            "_ClahePreviewDialog subclasses QDialog",
            "QDialog" in base_names,
        )
        dlg_src = ast.unparse(dialog_cls)
        check(
            "Dialog references cv2.createCLAHE",
            "cv2.createCLAHE" in dlg_src,
        )
        check(
            "Dialog references read_frm_of_video backend hook",
            "read_frm_of_video" in dlg_src,
        )
        check(
            "Dialog references get_video_meta_data backend hook",
            "get_video_meta_data" in dlg_src,
        )
        check(
            "Dialog has clip_limit, tile_size, and frame_slider",
            "self.clip_sp" in dlg_src
            and "self.tile_sp" in dlg_src
            and "self.frame_slider" in dlg_src,
        )
        check(
            "Dialog has Apply / Cancel buttons (QDialogButtonBox)",
            "QDialogButtonBox" in dlg_src
            and ("accepted" in dlg_src and "rejected" in dlg_src),
        )
        check(
            "Dialog exposes clip_limit + tile_size as properties",
            "@property" in dlg_src
            and "def clip_limit" in dlg_src
            and "def tile_size" in dlg_src,
        )

    # ==================================================================
    # on_run override
    # ==================================================================
    if form_cls is not None:
        on_run_src = ""
        target_src = ""
        for stmt in form_cls.body:
            if isinstance(stmt, ast.FunctionDef):
                if stmt.name == "on_run":
                    on_run_src = ast.unparse(stmt)
                elif stmt.name == "target":
                    target_src = ast.unparse(stmt)

        check(
            "VideoFiltersForm defines on_run()",
            on_run_src != "",
        )
        check(
            "on_run() opens _ClahePreviewDialog when interactive",
            "_ClahePreviewDialog" in on_run_src
            and "'clahe'" in on_run_src
            and "interactive" in on_run_src,
        )
        check(
            "on_run() falls through to run_with_progress",
            "run_with_progress" in on_run_src,
        )

        # target CLAHE branch cleanups
        check(
            "target()'s CLAHE branch no longer raises NotImplementedError",
            "NotImplementedError" not in target_src,
        )
        check(
            "target()'s CLAHE branch no longer has the "
            "'Interactive preview' raise message",
            "Interactive preview launches a dialog" not in target_src,
        )

    # ==================================================================
    # Doc updates
    # ==================================================================
    gaps = (REPO_ROOT / "docs" / "qt_form_runtime_gaps.md").read_text()
    check(
        "qt_form_runtime_gaps.md §1 summary marks CLAHE preview "
        "FIXED in 122ci",
        "122ci" in gaps and "Qt preview dialog" in gaps,
    )
    check(
        "qt_form_runtime_gaps.md §2b marks CLAHE row FIXED in 122ci",
        "FIXED in patch 122ci" in gaps,
    )
    check(
        "qt_form_runtime_gaps.md §2b post-122ci severity: none",
        "post-122ci): none" in gaps or "post-122ci): none." in gaps,
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
        f"smoke_122ci_clahe_preview_dialog: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
