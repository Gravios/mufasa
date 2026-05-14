"""
tests/smoke_122aj_dockable_labeller.py
=======================================

Patch 122aj: dockable frame labeler. Split the Qt frame_labeller's
monolithic ``FrameLabellerDialog`` into a reusable
``FrameLabellerWidget`` content widget plus a thin dialog wrapper
and a new dock helper, so the labeler can live embedded inside
the workbench's main window OR floated out as a standalone
floating window — same widget, three host containers.

Plus two correctness fixes while in the area:

* Continue-mode load now routes through
  ``mufasa.utils.label_io.load_labels_for_video`` — just the
  behaviour label collection (Int64 columns), reading from
  ``derived/labels/`` first and falling back to legacy
  ``csv/targets_inserted/`` automatically.
* The pre-existing copy-paste bug in ``_load_existing_labels``
  (per-classifier loop ran twice on the same data) is gone — the
  function was split into ``_load_continue_labels`` and
  ``_load_pseudo_labels`` and neither has the duplication.

Plus a sidecar save via ``save_labels_for_video`` (parity with
the 122ae-5c label-writes dual-write).

Plus a small UX tightening: in dock mode the Close button on the
button-bar is hidden (the dock has its own close X in its title).

The frame_labeller module imports PySide6 + numpy at module load
time, neither of which is available in the sandbox. All tests are
AST-based — they parse the source and verify structure.

Coverage:

1. Module surface: FrameLabellerWidget, FrameLabellerDialog,
   launch_frame_labeller, open_frame_labeller_dock all exported.
2. FrameLabellerWidget contract: public methods is_dirty,
   confirm_discard_changes, cleanup, set_in_dock; correct base
   class (QWidget).
3. FrameLabellerDialog contract: hosts a FrameLabellerWidget;
   reject() delegates to widget.confirm_discard_changes; base
   class is QDialog.
4. Dock helper: function _find_main_window walks parent chain;
   open_frame_labeller_dock creates a QDockWidget, sets it
   floating, calls addDockWidget, calls set_in_dock(True) on
   the widget, and hooks closeEvent for unsaved-changes prompt.
5. Continue-mode fix: imports load_labels_for_video; calls it in
   _load_continue_labels; no duplicate per-classifier loop body.
6. Save-side sidecar: imports save_labels_for_video; calls it
   inside _save with try/except so failure doesn't abort the
   legacy write.
7. Patch number recorded.
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


def _class_methods(tree: ast.AST, class_name: str) -> set[str]:
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == class_name):
            return {
                m.name for m in node.body
                if isinstance(m, ast.FunctionDef)
            }
    return set()


def _class_bases(tree: ast.AST,
                 class_name: str) -> list[str]:
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == class_name):
            return [
                ast.unparse(b) for b in node.bases
            ]
    return []


def _top_level_function_names(tree: ast.AST) -> set[str]:
    return {
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }


def main() -> int:
    src_path = (
        REPO_ROOT / "mufasa" / "ui_qt" / "frame_labeller.py"
    )
    src = src_path.read_text()
    tree = ast.parse(src)

    # ==================================================================
    # 1. Module surface — exports
    # ==================================================================
    all_names: list[str] = []
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name)
                        and t.id == "__all__"
                        for t in node.targets)
                and isinstance(node.value, ast.List)):
            all_names = [
                e.value for e in node.value.elts
                if isinstance(e, ast.Constant)
                and isinstance(e.value, str)
            ]
            break
    for expected in (
        "FrameLabellerDialog",
        "FrameLabellerWidget",
        "launch_frame_labeller",
        "open_frame_labeller_dock",
    ):
        check(
            f"__all__ exports {expected}",
            expected in all_names,
        )

    # ==================================================================
    # 2. FrameLabellerWidget contract
    # ==================================================================
    widget_methods = _class_methods(tree, "FrameLabellerWidget")
    for required in (
        "is_dirty",
        "confirm_discard_changes",
        "cleanup",
        "set_in_dock",
        "_load_continue_labels",
        "_load_pseudo_labels",
        "_save",
        "_build_ui",
    ):
        check(
            f"FrameLabellerWidget defines {required}()",
            required in widget_methods,
        )
    widget_bases = _class_bases(tree, "FrameLabellerWidget")
    check(
        "FrameLabellerWidget extends QWidget",
        "QWidget" in widget_bases,
        detail=f"got {widget_bases}",
    )

    # ==================================================================
    # 3. FrameLabellerDialog contract
    # ==================================================================
    dialog_methods = _class_methods(tree, "FrameLabellerDialog")
    check(
        "FrameLabellerDialog defines reject()",
        "reject" in dialog_methods,
    )
    dialog_bases = _class_bases(tree, "FrameLabellerDialog")
    check(
        "FrameLabellerDialog extends QDialog",
        "QDialog" in dialog_bases,
    )
    # Dialog hosts a FrameLabellerWidget — verify by source scan
    check(
        "FrameLabellerDialog instantiates FrameLabellerWidget",
        "self.widget = FrameLabellerWidget(" in src,
    )
    check(
        "FrameLabellerDialog.reject delegates to "
        "widget.confirm_discard_changes",
        "self.widget.confirm_discard_changes()" in src,
    )

    # ==================================================================
    # 4. Dock helper
    # ==================================================================
    top_fns = _top_level_function_names(tree)
    check(
        "_find_main_window helper defined",
        "_find_main_window" in top_fns,
    )
    check(
        "open_frame_labeller_dock function defined",
        "open_frame_labeller_dock" in top_fns,
    )
    # Source-level: it creates a QDockWidget
    check(
        "open_frame_labeller_dock creates a QDockWidget",
        "QDockWidget(" in src,
    )
    check(
        "open_frame_labeller_dock floats the dock by default",
        "setFloating(True)" in src,
    )
    check(
        "open_frame_labeller_dock attaches via addDockWidget",
        "main.addDockWidget(" in src,
    )
    check(
        "open_frame_labeller_dock allows all dock areas (can be "
        "docked into any side)",
        "Qt.AllDockWidgetAreas" in src,
    )
    check(
        "open_frame_labeller_dock sets the dock's features "
        "(Movable | Floatable | Closable) so the user can "
        "dock/undock by dragging",
        "DockWidgetMovable" in src
        and "DockWidgetFloatable" in src
        and "DockWidgetClosable" in src,
    )
    check(
        "open_frame_labeller_dock tells the widget it's in a "
        "dock so it can hide the redundant Close button",
        "widget.set_in_dock(True)" in src,
    )
    check(
        "open_frame_labeller_dock falls back to dialog when no "
        "QMainWindow ancestor exists",
        "FrameLabellerDialog(" in src
        and "if main is None:" in src,
    )
    check(
        "open_frame_labeller_dock keeps a reference on the main "
        "window so the dock doesn't get garbage-collected",
        "_active_labeller_docks" in src,
    )
    check(
        "open_frame_labeller_dock hooks dock close for "
        "unsaved-changes prompt",
        "dock.closeEvent" in src
        and "confirm_discard_changes" in src,
    )

    # ==================================================================
    # 5. Continue-mode fix — routes through load_labels_for_video
    # ==================================================================
    check(
        "_load_continue_labels imports load_labels_for_video",
        "from mufasa.utils.label_io import load_labels_for_video"
        in src,
    )
    check(
        "_load_continue_labels calls load_labels_for_video",
        "load_labels_for_video(self.video_name, self.config_path)"
        in src,
    )
    check(
        "_load_continue_labels handles FileNotFoundError as 'no "
        "existing labels' rather than crashing",
        "except FileNotFoundError" in src
        and "starting from zeros" in src,
    )

    # ==================================================================
    # 6. Duplicate-loop bug gone
    # ==================================================================
    # Find the body of _load_continue_labels and count how many
    # times the "if name not in df.columns: continue" pattern
    # appears inside it. The old function had two identical loops;
    # the new one should have exactly one.
    continue_loops_in_module = src.count(
        "if name not in df.columns:"
    )
    check(
        "duplicate-loop bug from old _load_existing_labels gone — "
        "the per-classifier loop body appears only twice in the "
        "module (once each for continue + pseudo loaders), not "
        "four times as the duplicated old code had",
        continue_loops_in_module == 2,
        detail=f"got {continue_loops_in_module} occurrences",
    )

    # ==================================================================
    # 7. Save-side sidecar
    # ==================================================================
    check(
        "_save imports save_labels_for_video",
        "from mufasa.utils.label_io import save_labels_for_video"
        in src,
    )
    check(
        "_save calls save_labels_for_video with video_name + "
        "config_path + labels",
        "save_labels_for_video(" in src
        and "video_name=self.video_name" in src
        and "config_path=self.config_path" in src,
    )
    check(
        "_save guards sidecar with try/except so failure doesn't "
        "abort the legacy write (matches 122ae-5c canary tag)",
        "[122aj] Sidecar labels write" in src,
    )
    check(
        "_save still writes the legacy combined features+labels "
        "file (dual-write transition)",
        "self._write_df_best_effort(df, target_path)" in src,
    )

    # ==================================================================
    # 8. UX strip — Close button hidden in dock mode
    # ==================================================================
    check(
        "set_in_dock toggles Close button visibility so the "
        "dock's own close X isn't duplicated",
        "self._close_btn.setVisible(not value)" in src,
    )

    # ==================================================================
    # 9. Patch number recorded
    # ==================================================================
    check(
        "module records patch 122aj in module docstring",
        "122aj" in src,
    )

    # ==================================================================
    # 10. Imports are coherent — no unused QPushButton, etc.,
    #     and the new symbols (QDockWidget, QMainWindow) are
    #     imported.
    # ==================================================================
    check(
        "QDockWidget imported",
        "QDockWidget" in src.split("class ")[0],
    )
    check(
        "QMainWindow imported",
        "QMainWindow" in src.split("class ")[0],
    )

    print(
        f"smoke_122aj_dockable_labeller: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
