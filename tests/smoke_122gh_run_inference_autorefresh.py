"""
tests/smoke_122gh_run_inference_autorefresh.py
==============================================

Patch 122gh — Run inference table auto-refreshes when classifiers change.

USER-REPORTED BUG
=================
"If I change classifiers in Manage classifiers it doesn't update the table
in Inference: run inference."

ROOT CAUSE
==========
The classifier page is a QToolBox whose sections are instantiated lazily
and ONCE (WorkflowPage._instantiate guards on a `_instantiated` set). So
RunInferenceForm builds its per-classifier table on first expand and never
rebuilds it. "Manage classifiers" is a sibling section on the same page;
adding/removing a classifier there writes the project config but leaves the
already-built inference table stale. A manual "Reload classifier list"
button existed, but the table never refreshed on its own.

FIX
===
Override showEvent on RunInferenceForm to call the existing _reload().
QToolBox shows/hides item widgets on navigation, so navigating (back) to
the Run inference section fires showEvent and re-reads the classifier list.
_reload() snapshots current rows first and lets in-progress per-row edits
win over on-disk INI values, so refreshing on show preserves model
paths / thresholds the user already entered for classifiers that still
exist; it only adds rows for new classifiers and drops rows for removed
ones. Guarded with hasattr(self, "table") since showEvent can fire before
build() creates the table. (A popped-out floating instance stays visible
and won't get showEvent from sibling edits — the Reload button covers
that case.)

NEW SMOKE: smoke_122gh_run_inference_autorefresh.py (4 checks)
* run_inference.py parses cleanly
* RunInferenceForm defines showEvent
* showEvent calls super().showEvent AND self._reload(), guarded by
  hasattr(self, "table")
* _reload still re-reads the classifier targets (the refresh source)
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FORM = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "run_inference.py"

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
    src = FORM.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"FAIL: run_inference.py parses — {e}")
        print("smoke_122gh_run_inference_autorefresh: 0/4 checks passed")
        return 1
    check("run_inference.py parses cleanly", True)

    # locate RunInferenceForm.showEvent and RunInferenceForm._reload
    show_fn = reload_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RunInferenceForm":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == "showEvent":
                    show_fn = m
                if isinstance(m, ast.FunctionDef) and m.name == "_reload":
                    reload_fn = m

    check("RunInferenceForm defines showEvent", show_fn is not None)

    if show_fn is not None:
        body = ast.unparse(show_fn)
        check(
            "showEvent calls super().showEvent and self._reload(), "
            "guarded by hasattr(self, 'table')",
            "super().showEvent" in body
            and "self._reload()" in body
            and "hasattr(self, 'table')" in body.replace('"', "'"),
            detail=body[:160],
        )
    else:
        check("showEvent calls super().showEvent and self._reload() (guarded)", False)

    if reload_fn is not None:
        rbody = ast.unparse(reload_fn)
        check(
            "_reload still re-reads the classifier targets (refresh source)",
            "_read_classifier_targets" in rbody,
        )
    else:
        check("_reload still re-reads the classifier targets", False)

    print(
        f"smoke_122gh_run_inference_autorefresh: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
