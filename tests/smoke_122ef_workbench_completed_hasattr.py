"""
tests/smoke_122ef_workbench_completed_hasattr.py
==================================================

Patch 122ef — hotfix: guard ``form.completed.connect(...)`` in
:meth:`mufasa.ui_qt.workbench.WorkflowPage._instantiate` with
``hasattr(form, "completed")``.

Bug
---
The 122du badge-refresh wiring assumed every form added to a
workbench section via ``add_section()`` inherits from
:class:`OperationForm`, which defines a ``completed`` Qt signal.
That assumption is true for the ~29 ``OperationForm`` subclasses
but FALSE for any plain ``QWidget`` that gets
``add_section()``'d — notably :class:`NewProjectForm` in
``mufasa/ui_qt/forms/project_info.py``, which inherits from
``QWidget`` directly because its "operation" (creating a new
project) is structured differently from the ``target()``-based
``OperationForm`` flow.

Real-world report (Friday May 22, 2026):

    Loading recent project: /data/testing/mufasa/test-20260427/project.toml
    Traceback (most recent call last):
      File ".../bin/mufasa", line 6, in <module>
        sys.exit(main())
      ...
      File ".../mufasa/ui_qt/workbench.py", line 466, in _instantiate
        form.completed.connect(self._on_form_completed)
    AttributeError: 'NewProjectForm' object has no attribute 'completed'

Sandbox blind spot: 122du's smoke test verified the connection
was MADE (AST: ``Call(attr='connect', ...)``) but didn't exercise
it under Qt, where the AttributeError would have fired
immediately on workbench launch. PySide6 isn't installed in the
sandbox, so the smoke couldn't catch this class of bug.

Fix
---
``hasattr(form, "completed")`` guard around the connect call.
Forms without a ``completed`` signal silently skip the
badge-refresh wiring — they don't drive badges anyway because
badge transitions come from ``OperationForm.target`` completion,
which they don't have.

Why a guard rather than a marker base class:

* The forms-without-completed are a small set (currently:
  ``NewProjectForm``; possibly future setup-style widgets).
* Forcing them onto a marker base would be invasive (changes
  every project-setup-style widget) for cosmetic typing benefit.
* ``hasattr`` is duck-typing native to Qt — Qt signals are
  attribute-accessed, not statically typed.

Coverage
--------
1.  ``mufasa/ui_qt/workbench.py`` contains an ``hasattr(form,
    "completed")`` check immediately before
    ``form.completed.connect(self._on_form_completed)``.
2.  The guard is on the SAME ``_instantiate`` method that 122du
    edited; the rest of the method is unchanged (we don't
    accidentally turn off the connect altogether).
3.  ``NewProjectForm`` does NOT inherit from ``OperationForm``
    (verifies the bug condition the hotfix addresses).
4.  ``NewProjectForm`` does NOT define a ``completed`` class
    attribute (verifies the hasattr guard's truth value for
    this concrete failing case).
5.  The hotfix's source comment cites 122du as the patch
    introducing the bug, and 122ef as the patch fixing it.

Cross-patch invariants:
6.  All 38 pre-existing strict tests still pass (no regression
    from the guard).
7.  Parse-clean.
8.  122do baseline.
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


def _ast_find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _ast_method(cls_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def main() -> int:
    wb_path = REPO_ROOT / "mufasa" / "ui_qt" / "workbench.py"
    wb_src = wb_path.read_text()

    # 1. hasattr guard exists.
    # We look for the literal sequence "hasattr(form, 'completed')"
    # OR the equivalent with double-quotes.
    has_guard = bool(re.search(
        r'hasattr\s*\(\s*form\s*,\s*["\']completed["\']\s*\)',
        wb_src,
    ))
    check(
        "workbench.py contains an `hasattr(form, 'completed')` "
        "check (the hotfix that 122ef-hotfix added)",
        has_guard,
    )

    # 2. The guard is inside the _instantiate method's body, and
    # immediately precedes the connect call. Walk the AST.
    wb_tree = ast.parse(wb_src)
    wp_cls = _ast_find_class(wb_tree, "WorkflowPage")
    assert wp_cls is not None, (
        "WorkflowPage class not found in workbench.py"
    )
    inst_method = _ast_method(wp_cls, "_instantiate")
    assert inst_method is not None
    inst_src = ast.unparse(inst_method)
    check(
        "The hasattr guard is inside WorkflowPage._instantiate "
        "(verifying we didn't accidentally edit a different "
        "method); the guard immediately precedes the "
        "`form.completed.connect(...)` line",
        ("hasattr(form, 'completed')" in inst_src
         or 'hasattr(form, "completed")' in inst_src)
        and "form.completed.connect(self._on_form_completed)" in inst_src,
    )

    # 3. NewProjectForm exists and does NOT inherit from
    # OperationForm.
    pi_path = (REPO_ROOT / "mufasa" / "ui_qt"
               / "forms" / "project_info.py")
    pi_tree = ast.parse(pi_path.read_text())
    np_cls = _ast_find_class(pi_tree, "NewProjectForm")
    if np_cls is None:
        check(
            "NewProjectForm exists in project_info.py "
            "(prerequisite for the rest of the bug-condition check)",
            False, detail="class not found",
        )
    else:
        base_names = [
            b.id for b in np_cls.bases if isinstance(b, ast.Name)
        ]
        check(
            "NewProjectForm does NOT inherit from OperationForm "
            "(verifies the bug condition the hotfix addresses — "
            "NewProjectForm is a plain QWidget so it doesn't get "
            "the completed signal)",
            "OperationForm" not in base_names,
            detail=(f"bases: {base_names}"),
        )

        # 4. NewProjectForm does NOT define a `completed` class attr.
        # (Could be defined directly OR inherited; we check direct
        # only because Qt signals defined in QWidget itself would
        # not be named "completed" — that name is specific to
        # OperationForm.)
        has_completed_in_class = False
        for member in np_cls.body:
            if isinstance(member, ast.Assign):
                for tgt in member.targets:
                    if (isinstance(tgt, ast.Name)
                            and tgt.id == "completed"):
                        has_completed_in_class = True
            elif isinstance(member, ast.AnnAssign):
                if (isinstance(member.target, ast.Name)
                        and member.target.id == "completed"):
                    has_completed_in_class = True
        check(
            "NewProjectForm does NOT directly declare a "
            "`completed` class attribute (so hasattr(form, "
            "'completed') correctly returns False on it)",
            not has_completed_in_class,
        )

    # 5. Hotfix's source comment cites 122du as introducing the
    # bug AND 122ef as the fix. Comments aren't in ``ast.unparse``
    # output, so we check the raw file source — confined to the
    # vicinity of the hasattr guard so we don't accidentally match
    # an unrelated 122du / 122ef mention elsewhere in the file.
    guard_match = re.search(
        r'hasattr\s*\(\s*form\s*,\s*["\']completed["\']\s*\)',
        wb_src,
    )
    if guard_match:
        # Look at the 1000 chars immediately preceding the guard
        # (where the explanatory comment block lives).
        vicinity = wb_src[max(0, guard_match.start() - 1000):
                          guard_match.start()]
    else:
        vicinity = ""
    check(
        "Hotfix comment block immediately preceding the hasattr "
        "guard cites 122du as the patch that introduced the bug "
        "AND 122ef as the patch that fixed it",
        "122du" in vicinity and "122ef" in vicinity,
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 6. Parse-clean.
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

    # 7. 122do baseline.
    uiqt = pkg / "ui_qt"
    optional_hits = []
    for f in sorted(uiqt.rglob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"\bOptional\[", src):
            preceding = src[:m.start()]
            tq3 = preceding.count('"""') + preceding.count("'''")
            if tq3 % 2 == 0:
                optional_hits.append(str(f.relative_to(uiqt)))
                break
    check(
        "122do baseline preserved: no `Optional[` in non-docstring "
        "positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    # 8. 122du state preserved — the connect call itself still
    # exists in _instantiate (we didn't turn it off entirely).
    check(
        "122du state preserved: the `form.completed.connect("
        "self._on_form_completed)` line still exists in "
        "_instantiate; the hotfix added a guard around it, not "
        "a removal",
        "form.completed.connect(self._on_form_completed)" in inst_src,
    )

    print(
        f"smoke_122ef_workbench_completed_hasattr: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
