"""
tests/smoke_inline_project_create.py
====================================

Patch 122l: regression guard for the inline integration of
:class:`ProjectCreateForm` into the Projects page's
"Create or open project" frame.

Coverage (AST-only — these modules import PySide6 at module
top, which isn't always available in the sandbox):

1. ``mufasa/ui_qt/forms/project_create.py`` defines
   :class:`ProjectCreateForm` with:
   * a ``submit()`` method (the validate-and-create entry point)
   * a ``project_created`` :class:`Signal`
   * the expected field-widget attributes
   * a ``show_create_button`` kwarg on ``__init__`` so the
     modal-dialog wrapper can hide the inline button

2. :class:`CreateProjectDialog` (formerly ~300 lines of fields +
   validation) is now a slim wrapper that embeds
   ``ProjectCreateForm`` and connects its
   ``project_created`` signal to ``self.accept()``.

3. :class:`NewProjectForm` on the Projects page:
   * Drops the standalone "New project…" button (the inline
     form replaces it).
   * Keeps "Open project…" + the conditional recent shortcut.
   * Embeds ``ProjectCreateForm`` with ``show_create_button=True``.
   * Frames Open and Create as two :class:`QGroupBox` sections
     (formal section divider, matching the user's "formally
     formatted" request).
   * Wires the inline form's ``project_created`` signal to
     ``workbench._switch_to_project``.

The behavioural side (clicking Create actually creates a
project) is exercised by the existing
``smoke_config_creator_v1`` suite — ``ProjectCreateForm.submit``
delegates to ``ProjectConfigCreator`` unchanged, so that
suite's 38 assertions still apply.
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


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None  # type: ignore[return-value]


def main() -> int:
    # ------------------------------------------------------------------
    # 1. ProjectCreateForm shape
    # ------------------------------------------------------------------
    pc_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "project_create.py"
    pc_src = pc_path.read_text()
    pc_tree = ast.parse(pc_src)

    cls = _find_class(pc_tree, "ProjectCreateForm")
    check("ProjectCreateForm class defined", cls is not None)

    if cls is not None:
        methods = {
            n.name for n in cls.body if isinstance(n, ast.FunctionDef)
        }
        for required in ("__init__", "submit", "_pick_dir",
                         "_preset_changed", "_autodetect_from_dlc",
                         "_clear_autodetect"):
            check(
                f"ProjectCreateForm.{required} defined",
                required in methods,
            )

        # Signal declared at class level
        class_src = ast.unparse(cls)
        check(
            "ProjectCreateForm declares 'project_created' Signal(str)",
            "project_created" in class_src
            and "Signal(str)" in class_src,
        )

        # __init__ takes show_create_button kwarg
        init = next(
            (n for n in cls.body
             if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
            None,
        )
        if init is not None:
            arg_names = (
                [a.arg for a in init.args.args]
                + [a.arg for a in init.args.kwonlyargs]
            )
            check(
                "ProjectCreateForm.__init__ accepts show_create_button",
                "show_create_button" in arg_names,
            )

        # Field-widget attrs present
        for field in (
            "self._dir_edit", "self._name_edit", "self._preset_combo",
            "self._animal_count", "self._clf_edit",
            "self._file_type_combo", "self._autodetect_label",
            "self._autodetected_bps",
        ):
            check(
                f"ProjectCreateForm sets {field}",
                field in class_src,
            )

        # submit() emits project_created
        submit = next(
            (n for n in cls.body
             if isinstance(n, ast.FunctionDef) and n.name == "submit"),
            None,
        )
        if submit is not None:
            submit_src = ast.unparse(submit)
            check(
                "ProjectCreateForm.submit emits project_created",
                "self.project_created.emit(" in submit_src,
            )
            check(
                "ProjectCreateForm.submit invokes ProjectConfigCreator",
                "ProjectConfigCreator(" in submit_src,
            )

    # ------------------------------------------------------------------
    # 2. CreateProjectDialog is now a slim wrapper
    # ------------------------------------------------------------------
    dlg_path = REPO_ROOT / "mufasa" / "ui_qt" / "create_project_dialog.py"
    dlg_src = dlg_path.read_text()
    dlg_tree = ast.parse(dlg_src)

    dlg_cls = _find_class(dlg_tree, "CreateProjectDialog")
    check("CreateProjectDialog class defined", dlg_cls is not None)
    if dlg_cls is not None:
        dlg_src_unparsed = ast.unparse(dlg_cls)
        check(
            "CreateProjectDialog embeds ProjectCreateForm",
            "ProjectCreateForm(" in dlg_src_unparsed,
        )
        check(
            "CreateProjectDialog wires project_created to its own slot",
            "project_created.connect" in dlg_src_unparsed,
        )
        # The dialog should hide the form's own Create button —
        # it provides its own via QDialogButtonBox.
        check(
            "CreateProjectDialog disables the form's inline Create "
            "button (show_create_button=False)",
            "show_create_button=False" in dlg_src_unparsed,
        )
        # Slim: the previous ~300-line dialog body should be drastically
        # smaller now. Rough check: ProjectConfigCreator should NOT be
        # called directly in the dialog any more — that moved into the
        # form's submit().
        check(
            "CreateProjectDialog no longer calls ProjectConfigCreator "
            "directly (delegates to ProjectCreateForm.submit)",
            "ProjectConfigCreator(" not in dlg_src_unparsed,
        )

    # File-level: the dialog file shrank substantially. Use a soft
    # ceiling rather than an exact line count.
    line_count = len(dlg_src.splitlines())
    check(
        "create_project_dialog.py is now under 200 lines "
        "(was ~300 pre-122l)",
        line_count < 200,
        detail=f"got {line_count} lines",
    )

    # ------------------------------------------------------------------
    # 3. NewProjectForm wiring
    # ------------------------------------------------------------------
    info_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "project_info.py"
    info_src = info_path.read_text()
    info_tree = ast.parse(info_src)

    npf = _find_class(info_tree, "NewProjectForm")
    check("NewProjectForm class defined", npf is not None)
    if npf is not None:
        npf_src = ast.unparse(npf)
        check(
            "NewProjectForm embeds ProjectCreateForm",
            "ProjectCreateForm(" in npf_src,
        )
        check(
            "NewProjectForm passes show_create_button=True "
            "(inline button visible)",
            "show_create_button=True" in npf_src,
        )
        check(
            "NewProjectForm connects ProjectCreateForm.project_created "
            "to a slot",
            "project_created.connect" in npf_src,
        )
        check(
            "NewProjectForm uses QGroupBox for formal section framing",
            "QGroupBox(" in npf_src,
        )
        # The inline 'New project…' standalone button is removed —
        # the form itself is the create surface now.
        check(
            "NewProjectForm no longer has a standalone "
            "'New project…' button",
            "'New project…'" not in npf_src
            and '"New project…"' not in npf_src,
        )
        # Open project + Open most recent (conditional) still there
        check(
            "NewProjectForm still has the 'Open project…' button",
            "'Open project…'" in npf_src
            or '"Open project…"' in npf_src,
        )
        check(
            "NewProjectForm still surfaces the recent-project shortcut",
            "load_recent_project" in npf_src,
        )
        # The new slot routes project_created → workbench's switch
        check(
            "NewProjectForm defines _on_project_created slot",
            "_on_project_created" in npf_src,
        )
        check(
            "NewProjectForm's _on_project_created routes to "
            "workbench._switch_to_project",
            "_switch_to_project" in npf_src,
        )

    # ------------------------------------------------------------------
    # 4. QGroupBox import in project_info.py
    # ------------------------------------------------------------------
    check(
        "project_info.py imports QGroupBox",
        "QGroupBox" in info_src,
    )

    print(
        f"smoke_inline_project_create: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
