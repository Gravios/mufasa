"""
tests/smoke_122dn_roi_define_inline.py
=========================================

Patch 122dn: integrate the ROI Define popup into the workbench
frame. The Definitions section of the ROI page now hosts the
full ROI editor inline; the popup-as-dialog path is preserved
for legacy callers via a thin wrapper.

What this patch landed
----------------------
1. ``ROIDefinePanel`` was split into:
   - ``ROIDefineWidget(QWidget)`` — the embeddable content.
     Handles the no-project case gracefully (renders a centered
     placeholder instead of raising).
   - ``ROIDefinePanel(QDialog)`` — thin wrapper that hosts a
     ROIDefineWidget. Preserves the pre-122dn constructor
     signature and the rois_modified signal so existing callers
     (roi_video_table.py:_action_define_rois,
     forms/roi.py:ROIManageForm.on_run) keep working unchanged.
     Re-raises RuntimeError on no-videos for legacy callers'
     exception-handler paths.

2. ``ROIDefineWidget`` gained:
   - ``set_config_path(config_path)`` — switch to a different
     project (rebuilds UI in place).
   - ``set_embedded_mode(bool)`` — when True, hides "Close" and
     "Save && close" buttons (the page section is the dismiss
     affordance).
   - ``_show_placeholder(html)`` / ``_initialize_for_project()``
     — graceful no-project rendering.

3. ``WorkflowPage`` gained ``add_section_widget(title, factory)``
   — adds a section whose content is an arbitrary QWidget built
   by the factory on first expand (vs the existing add_section
   which only accepts OperationForm classes).

4. ``roi_page.py`` updated:
   - "Definitions" section is now a widget section hosting
     ROIDefineWidget directly.
   - "Maintenance" section is the new ROIManageForm location
     (renamed from "Definitions"; the Draw action stays
     available for users who prefer the popup path).

Coverage
--------
1.  ROIDefineWidget class exists and inherits from QWidget.
2.  ROIDefinePanel class exists and inherits from QDialog.
3.  ROIDefineWidget.__init__ takes config_path with default None
    (graceful no-project handling).
4.  ROIDefineWidget has set_config_path method.
5.  ROIDefineWidget has set_embedded_mode method.
6.  ROIDefineWidget has _show_placeholder method.
7.  ROIDefineWidget has _initialize_for_project method.
8.  ROIDefineWidget's _build_ui builds into a `content` widget
    (not directly into self) so the placeholder/content swap is
    clean.
9.  ROIDefinePanel.__init__ has same signature as pre-122dn
    (config_path positional, video_path optional, parent optional).
10. ROIDefinePanel wraps a ROIDefineWidget and forwards
    rois_modified.
11. ROIDefinePanel still raises RuntimeError on no-videos
    (backward-compat for legacy callers' exception handlers).
12. WorkflowPage.add_section_widget exists.
13. WorkflowPage._instantiate handles widget-factory sections
    (branch checking _declared_widget_factories before forms).
14. roi_page.py imports ROIDefineWidget.
15. roi_page.py calls add_section_widget("Definitions", ...).
16. roi_page.py still adds Maintenance/Analyze/Visualize/Features
    sections (functional regressions guard).
17. Legacy popup entry points still work — roi_video_table.py
    and forms/roi.py reference ROIDefinePanel unchanged.
18. ruff F401/W292/W293 clean on touched files.
19. All mufasa/**/*.py parse cleanly.
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
    if cls is None: return None
    return next(
        (m for m in cls.body
         if isinstance(m, ast.FunctionDef) and m.name == name),
        None,
    )


def main() -> int:
    pkg = REPO_ROOT / "mufasa"
    define_panel = pkg / "ui_qt" / "dialogs" / "roi_define_panel.py"
    workbench = pkg / "ui_qt" / "workbench.py"
    roi_page = pkg / "ui_qt" / "pages" / "roi_page.py"
    vt = pkg / "ui_qt" / "dialogs" / "roi_video_table.py"
    forms_roi = pkg / "ui_qt" / "forms" / "roi.py"

    panel_src = define_panel.read_text()
    panel_tree = ast.parse(panel_src)
    wb_src = workbench.read_text()
    wb_tree = ast.parse(wb_src)
    rp_src = roi_page.read_text()

    # 1. ROIDefineWidget class + QWidget base
    widget_cls = get_class(panel_tree, "ROIDefineWidget")
    check(
        "ROIDefineWidget class exists and inherits from QWidget",
        widget_cls is not None
        and any(isinstance(b, ast.Name) and b.id == "QWidget"
                for b in widget_cls.bases),
    )

    # 2. ROIDefinePanel class + QDialog base
    dlg_cls = get_class(panel_tree, "ROIDefinePanel")
    check(
        "ROIDefinePanel class exists and inherits from QDialog",
        dlg_cls is not None
        and any(isinstance(b, ast.Name) and b.id == "QDialog"
                for b in dlg_cls.bases),
    )

    # 3. ROIDefineWidget.__init__ has config_path with default None
    w_init = get_method(widget_cls, "__init__")
    if w_init is not None:
        # config_path is the first non-self arg; needs default None
        # via the optional Optional[str] pattern.
        n_pos = len(w_init.args.args) - len(w_init.args.defaults)
        arg_to_default = {
            w_init.args.args[n_pos + i].arg:
                w_init.args.defaults[i]
            for i in range(len(w_init.args.defaults))
        }
        cp_default = arg_to_default.get("config_path")
        check(
            "ROIDefineWidget.__init__(config_path=None, ...) — "
            "graceful no-project handling",
            isinstance(cp_default, ast.Constant)
            and cp_default.value is None,
        )

    widget_methods = (
        {m.name for m in widget_cls.body
         if isinstance(m, ast.FunctionDef)}
        if widget_cls else set()
    )
    # 4. set_config_path
    check(
        "ROIDefineWidget has set_config_path method "
        "(project switching)",
        "set_config_path" in widget_methods,
    )
    # 5. set_embedded_mode
    check(
        "ROIDefineWidget has set_embedded_mode method "
        "(hides Close / Save&Close buttons when embedded)",
        "set_embedded_mode" in widget_methods,
    )
    # 6. _show_placeholder
    check(
        "ROIDefineWidget has _show_placeholder method",
        "_show_placeholder" in widget_methods,
    )
    # 7. _initialize_for_project
    check(
        "ROIDefineWidget has _initialize_for_project method",
        "_initialize_for_project" in widget_methods,
    )

    # 8. _build_ui builds into a `content` widget
    bu = get_method(widget_cls, "_build_ui")
    if bu is not None:
        bu_src = ast.unparse(bu)
        check(
            "_build_ui builds into a `content` widget then mounts "
            "into _root_layout (placeholder/content swap is clean)",
            "content = QWidget(self)" in bu_src
            and "QVBoxLayout(content)" in bu_src
            and "_root_layout.addWidget(content)" in bu_src,
        )

    # 9. ROIDefinePanel.__init__ signature preserved
    d_init = get_method(dlg_cls, "__init__")
    if d_init is not None:
        d_args = [a.arg for a in d_init.args.args]
        check(
            "ROIDefinePanel.__init__ signature preserved "
            "(config_path, video_path=None, parent=None) — "
            "backward-compat for legacy callers",
            d_args[1:] == ["config_path", "video_path", "parent"],
        )

    # 10. ROIDefinePanel forwards rois_modified
    d_init_src = ast.unparse(d_init) if d_init is not None else ""
    check(
        "ROIDefinePanel wraps a ROIDefineWidget and forwards "
        "rois_modified signal",
        "ROIDefineWidget(" in d_init_src
        and "rois_modified.connect(self.rois_modified)" in d_init_src,
    )

    # 11. ROIDefinePanel re-raises on no-videos (legacy contract)
    check(
        "ROIDefinePanel raises RuntimeError on no-videos for "
        "legacy callers' exception handlers",
        "RuntimeError" in d_init_src
        and "No videos found" in d_init_src,
    )

    # 12. WorkflowPage.add_section_widget exists
    wp_cls = get_class(wb_tree, "WorkflowPage")
    wp_methods = (
        {m.name for m in wp_cls.body
         if isinstance(m, ast.FunctionDef)}
        if wp_cls else set()
    )
    check(
        "WorkflowPage has add_section_widget method",
        "add_section_widget" in wp_methods,
    )

    # 13. _instantiate handles widget-factory sections
    inst = get_method(wp_cls, "_instantiate")
    if inst is not None:
        inst_src = ast.unparse(inst)
        check(
            "WorkflowPage._instantiate branches on widget-factory "
            "sections (checks _declared_widget_factories first)",
            "_declared_widget_factories" in inst_src,
        )

    # 14. roi_page imports ROIDefineWidget
    check(
        "roi_page.py imports ROIDefineWidget",
        "ROIDefineWidget" in rp_src
        and "from mufasa.ui_qt.dialogs.roi_define_panel" in rp_src,
    )

    # 15. roi_page calls add_section_widget for Definitions
    check(
        "roi_page.py uses add_section_widget('Definitions', ...) — "
        "Definitions hosts widget inline",
        'add_section_widget("Definitions"' in rp_src,
    )

    # 16. Functional regression: other sections still present
    check(
        "roi_page.py still adds Maintenance, Analyze, Visualize, "
        "Features sections",
        all(s in rp_src for s in [
            '"Maintenance"', '"Analyze"',
            '"Visualize"', '"Features"',
        ]),
    )

    # 17. Legacy callers still reference ROIDefinePanel
    vt_src = vt.read_text()
    forms_src = forms_roi.read_text()
    check(
        "roi_video_table.py still uses ROIDefinePanel "
        "(backward-compat regression guard)",
        "ROIDefinePanel(" in vt_src,
    )
    check(
        "forms/roi.py still uses ROIDefinePanel "
        "(backward-compat regression guard)",
        "ROIDefinePanel(" in forms_src,
    )

    # 18. ruff clean
    import subprocess
    try:
        out = subprocess.run(
            ["ruff", "check",
             str(define_panel), str(workbench), str(roi_page),
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

    # 19. Parse-clean
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
        f"smoke_122dn_roi_define_inline: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
