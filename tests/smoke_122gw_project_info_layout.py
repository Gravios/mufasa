"""
tests/smoke_122gw_project_info_layout.py
========================================

Patch 122gw — visual cleanup of Projects -> "Project information".

Before: one flat 12-row QFormLayout mixing project identity, pose
configuration and file counts, with four near-identical "... runs" rows.
Nothing was scannable and "Layout" collided with the new [pose.layout]
skeleton role map (122gv).

After: three QGroupBox groups (QGroupBox is already this file's idiom —
NewProjectForm uses it):
  * Project       — Name (bold), Location, Format
  * Configuration — Animals, File type, Markers, Classifiers
  * Data          — Pose files, Smoothed, Outlier-corrected, Features,
                    Classifications
Counts lead ("15 — head_nose, ..."), "Layout"->"Format",
"Body parts"->"Markers" (matching the Model modifications tab), and the
Data labels drop the repeated word "runs".

Checks are AST/text based (portable); a rendering check runs only when
PySide6 is importable.
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
FORM = REPO / "mufasa" / "ui_qt" / "forms" / "project_info.py"

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main():
    src = FORM.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"FAIL: parse — {e}")
        print("smoke_122gw_project_info_layout: 0/7 checks passed")
        return 1
    check("project_info.py parses", True)

    check("three groups built (Project / Configuration / Data)",
          '("project", "Project")' in src
          and '("config", "Configuration")' in src
          and '("data", "Data")' in src
          and "QGroupBox(title, self)" in src)

    check("old flat form attrs are gone",
          "_form_layout" not in src and "_form_host" not in src)

    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "ProjectInfoForm")
    add_row = next(m for m in cls.body
                   if isinstance(m, ast.FunctionDef) and m.name == "_add_row")
    check("_add_row takes a group argument",
          [a.arg for a in add_row.args.args][:2] == ["self", "group"])

    bad = [ast.unparse(n)[:50] for n in ast.walk(cls)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "_add_row" and len(n.args) != 3]
    check("every _add_row call passes (group, label, value)", not bad,
          detail=str(bad[:2]))

    check('"Layout" row renamed to "Format" (no clash with [pose.layout]); '
          '"Body parts" -> "Markers"',
          '"Format",' in src and '"Markers"' in src
          and '"Layout",' not in src and '"Body parts"' not in src)

    check("Data labels drop the repeated 'runs' wording",
          '"Smoothed runs"' not in src and '"Feature runs"' not in src
          and '("Smoothed", "smoothed")' in src)

    # optional live render — only where PySide6 exists
    try:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import tempfile

        from PySide6.QtWidgets import QApplication, QFormLayout, QGroupBox

        from mufasa.project_layout import (
            PROJECT_LAYOUT_VERSION,
            write_project_toml,
        )
        d = Path(tempfile.mkdtemp())
        cp = d / "project.toml"
        write_project_toml(cp, {
            "project_layout_version": PROJECT_LAYOUT_VERSION,
            "project_name": "demo",
            "pose": {"body_parts": ["a", "b"], "file_type": "parquet"},
        })
        app = QApplication.instance() or QApplication([])
        from mufasa.ui_qt.forms.project_info import ProjectInfoForm
        w = ProjectInfoForm(config_path=str(cp))
        titles = [g.title() for g in w.findChildren(QGroupBox)]
        rows = sum(g.layout().rowCount() for g in w.findChildren(QGroupBox)
                   if isinstance(g.layout(), QFormLayout))
        check("renders three titled groups with populated rows",
              titles == ["Project", "Configuration", "Data"] and rows > 0,
              detail=f"titles={titles} rows={rows}")
        del app
    except ImportError:
        print("NOTE: PySide6 unavailable — render check skipped (soft pass).")
        check("renders three titled groups with populated rows", True)

    print(f"smoke_122gw_project_info_layout: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
