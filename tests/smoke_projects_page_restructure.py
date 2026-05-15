"""
tests/smoke_projects_page_restructure.py
========================================

Patch 122i: regression guard for the Projects-page restructure
and the relocation of Batch pre-process videos to the Data
Import page.

Coverage (AST-only — these modules import PySide6 at module top
which isn't always available in the sandbox):

1. ``project_setup_page.build_project_setup_page`` registers the
   page under the label "Projects" (not the historical
   "Project setup").
2. It branches on ``config_path``: when set, the first section is
   "Project information" with ProjectInfoForm; when not set, the
   first section is "Create or open project" with NewProjectForm
   (and the workbench reference is passed through).
3. It no longer adds a "Batch pre-process videos" section.
4. ``data_import_page.build_data_import_page`` adds a
   "Batch pre-process videos" section with BatchPreProcessLauncher.
5. ProjectInfoForm + NewProjectForm exist as plain QWidget
   subclasses (not OperationForm — they don't run operations) with
   the expected constructor surface.

These checks pin the structure so a careless future edit doesn't
silently re-introduce the duplication or lose the empty-state
surface.
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
    # ------------------------------------------------------------------
    # 1. Projects page label + section layout
    # ------------------------------------------------------------------
    proj_path = (
        REPO_ROOT / "mufasa" / "ui_qt" / "pages" / "project_setup_page.py"
    )
    src = proj_path.read_text()
    tree = ast.parse(src)

    check(
        "project_setup_page uses 'Projects' as the page label",
        'add_page("Projects"' in src,
    )
    check(
        "project_setup_page does NOT use the historical "
        "'Project setup' label",
        'add_page("Project setup"' not in src,
    )
    check(
        "project_setup_page no longer adds 'Batch pre-process videos'",
        'add_section("Batch pre-process videos"' not in src,
    )
    check(
        "project_setup_page no longer imports BatchPreProcessLauncher",
        "BatchPreProcessLauncher" not in src,
    )
    check(
        "project_setup_page imports ProjectInfoForm",
        "ProjectInfoForm" in src,
    )
    check(
        "project_setup_page imports NewProjectForm",
        "NewProjectForm" in src,
    )

    # Inspect the build function for the if/else branch
    builder = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "build_project_setup_page"):
            builder = node
            break
    check("build_project_setup_page exists", builder is not None)
    if builder is not None:
        body_src = ast.unparse(builder)
        # NewProjectForm should be registered unconditionally
        # (always at the top of the page after patch 122k), not
        # gated behind an `if not config_path:` branch.
        check(
            "builder registers 'Create or open project' "
            "unconditionally (always at top, post-122k)",
            "Create or open project" in body_src
            and "NewProjectForm" in body_src,
        )
        check(
            "builder passes workbench reference to NewProjectForm",
            "'workbench': workbench" in body_src
            or '"workbench": workbench' in body_src,
        )
        # Project information section gated on config_path
        check(
            "builder branches on config_path to register "
            "'Project information' (only when a project is loaded)",
            "if config_path" in body_src
            and "Project information" in body_src
            and "ProjectInfoForm" in body_src,
        )
        # Focus the Project information section when present —
        # users continuing on a recent project want to land there.
        check(
            "builder focuses Project information via setCurrentIndex(1)",
            "setCurrentIndex(1)" in body_src,
        )
        check(
            "builder no longer registers 'Archive processed files' "
            "(removed in patch 122m — legacy model doesn't fit v1 "
            "per-run provenance)",
            "Archive processed files" not in body_src
            and "ArchiveFilesForm" not in body_src,
        )
        # Sanity: no stray duplicate add_section for the Create
        # surface. Count only quoted occurrences — the function
        # docstring may mention the section name in prose.
        check(
            "builder doesn't duplicate add_section('Create or open project', ...)",
            body_src.count("'Create or open project'") == 1
            or body_src.count('"Create or open project"') == 1,
        )

    # ------------------------------------------------------------------
    # 2. Batch pre-process moved off Project Setup. As of:
    #    - 122i: relocated to Data Import page.
    #    - 122x: moved again, now to the Preprocessing page where it
    #            sits alongside the other upstream-of-features stages.
    #    Both moves agree the section is NOT on Project Setup; this
    #    block now verifies it's on Preprocessing (the current home).
    # ------------------------------------------------------------------
    pcp_path = (
        REPO_ROOT / "mufasa" / "ui_qt" / "pages"
        / "pose_cleanup_page.py"
    )
    pcp_src = pcp_path.read_text()
    check(
        "Preprocessing page imports BatchPreProcessForm "
        "(post-122x home; class renamed from "
        "BatchPreProcessLauncher → BatchPreProcessForm during "
        "the Qt port)",
        "BatchPreProcessForm" in pcp_src,
    )
    check(
        "Preprocessing page registers 'Preprocess Videos' section "
        "(post-122x rename)",
        '"Preprocess Videos"' in pcp_src,
    )

    # Data Import page should now host the renamed Pose Data section
    # (122w) but NOT the calibration / batch sections (moved in 122x).
    di_path = (
        REPO_ROOT / "mufasa" / "ui_qt" / "pages" / "data_import_page.py"
    )
    di_src = di_path.read_text()
    check(
        "data_import_page has 'Import Pose Data' (post-122w rename)",
        '"Import Pose Data"' in di_src,
    )
    check(
        "data_import_page no longer registers 'Video parameters & "
        "calibration' (moved to Preprocessing in 122x)",
        'add_section("Video parameters & calibration"' not in di_src,
    )

    # ------------------------------------------------------------------
    # 3. Project_info forms — shape
    # ------------------------------------------------------------------
    info_path = (
        REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "project_info.py"
    )
    info_src = info_path.read_text()
    info_tree = ast.parse(info_src)

    classes = {
        n.name: n for n in info_tree.body
        if isinstance(n, ast.ClassDef)
    }
    check("ProjectInfoForm defined", "ProjectInfoForm" in classes)
    check("NewProjectForm defined", "NewProjectForm" in classes)

    if "ProjectInfoForm" in classes:
        pif = classes["ProjectInfoForm"]
        bases = [
            (b.id if isinstance(b, ast.Name) else getattr(b, "attr", ""))
            for b in pif.bases
        ]
        check(
            "ProjectInfoForm extends QWidget (not OperationForm)",
            "QWidget" in bases and "OperationForm" not in bases,
        )
        methods = {
            n.name for n in pif.body if isinstance(n, ast.FunctionDef)
        }
        # __init__ must accept parent + config_path
        if "__init__" in {n.name for n in pif.body
                          if isinstance(n, ast.FunctionDef)}:
            init = next(
                n for n in pif.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            )
            arg_names = [a.arg for a in init.args.args]
            check(
                "ProjectInfoForm.__init__ accepts parent + config_path",
                "parent" in arg_names and "config_path" in arg_names,
            )
        # Must have a _populate / refresh path
        check(
            "ProjectInfoForm has a _populate method",
            "_populate" in methods,
        )

    if "NewProjectForm" in classes:
        npf = classes["NewProjectForm"]
        bases = [
            (b.id if isinstance(b, ast.Name) else getattr(b, "attr", ""))
            for b in npf.bases
        ]
        check(
            "NewProjectForm extends QWidget (not OperationForm)",
            "QWidget" in bases and "OperationForm" not in bases,
        )
        init = next(
            (n for n in npf.body
             if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
            None,
        )
        if init is not None:
            arg_names = [a.arg for a in init.args.args]
            check(
                "NewProjectForm.__init__ accepts parent + config_path "
                "+ workbench",
                "parent" in arg_names
                and "config_path" in arg_names
                and "workbench" in arg_names,
            )
        body_src = ast.unparse(npf)
        check(
            "NewProjectForm wires New project button to "
            "workbench._on_new_project",
            "_on_new_project" in body_src,
        )
        check(
            "NewProjectForm wires Open project button to "
            "workbench._on_open_project",
            "_on_open_project" in body_src,
        )
        check(
            "NewProjectForm surfaces recent-project quick-open",
            "load_recent_project" in body_src,
        )

    print(
        f"smoke_projects_page_restructure: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
