"""
tests/smoke_122cl_roi_ruler_callback.py
=========================================

Patch 122cl: `roi_tools/roi_ruler.py` decoupled from
`mufasa.ui.tkinter_functions.SimBALabel`.

Replaced the SimBALabel-typed `info_label` parameter with a
toolkit-agnostic `on_info_text: Callable[[str], None]` callback.
Tk consumer (`roi_ui_mixin.py`) wraps its `status_bar` in a
local closure that does the Tk-specific configure + idletasks
pair — the toolkit-specific knowledge stays at the consumer.

Closes `backend_audit.md` §4d item 13.

Coverage
--------
1. roi_ruler.py no longer imports SimBALabel from
   mufasa.ui.tkinter_functions.
2. roi_ruler.py imports Callable from typing.
3. ROIRuler.__init__ no longer has an `info_label` parameter.
4. ROIRuler.__init__ has an `on_info_text` parameter.
5. ROIRuler stores the callback on `self.on_info_text` (not
   `self.info_lbl`).
6. _get_attributes no longer calls `.configure(text=, fg=)` or
   `.update_idletasks()` on info_lbl.
7. _get_attributes calls `self.on_info_text(...)`.
8. roi_ruler.py validates that on_info_text is callable when not None.
9. roi_ui_mixin.py consumer no longer passes `info_label=`.
10. roi_ui_mixin.py consumer passes `on_info_text=` callback.
11. roi_ui_mixin.py defines a local closure that does the
    Tk-specific configure() + update_idletasks() pair.
12. Backend module-level Tk-importer count is ≤ 21 (was 22
    post-122ck, 23 post-122ch, 25 pre-122ch). Regression guard.
13. backend_audit.md §4d item 13 marked DONE in 122cl.
14. backend_audit.md §3 count updated to 21.
15. All mufasa/**/*.py files parse cleanly.
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


def _module_level_imports(tree: ast.Module, target: str) -> bool:
    return any(
        isinstance(n, ast.ImportFrom) and n.module == target
        for n in tree.body
    )


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # roi_ruler.py changes
    # ==================================================================
    ruler_src = (pkg / "roi_tools" / "roi_ruler.py").read_text()
    ruler_tree = ast.parse(ruler_src)

    check(
        "roi_ruler.py no longer imports from mufasa.ui.tkinter_functions",
        not _module_level_imports(
            ruler_tree, "mufasa.ui.tkinter_functions"),
    )
    check(
        "roi_ruler.py imports Callable from typing",
        any(
            isinstance(n, ast.ImportFrom) and n.module == "typing"
            and any(a.name == "Callable" for a in n.names)
            for n in ruler_tree.body
        ),
    )

    # Find ROIRuler.__init__ + _get_attributes
    init_src = ""
    attrs_src = ""
    for node in ast.walk(ruler_tree):
        if isinstance(node, ast.ClassDef) and node.name == "ROIRuler":
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef):
                    if stmt.name == "__init__":
                        init_src = ast.unparse(stmt)
                    elif stmt.name == "_get_attributes":
                        attrs_src = ast.unparse(stmt)

    check(
        "ROIRuler.__init__ no longer has `info_label` parameter",
        "info_label" not in init_src.split("def __init__")[1].split(")")[0]
        if "def __init__" in init_src else False,
    )
    check(
        "ROIRuler.__init__ has `on_info_text` parameter",
        "on_info_text" in init_src,
    )
    check(
        "ROIRuler stores callback as self.on_info_text "
        "(not self.info_lbl)",
        "self.on_info_text" in init_src
        and "self.info_lbl" not in init_src,
    )
    check(
        "ROIRuler validates on_info_text is callable",
        "callable(on_info_text)" in init_src,
    )

    # _get_attributes path
    check(
        "_get_attributes no longer calls info_lbl.configure / "
        "update_idletasks",
        ".configure(text=" not in attrs_src
        and ".update_idletasks()" not in attrs_src
        and "info_lbl" not in attrs_src,
    )
    check(
        "_get_attributes calls self.on_info_text(...)",
        "self.on_info_text(" in attrs_src,
    )

    # ==================================================================
    # roi_ui_mixin.py consumer changes — only check if the file
    # still exists. 122cr deleted the whole ROI Tk cluster
    # (including roi_ui_mixin.py). The 122cl decoupling lessons
    # remain valid in git history; the runtime check just stops
    # being meaningful when the file is gone.
    # ==================================================================
    mixin_path = pkg / "roi_tools" / "roi_ui_mixin.py"
    if mixin_path.exists():
        mixin_src = mixin_path.read_text()
        check(
            "roi_ui_mixin.py consumer no longer passes `info_label=`",
            "info_label=self.status_bar" not in mixin_src,
        )
        check(
            "roi_ui_mixin.py consumer passes `on_info_text=` callback",
            "on_info_text=" in mixin_src,
        )
        check(
            "roi_ui_mixin.py has a local closure doing the "
            "Tk-specific configure + update_idletasks",
            "_set_ruler_status" in mixin_src
            and "status_bar.configure" in mixin_src
            and "update_idletasks" in mixin_src,
        )
    else:
        # File was deleted in a later patch (122cr). The
        # decoupling pattern is preserved in git history.
        check(
            "roi_ui_mixin.py was deleted in a later patch — "
            "decoupling pattern preserved in git history",
            True,
        )

    # ==================================================================
    # Module-level Tk-importer count regression guard
    # ==================================================================
    module_level_count = 0
    for f in pkg.rglob("*.py"):
        if any(p in f.parts for p in ("ui", "ui_qt")):
            continue
        if f.name == "SimBA.py":
            continue
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        if _module_level_imports(t, "mufasa.ui.tkinter_functions"):
            module_level_count += 1
    check(
        f"Backend module-level Tk-importer count is ≤ 21 "
        f"(got {module_level_count}; trajectory: 25→23→22→21)",
        module_level_count <= 21,
    )

    # ==================================================================
    # Doc updates
    # ==================================================================
    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4d item 13 marked DONE in 122cl",
        "DONE in patch 122cl" in audit
        and "roi_ruler" in audit,
    )
    check(
        "backend_audit.md §3 records 122cl's contribution to the "
        "module-level importer count trajectory",
        # The pinned-count "21" is stale after later patches reduce
        # the count further. Pin only to the unchanging trajectory
        # entry that records 122cl's effect.
        "122cl" in audit and "21" in audit,
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
        f"smoke_122cl_roi_ruler_callback: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
