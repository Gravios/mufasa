"""
tests/smoke_video_import_polish.py
==================================

Patch 122v: regression guard for the three Import Videos
polish items added on top of the 122o VideoImportForm scaffold:

1. **Symlink mode is the default** — the symlink checkbox is
   constructed with ``setChecked(True)`` and the form's
   description says so.
2. **Pre-flight duplicate detection** — collect_args enumerates
   candidate basenames, compares against the existing video
   tree, and prompts the user via QMessageBox before the
   backend runs.
3. **Already-imported table** — a QTableWidget with the columns
   Filename / Size / Modified / Symlink target is built, has a
   Refresh button, populates from the project's video_dir, and
   refreshes after each successful import.

AST-only — PySide6 isn't reachable in the sandbox.
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


def _find_class(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        n.name: n for n in cls.body
        if isinstance(n, ast.FunctionDef)
    }


def main() -> int:
    vi_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "video_import.py")
    src = vi_path.read_text()
    tree = ast.parse(src)

    cls = _find_class(tree, "VideoImportForm")
    check("VideoImportForm class defined", cls is not None)

    if cls is not None:
        class_src = ast.unparse(cls)

        # ------------------ 1. Symlink default ON ------------------
        # The symlink checkbox is constructed AND explicitly set to
        # checked. The form's description advertises this so users
        # see the default in the page intro.
        check(
            "VideoImportForm sets self._symlink",
            "self._symlink" in class_src,
        )
        # Look for `self._symlink.setChecked(True)` — this is what
        # turns the default on.
        check(
            "VideoImportForm sets symlink checkbox to True by "
            "default",
            "self._symlink.setChecked(True)" in class_src,
        )
        # Description string mentions symlink-by-default
        check(
            "VideoImportForm description mentions 'Symlink mode "
            "is on by default'",
            "symlink" in class_src.lower()
            and "default" in class_src.lower(),
        )

        # ------------------ 2. Duplicate detection ------------------
        # Helper methods exist
        methods = _methods(cls)
        for m in ("_existing_video_names", "_candidate_basenames",
                  "_confirm_duplicates"):
            check(
                f"VideoImportForm has {m}() helper",
                m in methods,
            )

        # collect_args invokes the duplicate-detection flow
        if "collect_args" in methods:
            ca_src = ast.unparse(methods["collect_args"])
            check(
                "collect_args() enumerates candidate basenames",
                "_candidate_basenames" in ca_src,
            )
            check(
                "collect_args() compares against the existing tree",
                "_existing_video_names" in ca_src,
            )
            check(
                "collect_args() pops the duplicate confirmation "
                "dialog when overlaps exist",
                "_confirm_duplicates" in ca_src,
            )
            check(
                "collect_args() raises RuntimeError on cancel",
                "Import cancelled" in ca_src
                or "duplicate filenames" in ca_src,
            )

        # _confirm_duplicates uses QMessageBox.question
        if "_confirm_duplicates" in methods:
            cd_src = ast.unparse(methods["_confirm_duplicates"])
            check(
                "_confirm_duplicates() pops a QMessageBox.question",
                "QMessageBox.question" in cd_src,
            )
            check(
                "_confirm_duplicates() defaults to No",
                "QMessageBox.No" in cd_src,
            )

        # ------------------ 3. Already-imported table ------------------
        check(
            "VideoImportForm creates self._table (QTableWidget)",
            "self._table = QTableWidget(" in class_src,
        )
        # Four columns named explicitly (ast.unparse single-quotes
        # strings; check both quote styles)
        for col in ("Filename", "Size", "Modified", "Symlink target"):
            check(
                f"VideoImportForm table has '{col}' column",
                f'"{col}"' in class_src or f"'{col}'" in class_src,
            )
        # _refresh_table is the populator
        check(
            "VideoImportForm has _refresh_table() helper",
            "_refresh_table" in methods,
        )
        if "_refresh_table" in methods:
            rt_src = ast.unparse(methods["_refresh_table"])
            check(
                "_refresh_table() reads _videos_dir()",
                "_videos_dir" in rt_src,
            )
            check(
                "_refresh_table() iterates iterdir() and inspects "
                "stat / symlink",
                "iterdir" in rt_src
                and ("is_symlink" in rt_src
                     or "readlink" in rt_src),
            )
            check(
                "_refresh_table() humanises sizes",
                "_humanize_bytes" in rt_src,
            )
        # target() refreshes the table after a successful import
        if "target" in methods:
            target_src = ast.unparse(methods["target"])
            check(
                "target() refreshes the table after import",
                "_refresh_table" in target_src,
            )

        # build() creates a Refresh button wired to _refresh_table
        if "build" in methods:
            b_src = ast.unparse(methods["build"])
            check(
                "build() creates a 'Refresh' button wired to "
                "_refresh_table",
                "Refresh" in b_src and "_refresh_table" in b_src,
            )

    # _humanize_bytes utility correctness
    # (Module-level helper; we can exercise it by evaluating the
    # function via exec into a fresh ns — it's pure-Python.)
    func_src = None
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_humanize_bytes"):
            func_src = ast.unparse(node)
            break
    check(
        "_humanize_bytes() defined at module scope",
        func_src is not None,
    )
    if func_src is not None:
        ns: dict = {}
        exec(func_src, ns)
        humanize = ns["_humanize_bytes"]
        check(
            "_humanize_bytes(0) → '0 B'",
            humanize(0) == "0 B",
        )
        check(
            "_humanize_bytes(1023) → '1023 B' (no premature KB)",
            humanize(1023) == "1023 B",
        )
        check(
            "_humanize_bytes(1024) → '1.0 KB'",
            humanize(1024) == "1.0 KB",
        )
        check(
            "_humanize_bytes(1024 * 1024 * 5) → '5.0 MB'",
            humanize(1024 * 1024 * 5) == "5.0 MB",
        )

    print(
        f"smoke_video_import_polish: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
