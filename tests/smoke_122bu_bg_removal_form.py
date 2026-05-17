"""
tests/smoke_122bu_bg_removal_form.py
======================================

Patch 122bu: Tier 2 lane 1 of the Tk → Qt consolidation plan.
Consolidates two Tk popups
(:class:`BackgroundRemoverSingleVideoPopUp`,
:class:`BackgroundRemoverDirectoryPopUp`) into a single Qt
section under Video Processing → "Background removal".

The previous `_BgRemoverPanel` inside `VideoFiltersForm` was a
stub: it surfaced ``bg_method`` and ``parallel`` kwargs that the
backend (:func:`video_bg_subtraction`) does not accept, and never
wired bg_color / fg_color / threshold / time-window / core_cnt.
This patch removes the stub and replaces it with a fully-wired
:class:`BackgroundRemovalForm` that maps to the real backend
signature.

Coverage
--------
1. New file mufasa/ui_qt/forms/video_bg_removal.py exists and
   defines BackgroundRemovalForm.
2. BackgroundRemovalForm subclasses OperationForm (AST check).
3. BackgroundRemovalForm has build(), collect_args(), and target()
   methods.
4. The form's target() imports both backend functions
   (video_bg_subtraction and video_bg_subtraction_mp).
5. The form's target() handles the directory branch via
   find_all_videos_in_directory.
6. video_filters.py no longer defines _BgRemoverPanel.
7. video_filters.py OPS list no longer contains 'bg_remove'.
8. video_filters.py target() no longer has a 'bg_remove' branch.
9. video_filters.py docstring no longer claims to replace the
   BackgroundRemover popups.
10. video_processing_page.py imports BackgroundRemovalForm.
11. video_processing_page.py registers a "Background removal"
    section.
12. The new section uses BackgroundRemovalForm.
13. video_processing_page.py is at >= 15 sections (was 14).
14. All mufasa/**/*.py files parse cleanly.
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

    # ==================================================================
    # 1-5. New form file + class shape
    # ==================================================================
    bg_form = pkg / "ui_qt" / "forms" / "video_bg_removal.py"
    check("video_bg_removal.py exists", bg_form.exists())

    if bg_form.exists():
        bg_src = bg_form.read_text()
        tree = ast.parse(bg_src)

        # Find BackgroundRemovalForm class
        form_cls = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.ClassDef)
                    and node.name == "BackgroundRemovalForm"):
                form_cls = node
                break
        check(
            "BackgroundRemovalForm class defined",
            form_cls is not None,
        )
        if form_cls is not None:
            base_names = {ast.unparse(b) for b in form_cls.bases}
            check(
                "BackgroundRemovalForm subclasses OperationForm",
                "OperationForm" in base_names,
                detail=f"got bases: {base_names}",
            )

            methods = {
                stmt.name for stmt in form_cls.body
                if isinstance(stmt, ast.FunctionDef)
            }
            for m in ("build", "collect_args", "target"):
                check(
                    f"BackgroundRemovalForm defines {m}()",
                    m in methods,
                )

        # target() imports both backend functions
        check(
            "target() imports video_bg_subtraction",
            "video_bg_subtraction" in bg_src,
        )
        check(
            "target() imports video_bg_subtraction_mp",
            "video_bg_subtraction_mp" in bg_src,
        )
        check(
            "target() handles directory branch via "
            "find_all_videos_in_directory",
            "find_all_videos_in_directory" in bg_src,
        )

        # Tk popup names referenced in the docstring as "replaces…"
        for cls in ("BackgroundRemoverSingleVideoPopUp",
                    "BackgroundRemoverDirectoryPopUp"):
            check(
                f"docstring references replaced Tk class '{cls}'",
                cls in bg_src,
            )

    # ==================================================================
    # 6-9. video_filters.py stub removed
    # ==================================================================
    vf_path = pkg / "ui_qt" / "forms" / "video_filters.py"
    vf_src = vf_path.read_text()
    check(
        "video_filters.py no longer defines _BgRemoverPanel",
        "class _BgRemoverPanel" not in vf_src,
    )
    check(
        "video_filters.py OPS list no longer contains 'bg_remove'",
        '"bg_remove"' not in vf_src and "'bg_remove'" not in vf_src,
    )
    check(
        "video_filters.py docstring no longer claims to replace "
        "BackgroundRemoverSingleVideoPopUp",
        ("Inline form replacing five filter/enhancement popups"
         in vf_src or "five filter/enhancement popups" in vf_src),
    )
    check(
        "video_filters.py no longer registers _BgRemoverPanel "
        "in the QStackedWidget",
        "_BgRemoverPanel(self)" not in vf_src,
    )

    # ==================================================================
    # 10-13. Page wiring
    # ==================================================================
    page_path = pkg / "ui_qt" / "pages" / "video_processing_page.py"
    page_src = page_path.read_text()
    check(
        "video_processing_page.py imports BackgroundRemovalForm",
        "from mufasa.ui_qt.forms.video_bg_removal import "
        "BackgroundRemovalForm" in page_src,
    )
    check(
        "video_processing_page.py registers 'Background removal' "
        "section",
        '"Background removal"' in page_src,
    )
    check(
        "Section uses BackgroundRemovalForm",
        "(BackgroundRemovalForm, {})" in page_src,
    )
    # Section count
    section_count = page_src.count("page.add_section(")
    check(
        f"video_processing_page.py has >= 15 sections "
        f"(got {section_count})",
        section_count >= 15,
    )

    # ==================================================================
    # 14. All files parse cleanly
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
        f"smoke_122bu_bg_removal_form: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
