"""
tests/smoke_annotation_polish.py
================================

Patch 122aa: regression guard for the annotation page polish.

Coverage:

1. ``FrameLabellingLauncher`` button text is the short verb form
   'Label' (was 'Select video and launch labeller…').
2. ``ClipReviewLauncher`` button text is the short verb form
   'Review' (was 'Select video and launch reviewer…').
3. ``FrameLabellingLauncher.description`` mentions all three
   path conventions the labeller uses: ``csv/features_extracted/``,
   ``csv/targets_inserted/``, ``csv/machine_results/``.
4. ``FrameLabellingLauncher.description`` acknowledges the
   layout-awareness gap (deferred run-id allocation for v1).
5. The 'No project' QMessageBox warning in both launchers
   mentions both v1 (``project.toml``) and legacy
   (``project_config.ini``) project files.
6. The 122aa patch note appears in both the
   FrameLabellingLauncher class docstring and the page
   docstring.
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
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


def main() -> int:
    ann_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "annotation.py").read_text()
    ann_tree = ast.parse(ann_src)

    # ---------- FrameLabellingLauncher ----------
    cls = _find_class(ann_tree, "FrameLabellingLauncher")
    check("FrameLabellingLauncher class defined", cls is not None)

    if cls is not None:
        class_src = ast.unparse(cls)

        # Button rename: present in BOTH single + double quotes per
        # ast.unparse output
        check(
            "FrameLabellingLauncher button text is 'Label' "
            "(short verb form)",
            "'  Label'" in class_src
            or '"  Label"' in class_src,
        )
        check(
            "FrameLabellingLauncher no longer constructs a "
            "QPushButton with the old verbose label "
            "(historical mention in the docstring is fine)",
            "QPushButton('  Select video and launch labeller"
            not in class_src
            and 'QPushButton("  Select video and launch labeller'
            not in class_src,
        )

        # Description carries path note
        for path in ("csv/features_extracted/",
                     "csv/targets_inserted/",
                     "csv/machine_results/"):
            check(
                f"FrameLabellingLauncher description mentions "
                f"{path!r}",
                path in class_src,
            )
        check(
            "FrameLabellingLauncher description acknowledges the "
            "v1 layout-awareness gap (deferred run-id allocation)",
            "deferred" in class_src.lower()
            and ("v1" in class_src or "layout" in class_src),
        )

        # Project-load warning mentions both project file kinds
        check(
            "FrameLabellingLauncher _launch warning mentions v1 "
            "'project.toml'",
            "project.toml" in class_src,
        )
        check(
            "FrameLabellingLauncher _launch warning still mentions "
            "legacy 'project_config.ini'",
            "project_config.ini" in class_src,
        )

        # 122aa patch note in class docstring
        check(
            "FrameLabellingLauncher class docstring carries the "
            "122aa note",
            "122aa" in class_src,
        )

    # ---------- ClipReviewLauncher ----------
    cls = _find_class(ann_tree, "ClipReviewLauncher")
    check("ClipReviewLauncher class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        check(
            "ClipReviewLauncher button text is 'Review' "
            "(short verb form)",
            "'  Review'" in class_src
            or '"  Review"' in class_src,
        )
        check(
            "ClipReviewLauncher no longer constructs a "
            "QPushButton with the old verbose label "
            "(historical mention in docstring is fine)",
            "QPushButton('  Select video and launch reviewer"
            not in class_src
            and 'QPushButton("  Select video and launch reviewer'
            not in class_src,
        )
        check(
            "ClipReviewLauncher _launch warning mentions v1 "
            "'project.toml'",
            "project.toml" in class_src,
        )
        check(
            "ClipReviewLauncher class docstring carries the 122aa "
            "note",
            "122aa" in class_src,
        )

    # ---------- annotation_page docstring ----------
    page_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
                / "annotation_page.py").read_text()
    check(
        "annotation_page docstring carries the 122aa note",
        "122aa" in page_src,
    )
    check(
        "annotation_page docstring documents the labeller path "
        "convention",
        "features_extracted" in page_src
        or "targets_inserted" in page_src
        or "machine_results" in page_src,
    )

    print(
        f"smoke_annotation_polish: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
