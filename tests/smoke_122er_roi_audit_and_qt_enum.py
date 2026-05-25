"""
tests/smoke_122er_roi_audit_and_qt_enum.py
=============================================

Patch 122er-hotfix — two bugs from a single user report
during manual testing (Mon May 25, 2026):

1. ``KeyError: 'Video'`` when clicking Run in the ROI page's
   "Apply ROIs to selected videos" dialog on a rectangles-
   only project. Caused by ``get_roi_data_for_video_name``
   (and a sister function ``get_roi_data``) in
   ``mufasa/roi_tools/roi_utils.py`` doing naked
   ``df['Video']`` filters that were MISSED in the 122ek
   audit.

2. ``AttributeError: 'PixelCalibrationDialog' object has no
   attribute 'Accepted'`` (from the earlier launch's
   terminal output, visible above the screenshot). Caused
   by ``dlg.Accepted`` instance-attribute access on a Qt
   enum that lives on the class in PySide6.

Both bugs are documented as KNOWN classes that earlier
patches should have caught:

* Bug 1 is the SAME class as 122ej + 122ek (empty-DataFrame
  KeyError when only one ROI shape type was drawn). 122ek
  audited roi_utils.py and found ``multiply_ROIs`` and
  ``reset_video_ROIs`` but MISSED ``get_roi_data`` and
  ``get_roi_data_for_video_name`` — both have the same
  shape of bug.

* Bug 2 has an explicit comment block at workbench.py:750-752
  documenting the pattern, citing PySide6 semantics: "the
  enum lives on the class, not the instance — dlg.Accepted
  raises AttributeError." That fix only addressed ONE call
  site; two others (video_info.py and video_processing_page.py)
  still used the buggy form.

Lesson — both are instances of "class-of-bug noted, audit
incomplete." Adds to the session's running pattern of "find
the class, then audit for instances."

What this patch landed
----------------------
mufasa/roi_tools/roi_utils.py:

* ``get_roi_data`` (lines 315-338): 6 naked ``df[df['Video']
  == ...]`` / ``df[df['Video'] != ...]`` filters + 3 naked
  ``df['Video'].unique()`` calls replaced with the
  ``safe_filter_by_video`` / ``safe_filter_video_neq`` /
  ``safe_videos_in`` helpers from 122ek. Also guarded the
  inner ``rectangles_df['Name'].unique()`` reads.

* ``get_roi_data_for_video_name`` (lines 341-347): 3 naked
  filters replaced with ``safe_filter_by_video``.

mufasa/ui_qt/forms/video_info.py:

* ``PixelCalibrationDialog`` use site at line 538: was
  ``dlg.Accepted``; now ``QDialog.Accepted`` (class-scoped).
* Added ``QDialog`` to the PySide6.QtWidgets import block.

mufasa/ui_qt/pages/video_processing_page.py:

* Similar fix at line 247: ``dlg.Accepted`` → ``QDialog.
  Accepted``. Same import addition.

The workbench.py:753 site (``QDialog.Accepted``) was already
correct from earlier work — that's where the historical
comment lives explaining the pattern.

Coverage
--------
ROI audit gap-fill (3 checks):
1.  ``get_roi_data`` no longer contains the naked
    ``df['Video'].unique()`` pattern.
2.  ``get_roi_data`` no longer contains the naked
    ``df[df['Video'] == ...]`` filter pattern.
3.  ``get_roi_data_for_video_name`` no longer contains
    naked ``df[df['Video'] == ...]`` filters.

Helper usage (2 checks):
4.  ``get_roi_data`` uses ``safe_filter_by_video`` AND
    ``safe_filter_video_neq`` AND ``safe_videos_in``.
5.  ``get_roi_data_for_video_name`` uses
    ``safe_filter_by_video``.

Qt enum (3 checks):
6.  ``video_info.py`` no longer has any ``dlg.Accepted``
    instance-access; uses ``QDialog.Accepted`` (class-
    scoped).
7.  ``video_processing_page.py`` same.
8.  No ``dlg.Accepted`` / ``dlg.Rejected`` instance
    accesses remain anywhere in mufasa/ (the audit
    contract — catches future drift).

Cross-patch invariants:
9.  122ek state preserved: ``multiply_ROIs`` /
    ``reset_video_ROIs`` still use the safe helpers.
10. 122ep state preserved: 8 of 11 ui_bound sections have
    detect_path.
11. 122en state preserved: v1_project_paths canonical.
12. Parse-clean.
13. 122do baseline.
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


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def main() -> int:
    ru_path = REPO_ROOT / "mufasa" / "roi_tools" / "roi_utils.py"
    ru_src = ru_path.read_text()
    ru_tree = ast.parse(ru_src)

    get_data = _find_function(ru_tree, "get_roi_data")
    get_data_video = _find_function(
        ru_tree, "get_roi_data_for_video_name",
    )
    assert get_data is not None
    assert get_data_video is not None
    get_data_src = ast.unparse(get_data)
    get_data_video_src = ast.unparse(get_data_video)

    # -----------------------------------------------------------------
    # ROI audit gap-fill
    # -----------------------------------------------------------------
    check(
        "get_roi_data no longer contains the naked "
        "`df['Video'].unique()` pattern (122er-hotfix audit "
        "gap-fill — replaced with safe_videos_in)",
        not re.search(
            r"_df\['Video'\]\.unique\(\)", get_data_src,
        ),
    )

    check(
        "get_roi_data no longer contains the naked "
        "`df[df['Video'] == ...]` filter pattern (122er-hotfix "
        "audit gap-fill — replaced with safe_filter_by_video)",
        not re.search(
            r"_df\[in_\w+_df\['Video'\]\s*==", get_data_src,
        ),
    )

    check(
        "get_roi_data_for_video_name no longer contains "
        "naked df[df['Video'] == ...] filters (the function "
        "that triggered the user's KeyError 'Video' on the "
        "Apply-to-selected dialog)",
        not re.search(
            r"_df\[in_\w+_df\['Video'\]\s*==", get_data_video_src,
        ),
    )

    # -----------------------------------------------------------------
    # Helper usage
    # -----------------------------------------------------------------
    check(
        "get_roi_data uses all three safe helpers "
        "(safe_filter_by_video, safe_filter_video_neq, "
        "safe_videos_in — covers all three patterns in this "
        "function)",
        "safe_filter_by_video" in get_data_src
        and "safe_filter_video_neq" in get_data_src
        and "safe_videos_in" in get_data_src,
    )

    check(
        "get_roi_data_for_video_name uses safe_filter_by_video",
        "safe_filter_by_video" in get_data_video_src,
    )

    # -----------------------------------------------------------------
    # Qt enum
    # -----------------------------------------------------------------
    # AST-walk a file looking for Attribute nodes with attr ==
    # 'Accepted' and check whether the .value is the class
    # 'QDialog' (correct) or the instance 'dlg' (buggy).
    def _qt_enum_state(path: Path) -> tuple[bool, bool]:
        """Return (uses_QDialog_Accepted, has_buggy_dlg_Accepted)."""
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            return (False, False)
        uses_class = False
        has_instance = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr == "Accepted"
                    and isinstance(node.value, ast.Name)):
                if node.value.id == "QDialog":
                    uses_class = True
                elif node.value.id == "dlg":
                    has_instance = True
        return (uses_class, has_instance)

    vi_path = (REPO_ROOT / "mufasa" / "ui_qt"
               / "forms" / "video_info.py")
    vi_class, vi_inst = _qt_enum_state(vi_path)
    check(
        "video_info.py uses `QDialog.Accepted` (class-scoped, "
        "PySide6-correct) AND has no `dlg.Accepted` "
        "(instance-scoped, AttributeError-causing) — checked "
        "via AST so comments documenting the old form don't "
        "create false positives",
        vi_class and not vi_inst,
        detail=(f"uses_QDialog={vi_class} buggy_dlg={vi_inst}"),
    )

    vp_path = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
               / "video_processing_page.py")
    vp_class, vp_inst = _qt_enum_state(vp_path)
    check(
        "video_processing_page.py same fix (AST-checked)",
        vp_class and not vp_inst,
        detail=(f"uses_QDialog={vp_class} buggy_dlg={vp_inst}"),
    )

    # 8. No instance-access dlg.Accepted/Rejected anywhere.
    pkg = REPO_ROOT / "mufasa"
    instance_access_hits = []
    pattern = re.compile(
        r"\bdlg\.(Accepted|Rejected)\b"
        r"|\bself\.(Accepted|Rejected)\b"
    )
    for f in sorted(pkg.rglob("*.py")):
        src = f.read_text()
        # Skip occurrences that are inside comments OR docstrings.
        # Simple heuristic: strip out lines starting with # (post-
        # whitespace) and bulk-skip docstring-quoted regions.
        # For this audit, the AST-level approach is more reliable:
        # walk for Attribute nodes whose attr is Accepted/Rejected
        # and whose .value is a Name 'dlg' or 'self'.
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr in ("Accepted", "Rejected")
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("dlg", "self")):
                instance_access_hits.append(
                    f"{f.relative_to(REPO_ROOT)}:{node.lineno}"
                )
    check(
        "No instance-access `dlg.Accepted` / `dlg.Rejected` / "
        "`self.Accepted` patterns anywhere in mufasa/ (the "
        "class-of-bug audit contract — catches future drift "
        "from the same PySide6 enum issue)",
        not instance_access_hits,
        detail=("; ".join(instance_access_hits[:3])),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    multiply = _find_function(ru_tree, "multiply_ROIs")
    reset = _find_function(ru_tree, "reset_video_ROIs")
    if multiply is not None and reset is not None:
        mt = ast.unparse(multiply)
        rt = ast.unparse(reset)
        check(
            "122ek state preserved: multiply_ROIs uses "
            "safe_videos_in and safe_filter_by_video; "
            "reset_video_ROIs uses safe_filter_by_video and "
            "safe_filter_video_neq",
            ("safe_videos_in" in mt
             and "safe_filter_by_video" in mt
             and "safe_filter_by_video" in rt
             and "safe_filter_video_neq" in rt),
        )
    else:
        check("(122ek check skipped — functions missing)",
              False, detail="multiply_ROIs/reset_video_ROIs not found")

    from mufasa.section_provenance import SECTIONS
    ui_bound = [s for s in SECTIONS.values() if s.ui_bound]
    with_detect = [s for s in ui_bound if s.detect_path is not None]
    check(
        "122ep state preserved: 8 of 11 ui_bound sections "
        "have detect_path",
        len(with_detect) == 8 and len(ui_bound) == 11,
        detail=(f"got {len(with_detect)}/{len(ui_bound)}"),
    )

    pl_src = (REPO_ROOT / "mufasa"
              / "project_layout.py").read_text()
    check(
        "122en state preserved: v1_project_paths canonical helper",
        "def v1_project_paths" in pl_src,
    )

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
        "122do baseline preserved: no `Optional[` in non-"
        "docstring positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122er_roi_audit_and_qt_enum: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
