"""
tests/smoke_122bx_path_b_tier4_prep.py
========================================

Patch 122bx: Path-B Tier-4 prep.

Three small things rolled into one patch:

1. Delete `mufasa/ui/pop_ups/helpers.py` (13 lines, 1 function,
   UNREFERENCED per `tk_surface_audit.md`).
2. Delete `mufasa/ui/user_defined_pose_creator.py` (156 lines, 1
   class, UNREFERENCED per `tk_surface_audit.md`).
3. Port :class:`CheckVideoSeekablePopUp` to Qt as
   :class:`CheckVideoSeekableForm` in `forms/video_utilities.py`.
   Register as a 4th form in the "Utilities" section under
   Video Processing.

Coverage
--------
1. helpers.py is gone.
2. user_defined_pose_creator.py is gone.
3. No remaining import references to either deleted file in
   mufasa/.
4. CheckVideoSeekableForm class defined in video_utilities.py.
5. The form subclasses OperationForm.
6. The form has build(), collect_args(), target() methods.
7. target() imports is_video_seekable backend.
8. The form is exported in __all__ of video_utilities.py.
9. video_processing_page.py imports CheckVideoSeekableForm.
10. The Utilities section in video_processing_page.py now has
    4 forms (was 3).
11. The module docstring of video_utilities.py mentions the
    fourth form.
12. All mufasa/**/*.py files parse cleanly.
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


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # 1-3. Unreferenced files deleted + nothing imports them
    # ==================================================================
    helpers_path = pkg / "ui" / "pop_ups" / "helpers.py"
    udpc_path = pkg / "ui" / "user_defined_pose_creator.py"
    check(
        "mufasa/ui/pop_ups/helpers.py is gone",
        not helpers_path.exists(),
    )
    check(
        "mufasa/ui/user_defined_pose_creator.py is gone",
        not udpc_path.exists(),
    )

    # No remaining references in any .py file under mufasa/
    deleted_modules = [
        "mufasa.ui.pop_ups.helpers",
        "mufasa.ui.user_defined_pose_creator",
    ]
    for mod in deleted_modules:
        offenders: list[Path] = []
        for f in pkg.rglob("*.py"):
            src = f.read_text()
            # Any import statement referencing the deleted module
            if re.search(rf"(from\s+{re.escape(mod)})|(import\s+{re.escape(mod)})", src):
                offenders.append(f)
        check(
            f"No remaining imports of deleted module '{mod}'",
            offenders == [],
            detail=(f"first offender: {offenders[0]}"
                    if offenders else ""),
        )

    # ==================================================================
    # 4-8. CheckVideoSeekableForm
    # ==================================================================
    util_path = pkg / "ui_qt" / "forms" / "video_utilities.py"
    util_src = util_path.read_text()
    tree = ast.parse(util_src)
    form_cls = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "CheckVideoSeekableForm"):
            form_cls = node
            break
    check(
        "CheckVideoSeekableForm class defined",
        form_cls is not None,
    )
    if form_cls is not None:
        base_names = {ast.unparse(b) for b in form_cls.bases}
        check(
            "CheckVideoSeekableForm subclasses OperationForm",
            "OperationForm" in base_names,
        )
        methods = {
            stmt.name for stmt in form_cls.body
            if isinstance(stmt, ast.FunctionDef)
        }
        for m in ("build", "collect_args", "target"):
            check(
                f"CheckVideoSeekableForm defines {m}()",
                m in methods,
            )

    check(
        "target() imports is_video_seekable backend",
        "is_video_seekable" in util_src,
    )
    check(
        "CheckVideoSeekableForm exported in __all__",
        '"CheckVideoSeekableForm"' in util_src,
    )

    # ==================================================================
    # 9-10. Page wiring
    # ==================================================================
    page_path = pkg / "ui_qt" / "pages" / "video_processing_page.py"
    page_src = page_path.read_text()
    check(
        "video_processing_page.py imports CheckVideoSeekableForm",
        "CheckVideoSeekableForm" in page_src,
    )
    # The Utilities section now has 4 forms (was 3). Match the
    # registration tuple syntax loosely.
    utilities_section = re.search(
        r'add_section\("Utilities".*?\]\)', page_src,
        flags=re.DOTALL)
    if utilities_section:
        forms_in_utilities = utilities_section.group(0).count("Form, {})")
        check(
            f"Utilities section has 4 forms (got {forms_in_utilities})",
            forms_in_utilities == 4,
        )
    else:
        check("Utilities section located in page", False)

    # ==================================================================
    # 11. Docstring update
    # ==================================================================
    check(
        "video_utilities.py docstring mentions CheckVideoSeekableForm",
        "CheckVideoSeekableForm" in util_src[:1200],
    )

    # ==================================================================
    # 12. All files parse cleanly
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
        f"smoke_122bx_path_b_tier4_prep: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
