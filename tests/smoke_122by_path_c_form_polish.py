"""
tests/smoke_122by_path_c_form_polish.py
=========================================

Patch 122by: Path-C form polish. Documentation + consistency
audit for the Qt workbench forms.

Three small pieces:

1. New audit doc `docs/qt_form_runtime_gaps.md` — inventories
   the 4 forms / 7 operations that raise NotImplementedError
   at runtime. Per-form root cause, fix scope estimate, and
   stop-gap recommendations.
2. Polish fix: `CropVideosForm` docstring now lists its 4
   replaced Tk popups by `:class:` name (matching the
   convention used in BackgroundRemovalForm / BlobQuickCheckForm).
   Plus a "Known gap" note pointing to the new audit.
3. `docs/README.md` indexes the new audit.

Coverage
--------
1. docs/qt_form_runtime_gaps.md exists.
2. Audit names the 4 affected forms.
3. Audit references the AverageFrameForm spelling-mismatch root
   cause (the most concrete bug discovered in the audit).
4. Audit has §3 Recommendations (priority order).
5. Audit has §4 Audit methodology (reproducible script).
6. CropVideosForm docstring lists all 4 replaced Tk popups.
7. CropVideosForm docstring references the new audit doc.
8. docs/README.md references qt_form_runtime_gaps.md.
9. All mufasa/**/*.py files parse cleanly.
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
    audit_path = REPO_ROOT / "docs" / "qt_form_runtime_gaps.md"

    # ==================================================================
    # 1-5. Audit doc
    # ==================================================================
    check("docs/qt_form_runtime_gaps.md exists", audit_path.exists())
    if audit_path.exists():
        audit_text = audit_path.read_text()

        affected_forms = ["AverageFrameForm", "VideoFiltersForm",
                          "CropVideosForm", "DropBodypartsForm",
                          "ROIFeaturesForm"]
        for form in affected_forms:
            check(
                f"audit names affected form '{form}'",
                form in audit_text,
            )

        # AverageFrameForm root cause
        check(
            "audit mentions create_average_frm vs create_average_frame "
            "spelling mismatch",
            "create_average_frm" in audit_text
            and "create_average_frame" in audit_text,
        )

        # Structural sections
        check(
            "audit has §3 Recommendations section",
            "## 3. Recommendations" in audit_text
            or "Recommendations" in audit_text,
        )
        check(
            "audit has §4 Audit methodology section",
            "Audit methodology" in audit_text,
        )

        # Stop-gap recommendation
        check(
            "audit mentions the stop-gap disable recommendation",
            "stop-gap" in audit_text.lower()
            or "disable" in audit_text.lower(),
        )

    # ==================================================================
    # 6-7. CropVideosForm docstring fix
    # ==================================================================
    ve_path = pkg / "ui_qt" / "forms" / "video_editing.py"
    ve_src = ve_path.read_text()
    tree = ast.parse(ve_src)
    crop_form = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "CropVideosForm"):
            crop_form = node
            break
    check("CropVideosForm class found", crop_form is not None)
    if crop_form is not None:
        doc = ast.get_docstring(crop_form) or ""
        # All 4 Tk popups listed by :class: name
        tk_popups = ["CropVideoPopUp", "CropVideoCirclesPopUp",
                     "CropVideoPolygonsPopUp", "MultiCropPopUp"]
        for cls in tk_popups:
            check(
                f"CropVideosForm docstring lists Tk popup '{cls}'",
                cls in doc,
            )
        # References the audit doc OR a later resolution patch.
        # Post-122cf the form's docstring was updated from
        # "Known gap (patch 122by audit)" to a "Resolved in 122cf"
        # note. The check is "documents its history" not "still
        # marked as a known gap."
        check(
            "CropVideosForm docstring references either the "
            "runtime-gaps audit or a later resolution",
            "qt_form_runtime_gaps.md" in doc
            or "122by audit" in doc
            or "122cf" in doc,
        )

    # ==================================================================
    # 8. docs/README.md indexes the audit
    # ==================================================================
    readme = REPO_ROOT / "docs" / "README.md"
    check(
        "docs/README.md references qt_form_runtime_gaps.md",
        "qt_form_runtime_gaps.md" in readme.read_text(),
    )

    # ==================================================================
    # 9. All files parse cleanly
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
        f"smoke_122by_path_c_form_polish: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
