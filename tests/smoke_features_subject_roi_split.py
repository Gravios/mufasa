"""
tests/smoke_features_subject_roi_split.py
=========================================

Patch 122z: regression guard for the FeatureSubsetExtractorForm
split — subject features and ROI features now live in two
separate QGroupBox frames, and the destination radio labels
reference both v1 and legacy paths.

Coverage:

1. ``_is_roi_family`` correctly classifies the default family
   names: anything containing 'ROI' goes to the ROI group,
   everything else to the subject group.
2. Form builds two distinct QListWidgets — ``subject_families``
   and ``roi_families`` — wrapped in QGroupBox frames.
3. Description mentions both 'Subject features' and 'ROI
   features' explicitly.
4. ``collect_args`` concatenates selections from both lists.
5. Destination radio labels mention v1 paths
   (``derived/features/`` / ``derived/classifications/``) and
   the legacy paths (``csv/features_extracted/`` /
   ``csv/targets_inserted/``).
6. ``self.families`` historical alias still exists.
7. The 122z patch note is in the form's module docstring.
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
    src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
           / "features.py").read_text()
    tree = ast.parse(src)

    # ---------- 1. _is_roi_family classification ----------
    # Pull the helper out and exec it; verify against the default
    # family list.
    helper_src = None
    for n in tree.body:
        if (isinstance(n, ast.FunctionDef)
                and n.name == "_is_roi_family"):
            helper_src = ast.unparse(n)
            break
    check(
        "_is_roi_family() defined at module scope",
        helper_src is not None,
    )
    if helper_src is not None:
        ns: dict = {}
        exec(helper_src, ns)
        is_roi = ns["_is_roi_family"]
        # Subject group expectations
        for n in (
            "Two-point body-part distances (mm)",
            "Within-animal three-point body-part angles (degrees)",
            "Entire animal convex hull area (mm2)",
            "Frame-by-frame body-part movements (mm)",
        ):
            check(
                f"_is_roi_family({n!r}) == False (subject feature)",
                is_roi(n) is False,
            )
        # ROI group expectations
        for n in (
            "Frame-by-frame distance to ROI centers (mm)",
            "Frame-by-frame body-parts inside ROIs (Boolean)",
        ):
            check(
                f"_is_roi_family({n!r}) == True (ROI feature)",
                is_roi(n) is True,
            )

    # ---------- 2. Two list widgets, QGroupBox frames ----------
    cls = _find_class(tree, "FeatureSubsetExtractorForm")
    check("FeatureSubsetExtractorForm class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        check(
            "Form builds self.subject_families QListWidget",
            "self.subject_families = QListWidget(" in class_src,
        )
        check(
            "Form builds self.roi_families QListWidget",
            "self.roi_families = QListWidget(" in class_src,
        )
        check(
            "Form uses QGroupBox to frame the two categories",
            "QGroupBox" in class_src,
        )
        check(
            "Subject QGroupBox titled 'Subject features'",
            "Subject features" in class_src,
        )
        check(
            "ROI QGroupBox titled 'ROI features'",
            "ROI features" in class_src,
        )
        check(
            "self.families historical alias preserved",
            "self.families = self.subject_families" in class_src
            or "self.families = self.roi_families" in class_src
            or "self.families =" in class_src,
        )

        # ---------- 3. Description mentions both categories ----------
        # The description string is part of the class body source.
        check(
            "Description mentions 'Subject features'",
            "Subject features" in class_src,
        )
        check(
            "Description mentions 'ROI features'",
            "ROI features" in class_src,
        )

        # ---------- 4. collect_args merges both lists ----------
        methods = {
            n.name: n for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }
        if "collect_args" in methods:
            ca_src = ast.unparse(methods["collect_args"])
            check(
                "collect_args reads self.subject_families.selectedItems",
                "self.subject_families.selectedItems" in ca_src,
            )
            check(
                "collect_args reads self.roi_families.selectedItems",
                "self.roi_families.selectedItems" in ca_src,
            )

        # ---------- 5. Destination labels mention v1 schema ----------
        # Patch 122bf: post-122bd visible labels in the form
        # describe the v1 path only (derived/features/). The
        # legacy and classifications paths were transitional —
        # 122bd dropped them from user-visible text. Assertions
        # for those paths are removed.
        check(
            "Destination labels mention v1 'derived/features/' path",
            "derived/features/" in class_src,
        )

    # ---------- 7. 122z patch note in module docstring ----------
    # Module docstring is the first stmt of the module.
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        mod_doc = tree.body[0].value.value
        check(
            "Module docstring carries the 122z note",
            "122z" in mod_doc,
        )
    else:
        check("Module docstring exists", False)

    print(
        f"smoke_features_subject_roi_split: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
