"""
tests/smoke_122d7_qwi3_max_workers.py
=======================================

Patch 122d7: fix QWI-3 — `max_workers=0` crash on empty
project in feature subsets.

Two-sided fix:
- Backend (feature_subsets.py): short-circuit on n_videos==0
  + clamp n_workers to ≥ 1.
- Qt form (features.py): preflight returns None on empty
  project → on_run surfaces a clear QMessageBox.

Coverage
--------
1.  feature_subsets.py contains the empty-project short-circuit
    (no `ProcessPoolExecutor(max_workers=0)` reachable when
    n_videos==0).
2.  feature_subsets.py applies `max(1, …)` clamp on n_workers.
3.  ui_qt/forms/features.py preflight returns `None` when
    `calc.data_paths` is empty.
4.  ui_qt/forms/features.py on_run handles the `conflicts is
    None` case and shows a QMessageBox.warning (NOT a crash).
5.  on_run's empty-project message references the data-pipeline
    prerequisite (outlier correction / pose data) — the
    user-helpful detail, not just "no videos".
6.  qt_workbench_known_issues.md marks QWI-3 as Fixed 122d7.
7.  All mufasa/**/*.py parse cleanly.
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

    fs_path = pkg / "feature_extractors" / "feature_subsets.py"
    fs_src = fs_path.read_text()
    feat_path = pkg / "ui_qt" / "forms" / "features.py"
    feat_src = feat_path.read_text()

    # 1. Backend short-circuit on n_videos==0
    check(
        "feature_subsets.py: short-circuits when n_videos==0 "
        "(QWI-3 backend fix)",
        ("if n_videos == 0:" in fs_src
         and "no eligible videos" in fs_src.lower()),
    )

    # 2. n_workers clamp to ≥ 1
    check(
        "feature_subsets.py: clamps n_workers ≥ 1 with max(1, …)",
        "max(1, min(self.n_workers, n_videos))" in fs_src,
    )

    # 3. Qt form preflight returns None on empty project
    # Look for the empty-project branch in _run_preflight.
    feat_tree = ast.parse(feat_src)
    found_preflight_none = False
    for node in ast.walk(feat_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_run_preflight"):
            body_src = ast.unparse(node)
            if ("not calc.data_paths" in body_src
                    and "return None" in body_src):
                found_preflight_none = True
                break
    check(
        "_run_preflight returns None when calc.data_paths is "
        "empty (QWI-3 Qt-form signal)",
        found_preflight_none,
    )

    # 4. on_run handles `conflicts is None`
    check(
        "on_run handles `conflicts is None` from preflight "
        "with a QMessageBox.warning",
        "conflicts is None" in feat_src
        and "QMessageBox.warning" in feat_src,
    )

    # 5. Message mentions the prereqs the user actually needs
    check(
        "Empty-project message mentions outlier-correction / "
        "pose-data prerequisite (not just 'no videos')",
        ("outlier" in feat_src.lower()
         and "pose data" in feat_src.lower()),
    )

    # 6. Known-issues doc marks QWI-3 fixed
    qwi_doc = (REPO_ROOT / "docs"
               / "qt_workbench_known_issues.md").read_text()
    check(
        "qt_workbench_known_issues.md marks QWI-3 Fixed 122d7",
        "QWI-3" in qwi_doc
        and "Fixed 122d7" in qwi_doc,
    )

    # 7. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122d7_qwi3_max_workers: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
