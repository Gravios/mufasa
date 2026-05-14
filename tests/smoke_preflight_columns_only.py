"""
tests/smoke_preflight_columns_only.py
======================================

Originally asserted that ``preflight_check`` used a fast
header-only ``_read_columns_only`` helper (instead of full
``read_df``) when probing existing files in
``features_extracted/`` / ``targets_inserted/`` for column
overlaps.

Patch 122an (B1) removed that whole code path along with the
``append_to_*`` kwargs. ``_read_columns_only`` may still exist
in the module for other callers, but preflight no longer
needs it.

This test confirms the column-probing call site is gone from
``preflight_check``.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    pf_src = (REPO_ROOT / "mufasa" / "feature_extractors"
              / "feature_subsets.py").read_text()

    # Parse and find preflight_check
    tree = ast.parse(pf_src)
    preflight_src = ""
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "preflight_check"):
            preflight_src = ast.unparse(node)
            break
    if not preflight_src:
        print("FAIL: preflight_check method not found")
        return 1

    checks = 0
    passed = 0

    checks += 1
    if "_read_columns_only" not in preflight_src:
        passed += 1
    else:
        print("FAIL: preflight_check still calls _read_columns_only "
              "(should be removed in patch 122an)")

    checks += 1
    # The save_dir filename-collision check uses os.path.isfile
    if "os.path.isfile" in preflight_src:
        passed += 1
    else:
        print("FAIL: save_dir filename-collision check missing")

    print(
        f"smoke_preflight_columns_only: "
        f"{passed}/{checks} checks passed"
    )
    return 0 if passed == checks else 1


if __name__ == "__main__":
    sys.exit(main())
