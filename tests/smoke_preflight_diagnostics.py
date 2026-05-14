"""
tests/smoke_preflight_diagnostics.py
=====================================

Originally asserted that ``preflight_check`` printed several
diagnostics — column count from the probe, per-target file
counts, and the starting destination summary.

Patch 122an (B1) removed the column-collision probe and its
diagnostics. The remaining preflight diagnostics — the
starting summary and the save_dir check result — survive.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    backend_src = (REPO_ROOT / "mufasa" / "feature_extractors"
                   / "feature_subsets.py").read_text()

    checks = 0
    passed = 0

    # Surviving diagnostics
    for label, needle in (
        ("[preflight] starting check banner",
         "[preflight] starting check"),
        ("[preflight] save_dir result counter",
         "[preflight] save_dir check"),
    ):
        checks += 1
        if needle in backend_src:
            passed += 1
        else:
            print(f"FAIL: {label} — diagnostic missing")

    # Removed diagnostics — should NOT appear
    for label, needle in (
        ("[preflight] probe diagnostic removed",
         "[preflight] probe produced"),
    ):
        checks += 1
        if needle not in backend_src:
            passed += 1
        else:
            print(f"FAIL: {label} — should be removed")

    print(
        f"smoke_preflight_diagnostics: "
        f"{passed}/{checks} checks passed"
    )
    return 0 if passed == checks else 1


if __name__ == "__main__":
    sys.exit(main())
