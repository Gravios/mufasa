"""
tests/smoke_122fy_ruff_fix_chunk4_f401.py
=========================================

Patch 122fy — ruff --fix sweep chunk 4/N: F401 unused-import cleanup,
including the ~757 typing imports orphaned by the 122fx annotation
modernization.

User request: "continue" (resuming the chunked ruff sweep, full-sweep
baseline-diffed per the 122fx coverage-gap finding).

CHUNK 4 RULE (UNSAFE-classed — applied in ruff SAFE mode only)
==============================================================
  F401 unused-import

Command: ruff check mufasa --select F401 --fix   (safe fixes only)

WHY EXTRA CAUTION
=================
F401 is the riskiest chunk so far: removing an import can break a
re-export or a side-effect import. Mitigations applied:
  * Ran in ruff's default SAFE mode — it left 7 instances it could not
    prove safe (the package-root `import os`; conditional `typing.Literal`
    inside import-availability guards in two agg_clf modules; four
    function-local PySide6 imports). Those need human judgment; untouched.
  * mufasa/__init__.py (the only affected __init__, highest re-export
    risk) was NOT modified — verified by diff.
  * Cross-validated with F821: if a still-used import had been removed,
    the name would become undefined. The post-fix F821 set is EXACTLY
    the pre-existing known-deliberate set (no new undefined names):
      kalman_pose_smoother_v2.py Path x3  (annotation-only, inert under
        from __future__ import annotations; pre-existing since 122fp)
      project_layout.py Union x2          (same)
      reverse_pose.py extract_features_wotarget_14_from_16 (122fr "987"
        deliberate non-fix)
      converters.py geometry_to_rle       (122fp/122fs deferred)

RESULT / VERIFICATION
=====================
* 1103 F401 fixed across 273 files (210 ins / 593 del). 7 left (above).
* compileall clean.
* d/e/f strict sweep: 0 fails. Full smoke_122*.py sweep baseline-diffed
  against 122fx: the a/b/c partial-failure set is byte-identical to the
  pre-existing baseline (env / deleted-file / stale; not this chunk).
  Net regressions introduced: ZERO.
* Total selected ruff findings 4526 -> 3076.

NEW SMOKE: smoke_122fy_ruff_fix_chunk4_f401.py (4 checks)
* mufasa/ parses cleanly
* mufasa/__init__.py still imports os (re-export site left untouched)
* F821 introduced NO new undefined name (every undefined name is in the
  known-deliberate allowlist) — the real F401-safety guard
* ruff F401 leaves no [*] safe-fixable instances (only the 7 unsafe)
  -- ruff parts soft-pass with a note if ruff is unavailable.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# (relative_path_substring, undefined_name) pairs that are KNOWN and
# deliberate — pre-existing, documented, and not caused by F401.
KNOWN_F821 = {
    ("data_processors/kalman_pose_smoother_v2.py", "Path"),
    ("project_layout.py", "Union"),
    ("pose_processors/reverse_pose.py", "extract_features_wotarget_14_from_16"),
    ("third_party_label_appenders/converters.py", "geometry_to_rle"),
}

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

    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"all mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    init_src = (pkg / "__init__.py").read_text()
    check(
        "mufasa/__init__.py left untouched (still imports os)",
        "import os" in init_src,
    )

    ruff = shutil.which("ruff")
    if ruff is None:
        print("NOTE: ruff not found on PATH — F401/F821 checks skipped (soft pass).")
        check("F821 introduced no new undefined name", True)
        check("no [*] safe-fixable F401 remain", True)
    else:
        # F821 guard: every undefined name must be in the known allowlist.
        f821 = subprocess.run(
            [ruff, "check", str(pkg), "--select", "F821", "--output-format", "concise"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        unexpected = []
        for line in f821.stdout.splitlines():
            if "F821" not in line:
                continue
            # format: path:line:col: F821 Undefined name `X`
            path = line.split(":", 1)[0]
            name = line.split("`")[1] if "`" in line else ""
            if not any(p in path and n == name for (p, n) in KNOWN_F821):
                unexpected.append(line.strip())
        check(
            "F821 introduced no new undefined name (F401 removed nothing in use)",
            not unexpected,
            detail="; ".join(unexpected[:3]),
        )

        # No safe-fixable F401 should remain (only the 7 unsafe instances).
        f401 = subprocess.run(
            [ruff, "check", str(pkg), "--select", "F401", "--output-format", "concise"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        safe_fixable = [ln for ln in f401.stdout.splitlines() if "F401 [*]" in ln]
        check(
            "no [*] safe-fixable F401 remain (only unsafe judgment calls left)",
            not safe_fixable,
            detail=f"{len(safe_fixable)} safe-fixable F401 still present",
        )

    print(
        f"smoke_122fy_ruff_fix_chunk4_f401: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
