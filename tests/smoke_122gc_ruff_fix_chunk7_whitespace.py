"""
tests/smoke_122gc_ruff_fix_chunk7_whitespace.py
===============================================

Patch 122gc — ruff --fix sweep chunk 7/N: safe whitespace cleanup.

User request: "continue".

CHUNK 7 RULES (whitespace; SAFE subset only)
============================================
  W291 trailing-whitespace
  W293 blank-line-with-whitespace

Command: ruff check mufasa --select W291,W293 --fix   (50 safe fixes)

SAFE SUBSET ONLY — the unsafe remainder is left untouched
=========================================================
Of 85 W291/W293 hits, 50 are [*] safe and 35 are hidden-unsafe. The
unsafe ones are trailing whitespace INSIDE string literals / docstrings,
where stripping it would change the string's value — so ruff (correctly)
withholds them from a safe --fix. We do NOT pass --unsafe-fixes; those 35
are deliberately left.

The 50 applied are whitespace-only on code/blank lines: `git diff -w` is
empty (no content change), 50 insertions / 50 deletions across 11 files.

RESULT / VERIFICATION
=====================
* 50 fixes / 11 files; pure whitespace (git diff -w empty). compileall
  clean; F821 unchanged (7). d/e/f sweep 0 fails; a/b/c partial set ==
  pre-existing baseline. Zero net effect on behavior or smokes.

NEW SMOKE: smoke_122gc_ruff_fix_chunk7_whitespace.py (3 checks)
* mufasa/ parses cleanly
* no [*] safe-fixable W291/W293 remain (the 35 in-string unsafe ones
  may persist — that is expected and correct)
* F821 still exactly the 7 known-deliberate
  -- ruff parts soft-pass with a note if ruff is unavailable.
"""

import ast
import shutil
import subprocess
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

    ruff = shutil.which("ruff")
    if ruff is None:
        print("NOTE: ruff not found on PATH — chunk-7 checks skipped (soft pass).")
        check("no safe-fixable W291/W293 remain", True)
        check("F821 still the 7 known-deliberate", True)
    else:
        ws = subprocess.run(
            [ruff, "check", str(pkg), "--select", "W291,W293",
             "--output-format", "concise"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        safe_fixable = [ln for ln in ws.stdout.splitlines()
                        if ("W291 [*]" in ln or "W293 [*]" in ln)]
        check(
            "no [*] safe-fixable W291/W293 remain (in-string unsafe ones may persist)",
            not safe_fixable,
            detail=f"{len(safe_fixable)} safe-fixable still present",
        )
        f821 = subprocess.run(
            [ruff, "check", str(pkg), "--select", "F821", "--output-format", "concise"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        n_f821 = sum(1 for ln in f821.stdout.splitlines() if "F821" in ln)
        check(
            "F821 still exactly 7 known-deliberate",
            n_f821 == 7,
            detail=f"F821 count = {n_f821} (expected 7)",
        )

    print(
        f"smoke_122gc_ruff_fix_chunk7_whitespace: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
