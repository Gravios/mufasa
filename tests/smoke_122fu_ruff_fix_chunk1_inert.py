"""
tests/smoke_122fu_ruff_fix_chunk1_inert.py
==========================================

Patch 122fu — first chunk of the ruff --fix sweep: apply SAFE,
semantically-inert autofixes only.

User request (Fri May 30, 2026):
> start with the ruff --fix, move through it slowly and carefully

APPROACH
========
The sweep is being done chunked by rule family, safest first, one patch
per chunk, with a full parse + strict-smoke sweep + ruff-delta check
after each. Chunk 1 is restricted to rules whose fix cannot change
runtime behaviour:

  W292  missing-newline-at-end-of-file
  UP004 useless-object-inheritance      (class X(object) -> class X)
  UP015 redundant-open-modes            (open(p, "r") -> open(p))
  UP034 extraneous-parentheses
  UP039 unnecessary-class-parentheses

228 fixes across 140 files. Verified: total selected ruff findings
9270 -> 9043 (delta -227, no collateral rule changes); the five rules
report 0; mufasa/ compiles; all 79 prior strict smokes stay green.

DELIBERATELY HELD BACK (later chunks)
=====================================
* UP037 quoted-annotation removal — can break forward refs in modules
  without `from __future__ import annotations`.
* UP006/UP007/UP045 annotation modernization (~4700) — own chunk.
* UP032/F541 f-strings — own chunk.
* All [-] unsafe fixes (F401 unused-import, I001 import-sort, UP035
  deprecated-import, F841 unused-var, SIM118) — need per-file review.
* No-fix rules (E701/E702/E402/B905/B904/E722/E741/E711/E721/B007 …) —
  manual, out of scope for --fix.

NEW SMOKE: smoke_122fu_ruff_fix_chunk1_inert.py (2 checks)
* mufasa/ parses cleanly
* ruff reports 0 for the chunk-1 rules (regression tripwire; soft-passes
  with a printed note if the ruff binary is unavailable in the env)
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHUNK_RULES = "W292,UP004,UP015,UP034,UP039"

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

    # --- package parses ---------------------------------------------------
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

    # --- ruff regression tripwire ----------------------------------------
    ruff = shutil.which("ruff")
    if ruff is None:
        # Can't verify without the binary; don't block envs that lack it.
        print("NOTE: ruff not found on PATH — chunk-1 rule check skipped (soft pass).")
        check("ruff reports 0 for chunk-1 inert rules", True)
    else:
        proc = subprocess.run(
            [ruff, "check", str(pkg), "--select", CHUNK_RULES, "--quiet"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        check(
            "ruff reports 0 for chunk-1 inert rules (W292,UP004,UP015,UP034,UP039)",
            proc.returncode == 0,
            detail=proc.stdout.strip()[:200],
        )

    print(
        f"smoke_122fu_ruff_fix_chunk1_inert: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
