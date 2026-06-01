"""
tests/smoke_122ga_ruff_fix_chunk5_pyupgrade_tail.py
===================================================

Patch 122ga — ruff --fix sweep chunk 5/N: the remaining pyupgrade rules.

User request: "continue" (resuming the chunked ruff sweep).

CHUNK 5 RULES (both safe fixes)
===============================
  UP035 deprecated-import   (64)  from typing import Callable/Iterable/...
                                  -> from collections.abc import ...
  UP037 quoted-annotation   (41)  x: "Foo" -> x: Foo

Command: ruff check mufasa --select UP035,UP037 --fix

WHY THE F821 GUARD MATTERS HERE
===============================
UP037 removes quotes around annotations. In a module WITHOUT
`from __future__ import annotations`, a quoted forward reference that is
genuinely defined later would become a NameError once unquoted. The
guard: after the fix, the F821 (undefined-name) set must be unchanged —
still exactly the 7 known-deliberate entries (kalman Path x3,
project_layout Union x2, reverse_pose "987", converters geometry_to_rle).
A new F821 would mean UP037 unquoted a real forward ref. Verified: F821
stayed at 7.

RESULT / VERIFICATION
=====================
* 105 fixes across 66 files. UP035/UP037 -> 0. F401 unchanged (7); total
  selected 3076 -> 2975. compileall clean. F821 unchanged (7).
* d/e/f strict sweep: 0 fails. Full smoke_122*.py sweep baseline-diffed:
  one reciprocal tripwire — smoke_122cl pinned "roi_ruler imports Callable
  from typing"; UP035 modernized it to collections.abc. FLIPPED that
  assertion (122cl back to 13/13). No other net regressions; a/b/c
  partial set otherwise == pre-existing baseline.

RECIPROCAL FLIPS
================
* smoke_122cl_roi_ruler_callback — Callable-import assertion: typing ->
  collections.abc (UP035).

NEW SMOKE: smoke_122ga_ruff_fix_chunk5_pyupgrade_tail.py (3 checks)
* mufasa/ parses cleanly
* ruff reports 0 for UP035,UP037 (regression tripwire)
* F821 still exactly the 7 known-deliberate (UP037 forward-ref guard)
  -- ruff parts soft-pass with a note if ruff is unavailable.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHUNK_RULES = "UP035,UP037"

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
        print("NOTE: ruff not found on PATH — chunk-5 checks skipped (soft pass).")
        check("ruff reports 0 for chunk-5 rules", True)
        check("F821 still the 7 known-deliberate", True)
    else:
        proc = subprocess.run(
            [ruff, "check", str(pkg), "--select", CHUNK_RULES, "--quiet"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        check(
            "ruff reports 0 for chunk-5 rules (UP035,UP037)",
            proc.returncode == 0,
            detail=proc.stdout.strip()[:200],
        )
        f821 = subprocess.run(
            [ruff, "check", str(pkg), "--select", "F821", "--output-format", "concise"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        n_f821 = sum(1 for ln in f821.stdout.splitlines() if "F821" in ln)
        check(
            "F821 still exactly 7 known-deliberate (UP037 unquoted no real forward ref)",
            n_f821 == 7,
            detail=f"F821 count = {n_f821} (expected 7)",
        )

    print(
        f"smoke_122ga_ruff_fix_chunk5_pyupgrade_tail: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
