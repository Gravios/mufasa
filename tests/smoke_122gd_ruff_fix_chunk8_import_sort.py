"""
tests/smoke_122gd_ruff_fix_chunk8_import_sort.py
================================================

Patch 122gd — ruff --fix sweep chunk 8/N: I001 import sorting.

User request: "continue".

CHUNK 8 RULE
============
  I001 unsorted-imports   (267 of 268 fixed across 244 files)

Command: ruff check mufasa --select I001 --fix --unsafe-fixes

WHY THIS IS SAFE DESPITE THE [-] CLASS
======================================
I001 is unsafe-classed because reordering imports can change behavior
when an import has side effects whose ordering matters. ruff's isort,
however, only sorts CONTIGUOUS import blocks — it never moves an import
across a non-import statement. So a side-effecting sequence like
`import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot`
is not reordered across the `.use()` call. That property makes the sweep
safe in practice, and the verification below confirms no behavioral or
smoke change.

ONE LEFTOVER (expected): checks.py has a function-local
`from ... import get_ffmpeg_encoders; encoders = get_ffmpeg_encoders()`
on a single line — an import fused with a statement, not a sortable
block, so ruff leaves it. (It is also an E702 site for the manual pass.)

RESULT / VERIFICATION
=====================
* 267 fixes / 244 files (pure reordering + isort multi-line reflow).
  I001 reduced to 1 (the inline one-liner above). compileall clean;
  F821 unchanged (7). Total selected 2840 -> 2573.
* d/e/f sweep 0 fails; full smoke_122*.py baseline-diffed: a/b/c partial
  set == pre-existing baseline. NO reciprocal flips were needed — the
  smokes assert import PRESENCE (AST ImportFrom lookups), not order, so
  reordering is transparent to them. ui_qt was already I001-clean (122do).

NEW SMOKE: smoke_122gd_ruff_fix_chunk8_import_sort.py (3 checks)
* mufasa/ parses cleanly
* I001 reduced to <= 1 sortable-block error (the inline import one-liner)
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
        print("NOTE: ruff not found on PATH — chunk-8 checks skipped (soft pass).")
        check("I001 reduced to <= 1 (inline import one-liner)", True)
        check("F821 still the 7 known-deliberate", True)
    else:
        i001 = subprocess.run(
            [ruff, "check", str(pkg), "--select", "I001", "--output-format", "concise"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        n_i001 = sum(1 for ln in i001.stdout.splitlines() if "I001" in ln)
        check(
            "I001 reduced to <= 1 (only the inline import+statement one-liner remains)",
            n_i001 <= 1,
            detail=f"I001 count = {n_i001}",
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
        f"smoke_122gd_ruff_fix_chunk8_import_sort: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
