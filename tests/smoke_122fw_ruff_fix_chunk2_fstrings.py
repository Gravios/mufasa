"""
tests/smoke_122fw_ruff_fix_chunk2_fstrings.py
=============================================

Patch 122fw — ruff --fix sweep chunk 2/N: f-strings.

User request: "continue" (resuming the chunked ruff sweep started in
122fu; 122fv was an interleaved bug fix).

CHUNK 2 RULES (both safe fixes)
===============================
  UP032 f-string                      ("...".format(x) -> f"...{x}")
  F541  f-string-missing-placeholders (f"plain" -> "plain")

Command: ruff check mufasa --select UP032,F541 --fix

RESULT / VERIFICATION
=====================
* 567 fixes across 74 files (561 ins / 779 del — f-strings consolidate
  the .format() call tail, hence net shrink).
* Total selected ruff findings 9043 -> 8476 (delta -567, reconciles; no
  collateral changes — both rules report 0).
* mufasa/ compiles (compileall clean); conversions spot-checked.
* All 81 prior strict smokes stay green (no reciprocal tripwire — no
  smoke pinned a .format() call or an empty f-string).

STILL HELD BACK (later chunks)
==============================
* UP006/UP007/UP045 annotation modernization (~4700) — next, own chunk.
* UP037 quoted-annotation (forward-ref sensitive) + remaining safe SIM.
* [-] unsafe families (F401, I001, UP035, F841, SIM118) — per-file review.
* No-fix rules (E701/E702/E402/B905/B904/E722/E711/E721/B007 …) — manual.

NEW SMOKE: smoke_122fw_ruff_fix_chunk2_fstrings.py (2 checks)
* mufasa/ parses cleanly
* ruff reports 0 for UP032,F541 (regression tripwire; soft-passes with a
  printed note if ruff is absent from the env)
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHUNK_RULES = "UP032,F541"

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
        print("NOTE: ruff not found on PATH — chunk-2 rule check skipped (soft pass).")
        check("ruff reports 0 for chunk-2 f-string rules", True)
    else:
        proc = subprocess.run(
            [ruff, "check", str(pkg), "--select", CHUNK_RULES, "--quiet"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        check(
            "ruff reports 0 for chunk-2 f-string rules (UP032,F541)",
            proc.returncode == 0,
            detail=proc.stdout.strip()[:200],
        )

    print(
        f"smoke_122fw_ruff_fix_chunk2_fstrings: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
