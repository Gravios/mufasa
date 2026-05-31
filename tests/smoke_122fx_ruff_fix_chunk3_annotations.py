"""
tests/smoke_122fx_ruff_fix_chunk3_annotations.py
================================================

Patch 122fx — ruff --fix sweep chunk 3/N: PEP 585/604 annotation
modernization across the whole package.

User request: "continue" (resuming the chunked ruff sweep).

CHUNK 3 RULES (all safe fixes; valid at runtime on target py311)
================================================================
  UP045 Optional[X]  -> X | None       (1805)
  UP006 List/Dict/..  -> list/dict/...  (1636)
  UP007 Union[X, Y]  -> X | Y           (1266)

Command: ruff check mufasa --select UP006,UP007,UP045 --fix

RESULT / VERIFICATION
=====================
* 4707 fixes across 267 files.
* mufasa/ui_qt/ was already modernized by patch 122do (its smoke is
  scoped to ui_qt and stays 15/15); this chunk completed the rest of
  the package.
* Rule deltas: UP006/UP007/UP045 -> 0. F401 ROSE 333 -> 1090 (+757):
  expected — rewriting Optional/Union/List usages orphans their
  `from typing import ...` names. Those are cleaned in the dedicated
  F401 chunk later (an unsafe [-] fix needing per-file review), NOT
  here. Total selected: 8476 -> 4526 (= -4707 UP + 757 F401, exact).
* mufasa/ compiles (compileall clean).

REGRESSION ANALYSIS (full 158-smoke sweep, baseline-diffed)
===========================================================
A discovery this chunk surfaced: the working sweep glob had been
122d*/122e*/122f* (82 smokes); the 122a*/122b*/122c* families (76
more) were never being run. A complete `tests/smoke_122*.py` sweep,
diffed against a stash-baseline at 122fw, shows:
  * This chunk introduced exactly ONE regression: smoke_122dm
    (22 -> 21) — its embedded "ruff F401 clean on touched files"
    check, tripped because UP006 orphaned `typing.Dict` (et al.) in
    roi_tools/roi_logic.py (outside 122do's ui_qt scope). FIXED here
    by trimming that file's typing import to `Any, Literal`.
  * Every other red 122a/b/c smoke was ALREADY red at 122fw —
    pre-existing and unrelated to this chunk: sandbox lacks
    pyarrow/fastparquet (parquet-dependent smokes crash on import);
    several reference files deleted in the SimBA/Tk death cascade;
    a handful carry stale pins from earlier refactors. 122ch is a
    false flag (19/19; interactive-prompt noise on the last line).
  These pre-existing failures are NOT addressed here (out of scope;
  several are purely environmental) but are flagged for follow-up —
  the sweep glob should widen to smoke_122*.py with a known-failing
  allowlist.

NEW SMOKE: smoke_122fx_ruff_fix_chunk3_annotations.py (3 checks)
* mufasa/ parses cleanly
* ruff reports 0 for UP006,UP007,UP045 (regression tripwire)
* roi_logic.py typing import trimmed (122dm contract: F401-clean)
  -- soft-passes the ruff parts with a note if ruff is unavailable.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHUNK_RULES = "UP006,UP007,UP045"

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

    # roi_logic typing import trimmed (122dm contract)
    roi_logic = (REPO_ROOT / "mufasa" / "roi_tools" / "roi_logic.py").read_text()
    check(
        "roi_logic.py typing import trimmed to used names (Any, Literal)",
        "from typing import Any, Literal" in roi_logic
        and "from typing import Any, Dict, List" not in roi_logic,
    )

    ruff = shutil.which("ruff")
    if ruff is None:
        print("NOTE: ruff not found on PATH — chunk-3 rule check skipped (soft pass).")
        check("ruff reports 0 for chunk-3 annotation rules", True)
    else:
        proc = subprocess.run(
            [ruff, "check", str(pkg), "--select", CHUNK_RULES, "--quiet"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        check(
            "ruff reports 0 for chunk-3 annotation rules (UP006,UP007,UP045)",
            proc.returncode == 0,
            detail=proc.stdout.strip()[:200],
        )

    print(
        f"smoke_122fx_ruff_fix_chunk3_annotations: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
