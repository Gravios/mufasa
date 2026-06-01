"""
tests/smoke_122gb_ruff_fix_chunk6_safe_tail.py
==============================================

Patch 122gb — ruff --fix sweep chunk 6/N: the remaining SAFE autofixes
(the inert tail of the [*]-classified rules).

User request: "continue" (resuming the chunked ruff sweep).

CHUNK 6 RULES (all [*] safe; behavior-preserving)
=================================================
  E713  not-in-test               not x in y     -> x not in y
  E703  useless-semicolon
  E714  not-is-test               not x is y     -> x is not y
  SIM114 if-with-same-arms        merge identical branches with `or`
  SIM300 yoda-conditions          5 == x         -> x == 5
  UP024 os-error-alias            IOError        -> OSError
  UP017 datetime-timezone-utc     timezone.utc   -> datetime.UTC
  UP010 unnecessary-future-import
  UP009 utf8-encoding-declaration (drop the coding cookie)
  UP021 replace-universal-newlines  universal_newlines= -> text=
  B013  redundant-tuple-in-exception-handler  except (X,): -> except X:

Command: ruff check mufasa --select <those 11 rules> --fix   (81 fixes)

DELIBERATELY EXCLUDED: UP033 (lru-cache-with-maxsize-none -> @cache).
Although [*] safe, its 3 occurrences live in mufasa/ui_qt/ (which has
strict F401/I001/W292 lint smokes — 122dg/122do) and one is pinned by
smoke_122du ("icon_for_status is @lru_cache-decorated"). Rewriting it
orphans the `lru_cache` import and breaks those four smokes for a
cosmetic gain. Left for a deliberate, ui_qt-aware change later.

ONE FOLLOW-UP CLEANUP
=====================
UP017 rewrote `datetime.now(timezone.utc)` -> `datetime.now(UTC)` in
section_provenance.py, orphaning the `timezone` import. Trimmed that
import to `from datetime import datetime, UTC` so the F401 guard
(122fy: "no safe-fixable F401 remain") stays satisfied.

RESULT / VERIFICATION
=====================
* 81 fixes / 38 files. Selected rules -> 0. F821 unchanged (7);
  F401 safe-fixable -> 0; total selected 2975 -> ~2891. compileall clean.
* d/e/f sweep 0 fails; full smoke_122*.py baseline-diffed: a/b/c partial
  set == pre-existing baseline, no net regressions.

NEW SMOKE: smoke_122gb_ruff_fix_chunk6_safe_tail.py (4 checks)
* mufasa/ parses cleanly
* ruff reports 0 for the 11 selected safe rules (regression tripwire)
* F821 still exactly the 7 known-deliberate
* section_provenance.py no longer imports `timezone` (UP017 follow-up)
  -- ruff parts soft-pass with a note if ruff is unavailable.

This essentially exhausts the SAFE ruff autofixes. What remains is
[-] unsafe-classed (I001 import-sort, SIM118, F841, W293) and the no-fix
manual rules (E701/E702/E402/B007/B904/B905/E722/...), which need
deliberate per-rule review rather than a blind sweep.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHUNK_RULES = "E713,E703,E714,SIM114,SIM300,UP024,UP017,UP010,UP009,UP021,B013"

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

    prov_tree = ast.parse((pkg / "section_provenance.py").read_text())
    dt_imports = [
        {a.name for a in n.names}
        for n in ast.walk(prov_tree)
        if isinstance(n, ast.ImportFrom) and n.module == "datetime"
    ]
    dt_names = set().union(*dt_imports) if dt_imports else set()
    check(
        "section_provenance.py imports UTC, not timezone (UP017 follow-up)",
        "UTC" in dt_names and "timezone" not in dt_names,
        detail=f"datetime imports = {sorted(dt_names)}",
    )

    ruff = shutil.which("ruff")
    if ruff is None:
        print("NOTE: ruff not found on PATH — chunk-6 rule checks skipped (soft pass).")
        check("ruff reports 0 for chunk-6 safe rules", True)
        check("F821 still the 7 known-deliberate", True)
    else:
        proc = subprocess.run(
            [ruff, "check", str(pkg), "--select", CHUNK_RULES, "--quiet"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        check(
            "ruff reports 0 for the 11 selected safe rules",
            proc.returncode == 0,
            detail=proc.stdout.strip()[:200],
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
        f"smoke_122gb_ruff_fix_chunk6_safe_tail: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
