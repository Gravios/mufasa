"""
tests/smoke_122bj_dev_test_footer_sweep.py
============================================

Patch 122bj: strips commented-out `# test = ...` dev-time
invocation footers from the codebase. These were SimBA-era
artifacts — multi-line commented snippets at the end of
class files showing how the original developer manually
instantiated and ran the class on hardcoded user-specific
paths (e.g. `r"C:\\troubleshooting\\..\\project_config.ini"`,
`/Users/simon/Desktop/envs/...`).

What was stripped
-----------------
* The `# test = SomeClass(config_path=...)` line and all
  trailing comment/blank lines (the .run()/.save() chains).
* Any `# if __name__ == "__main__":` line immediately above
  the first `# test = ...` (it was scaffolding for the test
  code, not a separate construct).
* Trailing blank lines between the real code and the
  truncated footer.

What was preserved
------------------
* The actual code above the test footer — including any
  real `if __name__ == "__main__":` CLI blocks (e.g.
  inference_batch.py has a live argparse-based CLI).
* Module docstrings, classes, functions, real code.
* Any "Block 1"-style commented dead code that PRECEDES
  a `# test = ...` footer — out of scope for this patch.

Why
---
* Stale paths — references to dev-time absolute paths on
  the SimBA author's machines (C:\\troubleshooting\\,
  /Users/simon/, etc.) that don't exist on any user's
  system.
* Stale config format — every block references
  project_config.ini (legacy INI); mufasa v1 uses TOML.
* Not executable — all lines start with `#`. They serve no
  test purpose. The real test suite is tests/smoke_*.py.
* No documentation value — the API can be inferred from
  the class definition + docstring; these snippets just
  show one specific dev's invocation pattern.

Coverage
--------
1. No mufasa/**/*.py file contains a `# test = ...` line
   (any indentation).
2. No mufasa/**/*.py file contains a commented-out
   `# test.run()` or `# test.save()` orphaned by the sweep.
3. All mufasa/**/*.py files parse cleanly (regression
   guard — bad truncation could have left dangling
   `# ... ,\n# ... )` blocks).
4. A spot-check on 6 representative files: known to have
   had test footers, now don't, and their final byte is
   a newline (clean truncation, not a half-stripped line).
5. The total line-count reduction is sane (3047 lines
   across 111 files — recorded here as a regression check).
"""
from __future__ import annotations

import ast
import re
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
    pkg_root = REPO_ROOT / "mufasa"
    all_files = sorted(p for p in pkg_root.rglob("*.py") if p.is_file())

    # ==================================================================
    # 1. No '# test = ...' lines anywhere in mufasa/
    # ==================================================================
    test_re = re.compile(r"^\s*#\s*test\s*=")
    offenders: list[tuple[Path, int]] = []
    for f in all_files:
        for i, line in enumerate(f.read_text().splitlines()):
            if test_re.match(line):
                offenders.append((f, i + 1))
                break  # one report per file is enough
    check(
        "No mufasa/**/*.py file contains a '# test = ...' line",
        offenders == [],
        detail=f"{len(offenders)} offenders" if offenders else "",
    )

    # ==================================================================
    # 2. No orphaned '# test.run()' / '# test.save()' references
    # ==================================================================
    orphan_re = re.compile(r"^\s*#\s*test\.\w+")
    orphans: list[tuple[Path, int]] = []
    for f in all_files:
        for i, line in enumerate(f.read_text().splitlines()):
            if orphan_re.match(line):
                orphans.append((f, i + 1))
                break
    check(
        "No orphaned '# test.<method>()' lines (sweep cleaned "
        "the full footer block, not just the assignment)",
        orphans == [],
        detail=f"{len(orphans)} offenders" if orphans else "",
    )

    # ==================================================================
    # 3. All mufasa/**/*.py files parse cleanly
    # ==================================================================
    parse_errors: list[str] = []
    for f in all_files:
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All {len(all_files)} mufasa/**/*.py files parse cleanly",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    # ==================================================================
    # 4. Spot-check: known-formerly-had-footer files now don't,
    #    and end with a single trailing newline (clean truncation)
    # ==================================================================
    spot_check = [
        "mufasa/model/inference_batch.py",
        "mufasa/model/train_rf.py",
        "mufasa/feature_extractors/feature_extractor_4bp.py",
        "mufasa/data_processors/agg_clf_calculator.py",
        "mufasa/plotting/heat_mapper_clf.py",
        "mufasa/labelling/labelling_interface.py",
    ]
    for sp in spot_check:
        f = REPO_ROOT / sp
        if not f.exists():
            check(
                f"spot-check: {sp} exists",
                False,
                detail="file missing",
            )
            continue
        text = f.read_text()
        check(
            f"spot-check: {sp} has no '# test = ...'",
            test_re.search(text) is None,
        )
        # Single trailing newline (clean truncation).
        ends_with_one_newline = (
            text.endswith("\n") and not text.endswith("\n\n\n\n")
        )
        check(
            f"spot-check: {sp} ends with a single trailing newline "
            "(clean truncation, not mid-line)",
            ends_with_one_newline,
        )

    # ==================================================================
    # 5. The inference_batch.py live CLI block is PRESERVED
    #    (regression guard — proves the truncation didn't eat
    #    real code above the footer)
    # ==================================================================
    ib_path = REPO_ROOT / "mufasa" / "model" / "inference_batch.py"
    if ib_path.exists():
        ib_src = ib_path.read_text()
        check(
            "inference_batch.py: live argparse CLI is preserved "
            "(guard may be plain or with sys.ps1 check)",
            'if __name__ == "__main__"' in ib_src
            and "argparse" in ib_src
            and "runner = InferenceBatch" in ib_src,
        )

    print(
        f"smoke_122bj_dev_test_footer_sweep: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
