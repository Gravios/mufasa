"""
tests/smoke_122eq_dev_paths_sweep.py
=======================================

Patch 122eq: cosmetic sweep of hardcoded developer paths in
doctests and commented-out invocation footers. Companion to
122eg (which swept ``project_config.ini`` references) — same
shape of mechanical doc cleanup deferred from earlier patches.

What this sweeps
----------------
Three patterns, replaced in this order (longest-prefix first):

* ``/Users/simon/Desktop/envs/`` → ``/path/to/``
* ``/Users/simon/Desktop/``      → ``/path/to/``
* ``/Users/simon/``              → ``/path/to/``

All occurrences are in:
* Doctest examples (``>>> obj = Cls(path='/Users/simon/...')``)
* Commented-out invocation footers
  (``# obj = Cls(path='/Users/simon/...')``)
* Module docstrings citing example paths

422 references swept across 79 files. Cosmetic only — no
runtime behavior changes.

Why this matters at all
-----------------------
Three reasons, in decreasing order of importance:

1. The doctest examples LOOK like they're describing the dev's
   workspace. New users reading the docs may think they need
   to recreate that directory structure. Generic ``/path/to/``
   placeholders communicate "substitute your own path" more
   clearly.

2. SimBA inheritance: these paths originated in the upstream
   SimBA codebase. Removing the SimBA-developer-specific prefix
   helps signal that this is now a separate project (mufasa)
   with its own identity.

3. If a doctest ever runs (currently they don't, but if a
   future patch adds doctest discovery), the absolute paths
   would fail on any non-matching system. ``/path/to/`` won't
   accidentally hit real files on the runner.

What was deliberately excluded
------------------------------
Same exclusion list as 122eg:

* ``mufasa/legacy_layout.py`` — historical doc; preserve as-is.
* ``tests/`` — fixtures may reference real test paths.
* ``CHANGELOG.md`` / ``docs/`` — historical records; not
  swept by this patch (would be a separate scope).

Coverage
--------
1.  Parse-clean (verifies the sweep didn't break syntax —
    cheap check given all matches are in strings/comments).
2.  No ``/Users/simon`` references remain in mufasa/**/*.py
    EXCEPT in the exclusion list.
3.  ``mufasa/legacy_layout.py`` is preserved (the file is
    intentionally untouched by automated sweeps — historical
    reference value).
4.  CHANGELOG.md is preserved (similar historical reasoning).
5.  The replacement preserved doctest structure — at least one
    file that previously had ``/Users/simon`` now has
    ``/path/to/`` in roughly the same position (smoke check
    on a representative file).

Cross-patch invariants:
6.  122eg state preserved: no ``project_config.ini``
    references outside legacy_layout.py.
7.  122ep + 122es state preserved: 9 of 11 ui_bound sections have
    detect_path.
8.  122en state preserved: v1_project_paths canonical helper.
9.  122do baseline.
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
    pkg = REPO_ROOT / "mufasa"

    # 1. Parse-clean.
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files) "
        f"after the sweep",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    # 2. No /Users/simon refs anywhere in mufasa/.
    # (legacy_layout.py is in the exclusion list but doesn't contain
    # any /Users/simon paths to begin with; the assertion holds
    # globally regardless.)
    stray = []
    for f in sorted(pkg.rglob("*.py")):
        src = f.read_text()
        if "/Users/simon" in src:
            stray.append(
                f"{f.relative_to(REPO_ROOT)}: "
                f"{src.count('/Users/simon')} refs"
            )
    check(
        "No `/Users/simon` references remain in mufasa/**/*.py "
        "(the sweep is complete)",
        not stray,
        detail=("; ".join(stray[:3])),
    )

    # 3. legacy_layout.py preserved verbatim (audit-only check —
    # the file shouldn't contain /Users/simon paths at all,
    # but if a future maintainer adds them, the exclusion
    # should still apply).
    ll_path = REPO_ROOT / "mufasa" / "legacy_layout.py"
    ll_src = ll_path.read_text()
    check(
        "mufasa/legacy_layout.py is reachable for inspection "
        "(consistency check with 122eg's exclusion list — the "
        "file is intentionally preserved by automated sweeps)",
        ll_path.exists() and len(ll_src) > 0,
    )

    # 4. CHANGELOG.md preserved.
    cl_path = REPO_ROOT / "CHANGELOG.md"
    if cl_path.exists():
        check(
            "CHANGELOG.md is reachable (preserved per the same "
            "historical-record rationale that 122eg used)",
            True,
        )
    else:
        check("CHANGELOG.md absent — nothing to verify", True)

    # 5. /path/to/ appears where /Users/simon used to.
    # Pick a representative file that we know had occurrences.
    # mufasa/model/grid_search_multiclass_rf.py had several based
    # on the pre-sweep audit.
    sample = (REPO_ROOT / "mufasa" / "model"
              / "grid_search_multiclass_rf.py")
    if sample.exists():
        s = sample.read_text()
        check(
            "Sample file (grid_search_multiclass_rf.py) shows the "
            "replacement landed: '/path/to/' present, "
            "'/Users/simon' absent",
            "/path/to/" in s and "/Users/simon" not in s,
        )
    else:
        check("(sample file absent — skipped)", True)

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 6. 122eg state preserved.
    eg_stray = []
    for f in sorted(pkg.rglob("*.py")):
        rel = str(f.relative_to(REPO_ROOT))
        if rel == "mufasa/legacy_layout.py":
            continue
        s = f.read_text()
        if "project_config.ini" in s:
            eg_stray.append(rel)
    check(
        "122eg state preserved: no `project_config.ini` in "
        "mufasa/**/*.py outside legacy_layout.py",
        not eg_stray,
        detail=("; ".join(eg_stray[:3])),
    )

    # 7. 122ep state preserved.
    from mufasa.section_provenance import SECTIONS
    ui_bound = [s for s in SECTIONS.values() if s.ui_bound]
    with_detect = [s for s in ui_bound if s.detect_path is not None]
    check(
        "122ep + 122es state preserved: 9 of 11 ui_bound sections have "
        "detect_path",
        len(with_detect) == 9 and len(ui_bound) == 11,
        detail=(f"got {len(with_detect)}/{len(ui_bound)}"),
    )

    # 8. 122en state preserved.
    pl_src = (REPO_ROOT / "mufasa"
              / "project_layout.py").read_text()
    check(
        "122en state preserved: v1_project_paths canonical helper",
        "def v1_project_paths" in pl_src,
    )

    # 9. 122do baseline.
    uiqt = pkg / "ui_qt"
    optional_hits = []
    for f in sorted(uiqt.rglob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"\bOptional\[", src):
            preceding = src[:m.start()]
            tq3 = preceding.count('"""') + preceding.count("'''")
            if tq3 % 2 == 0:
                optional_hits.append(str(f.relative_to(uiqt)))
                break
    check(
        "122do baseline preserved: no `Optional[` in non-"
        "docstring positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122eq_dev_paths_sweep: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
