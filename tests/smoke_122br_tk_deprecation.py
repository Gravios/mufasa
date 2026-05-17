"""
tests/smoke_122br_tk_deprecation.py
=====================================

Patch 122br: adds deprecation banners to the Tk surface
(mufasa/SimBA.py, mufasa/ui/__init__.py, mufasa/ui/tkinter_functions.py)
and ships docs/tk_surface_audit.md.

No renames, no code-path changes. The banners emit a
DeprecationWarning on import and the docstrings explain the
removal plan. The audit doc inventories all 96 files in
mufasa/ui/ + SimBA.py and identifies the dependency chain
that has to be broken before the Tk surface can be removed.

This patch pivots the cleanup strategy: instead of continuing
SimBA → Mufasa renames in Tk-only code (which would just add
permanent back-compat alias debt for vanishing code), the
work shifts to documenting and planning removal.

Coverage
--------
1. docs/tk_surface_audit.md exists and references the four
   status categories.
2. docs/README.md references tk_surface_audit.md (so the
   doc is discoverable from the index).
3. mufasa/SimBA.py has a deprecation docstring + emits
   DeprecationWarning on import.
4. mufasa/ui/__init__.py exists with deprecation docstring +
   DeprecationWarning.
5. mufasa/ui/tkinter_functions.py has a deprecation docstring +
   DeprecationWarning.
6. The deprecation warnings are emitted BEFORE any
   `warnings.filterwarnings("ignore", ...)` calls that
   would suppress them.
7. All mufasa/**/*.py files parse cleanly.
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


def first_deprecation_warn_line(src: str) -> int | None:
    """Find the line number where warnings.warn(..., DeprecationWarning, ...) appears."""
    for i, line in enumerate(src.splitlines(), 1):
        if "warnings.warn(" in line.replace(" ", "") or "_warnings.warn(" in line:
            # crude but adequate — we'll verify DeprecationWarning is on
            # one of the next few lines
            for j in range(i, min(i + 8, len(src.splitlines()) + 1)):
                if "DeprecationWarning" in src.splitlines()[j - 1]:
                    return i
    return None


def first_filterwarnings_ignore_line(src: str) -> int | None:
    """Find the line where warnings.filterwarnings('ignore', ...) appears."""
    for i, line in enumerate(src.splitlines(), 1):
        if ("filterwarnings" in line
                and "ignore" in line):
            return i
    return None


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # 1. tk_surface_audit.md exists and is well-formed
    # ==================================================================
    audit_path = REPO_ROOT / "docs" / "tk_surface_audit.md"
    check(
        "docs/tk_surface_audit.md exists",
        audit_path.exists(),
    )
    if audit_path.exists():
        audit_text = audit_path.read_text()
        for status in ("LOAD-BEARING-FOR-QT", "TK-REACHABLE",
                       "TK-INTERNAL-ONLY", "UNREFERENCED"):
            check(
                f"audit doc references status '{status}'",
                status in audit_text,
            )
        check(
            "audit doc has a 'Removal-dependency graph' section "
            "(step-ordered)",
            "Removal-dependency graph" in audit_text,
        )
        check(
            "audit doc explicitly says don't rename Tk-only code",
            "Don't rename Tk-only code" in audit_text
            or "don't rename Tk-only code" in audit_text.lower(),
        )

    # ==================================================================
    # 2. docs/README.md references the audit
    # ==================================================================
    readme_path = REPO_ROOT / "docs" / "README.md"
    check(
        "docs/README.md references tk_surface_audit.md",
        readme_path.exists()
        and "tk_surface_audit.md" in readme_path.read_text(),
    )

    # ==================================================================
    # 3. mufasa/SimBA.py deprecation
    # ==================================================================
    simba_path = pkg / "SimBA.py"
    simba_src = simba_path.read_text()
    check(
        "SimBA.py: module docstring mentions deprecated/DEPRECATED",
        re.search(r"deprecat", simba_src[:1500], re.IGNORECASE) is not None,
    )
    check(
        "SimBA.py: emits DeprecationWarning on import",
        "DeprecationWarning" in simba_src[:1500],
    )
    check(
        "SimBA.py: references tk_surface_audit.md in deprecation note",
        "tk_surface_audit.md" in simba_src,
    )

    # ==================================================================
    # 4. mufasa/ui/__init__.py deprecation
    # ==================================================================
    ui_init = pkg / "ui" / "__init__.py"
    check(
        "mufasa/ui/__init__.py exists with content",
        ui_init.exists() and ui_init.read_text().strip() != "",
    )
    ui_init_src = ui_init.read_text()
    check(
        "ui/__init__.py: emits DeprecationWarning",
        "DeprecationWarning" in ui_init_src,
    )
    check(
        "ui/__init__.py: references tk_surface_audit.md",
        "tk_surface_audit.md" in ui_init_src,
    )

    # ==================================================================
    # 5. mufasa/ui/tkinter_functions.py deprecation
    # ==================================================================
    tf = pkg / "ui" / "tkinter_functions.py"
    tf_src = tf.read_text()
    check(
        "tkinter_functions.py: module docstring mentions deprecated",
        re.search(r"deprecat", tf_src[:1500], re.IGNORECASE) is not None,
    )
    check(
        "tkinter_functions.py: emits DeprecationWarning on import",
        "DeprecationWarning" in tf_src[:1500],
    )
    check(
        "tkinter_functions.py: references tk_surface_audit.md",
        "tk_surface_audit.md" in tf_src,
    )

    # ==================================================================
    # 6. Deprecation warnings fire BEFORE any filterwarnings('ignore')
    #    in the same file. (SimBA.py has a filterwarnings call later in
    #    the module; the warning must be emitted before that runs.)
    # ==================================================================
    warn_ln = first_deprecation_warn_line(simba_src)
    filter_ln = first_filterwarnings_ignore_line(simba_src)
    check(
        f"SimBA.py: DeprecationWarning (line ~{warn_ln}) is emitted "
        f"BEFORE filterwarnings('ignore') (line ~{filter_ln})",
        warn_ln is not None and filter_ln is not None and warn_ln < filter_ln,
        detail=f"warn={warn_ln}, filter={filter_ln}",
    )

    # ==================================================================
    # 7. All files parse cleanly
    # ==================================================================
    parse_errors: list[str] = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py files parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122br_tk_deprecation: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
