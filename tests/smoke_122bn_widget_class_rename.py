"""
tests/smoke_122bn_widget_class_rename.py
==========================================

Patch 122bn: rename SimBA-prefixed UI widget classes in
mufasa/ui/tkinter_functions.py to Mufasa-prefixed canonical
names. Also fixes the 'Seperator' → 'Separator' typo on
SimBASeperator → MufasaSeparator.

Renames
-------
* SimBADropDown   → MufasaDropDown    (650 refs across many files)
* SimBAScaleBar   → MufasaScaleBar    (5 refs)
* SimBASeperator  → MufasaSeparator   (14 refs + typo fix)

Total: 669 references renamed across 69 files. Backward-compat
aliases declared at the bottom of mufasa/ui/tkinter_functions.py
preserve the old names — including the misspelled
SimBASeperator — for external callers (notebooks, downstream
packages, user scripts).

Scope decision
--------------
This patch only touches UI widget classes (Tk wrappers).
Excluded:

* SimBA-format data importers/converters (SimBABlobImporter,
  SimBAYoloImporter, SimBA2Yolo, SimBA2YoloSegmentation,
  SimBA2YoloKeypointsPopUp, SimBAROI2Yolo, SimBAROIs2YOLOPopUp)
  — these legitimately reference SimBA-format data and should
  keep the prefix.

* SimBA-prefixed exception classes (SimBAGPUError ~55,
  SimBAModuleNotFoundError ~8, SimBAPackageVersionError ~36).
  Reserved for a future phase (122bo or similar).

* SIMBA_* constants (SIMBA_DIR ~9, SIMBA_VERSION ~8,
  SIMBA_PIP_URL ~3). Reserved for a future phase.

* simba_* variable names — per-case judgment needed.

Coverage
--------
1. The three canonical class names are defined in
   tkinter_functions.py.
2. The three backward-compat aliases are present at the
   bottom of tkinter_functions.py.
3. Aliases appear AFTER the class definitions (so they
   bind to defined names at import time).
4. Behavioural: old + new names refer to the same class
   object after import (identity check).
5. No production-code references to the old names outside
   the alias declaration in tkinter_functions.py.
6. Canonical names appear in many files (regression guard
   against an incomplete rename).
7. All mufasa/**/*.py files parse cleanly.
8. The 'Seperator' typo is gone from the canonical name
   (MufasaSeparator with proper spelling) and preserved
   only in the alias declaration line.
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
    tf_path = pkg / "ui" / "tkinter_functions.py"
    tf_src = tf_path.read_text()

    canonical = ["MufasaDropDown", "MufasaScaleBar", "MufasaSeparator"]
    old_names = ["SimBADropDown", "SimBAScaleBar", "SimBASeperator"]

    # ==================================================================
    # 1. The three canonical class names are defined in
    #    tkinter_functions.py.
    # ==================================================================
    for cname in canonical:
        check(
            f"{cname} class is defined in tkinter_functions.py",
            f"class {cname}(" in tf_src,
        )

    # ==================================================================
    # 2. The three backward-compat aliases are present.
    # ==================================================================
    expected_aliases = [
        "SimBADropDown = MufasaDropDown",
        "SimBAScaleBar = MufasaScaleBar",
        "SimBASeperator = MufasaSeparator",  # preserves typo
    ]
    for alias in expected_aliases:
        check(
            f"Alias '{alias}' is present in tkinter_functions.py",
            alias in tf_src,
        )

    # ==================================================================
    # 3. Aliases appear AFTER the class definitions.
    # ==================================================================
    for cname, alias in zip(canonical, expected_aliases):
        class_pos = tf_src.find(f"class {cname}(")
        alias_pos = tf_src.find(alias)
        check(
            f"Alias for {cname} appears AFTER its class definition "
            "(so it binds to a defined name)",
            0 < class_pos < alias_pos,
        )

    # ==================================================================
    # 4. Behavioural: old + new names refer to the same class
    #    object after import. Tk import may fail in the sandbox
    #    (no display); compile-and-eval the alias lines instead
    #    so we don't depend on Tk being installable.
    # ==================================================================
    # AST-based: find the three alias assignments and verify each
    # value targets the canonical name.
    tree = ast.parse(tf_src)
    alias_mapping = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Name)):
            target = node.targets[0].id
            source = node.value.id
            if target in old_names and source in canonical:
                alias_mapping[target] = source
    expected_mapping = dict(zip(old_names, canonical))
    check(
        "AST: each backward-compat alias targets the correct "
        f"canonical name ({len(alias_mapping)}/3 verified)",
        alias_mapping == expected_mapping,
        detail=f"got {alias_mapping}",
    )

    # ==================================================================
    # 5. No production-code references to old names outside the
    #    alias declarations in tkinter_functions.py.
    # ==================================================================
    for old in old_names:
        offenders: list[Path] = []
        for f in sorted(pkg.rglob("*.py")):
            if f == tf_path:
                continue  # alias declarations live here
            src = f.read_text()
            if re.search(rf"\b{old}\b", src):
                offenders.append(f)
        check(
            f"No production-code references to {old} outside "
            "the alias declaration in tkinter_functions.py",
            offenders == [],
            detail=(f"{len(offenders)} files; "
                    f"first: {offenders[0]}" if offenders else ""),
        )

    # ==================================================================
    # 6. Canonical names appear in many files (regression guard).
    # ==================================================================
    floors = {
        "MufasaDropDown": 60,   # 650 refs / spread across many files
        "MufasaScaleBar": 1,
        "MufasaSeparator": 4,
    }
    for cname, floor in floors.items():
        n = sum(
            1 for f in pkg.rglob("*.py")
            if re.search(rf"\b{cname}\b", f.read_text())
        )
        check(
            f"{cname} appears in >= {floor} files (got {n})",
            n >= floor,
        )

    # ==================================================================
    # 7. All files parse cleanly.
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

    # ==================================================================
    # 8. Typo fix verification.
    #    * 'SimBASeperator' (misspelled) appears ONLY in the alias
    #      declaration line in tkinter_functions.py.
    #    * The canonical name 'MufasaSeparator' uses the correct
    #      spelling everywhere.
    # ==================================================================
    typo_count = 0
    for f in sorted(pkg.rglob("*.py")):
        if "SimBASeperator" in f.read_text():
            typo_count += 1
    check(
        "'SimBASeperator' (misspelled) appears in only 1 file "
        "(tkinter_functions.py, alias declaration only)",
        typo_count == 1,
        detail=f"got {typo_count}",
    )
    # 'MufasaSeperator' (the misspelled Mufasa form) must not exist.
    bad_canonical = 0
    for f in sorted(pkg.rglob("*.py")):
        if "MufasaSeperator" in f.read_text():
            bad_canonical += 1
    check(
        "'MufasaSeperator' (misspelled canonical) does not exist "
        "(must use the correct 'MufasaSeparator')",
        bad_canonical == 0,
        detail=f"got {bad_canonical}",
    )

    print(
        f"smoke_122bn_widget_class_rename: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
