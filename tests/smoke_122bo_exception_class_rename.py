"""
tests/smoke_122bo_exception_class_rename.py
=============================================

Patch 122bo: rename SimBA-prefixed exception classes (and the
base class SimbaError) to Mufasa-prefixed canonical names.
Also drops 'SIMBA ' from the message-formatter prefixes inside
each exception's __init__ body (47 occurrences), consistent
with 122bl's drop of "SIMBA ERROR:" / "SIMBA WARNING:" at the
call sites.

Renames
-------
* SimbaError                → MufasaError                (58 refs, base class)
* SimBAGPUError             → MufasaGPUError             (55 refs)
* SimBAModuleNotFoundError  → MufasaModuleNotFoundError  (8 refs)
* SimBAPackageVersionError  → MufasaPackageVersionError  (36 refs)

Total: 157 refs renamed across 29 files. Plus 55 subclass
parent-class declarations updated in errors.py (every
`class X(SimbaError)` → `class X(MufasaError)`).

Plus the 47-occurrence message-formatter prefix drop:
  Before: msg = f"SIMBA NO DATA ERROR: {msg}"
  After:  msg = f"NO DATA ERROR: {msg}"
  (and similar for each exception class)

Backward-compat aliases declared at the bottom of
mufasa/utils/errors.py preserve the old names:

    SimbaError                = MufasaError
    SimBAGPUError             = MufasaGPUError
    SimBAModuleNotFoundError  = MufasaModuleNotFoundError
    SimBAPackageVersionError  = MufasaPackageVersionError

(plus the existing 122bl alias SimBAPAckageVersionError =
MufasaPackageVersionError for the long-standing typo)

Coverage
--------
1. Four canonical class names defined in errors.py.
2. Four backward-compat aliases present in errors.py.
3. Aliases appear AFTER class definitions.
4. AST: each alias targets the correct canonical name.
5. The pre-existing 122bl typo alias still resolves to the
   correctly-spelled canonical class.
6. No production-code references to the old names outside
   the alias declarations in errors.py.
7. Canonical names appear in many files.
8. All mufasa/**/*.py files parse cleanly.
9. The 'SIMBA ' prefix is gone from the message-formatter
   strings inside __init__ bodies (47 occurrences).
10. Pre-existing class parent-class declarations all reference
    MufasaError now (no `class X(SimbaError):` remains except
    via the alias).
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
    errors_path = pkg / "utils" / "errors.py"
    errors_src = errors_path.read_text()

    canonical = [
        "MufasaError",
        "MufasaGPUError",
        "MufasaModuleNotFoundError",
        "MufasaPackageVersionError",
    ]
    old_names = [
        "SimbaError",
        "SimBAGPUError",
        "SimBAModuleNotFoundError",
        "SimBAPackageVersionError",
    ]

    # ==================================================================
    # 1. Canonical class names are defined.
    # ==================================================================
    for cname in canonical:
        check(
            f"{cname} class is defined in errors.py",
            f"class {cname}(" in errors_src,
        )

    # ==================================================================
    # 2. Four backward-compat aliases are present.
    # ==================================================================
    expected_aliases = [
        "SimbaError = MufasaError",
        "SimBAGPUError = MufasaGPUError",
        "SimBAModuleNotFoundError = MufasaModuleNotFoundError",
        "SimBAPackageVersionError = MufasaPackageVersionError",
    ]
    for alias in expected_aliases:
        check(
            f"Alias '{alias}' is present in errors.py",
            alias in errors_src,
        )

    # ==================================================================
    # 3. Aliases appear AFTER class definitions.
    # ==================================================================
    for cname, alias in zip(canonical, expected_aliases):
        class_pos = errors_src.find(f"class {cname}(")
        alias_pos = errors_src.find(alias)
        check(
            f"Alias '{alias}' appears AFTER class {cname}",
            0 < class_pos < alias_pos,
        )

    # ==================================================================
    # 4. AST: each alias targets the correct canonical name.
    # ==================================================================
    tree = ast.parse(errors_src)
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
        f"AST: each alias targets correct canonical name "
        f"({len(alias_mapping)}/4 verified)",
        alias_mapping == expected_mapping,
        detail=f"got {alias_mapping}",
    )

    # ==================================================================
    # 5. Pre-existing 122bl typo alias still resolves correctly.
    # ==================================================================
    typo_alias = "SimBAPAckageVersionError = MufasaPackageVersionError"
    check(
        f"Pre-existing 122bl typo alias resolves to "
        "MufasaPackageVersionError",
        typo_alias in errors_src,
    )

    # ==================================================================
    # 6. No production-code refs to the old names outside the
    #    alias declarations in errors.py.
    # ==================================================================
    for old in old_names:
        offenders: list[Path] = []
        for f in sorted(pkg.rglob("*.py")):
            if f == errors_path:
                continue  # alias decls live here
            src = f.read_text()
            if re.search(rf"\b{old}\b", src):
                offenders.append(f)
        check(
            f"No production-code references to {old} outside "
            "errors.py alias declarations",
            offenders == [],
            detail=(f"{len(offenders)} files; "
                    f"first: {offenders[0]}" if offenders else ""),
        )

    # ==================================================================
    # 7. Canonical names appear in many files.
    # ==================================================================
    floors = {
        "MufasaError":               1,   # mostly inside errors.py
        "MufasaGPUError":            5,
        "MufasaModuleNotFoundError": 2,
        "MufasaPackageVersionError": 10,
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
    # 8. All files parse cleanly.
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
    # 9. 'SIMBA ' prefix is gone from message-formatter strings
    #    inside __init__ bodies.
    # ==================================================================
    msg_simba_count = errors_src.count('msg = f"SIMBA ')
    check(
        "No 'msg = f\"SIMBA ' prefix remaining in errors.py "
        "(__init__ message formatters cleaned)",
        msg_simba_count == 0,
        detail=f"got {msg_simba_count}",
    )

    # ==================================================================
    # 10. All subclass parent-class declarations now reference
    #     MufasaError, not SimbaError. The alias declaration line
    #     itself uses SimbaError but is on the LHS of `=`, not in
    #     a `class X(...)` parent.
    # ==================================================================
    simba_parent_count = len(re.findall(
        r"class \w+\(SimbaError\)",
        errors_src,
    ))
    mufasa_parent_count = len(re.findall(
        r"class \w+\(MufasaError\)",
        errors_src,
    ))
    check(
        f"No class definitions use SimbaError as parent "
        f"(got {simba_parent_count})",
        simba_parent_count == 0,
    )
    check(
        f"MufasaError is used as parent in >= 50 class "
        f"definitions (got {mufasa_parent_count})",
        mufasa_parent_count >= 50,
    )

    print(
        f"smoke_122bo_exception_class_rename: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
