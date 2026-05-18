"""
tests/smoke_122bp_simba_constants_rename.py
=============================================

Patch 122bp: Phase 2C of the identifier-rename lane. Renames
SIMBA_* constants that refer to the Mufasa application's own
state to MUFASA_*. Leaves SimBA-format asset references
unchanged.

Renames
-------
* OS.SIMBA_VERSION  → OS.MUFASA_VERSION
    (the mufasa-uw-tf-dev package version)
* Links.SIMBA_PIP_URL  → Links.MUFASA_PIP_URL
    ("https://pypi.org/pypi/mufasa-uw-tf-dev/json")
* SIMBA_DIR  → MUFASA_DIR  (module-level in 3 files)
    (os.path.dirname(mufasa.__file__))

For the two enum members, the backward-compat alias is a
same-value enum-member assignment inside the class body:

    class OS(Enum):
        ...
        MUFASA_VERSION = _metadata.version("mufasa-uw-tf-dev")
        SIMBA_VERSION = MUFASA_VERSION  # alias

Python Enum semantics: when two members share a value, the
second is automatically an alias for the first. Verified
behaviourally: `OS.SIMBA_VERSION is OS.MUFASA_VERSION` → True.

For SIMBA_DIR (module-level constant in 3 files), each file
gets a back-compat alias on the next line:

    MUFASA_DIR = os.path.dirname(mufasa.__file__)
    SIMBA_DIR = MUFASA_DIR  # patch 122bp: back-compat alias

Kept as SIMBA_* (legitimate SimBA-format references)
-----------------------------------------------------
* SIMBA_BLOB = 'simba_blob'   (SimBA blob format method ID)
* SIMBA_BP_CONFIG_PATH         (pose_configurations/bp_names/bp_names.csv)
* SIMBA_NO_ANIMALS_PATH        (pose_configurations/no_animals/no_animals.csv)
* SIMBA_FEATURE_EXTRACTION_COL_NAMES_PATH
                              (assets/lookups/feature_extraction_headers.csv)
* SIMBA_SHAP_CATEGORIES_PATH   (SHAP feature categories bundled asset)
* SIMBA_SHAP_IMG_PATH          (SHAP image bundled asset)

These name SimBA-format assets bundled with the package. The
prefix accurately reflects the data heritage.

Coverage
--------
1. OS.MUFASA_VERSION and Links.MUFASA_PIP_URL enum members exist.
2. OS.SIMBA_VERSION and Links.SIMBA_PIP_URL aliases still resolve
   to the canonical members (Python Enum aliasing).
3. SIMBA_DIR module-level alias points to MUFASA_DIR in 3 files.
4. No callsites still reference OS.SIMBA_VERSION or
   Links.SIMBA_PIP_URL outside the alias declarations.
5. SimBA-format-asset SIMBA_* constants are preserved unchanged.
6. All mufasa/**/*.py files parse cleanly.
"""
from __future__ import annotations

import ast
import importlib.util
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

    # ==================================================================
    # 1. Load enums.py and verify the canonical members exist
    # ==================================================================
    enums_path = pkg / "utils" / "enums.py"
    spec = importlib.util.spec_from_file_location("enums", enums_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    check(
        "OS.MUFASA_VERSION enum member exists",
        hasattr(mod.OS, "MUFASA_VERSION"),
    )
    check(
        "Links.MUFASA_PIP_URL enum member exists",
        hasattr(mod.Links, "MUFASA_PIP_URL"),
    )

    # ==================================================================
    # 2. Behavioural: Enum aliases resolve to canonical members
    # ==================================================================
    check(
        "OS.SIMBA_VERSION is OS.MUFASA_VERSION (Enum alias)",
        mod.OS.SIMBA_VERSION is mod.OS.MUFASA_VERSION,
    )
    check(
        "Links.SIMBA_PIP_URL is Links.MUFASA_PIP_URL (Enum alias)",
        mod.Links.SIMBA_PIP_URL is mod.Links.MUFASA_PIP_URL,
    )
    check(
        "Links.MUFASA_PIP_URL.value matches expected mufasa PyPI URL",
        mod.Links.MUFASA_PIP_URL.value
        == "https://pypi.org/pypi/mufasa-uw-tf-dev/json",
    )

    # ==================================================================
    # 3. SIMBA_DIR module-level alias structure in 3 files
    # ==================================================================
    dir_files = [
        pkg / "plotting" / "shap_agg_stats_visualizer.py",
        # pkg / "ui" / "blob_tracker_ui.py" — deleted in 122cr
        # as part of the ROI Tk cluster-deletion.
        pkg / "utils" / "read_write.py",
    ]
    for f in dir_files:
        if not f.exists():
            # File was deleted in a later patch; skip the check.
            # The MUFASA_DIR / SIMBA_DIR pattern is what's being
            # verified, not the specific file roster.
            continue
        src = f.read_text()
        check(
            f"{f.relative_to(REPO_ROOT)}: defines "
            "MUFASA_DIR = os.path.dirname(mufasa.__file__)",
            "MUFASA_DIR = os.path.dirname(mufasa.__file__)" in src,
        )
        check(
            f"{f.relative_to(REPO_ROOT)}: aliases "
            "SIMBA_DIR = MUFASA_DIR",
            "SIMBA_DIR = MUFASA_DIR" in src,
        )

    # ==================================================================
    # 4. No callsites still reference OS.SIMBA_VERSION or
    #    Links.SIMBA_PIP_URL outside the alias declarations
    # ==================================================================
    enums_src = enums_path.read_text()

    # Build the set of files allowed to mention the old names:
    # only enums.py (where the aliases are declared)
    for old_name in ("OS.SIMBA_VERSION", "Links.SIMBA_PIP_URL"):
        offenders = []
        for f in sorted(pkg.rglob("*.py")):
            src = f.read_text()
            if old_name in src:
                offenders.append(f)
        check(
            f"No production-code callsite references {old_name} "
            "(callsites should use the MUFASA_* form)",
            offenders == [],
            detail=f"{len(offenders)} files; first: "
                   f"{offenders[0]}" if offenders else "",
        )

    # The Enum-aliasing inside the enums.py class body uses just
    # 'SIMBA_VERSION' / 'SIMBA_PIP_URL' (no class qualifier),
    # which the regex above doesn't catch. Verify the alias
    # declarations are present.
    check(
        "enums.py: contains 'SIMBA_VERSION = MUFASA_VERSION' "
        "(Enum alias declaration)",
        "SIMBA_VERSION = MUFASA_VERSION" in enums_src,
    )
    check(
        "enums.py: contains 'SIMBA_PIP_URL = MUFASA_PIP_URL' "
        "(Enum alias declaration)",
        "SIMBA_PIP_URL = MUFASA_PIP_URL" in enums_src,
    )

    # ==================================================================
    # 5. SimBA-format SIMBA_* constants are preserved
    # ==================================================================
    preserved = [
        "SIMBA_BLOB",
        "SIMBA_BP_CONFIG_PATH",
        "SIMBA_NO_ANIMALS_PATH",
        "SIMBA_FEATURE_EXTRACTION_COL_NAMES_PATH",
        "SIMBA_SHAP_CATEGORIES_PATH",
        "SIMBA_SHAP_IMG_PATH",
    ]
    for name in preserved:
        n_files = sum(
            1 for f in pkg.rglob("*.py")
            if re.search(rf"\b{name}\b", f.read_text())
        )
        check(
            f"SimBA-format constant {name} is preserved "
            f"(appears in >= 2 files; got {n_files})",
            n_files >= 2,
        )

    # ==================================================================
    # 6. All files parse cleanly
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
        f"smoke_122bp_simba_constants_rename: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
