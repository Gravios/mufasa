"""
tests/smoke_122bm_simba_to_mufasa_rebrand_phase1.py
=====================================================

Patch 122bm: phase-1 SimBA → Mufasa rebrand of user-facing
strings. Targeted regex replacements applied only to
non-comment, non-docstring lines in mufasa/**/*.py.

Conservative scope:
* Print, stdout_*, argparse help/description, exception msg=,
  error_msg=, and inline raise messages.
* Patterns target phrasings unambiguously about the
  application ("SimBA expects", "SimBA tried to", "SimBA
  could not", "Updating settings in SimBA project config",
  "A new version of SimBA is available", etc.).
* Negative lookahead preserves "SimBA project_config.ini"
  (legitimate legacy-format file reference).

NOT touched in this phase (separate future lanes):
* Identifier renames (SimBALabel, SimBADropDown, SIMBA_*
  enum names, simba_* function names — ~872 references).
  Class renaming has ripple effects across imports.
* Legacy Tk launcher UI labels (window title, menu items,
  asset names like "SimBA_logo_3_small").
* External link labels in the legacy help menu ("SimBA
  Github", "SimBA Gitter Support Chatroom") — these
  legitimately point to SimBA project resources.
* Docstring text (separate doc-harmonisation lane).
* Variable/function/module names (e.g., `simba_pip_data`,
  `mufasa.SimBA`, `SimBA_PIP_URL`).

Coverage
--------
1. Zero matches for patterns the rename targeted.
2. "SimBA project_config.ini" references preserved (must
   not have been touched — regression guard).
3. Version-check banner now says "A new version of Mufasa
   is available" (clearly user-application banner; the
   install command already says mufasa-uw-tf-dev).
4. Specific high-value rewrites are in place
   (config_reader error msg, BORIS appender warning, video
   processing overwrite message, argparse help text).
5. All mufasa/**/*.py files parse cleanly (regression
   guard against bad regex slips).
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


# Helper: walk a file's non-comment, non-docstring lines
def non_doc_lines(src: str):
    in_doc = False
    doc_q = None
    for line in src.splitlines():
        for q in ('"""', "'''"):
            if line.count(q) % 2 == 1:
                if not in_doc:
                    in_doc = True
                    doc_q = q
                elif doc_q == q:
                    in_doc = False
                    doc_q = None
        stripped = line.lstrip()
        if stripped.startswith("#") or in_doc:
            continue
        yield line


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # 1. Zero matches for renamed patterns
    # ==================================================================
    renamed_patterns = [
        ("SimBA expects",            "Mufasa expects"),
        ("SimBA tried to grab frame", "Mufasa tried to grab frame"),
        ("SimBA could not",          "Mufasa could not"),
        ("SimBA could only find",    "Mufasa could only find"),
        ("SimBA found",              "Mufasa found"),
        ("SimBA is overwriting",     "Mufasa is overwriting"),
        ("SimBA is not allowed",     "Mufasa is not allowed"),
        ("SimBA was launched",       "Mufasa was launched"),
        ("SimBA CUML enabled",       "Mufasa CUML enabled"),
        ("SimBA blob tracking subprocess",
         "Mufasa blob tracking subprocess"),
        ("SimBA load time",          "Mufasa load time"),
        ("SimBA environment variables",
         "Mufasa environment variables"),
        ("A new version of SimBA",   "A new version of Mufasa"),
        ("SimBA Custom Feature Extractor",
         "Mufasa Custom Feature Extractor"),
        ("Updating settings in SimBA project config",
         "Updating settings in project config"),
        ("Path to SimBA Project config",
         "Path to project config (Mufasa TOML or SimBA INI)"),
        ("Path to SimBA project config",
         "Path to project config (Mufasa TOML or SimBA INI)"),
    ]
    for old, _new in renamed_patterns:
        offenders = []
        for f in sorted(pkg.rglob("*.py")):
            src = f.read_text()
            if old not in src:
                continue
            for line in non_doc_lines(src):
                if old in line:
                    offenders.append(f)
                    break
        check(
            f"Old phrasing {old!r} is gone from "
            "non-doc/non-comment lines",
            offenders == [],
            detail=(f"{len(offenders)} files: "
                    f"{offenders[0]}" if offenders else ""),
        )

    # ==================================================================
    # 2. "SimBA project_config.ini" references PRESERVED
    # ==================================================================
    preserved_count = 0
    for f in sorted(pkg.rglob("*.py")):
        src = f.read_text()
        preserved_count += src.count("SimBA project_config.ini")
    check(
        f"'SimBA project_config.ini' references preserved "
        f"(legitimate file format ref; found {preserved_count})",
        preserved_count >= 5,  # 19 at audit time; floor of 5 is conservative
        detail=f"got {preserved_count}",
    )

    # Negative-lookahead correctness check: the rename should
    # NOT have transformed "SimBA project_config.ini" into
    # "Mufasa project_config.ini". The file-format string IS
    # legitimately "project_config.ini" in both contexts:
    # * "SimBA project_config.ini" = "an INI from a SimBA project"
    # * "Mufasa project_config.ini" = "your Mufasa project's INI"
    # Pre-existing references to either are kept as-is. We only
    # verify that the *count* of SimBA-qualified references didn't
    # drop (i.e., the rename didn't accidentally swallow some).
    simba_form_count = 0
    for f in sorted(pkg.rglob("*.py")):
        simba_form_count += f.read_text().count("SimBA project_config.ini")
    check(
        f"'SimBA project_config.ini' references preserved at "
        f"audit-time floor (>= 15). Found {simba_form_count}",
        simba_form_count >= 15,
    )

    # ==================================================================
    # 3. Specific high-value rewrites are in place
    # ==================================================================
    simba_py = (pkg / "SimBA.py").read_text()
    check(
        "SimBA.py: version-check banner says 'A new version of Mufasa'",
        "A new version of Mufasa is available" in simba_py,
    )

    config_reader = (pkg / "mixins" / "config_reader.py").read_text()
    check(
        "config_reader.py: error msg says 'Mufasa expects'",
        "Mufasa expects" in config_reader,
    )
    check(
        "config_reader.py: error msg says 'Mufasa found'",
        "Mufasa found" in config_reader,
    )

    boris = (pkg / "third_party_label_appenders"
             / "BORIS_appender.py").read_text()
    check(
        "BORIS_appender.py: warning says 'Mufasa will set'",
        "Mufasa will set" in boris,
    )

    vproc = (pkg / "video_processors" / "video_processing.py").read_text()
    check(
        "video_processing.py: overwrite msg says 'Mufasa is overwriting'",
        "Mufasa is overwriting" in vproc,
    )

    movement = (pkg / "data_processors"
                / "movement_calculator.py").read_text()
    check(
        "movement_calculator.py: argparse help mentions TOML + INI",
        "Mufasa TOML or SimBA INI" in movement,
    )

    cuml = (pkg / "mixins" / "__init__.py").read_text()
    check(
        "mixins/__init__.py: prints 'Mufasa CUML enabled.'",
        "Mufasa CUML enabled." in cuml,
    )

    # ==================================================================
    # 4. All files parse cleanly
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
    # 5. Sanity: no obvious half-rename artifacts in file-format
    #    references. The rename targets WHOLE phrases like
    #    "SimBA project" — a half-rename would leave behind
    #    e.g. "Mufasa project_config_thing" with `_thing` being
    #    the leftover bit. Spot-check for a few suspicious forms.
    # ==================================================================
    # If the negative lookahead had failed, we'd see "Mufasa "
    # immediately before "_config" — i.e., the string
    # "Mufasa _config" (with a space then underscore). That's
    # syntactically impossible to produce by accident with our
    # rename rules, but explicitly verify.
    n_weird = 0
    for f in sorted(pkg.rglob("*.py")):
        text = f.read_text()
        if "Mufasa _config" in text or "Mufasa _proj" in text:
            n_weird += 1
    check(
        "No weird 'Mufasa _config' / 'Mufasa _proj' half-rename leaks",
        n_weird == 0,
        detail=f"got {n_weird}",
    )

    print(
        f"smoke_122bm_simba_to_mufasa_rebrand_phase1: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
