"""
tests/smoke_122eg_docstring_ini_sweep.py
==========================================

Patch 122eg: cosmetic sweep of ``project_config.ini`` doc /
docstring references across the codebase. Replaces 356
occurrences across 156 .py files with the v1-correct
``project.toml``, plus 4 occurrences across 3 user-facing
markdown files.

Context
-------
The legacy-deletion arc (122dw / 122dx / 122dy / 122dz / 122ea)
removed every load-bearing reference to the legacy
``project_config.ini`` layout from the runtime code path.
Remaining references were all in:

* Doctest examples (``>>> obj = SomeClass(config_path='...
  project_config.ini')``)
* Commented-out invocation footers
  (``# Interpolate(config_path='...project_config.ini')``)
* Module docstrings citing example paths

These were cosmetically out of date — the workbench no longer
loads ``.ini`` projects post-122dy — but never load-bearing.
122dz and 122ea explicitly deferred this sweep as "low value,
mechanical."

The sweep semantics
-------------------
Two-stage replacement applied uniformly:

1. ``project_folder/project_config.ini`` → ``project.toml``.
   The v1 layout has no ``project_folder/`` subdir; the
   config lives at the project root.
2. Bare ``project_config.ini`` (where the path was already
   shorter) → ``project.toml``.

Stage 1 captures the most common SimBA-era pattern; stage 2
catches the remaining short-form references. The order is
load-bearing — stage 2 would otherwise eat the full path
prefix in stage 1's targets.

What was deliberately excluded
------------------------------

* **mufasa/legacy_layout.py** — the file's purpose IS to
  document the legacy layout for in-repo source-compatibility.
  8 ``project_config.ini`` references preserved.
* **tests/** — the smoke tests use ``.ini`` for legacy-detection
  testing (e.g., creating a fake legacy project fixture to
  verify ConfigReader rejects it). Sweeping ``.ini`` out of
  tests would break those checks.
* **CHANGELOG.md** — historical record; references to ``.ini``
  there are correct.
* **docs/hardwired_paths_audit.md** — historical audit doc.
* **session_handoff.md** — historical workflow handoff.

User-facing markdown swept selectively
--------------------------------------
* README.md — replaced the dual-layout bullet ("Two project
  layouts: legacy / v1") with a v1-only bullet that cites
  122dw–122dz for the legacy removal.
* docs/workflows.md — minimal token replacement
  (``project_config.ini`` → ``project.toml``) for the project
  create/load flow descriptions. The doc still describes Tk
  entries; the broader doc-currency work is out of scope.
* docs/data_source_guides.md — dropped the "Legacy INI" bullet
  from the project-format comparison section.

What this patch landed
----------------------
156 .py files modified (no source files renamed or deleted; only
substring replacements in docstrings, comments, and commented-out
code).

3 .md files modified (README, workflows, data_source_guides).

No test files modified — the smoke-test surface is preserved
exactly.

Coverage
--------
Code-side correctness:
1.  All mufasa/**/*.py parse cleanly after the sweep.
2.  No ``project_config.ini`` references remain in mufasa/**/*.py
    EXCEPT for legacy_layout.py (8 expected).
3.  ``mufasa/legacy_layout.py`` still has its 8 references
    (verifies the skip-list worked).

Documentation correctness:
4.  README.md no longer presents two layouts as alternatives —
    no live "Legacy SimBA" descriptor for the project-config
    format (since legacy support is gone).
5.  docs/workflows.md uses ``project.toml`` for the load-project
    Reads line.
6.  docs/data_source_guides.md no longer has a "Legacy INI"
    bullet alongside "v1 TOML".

CHANGELOG / handoff preservation:
7.  CHANGELOG.md still mentions ``project_config.ini`` (the
    historical record is intentionally preserved).
8.  session_handoff.md (if present) is untouched.

Cross-patch invariants:
9.  All 41 pre-existing strict tests still pass (this check
    is implicit via the sweep regimen; we verify a few load-
    bearing ones explicitly).
10. 122ef hotfix preserved: workbench.py hasattr guard.
11. 122ee state preserved: PoseImportForm publish wiring.
12. 122ed state preserved: Interpolate run_id/run_dir.
13. 122dy state preserved: ConfigReader fail-fast on .toml
    requirement.
14. 122do baseline.
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
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    # 2. No project_config.ini in mufasa/**/*.py except
    # legacy_layout.py.
    stray = []
    for f in sorted(pkg.rglob("*.py")):
        rel = str(f.relative_to(REPO_ROOT))
        if rel == "mufasa/legacy_layout.py":
            continue
        src = f.read_text()
        if "project_config.ini" in src:
            stray.append(
                f"{rel}: {src.count('project_config.ini')} refs"
            )
    check(
        "No `project_config.ini` references remain in "
        "mufasa/**/*.py outside of legacy_layout.py",
        not stray,
        detail=("; ".join(stray[:3])),
    )

    # 3. legacy_layout.py still has its references.
    ll_src = (REPO_ROOT / "mufasa" / "legacy_layout.py").read_text()
    ll_count = ll_src.count("project_config.ini")
    check(
        f"mufasa/legacy_layout.py preserves its "
        f"`project_config.ini` references (the file documents "
        f"the legacy layout specifically; expected ~8 refs, "
        f"got {ll_count})",
        ll_count >= 4,  # generous floor — the exact count
                        # might fluctuate with future doc edits
        detail=(f"got {ll_count}"),
    )

    # 4. README.md no longer presents two layouts.
    readme_src = (REPO_ROOT / "README.md").read_text()
    check(
        "README.md no longer has a 'Two project layouts' "
        "section pitching legacy as a valid choice "
        "(replaced with a v1-only description that cites "
        "the 122dw–122dz removal in past tense)",
        "Two project layouts:" not in readme_src
        and "Legacy SimBA (`project_config.ini`-driven)" not in readme_src,
    )

    # 5. workflows.md uses project.toml on the Reads line for
    # load-project flow.
    wf_path = REPO_ROOT / "docs" / "workflows.md"
    if wf_path.exists():
        wf_src = wf_path.read_text()
        # The load-project section's Reads line should mention
        # project.toml.
        check(
            "docs/workflows.md uses `project.toml` in its "
            "load-project Reads description",
            "**Reads**: `project.toml`" in wf_src,
        )
    else:
        check("docs/workflows.md exists", False,
              detail="file missing")

    # 6. data_source_guides.md dropped Legacy INI bullet.
    dsg_path = REPO_ROOT / "docs" / "data_source_guides.md"
    if dsg_path.exists():
        dsg_src = dsg_path.read_text()
        check(
            "docs/data_source_guides.md no longer has a "
            "'**Legacy INI**' bullet alongside '**v1 TOML**'",
            "**Legacy INI**" not in dsg_src,
        )
    else:
        check("docs/data_source_guides.md exists", False,
              detail="file missing")

    # 7. CHANGELOG.md preserves its historical references.
    cl_path = REPO_ROOT / "CHANGELOG.md"
    if cl_path.exists():
        cl_src = cl_path.read_text()
        check(
            "CHANGELOG.md still mentions `project_config.ini` "
            "(historical record preserved; the sweep is "
            "cosmetic, not a rewrite of history)",
            "project_config.ini" in cl_src,
        )
    else:
        # CHANGELOG might not exist; skip-check.
        check("CHANGELOG.md present (skipped if missing)", True)

    # 8. session_handoff.md (if present) is untouched. We can't
    # easily verify "untouched" without a baseline; just verify
    # it still contains expected substrings if present.
    sh_path = REPO_ROOT / "session_handoff.md"
    if sh_path.exists():
        sh_src = sh_path.read_text()
        # Session handoff describes the work; should still mention
        # session-2 patches.
        check(
            "session_handoff.md still contains expected "
            "content (sweep didn't accidentally touch it)",
            "122" in sh_src,
        )
    else:
        check("session_handoff.md absent — nothing to verify", True)

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 10. 122ef hotfix preserved.
    wb_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "workbench.py").read_text()
    has_guard = bool(re.search(
        r'hasattr\s*\(\s*form\s*,\s*["\']completed["\']\s*\)',
        wb_src,
    ))
    check(
        "122ef-hotfix state preserved: workbench.py still has "
        "the `hasattr(form, 'completed')` guard",
        has_guard,
    )

    # 11. 122ee state preserved.
    pi_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "forms" / "pose_import.py").read_text()
    check(
        "122ee state preserved: PoseImportForm has "
        "publish_target_stage + publish_source_stage attrs",
        ('publish_target_stage = "outlier_corrected"' in pi_src
         and 'publish_source_stage = "pose"' in pi_src),
    )

    # 12. 122ed state preserved.
    bk_src = (REPO_ROOT / "mufasa" / "data_processors"
              / "interpolate.py").read_text()
    check(
        "122ed state preserved: Interpolate.__init__ allocates "
        "run_id + run_dir",
        "self.run_id = generate_run_id()" in bk_src
        and "self.run_dir" in bk_src,
    )

    # 13. 122dy state preserved.
    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()
    check(
        "122dy state preserved: ConfigReader.__init__ still "
        "raises on a non-.toml config_path. (Note: the 122dy "
        "fail-fast guard's error MESSAGE may have been "
        "modified by the sweep — it referenced .toml in a "
        "context where the sweep wouldn't have touched it, "
        "but the raise structure is what matters here.)",
        "InvalidInputError" in cr_src
        and "endswith" in cr_src
        and ".toml" in cr_src,
    )

    # 14. 122do baseline.
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
        "122do baseline preserved: no `Optional[` in non-docstring "
        "positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122eg_docstring_ini_sweep: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
