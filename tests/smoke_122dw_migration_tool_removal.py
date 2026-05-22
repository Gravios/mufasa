"""
tests/smoke_122dw_migration_tool_removal.py
=============================================

Patch 122dw: delete the legacy → v1 migration tool and every
in-repo reference that still treated it as live infrastructure.

What this patch landed
----------------------
* ``mufasa/cli/migrate_project.py`` — deleted (was 379 LoC).
* ``mufasa-migrate-project`` entry point — removed from
  ``pyproject.toml`` (reverses patch 122dp).
* ``docs/migration_guide.md`` — deleted (was the user-facing
  workflow doc for the now-gone tool).
* ``tests/smoke_migrate_project.py`` — deleted (the pre-strict
  smoke test for the tool's correctness).
* ``tests/smoke_122dp_migrate_console_entry.py`` — deleted (the
  strict-format smoke test that asserted the entry point existed;
  meaningless now that the entry point is gone).

In-source docstring cleanups (no behavior change):
* ``mufasa/project_layout.py`` — module header no longer points
  callers at the migration tool; the ProjectPaths error message
  for non-v1 layouts now says "create a fresh v1 project and copy
  source data" instead of "run the migration tool".
* ``mufasa/section_provenance.py`` — both retro-fill mentions of
  ``migrate_project`` excised; the docstring now explains that
  pre-122ds projects start with no [provenance] table and accrue
  one as the user runs operations.
* ``mufasa/utils/config_creator.py`` — header docstring updated.
* ``tests/smoke_122df_readme_rebrand_and_v1_docs.py`` — README and
  docs/README.md checks inverted to deletion tripwires.

Documentation sweep:
* ``README.md`` — migration entry removed from console-scripts
  table; the migration-workflow code block deleted; "Legacy
  SimBA layout is preserved for backward compatibility" paragraph
  removed (it pointed at the migration tool); v1 framed as the
  only supported layout.
* ``docs/README.md`` — migration_guide entry removed from the
  index.
* ``docs/v1_project_layout.md`` — "Migrating an existing legacy
  project" section replaced with a deprecation notice; the
  References section's migration-tool bullet removed; the DEFINITION
  table row for cli/migrate_project.py annotated as deleted.
* ``docs/hardwired_paths_audit.md`` — DEFINITION row updated to
  reflect the deletion.
* ``docs/testing_workflow.md`` — Workaround 2 no longer suggests
  the ``cli/migrate_project.py --v1-root`` flag; uses "manual
  copy" wording.

Why a clean delete (rather than move-to-tools-as-rescue)
---------------------------------------------------------
User directive locked in earlier this session: "1. Delete." The
rescue option (move to ``tools/legacy_migrate.py``, drop entry
point, document as "unsupported but available") was offered and
explicitly declined. v1 is the only supported layout going
forward; users with legacy projects copy source data into a fresh
v1 tree manually.

Coverage
--------
Source deletions:
1.  mufasa/cli/migrate_project.py no longer exists on disk.
2.  mufasa.cli.migrate_project module is no longer importable.

Entry-point deletion:
3.  pyproject.toml [project.scripts] no longer declares
    mufasa-migrate-project.
4.  pyproject.toml's deprecation breadcrumb is present.

Doc deletions:
5.  docs/migration_guide.md no longer exists on disk.

Test deletions:
6.  tests/smoke_migrate_project.py no longer exists.
7.  tests/smoke_122dp_migrate_console_entry.py no longer exists.

README + index sweep:
8.  README.md no longer links to docs/migration_guide.md.
9.  README.md no longer lists mufasa-migrate-project as a live
    console-script entry.
10. README.md mentions the 122dw removal (deprecation breadcrumb).
11. docs/README.md no longer indexes migration_guide.md.

In-source docstring sweep:
12. mufasa/project_layout.py module docstring no longer says the
    migration tool exists.
13. mufasa/project_layout.py ProjectPaths error message no longer
    tells users to run the migration tool.
14. mufasa/section_provenance.py no longer mentions migrate_project
    retro-fill in either of its two prior references.
15. mufasa/utils/config_creator.py docstring no longer mentions
    the migration tool as a live system.

v1_project_layout.md sweep:
16. "Migrating an existing legacy project" section in
    v1_project_layout.md is now a deprecation notice (mentions
    122dw + "no longer supported").
17. v1_project_layout.md References list no longer points at
    cli/migrate_project.py.

testing_workflow.md sweep:
18. testing_workflow.md Workaround 2 no longer mentions
    migrate_project.py.

122dp baseline reversal:
19. The 122dp commit's entry-point addition has been reversed
    (cross-check: only one place in the codebase mentions the
    short form mufasa-migrate-project, and it's in the deprecation
    breadcrumbs).

Cross-patch invariants:
20. The 122dv smoke test still finds no SkipOutlierCorrectionForm
    (122dv state preserved through 122dw).
21. The 122ds SECTIONS DAG still validates at module import
    (provenance infrastructure unaffected).
22. Parse-clean across mufasa/**/*.py.
23. 122do baseline: no ``Optional[`` in non-docstring positions
    across mufasa/ui_qt/.
"""
from __future__ import annotations

import ast
import importlib
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
    # 1. Source file deleted.
    check(
        "mufasa/cli/migrate_project.py no longer exists on disk",
        not (REPO_ROOT / "mufasa" / "cli" / "migrate_project.py").exists(),
    )

    # 2. Module no longer importable.
    importable = True
    try:
        importlib.import_module("mufasa.cli.migrate_project")
    except (ImportError, ModuleNotFoundError):
        importable = False
    check(
        "mufasa.cli.migrate_project no longer importable "
        "(ModuleNotFoundError on the import attempt)",
        not importable,
    )

    # 3. pyproject.toml entry-point removed.
    pp_src = (REPO_ROOT / "pyproject.toml").read_text()
    scripts_match = re.search(
        r"\[project\.scripts\](.*?)(?=\n\[|\Z)",
        pp_src, flags=re.DOTALL,
    )
    scripts_block = scripts_match.group(1) if scripts_match else ""
    has_entry = bool(re.search(
        r"^\s*mufasa-migrate-project\s*=\s*[\"']",
        scripts_block, flags=re.MULTILINE,
    ))
    check(
        "pyproject.toml [project.scripts] no longer declares "
        "mufasa-migrate-project",
        not has_entry,
    )

    # 4. pyproject.toml deprecation breadcrumb.
    check(
        "pyproject.toml mentions the 122dw removal "
        "(deprecation breadcrumb)",
        "patch 122dw" in pp_src.lower()
        or "122dw" in pp_src,
    )

    # 5. Doc deletion.
    check(
        "docs/migration_guide.md no longer exists on disk",
        not (REPO_ROOT / "docs" / "migration_guide.md").exists(),
    )

    # 6-7. Test deletions.
    check(
        "tests/smoke_migrate_project.py no longer exists",
        not (REPO_ROOT / "tests" / "smoke_migrate_project.py").exists(),
    )
    check(
        "tests/smoke_122dp_migrate_console_entry.py no longer exists "
        "(the 122dp strict test that pinned the entry-point's "
        "existence is now meaningless)",
        not (REPO_ROOT / "tests"
             / "smoke_122dp_migrate_console_entry.py").exists(),
    )

    # 8-10. README.md sweep.
    readme = (REPO_ROOT / "README.md").read_text()
    check(
        "README.md no longer links to docs/migration_guide.md",
        "migration_guide.md" not in readme,
    )
    # We want the OLD console-scripts table row gone. Distinguish
    # from the deprecation breadcrumb sentence that mentions the
    # name historically. Check that no live row table line
    # introduces the entry as if it still exists.
    has_table_row = bool(re.search(
        r"^\|\s*`mufasa-migrate-project`\s*\|",
        readme, flags=re.MULTILINE,
    ))
    check(
        "README.md no longer lists mufasa-migrate-project in the "
        "console-scripts table (the row introducing it as a live "
        "entry is gone)",
        not has_table_row,
    )
    check(
        "README.md mentions the 122dw removal (deprecation "
        "breadcrumb so the change is discoverable for users with "
        "stale install instructions)",
        "122dw" in readme,
    )

    # 11. docs/README.md no longer indexes the deleted guide.
    docs_index = (REPO_ROOT / "docs" / "README.md").read_text()
    check(
        "docs/README.md no longer indexes migration_guide.md",
        "migration_guide.md" not in docs_index,
    )

    # 12. mufasa/project_layout.py module docstring no longer says
    # the migration tool exists.
    pl_src = (REPO_ROOT / "mufasa" / "project_layout.py").read_text()
    pl_header = pl_src.split('"""', 2)[1] if '"""' in pl_src else ""
    # The header may legitimately reference migrate_project in
    # past tense (e.g., "Patch 122dw deleted ..."). What we want
    # is to verify the header DOES mention the deletion. If
    # there's a mention without "deleted" / "removed" / "no
    # longer", that's a regression.
    mentions_tool = "migrate_project" in pl_header
    deletes_explicitly = any(
        word in pl_header.lower()
        for word in ("deleted", "removed", "no longer")
    )
    check(
        "mufasa/project_layout.py header docstring no longer "
        "presents migrate_project as a live tool (mentions are "
        "in past-tense / deletion context only)",
        (not mentions_tool) or deletes_explicitly,
    )

    # 13. ProjectPaths error message no longer says "run the
    # migration tool" — it now says "create a fresh v1 project".
    check(
        "mufasa/project_layout.py ProjectPaths error message no "
        "longer tells users to run `python -m mufasa.cli.migrate_project`",
        "Run `python -m mufasa.cli.migrate_project" not in pl_src,
    )

    # 14. section_provenance.py — no retro-fill mentions.
    sp_src = (REPO_ROOT / "mufasa" / "section_provenance.py").read_text()
    # Same past-tense gate as for project_layout.py.
    sp_mentions = sp_src.count("migrate_project")
    # Each remaining mention should be inside a past-tense /
    # deletion sentence. We approximate that by checking the
    # surrounding 80 chars for the deletion vocabulary.
    bad_mentions = []
    for m in re.finditer(r"migrate_project", sp_src):
        window = sp_src[max(0, m.start() - 80):
                        m.end() + 80].lower()
        if not any(w in window for w in
                   ("deleted", "removed", "no longer")):
            bad_mentions.append(m.start())
    check(
        "section_provenance.py no longer treats migrate_project as "
        "live infrastructure (every remaining mention is in a "
        "deletion-context sentence)",
        not bad_mentions,
        detail=(f"unflagged mentions at offsets "
                f"{bad_mentions[:3]}"),
    )

    # 15. config_creator.py — same gate.
    cc_src = (REPO_ROOT / "mufasa" / "utils"
              / "config_creator.py").read_text()
    bad_mentions = []
    for m in re.finditer(r"migrate_project", cc_src):
        window = cc_src[max(0, m.start() - 80):
                        m.end() + 80].lower()
        if not any(w in window for w in
                   ("deleted", "removed", "no longer")):
            bad_mentions.append(m.start())
    check(
        "mufasa/utils/config_creator.py no longer treats "
        "migrate_project as live infrastructure",
        not bad_mentions,
    )

    # 16. v1_project_layout.md migration section is now a deprecation
    # notice.
    v1_src = (REPO_ROOT / "docs" / "v1_project_layout.md").read_text()
    # Look for the "Migrating an existing legacy project" section.
    mig_section_match = re.search(
        r"## Migrating an existing legacy project\s*\n(.*?)(?=\n##|\Z)",
        v1_src, flags=re.DOTALL,
    )
    mig_section = mig_section_match.group(1) if mig_section_match else ""
    check(
        "v1_project_layout.md 'Migrating an existing legacy project' "
        "section is now a deprecation notice (mentions 122dw + "
        "'no longer supported')",
        "122dw" in mig_section and "no longer supported" in mig_section.lower(),
    )

    # 17. v1_project_layout.md References list no longer points at
    # cli/migrate_project.py.
    refs_match = re.search(
        r"## References\s*\n(.*?)(?=\n##|\Z)",
        v1_src, flags=re.DOTALL,
    )
    refs_section = refs_match.group(1) if refs_match else ""
    check(
        "v1_project_layout.md References list no longer links to "
        "cli/migrate_project.py",
        "cli/migrate_project.py" not in refs_section,
    )

    # 18. testing_workflow.md Workaround 2 sweep.
    tw_src = (REPO_ROOT / "docs" / "testing_workflow.md").read_text()
    check(
        "testing_workflow.md no longer suggests "
        "cli/migrate_project.py --v1-root as a workaround",
        "cli/migrate_project.py" not in tw_src,
    )

    # 19. Only deprecation-context mentions of the short form
    # mufasa-migrate-project remain (no live usage instructions).
    # Exclusions: session_handoff.md is a historical record of a
    # past session's state and shouldn't be back-edited; this test
    # file itself naturally mentions the name.
    all_short_form_mentions = []
    for f in sorted((REPO_ROOT).rglob("*")):
        if not (f.is_file() and f.suffix in (".py", ".md", ".toml")):
            continue
        rel = f.relative_to(REPO_ROOT)
        if rel.name == "session_handoff.md":
            continue
        if rel.name.startswith("smoke_122dw_"):
            continue
        try:
            src = f.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue
        for m in re.finditer(r"mufasa-migrate-project", src):
            window = src[max(0, m.start() - 100):
                         m.end() + 100].lower()
            if not any(w in window for w in
                       ("removed", "deleted", "122dw")):
                all_short_form_mentions.append(
                    f"{rel}:"
                    f"{src[:m.start()].count(chr(10)) + 1}"
                )
    check(
        "Every remaining mention of 'mufasa-migrate-project' "
        "across the repo is in a deletion-context sentence "
        "(no live usage instructions linger; "
        "session_handoff.md and this test file excluded)",
        not all_short_form_mentions,
        detail=("; ".join(all_short_form_mentions[:3])),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants.
    # -----------------------------------------------------------------

    # 20. 122dv state preserved — no SkipOutlierCorrectionForm.
    pc_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "forms" / "pose_cleanup.py").read_text()
    check(
        "122dv state preserved: no SkipOutlierCorrectionForm "
        "class in pose_cleanup.py (only the deprecation comment)",
        "class SkipOutlierCorrectionForm" not in pc_src,
    )

    # 21. 122ds SECTIONS still validates at import.
    try:
        from mufasa.section_provenance import SECTIONS
        valid = len(SECTIONS) > 0
    except Exception:
        valid = False
    check(
        "122ds state preserved: section_provenance.SECTIONS still "
        "imports cleanly and the DAG validates",
        valid,
    )

    # 22. Parse-clean.
    pkg = REPO_ROOT / "mufasa"
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

    # 23. 122do baseline tripwire.
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
        f"smoke_122dw_migration_tool_removal: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
