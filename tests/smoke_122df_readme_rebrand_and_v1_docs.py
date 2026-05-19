"""
tests/smoke_122df_readme_rebrand_and_v1_docs.py
=================================================

Patch 122df: README rebrand + new v1 user docs.

What this patch landed
----------------------
1. README.md rebranded — drops the SimBA-dominated tutorial /
   release-note bulk; keeps the scientific citation, license,
   and acknowledgments. Adds Mufasa-first install instructions,
   project-layout summary, and pointers to the new docs.
2. `docs/v1_project_layout.md` created — user/dev reference for
   the v1 layout: directory structure, run-id semantics, path-
   abstraction layer, triage rules for hardwired-path audits,
   how to create a v1 project programmatically.
3. `docs/migration_guide.md` created — workflow for using
   `python -m mufasa.cli.migrate_project` to move a legacy
   project to v1. Dry-run / commit / verify / clean-up steps;
   troubleshooting; rollback notes.
4. `docs/README.md` index updated — adds a "User-facing entry
   points" section at the top pointing at the two new docs.

Coverage
--------
1.  README.md exists.
2.  README has a Citation section with the SimBA paper bibtex
    (preserves scientific reference).
3.  README has a License section preserving GPL v3.
4.  README has an Acknowledgments section crediting SimBA + key
    contributors.
5.  README has a "Running Mufasa" section.
6.  README has a "Project layout" section.
7.  README points at the new docs (v1_project_layout.md,
    migration_guide.md).
8.  README dropped the SimBA tutorial bulk (no more
    "Tutorial 📚", "What is SimBA?", date-stamped release notes).
9.  README dropped simba-uw-tf-dev pip-install instructions
    (Mufasa is install-from-source).
10. docs/v1_project_layout.md exists and covers run-id format,
    `project_paths_from_config`, directory contents.
11. docs/migration_guide.md exists and covers dry-run / --commit
    / `MIGRATION.toml` audit trail.
12. docs/README.md indexes both new docs.
13. All mufasa/**/*.py still parse cleanly (no code changes
    in this patch, but check anyway).
"""
from __future__ import annotations

import ast
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
    readme = (REPO_ROOT / "README.md").read_text()
    v1_doc = REPO_ROOT / "docs" / "v1_project_layout.md"
    mig_doc = REPO_ROOT / "docs" / "migration_guide.md"
    docs_index = REPO_ROOT / "docs" / "README.md"

    # 1. README exists
    check("README.md exists", (REPO_ROOT / "README.md").exists())

    # 2. Citation preserved
    check(
        "README preserves SimBA paper bibtex citation",
        "@article{Goodwin2024" in readme
        and "10.1038/s41593-024-01649-9" in readme,
    )

    # 3. License preserved
    check(
        "README has License section with GPL v3",
        "## License" in readme and "GPL v3" in readme,
    )

    # 4. Acknowledgments
    check(
        "README has Acknowledgments crediting SimBA + "
        "contributors",
        "## Acknowledgments" in readme
        and "Simon Nilsson" in readme
        and "Golden Lab" in readme,
    )

    # 5-6. Sections
    check(
        "README has 'Running Mufasa' section",
        "## Running Mufasa" in readme,
    )
    check(
        "README has 'Project layout' section",
        "## Project layout" in readme,
    )

    # 7. Links to new docs
    check(
        "README links to docs/v1_project_layout.md",
        "v1_project_layout.md" in readme,
    )
    check(
        "README links to docs/migration_guide.md",
        "migration_guide.md" in readme,
    )

    # 8. SimBA tutorial bulk dropped
    check(
        "README no longer has 'Tutorial 📚' section "
        "(SimBA-specific tutorials)",
        "## Tutorial 📚" not in readme,
    )
    check(
        "README no longer has 'What is SimBA?' section",
        "## What is SimBA?" not in readme,
    )
    check(
        "README no longer has SimBA date-stamped release notes "
        "(e.g., 'Apr-03-2025')",
        "Apr-03-2025" not in readme
        and "Apr-04-2023" not in readme,
    )

    # 9. Pip install legacy gone
    check(
        "README no longer instructs `pip install "
        "simba-uw-tf-dev` (Mufasa is install-from-source)",
        "pip install simba-uw-tf-dev" not in readme,
    )

    # 10. v1 project layout doc
    check(
        "docs/v1_project_layout.md exists",
        v1_doc.exists(),
    )
    if v1_doc.exists():
        v1_src = v1_doc.read_text()
        check(
            "v1 doc covers run-id format (YYYYMMDD-HHMMSS-XXXXXX)",
            "YYYYMMDD-HHMMSS-XXXXXX" in v1_src,
        )
        check(
            "v1 doc covers project_paths_from_config helper",
            "project_paths_from_config" in v1_src,
        )
        check(
            "v1 doc covers directory contents (sources/derived/"
            "models/logs)",
            "sources/" in v1_src and "derived/" in v1_src
            and "models/" in v1_src and "logs/" in v1_src,
        )
        check(
            "v1 doc has triage rules table for backend devs",
            "Triage rules" in v1_src or "triage rule" in v1_src,
        )

    # 11. Migration guide
    check(
        "docs/migration_guide.md exists",
        mig_doc.exists(),
    )
    if mig_doc.exists():
        mig_src = mig_doc.read_text()
        check(
            "Migration guide covers dry-run workflow",
            "dry run" in mig_src.lower() or "dry-run" in mig_src.lower(),
        )
        check(
            "Migration guide covers --commit flag",
            "--commit" in mig_src,
        )
        check(
            "Migration guide mentions MIGRATION.toml audit trail",
            "MIGRATION.toml" in mig_src,
        )
        check(
            "Migration guide has Troubleshooting section",
            "## Troubleshooting" in mig_src,
        )
        check(
            "Migration guide has Rollback section",
            "## Rollback" in mig_src,
        )

    # 12. docs/README index
    check(
        "docs/README.md indexes v1_project_layout.md",
        "v1_project_layout.md" in docs_index.read_text(),
    )
    check(
        "docs/README.md indexes migration_guide.md",
        "migration_guide.md" in docs_index.read_text(),
    )

    # 13. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122df_readme_rebrand_and_v1_docs: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
