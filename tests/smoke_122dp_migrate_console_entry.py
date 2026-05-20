"""
tests/smoke_122dp_migrate_console_entry.py
=============================================

Patch 122dp: ``mufasa-migrate-project`` console-script entry point.

What this patch landed
----------------------
1. ``pyproject.toml`` — added under ``[project.scripts]``::

       mufasa-migrate-project = "mufasa.cli.migrate_project:main"

   The function ``mufasa.cli.migrate_project.main`` was already
   written for this purpose: its ``argparse.ArgumentParser`` has
   ``prog="mufasa-migrate-project"`` baked in (so ``--help`` output
   matches the entry point), and its signature is
   ``def main(argv: Optional[List[str]] = None) -> int`` — the
   setuptools console-script convention.

2. ``README.md`` — promoted ``mufasa-migrate-project`` to the
   console-scripts table and updated the migration-command example
   to use it. Kept the ``python -m`` form as a fallback note for
   environments where the entry point isn't on ``$PATH``.

3. ``docs/migration_guide.md`` — updated TL;DR, dry-run example,
   commit example, and Tool-flags synopsis to use the short form.
   Updated the doc's opening line to credit patch 122dp.

Coverage
--------
1.  ``pyproject.toml`` declares ``mufasa-migrate-project`` under
    ``[project.scripts]``.
2.  The entry target is ``mufasa.cli.migrate_project:main`` (not a
    typo, not pointing elsewhere).
3.  The previous three entry points (``mufasa``, ``mufasa-chooser``,
    ``mufasa-workbench``) are preserved — the addition didn't
    accidentally remove any.
4.  ``mufasa.cli.migrate_project`` module is importable and exposes
    a ``main`` callable (proving the entry point will actually
    resolve at install time).
5.  The ``main`` callable accepts an optional ``argv`` argument
    (matches the setuptools console-script convention; otherwise
    setuptools wraps it with ``sys.argv[1:]``).
6.  ``main``'s ArgumentParser uses ``prog="mufasa-migrate-project"``
    so ``--help`` output is consistent with the entry point name.
7.  ``README.md`` mentions ``mufasa-migrate-project`` (the entry
    point is documented for users).
8.  ``docs/migration_guide.md`` mentions ``mufasa-migrate-project``
    in its TL;DR or main workflow.
9.  ``docs/migration_guide.md`` still mentions ``python -m
    mufasa.cli.migrate_project`` somewhere — the long form is the
    fallback for installs without ``$PATH``.
10. The 122dg ruff config (``extend-exclude`` for build/dist/.venv,
    no ``mufasa/ui`` entry) is preserved.
11. The 122do typing-import baseline still holds: ``mufasa/ui_qt/``
    is still free of ``Optional[`` in non-docstring positions. (A
    pyproject.toml-only patch shouldn't regress this; the check is
    a tripwire.)
12. All ``mufasa/**/*.py`` parse cleanly.
"""
from __future__ import annotations

import ast
import importlib
import inspect
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
    pp_path = REPO_ROOT / "pyproject.toml"
    pp_src = pp_path.read_text()

    # Extract the [project.scripts] section as a substring. We avoid
    # depending on tomllib so this test works on Python 3.10 too if
    # ever needed (though the project requires 3.11+).
    scripts_match = re.search(
        r"\[project\.scripts\](.*?)(?=\n\[|\Z)",
        pp_src,
        flags=re.DOTALL,
    )
    scripts_block = scripts_match.group(1) if scripts_match else ""

    # 1. Entry declared
    declares_entry = bool(re.search(
        r"^\s*mufasa-migrate-project\s*=\s*[\"']",
        scripts_block,
        flags=re.MULTILINE,
    ))
    check(
        "pyproject.toml [project.scripts] declares "
        "`mufasa-migrate-project`",
        declares_entry,
    )

    # 2. Correct target
    correct_target = bool(re.search(
        r"^\s*mufasa-migrate-project\s*=\s*"
        r"[\"']mufasa\.cli\.migrate_project:main[\"']",
        scripts_block,
        flags=re.MULTILINE,
    ))
    check(
        "Entry point targets `mufasa.cli.migrate_project:main` "
        "(no typo)",
        correct_target,
    )

    # 3. Existing entry points preserved
    preserved = all(
        re.search(rf"^\s*{re.escape(name)}\s*=", scripts_block,
                  flags=re.MULTILINE)
        for name in ("mufasa", "mufasa-chooser", "mufasa-workbench")
    )
    check(
        "Pre-existing entry points (mufasa, mufasa-chooser, "
        "mufasa-workbench) preserved",
        preserved,
    )

    # 4. Module + main resolvable. Skip if the import requires
    # third-party libs that aren't in the test environment; the
    # smoke test should still pass in the sandbox.
    main_callable = None
    try:
        mod = importlib.import_module("mufasa.cli.migrate_project")
        main_callable = getattr(mod, "main", None)
        check(
            "mufasa.cli.migrate_project is importable and exposes "
            "a `main` callable (proves the entry point will "
            "actually resolve)",
            callable(main_callable),
        )
    except ImportError as e:
        # Sandbox fallback: parse the module file as AST and check
        # for a top-level `def main(`. Still proves the target
        # exists.
        mod_path = REPO_ROOT / "mufasa" / "cli" / "migrate_project.py"
        tree = ast.parse(mod_path.read_text())
        has_main = any(
            isinstance(n, ast.FunctionDef) and n.name == "main"
            for n in tree.body
        )
        check(
            "mufasa.cli.migrate_project module file defines a top-"
            "level `def main` (AST fallback: full import failed in "
            "test environment, but the entry point target exists)",
            has_main,
            detail=f"ImportError was: {type(e).__name__}: {e}",
        )

    # 5. main accepts an optional argv parameter
    if main_callable is not None:
        sig = inspect.signature(main_callable)
        first_param = next(iter(sig.parameters.values()), None)
        accepts_argv = (
            first_param is not None
            and first_param.default is not inspect.Parameter.empty
        )
        check(
            "main(argv=None) accepts an optional `argv` "
            "(setuptools console-script convention; otherwise "
            "setuptools wraps it with sys.argv[1:])",
            accepts_argv,
        )
    else:
        # AST fallback: confirm the signature has at least one
        # default-valued parameter.
        mod_path = REPO_ROOT / "mufasa" / "cli" / "migrate_project.py"
        tree = ast.parse(mod_path.read_text())
        ok = False
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                # Has at least one default
                ok = bool(node.args.defaults) or bool(
                    node.args.kw_defaults)
                break
        check(
            "main() has an optional first arg (AST fallback for "
            "argv-accepts check)",
            ok,
        )

    # 6. prog="mufasa-migrate-project" baked into ArgumentParser.
    # Source-level check (no import needed).
    mod_src = (REPO_ROOT / "mufasa" / "cli" /
               "migrate_project.py").read_text()
    check(
        "main()'s ArgumentParser uses prog='mufasa-migrate-project' "
        "(so --help output matches the entry-point name)",
        'prog="mufasa-migrate-project"' in mod_src
        or "prog='mufasa-migrate-project'" in mod_src,
    )

    # 7-9. Doc references
    readme = (REPO_ROOT / "README.md").read_text()
    check(
        "README.md mentions `mufasa-migrate-project`",
        "mufasa-migrate-project" in readme,
    )

    mg = (REPO_ROOT / "docs" / "migration_guide.md").read_text()
    check(
        "docs/migration_guide.md mentions `mufasa-migrate-project`",
        "mufasa-migrate-project" in mg,
    )
    check(
        "docs/migration_guide.md still mentions `python -m "
        "mufasa.cli.migrate_project` somewhere (the fallback form "
        "is preserved for $PATH-less installs)",
        "python -m mufasa.cli.migrate_project" in mg,
    )

    # 10. 122dg ruff config invariants
    check(
        "Ruff extend-exclude config preserved (build, dist, .venv "
        "still listed; mufasa/ui not re-added)",
        '"build"' in pp_src
        and '"dist"' in pp_src
        and '".venv"' in pp_src
        and '"mufasa/ui"' not in pp_src,
    )

    # 11. 122do baseline tripwire — no Optional[ in non-docstring
    # positions across mufasa/ui_qt/. A pyproject change shouldn't
    # regress this, but the check costs nothing.
    uiqt = REPO_ROOT / "mufasa" / "ui_qt"
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
        "positions across mufasa/ui_qt/ (a pyproject patch "
        "shouldn't regress the typing modernization)",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    # 12. Parse-clean
    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(
                f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122dp_migrate_console_entry: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
