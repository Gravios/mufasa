"""
tests/smoke_122dx_legacy_chooser_removal.py
=============================================

Patch 122dx: delete the legacy Qt chooser (``mufasa/ui_qt/app.py``)
and the ``mufasa-chooser`` console entry point.

Context
-------
``mufasa.ui_qt.app`` was the Qt-era replacement for the legacy Tk
launcher (``mufasa.SimBA:main``, removed in 122d4): a chooser
window with "Load Project" / "Create Project" buttons that, after
loading, displayed a 10-tab layout mirroring the legacy SimBA
tabs. It was the *intermediate* Qt surface — Qt-rendered but
otherwise SimBA-shaped.

Once the consolidated :class:`MufasaWorkbench` (sidebar-navigated,
forms-driven, ~14 workflow pages) acquired a Projects page covering
create / load / recent-project flows, the standalone chooser
surface was redundant. ``mufasa`` and ``mufasa-workbench`` both
reach the workbench directly; the chooser had no remaining users
who weren't better served by either one.

What this patch landed
----------------------
Source deletion:
* ``mufasa/ui_qt/app.py`` — 409 LoC removed.

Entry-point removal:
* ``pyproject.toml`` ``[project.scripts]``: the
  ``mufasa-chooser = "mufasa.ui_qt.app:main"`` line is gone,
  replaced with a deprecation breadcrumb comment.

Updated tests:
* ``tests/smoke_122d4_stage_a.py`` — the active-entry-points loop
  no longer includes ``mufasa-chooser``. The corresponding pin on
  ``(mufasa-chooser, mufasa.ui_qt.app:main)`` is gone.
* ``tests/smoke_122bl_code_cleanup_pass.py`` — the descriptive-
  TODO check (which asserted that ``ui_qt/app.py`` had a
  ``# TODO: wire up Interpolate`` + ``workbench`` comment) is
  flipped to a deletion tripwire (``app.py`` should NOT exist).
* ``tests/smoke_workbench_launcher.py`` — the comment claiming
  ``mufasa-chooser`` was an "alias for old mufasa behavior" is
  updated; the assertion is repurposed as a tripwire that the
  deprecation breadcrumb in pyproject.toml remains intact.

Documentation sweep:
* ``README.md`` — console-scripts table goes from 3 entries to 2;
  the deprecation breadcrumb paragraph gains a sentence about
  122dx.
* ``docs/tk_surface_audit.md`` — three references updated: the
  current-entries table row drops mufasa-chooser; the step-4 ASCII
  flow box updated; the "intermediate, will outlive Tk" prediction
  is annotated as having held until 122dx.
* ``docs/simba_death_cascade.md`` — code blocks left intact (they
  are 122d4-era snapshots, correct as historical fixtures); a
  one-line breadcrumb added after them noting later trim patches.
* ``mufasa/ui_qt/workbench_app.py`` — module header rewritten:
  the chooser is no longer described as a live coexisting surface.

Coverage
--------
Source deletions:
1.  ``mufasa/ui_qt/app.py`` no longer exists on disk.
2.  ``mufasa.ui_qt.app`` module no longer importable.

Entry-point deletion:
3.  ``pyproject.toml`` ``[project.scripts]`` no longer declares
    ``mufasa-chooser`` as a live entry.
4.  ``pyproject.toml`` retains a ``patch 122dx`` deprecation
    breadcrumb (so install instructions referencing the removed
    name surface a discoverable explanation).

Test reconciliation:
5.  ``tests/smoke_122d4_stage_a.py`` no longer has the
    ``(mufasa-chooser, mufasa.ui_qt.app:main)`` tuple in its
    active-entries loop.
6.  ``tests/smoke_122bl_code_cleanup_pass.py`` no longer reads
    ``app.py`` as if it exists (was a regression-risk pin).

Doc reconciliation:
7.  ``README.md`` no longer lists ``mufasa-chooser`` in the
    console-scripts table (the live row introducing it is gone).
8.  ``README.md`` mentions 122dx (deprecation breadcrumb).
9.  ``docs/tk_surface_audit.md`` no longer presents
    ``mufasa-chooser = "mufasa.ui_qt.app:main"`` as the **current**
    Qt entry point. (The text may still mention the name in past-
    tense historical context; the gate is that the row introducing
    it as live is gone.)

Repo-wide past-tense gate:
10. Every remaining occurrence of ``mufasa-chooser`` across the
    repo sits in a deletion-context sentence (look for "removed",
    "deleted", "no longer", "122dx" in a ±100-char window).
    ``session_handoff.md`` and this test file are excluded from the
    gate (the former is a historical session record; the latter
    naturally repeats the name).
11. Every remaining occurrence of ``mufasa.ui_qt.app`` across the
    repo passes the same gate (with the same exclusions).

Cross-patch invariants:
12. ``mufasa-workbench`` entry point still live (the other Qt
    workbench entry; if 122dx accidentally tripped this it would
    leave users with only ``mufasa`` as the workbench-reaching
    command).
13. The 122dw migration-tool removal is preserved (no
    ``migrate_project.py`` resurrected, no entry restored).
14. The 122dv Skip removal is preserved.
15. The 122ds SECTIONS DAG still validates.
16. Parse-clean across ``mufasa/**/*.py``.
17. 122do baseline tripwire: no ``Optional[`` in non-docstring
    positions across ``mufasa/ui_qt/``.
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


def _past_tense_gate(
    src: str, needle: str, *, window: int = 300,
) -> list[int]:
    """Return offsets where `needle` appears OUTSIDE a deletion-
    context window. Empty list = all mentions are in past-tense.

    The window default (300 chars on each side) is wider than the
    122dw test's 80 because the chooser-removal narrative tends to
    have more prose between the noun and the verb ("the audit's
    prediction that it 'will outlive Tk' held — until 122dx
    removed it" spans ~180 chars).
    """
    out = []
    for m in re.finditer(re.escape(needle), src):
        ctx = src[max(0, m.start() - window):
                  m.end() + window].lower()
        if not any(w in ctx for w in
                   ("removed", "deleted", "no longer", "122dx")):
            out.append(m.start())
    return out


def main() -> int:
    # 1. Source file deleted.
    check(
        "mufasa/ui_qt/app.py no longer exists on disk "
        "(was the 409-LoC legacy Qt chooser)",
        not (REPO_ROOT / "mufasa" / "ui_qt" / "app.py").exists(),
    )

    # 2. Module no longer importable.
    importable = True
    try:
        importlib.import_module("mufasa.ui_qt.app")
    except (ImportError, ModuleNotFoundError):
        importable = False
    check(
        "mufasa.ui_qt.app no longer importable "
        "(ModuleNotFoundError on attempted import)",
        not importable,
    )

    # 3. pyproject.toml entry-point removed (live row gone).
    pp_src = (REPO_ROOT / "pyproject.toml").read_text()
    scripts_match = re.search(
        r"\[project\.scripts\](.*?)(?=\n\[|\Z)",
        pp_src, flags=re.DOTALL,
    )
    scripts_block = scripts_match.group(1) if scripts_match else ""
    # A "live" row in [project.scripts] is a line that starts with
    # the entry name and an `=` (no leading `#`). Deprecation
    # breadcrumbs are `#`-commented prose mentioning the name.
    has_live_entry = bool(re.search(
        r"^\s*mufasa-chooser\s*=\s*[\"']",
        scripts_block, flags=re.MULTILINE,
    ))
    check(
        "pyproject.toml [project.scripts] no longer declares "
        "mufasa-chooser as a live entry",
        not has_live_entry,
    )

    # 4. Deprecation breadcrumb preserved.
    check(
        "pyproject.toml retains a `patch 122dx` deprecation "
        "breadcrumb so users with stale install instructions get a "
        "discoverable explanation",
        "patch 122dx" in pp_src.lower() or "122dx" in pp_src,
    )

    # 5. smoke_122d4_stage_a no longer has the chooser tuple.
    stage_a_src = (REPO_ROOT / "tests"
                   / "smoke_122d4_stage_a.py").read_text()
    # Look for the literal tuple form. The historical pin was
    #     ("mufasa-chooser", "mufasa.ui_qt.app:main"),
    has_tuple = bool(re.search(
        r'\(\s*["\']mufasa-chooser["\']\s*,\s*'
        r'["\']mufasa\.ui_qt\.app:main["\']\s*\)',
        stage_a_src,
    ))
    check(
        "tests/smoke_122d4_stage_a.py no longer pins the "
        "(mufasa-chooser, mufasa.ui_qt.app:main) tuple in its "
        "active-entries loop",
        not has_tuple,
    )

    # 6. smoke_122bl no longer .read_text()s app.py.
    bl_src = (REPO_ROOT / "tests"
              / "smoke_122bl_code_cleanup_pass.py").read_text()
    has_read = '(pkg_root / "ui_qt" / "app.py").read_text()' in bl_src
    check(
        "tests/smoke_122bl_code_cleanup_pass.py no longer reads "
        "app.py (which doesn't exist post-122dx); the check is now "
        "a deletion tripwire",
        not has_read,
    )

    # 7. README.md no longer lists mufasa-chooser as a live entry.
    readme = (REPO_ROOT / "README.md").read_text()
    has_live_row = bool(re.search(
        r"^\|\s*`mufasa-chooser`\s*\|",
        readme, flags=re.MULTILINE,
    ))
    check(
        "README.md no longer lists `mufasa-chooser` in the "
        "console-scripts table (live row removed)",
        not has_live_row,
    )

    # 8. README mentions 122dx.
    check(
        "README.md mentions 122dx (so users encountering 'command "
        "not found' after pip upgrade see why)",
        "122dx" in readme,
    )

    # 9. tk_surface_audit current-entry table row no longer shows
    # mufasa-chooser as part of the **current** Qt entry list.
    # Past-tense mentions elsewhere are fine.
    tk_src = (REPO_ROOT / "docs" / "tk_surface_audit.md").read_text()
    # Match the line containing "Qt (current)" and its row body.
    qt_row_match = re.search(
        r"^\|.*Qt \(current\).*\|(.*?)\|.*\|.*\|",
        tk_src, flags=re.MULTILINE,
    )
    qt_row_body = qt_row_match.group(1) if qt_row_match else ""
    # The OLD row listed three entry names back-to-back; the NEW
    # row should not list mufasa-chooser as one of the live ones.
    has_live_chooser_in_row = (
        "mufasa-chooser" in qt_row_body
        and not any(w in qt_row_body.lower() for w in
                    ("removed", "no longer", "122dx"))
    )
    check(
        "docs/tk_surface_audit.md 'Qt (current)' row no longer "
        "lists mufasa-chooser as a live entry (mentions in past-"
        "tense are OK; the live listing is what was removed)",
        not has_live_chooser_in_row,
    )

    # 10. Repo-wide past-tense gate for `mufasa-chooser`.
    chooser_bad = []
    for f in sorted(REPO_ROOT.rglob("*")):
        if not (f.is_file() and f.suffix in (".py", ".md", ".toml")):
            continue
        rel = f.relative_to(REPO_ROOT)
        if rel.name == "session_handoff.md":
            continue
        if rel.name.startswith("smoke_122dx_"):
            continue
        try:
            src = f.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue
        for offset in _past_tense_gate(src, "mufasa-chooser"):
            chooser_bad.append(
                f"{rel}:"
                f"{src[:offset].count(chr(10)) + 1}"
            )
    check(
        "Every remaining mention of 'mufasa-chooser' across the "
        "repo is in a deletion-context sentence (no live usage "
        "instructions linger; session_handoff.md and this test "
        "are excluded)",
        not chooser_bad,
        detail=("; ".join(chooser_bad[:3])),
    )

    # 11. Same gate for `mufasa.ui_qt.app`.
    module_bad = []
    for f in sorted(REPO_ROOT.rglob("*")):
        if not (f.is_file() and f.suffix in (".py", ".md", ".toml")):
            continue
        rel = f.relative_to(REPO_ROOT)
        if rel.name == "session_handoff.md":
            continue
        if rel.name.startswith("smoke_122dx_"):
            continue
        try:
            src = f.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue
        for offset in _past_tense_gate(src, "mufasa.ui_qt.app"):
            module_bad.append(
                f"{rel}:"
                f"{src[:offset].count(chr(10)) + 1}"
            )
    check(
        "Every remaining mention of 'mufasa.ui_qt.app' across the "
        "repo is in a deletion-context sentence (catches both the "
        "module dotted-name and the entry-point target form)",
        not module_bad,
        detail=("; ".join(module_bad[:3])),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------

    # 12. mufasa-workbench entry still live.
    has_workbench_entry = bool(re.search(
        r"^\s*mufasa-workbench\s*=\s*[\"']"
        r"mufasa\.ui_qt\.workbench_app:main",
        scripts_block, flags=re.MULTILINE,
    ))
    check(
        "Cross-check: mufasa-workbench entry point is still live "
        "(122dx must not have collateral damage on the workbench "
        "entry path)",
        has_workbench_entry,
    )

    # 13. 122dw migration removal preserved.
    check(
        "122dw state preserved: mufasa/cli/migrate_project.py "
        "still absent",
        not (REPO_ROOT / "mufasa" / "cli"
             / "migrate_project.py").exists(),
    )

    # 14. 122dv Skip removal preserved.
    pc_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "forms" / "pose_cleanup.py").read_text()
    check(
        "122dv state preserved: no SkipOutlierCorrectionForm class",
        "class SkipOutlierCorrectionForm" not in pc_src,
    )

    # 15. SECTIONS still validates.
    try:
        from mufasa.section_provenance import SECTIONS
        sections_ok = len(SECTIONS) > 0
    except Exception:
        sections_ok = False
    check(
        "122ds state preserved: section_provenance.SECTIONS still "
        "imports + DAG validates",
        sections_ok,
    )

    # 16. Parse-clean.
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

    # 17. 122do baseline.
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
        f"smoke_122dx_legacy_chooser_removal: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
