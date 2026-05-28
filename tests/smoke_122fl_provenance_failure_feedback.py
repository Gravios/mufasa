"""
tests/smoke_122fl_provenance_failure_feedback.py
===================================================

Patch 122fl — surface provenance-recording failures to the user.

Standing item from 122em ("louder _record_provenance feedback").
The KeyError case was already hardened in 122eo (logging.error).
This patch closes the REMAINING silent-failure gap: the broad
``except Exception`` branch previously did a bare ``print`` to
stdout — invisible in a packaged app — leaving the user staring
at a status badge that stayed white after a successful run, with
no explanation.

THE GAP
=======

OperationForm._record_provenance was fire-and-forget: it returned
None and swallowed all failures. When record_run failed (a
section_id typo → KeyError, or a runtime issue like a read-only
project.toml → Exception), the operation's "Done." dialog still
showed plain "Done." and the badge stayed white. The user had no
way to know the badge was stale vs. the operation having silently
not done what they expected.

This matters MORE after this session's work: 122ex, 122fj wired
several new section_ids. A typo or a registration miss in any of
them would manifest as exactly this confusing symptom.

THE FIX
=======

mufasa/ui_qt/workbench.py::OperationForm:

* _record_provenance now returns ``str | None``:
  - None when there's nothing to record (no section_id / no
    config_path — not failures) OR on success.
  - A human-readable error string when a declared section_id
    failed to record.

* The broad ``except Exception`` branch:
  - Upgraded ``print(...)`` → ``logging.warning(...)`` (visible
    in packaged apps; print to stdout is not).
  - Sets a prov_error string explaining the operation succeeded
    but the badge couldn't update, with a hint to re-run / check
    that the project folder is writable.

* The KeyError branch (from 122eo) keeps logging.error and now
  ALSO sets a prov_error string (internal-configuration-error
  wording).

* The publish_to_stage failure branch: print → logging.warning
  too (same packaged-app-visibility rationale), but NOT
  surfaced to the user — publishing is a secondary convenience,
  not the primary output, and a publish failure doesn't make the
  badge lie.

* OperationForm.on_run's _on_success appends prov_error to the
  "Done." dialog when present:
      "Done.\\n\\n<prov_error>"
  so the user learns WHY the badge didn't update.

COVERAGE
========

Return-value contract (3 checks):
1.  _record_provenance has return annotation str | None.
2.  The KeyError handler sets prov_error (a returned string).
3.  The Exception handler sets prov_error AND uses
    logging.warning (not bare print).

Success-path wiring (3 checks):
4.  _on_success captures the _record_provenance return value.
5.  _on_success branches the dialog text on prov_error
    (appends it when present).
6.  The non-error path still shows plain "Done." (no
    regression).

print → logging migration (2 checks):
7.  No bare print( remains in _record_provenance (both the
    record_run failure and publish failure paths now use
    logging).
8.  logging.warning appears in both the Exception branch and
    the publish-failure branch.

Cross-patch invariants (3 checks):
9.  122eo state preserved: KeyError handler still uses
    logging.error and does not raise.
10. 122fk state preserved: SectionSpec has content_predicate.
11. Parse-clean.
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


def _get_method(cls_node: ast.ClassDef, name: str):
    for m in cls_node.body:
        if isinstance(m, ast.FunctionDef) and m.name == name:
            return m
    return None


def main() -> int:
    wb_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "workbench.py").read_text()
    wb_tree = ast.parse(wb_src)

    op_form = None
    for node in ast.walk(wb_tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "OperationForm"):
            op_form = node
            break
    assert op_form is not None, "OperationForm class missing"

    rec = _get_method(op_form, "_record_provenance")
    assert rec is not None, "_record_provenance missing"
    rec_src = ast.unparse(rec)

    # -----------------------------------------------------------------
    # Return-value contract
    # -----------------------------------------------------------------
    # 1. Return annotation str | None.
    ann_ok = False
    if rec.returns is not None:
        ann_src = ast.unparse(rec.returns)
        ann_ok = "str" in ann_src and "None" in ann_src
    check(
        "_record_provenance has return annotation `str | None` "
        "(was `None` — now returns an error string on failure)",
        ann_ok,
        detail=(
            f"got {ast.unparse(rec.returns) if rec.returns else None!r}"
        ),
    )

    # Locate the try/except inside _record_provenance.
    handlers = []
    for node in ast.walk(rec):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                handlers.append(h)
    # The first try has KeyError + Exception handlers.
    keyerror_src = ""
    exception_src = ""
    for h in handlers:
        htype = ast.unparse(h.type) if h.type else ""
        if htype == "KeyError" and not keyerror_src:
            keyerror_src = ast.unparse(h)
        elif htype == "Exception" and not exception_src:
            exception_src = ast.unparse(h)

    # 2. KeyError handler sets prov_error.
    check(
        "The KeyError handler sets a prov_error string (so a "
        "section_id-not-registered bug surfaces to the user, "
        "not just to the dev log)",
        "prov_error" in keyerror_src,
    )

    # 3. Exception handler sets prov_error AND uses logging.warning.
    check(
        "The Exception handler sets prov_error AND uses "
        "logging.warning (122fl upgrade — replaces the bare "
        "print that was invisible in packaged apps)",
        ("prov_error" in exception_src
         and "logging.warning" in exception_src),
        detail=(
            f"has_prov_error={'prov_error' in exception_src} "
            f"has_logging_warning="
            f"{'logging.warning' in exception_src}"
        ),
    )

    # -----------------------------------------------------------------
    # Success-path wiring
    # -----------------------------------------------------------------
    on_run = _get_method(op_form, "on_run")
    assert on_run is not None, "on_run missing"
    on_run_src = ast.unparse(on_run)

    check(
        "_on_success captures the _record_provenance return "
        "value (prov_error = self._record_provenance())",
        "prov_error = self._record_provenance()" in on_run_src,
    )

    check(
        "_on_success branches the dialog text on prov_error "
        "(appends it to 'Done.' when present)",
        ("if prov_error" in on_run_src
         and "Done." in on_run_src),
    )

    check(
        "_on_success still shows plain 'Done.' on the no-error "
        "path (no regression for the common success case)",
        "'Done.'" in on_run_src or '"Done."' in on_run_src,
    )

    # -----------------------------------------------------------------
    # print → logging migration
    # -----------------------------------------------------------------
    check(
        "No bare print( remains in _record_provenance (both the "
        "record_run-failure and publish-failure paths migrated "
        "to logging — print to stdout is invisible in packaged "
        "apps)",
        "print(" not in rec_src,
        detail=("print( still present" if "print(" in rec_src
                else ""),
    )

    check(
        "logging.warning appears in BOTH the Exception branch "
        "and the publish-failure branch",
        rec_src.count("logging.warning") >= 2,
        detail=(
            f"logging.warning count: "
            f"{rec_src.count('logging.warning')}"
        ),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 122eo: KeyError handler still uses logging.error + no raise.
    check(
        "122eo state preserved: KeyError handler uses "
        "logging.error and does not raise",
        ("logging.error" in keyerror_src
         and "raise" not in keyerror_src),
    )

    from mufasa.section_provenance import SectionSpec
    check(
        "122fk state preserved: SectionSpec has the "
        "content_predicate field",
        "content_predicate" in SectionSpec.__dataclass_fields__,
    )

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

    print(
        f"smoke_122fl_provenance_failure_feedback: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
