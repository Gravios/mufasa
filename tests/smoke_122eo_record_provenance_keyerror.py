"""
tests/smoke_122eo_record_provenance_keyerror.py
==================================================

Patch 122eo: louder failure mode for ``_record_provenance``
KeyError. Filed by 122em as deferred — implementing it now.

Background
----------
``mufasa.ui_qt.workbench.OperationForm._record_provenance``
wrapped ``record_run`` calls in a catch-all-Exception block,
printing the failure to stdout::

    try:
        record_run(self.config_path, self.section_id, run_id)
    except Exception as exc:
        print(f"[provenance] record_run failed for ...")

``record_run`` raises ``KeyError`` for unknown section_ids
(the 122dt "central declaration is authoritative" semantic).
That's a PROGRAMMING bug — the form class attribute is wrong
— but the catch-all-and-print made it indistinguishable from
a transient IO error.

The smoke_122em_section_id_audit smoke catches this at commit
time. 122eo provides a runtime fallback for the case where
the smoke wasn't run before shipping.

Why ``logging.error`` and not ``raise`` or ``QMessageBox``
----------------------------------------------------------
* **raise** would propagate to the success-callback wrapper
  and crash the form. Heavy-handed for a programming bug
  the user can't fix anyway.
* **QMessageBox.critical** would interrupt the user with a
  modal dialog. Helpful for developers running the app
  during testing, but disruptive when the bug ships to
  end-users (which it shouldn't, but might).
* **logging.error** is the standard balance: visible in
  stderr where errors belong; doesn't interrupt the user;
  controllable via standard logging config; distinguishable
  from regular stdout output.

Non-KeyError exceptions keep the prior behavior (print to
stdout). Those ARE transient (IO errors, file locks); the
user-facing UX should be unobtrusive.

What this patch landed
----------------------
mufasa/ui_qt/workbench.py:

* The single ``except Exception`` block in
  ``OperationForm._record_provenance`` split into two
  handlers:
  1. ``except KeyError as exc:`` — logs via ``logging.error``
     with an explanatory message naming the offending
     section_id and pointing developers at SECTIONS.
  2. ``except Exception as exc:`` — unchanged print behavior
     for transient runtime issues.

* The KeyError handler explicitly does NOT ``return`` or
  ``raise`` — the publish-to-stage call below stays
  attempted, since it's an independent concern (it might
  succeed even if record_run failed).

The 122do-style cascade in publish_to_stage block (the
SECOND ``try/except`` in the method) is NOT touched —
publish_to_stage doesn't have a KeyError-class failure mode
to distinguish.

Coverage
--------
1.  ``_record_provenance`` method has TWO except handlers
    (was one) — verified via AST inspection.
2.  The first handler is ``except KeyError`` (specific
    before general — Python catches the first matching
    handler).
3.  The second handler is ``except Exception`` (still
    catches everything else).
4.  The KeyError handler uses ``logging.error`` (the
    chosen vehicle for programming-bug surfacing).
5.  The KeyError handler's message mentions both the
    offending section_id AND points at the central
    SECTIONS declaration (helps developers find the fix).
6.  The KeyError handler does NOT raise — control flow
    continues to the publish_to_stage attempt below.
7.  The Exception handler retains its prior print-based
    behavior (no regression for transient failures).

Cross-patch invariants:
8.  smoke_122em audit still in place (the commit-time
    drift detector).
9.  122en state preserved: v1_project_paths is the
    canonical layout helper.
10. 122el state preserved: SectionSpec.ui_bound field
    exists.
11. Parse-clean.
12. 122do baseline.
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
    wb_path = REPO_ROOT / "mufasa" / "ui_qt" / "workbench.py"
    wb_src = wb_path.read_text()
    wb_tree = ast.parse(wb_src)

    # Locate OperationForm._record_provenance.
    method = None
    for cls in ast.walk(wb_tree):
        if (isinstance(cls, ast.ClassDef)
                and cls.name == "OperationForm"):
            for m in cls.body:
                if (isinstance(m, ast.FunctionDef)
                        and m.name == "_record_provenance"):
                    method = m
                    break
            break
    assert method is not None, (
        "OperationForm._record_provenance not found"
    )

    # Walk for Try nodes; the first one wraps the record_run call.
    try_nodes = [n for n in ast.walk(method)
                 if isinstance(n, ast.Try)]
    assert len(try_nodes) >= 1, (
        "No try blocks found in _record_provenance"
    )
    first_try = try_nodes[0]
    handlers = first_try.handlers

    # 1. TWO handlers.
    check(
        "_record_provenance's first try block has TWO except "
        "handlers (was one — patch 122eo split it to "
        "distinguish KeyError from other failures)",
        len(handlers) == 2,
        detail=(f"got {len(handlers)} handler(s)"),
    )

    # 2. First handler is KeyError.
    h0_type = (handlers[0].type.id
               if (handlers and isinstance(handlers[0].type, ast.Name))
               else None)
    check(
        "First handler is `except KeyError` (specific before "
        "general — Python catches the first matching handler)",
        h0_type == "KeyError",
        detail=(f"got {h0_type!r}"),
    )

    # 3. Second handler is Exception.
    h1_type = (handlers[1].type.id
               if (len(handlers) > 1
                   and isinstance(handlers[1].type, ast.Name))
               else None)
    check(
        "Second handler is `except Exception` (still catches "
        "transient runtime failures)",
        h1_type == "Exception",
        detail=(f"got {h1_type!r}"),
    )

    # 4. KeyError handler uses logging.error.
    if handlers:
        h0_src = ast.unparse(handlers[0])
    else:
        h0_src = ""
    check(
        "KeyError handler uses `logging.error` (the chosen "
        "vehicle for programming-bug surfacing — visible in "
        "stderr, doesn't interrupt the user)",
        "logging.error" in h0_src,
    )

    # 5. KeyError handler message mentions section_id and SECTIONS.
    check(
        "KeyError handler message names both the offending "
        "section_id AND points developers at SECTIONS",
        "section_id" in h0_src and "SECTIONS" in h0_src,
    )

    # 6. KeyError handler does NOT raise.
    has_raise = any(
        isinstance(n, ast.Raise) for n in ast.walk(handlers[0])
    )
    check(
        "KeyError handler does NOT raise (control flow "
        "continues to the publish_to_stage attempt below — "
        "the two are independent concerns)",
        not has_raise,
    )

    # 7. Exception handler retains print behavior.
    if len(handlers) > 1:
        h1_src = ast.unparse(handlers[1])
    else:
        h1_src = ""
    check(
        "Exception handler retains its prior `print(...)` "
        "behavior (no regression for transient runtime "
        "failures)",
        "print(" in h1_src,
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    em_smoke = REPO_ROOT / "tests" / "smoke_122em_section_id_audit.py"
    check(
        "smoke_122em audit still in place (the commit-time "
        "drift detector; this patch is the runtime fallback)",
        em_smoke.exists(),
    )

    pl_src = (REPO_ROOT / "mufasa"
              / "project_layout.py").read_text()
    check(
        "122en state preserved: v1_project_paths is the "
        "canonical v1 layout helper",
        "def v1_project_paths" in pl_src,
    )

    from mufasa.section_provenance import SectionSpec
    check(
        "122el state preserved: SectionSpec.ui_bound field",
        any(f.name == "ui_bound"
            for f in SectionSpec.__dataclass_fields__.values()),
    )

    # 11. Parse-clean.
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

    # 12. 122do baseline.
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
        "122do baseline preserved: no `Optional[` in non-"
        "docstring positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122eo_record_provenance_keyerror: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
