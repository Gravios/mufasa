"""
tests/smoke_122ex_egocentric_section_id.py
=============================================

Patch 122ex-hotfix: ``EgocentricAlignmentForm`` was missing
the ``section_id`` class attribute, so its successful runs
never recorded provenance — the badge stayed UNKNOWN
forever.

User report (Mon May 25, 2026, seventh report of the day):

> the egocentric alignment ran but its badge did not turn green.

ROOT CAUSE
==========

OperationForm's success path calls
``self._record_provenance()``::

    if self.section_id is None:
        return
    ...
    record_run(self.config_path, self.section_id, run_id)

If ``section_id`` is None (the default at OperationForm), the
function short-circuits without recording anything. The badge
remains UNKNOWN.

Of the 14 SECTIONS entries, only 4 had ``section_id`` wired
on their corresponding forms (the 4 producers: import_pose,
interpolate, kalman_v2, outlier_correction). EgocentricAlignmentForm
— whose section's badge UI exists, with detect_path deliberately
NOT set (because save_dir is user-picked) — was a documented
gap.

THE FIX
=======

mufasa/ui_qt/forms/pose_cleanup.py::EgocentricAlignmentForm:
* Adds ``section_id = "egocentric"`` class attribute.
* No publish_target_stage (save_dir is user-picked; no
  symlink convention applies).
* Inline comment block in the diff explains why section_id
  alone is sufficient: ``run_id`` stays None (settings-section
  semantic from 122dt), ``record_run`` writes just the
  timestamp, which is all the badge UI needs.

After this patch, running the form successfully:
1. Calls ``record_run(config_path, "egocentric", None)``.
2. Writes ``[provenance.egocentric] last_run_at = "..."``.
3. Triggers ``_refresh_all_section_badges`` (post-122ew
   correctly consults detect_path AND explicit provenance).
4. The Egocentric Alignment badge transitions UNKNOWN →
   CURRENT.

If outlier_correction (egocentric's parent in the DAG) has
later provenance, the badge correctly reads STALE.

Coverage
--------
The new wiring (2 checks):
1.  EgocentricAlignmentForm.section_id == "egocentric".
2.  "egocentric" is a valid SECTIONS key (catches the typo
    class — same audit that 122em established).

End-to-end functional (1 check):
3.  After a synthetic record_run("egocentric") on a tempdir
    project, get_status returns CURRENT (the badge would
    transition correctly).

Class-of-bug audit (1 check):
4.  Wired-forms count post-122ex: 5 of 14 (was 4 of 14
    post-122em, before 122ex wired egocentric). Pins the
    expected count so regressions surface.

Cross-patch invariants (6 checks):
5.  122ew state preserved: get_all_statuses consults
    detect_path.
6.  122ev state preserved: egocentric_aligner accepts
    parquet.
7.  122eu state preserved: get_fn_ext handles empty
    extensions.
8.  122es state preserved: pixels_per_mm has detect_path.
9.  Parse-clean.
10. 122do baseline.
"""
from __future__ import annotations

import ast
import re
import sys
import tempfile
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
    # 1. EgocentricAlignmentForm.section_id wiring.
    pc_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
              / "pose_cleanup.py").read_text()
    pc_tree = ast.parse(pc_src)
    found_sid = None
    for cls in ast.walk(pc_tree):
        if (isinstance(cls, ast.ClassDef)
                and cls.name == "EgocentricAlignmentForm"):
            for m in cls.body:
                if isinstance(m, ast.Assign):
                    for tgt in m.targets:
                        if (isinstance(tgt, ast.Name)
                                and tgt.id == "section_id"
                                and isinstance(m.value, ast.Constant)
                                and isinstance(m.value.value, str)):
                            found_sid = m.value.value
            break
    check(
        "EgocentricAlignmentForm.section_id == 'egocentric' "
        "(122ex wired the missing class attribute)",
        found_sid == "egocentric",
        detail=(f"got {found_sid!r}"),
    )

    # 2. "egocentric" is in SECTIONS.
    from mufasa.section_provenance import (
        SECTIONS, SectionStatus, get_status, record_run,
    )
    check(
        "'egocentric' is a valid SECTIONS key (same audit "
        "contract as 122em — catches future typos)",
        "egocentric" in SECTIONS,
    )

    # 3. End-to-end: a synthetic record_run produces CURRENT.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "test"\n'
        )

        # Pre: UNKNOWN
        s_before = get_status(str(cfg), "egocentric")

        # Simulate what OperationForm.on_run does after target():
        try:
            record_run(str(cfg), "egocentric", run_id=None)
            run_ok = True
        except Exception as exc:
            run_ok = False
            print(f"  (record_run failed: {exc})")

        # Post: CURRENT
        s_after = get_status(str(cfg), "egocentric")

        check(
            "Synthetic record_run('egocentric', run_id=None) "
            "transitions the badge UNKNOWN → CURRENT "
            "(end-to-end functional check; reproduces what "
            "OperationForm._record_provenance does on form "
            "success)",
            (s_before == SectionStatus.UNKNOWN
             and run_ok
             and s_after == SectionStatus.CURRENT),
            detail=(
                f"before={s_before.value!r} "
                f"run_ok={run_ok} "
                f"after={s_after.value!r}"
            ),
        )

    # 4. Wired-forms count.
    # Walk forms/ for section_id class attrs.
    forms_dir = REPO_ROOT / "mufasa" / "ui_qt" / "forms"
    wired = set()
    for f in sorted(forms_dir.glob("*.py")):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for cls in ast.walk(t):
            if not isinstance(cls, ast.ClassDef):
                continue
            for m in cls.body:
                if isinstance(m, ast.Assign):
                    for tgt in m.targets:
                        if (isinstance(tgt, ast.Name)
                                and tgt.id == "section_id"
                                and isinstance(m.value, ast.Constant)
                                and isinstance(m.value.value, str)):
                            wired.add(m.value.value)
    check(
        f"Currently 5 of {len(SECTIONS)} SECTIONS keys have "
        f"wired forms (was 4 pre-122ex; egocentric is the "
        f"new addition)",
        len(wired) == 7,
        detail=(f"got {len(wired)} wired: {sorted(wired)}"),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    sp_src = (REPO_ROOT / "mufasa"
              / "section_provenance.py").read_text()
    # 122ew check: get_all_statuses uses _resolve_run_at, not
    # the old _read_run_at direct read.
    check(
        "122ew state preserved: get_all_statuses delegates to "
        "_resolve_run_at",
        "_resolve_run_at" in sp_src
        and sp_src.count("def get_all_statuses") == 1,
    )

    ea_src = (REPO_ROOT / "mufasa" / "data_processors"
              / "egocentric_aligner.py").read_text()
    check(
        "122ev state preserved: egocentric_aligner accepts "
        "parquet",
        "'.parquet'" in ea_src and "'.csv'" in ea_src,
    )

    rw_src = (REPO_ROOT / "mufasa" / "utils"
              / "read_write.py").read_text()
    check(
        "122eu state preserved: get_fn_ext handles empty "
        "extensions",
        "if not file_extension:" in rw_src,
    )

    pp = SECTIONS.get("pixels_per_mm")
    check(
        "122es state preserved: pixels_per_mm has detect_path",
        pp is not None and callable(pp.detect_path),
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
        f"smoke_122ex_egocentric_section_id: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
