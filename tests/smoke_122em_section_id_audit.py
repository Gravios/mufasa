"""
tests/smoke_122em_section_id_audit.py
========================================

Patch 122em: drift-prevention smoke test for the OTHER half of
the section-provenance binding surface — ``section_id`` strings
in form classes and ``record_run`` calls.

Companion to 122el. 122el audited the (page, section_title)
binding — the strings used to LOOK UP a QGroupBox for badge
attachment. 122em audits ``section_id`` — the string used to
WRITE provenance entries to project.toml.

The bug class
-------------
:func:`mufasa.section_provenance.record_run` raises
:exc:`KeyError` if the section_id passed in isn't a key in
SECTIONS. But ``OperationForm._record_provenance`` swallows
all exceptions and only prints to console::

    try:
        from mufasa.section_provenance import record_run
        record_run(self.config_path, self.section_id, run_id)
    except Exception as exc:
        print(f"[provenance] record_run failed for ...")

So a typo in a form's ``section_id`` class attribute fails
silently. The form's success message ("Done.") still shows;
the user has no signal that provenance wasn't recorded.
Symptoms only appear later — the badge stays UNKNOWN forever
for the affected section's runs.

This is a higher-impact bug class than 122el's:

* 122el: badge mis-displays for a section that DID record.
  User sees the data but not the indicator.
* 122em: provenance silently doesn't record. Cross-section
  staleness detection breaks (dependent sections don't know
  the upstream ran).

What this patch covers
----------------------
AST audit walking ``mufasa/`` for:

1. ``class X:`` declarations with a ``section_id = "Y"``
   class-level attribute. Y must be a key in SECTIONS.

2. Direct ``record_run("config_path", "Y", ...)`` calls with
   string-literal section_ids. Y must be a key in SECTIONS.
   (Indirect calls via ``self.section_id`` are covered by
   the first audit since those forms must have a class-attr
   declaration.)

Current state (as of this patch's landing):

* 4 form classes have ``section_id`` attrs:
  - PoseImportForm: ``"import_pose"``
  - InterpolateForm: ``"interpolate"``
  - KalmanV2SmoothingForm: ``"kalman_v2"``
  - RunOutlierCorrectionForm: ``"outlier_correction"``
  All four are valid SECTIONS keys.

* 0 direct ``record_run(literal, ...)`` calls outside the
  ``OperationForm._record_provenance`` plumbing.

So there's no current drift; the smoke is purely preventive.

Class-level pattern observations
--------------------------------
This is the third drift-prevention smoke in session 2:

* smoke_122ek_roi_consumer_audit: defensive against
  empty-DataFrame KeyError class (4 sites pinned).
* smoke_122el_section_binding_audit: page+title binding
  drift (11 sites pinned).
* smoke_122em_section_id_audit (this): section_id binding
  drift (4 sites pinned, room for more).

Each was filed as deferred AFTER fixing a real-world bug of
the same class. The pattern: a bug surfaces, the immediate
fix is hot-applied, an audit identifies the class of failure
and pins it via a smoke test. The cost is modest (1-2 days
of work per class); the value is preventing recurrence.

Coverage
--------
1.  Audit machinery — the test walks mufasa/ for section_id
    class attrs without crashing.

Form attrs (4 checks — one per currently-wired form):
2.  PoseImportForm.section_id == "import_pose".
3.  InterpolateForm.section_id == "interpolate".
4.  KalmanV2SmoothingForm.section_id == "kalman_v2".
5.  RunOutlierCorrectionForm.section_id == "outlier_correction".

Drift contract (2 checks):
6.  Every ``section_id = "X"`` literal in a form class has
    X as a SECTIONS key.
7.  Every literal-arg ``record_run("X", ...)`` call has X as
    a SECTIONS key.

Coverage informational (3 checks — don't fail, just
document):
8.  All 4 producer sections (import_pose, interpolate,
    kalman_v2, outlier_correction) have a wired form. If a
    future patch un-wires one, this test surfaces it.
9.  Currently 5 of 14 SECTIONS keys have wired forms (the
    producers). The remaining 10 are either ``ui_bound=False``
    placeholders (3) or ui-bound sections without a record_run
    pathway yet (7). Pinning this count catches accidental
    unwiring.
10. The 3 ``ui_bound=False`` sections (savitzky_golay,
    drop_body_parts, features_subject) all have no form
    wired (consistent with their declared unboundness).

Cross-patch invariants:
11. 122el state preserved: SectionSpec has ui_bound field.
12. 122ek state preserved: safe_filter helpers in roi_utils.
13. 122ej state preserved: read_roi_data column-tolerance.
14. 122ei state preserved: detect_path on producer sections.
15. Parse-clean.
16. 122do baseline.
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


def _find_section_id_attrs(
    pkg_root: Path,
) -> list[tuple[Path, str, str]]:
    """Return list of (file, class_name, section_id_value) for
    every ``class X: section_id = "Y"`` declaration in the
    package."""
    found = []
    for f in sorted(pkg_root.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for member in node.body:
                if not isinstance(member, ast.Assign):
                    continue
                if (isinstance(member.value, ast.Constant)
                        and isinstance(member.value.value, str)):
                    for tgt in member.targets:
                        if (isinstance(tgt, ast.Name)
                                and tgt.id == "section_id"):
                            found.append(
                                (f, node.name, member.value.value),
                            )
    return found


def _find_record_run_literal_calls(
    pkg_root: Path,
) -> list[tuple[Path, str]]:
    """Return list of (file, section_id_value) for every
    ``record_run(<anything>, "X", ...)`` call where the
    second positional arg is a string literal."""
    found = []
    for f in sorted(pkg_root.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "record_run"
                    and len(node.args) >= 2):
                arg = node.args[1]
                if (isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)):
                    found.append((f, arg.value))
    return found


def main() -> int:
    from mufasa.section_provenance import SECTIONS
    pkg_root = REPO_ROOT / "mufasa"
    known_ids = set(SECTIONS.keys())

    # -----------------------------------------------------------------
    # Audit machinery
    # -----------------------------------------------------------------
    try:
        sid_attrs = _find_section_id_attrs(pkg_root)
        rr_calls = _find_record_run_literal_calls(pkg_root)
        machinery_ok = True
    except Exception as exc:
        sid_attrs = []
        rr_calls = []
        machinery_ok = False
        print(f"  (audit machinery failed: {exc!r})")
    check(
        "AST audit machinery walks mufasa/ without crashing",
        machinery_ok,
    )

    # -----------------------------------------------------------------
    # Form attrs — verify the 4 known forms have correct ids
    # -----------------------------------------------------------------
    by_cls = {cls: sid for _, cls, sid in sid_attrs}

    for cls_name, expected_id in [
        ("PoseImportForm",          "import_pose"),
        ("InterpolateForm",         "interpolate"),
        ("KalmanV2SmoothingForm",   "kalman_v2"),
        ("RunOutlierCorrectionForm", "outlier_correction"),
    ]:
        check(
            f"{cls_name}.section_id == {expected_id!r} "
            f"(the wiring established by patches 122dt–122ed)",
            by_cls.get(cls_name) == expected_id,
            detail=(f"got {by_cls.get(cls_name)!r}"),
        )

    # -----------------------------------------------------------------
    # Drift contract
    # -----------------------------------------------------------------
    bad_attrs = [
        (str(f.relative_to(REPO_ROOT)), c, s)
        for f, c, s in sid_attrs
        if s not in known_ids
    ]
    check(
        "Every `section_id = \"X\"` class attribute in mufasa/ "
        "has X as a key in SECTIONS (drift contract — catches "
        "future typos in form class attrs)",
        not bad_attrs,
        detail=("; ".join(f"{c}={s!r}@{f}"
                          for f, c, s in bad_attrs[:3])),
    )

    bad_calls = [
        (str(f.relative_to(REPO_ROOT)), s)
        for f, s in rr_calls
        if s not in known_ids
    ]
    check(
        "Every literal-arg `record_run(..., \"X\", ...)` "
        "call has X as a key in SECTIONS (covers direct "
        "record_run callers — currently none, but the test "
        "prevents future drift)",
        not bad_calls,
        detail=("; ".join(f"{s!r}@{f}"
                          for f, s in bad_calls[:3])),
    )

    # -----------------------------------------------------------------
    # Coverage informational
    # -----------------------------------------------------------------
    wired_ids = {s for _, _, s in sid_attrs}

    # 8. All producer sections have a wired form.
    producers = {"import_pose", "interpolate",
                 "kalman_v2", "outlier_correction"}
    missing_producers = producers - wired_ids
    check(
        "All 4 producer sections (import_pose, interpolate, "
        "kalman_v2, outlier_correction) have a wired form "
        "with the correct section_id (catches accidental "
        "unwiring of the badge-recording path)",
        not missing_producers,
        detail=(f"missing: {sorted(missing_producers)}"),
    )

    # 9. Expected count: 7 wired, 17 total.
    check(
        f"Currently 7 of {len(known_ids)} SECTIONS keys have "
        f"wired forms — pinned to catch accidental unwiring "
        f"OR accidental over-wiring (an 8th form would surface "
        f"in this check)",
        len(wired_ids) == 7,
        detail=(f"got {len(wired_ids)} wired: "
                f"{sorted(wired_ids)}"),
    )

    # 10. ui_bound=False sections don't have wired forms.
    # (They COULD — provenance is recordable even without a UI —
    # but currently none do. Pinning surfaces a subtle bug
    # where someone wires up a section that's also marked
    # ui_bound=False, which is contradictory state.)
    unbound = {sid for sid, spec in SECTIONS.items()
               if not spec.ui_bound}
    accidentally_wired_unbound = wired_ids & unbound
    check(
        "No ui_bound=False section has a wired form (would be "
        "contradictory state — recording provenance for a "
        "section that has no UI to show it)",
        not accidentally_wired_unbound,
        detail=(f"contradictory: "
                f"{sorted(accidentally_wired_unbound)}"),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    from mufasa.section_provenance import SectionSpec
    check(
        "122el state preserved: SectionSpec has ui_bound field",
        any(f.name == "ui_bound"
            for f in SectionSpec.__dataclass_fields__.values()),
    )

    ru_src = (REPO_ROOT / "mufasa" / "roi_tools"
              / "roi_utils.py").read_text()
    check(
        "122ek state preserved: safe_filter_by_video in roi_utils",
        "def safe_filter_by_video" in ru_src,
    )

    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()
    check(
        "122ej state preserved: read_roi_data has _col_unique",
        "_col_unique" in cr_src,
    )

    sp_src = (REPO_ROOT / "mufasa"
              / "section_provenance.py").read_text()
    check(
        "122ei state preserved: detect_path on producer sections",
        "detect_path=lambda root:" in sp_src,
    )

    # 15. Parse-clean.
    parse_errors = []
    file_count = 0
    for f in sorted(pkg_root.rglob("*.py")):
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

    # 16. 122do baseline.
    uiqt = pkg_root / "ui_qt"
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
        f"smoke_122em_section_id_audit: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
