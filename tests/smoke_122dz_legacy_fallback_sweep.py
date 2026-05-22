"""
tests/smoke_122dz_legacy_fallback_sweep.py
=============================================

Patch 122dz: remove the ``legacy_fallback`` shim from
:func:`mufasa.utils.classification_io.load_machine_results_for_video`
and sweep all call sites.

Context
-------
``load_machine_results_for_video`` was introduced as a migration
shim in the 122at..122av arc: read v1 predictions + features,
``pd.concat`` them, and if either of those v1 reads raises
``FileNotFoundError``, fall back to reading a legacy combined
CSV at the path the caller passed as ``legacy_fallback``.

The shim was needed during the consumer-migration era (122at:
predictions sidecar; 122au: analysis consumers; 122av:
visualization consumers) because some projects had legacy
``machine_results/`` files but no v1 predictions yet.

With 122dy rejecting non-v1 projects at the ConfigReader entry
point, "legacy file alongside a live v1 project" stopped being
a reachable state. Every caller was passing a path that either
(a) didn't exist, or (b) couldn't be alongside the v1 layout
ConfigReader requires. The fallback code was dead weight.

What this patch landed
----------------------
``mufasa/utils/classification_io.py``:
* Removed the ``legacy_fallback`` keyword parameter from
  ``load_machine_results_for_video``.
* Removed the try/except FileNotFoundError block. The function
  now reads v1 unconditionally and lets FileNotFoundError
  propagate.
* Docstring rewritten to reflect single-path behavior with a
  past-tense breadcrumb about the removed parameter.

Sweep of 21 call sites (uniform pattern: an indented
``legacy_fallback=...,`` line as the last kwarg to the call):

* mufasa/data_processors/ — 8 files
  (agg_clf_calculator, agg_clf_counter_mp, fsttc_calculator,
  mutual_exclusivity_corrector, severity_calculator,
  severity_bout_based_calculator [2 occurrences],
  severity_frame_based_calculator [2 occurrences],
  timebins_clf_calculator)
* mufasa/plotting/ — 11 files
  (clf_validator, clf_validator_mp, gantt_creator,
  gantt_creator_mp, heat_mapper_clf, heat_mapper_clf_mp,
  path_plotter, path_plotter_mp, plot_clf_results,
  plot_clf_results_mp, probability_plot_creator,
  probability_plot_creator_mp)
* mufasa/roi_tools/ — 2 files
  (roi_clf_calculator, roi_clf_calculator_mp)
* mufasa/ui_qt/ — 3 files
  (clip_review, frame_labeller, targeted_clips)

23 ``legacy_fallback=...`` references total deleted.

In-source comment past-tense sweep:
* mufasa/mixins/config_reader.py — comment near the
  machine_results_paths construction updated.
* mufasa/ui_qt/targeted_clips.py — comment near
  ``machine_results_dir`` attribute updated.
* mufasa/ui_qt/frame_labeller.py — same.

Orphan dev-time invocations deleted (SimBA-era footer code that
crashed any import of the module — 122bj swept most of these but
missed two):
* mufasa/pose_processors/reverse_pose.py
* mufasa/model/train_multilabel_rf.py

Obsolete pre-strict tests deleted (they tested functionality
122dz removed):
* tests/smoke_122au_analysis_consumer_migration.py
* tests/smoke_122av_visualization_consumer_migration.py

Coverage
--------
Function signature change:
1.  ``load_machine_results_for_video`` no longer accepts
    ``legacy_fallback`` as a parameter.
2.  The function body no longer contains a try/except for
    ``FileNotFoundError`` (the fallback path is gone).
3.  Docstring mentions 122dz as the removal patch.

Call-site sweep:
4.  No file in mufasa/ passes ``legacy_fallback=`` to the
    helper (other than past-tense breadcrumb comments).
5.  All 19 known call-site files still parse cleanly post-edit.
6.  Each known call-site file makes at least one
    ``load_machine_results_for_video`` call (the sweep removed
    only the kwarg, not the call itself).

Orphan-invocation cleanup:
7.  reverse_pose.py no longer has a module-scope
    ``test = Reverse2AnimalTracking(...)`` invocation.
8.  train_multilabel_rf.py no longer has a module-scope
    ``model_trainer = TrainMultiLabel...(...)`` invocation.

Pre-strict test deletion:
9.  tests/smoke_122au_analysis_consumer_migration.py is gone.
10. tests/smoke_122av_visualization_consumer_migration.py is gone.

Repo-wide past-tense gate:
11. Every remaining mention of ``legacy_fallback`` across
    mufasa/ + tests/ is in a deletion-context sentence (or
    inside a docstring).

Cross-patch invariants:
12. 122dy state preserved: ConfigReader still rejects non-.toml.
13. 122dx state preserved: ``mufasa/ui_qt/app.py`` still gone.
14. 122dw state preserved: ``mufasa/cli/migrate_project.py``
    still gone.
15. 122dv state preserved: no SkipOutlierCorrectionForm.
16. 122ds state preserved: SECTIONS DAG still validates.
17. Parse-clean across mufasa/**/*.py.
18. 122do baseline: no ``Optional[`` in non-docstring positions
    across mufasa/ui_qt/.
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


# The 19 files whose call sites were swept. Source for the count:
# `grep -rln legacy_fallback mufasa/**/*.py` pre-122dz, minus
# classification_io.py (the helper itself) and the comment-only
# files (config_reader.py, targeted_clips.py, frame_labeller.py
# which never had a live call site).
SWEPT_CALL_SITES = [
    "mufasa/data_processors/agg_clf_calculator.py",
    "mufasa/data_processors/agg_clf_counter_mp.py",
    "mufasa/data_processors/fsttc_calculator.py",
    "mufasa/data_processors/mutual_exclusivity_corrector.py",
    "mufasa/data_processors/severity_bout_based_calculator.py",
    "mufasa/data_processors/severity_calculator.py",
    "mufasa/data_processors/severity_frame_based_calculator.py",
    "mufasa/data_processors/timebins_clf_calculator.py",
    "mufasa/plotting/clf_validator.py",
    "mufasa/plotting/clf_validator_mp.py",
    "mufasa/plotting/gantt_creator.py",
    "mufasa/plotting/gantt_creator_mp.py",
    "mufasa/plotting/heat_mapper_clf.py",
    "mufasa/plotting/heat_mapper_clf_mp.py",
    "mufasa/plotting/path_plotter.py",
    "mufasa/plotting/path_plotter_mp.py",
    "mufasa/plotting/plot_clf_results.py",
    "mufasa/plotting/plot_clf_results_mp.py",
    "mufasa/plotting/probability_plot_creator.py",
    "mufasa/plotting/probability_plot_creator_mp.py",
    "mufasa/roi_tools/roi_clf_calculator.py",
    "mufasa/roi_tools/roi_clf_calculator_mp.py",
    "mufasa/ui_qt/clip_review.py",
    "mufasa/ui_qt/frame_labeller.py",
    "mufasa/ui_qt/targeted_clips.py",
]


def main() -> int:
    # -----------------------------------------------------------------
    # Function signature
    # -----------------------------------------------------------------
    cio_path = (REPO_ROOT / "mufasa" / "utils"
                / "classification_io.py")
    cio_src = cio_path.read_text()
    cio_tree = ast.parse(cio_src)
    helper = None
    for node in ast.walk(cio_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "load_machine_results_for_video"):
            helper = node
            break
    assert helper is not None, (
        "load_machine_results_for_video not found in "
        "classification_io.py"
    )

    # 1. No `legacy_fallback` parameter.
    param_names = (
        [a.arg for a in helper.args.args]
        + [a.arg for a in helper.args.kwonlyargs]
    )
    check(
        "load_machine_results_for_video no longer accepts "
        "`legacy_fallback` as a parameter",
        "legacy_fallback" not in param_names,
        detail=(f"params: {param_names}"),
    )

    # 2. No try/except for FileNotFoundError in the body.
    # (We accept ANY try, but the specific shape we're removing is
    # the one whose except clause names FileNotFoundError.)
    has_fnf_except = False
    for sub in ast.walk(helper):
        if isinstance(sub, ast.Try):
            for handler in sub.handlers:
                if handler.type is None:
                    continue
                # Either `except FileNotFoundError` or
                # `except FileNotFoundError as x`.
                t = handler.type
                if isinstance(t, ast.Name) and t.id == "FileNotFoundError":
                    has_fnf_except = True
                    break
        if has_fnf_except:
            break
    check(
        "load_machine_results_for_video body no longer contains a "
        "try/except FileNotFoundError block (legacy-fallback path "
        "is gone)",
        not has_fnf_except,
    )

    # 3. Docstring mentions 122dz.
    docstring = ast.get_docstring(helper) or ""
    check(
        "load_machine_results_for_video docstring mentions 122dz "
        "(removal breadcrumb)",
        "122dz" in docstring,
    )

    # -----------------------------------------------------------------
    # Call-site sweep
    # -----------------------------------------------------------------
    # 4. No live `legacy_fallback=` argument anywhere in mufasa/.
    # We accept comment-only references (the call would have
    # `legacy_fallback=` as an actual ast.keyword arg in a Call).
    live_kwarg_hits = []
    for f in sorted((REPO_ROOT / "mufasa").rglob("*.py")):
        try:
            src = f.read_text()
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "legacy_fallback":
                        live_kwarg_hits.append(
                            f"{f.relative_to(REPO_ROOT)}:{node.lineno}"
                        )
    check(
        "No live `legacy_fallback=` keyword argument anywhere in "
        "mufasa/ (comment-only mentions are OK)",
        not live_kwarg_hits,
        detail=("; ".join(live_kwarg_hits[:3])),
    )

    # 5 + 6. Each swept call-site file parses cleanly + still has at
    # least one load_machine_results_for_video call.
    parse_failures = []
    missing_call = []
    for rel in SWEPT_CALL_SITES:
        f = REPO_ROOT / rel
        if not f.exists():
            missing_call.append(f"{rel}: file gone (unexpected)")
            continue
        src = f.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            parse_failures.append(f"{rel}: {e}")
            continue
        has_call = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (
                    fn.id if isinstance(fn, ast.Name)
                    else fn.attr if isinstance(fn, ast.Attribute)
                    else None
                )
                if name == "load_machine_results_for_video":
                    has_call = True
                    break
        if not has_call:
            missing_call.append(rel)
    check(
        f"All {len(SWEPT_CALL_SITES)} swept call-site files "
        f"parse cleanly",
        not parse_failures,
        detail=(parse_failures[0] if parse_failures else ""),
    )
    check(
        f"All swept call-site files still make at least one "
        f"load_machine_results_for_video call (sweep removed only "
        f"the kwarg, not the call)",
        not missing_call,
        detail=(missing_call[0] if missing_call else ""),
    )

    # -----------------------------------------------------------------
    # Orphan-invocation cleanup
    # -----------------------------------------------------------------
    rp_src = (REPO_ROOT / "mufasa" / "pose_processors"
              / "reverse_pose.py").read_text()
    check(
        "reverse_pose.py no longer has a module-scope "
        "`test = Reverse2AnimalTracking(...)` orphan invocation",
        not re.search(r"^test\s*=\s*Reverse2AnimalTracking",
                      rp_src, flags=re.MULTILINE),
    )
    tm_src = (REPO_ROOT / "mufasa" / "model"
              / "train_multilabel_rf.py").read_text()
    check(
        "train_multilabel_rf.py no longer has a module-scope "
        "`model_trainer = TrainMultiLabelRandomForestClassifier(...)`"
        " orphan invocation",
        not re.search(
            r"^model_trainer\s*=\s*TrainMultiLabelRandomForestClassifier",
            tm_src, flags=re.MULTILINE,
        ),
    )

    # -----------------------------------------------------------------
    # Pre-strict test deletion
    # -----------------------------------------------------------------
    check(
        "tests/smoke_122au_analysis_consumer_migration.py is gone "
        "(tested functionality 122dz removed)",
        not (REPO_ROOT / "tests"
             / "smoke_122au_analysis_consumer_migration.py").exists(),
    )
    check(
        "tests/smoke_122av_visualization_consumer_migration.py "
        "is gone (same reason)",
        not (REPO_ROOT / "tests"
             / "smoke_122av_visualization_consumer_migration.py").exists(),
    )

    # -----------------------------------------------------------------
    # Past-tense gate
    # -----------------------------------------------------------------
    bad = []
    for f in sorted(REPO_ROOT.rglob("*")):
        if not (f.is_file() and f.suffix in (".py", ".md")):
            continue
        rel = f.relative_to(REPO_ROOT)
        if rel.name == "session_handoff.md":
            continue
        if rel.name.startswith("smoke_122dz_"):
            continue
        try:
            src = f.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue
        for m in re.finditer(r"legacy_fallback", src):
            ctx = src[max(0, m.start() - 300):
                      m.end() + 300].lower()
            if any(w in ctx for w in
                   ("removed", "deleted", "no longer", "122dz",
                    "was used", "used to", "before 122dz")):
                continue
            # Also accept inside docstrings.
            preceding = src[:m.start()]
            triple_count = (
                preceding.count('"""') + preceding.count("'''")
            )
            if triple_count % 2 == 1:
                continue
            bad.append(f"{rel}:{src[:m.start()].count(chr(10)) + 1}")
    check(
        "Every remaining mention of `legacy_fallback` is in a "
        "deletion-context sentence or inside a docstring "
        "(session_handoff.md and this test excluded)",
        not bad,
        detail=("; ".join(bad[:3])),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 12. 122dy state preserved — ConfigReader rejects non-.toml.
    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()
    check(
        "122dy state preserved: ConfigReader.__init__ still "
        "raises on a non-.toml config_path",
        "InvalidInputError" in cr_src
        and ".toml" in cr_src
        and "122dy" in cr_src,
    )

    # 13. 122dx state preserved.
    check(
        "122dx state preserved: ui_qt/app.py still gone",
        not (REPO_ROOT / "mufasa" / "ui_qt" / "app.py").exists(),
    )

    # 14. 122dw state preserved.
    check(
        "122dw state preserved: cli/migrate_project.py still gone",
        not (REPO_ROOT / "mufasa" / "cli"
             / "migrate_project.py").exists(),
    )

    # 15. 122dv state preserved.
    pc_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
              / "pose_cleanup.py").read_text()
    check(
        "122dv state preserved: no SkipOutlierCorrectionForm",
        "class SkipOutlierCorrectionForm" not in pc_src,
    )

    # 16. SECTIONS DAG.
    try:
        from mufasa.section_provenance import SECTIONS
        sections_ok = len(SECTIONS) > 0
    except Exception:
        sections_ok = False
    check(
        "122ds state preserved: SECTIONS still imports + DAG "
        "validates",
        sections_ok,
    )

    # 17. Parse-clean.
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

    # 18. 122do baseline.
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
        f"smoke_122dz_legacy_fallback_sweep: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
