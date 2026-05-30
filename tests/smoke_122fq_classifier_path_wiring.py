"""
tests/smoke_122fq_classifier_path_wiring.py
===========================================

Patch 122fq — correct stale legacy-path references across the
classifier/annotation surface, plus a save-guard cleanup.

User request (Fri May 30, 2026):

> give me an overview of how to use the classifiers, and also check
> that they are wired correctly for the new directory structure
  (then: "yes" — fix the issues found)

CONTEXT / WHY
=============
Tracing the annotate -> train -> infer -> pseudo loop confirmed the
DATA FLOW is correctly wired to the v1 layout:
  * annotator saves -> derived/labels/<video>.parquet (label_io)
  * training reads   -> derived/labels/ via load_labels_for_video
  * inference writes -> derived/classifications/<video>.parquet
  * pseudo seeds     -> derived/classifications/ via classification_io
But a layer of STALE STRINGS still pointed at the legacy SimBA
csv/targets_inserted/ tree (removed for labels in 122ak):
  * the Frame Labeling launcher's on-screen "Path note" told users
    labels save to csv/targets_inserted/ (they save to derived/labels/)
  * frame_labeller docstrings claimed a load fallback / save dual-write
    that 122ak removed, and a pseudo comment that the loader's own
    docstring already calls "no longer true post-122at"
  * train_model_mixin error messages sent users to
    project_folder/csv/targets_inserted on feature/annotation faults

These mislead during exactly the workflow being tested.

WHAT THIS PATCH LANDED
======================
mufasa/model/train_rf.py
* save(): replaced the inverted/fragile guard
    `if not os.listdir(self.model_dir_out): os.makedirs(...)`
  with `os.makedirs(self.model_dir_out, exist_ok=True)`. NB: this is
  HYGIENE, not a crash fix — read_model_settings_from_config already
  creates generated_models/ (+ a model_evaluations subdir) at init, so
  the old guard's crashing branches were unreachable in normal flow.
  The replacement is correct in all dir states and survives init-order
  refactors.

mufasa/ui_qt/forms/annotation.py
* FrameLabellingLauncher docstring + on-screen "Path note": labels now
  correctly shown saving to derived/labels/. Pseudo mode label reads
  "seed from classifications" (was "machine_results").

mufasa/ui_qt/frame_labeller.py
* module docstring: load reads derived/labels/ only (122ak removed the
  csv/targets_inserted fallback); save writes labels-only to
  derived/labels/ (122ak removed the dual-write).
* _load_continue_labels docstring + the pseudo-branch comment corrected.

mufasa/mixins/train_model_mixin.py
* both check_all_dfs_in_list_has_same_cols source labels and seven
  FaultyTrainingSet / NaN / "0 observations" error messages now point
  at derived/features/ (feature faults) or derived/labels/ (annotation
  faults) instead of csv/targets_inserted/.

WHAT THIS PATCH DID NOT CHANGE
==============================
* Three docstring `>>>` usage examples in train_model_mixin still show
  csv/targets_inserted paths — illustrative only, deferred.
* train_multiclass_rf.save_model has no makedirs guard but is SAFE for
  the same init reason; left as-is.
* grid_search_rf / grid_search_multiclass_rf already use the correct
  `if not os.path.exists: makedirs` guard — verified, untouched.
* The broad SimBA->mufasa rebrand of remaining strings is out of scope.

NEW SMOKE: smoke_122fq_classifier_path_wiring.py (10 checks)
"""

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


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _func_src(src: str, fn_name: str) -> str:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            return ast.unparse(node)
    return ""


def main() -> int:
    # --- train_rf save guard cleanup -------------------------------------
    tr_src = _read("mufasa/model/train_rf.py")
    save_src = _func_src(tr_src, "save")
    check(
        "train_rf.save uses makedirs(exist_ok=True), no os.listdir guard",
        "exist_ok=True" in save_src and "os.listdir(self.model_dir_out)" not in save_src,
    )

    # --- annotation.py launcher text -------------------------------------
    an_src = _read("mufasa/ui_qt/forms/annotation.py")
    # Any surviving csv/targets_inserted mention must be a historical note.
    ti_historical = all(
        ("122ak" in ln or "dropped" in ln or "legacy" in ln.lower())
        for ln in an_src.splitlines()
        if "csv/targets_inserted" in ln
    )
    check(
        "annotation.py launcher: labels save to derived/labels/ (targets_inserted only historical)",
        "<code>derived/labels/</code>" in an_src and ti_historical,
    )
    check(
        "annotation.py pseudo mode label says 'seed from classifications'",
        "seed from classifications" in an_src and "seed from machine_results" not in an_src,
    )

    # --- frame_labeller docstrings/comments ------------------------------
    fl_src = _read("mufasa/ui_qt/frame_labeller.py")
    cont_src = _func_src(fl_src, "_load_continue_labels")
    check(
        "frame_labeller _load_continue_labels: derived/labels only (122ak)",
        "derived/labels/" in cont_src and "removed in 122ak" in cont_src,
    )
    check(
        "frame_labeller pseudo comment corrected (classification_io)",
        "reads machine_results which doesn't have" not in fl_src
        and "resolves derived/classifications/ (v1) first" in fl_src,
    )
    check(
        "frame_labeller module docstring: no live dual-write claim",
        "the legacy\n  ``targets_inserted/<video>.<ext>`` write stays" not in fl_src,
    )

    # --- train_model_mixin error-message paths ---------------------------
    tmm_src = _read("mufasa/mixins/train_model_mixin.py")
    check(
        "train_model_mixin col-consistency source = derived/features + derived/labels",
        tmm_src.count("source='derived/features + derived/labels'") == 2
        and "source='/project_folder/csv/targets_inserted'" not in tmm_src,
    )
    # Every surviving csv/targets_inserted mention must be a docstring example (>>>).
    stale_runtime = [
        ln.strip()
        for ln in tmm_src.splitlines()
        if "csv/targets_inserted" in ln and not ln.lstrip().startswith(">>>")
    ]
    check(
        "train_model_mixin: no csv/targets_inserted outside >>> examples",
        not stale_runtime,
        detail=f"{len(stale_runtime)} runtime line(s): {stale_runtime[:2]}",
    )
    check(
        "train_model_mixin error msgs reference derived/features and derived/labels",
        "project_folder/derived/features" in tmm_src
        and "project_folder/derived/labels" in tmm_src,
    )

    # --- everything still parses -----------------------------------------
    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"all mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122fq_classifier_path_wiring: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
