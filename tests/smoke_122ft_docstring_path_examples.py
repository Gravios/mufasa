"""
tests/smoke_122ft_docstring_path_examples.py
============================================

Patch 122ft — correct the four stale csv/targets_inserted references in
train_model_mixin docstring examples (the last documented item in the
"path-correctness" thread started in 122fq).

User request: "continue" (autonomous backlog selection — lowest-risk
remaining cleanup; the functional items left, converters.geometry_to_rle
and reverse_pose "987", need maintainer input).

CONTEXT / WHY
=============
122fq fixed the runtime/error-message legacy paths but deliberately left
four `>>>` docstring examples pointing at csv/targets_inserted, guarded
by the 122fq invariant "no csv/targets_inserted outside >>> examples".
This patch clears the examples themselves so the file has zero stale
references:

  * read_all_files_in_folder example (line ~130): the 122ak comment
    states file_paths are pseudo-paths used only for stem extraction, so
    the targets_inserted/ prefix was doubly misleading. Reduced to bare
    stems: file_paths=['Video_1.csv', 'Video_2.csv'].
  * random_multiclass_frm_sampler / random_multiclass_bout_sampler /
    create_shap_log_concurrent_mp examples: pd.read_csv input paths
    repointed csv/targets_inserted -> derived/labels (the v1 annotation
    location). read_csv mechanics left untouched — illustrative only.

WHAT THIS PATCH DID NOT CHANGE
==============================
* No code paths — docstrings only. Behaviour is byte-for-byte identical.
* Functional backlog (converters.geometry_to_rle pycocotools decision;
  reverse_pose "987" extractor) — still needs maintainer input.
* Broad ruff --fix (~6.4k autofixable) + SimBA->mufasa rebrand — pending
  an explicit scoping decision (large diff; some rule classes, e.g. B905
  strict= and E722, are behaviour-changing and should not be applied
  blindly).

NEW SMOKE: smoke_122ft_docstring_path_examples.py (4 checks)
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


def main() -> int:
    tmm = _read("mufasa/mixins/train_model_mixin.py")

    check(
        "train_model_mixin has zero targets_inserted references",
        "targets_inserted" not in tmm,
        detail=f"{tmm.count('targets_inserted')} remaining",
    )
    check(
        "read_all_files_in_folder example uses bare stems",
        "file_paths=['Video_1.csv', 'Video_2.csv']" in tmm,
    )
    check(
        "pd.read_csv examples repointed to derived/labels",
        tmm.count("derived/labels") >= 3 and "csv/targets_inserted" not in tmm,
    )

    # still compiles/parses
    parse_ok = True
    try:
        ast.parse(tmm)
    except SyntaxError:
        parse_ok = False
    check("train_model_mixin parses cleanly", parse_ok)

    print(
        f"smoke_122ft_docstring_path_examples: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
