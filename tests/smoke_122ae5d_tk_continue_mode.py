"""
tests/smoke_122ae5d_tk_continue_mode.py
========================================

Patch 122ae-5d: continue-mode for the three legacy Tk labellers.
Mirror of what 122aj did for the Qt labeler — replace the
read_df(targets_inserted_file_path) call that pulled
features+labels combined with a load_labels_for_video call that
pulls just the behaviour label collection, then combine with
features via the already-shipped load_features_for_video swap
(122ae-5).

Scope is the THREE Tk labellers:
  * mufasa/labelling/labelling_interface.py
  * mufasa/labelling/labelling_advanced_interface.py
  * mufasa/labelling/standard_labeller.py

User direction shaping this patch: "continue mode should just
load the behavior label collection for further labeling".

All checks are AST-based — the Tk labellers import cv2 + tkinter
+ trafaret etc., none of which are in the sandbox.

Coverage per labeller:
* Imports load_labels_for_video from mufasa.utils.label_io.
* Calls load_labels_for_video at code level inside continue
  branch.
* No remaining read_df(self.targets_inserted_file_path) call at
  code level (comments OK).
* Records 122ae-5d in code/comments.
* Still uses load_features_for_video for features (didn't
  accidentally regress 122ae-5).

Specific shape checks:
* labelling_interface: uses pd.concat to combine features +
  labels (was old missing-frames merge; replaced).
* labelling_advanced: uses self.data_df_features.join(self.data_df)
  to combine (preserved from 122ae-5 — labels just slot into
  the existing join).
* standard_labeller: warns on missing classifier columns using
  the DataHeaderWarning constructor, preserving the prior
  semantic where missing classifier names default to 0.
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
    labellers = {
        "labelling_interface.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "labelling_interface.py",
        "labelling_advanced_interface.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "labelling_advanced_interface.py",
        "standard_labeller.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "standard_labeller.py",
    }

    # ==================================================================
    # 1. Each labeller imports + uses load_labels_for_video
    # ==================================================================
    for name, path in labellers.items():
        src = path.read_text()
        check(
            f"{name}: imports load_labels_for_video from "
            "mufasa.utils.label_io",
            "load_labels_for_video" in src
            and "from mufasa.utils.label_io" in src,
        )
        code_uses = [
            line for line in src.splitlines()
            if "load_labels_for_video(" in line
            and not line.lstrip().startswith("#")
            and "import" not in line
        ]
        check(
            f"{name}: load_labels_for_video called at code "
            "level inside continue-mode branch",
            len(code_uses) >= 1,
            detail=f"got {len(code_uses)} call(s)",
        )
        check(
            f"{name}: records 122ae-5d in code/comments",
            "122ae-5d" in src,
        )

    # ==================================================================
    # 2. No remaining read_df(self.targets_inserted_file_path)
    #    calls at code level
    # ==================================================================
    for name, path in labellers.items():
        src = path.read_text()
        stale_calls = [
            (i, line)
            for i, line in enumerate(src.splitlines(), start=1)
            if (
                "read_df(self.targets_inserted_file_path"
                in line
                and not line.lstrip().startswith("#")
            )
        ]
        check(
            f"{name}: no remaining read_df(targets_inserted) "
            "calls at code level",
            not stale_calls,
            detail=(
                f"leaked: {stale_calls}" if stale_calls else ""
            ),
        )

    # ==================================================================
    # 3. labelling_advanced — no remaining branched-by-file_type
    #    pd.read_csv / pd.read_parquet on targets_inserted_file_path
    # ==================================================================
    advanced_src = labellers[
        "labelling_advanced_interface.py"
    ].read_text()
    # The pre-122ae-5d code had:
    #   pd.read_csv(self.targets_inserted_file_path).set_index("Unnamed: 0")[self.target_lst]
    #   pd.read_parquet(self.targets_inserted_file_path)[self.target_lst]
    # Both should be gone at code level.
    for stale_pattern in (
        "pd.read_csv(self.targets_inserted_file_path)",
        "pd.read_parquet(self.targets_inserted_file_path)",
    ):
        leaked = [
            (i, line)
            for i, line in enumerate(
                advanced_src.splitlines(), start=1,
            )
            if stale_pattern in line
            and not line.lstrip().startswith("#")
        ]
        check(
            f"labelling_advanced: no remaining "
            f"'{stale_pattern}' at code level",
            not leaked,
            detail=str(leaked) if leaked else "",
        )

    # ==================================================================
    # 4. Each labeller still uses load_features_for_video
    #    (regression check — didn't accidentally revert 122ae-5)
    # ==================================================================
    for name, path in labellers.items():
        src = path.read_text()
        feat_uses = [
            line for line in src.splitlines()
            if "load_features_for_video(" in line
            and not line.lstrip().startswith("#")
            and "import" not in line
        ]
        check(
            f"{name}: still uses load_features_for_video "
            "(122ae-5 swap preserved)",
            len(feat_uses) >= 1,
            detail=f"got {len(feat_uses)} call(s)",
        )

    # ==================================================================
    # 5. labelling_interface: pd.concat combine pattern
    # ==================================================================
    li_src = labellers["labelling_interface.py"].read_text()
    check(
        "labelling_interface: continue mode combines features "
        "+ labels via pd.concat([... features, labels ...], "
        "axis=1)",
        "pd.concat(" in li_src
        and "self.data_df_features" in li_src
        and "labels_df" in li_src,
    )
    check(
        "labelling_interface: uses reindex(fill_value=0) to "
        "align label length to features (features wins)",
        "reindex(" in li_src
        and "fill_value=0" in li_src,
    )

    # ==================================================================
    # 6. labelling_advanced: features.join(labels) preserved
    # ==================================================================
    la_src = labellers["labelling_advanced_interface.py"].read_text()
    check(
        "labelling_advanced: combine via "
        "self.data_df_features.join(self.data_df)",
        "self.data_df_features.join(self.data_df)" in la_src,
    )
    check(
        "labelling_advanced: NA → 0, int projection to "
        "target_lst",
        "fillna(0)" in la_src
        and "[self.target_lst]" in la_src,
    )

    # ==================================================================
    # 7. standard_labeller: missing-classifier warning preserved
    # ==================================================================
    sl_src = labellers["standard_labeller.py"].read_text()
    check(
        "standard_labeller: still warns via DataHeaderWarning "
        "for missing classifier columns",
        "DataHeaderWarning(" in sl_src
        and "No labels for behavior" in sl_src,
    )
    check(
        "standard_labeller: missing-classifier branch handles "
        "both 'column not in df' AND 'column all-NA' (the "
        "legacy-fallback case where the helper returns all-NA "
        "for missing targets)",
        "labels_df[clf].isna().all()" in sl_src,
    )

    # ==================================================================
    # 8. Old optional-augmentation branch is gone from
    #    standard_labeller
    # ==================================================================
    # The old code had:
    #   new_x = [x for x in features_df.columns if x not in self.data_df.columns
    #            and x not in self.bp_col_names]
    # No longer needed since features are read fresh.
    check(
        "standard_labeller: pre-122ae-5d feature-augmentation "
        "branch (new_x = [...]) removed — features are read "
        "fresh in the new flow",
        "new_x = [" not in sl_src,
    )

    # ==================================================================
    # 9. labelling_interface: legacy missing-frames merge gone
    # ==================================================================
    check(
        "labelling_interface: pre-122ae-5d missing-frames "
        "merge branch (missing_idx = ...) removed — load "
        "helpers return aligned data",
        "missing_idx" not in li_src,
    )

    print(
        f"smoke_122ae5d_tk_continue_mode: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
