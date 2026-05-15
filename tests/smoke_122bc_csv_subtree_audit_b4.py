"""
tests/smoke_122bc_csv_subtree_audit_b4.py
==========================================

Patch 122bc: closes out gaps surfaced by a fresh csv-subtree
audit (B4) after 122ax (machine_results migration core arc)
and 122bb (features migration close-out) had landed.

Audit found two ACTIVE code paths missed by 122ax
--------------------------------------------------
1. ``model/inference_multiclass_batch.py``
       Still wrote legacy CSV to ``self.machine_results_dir``
       via ``write_df(out_df, self.file_type, file_save_path)``.
       The single-classifier inference (InferenceBatch) was
       migrated in 122ax; the multi-class variant was not.

2. ``data_processors/mutual_exclusivity_corrector.py``
       Read-side migrated in 122au via
       ``load_machine_results_for_video``. Write-back side
       still did ``write_df(self.data_df, self.file_type,
       save_path=file_path)`` — the backup-and-rewrite-in-place
       flow. Backup move via shutil.move is layout-neutral
       (works against any path). The rewrite needs the v1
       helper.

Plus visible text drift in the Qt workbench and tooltips:
* ``ui_qt/forms/run_inference.py`` — module docstring,
  description HTML, and preview text all mentioned
  ``csv/machine_results/``.
* ``ui_qt/forms/annotation.py`` — module docstring,
  description HTML, and the per-bout reviewer hint all
  mentioned legacy paths.
* ``assets/lookups/tooptips.json`` — 2 tooltip strings
  referenced ``project_folder/csv/machine_results``.

Migration shape
---------------
Both code-path fixes follow the established 122ax pattern:
* Drop the legacy ``write_df(...)`` call.
* Use ``save_classifications_for_video(...)`` from
  ``classification_io`` to write the v1 parquet.
* Filter to the prediction columns via ``_prediction_columns``
  so the v1 file only contains ``Probability_<clf>`` / ``<clf>``
  columns (matches the contract for
  ``derived/classifications/<video>.parquet`` post-122ax).
* Drop ``write_df`` from imports.
* Update stdout_success messages to point at
  ``derived/classifications/``.

Coverage
--------
1. inference_multiclass_batch:
    - No remaining write_df references.
    - Imports save_classifications_for_video.
    - Records 122bc.
    - stdout_success now mentions derived/classifications/.
2. mutual_exclusivity_corrector:
    - No remaining write_df references.
    - Imports save_classifications_for_video.
    - Records 122bc.
    - shutil.move backup-to-prior-dir behaviour preserved.
    - stdout_success now mentions derived/classifications/.
3. Visible text — no remaining csv/machine_results refs in:
    - ui_qt/forms/run_inference.py active strings
    - ui_qt/forms/annotation.py active strings
    - assets/lookups/tooptips.json
"""
from __future__ import annotations

import json
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
    # ==================================================================
    # 1. inference_multiclass_batch.py
    # ==================================================================
    imb_path = (REPO_ROOT / "mufasa" / "model"
                / "inference_multiclass_batch.py")
    imb_src = imb_path.read_text()
    check(
        "inference_multiclass_batch: no write_df references",
        "write_df" not in imb_src,
    )
    check(
        "inference_multiclass_batch: imports "
        "save_classifications_for_video",
        "save_classifications_for_video" in imb_src,
    )
    check(
        "inference_multiclass_batch: imports _prediction_columns",
        "_prediction_columns" in imb_src,
    )
    check(
        "inference_multiclass_batch: records 122bc",
        "122bc" in imb_src,
    )
    check(
        "inference_multiclass_batch: stdout_success now mentions "
        "derived/classifications/",
        "project_folder/derived/classifications/" in imb_src,
    )
    check(
        "inference_multiclass_batch: no more "
        "project_folder/csv/machine_results in stdout",
        "saved in project_folder/csv/machine_results" not in imb_src,
    )

    # ==================================================================
    # 2. mutual_exclusivity_corrector.py
    # ==================================================================
    mec_path = (REPO_ROOT / "mufasa" / "data_processors"
                / "mutual_exclusivity_corrector.py")
    mec_src = mec_path.read_text()
    check(
        "mutual_exclusivity_corrector: no write_df references",
        "write_df" not in mec_src,
    )
    check(
        "mutual_exclusivity_corrector: imports "
        "save_classifications_for_video",
        "save_classifications_for_video" in mec_src,
    )
    check(
        "mutual_exclusivity_corrector: imports _prediction_columns",
        "_prediction_columns" in mec_src,
    )
    check(
        "mutual_exclusivity_corrector: records 122bc",
        "122bc" in mec_src,
    )
    check(
        "mutual_exclusivity_corrector: preserves shutil.move "
        "backup flow",
        "shutil.move" in mec_src
        and "self.save_dir" in mec_src,
    )
    check(
        "mutual_exclusivity_corrector: stdout now mentions "
        "derived/classifications/",
        "project_folder/derived/classifications/" in mec_src,
    )
    check(
        "mutual_exclusivity_corrector: no more "
        "csv/machine_results in stdout msgs",
        "saved in the project_folder/csv/machine_results"
        not in mec_src,
    )

    # ==================================================================
    # 3. ui_qt/forms/run_inference.py — visible text drift
    # ==================================================================
    ri_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "run_inference.py")
    ri_src = ri_path.read_text()
    check(
        "run_inference: no remaining csv/machine_results in "
        "module docstring",
        "csv/machine_results/`` (still legacy" not in ri_src,
    )
    check(
        "run_inference: description HTML points at "
        "derived/classifications/",
        "<code>derived/classifications/</code>" in ri_src,
    )
    check(
        "run_inference: preview HTML points at "
        "derived/classifications/",
        ri_src.count("<code>derived/classifications/</code>") >= 2,
    )
    check(
        "run_inference: description mentions both v1 and legacy "
        "settings persistence",
        "project.toml" in ri_src and "project_config.ini" in ri_src,
    )

    # ==================================================================
    # 4. ui_qt/forms/annotation.py — visible text drift
    # ==================================================================
    an_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "annotation.py")
    an_src = an_path.read_text()
    check(
        "annotation: no csv/machine_results in active strings "
        "(description / hint)",
        # The Path note + reviewer hint should no longer say
        # csv/machine_results; only references retained should
        # be in comments or docstrings clearly framed as
        # post-122ax retrospectives.
        "from <code>csv/machine_results/</code>" not in an_src
        and "from <code>csv/machine_results/</code>). The"
        not in an_src,
    )
    check(
        "annotation: description HTML points at "
        "derived/classifications/",
        "<code>derived/classifications/</code>" in an_src,
    )
    check(
        "annotation: description HTML points at "
        "derived/features/",
        "<code>derived/features/</code>" in an_src,
    )

    # ==================================================================
    # 5. assets/lookups/tooptips.json — tooltip strings
    # ==================================================================
    tip_path = (REPO_ROOT / "mufasa" / "assets" / "lookups"
                / "tooptips.json")
    tip_src = tip_path.read_text()
    check(
        "tooltips: valid JSON",
        bool(json.loads(tip_src)),
    )
    check(
        "tooltips: no remaining project_folder/csv/machine_results",
        "project_folder/csv/machine_results" not in tip_src,
    )
    tip_data = json.loads(tip_src)
    check(
        "tooltips: KLEINBERG_SAVE_ORIGINALS points at "
        "derived/classifications/",
        "derived/classifications"
        in tip_data.get("KLEINBERG_SAVE_ORIGINALS", ""),
    )
    check(
        "tooltips: CLF_PLOT_VIDEO_PATH points at "
        "derived/classifications/",
        "derived/classifications"
        in tip_data.get("CLF_PLOT_VIDEO_PATH", ""),
    )

    print(
        f"smoke_122bc_csv_subtree_audit_b4: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
