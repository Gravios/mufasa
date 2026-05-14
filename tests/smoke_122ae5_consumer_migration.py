"""
tests/smoke_122ae5_consumer_migration.py
=========================================

Patch 122ae-5: consumer migration — switch the three frame
labellers (labelling_interface, labelling_advanced_interface,
standard_labeller) from direct
``read_df(self.features_extracted_file_path, self.file_type)``
calls to the layout-aware
:func:`mufasa.utils.feature_io.load_features_for_video`.

Behavioural verifies that:

* feature_io._read_legacy now strips the leading pad column
  for CSVs (matching read_df's default has_index=True
  semantic), so load_features_for_video is a drop-in
  replacement for read_df at the call sites we swapped.
* label_io._read_legacy got the same alignment for consistency.

AST verifies that:

* All 3 labellers import load_features_for_video.
* All 8 read sites swapped (no remaining
  ``read_df(self.features_extracted_file_path, ...)`` calls
  at code level — comments OK).
* The 3 ``check_file_exist_and_readable`` /
  ``os.path.isfile`` pre-checks that previously fronted the
  reads got either removed (since load_features_for_video
  does its own probing across all layouts) or kept as
  intentional augmentation guards (the standard labeller's
  continuing-mode "if legacy CSV is present, augment with
  new features" branch — preserved on purpose).
* All 3 labellers record 122ae-5 in code/comments.

Scope: only the 3 labellers in this patch. Other consumers
that read csv/features_extracted/ (inference_batch, ROI
analyzer, BENTO appender, directing_other_animals_calculator)
remain on their legacy paths for now — load_features_for_video
falls back to legacy CSV so they keep working unchanged.
Migration of those consumers is follow-up scope.
"""
from __future__ import annotations

import ast
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _make_v1_project(tmp: Path, file_type: str = "csv") -> Path:
    toml = tmp / "project.toml"
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122ae5"
        version = "0.0.1"

        [pose]
        file_type = "{file_type}"
        animal_count = 1
        body_parts = ["nose", "tail"]
    """).strip() + "\n")
    return toml


def main() -> int:
    # ==================================================================
    # 1. Legacy-only seed → FileNotFoundError (patch 122ak removed
    #    the _read_legacy strip-alignment helpers along with the
    #    fallback they served)
    # ==================================================================
    from mufasa.utils.feature_io import load_features_for_video
    from mufasa.utils.label_io import load_labels_for_video

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, file_type="csv")
        legacy_dir = tmp / "csv" / "features_extracted"
        legacy_dir.mkdir(parents=True)
        pd.DataFrame({
            "Unnamed: 0":  [0, 1, 2, 3],
            "feat_a":      [10.0, 20.0, 30.0, 40.0],
            "feat_b":      [100.0, 200.0, 300.0, 400.0],
        }).to_csv(legacy_dir / "v_legacy.csv", index=False)

        raised = False
        try:
            load_features_for_video("v_legacy", str(toml))
        except FileNotFoundError:
            raised = True
        check(
            "patch 122ak: legacy-CSV-only project raises "
            "FileNotFoundError (legacy fallback removed)",
            raised,
        )

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = tmp / "project.toml"
        toml.write_text(textwrap.dedent("""
            project_layout_version = 1

            [project]
            name = "smoke_122ae5_lab"
            version = "0.0.1"

            [pose]
            file_type = "csv"
            animal_count = 1
            body_parts = ["nose"]

            [classifiers]
            targets = ["sniff", "rear"]
        """).strip() + "\n")
        legacy_dir = tmp / "csv" / "targets_inserted"
        legacy_dir.mkdir(parents=True)
        pd.DataFrame({
            "Unnamed: 0":  [0, 1, 2],
            "feat_x":      [1.0, 2.0, 3.0],
            "sniff":       [0, 1, 0],
            "rear":        [1, 1, 0],
        }).to_csv(legacy_dir / "v_lab.csv", index=False)

        raised = False
        try:
            load_labels_for_video("v_lab", str(toml))
        except FileNotFoundError:
            raised = True
        check(
            "patch 122ak: legacy-targets-only project raises "
            "FileNotFoundError (legacy fallback removed)",
            raised,
        )

    # Parquet at the v1 location: still works (regression check).
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, file_type="parquet")
        feat_dir = tmp / "derived" / "features"
        feat_dir.mkdir(parents=True)
        pd.DataFrame({
            "feat_a": [1.0, 2.0],
            "feat_b": [3.0, 4.0],
        }).to_parquet(feat_dir / "v_parq.parquet", index=False)

        result = load_features_for_video("v_parq", str(toml))
        check(
            "v1 wide-parquet read still works after 122ak cleanup",
            set(result.columns) == {"feat_a", "feat_b"},
        )

    # ==================================================================
    # 2. AST — labellers swapped, imports added, no stale call sites
    # ==================================================================
    labellers = {
        "labelling_interface.py":
            REPO_ROOT / "mufasa" / "labelling" / "labelling_interface.py",
        "labelling_advanced_interface.py":
            REPO_ROOT / "mufasa" / "labelling"
            / "labelling_advanced_interface.py",
        "standard_labeller.py":
            REPO_ROOT / "mufasa" / "labelling" / "standard_labeller.py",
    }

    for name, path in labellers.items():
        src = path.read_text()
        check(
            f"{name}: imports load_features_for_video",
            "from mufasa.utils.feature_io import "
            "load_features_for_video" in src,
        )
        check(
            f"{name}: records 122ae-5 in code/comments",
            "122ae-5" in src,
        )
        # No remaining code-level
        # `read_df(self.features_extracted_file_path, ...)` calls
        # — comments OK (they document the swap).
        stale_calls = [
            (i, line)
            for i, line in enumerate(src.splitlines(), start=1)
            if (
                "read_df(self.features_extracted_file_path"
                in line
                and not line.lstrip().startswith("#")
            )
        ]
        check(
            f"{name}: no remaining read_df(features_extracted) "
            "calls at code level",
            not stale_calls,
            detail=(
                f"leaked: {stale_calls}" if stale_calls else ""
            ),
        )
        # load_features_for_video appears at code level (not just
        # in imports/comments).
        code_uses = [
            line for line in src.splitlines()
            if "load_features_for_video(" in line
            and not line.lstrip().startswith("#")
            and "import" not in line
        ]
        check(
            f"{name}: load_features_for_video called at least "
            "once at code level",
            len(code_uses) >= 1,
            detail=f"got {len(code_uses)} calls",
        )

    # ==================================================================
    # 3. The 3 labellers' specific call counts match expected
    # ==================================================================
    # Patch 122ak: labellers' save methods no longer call
    # load_features_for_video (labels-only save). The 3 → 2,
    # 3 → 2, 2 → 2 drops reflect:
    #   * labelling_interface: __save_results dropped its
    #     load_features_for_video call.
    #   * labelling_advanced: save_results dropped its
    #     load_features_for_video call.
    #   * standard_labeller: __save_results never had one;
    #     count unchanged.
    expected_call_counts = {
        "labelling_interface.py":          2,
        "labelling_advanced_interface.py": 2,
        "standard_labeller.py":            2,
    }
    for name, expected in expected_call_counts.items():
        src = labellers[name].read_text()
        count = sum(
            1 for line in src.splitlines()
            if "load_features_for_video(" in line
            and not line.lstrip().startswith("#")
            and "import" not in line
        )
        check(
            f"{name}: load_features_for_video call count "
            f"is {expected} (matches 122ae-3 read-site survey)",
            count == expected,
            detail=f"got {count}",
        )

    # ==================================================================
    # 4. _read_legacy alignment recorded in both modules
    # ==================================================================
    fio_src = (REPO_ROOT / "mufasa" / "utils"
               / "feature_io.py").read_text()
    lio_src = (REPO_ROOT / "mufasa" / "utils"
               / "label_io.py").read_text()
    # Patch 122ak: _read_legacy was removed from both modules.
    # Confirm the 122ae-5 alignment note isn't a stale leftover
    # asserting a function that no longer exists.
    check(
        "feature_io: _read_legacy removed (patch 122ak)",
        "def _read_legacy" not in fio_src,
    )
    check(
        "label_io: _read_legacy removed (patch 122ak)",
        "def _read_legacy" not in lio_src,
    )

    print(
        f"smoke_122ae5_consumer_migration: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
