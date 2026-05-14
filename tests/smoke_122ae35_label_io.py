"""
tests/smoke_122ae35_label_io.py
================================

Patch 122ae-3.5: behavioural verification of
:mod:`mufasa.utils.label_io` — the labels split helpers that
consumers (frame labeller, classifier training) will switch
to in patch 122ae-5.

Same coverage shape as smoke_122ae2_feature_io but for labels:
both read branches (v1 per-video parquet, legacy wide CSV
fallback), the write helper's overwrite + merge behaviours,
edge cases.

Sections:

1. **v1 per-video parquet branch** — set up a project with a
   ``derived/labels/<video>.parquet`` file; verify
   ``load_labels_for_video`` returns its contents projected
   to the requested classifier targets.

2. **Legacy CSV fallback** — set up a project with no
   ``derived/labels/`` tree but a populated
   ``csv/targets_inserted/<video>.csv`` (the legacy wide file
   with feature + label columns). Verify the helper extracts
   only the classifier-target columns and discards the rest.

3. **Precedence** — both v1 + legacy present → v1 wins.

4. **Missing on both** — clean FileNotFoundError.

5. **Targets filter** — passing an explicit ``targets`` list
   projects the output to just those columns.

6. **Missing-target stability** — requesting a target the file
   doesn't have yields an all-NA Int64 column rather than
   dropping the column.

7. **Save: fresh file** — call ``save_labels_for_video`` with
   no existing file; verify the parquet lands at the expected
   path with Int64 dtype.

8. **Save: merge mode (default)** — call save twice: first
   with target A only, then with target B only. Verify the
   resulting file has BOTH columns (the merge preserves
   target A across the second write).

9. **Save: overwrite mode** — call save with merge=False and
   verify the existing file is replaced.

10. **Save: dtype coercion** — input DataFrame with float
    NaN-bearing columns gets coerced to Int64 nullable on
    disk so 0 / 1 / NaN round-trip exactly.

11. **AST surface** — public API (load + save), uses
    project_paths_from_config + project_metadata_from_config,
    references the 122ae-3.5 patch.
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


def _make_v1_project(tmp: Path,
                     classifier_targets: list[str],
                     file_type: str = "csv") -> Path:
    """Make a v1 project_root with a project.toml; return the
    toml path. Classifier targets are injected so
    project_metadata_from_config sees them.
    """
    toml = tmp / "project.toml"
    targets_str = ", ".join(f'"{t}"' for t in classifier_targets)
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122ae35"
        version = "0.0.1"

        [pose]
        file_type = "{file_type}"
        animal_count = 1
        body_parts = ["nose", "tail"]

        [classifiers]
        targets = [{targets_str}]
    """).strip() + "\n")
    return toml


def main() -> int:
    from mufasa.utils.label_io import (load_labels_for_video,
                                       save_labels_for_video)

    # ==================================================================
    # 1. v1 per-video parquet branch
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(
            tmp, classifier_targets=["sniff", "rear"],
        )
        labels_dir = tmp / "derived" / "labels"
        labels_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "sniff": pd.array([0, 1, 1, 0], dtype="Int64"),
            "rear":  pd.array([0, 0, 1, 1], dtype="Int64"),
        })
        df.to_parquet(labels_dir / "video_001.parquet", index=False)

        result = load_labels_for_video("video_001", str(toml))
        check(
            "v1 parquet branch: returns DataFrame",
            isinstance(result, pd.DataFrame),
        )
        check(
            "v1 parquet branch: both classifier columns present",
            set(result.columns) == {"sniff", "rear"},
            detail=f"got {list(result.columns)}",
        )
        check(
            "v1 parquet branch: 4 rows preserved",
            len(result) == 4,
        )
        check(
            "v1 parquet branch: Int64 dtype preserved on read",
            result["sniff"].dtype.name == "Int64"
            and result["rear"].dtype.name == "Int64",
        )
        check(
            "v1 parquet branch: values round-tripped",
            result["sniff"].tolist() == [0, 1, 1, 0]
            and result["rear"].tolist() == [0, 0, 1, 1],
        )

    # ==================================================================
    # 2. Legacy CSV fallback
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(
            tmp, classifier_targets=["sniff", "rear"],
            file_type="csv",
        )
        legacy = tmp / "csv" / "targets_inserted"
        legacy.mkdir(parents=True)
        # Wide CSV: features + pose + label columns mixed
        wide = pd.DataFrame({
            "nose_x":        [10.0, 11.0, 12.0, 13.0],
            "nose_y":        [20.0, 21.0, 22.0, 23.0],
            "feature_alpha": [0.1, 0.2, 0.3, 0.4],
            "sniff":         [1, 0, 1, 0],
            "rear":          [0, 0, 1, 1],
        })
        wide.to_csv(legacy / "video_002.csv", index=False)

        result = load_labels_for_video("video_002", str(toml))
        check(
            "legacy fallback: only classifier columns returned "
            "(features + pose discarded)",
            set(result.columns) == {"sniff", "rear"},
            detail=f"got {list(result.columns)}",
        )
        check(
            "legacy fallback: row count matches input (4)",
            len(result) == 4,
        )
        check(
            "legacy fallback: values projected correctly",
            result["sniff"].tolist() == [1, 0, 1, 0],
        )
        check(
            "legacy fallback: dtype coerced to Int64 on read",
            result["sniff"].dtype.name == "Int64",
        )

    # ==================================================================
    # 3. Precedence — v1 wins when both exist
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, classifier_targets=["sniff"])
        # v1
        labels_dir = tmp / "derived" / "labels"
        labels_dir.mkdir(parents=True)
        pd.DataFrame({"sniff": [9, 9, 9]}).astype("Int64").to_parquet(
            labels_dir / "video_003.parquet", index=False,
        )
        # Legacy
        legacy = tmp / "csv" / "targets_inserted"
        legacy.mkdir(parents=True)
        pd.DataFrame({"sniff": [1, 2, 3]}).to_csv(
            legacy / "video_003.csv", index=False,
        )

        result = load_labels_for_video("video_003", str(toml))
        check(
            "precedence: v1 parquet wins over legacy when both "
            "exist",
            result["sniff"].tolist() == [9, 9, 9],
        )

    # ==================================================================
    # 4. Missing on both — FileNotFoundError
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, classifier_targets=["sniff"])
        raised = False
        msg = ""
        try:
            load_labels_for_video("video_missing", str(toml))
        except FileNotFoundError as exc:
            raised = True
            msg = str(exc)
        check("missing-on-both: raises FileNotFoundError", raised)
        check(
            "missing-on-both: error mentions both probed paths",
            "derived" in msg and "labels" in msg
            and "targets_inserted" in msg,
            detail=f"got {msg!r}",
        )

    # ==================================================================
    # 5. Targets filter
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(
            tmp, classifier_targets=["sniff", "rear", "groom"],
        )
        labels_dir = tmp / "derived" / "labels"
        labels_dir.mkdir(parents=True)
        pd.DataFrame({
            "sniff": [1, 0], "rear": [0, 1], "groom": [1, 1],
        }).astype("Int64").to_parquet(
            labels_dir / "video_004.parquet", index=False,
        )

        # Default: all 3
        all_targets = load_labels_for_video(
            "video_004", str(toml),
        )
        check(
            "targets=None: all 3 classifier targets returned",
            set(all_targets.columns) == {"sniff", "rear", "groom"},
        )

        # Filtered: just one
        just_one = load_labels_for_video(
            "video_004", str(toml), targets=["rear"],
        )
        check(
            "targets filter: only requested column returned",
            set(just_one.columns) == {"rear"},
        )

    # ==================================================================
    # 6. Missing-target stability — emit all-NA Int64 column
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        # Project knows about sniff + rear; file only has sniff.
        toml = _make_v1_project(
            tmp, classifier_targets=["sniff", "rear"],
        )
        labels_dir = tmp / "derived" / "labels"
        labels_dir.mkdir(parents=True)
        pd.DataFrame({"sniff": [1, 0, 1]}).astype("Int64").to_parquet(
            labels_dir / "video_005.parquet", index=False,
        )

        result = load_labels_for_video("video_005", str(toml))
        check(
            "missing-target: 'rear' column emitted (not dropped) "
            "even though file lacks it",
            "rear" in result.columns,
        )
        check(
            "missing-target: 'rear' values are all pd.NA",
            result["rear"].isna().all(),
        )
        check(
            "missing-target: dtype still Int64",
            result["rear"].dtype.name == "Int64",
        )

    # ==================================================================
    # 7. Save: fresh file
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, classifier_targets=["sniff"])
        new_labels = pd.DataFrame({"sniff": [0, 1, 1, 0]})
        path_returned = save_labels_for_video(
            "video_006", str(toml), new_labels,
        )
        out_path = tmp / "derived" / "labels" / "video_006.parquet"
        check(
            "save fresh: returns the path it wrote to",
            Path(path_returned) == out_path,
        )
        check(
            "save fresh: parquet file exists on disk",
            out_path.is_file(),
        )
        on_disk = pd.read_parquet(out_path)
        check(
            "save fresh: column round-tripped",
            "sniff" in on_disk.columns,
        )
        check(
            "save fresh: dtype is Int64 (nullable)",
            on_disk["sniff"].dtype.name == "Int64",
        )

    # ==================================================================
    # 8. Save: merge mode (default) — two writes, both columns survive
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(
            tmp, classifier_targets=["sniff", "rear"],
        )
        # First write: just sniff
        save_labels_for_video(
            "video_007", str(toml),
            pd.DataFrame({"sniff": [1, 0, 1]}),
        )
        # Second write: just rear
        save_labels_for_video(
            "video_007", str(toml),
            pd.DataFrame({"rear": [0, 1, 1]}),
        )
        out_path = tmp / "derived" / "labels" / "video_007.parquet"
        on_disk = pd.read_parquet(out_path)
        check(
            "merge: BOTH columns survive after two single-"
            "column writes",
            set(on_disk.columns) == {"sniff", "rear"},
            detail=f"got {list(on_disk.columns)}",
        )
        check(
            "merge: sniff values preserved from first write",
            on_disk["sniff"].tolist() == [1, 0, 1],
        )
        check(
            "merge: rear values from second write",
            on_disk["rear"].tolist() == [0, 1, 1],
        )

    # ==================================================================
    # 9. Save: overwrite mode (merge=False)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(
            tmp, classifier_targets=["sniff", "rear"],
        )
        save_labels_for_video(
            "video_008", str(toml),
            pd.DataFrame({"sniff": [1, 1, 1], "rear": [0, 0, 0]}),
        )
        # Overwrite with merge=False — should DROP sniff
        save_labels_for_video(
            "video_008", str(toml),
            pd.DataFrame({"rear": [1, 1, 1]}),
            merge=False,
        )
        out_path = tmp / "derived" / "labels" / "video_008.parquet"
        on_disk = pd.read_parquet(out_path)
        check(
            "overwrite: only the second write's columns survive",
            list(on_disk.columns) == ["rear"],
            detail=f"got {list(on_disk.columns)}",
        )

    # ==================================================================
    # 10. Save: float-NaN input coerces to Int64 nullable
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, classifier_targets=["sniff"])
        # Input is float with NaN — common shape coming out of
        # the labeller's intermediate state.
        import numpy as np
        save_labels_for_video(
            "video_009", str(toml),
            pd.DataFrame({"sniff": [1.0, np.nan, 0.0, 1.0]}),
        )
        out_path = tmp / "derived" / "labels" / "video_009.parquet"
        on_disk = pd.read_parquet(out_path)
        check(
            "dtype coercion: float+NaN input lands as Int64 "
            "nullable on disk",
            on_disk["sniff"].dtype.name == "Int64",
        )
        check(
            "dtype coercion: NaN became pd.NA",
            on_disk["sniff"].isna().tolist()
            == [False, True, False, False],
        )

    # ==================================================================
    # 11. AST surface
    # ==================================================================
    src = (REPO_ROOT / "mufasa" / "utils"
           / "label_io.py").read_text()
    tree = ast.parse(src)
    top_names = [
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    check(
        "label_io exports load_labels_for_video",
        "load_labels_for_video" in top_names,
    )
    check(
        "label_io exports save_labels_for_video",
        "save_labels_for_video" in top_names,
    )
    check(
        "label_io imports project_paths_from_config",
        "project_paths_from_config" in src,
    )
    check(
        "label_io imports project_metadata_from_config",
        "project_metadata_from_config" in src,
    )
    check(
        "label_io reads 'derived_labels_dir' from helper",
        "derived_labels_dir" in src,
    )
    check(
        "label_io reads 'targets_inserted_dir' for legacy "
        "fallback",
        "targets_inserted_dir" in src,
    )
    check(
        "label_io docstring records the 122ae-3.5 patch number",
        "122ae-3.5" in src,
    )

    print(
        f"smoke_122ae35_label_io: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
