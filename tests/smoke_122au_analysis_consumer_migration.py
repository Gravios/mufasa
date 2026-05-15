"""
tests/smoke_122au_analysis_consumer_migration.py
=================================================

Patch 122au: migrate 9 analysis consumers to v1-aware reads
via :func:`mufasa.utils.classification_io.load_machine_results_for_video`.
The helper is a dual-read shim:

* **v1 path**: load features + load classifications + concat → the
  combined-shape DataFrame consumers historically expected.

* **Legacy fallback**: if either of the v1 reads raises
  :exc:`FileNotFoundError` (typical when the project hasn't run
  inference under 122at), read the legacy combined CSV at
  ``legacy_fallback``.

Consumers migrated in this patch:

1. data_processors/agg_clf_calculator.py
2. data_processors/timebins_clf_calculator.py
3. data_processors/fsttc_calculator.py
4. data_processors/mutual_exclusivity_corrector.py
5. data_processors/severity_calculator.py
6. data_processors/severity_bout_based_calculator.py (2 read sites)
7. data_processors/severity_frame_based_calculator.py (2 read sites)
8. roi_tools/roi_clf_calculator.py

(9 modules, 11 read sites total. Single-core in-process flows
only. The multiprocess workers in agg_clf_counter_mp,
roi_clf_calculator_mp, etc. stay on the legacy read path for
now — porting them needs config_path threaded into the pickled
worker args, which is a separate concern.)

Coverage
--------
1. ``load_machine_results_for_video`` exists and works end-to-end:
   * v1 path: features + predictions present → returns combined
     DataFrame.
   * Fallback path: v1 predictions missing + legacy CSV present →
     returns legacy CSV.
   * Both missing → FileNotFoundError.
2. Each of the 9 migrated modules:
   * Imports the helper at code level.
   * Read site no longer does the raw ``read_df(file_path,
     self.file_type)`` for machine_results.
   * Records 122au in the file.
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
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


def _write_v1_toml(tmp: Path, classifiers: list[str]) -> Path:
    toml = tmp / "project.toml"
    target_lines = "\n".join(f'    "{c}",' for c in classifiers)
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122au"
        version = "0.0.1"

        [pose]
        file_type = "csv"
        animal_count = 1
        body_parts = ["nose"]

        [classifiers]
        targets = [
        {target_lines}
        ]
    """).strip() + "\n")
    return toml


def main() -> int:
    # ==================================================================
    # 1. Helper exists + importable
    # ==================================================================
    try:
        from mufasa.utils.classification_io import (
            load_machine_results_for_video,
            save_classifications_for_video,
        )
        check("load_machine_results_for_video importable", True)
    except ImportError as exc:
        check("load_machine_results_for_video importable",
              False, detail=str(exc))
        return 1

    import pandas as pd

    # ==================================================================
    # 2. v1 path: features + predictions → combined
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack"])

        # Write a feature parquet to derived/features/
        from mufasa.utils.feature_io import write_wide_features_v1
        features_df = pd.DataFrame({
            "Nose_x": [10.0, 11.0, 12.0],
            "Nose_y": [20.0, 21.0, 22.0],
        })
        write_wide_features_v1(
            df=features_df, video_name="video_001",
            config_path=str(toml),
        )

        # Write a predictions parquet to derived/classifications/
        predictions_df = pd.DataFrame({
            "Probability_attack": [0.1, 0.8, 0.4],
            "attack":             [0, 1, 0],
        })
        save_classifications_for_video(
            video_name="video_001",
            config_path=str(toml),
            predictions=predictions_df,
        )

        combined = load_machine_results_for_video(
            video_name="video_001",
            config_path=str(toml),
            legacy_fallback=None,
        )
        check(
            "v1 path: combined DataFrame has feature columns",
            "Nose_x" in combined.columns
            and "Nose_y" in combined.columns,
        )
        check(
            "v1 path: combined DataFrame has prediction columns",
            "Probability_attack" in combined.columns
            and "attack" in combined.columns,
        )
        check(
            "v1 path: row count matches input (3 frames)",
            len(combined) == 3,
        )
        check(
            "v1 path: feature values preserved",
            list(combined["Nose_x"]) == [10.0, 11.0, 12.0],
        )
        check(
            "v1 path: prediction values preserved",
            list(combined["attack"]) == [0, 1, 0],
        )

    # ==================================================================
    # 3. Legacy fallback: v1 predictions missing → read legacy CSV
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack"])
        # No v1 features or predictions written
        legacy_csv = tmp / "video_001.csv"
        legacy_df = pd.DataFrame({
            "Nose_x":             [10.0, 11.0, 12.0],
            "Probability_attack": [0.1, 0.8, 0.4],
            "attack":             [0, 1, 0],
        })
        legacy_df.to_csv(legacy_csv, index=False)

        combined = load_machine_results_for_video(
            video_name="video_001",
            config_path=str(toml),
            legacy_fallback=str(legacy_csv),
        )
        check(
            "Legacy fallback: returns the legacy CSV's columns",
            "Nose_x" in combined.columns
            and "attack" in combined.columns,
        )
        check(
            "Legacy fallback: row count matches CSV",
            len(combined) == 3,
        )

    # ==================================================================
    # 4. Both missing → FileNotFoundError
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack"])
        try:
            load_machine_results_for_video(
                video_name="never_inferenced",
                config_path=str(toml),
                legacy_fallback=None,
            )
            raised = False
        except FileNotFoundError:
            raised = True
        check(
            "Both missing → FileNotFoundError",
            raised,
        )

    # ==================================================================
    # 5. Consumer migrations — each module references the helper
    #    and dropped the raw read_df call
    # ==================================================================
    migrated_files = [
        REPO_ROOT / "mufasa" / "data_processors"
        / "agg_clf_calculator.py",
        REPO_ROOT / "mufasa" / "data_processors"
        / "timebins_clf_calculator.py",
        REPO_ROOT / "mufasa" / "data_processors"
        / "fsttc_calculator.py",
        REPO_ROOT / "mufasa" / "data_processors"
        / "mutual_exclusivity_corrector.py",
        REPO_ROOT / "mufasa" / "data_processors"
        / "severity_calculator.py",
        REPO_ROOT / "mufasa" / "data_processors"
        / "severity_bout_based_calculator.py",
        REPO_ROOT / "mufasa" / "data_processors"
        / "severity_frame_based_calculator.py",
        REPO_ROOT / "mufasa" / "roi_tools"
        / "roi_clf_calculator.py",
    ]
    for path in migrated_files:
        src = path.read_text()
        check(
            f"{path.name}: imports load_machine_results_for_video",
            "load_machine_results_for_video" in src,
        )
        check(
            f"{path.name}: records 122au",
            "122au" in src,
        )

    # ==================================================================
    # 6. No more raw `read_df(file_path, self.file_type)` calls in
    #    the migrated files (the analysis-side reads).
    #    Some files may still have read_df for OTHER files (e.g.
    #    video_info.csv, intermediate writes) — we filter by checking
    #    that there are no remaining reads of `file_path` or
    #    `data_path` (the iteration variables) via read_df.
    # ==================================================================
    for path in migrated_files:
        src = path.read_text()
        offending = []
        for lineno, line in enumerate(src.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if (
                "read_df(file_path" in stripped.replace(" ", "")
                or "read_df(data_path" in stripped.replace(" ", "")
                or "read_df(file_path=file_path" in stripped
                or "read_df(file_path=data_path" in stripped
            ):
                offending.append((lineno, line.strip()))
        check(
            f"{path.name}: no remaining read_df(file_path / "
            "data_path, ...) calls in code",
            len(offending) == 0,
            detail=(
                "; ".join(f"L{n}: {l}" for n, l in offending)
                if offending else ""
            ),
        )

    print(
        f"smoke_122au_analysis_consumer_migration: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
