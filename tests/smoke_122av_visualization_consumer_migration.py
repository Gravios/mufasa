"""
tests/smoke_122av_visualization_consumer_migration.py
======================================================

Patch 122av: migrate 6 visualization consumers to v1-aware reads
via :func:`mufasa.utils.classification_io.load_machine_results_for_video`.
Third patch in the machine_results migration arc, following 122at
(open dual-write era) and 122au (analysis consumers).

Consumers migrated in this patch
--------------------------------
1. plotting/clf_validator.py — validation clip generator
2. plotting/gantt_creator.py — gantt overlay video
   (.reset_index(drop=True) chain preserved)
3. plotting/heat_mapper_clf.py — classifier-driven heatmap
4. plotting/path_plotter.py — pose path with optional clf
   overlay
5. plotting/plot_clf_results.py — full annotated video
   (.reset_index(drop=True).fillna(0) chain preserved — the
   rendering loop downstream needs a contiguous integer index
   and NaN-free predictions)
6. plotting/probability_plot_creator.py — per-classifier
   probability time-series

What's NOT migrated
-------------------
* distance_plotter.py — reads pose data only (assigns
  ``data_df.columns = self.bp_headers`` directly, so it expects
  pose-shaped data not the combined features+predictions
  shape). Not a machine_results consumer.
* heat_mapper_location.py — same: location heatmap from pose,
  not classifier predictions.
* The 6 ``*_mp.py`` variants (clf_validator_mp, gantt_creator_mp,
  heat_mapper_clf_mp, path_plotter_mp, plot_clf_results_mp,
  probability_plot_creator_mp) — multiprocessing workers that
  need ``config_path`` threaded into the pickled worker args.
  Same deferral as the analysis MP workers in 122au.

Coverage
--------
1. Helper round-trip (sanity, since the helper is exercised in
   anger across consumer code now).
2. Each of the 6 migrated modules:
   * Imports the helper at code level.
   * Records 122av.
   * No remaining raw ``read_df(file_path / data_path / self.data_path, ...)``
     calls.
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
        name = "smoke_122av"
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
    # 1. Helper still works end-to-end (sanity)
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
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack"])

        from mufasa.utils.feature_io import write_wide_features_v1
        features_df = pd.DataFrame({
            "Nose_x": [10.0, 11.0, 12.0],
            "Nose_y": [20.0, 21.0, 22.0],
        })
        write_wide_features_v1(
            df=features_df, video_name="video_001",
            config_path=str(toml),
        )
        save_classifications_for_video(
            video_name="video_001",
            config_path=str(toml),
            predictions=pd.DataFrame({
                "Probability_attack": [0.1, 0.8, 0.4],
                "attack":             [0, 1, 0],
            }),
        )
        combined = load_machine_results_for_video(
            video_name="video_001",
            config_path=str(toml),
            legacy_fallback=None,
        )
        check(
            "Helper v1 path returns combined features + predictions",
            ("Nose_x" in combined.columns and
             "Probability_attack" in combined.columns and
             "attack" in combined.columns),
        )

        # Legacy fallback still works
        legacy_csv = tmp / "video_002.csv"
        pd.DataFrame({
            "Nose_x": [1, 2], "attack": [0, 1],
            "Probability_attack": [0.2, 0.7],
        }).to_csv(legacy_csv, index=False)
        legacy_combined = load_machine_results_for_video(
            video_name="video_002",
            config_path=str(toml),
            legacy_fallback=str(legacy_csv),
        )
        check(
            "Helper legacy fallback still works",
            len(legacy_combined) == 2
            and "Nose_x" in legacy_combined.columns,
        )

    # ==================================================================
    # 2. Each migrated module imports the helper + records 122av
    # ==================================================================
    migrated_files = [
        REPO_ROOT / "mufasa" / "plotting" / "clf_validator.py",
        REPO_ROOT / "mufasa" / "plotting" / "gantt_creator.py",
        REPO_ROOT / "mufasa" / "plotting" / "heat_mapper_clf.py",
        REPO_ROOT / "mufasa" / "plotting" / "path_plotter.py",
        REPO_ROOT / "mufasa" / "plotting" / "plot_clf_results.py",
        REPO_ROOT / "mufasa" / "plotting"
        / "probability_plot_creator.py",
    ]
    for path in migrated_files:
        src = path.read_text()
        check(
            f"{path.name}: imports load_machine_results_for_video",
            "load_machine_results_for_video" in src,
        )
        check(
            f"{path.name}: records 122av",
            "122av" in src,
        )

    # ==================================================================
    # 3. No remaining raw read_df(file_path / data_path / self.data_path)
    #    calls for the migrated read sites. Other read_df calls (for
    #    e.g. video_info.csv) are allowed — we filter by checking the
    #    var name being passed.
    # ==================================================================
    for path in migrated_files:
        src = path.read_text()
        offending = []
        for lineno, line in enumerate(src.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            compact = stripped.replace(" ", "")
            if (
                "read_df(file_path," in compact
                or "read_df(file_path=file_path" in stripped
                or "read_df(data_path," in compact
                or "read_df(self.data_path" in stripped
                or "read_df(file_path,self." in compact
            ):
                offending.append((lineno, line.strip()))
        check(
            f"{path.name}: no remaining raw read_df calls for the "
            "migrated read site",
            len(offending) == 0,
            detail=(
                "; ".join(f"L{n}: {l}" for n, l in offending)
                if offending else ""
            ),
        )

    # ==================================================================
    # 4. Preserved chains — gantt_creator keeps .reset_index(drop=True);
    #    plot_clf_results keeps .reset_index(drop=True).fillna(0).
    # ==================================================================
    gantt_src = (REPO_ROOT / "mufasa" / "plotting"
                 / "gantt_creator.py").read_text()
    check(
        "gantt_creator preserves .reset_index(drop=True) chain "
        "after the migrated read",
        ".reset_index(drop=True)" in gantt_src
        and "load_machine_results_for_video" in gantt_src,
    )
    pcr_src = (REPO_ROOT / "mufasa" / "plotting"
               / "plot_clf_results.py").read_text()
    check(
        "plot_clf_results preserves .reset_index(drop=True)."
        "fillna(0) chain after the migrated read",
        ".reset_index(drop=True).fillna(0)" in pcr_src
        and "load_machine_results_for_video" in pcr_src,
    )

    print(
        f"smoke_122av_visualization_consumer_migration: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
