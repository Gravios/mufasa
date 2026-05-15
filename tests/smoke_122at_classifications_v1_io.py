"""
tests/smoke_122at_classifications_v1_io.py
============================================

Patch 122at: open the v1 classifier-predictions migration arc.

New module ``mufasa.utils.classification_io`` mirrors the labels
and features helpers — providing read/write/list helpers backed
by ``derived/classifications/<video>.parquet``. The layout
helper :func:`project_paths_from_config` gains
``derived_classifications_dir``. :class:`InferenceBatch` opens
the dual-write era: every per-video write now produces a v1
predictions-only parquet sidecar alongside its existing combined
CSV at ``csv/machine_results/``.

Coverage
--------
* Layout helper exposes ``derived_classifications_dir`` for both
  v1 TOML and legacy INI projects.
* classification_io API surface: ``save_*``, ``load_*``,
  ``list_video_stems_with_classifications``.
* Round-trip: write a predictions DataFrame, read it back, get
  the same shape + values.
* targets-filtered read returns Probability_<T> + <T> per target
  in the requested order; missing columns surface as NaN-filled.
* ``list_video_stems_with_classifications`` enumerates parquet
  stems under the v1 directory.
* InferenceBatch references the v1 helper at code level (the
  sidecar write block is in place).
* Sidecar write block is wrapped in try/except — never blocks
  the legacy CSV write path.
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
        name = "smoke_122at"
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


def _write_legacy_ini(tmp: Path) -> Path:
    proj = tmp / "project_folder"
    proj.mkdir()
    ini = tmp / "project_config.ini"
    lines = [
        "[General settings]",
        f"project_path = {proj}",
        "workflow_file_type = csv",
    ]
    ini.write_text("\n".join(lines) + "\n")
    return ini


def main() -> int:
    # ==================================================================
    # 1. Layout helper exposes derived_classifications_dir
    # ==================================================================
    from mufasa.project_layout import project_paths_from_config

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack"])
        paths = project_paths_from_config(str(toml))
        check(
            "v1 layout: derived_classifications_dir present",
            "derived_classifications_dir" in paths,
        )
        expected = str(toml.parent / "derived" / "classifications")
        check(
            "v1 layout: derived_classifications_dir = "
            "<root>/derived/classifications",
            paths.get("derived_classifications_dir") == expected,
        )

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        ini = _write_legacy_ini(tmp)
        paths = project_paths_from_config(str(ini))
        check(
            "legacy layout: derived_classifications_dir present",
            "derived_classifications_dir" in paths,
        )
        expected = str(tmp / "project_folder" / "derived"
                       / "classifications")
        check(
            "legacy layout: derived_classifications_dir = "
            "<project_path>/derived/classifications",
            paths.get("derived_classifications_dir") == expected,
        )

    # ==================================================================
    # 2. classification_io API surface
    # ==================================================================
    try:
        from mufasa.utils.classification_io import (
            load_classifications_for_video,
            save_classifications_for_video,
            list_video_stems_with_classifications,
        )
        check("classification_io: 3 helpers importable", True)
    except ImportError as exc:
        check("classification_io: 3 helpers importable",
              False, detail=str(exc))
        return 1

    # ==================================================================
    # 3. Round-trip — TOML project
    # ==================================================================
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack", "grooming"])
        df = pd.DataFrame({
            "Probability_attack":   [0.1, 0.6, 0.9, 0.2, 0.5],
            "attack":               [0, 1, 1, 0, 0],
            "Probability_grooming": [0.4, 0.3, 0.1, 0.8, 0.7],
            "grooming":             [0, 0, 0, 1, 1],
        })
        out_path = save_classifications_for_video(
            video_name="video_001",
            config_path=str(toml),
            predictions=df,
        )
        check(
            "save_classifications_for_video returns a path under "
            "derived/classifications/",
            "derived/classifications/video_001.parquet"
            in out_path.replace("\\", "/"),
        )
        check(
            "save_classifications_for_video creates the file",
            Path(out_path).is_file(),
        )
        read_back = load_classifications_for_video(
            video_name="video_001",
            config_path=str(toml),
        )
        check(
            "Round-trip: row count preserved",
            len(read_back) == 5,
        )
        check(
            "Round-trip: all 4 columns preserved",
            set(read_back.columns) == {
                "Probability_attack", "attack",
                "Probability_grooming", "grooming",
            },
        )
        check(
            "Round-trip: probability values preserved",
            list(read_back["Probability_attack"]) == [0.1, 0.6,
                                                     0.9, 0.2, 0.5],
        )
        check(
            "Round-trip: prediction values preserved",
            list(read_back["attack"]) == [0, 1, 1, 0, 0],
        )

        # 4. targets-filtered read
        sub = load_classifications_for_video(
            video_name="video_001",
            config_path=str(toml),
            targets=["attack"],
        )
        check(
            "targets-filtered: returns only Probability_attack + "
            "attack columns",
            list(sub.columns) == ["Probability_attack", "attack"],
        )
        check(
            "targets-filtered: row count unchanged",
            len(sub) == 5,
        )

        # 5. targets-filtered with a missing target → NaN-filled
        missing = load_classifications_for_video(
            video_name="video_001",
            config_path=str(toml),
            targets=["never_trained"],
        )
        check(
            "targets-filtered: missing target → 2 NaN-filled "
            "columns",
            list(missing.columns)
            == ["Probability_never_trained", "never_trained"]
            and missing["never_trained"].isna().all()
            and missing["Probability_never_trained"].isna().all(),
        )

        # 6. list_video_stems_with_classifications
        save_classifications_for_video(
            video_name="video_002",
            config_path=str(toml),
            predictions=df,
        )
        stems = list_video_stems_with_classifications(str(toml))
        check(
            "list_video_stems_with_classifications: returns both "
            "stems, sorted",
            stems == ["video_001", "video_002"],
        )

        # 7. video_name with .mp4 suffix → stripped to bare stem
        save_classifications_for_video(
            video_name="video_003.mp4",
            config_path=str(toml),
            predictions=df,
        )
        check(
            ".mp4 suffix stripped from video_name on save",
            (Path(out_path).parent / "video_003.parquet").is_file(),
        )

        # 8. missing predictions → FileNotFoundError
        try:
            load_classifications_for_video(
                video_name="never_inferenced",
                config_path=str(toml),
            )
            raised = False
        except FileNotFoundError:
            raised = True
        check(
            "load: missing video → FileNotFoundError",
            raised,
        )

    # ==================================================================
    # 9. InferenceBatch dual-write site — code level
    # ==================================================================
    ib_path = REPO_ROOT / "mufasa" / "model" / "inference_batch.py"
    ib_src = ib_path.read_text()
    check(
        "InferenceBatch imports save_classifications_for_video",
        "save_classifications_for_video" in ib_src,
    )
    check(
        "InferenceBatch sidecar references "
        "_prediction_columns helper",
        "_prediction_columns" in ib_src,
    )
    check(
        "InferenceBatch sidecar wrapped in try/except — never "
        "blocks legacy CSV write",
        "save_classifications_for_video" in ib_src
        and "except Exception" in ib_src,
    )
    check(
        "InferenceBatch records 122at patch number",
        "122at" in ib_src,
    )

    print(
        f"smoke_122at_classifications_v1_io: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
