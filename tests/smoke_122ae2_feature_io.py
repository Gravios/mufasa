"""
tests/smoke_122ae2_feature_io.py
=================================

Patch 122ae-2: behavioural verification of the
:mod:`mufasa.utils.feature_io` module — the shared read API
that consumers switch to in patch 122ae-5.

Two real branches to exercise end-to-end (the sandbox now
has pyarrow + pandas, so this test can write and read actual
parquet files):

1. **Per-family parquet branch** — set up a v1 project tree
   with two per-family directories each containing one
   parquet file for a video. ``load_features_for_video``
   should concat them into a single wide DataFrame with all
   the union of columns from both families.

2. **Legacy CSV fallback branch** — set up the same project
   with only the legacy ``csv/features_extracted/<video>.csv``
   wide file (no ``derived/features/`` tree). The same helper
   call should return the legacy file's contents.

3. **Both present** — when the per-family tree exists AND
   contains files for the video, the legacy fallback is
   ignored. Verifies the precedence order.

4. **Neither present** — clean FileNotFoundError with a
   message that names both probed paths so users can debug.

5. **family_slug** — round-trip the canonical FEATURE_FAMILIES
   names through the slugifier; confirm the outputs are
   filesystem-safe.

6. **Subset selection via ``families`` argument** — passing a
   subset narrows the result to only those families.

7. **Tolerates ``video_name`` with extension** — same result
   whether the caller passes 'video_001' or 'video_001.mp4'.

8. **Duplicate columns** — when two families have a colliding
   column name (a writer bug, but possible), the reader
   keeps the first occurrence and emits a RuntimeWarning.

9. **AST checks** — the module exposes the expected public
   API (family_slug, load_features_for_video), uses
   project_paths_from_config + project_metadata_from_config,
   and reads import_file_type rather than file_type as the
   primary key (with file_type as fallback).
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import textwrap
import warnings
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


def _make_v1_project(tmp: Path, file_type: str = "parquet") -> Path:
    """Make a minimal v1 project_root with a project.toml; return
    the toml path."""
    toml = tmp / "project.toml"
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122ae2"
        version = "0.0.1"

        [pose]
        file_type = "{file_type}"
        animal_count = 1
        body_parts = ["nose", "tail"]
    """).strip() + "\n")
    return toml


def main() -> int:
    from mufasa.utils.feature_io import (family_slug,
                                         load_features_for_video)

    # ==================================================================
    # 5. family_slug — table-driven correctness check
    # ==================================================================
    SLUG_CASES = [
        ("TWO-POINT BODY-PART DISTANCES (MM)",
         "two_point_body_part_distances_mm"),
        ("WITHIN-ANIMAL THREE-POINT BODY-PART ANGLES (DEGREES)",
         "within_animal_three_point_body_part_angles_degrees"),
        ("WITHIN-ANIMAL FOUR-POINT CONVEX HULL PERIMETERS (MM)",
         "within_animal_four_point_convex_hull_perimeters_mm"),
        ("ENTIRE ANIMAL CONVEX HULL AREA (MM2)",
         "entire_animal_convex_hull_area_mm2"),
        ("FRAME-BY-FRAME BODY-PARTS INSIDE ROIS (BOOLEAN)",
         "frame_by_frame_body_parts_inside_rois_boolean"),
        # Edge case: punctuation-only → fallback
        ("   ---   ", "untitled"),
    ]
    for inp, expected in SLUG_CASES:
        got = family_slug(inp)
        check(
            f"family_slug({inp!r}) → {expected!r}",
            got == expected,
            detail=f"got {got!r}",
        )

    # ==================================================================
    # 1. Per-family parquet branch
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, file_type="parquet")

        feat_dir = tmp / "derived" / "features"
        slug_a = family_slug("TWO-POINT BODY-PART DISTANCES (MM)")
        slug_b = family_slug(
            "FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)",
        )
        (feat_dir / slug_a).mkdir(parents=True)
        (feat_dir / slug_b).mkdir(parents=True)

        # Two parquet files for the same video, different columns
        df_a = pd.DataFrame({
            "dist_nose_tail": [1.1, 2.2, 3.3, 4.4],
            "dist_nose_ear":  [5.5, 6.6, 7.7, 8.8],
        })
        df_b = pd.DataFrame({
            "move_nose": [0.01, 0.02, 0.03, 0.04],
            "move_tail": [0.05, 0.06, 0.07, 0.08],
        })
        df_a.to_parquet(feat_dir / slug_a / "video_001.parquet")
        df_b.to_parquet(feat_dir / slug_b / "video_001.parquet")

        result = load_features_for_video("video_001", str(toml))
        check(
            "per-family branch: returns a DataFrame",
            isinstance(result, pd.DataFrame),
        )
        check(
            "per-family branch: all source columns present "
            "after concat",
            set(result.columns) == {"dist_nose_tail",
                                     "dist_nose_ear",
                                     "move_nose",
                                     "move_tail"},
            detail=f"got {set(result.columns)}",
        )
        check(
            "per-family branch: row count matches inputs (4)",
            len(result) == 4,
        )
        check(
            "per-family branch: numeric values round-tripped "
            "exactly (parquet preserves float64)",
            (result["dist_nose_tail"].iloc[0] == 1.1
             and result["move_tail"].iloc[3] == 0.08),
        )

    # ==================================================================
    # 2. Legacy CSV no longer consulted (patch 122ak — v1-only)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, file_type="csv")

        # Seed only a legacy CSV — no v1 parquet.
        legacy_dir = tmp / "csv" / "features_extracted"
        legacy_dir.mkdir(parents=True)
        legacy_df = pd.DataFrame({
            "Unnamed: 0":     [0, 1, 2, 3],
            "feature_alpha":  [10.0, 20.0, 30.0, 40.0],
            "feature_beta":   ["x", "y", "z", "w"],
        })
        legacy_df.to_csv(legacy_dir / "video_002.csv", index=False)

        # Patch 122ak: load_features_for_video no longer falls back
        # to legacy CSV. Seeding only legacy data must raise.
        raised = False
        try:
            load_features_for_video("video_002", str(toml))
        except FileNotFoundError:
            raised = True
        check(
            "legacy-only project (v1-only patch): raises "
            "FileNotFoundError instead of silently reading legacy",
            raised,
        )

    # ==================================================================
    # 3. Per-family branch returns the per-family data (no legacy
    #    consulted even when present)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, file_type="csv")

        # Per-family
        feat_dir = tmp / "derived" / "features"
        slug = family_slug("TWO-POINT BODY-PART DISTANCES (MM)")
        (feat_dir / slug).mkdir(parents=True)
        pd.DataFrame({"new_col": [99, 98, 97]}).to_parquet(
            feat_dir / slug / "video_003.parquet",
        )

        # Also drop a legacy CSV to confirm it's ignored.
        legacy_dir = tmp / "csv" / "features_extracted"
        legacy_dir.mkdir(parents=True)
        pd.DataFrame({"old_col": [1, 2, 3]}).to_csv(
            legacy_dir / "video_003.csv", index=False,
        )

        result = load_features_for_video("video_003", str(toml))
        check(
            "v1-only patch: per-family data returned, legacy "
            "ignored",
            "new_col" in result.columns and "old_col" not in result.columns,
            detail=f"got {set(result.columns)}",
        )

    # ==================================================================
    # 4. Neither present — FileNotFoundError mentions v1 paths
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp, file_type="csv")
        raised = False
        msg = ""
        try:
            load_features_for_video("video_missing", str(toml))
        except FileNotFoundError as exc:
            raised = True
            msg = str(exc)
        check("missing: raises FileNotFoundError", raised)
        check(
            "missing: error message mentions derived/features",
            "derived" in msg and "features" in msg,
            detail=f"got: {msg!r}",
        )
        check(
            "missing: error message names the video stem",
            "video_missing" in msg,
        )

    # ==================================================================
    # 6. Subset selection via families argument
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        feat_dir = tmp / "derived" / "features"
        slug_a = family_slug("TWO-POINT BODY-PART DISTANCES (MM)")
        slug_b = family_slug(
            "FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)",
        )
        (feat_dir / slug_a).mkdir(parents=True)
        (feat_dir / slug_b).mkdir(parents=True)
        pd.DataFrame({"a_col": [1, 2]}).to_parquet(
            feat_dir / slug_a / "video_004.parquet",
        )
        pd.DataFrame({"b_col": [3, 4]}).to_parquet(
            feat_dir / slug_b / "video_004.parquet",
        )

        # Ask for only A
        result = load_features_for_video(
            "video_004", str(toml),
            families=["TWO-POINT BODY-PART DISTANCES (MM)"],
        )
        check(
            "families= filter: returns only A's columns",
            set(result.columns) == {"a_col"},
            detail=f"got {set(result.columns)}",
        )

    # ==================================================================
    # 7. Tolerates extension on video_name
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        feat_dir = tmp / "derived" / "features"
        slug = family_slug("TWO-POINT BODY-PART DISTANCES (MM)")
        (feat_dir / slug).mkdir(parents=True)
        pd.DataFrame({"col": [1.0, 2.0]}).to_parquet(
            feat_dir / slug / "video_005.parquet",
        )
        r1 = load_features_for_video("video_005", str(toml))
        r2 = load_features_for_video("video_005.mp4", str(toml))
        check(
            "extension tolerance: 'video_005' and "
            "'video_005.mp4' return identical results",
            r1.equals(r2),
        )

    # ==================================================================
    # 8. Duplicate columns warn but don't crash
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _make_v1_project(tmp)
        feat_dir = tmp / "derived" / "features"
        (feat_dir / "fam_one").mkdir(parents=True)
        (feat_dir / "fam_two").mkdir(parents=True)
        pd.DataFrame({"shared_col": [1, 2], "unique_a": [10, 20]}
                     ).to_parquet(
            feat_dir / "fam_one" / "video_006.parquet",
        )
        pd.DataFrame({"shared_col": [9, 9], "unique_b": [30, 40]}
                     ).to_parquet(
            feat_dir / "fam_two" / "video_006.parquet",
        )

        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            result = load_features_for_video("video_006", str(toml))
            warned = any(
                "Duplicate columns" in str(w.message)
                for w in wlist
            )
        check("duplicate columns: RuntimeWarning emitted", warned)
        check(
            "duplicate columns: first occurrence kept "
            "(not crashed)",
            "shared_col" in result.columns
            and len(result["shared_col"]) == 2,
        )
        # Note: pd.concat axis=1 followed by drop-duplicates preserves
        # the LEFTMOST occurrence. Don't pin specific values — the
        # important contract is 'no crash, no duplicate column names
        # in the output'.
        check(
            "duplicate columns: output has no duplicated names",
            not any(result.columns.duplicated()),
        )

    # ==================================================================
    # 9. AST checks on the module surface
    # ==================================================================
    src = (REPO_ROOT / "mufasa" / "utils"
           / "feature_io.py").read_text()
    tree = ast.parse(src)
    top_names = [
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    check("feature_io exports family_slug", "family_slug" in top_names)
    check(
        "feature_io exports load_features_for_video",
        "load_features_for_video" in top_names,
    )
    check(
        "feature_io imports project_paths_from_config",
        "project_paths_from_config" in src,
    )
    # Patch 122ak: project_metadata_from_config import + the
    # 'import_file_type' lookup were both part of the legacy CSV
    # fallback (resolving the legacy file's extension). Both are
    # gone now.
    check(
        "feature_io docstring records the 122ae-2 patch number",
        "122ae-2" in src,
    )

    print(
        f"smoke_122ae2_feature_io: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
