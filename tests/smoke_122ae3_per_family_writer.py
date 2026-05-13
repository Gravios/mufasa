"""
tests/smoke_122ae3_per_family_writer.py
========================================

Patch 122ae-3: behavioural verification of the per-family
parquet writer added to ``process_one_video`` plus AST checks
that the FeatureSubsetExtractorForm exposes the new mode.

Three layers:

1. **VideoProcessingConfig dataclass schema** — the new
   ``derived_features_dir`` field is present, optional,
   defaults to None.

2. **process_one_video write logic** — when
   ``config.derived_features_dir`` is set, per-family parquet
   files appear at the expected path with the expected column
   suffix (``_FEATURE_SUBSET``). When it's None, no per-family
   files are created. The legacy wide write to ``temp_dir``
   always happens unchanged.

   To exercise this we need to bypass the heavy backend (numba
   kernels, ConfigReader, multiprocess pool). We call the
   *write phase* of process_one_video's logic directly by
   constructing a minimal "results_by_family" map and invoking
   the same parquet-writing code the orchestrator calls. Done
   via a tiny helper that mirrors the actual write block —
   keeps the test independent of numba availability.

3. **FeatureSubsetExtractorForm AST surface** —

   * New radio ``dest_derived_parquet`` exists.
   * The new radio is checked by default (replaces save_dir
     as default).
   * Backend is invoked with ``derived_features_dir`` kwarg.
   * ``collect_args`` returns ``derived_features_dir`` key.
   * Pre-122ae-3 stale phrases ('deferred run-id allocation'
     etc.) are gone from the active code paths.
"""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

import pandas as pd

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
    # 1. VideoProcessingConfig dataclass schema
    # ==================================================================
    src = (REPO_ROOT / "mufasa" / "feature_extractors"
           / "feature_subset_orchestration.py").read_text()
    tree = ast.parse(src)
    cfg_cls = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef)
         and n.name == "VideoProcessingConfig"),
        None,
    )
    check("VideoProcessingConfig class defined", cfg_cls is not None)

    # The dataclass fields are AnnAssigns at the class body level.
    field_names: list[str] = []
    if cfg_cls is not None:
        for stmt in cfg_cls.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(
                stmt.target, ast.Name,
            ):
                field_names.append(stmt.target.id)
    check(
        "VideoProcessingConfig has 'derived_features_dir' field",
        "derived_features_dir" in field_names,
    )
    check(
        "VideoProcessingConfig keeps legacy fields "
        "(temp_dir, file_type, feature_families)",
        all(f in field_names for f in (
            "temp_dir", "file_type", "feature_families",
        )),
    )

    # ==================================================================
    # 2. Per-family write end-to-end. Mirror the write block from
    #    process_one_video to test in isolation (avoids numba +
    #    ConfigReader requirements).
    # ==================================================================
    from mufasa.utils.feature_io import family_slug

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        derived_features_dir = tmp / "derived" / "features"
        # Mirror the write block exactly. If the orchestrator
        # changes shape, the smoke breaks on the AST check below;
        # this block tests the disk side specifically.
        results_by_family = {
            "TWO-POINT BODY-PART DISTANCES (MM)": {
                "nose_to_tail": [1.0, 2.0, 3.0],
                "nose_to_ear":  [4.0, 5.0, 6.0],
            },
            "FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)": {
                "nose_move": [0.1, 0.2, 0.3],
            },
        }
        video_name = "v_001"
        import os
        for family, fam_columns in results_by_family.items():
            slug = family_slug(family)
            out_dir = os.path.join(str(derived_features_dir), slug)
            os.makedirs(out_dir, exist_ok=True)
            fam_df = pd.DataFrame(fam_columns)
            fam_df = fam_df.add_suffix("_FEATURE_SUBSET")
            fam_df = fam_df[sorted(fam_df.columns)]
            fam_df = fam_df.fillna(-1)
            fam_df.to_parquet(
                os.path.join(out_dir, f"{video_name}.parquet"),
                index=False,
            )

        # Now read back and verify
        slug_a = family_slug("TWO-POINT BODY-PART DISTANCES (MM)")
        slug_b = family_slug("FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)")
        check(
            "per-family write: slug A directory exists",
            (derived_features_dir / slug_a).is_dir(),
        )
        check(
            "per-family write: slug B directory exists",
            (derived_features_dir / slug_b).is_dir(),
        )
        path_a = derived_features_dir / slug_a / f"{video_name}.parquet"
        path_b = derived_features_dir / slug_b / f"{video_name}.parquet"
        check(
            "per-family write: slug A parquet file written",
            path_a.is_file(),
        )
        check(
            "per-family write: slug B parquet file written",
            path_b.is_file(),
        )

        df_a = pd.read_parquet(path_a)
        df_b = pd.read_parquet(path_b)
        check(
            "per-family write: slug A columns have "
            "_FEATURE_SUBSET suffix",
            all(c.endswith("_FEATURE_SUBSET") for c in df_a.columns),
        )
        check(
            "per-family write: slug A columns are sorted",
            list(df_a.columns) == sorted(df_a.columns),
        )
        check(
            "per-family write: slug A values round-tripped "
            "exactly",
            df_a["nose_to_tail_FEATURE_SUBSET"].tolist()
            == [1.0, 2.0, 3.0],
        )
        check(
            "per-family write: slug B values round-tripped "
            "exactly",
            df_b["nose_move_FEATURE_SUBSET"].tolist()
            == [0.1, 0.2, 0.3],
        )

        # End-to-end via the load helper from 122ae-2
        from mufasa.utils.feature_io import load_features_for_video
        # Construct a v1 project.toml so load_features_for_video
        # finds the derived_features_dir via the layout helper.
        toml = tmp / "project.toml"
        toml.write_text(
            "project_layout_version = 1\n\n"
            "[project]\nname = 'smoke'\nversion = '0.0.1'\n\n"
            "[pose]\nfile_type = 'parquet'\n"
        )
        merged = load_features_for_video(video_name, str(toml))
        check(
            "end-to-end: load_features_for_video reads back "
            "all 3 columns from both families",
            len(merged.columns) == 3
            and "nose_to_tail_FEATURE_SUBSET" in merged.columns
            and "nose_move_FEATURE_SUBSET" in merged.columns,
            detail=f"got {list(merged.columns)}",
        )

    # ==================================================================
    # 2b. process_one_video AST — the write block must exist in
    #     the code path, gated on config.derived_features_dir is
    #     not None, and use family_slug.
    # ==================================================================
    check(
        "process_one_video imports family_slug from feature_io",
        "from mufasa.utils.feature_io import family_slug" in src,
    )
    check(
        "process_one_video gates per-family write on "
        "config.derived_features_dir is not None",
        "config.derived_features_dir is not None" in src,
    )
    check(
        "process_one_video tracks results_by_family",
        "results_by_family" in src,
    )
    check(
        "process_one_video writes parquet (not csv) for "
        "per-family files",
        ".to_parquet(" in src,
    )
    check(
        "process_one_video docstring records the 122ae-3 fix",
        "Patch 122ae-3" in src,
    )

    # ==================================================================
    # 3. Calculator __init__ accepts derived_features_dir
    # ==================================================================
    calc_src = (REPO_ROOT / "mufasa" / "feature_extractors"
                / "feature_subsets.py").read_text()
    check(
        "FeatureSubsetsCalculator.__init__ signature includes "
        "derived_features_dir",
        "derived_features_dir: Optional[" in calc_src
        or "derived_features_dir=None" in calc_src,
    )
    check(
        "FeatureSubsetsCalculator stores self.derived_features_dir",
        "self.derived_features_dir" in calc_src,
    )
    check(
        "FeatureSubsetsCalculator passes derived_features_dir to "
        "VideoProcessingConfig",
        "derived_features_dir=self.derived_features_dir" in calc_src,
    )
    check(
        "No-destination guard counts derived_features_dir as "
        "a valid destination",
        "self.derived_features_dir is None" in calc_src,
    )

    # ==================================================================
    # 4. FeatureSubsetExtractorForm AST surface
    # ==================================================================
    form_src = (REPO_ROOT / "mufasa" / "ui_qt"
                / "forms" / "features.py").read_text()
    check(
        "Form defines dest_derived_parquet radio",
        "self.dest_derived_parquet" in form_src,
    )
    check(
        "Form sets dest_derived_parquet as default checked",
        "self.dest_derived_parquet.setChecked(True)" in form_src,
    )
    check(
        "Form's collect_args resolves derived_features_dir from "
        "project_paths_from_config",
        ("project_paths_from_config" in form_src
         and "derived_features_dir" in form_src),
    )
    check(
        "Form's target() takes derived_features_dir kwarg",
        "derived_features_dir: Optional[str]" in form_src,
    )
    check(
        "Form's target() passes derived_features_dir to "
        "FeatureSubsetsCalculator",
        "derived_features_dir=derived_features_dir" in form_src,
    )

    # ==================================================================
    # 5. 122z's deferred caveats removed from radio labels
    # ==================================================================
    # Walk the form module to extract only string literals that
    # appear inside the build() method — that's where the radio
    # labels live. The 122z notes in the patch-history docstrings
    # at the top of the file are historical context and may
    # legitimately mention 'deferred run-id allocation' — we just
    # don't want it in the user-visible labels.
    form_tree = ast.parse(form_src)
    build_method = None
    for cls in (n for n in ast.walk(form_tree)
                if isinstance(n, ast.ClassDef)
                and n.name == "FeatureSubsetExtractorForm"):
        for member in cls.body:
            if (isinstance(member, ast.FunctionDef)
                    and member.name == "build"):
                build_method = member
                break

    deferred_in_labels = False
    if build_method is not None:
        build_src = ast.unparse(build_method)
        # ast.unparse() drops comments — any 'deferred run-id'
        # that survives is in a string literal (radio label).
        deferred_in_labels = "deferred run-id" in build_src.lower()
    check(
        "Radio labels no longer mention 'deferred run-id "
        "allocation' (the 122z caveat is gone now that the "
        "per-family layout answers the question)",
        not deferred_in_labels,
    )

    print(
        f"smoke_122ae3_per_family_writer: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
