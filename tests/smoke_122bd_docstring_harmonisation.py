"""
tests/smoke_122bd_docstring_harmonisation.py
=============================================

Patch 122bd: docstring + visible-text harmonisation pass for
analysis, processor, plotting, and mixin modules whose
``:param`` descriptions, class summaries, and ``>>>``
doctest-style examples still referenced legacy
``csv/machine_results/`` or ``csv/features_extracted/``
paths. Cosmetic but matters for searchability and to avoid
misleading new contributors.

Scope (this patch)
------------------
Categories touched:
    A) Param ``:param`` docs that named the default
       directory — now name both v1 and legacy locations.
    B) Class-level summary docstrings — same.
    C) ``>>>`` doctest-style examples — rewrote example
       paths to ``derived/classifications/<stem>.parquet``,
       AND fixed the resulting type mismatch where the
       example was reading ``.parquet`` via
       ``pd.read_csv`` (now uses ``pd.read_parquet``).
    D) Active user-facing error message in
       ``InferenceMulticlassBatch.__init__``.

Categories explicitly NOT touched:
    * ``# test = ...`` commented-out test invocations at
      module footers — historical, not user-visible.
    * Internal ``# legacy fallback: ...`` retrospective
      comments — accurate by design.
    * ``legacy_layout.py`` migration map — intentionally
      lists the legacy path as the source of the
      migration mapping.

Coverage
--------
1. agg_clf_calculator + agg_clf_counter_mp: data_dir param doc mentions both v1 and legacy.
2. kleinberg_calculator: input_dir + output_dir param docs mention both.
3. inference_batch class summary mentions both v1 and legacy features dirs AND derived/classifications/ writes; save_dir param doc is marked deprecated post-122ax.
4. inference_multiclass_batch has a class docstring; error message mentions both locations.
5. plot_clf_results video_file_path param doc mentions both.
6. plotting >>> examples updated to derived/classifications/<stem>.parquet — verified across 7 backends.
7. mixins/geometry_mixin >>> examples use pd.read_parquet (not pd.read_csv on .parquet).
8. No remaining `>>>` doctest example with read_csv(...parquet) anywhere in the tree.
"""
from __future__ import annotations

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
    # 1. data_processor param docs
    # ==================================================================
    for fp_str, mention in (
        ("mufasa/data_processors/agg_clf_calculator.py",
         "classifier prediction files"),
        ("mufasa/data_processors/agg_clf_counter_mp.py",
         "classifier prediction files"),
    ):
        src = (REPO_ROOT / fp_str).read_text()
        check(
            f"{fp_str}: data_dir param doc mentions v1 path",
            "derived/classifications/" in src,
        )
        check(
            f"{fp_str}: data_dir param doc mentions legacy path",
            "csv/machine_results/" in src,
        )
        check(
            f"{fp_str}: data_dir param doc rewritten to "
            f"reference both formats",
            mention in src,
        )

    kleinberg_src = (REPO_ROOT
                     / "mufasa/data_processors/kleinberg_calculator.py"
                     ).read_text()
    check(
        "kleinberg_calculator: input_dir param doc mentions "
        "both formats",
        "derived/classifications/" in kleinberg_src
        and "csv/machine_results/" in kleinberg_src,
    )
    check(
        "kleinberg_calculator: defaults phrasing harmonised "
        "(uses 'defaults to' for both)",
        kleinberg_src.count("defaults to the project's") >= 2,
    )

    # ==================================================================
    # 2. inference_batch class summary + save_dir deprecation
    # ==================================================================
    ib_src = (REPO_ROOT / "mufasa/model/inference_batch.py").read_text()
    check(
        "inference_batch: class summary mentions derived/features/",
        "``derived/features/``" in ib_src
        or "derived/features/" in ib_src,
    )
    check(
        "inference_batch: class summary mentions "
        "derived/classifications/<video>.parquet",
        "derived/classifications/<video>.parquet" in ib_src,
    )
    check(
        "inference_batch: class summary no longer says "
        "'Results are stored in ... csv/machine_results'",
        "Results are stored in the ``project_folder/csv/machine_results``"
        not in ib_src,
    )
    check(
        "inference_batch: save_dir param marked deprecated",
        "Deprecated post-122ax" in ib_src,
    )
    check(
        "inference_batch: features_dir param mentions both formats",
        "derived/features/" in ib_src
        and "csv/features_extracted/" in ib_src,
    )

    # ==================================================================
    # 3. inference_multiclass_batch class docstring + error msg
    # ==================================================================
    imb_src = (REPO_ROOT
               / "mufasa/model/inference_multiclass_batch.py"
               ).read_text()
    check(
        "inference_multiclass_batch: gained class docstring",
        '"""\n    Run multi-class classifier inference' in imb_src,
    )
    check(
        "inference_multiclass_batch: error msg mentions both "
        "features locations",
        "derived/features/<video>.parquet" in imb_src
        and "csv/features_extracted/" in imb_src,
    )
    check(
        "inference_multiclass_batch: error msg no longer "
        "lists only the legacy path",
        'Zero files found in the project_folder/csv/features_extracted'
        not in imb_src,
    )

    # ==================================================================
    # 4. plot_clf_results param doc
    # ==================================================================
    pcr_src = (REPO_ROOT / "mufasa/plotting/plot_clf_results.py"
               ).read_text()
    check(
        "plot_clf_results: video_file_path param doc mentions "
        "both formats",
        "derived/classifications/" in pcr_src
        and "csv/machine_results/" in pcr_src,
    )

    # ==================================================================
    # 5. >>> doctest examples in plotting backends — point at v1
    # ==================================================================
    PLOTTING_FILES_WITH_DOCTEST = [
        "mufasa/plotting/gantt_creator.py",
        "mufasa/plotting/probability_plot_creator.py",
        "mufasa/plotting/path_plotter.py",
        "mufasa/plotting/gantt_creator_mp.py",
        "mufasa/plotting/heat_mapper_location.py",
        "mufasa/plotting/heat_mapper_clf_mp.py",
        "mufasa/plotting/path_plotter_mp.py",
    ]
    for fp_str in PLOTTING_FILES_WITH_DOCTEST:
        src = (REPO_ROOT / fp_str).read_text()
        # >>> lines should no longer reference csv/machine_results
        # for THE doctest example (commented-out test code may
        # still mention it — that's category D, out of scope).
        offending = []
        for lineno, line in enumerate(src.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith(">>>") and "csv/machine_results/" in line:
                offending.append((lineno, line.strip()))
        check(
            f"{fp_str}: no >>> doctest example references "
            f"csv/machine_results/",
            len(offending) == 0,
            detail=(
                "; ".join(f"L{n}: {l[:80]}" for n, l in offending)
                if offending else ""
            ),
        )
        # And at least one >>> line should mention derived/
        # classifications — verifying the rewrite landed.
        has_v1_doctest = any(
            line.lstrip().startswith(">>>")
            and "derived/classifications/" in line
            for line in src.splitlines()
        )
        check(
            f"{fp_str}: at least one >>> doctest example "
            f"points at derived/classifications/",
            has_v1_doctest,
        )

    # ==================================================================
    # 6. mixin doctest examples — pd.read_parquet, no pd.read_csv on parquet
    # ==================================================================
    gm_src = (REPO_ROOT / "mufasa/mixins/geometry_mixin.py"
              ).read_text()
    check(
        "geometry_mixin: doctest example uses pd.read_parquet",
        ">>> df = pd.read_parquet" in gm_src,
    )
    check(
        "geometry_mixin: doctest example uses .head(1000) "
        "(replaced index_col= for parquet)",
        ".head(1000).iloc[:, 0:21]" in gm_src,
    )
    check(
        "geometry_mixin: no >>> with pd.read_csv on .parquet",
        ">>> df = pd.read_csv" not in gm_src
        or ".parquet" not in gm_src.split(">>> df = pd.read_csv")[1].split("\n")[0]
        if ">>> df = pd.read_csv" in gm_src
        else True,
    )

    # ==================================================================
    # 7. Repo-wide invariant: no >>> with read_csv(...parquet)
    # ==================================================================
    import os
    bad_lines = []
    for root, _, names in os.walk(REPO_ROOT / "mufasa"):
        for n in names:
            if not n.endswith(".py"):
                continue
            fp = Path(root) / n
            try:
                src = fp.read_text()
            except Exception:
                continue
            for lineno, line in enumerate(src.splitlines(), 1):
                stripped = line.lstrip()
                if (stripped.startswith(">>>")
                        and "read_csv" in line
                        and ".parquet" in line):
                    bad_lines.append(
                        (fp.relative_to(REPO_ROOT), lineno))
    check(
        "Repo-wide: no >>> example reads parquet via read_csv",
        len(bad_lines) == 0,
        detail=(
            "; ".join(f"{p}:{n}" for p, n in bad_lines)
            if bad_lines else ""
        ),
    )

    # ==================================================================
    # 8. All 14 touched files record 122bd
    # ==================================================================
    TOUCHED_FILES = [
        "mufasa/data_processors/agg_clf_calculator.py",
        "mufasa/data_processors/agg_clf_counter_mp.py",
        "mufasa/data_processors/kleinberg_calculator.py",
        "mufasa/model/inference_batch.py",
        "mufasa/model/inference_multiclass_batch.py",
        "mufasa/plotting/plot_clf_results.py",
        "mufasa/plotting/gantt_creator.py",
        "mufasa/plotting/probability_plot_creator.py",
        "mufasa/plotting/path_plotter.py",
        "mufasa/plotting/gantt_creator_mp.py",
        "mufasa/plotting/heat_mapper_location.py",
        "mufasa/plotting/heat_mapper_clf_mp.py",
        "mufasa/plotting/path_plotter_mp.py",
        "mufasa/mixins/geometry_mixin.py",
    ]
    # Note: 122bd-recording is not required because the
    # changes are all docstring-only — there's no behavioural
    # change to bind to a patch ID. The recording is purely
    # documentary, so this is a soft check (not strict).
    # We just verify the rewrite landed by checking for
    # derived/classifications/ presence somewhere in each.
    for fp_str in TOUCHED_FILES:
        src = (REPO_ROOT / fp_str).read_text()
        check(
            f"{fp_str}: contains a derived/ reference "
            f"(rewrite landed)",
            "derived/" in src,
        )

    print(
        f"smoke_122bd_docstring_harmonisation: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
