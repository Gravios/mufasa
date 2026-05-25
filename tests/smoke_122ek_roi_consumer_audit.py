"""
tests/smoke_122ek_roi_consumer_audit.py
==========================================

Patch 122ek: ROI-consumer audit. Extends 122ej's column-
tolerance fix in :meth:`ConfigReader.read_roi_data` to four
additional ROI consumers that had the same shape of bug.

Context
-------
122ej fixed ``ConfigReader.read_roi_data`` to tolerate empty
DataFrames returned by ``pd.read_hdf`` when the user only drew
one shape type. The audit asked: which OTHER places in the
codebase do naked ``df["Video"] == ...`` filters or
``df["Video"].unique()`` calls on possibly-empty ROI
DataFrames?

The audit found four:

1. ``mufasa/roi_tools/roi_utils.py / multiply_ROIs``
   The legacy "apply to all" path. Used to crash on
   ``check_valid_dataframe(circles_df, required_fields=
   ['Video', 'Name'])`` for rectangles-only projects.

2. ``mufasa/roi_tools/roi_utils.py / reset_video_ROIs``
   The "clear ROIs for one video" path. Same
   ``check_valid_dataframe`` failure plus naked
   ``df[df["Video"] == video_name]`` filters.

3. ``mufasa/ui_qt/dialogs/duplicate_rois_source_target.py``
   The Apply body in the Duplicate ROIs dialog. Naked
   ``df[df["Video"] != target_video]`` filters in the
   overwrite loop. 122ej fixed the dialog OPEN path (via
   read_roi_data); 122ek finishes by fixing the APPLY path.

4. ``mufasa/roi_tools/ROI_size_standardizer.py``
   The size-standardizer's per-video filter loop. Same
   ``df[df["Video"] == video_name]`` pattern as the dialog.

Other consumers audited and cleared:
- ``ROI_analyzer.py``, ``ROI_directing_analyzer.py``,
  ``ROI_feature_analyzer.py``, ``ROI_aggregate_*``,
  ``ROI_feature_visualizer.py``, ``roi_plotter.py``,
  ``cue_light_analyzer.py``, ``cue_light_visualizer.py``,
  plotting mixins, geometry mixin, ``reconfigure_dialog.py``,
  ``roi_video_table.py``, ``roi_size_standardizer.py``
  (in ``ui_qt/dialogs/``). All of these either:
  * Iterate via ``df.iterrows()`` (safe — empty df → no
    iterations), or
  * Operate on ``self.roi_dict`` populated by
    ``read_roi_data`` (now defensive after 122ej), or
  * Use guarded column access (``"X" in df.columns``).

Design: centralized helpers
---------------------------
With 4 consumers needing the same defensive pattern, helpers
make sense over per-function inline closures. Three helpers
added to ``mufasa/roi_tools/roi_utils.py``:

* ``safe_filter_by_video(df, video_name)`` — equality filter
  returning empty df-with-same-columns if Video column missing.
* ``safe_filter_video_neq(df, video_name)`` — inequality
  filter (for rewrite loops).
* ``safe_videos_in(df)`` — unique values, returning empty
  list if column missing.

Plus a helper ``_empty_like(df)`` that returns ``df.iloc[0:0]``
— keeps the return type a DataFrame even when the column
isn't there, so callers don't need to special-case the
result.

``ConfigReader.read_roi_data``'s own ``_col_unique`` /
``_col_list`` inline closures from 122ej are NOT refactored
to use these helpers — they're slightly different shape
(unique-values vs. as-list) and the mixin should not depend
on roi_tools.

Coverage
--------
Helper definitions (4 checks):
1.  ``safe_filter_by_video`` defined in roi_utils.
2.  ``safe_filter_video_neq`` defined in roi_utils.
3.  ``safe_videos_in`` defined in roi_utils.
4.  ``_empty_like`` defined in roi_utils.

Helper implementation correctness (3 AST checks):
5.  ``safe_filter_by_video`` checks ``"Video" in df.columns``
    before filtering.
6.  ``safe_filter_video_neq`` checks ``"Video" in df.columns``
    before filtering.
7.  ``safe_videos_in`` checks ``"Video" in df.columns``
    before ``.unique()``.

Call-site migration (5 checks):
8.  ``multiply_ROIs`` uses ``safe_videos_in`` (replaces 3 naked
    ``["Video"].unique()`` reads).
9.  ``multiply_ROIs`` uses ``safe_filter_by_video`` (replaces
    naked ``df[df["Video"] == ...]`` filters).
10. ``reset_video_ROIs`` uses ``safe_filter_by_video`` AND
    ``safe_filter_video_neq``.
11. ``duplicate_rois_source_target.py`` uses
    ``safe_filter_video_neq`` in the Apply body.
12. ``ROI_size_standardizer.py`` uses ``safe_filter_by_video``
    in the per-video filter loop.

check_valid_dataframe guards (2 checks):
13. ``multiply_ROIs`` guards each ``check_valid_dataframe``
    call with ``len(df) > 0`` (so empty DataFrames don't
    raise on the missing-columns assertion).
14. ``reset_video_ROIs`` guards each ``check_valid_dataframe``
    call.

No remaining bare-column patterns (2 checks):
15. ``multiply_ROIs`` no longer contains
    ``rectangles_df["Video"]`` or analogous naked accesses
    (sanity — the migration is complete).
16. ``ROI_size_standardizer.py`` no longer contains the
    naked ``rectangles_df["Video"] == video_name`` filter.

Cross-patch invariants (4 checks):
17. 122ej state preserved: ``read_roi_data`` still has
    ``_col_unique`` and ``_col_list`` helpers.
18. 122ei state preserved: ``import_pose`` has a
    ``detect_path``.
19. 122eh state preserved: ``roi_coordinates_path`` resolves
    to logs/measures/ROI_definitions.h5.
20. 122do baseline: no ``Optional[`` in non-docstring positions
    across mufasa/ui_qt/.
"""
from __future__ import annotations

import ast
import re
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


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _find_method(tree: ast.Module, cls: str, m: str) -> ast.FunctionDef | None:
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef) and c.name == cls:
            for mem in c.body:
                if (isinstance(mem, ast.FunctionDef)
                        and mem.name == m):
                    return mem
    return None


def main() -> int:
    ru_path = REPO_ROOT / "mufasa" / "roi_tools" / "roi_utils.py"
    ru_src = ru_path.read_text()
    ru_tree = ast.parse(ru_src)

    # -----------------------------------------------------------------
    # Helper definitions
    # -----------------------------------------------------------------
    for name in ("safe_filter_by_video", "safe_filter_video_neq",
                 "safe_videos_in", "_empty_like"):
        fn = _find_function(ru_tree, name)
        check(
            f"roi_utils.{name} is defined as a module-level "
            f"function",
            fn is not None,
        )

    # -----------------------------------------------------------------
    # Helper implementation correctness
    # -----------------------------------------------------------------
    for fname, label in [
        ("safe_filter_by_video", "safe_filter_by_video"),
        ("safe_filter_video_neq", "safe_filter_video_neq"),
        ("safe_videos_in", "safe_videos_in"),
    ]:
        fn = _find_function(ru_tree, fname)
        if fn is None:
            check(
                f"(skipped — {fname} missing)",
                False, detail="helper not found",
            )
            continue
        body_src = ast.unparse(fn)
        # Either form is fine: positive check then access, OR
        # early-return on not-in. The helpers chose the early-
        # return form.
        has_guard = (
            "'Video' in df.columns" in body_src
            or '"Video" in df.columns' in body_src
            or "'Video' not in df.columns" in body_src
            or '"Video" not in df.columns' in body_src
        )
        check(
            f"{label} body guards against the missing 'Video' "
            f"column before the column access (either form: "
            f"positive-check or early-return-on-not-in)",
            has_guard,
        )

    # -----------------------------------------------------------------
    # Call-site migration in roi_utils.py
    # -----------------------------------------------------------------
    multiply = _find_function(ru_tree, "multiply_ROIs")
    reset = _find_function(ru_tree, "reset_video_ROIs")
    assert multiply is not None and reset is not None
    multiply_src = ast.unparse(multiply)
    reset_src = ast.unparse(reset)

    check(
        "multiply_ROIs uses safe_videos_in (replaces the 3 "
        "naked ['Video'].unique() reads)",
        "safe_videos_in" in multiply_src,
    )
    check(
        "multiply_ROIs uses safe_filter_by_video (replaces "
        "the naked df[df['Video'] == ...] filters)",
        "safe_filter_by_video" in multiply_src,
    )
    check(
        "reset_video_ROIs uses BOTH safe_filter_by_video AND "
        "safe_filter_video_neq (the read-then-rewrite pattern)",
        "safe_filter_by_video" in reset_src
        and "safe_filter_video_neq" in reset_src,
    )

    # -----------------------------------------------------------------
    # Call-site migration in dialog + standardizer
    # -----------------------------------------------------------------
    dlg_src = (REPO_ROOT / "mufasa" / "ui_qt" / "dialogs"
               / "duplicate_rois_source_target.py").read_text()
    check(
        "duplicate_rois_source_target.py uses "
        "safe_filter_video_neq in the Apply body (the "
        "overwrite-by-target loop that crashed pre-122ek "
        "when applying to a rectangles-only project)",
        "safe_filter_video_neq" in dlg_src,
    )

    rss_src = (REPO_ROOT / "mufasa" / "roi_tools"
               / "ROI_size_standardizer.py").read_text()
    check(
        "ROI_size_standardizer.py uses safe_filter_by_video "
        "(the per-video filter loop in standardize())",
        "safe_filter_by_video" in rss_src,
    )

    # -----------------------------------------------------------------
    # check_valid_dataframe guards
    # -----------------------------------------------------------------
    # Look for `if len(<df>) > 0:` paired with check_valid_dataframe.
    # Simplest pinning: the check_valid_dataframe lines are now
    # preceded by `if len(...) > 0:` blocks.
    check(
        "multiply_ROIs guards check_valid_dataframe with "
        "`if len(rectangles_df) > 0`/etc (so empty DataFrames "
        "don't raise on missing-required-columns)",
        "if len(rectangles_df) > 0:" in multiply_src
        and "check_valid_dataframe" in multiply_src,
    )
    check(
        "reset_video_ROIs similarly guards "
        "check_valid_dataframe with `if len(...) > 0`",
        "if len(rectangles_df) > 0:" in reset_src
        and "check_valid_dataframe" in reset_src,
    )

    # -----------------------------------------------------------------
    # No remaining bare-column patterns
    # -----------------------------------------------------------------
    # multiply_ROIs: no naked `rectangles_df["Video"]` after the
    # migration. (The helper calls don't contain that substring
    # because they pass the df as an argument.)
    bare_video = re.search(
        r'rectangles_df\["Video"\]\.unique\(\)',
        multiply_src,
    )
    check(
        "multiply_ROIs no longer contains naked "
        "`rectangles_df[\"Video\"].unique()` (sanity — the "
        "migration is complete)",
        bare_video is None,
    )

    bare_filter = re.search(
        r'self\.rectangles_df\[self\.rectangles_df\["Video"\]\s*==',
        rss_src,
    )
    check(
        "ROI_size_standardizer no longer contains the naked "
        "`self.rectangles_df[self.rectangles_df[\"Video\"] == "
        "video_name]` filter",
        bare_filter is None,
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    cr_src = (REPO_ROOT / "mufasa" / "mixins"
              / "config_reader.py").read_text()
    cr_tree = ast.parse(cr_src)
    rrd = _find_method(cr_tree, "ConfigReader", "read_roi_data")
    assert rrd is not None
    rrd_src = ast.unparse(rrd)
    check(
        "122ej state preserved: read_roi_data still has "
        "_col_unique and _col_list helpers",
        "_col_unique" in rrd_src and "_col_list" in rrd_src,
    )

    sp_src = (REPO_ROOT / "mufasa"
              / "section_provenance.py").read_text()
    check(
        "122ei state preserved: import_pose has a detect_path "
        "(via lambda root: root / 'sources' / 'pose')",
        "detect_path=lambda root:" in sp_src,
    )

    check(
        "122eh state preserved: roi_coordinates_path resolves "
        "to logs/measures/ROI_definitions.h5",
        '"measures"' in cr_src and '"ROI_definitions.h5"' in cr_src,
    )

    uiqt = REPO_ROOT / "mufasa" / "ui_qt"
    optional_hits = []
    for f in sorted(uiqt.rglob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"\bOptional\[", src):
            preceding = src[:m.start()]
            tq3 = preceding.count('"""') + preceding.count("'''")
            if tq3 % 2 == 0:
                optional_hits.append(str(f.relative_to(uiqt)))
                break
    check(
        "122do baseline preserved: no `Optional[` in non-"
        "docstring positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122ek_roi_consumer_audit: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
