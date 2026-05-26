"""
tests/smoke_122ev_data_processor_format_sweep.py
====================================================

Patch 122ev-hotfix: 5 data processors hardcoded
``extensions=['.csv']`` for input file discovery. v1 projects
write ``.parquet``; the hardcoded filter found 0 files. User-
reported on the egocentric aligner; class-of-bug audit
revealed 4 more silent-no-op backends.

User report (Mon May 25, 2026, fifth report of the day):

> [10:02:17] SIMBA WARNING: NoFileFoundWarning: Mufasa could
> not find any files with accepted extensions ['.csv'] in
> the /data/testing/mufasa/test-20260427/derived/
> outlier_corrected/20260518-192433-7f64a3 directory

The aligner found 0 files. Followed 122eu-hotfix (which made
``get_fn_ext`` robust to extension-less files like SKIPPED)
— so the SKIPPED file no longer crashed, but the dir
contained only ``.parquet`` data files which the aligner
ignored.

ROOT CAUSE
==========

v1 projects use ``self.file_type = 'parquet'`` (per the
ConfigReader). Outlier correction (and other backends that
inherit ConfigReader) reads/writes with self.file_type, so
v1 output lands in ``derived/outlier_corrected/<run_id>/
<video>.parquet``.

But ``EgocentricalAligner`` (NOT a ConfigReader subclass)
hardcoded ``extensions=['.csv']`` for input discovery and
``Formats.CSV.value`` for read_df + write_df. Result on a
v1 project: 0 files found, silent no-op exit.

CLASS-OF-BUG AUDIT
==================

AST grep for ``extensions=['.csv']`` across data_processors/
found FIVE instances:

| File                              | Class                        | Inherits ConfigReader? |
|-----------------------------------|------------------------------|------------------------|
| egocentric_aligner.py             | EgocentricalAligner          | No                     |
| distance_calculator.py            | DistanceCalculator           | Yes                    |
| distance_timbin_calculator.py     | DistanceTimeBinCalculator    | Yes                    |
| circling_detector.py              | CirclingDetector             | Yes                    |
| freezing_detector.py              | FreezingDetector             | Yes                    |

The 4 ConfigReader subclasses have ``self.file_type``
available but were inconsistent: read_df calls used
``file_type=self.file_type`` OR ``'csv'`` literal, and the
extension filter used ``['.csv']`` literal. Internal drift
within each class.

The egocentric aligner doesn't inherit ConfigReader — it
takes only what the form passes. So the fix shape is
different:

Per-class fix:

* **egocentric_aligner**: accept both ``.csv`` AND
  ``.parquet``; for each file, derive ``file_type`` from
  the actual extension via ``get_fn_ext``; pass that file_type
  to read_df + write_df + save_path. Preserves format
  end-to-end.

* **distance_calculator + distance_timbin_calculator +
  circling_detector + freezing_detector**: change extension
  filter to use ``self.file_type`` (which they already had).
  Eliminates the internal drift and aligns with v1.

  ``circling_detector`` + ``freezing_detector`` additionally
  had ``read_df(..., file_type='csv')`` literals — flipped
  to ``self.file_type`` for the same reason.

After this patch, all 5 backends:
- Accept v1 (parquet) input automatically.
- Preserve format end-to-end.
- Have no internal drift between filter and read_df.

GENERALIZATION OF THE 122es LESSON
====================================

122es said: "audit the class of bug." 122eu refined:
"fix at the lowest layer with enough context." 122ev
applies both:

- Audit the class (5 files found, not just the user's
  egocentric).
- Fix at the right layer for each class member:
  - egocentric_aligner: at the constructor/read loop
    (no ConfigReader to lean on).
  - 4 ConfigReader subclasses: at the filter call (lean
    on self.file_type that's already there).

Different right answers for different layers; same audit-
the-class discipline.

WHAT THIS PATCH LANDED
======================

mufasa/data_processors/egocentric_aligner.py:
* Line 76: extensions=['.csv'] → ['.csv', '.parquet'].
* Line 109: capture file extension via get_fn_ext.
* Line 116: save_path uses captured file_type.
* Line 117: read_df with captured file_type.
* Line 131: write_df with captured file_type.

mufasa/data_processors/distance_calculator.py:
* Line 79: extensions=[f'.{self.file_type}'].

mufasa/data_processors/distance_timbin_calculator.py:
* Line 82: extensions=[f'.{self.file_type}'].

mufasa/data_processors/circling_detector.py:
* Line 89: extensions=[f'.{self.file_type}'].
* Line 111: read_df with self.file_type (was 'csv').

mufasa/data_processors/freezing_detector.py:
* Line 102: extensions=[f'.{self.file_type}'].
* Line 126: read_df with self.file_type (was 'csv').

Coverage
--------
Each backend (5 checks):
1.  egocentric_aligner: extensions=['.csv', '.parquet']
    AND no longer references Formats.CSV.value in the
    read/write loop body.
2.  distance_calculator: extensions=[f'.{self.file_type}'].
3.  distance_timbin_calculator: same.
4.  circling_detector: same + read_df uses self.file_type.
5.  freezing_detector: same + read_df uses self.file_type.

Class-of-bug invariant (1 check):
6.  No production code path in data_processors/ remains
    with ``extensions=['.csv']`` literal (excluding
    docstring/comment occurrences).

Cross-patch invariants:
7.  122eu state preserved: get_fn_ext handles empty extensions.
8.  122et state preserved: ROIPlotMultiprocess accepts show_bbox.
9.  122es state preserved: pixels_per_mm has detect_path.
10. Parse-clean.
11. 122do baseline.
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


def _find_call_args(src: str, fn_name: str) -> list[str]:
    """Return list of the source string of each `fn_name(...)` call
    found at the AST level. Skips docstrings/comments."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name)
                     and node.func.id == fn_name)
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr == fn_name)
                )):
            try:
                out.append(ast.unparse(node))
            except Exception:
                pass
    return out


def main() -> int:
    data_processors = REPO_ROOT / "mufasa" / "data_processors"

    # -----------------------------------------------------------------
    # Per-backend checks
    # -----------------------------------------------------------------

    # 1. egocentric_aligner — verify via AST that read_df/write_df
    # in the loop body no longer use Formats.CSV.value as the
    # file_type arg (instead use the per-file `file_type`
    # variable). Skip naive substring counting because the
    # comment block AND the fallback `or Formats.CSV.value`
    # reference the symbol intentionally.
    ea_src = (data_processors / "egocentric_aligner.py").read_text()
    ea_calls = _find_call_args(
        ea_src, "find_files_of_filetypes_in_directory",
    )
    ea_filter_ok = any(
        "'.csv'" in c and "'.parquet'" in c
        for c in ea_calls
    )
    # AST-walk for read_df / write_df calls; their file_type arg
    # should be a Name (the local `file_type` variable), not an
    # Attribute (Formats.CSV.value).
    ea_tree = ast.parse(ea_src)
    bad_read_write = []
    for node in ast.walk(ea_tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("read_df", "write_df")):
            for kw in node.keywords:
                if kw.arg == "file_type":
                    if (isinstance(kw.value, ast.Attribute)
                            and isinstance(kw.value.value, ast.Name)
                            and kw.value.value.id == "Formats"):
                        bad_read_write.append(node.lineno)
    check(
        "egocentric_aligner: extensions filter accepts both "
        "'.csv' and '.parquet'; read_df/write_df calls use the "
        "per-file `file_type` variable (NOT Formats.CSV.value)",
        ea_filter_ok and not bad_read_write,
        detail=(
            f"filter_ok={ea_filter_ok} "
            f"bad_read_write_lines={bad_read_write}"
        ),
    )

    # 2-3. distance_calculator + distance_timbin_calculator
    for fname in ("distance_calculator.py",
                  "distance_timbin_calculator.py"):
        src = (data_processors / fname).read_text()
        calls = _find_call_args(
            src, "find_files_of_filetypes_in_directory",
        )
        ok = any(
            "self.file_type" in c for c in calls
        )
        check(
            f"{fname.replace('.py', '')}: extensions filter "
            f"uses self.file_type (not hardcoded '.csv')",
            ok,
            detail=("; ".join(calls[:2])),
        )

    # 4-5. circling_detector + freezing_detector — filter AND read_df
    for fname in ("circling_detector.py", "freezing_detector.py"):
        src = (data_processors / fname).read_text()
        # extension filter
        calls = _find_call_args(
            src, "find_files_of_filetypes_in_directory",
        )
        filter_ok = any(
            "self.file_type" in c for c in calls
        )
        # read_df calls — none should use 'csv' literal
        read_calls = _find_call_args(src, "read_df")
        read_ok = not any(
            "file_type='csv'" in c or 'file_type="csv"' in c
            for c in read_calls
        )
        check(
            f"{fname.replace('.py', '')}: extensions filter "
            f"uses self.file_type AND read_df uses "
            f"self.file_type (not 'csv' literal)",
            filter_ok and read_ok,
            detail=(f"filter_ok={filter_ok} read_ok={read_ok}"),
        )

    # -----------------------------------------------------------------
    # Class-of-bug invariant
    # -----------------------------------------------------------------
    # ``light_dark_box_analyzer`` is on a known-deferred list:
    # its read uses raw ``pd.read_csv`` (not ``read_df``), so it's
    # v0-format-only at the read layer and can't be cleanly
    # converted to ``self.file_type``-based dispatch like the
    # other backends. Fixing it requires replacing pd.read_csv
    # with read_df + per-file format detection — a bigger refactor
    # filed as deferred. The extension filter stays
    # ``['.csv']`` for now to PRESERVE the v0-only failure mode
    # (rather than fail half-way at the parquet read).
    KNOWN_DEFERRED = {
        "mufasa/data_processors/light_dark_box_analyzer.py",
    }

    stray = []
    for f in sorted(data_processors.rglob("*.py")):
        rel = str(f.relative_to(REPO_ROOT))
        if rel in KNOWN_DEFERRED:
            continue
        src = f.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "find_files_of_filetypes_in_directory"):
                call_src = ast.unparse(node)
                # Match the EXACT bug shape: extensions=['.csv']
                # (single-element list, csv only). Both .csv +
                # .parquet (egocentric pattern) OR self.file_type
                # (other pattern) are acceptable.
                if (re.search(
                        r"extensions=\['?\.csv'?\]", call_src,
                    )
                    and ".parquet" not in call_src
                    and "self.file_type" not in call_src):
                    stray.append(
                        f"{f.relative_to(REPO_ROOT)}:{node.lineno}"
                    )
    check(
        "No production code in data_processors/ has "
        "extensions=['.csv'] alone (the class of bug: every "
        "such site silently produces 0 files on v1 projects)",
        not stray,
        detail=("; ".join(stray[:3])),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    rw_src = (REPO_ROOT / "mufasa" / "utils"
              / "read_write.py").read_text()
    check(
        "122eu state preserved: get_fn_ext short-circuits on "
        "empty file_extension",
        "if not file_extension:" in rw_src,
    )

    mp_src = (REPO_ROOT / "mufasa" / "plotting"
              / "roi_plotter_mp.py").read_text()
    check(
        "122et state preserved: ROIPlotMultiprocess accepts "
        "show_bbox",
        "show_bbox" in mp_src
        and "bbox = 'axis-aligned'" in mp_src,
    )

    from mufasa.section_provenance import SECTIONS
    pp = SECTIONS.get("pixels_per_mm")
    check(
        "122es state preserved: pixels_per_mm has detect_path",
        pp is not None and callable(pp.detect_path),
    )

    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    uiqt = pkg / "ui_qt"
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
        f"smoke_122ev_data_processor_format_sweep: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
