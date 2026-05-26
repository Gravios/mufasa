"""
tests/smoke_122et_roi_plotter_mp_kwargs.py
=============================================

Patch 122et-hotfix: ``ROIPlotMultiprocess.__init__()`` got an
unexpected keyword argument ``'show_bbox'``.

User report (Mon May 25, 2026, third report of the day):

> Another error in roi : visualize : [screenshot of the
> Visualize ROI on video dialog with "Show bounding boxes"
> checked, Worker cores: 4, crashing with the title-cased
> error above].

Root cause
----------
The Qt form (``mufasa/ui_qt/forms/roi.py:_run``) calls one of
two sibling backends depending on the user-chosen worker
count:

* ``cores == 1``: ``ROIPlotter`` (single-core).
* ``cores > 1``:  ``ROIPlotMultiprocess`` (the MP variant).

The form was written assuming the two backends had identical
kwargs. They don't:

* ``ROIPlotter.__init__`` accepts ``show_bbox: bool = False``
  — a simple binary toggle.
* ``ROIPlotMultiprocess.__init__`` accepts ``bbox:
  Optional[Literal['axis-aligned', 'animal-aligned']] = None``
  — a richer enum. No ``show_bbox`` kwarg AT ALL.

So the form's binary checkbox worked on single-core, raised
TypeError on multi. The user picked 4 workers → crash.

The fix
-------
Add ``show_bbox: bool = False`` to ROIPlotMultiprocess'
signature AS AN ADDITIONAL kwarg (not replacing ``bbox``).
Inside ``__init__``, bridge: when ``bbox is None and
show_bbox``, set ``bbox = 'axis-aligned'``. Explicit
``bbox=...`` always wins over ``show_bbox`` — preserves the
richer enum for script-level callers who want
animal-aligned bboxes.

This restores API symmetry between the two backends
without forcing script-level callers to switch to the bool.

CLASS-OF-BUG AUDIT
==================

Same shape of bug: sibling pairs in ``mufasa/plotting/``
where ``<name>.py`` and ``<name>_mp.py`` have drifted kwargs.
Quick AST audit across all 13 pairs revealed THREE pairs
with this shape of drift:

1. **roi_plotter** (the user's bug): ``show_bbox`` (single)
   ↔ ``bbox`` enum (MP). **Live** — form path triggers it.
   **Fixed by this patch.**

2. **plot_clf_results**: ``show_bbox``, ``print_timers``
   (single) ↔ ``bbox``, ``print_timer`` (MP). Plural-vs-
   singular on the timer kwarg AND bbox kwarg.
   **Dormant** — no form calls these with the bad kwargs
   currently. Drift exists but isn't triggered by user
   action. Pinned in this smoke as known drift.

3. **single_run_model_validation_video**:
   ``show_animal_bounding_boxes`` (single) ↔ ``bbox`` (MP).
   **Dormant** — same situation.

The other 10 sibling pairs have only LEGITIMATE differences
(``core_cnt``, ``gpu``, ``verbose``, ``time_slice``) —
multiprocess-specific kwargs that aren't relevant to single-
core. Those are by design.

The two dormant pairs are NOT fixed by this patch because
the fix would require touching backend signatures with no
user-visible payoff (no form currently triggers them). They
ARE pinned in this smoke so:
- if a future form adds a UI that calls them with the bad
  kwargs, the smoke catches it at commit time;
- if a future patch tries to "fix" the drift by removing
  the existing enum-style API, the smoke catches the
  signature regression.

Coverage
--------
The user's bug fix (3 checks):
1.  ROIPlotMultiprocess.__init__ now accepts ``show_bbox``.
2.  ROIPlotMultiprocess.__init__ still accepts ``bbox``
    (the richer enum is preserved for script callers).
3.  ROIPlotMultiprocess.__init__ body has the bridge line
    (``bbox = 'axis-aligned'`` when show_bbox True and bbox
    None).

API symmetry (1 check):
4.  ROIPlotter.__init__ still accepts ``show_bbox`` (was
    always there; pin to prevent removal regression).

Dormant drift pins (2 checks):
5.  plot_clf_results MP class accepts ``bbox`` (NOT
    ``show_bbox``) and ``print_timer`` (NOT
    ``print_timers``) — confirms the dormant drift is
    where 122et's audit found it; if a fix lands later
    this check flips.
6.  single_run_model_validation_video MP class accepts
    ``bbox`` (NOT ``show_animal_bounding_boxes``) —
    same.

Cross-patch invariants:
7.  122es state preserved: pixels_per_mm has detect_path.
8.  122er state preserved: get_roi_data uses safe helpers.
9.  122en state preserved: v1_project_paths canonical helper.
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


def _init_kwargs(path: Path, class_name: str) -> tuple[set[str], str]:
    """Return (kwarg set, __init__ body src) for the named class."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return (set(), "")
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for m in cls.body:
                if (isinstance(m, ast.FunctionDef)
                        and m.name == "__init__"):
                    kws = {arg.arg for arg in m.args.args}
                    kws |= {arg.arg for arg in m.args.kwonlyargs}
                    kws.discard("self")
                    return (kws, ast.unparse(m))
    return (set(), "")


def main() -> int:
    plotting = REPO_ROOT / "mufasa" / "plotting"

    mp_kws, mp_body = _init_kwargs(
        plotting / "roi_plotter_mp.py", "ROIPlotMultiprocess",
    )
    sb_kws, _ = _init_kwargs(
        plotting / "roi_plotter.py", "ROIPlotter",
    )

    # 1. show_bbox in MP.
    check(
        "ROIPlotMultiprocess.__init__ now accepts `show_bbox` "
        "(122et-hotfix added it — was the kwarg the user's "
        "form was passing in)",
        "show_bbox" in mp_kws,
        detail=(f"kwargs: {sorted(mp_kws)[:5]}..."),
    )

    # 2. bbox enum still in MP.
    check(
        "ROIPlotMultiprocess.__init__ still accepts `bbox` "
        "(the richer Literal enum — script-level callers "
        "who want 'animal-aligned' bboxes can still ask for "
        "them)",
        "bbox" in mp_kws,
    )

    # 3. Bridge line present.
    check(
        "ROIPlotMultiprocess.__init__ body bridges "
        "`show_bbox=True` to `bbox='axis-aligned'` when bbox "
        "is unspecified (the actual fix logic)",
        "bbox = 'axis-aligned'" in mp_body
        and "show_bbox" in mp_body,
    )

    # 4. Single-core API symmetry.
    check(
        "ROIPlotter.__init__ still accepts `show_bbox` "
        "(was already there pre-122et; pin to prevent "
        "future removal regression)",
        "show_bbox" in sb_kws,
    )

    # -----------------------------------------------------------------
    # Dormant drift pins
    # -----------------------------------------------------------------
    pcr_mp_kws, _ = _init_kwargs(
        plotting / "plot_clf_results_mp.py",
        "PlotSklearnResultsMultiProcess",
    )
    check(
        "plot_clf_results_mp.PlotSklearnResultsMultiProcess "
        "uses `bbox` and `print_timer` (NOT show_bbox / "
        "print_timers — dormant drift from the single-core "
        "sibling; no form path triggers this currently, but "
        "if one does the bug is the same shape as 122et's)",
        ("bbox" in pcr_mp_kws
         and "print_timer" in pcr_mp_kws
         and "show_bbox" not in pcr_mp_kws
         and "print_timers" not in pcr_mp_kws),
    )

    val_mp_kws, _ = _init_kwargs(
        plotting / "single_run_model_validation_video_mp.py",
        "ValidateModelOneVideoMultiprocess",
    )
    check(
        "single_run_model_validation_video_mp."
        "ValidateModelOneVideoMultiprocess uses `bbox` (NOT "
        "show_animal_bounding_boxes — dormant drift, same "
        "class of bug)",
        ("bbox" in val_mp_kws
         and "show_animal_bounding_boxes" not in val_mp_kws),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    from mufasa.section_provenance import SECTIONS
    pp = SECTIONS.get("pixels_per_mm")
    check(
        "122es state preserved: pixels_per_mm has detect_path",
        pp is not None and callable(pp.detect_path),
    )

    ru_src = (REPO_ROOT / "mufasa" / "roi_tools"
              / "roi_utils.py").read_text()
    check(
        "122er state preserved: get_roi_data uses safe helpers",
        "safe_filter_by_video" in ru_src
        and "safe_videos_in" in ru_src,
    )

    pl_src = (REPO_ROOT / "mufasa"
              / "project_layout.py").read_text()
    check(
        "122en state preserved: v1_project_paths canonical helper",
        "def v1_project_paths" in pl_src,
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
        f"smoke_122et_roi_plotter_mp_kwargs: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
