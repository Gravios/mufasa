"""
tests/smoke_122fg_label_timeseries_plot.py
==============================================

Patch 122fg — Label timeseries plot widget + integration into
FrameLabellerWidget.

User request (Tue May 26, 2026):

> Annotation : Frame labelling. I still need an active plot,
> +- 2 range with lines where a behavior has been labeled.

122ff shipped the playback-direction keys + continuous label/
delete state. This patch completes the user's annotation
redesign with the visual piece: a ±2-second window of label
state around the current frame, repainting as the playhead
moves.

WHAT THIS PATCH LANDED
======================

mufasa/ui_qt/label_timeseries_plot.py — NEW widget.

* ``LabelTimeseriesPlot(QWidget)`` — render-only widget. Owns
  no playback state or label data; the host feeds it three
  setters:
  - ``set_labels(labels, classifier_names, classifier_keys)`` —
    one-shot on load.
  - ``set_fps(fps)`` — one-shot on video load.
  - ``set_current_frame(idx)`` — once per frame_changed event.
  Plus an optional ``set_window_seconds(s)`` (default 2.0).
* Custom paintEvent renders one horizontal lane per classifier.
  Spans where ``labels[name][i] == 1`` are filled rectangles in
  the lane's colour (from _LANE_COLORS palette, cycling).
* Vertical "now" cursor (red, 2px) at current frame position.
* Status row at the bottom: ``frame N · window ±2s · fps 30``.
* No matplotlib dependency — pure QPainter. Cheap, no extra
  install footprint.

mufasa/ui_qt/frame_labeller.py — INTEGRATION.

* New attribute ``self.timeseries_plot`` constructed in
  ``_build_ui`` between the scrubber and the classifier-checkbox
  bar (so the timeseries appears immediately under the video,
  above the per-frame toggles).
* In ``__init__`` after ``_initialize_labels`` + ``scrubber.load``:
  - ``timeseries_plot.set_labels(self._labels, ...)`` once.
  - ``timeseries_plot.set_fps(self.scrubber.fps)`` once.
* In ``_on_frame_changed``:
  - ``timeseries_plot.set_current_frame(frame_idx)`` on every
    frame change (i.e., during scrubbing, single-frame seeks,
    AND continuous playback driven by 122ff's arrow keys).

The plot widget repaints whenever any of its three setters
fires — Qt's update() handles batching at the event-loop level.

COVERAGE
========

Widget shape (5 checks):
1.  LabelTimeseriesPlot class exists in label_timeseries_plot.py.
2.  Declares the public setter API:
    set_labels, set_fps, set_current_frame, set_window_seconds.
3.  Default _window_seconds == 2.0 (the user-requested ±2 range).
4.  paintEvent method exists (custom rendering).
5.  _LANE_COLORS palette defined (multiple colours, cycling).

Integration (4 checks):
6.  FrameLabellerWidget._build_ui imports LabelTimeseriesPlot
    and constructs self.timeseries_plot.
7.  FrameLabellerWidget.__init__ calls
    timeseries_plot.set_labels AND timeseries_plot.set_fps.
8.  FrameLabellerWidget._on_frame_changed calls
    timeseries_plot.set_current_frame.
9.  The plot is built BEFORE the clf_bar (visual ordering:
    scrubber on top, plot below it, then checkboxes).

Cross-patch invariants (3 checks):
10. 122ff state preserved: FrameLabellerWidget declares
    _active_label and _active_mode.
11. 122fe state preserved: _read_classifier_keys exists.
12. Parse-clean.
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


def _get_method_src(
    cls_node: ast.ClassDef, method_name: str,
) -> str:
    for m in cls_node.body:
        if isinstance(m, ast.FunctionDef) and m.name == method_name:
            return ast.unparse(m)
    return ""


def main() -> int:
    # -----------------------------------------------------------------
    # Widget shape
    # -----------------------------------------------------------------
    plot_path = (REPO_ROOT / "mufasa" / "ui_qt"
                 / "label_timeseries_plot.py")
    plot_src = plot_path.read_text()
    plot_tree = ast.parse(plot_src)

    plot_cls = None
    for node in ast.walk(plot_tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "LabelTimeseriesPlot"):
            plot_cls = node
            break
    check(
        "LabelTimeseriesPlot class exists in "
        "mufasa/ui_qt/label_timeseries_plot.py",
        plot_cls is not None,
    )

    methods = (
        {m.name for m in plot_cls.body
         if isinstance(m, ast.FunctionDef)}
        if plot_cls else set()
    )
    expected_api = {
        "set_labels", "set_fps", "set_current_frame",
        "set_window_seconds", "paintEvent",
    }
    missing = expected_api - methods
    check(
        "LabelTimeseriesPlot declares the public setter API: "
        "set_labels, set_fps, set_current_frame, "
        "set_window_seconds, plus paintEvent",
        not missing,
        detail=(f"missing: {sorted(missing)}"),
    )

    # Default window = 2.0s.
    init_src = (
        _get_method_src(plot_cls, "__init__") if plot_cls else ""
    )
    check(
        "LabelTimeseriesPlot defaults _window_seconds to 2.0 "
        "(the user's '+- 2 range' requirement)",
        "self._window_seconds = 2.0" in init_src
        or "self._window_seconds: float = 2.0" in init_src,
    )

    check(
        "LabelTimeseriesPlot has a paintEvent method "
        "(custom QPainter rendering — no matplotlib dep)",
        "paintEvent" in methods,
    )

    # Lane-colour palette.
    check(
        "_LANE_COLORS palette defined in label_timeseries_plot.py "
        "(per-lane colour cycling for multi-classifier projects)",
        "_LANE_COLORS" in plot_src
        and "Sequence" in plot_src,
    )

    # -----------------------------------------------------------------
    # Integration
    # -----------------------------------------------------------------
    fl_path = (REPO_ROOT / "mufasa" / "ui_qt"
               / "frame_labeller.py")
    fl_src = fl_path.read_text()
    fl_tree = ast.parse(fl_src)

    flw = None
    for node in ast.walk(fl_tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "FrameLabellerWidget"):
            flw = node
            break
    assert flw is not None, "FrameLabellerWidget class missing"

    build_src = _get_method_src(flw, "_build_ui")
    check(
        "FrameLabellerWidget._build_ui imports LabelTimeseriesPlot "
        "and constructs self.timeseries_plot",
        ("from mufasa.ui_qt.label_timeseries_plot import" in build_src
         and "self.timeseries_plot" in build_src
         and "LabelTimeseriesPlot(self)" in build_src),
    )

    init_src = _get_method_src(flw, "__init__")
    check(
        "FrameLabellerWidget.__init__ calls "
        "timeseries_plot.set_labels AND set_fps (one-shot wiring "
        "after _initialize_labels + scrubber.load)",
        ("self.timeseries_plot.set_labels(" in init_src
         and "self.timeseries_plot.set_fps(" in init_src),
    )

    on_frame_src = _get_method_src(flw, "_on_frame_changed")
    check(
        "FrameLabellerWidget._on_frame_changed calls "
        "timeseries_plot.set_current_frame (so the playhead "
        "cursor + window contents update as the user scrubs or "
        "plays)",
        "set_current_frame(frame_idx)" in on_frame_src,
    )

    # Ordering check: timeseries_plot constructed BEFORE clf_bar.
    plot_pos = build_src.find("self.timeseries_plot = ")
    clf_bar_pos = build_src.find("clf_bar = QHBoxLayout()")
    check(
        "_build_ui constructs self.timeseries_plot BEFORE the "
        "clf_bar (visual ordering: scrubber on top, plot below "
        "it, then per-frame checkboxes)",
        (plot_pos > 0 and clf_bar_pos > 0 and plot_pos < clf_bar_pos),
        detail=(
            f"plot_pos={plot_pos}, clf_bar_pos={clf_bar_pos}"
        ),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    check(
        "122ff state preserved: FrameLabellerWidget declares "
        "_active_label and _active_mode (the continuous-mode "
        "state)",
        ("self._active_label" in init_src
         and "self._active_mode" in init_src),
    )

    cls_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "classifier.py").read_text()
    check(
        "122fe state preserved: classifier.py defines "
        "_read_classifier_keys",
        "def _read_classifier_keys(" in cls_src,
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

    print(
        f"smoke_122fg_label_timeseries_plot: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
