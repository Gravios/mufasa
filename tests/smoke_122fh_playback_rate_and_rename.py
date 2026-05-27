"""
tests/smoke_122fh_playback_rate_and_rename.py
================================================

Patch 122fh — Up/Down arrow controls playback rate + "Frame
labelling" → "Frame Labeling" rename.

User request (Wed May 27, 2026):

> Annotation : Frame labelling (should be Frame Labeling,
> Right?) the up and down arrows should control the frame rate.
> (perhaps +- 1/4 the current frame rate).

Two changes shipped together:

1. **Rename**. User-facing strings: "Frame labelling" → "Frame
   Labeling" (American spelling + title case). The class name
   ``FrameLabellingLauncher`` and module path
   ``mufasa.labelling.*`` keep their British spellings — renaming
   import paths is out of scope.

2. **Up/Down playback rate**. ×1.25 / ÷1.25 per press. Symmetric
   (Up then Down returns to native). Clamped to [1.0, 240.0] fps.

WHAT THIS PATCH LANDED
======================

mufasa/ui_qt/frame_scrubber.py::FrameScrubberWidget:

* New ``_playback_fps`` instance field. Distinct from ``_fps``
  (the video's NATIVE rate, used for time-label math). Starts
  equal to ``_fps`` on each load.
* Timer interval is now derived from ``_playback_fps`` not
  ``_fps``.
* New ``get_playback_fps()`` / ``set_playback_fps(new_fps)``.
  set_playback_fps clamps to [1.0, 240.0] and restarts the
  timer if currently playing so the change takes effect on the
  next tick.

mufasa/ui_qt/frame_labeller.py:

* New Qt.Key_Up and Qt.Key_Down shortcuts in
  ``_setup_shortcuts`` calling new ``_adjust_playback_fps``.
* ``_adjust_playback_fps(factor)`` multiplies the scrubber's
  playback FPS by ``factor`` and updates the status label with
  the new rate + percentage of native.
* Hint label updated to include the Up/Down shortcut.

mufasa/section_provenance.py:
* ``SECTIONS['annotation'].section_title``: 'Frame labelling'
  → 'Frame Labeling'.

mufasa/ui_qt/pages/annotation_page.py:
* ``add_section('Frame labelling', ...)`` → 'Frame Labeling'.

mufasa/ui_qt/forms/annotation.py:
* ``title = 'Frame labelling'`` → 'Frame Labeling'.
* MODES strings: 'New labelling' → 'New labeling', etc.
* "before labelling." message → "before labeling.".

mufasa/ui_qt/frame_labeller.py:
* Two user-visible messages ("…before labelling.") flipped.

mufasa/ui_qt/forms/classifier.py:
* Description: "Frame labelling" → "Frame Labeling".

WHAT IS *NOT* RENAMED
=====================

* ``FrameLabellingLauncher`` class name — would cascade through
  imports.
* ``mufasa.labelling.*`` module paths — same reason.
* Comments containing "labelling" — internal docs, low cost,
  left as-is unless natural to fix.

COVERAGE
========

Rename (4 checks):
1.  SECTIONS['annotation'].section_title == 'Frame Labeling'.
2.  annotation_page.py registers 'Frame Labeling' section.
3.  forms/annotation.py FrameLabellingLauncher.title is
    'Frame Labeling'.
4.  No "Frame labelling" string remains in any user-facing
    string position (string-literal or quote-wrapped).

Scrubber API (3 checks):
5.  FrameScrubberWidget declares _playback_fps in __init__.
6.  FrameScrubberWidget exposes get_playback_fps and
    set_playback_fps methods.
7.  set_playback_fps clamps to [1.0, 240.0] (the safety
    bounds documented in the patch).

FrameLabellerWidget wiring (3 checks):
8.  _setup_shortcuts wires Qt.Key_Up to _adjust_playback_fps
    with factor=1.25.
9.  _setup_shortcuts wires Qt.Key_Down to _adjust_playback_fps
    with factor=1/1.25 ≈ 0.8.
10. _adjust_playback_fps reads + writes via the scrubber's
    new playback-fps API (not by mutating private fields).

Cross-patch invariants (3 checks):
11. 122fg state preserved: LabelTimeseriesPlot is instantiated
    in FrameLabellerWidget._build_ui.
12. 122ff state preserved: _active_label and _active_mode
    declared in __init__.
13. Parse-clean.
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
    # Rename
    # -----------------------------------------------------------------
    from mufasa.section_provenance import SECTIONS
    a_spec = SECTIONS.get("annotation")
    check(
        "SECTIONS['annotation'].section_title == 'Frame Labeling' "
        "(American spelling + title case per user request)",
        a_spec is not None
        and a_spec.section_title == "Frame Labeling",
        detail=(
            f"got {getattr(a_spec, 'section_title', None)!r}"
        ),
    )

    ap_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
              / "annotation_page.py").read_text()
    check(
        "annotation_page.py registers section title "
        "'Frame Labeling' (rename per user request)",
        '"Frame Labeling"' in ap_src
        and '"Frame labelling"' not in ap_src,
    )

    an_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
              / "annotation.py").read_text()
    an_tree = ast.parse(an_src)
    flw_launcher = None
    for node in ast.walk(an_tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "FrameLabellingLauncher"):
            flw_launcher = node
            break
    title_value = None
    if flw_launcher is not None:
        for m in flw_launcher.body:
            if isinstance(m, ast.Assign):
                for tgt in m.targets:
                    if (isinstance(tgt, ast.Name)
                            and tgt.id == "title"
                            and isinstance(m.value, ast.Constant)):
                        title_value = m.value.value
    check(
        "FrameLabellingLauncher.title == 'Frame Labeling' "
        "(form-level title; class name itself unchanged to avoid "
        "import-path cascade)",
        title_value == "Frame Labeling",
        detail=(f"got {title_value!r}"),
    )

    # 4. No "Frame labelling" string remains in user-facing
    # string-LITERAL positions across the rename-targeted files.
    # We use a regex that catches the exact phrase in any
    # quoting style.
    user_facing_paths = [
        "mufasa/section_provenance.py",
        "mufasa/ui_qt/pages/annotation_page.py",
        "mufasa/ui_qt/forms/annotation.py",
        "mufasa/ui_qt/forms/classifier.py",
        "mufasa/ui_qt/frame_labeller.py",
    ]
    rename_violations = []
    for rel in user_facing_paths:
        src = (REPO_ROOT / rel).read_text()
        # The pattern catches any line where a non-comment context
        # uses the old phrase. We allow it in comments because the
        # patch documentation references the old name.
        for ln_no, line in enumerate(src.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Skip lines that are inside a docstring — heuristic:
            # if the line contains triple-quote, it's likely a
            # docstring marker; the surrounding lines may be doc
            # text. For simplicity we just check for the literal
            # phrase wrapped in quotes (string-literal) anywhere.
            if (re.search(r'"Frame labelling"', line)
                    or re.search(r"'Frame labelling'", line)):
                rename_violations.append(f"{rel}:{ln_no}")
    check(
        "No 'Frame labelling' string-literal remains in any "
        "rename-targeted source file (user-facing strings have "
        "all flipped to 'Frame Labeling')",
        not rename_violations,
        detail=(
            "; ".join(rename_violations[:3])
            if rename_violations else ""
        ),
    )

    # -----------------------------------------------------------------
    # Scrubber playback-fps API
    # -----------------------------------------------------------------
    sc_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "frame_scrubber.py").read_text()
    sc_tree = ast.parse(sc_src)
    scrubber_cls = None
    for node in ast.walk(sc_tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "FrameScrubberWidget"):
            scrubber_cls = node
            break
    init_src = (
        _get_method_src(scrubber_cls, "__init__")
        if scrubber_cls else ""
    )
    check(
        "FrameScrubberWidget.__init__ declares _playback_fps "
        "(distinct from _fps which holds the video's NATIVE "
        "rate)",
        "self._playback_fps" in init_src,
    )

    scrubber_methods = (
        {m.name for m in scrubber_cls.body
         if isinstance(m, ast.FunctionDef)}
        if scrubber_cls else set()
    )
    check(
        "FrameScrubberWidget exposes get_playback_fps + "
        "set_playback_fps (public API for adjusting playback "
        "rate without touching internals)",
        {"get_playback_fps", "set_playback_fps"}
        <= scrubber_methods,
        detail=(f"methods: {sorted(scrubber_methods)}"),
    )

    set_pb_src = _get_method_src(scrubber_cls, "set_playback_fps")
    has_clamp = (
        ("max(1.0" in set_pb_src or "max(1," in set_pb_src)
        and ("min(240" in set_pb_src or "240.0" in set_pb_src)
    )
    check(
        "set_playback_fps clamps to [1.0, 240.0] (safety bounds "
        "to keep timer math sane near Qt's 1ms resolution and "
        "to prevent < 1fps stalls)",
        has_clamp,
    )

    # -----------------------------------------------------------------
    # FrameLabellerWidget wiring
    # -----------------------------------------------------------------
    fl_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "frame_labeller.py").read_text()
    fl_tree = ast.parse(fl_src)
    flw = None
    for node in ast.walk(fl_tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "FrameLabellerWidget"):
            flw = node
            break
    setup_src = _get_method_src(flw, "_setup_shortcuts")
    check(
        "_setup_shortcuts wires Qt.Key_Up to "
        "_adjust_playback_fps(factor=1.25) — the ×1.25 Up step",
        "Qt.Key_Up" in setup_src
        and "factor=1.25" in setup_src,
    )

    check(
        "_setup_shortcuts wires Qt.Key_Down to "
        "_adjust_playback_fps(factor=1.0 / 1.25) — the symmetric "
        "÷1.25 Down step (Up + Down returns to original rate)",
        "Qt.Key_Down" in setup_src
        and "1.0 / 1.25" in setup_src,
    )

    adjust_src = _get_method_src(flw, "_adjust_playback_fps")
    check(
        "_adjust_playback_fps reads + writes via the scrubber's "
        "get_playback_fps / set_playback_fps API (not by mutating "
        "_playback_fps directly)",
        "get_playback_fps()" in adjust_src
        and "set_playback_fps(" in adjust_src,
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    build_src = _get_method_src(flw, "_build_ui")
    check(
        "122fg state preserved: LabelTimeseriesPlot is "
        "instantiated in FrameLabellerWidget._build_ui",
        "LabelTimeseriesPlot(self)" in build_src,
    )

    init_flw_src = _get_method_src(flw, "__init__")
    check(
        "122ff state preserved: _active_label and _active_mode "
        "declared in FrameLabellerWidget.__init__",
        ("self._active_label" in init_flw_src
         and "self._active_mode" in init_flw_src),
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
        f"smoke_122fh_playback_rate_and_rename: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
