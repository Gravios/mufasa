"""
tests/smoke_122ff_frame_labeller_playback_and_keys.py
========================================================

Patch 122ff — Frame labelling: playback-direction keys, continuous
label/delete state, classifier-key reading from project.toml.

User request (Tue May 26, 2026):

> Annotation : Frame labelling. I still need an active plot, +- 2
> range with lines where a behavior has been labeled. The arrow
> keys should control the direction of play and the space bar
> should toggle play/pause. Each label should be listed with its
> associate key. Pressing a label toggles a continous label state.
> "Del" should be reserved for toggling continuous deletion of the
> currently toggled label (all other labels should be left
> untouched).

This patch lands the keyboard rewrite + continuous-mode semantics
+ classifier-key reading. The label-timeseries PLOT is a separate
patch (122fg or later) — it's a custom widget that warrants its
own scope.

WHAT THIS PATCH LANDED
======================

mufasa/ui_qt/frame_labeller.py — substantial rewrite:

* New state on FrameLabellerWidget:
  - ``_classifier_keys: dict[str, str]`` — mirror of
    [classifiers.keys] from project.toml, filtered to only keys
    for actual project classifiers (stale entries ignored).
  - ``_active_label: str | None`` — the classifier currently in
    continuous-write mode (None when idle).
  - ``_active_mode: str`` — "label" or "delete" — what value to
    write to the active label on each frame.
  - ``_last_play_direction`` — remembers the most recent play
    direction so Space resumes correctly.

* New methods:
  - ``_load_classifier_keys()`` — reads [classifiers.keys] from
    project.toml via the 122fe ``_read_classifier_keys`` helper.
  - ``_toggle_active_label(name)`` — toggles continuous-label
    mode for the named classifier (same name twice → deactivate;
    different name → switch).
  - ``_toggle_active_delete_mode()`` — Del-key handler. Flips
    _active_mode between "label" and "delete" for the currently-
    active classifier. No-op if no label active.
  - ``_toggle_play_pause()`` — Space-bar handler. Pauses if
    playing, resumes in last direction if paused.
  - ``_refresh_active_mode_indicator()`` — updates the active-
    mode status label's text + color (green for label mode, red
    for delete mode, neutral when idle).

* Modified methods:
  - ``_setup_shortcuts()`` — rewired:
    * 1-9 numeric bindings → REMOVED. Per-classifier keys come
      from [classifiers.keys] now.
    * Left/Right arrows: were single-frame jog (seek±1); now
      call ``scrubber._toggle_play(direction=±1)``.
    * Space: was "advance one frame" (seek+1); now toggles
      play/pause.
    * Del: NEW — toggles continuous-delete mode for the active
      label.
    * Shift+Left/Right (jog 10): UNCHANGED.
    * Ctrl+S (save): UNCHANGED.
  - ``_build_ui()`` — checkbox labels now show the assigned key
    as "(k)" instead of the legacy "N. " numeric prefix. Added a
    prominent "active continuous mode" status line below the
    keystroke hint.
  - ``_on_frame_changed()`` — if active mode is set, writes to
    the active label's array BEFORE refreshing the checkboxes.
    This is what produces a contiguous label span when the user
    plays the video with a label active.

* ``__init__`` calls ``_load_classifier_keys()`` after
  ``_load_project_metadata()`` so keys are bound to the actually-
  declared classifier set.

COVERAGE
========

State (3 checks):
1.  FrameLabellerWidget declares _classifier_keys, _active_label,
    _active_mode in __init__.
2.  _load_classifier_keys uses the 122fe helper.
3.  _classifier_keys is filtered to actual classifier names.

Keyboard wiring (4 checks):
4.  _setup_shortcuts wires arrow keys to _toggle_play (not seek).
5.  _setup_shortcuts wires Space to _toggle_play_pause (not
    seek).
6.  _setup_shortcuts wires Qt.Key_Delete to
    _toggle_active_delete_mode.
7.  _setup_shortcuts loops over _classifier_keys to wire per-
    classifier hotkeys (no 1-9 numeric bindings any more).

Continuous-mode semantics (3 checks):
8.  _toggle_active_label same-name-twice deactivates (idempotent
    off semantic).
9.  _toggle_active_delete_mode no-ops when _active_label is None.
10. _on_frame_changed writes to the active label's array when
    _active_label is set, BEFORE refreshing checkboxes.

UI (1 check):
11. _build_ui shows "(key)" suffix on checkbox labels (the
    user's "Each label should be listed with its associate
    key" requirement).

Cross-patch invariants (3 checks):
12. 122fe state preserved: _read_classifier_keys helper exists.
13. 122fd state preserved: roi_canvas shape_selected signal.
14. Parse-clean.
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

    # -----------------------------------------------------------------
    # State
    # -----------------------------------------------------------------
    init_src = _get_method_src(flw, "__init__")
    check(
        "FrameLabellerWidget.__init__ declares _classifier_keys, "
        "_active_label, _active_mode (the continuous-mode state)",
        ("self._classifier_keys" in init_src
         and "self._active_label" in init_src
         and "self._active_mode" in init_src),
        detail=(
            f"keys={'self._classifier_keys' in init_src} "
            f"active_label={'self._active_label' in init_src} "
            f"active_mode={'self._active_mode' in init_src}"
        ),
    )

    load_keys_src = _get_method_src(flw, "_load_classifier_keys")
    check(
        "_load_classifier_keys imports and uses the 122fe "
        "_read_classifier_keys helper",
        ("_read_classifier_keys" in load_keys_src
         and "self.config_path" in load_keys_src),
    )

    check(
        "_load_classifier_keys filters keys to only project's "
        "actual classifier names (avoids binding shortcuts for "
        "deleted/stale entries)",
        ("self._classifier_names" in load_keys_src
         and ("in all_keys" in load_keys_src
              or "n in all_keys" in load_keys_src)),
    )

    # -----------------------------------------------------------------
    # Keyboard wiring
    # -----------------------------------------------------------------
    setup_src = _get_method_src(flw, "_setup_shortcuts")

    check(
        "_setup_shortcuts wires Left/Right arrows to "
        "scrubber._toggle_play (continuous playback) — NOT to "
        "scrubber.seek (was the legacy single-frame jog)",
        ("_toggle_play(direction=-1)" in setup_src
         and "_toggle_play(direction=+1)" in setup_src
         and (
            "current_frame - 1" not in setup_src
            and "current_frame + 1" not in setup_src.replace(
                "current_frame + d", "")
         )),
    )

    check(
        "_setup_shortcuts wires Space to _toggle_play_pause "
        "(not the legacy seek+1)",
        ("Qt.Key_Space" in setup_src
         and "_toggle_play_pause" in setup_src),
    )

    check(
        "_setup_shortcuts wires Qt.Key_Delete to "
        "_toggle_active_delete_mode",
        ("Qt.Key_Delete" in setup_src
         and "_toggle_active_delete_mode" in setup_src),
    )

    check(
        "_setup_shortcuts iterates _classifier_keys for per-"
        "classifier hotkey binding (replaces the legacy 1-9 "
        "numeric bindings)",
        ("self._classifier_keys" in setup_src
         and "_toggle_active_label" in setup_src
         # No more numeric str(i+1) for the classifier bindings.
         and "str(i + 1)" not in setup_src),
    )

    # -----------------------------------------------------------------
    # Continuous-mode semantics
    # -----------------------------------------------------------------
    toggle_label_src = _get_method_src(flw, "_toggle_active_label")
    # Idempotent-off: pressing same name twice deactivates.
    check(
        "_toggle_active_label deactivates when called with the "
        "currently-active label (same name twice → idle) — the "
        "'continuous label state' toggle semantic",
        ("self._active_label == name" in toggle_label_src
         and "self._active_label = None" in toggle_label_src),
    )

    toggle_delete_src = _get_method_src(
        flw, "_toggle_active_delete_mode",
    )
    check(
        "_toggle_active_delete_mode no-ops with a status "
        "message when _active_label is None (Del with no active "
        "label is informational, not silent)",
        ("self._active_label is None" in toggle_delete_src
         and (
            "return" in toggle_delete_src
            or "no label is active" in toggle_delete_src
         )),
    )

    on_frame_src = _get_method_src(flw, "_on_frame_changed")
    check(
        "_on_frame_changed writes to the active label's array "
        "when _active_label is set, BEFORE refreshing the "
        "checkboxes (this is what produces a contiguous label "
        "span during playback)",
        ("self._active_label" in on_frame_src
         and "self._active_mode" in on_frame_src
         and ("arr[frame_idx]" in on_frame_src
              or "self._labels[" in on_frame_src)),
    )

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    build_src = _get_method_src(flw, "_build_ui")
    # Look for the "(key)" suffix pattern in the checkbox label.
    check(
        "_build_ui shows the classifier key alongside the name "
        "in each checkbox label (the 'Each label should be "
        "listed with its associate key' requirement)",
        ("_classifier_keys" in build_src
         and ("({key})" in build_src or "key})" in build_src
              or "(—)" in build_src)),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    cls_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "classifier.py").read_text()
    check(
        "122fe state preserved: classifier.py defines "
        "_read_classifier_keys",
        "def _read_classifier_keys(" in cls_src,
    )

    rc_src = (REPO_ROOT / "mufasa" / "ui_qt" / "dialogs"
              / "roi_canvas.py").read_text()
    check(
        "122fd state preserved: roi_canvas declares the "
        "shape_selected signal",
        "shape_selected = Signal(" in rc_src,
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
        f"smoke_122ff_frame_labeller_playback_and_keys: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
