"""
tests/smoke_122aw_labelling_consumer_migration.py
==================================================

Patch 122aw: migrate 6 labelling consumers to v1-aware reads via
:func:`mufasa.utils.classification_io.load_machine_results_for_video`.
Fourth patch in the machine_results migration arc, following
122at (open dual-write), 122au (analysis), 122av (visualization).

Consumers migrated in this patch
--------------------------------
1. labelling/labelling_interface.py — Tk pseudo-labelling path
2. labelling/standard_labeller.py — Tk standard labeller's
   pseudo-labelling path
3. ui/pop_ups/select_video_for_pseudo_labelling_popup.py — Tk
   popup that launches LabellingInterface in pseudo mode
4. ui_qt/clip_review.py — Qt clip review (loads predictions
   for the rating UI)
5. ui_qt/targeted_clips.py — Qt targeted-clips data slicing
6. ui_qt/frame_labeller.py — Qt frame labeller's pseudo-label
   seeding (had a stale docstring saying 'machine_results
   doesn't have a v1 derived/ location yet' which is fixed)

What's NOT migrated
-------------------
* labelling/labelling_advanced_interface.py and
  labelling/targeted_annotations_clips.py — they build the path
  string ``machine_results_file_path = os.path.join(...)`` but
  never actually call read_df on it. The path stays for
  downstream code that may reference it; no read site to migrate.

After this patch
----------------
All in-process readers of machine_results are on the v1 helper.
The remaining legacy reads are in the ``_mp.py`` multiprocessing
workers (~10 files across analysis + viz). Those need
``config_path`` threaded into the pickled worker args — separate
small lane.

Coverage
--------
1. Each migrated file imports load_machine_results_for_video.
2. Each migrated file records 122aw.
3. No remaining raw read_df(machine_results_file_path, ...) /
   read_df(mr_path, ...) / read_df(src, ...) calls in the
   migrated files for the machine_results read site.
4. frame_labeller's stale docstring is updated.
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
    # 1. Each migrated file imports the helper + records 122aw
    # ==================================================================
    migrated_files = [
        REPO_ROOT / "mufasa" / "labelling"
        / "labelling_interface.py",
        REPO_ROOT / "mufasa" / "labelling"
        / "standard_labeller.py",
        REPO_ROOT / "mufasa" / "ui" / "pop_ups"
        / "select_video_for_pseudo_labelling_popup.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "clip_review.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "targeted_clips.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "frame_labeller.py",
    ]
    for path in migrated_files:
        check(
            f"{path.name}: exists",
            path.is_file(),
        )
        if not path.is_file():
            continue
        src = path.read_text()
        check(
            f"{path.name}: imports load_machine_results_for_video",
            "load_machine_results_for_video" in src,
        )
        check(
            f"{path.name}: records 122aw",
            "122aw" in src,
        )

    # ==================================================================
    # 2. No remaining raw read_df calls for the machine_results read
    #    site. (Other read_df calls in the file — e.g. for
    #    features_extracted — are allowed; we filter by the var name.)
    # ==================================================================
    target_var_names = [
        "machine_results_file_path",
        "self.machine_results_file_path",
        "self.machine_results_path",
        "mr_path",
        # frame_labeller used `src` as the variable holding the
        # machine_results path; check that too
        "src",
    ]
    for path in migrated_files:
        if not path.is_file():
            continue
        src_text = path.read_text()
        offending = []
        for lineno, line in enumerate(src_text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Skip docstring-ish lines
            if stripped.startswith(('"', "'")):
                continue
            # Look for read_df calls whose first arg is one of the
            # legacy machine_results variable names.
            for var in target_var_names:
                # The literal substring `read_df(<var>` or
                # `read_df(<var>,` or `read_df(file_path=<var>`
                candidates = (
                    f"read_df({var}",
                    f"read_df({var},",
                    f"read_df(file_path={var}",
                )
                if any(c in stripped for c in candidates):
                    # `src` is also used in some files as a different
                    # variable; only count it for frame_labeller
                    if var == "src" and "frame_labeller" not in path.name:
                        continue
                    offending.append(
                        (lineno, line.strip(), var),
                    )
                    break
        check(
            f"{path.name}: no remaining raw read_df calls for "
            "machine_results read site",
            len(offending) == 0,
            detail=(
                "; ".join(f"L{n}: {l} [{v}]"
                          for n, l, v in offending)
                if offending else ""
            ),
        )

    # ==================================================================
    # 3. frame_labeller's stale docstring updated
    # ==================================================================
    fl_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "frame_labeller.py").read_text()
    check(
        "frame_labeller: stale 'machine_results doesn't have a v1 "
        "derived/ location yet' docstring is no longer present",
        "doesn't have a v1 derived/ location yet" not in fl_src,
    )

    # ==================================================================
    # 4. Migration arc consistency: classification_io still defines
    #    the helper (sanity)
    # ==================================================================
    try:
        from mufasa.utils.classification_io import (
            load_machine_results_for_video,
        )
        check(
            "load_machine_results_for_video still importable "
            "from classification_io",
            callable(load_machine_results_for_video),
        )
    except ImportError as exc:
        check(
            "load_machine_results_for_video still importable "
            "from classification_io",
            False, detail=str(exc),
        )

    print(
        f"smoke_122aw_labelling_consumer_migration: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
