"""
tests/smoke_122es_video_calibration_detect_path.py
=====================================================

Patch 122es-hotfix: a third real-world user report during
manual testing of the session-2 stack:

> Also Video calibration does not show green even after I
> saved the values in the table to csv.

The user expected the Preprocessing → Video Calibration
section to show CURRENT after saving the calibration table
to ``sources/video_info.csv``. It didn't, because the
``pixels_per_mm`` SectionSpec was twice marked "settings-only,
no on-disk artifact":

* In 122ei when the detect_path mechanism was introduced —
  pixels_per_mm was explicitly left without detect_path:
  "pure-settings section like Pixels-per-mm calibration that
  don't produce a file."
* In 122ep when 4 more sections gained detect_path —
  pixels_per_mm was again explicitly listed under "settings-
  only or user-picked save dir (3 — stay None)."

Both wrong. The Video Calibration form's Save action writes
to ``sources/video_info.csv`` — the canonical
``video_info_path`` from ``v1_project_paths`` (122en).
``video_info.csv`` IS the on-disk artifact.

THE LESSON HERE
---------------
This is the third time in session 2 the same shape of bug
surfaced (and the second time within the same FEATURE — the
detect_path infrastructure):

1. 122eh: "the right long-term fix is centralization, filed
   as deferred." Done in 122en. Took 5 patches.
2. 122ek: audited roi_utils.py but missed two functions.
   Fixed by 122er. Took 3 patches between miss and fix.
3. 122ei/122ep: marked pixels_per_mm "settings-only" twice.
   Real workflow has a file. Fixed by 122es. Took 2
   patches between miss and fix.

The pattern from 122er's commit message ("don't trust
'cleared by inspection'") generalizes further: don't trust
"settings-only" either. Real user workflows produce file
artifacts even for sections that LOOK pure-configurative.
When marking a section as no-detect-path, the burden of
proof is on the marker: name the alternative signal that
proves the section ran.

For pixels_per_mm, the alternative signal IS
``video_info.csv`` — present in the v1 layout, written by
the Save action, recoverable via filesystem evidence. No
mystery, just an audit oversight.

What this patch landed
----------------------
mufasa/section_provenance.py:

* ``pixels_per_mm`` SectionSpec gains:
  - ``detect_path=lambda root: root / "sources" / "video_info.csv"``
  - Inline comment block explaining the prior misjudgments
    in 122ei and 122ep, the user report that surfaced the
    fix, and the acceptable false-positive case (old
    uncalibrated project with the CSV present at
    creation-time mtime).

Reciprocal-tripwire flips (3):

* smoke_122ei_detect_path_fallback.py:
  - Check 6 was "SECTIONS['pixels_per_mm'].detect_path is
    None"; now "is callable" (post-122es flip).
  - Check 13 was "pixels_per_mm reads UNKNOWN regardless
    of filesystem state"; now "reads UNKNOWN when
    sources/video_info.csv is absent" (the file's presence
    is the signal).

* smoke_122ep_detect_path_coverage.py:
  - Removed pixels_per_mm from the "deliberately not wired"
    loop.
  - Coverage count: 8 → 9 of 11 ui_bound sections.

* smoke_122eq_dev_paths_sweep.py + smoke_122er_roi_audit_
  and_qt_enum.py:
  - "8 of 11" → "9 of 11" in their cross-patch invariant
    checks.

Coverage
--------
The new detect_path (3 checks):
1.  SECTIONS["pixels_per_mm"].detect_path is callable.
2.  It resolves to sources/video_info.csv (relative to project
    root).
3.  An empty project reads UNKNOWN; a project with
    sources/video_info.csv reads CURRENT.

Tripwire-flip verification (2 checks):
4.  smoke_122ei post-flip: check 6 verifies callable (not
    None).
5.  smoke_122ep post-flip: coverage count is 9 of 11.

Cross-patch invariants:
6.  122er state preserved: get_roi_data uses safe helpers.
7.  122en state preserved: v1_project_paths canonical helper.
8.  pixels_per_mm's section_title / page are unchanged from
    122el (the binding fix landed; this patch doesn't
    re-disturb it).
9.  Parse-clean.
10. 122do baseline.
"""
from __future__ import annotations

import ast
import re
import sys
import tempfile
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
    from mufasa.section_provenance import (
        SECTIONS, SectionStatus, get_status,
    )

    # -----------------------------------------------------------------
    # The new detect_path
    # -----------------------------------------------------------------
    pp = SECTIONS.get("pixels_per_mm")
    check(
        "SECTIONS['pixels_per_mm'].detect_path is callable "
        "(was None pre-122es; the user's manual-test report "
        "surfaced that video_info.csv IS the on-disk artifact)",
        pp is not None and callable(pp.detect_path),
    )

    sample_root = Path("/tmp/fake_project_root")
    if pp is not None and callable(pp.detect_path):
        resolved = pp.detect_path(sample_root)
        try:
            rel = resolved.relative_to(sample_root)
            rel_parts = rel.parts
        except ValueError:
            rel_parts = ()
        check(
            "pixels_per_mm.detect_path resolves to "
            "sources/video_info.csv (the canonical "
            "video_info_path from v1_project_paths)",
            rel_parts == ("sources", "video_info.csv"),
            detail=(f"got parts={rel_parts!r}"),
        )
    else:
        check("(detect_path not callable — skipping resolve check)",
              False)

    # 3. Functional.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "test"\n'
        )

        s_empty = get_status(str(cfg), "pixels_per_mm")
        empty_unknown = (s_empty == SectionStatus.UNKNOWN)

        (root / "sources").mkdir()
        (root / "sources" / "video_info.csv").write_text(
            "Video,fps,Resolution_width,Resolution_height,"
            "Distance_in_mm,pixels_per_mm\nvideo1,30,640,480,100,12.5\n"
        )
        s_present = get_status(str(cfg), "pixels_per_mm")
        present_current = (s_present == SectionStatus.CURRENT)

        check(
            "Functional: empty project → pixels_per_mm reads "
            "UNKNOWN; project with sources/video_info.csv → "
            "reads CURRENT (the user-reported workflow)",
            empty_unknown and present_current,
            detail=(f"empty={s_empty.value!r} "
                    f"present={s_present.value!r}"),
        )

    # -----------------------------------------------------------------
    # Tripwire-flip verification
    # -----------------------------------------------------------------
    ei_path = (REPO_ROOT / "tests"
               / "smoke_122ei_detect_path_fallback.py")
    ei_src = ei_path.read_text()
    check(
        "smoke_122ei was flipped post-122es: the pixels_per_mm "
        "check now asserts 'detect_path is callable' (was "
        "'is None')",
        ("callable(pp.detect_path)" in ei_src
         or "callable(spec.detect_path)" in ei_src),
    )

    ep_path = (REPO_ROOT / "tests"
               / "smoke_122ep_detect_path_coverage.py")
    ep_src = ep_path.read_text()
    check(
        "smoke_122ep was flipped post-122ez: coverage count "
        "is 10 of 11 (was 9 of 11 pre-122ez; 122ez wired "
        "egocentric.detect_path)",
        ("len(with_detect) == 10" in ep_src
         and "10 of 11" in ep_src),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    ru_src = (REPO_ROOT / "mufasa" / "roi_tools"
              / "roi_utils.py").read_text()
    check(
        "122er state preserved: get_roi_data uses safe_filter "
        "helpers",
        "safe_filter_by_video" in ru_src
        and "safe_filter_video_neq" in ru_src
        and "safe_videos_in" in ru_src,
    )

    pl_src = (REPO_ROOT / "mufasa"
              / "project_layout.py").read_text()
    check(
        "122en state preserved: v1_project_paths canonical "
        "helper (the source of the video_info_path key 122es "
        "matches against)",
        "def v1_project_paths" in pl_src
        and '"video_info_path"' in pl_src,
    )

    # 8. Binding from 122el unchanged.
    check(
        "122el binding preserved: pixels_per_mm.page is "
        "'Preprocessing' and section_title is "
        "'Video Calibration' (the rebinding that 122el did; "
        "122es only adds detect_path, doesn't disturb the "
        "binding)",
        (pp is not None
         and pp.page == "Preprocessing"
         and pp.section_title == "Video Calibration"),
    )

    # 9. Parse-clean.
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

    # 10. 122do baseline.
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
        f"smoke_122es_video_calibration_detect_path: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
