"""
tests/smoke_122ez_egocentric_detect_path.py
==============================================

Patch 122ez — defensive ``detect_path`` for the ``egocentric``
section, pointing at the form's default save_dir
(``<project>/rotated/``).

User report (Tue May 26, 2026, follow-up to 122ex):

> Preprocessing: Egocentric alignment, saved to
> /data/testing/mufasa/test-20260427/rotated and contains the
> mp4 and parquet files but the badge is still white.

122ex wired ``section_id = "egocentric"`` on
``EgocentricAlignmentForm``. That alone should have made the
badge transition UNKNOWN → CURRENT via ``record_run``. The
user reports it didn't. Possible causes:
* User's running workbench has the old class definition cached
  (PySide6 process needs a restart after a pip install -e .
  for class-attribute changes to take effect).
* record_run path has a runtime issue specific to the user's
  environment (project.toml permissions, etc.).
* My 122ex patch didn't reach the user's runtime for some
  other reason.

This patch adds defense-in-depth: detect_path. The badge will
go CURRENT via filesystem evidence (files in
``<project>/rotated/``) regardless of whether record_run was
called.

WHY 122ep DELIBERATELY OMITTED IT
=================================

The 122ep audit (which doubled detect_path coverage from 5/11
to 9/11) explicitly skipped egocentric because the form's
``save_dir`` is user-picked, not a fixed
``derived/<stage>/<run_id>/`` convention. The reasoning was:
"if the user picks a non-default dir, detect_path can't find
their output, badge stays UNKNOWN forever — misleading."

The 122ez tradeoff: prefer the COMMON CASE going CURRENT
(default save_dir, which is what the user has) over the EDGE
CASE staying defensively UNKNOWN (user picked a non-default
dir). The form's default is
``<project_root>/rotated/`` per ``EgocentricAlignmentForm.build``
at line 928 of pose_cleanup.py. The user's actual save dir
matches.

For users who pick a non-default save_dir, the badge falls
back to record_run via 122ex's section_id wiring. Worst
case: UNKNOWN. Not strictly worse than before 122ez.

THE PATTERN — DETECT_PATH FOR USER-CHOSEN DIRS
================================================

This is the second user-picked save_dir section in the
codebase. The other is annotation (derived/labels/). 122ep
gave annotation a detect_path on ``derived/labels`` because
that's a known convention (not user-pickable in the form
sense — annotation outputs always go there).

Egocentric is a true user-pickable destination. The default
matches a known convention only by default. Worth noting:
the right pattern for future user-pickable producers is to
either:
  (a) constrain to a stable known location and add detect_path
      pointing at it, OR
  (b) rely on record_run via section_id (no detect_path) and
      accept the cached-class-definition risk.

122ez takes the implicit (a) for egocentric because the
default is well-known.

Coverage
--------
The fix (2 checks):
1.  SECTIONS["egocentric"].detect_path is callable.
2.  SECTIONS["egocentric"].detect_path(root) returns
    root / "rotated".

End-to-end (1 check):
3.  Synthetic project with files at <root>/rotated/* but NO
    [provenance.egocentric] block → badge reads CURRENT
    via detect_path (the user's exact scenario).

Edge case (1 check):
4.  Synthetic project with NO files at <root>/rotated/ and
    no provenance → badge reads UNKNOWN (the inverse — no
    false positive when egocentric never ran).

Dependency math (1 check):
5.  Synthetic project where outlier_correction's detect_path
    mtime is LATER than rotated/'s mtime → egocentric reads
    STALE (the staleness rule still composes correctly).

Cross-patch invariants (5 checks):
6.  122ey state preserved: classifier_page.py has 6 build
    functions.
7.  122ex state preserved: EgocentricAlignmentForm.section_id
    == "egocentric".
8.  122ew state preserved: get_all_statuses uses
    _resolve_run_at.
9.  Parse-clean.
10. 122do baseline.
"""
from __future__ import annotations

import ast
import re
import sys
import tempfile
import time
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
        SECTIONS, SectionStatus, get_status, get_all_statuses,
    )

    # 1. detect_path is callable.
    egospec = SECTIONS["egocentric"]
    check(
        "SECTIONS['egocentric'].detect_path is callable "
        "(was None pre-122ez)",
        callable(egospec.detect_path),
    )

    # 2. detect_path returns root / 'rotated'.
    test_root = Path("/fake/project")
    expected = test_root / "rotated"
    got = egospec.detect_path(test_root)
    check(
        "SECTIONS['egocentric'].detect_path(root) returns "
        "root / 'rotated' (the form's default save_dir)",
        Path(got) == expected,
        detail=(f"got {got}"),
    )

    # 3. End-to-end: rotated/ has files → CURRENT.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n'
        )
        # User's actual scenario: mp4 and parquet in rotated/
        (root / "rotated").mkdir()
        (root / "rotated" / "v1.mp4").write_text("video")
        (root / "rotated" / "v1.parquet").write_text("data")
        # outlier_correction has detect_path; must exist for STALE
        # math, but mtime EARLIER than the rotated/ files.
        (root / "derived" / "outlier_corrected"
         / "old-run").mkdir(parents=True)
        # Touch with an explicit older time
        old_file = (root / "derived" / "outlier_corrected"
                    / "old-run" / "v1.parquet")
        old_file.write_text("d")
        old_time = time.time() - 86400  # 1 day ago
        import os
        os.utime(old_file, (old_time, old_time))

        s = get_status(str(cfg), "egocentric")
        check(
            "End-to-end: project with files at <root>/rotated/* "
            "but no [provenance.egocentric] block reads CURRENT "
            "via the new detect_path (the user's exact scenario)",
            s == SectionStatus.CURRENT,
            detail=(f"got {s.value!r}"),
        )

    # 4. Inverse: no rotated/ → UNKNOWN.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n'
        )
        # NO rotated/ dir, no provenance.
        s = get_status(str(cfg), "egocentric")
        check(
            "Inverse: project without rotated/ dir AND no "
            "provenance reads UNKNOWN (no false positive when "
            "egocentric was never run)",
            s == SectionStatus.UNKNOWN,
            detail=(f"got {s.value!r}"),
        )

    # 5. Dependency math: outlier_correction newer → STALE.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n'
        )
        # rotated/ files exist with OLD mtime
        (root / "rotated").mkdir()
        rotated_file = root / "rotated" / "v1.parquet"
        rotated_file.write_text("data")
        old_time = time.time() - 86400
        import os
        os.utime(rotated_file, (old_time, old_time))

        # outlier_correction has a more recent mtime
        time.sleep(0.05)
        (root / "derived" / "outlier_corrected"
         / "recent-run").mkdir(parents=True)
        (root / "derived" / "outlier_corrected" / "recent-run"
         / "v1.parquet").write_text("d")

        s = get_status(str(cfg), "egocentric")
        check(
            "Dependency math: when outlier_correction's "
            "detect_path mtime is later than rotated/'s mtime, "
            "egocentric reads STALE (staleness rule composes "
            "correctly with detect_path)",
            s == SectionStatus.STALE,
            detail=(f"got {s.value!r}"),
        )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    cp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
              / "classifier_page.py").read_text()
    check(
        "122ey state preserved: classifier_page.py exposes "
        "build_manage_classifiers_page",
        "def build_manage_classifiers_page" in cp_src,
    )

    pc_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
              / "pose_cleanup.py").read_text()
    pc_tree = ast.parse(pc_src)
    egocentric_sid = None
    for cls in ast.walk(pc_tree):
        if (isinstance(cls, ast.ClassDef)
                and cls.name == "EgocentricAlignmentForm"):
            for m in cls.body:
                if isinstance(m, ast.Assign):
                    for tgt in m.targets:
                        if (isinstance(tgt, ast.Name)
                                and tgt.id == "section_id"
                                and isinstance(m.value, ast.Constant)):
                            egocentric_sid = m.value.value
    check(
        "122ex state preserved: EgocentricAlignmentForm."
        "section_id == 'egocentric'",
        egocentric_sid == "egocentric",
    )

    sp_src = (REPO_ROOT / "mufasa"
              / "section_provenance.py").read_text()
    check(
        "122ew state preserved: get_all_statuses delegates to "
        "_resolve_run_at",
        "_resolve_run_at" in sp_src,
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
        f"smoke_122ez_egocentric_detect_path: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
