"""
tests/smoke_122ds_section_provenance.py
=========================================

Patch 122ds: foundation for workflow-section provenance tracking.

Two pieces of new infrastructure:

1. ``mufasa.project_layout.publish_to_stage`` — atomic relative-
   symlink publish from one stage's run dir into another. Lets
   producers (Data Import, Interpolate, Kalman v2) make their
   output discoverable under ``derived/outlier_corrected/`` so
   the legacy downstream contract is satisfied without each
   backend having to be re-plumbed onto the input-source picker.

2. ``mufasa.section_provenance`` — new module:
     * SECTIONS DAG (14 sections; central declaration).
     * SectionSpec / SectionStatus types.
     * record_run / get_status / get_all_statuses helpers.
     * _validate_dag runs at import time and rejects cycles,
       unknown depends_on, self-references, and key/id mismatch.

Future patches (122dt = backend wiring, 122du = UI badges, 122dv =
remove Skip outlier correction) consume this foundation. They are
NOT in scope for 122ds.

Coverage
--------
publish_to_stage:
1.  Happy path: source exists, target doesn't, symlink is created.
2.  Symlink target is RELATIVE (not absolute) — survives project move.
3.  Idempotent re-publish: calling twice with same args is a no-op.
4.  Re-publish to a different source: atomic replace via temp+rename.
5.  Refuses to clobber a real directory at the target.
6.  Raises FileNotFoundError if the source run doesn't exist.
7.  Raises ValueError on stage names containing path separators.
8.  Symlink follows: globbing ``*.parquet`` through the link finds
    files in the underlying directory.

section_provenance — DAG declaration:
9.  Module imports cleanly (DAG validation passes on the shipped
    SECTIONS).
10. Every section's depends_on references a known section_id.
11. The DAG is acyclic.
12. _validate_dag rejects a synthetic cyclic graph.
13. _validate_dag rejects a synthetic self-reference.
14. _validate_dag rejects a synthetic unknown depends_on.

section_provenance — record_run + get_status:
15. record_run on a fresh project creates the [provenance.<id>]
    entry with correct fields.
16. record_run is round-trip-stable through read_project_toml /
    write_project_toml (file is still valid TOML, no other
    project.toml content is lost).
17. record_run raises KeyError on unknown section_id.
18. record_run with run_id=None records last_run_at only (settings-
    section case).
19. get_status returns UNKNOWN when project.toml has no provenance
    entry for the section.
20. get_status returns CURRENT when this section ran AFTER all its
    declared dependencies.
21. get_status returns STALE when a dependency ran AFTER this section.
22. get_status ignores UNKNOWN dependencies (a dependency with no
    provenance entry doesn't mark this section STALE).
23. get_all_statuses returns one entry per SECTIONS key.

section_provenance — robustness:
24. get_status soft-fails to UNKNOWN if project.toml is missing
    (UI tolerance).
25. get_status soft-fails to UNKNOWN on a malformed last_run_at
    string.

Project-wide:
26. All mufasa/**/*.py parse cleanly.
27. 122do baseline tripwire: no ``Optional[`` in non-docstring
    positions across mufasa/ui_qt/.
28. 122dr baseline preserved: latest_populated_run_or_parent still
    importable.
"""
from __future__ import annotations

import ast
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
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


def _make_project(td: Path) -> Path:
    """Drop a minimal valid v1 project.toml into ``td`` and return its path."""
    p = td / "project.toml"
    p.write_text(
        'project_layout_version = 1\n'
        'file_type = "parquet"\n'
    )
    return p


def main() -> int:
    from mufasa.project_layout import (
        latest_populated_run_or_parent,  # tripwire
        publish_to_stage,
        read_project_toml,
    )
    from mufasa.section_provenance import (
        SECTIONS,
        SectionSpec,
        SectionStatus,
        ProvenanceError,
        get_all_statuses,
        get_status,
        record_run,
        _validate_dag,
    )

    # -----------------------------------------------------------------
    # publish_to_stage
    # -----------------------------------------------------------------

    # 1. Happy path.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src_run = root / "derived" / "smoothed" / "20260520-120000-aabbcc"
        src_run.mkdir(parents=True)
        (src_run / "video1.parquet").write_text("data1")

        final = publish_to_stage(
            root, "smoothed", "outlier_corrected",
            "20260520-120000-aabbcc",
        )
        check(
            "publish_to_stage happy path: target is a symlink "
            "located under derived/<target_stage>/",
            final.is_symlink()
            and final.parent.name == "outlier_corrected"
            and final.name == "20260520-120000-aabbcc",
        )

        # 2. Relative target.
        link_target = os.readlink(final)
        check(
            "publish_to_stage creates a RELATIVE symlink "
            "(starts with '..'; survives project move)",
            link_target.startswith(".."),
            detail=f"got {link_target!r}",
        )

        # 8. Follows for glob.
        files_through_link = list(final.glob("*.parquet"))
        check(
            "Files in the source run are reachable via the published "
            "symlink (glob through the link finds them)",
            len(files_through_link) == 1
            and files_through_link[0].name == "video1.parquet",
        )

        # 3. Idempotent.
        final2 = publish_to_stage(
            root, "smoothed", "outlier_corrected",
            "20260520-120000-aabbcc",
        )
        check(
            "publish_to_stage is idempotent: second call with "
            "the same args is a no-op (same symlink path, still "
            "a symlink)",
            final2 == final and final2.is_symlink(),
        )

    # 4. Atomic replace when source changes.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Two source runs in different stages.
        smoothed_run = root / "derived" / "smoothed" / "20260520-120000-aabbcc"
        smoothed_run.mkdir(parents=True)
        (smoothed_run / "from_smoothed.parquet").write_text("smoothed-data")
        interp_run = root / "derived" / "interpolated" / "20260520-120000-aabbcc"
        interp_run.mkdir(parents=True)
        (interp_run / "from_interpolated.parquet").write_text(
            "interpolated-data")

        # First publish from smoothed.
        publish_to_stage(
            root, "smoothed", "outlier_corrected",
            "20260520-120000-aabbcc",
        )
        # Re-publish from interpolated.
        final = publish_to_stage(
            root, "interpolated", "outlier_corrected",
            "20260520-120000-aabbcc",
        )
        target = os.readlink(final)
        check(
            "publish_to_stage atomic-replaces an existing symlink "
            "when re-published from a different source",
            "interpolated" in target,
            detail=f"got {target!r}",
        )

    # 5. Refuses to clobber a real directory.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src_run = root / "derived" / "smoothed" / "20260520-120000-aabbcc"
        src_run.mkdir(parents=True)
        (src_run / "video.parquet").write_text("data")
        # Pre-create a REAL directory at the target.
        real_dir = root / "derived" / "outlier_corrected" / "20260520-120000-aabbcc"
        real_dir.mkdir(parents=True)
        (real_dir / "preexisting.parquet").write_text("don't lose me")

        raised = False
        try:
            publish_to_stage(
                root, "smoothed", "outlier_corrected",
                "20260520-120000-aabbcc",
            )
        except FileExistsError:
            raised = True
        check(
            "publish_to_stage refuses to clobber a REAL directory at "
            "the target (raises FileExistsError; protects user data)",
            raised,
        )
        check(
            "Pre-existing data in the un-clobbered target is preserved",
            (real_dir / "preexisting.parquet").read_text() == "don't lose me",
        )

    # 6. Source missing → FileNotFoundError.
    with tempfile.TemporaryDirectory() as td:
        raised = False
        try:
            publish_to_stage(
                Path(td), "smoothed", "outlier_corrected",
                "20260520-120000-aabbcc",
            )
        except FileNotFoundError:
            raised = True
        check(
            "publish_to_stage raises FileNotFoundError when the "
            "source run doesn't exist",
            raised,
        )

    # 7. Bad stage names rejected.
    with tempfile.TemporaryDirectory() as td:
        n_raised = 0
        for src, tgt in [
            ("foo/bar", "outlier_corrected"),
            ("smoothed", "foo/bar"),
            ("foo\\bar", "outlier_corrected"),
        ]:
            try:
                publish_to_stage(Path(td), src, tgt, "run")
            except ValueError:
                n_raised += 1
        check(
            "publish_to_stage rejects stage names containing path "
            "separators (3/3 ValueError cases)",
            n_raised == 3,
            detail=f"raised {n_raised}/3",
        )

    # -----------------------------------------------------------------
    # SECTIONS / DAG validation
    # -----------------------------------------------------------------

    # 9. Module imported cleanly already — DAG validation passed.
    check(
        "section_provenance module imports cleanly "
        "(_validate_dag passes on shipped SECTIONS)",
        len(SECTIONS) > 0,
    )

    # 10. Every depends_on references known IDs.
    bad_deps = []
    for sid, spec in SECTIONS.items():
        for dep in spec.depends_on:
            if dep not in SECTIONS:
                bad_deps.append(f"{sid} -> {dep}")
    check(
        "Every depends_on references a known section_id",
        not bad_deps,
        detail=("; ".join(bad_deps[:3])),
    )

    # 11. Acyclic — verified by import; double-check via topological
    # sort (Kahn). If acyclic, every node ends up in the order list.
    indeg = {sid: 0 for sid in SECTIONS}
    for sid, spec in SECTIONS.items():
        for dep in spec.depends_on:
            indeg[sid] += 1
    queue = [s for s, d in indeg.items() if d == 0]
    order: list[str] = []
    while queue:
        n = queue.pop()
        order.append(n)
        for other_sid, other_spec in SECTIONS.items():
            if n in other_spec.depends_on:
                indeg[other_sid] -= 1
                if indeg[other_sid] == 0:
                    queue.append(other_sid)
    check(
        "SECTIONS graph is acyclic (topological sort enumerates all "
        f"{len(SECTIONS)} nodes)",
        len(order) == len(SECTIONS),
        detail=f"sorted {len(order)}/{len(SECTIONS)}",
    )

    # 12. Cyclic graph rejected.
    bad = {
        "a": SectionSpec(section_id="a", page="p", section_title="A",
                         depends_on=("b",)),
        "b": SectionSpec(section_id="b", page="p", section_title="B",
                         depends_on=("a",)),
    }
    raised = False
    try:
        _validate_dag(bad)
    except ProvenanceError:
        raised = True
    check(
        "_validate_dag rejects a cyclic graph with ProvenanceError",
        raised,
    )

    # 13. Self-reference rejected.
    bad = {
        "a": SectionSpec(section_id="a", page="p", section_title="A",
                         depends_on=("a",)),
    }
    raised = False
    try:
        _validate_dag(bad)
    except ProvenanceError:
        raised = True
    check(
        "_validate_dag rejects a self-referencing section",
        raised,
    )

    # 14. Unknown depends_on rejected.
    bad = {
        "a": SectionSpec(section_id="a", page="p", section_title="A",
                         depends_on=("nonexistent",)),
    }
    raised = False
    try:
        _validate_dag(bad)
    except ProvenanceError:
        raised = True
    check(
        "_validate_dag rejects depends_on referring to an undeclared "
        "section",
        raised,
    )

    # -----------------------------------------------------------------
    # record_run / get_status
    # -----------------------------------------------------------------

    # 15. record_run writes the right TOML block.
    with tempfile.TemporaryDirectory() as td:
        p = _make_project(Path(td))
        t = datetime(2026, 5, 20, 23, 36, 10, tzinfo=timezone.utc)
        record_run(p, "kalman_v2", "20260520-233610-6203f1", run_at=t)
        data = read_project_toml(p)
        entry = data.get("provenance", {}).get("kalman_v2", {})
        check(
            "record_run writes [provenance.<id>] with last_run_id "
            "and last_run_at",
            entry.get("last_run_id") == "20260520-233610-6203f1"
            and "2026-05-20T23:36:10" in str(entry.get("last_run_at",
                                                       "")),
        )

    # 16. Round-trip stability: other fields preserved.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "project.toml"
        p.write_text(
            'project_layout_version = 1\n'
            'file_type = "parquet"\n'
            'project_name = "TestProject"\n'
        )
        t = datetime(2026, 5, 20, 23, 36, 10, tzinfo=timezone.utc)
        record_run(p, "kalman_v2", "run1", run_at=t)
        data = read_project_toml(p)
        check(
            "record_run preserves other project.toml fields "
            "(project_name, file_type, project_layout_version)",
            data.get("project_name") == "TestProject"
            and data.get("file_type") == "parquet"
            and data.get("project_layout_version") == 1,
        )

    # 17. Unknown section_id raises KeyError.
    with tempfile.TemporaryDirectory() as td:
        p = _make_project(Path(td))
        raised = False
        try:
            record_run(p, "definitely_not_a_section", "run1")
        except KeyError:
            raised = True
        check(
            "record_run raises KeyError on unknown section_id "
            "(central declaration is authoritative)",
            raised,
        )

    # 18. run_id=None records last_run_at only.
    with tempfile.TemporaryDirectory() as td:
        p = _make_project(Path(td))
        t = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        record_run(p, "pixels_per_mm", run_id=None, run_at=t)
        data = read_project_toml(p)
        entry = data.get("provenance", {}).get("pixels_per_mm", {})
        check(
            "record_run(run_id=None) records last_run_at and "
            "OMITS last_run_id (settings-section case)",
            "last_run_id" not in entry
            and "last_run_at" in entry,
        )

    # 19. UNKNOWN when no entry.
    with tempfile.TemporaryDirectory() as td:
        p = _make_project(Path(td))
        check(
            "get_status returns UNKNOWN for a section with no "
            "provenance entry",
            get_status(p, "kalman_v2") == SectionStatus.UNKNOWN,
        )

    # 20 & 21. CURRENT vs STALE.
    with tempfile.TemporaryDirectory() as td:
        p = _make_project(Path(td))
        t_old = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
        t_new = datetime(2026, 5, 20, 0, 0, 0, tzinfo=timezone.utc)
        # Parent ran earlier, child ran later → CURRENT.
        record_run(p, "import_pose", "imp1", run_at=t_old)
        record_run(p, "kalman_v2", "kal1", run_at=t_new)
        check(
            "get_status returns CURRENT when this section ran AFTER "
            "its dependency",
            get_status(p, "kalman_v2") == SectionStatus.CURRENT,
        )
        # Now re-run parent later → child becomes STALE.
        t_newer = datetime(2026, 5, 21, 0, 0, 0, tzinfo=timezone.utc)
        record_run(p, "import_pose", "imp2", run_at=t_newer)
        check(
            "get_status returns STALE when a dependency was re-run "
            "AFTER this section",
            get_status(p, "kalman_v2") == SectionStatus.STALE,
        )

    # 22. UNKNOWN dependencies are ignored.
    with tempfile.TemporaryDirectory() as td:
        p = _make_project(Path(td))
        # Record kalman_v2 but NOT its dependency (import_pose).
        t = datetime(2026, 5, 20, 0, 0, 0, tzinfo=timezone.utc)
        record_run(p, "kalman_v2", "kal1", run_at=t)
        check(
            "get_status returns CURRENT when this section is recorded "
            "but its dependency is UNKNOWN (unverified parents are "
            "ignored — see module docstring)",
            get_status(p, "kalman_v2") == SectionStatus.CURRENT,
        )

    # 23. get_all_statuses returns one entry per section.
    with tempfile.TemporaryDirectory() as td:
        p = _make_project(Path(td))
        statuses = get_all_statuses(p)
        check(
            f"get_all_statuses returns one entry per SECTIONS key "
            f"({len(statuses)} returned, {len(SECTIONS)} declared)",
            set(statuses.keys()) == set(SECTIONS.keys()),
        )

    # 24. Missing project.toml soft-fails.
    missing = Path("/tmp/definitely_not_a_real_project_path/project.toml")
    check(
        "get_status soft-fails to UNKNOWN when project.toml is "
        "missing (UI tolerance)",
        get_status(missing, "kalman_v2") == SectionStatus.UNKNOWN,
    )

    # 25. Malformed last_run_at soft-fails.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "project.toml"
        p.write_text(
            'project_layout_version = 1\n'
            'file_type = "parquet"\n'
            '\n[provenance]\n'
            '\n[provenance.kalman_v2]\n'
            'last_run_at = "not a real timestamp"\n'
        )
        check(
            "get_status soft-fails to UNKNOWN on a malformed "
            "last_run_at (corrupted entry doesn't crash UI)",
            get_status(p, "kalman_v2") == SectionStatus.UNKNOWN,
        )

    # -----------------------------------------------------------------
    # Project-wide invariants
    # -----------------------------------------------------------------

    # 26. Parse-clean.
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

    # 27. 122do baseline tripwire.
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
        "122do baseline preserved: no `Optional[` in non-docstring "
        "positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    # 28. 122dr helper still importable.
    check(
        "122dr baseline preserved: latest_populated_run_or_parent "
        "still exported from mufasa.project_layout",
        callable(latest_populated_run_or_parent),
    )

    print(
        f"smoke_122ds_section_provenance: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
