"""
tests/smoke_122ee_pose_import_publishing.py
==============================================

Patch 122ee: Pose Import publish-to-stage. The 2nd half of the
backend-refactor track deferred from 122eb / 122ec. Differs from
122ed's Interpolate backend refactor: rather than touching all
12 per-tracker importer backends, this patch wires publishing at
the FORM level via a post-import snapshot of ``sources/pose/``.

Why a form-level snapshot rather than per-backend allocation
-------------------------------------------------------------
Pose Import has 12 per-tracker importer backends (DLC h5/csv,
DLC importer csv, SLEAP h5/csv/slp, maDLC, MARS, SuperAnimal,
facemap, SimBA blob, SimBA YOLO). Each writes pose data to
``sources/pose/`` with its own filename and format convention.
A v1-correct per-backend approach would mean either:

a) Each backend pre-allocates a v1 run-id, creates
   ``sources/pose/<run_id>/``, and writes there. 12 backends ×
   per-format quirks; medium-large surgery.
b) The form passes a pre-allocated run-id to each backend.
   Requires every backend to accept the kwarg.

122ee takes a third approach: AFTER the backend's ``.run()``
returns, the form takes a snapshot of the current state of
``sources/pose/`` by symlinking each file into
``derived/pose/<run_id>/``. The base class's success hook then
publishes ``derived/outlier_corrected/<run_id> ->
../pose/<run_id>``. No backend changes; one form change.

Snapshot semantics
------------------
The snapshot reflects ``sources/pose/`` AS-OF the moment after
this import returned, NOT just the files this import wrote. If
the user has previously imported video1, video2, and now imports
video3, the snapshot for video3's run_id includes all three.

That's intentional. Downstream consumers reading
``outlier_corrected/`` want the CURRENT pose state, not just the
just-imported delta. The semantic matches Kalman v2's:
``smoothed/kalman_v2/<run_id>/`` contains every video the
smoother ran on in that invocation.

Re-imports and updates: each file in the snapshot is a relative
symlink to ``sources/pose/<file>``, so if the user re-imports a
video later (overwriting the file in sources/pose/), the
existing snapshots' symlinks transparently see the new content.
That's also intentional — downstream consumers want the current
state of each pose file.

Effect
------
After this patch, the "Skip outlier correction" replacement story
is fully closed: users can run just the Import step and the
imported pose data is discoverable to downstream backends
(Features etc.) via ``outlier_corrected/<run_id>``. No outlier
correction, Kalman v2, or interpolate run required. The latest
producer wins; if the user runs interpolate next, the
``outlier_corrected/<run_id>`` symlink updates to point at
``../interpolated/<new_run_id>`` instead of ``../pose/...``.

What this patch landed
----------------------
mufasa/ui_qt/forms/pose_import.py:

* ``PoseImportForm`` gained class-level attrs:
  - ``publish_target_stage = "outlier_corrected"``
  - ``publish_source_stage = "pose"``
  No ``publish_source_flavor`` (only one flavor of import per
  run).
* ``target()`` now calls ``self._snapshot_and_set_run_id(
  config_path)`` immediately after the backend's ``.run()``
  returns. The new helper method:
  - Reads ``sources/pose/`` via
    ``project_paths_from_config(config_path)``.
  - No-op if the directory is missing or empty (record-only
    fallback; provenance still recorded with last_run_at-only).
  - Else: generates a v1 run-id, creates
    ``derived/pose/<run_id>/``, symlinks every file from
    ``sources/pose/`` into it using relative paths
    (``../../../sources/pose/<file>``).
  - Sets ``self._last_run_id = run_id``.
* Class docstring updated: "Provenance (patches 122eb / 122ee)"
  with a history section explaining the record-only → snapshot
  transition.

Reciprocal tripwire flips (already in 122eb / 122ec smokes):
* smoke_122eb_pose_import_provenance check 3 ("no
  publish_target_stage") → "is 'outlier_corrected'".
* smoke_122eb check 4 ("target doesn't set _last_run_id") →
  "form sets _last_run_id somewhere" (via the snapshot helper).
* smoke_122ec_interpolate_provenance check 10 ("exactly 2
  publishers") → "exactly 3 publishers" (kalman + interpolate +
  import_pose).

Coverage
--------
PoseImportForm publish wiring (4 checks):
1.  ``publish_target_stage == "outlier_corrected"``.
2.  ``publish_source_stage == "pose"``.
3.  No ``publish_source_flavor`` declaration (only one
    import flavor; matches Interpolate's choice from 122ed).
4.  ``section_id`` unchanged from 122eb (``"import_pose"``).

Snapshot helper (5 checks):
5.  ``_snapshot_and_set_run_id`` method exists on the class.
6.  The helper takes ``config_path`` as its only non-self arg.
7.  The helper uses ``generate_run_id`` from project_layout
    (verified via substring in unparsed source — the imports
    are lazy / inline so AST-walking the imports list isn't
    sufficient).
8.  The helper creates the run dir via ``mkdir(parents=True,
    exist_ok=False)`` — exist_ok=False so a generate_run_id
    collision surfaces rather than silently merging.
9.  The helper uses ``os.symlink`` with a relative ``..``
    prefix (the relative-symlink contract).

target() integration (2 checks):
10. ``target()`` calls ``self._snapshot_and_set_run_id(...)``
    after the runner's ``.run()`` method.
11. ``target()`` does NOT directly assign ``self._last_run_id``
    (the assignment is encapsulated in the helper).

Reciprocal tripwire flips (3 checks):
12. smoke_122eb check 3 now asserts publish_target_stage value.
13. smoke_122eb check 4 now asserts _last_run_id set somewhere.
14. smoke_122ec check 10 now asserts "exactly 3 publishers".

Arc completion (1 check):
15. All 4 producers now have publish_target_stage EXCEPT
    RunOutlierCorrectionForm (which writes directly to
    outlier_corrected/, no symlink needed).

Cross-patch invariants (5 checks):
16. 122ef state preserved: hasattr guard in workbench.py.
17. 122ed state preserved: Interpolate has run_id + run_dir.
18. 122dz state preserved: load_machine_results_for_video has
    no legacy_fallback parameter.
19. Parse-clean.
20. 122do baseline.
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


def _ast_find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _ast_method(cls_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _ast_class_attr(cls_node: ast.ClassDef, name: str) -> ast.AST | None:
    for node in cls_node.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return node.value
        elif isinstance(node, ast.AnnAssign):
            if (isinstance(node.target, ast.Name)
                    and node.target.id == name
                    and node.value is not None):
                return node.value
    return None


def main() -> int:
    pi_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "pose_import.py"
    pi_src = pi_path.read_text()
    pi_tree = ast.parse(pi_src)
    pi_cls = _ast_find_class(pi_tree, "PoseImportForm")
    assert pi_cls is not None

    # -----------------------------------------------------------------
    # Publish wiring
    # -----------------------------------------------------------------
    # 1. publish_target_stage.
    pub_tgt = _ast_class_attr(pi_cls, "publish_target_stage")
    check(
        "PoseImportForm.publish_target_stage == 'outlier_corrected'",
        isinstance(pub_tgt, ast.Constant)
        and pub_tgt.value == "outlier_corrected",
        detail=(f"got {pub_tgt.value!r}"
                if isinstance(pub_tgt, ast.Constant) else "missing"),
    )

    # 2. publish_source_stage.
    pub_src = _ast_class_attr(pi_cls, "publish_source_stage")
    check(
        "PoseImportForm.publish_source_stage == 'pose'",
        isinstance(pub_src, ast.Constant)
        and pub_src.value == "pose",
        detail=(f"got {pub_src.value!r}"
                if isinstance(pub_src, ast.Constant) else "missing"),
    )

    # 3. No publish_source_flavor.
    pub_flav = _ast_class_attr(pi_cls, "publish_source_flavor")
    check(
        "PoseImportForm has NO publish_source_flavor declaration "
        "(only one flavor of import per run — no need to "
        "disambiguate within the pose/ stage)",
        pub_flav is None,
    )

    # 4. section_id unchanged from 122eb.
    sid = _ast_class_attr(pi_cls, "section_id")
    check(
        "PoseImportForm.section_id unchanged from 122eb "
        "('import_pose')",
        isinstance(sid, ast.Constant) and sid.value == "import_pose",
    )

    # -----------------------------------------------------------------
    # Snapshot helper
    # -----------------------------------------------------------------
    helper = _ast_method(pi_cls, "_snapshot_and_set_run_id")

    # 5. Helper exists.
    check(
        "PoseImportForm._snapshot_and_set_run_id method exists",
        helper is not None,
    )

    if helper is not None:
        # 6. Signature: only non-self arg is config_path.
        non_self_args = [
            a.arg for a in helper.args.args if a.arg != "self"
        ]
        check(
            "_snapshot_and_set_run_id signature: only non-self arg "
            "is `config_path`",
            non_self_args == ["config_path"],
            detail=(f"got {non_self_args!r}"),
        )

        helper_src = ast.unparse(helper)

        # 7. Uses generate_run_id.
        check(
            "_snapshot_and_set_run_id uses generate_run_id from "
            "project_layout",
            "generate_run_id" in helper_src,
        )

        # 8. mkdir with exist_ok=False.
        # The pattern is `run_dir.mkdir(parents=True, exist_ok=False)`.
        check(
            "_snapshot_and_set_run_id creates the run dir with "
            "exist_ok=False (so a generate_run_id collision "
            "surfaces rather than silently merging into a "
            "previous run)",
            "exist_ok=False" in helper_src,
        )

        # 9. os.symlink with a relative ".." prefix.
        check(
            "_snapshot_and_set_run_id uses os.symlink with a "
            "relative-path prefix (the project survives moves / "
            "rsync copies; downstream consumers transparently "
            "follow the link)",
            "os.symlink" in helper_src and ".." in helper_src,
        )
    else:
        # Skip the dependent checks if the helper is missing.
        for _ in range(4):
            check("(skipped — _snapshot_and_set_run_id missing)", False)

    # -----------------------------------------------------------------
    # target() integration
    # -----------------------------------------------------------------
    target_method = _ast_method(pi_cls, "target")
    assert target_method is not None
    target_src = ast.unparse(target_method)

    # 10. target() calls the snapshot helper.
    check(
        "PoseImportForm.target() calls "
        "self._snapshot_and_set_run_id(...) after the backend's "
        "run()",
        "self._snapshot_and_set_run_id" in target_src,
    )

    # 11. target() does NOT directly assign self._last_run_id
    # (the assignment lives in the helper).
    target_self_assigns = set()
    for node in ast.walk(target_method):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"):
                    target_self_assigns.add(tgt.attr)
    check(
        "PoseImportForm.target() does NOT directly assign "
        "self._last_run_id (encapsulation: the assignment is "
        "in _snapshot_and_set_run_id where the run-id is "
        "allocated)",
        "_last_run_id" not in target_self_assigns,
    )

    # -----------------------------------------------------------------
    # Reciprocal tripwire flips
    # -----------------------------------------------------------------
    eb_src = (REPO_ROOT / "tests"
              / "smoke_122eb_pose_import_provenance.py").read_text()
    ec_src = (REPO_ROOT / "tests"
              / "smoke_122ec_interpolate_provenance.py").read_text()

    # 12. 122eb check 3 flipped.
    check(
        "smoke_122eb check 3 now asserts "
        "PoseImportForm.publish_target_stage == 'outlier_corrected' "
        "(was 'no publish target' in 122eb's record-only contract)",
        (
            "pub.value == 'outlier_corrected'" in eb_src
            or 'pub.value == "outlier_corrected"' in eb_src
        ),
    )

    # 13. 122eb check 4 flipped — now asserts _last_run_id IS set.
    check(
        "smoke_122eb check 4 now asserts the form sets "
        "_last_run_id somewhere (was 'never set' in 122eb)",
        "sets_run_id = True" in eb_src,
    )

    # 14. 122ec check 10 flipped to 3 publishers.
    check(
        "smoke_122ec check 10 now asserts 'exactly 3' publishers "
        "(was 'exactly 2' after 122ed; now kalman_v2 + interpolate "
        "+ import_pose)",
        "Exactly 3 of the 4 producers" in ec_src,
    )

    # -----------------------------------------------------------------
    # Arc completion
    # -----------------------------------------------------------------
    pc_path = (REPO_ROOT / "mufasa" / "ui_qt"
               / "forms" / "pose_cleanup.py")
    pc_tree = ast.parse(pc_path.read_text())
    roc = _ast_find_class(pc_tree, "RunOutlierCorrectionForm")
    kv2 = _ast_find_class(pc_tree, "KalmanV2SmoothingForm")
    interp = _ast_find_class(pc_tree, "InterpolateForm")
    assert all(x is not None for x in (roc, kv2, interp))

    publishers = []
    for name, cls in [
        ("RunOutlierCorrectionForm", roc),
        ("KalmanV2SmoothingForm", kv2),
        ("InterpolateForm", interp),
        ("PoseImportForm", pi_cls),
    ]:
        if _ast_class_attr(cls, "publish_target_stage") is not None:
            publishers.append(name)

    # 15. RunOutlierCorrectionForm is the only non-publisher.
    check(
        "All 4 producers have publish_target_stage EXCEPT "
        "RunOutlierCorrectionForm (which writes directly to "
        "outlier_corrected/ and so doesn't need a symlink-publish)",
        sorted(publishers) == [
            "InterpolateForm",
            "KalmanV2SmoothingForm",
            "PoseImportForm",
        ],
        detail=(f"publishers: {sorted(publishers)}"),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    # 16. 122ef state preserved.
    wb_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "workbench.py").read_text()
    has_guard = bool(re.search(
        r'hasattr\s*\(\s*form\s*,\s*["\']completed["\']\s*\)',
        wb_src,
    ))
    check(
        "122ef-hotfix state preserved: workbench.py still has "
        "the `hasattr(form, 'completed')` guard",
        has_guard,
    )

    # 17. 122ed state preserved.
    bk_src = (REPO_ROOT / "mufasa" / "data_processors"
              / "interpolate.py").read_text()
    bk_tree = ast.parse(bk_src)
    bk_cls = _ast_find_class(bk_tree, "Interpolate")
    assert bk_cls is not None
    bk_init = _ast_method(bk_cls, "__init__")
    assert bk_init is not None
    bk_init_assigns = set()
    for node in ast.walk(bk_init):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"):
                    bk_init_assigns.add(tgt.attr)
    check(
        "122ed state preserved: Interpolate.__init__ still "
        "assigns run_id and run_dir",
        "run_id" in bk_init_assigns and "run_dir" in bk_init_assigns,
    )

    # 18. 122dz state preserved.
    cio_src = (REPO_ROOT / "mufasa" / "utils"
               / "classification_io.py").read_text()
    cio_tree = ast.parse(cio_src)
    h = None
    for node in ast.walk(cio_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "load_machine_results_for_video"):
            h = node
            break
    assert h is not None
    h_params = [a.arg for a in h.args.args] + [
        a.arg for a in h.args.kwonlyargs
    ]
    check(
        "122dz state preserved: load_machine_results_for_video "
        "has no legacy_fallback parameter",
        "legacy_fallback" not in h_params,
    )

    # 19. Parse-clean.
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

    # 20. 122do baseline.
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

    print(
        f"smoke_122ee_pose_import_publishing: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
