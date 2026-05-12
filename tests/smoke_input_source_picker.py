"""
tests/smoke_input_source_picker.py
==================================

Patch 122b: tests the pure discovery function in
:mod:`mufasa.ui_qt.input_source_picker`. The Qt widget itself is
covered by AST-level assertions and not instantiated (PySide6 isn't
available in the sandbox).

Coverage:

* Empty / unknown projects return ``[]``.
* Legacy projects surface only the legacy ``csv/*`` dirs that
  exist and contain pose files.
* v1 projects surface ``sources/pose/`` plus one entry per
  smoothed / outlier-corrected run, sorted newest-first.
* Post-migration projects (both v1 and legacy paths reachable)
  list v1 candidates first, legacy after.
* Exactly one default per call, picked by ``_DEFAULT_PREFER_ORDER``.
* Empty run dirs are skipped.
* AST sanity check on the Qt widget class.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mufasa.project_layout import (  # noqa: E402
    PROJECT_CONFIG_FILENAME,
    ProjectPaths,
    SmoothingFlavors,
    Stages,
    generate_run_id,
    write_project_toml,
)
from mufasa.ui_qt.input_source_picker import (  # noqa: E402
    InputSource,
    SourceKinds,
    discover_input_sources,
)


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _make_pose_file(dir_path: Path, name: str = "video1.csv") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / name).write_text("scorer,bp\n")


def _make_legacy_project(root: Path) -> str:
    """Build a SimBA-style legacy project tree under ``root`` and
    return the path to its project_config.ini.
    """
    project_folder = root / "project_folder"
    project_folder.mkdir(parents=True)
    ini_path = project_folder / "project_config.ini"
    ini_path.write_text(
        "[General settings]\n"
        f"project_path = {project_folder}\n"
    )
    return str(ini_path)


def _make_v1_project(root: Path) -> ProjectPaths:
    """Build a v1 project skeleton under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(root)
    paths.ensure_skeleton()
    write_project_toml(root / PROJECT_CONFIG_FILENAME, {
        "project_layout_version": 1,
        "project_name": "smoke",
    })
    return paths


def main() -> int:
    # -----------------------------------------------------------
    # 1. Nothing → empty
    # -----------------------------------------------------------
    check(
        "discover() with no args returns []",
        discover_input_sources() == [],
    )
    check(
        "discover(None, None) returns []",
        discover_input_sources(None, None) == [],
    )

    # -----------------------------------------------------------
    # 2. Legacy project: surface only dirs with content
    # -----------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ini = _make_legacy_project(tmp / "L1")
        project_folder = tmp / "L1" / "project_folder"

        # Pre-create the dirs but only put files in some.
        _make_pose_file(project_folder / "csv" / "input_csv")
        _make_pose_file(
            project_folder / "csv" / "outlier_corrected_movement_location",
        )
        # Make this dir exist but stay empty — should be skipped.
        (project_folder / "csv" / "outlier_corrected_movement").mkdir(
            parents=True,
        )
        # Don't create csv/smoothed_v2 at all.

        sources = discover_input_sources(config_path=ini)
        kinds = [s.kind for s in sources]
        labels = [s.label for s in sources]
        check(
            "legacy: surfaces 2 sources (input_csv + outlier_location)",
            len(sources) == 2,
            detail=f"got {len(sources)}: {labels}",
        )
        check(
            "legacy: contains RAW source",
            SourceKinds.RAW in kinds,
        )
        check(
            "legacy: contains OUTLIER_CORRECTED source",
            SourceKinds.OUTLIER_CORRECTED in kinds,
        )
        check(
            "legacy: skips empty outlier_corrected_movement/",
            not any(
                "movement only" in s.label.lower() for s in sources
            ),
        )
        # Default preference: OUTLIER_CORRECTED beats RAW per
        # _DEFAULT_PREFER_ORDER.
        defaults = [s for s in sources if s.is_default]
        check(
            "legacy: exactly one default marked",
            len(defaults) == 1,
            detail=f"got {len(defaults)}: {[d.label for d in defaults]}",
        )
        check(
            "legacy: default is OUTLIER_CORRECTED (preferred over RAW)",
            defaults and defaults[0].kind == SourceKinds.OUTLIER_CORRECTED,
        )

    # -----------------------------------------------------------
    # 3. v1 project with multiple runs
    # -----------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = _make_v1_project(tmp / "V1")

        # Raw under sources/pose/
        _make_pose_file(paths.sources_pose)

        # Two Kalman v2 runs — generate ids that are explicitly
        # ordered so newest-first is verifiable. We can't rely on
        # generate_run_id() being monotonic across same-second
        # invocations, so synthesize them by hand.
        run1 = "20260510-120000-aaaaaa"
        run2 = "20260511-120000-bbbbbb"  # newer
        kv2_run1 = paths.smoothed_run_dir(
            SmoothingFlavors.KALMAN_V2, run_id=run1,
        )
        kv2_run2 = paths.smoothed_run_dir(
            SmoothingFlavors.KALMAN_V2, run_id=run2,
        )
        _make_pose_file(kv2_run1)
        _make_pose_file(kv2_run2)

        # One Savitzky run
        sg_run = paths.smoothed_run_dir(
            SmoothingFlavors.SAVITZKY_GOLAY,
            run_id="20260509-120000-cccccc",
        )
        _make_pose_file(sg_run)

        # One outlier-corrected run
        oc_run = paths.stage_run_dir(
            Stages.OUTLIER_CORRECTED,
            run_id="20260508-120000-dddddd",
        )
        _make_pose_file(oc_run)

        # An empty Kalman run that should be skipped
        empty_run = paths.smoothed_run_dir(
            SmoothingFlavors.KALMAN_V2,
            run_id="20260507-120000-eeeeee",
        )
        # don't create any files in it

        sources = discover_input_sources(project_root=paths.root)
        labels = [s.label for s in sources]
        check(
            "v1: total source count = 1 raw + 2 kv2 + 1 sg + 1 oc",
            len(sources) == 5,
            detail=f"got {len(sources)}: {labels}",
        )
        check(
            "v1: empty kv2 run skipped",
            not any("eeeeee" in s.label for s in sources),
        )

        # Verify kv2 ordering: newer run appears before older
        kv2_indices = [
            i for i, s in enumerate(sources)
            if s.kind == SourceKinds.SMOOTHED_KALMAN_V2
        ]
        check(
            "v1: 2 kv2 sources",
            len(kv2_indices) == 2,
        )
        check(
            "v1: kv2 newest-first",
            len(kv2_indices) == 2
            and "bbbbbb" in sources[kv2_indices[0]].label
            and "aaaaaa" in sources[kv2_indices[1]].label,
        )

        # Default: should be the newest kv2 run (preferred kind +
        # newest first).
        defaults = [s for s in sources if s.is_default]
        check(
            "v1: exactly one default",
            len(defaults) == 1,
        )
        check(
            "v1: default is the newest Kalman v2 run",
            defaults
            and defaults[0].kind == SourceKinds.SMOOTHED_KALMAN_V2
            and "bbbbbb" in defaults[0].label,
        )

    # -----------------------------------------------------------
    # 4. Post-migration: both layouts present
    # -----------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        proj = tmp / "post_mig"
        proj.mkdir()
        # v1 layout files
        paths = _make_v1_project(proj)
        _make_pose_file(paths.sources_pose)
        # Plus a legacy INI at the same root pointing to a
        # legacy-style csv/ tree inside it.
        legacy_root = proj / "legacy_root"
        legacy_root.mkdir()
        _make_pose_file(legacy_root / "csv" / "input_csv")
        ini = proj / "project_config.ini"
        ini.write_text(
            "[General settings]\n"
            f"project_path = {legacy_root}\n"
        )

        sources = discover_input_sources(config_path=str(ini))
        check(
            "post-migration: both v1 and legacy sources surfaced",
            len(sources) == 2,
            detail=f"got {len(sources)}: "
                   f"{[s.kind for s in sources]}",
        )
        check(
            "post-migration: v1 source comes first",
            sources[0].path == paths.sources_pose,
        )
        check(
            "post-migration: legacy source comes second",
            "csv/input_csv" in str(sources[1].path).replace("\\", "/"),
        )

    # -----------------------------------------------------------
    # 5. AST sanity on the Qt widget
    # -----------------------------------------------------------
    src_path = REPO_ROOT / "mufasa" / "ui_qt" / "input_source_picker.py"
    src = src_path.read_text()
    tree = ast.parse(src)
    classes = {
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
    }
    check(
        "widget module: InputSourcePicker class defined",
        "InputSourcePicker" in classes,
    )
    check(
        "widget module: InputSource dataclass defined",
        "InputSource" in classes,
    )
    check(
        "widget module: SourceKinds defined",
        "SourceKinds" in classes,
    )
    # Public methods on the widget
    picker_class = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef) and n.name == "InputSourcePicker"),
        None,
    )
    if picker_class is not None:
        methods = {
            n.name for n in picker_class.body
            if isinstance(n, ast.FunctionDef)
        }
        for required in (
            "selected_source", "selected_path", "selected_kind", "refresh",
        ):
            check(
                f"widget: defines {required}()",
                required in methods,
            )

    # -----------------------------------------------------------
    # 6. Egocentric form consumes the picker
    # -----------------------------------------------------------
    pc_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "pose_cleanup.py"
    pc_src = pc_path.read_text()
    pc_tree = ast.parse(pc_src)
    ego_class = next(
        (n for n in ast.walk(pc_tree)
         if isinstance(n, ast.ClassDef)
         and n.name == "EgocentricAlignmentForm"),
        None,
    )
    check(
        "EgocentricAlignmentForm class exists",
        ego_class is not None,
    )
    if ego_class is not None:
        ego_src = ast.unparse(ego_class)
        check(
            "Egocentric form references InputSourcePicker",
            "InputSourcePicker" in ego_src,
        )
        check(
            "Egocentric form constructs self.source_picker in build()",
            "self.source_picker" in ego_src
            and "source_picker = InputSourcePicker" in ego_src,
        )
        check(
            "Egocentric form's collect_args calls source_picker.selected_path()",
            "source_picker.selected_path" in ego_src,
        )
        check(
            "Egocentric form no longer hard-codes the legacy auto-detect",
            "outlier_corrected_movement_location" not in ego_src
            or "InputSourcePicker" in ego_src,
            detail="should no longer special-case outlier-corrected dir",
        )

    print(
        f"smoke_input_source_picker: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
