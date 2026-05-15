"""
tests/smoke_layout_paths_extension.py
======================================

Patch 122ab: behavioural verification of the four new keys
added to :func:`mufasa.project_layout.project_paths_from_config`
plus AST checks that the two known consumers
(:mod:`mufasa.ui_qt.frame_labeller` and
:mod:`mufasa.ui_qt.dialogs.roi_video_table`) actually route
through the helper instead of reading the legacy INI directly.

Behavioural coverage (project_layout.py imports cleanly in
the sandbox — no PySide6 / cv2 dependency):

1. v1 layout (project.toml) returns the four new keys with
   ``<root>/csv/...`` and ``<root>/logs/measures/...`` shapes.
2. Legacy layout (project_config.ini) returns the same key
   names with ``<project_path>/csv/...`` /
   ``<project_path>/logs/measures/...`` shapes.
3. Empty INI raises ValueError (existing behaviour preserved).
4. The four new keys round-trip through pathlib without
   slash-direction surprises (str paths only, no Path objects
   leaked).

AST coverage:

5. frame_labeller.py calls project_paths_from_config in
   _load_project_metadata + project_metadata_from_config for
   file_type.
6. frame_labeller.py no longer reads
   ``[General settings].project_path`` via configparser.
7. roi_video_table.py routes through the layout helper +
   exposes self.video_dir + self.roi_h5_path from helper keys.
8. roi_video_table.py no longer reads
   ``[General settings].project_path`` via configparser.
9. configparser import is gone from both modules.
"""
from __future__ import annotations

import ast
import sys
import tempfile
import textwrap
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
    # 1-3. Behavioural
    # ==================================================================
    from mufasa.project_layout import project_paths_from_config

    # ----- v1 (project.toml) -----
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        toml = root / "project.toml"
        toml.write_text(textwrap.dedent("""
            [project]
            name = "smoke_test_v1"
            version = "0.0.1"
        """).strip() + "\n")
        v1 = project_paths_from_config(toml)
        # Patch 122bf: 122ao removed features_extracted_dir +
        # targets_inserted_dir from v1 (csv-subtree audit B3),
        # and 122ax removed machine_results_dir (close-out).
        # Only roi_definitions_path remains of the 122ab 4-key
        # set on v1. Negative guards on the removed keys lock
        # in the cleanup.
        check(
            "v1 layout returns key 'roi_definitions_path'",
            "roi_definitions_path" in v1,
        )
        for removed_key in ("features_extracted_dir",
                            "targets_inserted_dir",
                            "machine_results_dir"):
            check(
                f"v1 layout: key {removed_key!r} is REMOVED "
                f"(was 122ab; removed in 122ao/122ax)",
                removed_key not in v1,
            )
        # Patch 122bf: only the surviving v1 key shape is checked.
        # The previous v1 csv/features_extracted, csv/targets_inserted,
        # csv/machine_results path-shape assertions are removed since
        # the keys are gone.
        check(
            "v1 roi_definitions_path is "
            "<root>/logs/measures/ROI_definitions.h5",
            v1.get("roi_definitions_path", "").replace("\\", "/").endswith(
                "/logs/measures/ROI_definitions.h5"
            ),
        )
        # Anchored under the project root — switched from
        # features_extracted_dir to roi_definitions_path since
        # the former is gone on v1.
        check(
            "v1 roi_definitions_path is under project_root",
            v1.get("roi_definitions_path", "")
              .startswith(v1.get("project_root", "x")),
        )

    # ----- Legacy (project_config.ini) -----
    with tempfile.TemporaryDirectory() as tmp:
        proj_dir = Path(tmp) / "Legacy_Project"
        proj_dir.mkdir()
        ini = proj_dir / "project_config.ini"
        ini.write_text(textwrap.dedent(f"""
            [General settings]
            project_path = {proj_dir}
            workflow_file_type = csv
        """).strip() + "\n")
        legacy = project_paths_from_config(ini)
        # Patch 122bf: 122ao removed features_extracted_dir +
        # targets_inserted_dir from legacy too. Only the
        # machine_results_dir + roi_definitions_path pair
        # remains of the 122ab 4-key set on legacy. Negative
        # guards lock in the cleanup.
        for key in ("machine_results_dir", "roi_definitions_path"):
            check(
                f"legacy layout returns key {key!r}",
                key in legacy,
            )
        for removed_key in ("features_extracted_dir",
                            "targets_inserted_dir"):
            check(
                f"legacy layout: key {removed_key!r} is REMOVED "
                f"(was 122ab; removed in 122ao)",
                removed_key not in legacy,
            )
        # Patch 122bf: legacy-features-extracted path-shape
        # assertion removed since the key is gone.
        check(
            "legacy machine_results_dir is "
            "<project>/csv/machine_results",
            legacy.get("machine_results_dir", "")
                  .endswith(f"{proj_dir.name}/csv/machine_results")
            or legacy.get("machine_results_dir", "")
                  .endswith(f"{proj_dir.name}\\csv\\machine_results"),
        )
        check(
            "legacy roi_definitions_path is "
            "<project>/logs/measures/ROI_definitions.h5",
            legacy.get("roi_definitions_path", "")
                  .replace("\\", "/")
                  .endswith("/logs/measures/ROI_definitions.h5"),
        )

    # ----- Empty INI raises ValueError -----
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.ini"
        empty.write_text("")
        raised = False
        try:
            project_paths_from_config(empty)
        except ValueError:
            raised = True
        check("empty INI raises ValueError (existing behaviour)", raised)

    # ----- Return values are strings, not Path objects -----
    with tempfile.TemporaryDirectory() as tmp:
        toml = Path(tmp) / "project.toml"
        toml.write_text("[project]\nname = 'x'\n")
        all_strings = all(
            isinstance(v, str)
            for v in project_paths_from_config(toml).values()
        )
        check(
            "All return values are str (no Path-object leaks)",
            all_strings,
        )

    # ==================================================================
    # 5-9. AST checks on the consumers
    # ==================================================================
    # frame_labeller.py
    fl_src = (REPO_ROOT / "mufasa" / "ui_qt"
              / "frame_labeller.py").read_text()
    fl_tree = ast.parse(fl_src)
    check(
        "frame_labeller imports project_paths_from_config",
        "project_paths_from_config" in fl_src,
    )
    check(
        "frame_labeller imports project_metadata_from_config "
        "(for v1 file_type)",
        "project_metadata_from_config" in fl_src,
    )
    check(
        "frame_labeller no longer imports configparser",
        not any(
            isinstance(n, ast.Import)
            and any(a.name == "configparser" for a in n.names)
            for n in fl_tree.body
        ),
    )
    check(
        "frame_labeller no longer reads "
        "[General settings].project_path",
        "'General settings'" not in fl_src
        and '"General settings"' not in fl_src,
    )
    # roi_video_table.py
    rvt_src = (REPO_ROOT / "mufasa" / "ui_qt" / "dialogs"
               / "roi_video_table.py").read_text()
    rvt_tree = ast.parse(rvt_src)
    check(
        "roi_video_table calls project_paths_from_config",
        "project_paths_from_config" in rvt_src,
    )
    check(
        "roi_video_table no longer imports configparser",
        not any(
            isinstance(n, ast.Import)
            and any(a.name == "configparser" for a in n.names)
            for n in rvt_tree.body
        ),
    )
    check(
        "roi_video_table no longer reads "
        "[General settings].project_path",
        "'General settings'" not in rvt_src
        and '"General settings"' not in rvt_src,
    )
    # Dialog stores video_dir and roi_h5_path from helper
    check(
        "ROIVideoTableDialog stores self.video_dir from helper",
        "self.video_dir = paths" in rvt_src
        or 'paths.get("video_dir"' in rvt_src,
    )
    check(
        "ROIVideoTableDialog stores self.roi_h5_path from helper "
        "'roi_definitions_path' key",
        "roi_definitions_path" in rvt_src,
    )

    # 122ab note in helper docstring
    layout_src = (REPO_ROOT / "mufasa"
                  / "project_layout.py").read_text()
    check(
        "project_layout docstring records the 122ab note",
        "122ab" in layout_src,
    )

    print(
        f"smoke_layout_paths_extension: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
