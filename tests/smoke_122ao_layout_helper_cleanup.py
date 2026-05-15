"""
tests/smoke_122ao_layout_helper_cleanup.py
============================================

Patch 122ao: csv-subtree audit Category B3 — drop the dead
``glob.glob(self.features_dir + ...)`` / ``glob.glob(self.
targets_folder + ...)`` populations in
:meth:`ConfigReader.__init__` (and the v1 override path's
matching globs), and drop ``features_extracted_dir`` /
``targets_inserted_dir`` from :func:`project_paths_from_config`.

After 122ao the only csv/ subtree key remaining in the layout
helper is ``machine_results_dir`` — separate work tied to
classifier inference output migration.

What the patch keeps and why
----------------------------
* ``self.features_dir`` and ``self.targets_folder`` attribute
  setters in :class:`ConfigReader` are preserved (sourced from
  the ``Paths.*`` enum, not the layout helper). The 7 specialty
  feature extractors still emit legacy CSVs alongside their v1
  sidecars (122am), and 16 f-string error messages across the
  codebase interpolate these paths. Both keep working.

* The :attr:`feature_file_paths` / :attr:`target_file_paths`
  list attributes are still populated — but via the v1-aware
  :func:`list_video_stems_with_features` /
  :func:`list_video_stems_with_labels` helpers, producing
  pseudo-paths under ``self.features_dir`` /
  ``self.targets_folder`` with v1-discovered stems.
  Consumers that count entries (UI displays) or iterate for
  stem extraction get correct results. Consumers that actually
  open files go through the v1 load helpers in the 122ae-5
  series migrations.

What's now gone
---------------
* ``glob.glob(self.features_dir + f"/*.{self.file_type}")``
  at the top of :meth:`ConfigReader.__init__`.
* ``glob.glob(self.targets_folder + ...)`` at the same site.
* The matching pair in ConfigReader's v1 override branch.
* Two keys in :func:`project_paths_from_config` (v1 + legacy
  branches both lose ``features_extracted_dir`` and
  ``targets_inserted_dir``).
* :attr:`features_dir` / :attr:`targets_dir` reads in
  :class:`FrameLabellerWidget._load_project_metadata` (dead
  after 122ak's labels-only save).

All AST-based — heavy deps (Qt, ConfigParser fixtures, etc.)
not available in sandbox.
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
    # 1. project_paths_from_config — legacy keys gone from both
    #    layout branches
    # ==================================================================
    pl_path = REPO_ROOT / "mufasa" / "project_layout.py"
    pl_src = pl_path.read_text()
    check(
        "project_layout: 'features_extracted_dir' key no longer "
        "appears in any dict literal (only in comments / docstring)",
        not any(
            '"features_extracted_dir":' in line
            and not line.lstrip().startswith("#")
            for line in pl_src.splitlines()
        ),
    )
    check(
        "project_layout: 'targets_inserted_dir' key no longer "
        "appears in any dict literal",
        not any(
            '"targets_inserted_dir":' in line
            and not line.lstrip().startswith("#")
            for line in pl_src.splitlines()
        ),
    )
    check(
        "project_layout: machine_results_dir + roi_definitions_path "
        "+ derived_features_dir + derived_labels_dir all retained",
        all(
            f'"{k}":' in pl_src
            for k in (
                "machine_results_dir", "roi_definitions_path",
                "derived_features_dir", "derived_labels_dir",
            )
        ),
    )
    check(
        "project_layout: records 122ao patch number",
        "122ao" in pl_src,
    )

    # ==================================================================
    # 2. Behavioural — project_paths_from_config no longer returns
    #    the removed keys
    # ==================================================================
    from mufasa.project_layout import project_paths_from_config

    # 2a — v1 project
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = tmp / "project.toml"
        toml.write_text(textwrap.dedent("""
            project_layout_version = 1

            [project]
            name = "smoke_122ao"
            version = "0.0.1"

            [pose]
            file_type = "csv"
            animal_count = 1
            body_parts = ["nose"]

            [classifiers]
            targets = ["sniff"]
        """).strip() + "\n")
        paths = project_paths_from_config(str(toml))
        check(
            "v1 layout: 'features_extracted_dir' NOT in returned dict",
            "features_extracted_dir" not in paths,
        )
        check(
            "v1 layout: 'targets_inserted_dir' NOT in returned dict",
            "targets_inserted_dir" not in paths,
        )
        check(
            "v1 layout: 'derived_features_dir' / 'derived_labels_dir' "
            "still present",
            "derived_features_dir" in paths
            and "derived_labels_dir" in paths,
        )
        check(
            "v1 layout: 'machine_results_dir' still present "
            "(out of scope for 122ao)",
            "machine_results_dir" in paths,
        )

    # 2b — legacy INI project
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        ini = tmp / "project_config.ini"
        proj_path = tmp / "project_folder"
        proj_path.mkdir()
        ini.write_text(textwrap.dedent(f"""
            [General settings]
            project_path = {proj_path}
            workflow_file_type = csv
        """).strip() + "\n")
        paths = project_paths_from_config(str(ini))
        check(
            "legacy layout: 'features_extracted_dir' NOT in dict",
            "features_extracted_dir" not in paths,
        )
        check(
            "legacy layout: 'targets_inserted_dir' NOT in dict",
            "targets_inserted_dir" not in paths,
        )
        check(
            "legacy layout: 'machine_results_dir' still present",
            "machine_results_dir" in paths,
        )

    # ==================================================================
    # 3. ConfigReader globs replaced with v1 enumeration
    # ==================================================================
    cr_path = REPO_ROOT / "mufasa" / "mixins" / "config_reader.py"
    cr_src = cr_path.read_text()
    check(
        "ConfigReader: no remaining glob.glob(self.features_dir ...) "
        "at code level",
        not any(
            "glob.glob(self.features_dir" in line
            and not line.lstrip().startswith("#")
            for line in cr_src.splitlines()
        ),
    )
    check(
        "ConfigReader: no remaining glob.glob(self.targets_folder ...) "
        "at code level",
        not any(
            "glob.glob(self.targets_folder" in line
            and not line.lstrip().startswith("#")
            for line in cr_src.splitlines()
        ),
    )
    # The v1 helpers are now imported (lazy in two methods, so search
    # for the import strings)
    check(
        "ConfigReader: imports list_video_stems_with_features",
        "list_video_stems_with_features" in cr_src,
    )
    check(
        "ConfigReader: imports list_video_stems_with_labels",
        "list_video_stems_with_labels" in cr_src,
    )
    check(
        "ConfigReader: defensive try/except around v1 enumeration "
        "falls back to empty lists",
        "self.feature_file_paths = []" in cr_src
        and "self.target_file_paths = []" in cr_src,
    )
    check(
        "ConfigReader: still sets self.features_dir from Paths."
        "FEATURES_EXTRACTED_DIR (preserved for legacy writes + "
        "16 ERROR_MSG sites)",
        "Paths.FEATURES_EXTRACTED_DIR" in cr_src,
    )
    check(
        "ConfigReader: still sets self.targets_folder from Paths."
        "TARGETS_INSERTED_DIR (preserved)",
        "Paths.TARGETS_INSERTED_DIR" in cr_src,
    )
    check(
        "ConfigReader: records 122ao patch number",
        "122ao" in cr_src,
    )

    # ==================================================================
    # 4. frame_labeller: dead reads of features_extracted_dir +
    #    targets_inserted_dir keys removed
    # ==================================================================
    fl_path = REPO_ROOT / "mufasa" / "ui_qt" / "frame_labeller.py"
    fl_src = fl_path.read_text()
    check(
        "frame_labeller: paths['features_extracted_dir'] read gone",
        'paths["features_extracted_dir"]' not in fl_src
        and "paths['features_extracted_dir']" not in fl_src,
    )
    check(
        "frame_labeller: paths['targets_inserted_dir'] read gone",
        'paths["targets_inserted_dir"]' not in fl_src
        and "paths['targets_inserted_dir']" not in fl_src,
    )
    check(
        "frame_labeller: machine_results_dir read retained "
        "(out of scope)",
        'paths["machine_results_dir"]' in fl_src
        or "paths['machine_results_dir']" in fl_src,
    )
    check(
        "frame_labeller: records 122ao patch number",
        "122ao" in fl_src,
    )

    # ==================================================================
    # 5. Sanity — all touched files parse
    # ==================================================================
    for p in (pl_path, cr_path, fl_path):
        try:
            ast.parse(p.read_text())
            ok = True
        except SyntaxError:
            ok = False
        check(f"AST parses: {p.name}", ok)

    print(
        f"smoke_122ao_layout_helper_cleanup: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
