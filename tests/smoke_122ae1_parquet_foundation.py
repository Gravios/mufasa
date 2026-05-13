"""
tests/smoke_122ae1_parquet_foundation.py
========================================

Patch 122ae-1: foundation patch for the parquet-only derived
storage migration. Three behavioural verifications:

1. ``project_paths_from_config`` returns two new keys
   ``derived_features_dir`` and ``derived_labels_dir`` for both
   v1 and legacy layouts, anchored under the project root with
   shape ``derived/features/`` / ``derived/labels/``.

2. ``project_metadata_from_config`` returns ``import_file_type``
   alongside the existing ``file_type`` key, both carrying the
   same value. This is the alias that disambiguates IMPORT
   format from on-disk STORAGE format now that the latter is
   moving to parquet-only.

3. The 122ae-1 documentation appears in the relevant docstrings
   so the design decision is discoverable.

All three checks run against real tempfile-on-disk project
configs — project_layout.py has no heavy deps and imports
cleanly in the sandbox.
"""
from __future__ import annotations

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
    from mufasa.project_layout import (project_metadata_from_config,
                                       project_paths_from_config)

    # ==================================================================
    # 1. New keys in project_paths_from_config (both layouts)
    # ==================================================================
    # v1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        toml = root / "project.toml"
        toml.write_text(textwrap.dedent("""
            project_layout_version = 1

            [project]
            name = "smoke_122ae1_v1"
            version = "0.0.1"

            [pose]
            file_type = "parquet"
        """).strip() + "\n")
        paths = project_paths_from_config(toml)

        check(
            "v1 layout returns 'derived_features_dir' key",
            "derived_features_dir" in paths,
        )
        check(
            "v1 layout returns 'derived_labels_dir' key",
            "derived_labels_dir" in paths,
        )
        # Shape: '<root>/derived/features/' / '<root>/derived/labels/'
        check(
            "v1 derived_features_dir is "
            "'<root>/derived/features'",
            paths.get("derived_features_dir", "")
                 .replace("\\", "/").endswith("/derived/features"),
        )
        check(
            "v1 derived_labels_dir is "
            "'<root>/derived/labels'",
            paths.get("derived_labels_dir", "")
                 .replace("\\", "/").endswith("/derived/labels"),
        )
        # Anchored under project_root
        check(
            "v1 derived_features_dir is under project_root",
            paths.get("derived_features_dir", "")
                 .startswith(paths.get("project_root", "x")),
        )
        check(
            "v1 derived_labels_dir is under project_root",
            paths.get("derived_labels_dir", "")
                 .startswith(paths.get("project_root", "x")),
        )

    # Legacy
    with tempfile.TemporaryDirectory() as tmp:
        proj_dir = Path(tmp) / "LegacyProj"
        proj_dir.mkdir()
        ini = proj_dir / "project_config.ini"
        ini.write_text(textwrap.dedent(f"""
            [General settings]
            project_path = {proj_dir}
            workflow_file_type = csv
        """).strip() + "\n")
        paths = project_paths_from_config(ini)

        check(
            "legacy layout returns 'derived_features_dir' key",
            "derived_features_dir" in paths,
        )
        check(
            "legacy layout returns 'derived_labels_dir' key",
            "derived_labels_dir" in paths,
        )
        check(
            "legacy derived_features_dir is "
            "'<project>/derived/features'",
            paths.get("derived_features_dir", "")
                 .replace("\\", "/")
                 .endswith(f"{proj_dir.name}/derived/features"),
        )
        check(
            "legacy derived_labels_dir is "
            "'<project>/derived/labels'",
            paths.get("derived_labels_dir", "")
                 .replace("\\", "/")
                 .endswith(f"{proj_dir.name}/derived/labels"),
        )
        # The previous 122ab keys must still be present — no
        # regression.
        for key in ("features_extracted_dir",
                    "targets_inserted_dir",
                    "machine_results_dir",
                    "roi_definitions_path"):
            check(
                f"legacy layout still returns 122ab key {key!r}",
                key in paths,
            )

    # ==================================================================
    # 2. import_file_type alias in metadata (both layouts)
    # ==================================================================
    # v1: explicit file_type = parquet
    with tempfile.TemporaryDirectory() as tmp:
        toml = Path(tmp) / "project.toml"
        toml.write_text(textwrap.dedent("""
            project_layout_version = 1

            [project]
            name = "smoke_122ae1_meta_v1"
            version = "0.0.1"

            [pose]
            file_type = "parquet"
            animal_count = 1
            body_parts = ["nose", "tail"]
        """).strip() + "\n")
        meta = project_metadata_from_config(toml)
        check(
            "v1 metadata includes 'import_file_type' key",
            "import_file_type" in meta,
        )
        check(
            "v1 metadata still includes 'file_type' key "
            "(back-compat)",
            "file_type" in meta,
        )
        check(
            "v1 'import_file_type' equals 'file_type' "
            "(same source-of-truth)",
            meta.get("import_file_type") == meta.get("file_type"),
        )
        check(
            "v1 'import_file_type' is 'parquet' from "
            "[pose].file_type",
            meta.get("import_file_type") == "parquet",
        )

    # Legacy: workflow_file_type = csv
    with tempfile.TemporaryDirectory() as tmp:
        proj_dir = Path(tmp) / "LegacyMeta"
        proj_dir.mkdir()
        ini = proj_dir / "project_config.ini"
        ini.write_text(textwrap.dedent(f"""
            [General settings]
            project_path = {proj_dir}
            workflow_file_type = csv
            animal_no = 1

            [SML settings]
            no_targets = 0
        """).strip() + "\n")
        meta = project_metadata_from_config(ini)
        check(
            "legacy metadata includes 'import_file_type' key",
            "import_file_type" in meta,
        )
        check(
            "legacy metadata still includes 'file_type' key "
            "(back-compat)",
            "file_type" in meta,
        )
        check(
            "legacy 'import_file_type' equals 'file_type'",
            meta.get("import_file_type") == meta.get("file_type"),
        )
        check(
            "legacy 'import_file_type' is 'csv' from "
            "workflow_file_type",
            meta.get("import_file_type") == "csv",
        )

    # ==================================================================
    # 3. 122ae-1 documentation discoverability
    # ==================================================================
    layout_src = (REPO_ROOT / "mufasa"
                  / "project_layout.py").read_text()
    check(
        "project_layout module docstring records the 122ae "
        "direction (parquet-only derived storage)",
        "122ae" in layout_src and "parquet-only" in layout_src,
    )
    check(
        "project_layout module docstring updates the v1 tree "
        "to show per-family features",
        "features/<family>/" in layout_src,
    )
    check(
        "project_layout module docstring updates the v1 tree "
        "to show labels split out from targets_inserted",
        "labels/<video>.parquet" in layout_src,
    )
    check(
        "project_paths_from_config docstring documents "
        "'derived_features_dir' key",
        "derived_features_dir" in layout_src
        and "derived/features" in layout_src,
    )
    check(
        "project_paths_from_config docstring documents "
        "'derived_labels_dir' key",
        "derived_labels_dir" in layout_src,
    )
    check(
        "project_metadata_from_config docstring explains the "
        "import_file_type semantic",
        "import_file_type" in layout_src
        and "IMPORTER" in layout_src,
    )

    print(
        f"smoke_122ae1_parquet_foundation: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
