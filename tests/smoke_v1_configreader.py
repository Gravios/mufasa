"""
tests/smoke_v1_configreader.py
==============================

Patch 122e: smoke test for the TOML-aware ConfigReader path.

Three layers covered:

1. **Behavioral** — :func:`project_toml_to_configparser` produces
   a :class:`ConfigParser` with the legacy section/key names from
   a v1 ``project.toml``. Empty-classifier handling preserved.
2. **Routing** — :func:`mufasa.utils.read_write.read_config_file`
   branches on suffix: ``.toml`` → synthetic ConfigParser via the
   shim; everything else → legacy INI parsing.
3. **AST** — :class:`mufasa.mixins.config_reader.ConfigReader`
   carries a v1 detection block in ``__init__``, a v1 body-parts
   branch, and an ``_apply_v1_path_overrides`` method covering
   the path attributes legacy backends consume.

The full behavioral test (instantiate ConfigReader against a v1
``project.toml`` and inspect its attribute surface) requires
heavy runtime deps (cv2, h5py, trafaret, …) that aren't always
available. We surface that limitation honestly via the AST
checks: the wiring is verifiable; the runtime behavior is the
user's to confirm on a full install.
"""
from __future__ import annotations

import ast
import configparser
import os
import sys
import tempfile
import tomllib
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
    from mufasa.utils.config_creator import ProjectConfigCreator
    from mufasa.utils.toml_to_configparser import (
        project_toml_to_configparser,
        read_project_toml_as_configparser,
    )

    # ==================================================================
    # Layer 1 — TOML → ConfigParser behavioral
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1a: project with classifiers + multiple animals
        creator = ProjectConfigCreator(
            project_path=str(tmp),
            project_name="exp_a",
            target_list=["Attack", "Sniffing"],
            pose_estimation_bp_cnt="7",
            body_part_config_idx=1,
            animal_cnt=2,
            file_type="csv",
        )
        cp = read_project_toml_as_configparser(creator.config_path)
        check(
            "shim returns a ConfigParser instance",
            isinstance(cp, configparser.ConfigParser),
        )
        # Legacy section names present
        for section in (
            "General settings", "SML settings", "threshold_settings",
            "Minimum_bout_lengths", "create ensemble settings",
            "Multi animal IDs", "Outlier settings",
        ):
            check(
                f"shim section [{section}] present",
                cp.has_section(section),
            )
        check(
            "[General settings] project_path = v1 root",
            cp.get("General settings", "project_path")
            == str(creator.project_root),
        )
        check(
            "[General settings] file_type = csv",
            cp.get("General settings", "file_type") == "csv",
        )
        check(
            "[General settings] animal_no = 2",
            cp.get("General settings", "animal_no") == "2",
        )
        check(
            "[SML settings] no_targets = 2",
            cp.get("SML settings", "no_targets") == "2",
        )
        check(
            "[SML settings] target_name_1 = Attack",
            cp.get("SML settings", "target_name_1") == "Attack",
        )
        check(
            "[SML settings] target_name_2 = Sniffing",
            cp.get("SML settings", "target_name_2") == "Sniffing",
        )
        check(
            "[SML settings] model_path_1 ends in Attack.sav",
            cp.get("SML settings", "model_path_1").endswith("Attack.sav"),
        )
        check(
            "[Multi animal IDs] id_list comma-joined",
            cp.get("Multi animal IDs", "id_list") == "Animal_1,Animal_2",
        )
        check(
            "[Outlier settings] movement_criterion preserved",
            cp.get("Outlier settings", "movement_criterion") == "NaN",
        )
        # Per-classifier thresholds and min_bouts auto-populated
        check(
            "[threshold_settings] threshold_1 = NaN",
            cp.get("threshold_settings", "threshold_1") == "NaN",
        )
        check(
            "[threshold_settings] threshold_2 = NaN",
            cp.get("threshold_settings", "threshold_2") == "NaN",
        )
        check(
            "[Minimum_bout_lengths] min_bout_2 = NaN",
            cp.get("Minimum_bout_lengths", "min_bout_2") == "NaN",
        )

        # 1b: empty classifier list — no target_name_* keys
        creator2 = ProjectConfigCreator(
            project_path=str(tmp),
            project_name="exp_empty",
            target_list=[],
            pose_estimation_bp_cnt="7",
            body_part_config_idx=1,
            animal_cnt=1,
            file_type="csv",
        )
        cp2 = read_project_toml_as_configparser(creator2.config_path)
        check(
            "empty classifier: no_targets = 0",
            cp2.get("SML settings", "no_targets") == "0",
        )
        offenders = [
            k for k in cp2["SML settings"]
            if k.startswith("target_name_")
        ]
        check(
            "empty classifier: no target_name_* keys",
            not offenders,
            detail=f"unexpected: {offenders}",
        )

        # ==============================================================
        # Layer 2 — read_config_file routing (AST-only; the module
        # pulls heavy runtime deps that aren't always installed in
        # the sandbox, so we verify the routing logic at the source
        # level. Behavioral verification happens in a full env.)
        # ==============================================================
        rw_path = REPO_ROOT / "mufasa" / "utils" / "read_write.py"
        rw_src = rw_path.read_text()
        rw_tree = ast.parse(rw_src)
        read_config_fn = None
        for node in ast.walk(rw_tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "read_config_file"
            ):
                read_config_fn = node
                break
        check(
            "read_config_file function exists in read_write.py",
            read_config_fn is not None,
        )
        if read_config_fn is not None:
            body_src = ast.unparse(read_config_fn)
            check(
                "read_config_file branches on .toml suffix",
                ".toml" in body_src,
            )
            check(
                "read_config_file delegates TOML to the shim",
                "read_project_toml_as_configparser" in body_src,
            )
            check(
                "read_config_file's TOML branch is case-insensitive",
                ".lower()" in body_src,
            )
            check(
                "read_config_file's legacy INI branch preserved",
                "configparser" in body_src.lower()
                and "config.read" in body_src,
            )

    # ==================================================================
    # Layer 3 — AST checks on ConfigReader
    # ==================================================================
    cr_src = (REPO_ROOT / "mufasa" / "mixins" / "config_reader.py").read_text()
    cr_tree = ast.parse(cr_src)
    cr_class = None
    for node in ast.walk(cr_tree):
        if isinstance(node, ast.ClassDef) and node.name == "ConfigReader":
            cr_class = node
            break
    check("ConfigReader class exists", cr_class is not None)
    if cr_class is None:
        print(f"smoke_v1_configreader: "
              f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed")
        return 1

    methods = {
        n.name: n for n in cr_class.body
        if isinstance(n, ast.FunctionDef)
    }
    check(
        "ConfigReader.__init__ defined",
        "__init__" in methods,
    )
    check(
        "ConfigReader._apply_v1_path_overrides defined",
        "_apply_v1_path_overrides" in methods,
    )

    init_src = ast.unparse(methods["__init__"])
    check(
        "__init__ sets self._is_v1 from config_path suffix",
        "self._is_v1" in init_src and ".toml" in init_src,
    )
    check(
        "__init__ pre-loads TOML data when v1",
        "tomllib.load" in init_src
        and "_v1_toml_data" in init_src,
    )
    check(
        "__init__ has v1 body_parts branch using TOML data",
        "if self._is_v1" in init_src
        and "body_parts_lst" in init_src
        and "body_parts" in init_src,
    )
    check(
        "__init__ invokes _apply_v1_path_overrides for v1 projects",
        "_apply_v1_path_overrides" in init_src,
    )

    override_src = ast.unparse(methods["_apply_v1_path_overrides"])
    # Path attributes that must be overridden for v1
    expected_attrs = [
        # source data
        "self.input_csv_dir",
        "self.video_dir",
        "self.video_info_path",
        # multi-run stages
        "self.outlier_corrected_dir",
        "self.outlier_corrected_movement_dir",
        "self.features_dir",
        "self.targets_folder",
        "self.machine_results_dir",
        # frames
        "self.input_frames_dir",
        "self.frames_output_dir",
        # plots
        "self.line_plot_dir",
        "self.gantt_plot_dir",
        "self.path_plot_dir",
        "self.probability_plot_dir",
        "self.heatmap_clf_location_dir",
        "self.heatmap_location_dir",
        "self.sklearn_plot_dir",
        # misc
        "self.shap_logs_path",
        "self.directionality_df_dir",
        "self.body_part_directionality_df_dir",
        "self.roi_features_save_dir",
        "self.detailed_roi_data_dir",
        "self.clf_validation_dir",
        "self.clf_data_validation_dir",
        "self.cue_lights_data_dir",
        # logs / roi / configs
        "self.logs_path",
        "self.roi_coordinates_path",
        "self.configs_meta_dir",
        # re-globbed file lists
        "self.input_csv_paths",
        "self.feature_file_paths",
        "self.target_file_paths",
        "self.outlier_corrected_paths",
        "self.machine_results_paths",
    ]
    for attr in expected_attrs:
        check(
            f"_apply_v1_path_overrides sets {attr}",
            attr in override_src,
        )
    check(
        "_apply_v1_path_overrides uses sources/ and derived/ paths",
        "sources" in override_src and "derived" in override_src,
    )
    check(
        "_apply_v1_path_overrides resolves latest-run for stages",
        "is_run_id" in override_src
        and "_latest_run_or_parent" in override_src,
    )

    # read_config_file routing is verified in Layer 2 above (AST).

    print(
        f"smoke_v1_configreader: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
