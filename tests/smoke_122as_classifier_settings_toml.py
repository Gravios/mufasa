"""
tests/smoke_122as_classifier_settings_toml.py
==============================================

Patch 122as: defines TOML schemas for per-classifier
``[classifier_inference.<name>]`` sub-tables and
``[classifier_training]`` flat table. Adds 4 helper
functions in :mod:`mufasa.project_layout`:

* :func:`read_classifier_inference_settings(config_path)`
* :func:`write_classifier_inference_settings(config_path, settings)`
* :func:`read_classifier_training_settings(config_path)`
* :func:`write_classifier_training_settings(config_path, settings)`

These transparently handle both ``.toml`` and ``.ini`` projects.
The TOML→CP translator
(:mod:`mufasa.utils.toml_to_configparser`) is extended to
populate the legacy CP sections from the new TOML data, so the
existing backends (:class:`InferenceBatch`,
:class:`TrainRandomForestClassifier`) read user-set values on
v1 projects without any backend code change.

122ap's :class:`RunInferenceForm` and 122aq's
:class:`TrainClassifierForm` now use these helpers and drop
their RuntimeError on TOML projects.

Coverage
--------
Behavioural — exercises the helpers with real TOML and INI
fixtures (no Qt dependency).

1. Helpers exist and are importable.
2. Inference settings round-trip on a TOML project: write →
   read back → equal.
3. Inference settings round-trip on an INI project (legacy
   classifier ordinal shape).
4. Training settings round-trip on a TOML project (bool / int
   / float types preserved).
5. Training settings round-trip on an INI project (values
   come back as strings, caller casts).
6. TOML→CP translator: ``[classifier_inference.<name>]`` →
   ``[SML settings] model_path_N`` / ``[threshold_settings]
   threshold_N`` / ``[Minimum_bout_lengths] min_bout_N``.
7. TOML→CP translator: ``[classifier_training]`` →
   ``[create ensemble settings]``, defaults preserved for
   keys the user didn't set.
8. Form code references the new helpers and no longer raises
   RuntimeError on TOML projects.
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


def _write_v1_toml(tmp: Path, classifiers: list[str]) -> Path:
    toml = tmp / "project.toml"
    target_lines = "\n".join(f'    "{c}",' for c in classifiers)
    toml.write_text(textwrap.dedent(f"""
        project_layout_version = 1

        [project]
        name = "smoke_122as"
        version = "0.0.1"

        [pose]
        file_type = "csv"
        animal_count = 1
        body_parts = ["nose"]

        [classifiers]
        targets = [
        {target_lines}
        ]
    """).strip() + "\n")
    return toml


def _write_legacy_ini(tmp: Path, classifiers: list[str]) -> Path:
    proj = tmp / "project_folder"
    proj.mkdir()
    ini = tmp / "project_config.ini"
    lines: list[str] = [
        "[General settings]",
        f"project_path = {proj}",
        "workflow_file_type = csv",
        "",
        "[SML settings]",
        f"no_targets = {len(classifiers)}",
    ]
    for i, name in enumerate(classifiers, start=1):
        lines.append(f"target_name_{i} = {name}")
        lines.append(f"model_path_{i} = ")
    lines.append("")
    lines.append("[threshold_settings]")
    for i in range(1, len(classifiers) + 1):
        lines.append(f"threshold_{i} = NaN")
    lines.append("")
    lines.append("[Minimum_bout_lengths]")
    for i in range(1, len(classifiers) + 1):
        lines.append(f"min_bout_{i} = NaN")
    ini.write_text("\n".join(lines) + "\n")
    return ini


def main() -> int:
    # ==================================================================
    # 1. Helpers exist + importable
    # ==================================================================
    try:
        from mufasa.project_layout import (
            read_classifier_inference_settings,
            write_classifier_inference_settings,
            read_classifier_training_settings,
            write_classifier_training_settings,
        )
        check("4 helpers importable from mufasa.project_layout", True)
    except ImportError as exc:
        check("4 helpers importable from mufasa.project_layout",
              False, detail=str(exc))
        return 1

    # ==================================================================
    # 2. Inference round-trip — TOML
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack", "grooming"])
        inference_settings = {
            "attack": {
                "model_path":  "/abs/path/to/attack.sav",
                "threshold":   0.55,
                "min_bout_ms": 200,
            },
            "grooming": {
                "model_path":  "/abs/path/to/grooming.sav",
                "threshold":   0.40,
                "min_bout_ms": 100,
            },
        }
        write_classifier_inference_settings(toml, inference_settings)
        read_back = read_classifier_inference_settings(toml)
        check(
            "TOML inference round-trip: attack settings preserved",
            read_back.get("attack", {}).get("model_path")
            == "/abs/path/to/attack.sav"
            and read_back.get("attack", {}).get("threshold") == 0.55
            and read_back.get("attack", {}).get("min_bout_ms") == 200,
        )
        check(
            "TOML inference round-trip: grooming settings preserved",
            read_back.get("grooming", {}).get("model_path")
            == "/abs/path/to/grooming.sav"
            and read_back.get("grooming", {}).get("threshold") == 0.40
            and read_back.get("grooming", {}).get("min_bout_ms") == 100,
        )

    # ==================================================================
    # 3. Inference round-trip — INI
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        ini = _write_legacy_ini(tmp, ["attack", "grooming"])
        inference_settings = {
            "attack": {
                "model_path":  "/abs/path/to/attack.sav",
                "threshold":   0.55,
                "min_bout_ms": 200,
            },
        }
        write_classifier_inference_settings(ini, inference_settings)
        read_back = read_classifier_inference_settings(ini)
        check(
            "INI inference round-trip: attack settings preserved",
            read_back.get("attack", {}).get("model_path")
            == "/abs/path/to/attack.sav"
            and abs(read_back.get("attack", {}).get("threshold", 0)
                    - 0.55) < 1e-6
            and read_back.get("attack", {}).get("min_bout_ms") == 200,
        )
        check(
            "INI inference round-trip: untouched grooming has no "
            "settings (only attack was written)",
            "grooming" not in read_back,
        )

    # ==================================================================
    # 4. Training round-trip — TOML (typed)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack"])
        training_settings = {
            "model_to_run":        "RF",
            "classifier":          "attack",
            "rf_n_estimators":     1500,
            "rf_max_features":     "sqrt",
            "rf_criterion":        "gini",
            "train_test_size":     "0.2",
            "train_test_split_type": "FRAMES",
            "rf_min_sample_leaf":  1,
            "under_sample_setting": "random undersample",
            "under_sample_ratio":  "1.0",
            "over_sample_setting": "None",
            "over_sample_ratio":   "1.0",
            "class_weights":       "balanced",
            "rf_metadata":         True,
            "generate_classification_report": True,
            "generate_features_importance_bar_graph": True,
            "n_feature_importance_bars": 10,
            "calculate_shap_scores": False,
        }
        write_classifier_training_settings(toml, training_settings)
        read_back = read_classifier_training_settings(toml)
        check(
            "TOML training round-trip: int preserved as int",
            read_back.get("rf_n_estimators") == 1500
            and isinstance(read_back.get("rf_n_estimators"), int),
        )
        check(
            "TOML training round-trip: bool preserved as bool",
            read_back.get("rf_metadata") is True
            and isinstance(read_back.get("rf_metadata"), bool),
        )
        check(
            "TOML training round-trip: str preserved",
            read_back.get("classifier") == "attack"
            and read_back.get("rf_max_features") == "sqrt",
        )
        check(
            "TOML training round-trip: undersample_setting + "
            "class_weights preserved",
            read_back.get("under_sample_setting")
            == "random undersample"
            and read_back.get("class_weights") == "balanced",
        )

    # ==================================================================
    # 5. Training round-trip — INI (stringified)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        ini = _write_legacy_ini(tmp, ["attack"])
        training_settings = {
            "rf_n_estimators": 1500,
            "rf_metadata":     True,
            "classifier":      "attack",
        }
        write_classifier_training_settings(ini, training_settings)
        read_back = read_classifier_training_settings(ini)
        # All come back as strings on INI
        check(
            "INI training round-trip: values come back as strings",
            read_back.get("rf_n_estimators") == "1500"
            and read_back.get("rf_metadata") == "True"
            and read_back.get("classifier") == "attack",
        )

    # ==================================================================
    # 6. TOML→CP translator picks up [classifier_inference.<name>]
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack", "grooming"])
        write_classifier_inference_settings(toml, {
            "attack": {
                "model_path":  "/abs/path/to/attack.sav",
                "threshold":   0.55,
                "min_bout_ms": 200,
            },
            "grooming": {
                "model_path":  "/abs/path/to/grooming.sav",
                "threshold":   0.40,
                "min_bout_ms": 100,
            },
        })
        from mufasa.utils.toml_to_configparser import (
            read_project_toml_as_configparser,
        )
        cp = read_project_toml_as_configparser(toml)
        check(
            "translator: model_path_1 = attack's path",
            cp.get("SML settings", "model_path_1")
            == "/abs/path/to/attack.sav",
        )
        check(
            "translator: model_path_2 = grooming's path",
            cp.get("SML settings", "model_path_2")
            == "/abs/path/to/grooming.sav",
        )
        check(
            "translator: threshold_1 = 0.55",
            cp.get("threshold_settings", "threshold_1") == "0.55",
        )
        check(
            "translator: threshold_2 = 0.4",
            cp.get("threshold_settings", "threshold_2") == "0.4",
        )
        check(
            "translator: min_bout_1 = 200",
            cp.get("Minimum_bout_lengths", "min_bout_1") == "200",
        )
        check(
            "translator: min_bout_2 = 100",
            cp.get("Minimum_bout_lengths", "min_bout_2") == "100",
        )

    # ==================================================================
    # 7. TOML→CP translator picks up [classifier_training] keys,
    #    defaults preserved for unset keys
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        toml = _write_v1_toml(tmp, ["attack"])
        write_classifier_training_settings(toml, {
            "rf_n_estimators": 5000,
            "rf_max_features": "log2",
            "rf_metadata":     True,
        })
        cp = read_project_toml_as_configparser(toml)
        check(
            "translator: rf_n_estimators overlay",
            cp.get("create ensemble settings", "rf_n_estimators")
            == "5000",
        )
        check(
            "translator: rf_max_features overlay",
            cp.get("create ensemble settings", "rf_max_features")
            == "log2",
        )
        check(
            "translator: rf_metadata bool → 'True'",
            cp.get("create ensemble settings", "rf_metadata")
            == "True",
        )
        check(
            "translator: rf_criterion default preserved (was not "
            "set in [classifier_training])",
            cp.get("create ensemble settings", "rf_criterion")
            == "entropy",
        )
        check(
            "translator: train_test_size default preserved",
            cp.get("create ensemble settings", "train_test_size")
            == "0.2",
        )

    # ==================================================================
    # 8. Forms no longer raise RuntimeError on TOML projects;
    #    helpers are referenced at code level
    # ==================================================================
    run_inf_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                   / "run_inference.py").read_text()
    train_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                 / "train_classifier.py").read_text()
    check(
        "run_inference.py uses write_classifier_inference_settings",
        "write_classifier_inference_settings" in run_inf_src,
    )
    check(
        "run_inference.py uses read_classifier_inference_settings",
        "read_classifier_inference_settings" in run_inf_src,
    )
    check(
        "run_inference.py no longer raises RuntimeError about "
        "[classifier_inference] section",
        not any(
            "raise RuntimeError" in line
            and not line.lstrip().startswith("#")
            for line in run_inf_src.splitlines()
        ),
    )
    check(
        "train_classifier.py uses write_classifier_training_settings",
        "write_classifier_training_settings" in train_src,
    )
    check(
        "train_classifier.py uses read_classifier_training_settings",
        "read_classifier_training_settings" in train_src,
    )
    check(
        "train_classifier.py no longer raises RuntimeError about "
        "[classifier_training] section",
        not any(
            "raise RuntimeError" in line
            and not line.lstrip().startswith("#")
            for line in train_src.splitlines()
        ),
    )
    check(
        "run_inference.py records 122as",
        "122as" in run_inf_src,
    )
    check(
        "train_classifier.py records 122as",
        "122as" in train_src,
    )

    print(
        f"smoke_122as_classifier_settings_toml: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
