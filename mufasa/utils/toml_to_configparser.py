"""
mufasa.utils.toml_to_configparser
=================================

Patch 122e: bridge between v1 ``project.toml`` and legacy
``configparser`` consumers.

Background
----------
:class:`mufasa.mixins.config_reader.ConfigReader` is inherited by
268+ files across the codebase. Its API reads project metadata
via ``configparser`` (sections like ``[General settings]`` and
``[SML settings]``). Converting every dependent to a new
TOML-native API would be weeks of work and a high regression
risk.

Strategy
--------
For v1 projects, we keep ConfigReader's API but synthesize a
:class:`configparser.ConfigParser` from the TOML data with the
legacy section/key names. Dependents call ``self.config.get(
"General settings", "project_path")`` and get back the v1
project root — same call shape, different file format under
the hood.

Only metadata translation lives here. Filesystem path
attributes (``input_csv_dir``, ``outlier_corrected_dir``, etc.)
are handled separately in
:meth:`ConfigReader._init_v1_paths` since their values depend
on filesystem state (e.g. which ``derived/<stage>/<run>/`` dirs
exist), not just on TOML content.

Schema coverage
---------------
The synthetic ConfigParser populates every section the legacy
``ProjectConfigCreator`` used to write, even when the v1 TOML
has no equivalent field. Missing values default to the legacy
``Dtypes.NONE.value`` sentinel (``"NaN"``) so existing
``read_config_entry(..., default_value=...)`` calls behave
identically.
"""
from __future__ import annotations

import platform
import tomllib
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict, Union


def _stringify(value: Any) -> str:
    """ConfigParser stores everything as strings. Match the legacy
    serialization for the few non-string types we deal with."""
    if isinstance(value, bool):
        # Legacy code writes Python bools as "True"/"False".
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def project_toml_to_configparser(
    toml_data: Dict[str, Any],
    project_root: Union[str, Path],
) -> ConfigParser:
    """Build a :class:`ConfigParser` that mirrors what the legacy
    ``ProjectConfigCreator`` used to write, populated from ``toml_data``.

    :param toml_data: parsed contents of ``project.toml`` (just call
        ``tomllib.load(open(p, "rb"))``; pass the result here).
    :param project_root: absolute path to the v1 project root. Used
        for ``[General settings] project_path`` and to point
        ``[SML settings] model_dir`` / ``model_path_N`` at the
        v1 ``<root>/models/`` tree.
    :returns: a fully-populated ConfigParser. Every legacy section
        is present even when the TOML has no corresponding field;
        missing values default to ``"NaN"`` (the legacy "not
        configured" sentinel).

    Notes
    -----
    * ``[General settings].project_path`` is set to the v1 root
      itself, not to a ``<root>/project_folder/`` subdir. v1 has
      no such subdir; ConfigReader's path-attribute computation
      handles the rest of the layout difference.
    * Classifier-related fields (``[SML settings] target_name_N``,
      ``model_path_N``; ``[threshold_settings] threshold_N``;
      ``[Minimum_bout_lengths] min_bout_N``) are generated from
      ``[classifiers].targets`` in the TOML. Empty target list
      → ``no_targets = "0"`` and no ``target_name_*`` keys, which
      matches the empty-classifier semantics tested in
      ``smoke_empty_classifier``.
    * The ``[create ensemble settings]`` block uses the same
      "all NaN by default" placeholders the legacy creator wrote,
      so downstream ML-param reads return the legacy sentinel
      until classifier training overwrites them.
    """
    project_root = str(Path(project_root))
    pose = toml_data.get("pose", {})
    classifiers = toml_data.get("classifiers", {})
    outlier = toml_data.get("outlier_settings", {})
    targets = list(classifiers.get("targets", []))

    cp = ConfigParser(allow_no_value=True)

    # ------------------------------------------------------------------
    # [General settings]
    # ------------------------------------------------------------------
    # project_path = v1 root (not <root>/project_folder/). ConfigReader
    # branches on the .toml suffix and overrides the path attributes
    # that previously assumed a project_folder subdir.
    cp.add_section("General settings")
    cp.set("General settings", "project_path", project_root)
    cp.set(
        "General settings", "project_name",
        _stringify(toml_data.get("project_name", "")),
    )
    cp.set(
        "General settings", "file_type",
        _stringify(pose.get("file_type", "csv")),
    )
    cp.set(
        "General settings", "workflow_file_type",
        _stringify(pose.get("file_type", "csv")),
    )
    cp.set(
        "General settings", "animal_no",
        _stringify(pose.get("animal_count", 1)),
    )
    cp.set(
        "General settings", "os_system",
        _stringify(toml_data.get("os_platform", platform.system())),
    )

    # ------------------------------------------------------------------
    # [SML settings]
    # ------------------------------------------------------------------
    # Classifier list + per-classifier model paths. The legacy creator
    # set model_path_N = <models_dir>/<name>.sav; we mirror that even
    # though the actual model lives at <project>/models/<name>/model.npz
    # in v1 — legacy consumers expect the .sav extension and we don't
    # want to surprise them. The model file may not exist at this path
    # until classifier training runs; reads through ConfigReader's
    # `model_path_N` attribute should be treated as "the path where the
    # model is expected to land".
    #
    # Patch 122as: per-classifier inference settings from the v1
    # [classifier_inference.<name>] sub-tables override these defaults
    # so the InferenceBatch reader sees real model paths / thresholds
    # / min bouts (not NaN placeholders) when the user has run the
    # RunInferenceForm against a TOML project.
    models_dir = str(Path(project_root) / "models")
    inference_settings = toml_data.get("classifier_inference", {})
    if not isinstance(inference_settings, dict):
        inference_settings = {}
    cp.add_section("SML settings")
    cp.set("SML settings", "model_dir", models_dir)
    cp.set("SML settings", "no_targets", str(len(targets)))
    for i, target in enumerate(targets, start=1):
        cp.set("SML settings", f"target_name_{i}", target)
        entry = inference_settings.get(target, {})
        if not isinstance(entry, dict):
            entry = {}
        model_path = entry.get(
            "model_path",
            str(Path(models_dir) / f"{target}.sav"),
        )
        cp.set("SML settings", f"model_path_{i}", str(model_path))

    # ------------------------------------------------------------------
    # [threshold_settings]
    # ------------------------------------------------------------------
    cp.add_section("threshold_settings")
    for i, target in enumerate(targets, start=1):
        entry = inference_settings.get(target, {})
        if not isinstance(entry, dict):
            entry = {}
        thr = entry.get("threshold")
        cp.set(
            "threshold_settings", f"threshold_{i}",
            _stringify(thr) if thr is not None else "NaN",
        )
    cp.set("threshold_settings", "sklearn_bp_prob_thresh", "0.0")

    # ------------------------------------------------------------------
    # [Minimum_bout_lengths]
    # ------------------------------------------------------------------
    cp.add_section("Minimum_bout_lengths")
    for i, target in enumerate(targets, start=1):
        entry = inference_settings.get(target, {})
        if not isinstance(entry, dict):
            entry = {}
        mb = entry.get("min_bout_ms")
        cp.set(
            "Minimum_bout_lengths", f"min_bout_{i}",
            _stringify(mb) if mb is not None else "NaN",
        )

    # ------------------------------------------------------------------
    # Plot / ROI / directionality / movement sections — empty by
    # default; populated later by their respective workflow forms
    # writing back into project.toml. The legacy creator added these
    # sections without keys so `config.has_section(...)` succeeds.
    # ------------------------------------------------------------------
    for s in (
        "Frame settings",
        "Line plot settings",
        "Path plot settings",
        "ROI settings",
        "Directionality settings",
        "process movement settings",
    ):
        cp.add_section(s)
    cp.set("Frame settings", "distance_mm", "0.0")

    # ------------------------------------------------------------------
    # [create ensemble settings]
    # ------------------------------------------------------------------
    # Pose preset + ML training knobs. The legacy creator dumped
    # ~25 keys here, mostly with "NaN" placeholders. Match exactly so
    # downstream `read_config_entry(...)` calls return the same
    # values they always have.
    #
    # Patch 122as: training settings from the v1 [classifier_training]
    # table override these defaults so TrainRandomForestClassifier
    # sees the user's saved hyperparameters / evaluation toggles on
    # TOML projects.
    training_settings = toml_data.get("classifier_training", {})
    if not isinstance(training_settings, dict):
        training_settings = {}
    cp.add_section("create ensemble settings")
    _ce_defaults = {
        "pose_estimation_body_parts":
            _stringify(pose.get("pose_config_code", "user_defined")),
        "classifier":                 "NaN",
        "train_test_size":            "0.2",
        "under_sample_setting":       "NaN",
        "under_sample_ratio":         "NaN",
        "over_sample_setting":        "NaN",
        "over_sample_ratio":          "NaN",
        "rf_n_estimators":            "2000",
        "rf_min_sample_leaf":         "1",
        "rf_max_features":            "sqrt",
        "rf_n_jobs":                  "-1",
        "rf_criterion":               "entropy",
        "rf_metadata":                "NaN",
        "ex_decision_tree":           "NaN",
        "ex_decision_tree_fancy":     "NaN",
        "rf_feature_importance_log":  "NaN",
        "feature_importance_bar_chart": "NaN",
        "permutation_feature_importance": "NaN",
        "learning_curve":             "NaN",
        "precision_recall":           "NaN",
        "n_feature_importance_bars":  "NaN",
        "learning_curve_k_splits":    "NaN",
        "learningcurve_data_splits":  "NaN",
    }
    # Defaults first, then overlay the user's [classifier_training]
    # values. Stringify since ConfigParser is string-only on the wire.
    for key, default in _ce_defaults.items():
        cp.set("create ensemble settings", key, default)
    for key, value in training_settings.items():
        cp.set(
            "create ensemble settings", key, _stringify(value),
        )

    # ------------------------------------------------------------------
    # [Multi animal IDs]
    # ------------------------------------------------------------------
    # Legacy stored a comma-separated string. v1 stores a list.
    # Round-trip preserves the user's animal names if they were set.
    cp.add_section("Multi animal IDs")
    animal_ids = pose.get("animal_ids", [])
    if animal_ids:
        cp.set(
            "Multi animal IDs", "id_list",
            ",".join(str(a) for a in animal_ids),
        )
    else:
        cp.set("Multi animal IDs", "id_list", "NaN")

    # ------------------------------------------------------------------
    # [Outlier settings]
    # ------------------------------------------------------------------
    cp.add_section("Outlier settings")
    cp.set(
        "Outlier settings", "movement_criterion",
        _stringify(outlier.get("movement_criterion", "NaN")),
    )
    cp.set(
        "Outlier settings", "location_criterion",
        _stringify(outlier.get("location_criterion", "NaN")),
    )

    return cp


def read_project_toml_as_configparser(
    config_path: Union[str, Path],
) -> ConfigParser:
    """Convenience: load ``config_path`` (a ``.toml`` file) and
    return the synthesized :class:`ConfigParser`.

    Used by :func:`mufasa.utils.read_write.read_config_file` when
    the suffix is ``.toml``. Standalone so other callers (tests,
    one-off scripts) can also use it.
    """
    p = Path(config_path)
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return project_toml_to_configparser(data, project_root=p.parent)


__all__ = [
    "project_toml_to_configparser",
    "read_project_toml_as_configparser",
]
