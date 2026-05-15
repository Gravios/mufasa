"""
mufasa.ui_qt.forms.train_classifier
====================================

Qt port of :class:`mufasa.ui.machine_model_settings_ui.MachineModelSettingsPopUp`
plus a Train button that drives :class:`TrainRandomForestClassifier`.

Patch 122aq (this file)
-----------------------
The Tk popup was 556 lines of grid-laid-out :class:`Entry_Box` /
:class:`SimBADropDown` widgets with hand-rolled enable/disable
plumbing for sub-fields gated on parent checkboxes. Replaced
here with a declarative :class:`QFormLayout` per group +
helper functions for the enable/disable cascade.

The Tk popup only persisted settings to ``project_config.ini``
— it never actually trained. Training was invoked separately
from SimBA.py's menu hooks. The Qt port consolidates: the same
Save settings button persists to INI, and a new Train button
runs the trainer end-to-end after persisting.

In-frame + dockable
-------------------
Subclasses :class:`OperationForm` so it lives inline on the
Classifier page like any other section. Pop-out button
re-parents into a :class:`QDockWidget` attached to the
workbench main window — same pattern as 122aj, 122al, 122ap.

What's persisted
----------------
All fields write to whichever config format the project uses.
Legacy ``.ini`` projects write to ``[create_ensemble_settings]``
using the same keys :class:`TrainRandomForestClassifier`'s
settings reader expects. v1 ``.toml`` projects write to the
``[classifier_training]`` table with the same key names
(proper-typed values though — bools as bools, ints as ints).
The TOML→CP translator in 122as injects the values into the
legacy CP section name the trainer reads from.

What's deferred from Tk parity
------------------------------
* **Class-weights "custom" table.** The Tk popup popped a separate
  Toplevel table for per-class weight values when "custom" was
  selected. Custom weights are a corner case (most users pick
  "balanced" or "None"); for now selecting "custom" persists the
  value but doesn't open a Qt editor — user can hand-edit the
  weights in the INI if needed. A QDialog port is a clean
  follow-up.
* **Load / Save preset buttons.** The Tk popup let you load a
  saved meta-config from ``configs/`` and save the current
  settings there. Useful for parameter sweeps but separate scope.
* **Algorithm dropdown.** Tk listed RF/GBC/XGBoost but only
  passed ``[self.clf_options[0]]`` (just RF) to the dropdown —
  the others aren't implemented in TrainRandomForestClassifier.
  Dropped in the Qt port; RF is the only algorithm.
"""
from __future__ import annotations

import configparser
import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from mufasa.ui_qt.workbench import OperationForm


# Algorithm and option vocabulary mirrors Options.* in enums.py
_MAX_FEATURES = ["sqrt", "log2", "None"]
_CRITERION    = ["gini", "entropy"]
_UNDERSAMPLE  = ["None", "random undersample"]
_OVERSAMPLE   = ["None", "SMOTE", "SMOTEENN"]
_CLASS_WEIGHT = ["None", "balanced", "balanced_subsample", "custom"]
_TEST_SIZES   = ["0.1", "0.2", "0.3", "0.4", "0.5"]
_TT_SPLIT     = ["FRAMES", "BOUTS"]
_SHAP_CADENCE = ["1", "10", "100", "1000", "ALL FRAMES"]


class TrainClassifierForm(OperationForm):
    """In-frame Qt port of the model-training settings + runner."""

    title = "Train classifier"
    description = (
        "Configure hyperparameters and evaluation outputs for the "
        "Random Forest classifier trainer, then either save the "
        "settings to <code>project_config.ini</code> or kick off "
        "a full training run. Saved settings are picked up by the "
        "trainer the next time it runs."
    )

    # ----------------------------------------------------------- State
    def __init__(self,
                 parent: Optional[QWidget] = None,
                 config_path: Optional[str] = None) -> None:
        self._docked_widget: Optional[QDockWidget] = None
        super().__init__(parent=parent, config_path=config_path)

    # ----------------------------------------------------------- UI
    def build(self) -> None:
        # --- Behavior picker --------------------------------------- #
        targets = self._read_classifier_targets()
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Classifier (target behavior):", self))
        self.behavior_combo = QComboBox(self)
        if targets:
            self.behavior_combo.addItems(targets)
        else:
            self.behavior_combo.addItem("⟨no classifiers defined⟩")
            self.behavior_combo.setEnabled(False)
        target_row.addWidget(self.behavior_combo, 1)
        target_row.addStretch()
        self.body_layout.addLayout(target_row)

        # --- Two column layout: hyperparams | evaluations ---------- #
        cols = QHBoxLayout()
        cols.addWidget(self._build_hyperparams_group(), 1)
        cols.addWidget(self._build_evaluations_group(), 1)
        self.body_layout.addLayout(cols, 1)

        # --- Action row ------------------------------------------- #
        actions = QHBoxLayout()
        self.save_btn = QPushButton("Save settings only", self)
        self.save_btn.setToolTip(
            "Persist the current settings to project_config.ini "
            "without running training."
        )
        self.save_btn.clicked.connect(self._on_save_only)
        self.pop_out_btn = QPushButton("Pop out ⇱", self)
        self.pop_out_btn.setToolTip(
            "Detach this form into a floating dockable window."
        )
        self.pop_out_btn.clicked.connect(self._toggle_pop_out)
        actions.addWidget(self.save_btn)
        actions.addStretch()
        actions.addWidget(self.pop_out_btn)
        self.body_layout.addLayout(actions)

        self.run_btn.setText("  Train classifier")

        # Hydrate field values from project_config.ini if present
        self._load_from_ini()

    # ----------------------------------------------------------- Hyperparams group
    def _build_hyperparams_group(self) -> QGroupBox:
        box = QGroupBox("Hyperparameters", self)
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        # Integer estimators
        self.estimators = QLineEdit("2000", self)
        self.estimators.setValidator(QIntValidator(1, 100_000, self))
        form.addRow("RF estimators:", self.estimators)

        self.max_features = QComboBox(self)
        self.max_features.addItems(_MAX_FEATURES)
        form.addRow("Max features:", self.max_features)

        self.criterion = QComboBox(self)
        self.criterion.addItems(_CRITERION)
        form.addRow("Criterion:", self.criterion)

        self.test_size = QComboBox(self)
        self.test_size.addItems(_TEST_SIZES)
        self.test_size.setCurrentText("0.2")
        form.addRow("Test size:", self.test_size)

        self.tt_split = QComboBox(self)
        self.tt_split.addItems(_TT_SPLIT)
        form.addRow("Train-test split type:", self.tt_split)

        self.min_leaf = QLineEdit("1", self)
        self.min_leaf.setValidator(QIntValidator(1, 10_000, self))
        form.addRow("Min samples per leaf:", self.min_leaf)

        # Under-sample
        self.undersample_setting = QComboBox(self)
        self.undersample_setting.addItems(_UNDERSAMPLE)
        self.undersample_setting.currentTextChanged.connect(
            self._toggle_undersample,
        )
        form.addRow("Under-sample setting:", self.undersample_setting)
        self.undersample_ratio = QLineEdit("1.0", self)
        self.undersample_ratio.setValidator(
            QDoubleValidator(0.0, 100.0, 3, self),
        )
        self.undersample_ratio.setEnabled(False)
        form.addRow("Under-sample ratio:", self.undersample_ratio)

        # Over-sample
        self.oversample_setting = QComboBox(self)
        self.oversample_setting.addItems(_OVERSAMPLE)
        self.oversample_setting.currentTextChanged.connect(
            self._toggle_oversample,
        )
        form.addRow("Over-sample setting:", self.oversample_setting)
        self.oversample_ratio = QLineEdit("1.0", self)
        self.oversample_ratio.setValidator(
            QDoubleValidator(0.0, 100.0, 3, self),
        )
        self.oversample_ratio.setEnabled(False)
        form.addRow("Over-sample ratio:", self.oversample_ratio)

        # Class weights
        self.class_weight = QComboBox(self)
        self.class_weight.addItems(_CLASS_WEIGHT)
        form.addRow("Class weights:", self.class_weight)

        return box

    def _toggle_undersample(self, text: str) -> None:
        self.undersample_ratio.setEnabled(text != "None")

    def _toggle_oversample(self, text: str) -> None:
        self.oversample_ratio.setEnabled(text != "None")

    # ----------------------------------------------------------- Evaluations group
    def _build_evaluations_group(self) -> QGroupBox:
        box = QGroupBox("Evaluations", self)
        layout = QVBoxLayout(box)

        self.eval_meta_data    = QCheckBox("Save model meta-data file",
                                           self)
        self.eval_dtree_gviz   = QCheckBox(
            "Example decision tree (graphviz)", self,
        )
        self.eval_dtree_dtviz  = QCheckBox(
            "Example decision tree (dtreeviz)", self,
        )
        self.eval_clf_report   = QCheckBox("Classification report",
                                           self)
        self.eval_bar_graph    = QCheckBox(
            "Feature importance bar graph", self,
        )
        self.eval_bar_graph.toggled.connect(self._toggle_bar_graph_n)
        self.eval_bar_graph_n  = QLineEdit("15", self)
        self.eval_bar_graph_n.setValidator(QIntValidator(1, 10_000, self))
        self.eval_bar_graph_n.setEnabled(False)
        self.eval_permutation  = QCheckBox(
            "Feature permutation importance (intensive)", self,
        )

        # Learning curves + sub-fields
        self.eval_learning     = QCheckBox(
            "Learning curves (intensive)", self,
        )
        self.eval_learning.toggled.connect(self._toggle_learning)
        self.eval_learning_k   = QLineEdit("5", self)
        self.eval_learning_k.setValidator(QIntValidator(2, 100, self))
        self.eval_learning_k.setEnabled(False)
        self.eval_learning_d   = QLineEdit("10", self)
        self.eval_learning_d.setValidator(QIntValidator(2, 100, self))
        self.eval_learning_d.setEnabled(False)

        self.eval_pr_curve     = QCheckBox("Precision-recall curves",
                                           self)
        self.eval_partial_dep  = QCheckBox(
            "Partial dependencies (intensive)", self,
        )

        # SHAP + sub-fields
        self.eval_shap         = QCheckBox("Compute SHAP scores", self)
        self.eval_shap.toggled.connect(self._toggle_shap)
        self.eval_shap_present = QLineEdit("100", self)
        self.eval_shap_present.setValidator(QIntValidator(1, 1_000_000,
                                                         self))
        self.eval_shap_present.setEnabled(False)
        self.eval_shap_absent  = QLineEdit("100", self)
        self.eval_shap_absent.setValidator(QIntValidator(1, 1_000_000,
                                                        self))
        self.eval_shap_absent.setEnabled(False)
        self.eval_shap_cadence = QComboBox(self)
        self.eval_shap_cadence.addItems(_SHAP_CADENCE)
        self.eval_shap_cadence.setCurrentText("ALL FRAMES")
        self.eval_shap_cadence.setEnabled(False)
        self.eval_shap_mp      = QComboBox(self)
        self.eval_shap_mp.addItems(["FALSE", "TRUE"])
        self.eval_shap_mp.setEnabled(False)

        # Lay out — checkboxes top to bottom, with inline sub-field
        # rows where applicable
        for w in (self.eval_meta_data, self.eval_dtree_gviz,
                  self.eval_dtree_dtviz, self.eval_clf_report,
                  self.eval_bar_graph):
            layout.addWidget(w)
        sub_bar = QHBoxLayout()
        sub_bar.addWidget(QLabel("    # of features in bar:", self))
        sub_bar.addWidget(self.eval_bar_graph_n)
        sub_bar.addStretch()
        layout.addLayout(sub_bar)

        layout.addWidget(self.eval_permutation)
        layout.addWidget(self.eval_learning)
        sub_l = QHBoxLayout()
        sub_l.addWidget(QLabel("    K splits:", self))
        sub_l.addWidget(self.eval_learning_k)
        sub_l.addWidget(QLabel("    Data splits:", self))
        sub_l.addWidget(self.eval_learning_d)
        sub_l.addStretch()
        layout.addLayout(sub_l)

        layout.addWidget(self.eval_pr_curve)
        layout.addWidget(self.eval_partial_dep)
        layout.addWidget(self.eval_shap)
        sub_s = QHBoxLayout()
        sub_s.addWidget(QLabel("    SHAP target frames:", self))
        sub_s.addWidget(self.eval_shap_present)
        sub_s.addWidget(QLabel("    Non-target frames:", self))
        sub_s.addWidget(self.eval_shap_absent)
        sub_s.addStretch()
        layout.addLayout(sub_s)
        sub_s2 = QHBoxLayout()
        sub_s2.addWidget(QLabel("    SHAP cadence:", self))
        sub_s2.addWidget(self.eval_shap_cadence)
        sub_s2.addWidget(QLabel("    Multi-process:", self))
        sub_s2.addWidget(self.eval_shap_mp)
        sub_s2.addStretch()
        layout.addLayout(sub_s2)

        layout.addStretch()
        return box

    def _toggle_bar_graph_n(self, checked: bool) -> None:
        self.eval_bar_graph_n.setEnabled(checked)

    def _toggle_learning(self, checked: bool) -> None:
        self.eval_learning_k.setEnabled(checked)
        self.eval_learning_d.setEnabled(checked)

    def _toggle_shap(self, checked: bool) -> None:
        for w in (self.eval_shap_present, self.eval_shap_absent,
                  self.eval_shap_cadence, self.eval_shap_mp):
            w.setEnabled(checked)

    # ----------------------------------------------------------- Pop-out
    def _toggle_pop_out(self) -> None:
        """Re-parent the form between the inline section and a
        floating QDockWidget. Mirrors 122al + 122ap."""
        if self._docked_widget is None:
            main_window = self._find_main_window()
            if main_window is None:
                QMessageBox.information(
                    self, "Pop out",
                    "No main workbench window available; the form "
                    "must stay inline.",
                )
                return
            dock = QDockWidget("Train classifier", main_window)
            dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            dock.setFeatures(
                QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
                | QDockWidget.DockWidgetClosable
            )
            self.setParent(dock)
            dock.setWidget(self)
            main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.setFloating(True)
            dock.show()
            self._docked_widget = dock
            self.pop_out_btn.setText("Re-dock ⇲")
            store = getattr(main_window, "_train_clf_docks", [])
            store.append(dock)
            main_window._train_clf_docks = store
        else:
            dock = self._docked_widget
            self._docked_widget = None
            section_host = getattr(self, "_section_host", None)
            if section_host is not None:
                self.setParent(section_host)
                section_host.layout().addWidget(self)
            dock.setWidget(None)
            dock.close()
            self.pop_out_btn.setText("Pop out ⇱")

    def _find_main_window(self) -> Optional[QWidget]:
        from PySide6.QtWidgets import QMainWindow
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QMainWindow):
                return w
            w = w.parentWidget()
        return None

    # ----------------------------------------------------------- Config IO
    def _read_classifier_targets(self) -> list[str]:
        try:
            from mufasa.project_layout import project_metadata_from_config
            return list(
                project_metadata_from_config(self.config_path)
                .get("classifier_targets", [])
            )
        except Exception:
            return []

    def _load_from_ini(self) -> None:
        """Hydrate field values from whichever config format the
        project uses (v1 ``[classifier_training]`` table or legacy
        ``[create_ensemble_settings]`` section).

        Patch 122as: this now delegates to
        :func:`mufasa.project_layout.read_classifier_training_settings`,
        which transparently handles both formats. Pre-122as this
        method was INI-only and the form showed defaults on TOML
        projects.
        """
        if not self.config_path:
            return
        try:
            from mufasa.project_layout import (
                read_classifier_training_settings,
            )
            cfg = read_classifier_training_settings(self.config_path)
        except Exception:
            return
        if not cfg:
            return

        # Helpers — cfg values can be native types (TOML) or str (INI)
        def s(key: str, default: str = "") -> str:
            v = cfg.get(key, default)
            return str(v).strip() if v is not None else default
        def b(key: str) -> bool:
            v = cfg.get(key)
            if isinstance(v, bool):
                return v
            return str(v or "False").strip().lower() in (
                "true", "yes", "1",
            )

        # Hydrate behavior
        clf = s("classifier")
        idx = self.behavior_combo.findText(clf)
        if idx >= 0:
            self.behavior_combo.setCurrentIndex(idx)

        if s("rf_n_estimators"):
            self.estimators.setText(s("rf_n_estimators"))
        if s("rf_max_features"):
            idx = self.max_features.findText(s("rf_max_features"))
            if idx >= 0:
                self.max_features.setCurrentIndex(idx)
        if s("rf_criterion"):
            idx = self.criterion.findText(s("rf_criterion"))
            if idx >= 0:
                self.criterion.setCurrentIndex(idx)
        if s("train_test_size"):
            idx = self.test_size.findText(s("train_test_size"))
            if idx >= 0:
                self.test_size.setCurrentIndex(idx)
        if s("train_test_split_type"):
            idx = self.tt_split.findText(s("train_test_split_type"))
            if idx >= 0:
                self.tt_split.setCurrentIndex(idx)
        if s("rf_min_sample_leaf"):
            self.min_leaf.setText(s("rf_min_sample_leaf"))
        if s("under_sample_setting"):
            idx = self.undersample_setting.findText(
                s("under_sample_setting"),
            )
            if idx >= 0:
                self.undersample_setting.setCurrentIndex(idx)
        if s("under_sample_ratio"):
            self.undersample_ratio.setText(s("under_sample_ratio"))
        if s("over_sample_setting"):
            idx = self.oversample_setting.findText(
                s("over_sample_setting"),
            )
            if idx >= 0:
                self.oversample_setting.setCurrentIndex(idx)
        if s("over_sample_ratio"):
            self.oversample_ratio.setText(s("over_sample_ratio"))
        if s("class_weights"):
            idx = self.class_weight.findText(s("class_weights"))
            if idx >= 0:
                self.class_weight.setCurrentIndex(idx)

        # Eval checkboxes
        self.eval_meta_data.setChecked(b("rf_metadata"))
        self.eval_dtree_gviz.setChecked(b("generate_example_decision_tree"))
        self.eval_dtree_dtviz.setChecked(
            b("generate_example_decision_tree_fancy"),
        )
        self.eval_clf_report.setChecked(b("generate_classification_report"))
        self.eval_bar_graph.setChecked(
            b("generate_features_importance_bar_graph"),
        )
        if s("n_feature_importance_bars"):
            self.eval_bar_graph_n.setText(s("n_feature_importance_bars"))
        self.eval_permutation.setChecked(b("compute_feature_permutation_importance"))
        self.eval_learning.setChecked(b("generate_learning_curve"))
        if s("learning_curve_k_splits"):
            self.eval_learning_k.setText(s("learning_curve_k_splits"))
        if s("learning_curve_data_splits"):
            self.eval_learning_d.setText(s("learning_curve_data_splits"))
        self.eval_pr_curve.setChecked(b("generate_precision_recall_curve"))
        self.eval_partial_dep.setChecked(b("partial_dependency"))
        self.eval_shap.setChecked(b("calculate_shap_scores"))
        if s("shap_target_present_no"):
            self.eval_shap_present.setText(s("shap_target_present_no"))
        if s("shap_target_absent_no"):
            self.eval_shap_absent.setText(s("shap_target_absent_no"))
        if s("shap_save_iteration"):
            idx = self.eval_shap_cadence.findText(s("shap_save_iteration"))
            if idx >= 0:
                self.eval_shap_cadence.setCurrentIndex(idx)
        if s("shap_multiprocess"):
            idx = self.eval_shap_mp.findText(s("shap_multiprocess"))
            if idx >= 0:
                self.eval_shap_mp.setCurrentIndex(idx)

    def _write_to_ini(self) -> None:
        """Persist settings to whichever config format the project
        uses (v1 ``[classifier_training]`` table or legacy
        ``[create_ensemble_settings]`` section).

        Patch 122as: previously raised RuntimeError on TOML
        projects. Now delegates to
        :func:`mufasa.project_layout.write_classifier_training_settings`
        which handles both formats. v1 TOML projects get proper
        typed values (bool / int / float); legacy INI projects get
        the stringified shape the trainer's reader expects.
        """
        from mufasa.project_layout import (
            write_classifier_training_settings,
        )
        settings: dict = {
            # Algorithm fixed to RF (Tk dropdown was cosmetic)
            "model_to_run":           "RF",
            "classifier":             self.behavior_combo.currentText(),
            "rf_n_estimators":        int(self.estimators.text().strip()),
            "rf_max_features":        self.max_features.currentText(),
            "rf_criterion":           self.criterion.currentText(),
            "train_test_size":        self.test_size.currentText(),
            "train_test_split_type":  self.tt_split.currentText(),
            "rf_min_sample_leaf":     int(self.min_leaf.text().strip()),
            "under_sample_setting":   self.undersample_setting.currentText(),
            "under_sample_ratio":     self.undersample_ratio.text().strip(),
            "over_sample_setting":    self.oversample_setting.currentText(),
            "over_sample_ratio":      self.oversample_ratio.text().strip(),
            "class_weights":          self.class_weight.currentText(),
            # Evaluations — bool checkboxes
            "rf_metadata":            bool(self.eval_meta_data.isChecked()),
            "generate_example_decision_tree":
                bool(self.eval_dtree_gviz.isChecked()),
            "generate_example_decision_tree_fancy":
                bool(self.eval_dtree_dtviz.isChecked()),
            "generate_classification_report":
                bool(self.eval_clf_report.isChecked()),
            "generate_features_importance_bar_graph":
                bool(self.eval_bar_graph.isChecked()),
            "n_feature_importance_bars":
                int(self.eval_bar_graph_n.text().strip()
                    or "15"),
            "compute_feature_permutation_importance":
                bool(self.eval_permutation.isChecked()),
            "generate_learning_curve":
                bool(self.eval_learning.isChecked()),
            "learning_curve_k_splits":
                int(self.eval_learning_k.text().strip() or "5"),
            "learning_curve_data_splits":
                int(self.eval_learning_d.text().strip() or "10"),
            "generate_precision_recall_curve":
                bool(self.eval_pr_curve.isChecked()),
            "partial_dependency":
                bool(self.eval_partial_dep.isChecked()),
            "calculate_shap_scores":
                bool(self.eval_shap.isChecked()),
            "shap_target_present_no":
                int(self.eval_shap_present.text().strip()
                    or "100"),
            "shap_target_absent_no":
                int(self.eval_shap_absent.text().strip()
                    or "100"),
            "shap_save_iteration":
                self.eval_shap_cadence.currentText(),
            "shap_multiprocess":
                self.eval_shap_mp.currentText(),
        }
        write_classifier_training_settings(
            self.config_path, settings,
        )

    def _validate(self) -> None:
        if (self.behavior_combo.currentText()
                == "⟨no classifiers defined⟩"
                or not self._read_classifier_targets()):
            raise ValueError(
                "No classifiers defined. Add at least one classifier "
                "via the 'Manage classifiers' section first."
            )
        # Quick sanity on the int fields (QIntValidator should have
        # caught these but we re-check at submit-time)
        for label, w in (
            ("RF estimators", self.estimators),
            ("min sample leaf", self.min_leaf),
        ):
            text = w.text().strip()
            try:
                int(text)
            except ValueError:
                raise ValueError(f"{label} must be an integer; got {text!r}")
        for label, setting_w, ratio_w in (
            ("under-sample", self.undersample_setting,
             self.undersample_ratio),
            ("over-sample", self.oversample_setting,
             self.oversample_ratio),
        ):
            if setting_w.currentText() != "None":
                try:
                    float(ratio_w.text().strip())
                except ValueError:
                    raise ValueError(
                        f"{label} ratio must be a float when "
                        f"setting is not None; got "
                        f"{ratio_w.text()!r}"
                    )

    # ----------------------------------------------------------- Save-only handler
    def _on_save_only(self) -> None:
        try:
            self._validate()
            self._write_to_ini()
        except Exception as exc:
            QMessageBox.critical(
                self, "Save failed", str(exc),
            )
            return
        QMessageBox.information(
            self, "Saved",
            f"Training settings saved to project_config.ini.",
        )

    # ----------------------------------------------------------- Execute
    def collect_args(self) -> dict:
        self._validate()
        self._write_to_ini()
        return {"config_path": self.config_path}

    def target(self, *, config_path: str) -> None:
        from mufasa.model.train_rf import TrainRandomForestClassifier
        TrainRandomForestClassifier(config_path=config_path).run()


__all__ = ["TrainClassifierForm"]
