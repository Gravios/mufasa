"""
mufasa.ui_qt.forms.features
===========================

Inline form wrapping :class:`FeatureSubsetsCalculator`. Replaces
:class:`SubsetFeatureExtractorPopUp` (1 popup).

The legacy popup lets the user pick which feature families
(movement / ROI / distance / circular / etc.) to recompute and
whether to append the result to the existing features_extracted/
targets_inserted CSVs. Same in the Qt form, with the feature-family
list populated from the backend's constants.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QLabel, QListWidget,
                               QListWidgetItem, QSpinBox, QVBoxLayout,
                               QWidget)

from mufasa.ui_qt.forms.data_import import _PathField
from mufasa.ui_qt.workbench import OperationForm


# Default feature families — mirrors the hard-coded list in the legacy
# popup. If the backend later exposes a registry, swap to that.
_DEFAULT_FAMILIES = [
    "Two-point body-part distances (mm)",
    "Within-animal three-point body-part angles (degrees)",
    "Within-animal three-point convex hull perimeters (mm)",
    "Within-animal four-point convex hull perimeters (mm)",
    "Entire animal convex hull perimeters (mm)",
    "Entire animal convex hull area (mm2)",
    "Frame-by-frame body-part movements (mm)",
    "Frame-by-frame distance to ROI centers (mm)",
    "Frame-by-frame body-parts inside ROIs (Boolean)",
]


class FeatureSubsetExtractorForm(OperationForm):
    """Compute a subset of feature families and merge into existing
    feature CSVs."""

    title = "Compute feature subsets"
    description = ("Recompute one or more feature families and "
                   "optionally append them to the existing "
                   "feature/targets CSVs. Useful when new feature "
                   "families are introduced without re-running the full "
                   "extractor.")

    def build(self) -> None:
        outer = QVBoxLayout()

        # Feature-family multi-select
        outer.addWidget(self._bold("Feature families"))
        self.families = QListWidget(self)
        self.families.setSelectionMode(QListWidget.MultiSelection)
        self.families.setMinimumHeight(180)
        for fam in self._discover_families():
            item = QListWidgetItem(fam)
            item.setData(Qt.UserRole, fam)
            self.families.addItem(item)
        outer.addWidget(self.families)

        # Options
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.save_dir = _PathField(
            is_file=False,
            placeholder=("Save directory (required unless an "
                         "append checkbox below is checked)…"),
        )
        form.addRow("Save to:", self.save_dir)

        self.file_checks = QCheckBox("Run file checks before computing", self)
        self.file_checks.setChecked(True)
        form.addRow("", self.file_checks)

        self.append_features = QCheckBox(
            "Append to existing features_extracted CSVs", self,
        )
        form.addRow("", self.append_features)

        self.append_targets = QCheckBox(
            "Append to existing targets_inserted CSVs", self,
        )
        form.addRow("", self.append_targets)

        # Parallel workers — sits as the last row in the form so it's
        # visually adjacent to the Run button row (which the parent
        # OperationForm appends right after self.body_layout).
        # Default 1 = sequential (byte-equivalent to pre-step-6
        # behavior). Cap is the number of physical cores reported by
        # the OS, falling back to 8 if os.cpu_count() returns None.
        # Going above physical core count rarely helps because the
        # numba kernels are compute-bound and SMT threads contend.
        import os
        max_workers_cap = os.cpu_count() or 8
        self.n_workers = QSpinBox(self)
        self.n_workers.setMinimum(1)
        self.n_workers.setMaximum(max_workers_cap)
        self.n_workers.setValue(1)
        self.n_workers.setSuffix("  worker(s)")
        self.n_workers.setToolTip(
            "Number of parallel processes. 1 = sequential (default, "
            "byte-equivalent to single-threaded). Higher values run "
            "videos in parallel; 5-7× speedup typical on 8-core "
            "systems. IMPORTANT: verify parallel output matches "
            "sequential output before relying on n_workers > 1 for "
            "production analysis (see "
            "tests/smoke_feature_parallel_verify.py)."
        )
        form.addRow("Parallel workers:", self.n_workers)
        # Hint label visible in the form (not just in the tooltip).
        # Verification matters because a botched parallel implementation
        # produces silently-wrong feature values that look plausible
        # but corrupt downstream analysis. The hint is muted styling
        # so it doesn't fight for attention but is reachable when
        # somebody bumps the spinbox above 1.
        hint = QLabel(
            "<i>If n_workers &gt; 1: verify against sequential output "
            "first via <code>tests/smoke_feature_parallel_verify.py</code>.</i>"
        )
        hint.setStyleSheet("color: #666; font-size: 9pt;")
        hint.setWordWrap(True)
        form.addRow("", hint)

        outer.addLayout(form)
        self.body_layout.addLayout(outer)

    def _bold(self, text: str) -> "QWidget":
        from PySide6.QtWidgets import QLabel
        lbl = QLabel(f"<b>{text}</b>")
        return lbl

    def _discover_families(self) -> list[str]:
        """Try to read the family list from the backend; fall back to
        the static default."""
        try:
            from mufasa.feature_extractors.feature_subsets import (
                FEATURE_FAMILIES,
            )
            if FEATURE_FAMILIES:
                return list(FEATURE_FAMILIES)
        except Exception:
            pass
        return list(_DEFAULT_FAMILIES)

    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError("No project loaded.")
        selected = [it.data(Qt.UserRole)
                    for it in self.families.selectedItems()]
        if not selected:
            raise ValueError("Select at least one feature family.")
        save_dir_value = self.save_dir.path or None
        append_features = bool(self.append_features.isChecked())
        append_targets = bool(self.append_targets.isChecked())
        # Match the backend strict-mode validation: refuse to start
        # without an explicit destination. Before this check landed
        # in the backend, the form's old placeholder text claimed
        # "blank = project log dir" — incorrect; blank meant the
        # output was silently discarded after compute. Surfacing the
        # error here (rather than in the backend) gives the user a
        # nicer dialog than a stack trace.
        if save_dir_value is None and not append_features and not append_targets:
            raise ValueError(
                "Specify a save destination: either set 'Save to:' "
                "to a directory, or check 'Append to existing "
                "features_extracted CSVs' / 'Append to existing "
                "targets_inserted CSVs'. Without one of these, "
                "the computed features would be discarded after "
                "the run completes."
            )
        return {
            "config_path":      self.config_path,
            "feature_families": selected,
            "file_checks":      bool(self.file_checks.isChecked()),
            "save_dir":         save_dir_value,
            "append_features":  append_features,
            "append_targets":   append_targets,
            "n_workers":        int(self.n_workers.value()),
        }

    def target(self, *, config_path: str, feature_families: list[str],
               file_checks: bool, save_dir: Optional[str],
               append_features: bool, append_targets: bool,
               n_workers: int) -> None:
        from mufasa.feature_extractors.feature_subsets import (
            FeatureSubsetsCalculator,
        )
        FeatureSubsetsCalculator(
            config_path=config_path,
            feature_families=feature_families,
            file_checks=file_checks,
            save_dir=save_dir,
            data_dir=None,
            append_to_features_extracted=append_features,
            append_to_targets_inserted=append_targets,
            n_workers=n_workers,
        ).run()


__all__ = ["FeatureSubsetExtractorForm"]
