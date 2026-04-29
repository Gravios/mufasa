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
            "overwrite_existing": False,  # default; on_run may flip
        }

    def on_run(self) -> None:
        """Override OperationForm.on_run to run a preflight check
        for column collisions before kicking off the multi-hour
        compute job. If conflicts found, prompt the user and
        either abort or proceed with overwrite_existing=True.

        The preflight runs synchronously and freezes the UI for a
        few seconds (it processes the first video to discover
        what columns this run would produce). Acceptable trade
        — preflight cost is sub-1% of total run time, and the
        user just clicked Run so a few seconds of "checking..."
        is expected.
        """
        from PySide6.QtWidgets import QMessageBox
        from mufasa.ui_qt.progress import run_with_progress

        try:
            kwargs = self.collect_args()
        except Exception as exc:
            QMessageBox.warning(
                self, f"{self.title}: invalid input", str(exc),
            )
            return

        # Diagnostic: confirms the form's on_run override is being
        # called at all. If the user reports no preflight prompt
        # AND doesn't see this line in the console, the override
        # isn't taking effect (most likely cause: stale __pycache__
        # — `find . -name __pycache__ -exec rm -rf {} +` then
        # `pip install -e .` to fix).
        print(
            f"[on_run] FeatureSubsetExtractorForm dispatching: "
            f"save_dir={kwargs['save_dir']!r}, "
            f"append_features={kwargs['append_features']}, "
            f"append_targets={kwargs['append_targets']}, "
            f"families={len(kwargs['feature_families'])}"
        )

        # Run preflight whenever any destination is set. The
        # preflight checks both save_dir filename collisions AND
        # append-mode column collisions — both are real ways to
        # silently destroy previous run output. Pre-fix: this only
        # ran for append flags, missing the save_dir-overwrite case
        # where shutil.copy silently overwrote files from a prior run.
        needs_preflight = (
            kwargs["save_dir"] is not None
            or kwargs["append_features"]
            or kwargs["append_targets"]
        )

        if needs_preflight:
            # Run preflight synchronously. This blocks the UI for a
            # few seconds; show a busy cursor so it's obvious
            # something is happening.
            from PySide6.QtCore import Qt as _Qt
            from PySide6.QtGui import QGuiApplication, QCursor
            QGuiApplication.setOverrideCursor(QCursor(_Qt.WaitCursor))
            try:
                conflicts = self._run_preflight(kwargs)
            except Exception as exc:
                QGuiApplication.restoreOverrideCursor()
                QMessageBox.critical(
                    self, f"{self.title}: preflight failed",
                    f"Could not check for column conflicts before "
                    f"starting the run.\n\n"
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"You can still proceed (the run itself will "
                    f"detect conflicts at the end and either fail "
                    f"or skip them depending on overwrite_existing), "
                    f"but the early-warning safety net is unavailable.",
                )
                return
            finally:
                QGuiApplication.restoreOverrideCursor()

            if conflicts:
                # Categorize conflicts: file-exists in save_dir vs
                # column collisions in append destinations. Different
                # wording for each.
                save_dir_collisions = [
                    f for f, r in conflicts.items()
                    if r == ['file exists']
                ]
                column_collisions = {
                    f: r for f, r in conflicts.items()
                    if r != ['file exists']
                }
                msg_parts = []
                if save_dir_collisions:
                    sample_files = save_dir_collisions[:5]
                    more = ""
                    if len(save_dir_collisions) > 5:
                        more = (
                            f"<br>  ... ({len(save_dir_collisions) - 5} "
                            f"more files)"
                        )
                    msg_parts.append(
                        f"<b>{len(save_dir_collisions)} file(s) "
                        f"already exist in <code>save_dir</code></b> "
                        f"and would be overwritten:"
                        f"<pre>  " + "\n  ".join(sample_files) + more
                        + "</pre>"
                    )
                if column_collisions:
                    n_files = len(column_collisions)
                    sample_file = next(iter(column_collisions))
                    sample_cols = column_collisions[sample_file]
                    cols_preview = "\n  ".join(sample_cols[:8])
                    col_more = ""
                    if len(sample_cols) > 8:
                        col_more = (
                            f"\n  ... ({len(sample_cols) - 8} more "
                            f"columns)"
                        )
                    msg_parts.append(
                        f"<b>{n_files} file(s) in the append "
                        f"destination already contain feature columns "
                        f"this run would produce.</b><br>"
                        f"Example: <code>{sample_file}</code> has "
                        f"<b>{len(sample_cols)}</b> conflicting "
                        f"column(s):"
                        f"<pre>  {cols_preview}{col_more}</pre>"
                    )
                msg = (
                    "<br><br>".join(msg_parts)
                    + "<br>Overwrite the existing output?"
                )
                response = QMessageBox.question(
                    self, "Confirm overwrite",
                    msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,  # safer default
                )
                if response != QMessageBox.Yes:
                    # User said no — abort cleanly without raising
                    return
                kwargs["overwrite_existing"] = True

        # Dispatch as the parent OperationForm would
        def _work() -> None:
            self.target(**kwargs)

        run_with_progress(
            parent=self.window(),
            title=f"{self.title}…",
            target=_work,
            on_success=lambda: (
                self.completed.emit(),
                QMessageBox.information(self, self.title, "Done."),
            ),
        )

    def _run_preflight(self, kwargs: dict) -> dict:
        """Build a FeatureSubsetsCalculator and call its preflight
        check. Lives in the form so the UI can prompt before
        target() (which dispatches into a worker) is invoked."""
        from mufasa.feature_extractors.feature_subsets import (
            FeatureSubsetsCalculator,
        )
        calc = FeatureSubsetsCalculator(
            config_path=kwargs["config_path"],
            feature_families=kwargs["feature_families"],
            file_checks=kwargs["file_checks"],
            save_dir=kwargs["save_dir"],
            data_dir=None,
            append_to_features_extracted=kwargs["append_features"],
            append_to_targets_inserted=kwargs["append_targets"],
            n_workers=kwargs["n_workers"],
            overwrite_existing=False,  # preflight runs in non-overwrite mode
        )
        return calc.preflight_check()

    def target(self, *, config_path: str, feature_families: list[str],
               file_checks: bool, save_dir: Optional[str],
               append_features: bool, append_targets: bool,
               n_workers: int, overwrite_existing: bool = False) -> None:
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
            overwrite_existing=overwrite_existing,
        ).run()


__all__ = ["FeatureSubsetExtractorForm"]
