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

Patch 122z notes
----------------
* Feature families are now split into two grouped frames —
  **Subject features** (within-animal: distances, angles, hulls,
  movements) and **ROI features** (distance-to-ROI, body-parts-
  inside-ROIs). Same backend; the form just renders the
  selection in two QGroupBoxes so users see the two categories
  separately. ROI features won't compute without ROI definitions
  in the project; the ROI group's frame note says so.
* Destination radio labels previously referenced both the v1
  layout and the legacy layout with a deferred-runid caveat.

Patch 122ae-3 supersedes the 122z destination notes
----------------------------------------------------
* New default destination: **write per-family parquet** to
  ``derived/features/<family_slug>/<video>.parquet`` (the v1-
  native shape, in line with the parquet-only foundation laid
  in 122ae-1).
* The two legacy append modes
  (``csv/features_extracted/`` / ``csv/targets_inserted/``)
  remain available for the transition window but no longer
  carry the "deferred run-id allocation" caveat — the v1
  layout question is answered by the per-family layout, so
  there's no run-id question to defer.
* The standalone-save-dir mode is unchanged: write wide files
  to a user-picked directory outside the project tree.
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


def _is_roi_family(name: str) -> bool:
    """Classify a feature-family name as ROI-related.

    Heuristic: the family name contains the case-insensitive
    substring ``ROI``. Works for every entry in
    ``_DEFAULT_FAMILIES`` and for the canonical SimBA family
    names exposed by the backend's ``FEATURE_FAMILIES`` constant.
    If the backend adds families that don't follow this naming
    convention (none currently do), they'll land in the Subject
    group by default — which is the safer fallback since Subject
    features don't require ROI definitions to compute.
    """
    return "ROI" in name


class FeatureSubsetExtractorForm(OperationForm):
    """Compute a subset of feature families and merge into existing
    feature CSVs."""

    title = "Compute feature subsets"
    description = (
        "Recompute one or more feature families and optionally "
        "append them to the existing feature / targets CSVs. "
        "Useful when new feature families are introduced without "
        "re-running the full extractor. "
        "<br><br>"
        "<b>Subject features</b> (within-animal distances, angles, "
        "hulls, movements) and <b>ROI features</b> (distance to "
        "ROI centers, body-parts inside ROIs) live in separate "
        "frames below — pick zero, one, or many from either or "
        "both. ROI features need ROI definitions in the project; "
        "draw or import them on the ROI page first."
    )

    def build(self) -> None:
        from PySide6.QtWidgets import QGroupBox
        outer = QVBoxLayout()

        # ----- Feature families: two grouped frames ----- #
        # Patch 122z: previously one big multi-select list. Split
        # into Subject vs ROI families so users see the two
        # categories distinctly. Two QListWidgets feeding one
        # combined selection in collect_args().
        families = self._discover_families()
        subject_families = [f for f in families
                            if not _is_roi_family(f)]
        roi_families = [f for f in families if _is_roi_family(f)]

        # Subject frame
        subj_box = QGroupBox("Subject features", self)
        subj_layout = QVBoxLayout(subj_box)
        subj_layout.setContentsMargins(8, 8, 8, 8)
        self.subject_families = QListWidget(self)
        self.subject_families.setSelectionMode(
            QListWidget.MultiSelection,
        )
        self.subject_families.setMinimumHeight(140)
        for fam in subject_families:
            item = QListWidgetItem(fam)
            item.setData(Qt.UserRole, fam)
            self.subject_families.addItem(item)
        subj_layout.addWidget(self.subject_families)
        subj_hint = QLabel(
            "<i>Within-animal geometry: distances, angles, "
            "convex hulls, frame-to-frame body-part movements. "
            "Need only pose data.</i>",
            self,
        )
        subj_hint.setStyleSheet("color: palette(placeholder-text);")
        subj_hint.setWordWrap(True)
        subj_layout.addWidget(subj_hint)
        outer.addWidget(subj_box)

        # ROI frame
        roi_box = QGroupBox("ROI features", self)
        roi_layout = QVBoxLayout(roi_box)
        roi_layout.setContentsMargins(8, 8, 8, 8)
        self.roi_families = QListWidget(self)
        self.roi_families.setSelectionMode(
            QListWidget.MultiSelection,
        )
        self.roi_families.setMinimumHeight(90)
        for fam in roi_families:
            item = QListWidgetItem(fam)
            item.setData(Qt.UserRole, fam)
            self.roi_families.addItem(item)
        roi_layout.addWidget(self.roi_families)
        roi_hint = QLabel(
            "<i>Require ROI definitions in the current project. "
            "Draw or import ROIs on the ROI page first; results "
            "are written into the same CSV alongside the subject "
            "features.</i>",
            self,
        )
        roi_hint.setStyleSheet("color: palette(placeholder-text);")
        roi_hint.setWordWrap(True)
        roi_layout.addWidget(roi_hint)
        outer.addWidget(roi_box)

        # Back-compat alias: some external code (and the prior
        # smoke test) reads self.families. Expose a combined
        # convenience reference. Selection is read directly off
        # the two child lists in collect_args, so this is purely
        # for "did the form build a families list" structural
        # checks.
        self.families = self.subject_families  # historical handle

        # Options
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # Destination radio group. Pre-fix: three independent
        # widgets (save_dir field + two checkboxes) made it
        # possible to accidentally configure dual destinations
        # — output went to BOTH save_dir AND the project's
        # features_extracted/. Most users didn't intend that.
        # Radio buttons enforce a single choice and make each
        # mode's behavior explicit in the label.
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        dest_label = QLabel("<b>Destination:</b>", self)
        form.addRow("", dest_label)

        # Mode 0 (new in 122ae-3, default): per-family parquet
        # under the v1 derived/features/ tree. Each feature family
        # gets its own subdirectory; one parquet file per video per
        # family. Reads happen transparently via the shared
        # load_features_for_video() helper (122ae-2). This is the
        # native v1 shape and the path forward for the parquet-
        # only direction.
        self.dest_derived_parquet = QRadioButton(
            "Write per-family parquet to "
            "<code>derived/features/&lt;family&gt;/&lt;video&gt;.parquet</code>"
            " <i>(recommended, v1-native)</i>", self,
        )
        form.addRow("", self.dest_derived_parquet)

        # Mode 1 (legacy): save_dir — write standalone wide files
        # containing ONLY the new feature columns. Doesn't modify
        # project files. Useful as an escape hatch for export to a
        # custom directory outside the project tree.
        self.dest_save_dir = QRadioButton(
            "Save standalone files (new columns only) to:", self,
        )
        form.addRow("", self.dest_save_dir)
        self.save_dir = _PathField(
            is_file=False,
            placeholder="…directory for the standalone CSVs/parquets",
        )
        form.addRow("", self.save_dir)

        # Patch 122an (B1): the two legacy radio modes
        # ("Append new columns to legacy per-video feature files"
        # and "...to legacy per-video targets files") are gone.
        # The kwargs they passed (append_to_features_extracted /
        # append_to_targets_inserted) were inert after 122ae-3 —
        # they only ran pre-flight checks and never produced a
        # real append write. v1 layout has two real destinations:
        # derived/features/ (per-family parquet, default) or a
        # user-picked save_dir.

        # Group enforces single-choice. Default to per-family
        # parquet (v1-native, the recommended path).
        self._dest_group = QButtonGroup(self)
        self._dest_group.addButton(self.dest_derived_parquet, 0)
        self._dest_group.addButton(self.dest_save_dir, 1)
        self.dest_derived_parquet.setChecked(True)

        # Disable the path field unless save_dir mode is selected.
        def _update_save_dir_enabled() -> None:
            self.save_dir.setEnabled(self.dest_save_dir.isChecked())
        self.dest_derived_parquet.toggled.connect(lambda _: _update_save_dir_enabled())
        self.dest_save_dir.toggled.connect(lambda _: _update_save_dir_enabled())
        _update_save_dir_enabled()

        # Spacer after destination block before checks/workers
        form.addRow("", QLabel("", self))

        self.file_checks = QCheckBox("Run file checks before computing", self)
        self.file_checks.setChecked(True)
        form.addRow("", self.file_checks)

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
        hint.setStyleSheet("color: palette(placeholder-text); font-size: 9pt;")
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
        # Patch 122z: read from BOTH subject and ROI lists. Order:
        # subject first, then ROI — matches the visual top-to-
        # bottom layout. Duplicates impossible because the two
        # lists are disjoint by _is_roi_family classification.
        selected = (
            [it.data(Qt.UserRole)
             for it in self.subject_families.selectedItems()]
            + [it.data(Qt.UserRole)
               for it in self.roi_families.selectedItems()]
        )
        if not selected:
            raise ValueError("Select at least one feature family.")

        # Derive backend kwargs from radio button state. Single-
        # choice radio means at most one of the four is True at
        # any time.
        # Patch 122ae-3 added the derived_parquet mode as the new
        # default. The legacy 3 modes are unchanged.
        derived_features_dir = None
        if self.dest_derived_parquet.isChecked():
            # Look up the v1 path via project_paths_from_config —
            # works for both v1 (project.toml) and legacy
            # (project_config.ini) projects. For legacy projects
            # this resolves to '<project>/derived/features/' which
            # didn't exist before but is created on-demand by
            # process_one_video.
            from mufasa.project_layout import project_paths_from_config
            try:
                paths = project_paths_from_config(self.config_path)
            except Exception as exc:
                raise ValueError(
                    f"Couldn't resolve project paths from "
                    f"{self.config_path!r}: {exc}",
                )
            derived_features_dir = paths["derived_features_dir"]
            save_dir_value = None
        elif self.dest_save_dir.isChecked():
            save_dir_value = self.save_dir.path or None
            if save_dir_value is None:
                raise ValueError(
                    "Save destination is required: pick a directory "
                    "in the 'Save standalone files' field, or switch "
                    "the destination to derived/features/."
                )
        else:
            # Shouldn't happen — QButtonGroup forces a selection
            # since one is checked by default. But defensive.
            raise ValueError(
                "Internal error: no destination radio is selected. "
                "Please select a destination."
            )

        return {
            "config_path":      self.config_path,
            "feature_families": selected,
            "file_checks":      bool(self.file_checks.isChecked()),
            "save_dir":         save_dir_value,
            "derived_features_dir": derived_features_dir,
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
            f"derived_features_dir={kwargs['derived_features_dir']!r}, "
            f"families={len(kwargs['feature_families'])}"
        )

        # Patch 122an (B1): preflight runs whenever a destination
        # is set. The append_features / append_targets gates from
        # the legacy modes are gone — they only fired for dead
        # destinations anyway.
        needs_preflight = (
            kwargs["save_dir"] is not None
            or kwargs["derived_features_dir"] is not None
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
            derived_features_dir=kwargs["derived_features_dir"],
            n_workers=kwargs["n_workers"],
            overwrite_existing=False,  # preflight runs in non-overwrite mode
        )
        return calc.preflight_check()

    def target(self, *, config_path: str, feature_families: list[str],
               file_checks: bool, save_dir: Optional[str],
               derived_features_dir: Optional[str],
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
            derived_features_dir=derived_features_dir,
            n_workers=n_workers,
            overwrite_existing=overwrite_existing,
        ).run()


__all__ = ["FeatureSubsetExtractorForm"]
