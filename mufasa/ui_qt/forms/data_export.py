"""
mufasa.ui_qt.forms.data_export
==============================

Project-scoped Tools-page form for exporting v1-layout data
(per-family or wide parquets in ``derived/features/``,
per-video labels in ``derived/labels/``) back to legacy-shape
wide CSVs. Useful when v1 data needs to flow into external
tools that expect the SimBA ``csv/features_extracted/`` or
``csv/targets_inserted/`` layout.

Patch 122ae-6. The form is thin: it picks one of three
backend functions in :mod:`mufasa.utils.csv_export` based on
the user's "What to export" choice and calls it once per
selected video. Per-video errors are collected and surfaced
at the end so a single missing video doesn't abort a batch.

The form lives on the Tools page (via
``mufasa.ui_qt.pages.tools_page.build_tools_page``) so users
can reach it independently of the Annotate / Features / etc.
flows. Requires an open project (uses ``self.config_path``);
the form raises clearly if invoked without one rather than
silently working on the wrong directory.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QHBoxLayout,
                               QLineEdit, QPushButton)

from mufasa.ui_qt.workbench import OperationForm


# Export "what" choices — string constants keep the dropdown labels
# and backend dispatch in sync without an enum import dance.
WHAT_FEATURES  = "Features only"
WHAT_LABELS    = "Labels only"
WHAT_COMBINED  = "Features + labels combined"

# Video-pick mode
VIDS_ALL    = "All videos in project"
VIDS_SINGLE = "Single video…"


class ExportToCSVForm(OperationForm):
    """Export v1 features / labels / combined to CSV files.

    Single-video or batch — runs one backend call per video.
    Per-video failures (missing features, row-count mismatch in
    combined mode, etc.) accumulate; the run-result dialog
    summarises successes and failures at the end rather than
    aborting at the first error.
    """

    title = "Export to CSV"
    description = (
        "Export this project's features, labels, or both to "
        "wide CSV files. Reads from the v1 layout "
        "(<code>derived/features/</code>, <code>derived/labels/</code>) "
        "when present and falls back to the legacy "
        "(<code>csv/features_extracted/</code>, "
        "<code>csv/targets_inserted/</code>) layout transparently. "
        "Useful for handing v1 data off to external tools that "
        "expect the SimBA-style wide CSV shape."
    )

    def build(self) -> None:
        form = QFormLayout()
        self.body_layout.addLayout(form)

        # ----- What to export -----
        self.what_combo = QComboBox(self)
        self.what_combo.addItems([
            WHAT_FEATURES, WHAT_LABELS, WHAT_COMBINED,
        ])
        form.addRow("What to export:", self.what_combo)

        # ----- Which videos -----
        self.vids_combo = QComboBox(self)
        self.vids_combo.addItems([VIDS_ALL, VIDS_SINGLE])
        form.addRow("Scope:", self.vids_combo)

        # Single-video picker — visible only when VIDS_SINGLE is
        # selected. Populated from project_paths_from_config's
        # video_dir on form mount.
        self.single_video_combo = QComboBox(self)
        self.single_video_combo.setVisible(False)
        form.addRow("Video:", self.single_video_combo)
        # Tag the label so we can toggle visibility together.
        self._single_video_label = form.labelForField(
            self.single_video_combo,
        )
        if self._single_video_label is not None:
            self._single_video_label.setVisible(False)

        self.vids_combo.currentTextChanged.connect(
            self._on_scope_changed,
        )

        # ----- Destination -----
        dest_row = QHBoxLayout()
        self.dest_edit = QLineEdit(self)
        self.dest_edit.setPlaceholderText(
            "…directory for the exported CSVs",
        )
        dest_row.addWidget(self.dest_edit, stretch=1)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._pick_dest)
        dest_row.addWidget(browse)
        form.addRow("Destination:", dest_row)

        # ----- Options -----
        self.include_index = QCheckBox(
            "Include leading index column (recommended for "
            "compatibility with SimBA's read_df)", self,
        )
        self.include_index.setChecked(True)
        form.addRow("", self.include_index)

        # Populate single-video combo from the project. Done at
        # the end of build so all widgets exist if we have to
        # show an error.
        self._populate_videos()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _populate_videos(self) -> None:
        """Fill the single-video combo from the project's videos
        directory. Empty list is acceptable — the user might be
        exporting in a project that holds only pose CSVs without
        videos."""
        self.single_video_combo.clear()
        if not self.config_path:
            return
        try:
            from mufasa.project_layout import \
                project_paths_from_config
            paths = project_paths_from_config(self.config_path)
            vid_dir = paths.get("video_dir")
        except Exception:
            return
        if not vid_dir or not os.path.isdir(vid_dir):
            return
        for name in sorted(os.listdir(vid_dir)):
            if name.startswith("."):
                continue
            stem, _ = os.path.splitext(name)
            self.single_video_combo.addItem(stem)

    def _on_scope_changed(self, text: str) -> None:
        is_single = text == VIDS_SINGLE
        self.single_video_combo.setVisible(is_single)
        if self._single_video_label is not None:
            self._single_video_label.setVisible(is_single)

    def _pick_dest(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Pick destination directory", "",
        )
        if path:
            self.dest_edit.setText(path)

    def _list_all_videos(self) -> list[str]:
        """List every video stem in the project. Empty list means
        no videos found — caller should warn."""
        if not self.config_path:
            return []
        try:
            from mufasa.project_layout import \
                project_paths_from_config
            paths = project_paths_from_config(self.config_path)
            vid_dir = paths.get("video_dir")
        except Exception:
            return []
        if not vid_dir or not os.path.isdir(vid_dir):
            return []
        out: list[str] = []
        for name in sorted(os.listdir(vid_dir)):
            if name.startswith("."):
                continue
            stem, _ = os.path.splitext(name)
            out.append(stem)
        return out

    # ------------------------------------------------------------------ #
    # OperationForm contract
    # ------------------------------------------------------------------ #
    def collect_args(self) -> dict:
        if not self.config_path:
            raise RuntimeError(
                "No project loaded. Open a project first."
            )
        dest = (self.dest_edit.text() or "").strip()
        if not dest:
            raise ValueError(
                "Destination directory is required."
            )
        what = self.what_combo.currentText()
        scope = self.vids_combo.currentText()
        if scope == VIDS_SINGLE:
            video = self.single_video_combo.currentText()
            if not video:
                raise ValueError(
                    "Pick a video for single-video export."
                )
            videos = [video]
        else:
            videos = self._list_all_videos()
            if not videos:
                raise ValueError(
                    "No videos found in the project."
                )
        return {
            "config_path":    self.config_path,
            "what":           what,
            "videos":         videos,
            "dest_dir":       dest,
            "include_index":  bool(self.include_index.isChecked()),
        }

    def target(self, *, config_path: str, what: str,
               videos: list[str], dest_dir: str,
               include_index: bool) -> None:
        """Run the export. Dispatches one backend call per
        video; per-video failures don't abort the batch but
        accumulate for the final summary."""
        from mufasa.utils.csv_export import (export_combined_csv,
                                             export_features_csv,
                                             export_labels_csv)
        # Patch 122ae-6: dispatch table keyed by user-facing
        # "what" string. Matches the dropdown values constructed
        # in build(); no enum coercion needed.
        dispatch = {
            WHAT_FEATURES: export_features_csv,
            WHAT_LABELS:   export_labels_csv,
            WHAT_COMBINED: export_combined_csv,
        }
        fn = dispatch.get(what)
        if fn is None:
            raise RuntimeError(
                f"Internal error: unknown export type {what!r}"
            )

        successes: list[str] = []
        failures: list[tuple[str, str]] = []
        for video in videos:
            try:
                out_path = fn(
                    video_name=video,
                    config_path=config_path,
                    dest_dir=dest_dir,
                    include_index=include_index,
                )
                successes.append(out_path)
            except Exception as exc:
                failures.append(
                    (video, f"{type(exc).__name__}: {exc}")
                )

        # Result summary. The base OperationForm shows the
        # returned dict in a dialog; we shape a readable summary.
        summary = (
            f"Exported {len(successes)}/{len(videos)} videos to "
            f"{dest_dir}."
        )
        if failures:
            details = "\n".join(
                f"  • {v}: {err}" for v, err in failures
            )
            summary += f"\n\nFailures:\n{details}"
        print(f"[ExportToCSVForm] {summary}")


__all__ = ["ExportToCSVForm"]
