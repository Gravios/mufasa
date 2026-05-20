"""
mufasa.ui_qt.dialogs.duplicate_rois_source_target
===================================================

Qt-native dialog for duplicating ROI definitions from one
"source" video to one or more "target" videos in a project.

Replaces the legacy Tk popup
``mufasa.ui.pop_ups.duplicate_rois_by_source_target_popup.DuplicateROIsBySourceTarget``
which was previously bridged into the Qt workbench via the
subprocess-launch pattern in
``roi_video_table.py:_action_duplicate``
(see ``docs/tk_surface_audit.md`` §2g for the broader context).

Replacement landed in patch 122cv. After that patch:
* The Tk popup file is deleted.
* ``roi_video_table.py:_action_duplicate`` calls this dialog
  directly.
* The ``_launch_tk_popup`` helper in ``roi_video_table.py``
  becomes dead code (last subprocess popup ported).

Functional differences from the Tk original
-------------------------------------------

**Selection model: QListWidget(ExtendedSelection) instead of
checkboxes.** The Tk version used a per-video checkbox grid and
hand-rolled ctrl/shift+click range-selection by tracking
``ctrl_pressed`` state and ``check_index`` / ``prior_check_index``
across click events (lines 87–117 of the Tk popup). The Qt port
uses ``QAbstractItemView.ExtendedSelection`` which gives the same
behavior natively:

* Single click → select one.
* Ctrl+click → toggle individual selection.
* Shift+click → extend selection to a range.

The selected items ARE the destination videos — no separate
"check-state" tracked. This collapses ~30 lines of bespoke
shift-click bookkeeping into one ``setSelectionMode`` call.

**Per-video icon indicator.** Each target row shows a green
check-icon if the video already has ROIs (overwrite warning),
or a generic file-icon if not. Same intent as the Tk version's
``green_check`` / ``black_cross`` imagery.

**Filter persists selection.** The filter text-box hides
non-matching items via ``setHidden(True)`` rather than rebuilding
the list. Hidden items retain their selection state, so changing
the filter doesn't drop the user's prior picks. The Tk version
rebuilt the destination frame on filter change but copied
selection state forward via the ``selected_destination_videos``
dict; the Qt port gets the same behavior with less code.

**Deselect-all** uses ``QMessageBox.question`` for confirmation,
mirroring the Tk ``TwoOptionQuestionPopUp``.

**Status feedback** as a status-bar ``QLabel`` at the bottom
(same as Tk) plus ``QMessageBox.information`` on success
(Tk only updated the status bar).
"""
from __future__ import annotations

import os
import re
from copy import deepcopy

import pandas as pd
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStyle,
    QVBoxLayout,
)

from mufasa.mixins.config_reader import ConfigReader
from mufasa.roi_tools.roi_utils import (
    change_roi_dict_video_name,
    get_roi_data_for_video_name,
    get_roi_df_from_dict,
)
from mufasa.utils.enums import Keys
from mufasa.utils.errors import NoFilesFoundError
from mufasa.utils.read_write import find_all_videos_in_directory


def _natural_sort(s: str) -> list:
    """Same natural sort key the Tk version used (line 124).
    Splits on digit runs so "Video_10" sorts after "Video_2".
    """
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


class DuplicateRoisDialog(QDialog):
    """Qt port of `DuplicateROIsBySourceTarget` (122cv)."""

    def __init__(self,
                 config_path: str | os.PathLike,
                 roi_data_path: str | os.PathLike | None = None,
                 parent: QDialog | None = None,
                 default_source: str | None = None,
                 window_title: str | None = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        # Patch 122dl: `window_title` lets callers override the
        # dialog's titlebar text. The default ("Duplicate ROIs from
        # source to target videos") is accurate but reads as
        # technical-jargon to users who came in via the "Apply to
        # selected videos…" entry point on `roi_define_panel`.
        # That entry passes a friendlier title that matches the
        # button label they clicked.
        self.setWindowTitle(
            window_title
            if window_title is not None
            else "Duplicate ROIs from source to target videos"
        )
        self.setModal(True)
        self.resize(640, 720)
        # Patch 122dl: `default_source` lets callers pre-select a
        # specific video as the source — used by the new "Apply to
        # selected videos…" button on `roi_define_panel` so the
        # user doesn't have to re-pick the video they were just
        # working on. None preserves the legacy default-to-first-
        # video behaviour for the menu entry point.
        self._default_source = (
            str(default_source) if default_source else None
        )

        # Load project state + ROI inventory.
        try:
            self._reader = ConfigReader(
                config_path=config_path,
                read_video_info=False,
                create_logger=False)
            if roi_data_path is not None:
                from mufasa.utils.checks import check_file_exist_and_readable
                check_file_exist_and_readable(file_path=roi_data_path)
                self._reader.roi_coordinates_path = deepcopy(roi_data_path)

            # Read ROI data (populates video_names_w_rois and the
            # three shape DataFrames on the reader).
            if not os.path.isfile(self._reader.roi_coordinates_path):
                raise NoFilesFoundError(
                    msg="Cannot duplicate ROIs: no ROI definitions "
                    "file found in this project.",
                    source=self.__class__.__name__)
            self._reader.read_roi_data()
            videos_w_rois = sorted(
                list(self._reader.video_names_w_rois),
                key=_natural_sort)
            if not videos_w_rois:
                raise NoFilesFoundError(
                    msg="Cannot duplicate ROIs: no video has ROIs "
                    "defined.",
                    source=self.__class__.__name__)
            self._videos_w_rois = videos_w_rois

            # All project videos (target candidates after filter).
            self._video_dict = find_all_videos_in_directory(
                directory=self._reader.video_dir,
                as_dict=True,
                raise_error=False,
                sort_alphabetically=True)
            if not self._video_dict:
                raise NoFilesFoundError(
                    msg=f"No video files in {self._reader.video_dir}.",
                    source=self.__class__.__name__)
            self._project_video_names = sorted(
                list(self._video_dict.keys()), key=_natural_sort)
        except Exception as exc:
            QMessageBox.critical(
                self, "Cannot open duplicator",
                f"{type(exc).__name__}: {exc}")
            self._init_failed = True
            return

        self._init_failed = False
        self._build_ui()
        self._refresh_targets()

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def init_failed(self) -> bool:
        return getattr(self, "_init_failed", False)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Source — searchable QComboBox.
        source_group = QGroupBox("Source video (copy ROIs FROM)")
        source_layout = QHBoxLayout(source_group)
        self._source_combo = QComboBox()
        self._source_combo.setEditable(True)
        self._source_combo.setInsertPolicy(QComboBox.NoInsert)
        self._source_combo.addItems(self._videos_w_rois)
        self._source_combo.setCurrentIndex(0)
        # Patch 122dl: if a default_source was passed by the
        # caller AND it matches a video that has ROIs, pre-select
        # it. Otherwise keep the index-0 default (alphabetic
        # first). The default_source's basename (no extension) is
        # the canonical form used in self._videos_w_rois; tolerate
        # callers passing either basename or full path.
        if self._default_source:
            from pathlib import Path as _P
            cand = _P(self._default_source).stem
            if cand in self._videos_w_rois:
                self._source_combo.setCurrentText(cand)
        self._source_combo.currentTextChanged.connect(
            self._on_source_changed)
        source_layout.addWidget(QLabel("Source:"))
        source_layout.addWidget(self._source_combo, 1)
        layout.addWidget(source_group)

        # Filter + deselect-all controls.
        filter_group = QGroupBox("Filter targets")
        filter_layout = QHBoxLayout(filter_group)
        filter_layout.addWidget(QLabel("Filter:"))
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText(
            "(substring match — case-insensitive)")
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._filter_edit, 1)
        deselect_btn = QPushButton("Deselect all")
        deselect_btn.clicked.connect(self._deselect_all)
        filter_layout.addWidget(deselect_btn)
        layout.addWidget(filter_group)

        # Target list — ExtendedSelection mode for native
        # shift+click range select + ctrl+click toggle.
        target_group = QGroupBox(
            "Target videos (copy ROIs TO) — shift+click for range, "
            "ctrl+click to add individual")
        target_layout = QVBoxLayout(target_group)
        self._target_list = QListWidget()
        self._target_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection)
        target_layout.addWidget(self._target_list)
        layout.addWidget(target_group, 1)

        # Status bar.
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            "QLabel { padding: 4px; border: 1px solid #aaa; }")
        layout.addWidget(self._status_label)

        # Run / Cancel.
        self._buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._run_btn = self._buttons.addButton(
            "Run", QDialogButtonBox.AcceptRole)
        self._buttons.accepted.connect(self._run)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    # ------------------------------------------------------------------ #
    # Target list population + filter
    # ------------------------------------------------------------------ #
    def _refresh_targets(self) -> None:
        """Rebuild the target list, excluding the current source video.

        Preserves prior selection by reading currently-selected
        names before clearing, then re-applying after.
        """
        prior_selected = {item.text() for item
                          in self._target_list.selectedItems()}
        source = self._source_combo.currentText().strip()

        style = QApplication.instance().style()
        has_roi_icon = style.standardIcon(
            QStyle.SP_DialogYesButton)
        no_roi_icon = style.standardIcon(
            QStyle.SP_FileIcon)

        self._target_list.clear()
        for video_name in self._project_video_names:
            if video_name == source:
                continue  # skip source from target list
            icon = (has_roi_icon
                    if video_name in self._videos_w_rois
                    else no_roi_icon)
            item = QListWidgetItem(icon, video_name)
            self._target_list.addItem(item)
            if video_name in prior_selected:
                item.setSelected(True)
        self._apply_filter(self._filter_edit.text())

    def _apply_filter(self, filter_str: str) -> None:
        filter_lower = filter_str.strip().lower()
        for i in range(self._target_list.count()):
            item = self._target_list.item(i)
            if not filter_lower:
                item.setHidden(False)
            else:
                item.setHidden(filter_lower not in item.text().lower())

    def _on_filter_changed(self, text: str) -> None:
        self._apply_filter(text)

    def _on_source_changed(self, _text: str) -> None:
        self._refresh_targets()

    def _deselect_all(self) -> None:
        if not self._target_list.selectedItems():
            return  # nothing to confirm
        ans = QMessageBox.question(
            self, "Deselect all videos",
            "Clear all selected target videos?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans == QMessageBox.Yes:
            self._target_list.clearSelection()

    # ------------------------------------------------------------------ #
    # Status helpers
    # ------------------------------------------------------------------ #
    def _set_status(self, text: str, *,
                    color: str = "black") -> None:
        self._status_label.setStyleSheet(
            f"QLabel {{ padding: 4px; border: 1px solid #aaa; "
            f"color: {color}; }}")
        self._status_label.setText(text)

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        # Only currently-VISIBLE selections count — hiding via
        # filter shouldn't silently apply ROIs to videos the user
        # can't see. (Note: ExtendedSelection on a list with
        # hidden items still allows them to retain selection, but
        # we filter them out at run time.)
        selected_names = [
            item.text() for item in self._target_list.selectedItems()
            if not item.isHidden()
        ]

        source_video = self._source_combo.currentText().strip()

        if not selected_names:
            self._set_status(
                "No destination video(s) selected.", color="red")
            QMessageBox.warning(
                self, "No targets",
                "Select at least one target video first.")
            return
        if source_video not in self._videos_w_rois:
            QMessageBox.critical(
                self, "Invalid source",
                f"Source video '{source_video}' does not have any "
                f"ROIs defined.")
            return

        try:
            source_rois = get_roi_data_for_video_name(
                roi_path=self._reader.roi_coordinates_path,
                video_name=source_video)
            source_roi_cnt = len(list(source_rois.keys()))

            for target_video in selected_names:
                video_roi_dict = change_roi_dict_video_name(
                    roi_dict=source_rois,
                    video_name=target_video)
                (video_rectangles_df, video_circles_df,
                 video_polygon_df) = get_roi_df_from_dict(
                    roi_dict=video_roi_dict)
                # Overwrite any existing entries for this target.
                self._reader.rectangles_df = self._reader.rectangles_df[
                    self._reader.rectangles_df["Video"] != target_video]
                self._reader.circles_df = self._reader.circles_df[
                    self._reader.circles_df["Video"] != target_video]
                self._reader.polygon_df = self._reader.polygon_df[
                    self._reader.polygon_df["Video"] != target_video]
                self._reader.rectangles_df = pd.concat(
                    [self._reader.rectangles_df, video_rectangles_df],
                    axis=0).reset_index(drop=True)
                self._reader.circles_df = pd.concat(
                    [self._reader.circles_df, video_circles_df],
                    axis=0).reset_index(drop=True)
                self._reader.polygon_df = pd.concat(
                    [self._reader.polygon_df, video_polygon_df],
                    axis=0).reset_index(drop=True)

            store = pd.HDFStore(self._reader.roi_coordinates_path,
                                mode="w")
            try:
                store[Keys.ROI_RECTANGLES.value] = (
                    self._reader.rectangles_df)
                store[Keys.ROI_CIRCLES.value] = self._reader.circles_df
                store[Keys.ROI_POLYGONS.value] = self._reader.polygon_df
            finally:
                store.close()
        except Exception as exc:
            self._set_status(
                f"Duplicate failed: {type(exc).__name__}: {exc}",
                color="red")
            QMessageBox.critical(
                self, "Duplicate failed",
                f"{type(exc).__name__}: {exc}")
            return

        msg = (f"{source_roi_cnt} ROI(s) from '{source_video}' "
               f"applied to {len(selected_names)} other video(s).")
        self._set_status(msg, color="blue")
        QMessageBox.information(self, "Duplicated", msg)
        self.accept()
