"""
mufasa.ui_qt.forms.video_info
==============================

Per-video calibration form: edits the ``logs/video_info.csv``
table that powers all distance-converting feature kernels
(framewise_euclidean_distance, framewise_euclidean_distance_roi,
border_distances). Without this file, distance features come out
in pixels rather than millimeters, which is meaningless for cross-
study comparison.

The legacy Tk UI implements this as ``mufasa.ui.video_info_ui.
VideoInfoTable`` (309 lines of grid-widget plumbing). This Qt port
replaces it with a single :class:`VideoInfoForm` containing a
``QTableWidget`` (one row per video) plus apply-to-all utility
buttons. The form lives in the Data Import workbench page, which
is the natural spot in the workflow: calibration is part of
preparing imported data for analysis.

Columns (matches ``Formats.EXPECTED_VIDEO_INFO_COLS``):
  Video, fps, Resolution_width, Resolution_height,
  Distance_in_mm, pixels/mm

The first three are auto-filled from the video file via
``get_video_meta_data`` and shown read-only (editing fps or
resolution would silently desync the project from the actual
video). The last two are the calibration: a known-distance
reference (the mm length of something visible in the frame) and
the pixels-per-mm value (computed by the per-row Calibrate
button which opens an OpenCV line-drawing widget).

Workflow
--------

1. Open the Data Import page → "Video parameters & calibration"
   section.
2. The table loads with one row per video in
   ``project_folder/videos/`` (or auto-detected from
   ``input_csv/`` filenames if videos haven't been copied in
   yet).
3. For each video, type the known reference distance in mm
   (e.g. 100 if you placed a 10 cm ruler in the frame).
4. Click the row's "Calibrate…" button — opens an OpenCV
   window. Click two points to define the reference line. The
   px/mm value is filled in automatically.
5. If all videos use the same camera setup, fill in row 1, then
   click "Apply row 1 to all" for known distance and px/mm.
6. Click Save. Writes ``logs/video_info.csv``.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QSizePolicy, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from mufasa.ui_qt.workbench import OperationForm


# Column indices for the table. Don't reorder without updating
# the cellChanged handler and the save() column-name mapping.
_COL_VIDEO = 0
_COL_FPS = 1
_COL_WIDTH = 2
_COL_HEIGHT = 3
_COL_DISTANCE = 4
_COL_PX_PER_MM = 5
_COL_CALIBRATE = 6  # button column

_HEADERS = [
    "Video", "fps", "Width", "Height",
    "Distance (mm)", "pixels/mm", "Calibrate",
]


class VideoInfoForm(OperationForm):
    """Per-video pixel calibration table. Reads/writes
    ``project_folder/logs/video_info.csv``.

    Required by all distance kernels; without it, distance
    features are in pixels rather than mm.
    """

    title = "Video parameters & calibration"
    description = (
        "Set per-video FPS, resolution, and pixels-per-mm "
        "calibration. Required by all distance-based feature "
        "kernels (Euclidean distances, ROI distances, border "
        "distances) — without it, distances come out in pixels "
        "rather than millimeters. FPS and resolution are read "
        "from the video file automatically; you only need to "
        "fill the Distance (mm) reference and click the per-row "
        "Calibrate button to derive pixels/mm."
    )

    def build(self) -> None:
        # Action row: Reload (rebuild table from disk + video meta),
        # Apply-to-all (replicate row 1's distance / px-per-mm to
        # all other rows), Save.
        action_row = QHBoxLayout()
        self.btn_reload = QPushButton("Reload from project", self)
        self.btn_reload.clicked.connect(self._reload)
        self.btn_reload.setToolTip(
            "Re-scan the project's videos/ directory and load any "
            "existing values from logs/video_info.csv. Discards "
            "unsaved edits."
        )
        action_row.addWidget(self.btn_reload)

        self.btn_apply_distance = QPushButton(
            "Apply row 1 distance to all", self,
        )
        self.btn_apply_distance.clicked.connect(
            lambda: self._apply_first_row_to_all(_COL_DISTANCE)
        )
        self.btn_apply_distance.setToolTip(
            "Copy the Distance (mm) value from row 1 to all other "
            "rows. Use when all videos share the same camera setup "
            "and reference distance."
        )
        action_row.addWidget(self.btn_apply_distance)

        self.btn_apply_px = QPushButton("Apply row 1 px/mm to all", self)
        self.btn_apply_px.clicked.connect(
            lambda: self._apply_first_row_to_all(_COL_PX_PER_MM)
        )
        self.btn_apply_px.setToolTip(
            "Copy the pixels/mm value from row 1 to all other rows. "
            "Use when all videos share the same camera setup."
        )
        action_row.addWidget(self.btn_apply_px)

        action_row.addStretch(1)

        self.btn_save = QPushButton("Save to video_info.csv", self)
        self.btn_save.clicked.connect(self._save)
        self.btn_save.setToolTip(
            "Validate all rows and write the table to "
            "project_folder/logs/video_info.csv. Required before "
            "any distance-based feature extraction will produce "
            "valid mm-scaled values."
        )
        action_row.addWidget(self.btn_save)
        self.body_layout.addLayout(action_row)

        # The main table.
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        # Stretch policy: video name takes most space, numeric
        # columns get fixed sizes, calibrate button is small.
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(_COL_VIDEO, QHeaderView.Stretch)
        for col in (_COL_FPS, _COL_WIDTH, _COL_HEIGHT,
                    _COL_DISTANCE, _COL_PX_PER_MM, _COL_CALIBRATE):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding,
        )
        self.table.setMinimumHeight(220)
        self.body_layout.addWidget(self.table)

        # Status label below the table — shows current file path
        # or hints (e.g. "no project loaded"). Doesn't change the
        # form's required height much; just text.
        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
        self.status_label.setWordWrap(True)
        self.body_layout.addWidget(self.status_label)

        self._populate_if_possible()

    # --------------------------------------------------------------- #
    # OperationForm doesn't apply to this form — there's no run
    # button (saving happens via the Save button inside the form).
    # We override on_run to no-op so the parent's auto-Run button
    # doesn't fire collect_args / target.
    # --------------------------------------------------------------- #
    def collect_args(self) -> dict:  # pragma: no cover — not used
        # Required override; never called because we don't use
        # the OperationForm Run button. Save flow goes through
        # _save() instead.
        return {}

    def target(self, **_kwargs) -> None:  # pragma: no cover
        # Same as collect_args — present for the abstract API,
        # not invoked by user actions on this form.
        pass

    # --------------------------------------------------------------- #
    # Project loading
    # --------------------------------------------------------------- #
    # NB: when the user switches projects, the entire workbench is
    # rebuilt (see _switch_to_project in workbench.py), so this form
    # always sees the right config_path via __init__. No explicit
    # set_config_path hook is needed.

    def _populate_if_possible(self) -> None:
        """If a project is loaded, populate the table. Otherwise
        clear it and show a hint."""
        if not self.config_path or not os.path.isfile(self.config_path):
            self.table.setRowCount(0)
            self.status_label.setText(
                "Open a project (File → Open project) to enable "
                "video calibration."
            )
            return
        try:
            self._reload()
        except Exception as exc:
            QMessageBox.warning(
                self, f"{self.title}: load failed",
                f"Could not load video info: {type(exc).__name__}: {exc}",
            )

    def _reload(self) -> None:
        """Rebuild table from project state. Discards unsaved edits."""
        if not self.config_path:
            return

        rows = self._discover_rows()
        existing = self._load_existing_csv()
        self._fill_table(rows, existing)

        info_path = self._video_info_path()
        if os.path.isfile(info_path):
            self.status_label.setText(
                f"Loaded existing values from {info_path}. "
                f"Save to write back."
            )
        else:
            self.status_label.setText(
                f"No existing video_info.csv found at {info_path}. "
                f"Fill in calibration values and Save to create it."
            )

    def _video_info_path(self) -> str:
        """Path to logs/video_info.csv for the current project.
        Doesn't require the file to exist."""
        # ConfigReader stores VIDEO_INFO under Paths enum but we
        # don't need to instantiate ConfigReader (slow + checks
        # files). Hardcode the relative path; matches what
        # config_reader.py does.
        project_folder = os.path.dirname(self.config_path)
        return os.path.join(project_folder, "logs", "video_info.csv")

    def _discover_rows(self) -> List[Tuple[str, str]]:
        """List of (video_name, video_path) for the current project.

        Looks in ``project_folder/videos/`` first. Falls back to
        deriving names from ``csv/input_csv/`` entries if the
        videos directory is empty (videos may not be copied into
        the project tree in some workflows).
        """
        project_folder = os.path.dirname(self.config_path)
        videos_dir = os.path.join(project_folder, "videos")
        rows: List[Tuple[str, str]] = []
        if os.path.isdir(videos_dir):
            for entry in sorted(os.listdir(videos_dir)):
                full = os.path.join(videos_dir, entry)
                if os.path.isfile(full) and not entry.startswith("."):
                    name, ext = os.path.splitext(entry)
                    if ext.lower() in (".mp4", ".avi", ".mov", ".mkv",
                                       ".webm"):
                        rows.append((name, full))
        if not rows:
            # Fallback: derive from csv/input_csv/ filenames so the
            # form is at least useful for projects where videos
            # haven't been copied in.
            input_csv_dir = os.path.join(
                project_folder, "csv", "input_csv",
            )
            if os.path.isdir(input_csv_dir):
                for entry in sorted(os.listdir(input_csv_dir)):
                    full = os.path.join(input_csv_dir, entry)
                    if os.path.isfile(full) and not entry.startswith("."):
                        name, ext = os.path.splitext(entry)
                        if ext.lower() in (".csv", ".parquet"):
                            # No path to the actual video file —
                            # auto-fill from video meta won't work
                            # for these rows. The user fills FPS
                            # and resolution manually.
                            rows.append((name, ""))
        return rows

    def _load_existing_csv(self) -> Dict[str, Dict[str, str]]:
        """Read existing video_info.csv into a dict keyed by Video
        name. Returns empty dict if the file doesn't exist or is
        unreadable. Values are stored as strings so the table can
        display them verbatim."""
        info_path = self._video_info_path()
        if not os.path.isfile(info_path):
            return {}
        try:
            import pandas as pd
            df = pd.read_csv(info_path)
            df = df.astype(str)
            return {row["Video"]: dict(row) for _, row in df.iterrows()}
        except Exception:
            return {}

    def _fill_table(
        self,
        rows: List[Tuple[str, str]],
        existing: Dict[str, Dict[str, str]],
    ) -> None:
        """Populate the table widget from discovered rows + existing
        CSV values."""
        self.table.setRowCount(len(rows))
        for idx, (video_name, video_path) in enumerate(rows):
            # Video name (read-only — derived from filename)
            name_item = QTableWidgetItem(video_name)
            name_item.setFlags(
                name_item.flags() & ~Qt.ItemIsEditable
            )
            self.table.setItem(idx, _COL_VIDEO, name_item)

            # FPS, width, height — auto-fill from video file if we
            # have the path; else from existing CSV; else blank.
            fps, width, height = self._video_meta(video_path)
            if not fps and video_name in existing:
                fps = existing[video_name].get("fps", "")
            if not width and video_name in existing:
                width = existing[video_name].get("Resolution_width", "")
            if not height and video_name in existing:
                height = existing[video_name].get("Resolution_height", "")
            for col, val in (
                (_COL_FPS, fps),
                (_COL_WIDTH, width),
                (_COL_HEIGHT, height),
            ):
                item = QTableWidgetItem(str(val))
                # Auto-detected metadata is read-only. If the user
                # genuinely needs to override (e.g. video has a
                # corrupt FPS in its container), they can edit the
                # CSV directly afterward — but in-form override
                # would silently desync the project from the file.
                if val:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(idx, col, item)

            # Distance and px/mm — editable, populated from existing
            # CSV when available.
            distance = existing.get(video_name, {}).get(
                "Distance_in_mm", "",
            )
            px_per_mm = existing.get(video_name, {}).get("pixels/mm", "")
            self.table.setItem(
                idx, _COL_DISTANCE,
                QTableWidgetItem(str(distance) if distance != "nan" else ""),
            )
            self.table.setItem(
                idx, _COL_PX_PER_MM,
                QTableWidgetItem(
                    str(px_per_mm) if px_per_mm != "nan" else "",
                ),
            )

            # Calibrate button — opens OpenCV line-drawing widget
            # for this row's video. Disabled if video_path is
            # empty (no actual video file in project).
            btn = QPushButton("Calibrate…", self)
            btn.setEnabled(bool(video_path))
            btn.setToolTip(
                "Open the OpenCV line-drawing widget for this "
                "video. Set the Distance (mm) cell first, then "
                "click Calibrate and click two points spanning "
                "that reference distance in the video frame."
                if video_path
                else "Calibration unavailable: video file not "
                "found in project_folder/videos/."
            )
            btn.clicked.connect(
                lambda _checked=False, r=idx, p=video_path:
                    self._calibrate_row(r, p)
            )
            self.table.setCellWidget(idx, _COL_CALIBRATE, btn)

    def _video_meta(self, video_path: str) -> Tuple[str, str, str]:
        """Return (fps, width, height) for a video file as strings.
        Empty strings if path is empty or read fails."""
        if not video_path or not os.path.isfile(video_path):
            return ("", "", "")
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            try:
                if not cap.isOpened():
                    return ("", "", "")
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                cap.release()
            return (
                f"{fps:.2f}" if fps > 0 else "",
                str(width) if width > 0 else "",
                str(height) if height > 0 else "",
            )
        except Exception:
            return ("", "", "")

    # --------------------------------------------------------------- #
    # Apply-to-all
    # --------------------------------------------------------------- #
    def _apply_first_row_to_all(self, col: int) -> None:
        """Copy column ``col`` from row 0 to every other row."""
        if self.table.rowCount() < 2:
            return
        first_item = self.table.item(0, col)
        if first_item is None:
            return
        first_value = first_item.text().strip()
        if not first_value:
            QMessageBox.warning(
                self, self.title,
                "Row 1's value is empty. Fill it in first, then "
                "click Apply.",
            )
            return
        # Validate as a positive float before broadcasting — bad
        # values shouldn't propagate.
        try:
            v = float(first_value)
            if v <= 0:
                raise ValueError("must be positive")
        except ValueError as exc:
            QMessageBox.warning(
                self, self.title,
                f"Row 1's value {first_value!r} isn't a valid "
                f"positive number ({exc}). Fix it first.",
            )
            return
        for row in range(1, self.table.rowCount()):
            self.table.setItem(
                row, col, QTableWidgetItem(first_value),
            )

    # --------------------------------------------------------------- #
    # Per-row calibrate (OpenCV)
    # --------------------------------------------------------------- #
    def _calibrate_row(self, row: int, video_path: str) -> None:
        """Open the OpenCV pixel-calibration widget for this row's
        video and write the result back to the px/mm cell."""
        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(
                self, self.title,
                f"Video file not found: {video_path}",
            )
            return
        # Need the known-distance value first
        distance_item = self.table.item(row, _COL_DISTANCE)
        distance_str = distance_item.text().strip() if distance_item else ""
        try:
            known_distance = float(distance_str)
            if known_distance <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(
                self, self.title,
                "Set the Distance (mm) cell first (a positive "
                "number — the real-world length of the reference "
                "you'll click in the video frame).",
            )
            return

        try:
            from mufasa.video_processors.calculate_px_dist import (
                GetPixelsPerMillimeterInterface,
            )
        except ImportError:
            try:
                # Fallback location used by some Mufasa branches
                from mufasa.video_processors.px_dist_widget import (
                    GetPixelsPerMillimeterInterface,
                )
            except ImportError as exc:
                QMessageBox.critical(
                    self, self.title,
                    f"Calibration backend unavailable: {exc}",
                )
                return

        try:
            interface = GetPixelsPerMillimeterInterface(
                video_path=video_path,
                known_metric_mm=known_distance,
            )
            interface.run()
            ppm = float(getattr(interface, "ppm", 0))
        except Exception as exc:
            QMessageBox.critical(
                self, self.title,
                f"Calibration failed: {type(exc).__name__}: {exc}",
            )
            return

        if ppm <= 0:
            QMessageBox.warning(
                self, self.title,
                "Calibration produced no value (window closed "
                "without two clicks?). Try again.",
            )
            return
        # Write rounded value back to the cell
        self.table.setItem(
            row, _COL_PX_PER_MM, QTableWidgetItem(f"{ppm:.4f}"),
        )

    # --------------------------------------------------------------- #
    # Save
    # --------------------------------------------------------------- #
    def _save(self) -> None:
        """Validate all rows and write logs/video_info.csv."""
        if not self.config_path:
            QMessageBox.warning(
                self, self.title, "No project loaded.",
            )
            return
        if self.table.rowCount() == 0:
            QMessageBox.warning(
                self, self.title,
                "Nothing to save: no videos in the project.",
            )
            return

        # Collect rows
        import pandas as pd
        records = []
        errors = []
        for row in range(self.table.rowCount()):
            video = self.table.item(row, _COL_VIDEO).text()
            try:
                fps = self._validate_float(
                    row, _COL_FPS, "fps", min_value=0.0,
                )
                width = self._validate_int(
                    row, _COL_WIDTH, "Width", min_value=1,
                )
                height = self._validate_int(
                    row, _COL_HEIGHT, "Height", min_value=1,
                )
                distance = self._validate_float(
                    row, _COL_DISTANCE, "Distance (mm)",
                    min_value=0.0, allow_zero=False,
                )
                ppm = self._validate_float(
                    row, _COL_PX_PER_MM, "pixels/mm",
                    min_value=0.0, allow_zero=False,
                )
            except ValueError as exc:
                errors.append(f"Row {row + 1} ({video}): {exc}")
                continue
            records.append({
                "Video": video,
                "fps": fps,
                "Resolution_width": width,
                "Resolution_height": height,
                "Distance_in_mm": distance,
                "pixels/mm": ppm,
            })

        if errors:
            QMessageBox.warning(
                self, f"{self.title}: validation failed",
                "Cannot save — fix these issues first:\n\n"
                + "\n".join(errors[:10])
                + ("\n..." if len(errors) > 10 else ""),
            )
            return

        df = pd.DataFrame(records)
        info_path = self._video_info_path()
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        try:
            df.to_csv(info_path, index=False)
        except PermissionError as exc:
            QMessageBox.critical(
                self, self.title,
                f"Could not write {info_path}. "
                f"Is the file open in another program?\n\n{exc}",
            )
            return

        QMessageBox.information(
            self, self.title,
            f"Saved {len(records)} row(s) to:\n{info_path}",
        )

    def _validate_float(
        self, row: int, col: int, label: str,
        min_value: float = 0.0, allow_zero: bool = True,
    ) -> float:
        item = self.table.item(row, col)
        text = item.text().strip() if item else ""
        if not text:
            raise ValueError(f"{label} is empty")
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{label} {text!r} is not a number")
        if value < min_value:
            raise ValueError(f"{label} {value} < {min_value}")
        if not allow_zero and value == 0:
            raise ValueError(f"{label} cannot be zero")
        return value

    def _validate_int(
        self, row: int, col: int, label: str,
        min_value: int = 1,
    ) -> int:
        item = self.table.item(row, col)
        text = item.text().strip() if item else ""
        if not text:
            raise ValueError(f"{label} is empty")
        try:
            # Accept "640.0" as 640 — pandas may have saved
            # integer columns as floats.
            value = int(float(text))
        except ValueError:
            raise ValueError(f"{label} {text!r} is not an integer")
        if value < min_value:
            raise ValueError(f"{label} {value} < {min_value}")
        return value


__all__ = ["VideoInfoForm"]
