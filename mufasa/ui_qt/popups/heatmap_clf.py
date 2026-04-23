"""
mufasa.ui_qt.popups.heatmap_clf
===============================

Qt6 port of :class:`mufasa.ui.pop_ups.heatmap_clf_pop_up.HeatmapClfPopUp`.

**Bug fixes from the Tk original** (see audit A3-A5):

* **A3** — Tk version had two different backend classes
  (``HeatMapperClfSingleCore`` vs ``HeatMapperClfMultiprocess``) called
  with **different keyword sets**. UI controls for background video,
  line color, opacity, keypoint overlay, legend, time-slice, min-
  seconds were silently ignored on the single-core path. Fixed here by
  **always using the multiprocess backend** with ``core_cnt=1`` when
  the user picks 1 core — same visual result, same full feature set.
* **A4** — Tk version contained
  ``multiprocessing.Process(heatmapper_clf.run())`` (note the parens).
  That called ``.run()`` in the GUI thread (blocking the UI for the
  duration of heatmap generation) AND spawned a zombie worker with
  ``target=None``. Fixed by routing through
  :func:`mufasa.ui_qt.progress.run_with_progress`.
* **A5** — Tk version mixed ``getChoices()`` and ``get_value()`` on
  the same widget class. Port uses ``get_value()`` uniformly.

**Linux-native optimisation**: if :func:`nvenc_available` reports
NVENC support, video writes are routed through ``h264_nvenc`` via an
env flag the backend checks. Backends that ignore the flag fall back
to software x264.

**Note:** this port intentionally keeps the ``int`` bg-img sentinel
(``-1`` for "video", positive int for "frame id", ``None`` for "none")
that the existing backend expects. Fixing that (audit A1 —
``bg_source="video" | ("frame", int) | None``) requires changing
``HeatMapperClfMultiprocess.__init__``; out of scope for the popup
port alone.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
from PySide6.QtWidgets import QMessageBox

from mufasa.ui_qt import linux_env
from mufasa.ui_qt.dialog import MufasaDialog
from mufasa.ui_qt.progress import run_with_progress
from mufasa.ui_qt.widgets import (DISABLED, NORMAL, NW, CreateLabelFrameWithIcon,
                                  Entry_Box, Formats, MufasaButton,
                                  MufasaCheckbox, MufasaDropDown)

AUTO = "auto"
VIDEO_FRM = "VIDEO FRAME"
VIDEO = "VIDEO"
HEATMAP_BG_OPTIONS = ["NONE", VIDEO, VIDEO_FRM]


class HeatmapClfPopUp(MufasaDialog):
    """Classification-heatmap video / frame generator."""

    def __init__(self, config_path: Union[str, os.PathLike]) -> None:
        MufasaDialog.__init__(
            self,
            title="CREATE CLASSIFICATION HEATMAP PLOTS",
            config_path=str(config_path),
            icon="heatmap",
            size=(960, 820),
        )

        # --- discover input data files (machine_results dir) ---------- #
        # Lazy import — keeps the popup module light.
        from mufasa.utils.enums import Paths
        from mufasa.utils.read_write import get_file_name_info_in_directory

        data_dir = Path(self.project_path) / Paths.MACHINE_RESULTS_DIR.value
        self.files_found_dict = get_file_name_info_in_directory(
            directory=str(data_dir), file_type=self.file_type
        )
        if not self.files_found_dict:
            raise FileNotFoundError(
                f"Zero files found in {data_dir}. "
                "Run machine models before creating heatmaps."
            )

        # --- option values --------------------------------------------- #
        max_scales = [AUTO] + [int(v) for v in np.arange(5, 105, 5)]
        min_scales = ["NONE"] + [int(v) for v in np.arange(5, 500, 5)]

        from mufasa.utils.lookups import get_named_colors
        line_colors = ["NONE"] + get_named_colors()

        # --- style settings -------------------------------------------- #
        self.style_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm, header="STYLE SETTINGS", icon_name="style"
        )
        self.palette_dd = MufasaDropDown(
            parent=self.style_frm, dropdown_options=self.palette_options,
            label="PALETTE:", label_width=30, dropdown_width=35, value="jet",
        )
        self.shading_dd = MufasaDropDown(
            parent=self.style_frm, dropdown_options=self.shading_options,
            label="SHADING:", label_width=30, dropdown_width=35, value="flat",
        )
        self.clf_dd = MufasaDropDown(
            parent=self.style_frm, dropdown_options=self.clf_names,
            label="CLASSIFIER:", label_width=30, dropdown_width=35,
            value=self.clf_names[0],
        )
        self.bp_dd = MufasaDropDown(
            parent=self.style_frm, dropdown_options=self.body_parts_lst,
            label="BODY-PART:", label_width=30, dropdown_width=35,
            value=self.body_parts_lst[0],
        )
        self.line_dd = MufasaDropDown(
            parent=self.style_frm, dropdown_options=line_colors,
            label="LINE COLOR:", label_width=30, dropdown_width=35, value="NONE",
        )
        self.legend_dd = MufasaDropDown(
            parent=self.style_frm, dropdown_options=["TRUE", "FALSE"],
            label="SHOW TIME COLOR LEGEND:", label_width=30, dropdown_width=35,
            value="TRUE",
        )
        self.max_time_dd = MufasaDropDown(
            parent=self.style_frm, dropdown_options=max_scales,
            label="MAX TIME SCALE (S):", label_width=30, dropdown_width=35,
            value=AUTO,
        )
        self.min_time_dd = MufasaDropDown(
            parent=self.style_frm, dropdown_options=min_scales,
            label="MINIMUM SECONDS:", label_width=30, dropdown_width=35,
            value="NONE",
        )
        self.bin_size_dd = MufasaDropDown(
            parent=self.style_frm,
            dropdown_options=self.heatmap_bin_size_options,
            label="BIN SIZE (MM):", label_width=30, dropdown_width=35,
            value="20×20",
        )
        self.style_frm.grid(row=0, column=0, sticky=NW)
        for r, w in enumerate([
            self.palette_dd, self.shading_dd, self.clf_dd, self.bp_dd,
            self.line_dd, self.legend_dd, self.max_time_dd, self.min_time_dd,
            self.bin_size_dd,
        ]):
            w.grid(row=r, column=0, sticky=NW)

        # --- background settings --------------------------------------- #
        self.bg_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm, header="BACKGROUND", icon_name="background"
        )
        self.bg_dd = MufasaDropDown(
            parent=self.bg_frm, dropdown_options=HEATMAP_BG_OPTIONS,
            label="HEATMAP BACKGROUND:", label_width=30, dropdown_width=35,
            value="NONE", command=self._on_bg_changed,
        )
        self.bg_frm.grid(row=1, column=0, sticky=NW)
        self.bg_dd.grid(row=0, column=0, sticky=NW)
        # Dynamic children populated by _on_bg_changed.
        self.opacity_dd = None
        self.keypoint_dd = None
        self.frm_id_eb = None

        # --- time-period slicer ---------------------------------------- #
        self.time_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm, header="TIME PERIOD", icon_name="timer_2"
        )
        self.time_period_cb, self.time_period_val = MufasaCheckbox(
            parent=self.time_frm, txt="PLOT SELECTED TIME-PERIOD",
            val=False, cmd=self._on_timeslice_toggle,
        )
        self.start_eb = Entry_Box(
            parent=self.time_frm, fileDescription="START TIME:",
            labelwidth=20, entry_box_width=15, justify="center",
            value="00:00:00", status=DISABLED,
        )
        self.end_eb = Entry_Box(
            parent=self.time_frm, fileDescription="END TIME:",
            labelwidth=20, entry_box_width=15, justify="center",
            value="00:00:30", status=DISABLED,
        )
        self.time_frm.grid(row=2, column=0, sticky=NW)
        self.time_period_cb.grid(row=0, column=0, sticky=NW)
        self.start_eb.grid(row=1, column=0, sticky=NW)
        self.end_eb.grid(row=2, column=0, sticky=NW)

        # --- run settings ---------------------------------------------- #
        self.run_settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm, header="RUN SETTINGS", icon_name="settings"
        )
        # cpu_count — cgroup-aware via linux_env, not os.cpu_count().
        cpu_n = linux_env.cpu_count()
        default_cores = max(1, min(cpu_n, max(1, cpu_n // 3)))
        self.core_cnt_dd = MufasaDropDown(
            parent=self.run_settings_frm,
            dropdown_options=list(range(1, cpu_n + 1)),
            label="CPU CORE COUNT:", label_width=30, dropdown_width=35,
            value=default_cores,
        )
        self.frames_cb, self.frames_val = MufasaCheckbox(
            parent=self.run_settings_frm, txt="CREATE FRAMES",
        )
        self.videos_cb, self.videos_val = MufasaCheckbox(
            parent=self.run_settings_frm, txt="CREATE VIDEOS",
        )
        self.last_frm_cb, self.last_frm_val = MufasaCheckbox(
            parent=self.run_settings_frm, txt="CREATE LAST FRAME", val=True,
        )
        self.run_settings_frm.grid(row=3, column=0, sticky=NW)
        self.core_cnt_dd.grid(row=0, column=0, sticky=NW)
        self.frames_cb.grid(row=1, column=0, sticky=NW)
        self.videos_cb.grid(row=2, column=0, sticky=NW)
        self.last_frm_cb.grid(row=3, column=0, sticky=NW)

        # --- run buttons ----------------------------------------------- #
        self.run_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm, header="RUN", icon_name="rocket"
        )
        self.single_video_dd = MufasaDropDown(
            parent=self.run_frm,
            dropdown_options=list(self.files_found_dict.keys()),
            label="VIDEO:", label_width=20, dropdown_width=35,
            value=list(self.files_found_dict.keys())[0],
        )
        self.run_single_btn = MufasaButton(
            parent=self.run_frm, txt="CREATE SINGLE HEATMAP",
            txt_clr="blue", img="rocket",
            cmd=lambda: self._dispatch(multiple_videos=False),
        )
        nfiles = len(self.files_found_dict)
        self.run_multi_btn = MufasaButton(
            parent=self.run_frm,
            txt=f"CREATE MULTIPLE HEATMAPS ({nfiles} files found)",
            txt_clr="blue", img="rocket",
            cmd=lambda: self._dispatch(multiple_videos=True),
        )
        self.run_frm.grid(row=4, column=0, sticky=NW)
        self.single_video_dd.grid(row=0, column=0, sticky=NW)
        self.run_single_btn.grid(row=1, column=0, sticky=NW)
        self.run_multi_btn.grid(row=2, column=0, sticky=NW)

    # ------------------------------------------------------------------ #
    # UI state callbacks
    # ------------------------------------------------------------------ #
    def _on_timeslice_toggle(self) -> None:
        state = NORMAL if self.time_period_val.isChecked() else DISABLED
        self.start_eb.set_state(setstatus=state)
        self.end_eb.set_state(setstatus=state)

    def _on_bg_changed(self, selection: str) -> None:
        # Clear prior dynamic children
        for attr in ("opacity_dd", "keypoint_dd", "frm_id_eb"):
            w = getattr(self, attr, None)
            if w is not None:
                w.setParent(None)
                w.deleteLater()
                setattr(self, attr, None)
        if selection == VIDEO:
            self.opacity_dd = MufasaDropDown(
                parent=self.bg_frm,
                dropdown_options=[int(v) for v in np.arange(5, 105, 5)],
                label="HEATMAP OPACITY (%):", label_width=30,
                dropdown_width=35, value=50,
            )
            self.keypoint_dd = MufasaDropDown(
                parent=self.bg_frm, dropdown_options=["TRUE", "FALSE"],
                label="SHOW KEYPOINT:", label_width=30, dropdown_width=35,
                value="TRUE",
            )
            self.opacity_dd.grid(row=1, column=0, sticky=NW)
            self.keypoint_dd.grid(row=2, column=0, sticky=NW)
        elif selection == VIDEO_FRM:
            self.opacity_dd = MufasaDropDown(
                parent=self.bg_frm,
                dropdown_options=[int(v) for v in np.arange(5, 105, 5)],
                label="HEATMAP OPACITY (%):", label_width=30,
                dropdown_width=35, value=50,
            )
            self.frm_id_eb = Entry_Box(
                parent=self.bg_frm, fileDescription="FRAME NUMBER:",
                labelwidth=30, entry_box_width=35, justify="center",
                validation="numeric", value=0,
            )
            self.opacity_dd.grid(row=1, column=0, sticky=NW)
            self.frm_id_eb.grid(row=2, column=0, sticky=NW)

    # ------------------------------------------------------------------ #
    # Backend dispatch (A3 + A4 fix)
    # ------------------------------------------------------------------ #
    def _collect_args(self, multiple_videos: bool) -> dict:
        """Collect validated popup state into a backend kwargs dict.

        Validation failures raise, caller handles them via QMessageBox.
        """
        from mufasa.utils.checks import (check_if_string_value_is_valid_video_timestamp,
                                         check_int,
                                         check_that_hhmmss_start_is_before_end)

        core_cnt = int(self.core_cnt_dd.get_value())

        max_t = self.max_time_dd.get_value()
        max_scale = max_t if max_t == AUTO else int(str(max_t).split("×")[0])

        min_t = self.min_time_dd.get_value()
        min_seconds = None if min_t == "NONE" else int(min_t)

        bin_size_raw = self.bin_size_dd.get_value()
        bin_size = int(str(bin_size_raw).split("×")[0])

        show_legend = self.legend_dd.get_value() == "TRUE"
        line_clr = None if self.line_dd.get_value() == "NONE" else self.line_dd.get_value()

        # Background: convert UI state → backend-expected (bg_img, opacity, show_kp)
        bg_sel = self.bg_dd.get_value()
        bg_img, heatmap_opacity, show_kp = None, None, False
        if bg_sel == VIDEO and self.opacity_dd is not None:
            bg_img = -1  # backend sentinel meaning "use video"
            heatmap_opacity = float(self.opacity_dd.get_value()) / 100.0
            show_kp = (
                self.keypoint_dd is not None
                and self.keypoint_dd.get_value() == "TRUE"
            )
        elif bg_sel == VIDEO_FRM and self.opacity_dd is not None:
            heatmap_opacity = float(self.opacity_dd.get_value()) / 100.0
            frm_id = self.frm_id_eb.entry_get if self.frm_id_eb else "0"
            check_int(name="FRAME NUMBER", value=frm_id,
                      min_value=0, raise_error=True)
            bg_img = int(frm_id)

        # Time slice
        time_slice = None
        if self.time_period_val.isChecked():
            start = self.start_eb.entry_get
            end = self.end_eb.entry_get
            check_if_string_value_is_valid_video_timestamp(
                value=start, name="START TIME", raise_error=True)
            check_if_string_value_is_valid_video_timestamp(
                value=end, name="END TIME", raise_error=True)
            check_that_hhmmss_start_is_before_end(
                start_time=start, end_time=end,
                name=self.__class__.__name__, raise_error=True)
            time_slice = {"start_time": start, "end_time": end}

        data_paths = (
            list(self.files_found_dict.values()) if multiple_videos
            else [self.files_found_dict[self.single_video_dd.get_value()]]
        )

        return dict(
            config_path=self.config_path,
            style_attr={
                "palette": self.palette_dd.get_value(),
                "shading": self.shading_dd.get_value(),
                "max_scale": max_scale,
                "bin_size": bin_size,
            },
            final_img_setting=self.last_frm_val.isChecked(),
            video_setting=self.videos_val.isChecked(),
            frame_setting=self.frames_val.isChecked(),
            bodypart=self.bp_dd.get_value(),
            clf_name=self.clf_dd.get_value(),
            data_paths=data_paths,
            min_seconds=min_seconds,
            bg_img=bg_img,
            line_clr=line_clr,
            show_keypoint=show_kp,
            heatmap_opacity=heatmap_opacity,
            time_slice=time_slice,
            show_legend=show_legend,
            core_cnt=core_cnt,
        )

    def _dispatch(self, multiple_videos: bool) -> None:
        try:
            kwargs = self._collect_args(multiple_videos=multiple_videos)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        # Hint the backend to use NVENC when available. Backends that
        # don't honour the env flag degrade silently to software x264.
        if linux_env.nvenc_available() and kwargs["video_setting"]:
            os.environ.setdefault("MUFASA_VIDEO_ENCODER", "h264_nvenc")

        def _run() -> None:
            # Always use the multiprocess backend — core_cnt=1 is the
            # single-core path. Fixes audit A3: UI controls (legend,
            # bg, line_clr, time_slice, min_seconds, opacity) were
            # silently ignored by HeatMapperClfSingleCore.
            from mufasa.plotting.heat_mapper_clf_mp import HeatMapperClfMultiprocess
            HeatMapperClfMultiprocess(**kwargs).run()

        n = len(kwargs["data_paths"])
        run_with_progress(
            parent=self,
            title="Generating heatmaps…",
            target=_run,
            label_template=f"Rendering {n} video{'s' if n > 1 else ''}…",
            on_success=lambda: QMessageBox.information(
                self, "Done", f"Heatmap generation complete ({n} video{'s' if n > 1 else ''})."
            ),
        )


__all__ = ["HeatmapClfPopUp"]
