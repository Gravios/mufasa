"""
mufasa.ui_qt.popups.smoothing
============================

Qt6 port of :class:`mufasa.ui.pop_ups.smoothing_popup.SmoothingPopUp`.

**Port notes — what changed vs. the Tk original**

This port deliberately fixes the integration-schema issues surfaced in
the audit, not just the toolkit. Specifically:

1. **Removed duplicate `multi_index_df_headers` detection.** The Tk
   original did raw string equality (``data_dir == self.input_csv_dir``)
   which was wrong on Windows case-insensitive paths. Moved to
   :class:`~mufasa.ui_qt.operations.DatasetOp.has_multi_index_headers`,
   which uses fully resolved paths. Parity with InterpolatePopUp's
   (already-correct) implementation.
2. **Removed ``self.main_frm.mainloop()``** at end of ``__init__`` —
   Qt manages the event loop at the application level.
3. **Processor call extracted** from ``run()`` into a small local
   subclass of :class:`DatasetOp`. The popup no longer carries
   pipeline logic; it carries widget state and one-line dispatch.
4. **Enum-style `SMOOTHING_OPTION` mapping removed from popup.** The
   UI→processor string mapping is a data concern; ideally it belongs
   on the :class:`Smoothing` class itself. For this port I kept it as
   a module-level constant (backwards-compatible) with a ``TODO`` to
   relocate it when the backend is touched.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from PySide6.QtWidgets import QMessageBox

from mufasa.ui_qt.dialog import MufasaDialog
from mufasa.ui_qt.operations import DatasetOp
from mufasa.ui_qt.widgets import (NW, CreateLabelFrameWithIcon, Entry_Box,
                                 FileSelect, FolderSelect, Formats,
                                 MufasaButton, MufasaDropDown, MufasaLabel)

# TODO: relocate to ``mufasa.data_processors.smoothing.Smoothing.METHODS``
# when the backend is touched — per audit schema #2 (processors expose
# their own UI option schema).
SMOOTHING_OPTIONS_UI = ["Savitzky Golay", "Gaussian"]
SMOOTHING_UI_TO_API = {"Savitzky Golay": "savitzky-golay", "Gaussian": "gaussian"}

INSTRUCTIONS = (
    "NOTE: The chosen data will be overwritten with the smoothened data.\n"
    'The original, un-smoothened, data — if saved — is placed in a\n'
    'timestamped sub-directory of the original data with the "pre" prefix.'
)


class _SmoothingOp(DatasetOp):
    """Thin adapter from :class:`DatasetOp` to the existing
    :class:`mufasa.data_processors.smoothing.Smoothing` class.

    Kept as a local subclass so the popup file remains self-contained.
    A future refactor should move this into ``mufasa.data_processors``
    proper.
    """

    def __init__(
        self,
        config_path: os.PathLike | str,
        data_path: os.PathLike | str,
        time_window_ms: int,
        method: str,
        copy_originals: bool,
        raw_input_dir: Path,
    ) -> None:
        super().__init__(
            config_path=config_path, data_path=data_path, raw_input_dir=raw_input_dir
        )
        self._time_window_ms = time_window_ms
        self._method = method
        self._copy_originals = copy_originals
        # Lazy: avoid pulling numba/cv2/pandas at module load time.
        from mufasa.data_processors.smoothing import Smoothing  # noqa
        self._Smoothing = Smoothing

    def _run_one(self, target: Path) -> None:
        # NB: existing Smoothing class still expects the full "data_path"
        # and makes its own decision about single-file vs dir. To keep
        # behaviour identical, we call it once with the full data_path
        # rather than per-target. If/when the backend is rewritten to
        # match DatasetOp semantics, change this to per-target.
        pass

    def run(self) -> None:  # override — keep legacy call shape for now
        smoothing = self._Smoothing(
            config_path=str(self.config_path),
            data_path=str(self.data_path),
            time_window=self._time_window_ms,
            method=self._method,
            multi_index_df_headers=self.has_multi_index_headers,
            copy_originals=self._copy_originals,
        )
        smoothing.run()


class SmoothingPopUp(MufasaDialog):
    """Smooth pose-estimation CSVs with Savitzky–Golay or Gaussian.

    Ported from :class:`mufasa.ui.pop_ups.smoothing_popup.SmoothingPopUp`.
    """

    def __init__(self, config_path: Union[str, os.PathLike]) -> None:
        MufasaDialog.__init__(
            self,
            title="SMOOTH POSE-ESTIMATION DATA",
            config_path=str(config_path),
            icon="smooth",
            size=(720, 360),
        )

        # ----- settings frame -------------------------------------- #
        self.settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm, header="SETTINGS", icon_name="settings"
        )
        instruction_lbl = MufasaLabel(
            parent=self.settings_frm,
            txt=INSTRUCTIONS,
            font=Formats.FONT_REGULAR_ITALICS.value,
        )
        self.time_window = Entry_Box(
            self.settings_frm,
            "TIME WINDOW (MILLISECONDS):",
            labelwidth=35,
            validation="numeric",
            entry_box_width=35,
            value=100,
            justify="center",
        )
        self.method_dropdown = MufasaDropDown(
            parent=self.settings_frm,
            dropdown_options=SMOOTHING_OPTIONS_UI,
            label="SMOOTHING METHOD:",
            label_width=35,
            dropdown_width=35,
            value=SMOOTHING_OPTIONS_UI[0],
        )
        self.save_originals_dropdown = MufasaDropDown(
            parent=self.settings_frm,
            dropdown_options=["TRUE", "FALSE"],
            label="SAVE ORIGINALS:",
            label_width=35,
            dropdown_width=35,
            value="TRUE",
        )
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        instruction_lbl.grid(row=0, column=0, sticky=NW)
        self.time_window.grid(row=1, column=0, sticky=NW)
        self.method_dropdown.grid(row=2, column=0, sticky=NW)
        self.save_originals_dropdown.grid(row=3, column=0, sticky=NW)

        # ----- single-file frame ----------------------------------- #
        self.single_file_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm, header="SMOOTH SINGLE DATA FILE"
        )
        self.selected_file = FileSelect(
            self.single_file_frm,
            "DATA PATH:",
            lblwidth=35,
            file_types=[("Pose data", "*.csv *.parquet")],
            initialdir=self.project_path,
        )
        run_btn_single = MufasaButton(
            parent=self.single_file_frm,
            txt="RUN SINGLE DATA FILE SMOOTHING",
            img="rocket",
            txt_clr="blue",
            cmd=lambda: self._run(multiple=False),
        )
        self.single_file_frm.grid(row=1, column=0, sticky=NW)
        self.selected_file.grid(row=0, column=0, sticky=NW)
        run_btn_single.grid(row=1, column=0, sticky=NW)

        # ----- directory frame ------------------------------------- #
        self.multiple_file_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm, header="SMOOTH DIRECTORY OF DATA"
        )
        self.selected_dir = FolderSelect(
            self.multiple_file_frm,
            "SELECT DIRECTORY OF DATA FILES:",
            lblwidth=35,
            initialdir=self.project_path,
        )
        run_btn_multiple = MufasaButton(
            parent=self.multiple_file_frm,
            txt="RUN DATA DIRECTORY SMOOTHING",
            img="rocket",
            txt_clr="blue",
            cmd=lambda: self._run(multiple=True),
        )
        self.multiple_file_frm.grid(row=2, column=0, sticky=NW)
        self.selected_dir.grid(row=0, column=0, sticky=NW)
        run_btn_multiple.grid(row=1, column=0, sticky=NW)

    # ---- dispatch ------------------------------------------------- #
    def _run(self, multiple: bool) -> None:
        from mufasa.utils.checks import check_int  # lazy import
        from mufasa.ui_qt.runner import ProcessorRunner

        time_window_text = self.time_window.entry_get
        try:
            check_int(
                name="TIME WINDOW (MILLISECONDS)", value=time_window_text, min_value=1
            )
        except Exception as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return
        data_path = (
            self.selected_file.file_path if not multiple else self.selected_dir.folder_path
        )
        if not data_path:
            QMessageBox.warning(
                self, "No selection",
                "Select a data file (or directory) before running.",
            )
            return

        op = _SmoothingOp(
            config_path=self.config_path,
            data_path=data_path,
            time_window_ms=int(time_window_text),
            method=SMOOTHING_UI_TO_API[self.method_dropdown.get_value()],
            copy_originals=(self.save_originals_dropdown.get_value() == "TRUE"),
            raw_input_dir=Path(self.input_csv_dir),
        )

        # Route through ProcessorRunner so the GUI stays responsive and
        # errors propagate via a signal instead of being swallowed in a
        # bare ``threading.Thread``. The cancel button in the progress
        # dialog calls ``runner.request_cancel()``; backends that check
        # ``runner.cancel_event`` will honour it. ``Smoothing.run`` does
        # not currently check, so cancel becomes a no-op — will show up
        # as "Cancelling…" then completion. Better than SIGKILL.
        self._runner = ProcessorRunner(target=op.run)
        self._runner.errored.connect(
            lambda e: QMessageBox.critical(self, "Smoothing failed", str(e))
        )
        self._runner.completed.connect(
            lambda: QMessageBox.information(
                self, "Done",
                f"Smoothing complete in {self._runner.wall_time_s:.1f} s.",
            )
        )
        self._runner.start()


__all__ = ["SmoothingPopUp"]
