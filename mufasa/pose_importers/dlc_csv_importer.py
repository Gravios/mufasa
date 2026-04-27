"""
mufasa.pose_importers.dlc_csv_importer
======================================

Importer for **single-animal** DeepLabCut pose-estimation output in
CSV format. Built as a parallel to
:class:`mufasa.pose_importers.dlc_h5_importer.DLCSingleAnimalH5Importer`
so the two formats share the same UI controls and behaviour
(likelihood thresholding, interpolation, smoothing).

Mufasa already shipped a CSV importer (``import_dlc_csv_data`` in
``dlc_importer_csv.py``), but it:

* doesn't support per-frame likelihood masking
* uses a copy-then-rename file flow that bypasses the
  IMPORTED_POSE multi-index normalization
* is a function rather than a class (no shared state with the form's
  threshold/interpolation/smoothing collection logic)

Rather than refactor that to bolt on these features, this module
implements the full pipeline in line with the H5 importer. The
existing function remains for backwards compatibility / scripting.

Public API matches the H5 importer:

    DLCSingleAnimalCSVImporter(
        config_path=..., data_folder=...,
        interpolation_settings=..., smoothing_settings=...,
        p_threshold=...,
    ).run()

DLC CSV format details handled:

* 3-row multi-header (scorer / bodyparts / coords) — read via
  ``header=[0, 1, 2]``
* Empty cells: DLC writes literal blanks for low-confidence frames
  in some export settings. ``pd.read_csv`` parses these as NaN, which
  ``df.fillna(0)`` then converts to (0, 0) — picked up by the
  interpolator as missing.
* DLC suffix in filename (``..._DLC_HrnetW32_...``) — stripped to
  match video stems.
"""
from __future__ import annotations

__author__ = "Gravio"

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mufasa.data_processors.interpolate import Interpolate
from mufasa.data_processors.smoothing import Smoothing
from mufasa.mixins.config_reader import ConfigReader
from mufasa.mixins.pose_importer_mixin import PoseImporterMixin
from mufasa.utils.checks import (check_file_exist_and_readable,
                                 check_if_dir_exists,
                                 check_if_keys_exist_in_dict, check_int,
                                 check_str)
from mufasa.utils.enums import Formats, Methods
from mufasa.utils.errors import BodypartColumnNotFoundError
from mufasa.utils.printing import SimbaTimer, stdout_success
from mufasa.utils.read_write import (find_all_videos_in_project, get_fn_ext,
                                     write_df)


class DLCSingleAnimalCSVImporter(ConfigReader, PoseImporterMixin):
    """Import single-animal DeepLabCut CSV pose-estimation data into a
    Mufasa project.

    Mirrors :class:`DLCSingleAnimalH5Importer`'s constructor and
    behaviour. See that class for the full parameter docs; the only
    difference is that ``data_folder`` here contains ``.csv`` files
    instead of ``.h5``.

    :example:
    >>> DLCSingleAnimalCSVImporter(
    ...     config_path="/data/project/project_config.ini",
    ...     data_folder="/data/dlc_output",
    ...     interpolation_settings={'type': 'body-parts', 'method': 'linear'},
    ...     p_threshold=0.5,
    ... ).run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_folder: Union[str, os.PathLike],
        interpolation_settings: Optional[Dict[str, str]] = None,
        smoothing_settings: Optional[Dict[str, Any]] = None,
        p_threshold: float = 0.0,
    ) -> None:
        check_file_exist_and_readable(file_path=config_path)
        check_if_dir_exists(in_dir=data_folder)
        if not (0.0 <= float(p_threshold) <= 1.0):
            raise ValueError(
                f"p_threshold must be in [0.0, 1.0], got {p_threshold}"
            )
        if interpolation_settings is not None:
            check_if_keys_exist_in_dict(
                data=interpolation_settings, key=['method', 'type'],
                name=f'{self.__class__.__name__} interpolation_settings',
            )
            check_str(
                name=f'{self.__class__.__name__} interpolation_settings type',
                value=interpolation_settings['type'],
                options=('body-parts', 'animals'),
            )
            check_str(
                name=f'{self.__class__.__name__} interpolation_settings method',
                value=interpolation_settings['method'],
                options=('linear', 'quadratic', 'nearest'),
            )
        if smoothing_settings is not None:
            check_if_keys_exist_in_dict(
                data=smoothing_settings, key=['method', 'time_window'],
                name=f'{self.__class__.__name__} smoothing_settings',
            )
            check_str(
                name=f'{self.__class__.__name__} smoothing_settings method',
                value=smoothing_settings['method'],
                options=('savitzky-golay', 'gaussian'),
            )
            check_int(
                name=f'{self.__class__.__name__} smoothing_settings time_window',
                value=smoothing_settings['time_window'], min_value=1,
            )

        ConfigReader.__init__(self, config_path=config_path,
                              read_video_info=False)
        PoseImporterMixin.__init__(self)
        self.interpolation_settings = interpolation_settings
        self.smoothing_settings = smoothing_settings
        self.p_threshold = float(p_threshold)
        self.data_folder = data_folder
        self.import_log_path = os.path.join(
            self.logs_path, f"data_import_log_{self.datetime}.csv"
        )

        self.video_paths = find_all_videos_in_project(
            videos_dir=self.video_dir, raise_error=False,
        )
        self.input_data_paths = self._find_csv_files(self.data_folder)
        if not self.input_data_paths:
            raise BodypartColumnNotFoundError(
                msg=f"No .csv files found in {self.data_folder}",
                source=self.__class__.__name__,
            )

        print(f"Importing {len(self.input_data_paths)} "
              f"single-animal DLC CSV file(s)...")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_csv_files(directory: Union[str, os.PathLike]) -> List[str]:
        """Sorted list of .csv paths in *directory* (non-recursive)."""
        out = []
        for name in sorted(os.listdir(directory)):
            if name.startswith("."):
                continue
            if name.lower().endswith(".csv"):
                out.append(os.path.join(directory, name))
        return out

    @staticmethod
    def _strip_dlc_suffix(stem: str) -> str:
        """See ``DLCSingleAnimalH5Importer._strip_dlc_suffix``."""
        for delim in ("DLC_", "DeepCut"):
            if delim in stem:
                return stem.split(delim)[0]
        return stem

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Import every ``.csv`` found in ``self.data_folder``."""
        import_log_rows = []
        mask_totals: Dict[str, Dict[str, int]] = {}
        for cnt, csv_path in enumerate(self.input_data_paths):
            video_timer = SimbaTimer(start=True)
            raw_stem = get_fn_ext(filepath=csv_path)[1]
            video_name = self._strip_dlc_suffix(raw_stem)
            print(f"Processing {video_name} "
                  f"({cnt + 1}/{len(self.input_data_paths)})...")

            # DLC CSV: 3 header rows (scorer / bodyparts / coords) +
            # an unnamed leading frame-index column.
            try:
                df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
            except Exception as exc:
                raise BodypartColumnNotFoundError(
                    msg=f"Could not read {csv_path}: "
                    f"{type(exc).__name__}: {exc}. Is the file a valid "
                    f"DLC CSV with a 3-row multi-header (scorer / "
                    f"bodyparts / coords)?",
                    source=self.__class__.__name__,
                )

            # Reject multi-animal CSV (4-level header). Same guard as
            # the H5 importer.
            if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 4:
                raise BodypartColumnNotFoundError(
                    msg=f"{csv_path} appears to be a multi-animal DLC "
                    f"CSV (column MultiIndex has 4 levels: "
                    f"{df.columns.names}). Use a maDLC importer "
                    f"instead.",
                    source=self.__class__.__name__,
                )

            # NaNs are converted to 0 here so downstream interpolation
            # treats them as missing. The likelihood mask (below)
            # additionally zeroes out (x, y) for points whose
            # likelihood is below the threshold.
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

            if len(df.columns) != len(self.bp_headers):
                raise BodypartColumnNotFoundError(
                    msg=(
                        f"Body-part count mismatch for {csv_path}: the "
                        f"DLC file has {len(df.columns) // 3} body-parts "
                        f"but this project expects "
                        f"{len(self.bp_headers) // 3}. Either recreate "
                        f"the project with the correct preset (or "
                        f"'user_defined' with your custom body-part "
                        f"list), or use File → Reconfigure project from "
                        f"DLC file… to autodetect from this CSV. "
                        f"Project body-parts are listed at "
                        f"{self.body_parts_path}."
                    ),
                    source=self.__class__.__name__,
                )

            df.columns = self.bp_headers

            # Likelihood masking (see dlc_h5_importer for the full why).
            if self.p_threshold > 0.0:
                from mufasa.pose_importers.likelihood_mask import (
                    apply_likelihood_threshold, summarize_mask_counts,
                )
                df, counts = apply_likelihood_threshold(
                    df, threshold=self.p_threshold,
                )
                summary = summarize_mask_counts(counts, n_frames=len(df))
                if summary:
                    print(summary)
                mask_totals[video_name] = counts

            out_df = self.insert_multi_idx_columns(df=df.fillna(0))

            save_path = os.path.join(
                self.input_csv_dir, f"{video_name}.{self.file_type}"
            )
            write_df(df=out_df, file_type=self.file_type,
                     save_path=save_path, multi_idx_header=True)

            if self.interpolation_settings is not None:
                interpolator = Interpolate(
                    config_path=self.config_path, data_path=save_path,
                    type=self.interpolation_settings['type'],
                    method=self.interpolation_settings['method'],
                    multi_index_df_headers=True, copy_originals=False,
                )
                interpolator.run()
            if self.smoothing_settings is not None:
                smoother = Smoothing(
                    config_path=self.config_path, data_path=save_path,
                    time_window=self.smoothing_settings['time_window'],
                    method=self.smoothing_settings['method'],
                    multi_index_df_headers=True, copy_originals=False,
                )
                smoother.run()

            video_timer.stop_timer()
            total_masked = sum(mask_totals.get(video_name, {}).values())
            import_log_rows.append({
                "VIDEO": video_name,
                "IMPORT_TIME": video_timer.elapsed_time_str,
                "IMPORT_SOURCE": csv_path,
                "P_THRESHOLD": self.p_threshold,
                "MASKED_POINTS": total_masked,
                "INTERPOLATION_SETTING": str(self.interpolation_settings),
                "SMOOTHING_SETTING": str(self.smoothing_settings),
            })
            stdout_success(
                msg=f"Video {video_name} data imported...",
                elapsed_time=video_timer.elapsed_time_str,
            )

        if import_log_rows:
            pd.DataFrame(import_log_rows).to_csv(self.import_log_path,
                                                 index=False)

        self.timer.stop_timer()
        stdout_success(
            msg=f"All single-animal DLC CSV data files imported to "
            f"{self.input_csv_dir} directory",
            elapsed_time=self.timer.elapsed_time_str,
        )
