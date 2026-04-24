"""
mufasa.pose_importers.dlc_h5_importer
=====================================

Importer for **single-animal** DeepLabCut pose-estimation output in H5
format. The maDLC importer (``madlc_importer.MADLCImporterH5``) handles
the multi-animal case; this module handles the common single-animal
case which was missing from the upstream fork.

Why a separate module:
  * ``MADLCImporterH5`` requires ``id_lst`` and branches through
    ``multianimal_identification`` UI — inappropriate for single-animal
    data, where there's nothing to identify.
  * Single-animal DLC H5 files have a 3-level column MultiIndex
    (``scorer → bodypart → coord``); maDLC has 4 levels (with an
    ``individuals`` level between scorer and bodypart).

This importer shares the filename-matching logic with the CSV
importer — DLC network suffixes like ``DLC_Resnet50_<project>...``
are stripped so pose files match video stems.

Output is written to ``project_folder/csv/input_csv/`` in the project's
configured file type (csv or parquet), with the SimBA-standard
3-level multi-index column layout (IMPORTED_POSE, IMPORTED_POSE,
<bp>_{x|y|likelihood}).

Example
-------
>>> DLCSingleAnimalH5Importer(
...     config_path="/path/to/project_config.ini",
...     data_folder="/path/to/dlc/output",
... ).run()
"""
from __future__ import annotations

__author__ = "Gravio"

import os
from typing import Any, Dict, Optional, Union

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


class DLCSingleAnimalH5Importer(ConfigReader, PoseImporterMixin):
    """Import single-animal DeepLabCut H5 pose-estimation data into a
    Mufasa project.

    :param Union[str, os.PathLike] config_path: Path to Mufasa project
        ``project_config.ini``.
    :param Union[str, os.PathLike] data_folder: Directory containing
        DLC ``.h5`` output files (one per video).
    :param Optional[Dict[str, str]] interpolation_settings: Dict with
        ``'type'`` ('body-parts' or 'animals') and ``'method'``
        ('linear', 'quadratic', 'nearest'). Default: ``None``.
    :param Optional[Dict[str, Any]] smoothing_settings: Dict with
        ``'time_window'`` (int, ms) and ``'method'`` ('savitzky-golay'
        or 'gaussian'). Default: ``None``.

    :example:
    >>> DLCSingleAnimalH5Importer(
    ...     config_path="/data/project/project_config.ini",
    ...     data_folder="/data/dlc_output",
    ...     interpolation_settings={'type': 'body-parts', 'method': 'linear'},
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

        # Video discovery is best-effort: if the user hasn't imported
        # videos yet we can still import pose; the filenames are
        # normalized via the DLC-suffix stripper below regardless.
        self.video_paths = find_all_videos_in_project(
            videos_dir=self.video_dir, raise_error=False,
        )
        self.input_data_paths = self._find_h5_files(self.data_folder)
        if not self.input_data_paths:
            raise BodypartColumnNotFoundError(
                msg=f"No .h5 files found in {self.data_folder}",
                source=self.__class__.__name__,
            )

        # User-defined / non-preset bp configs bump the animal_cnt from
        # the default of 1 if the caller went through the UI flow that
        # allows it. For single-animal we keep animal_cnt=1, matching
        # what ConfigReader parsed from project_config.ini.
        if self.pose_setting is Methods.USER_DEFINED.value:
            # User-defined projects set their own bp list; only call
            # the shared update hook if animal_cnt somehow disagrees
            # with the single-animal assumption (it shouldn't).
            pass

        print(f"Importing {len(self.input_data_paths)} "
              f"single-animal DLC H5 file(s)...")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_h5_files(directory: Union[str, os.PathLike]) -> list:
        """Return sorted list of ``*.h5`` paths in *directory* (non-
        recursive). Skips hidden files."""
        out = []
        for name in sorted(os.listdir(directory)):
            if name.startswith("."):
                continue
            if name.lower().endswith(".h5"):
                out.append(os.path.join(directory, name))
        return out

    @staticmethod
    def _strip_dlc_suffix(stem: str) -> str:
        """Strip DLC network suffix from a filename stem.

        DLC output filenames look like::

            mouse01DLC_Resnet50_myproj_shuffle1_snapshot_best-150

        We want ``mouse01`` so the pose file joins to ``mouse01.mp4``.
        The delimiter casing varies (``DLC`` vs ``DeepCut`` across
        DLC versions) so we split on either.
        """
        for delim in ("DLC_", "DLC_resnet", "DLC_Resnet", "DLC_dlc",
                      "DeepCut"):
            if delim in stem:
                return stem.split(delim)[0]
        return stem

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Import every ``.h5`` found in ``self.data_folder``."""
        import_log_rows = []
        mask_totals: Dict[str, Dict[str, int]] = {}
        for cnt, h5_path in enumerate(self.input_data_paths):
            video_timer = SimbaTimer(start=True)
            raw_stem = get_fn_ext(filepath=h5_path)[1]
            video_name = self._strip_dlc_suffix(raw_stem)
            print(f"Processing {video_name} "
                  f"({cnt + 1}/{len(self.input_data_paths)})...")

            try:
                df = pd.read_hdf(h5_path)
            except Exception as exc:
                raise BodypartColumnNotFoundError(
                    msg=f"Could not read {h5_path}: {type(exc).__name__}: "
                    f"{exc}. Is the file a valid DLC H5? If it's a "
                    f"multi-animal H5 (column nlevels == 4), use the "
                    f"maDLC importer instead.",
                    source=self.__class__.__name__,
                )

            # Guard against multi-animal H5 being fed into this
            # importer — the column structure differs and would write
            # garbage to disk.
            if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 4:
                raise BodypartColumnNotFoundError(
                    msg=f"{h5_path} appears to be a multi-animal DLC H5 "
                    f"(column MultiIndex has 4 levels: "
                    f"{df.columns.names}). Use the maDLC H5 importer "
                    f"(Data import → 'DLC (multi-animal H5)') instead.",
                    source=self.__class__.__name__,
                )

            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Validate bp count matches project config. bp_headers is
            # populated by ConfigReader from project_bp_names.csv;
            # length = 3 * n_bodyparts (x, y, likelihood per bp).
            if len(df.columns) != len(self.bp_headers):
                raise BodypartColumnNotFoundError(
                    msg=(
                        f"Body-part count mismatch for {h5_path}: the "
                        f"DLC file has {len(df.columns) // 3} body-parts "
                        f"but this project expects "
                        f"{len(self.bp_headers) // 3}. Either recreate "
                        f"the project with the correct preset (or "
                        f"'user_defined' with your custom body-part "
                        f"list), or import a DLC file matching the "
                        f"project's configured body-parts. "
                        f"Project body-parts are listed at "
                        f"{self.body_parts_path}."
                    ),
                    source=self.__class__.__name__,
                )

            # Replace the DLC column MultiIndex with the project's
            # flat bp-header list, then re-nest into SimBA's
            # IMPORTED_POSE multi-index. This is the shape write_df's
            # multi_idx_header=True path expects.
            df.columns = self.bp_headers

            # Likelihood masking: zero out (x, y) for points below the
            # confidence threshold so downstream interpolation picks
            # them up as the (0, 0) sentinel. Likelihood column itself
            # is preserved. This is where DLC's confidence signal
            # becomes actionable in Mufasa.
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
                "IMPORT_SOURCE": h5_path,
                "P_THRESHOLD": self.p_threshold,
                "MASKED_POINTS": total_masked,
                "INTERPOLATION_SETTING": str(self.interpolation_settings),
                "SMOOTHING_SETTING": str(self.smoothing_settings),
            })
            stdout_success(
                msg=f"Video {video_name} data imported...",
                elapsed_time=video_timer.elapsed_time_str,
            )

        # Persist the import log
        if import_log_rows:
            pd.DataFrame(import_log_rows).to_csv(self.import_log_path,
                                                 index=False)

        self.timer.stop_timer()
        stdout_success(
            msg=f"All single-animal DLC H5 data files imported to "
            f"{self.input_csv_dir} directory",
            elapsed_time=self.timer.elapsed_time_str,
        )
