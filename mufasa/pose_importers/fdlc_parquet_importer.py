"""
mufasa.pose_importers.fdlc_parquet_importer
===========================================

Importer for **single-animal** FreeDLC pose output in the long/tidy
Parquet format written by the modified DeepLabCut (FreeDLC) export:

    frame | individual | bodypart | x | y | likelihood

one row per (frame, individual, bodypart). Files are named
``<video_stem>.fdlc.parquet``.

This mirrors :class:`mufasa.pose_importers.dlc_h5_importer.DLCSingleAnimalH5Importer`
— same project alignment, likelihood masking, multi-index output,
interpolation/smoothing, and import log — but adapts the *read* step for
the long Parquet layout instead of DLC's wide 3-level-MultiIndex H5.

Two differences from the H5 path, both because the Parquet carries named
nodes (its "skeleton" node set):

* **Name-based alignment.** The H5 importer is positional
  (``df.columns = self.bp_headers``). Here the bodypart column lets us map
  the data's nodes to the project's configured body-parts by *name* and
  reorder into the project's order — so a differently-ordered export still
  lands correctly. Exact match is tried first, then case-insensitive, then
  a positional fallback (parity with H5) when the counts match but names
  don't, and finally a clear error.
* **Likelihood sentinel.** FreeDLC writes ``-1.0`` for "no detection";
  negative confidences are clamped to ``0.0`` so the ``p_threshold`` mask
  treats them as low-confidence points.

Multi-animal Parquet (``individual`` with more than one value) is rejected
with a pointer to a future maDLC-parquet path, matching how the single vs
multi H5 importers are kept separate.

Output: ``project_folder/csv/input_csv/<video>.<file_type>`` with the
SimBA-standard IMPORTED_POSE 3-level multi-index columns.
"""
from __future__ import annotations

__author__ = "Gravio"

import os
from typing import Any

import numpy as np
import pandas as pd

from mufasa.data_processors.interpolate import Interpolate
from mufasa.data_processors.smoothing import Smoothing
from mufasa.mixins.config_reader import ConfigReader
from mufasa.mixins.pose_importer_mixin import PoseImporterMixin
from mufasa.utils.checks import (
    check_file_exist_and_readable,
    check_if_dir_exists,
    check_if_keys_exist_in_dict,
    check_int,
    check_str,
)
from mufasa.utils.errors import BodypartColumnNotFoundError
from mufasa.utils.printing import SimbaTimer, stdout_success
from mufasa.utils.read_write import find_all_videos_in_project, get_fn_ext, write_df

FDLC_SUFFIX = ".fdlc.parquet"
_LONG_COLUMNS = ("frame", "individual", "bodypart", "x", "y", "likelihood")


class FDLCParquetImporter(ConfigReader, PoseImporterMixin):
    """Import single-animal FreeDLC ``*.fdlc.parquet`` pose data into a
    Mufasa project.

    :param Union[str, os.PathLike] config_path: Path to project ``project.toml``.
    :param Union[str, os.PathLike] data_folder: Directory of ``*.fdlc.parquet`` files.
    :param Optional[Dict[str, str]] interpolation_settings: ``{'type', 'method'}``.
    :param Optional[Dict[str, Any]] smoothing_settings: ``{'time_window', 'method'}``.
    :param float p_threshold: Confidence below which (x, y) is masked. Default 0.0.
    """

    def __init__(
        self,
        config_path: str | os.PathLike,
        data_folder: str | os.PathLike,
        interpolation_settings: dict[str, str] | None = None,
        smoothing_settings: dict[str, Any] | None = None,
        p_threshold: float = 0.0,
    ) -> None:
        check_file_exist_and_readable(file_path=config_path)
        check_if_dir_exists(in_dir=data_folder)
        if not (0.0 <= float(p_threshold) <= 1.0):
            raise ValueError(f"p_threshold must be in [0.0, 1.0], got {p_threshold}")
        if interpolation_settings is not None:
            check_if_keys_exist_in_dict(
                data=interpolation_settings, key=["method", "type"],
                name=f"{self.__class__.__name__} interpolation_settings",
            )
            check_str(
                name=f"{self.__class__.__name__} interpolation_settings type",
                value=interpolation_settings["type"],
                options=("body-parts", "animals"),
            )
            check_str(
                name=f"{self.__class__.__name__} interpolation_settings method",
                value=interpolation_settings["method"],
                options=("linear", "quadratic", "nearest"),
            )
        if smoothing_settings is not None:
            check_if_keys_exist_in_dict(
                data=smoothing_settings, key=["method", "time_window"],
                name=f"{self.__class__.__name__} smoothing_settings",
            )
            check_str(
                name=f"{self.__class__.__name__} smoothing_settings method",
                value=smoothing_settings["method"],
                options=("savitzky-golay", "gaussian"),
            )
            check_int(
                name=f"{self.__class__.__name__} smoothing_settings time_window",
                value=smoothing_settings["time_window"], min_value=1,
            )

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
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
        self.input_data_paths = self._find_fdlc_files(self.data_folder)
        if not self.input_data_paths:
            raise BodypartColumnNotFoundError(
                msg=f"No {FDLC_SUFFIX} files found in {self.data_folder}",
                source=self.__class__.__name__,
            )
        print(f"Importing {len(self.input_data_paths)} FreeDLC parquet file(s)...")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_fdlc_files(directory: str | os.PathLike) -> list:
        out = []
        for name in sorted(os.listdir(directory)):
            if name.startswith("."):
                continue
            if name.lower().endswith(FDLC_SUFFIX):
                out.append(os.path.join(directory, name))
        return out

    @staticmethod
    def _video_name(path: str) -> str:
        """Strip ``.fdlc`` (and the ``.parquet`` ext) to recover the video
        stem, so ``mouse01.fdlc.parquet`` joins to ``mouse01.mp4``."""
        stem = get_fn_ext(filepath=path)[1]
        if stem.lower().endswith(".fdlc"):
            stem = stem[: -len(".fdlc")]
        return stem

    @staticmethod
    def long_to_wide(
        df_long: pd.DataFrame,
        project_bodyparts: list[str],
        *,
        source: str = "FDLCParquetImporter",
    ) -> pd.DataFrame:
        """Pivot the long FreeDLC frame to a wide SimBA pose frame.

        Returns a DataFrame indexed 0..n_frames-1 whose columns are
        ``[<bp>_x, <bp>_y, <bp>_p, ...]`` in ``project_bodyparts`` order.
        Aligns the data's nodes to the project's body-parts by name (exact,
        then case-insensitive), falling back to positional order when the
        counts match but the names don't, and erroring otherwise.

        Static + dependency-free (pandas/numpy only) so it is unit-testable
        against a real export without a project.
        """
        missing = [c for c in _LONG_COLUMNS if c not in df_long.columns]
        if missing:
            raise BodypartColumnNotFoundError(
                msg=(
                    f"FreeDLC parquet is missing expected column(s) {missing}; "
                    f"found {list(df_long.columns)}. Expected long layout "
                    f"{list(_LONG_COLUMNS)}."
                ),
                source=source,
            )

        individuals = list(pd.unique(df_long["individual"]))
        if len(individuals) > 1:
            raise BodypartColumnNotFoundError(
                msg=(
                    f"FreeDLC parquet contains {len(individuals)} individuals "
                    f"({individuals}); this single-animal importer expects one. "
                    f"Use the multi-animal FreeDLC path."
                ),
                source=source,
            )

        df = df_long.copy()
        # -1.0 == "no detection" sentinel -> 0.0 confidence.
        df["likelihood"] = df["likelihood"].clip(lower=0.0)

        wide = df.pivot(index="frame", columns="bodypart",
                        values=["x", "y", "likelihood"])
        wide = wide.sort_index()
        # Guarantee a contiguous 0..max frame index.
        full_idx = range(int(wide.index.min()), int(wide.index.max()) + 1)
        wide = wide.reindex(full_idx)

        data_bps = list(wide["x"].columns)

        # ---- resolve data node -> project body-part order ----
        order = _resolve_bodypart_order(data_bps, project_bodyparts, source=source)

        out = pd.DataFrame(index=range(len(wide)))
        for proj_bp, data_bp in order:
            out[f"{proj_bp}_x"] = wide[("x", data_bp)].to_numpy()
            out[f"{proj_bp}_y"] = wide[("y", data_bp)].to_numpy()
            out[f"{proj_bp}_p"] = wide[("likelihood", data_bp)].to_numpy()
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Import every ``*.fdlc.parquet`` found in ``self.data_folder``."""
        import_log_rows = []
        mask_totals: dict[str, dict[str, int]] = {}
        for cnt, path in enumerate(self.input_data_paths):
            video_timer = SimbaTimer(start=True)
            video_name = self._video_name(path)
            print(f"Processing {video_name} ({cnt + 1}/{len(self.input_data_paths)})...")

            try:
                df_long = pd.read_parquet(path)
            except Exception as exc:
                raise BodypartColumnNotFoundError(
                    msg=f"Could not read {path}: {type(exc).__name__}: {exc}",
                    source=self.__class__.__name__,
                )

            df = self.long_to_wide(
                df_long, self.body_parts_lst, source=self.__class__.__name__,
            )

            # Column layout now equals self.bp_headers (project bp order,
            # x/y/p each) — the shape write_df's multi_idx_header path wants.
            if list(df.columns) != list(self.bp_headers):
                raise BodypartColumnNotFoundError(
                    msg=(
                        f"Body-part mismatch for {path}: produced "
                        f"{len(df.columns) // 3} body-parts, project expects "
                        f"{len(self.bp_headers) // 3}. Project body-parts are "
                        f"listed at {self.body_parts_path}."
                    ),
                    source=self.__class__.__name__,
                )

            if self.p_threshold > 0.0:
                from mufasa.pose_importers.likelihood_mask import (
                    apply_likelihood_threshold,
                    summarize_mask_counts,
                )
                df, counts = apply_likelihood_threshold(df, threshold=self.p_threshold)
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
                Interpolate(
                    config_path=self.config_path, data_path=save_path,
                    type=self.interpolation_settings["type"],
                    method=self.interpolation_settings["method"],
                    multi_index_df_headers=True,
                ).run()
            if self.smoothing_settings is not None:
                Smoothing(
                    config_path=self.config_path, data_path=save_path,
                    time_window=self.smoothing_settings["time_window"],
                    method=self.smoothing_settings["method"],
                    multi_index_df_headers=True,
                ).run()

            video_timer.stop_timer()
            total_masked = sum(mask_totals.get(video_name, {}).values())
            import_log_rows.append({
                "VIDEO": video_name,
                "IMPORT_TIME": video_timer.elapsed_time_str,
                "IMPORT_SOURCE": path,
                "P_THRESHOLD": self.p_threshold,
                "MASKED_POINTS": total_masked,
                "INTERPOLATION_SETTING": str(self.interpolation_settings),
                "SMOOTHING_SETTING": str(self.smoothing_settings),
            })
            stdout_success(msg=f"Video {video_name} data imported...",
                           elapsed_time=video_timer.elapsed_time_str)

        if import_log_rows:
            pd.DataFrame(import_log_rows).to_csv(self.import_log_path, index=False)

        self.timer.stop_timer()
        stdout_success(
            msg=f"All FreeDLC parquet data files imported to {self.input_csv_dir} directory",
            elapsed_time=self.timer.elapsed_time_str,
        )


def _resolve_bodypart_order(
    data_bps: list[str],
    project_bps: list[str],
    *,
    source: str,
) -> list[tuple[str, str]]:
    """Map project body-parts to data nodes, returning ordered
    ``(project_bp, data_bp)`` pairs. Precedence: exact name match ->
    case-insensitive -> positional (counts equal) -> error."""
    data_set, proj_set = set(data_bps), set(project_bps)

    if proj_set <= data_set:  # exact names present (extra data nodes tolerated)
        return [(bp, bp) for bp in project_bps]

    lower_map = {b.lower(): b for b in data_bps}
    if len(lower_map) == len(data_bps) and {b.lower() for b in project_bps} <= set(lower_map):
        return [(bp, lower_map[bp.lower()]) for bp in project_bps]

    if len(data_bps) == len(project_bps):
        # Positional fallback (H5 parity): trust the data's node order.
        return list(zip(project_bps, data_bps))

    missing = [b for b in project_bps if b not in data_set and b.lower() not in lower_map]
    raise BodypartColumnNotFoundError(
        msg=(
            f"FreeDLC nodes do not match project body-parts. Project expects "
            f"{len(project_bps)} ({project_bps}); data has {len(data_bps)} "
            f"({data_bps}). Missing from data: {missing}."
        ),
        source=source,
    )
