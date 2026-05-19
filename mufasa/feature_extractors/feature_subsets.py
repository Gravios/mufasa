__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from itertools import combinations

from mufasa.mixins.config_reader import ConfigReader
from mufasa.mixins.train_model_mixin import TrainModelMixin
from mufasa.roi_tools.roi_utils import get_roi_dict_from_dfs
from mufasa.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_dir_exists,
    check_same_files_exist_in_all_directories, check_valid_boolean,
    check_valid_lst, check_video_has_rois)
from mufasa.utils.errors import (DuplicationError, InvalidInputError,
                                NoFilesFoundError, NoROIDataError)
from mufasa.utils.printing import stdout_success
from mufasa.utils.read_write import (copy_files_in_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, remove_a_folder,
                                    remove_multiple_folders, write_df)


def _read_columns_only(file_path: str, file_type: str) -> set:
    """Return the column names of a tabular file WITHOUT reading
    its full contents. Significantly faster than ``read_df`` for
    the preflight check, where we only care about column names.

    For 67 multi-MB CSV files, this drops the cost from ~minutes
    to ~milliseconds because we read only the header line instead
    of every row. For parquet, we read just the schema metadata
    (no row data at all).

    Falls back to read_df on unknown file types or if the
    fast path raises.
    """
    if file_type == "csv":
        # pd.read_csv with nrows=0 reads only the header row.
        try:
            import pandas as pd
            df = pd.read_csv(file_path, nrows=0)
            return set(df.columns)
        except Exception:
            # Some malformed CSVs trip pandas but not pyarrow.
            # Fall through to the safe slow path.
            pass
    elif file_type == "parquet":
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(file_path)
            return set(schema.names)
        except Exception:
            pass
    # Slow but always works.
    df = read_df(file_path=file_path, file_type=file_type)
    return set(df.columns)


# Constants below are kept (not moved to kernels module) because they
# define the public API for `feature_families` parameter values.

SHAPE_TYPE = "Shape_type"

# Feature family display strings. These are the public API — UI
# code passes them as-is into `feature_families=[...]`. They appear
# in saved CSV column suffixes and in user-facing log lines.
TWO_POINT_BP_DISTANCES = 'TWO-POINT BODY-PART DISTANCES (MM)'
WITHIN_ANIMAL_THREE_POINT_ANGLES = 'WITHIN-ANIMAL THREE-POINT BODY-PART ANGLES (DEGREES)'
WITHIN_ANIMAL_THREE_POINT_HULL = "WITHIN-ANIMAL THREE-POINT CONVEX HULL PERIMETERS (MM)"
WITHIN_ANIMAL_FOUR_POINT_HULL = "WITHIN-ANIMAL FOUR-POINT CONVEX HULL PERIMETERS (MM)"
ANIMAL_CONVEX_HULL_PERIMETER = 'ENTIRE ANIMAL CONVEX HULL PERIMETERS (MM)'
ANIMAL_CONVEX_HULL_AREA = "ENTIRE ANIMAL CONVEX HULL AREA (MM2)"
FRAME_BP_MOVEMENT = "FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)"
FRAME_BP_TO_ROI_CENTER = "FRAME-BY-FRAME BODY-PART DISTANCES TO ROI CENTERS (MM)"
FRAME_BP_INSIDE_ROI = "FRAME-BY-FRAME BODY-PARTS INSIDE ROIS (BOOLEAN)"
ARENA_EDGE = "BODY-PART DISTANCES TO VIDEO FRAME EDGE (MM)"


class FeatureFamily(Enum):
    """Enum form of the feature family identifiers.

    The Enum's *value* is the public display string (matching the
    legacy module-level constants like TWO_POINT_BP_DISTANCES).
    This means:

    - New code can use ``FeatureFamily.TWO_POINT_BP_DISTANCES`` for
      type-safe references.
    - Old code passing the string ``'TWO-POINT BODY-PART DISTANCES (MM)'``
      still works — the string IS the enum's value, so
      ``FeatureFamily('TWO-POINT BODY-PART DISTANCES (MM)')``
      resolves cleanly. The UI code that builds feature_families
      from string lists does not need to change.

    - Internal dispatch in process_one_video can compare against
      either form.

    Intentionally NOT changing the public string constants — they're
    imported by the UI forms and removing them would be a breaking
    change with no compensating benefit.
    """

    TWO_POINT_BP_DISTANCES = TWO_POINT_BP_DISTANCES
    WITHIN_ANIMAL_THREE_POINT_ANGLES = WITHIN_ANIMAL_THREE_POINT_ANGLES
    WITHIN_ANIMAL_THREE_POINT_HULL = WITHIN_ANIMAL_THREE_POINT_HULL
    WITHIN_ANIMAL_FOUR_POINT_HULL = WITHIN_ANIMAL_FOUR_POINT_HULL
    ANIMAL_CONVEX_HULL_PERIMETER = ANIMAL_CONVEX_HULL_PERIMETER
    ANIMAL_CONVEX_HULL_AREA = ANIMAL_CONVEX_HULL_AREA
    FRAME_BP_MOVEMENT = FRAME_BP_MOVEMENT
    FRAME_BP_TO_ROI_CENTER = FRAME_BP_TO_ROI_CENTER
    FRAME_BP_INSIDE_ROI = FRAME_BP_INSIDE_ROI
    ARENA_EDGE = ARENA_EDGE




FEATURE_FAMILIES = [TWO_POINT_BP_DISTANCES,
                    WITHIN_ANIMAL_THREE_POINT_ANGLES,
                    WITHIN_ANIMAL_THREE_POINT_HULL,
                    WITHIN_ANIMAL_FOUR_POINT_HULL,
                    ANIMAL_CONVEX_HULL_PERIMETER,
                    ANIMAL_CONVEX_HULL_AREA,
                    FRAME_BP_MOVEMENT,
                    FRAME_BP_TO_ROI_CENTER,
                    FRAME_BP_INSIDE_ROI,
                    ARENA_EDGE]


class FeatureSubsetsCalculator(ConfigReader, TrainModelMixin):
    """
    Computes a subset of features from pose for non-ML downstream purposes.
    E.g., returns the size of animal convex hull in each frame.

    :param str config_path: path to SimBA project config file in Configparser format
    :param str save_dir: directory where to store results.
    :param List[str] feature_family: List of feature subtype to calculate. E.g., ['TWO-POINT BODY-PART DISTANCES (MM)"].
    :param bool file_checks: If true, checks that the files which the data is appended too contains the anticipated number of rows and no duplicate columns after appending. Default False.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to save the data. If None, then the data is only appended.
    :param Optional[Union[str, os.PathLike]] data_dir: Directory of pose-estimation data to compute feature subsets for. If None, then the `/project_folder/csv/outlier_corrected_movement_locations` directory.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/mufasa/blob/master/docs/feature_subsets.md>_`

    .. image:: _static/img/feature_subsets.png
       :width: 400
       :align: center

    :example:
    >>> test = FeatureSubsetsCalculator(config_path=r"C:/troubleshooting/mitra/project_folder/project_config.ini",
    >>>                               feature_families=[FRAME_BP_MOVEMENT, WITHIN_ANIMAL_THREE_POINT_ANGLES],
    >>>                               file_checks=False,
    >>>                               save_dir=r"C:/troubleshooting/mitra/project_folder/csv/new_features")
    >>> test.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 feature_families: List[str],
                 file_checks: bool = False,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 derived_features_dir: Optional[
                     Union[str, os.PathLike]
                 ] = None,
                 n_workers: int = 1,
                 raise_on_error: bool = False,
                 overwrite_existing: bool = False):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        check_valid_boolean(value=file_checks, source=f'{self.__class__.__name__} file_checks', raise_error=True)
        # Patch 122an (B1): append_to_features_extracted +
        # append_to_targets_inserted kwargs removed. The pre-flight
        # checks they gated never produced an actual append write —
        # the writer was retargeted to per-family parquet in
        # 122ae-3 and these flags became inert. UI form fields in
        # features.py + subset_feature_extractor_pop_up.py are
        # dropped in the same patch.
        check_valid_boolean(value=raise_on_error, source=f'{self.__class__.__name__} raise_on_error', raise_error=True)
        check_valid_boolean(value=overwrite_existing, source=f'{self.__class__.__name__} overwrite_existing', raise_error=True)
        if not isinstance(n_workers, int) or n_workers < 1:
            raise InvalidInputError(
                msg=f'n_workers must be a positive integer, got {n_workers!r}',
                source=self.__class__.__name__,
            )
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir)
        # Patch 122ae-3: derived_features_dir gets created on-demand
        # by process_one_video (one subdir per family), so we don't
        # check_if_dir_exists here — just normalise to str-or-None
        # for serialisation into the worker config.
        if derived_features_dir is not None:
            derived_features_dir = str(derived_features_dir)
        check_valid_lst(data=feature_families, source=f'{self.__class__.__name__} feature_families', valid_dtypes=(str,), valid_values=FEATURE_FAMILIES, min_len=1, raise_error=True)
        self.file_checks, self.feature_families, self.save_dir = file_checks, feature_families, save_dir
        self.derived_features_dir = derived_features_dir
        self.n_workers = n_workers
        self.raise_on_error = raise_on_error
        self.overwrite_existing = overwrite_existing
        # Refuse to start without an explicit save destination. Pre-fix
        # behavior: if save_dir was None, run() would compute features
        # into temp_dir and then delete temp_dir at the end —
        # silently discarding hours of compute. The Qt form's
        # placeholder text "blank = project log dir" was misleading:
        # blank meant `None` meant discard. This check fails fast
        # before any work happens.
        #
        # Patch 122an (B1): the dead append_to_features_extracted /
        # append_to_targets_inserted clauses are gone from this guard
        # along with the kwargs themselves.
        if (self.save_dir is None
                and self.derived_features_dir is None):
            raise InvalidInputError(
                msg=(
                    "No save destination specified. Feature output "
                    "would be silently discarded after compute. "
                    "Pass one of:\n"
                    "  - derived_features_dir=<path>  (recommended, "
                    "writes per-family parquet to <path>/<family>/)\n"
                    "  - save_dir=<path>  (writes per-video wide "
                    "files to <path>)"
                ),
                source=self.__class__.__name__,
            )
        if data_dir is None:
            self.data_dir = self.outlier_corrected_dir
            self.data_paths = self.outlier_corrected_paths
        else:
            self.data_dir = data_dir
            check_if_dir_exists(in_dir=data_dir)
            self.data_paths  = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['csv'], raise_error=True)
        # Patch 122an (B1): legacy append-checks against
        # self.features_dir / self.targets_folder removed.
        self.video_names = [get_fn_ext(filepath=x)[1] for x in self.data_paths]
        for file_path in self.data_paths: check_file_exist_and_readable(file_path=file_path)
        # Heavy setup (temp_dir creation, ROI loading, video
        # filtering, bp combinations) is deferred to _setup_run() —
        # called by run() rather than the constructor. This makes
        # the class cheap to instantiate (e.g. for inspection or
        # testing) without committing to filesystem side effects.
        # Pre-step-4 behavior: __init__ created temp_data_<datetime>
        # directories on every instantiation, which accumulated as
        # cruft if run() was never called (Bug C in
        # AUDIT_feature_subsets_calculator.md).

    def _setup_run(self):
        """Per-run setup: create temp_dir, load ROI data if needed,
        filter videos with missing ROIs, build body-part combinations.

        Called by run() before the per-video loop. Idempotent within
        a single run; not safe to call concurrently from two
        instances against the same data_dir (the temp_dir uses the
        instance's datetime, so two calls in the same second from
        the same process could collide — unlikely in practice).
        """
        self.temp_dir = os.path.join(self.data_dir, f"temp_data_{self.datetime}")
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)
        # Announce the save destination(s) upfront so the user can
        # see them BEFORE the run starts (and before potentially
        # hours of compute). Pre-fix behavior: the only output line
        # mentioning destinations was "Storing new features in X..."
        # at the very end of run(), after all the compute had already
        # happened.
        destinations = []
        if self.save_dir is not None:
            destinations.append(f"save_dir → {self.save_dir}")
        if self.derived_features_dir is not None:
            destinations.append(
                f"per-family parquet → {self.derived_features_dir}/<family>/"
            )
        print(
            f"Feature subsets will be written to:\n  - "
            + "\n  - ".join(destinations)
        )
        print(f"  (intermediate work: {self.temp_dir})")
        if (FRAME_BP_TO_ROI_CENTER in self.feature_families) or (FRAME_BP_INSIDE_ROI in self.feature_families):
            if not os.path.isfile(self.roi_coordinates_path):
                raise NoROIDataError(msg=f'Cannot compute ROI features: The Mufasa project has no ROI data defined.')
            self.read_roi_data()
            # Identify videos that are missing some of the ROIs the
            # feature pipeline expects. Default behavior of
            # check_video_has_rois unions every ROI name across every
            # labeled video — so if you defined "platform" on one
            # video, every other video is required to have "platform"
            # too. That's strict-by-default but matches how downstream
            # ROI-feature columns are laid out (one column per
            # (video × ROI) pair).
            #
            # Earlier versions raised NoROIDataError on the first
            # missing combination, dumping a giant dict to the user's
            # screen and aborting the whole 67-video run. The honest
            # research workflow is: label ROIs on the videos you care
            # about, run feature extraction on those, leave the rest
            # for later. So now we collect the missing list, skip
            # those videos, log what got skipped, and proceed with
            # the videos that have complete ROI coverage.
            check_result = check_video_has_rois(roi_dict=self.roi_dict,
                                                  raise_error=False)
            videos_with_complete_rois: set[str] = set()
            videos_skipped: dict[str, list[str]] = {}
            if check_result is True:
                videos_with_complete_rois = set(self.video_names)
            else:
                # check_video_has_rois returned (False, missing_rois)
                _, missing_rois = check_result
                for vname, missing_list in missing_rois.items():
                    if not missing_list:
                        videos_with_complete_rois.add(vname)
                    else:
                        videos_skipped[vname] = list(missing_list)

            self.roi_dict = get_roi_dict_from_dfs(rectangle_df=self.rectangles_df, circle_df=self.circles_df, polygon_df=self.polygon_df, video_name_nesting=True)
            # Also catch videos that have ZERO ROIs (don't appear in
            # the ROI dataframe at all). check_video_has_rois only
            # reports videos that ARE in the dataframe.
            videos_with_no_rois_at_all = [
                v for v in self.video_names
                if v not in self.roi_dict
            ]
            for v in videos_with_no_rois_at_all:
                videos_skipped[v] = ["(no ROIs defined for this video)"]

            # Filter data_paths / video_names down to the videos that
            # have complete ROI coverage. Keep both lists in sync.
            target_video_names = [
                v for v in self.video_names
                if v in self.roi_dict and v in videos_with_complete_rois
            ]
            if not target_video_names:
                raise NoROIDataError(
                    msg=(
                        "Cannot compute ROI features: none of the "
                        f"{len(self.video_names)} videos have all "
                        "required ROIs defined. Define ROIs on at "
                        "least one video before running ROI features."
                    ),
                    source=self.__class__.__name__,
                )

            # Surface the skip decisions: stdout for the workbench
            # log, plus a CSV in the temp dir for permanent record.
            if videos_skipped:
                n_skip = len(videos_skipped)
                n_run = len(target_video_names)
                print(
                    f"WARNING: ROI feature extraction will SKIP "
                    f"{n_skip} of {len(self.video_names)} videos that "
                    f"are missing one or more required ROIs; "
                    f"continuing with {n_run} videos that have "
                    f"complete ROI coverage."
                )
                # Write a CSV so the user can see exactly what got
                # skipped and which ROIs were missing for each.
                try:
                    import csv
                    skip_log_path = os.path.join(
                        self.temp_dir, "skipped_videos_missing_rois.csv",
                    )
                    with open(skip_log_path, "w", newline="") as fh:
                        writer = csv.writer(fh)
                        writer.writerow(["video", "missing_rois"])
                        for v, missing in sorted(videos_skipped.items()):
                            writer.writerow([v, ";".join(missing)])
                    print(f"Skipped-video details: {skip_log_path}")
                except Exception as exc:
                    # Don't let a logging failure abort the run.
                    print(f"(Could not write skip log: "
                          f"{type(exc).__name__}: {exc})")

            # Apply the filter to data_paths and video_names so the
            # main run() loop only processes the kept videos.
            kept = set(target_video_names)
            self.data_paths = [
                p for p in self.data_paths
                if get_fn_ext(filepath=p)[1] in kept
            ]
            self.video_names = target_video_names
        self.__get_bp_combinations()

    def __get_bp_combinations(self):
        ordered_bps = sorted(self.project_bps, key=str.lower)
        self.two_point_combs = np.array(list(combinations(ordered_bps, 2)))
        self.within_animal_three_point_combs = {}
        self.within_animal_four_point_combs = {}
        self.animal_bps = {}
        for animal, animal_data in self.animal_bp_dict.items():
            animal_bps = [x[:-2] for x in animal_data["X_bps"]]
            self.animal_bps[animal] = animal_bps
            self.within_animal_three_point_combs[animal] = np.array(list(combinations(animal_bps, 3)))
            self.within_animal_four_point_combs[animal] = np.array(list(combinations(animal_bps, 4)))


    def __check_files(self, x: pd.DataFrame, y: pd.DataFrame, path_x: str, path_y: str):
        if len(x) != len(y):
            remove_multiple_folders(folders=[self.temp_append_dir, self.temp_dir], raise_error=False)
            raise InvalidInputError(msg=f'The files at {path_x} and {path_y} do not contain the same number of rows: {len(x)} vs {len(y)}', source=self.__class__.__name__)
        duplicated_x_cols = [i for i in x.columns if i in y.columns]
        if len(duplicated_x_cols) > 0 and not self.overwrite_existing:
            # Hard fail unless the caller explicitly opted in to
            # overwrite. Pre-fix this raised unconditionally — which
            # was correct (silent overwrite is bad) but caused users
            # to lose hours of compute when they hit a column
            # collision at the very end of a run. The new
            # preflight_check (called at run() start) catches this
            # case BEFORE compute, with a clear error message that
            # tells the user to either pass overwrite_existing=True
            # or pick different feature families.
            remove_multiple_folders(folders=[self.temp_append_dir, self.temp_dir], raise_error=False)
            raise DuplicationError(msg=f'Cannot append the new features to {path_y}. This file already has the following columns: {duplicated_x_cols}', source=self.__class__.__name__)
        # If we DO have overlap and overwrite is allowed, the caller
        # of __check_files (the append helpers below) is responsible
        # for actually dropping the conflicting columns from y before
        # the concat. We don't mutate y here because pandas semantics
        # (column-name duplicates in concat result) are clearer when
        # the drop is explicit at the call site.

    def __append_to_data_in_dir(self, dir: Union[str, os.PathLike]):
        temp_files = find_files_of_filetypes_in_directory(directory=self.temp_dir, extensions=[f'.{self.file_type}'], as_dict=True)
        self.temp_append_dir = os.path.join(dir, f'temp_{self.datetime}')
        os.makedirs(self.temp_append_dir)
        for file_cnt, (file_name, file_path) in enumerate(temp_files.items()):
            print(f'Appending features to {file_name} ({file_cnt+1}/{len(list(temp_files.keys()))})')
            old_df = read_df(file_path=os.path.join(dir, f'{file_name}.{self.file_type}'), file_type=self.file_type).reset_index(drop=True)
            new_features_df = read_df(file_path=file_path, file_type=self.file_type).reset_index(drop=True)
            if self.file_checks:
                self.__check_files(x=new_features_df, y=old_df, path_x=file_path, path_y=os.path.join(dir, f'{file_name}.{self.file_type}'))
            # When overwrite_existing is set, drop any columns from
            # the existing file that the new features will replace.
            # Without this, pd.concat would produce duplicate column
            # names in the output DataFrame, which downstream code
            # handles unpredictably.
            if self.overwrite_existing:
                conflicts = [c for c in new_features_df.columns if c in old_df.columns]
                if conflicts:
                    print(f'  Overwriting {len(conflicts)} existing column(s): {conflicts}')
                    old_df = old_df.drop(columns=conflicts)
            save_path = os.path.join(self.temp_append_dir, f'{file_name}.{self.file_type}')
            out_df = pd.concat([old_df, new_features_df], axis=1)
            write_df(df=out_df, file_type=self.file_type, save_path=save_path)
        prior_dir = os.path.join(dir, f"Prior_to_feature_subset_append_{self.datetime}")
        os.makedirs(prior_dir)
        copy_files_in_directory(in_dir=dir, out_dir=prior_dir, filetype=self.file_type, raise_error=True)
        copy_files_in_directory(in_dir=self.temp_append_dir, out_dir=dir, filetype=self.file_type, raise_error=True)
        remove_a_folder(folder_dir=self.temp_append_dir, ignore_errors=False)

    def __append_to_targets_inserted(self, dir: Union[str, os.PathLike]):
        temp_files = find_files_of_filetypes_in_directory(directory=self.temp_dir, extensions=[f'.{self.file_type}'], as_dict=True)
        self.temp_append_dir = os.path.join(dir, f'temp_{self.datetime}')
        os.makedirs(self.temp_append_dir)
        for file_cnt, (file_name, file_path) in enumerate(temp_files.items()):
            old_df = read_df(file_path=os.path.join(dir, f'{file_name}.{self.file_type}'), file_type=self.file_type).reset_index(drop=True)
            new_features_df = read_df(file_path=file_path, file_type=self.file_type).reset_index(drop=True)
            if self.file_checks:
                self.__check_files(x=new_features_df, y=old_df, path_x=file_path, path_y=os.path.join(dir, f'{file_name}.{self.file_type}'))
            # Drop conflicting columns from the existing file when
            # overwrite_existing is set. Same rationale as in
            # __append_to_data_in_dir; see comment there.
            if self.overwrite_existing:
                conflicts = [c for c in new_features_df.columns if c in old_df.columns]
                if conflicts:
                    print(f'  Overwriting {len(conflicts)} existing column(s): {conflicts}')
                    old_df = old_df.drop(columns=conflicts)
            save_path = os.path.join(self.temp_append_dir, f'{file_name}.{self.file_type}')
            clf_cols = [x for x in self.clf_names if x in list(old_df.columns)]
            clf_df, old_df = old_df[clf_cols], old_df.drop(clf_cols, axis=1)
            out_df = pd.concat([old_df, new_features_df, clf_df], axis=1)
            write_df(df=out_df, file_type=self.file_type, save_path=save_path)
        prior_dir = os.path.join(dir, f"Prior_to_feature_subset_append_{self.datetime}")
        os.makedirs(prior_dir)
        copy_files_in_directory(in_dir=dir, out_dir=prior_dir, filetype=self.file_type, raise_error=True)
        copy_files_in_directory(in_dir=self.temp_append_dir, out_dir=dir, filetype=self.file_type, raise_error=True)
        remove_a_folder(folder_dir=self.temp_append_dir, ignore_errors=False)

    def _build_video_processing_config(self):
        """Build a VideoProcessingConfig from the current self state.

        Used by run() to hand off per-video work to process_one_video.
        Single source of truth for what state crosses the (currently
        in-process; in step 6, cross-process) boundary into the
        per-video worker.
        """
        from mufasa.feature_extractors.feature_subset_orchestration import (
            VideoProcessingConfig,
        )
        return VideoProcessingConfig(
            feature_families=list(self.feature_families),
            file_type=self.file_type,
            temp_dir=self.temp_dir,
            two_point_combs=self.two_point_combs,
            within_animal_three_point_combs=self.within_animal_three_point_combs,
            within_animal_four_point_combs=self.within_animal_four_point_combs,
            animal_bps=self.animal_bps,
            roi_dict=getattr(self, "roi_dict", None) if (
                FRAME_BP_TO_ROI_CENTER in self.feature_families
                or FRAME_BP_INSIDE_ROI in self.feature_families
            ) else None,
            video_info_df=self.video_info_df,
            # Patch 122ae-3: per-family parquet write destination.
            # None disables the new branch; legacy callers see no
            # change.
            derived_features_dir=self.derived_features_dir,
        )

    def preflight_check(self) -> Dict[str, List[str]]:
        """Discover whether the planned run would clobber existing
        output. Returns a dict mapping
        ``destination/filename → [reason_strings]``
        for every collision. An empty dict means no conflicts.

        Detects ``save_dir`` filename collisions: when ``save_dir``
        is set and files with the same names as the videos in
        this run already exist in that directory. Pre-fix
        behavior: ``copy_files_in_directory`` (via
        ``shutil.copy``) silently overwrote these. Reason
        string is ``"file exists"``.

        Patch 122an (B1): the append-mode column-collision probe
        is gone. It scanned ``features_extracted/`` and
        ``targets_inserted/`` for column overlaps, but the
        ``append_to_*`` kwargs that triggered it were removed
        from the constructor — v1 writes go to per-family
        parquet under ``derived_features_dir`` where collisions
        are structurally impossible (each family lives in its
        own subdir).

        Returns dict instead of raising so the caller can prompt
        the user before deciding whether to set
        ``overwrite_existing=True`` and re-run.
        """
        conflicts: Dict[str, List[str]] = {}

        # Diagnostic: emits a single line on stderr/stdout when
        # preflight is actually invoked. Distinguishes "preflight
        # ran but found nothing" from "preflight wasn't called at
        # all" (e.g. stale __pycache__, on_run override missing).
        # If you don't see this line in the console, preflight
        # didn't run.
        print(
            f"[preflight] starting check: "
            f"save_dir={'set' if self.save_dir else 'unset'}, "
            f"derived_features_dir={'set' if self.derived_features_dir else 'unset'}"
        )

        # ------------------------------------------------------------ #
        # Pre-check 1: save_dir filename collisions
        # ------------------------------------------------------------ #
        # Cheap — just stat per video filename. Do this FIRST so we
        # detect this case even when only save_dir is set (skipping
        # the more expensive column-discovery probe below).
        if self.save_dir is not None and os.path.isdir(self.save_dir):
            # Need data_paths to know the filenames. Run setup first
            # if not done.
            if not hasattr(self, 'data_paths'):
                self._setup_run()
            for file_path in self.data_paths:
                video_name = get_fn_ext(filepath=file_path)[1]
                target_path = os.path.join(
                    self.save_dir, f'{video_name}.{self.file_type}',
                )
                if os.path.isfile(target_path):
                    conflicts[f'save_dir/{video_name}'] = ['file exists']
            n_save_dir_collisions = sum(
                1 for k in conflicts if k.startswith('save_dir/')
            )
            print(
                f"[preflight] save_dir check: "
                f"{n_save_dir_collisions} file(s) already exist in "
                f"{self.save_dir}"
            )

        # Patch 122an (B1): the append-mode column-collision probe
        # (which scanned features_extracted/ and targets_inserted/
        # for column overlaps) is gone. It depended on the
        # append_to_* kwargs which are no longer accepted; v1
        # writes go to derived/features/<family>/<video>.parquet
        # under the user-supplied derived_features_dir, where the
        # per-family layout makes collisions structurally
        # impossible (each family lives in its own subdir).

        return conflicts

    def _run_sequential(self, config, process_one_video):
        """Run the per-video loop in the current process.

        This is the path used when n_workers <= 1 (default).
        Behavior is byte-equivalent to pre-step-6 code: each video
        is processed in order, prints interleave naturally,
        exceptions abort the loop (and bubble up).
        """
        for file_cnt, file_path in enumerate(self.data_paths):
            process_one_video(
                file_path=file_path, config=config,
                file_idx=file_cnt, n_total_files=len(self.data_paths),
                print_progress=True,
            )

    def _run_parallel(self, config, process_one_video):
        """Run the per-video loop across worker processes.

        Each worker receives the same VideoProcessingConfig (which
        is small, ~few KB serialized — not a concern). Workers
        compute features for their assigned video and write the
        result to ``config.temp_dir`` exactly as the sequential
        version does. Output files are byte-identical to the
        sequential path.

        Failure handling depends on self.raise_on_error:
        * raise_on_error=False (default): per-video failures are
          logged to stderr but do not abort the batch. The batch
          completes for all videos that succeeded; failed videos
          have no output file in temp_dir. A summary is printed
          at the end.
        * raise_on_error=True: the first failure cancels remaining
          futures and re-raises the exception in the main process.

        Worker count rationale: defaults to whatever the user
        passed to n_workers. On Ryzen 9800X3D (8 physical cores,
        16 SMT threads), a good value is 6-8 (leaves headroom for
        the OS, IO, and the main process). Going above n_workers=
        physical_cores tends to hurt because the kernels are
        compute-bound and SMT threads contend.

        Numba JIT cache: each worker re-JITs feature kernels on
        first use. Cost is ~5-10 seconds per worker, paid once.
        For the user's typical 67-video batch (~4 hours sequential)
        this is amortized to invisibility.
        """
        # Local import keeps the cost out of the import path for
        # callers who only ever use sequential mode.
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_videos = len(self.data_paths)
        if n_videos == 0:
            # Patch 122d7 (QWI-3): without this short-circuit,
            # `ProcessPoolExecutor(max_workers=0)` below would
            # raise `ValueError: max_workers must be greater than
            # 0`. Empty-project state (no eligible videos) is a
            # legitimate input — common right after Data Import
            # before any videos have been added. Return cleanly
            # so the caller can decide what to surface.
            print(
                "Feature extraction: no eligible videos in this "
                "project. Nothing to do."
            )
            return
        # Belt-and-braces: clamp to ≥ 1 even though the
        # short-circuit above should make this unreachable for the
        # known n_videos==0 path. Defensive against future paths
        # that might call _run_parallel with self.n_workers==0.
        n_workers = max(1, min(self.n_workers, n_videos))
        print(
            f"Running feature extraction on {n_videos} videos "
            f"across {n_workers} workers..."
        )
        failures: list[tuple[str, str]] = []  # (video_name, error_msg)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    process_one_video,
                    file_path,
                    config,
                    idx,
                    n_videos,
                    False,  # print_progress=False to avoid interleaved output
                ): (idx, file_path)
                for idx, file_path in enumerate(self.data_paths)
            }
            n_done = 0
            for fut in as_completed(futures):
                idx, file_path = futures[fut]
                video_name = get_fn_ext(filepath=file_path)[1]
                try:
                    fut.result()
                    n_done += 1
                    print(
                        f"  [{n_done}/{n_videos}] ✓ {video_name}"
                    )
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    failures.append((video_name, msg))
                    print(
                        f"  [{n_done}/{n_videos}] ✗ {video_name}: "
                        f"{msg}"
                    )
                    if self.raise_on_error:
                        # Cancel pending and re-raise. Note: as_completed
                        # in conjunction with the context manager will
                        # wait for in-flight workers to finish before
                        # re-raising the exception in the outer scope.
                        for f in futures:
                            f.cancel()
                        raise
        if failures:
            print(
                f"\n{len(failures)} of {n_videos} videos FAILED "
                f"during parallel feature extraction:"
            )
            for video_name, msg in failures:
                print(f"  - {video_name}: {msg}")
            print(
                f"Output files for the {n_videos - len(failures)} "
                f"successful videos are in {self.temp_dir}."
            )
            # Surface as a hard error if every video failed —
            # likely indicates a systematic problem rather than
            # per-video flakes.
            if len(failures) == n_videos:
                raise RuntimeError(
                    "All videos failed during parallel feature "
                    "extraction. See per-video errors above."
                )

    def run(self):
        from mufasa.feature_extractors.feature_subset_orchestration import (
            process_one_video,
        )
        # Heavy setup deferred from __init__ — see _setup_run docstring.
        # Idempotent within a single run.
        self._setup_run()
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)

        # Preflight: detect collisions in destinations BEFORE doing
        # the multi-hour compute. If conflicts exist and the caller
        # hasn't opted into overwrite, fail fast with a clear error
        # message. The Qt form catches this earlier (synchronous
        # preflight + confirm dialog), but headless callers (scripts,
        # tests) get the same protection here.
        #
        # Two collision types are detected (see preflight_check):
        # 1. save_dir/<video> = ['file exists'] — pre-fix this was
        #    silently overwritten by shutil.copy
        # 2. <append_dest>/<video> = [list of column names] — pre-fix
        #    this would explode at the END of the run via
        #    __check_files DuplicationError
        if not self.overwrite_existing:
            conflicts = self.preflight_check()
            if conflicts:
                sample_file = next(iter(conflicts))
                sample_reasons = conflicts[sample_file]
                # Categorize: file-exists (save_dir) vs columns (append)
                save_dir_collisions = [
                    f for f, r in conflicts.items()
                    if r == ['file exists']
                ]
                column_collisions = {
                    f: r for f, r in conflicts.items()
                    if r != ['file exists']
                }
                # Build adaptive message.
                summary_parts = []
                if save_dir_collisions:
                    summary_parts.append(
                        f"{len(save_dir_collisions)} file(s) already "
                        f"exist in save_dir (would be silently "
                        f"overwritten by shutil.copy)"
                    )
                if column_collisions:
                    n_files = len(column_collisions)
                    sample = next(iter(column_collisions))
                    sample_cols = column_collisions[sample]
                    cols_preview = sample_cols[:5]
                    more = ""
                    if len(sample_cols) > 5:
                        more = f" (and {len(sample_cols) - 5} more)"
                    summary_parts.append(
                        f"{n_files} append-destination file(s) "
                        f"already contain feature columns this run "
                        f"would produce; e.g. {sample} has columns "
                        f"{cols_preview}{more}"
                    )
                # Clean up temp_dir from _setup_run since we're
                # bailing out before the real compute starts.
                try:
                    remove_a_folder(folder_dir=self.temp_dir, ignore_errors=True)
                except Exception:
                    pass
                raise DuplicationError(
                    msg=(
                        "Run would overwrite existing output:\n  - "
                        + "\n  - ".join(summary_parts)
                        + "\n\nTo proceed, either:\n"
                        + "  1. Pass overwrite_existing=True to "
                        + "overwrite\n"
                        + "  2. Pick different feature families that "
                        + "don't overlap (for column collisions)\n"
                        + "  3. Use save_dir=<new path> to write to a "
                        + "fresh directory instead"
                    ),
                    source=self.__class__.__name__,
                )

        config = self._build_video_processing_config()
        if self.n_workers <= 1:
            self._run_sequential(config=config, process_one_video=process_one_video)
        else:
            self._run_parallel(config=config, process_one_video=process_one_video)

        # Track whether output made it to its final destination.
        # If any destination step failed, we must NOT delete temp_dir
        # — that would silently destroy hours of compute. Instead,
        # leave temp_dir on disk and tell the user where to find it
        # for manual recovery.
        output_persisted = False
        save_errors: list[str] = []

        # Patch 122an (B1): the two legacy append branches
        # (self.append_to_features_extracted → __append_to_data_in_dir
        # against features_dir, and self.append_to_targets_inserted
        # → __append_to_targets_inserted against targets_folder)
        # are gone. The kwargs no longer exist and the underlying
        # private append methods are unused — they're left in
        # place as private utilities in case future code needs
        # them, but no public code path calls them anymore.

        if self.save_dir is not None:
            print(f"Storing new features in {self.save_dir}...")
            try:
                copy_files_in_directory(in_dir=self.temp_dir, out_dir=self.save_dir, filetype=self.file_type, raise_error=True)
                output_persisted = True
            except Exception as exc:
                save_errors.append(
                    f"copy to save_dir failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        # Cleanup logic. ONLY delete temp_dir if at least one
        # destination succeeded. If all destinations failed, keep
        # temp_dir so the user can recover manually.
        if output_persisted:
            try:
                remove_a_folder(folder_dir=self.temp_dir, ignore_errors=False)
            except Exception as exc:
                # Cleanup failed but compute completed and at least
                # one save succeeded. Don't bubble up — the data is
                # already safe at the destination(s). Just inform.
                print(
                    f"Note: failed to remove temp directory "
                    f"{self.temp_dir} ({type(exc).__name__}: {exc}). "
                    f"Output was successfully written; you may delete "
                    f"the temp directory manually."
                )
        if save_errors:
            print(
                "\nERROR: one or more save destinations failed:\n  - "
                + "\n  - ".join(save_errors)
            )
            print(
                f"\nIntermediate output preserved in: {self.temp_dir}"
            )
            print(
                "Recover with: "
                f"cp {self.temp_dir}/*.{self.file_type} "
                f"<your-target-dir>/"
            )
            # Surface as an exception so callers (UI form, scripts)
            # know the run did not complete its save phase. compute
            # is intact in temp_dir; this isn't catastrophic but the
            # user must know.
            raise RuntimeError(
                f"Feature extraction compute completed but {len(save_errors)} "
                f"save destination(s) failed. See messages above. "
                f"Output preserved in {self.temp_dir}."
            )
        self.timer.stop_timer()
        stdout_success(msg="Feature sub-sets calculations complete!", elapsed_time=self.timer.elapsed_time_str)
