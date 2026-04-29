__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from enum import Enum
from typing import List, Optional, Union

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
    :param bool append_to_features_extracted: If True, appends the data to the file sin the `features_extracted` directory. Default: False.
    :param bool append_to_targets_inserted: If True, appends the data to the file sin the `targets_inserted` directory. Default: False.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/mufasa/blob/master/docs/feature_subsets.md>_`

    .. image:: _static/img/feature_subsets.png
       :width: 400
       :align: center

    :example:
    >>> test = FeatureSubsetsCalculator(config_path=r"C:/troubleshooting/mitra/project_folder/project_config.ini",
    >>>                               feature_families=[FRAME_BP_MOVEMENT, WITHIN_ANIMAL_THREE_POINT_ANGLES],
    >>>                               append_to_features_extracted=False,
    >>>                               file_checks=False,
    >>>                               append_to_targets_inserted=False,
    >>>                               save_dir=r"C:/troubleshooting/mitra/project_folder/csv/new_features")
    >>> test.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 feature_families: List[str],
                 file_checks: bool = False,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 append_to_features_extracted: bool = False,
                 append_to_targets_inserted: bool = False):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        check_valid_boolean(value=file_checks, source=f'{self.__class__.__name__} file_checks', raise_error=True)
        check_valid_boolean(value=append_to_features_extracted, source=f'{self.__class__.__name__} append_to_features_extracted', raise_error=True)
        check_valid_boolean(value=append_to_targets_inserted, source=f'{self.__class__.__name__} append_to_targets_inserted', raise_error=True)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir)
        check_valid_lst(data=feature_families, source=f'{self.__class__.__name__} feature_families', valid_dtypes=(str,), valid_values=FEATURE_FAMILIES, min_len=1, raise_error=True)
        self.file_checks, self.feature_families, self.save_dir = file_checks, feature_families, save_dir
        self.append_to_features_extracted = append_to_features_extracted
        self.append_to_targets_inserted = append_to_targets_inserted
        if data_dir is None:
            self.data_dir = self.outlier_corrected_dir
            self.data_paths = self.outlier_corrected_paths
        else:
            self.data_dir = data_dir
            check_if_dir_exists(in_dir=data_dir)
            self.data_paths  = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['csv'], raise_error=True)
        if self.append_to_features_extracted:
            if not check_same_files_exist_in_all_directories(dirs=[self.data_dir, self.features_dir], file_type=self.file_type, raise_error=False):
                raise NoFilesFoundError(msg=f'Cannot append feature subset to files in {self.features_dir} directory: To proceed, the files in the {self.features_dir} and the {self.data_dir} directories has to contain the same number of files with the same filenames.', source=self.__class__.__name__)
        if self.append_to_targets_inserted:
            if not check_same_files_exist_in_all_directories(dirs=[self.data_dir, self.targets_folder], file_type=self.file_type, raise_error=False):
                raise NoFilesFoundError(msg=f'Cannot append feature subset to files in {self.targets_folder} directory: To proceed, the files in the {self.targets_folder} and the {self.data_dir} directories has to contain the same number of files with the same filenames.', source=self.__class__.__name__)
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
        if (FRAME_BP_TO_ROI_CENTER in self.feature_families) or (FRAME_BP_INSIDE_ROI in self.feature_families):
            if not os.path.isfile(self.roi_coordinates_path):
                raise NoROIDataError(msg=f'Cannot compute ROI features: The SimBA project has no ROI data defined.')
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
        if len(duplicated_x_cols) > 0:
            remove_multiple_folders(folders=[self.temp_append_dir, self.temp_dir], raise_error=False)
            raise DuplicationError(msg=f'Cannot append the new features to {path_y}. This file already has the following columns: {duplicated_x_cols}', source=self.__class__.__name__)

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
        )

    def run(self):
        from mufasa.feature_extractors.feature_subset_orchestration import (
            process_one_video,
        )
        # Heavy setup deferred from __init__ — see _setup_run docstring.
        # Idempotent within a single run.
        self._setup_run()
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        config = self._build_video_processing_config()
        for file_cnt, file_path in enumerate(self.data_paths):
            # Per-video work is now in a module-level function. The
            # class no longer holds intermediate state (self.data_df,
            # self.results, self.video_info, etc.) during the loop —
            # those live on the local stack of process_one_video.
            #
            # Step 6 of the refactor will wrap this loop in a
            # ProcessPoolExecutor for parallel execution.
            process_one_video(
                file_path=file_path, config=config,
                file_idx=file_cnt, n_total_files=len(self.data_paths),
                print_progress=True,
            )
        if self.append_to_features_extracted:
            print(f'Appending new feature to files in {self.features_dir}...')
            self.__append_to_data_in_dir(dir=self.features_dir)
        if self.append_to_targets_inserted:
            print(f'Appending new feature to files in {self.targets_folder}...')
            self.__append_to_targets_inserted(dir=self.targets_folder)
        if self.save_dir is not None:
            print(f"Storing new features in {self.save_dir}...")
            copy_files_in_directory(in_dir=self.temp_dir, out_dir=self.save_dir, filetype=self.file_type, raise_error=True)
        remove_a_folder(folder_dir=self.temp_dir, ignore_errors=False)
        self.timer.stop_timer()
        stdout_success(msg="Feature sub-sets calculations complete!", elapsed_time=self.timer.elapsed_time_str)



# test = FeatureSubsetsCalculator(config_path=r"C:\troubleshooting\srami0619\project_folder\project_config.ini",
#                                 feature_families=[TWO_POINT_BP_DISTANCES],
#                                 append_to_features_extracted=False,
#                                 file_checks=True,
#                                 append_to_targets_inserted=False)
# test.run()



# test = FeatureSubsetsCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                 feature_families=[TWO_POINT_BP_DISTANCES],
#                                 append_to_features_extracted=False,
#                                 file_checks=True,
#                                 append_to_targets_inserted=False)
# test.run()

#
# test = FeatureSubsetsCalculator(config_path=r"D:\Stretch\Stretch\project_folder\project_config.ini",
#                                 feature_families=[TWO_POINT_BP_DISTANCES],
#                                 append_to_features_extracted=True,
#                                 file_checks=True,
#                                 append_to_targets_inserted=True,
#                                 save_dir=r"D:\Stretch\Stretch\project_folder\new_features")
# test.run()



# test = FeatureSubsetsCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                 feature_families=[TWO_POINT_BP_DISTANCES],
#                                 append_to_features_extracted=False,
#                                 file_checks=False,
#                                 append_to_targets_inserted=False,
#                                 save_dir=r"C:\troubleshooting\mitra\project_folder\csv\new_features")
# test.run()



# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_family='Frame-by-frame body-parts inside ROIs (Boolean)',
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data')
# test.run()
#
#
# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_family='Frame-by-frame body-part movements (mm)',
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data')
# test.run()
#
#
# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_families=['Frame-by-frame body-part distances to ROI centers (mm)', 'Frame-by-frame body-parts inside ROIs (Boolean)'],
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data',
#                                 include_file_checks=True,
#                                 append_to_features_extracted=False,
#                                 append_to_targets_inserted=True)
# test.run()
