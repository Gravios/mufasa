__author__ = "Simon Nilsson; sronilsson@gmail.com"

from datetime import datetime
from tkinter import messagebox as mb

from mufasa.utils.enums import Defaults, TagNames
from mufasa.utils.printing import log_event

WINDOW_TITLE = "SIMBA ERROR"


class MufasaError(Exception):
    def __init__(self, msg: str, source: str = " ", show_window: bool = False):
        self.msg, self.source, self.show_window = msg, source, show_window
        self.print_and_log_error()

    def __str__(self):
        return self.msg

    def print_and_log_error(self):
        log_event(logger_name=f"{self.source}.{self.__class__.__name__}", log_type=TagNames.ERROR.value,msg=self.msg)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.msg}{Defaults.STR_SPLIT_DELIMITER.value}{TagNames.ERROR.value}")
        if self.show_window:
            mb.showerror(title=WINDOW_TITLE, message=self.msg)


class NoSpecifiedOutputError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = True):
        super().__init__(msg=msg, source=source, show_window=show_window)


class ROICoordinatesNotFoundError(MufasaError):
    def __init__(self, expected_file_path: str, source: str = "", show_window: bool = False):
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] SIMBA ROI COORDINATES ERROR: No ROI coordinates found. Please use the [ROI] tab to define ROIs. Expected at location {expected_file_path}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoChoosenClassifierError(MufasaError):
    def __init__(self, source: str = "", show_window: bool = False):
        msg = f"Select at least one classifier"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoChoosenROIError(MufasaError):
    def __init__(self, source: str = "", show_window: bool = False):
        msg = f"Please select at least one ROI."
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoChoosenMeasurementError(MufasaError):
    def __init__(self, source: str = "", show_window: bool = False):
        msg = "SIMBA NoChoosenMeasurementError ERROR: Please select at least one measurement to calculate descriptive statistics for."
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoDataError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"NO DATA ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class SamplingError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"SAMPLING ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class PermissionError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"PERMISSION ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoROIDataError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"NO ROI DATA ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class MixedMosaicError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"MixedMosaicError ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ClassifierInferenceError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"CLASSIFIER INFERENCE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class AnimalNumberError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"ANIMAL NUMBER ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidFilepathError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"INVALID FILE PATH ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NoFilesFoundError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"NO FILES FOUND ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class DataHeaderError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"DATA HEADER ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class NotDirectoryError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"NOT A DIRECTORY ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class DirectoryExistError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"DIRECTORY ALREADY EXIST ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FileExistError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"FILE EXIST ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FrameRangeError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"FRAME RANGE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class AdvancedLabellingError(MufasaError):
    def __init__(
        self,
        frame: str,
        lbl_lst: list,
        unlabel_lst: list,
        source: str = "",
        show_window: bool = False,
    ):
        msg = (
            "SIMBA ADVANCED LABELLING ERROR: In advanced labelling of multiple behaviors, any annotated frame cannot have some "
            "behaviors annotated as present/absent, while other behaviors are un-labelled. All behaviors need "
            "labels for a frame with one or more labels. In frame {}, behaviors {} are labelled, while behaviors "
            "{} are un-labelled.".format(str(frame), lbl_lst, unlabel_lst)
        )
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidHyperparametersFileError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"HYPERPARAMETER FILE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidVideoFileError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"VIDEO FILE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidFileTypeError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"INVALID FILE TYPE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FaultyTrainingSetError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"INVALID ML TRAINING SET ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class CountError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"COUNT ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FeatureNumberMismatchError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"FEATURE NUMBER MISMATCH ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class DuplicationError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"DUPLICATION ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class InvalidInputError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"VALUE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class IntegerError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"INTEGER ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class StringError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"STRING ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FloatError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"FLOAT ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class MissingProjectConfigEntryError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"MISSING PROJECT CONFIG ENTRY ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class MissingColumnsError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"MISSING COLUMN ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class CorruptedFileError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"READ FILE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ParametersFileError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"VIDEO PARAMETERS FILE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ColumnNotFoundError(MufasaError):
    def __init__(
        self,
        column_name: str,
        file_name: str,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"Field name {column_name} could not be found in file {file_name}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class BodypartColumnNotFoundError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"BODY_PART COLUMN NOT FOUND ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class AnnotationFileNotFoundError(MufasaError):
    def __init__(self, video_name: str, source: str = "", show_window: bool = False):
        msg = f"THIRD-PARTY ANNOTATION ERROR: NO ANNOTATION DATA FOR VIDEO {video_name} FOUND"
        super().__init__(msg=msg, source=source, show_window=show_window)


class DirectoryNotEmptyError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"DIRECTORY NOT EMPTY ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


#####


class ThirdPartyAnnotationFileNotFoundError(MufasaError):
    def __init__(self, video_name: str, source: str = "", show_window: bool = False):
        msg = f"Could not find file in project_folder/csv/features_extracted directory representing annotations for video {video_name}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsFpsConflictError(MufasaError):
    def __init__(
        self,
        video_name: str,
        annotation_fps: int,
        video_fps: int,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"THIRD-PARTY ANNOTATION ERROR: The FPS for video {video_name} is set to {str(video_fps)} in SimBA and {str(annotation_fps)} in the annotation file"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsMissingAnnotationsError(MufasaError):
    def __init__(
        self,
        video_name: str,
        clf_names: list,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"THIRD-PARTY ANNOTATION ERROR: No annotations detected for SimBA classifier(s) named {clf_names} for video {video_name}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationOverlapError(MufasaError):
    def __init__(
        self,
        video_name: str,
        clf_name: str,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"THIRD-PARTY ANNOTATION ERROR: The annotations for behavior {clf_name} in video {video_name} contains behavior start events that are initiated PRIOR to the PRECEDING behavior event ending. SimBA requires a specific behavior event to end before another behavior event can start."
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsAdditionalClfError(MufasaError):
    def __init__(
        self,
        video_name: str,
        clf_names: list,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"THIRD-PARTY ANNOTATION ERROR: Annotations file for video {video_name} has annotations for the following behaviors {clf_names} that are NOT classifiers named in the Mufasa project."
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationEventCountError(MufasaError):
    def __init__(
        self,
        video_name: str,
        clf_name: str,
        start_event_cnt: int,
        stop_event_cnt: int,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"THIRD-PARTY ANNOTATION ERROR: The annotations for behavior {clf_name} in video {video_name} contains {str(start_event_cnt)} start events and {str(stop_event_cnt)} stop events. SimBA requires the number of stop and start event counts to be equal."
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsClfMissingError(MufasaError):
    def __init__(
        self,
        video_name: str,
        clf_name: str,
        source: str = "",
        show_window: bool = False,
    ):
        msg = f"THIRD-PARTY ANNOTATION WARNING: No annotations detected for video {video_name} and behavior {clf_name}."
        super().__init__(msg=msg, source=source, show_window=show_window)


class ThirdPartyAnnotationsOutsidePoseEstimationDataError(MufasaError):
    def __init__(
        self,
        video_name: str,
        frm_cnt: int,
        clf_name: str or None = None,
        annotation_frms: int or None = None,
        first_error_frm: int or None = None,
        ambiguous_cnt: int or None = None,
        source: str = "",
        show_window: bool = False,
    ):

        if clf_name:
            msg = (
                f"SIMBA THIRD-PARTY ANNOTATION WARNING: Mufasa found THIRD-PARTY annotations for behavior {clf_name} in video "
                f"{video_name} that are annotated to occur at times which is not present in the "
                f"video data you imported into SIMBA. The video you imported to SimBA has {str(frm_cnt)} frames. "
                f"However, in BORIS, you have annotated {clf_name} to happen at frame number {str(first_error_frm)}. "
                f"These ambiguous annotations occur in {str(ambiguous_cnt)} different frames for video {video_name} that SimBA will **remove** by default. "
                f"Please make sure you imported the same video as you annotated in BORIS into SimBA and the video is registered with the correct frame rate."
            )
        else:
            msg = f"THIRD-PARTY ANNOTATION WARNING: The annotations for video {video_name} contain data for {str(annotation_frms)} frames. The pose-estimation features for the same video contain data for {str(frm_cnt)} frames."

        super().__init__(msg=msg, source=source, show_window=show_window)


class FFMPEGCodecGPUError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"FFMPEG CODEC ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class FFMPEGNotFoundError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"FFMPEG NOT FOUND ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class ArrayError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"ARRAY SIZE ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)

class ResolutionError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"RESOLUTION ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class MufasaModuleNotFoundError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"MODULE NOT FOUND ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class MufasaGPUError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"GPU ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class MufasaPackageVersionError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"PACKAGE VERSION ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)


class CropError(MufasaError):
    def __init__(self, msg: str, source: str = "", show_window: bool = False):
        msg = f"CROP ERROR: {msg}"
        super().__init__(msg=msg, source=source, show_window=show_window)

# Patch 122bl: backward-compat alias for the
# misspelled exception class name (SimBAPAckage… →
# SimBAPackage…). External callers importing the old
# name keep working; the canonical name is preferred.
SimBAPAckageVersionError = MufasaPackageVersionError

# Patch 122bo: backward-compat aliases for the renamed exception
# classes. The canonical names are MufasaError (base class) and
# MufasaGPUError / MufasaModuleNotFoundError /
# MufasaPackageVersionError (subclasses). External callers
# importing the old SimBA-prefixed names continue to work.
SimbaError = MufasaError
SimBAGPUError = MufasaGPUError
SimBAModuleNotFoundError = MufasaModuleNotFoundError
SimBAPackageVersionError = MufasaPackageVersionError
