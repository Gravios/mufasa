__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from copy import deepcopy

import pandas as pd

from mufasa.mixins.config_reader import ConfigReader
from mufasa.mixins.train_model_mixin import TrainModelMixin
from mufasa.utils.enums import TagNames
from mufasa.utils.errors import NoFilesFoundError
from mufasa.utils.printing import SimbaTimer, log_event, stdout_success
from mufasa.utils.read_write import get_fn_ext, read_df


class InferenceMulticlassBatch(TrainModelMixin, ConfigReader):

    def __init__(self, config_path: str):

        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        log_event(
            logger_name=str(self.__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )

        if len(self.feature_file_paths) == 0:
            raise NoFilesFoundError(
                "Zero files found in the project_folder/csv/features_extracted directory. Create features before running classifier.",
                source=self.__class__.__name__,
            )
        print(
            f"Analyzing {len(self.feature_file_paths)} file(s) with {self.clf_cnt} classifier(s)..."
        )
        self.timer = SimbaTimer(start=True)
        self.model_dict = self.get_model_info(
            config=self.config, model_cnt=self.clf_cnt
        )

    def _create_p_col_headers(self, model_info: dict):
        self.p_col_headers = []
        for v in model_info["classifier_map"].values():
            self.p_col_headers.append(f"Probability_{v}")

    def run(self):
        for file_cnt, file_path in enumerate(self.feature_file_paths):
            video_timer = SimbaTimer(start=True)
            file_name = get_fn_ext(file_path)[1]
            print(f"Analyzing video {file_name}...")
            in_df = read_df(file_path, self.file_type)
            x_df = self.drop_bp_cords(df=in_df, raise_error=False).astype("float32")
            self.check_df_dataset_integrity(
                df=x_df, logs_path=self.logs_path, file_name=file_name
            )
            out_df = deepcopy(in_df)
            for m, m_hyp in self.model_dict.items():
                if not os.path.isfile(m_hyp["model_path"]):
                    raise NoFilesFoundError(
                        msg=f"{m_hyp['model_path']} is not a VALID model file path",
                        source=self.__class__.__name__,
                    )
                self._create_p_col_headers(model_info=m_hyp)
                clf = self.read_pickle(file_path=m_hyp["model_path"])
                probability_df = pd.DataFrame(
                    self.clf_predict_proba(
                        clf=clf,
                        x_df=x_df,
                        data_path=file_path,
                        model_name=m_hyp["model_name"],
                        multiclass=True,
                    ),
                    columns=self.p_col_headers,
                )
                out_df = pd.concat([out_df, probability_df], axis=1)
            # Patch 122bc: dropped the legacy
            # csv/machine_results/ write — same close-out as
            # 122ax did for InferenceBatch. The multi-class
            # variant was missed by 122ax and surfaced by the
            # 122bc csv-subtree audit. v1 helper handles both
            # project formats; predictions land at
            # derived/classifications/<video>.parquet.
            from mufasa.utils.classification_io import (
                save_classifications_for_video,
                _prediction_columns,
            )
            clf_names = []
            for m_hyp in self.model_dict.values():
                # Multi-class produces one prediction column per
                # classifier (the winning class label per frame
                # isn't materialised here, but the probability
                # columns ARE — _prediction_columns picks up
                # any non-probability classifier columns).
                clf_names.append(m_hyp["model_name"])
            pred_cols = _prediction_columns(out_df, clf_names)
            if pred_cols:
                save_classifications_for_video(
                    video_name=file_name,
                    config_path=self.config_path,
                    predictions=out_df[pred_cols],
                )
            video_timer.stop_timer()
            print(
                f"Predictions created for {file_name} (elapsed time: {video_timer.elapsed_time_str}) ..."
            )
        self.timer.stop_timer()
        stdout_success(
            msg=f"Multi-class machine predictions complete. {len(self.feature_file_paths)} file(s) saved in project_folder/derived/classifications/ directory",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


# test = InferenceMulticlassBatch(config_path='/Users/simon/Desktop/envs/troubleshooting/multilabel/project_folder/project_config.ini')
# test.run()


# test.clf.n_classes_
# test.model_dict
