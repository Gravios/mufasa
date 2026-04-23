__author__ = "Simon Nilsson; sronilsson@gmail.com"

import threading
from tkinter import *

from mufasa.mixins.config_reader import ConfigReader
from mufasa.mixins.pop_up_mixin import PopUpMixin
from mufasa.ui.tkinter_functions import DropDownMenu, FolderSelect
from mufasa.unsupervised.data_extractor import DataExtractor
from mufasa.unsupervised.enums import UMLOptions
from mufasa.utils.checks import check_if_dir_exists
from mufasa.utils.enums import Formats, Options
from mufasa.utils.errors import CountError
from mufasa.utils.read_write import find_files_of_filetypes_in_directory


class DataExtractorPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="DATA EXTRACTOR")
        ConfigReader.__init__(self, config_path=config_path)
        data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.data_dir_select = FolderSelect(
            data_frm,
            "DATA DIRECTORY (PICKLES):",
            lblwidth=25,
            initialdir=self.project_path,
        )
        data_frm.grid(row=0, column=0, sticky=NW)
        self.data_dir_select.grid(row=0, column=0, sticky=NW)

        self.cb_settings = self.create_cb_frame(
            main_frm=self.main_frm,
            cb_titles=UMLOptions.DATA_TYPES.value,
            frm_title="DATA TYPES",
        )

        self.create_run_frm(run_function=self.run, title="RUN")
        self.main_frm.mainloop()

    def run(self):
        selections = []
        for k, v in self.cb_settings.items():
            if v.get():
                selections.append(k)
        check_if_dir_exists(self.data_dir_select.folder_path)
        if len(selections) == 0:
            raise CountError(
                msg="Select at least 1 data-type checkbox",
                source=self.__class__.__name__,
            )
        check_if_dir_exists(self.data_dir_select.folder_path)
        _ = find_files_of_filetypes_in_directory(
            directory=self.data_dir_select.folder_path,
            extensions=[f".{Formats.PICKLE.value}"],
            raise_error=True,
        )
        data_extractor = DataExtractor(
            config_path=self.config_path,
            data_path=self.data_dir_select.folder_path,
            data_types=selections,
        )
        threading.Thread(target=data_extractor.run()).start()


# _ = DataExtractorPopUp(config_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini')
