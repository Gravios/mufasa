"""
mufasa.ui_qt.popups.clf_add_remove_print
=======================================

Qt6 ports of the three classifier-management popups from
:mod:`mufasa.ui.pop_ups.clf_add_remove_print_pop_up`:

* :class:`AddClfPopUp`         — add a new classifier target to a project
* :class:`RemoveAClassifierPopUp` — remove a classifier from a project
* :class:`PrintModelInfoPopUp`    — dump a trained model's metadata

The original module uses the ``(PopUpMixin, ConfigReader)`` MI pattern.
Here we use :class:`mufasa.ui_qt.dialog.MufasaDialog`'s composition +
``__getattr__`` proxy so ``self.clf_names``, ``self.clf_cnt``,
``self.config``, ``self.config_path`` resolve transparently — the
``run()`` methods port **with no logic changes**.
"""
from __future__ import annotations

import os
from typing import Union

from PySide6.QtWidgets import QMessageBox

from mufasa.ui_qt.dialog import MufasaDialog
from mufasa.ui_qt.widgets import (NW, CreateLabelFrameWithIcon, Entry_Box,
                                 FileSelect, MufasaDropDown, MufasaButton, W)

# These non-GUI imports are the same ones the Tk version uses. None of
# them pull tkinter — they were already neutral utilities.
from mufasa.utils.checks import check_str
from mufasa.utils.enums import ConfigKey, Keys, Links
from mufasa.utils.errors import DuplicationError, NoDataError
from mufasa.utils.printing import stdout_success, stdout_trash
from mufasa.utils.read_write import tabulate_clf_info


class AddClfPopUp(MufasaDialog):
    """Add a new classifier target to a Mufasa project.

    Ported from :class:`mufasa.ui.pop_ups.clf_add_remove_print_pop_up.AddClfPopUp`.

    **Port notes**

    * MI ``(PopUpMixin, ConfigReader)`` → composition. ``config_path`` flows
      into :class:`MufasaDialog`, which attaches a ``ConfigReader`` and
      proxies its attributes.
    * Tk ``self.main_frm`` is now a :class:`QWidget` whose layout is a
      :class:`QGridLayout` — so ``.grid(row=, column=, sticky=)`` on child
      widgets works unchanged via the widget shim's geometry mixin.
    * ``Entry_Box(labelwidth=25, entry_box_width=30)`` and
      ``MufasaButton(img='rocket')`` unchanged.
    * ``run()`` body is **literally unchanged** from the Tk version —
      every ``self.xxx`` is resolved either directly (widgets) or via the
      ConfigReader proxy.
    """

    def __init__(self, config_path: Union[str, os.PathLike]) -> None:
        MufasaDialog.__init__(
            self,
            title="ADD CLASSIFIER",
            config_path=str(config_path),
            icon="plus",
            size=(560, 160),
        )
        self.clf_eb = Entry_Box(
            parent=self.main_frm,
            fileDescription="CLASSIFIER NAME:",
            labelwidth=25,
            entry_box_width=30,
            justify="center",
        )
        add_btn = MufasaButton(
            parent=self.main_frm,
            txt="ADD CLASSIFIER",
            cmd=self.run,
            img="rocket",
        )
        self.clf_eb.grid(row=0, column=0, sticky=NW)
        add_btn.grid(row=1, column=0, sticky=NW)

    def run(self) -> None:
        # NB: body identical to the Tk version.
        clf_name = self.clf_eb.entry_get.strip()
        check_str(name="CLASSIFIER NAME", value=clf_name)
        if clf_name in self.clf_names:
            raise DuplicationError(
                msg=f"The classifier name {clf_name} already exist in the Mufasa project.",
                source=self.__class__.__name__,
            )
        self.config.set(
            ConfigKey.SML_SETTINGS.value,
            ConfigKey.TARGET_CNT.value,
            str(self.clf_cnt + 1),
        )
        self.config.set(
            ConfigKey.SML_SETTINGS.value,
            f"model_path_{self.clf_cnt + 1}",
            "",
        )
        self.config.set(
            ConfigKey.SML_SETTINGS.value,
            f"target_name_{self.clf_cnt + 1}",
            clf_name,
        )
        self.config.set(
            ConfigKey.THRESHOLD_SETTINGS.value,
            f"threshold_{self.clf_cnt + 1}",
            "None",
        )
        self.config.set(
            ConfigKey.MIN_BOUT_LENGTH.value,
            f"min_bout_{self.clf_cnt + 1}",
            "None",
        )
        with open(self.config_path, "w") as f:
            self.config.write(f)
        stdout_success(
            msg=f"{clf_name} classifier added to Mufasa project",
            source=self.__class__.__name__,
        )


class RemoveAClassifierPopUp(MufasaDialog):
    """Remove a classifier target from a Mufasa project.

    Ported from :class:`mufasa.ui.pop_ups.clf_add_remove_print_pop_up.RemoveAClassifierPopUp`.

    **Port note on the confirmation dialog:** the Tk original spawns a
    custom ``TwoOptionQuestionPopUp`` Toplevel. In Qt we use the native
    :class:`QMessageBox.question` which is shorter, accessible, and
    HiDPI-correct. Behavioural parity preserved: YES proceeds, anything
    else is a no-op.
    """

    def __init__(self, config_path: Union[str, os.PathLike]) -> None:
        # Create the ConfigReader first so we can validate *before*
        # building the UI (matching Tk version's ordering).
        MufasaDialog.__init__(
            self,
            title="WARNING: REMOVE CLASSIFIER",
            config_path=str(config_path),
            icon="trash_red",
            size=(560, 180),
        )
        if not isinstance(self.clf_names, (list, tuple)) or len(self.clf_names) < 1:
            raise NoDataError(
                msg="The Mufasa project has no classifiers: Cannot remove a classifier.",
                source=self.__class__.__name__,
            )
        self.remove_clf_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SELECT A CLASSIFIER TO REMOVE",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.REMOVE_CLF.value,
        )
        self.clf_dropdown = MufasaDropDown(
            parent=self.remove_clf_frm,
            dropdown_options=self.clf_names,
            label_width=20,
            dropdown_width=40,
            label="CLASSIFIER:",
            value=self.clf_names[0],
        )
        run_btn = MufasaButton(
            parent=self.main_frm,
            txt="REMOVE CLASSIFIER",
            cmd=self.run,
            img="trash",
        )
        self.remove_clf_frm.grid(row=0, column=0, sticky=W)
        self.clf_dropdown.grid(row=0, column=0, sticky=W)
        run_btn.grid(row=1, column=0, pady=10)

    def run(self) -> None:
        clf_to_remove = self.clf_dropdown.get_value()
        ans = QMessageBox.question(
            self,
            "WARNING!",
            f"Do you want to remove the {clf_to_remove}\nclassifier from the Mufasa project?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        # --- body below is literally the same as Tk version ------------- #
        for i in range(len(self.clf_names)):
            self.config.remove_option("SML settings", f"model_path_{i+1}")
            self.config.remove_option("SML settings", f"target_name_{i+1}")
            self.config.remove_option("threshold_settings", f"threshold_{i+1}")
            self.config.remove_option("Minimum_bout_lengths", f"min_bout_{i+1}")
        self.clf_names.remove(clf_to_remove)
        self.config.set("SML settings", "no_targets", str(len(self.clf_names)))
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.config.set("SML settings", f"model_path_{clf_cnt + 1}", "")
            self.config.set("SML settings", f"target_name_{clf_cnt + 1}", clf_name)
            self.config.set("threshold_settings", f"threshold_{clf_cnt + 1}", "None")
            self.config.set("Minimum_bout_lengths", f"min_bout_{clf_cnt + 1}", "None")
        with open(self.config_path, "w") as f:
            self.config.write(f)
        stdout_trash(
            msg=f"{clf_to_remove} classifier removed from Mufasa project.",
            source=self.__class__.__name__,
        )


class PrintModelInfoPopUp(MufasaDialog):
    """Print metadata of a trained classifier ``.sav`` file.

    Ported from :class:`mufasa.ui.pop_ups.clf_add_remove_print_pop_up.PrintModelInfoPopUp`.

    This popup does not need a project config — it takes a model file
    path and calls :func:`tabulate_clf_info` on it.
    """

    def __init__(self) -> None:
        MufasaDialog.__init__(
            self,
            title="PRINT MACHINE MODEL INFO",
            size=(520, 180),
            icon="print",
        )
        self.file_select = FileSelect(
            parent=self.main_frm,
            fileDescription="MODEL FILE PATH (.SAV):",
            file_types=[("Mufasa model", "*.sav")],
            lblwidth=30,
        )
        run_btn = MufasaButton(
            parent=self.main_frm,
            txt="PRINT MODEL INFO",
            cmd=self.run,
            img="print",
        )
        self.file_select.grid(row=0, column=0, sticky=NW)
        run_btn.grid(row=1, column=0, sticky=NW)

    def run(self) -> None:
        path = self.file_select.file_path
        if not path:
            QMessageBox.warning(self, "No file", "Select a model (.sav) file first.")
            return
        tabulate_clf_info(clf_path=path)


__all__ = ["AddClfPopUp", "RemoveAClassifierPopUp", "PrintModelInfoPopUp"]
