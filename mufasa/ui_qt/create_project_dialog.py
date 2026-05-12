"""
mufasa.ui_qt.create_project_dialog
==================================

Modal wrapper around :class:`mufasa.ui_qt.forms.project_create.ProjectCreateForm`.
Used by the File → New project menu action; produces the same
v1 project layout (``project.toml`` + ``sources/``,
``derived/``, ``models/``, ``logs/``) as the inline form on the
Projects page.

Patch 122l: refactored from a self-contained dialog (~300 lines
of widget construction and validation logic) into a thin shell
around :class:`ProjectCreateForm`. The dialog's responsibility
is now only:

* host the form as its central content
* provide modal Cancel via a :class:`QDialogButtonBox`
* listen for :pyattr:`ProjectCreateForm.project_created` and
  close itself with the new config path captured on
  :pyattr:`self.config_path`

The previous embedded validation + ProjectConfigCreator call
moved with the field widgets into :class:`ProjectCreateForm`,
so the inline Projects-page surface and the menu-launched modal
share the same code path.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel,
                               QVBoxLayout, QWidget)

from mufasa.ui_qt.forms.project_create import ProjectCreateForm


class CreateProjectDialog(QDialog):
    """Modal dialog that creates a new v1-layout Mufasa project.

    Use :pyattr:`config_path` after ``exec()`` returns
    :pyattr:`QDialog.Accepted` to retrieve the path to the
    generated ``project.toml``. Returns :pyattr:`QDialog.Rejected`
    if the user cancels; in that case :pyattr:`config_path`
    remains ``None``.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create new Mufasa project")
        self.setModal(True)
        self.resize(560, 540)
        self.config_path: Optional[str] = None

        # Embed the inline form. show_create_button=False because
        # the dialog supplies its own action button via
        # QDialogButtonBox (standard Qt OK / Cancel pattern).
        self._form = ProjectCreateForm(
            self, show_create_button=False,
        )
        self._form.project_created.connect(self._on_project_created)

        # OK / Cancel button row. "Create" label on the accept
        # button reads better than the default "OK" for a
        # creation action.
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, parent=self)
        create_btn = buttons.addButton(
            "Create project", QDialogButtonBox.AcceptRole,
        )
        # AcceptRole buttons don't auto-emit accepted() until the
        # dialog is dismissed; we want to validate first and only
        # accept on success. Wire directly to the form's submit().
        create_btn.clicked.connect(self._form.submit)
        buttons.rejected.connect(self.reject)

        # Top hint — explains what'll happen on Create.
        hint = QLabel(
            "This creates a new v1 project directory tree "
            "(<code>project.toml</code> + <code>sources/</code>, "
            "<code>derived/</code>, <code>models/</code>, "
            "<code>logs/</code>). The workbench will reopen "
            "pointing at the new project.",
            self,
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.RichText)

        root = QVBoxLayout(self)
        root.addWidget(hint)
        root.addWidget(self._form)
        root.addStretch(1)
        root.addWidget(buttons)

    def _on_project_created(self, config_path: str) -> None:
        """Slot: form succeeded. Capture the path + accept the
        dialog so the caller's ``exec()`` returns Accepted and
        can read ``self.config_path``."""
        self.config_path = config_path
        self.accept()
