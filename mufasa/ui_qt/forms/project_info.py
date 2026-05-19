"""
mufasa.ui_qt.forms.project_info
===============================

Two display widgets for the "Projects" workbench page
(patch 122i):

* :class:`ProjectInfoForm` — read-only summary of the currently-
  loaded project (name, layout version, root path, animal count,
  file type, body parts, classifier targets, plus quick counts
  of source / derived files).
* :class:`NewProjectForm` — empty-state surface shown when no
  project is loaded. Provides inline buttons that dispatch to
  the workbench's New / Open project slots, plus a one-click
  "Open most recent" if a recent project is on disk.

Neither widget is an :class:`OperationForm` — they don't run
operations. The workbench's :meth:`WorkflowPage._instantiate`
only requires a constructor accepting ``parent=`` and
``config_path=``; any QWidget satisfying that contract slots in.

Note on the "Pose cleanup" / "Preprocessing" / "Projects"
naming history: see ``pose_cleanup_page.py``'s docstring for the
parallel terminology rename in patch 122g. The page-builder file
for *this* widget is still :mod:`pose_cleanup_page` (file kept,
label changed). Same story for ``project_setup_page.py`` — the
file name persists; the user-facing label became "Projects"
in 122i.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)


# --------------------------------------------------------------------------- #
# ProjectInfoForm — read-only display
# --------------------------------------------------------------------------- #
class ProjectInfoForm(QWidget):
    """Read-only summary of the loaded project.

    Fields displayed:

    * **Layout**       — "v1 (project.toml)" or "legacy (project_config.ini)"
    * **Name**         — from project_name / [General settings].project_name
    * **Root**         — project root path (v1: project.toml's parent;
      legacy: <General settings>.project_path)
    * **Animals**      — animal count + ID list
    * **File type**    — csv / parquet / h5
    * **Body parts**   — flat list, truncated with "(+N more)" if long
    * **Classifiers**  — list of target names (or "none")
    * **Source files** — count of pose files in input_pose_dir
    * **Smoothed runs** — count of v1 runs under
      derived/smoothed/<flavor>/ (v1 only; legacy shows "-")

    A Refresh button at the bottom re-reads from disk so the
    user can update the view after editing project.toml or
    running a new pipeline step externally.
    """

    # Surface the title so WorkflowPage._instantiate uses it as
    # the bold header above the widget.
    title = ""  # blank — section heading already says "Project information"

    refreshed = Signal()

    def __init__(self, parent: Optional[QWidget] = None,
                 config_path: Optional[str] = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self._build_shell()
        self._populate()

    # ------------------------------------------------------------------ #
    # UI scaffolding
    # ------------------------------------------------------------------ #
    def _build_shell(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(8)

        self._form_host = QWidget(self)
        self._form_layout = QFormLayout(self._form_host)
        self._form_layout.setLabelAlignment(Qt.AlignRight)
        self._form_layout.setSpacing(6)
        outer.addWidget(self._form_host)

        row = QHBoxLayout()
        row.addStretch()
        # Patch 122n: Edit button — opens EditProjectMetadataDialog
        # so users can fix file type / body parts / animal IDs /
        # classifier targets without leaving the page. v1 only;
        # the dialog refuses on legacy projects with a clear
        # message.
        self._edit_btn = QPushButton("Edit…", self)
        self._edit_btn.setMinimumWidth(100)
        self._edit_btn.clicked.connect(self._open_edit_dialog)
        row.addWidget(self._edit_btn)
        self._refresh_btn = QPushButton("Refresh", self)
        self._refresh_btn.setMinimumWidth(100)
        self._refresh_btn.clicked.connect(self._populate)
        row.addWidget(self._refresh_btn)
        outer.addLayout(row)

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _open_edit_dialog(self) -> None:
        """Open the project-metadata editor. On successful save,
        re-populate the displayed rows so the change shows up
        without requiring a manual Refresh click."""
        if not self.config_path:
            return
        # Lazy import — avoids pulling the dialog code when the
        # button isn't used (and avoids a circular import risk
        # since the dialog reads project_layout helpers).
        from mufasa.ui_qt.dialogs.edit_project_metadata_dialog import (
            EditProjectMetadataDialog,
        )
        dlg = EditProjectMetadataDialog(self.config_path, self)
        dlg.metadata_updated.connect(lambda _path: self._populate())
        dlg.exec()

    # ------------------------------------------------------------------ #
    # Population
    # ------------------------------------------------------------------ #
    def _clear_form(self) -> None:
        # QFormLayout doesn't have a one-shot "remove all rows" method
        # that's portable across PySide6 versions; remove rows from the
        # end so indices stay stable.
        while self._form_layout.rowCount() > 0:
            self._form_layout.removeRow(self._form_layout.rowCount() - 1)

    def _add_row(self, label: str, value: str) -> None:
        # Labels are proper form labels — keep them at the theme's
        # default text color (which inherits from palette(text), the
        # high-contrast text role). Earlier versions dimmed these with
        # palette(mid) — that's a structural UI color, not a text
        # color, and reads as illegible-grey in both light and dark
        # themes on Ubuntu (Yaru / Adwaita) and many other systems.
        lbl = QLabel(label + ":")
        val = QLabel(value)
        val.setTextFormat(Qt.RichText)
        val.setWordWrap(True)
        val.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard,
        )
        val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._form_layout.addRow(lbl, val)

    def _populate(self) -> None:
        """Read metadata from disk and refresh the displayed rows.

        Soft-fails on any read error: shows a single
        '<error>' row rather than raising. The page builder is
        expected to swap this widget out for NewProjectForm when
        no project is loaded; we only show up when there's a
        config_path, but defend against the path turning
        non-existent between page build and Refresh click.
        """
        self._clear_form()
        if not self.config_path:
            self._add_row(
                "Status",
                "<i>No project loaded.</i>",
            )
            return

        cp = Path(self.config_path)
        if not cp.exists():
            self._add_row("Status", f"<i>{cp} no longer exists.</i>")
            return

        # Lazy import — keeps page-build cost minimal.
        try:
            from mufasa.project_layout import (
                project_metadata_from_config,
                project_paths_from_config,
            )
            paths = project_paths_from_config(self.config_path)
            meta = project_metadata_from_config(self.config_path)
        except (ValueError, OSError) as exc:
            self._add_row(
                "Error",
                f"<i>Could not read project config: "
                f"{type(exc).__name__}: {exc}</i>",
            )
            return

        is_v1 = str(cp).lower().endswith(".toml")
        self._add_row(
            "Layout",
            "v1 (<code>project.toml</code>)" if is_v1
            else "legacy (<code>project_config.ini</code>)",
        )
        # Project name — for v1 from project.toml, for legacy we
        # can't trivially get it without parsing the ini, so fall
        # back to the project_root directory name.
        if is_v1:
            try:
                import tomllib
                with open(cp, "rb") as f:
                    toml_data = tomllib.load(f)
                proj_name = toml_data.get("project_name") or ""
            except Exception:
                proj_name = ""
        else:
            proj_name = ""
        if not proj_name:
            proj_name = Path(paths["project_root"]).name
        self._add_row("Name", proj_name)
        self._add_row(
            "Root",
            f"<code>{paths['project_root']}</code>",
        )

        animal_ids = meta.get("animal_ids") or []
        animal_count = meta.get("animal_count", 0)
        if animal_ids:
            ids_summary = (
                ", ".join(animal_ids)
                if len(animal_ids) <= 4
                else f"{', '.join(animal_ids[:4])} (+{len(animal_ids) - 4} more)"
            )
            self._add_row(
                "Animals",
                f"{animal_count} ({ids_summary})",
            )
        else:
            self._add_row("Animals", str(animal_count))

        self._add_row("File type", meta.get("file_type", "csv"))

        bps = meta.get("body_parts") or []
        if bps:
            head = ", ".join(bps[:8])
            tail = (
                ""
                if len(bps) <= 8
                else f" <span style='color:palette(placeholder-text)'>(+{len(bps) - 8} more)</span>"
            )
            self._add_row("Body parts", f"{head}{tail}")
        else:
            self._add_row("Body parts", "<i>none</i>")

        clf = meta.get("classifier_targets") or []
        if clf:
            self._add_row("Classifiers", ", ".join(clf))
        else:
            self._add_row(
                "Classifiers",
                "<i>none</i>",
            )

        # File counts. Cheap glob; tolerate non-existent dirs.
        pose_dir = Path(paths["input_pose_dir"])
        if pose_dir.is_dir():
            ft = meta.get("file_type", "csv")
            n_pose = len(list(pose_dir.glob(f"*.{ft}")))
        else:
            n_pose = 0
        self._add_row(
            "Source pose files",
            f"{n_pose} <span style='color:palette(placeholder-text)'>"
            f"in <code>{pose_dir.name}/</code></span>",
        )

        # v1 derived-runs counts. Skip for legacy.
        if is_v1:
            root = Path(paths["project_root"])
            self._add_row(
                "Smoothed runs",
                self._count_runs(root / "derived" / "smoothed"),
            )
            self._add_row(
                "Outlier-corrected runs",
                self._count_runs(root / "derived" / "outlier_corrected"),
            )
            self._add_row(
                "Feature runs",
                self._count_runs(root / "derived" / "features"),
            )
            self._add_row(
                "Classification runs",
                self._count_runs(root / "derived" / "classifications"),
            )

        self.refreshed.emit()

    def _count_runs(self, stage_parent: Path) -> str:
        """Count run-id subdirs under ``stage_parent``. Returns
        a human string, including a hint at the latest run id
        when any exist.
        """
        if not stage_parent.is_dir():
            return "0"
        # is_run_id imported lazily to avoid surfacing the
        # project_layout import at module load time when no v1
        # project is present.
        from mufasa.project_layout import is_run_id

        # Stage parent may have flavor subdirs (smoothed has
        # kalman_v2/ etc.) OR direct run-id dirs. Try both.
        run_dirs = []
        for child in stage_parent.iterdir():
            if not child.is_dir():
                continue
            if is_run_id(child.name):
                run_dirs.append(child)
            else:
                # Could be a flavor; descend one level.
                for grand in child.iterdir():
                    if grand.is_dir() and is_run_id(grand.name):
                        run_dirs.append(grand)
        if not run_dirs:
            return "0"
        run_dirs.sort()
        latest = run_dirs[-1].name
        return (
            f"{len(run_dirs)} "
            f"<span style='color:palette(placeholder-text)'>latest: "
            f"<code>{latest}</code></span>"
        )


# --------------------------------------------------------------------------- #
# NewProjectForm — empty-state
# --------------------------------------------------------------------------- #
class NewProjectForm(QWidget):
    """Empty-state surface shown when no project is loaded.

    Three actions:

    * **New project…** → ``workbench._on_new_project`` (same slot
      File menu uses).
    * **Open project…** → ``workbench._on_open_project``.
    * **Open most recent: <path>** — visible only when
      :func:`mufasa.ui_qt.recent_project.load_recent_project`
      returns a valid path. One click reopens the workbench with
      that project. Useful for "I just closed my project, give
      it back."

    The widget needs the workbench reference to wire its buttons
    to the same slots the File menu uses. Pass it via the
    ``workbench=`` kwarg in :meth:`WorkflowPage.add_section`.
    """

    title = ""

    def __init__(self, parent: Optional[QWidget] = None,
                 config_path: Optional[str] = None,
                 workbench: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self._workbench = workbench
        self._build_shell()

    def _build_shell(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(12)

        # ---------------------------------------------------------- #
        # Intro / framing text — adapts to project-loaded state
        # ---------------------------------------------------------- #
        # Patch 122k: contextual messaging (no project vs loaded).
        # Patch 122l: trimmed to one short sentence; the per-group
        # framing (QGroupBox titles + inner hints) carries the
        # rest, so the intro doesn't repeat what's right below it.
        if self.config_path:
            intro = QLabel(
                "Switch to a different project, or create a new one.",
                self,
            )
        else:
            intro = QLabel(
                "<b>No project loaded.</b> Open an existing project "
                "or create a new one below.",
                self,
            )
        intro.setTextFormat(Qt.RichText)
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # ---------------------------------------------------------- #
        # Group 1: Open existing project
        # ---------------------------------------------------------- #
        open_group = QGroupBox("Open existing project", self)
        open_lay = QVBoxLayout(open_group)
        open_lay.setSpacing(8)

        open_hint = QLabel(
            "Open a v1 <code>project.toml</code> or a legacy "
            "<code>project_config.ini</code> from disk. Use "
            "<b>Open most recent</b> to jump back to the previous "
            "project without browsing for it.",
            open_group,
        )
        open_hint.setTextFormat(Qt.RichText)
        open_hint.setWordWrap(True)
        open_hint.setStyleSheet("color: palette(placeholder-text);")
        open_lay.addWidget(open_hint)

        actions = QHBoxLayout()
        open_btn = QPushButton("Open project…", open_group)
        open_btn.setMinimumWidth(140)
        if self._workbench is not None:
            open_btn.clicked.connect(self._workbench._on_open_project)
        else:
            open_btn.setEnabled(False)
            open_btn.setToolTip("Workbench reference missing.")
        actions.addWidget(open_btn)

        # Recent-project quick-open. Soft-fails on read errors.
        # Hidden when no recent exists OR when recent == current
        # (would be a no-op).
        try:
            from mufasa.ui_qt.recent_project import load_recent_project
            recent = load_recent_project()
        except Exception:
            recent = None
        if recent is not None:
            current_resolved: Optional[Path] = None
            if self.config_path:
                try:
                    current_resolved = Path(self.config_path).resolve()
                except OSError:
                    current_resolved = None
            recent_is_self = (
                current_resolved is not None
                and recent.resolve() == current_resolved
            )
            if not recent_is_self:
                recent_btn = QPushButton(
                    f"Open most recent: {recent.name}", open_group,
                )
                recent_btn.setToolTip(str(recent))
                if self._workbench is not None:
                    def _open_recent() -> None:
                        # Prefer _switch_to_project (the workbench's
                        # post-creation / post-open code path) when
                        # present; fall back to _load_project if a
                        # downstream variant exposes that instead;
                        # final fallback is the normal Open dialog.
                        switch = getattr(
                            self._workbench, "_switch_to_project", None,
                        )
                        if switch is not None:
                            switch(str(recent))
                            return
                        loader = getattr(
                            self._workbench, "_load_project", None,
                        )
                        if loader is not None:
                            loader(str(recent))
                            return
                        self._workbench._on_open_project()
                    recent_btn.clicked.connect(_open_recent)
                else:
                    recent_btn.setEnabled(False)
                actions.addWidget(recent_btn)
        actions.addStretch()
        open_lay.addLayout(actions)
        outer.addWidget(open_group)

        # ---------------------------------------------------------- #
        # Group 2: Create new project — embed ProjectCreateForm
        # ---------------------------------------------------------- #
        # Patch 122l: the create flow is now inline rather than a
        # modal dialog. ProjectCreateForm draws its own Create
        # button at the bottom (show_create_button=True). On
        # successful creation it emits project_created with the
        # new config_path; we route that to the workbench's
        # switch-to-project handler so the workbench reopens
        # pointed at the new project.
        from mufasa.ui_qt.forms.project_create import ProjectCreateForm

        create_group = QGroupBox("Create new project", self)
        create_lay = QVBoxLayout(create_group)
        create_lay.setSpacing(8)

        create_hint = QLabel(
            "Creates a v1 project directory tree "
            "(<code>project.toml</code> + <code>sources/</code>, "
            "<code>derived/</code>, <code>models/</code>, "
            "<code>logs/</code>). The workbench will reopen on "
            "the new project.",
            create_group,
        )
        create_hint.setTextFormat(Qt.RichText)
        create_hint.setWordWrap(True)
        create_hint.setStyleSheet("color: palette(placeholder-text);")
        create_lay.addWidget(create_hint)

        self._create_form = ProjectCreateForm(
            create_group, show_create_button=True,
        )
        self._create_form.project_created.connect(
            self._on_project_created,
        )
        create_lay.addWidget(self._create_form)
        outer.addWidget(create_group)

        outer.addStretch()

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_project_created(self, config_path: str) -> None:
        """The inline create form succeeded. Switch the workbench
        to point at the new project so all the other pages
        rebuild with its config_path."""
        if self._workbench is None:
            return
        switch = getattr(self._workbench, "_switch_to_project", None)
        if switch is not None:
            switch(config_path)


__all__ = ["ProjectInfoForm", "NewProjectForm"]
