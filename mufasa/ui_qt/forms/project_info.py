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
from PySide6.QtWidgets import (QFormLayout, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)


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
    * **Classifiers**  — list of target names (or "none configured")
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
        self._refresh_btn = QPushButton("Refresh", self)
        self._refresh_btn.setMinimumWidth(100)
        self._refresh_btn.clicked.connect(self._populate)
        row.addWidget(self._refresh_btn)
        outer.addLayout(row)

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
            self._add_row("Body parts", "<i>none configured</i>")

        clf = meta.get("classifier_targets") or []
        if clf:
            self._add_row("Classifiers", ", ".join(clf))
        else:
            self._add_row(
                "Classifiers",
                "<i>none configured</i>",
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
        outer.setSpacing(10)

        # Patch 122k: this form is now always visible at the top
        # of the Projects page, not just as an empty-state surface.
        # Tailor the framing message to the current state.
        if self.config_path:
            msg = QLabel(
                "Create a new v1 project (<code>project.toml</code> + "
                "<code>sources/</code>, <code>derived/</code>, "
                "<code>models/</code>, <code>logs/</code>), open a "
                "different one, or switch back to the most recent.",
                self,
            )
        else:
            msg = QLabel(
                "<b>No project loaded.</b><br>"
                "Create a new v1 project (<code>project.toml</code> + "
                "<code>sources/</code>, <code>derived/</code>, "
                "<code>models/</code>, <code>logs/</code>) or open an "
                "existing one. Legacy <code>project_config.ini</code> "
                "projects are still supported via Open.",
                self,
            )
        msg.setTextFormat(Qt.RichText)
        msg.setWordWrap(True)
        outer.addWidget(msg)

        actions = QHBoxLayout()
        new_btn = QPushButton("New project…", self)
        new_btn.setMinimumWidth(140)
        if self._workbench is not None:
            new_btn.clicked.connect(self._workbench._on_new_project)
        else:
            new_btn.setEnabled(False)
            new_btn.setToolTip("Workbench reference missing.")
        actions.addWidget(new_btn)

        open_btn = QPushButton("Open project…", self)
        open_btn.setMinimumWidth(140)
        if self._workbench is not None:
            open_btn.clicked.connect(self._workbench._on_open_project)
        else:
            open_btn.setEnabled(False)
        actions.addWidget(open_btn)
        actions.addStretch()
        outer.addLayout(actions)

        # Recent-project quick-open. Soft-fails on read errors.
        try:
            from mufasa.ui_qt.recent_project import load_recent_project
            recent = load_recent_project()
        except Exception:
            recent = None

        # Patch 122k: hide the recent shortcut when it would be
        # a no-op (recent path == currently-loaded project) or
        # when no recent exists. Both cases would leave a button
        # that either does nothing or doesn't apply.
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
                recent_row = QHBoxLayout()
                recent_btn = QPushButton(
                    f"Open most recent: {recent.name}", self,
                )
                recent_btn.setToolTip(str(recent))
                recent_btn.setStyleSheet("padding: 6px 10px;")
                if self._workbench is not None:
                    # The workbench's _on_open_project asks for a
                    # path via QFileDialog; we have one already, so
                    # dispatch directly to the load helper if
                    # available.
                    def _open_recent() -> None:
                        handler = getattr(
                            self._workbench, "_load_project", None,
                        )
                        if handler is not None:
                            handler(str(recent))
                        else:
                            # Fallback: still call _on_open_project
                            # so the workbench's normal Open dialog
                            # handles it.
                            self._workbench._on_open_project()
                    recent_btn.clicked.connect(_open_recent)
                else:
                    recent_btn.setEnabled(False)
                recent_row.addWidget(recent_btn)
                recent_row.addStretch()
                outer.addLayout(recent_row)

        outer.addStretch()


__all__ = ["ProjectInfoForm", "NewProjectForm"]
