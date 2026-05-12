"""
mufasa.ui_qt.input_source_picker
================================

Patch 122b: a small reusable widget for picking which directory
of pose data a downstream operation should read from.

Motivation
----------

Before the Kalman v2 smoother landed there were only two plausible
inputs for downstream stages: raw imported pose
(``csv/input_csv/``) or outlier-corrected pose
(``csv/outlier_corrected_movement_location/``). Forms hard-coded a
two-way fallback. Now there are at least four (raw, outlier-
corrected, legacy Savitzky-Golay smoothed, Kalman v2 smoothed)
and once project layout v1 is wired in there's potentially one
candidate per saved run.

Two parts here:

* :func:`discover_input_sources` — pure function that walks the
  legacy ``csv/*`` dirs and the v1 ``derived/*`` tree and returns
  a list of :class:`InputSource` candidates. Testable headless,
  no PySide6.

* :class:`InputSourcePicker` — Qt widget that wraps the discovery
  function in a combobox + path field + browse + refresh. Drop
  into any form that needs to ask "which version of the pose
  data?".

Consumers: :class:`EgocentricAlignmentForm` first (this patch),
then the outlier-correction-run form, then feature extraction,
then inference (Session B and beyond).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from mufasa.project_layout import (
    ProjectPaths,
    SmoothingFlavors,
    Stages,
    detect_layout,
    is_run_id,
    resolve_v1_project_root,
)


# --------------------------------------------------------------------------- #
# Source kinds + discovery
# --------------------------------------------------------------------------- #

class SourceKinds:
    """Stable identifiers for the kinds of input source the picker
    knows about. Use these in client code rather than string
    literals — adding a new kind is a one-place change.
    """
    RAW = "raw"
    OUTLIER_CORRECTED = "outlier_corrected"
    SMOOTHED_KALMAN_V2 = "smoothed_kalman_v2"
    SMOOTHED_SAVITZKY = "smoothed_savitzky"
    CUSTOM = "custom"


# Human-readable labels for the bare kind names. Used as a prefix
# in the picker's combobox items; the discovery function appends
# run-id / "(legacy)" suffixes where relevant.
_KIND_LABELS = {
    SourceKinds.RAW:                "Raw pose",
    SourceKinds.OUTLIER_CORRECTED:  "Outlier-corrected",
    SourceKinds.SMOOTHED_KALMAN_V2: "Smoothed — Kalman v2",
    SourceKinds.SMOOTHED_SAVITZKY:  "Smoothed — Savitzky-Golay",
    SourceKinds.CUSTOM:             "Custom path…",
}


@dataclass(frozen=True)
class InputSource:
    """A single candidate input directory.

    Attributes
    ----------
    label : str
        Human-readable; what the user sees in the picker.
    path : Path
        Absolute directory. Discovery only returns sources
        whose directory exists; emptiness (no pose files inside)
        is the caller's problem to detect.
    kind : str
        One of :class:`SourceKinds` values.
    is_default : bool
        True if the picker should initialize to this source.
        Exactly zero or one source will have this set per call;
        callers can detect "none preferred" by checking the
        return list.
    """
    label: str
    path: Path
    kind: str
    is_default: bool = False


# Discovery is parameterized to keep ranking explicit. The default
# preference order picks the "most processed" available source —
# the assumption being that downstream stages usually want the
# latest output of the previous stage. Override by passing a
# different ``prefer_order``.
_DEFAULT_PREFER_ORDER: Tuple[str, ...] = (
    SourceKinds.SMOOTHED_KALMAN_V2,
    SourceKinds.OUTLIER_CORRECTED,
    SourceKinds.SMOOTHED_SAVITZKY,
    SourceKinds.RAW,
)


def _dir_has_pose_files(d: Path) -> bool:
    """Truthy if ``d`` exists and contains at least one file that
    looks like pose data. Cheap; doesn't open the files.

    The extension list mirrors what the smoother and the feature
    pipeline accept. Hidden files and subdirectories are ignored.
    """
    if not d.is_dir():
        return False
    try:
        for entry in os.listdir(d):
            if entry.startswith("."):
                continue
            full = d / entry
            if not full.is_file():
                continue
            if entry.lower().endswith((".csv", ".tsv", ".parquet", ".h5")):
                return True
    except OSError:
        return False
    return False


def _legacy_sources(project_path: Path) -> List[InputSource]:
    """Discover legacy SimBA-style sources under
    ``<project_path>/csv/``.

    Order matches the legacy SimBA pipeline: input_csv (raw),
    outlier_corrected_movement, outlier_corrected_movement_location
    (the final outlier-corrected output), smoothed_v2 (the Kalman
    v2 form's default output dir under legacy layout). Savitzky-
    Golay smoothing in legacy SimBA writes back over input_csv
    in-place, so there's no distinct "savitzky" dir to list.
    """
    out: List[InputSource] = []
    csv_root = project_path / "csv"
    candidates = [
        (
            csv_root / "input_csv",
            SourceKinds.RAW,
            "Raw pose (csv/input_csv/)",
        ),
        (
            csv_root / "outlier_corrected_movement",
            SourceKinds.OUTLIER_CORRECTED,
            "Outlier-corrected (movement only)",
        ),
        (
            csv_root / "outlier_corrected_movement_location",
            SourceKinds.OUTLIER_CORRECTED,
            "Outlier-corrected (movement + location)",
        ),
        (
            csv_root / "smoothed_v2",
            SourceKinds.SMOOTHED_KALMAN_V2,
            "Smoothed — Kalman v2 (csv/smoothed_v2/)",
        ),
    ]
    for path, kind, label in candidates:
        if _dir_has_pose_files(path):
            out.append(InputSource(label=label, path=path, kind=kind))
    return out


def _v1_sources(project_root: Path) -> List[InputSource]:
    """Discover v1 sources under ``<project_root>/sources/`` and
    ``<project_root>/derived/``.

    Each multi-run stage contributes one entry per run, sorted
    newest-first (run-ids are lexically sortable by timestamp).
    Empty run dirs are skipped.
    """
    out: List[InputSource] = []
    paths = ProjectPaths(project_root)

    # Raw under sources/pose/
    if _dir_has_pose_files(paths.sources_pose):
        out.append(InputSource(
            label="Raw pose (sources/pose/)",
            path=paths.sources_pose,
            kind=SourceKinds.RAW,
        ))

    # Smoothed runs — Kalman v2, newest first
    kv2_runs = paths.list_runs(
        Stages.SMOOTHED, flavor=SmoothingFlavors.KALMAN_V2,
    )
    for run_dir in reversed(kv2_runs):  # newest first
        if _dir_has_pose_files(run_dir):
            out.append(InputSource(
                label=f"Smoothed — Kalman v2 (run {run_dir.name})",
                path=run_dir,
                kind=SourceKinds.SMOOTHED_KALMAN_V2,
            ))

    # Smoothed runs — Savitzky-Golay
    sg_runs = paths.list_runs(
        Stages.SMOOTHED, flavor=SmoothingFlavors.SAVITZKY_GOLAY,
    )
    for run_dir in reversed(sg_runs):
        if _dir_has_pose_files(run_dir):
            out.append(InputSource(
                label=f"Smoothed — Savitzky-Golay (run {run_dir.name})",
                path=run_dir,
                kind=SourceKinds.SMOOTHED_SAVITZKY,
            ))

    # Outlier-corrected runs
    oc_runs = paths.list_runs(Stages.OUTLIER_CORRECTED)
    for run_dir in reversed(oc_runs):
        if _dir_has_pose_files(run_dir):
            out.append(InputSource(
                label=f"Outlier-corrected (run {run_dir.name})",
                path=run_dir,
                kind=SourceKinds.OUTLIER_CORRECTED,
            ))

    return out


def _mark_default(
    sources: List[InputSource],
    prefer_order: Tuple[str, ...],
) -> List[InputSource]:
    """Return ``sources`` with at most one entry marked default.

    Walks ``prefer_order`` looking for the first kind that has any
    candidate; the first such candidate (already sorted newest-
    first by the discovery helpers) wins. Unchanged otherwise.
    """
    by_kind: dict[str, int] = {}
    for i, s in enumerate(sources):
        by_kind.setdefault(s.kind, i)
    for kind in prefer_order:
        if kind in by_kind:
            i = by_kind[kind]
            sources[i] = InputSource(
                label=sources[i].label,
                path=sources[i].path,
                kind=sources[i].kind,
                is_default=True,
            )
            break
    return sources


def discover_input_sources(
    config_path: Optional[str] = None,
    project_root: Optional[Path] = None,
    *,
    prefer_order: Tuple[str, ...] = _DEFAULT_PREFER_ORDER,
) -> List[InputSource]:
    """Return candidate input directories for a downstream operation.

    Either ``config_path`` (legacy INI) or ``project_root`` (v1 root)
    may be passed; if both are passed, v1 candidates come first
    (preferred). At most one returned source is marked
    ``is_default=True``, per ``prefer_order``. Sources whose
    directory doesn't exist or contains no pose files are omitted.

    Resolution rules:

    * If ``project_root`` is None but ``config_path`` resolves to a
      v1 project (via :func:`resolve_v1_project_root`), v1 sources
      are discovered there.
    * If ``config_path`` points to a legacy INI, the legacy
      ``project_path`` is read from it and legacy sources are
      added. The same project_root may yield both v1 and legacy
      sources during the post-migration transient state — that's
      expected and surfaced for the user to choose between.
    * If neither path resolves to anything usable, an empty list
      is returned — callers should fall back to a custom-only UI.
    """
    out: List[InputSource] = []

    # v1 first (preferred when present)
    eff_project_root = project_root
    if eff_project_root is None:
        eff_project_root = resolve_v1_project_root(config_path)
    if eff_project_root is not None and detect_layout(eff_project_root) == "v1":
        out.extend(_v1_sources(eff_project_root))

    # Legacy second
    if config_path:
        import configparser
        cfg = configparser.ConfigParser()
        try:
            cfg.read(config_path)
            legacy_root = cfg.get(
                "General settings", "project_path", fallback="",
            ).strip()
            if legacy_root and Path(legacy_root).is_dir():
                out.extend(_legacy_sources(Path(legacy_root)))
        except (configparser.Error, OSError):
            # Bad INI / unreadable — silently skip; the v1 path
            # may still yield something useful.
            pass

    return _mark_default(out, prefer_order)


# --------------------------------------------------------------------------- #
# Qt widget
# --------------------------------------------------------------------------- #
#
# The widget is imported lazily — :func:`discover_input_sources` is
# the pure-Python part of this module and shouldn't pay for PySide6
# import cost when used from CLI tools or smoke tests.
#
# Public API:
#
#   picker = InputSourcePicker(parent, config_path=..., project_root=...)
#   layout.addWidget(picker)
#   ...
#   path = picker.selected_path()  # raises ValueError if empty/invalid
#   kind = picker.selected_kind()
#
# Signals:
#
#   picker.selection_changed(InputSource | None)
#       Emitted whenever the user picks a different source or types
#       a new custom path. ``None`` means "Custom" is selected but
#       the path is empty/invalid.

try:
    from PySide6.QtCore import Qt, Signal  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QPushButton,
        QVBoxLayout, QWidget,
    )
    _HAVE_QT = True
except ImportError:  # pragma: no cover - sandbox has no PySide6
    _HAVE_QT = False


if _HAVE_QT:

    class InputSourcePicker(QWidget):
        """Combobox + path field + browse + refresh.

        Renders the list returned by :func:`discover_input_sources`
        plus a "Custom path…" sentinel. Selecting the sentinel
        un-greys the path field and the browse button. Selecting
        any concrete source pins the path to that source's dir
        (read-only). The refresh button re-runs discovery — useful
        after running a new smoother or outlier-correction pass
        without rebuilding the form.

        The widget caches the most-recent discovery result. Tests
        and lightweight callers may set ``self._sources`` directly
        and call ``self._rebuild_combo()`` to bypass discovery
        entirely.
        """

        selection_changed = Signal(object)  # InputSource | None

        def __init__(
            self,
            parent: Optional[QWidget] = None,
            *,
            config_path: Optional[str] = None,
            project_root: Optional[Path] = None,
            prefer_order: Optional[Tuple[str, ...]] = None,
        ) -> None:
            super().__init__(parent)
            self._config_path = config_path
            self._project_root = project_root
            self._prefer_order = (
                prefer_order if prefer_order is not None
                else _DEFAULT_PREFER_ORDER
            )
            self._sources: List[InputSource] = []

            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(4)

            self._combo = QComboBox(self)
            self._combo.currentIndexChanged.connect(
                self._on_combo_changed,
            )
            outer.addWidget(self._combo)

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            self._path_edit = QLineEdit(self)
            self._path_edit.textChanged.connect(self._on_path_edited)
            row.addWidget(self._path_edit)
            self._browse_btn = QPushButton("Browse…", self)
            self._browse_btn.clicked.connect(self._on_browse)
            row.addWidget(self._browse_btn)
            self._refresh_btn = QPushButton("↻", self)
            self._refresh_btn.setToolTip(
                "Re-scan for available source directories",
            )
            self._refresh_btn.setMaximumWidth(32)
            self._refresh_btn.clicked.connect(self.refresh)
            row.addWidget(self._refresh_btn)
            outer.addLayout(row)

            self.refresh()

        # --- Public API ----------------------------------------------------
        def refresh(self) -> None:
            """Re-run discovery and rebuild the combobox.

            Preserves the user's current selection by kind+path
            when possible, falls back to the default-marked source
            when not.
            """
            prev_path: Optional[str] = None
            if self._combo.count() > 0:
                # Preserve current path if user has typed one
                prev_path = self._path_edit.text().strip()
            self._sources = discover_input_sources(
                config_path=self._config_path,
                project_root=self._project_root,
                prefer_order=self._prefer_order,
            )
            self._rebuild_combo(prefer_existing=prev_path)

        def selected_source(self) -> Optional[InputSource]:
            """Return the picked :class:`InputSource`, or ``None``
            when Custom is selected (use :meth:`selected_path`
            then).
            """
            idx = self._combo.currentIndex()
            if 0 <= idx < len(self._sources):
                return self._sources[idx]
            return None

        def selected_kind(self) -> str:
            src = self.selected_source()
            return src.kind if src is not None else SourceKinds.CUSTOM

        def selected_path(self) -> Path:
            """Return the user's chosen directory.

            Raises :class:`ValueError` if the field is empty or the
            path doesn't exist. Caller is responsible for catching
            and surfacing this to the user as a validation error.
            """
            src = self.selected_source()
            if src is not None:
                if not src.path.is_dir():
                    raise ValueError(
                        f"Selected source no longer exists: {src.path}. "
                        f"Click ↻ to refresh."
                    )
                return src.path
            txt = self._path_edit.text().strip()
            if not txt:
                raise ValueError(
                    "No input directory selected — pick from the "
                    "list or type a path."
                )
            p = Path(txt)
            if not p.is_dir():
                raise ValueError(
                    f"Custom path does not exist or is not a "
                    f"directory: {p}"
                )
            return p

        # --- Internals -----------------------------------------------------
        def _rebuild_combo(
            self, *, prefer_existing: Optional[str] = None,
        ) -> None:
            self._combo.blockSignals(True)
            self._combo.clear()
            default_idx = 0
            for i, src in enumerate(self._sources):
                self._combo.addItem(src.label)
                if src.is_default:
                    default_idx = i
            # Always append the Custom sentinel
            self._combo.addItem(_KIND_LABELS[SourceKinds.CUSTOM])
            # Restore prior path selection if it matches a discovered
            # source, else use the default.
            chosen = default_idx
            if prefer_existing:
                for i, src in enumerate(self._sources):
                    if str(src.path) == prefer_existing:
                        chosen = i
                        break
            if not self._sources:
                # Nothing discovered → force Custom mode
                chosen = 0  # the only entry is Custom
            self._combo.setCurrentIndex(chosen)
            self._combo.blockSignals(False)
            self._on_combo_changed(chosen)

        def _on_combo_changed(self, idx: int) -> None:
            if 0 <= idx < len(self._sources):
                src = self._sources[idx]
                self._path_edit.setText(str(src.path))
                self._path_edit.setReadOnly(True)
                self._browse_btn.setEnabled(False)
                self.selection_changed.emit(src)
            else:
                # Custom mode
                self._path_edit.setReadOnly(False)
                self._browse_btn.setEnabled(True)
                # Don't clobber an existing custom path
                self.selection_changed.emit(None)

        def _on_path_edited(self, _text: str) -> None:
            # Only re-emit in custom mode; in concrete-source mode
            # the path edit is just a display.
            if self.selected_source() is None:
                self.selection_changed.emit(None)

        def _on_browse(self) -> None:
            start = self._path_edit.text().strip() or os.getcwd()
            d = QFileDialog.getExistingDirectory(
                self, "Choose input directory", start,
            )
            if d:
                self._path_edit.setText(d)

else:  # pragma: no cover - PySide6 missing

    class InputSourcePicker:  # type: ignore[no-redef]
        """Stub used when PySide6 isn't installed (e.g. CI / smoke
        tests). The pure :func:`discover_input_sources` function is
        unaffected and the rest of this module imports cleanly.
        """
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "InputSourcePicker requires PySide6; install it or "
                "use discover_input_sources() directly."
            )


__all__ = [
    "SourceKinds",
    "InputSource",
    "InputSourcePicker",
    "discover_input_sources",
]
