"""Patch 122a: Mufasa project layout (v1).

The legacy SimBA-derived layout bakes specific pipeline stages
into directory names (``csv/outlier_corrected_movement_location``
etc), has no separation between user-supplied source data and
Mufasa-generated derived data, and offers no per-run isolation
or provenance tracking. Stage outputs overwrite one another;
re-running the same stage with different parameters destroys
the previous result.

This module defines a replacement layout::

    <project_root>/
    ├── project.toml                     # was project_config.ini
    ├── README.md                        # optional
    ├── sources/                         # read-only inputs; never written
    │   ├── videos/
    │   ├── pose/                        # raw tracker output (DLC / SLEAP)
    │   └── annotations/
    ├── derived/                         # generated; safe to delete
    │   ├── smoothed/<flavor>/<run_id>/
    │   ├── outlier_corrected/<run_id>/
    │   ├── features/<run_id>/
    │   ├── classifications/<run_id>/
    │   └── frames/{extracted,annotated}/
    ├── models/                          # trained classifiers
    │   └── <model_name>/
    └── logs/<run_id>/

Each ``<run_id>`` directory carries a ``run.toml`` file with the
parameters, mufasa version, input file list, and timing. This
makes "what produced this output" answerable a year later.

Backward compatibility with the legacy layout lives in
``mufasa.legacy_layout``; a migration tool sits at
``mufasa.cli.migrate_project``. Code that doesn't know which
layout it's looking at should call :func:`detect_layout` first.
"""
from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


# Layout version written into project.toml. Bump when the
# layout changes incompatibly. ProjectPaths refuses to operate
# on a project with a higher version than this code understands.
PROJECT_LAYOUT_VERSION = 1

PROJECT_CONFIG_FILENAME = "project.toml"
RUN_PROVENANCE_FILENAME = "run.toml"

# Canonical stage names used under ``derived/``. Centralizing
# these prevents typos and gives downstream code (e.g. the
# workbench's "Load saved model" picker) a stable list to
# enumerate.
class Stages:
    SMOOTHED = "smoothed"
    OUTLIER_CORRECTED = "outlier_corrected"
    FEATURES = "features"
    CLASSIFICATIONS = "classifications"
    FRAMES = "frames"
    ANNOTATIONS = "annotations"


# Sub-flavor naming under ``derived/smoothed/``. The v2
# Kalman smoother and any future smoothers each get their
# own subfolder so per-flavor models, runs, and configs
# don't collide.
class SmoothingFlavors:
    KALMAN_V2 = "kalman_v2"
    SAVITZKY_GOLAY = "savitzky_golay"
    GAUSSIAN = "gaussian"


# ---------------------------------------------------------------------------
# Run id
# ---------------------------------------------------------------------------

# A run id is "YYYYMMDD-HHMMSS-<6 hex chars>". The hex suffix
# disambiguates runs started within the same second (parallel
# launches, automation). Sortable lexically == sortable
# chronologically, which makes "latest run" cheap.
_RUN_ID_PATTERN = re.compile(
    r"^\d{8}-\d{6}-[0-9a-f]{6}$"
)


def generate_run_id() -> str:
    """Return a fresh run id.

    Format: ``YYYYMMDD-HHMMSS-XXXXXX`` (timestamp then 6 hex
    characters). Sortable by string, monotonic-ish across
    parallel launches.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{ts}-{suffix}"


def is_run_id(name: str) -> bool:
    """True if ``name`` matches the run-id format."""
    return bool(_RUN_ID_PATTERN.match(name))


# ---------------------------------------------------------------------------
# project.toml read/write
# ---------------------------------------------------------------------------

def read_project_toml(path: Path) -> Dict[str, Any]:
    """Parse ``project.toml`` into a dict.

    Raises FileNotFoundError if the file is missing, and
    ProjectLayoutError if the file's ``project_layout_version``
    is higher than this code understands.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    version = data.get("project_layout_version", 0)
    if not isinstance(version, int) or version < 1:
        raise ProjectLayoutError(
            f"{path}: missing or invalid project_layout_version "
            f"(got {version!r})"
        )
    if version > PROJECT_LAYOUT_VERSION:
        raise ProjectLayoutError(
            f"{path}: project_layout_version={version} is newer "
            f"than this Mufasa supports "
            f"({PROJECT_LAYOUT_VERSION}). Upgrade Mufasa."
        )
    return data


def write_project_toml(path: Path, data: Dict[str, Any]) -> None:
    """Write ``data`` to ``project.toml`` using a focused TOML
    writer (avoids a third-party tomli_w dependency).

    Supports the subset of TOML the schema actually uses:
    scalars (str, int, float, bool), datetimes-as-strings,
    flat string-keyed tables, lists of scalars or strings,
    and dotted-section tables one level deep. Throws
    ``TypeError`` on anything outside that subset so we fail
    loudly rather than silently producing malformed TOML.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _format_toml(data)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _format_toml(data: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    # Bare keys (no section) come first
    bare = {k: v for k, v in data.items() if not isinstance(v, dict)}
    sectioned = {k: v for k, v in data.items() if isinstance(v, dict)}
    for k, v in bare.items():
        out.append(f"{k} = {_format_toml_value(v)}")
    if bare and sectioned:
        out.append("")
    for sec_name, sec_body in sectioned.items():
        # Nested-once: section body is itself a dict, possibly
        # with one more level of subsections under it.
        if not isinstance(sec_body, dict):
            raise TypeError(
                f"section {sec_name!r} must be a dict, "
                f"got {type(sec_body).__name__}"
            )
        flat = {
            k: v for k, v in sec_body.items()
            if not isinstance(v, dict)
        }
        subs = {
            k: v for k, v in sec_body.items()
            if isinstance(v, dict)
        }
        out.append(f"[{sec_name}]")
        for k, v in flat.items():
            out.append(f"{k} = {_format_toml_value(v)}")
        out.append("")
        for sub_name, sub_body in subs.items():
            out.append(f"[{sec_name}.{sub_name}]")
            for k, v in sub_body.items():
                if isinstance(v, dict):
                    raise TypeError(
                        f"nested-twice tables not supported "
                        f"(at {sec_name}.{sub_name}.{k})"
                    )
                out.append(f"{k} = {_format_toml_value(v)}")
            out.append("")
    # Trim trailing blank
    while out and out[-1] == "":
        out.pop()
    return out


def _format_toml_value(v: Any) -> str:
    if v is None:
        # TOML has no null; emit empty string. Callers should
        # avoid None for fields they care about — schema-level
        # absence is the right way to encode "no value."
        return '""'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return _toml_string(v)
    if isinstance(v, (list, tuple)):
        return (
            "["
            + ", ".join(_format_toml_value(x) for x in v)
            + "]"
        )
    raise TypeError(
        f"unsupported TOML value type {type(v).__name__}: {v!r}"
    )


def _toml_string(s: str) -> str:
    # Basic-string with escaping for the characters TOML
    # requires. Avoids edge cases by always quoting.
    escapes = {"\\": "\\\\", '"': '\\"', "\n": "\\n",
               "\t": "\\t", "\r": "\\r"}
    out = []
    for ch in s:
        out.append(escapes.get(ch, ch))
    return '"' + "".join(out) + '"'


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

class ProjectLayoutError(RuntimeError):
    """Raised when a project directory has an unsupported,
    malformed, or ambiguous layout.
    """


def detect_layout(path: Path) -> str:
    """Inspect ``path`` and return either ``"v1"``,
    ``"legacy"``, or ``"unknown"``.

    ``"v1"`` — has ``project.toml`` with a valid
    ``project_layout_version``.
    ``"legacy"`` — has SimBA-style ``project_folder/
    project_config.ini``.
    ``"unknown"`` — neither.
    """
    path = Path(path)
    if (path / PROJECT_CONFIG_FILENAME).is_file():
        return "v1"
    if (path / "project_folder" / "project_config.ini").is_file():
        return "legacy"
    # Edge case: caller might pass the project_folder itself
    # rather than its parent.
    if (path / "project_config.ini").is_file():
        return "legacy"
    return "unknown"


# ---------------------------------------------------------------------------
# ProjectPaths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectPaths:
    """Resolves all paths within a v1-layout project.

    The class is constructed from a project root and exposes
    each subdirectory as an attribute or method. Methods that
    accept a ``run_id`` create the run directory if it doesn't
    exist; passing ``None`` allocates a fresh id.

    Example::

        paths = ProjectPaths(Path("/data/rats/exp1"))
        out_dir = paths.smoothed_run_dir(
            SmoothingFlavors.KALMAN_V2,
        )  # auto-generates a run id
        write_run_toml(out_dir / RUN_PROVENANCE_FILENAME, {
            "stage": "smoothed.kalman_v2",
            "params": {"em_max_iter": 20},
        })
    """

    root: Path

    @classmethod
    def open(cls, root: Path) -> "ProjectPaths":
        """Validate that ``root`` is a v1 project, return a
        ProjectPaths.

        Raises ProjectLayoutError if the layout doesn't match.
        Use ``ProjectPaths(root)`` directly if you don't need
        validation (e.g. to construct paths before the
        project exists).
        """
        root = Path(root).resolve()
        kind = detect_layout(root)
        if kind != "v1":
            raise ProjectLayoutError(
                f"{root}: layout is {kind!r}, expected v1. "
                f"Run `python -m mufasa.cli.migrate_project "
                f"{root}` to migrate from legacy layout."
            )
        return cls(root)

    # ---- Top-level dirs ----
    @property
    def config_file(self) -> Path:
        return self.root / PROJECT_CONFIG_FILENAME

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    # ---- sources/ ----
    @property
    def sources_videos(self) -> Path:
        return self.sources_dir / "videos"

    @property
    def sources_pose(self) -> Path:
        return self.sources_dir / "pose"

    @property
    def sources_annotations(self) -> Path:
        return self.sources_dir / "annotations"

    # ---- derived/<stage>/<flavor>/<run_id>/ ----
    def stage_dir(self, stage: str) -> Path:
        return self.derived_dir / stage

    def stage_run_dir(
        self,
        stage: str,
        run_id: Optional[str] = None,
        flavor: Optional[str] = None,
    ) -> Path:
        """Return ``derived/<stage>[/<flavor>]/<run_id>/``,
        creating it if missing.

        ``run_id=None`` allocates a fresh id via
        :func:`generate_run_id`. Pass an explicit string to
        reuse an existing run dir.
        """
        if run_id is None:
            run_id = generate_run_id()
        parts: List[Path] = [self.derived_dir, Path(stage)]
        if flavor:
            parts.append(Path(flavor))
        parts.append(Path(run_id))
        path = Path(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def smoothed_run_dir(
        self,
        flavor: str,
        run_id: Optional[str] = None,
    ) -> Path:
        """Convenience for ``derived/smoothed/<flavor>/<run_id>/``."""
        return self.stage_run_dir(
            Stages.SMOOTHED, run_id=run_id, flavor=flavor,
        )

    # ---- Run enumeration ----
    def list_runs(
        self,
        stage: str,
        flavor: Optional[str] = None,
    ) -> List[Path]:
        """Return all run directories under ``stage[/flavor]``,
        sorted lexically (= chronologically by run-id format).

        Skips entries that don't look like run ids, so any
        ad-hoc subdirs (``latest``, ``scratch``, ``imported``)
        coexist without polluting the run list.
        """
        base = self.stage_dir(stage)
        if flavor:
            base = base / flavor
        if not base.is_dir():
            return []
        runs = [
            p for p in base.iterdir()
            if p.is_dir() and is_run_id(p.name)
        ]
        runs.sort(key=lambda p: p.name)
        return runs

    def latest_run(
        self,
        stage: str,
        flavor: Optional[str] = None,
    ) -> Optional[Path]:
        runs = self.list_runs(stage, flavor=flavor)
        return runs[-1] if runs else None

    # ---- Logs ----
    def log_dir(self, run_id: str) -> Path:
        path = self.logs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---- Models ----
    def model_dir(self, model_name: str) -> Path:
        path = self.models_dir / model_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---- Construction helpers ----
    def ensure_skeleton(self) -> None:
        """Create the top-level directory skeleton.

        Idempotent. Use after migrating or initializing a fresh
        project. Doesn't write project.toml — caller is
        responsible for that, since the config depends on what
        the user passed in.
        """
        for d in (
            self.sources_videos, self.sources_pose,
            self.sources_annotations,
            self.derived_dir, self.models_dir, self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# run.toml provenance
# ---------------------------------------------------------------------------

def write_run_toml(
    path: Path,
    *,
    stage: str,
    run_id: str,
    inputs: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
    mufasa_version: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a run-provenance file at ``path``.

    ``path`` should typically be the run directory's
    ``run.toml`` file. The schema captures enough information
    for "what produced this output" to be answerable: stage,
    timestamp, inputs (file paths with optional hashes),
    parameters (every non-default param), and results
    (counts, timings, convergence flags). All fields except
    ``stage`` and ``run_id`` are optional.
    """
    if mufasa_version is None:
        try:
            import mufasa
            mufasa_version = getattr(mufasa, "__version__", "unknown")
        except Exception:
            mufasa_version = "unknown"
    data: Dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mufasa_version": mufasa_version,
    }
    if inputs:
        data["inputs"] = {"files": list(inputs)}
    if params:
        data["params"] = dict(params)
    if results:
        data["results"] = dict(results)
    if extra:
        data.update(extra)
    write_project_toml(path, data)


def read_run_toml(path: Path) -> Dict[str, Any]:
    """Parse a run-provenance file.

    Reuses ``tomllib`` directly (no schema enforcement beyond
    being valid TOML). Callers can validate the fields they
    care about.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)
