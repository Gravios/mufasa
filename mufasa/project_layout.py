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
    │   ├── features/<family>/<video>.parquet   # 122ae: per-family
    │   ├── labels/<video>.parquet              # 122ae: split from
    │   │                                       #         targets_inserted
    │   ├── classifications/<run_id>/
    │   └── frames/{extracted,annotated}/
    ├── models/                          # trained classifiers
    │   └── <model_name>/
    └── logs/<run_id>/

Each ``<run_id>`` directory carries a ``run.toml`` file with the
parameters, mufasa version, input file list, and timing. This
makes "what produced this output" answerable a year later.

Patch 122ae departs from the ``<run_id>`` convention for the
**features** and **labels** trees: those operations don't fit
the per-run provenance model (features are updated incrementally
as new family extractors land; labels are hand-edited in place).
They live under flat per-family / per-video subtrees instead.
The run_id convention stays in place for smoothing, outlier
correction, and classifications (the operations that DO have
fresh-extract-from-pose semantics).

Internal storage under ``derived/`` is parquet-only as of the
122ae series. CSV and H5 stay as IMPORT formats — the user's
``import_file_type`` project setting controls what the importers
expect to ingest, not what's stored on disk after extraction.

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


# ---------------------------------------------------------------------------
# Model store (dual-save: project + global cache)
# ---------------------------------------------------------------------------
#
# Two model locations coexist:
#
# * **Global cache** at ``~/.config/mufasa/models/<name>.npz``.
#   Cross-project library; a model trained once is discoverable
#   from any project on the same machine. Flat layout — files,
#   not directories.
#
# * **Project store** at ``<project>/models/<name>/model.npz``
#   with a sibling ``card.toml`` carrying provenance. Each model
#   lives under its own subdirectory so future per-model artifacts
#   (eval reports, training-data manifests, classifier
#   confusion matrices) have a place to land.
#
# Whenever a model crosses from one side to the other it gets
# copied so both have it: training flows mirror saved models to
# the global cache, and loading a model from outside a project
# imports it into the project's store. The project store is the
# source of truth for "what produced this output"; the global
# cache exists to make models easy to find.
#
# Identity is content-hash (SHA-256). If a model with the same
# name already exists at the destination AND its hash matches,
# the import is a no-op (idempotent). If the hash differs the
# import raises ``FileExistsError`` rather than silently
# overwriting — callers decide whether to overwrite or rename.


GLOBAL_MODEL_CACHE_DIRNAME = ".config/mufasa/models"
MODEL_CARD_FILENAME = "card.toml"
MODEL_BLOB_FILENAME = "model.npz"


def global_model_cache_dir() -> Path:
    """Return ``~/.config/mufasa/models/`` (created if missing).

    Tolerates read-only-home and similar OS quirks by returning the
    expected path without raising — callers should still try the
    actual write and handle failure there. Centralizing the path
    here keeps callers in sync if the location ever changes.
    """
    p = Path.home() / GLOBAL_MODEL_CACHE_DIRNAME
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


def file_sha256(path: Path) -> str:
    """SHA-256 of ``path`` as a lowercase hex string.

    Streams the file in 1 MiB chunks; safe for the npz blobs we
    care about here (a few hundred MB at most).
    """
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_name_from_path(path: Path) -> str:
    """Derive the canonical model name from a .npz path.

    Strips ``.npz`` (and the legacy ``v2_`` prefix the form writes
    by default — a model named "v2_model.npz" becomes simply
    "model" which is unhelpful; preserve the "v2_" so it survives
    the round-trip).
    """
    name = path.stem
    if not name:
        raise ValueError(f"Cannot derive model name from {path!r}")
    return name


def import_model_into_project(
    src_path: Path,
    project_root: Path,
    *,
    model_name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Copy ``src_path`` into ``<project>/models/<model_name>/model.npz``
    and write a ``card.toml`` recording where it came from.

    Returns the in-project path to ``model.npz``. Idempotent on
    matching content hash (re-importing the same model is a no-op).
    Raises ``FileExistsError`` if a different-content model with the
    same name already exists, unless ``overwrite=True``.

    The project doesn't need to be fully constructed — ``project_root``
    just needs to exist; this function creates the ``models/`` subtree
    as needed. The caller is responsible for ensuring the project is
    actually a v1 layout (use ``detect_layout`` if unsure).
    """
    import shutil

    src_path = Path(src_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Model file not found: {src_path}")
    project_root = Path(project_root)
    if model_name is None:
        model_name = _model_name_from_path(src_path)

    paths = ProjectPaths(project_root)
    model_dir = paths.model_dir(model_name)
    dst_npz = model_dir / MODEL_BLOB_FILENAME
    card_path = model_dir / MODEL_CARD_FILENAME

    src_hash = file_sha256(src_path)

    if dst_npz.is_file():
        existing_hash = file_sha256(dst_npz)
        if existing_hash == src_hash:
            # Same content → no-op. Touch card.toml's "last seen"
            # if it exists, otherwise leave the dir as-is.
            return dst_npz
        if not overwrite:
            raise FileExistsError(
                f"{dst_npz} already exists with different content "
                f"(existing sha={existing_hash[:12]}, "
                f"new sha={src_hash[:12]}). "
                f"Pass overwrite=True to replace, or supply a "
                f"different model_name."
            )

    # Copy blob, write card.
    shutil.copy2(src_path, dst_npz)
    try:
        import mufasa
        mufasa_version = getattr(mufasa, "__version__", "unknown")
    except Exception:
        mufasa_version = "unknown"
    write_project_toml(card_path, {
        "model_name":     model_name,
        "source_path":    str(src_path),
        "sha256":         src_hash,
        "copied_at":      time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mufasa_version": mufasa_version,
    })
    return dst_npz


def mirror_model_to_global_cache(
    src_path: Path,
    *,
    model_name: Optional[str] = None,
) -> Optional[Path]:
    """Copy ``src_path`` to ``~/.config/mufasa/models/<name>.npz``.

    Returns the cache path on success, ``None`` if the cache dir
    couldn't be written (read-only home, etc.) — failure is silent
    by design; the global cache is a convenience, not a correctness
    requirement. Idempotent on matching content hash.

    If ``src_path`` is already inside the cache, returns it
    unchanged.
    """
    import shutil

    src_path = Path(src_path).resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"Model file not found: {src_path}")
    if model_name is None:
        model_name = _model_name_from_path(src_path)

    cache_dir = global_model_cache_dir()
    dst = (cache_dir / f"{model_name}.npz").resolve()

    # Already in the cache → nothing to do.
    if src_path == dst:
        return dst

    try:
        if dst.is_file() and file_sha256(dst) == file_sha256(src_path):
            return dst
        shutil.copy2(src_path, dst)
        return dst
    except OSError:
        # Read-only / permission / disk-full — degrade silently.
        return None


def resolve_v1_project_root(
    config_path: Optional[str],
) -> Optional[Path]:
    """Best-effort: locate the v1 project root reachable from
    ``config_path``, or ``None``.

    Handles three cases:

    * ``config_path`` is itself ``project.toml`` → parent is the root.
    * ``config_path`` is a legacy ``project_config.ini`` whose
      enclosing directory (or its parent, for the
      ``<project>/project_folder/project_config.ini`` layout) has
      a ``project.toml`` sibling → that directory is the root.
      This is the post-migration case where a project has both
      files transiently.
    * Neither — return ``None``. Forms running under a pure legacy
      project should treat ``None`` as "no v1 store available; skip
      v1-only behaviors."
    """
    if not config_path:
        return None
    cp = Path(config_path)
    candidates: List[Path] = []
    if cp.name == PROJECT_CONFIG_FILENAME:
        candidates.append(cp.parent)
    elif cp.name == "project_config.ini":
        candidates.append(cp.parent)
        candidates.append(cp.parent.parent)
    else:
        # Caller passed something exotic; nothing we can resolve.
        return None
    for c in candidates:
        if (c / PROJECT_CONFIG_FILENAME).is_file():
            return c.resolve()
    return None


# ---------------------------------------------------------------------------
# Layout-agnostic project paths (patch 122f)
# ---------------------------------------------------------------------------
#
# Forms used to reach into ``configparser`` directly to look up
# ``[General settings] project_path`` and then compose
# ``<project_path>/videos/``, ``<project_path>/csv/input_csv/`` etc.
# Those compositions assumed the legacy SimBA layout and break for
# v1 projects (which have ``sources/videos/``, ``sources/pose/``,
# ``derived/...``). The helpers below return the right paths for
# either layout so forms can stop branching themselves.
#
# Detection rule: ``config_path`` ending in ``.toml`` → v1; anything
# else → legacy. Same rule used by :func:`read_config_file`,
# :class:`ConfigReader`, and the InputSourcePicker discovery.


def project_paths_from_config(
    config_path: Union[str, Path],
) -> Dict[str, str]:
    """Return a dict of conventional project paths, working for
    both v1 ``project.toml`` and legacy ``project_config.ini``.

    Keys returned (all values are absolute paths as strings):

    * ``project_root`` — the v1 root (parent of project.toml) for
      v1; the legacy ``project_path`` (read from
      ``[General settings].project_path``) for legacy.
    * ``video_dir`` — ``<root>/sources/videos/`` (v1) or
      ``<project_path>/videos/`` (legacy).
    * ``input_pose_dir`` — ``<root>/sources/pose/`` (v1) or
      ``<project_path>/csv/input_csv/`` (legacy).
    * ``logs_dir`` — ``<root>/logs/`` (v1) or
      ``<project_path>/logs/`` (legacy).
    * ``video_info_path`` — ``<root>/sources/video_info.csv`` (v1)
      or ``<project_path>/logs/video_info.csv`` (legacy).
    * ``models_dir`` — ``<root>/models/`` (v1) or
      ``<project_path>/../models/`` (legacy).

    Patch 122ab originally added four keys consumed by the
    frame-labeller and ROI dialogs. Patch 122ao removed two of
    them (``features_extracted_dir`` + ``targets_inserted_dir``)
    after the v1 migration completed in 122ak — features and
    labels now live exclusively under ``derived/features/`` and
    ``derived/labels/``. The remaining two stay until classifier
    inference output migrates to v1:

    * ``machine_results_dir`` — ``<project_path>/csv/machine_results/``
      (**legacy only**, post-122ax). v1 projects no longer have
      this key — predictions live under
      ``derived_classifications_dir``. Consumers reading this key
      should use ``.get("machine_results_dir")`` and handle None.
      (both layouts).
    * ``roi_definitions_path`` — ``<root>/logs/measures/ROI_definitions.h5``
      (both layouts).

    Rationale for the v1 paths: hand-edited annotations, ROI
    definitions, and per-video classification outputs don't fit
    cleanly into the ``derived/<stage>/<run_id>/`` model that
    suits computed feature stages — they're project-level state,
    edited in place, not the output of a single reproducible
    run. Until the v1 schema decides on canonical homes for
    these (with or without per-edit run-id allocation), the
    backends use the conventional ``csv/`` and ``logs/measures/``
    subtrees under whichever project root is active. This is
    consistent with how 122aa described the labeller's behaviour
    to users.

    Patch 122ae-1 adds two keys that scope the per-family
    parquet trees for derived data:

    * ``derived_features_dir`` — ``<root>/derived/features/``
      (both layouts). Per-family subdirectories live underneath
      (e.g. ``derived/features/distances/<video>.parquet``).
      Writers in patches 122ae-3 and 122ae-4 will target this
      tree; readers from 122ae-2 onwards consult it before
      falling back to the legacy ``csv/features_extracted/``
      subtree. Created by the writer on first use — this
      function returns the path even when the directory doesn't
      exist yet.
    * ``derived_labels_dir`` — ``<root>/derived/labels/`` (both
      layouts). New home for hand-edited classifier labels,
      separated from the legacy ``csv/targets_inserted/`` tree
      (which conflates labels with the features they're paired
      with). The labels split lands in 122ae-3.5; this key
      reserves the path so the directory shape is stable
      from 122ae-1 forward.

    Rationale for the parquet-only direction (122ae series):
    parquet's native column projection makes "load just these
    columns" 5-10× faster than scanning a wide CSV; type round-
    tripping is exact (categoricals stay categorical, ints stay
    ints) where CSVs round-trip everything through strings;
    pyarrow is already a hard install dep, so making parquet
    the only on-disk storage format drops one branch from every
    read_df / write_df call site in the codebase without
    requiring new dependencies. CSV and H5 stay as IMPORT
    formats (the project_metadata_from_config 'file_type' /
    'import_file_type' values), but internal storage under
    ``derived/`` will be parquet-only once the writer patches
    land.

    Caller is responsible for creating directories as needed; this
    function returns paths even when they don't exist on disk.

    :raises ValueError: if ``config_path`` is unreadable / can't be
        resolved to a project at all.
    """
    cp = Path(config_path)
    cp_str = str(cp).lower()
    if cp_str.endswith(".toml"):
        root = cp.parent.resolve()
        return {
            "project_root":           str(root),
            "video_dir":              str(root / "sources" / "videos"),
            "input_pose_dir":         str(root / "sources" / "pose"),
            "logs_dir":               str(root / "logs"),
            "video_info_path":        str(root / "sources" / "video_info.csv"),
            "models_dir":             str(root / "models"),
            # Patch 122ax: machine_results_dir is gone from v1
            # projects. The csv/ subtree no longer exists at all
            # under a v1 layout — predictions live exclusively
            # under derived/classifications/<video>.parquet.
            # This is the last csv/<kind>/ key that was carried
            # through; with it dropped, the v1 layout is fully
            # the new shape (logs/, models/, sources/, derived/).
            "roi_definitions_path":   str(root / "logs" / "measures"
                                          / "ROI_definitions.h5"),
            # Patch 122ae-1: per-family parquet trees for derived data.
            "derived_features_dir":   str(root / "derived" / "features"),
            "derived_labels_dir":     str(root / "derived" / "labels"),
            # Patch 122at: per-video classifier predictions land
            # here (parquet, predictions-only — features stay in
            # derived/features/). Now the sole write location
            # post-122ax (was dual-write until then).
            "derived_classifications_dir":
                str(root / "derived" / "classifications"),
        }
    # Legacy: parse the [General settings] project_path.
    import configparser as _cp
    parser = _cp.ConfigParser()
    try:
        parser.read(config_path)
    except _cp.Error as exc:
        raise ValueError(
            f"Cannot parse legacy project config {config_path}: {exc}"
        )
    project_path = parser.get(
        "General settings", "project_path", fallback="",
    ).strip()
    if not project_path:
        raise ValueError(
            f"{config_path} has no [General settings].project_path. "
            f"Is this a valid Mufasa project_config.ini?"
        )
    proj = Path(project_path)
    return {
        "project_root":           str(proj),
        "video_dir":              str(proj / "videos"),
        "input_pose_dir":         str(proj / "csv" / "input_csv"),
        "logs_dir":               str(proj / "logs"),
        "video_info_path":        str(proj / "logs" / "video_info.csv"),
        "models_dir":             str(proj.parent / "models"),
        # Patch 122ao: features_extracted_dir +
        # targets_inserted_dir keys removed; see v1 branch above
        # for rationale.
        "machine_results_dir":    str(proj / "csv" / "machine_results"),
        "roi_definitions_path":   str(proj / "logs" / "measures"
                                      / "ROI_definitions.h5"),
        # Patch 122ae-1: per-family parquet trees for derived data.
        "derived_features_dir":   str(proj / "derived" / "features"),
        "derived_labels_dir":     str(proj / "derived" / "labels"),
        # Patch 122at: per-video classifier predictions (see v1
        # branch above for rationale). Legacy projects still also
        # write to machine_results_dir during the dual-write era.
        "derived_classifications_dir":
            str(proj / "derived" / "classifications"),
    }


def project_metadata_from_config(
    config_path: Union[str, Path],
) -> Dict[str, Any]:
    """Return the project's metadata (animal count, body parts,
    file type, classifier targets, animal IDs) for either layout.

    Schema of the returned dict:

    * ``animal_count`` — int
    * ``file_type``    — str (``"csv"``, ``"parquet"``, ``"h5"``).
      The format the project's IMPORTER expects to ingest — e.g.
      ``"csv"`` means pose data + third-party annotations arrive
      as CSV files. Going forward (patch 122ae and follow-ups)
      this is decoupled from internal storage, which is always
      parquet under ``derived/``.
    * ``import_file_type`` — str. Alias for ``file_type`` with
      the semantic spelled out. Added in 122ae-1 to disambiguate
      the importer-format from the storage-format now that
      derived data is parquet-only. Same value as ``file_type``;
      both keys present for back-compat across the ~10 consumers
      that already read ``file_type``.
    * ``body_parts``   — list[str], in order
    * ``animal_ids``   — list[str]
    * ``classifier_targets`` — list[str]
    * ``pose_config_code`` — str (the legacy preset code, e.g.
      ``"7"``, ``"user_defined"``)

    Used by forms that need project metadata without going through
    the full :class:`ConfigReader` (which pulls cv2, h5py, etc.).

    :raises ValueError: if the config can't be parsed.
    """
    cp = Path(config_path)
    cp_str = str(cp).lower()
    if cp_str.endswith(".toml"):
        try:
            data = read_project_toml(cp)
        except Exception as exc:
            raise ValueError(
                f"Cannot parse v1 project.toml {config_path}: {exc}"
            )
        pose = data.get("pose", {})
        classifiers = data.get("classifiers", {})
        file_type_val = str(pose.get("file_type", "csv"))
        return {
            "animal_count":       int(pose.get("animal_count", 1)),
            "file_type":          file_type_val,
            "import_file_type":   file_type_val,   # 122ae-1 alias
            "body_parts":         list(pose.get("body_parts", [])),
            "animal_ids":         list(pose.get("animal_ids", [])),
            "classifier_targets": list(classifiers.get("targets", [])),
            "pose_config_code":   str(pose.get(
                "pose_config_code", "user_defined",
            )),
        }
    # Legacy
    import configparser as _cp
    parser = _cp.ConfigParser()
    try:
        parser.read(config_path)
    except _cp.Error as exc:
        raise ValueError(
            f"Cannot parse legacy project config {config_path}: {exc}"
        )
    # animal_no
    animal_count = parser.getint(
        "General settings", "animal_no", fallback=1,
    )
    file_type = parser.get(
        "General settings", "file_type", fallback="csv",
    )
    # Classifier names
    n_targets = parser.getint(
        "SML settings", "no_targets", fallback=0,
    )
    classifier_targets = [
        parser.get("SML settings", f"target_name_{i+1}", fallback="")
        for i in range(n_targets)
    ]
    classifier_targets = [t for t in classifier_targets if t]
    # animal IDs — comma-separated in [Multi animal IDs] id_list
    id_list = parser.get(
        "Multi animal IDs", "id_list", fallback="",
    ).strip()
    if id_list and id_list != "NaN" and id_list != "None":
        animal_ids = [s.strip() for s in id_list.split(",") if s.strip()]
    else:
        animal_ids = [f"Animal_{i+1}" for i in range(animal_count)]
    # body_parts — read from logs/measures/pose_configs/bp_names/
    project_path = parser.get(
        "General settings", "project_path", fallback="",
    )
    body_parts: List[str] = []
    if project_path:
        bp_csv = (
            Path(project_path)
            / "logs" / "measures" / "pose_configs"
            / "bp_names" / "project_bp_names.csv"
        )
        if bp_csv.is_file():
            try:
                text = bp_csv.read_text().strip()
            except OSError:
                text = ""
            if text:
                # Tolerate both one-per-line and old-style
                # comma-separated single-row formats. Older
                # SimBA projects use the comma row; newer use
                # one body-part per line. Pick by inspecting the
                # first line.
                first_line = text.splitlines()[0]
                if "," in first_line:
                    body_parts = [
                        x.strip() for x in first_line.split(",")
                        if x.strip()
                    ]
                else:
                    body_parts = [
                        ln.strip() for ln in text.splitlines()
                        if ln.strip()
                    ]
    pose_config_code = parser.get(
        "create ensemble settings",
        "pose_estimation_body_parts",
        fallback="user_defined",
    )
    return {
        "animal_count":       animal_count,
        "file_type":          file_type,
        "import_file_type":   file_type,   # 122ae-1 alias
        "body_parts":         body_parts,
        "animal_ids":         animal_ids,
        "classifier_targets": classifier_targets,
        "pose_config_code":   pose_config_code,
    }


# ---------------------------------------------------------------------------
# Patch 122as: classifier-inference + classifier-training settings I/O
# ---------------------------------------------------------------------------
#
# Bridges the gap between the per-form settings the Qt forms collect
# (RunInferenceForm in 122ap, TrainClassifierForm in 122aq) and the
# storage they land in: ``[classifier_inference]`` / ``[classifier_training]``
# sub-tables in ``project.toml`` for v1 projects, and the canonical
# ``[SML settings]`` / ``[threshold_settings]`` / ``[Minimum_bout_lengths]``
# / ``[create_ensemble_settings]`` sections in ``project_config.ini``
# for legacy projects.
#
# Forms call these helpers without branching on file extension. The
# helpers detect the format and dispatch internally.
#
# v1 TOML schemas
# ---------------
# Inference settings live as per-classifier sub-tables:
#
#     [classifier_inference.attack]
#     model_path = "/abs/path/to/attack.sav"
#     threshold = 0.5
#     min_bout_ms = 200
#
#     [classifier_inference.grooming]
#     model_path = "/abs/path/to/grooming.sav"
#     threshold = 0.4
#     min_bout_ms = 100
#
# Training settings live as a flat table with the same key names the
# trainer's legacy reader expects (so the TOML→CP translator can dump
# them verbatim):
#
#     [classifier_training]
#     model_to_run = "RF"
#     classifier = "attack"
#     rf_n_estimators = 2000
#     rf_max_features = "sqrt"
#     ...
#     calculate_shap_scores = false
#     shap_target_present_no = 100
#     ...

import configparser as _cp_mod


def _is_toml_config(config_path) -> bool:
    return str(config_path).lower().endswith(".toml")


# Mapping from form-friendly TOML key → INI section + key shape for
# inference. The INI section/key shape was set in stone by the legacy
# Tk popup and the backend that reads it back.
_INFERENCE_INI_MAP = {
    "model_path":  ("SML settings",         "model_path_{idx}"),
    "threshold":   ("threshold_settings",   "threshold_{idx}"),
    "min_bout_ms": ("Minimum_bout_lengths", "min_bout_{idx}"),
}


def read_classifier_inference_settings(
    config_path,
) -> Dict[str, Dict[str, Any]]:
    """Return per-classifier inference settings (model path,
    discrimination threshold, minimum bout length in ms).

    Keys of the outer dict are classifier names. Missing classifiers
    are simply absent from the result; the form will leave their
    fields blank. Empty / NaN values from legacy INI projects are
    also skipped so the form starts blank when the user hasn't run
    inference before.

    Patch 122as. Replaces the form-internal
    ``_read_existing_ini_settings`` shim from 122ap with a
    layout-aware helper that handles both v1 TOML and legacy INI
    projects.
    """
    cp = Path(config_path)
    out: Dict[str, Dict[str, Any]] = {}
    if _is_toml_config(cp):
        try:
            data = read_project_toml(cp)
        except Exception:
            return out
        section = data.get("classifier_inference", {})
        if not isinstance(section, dict):
            return out
        for clf_name, settings in section.items():
            if not isinstance(settings, dict):
                continue
            row: Dict[str, Any] = {}
            mp = settings.get("model_path")
            if isinstance(mp, str) and mp:
                row["model_path"] = mp
            thr = settings.get("threshold")
            if isinstance(thr, (int, float)):
                row["threshold"] = float(thr)
            mb = settings.get("min_bout_ms")
            if isinstance(mb, int):
                row["min_bout_ms"] = int(mb)
            if row:
                out[str(clf_name)] = row
        return out
    # Legacy INI
    parser = _cp_mod.ConfigParser()
    try:
        parser.read(config_path)
    except _cp_mod.Error:
        return out
    n_targets = parser.getint(
        "SML settings", "no_targets", fallback=0,
    )
    for idx in range(1, n_targets + 1):
        name = parser.get(
            "SML settings", f"target_name_{idx}", fallback="",
        ).strip()
        if not name:
            continue
        row: Dict[str, Any] = {}
        mp = parser.get(
            "SML settings", f"model_path_{idx}", fallback="",
        ).strip()
        if mp and mp.lower() not in ("nan", "none", "select model (.sav) file"):
            row["model_path"] = mp
        thr_text = parser.get(
            "threshold_settings", f"threshold_{idx}", fallback="",
        ).strip()
        if thr_text and thr_text.lower() not in ("nan", "none", ""):
            try:
                row["threshold"] = float(thr_text)
            except ValueError:
                pass
        mb_text = parser.get(
            "Minimum_bout_lengths", f"min_bout_{idx}", fallback="",
        ).strip()
        if mb_text and mb_text.lower() not in ("nan", "none", ""):
            try:
                row["min_bout_ms"] = int(mb_text)
            except ValueError:
                pass
        if row:
            out[name] = row
    return out


def write_classifier_inference_settings(
    config_path,
    settings: Dict[str, Dict[str, Any]],
) -> None:
    """Persist per-classifier inference settings.

    For v1 ``.toml`` projects, writes ``[classifier_inference.<name>]``
    sub-tables. For legacy ``.ini`` projects, writes into the
    canonical sections using the classifier ordinal.

    Each entry's dict may have any subset of ``model_path``,
    ``threshold``, and ``min_bout_ms``. Missing keys are not
    written (preserving any earlier values).

    Patch 122as. Replaces the form-internal ``_write_settings_to_ini``
    shim from 122ap, which raised RuntimeError on TOML projects.
    """
    cp = Path(config_path)
    if _is_toml_config(cp):
        try:
            data = read_project_toml(cp)
        except FileNotFoundError:
            data = {"project_layout_version": PROJECT_LAYOUT_VERSION}
        section = dict(data.get("classifier_inference", {}))
        for clf_name, row in settings.items():
            entry = dict(section.get(clf_name, {}))
            if "model_path" in row:
                entry["model_path"] = str(row["model_path"])
            if "threshold" in row:
                entry["threshold"] = float(row["threshold"])
            if "min_bout_ms" in row:
                entry["min_bout_ms"] = int(row["min_bout_ms"])
            section[clf_name] = entry
        data["classifier_inference"] = section
        write_project_toml(cp, data)
        return
    # Legacy INI
    parser = _cp_mod.ConfigParser()
    parser.read(config_path)
    n_targets = parser.getint(
        "SML settings", "no_targets", fallback=0,
    )
    # Build name → ordinal map from the existing INI
    name_to_idx: Dict[str, int] = {}
    for idx in range(1, n_targets + 1):
        name = parser.get(
            "SML settings", f"target_name_{idx}", fallback="",
        ).strip()
        if name:
            name_to_idx[name] = idx
    for sec in ("SML settings", "threshold_settings",
                "Minimum_bout_lengths"):
        if not parser.has_section(sec):
            parser.add_section(sec)
    for clf_name, row in settings.items():
        idx = name_to_idx.get(clf_name)
        if idx is None:
            # Classifier not registered in [SML settings] —
            # skip rather than corrupt the ordinal sequence.
            # Callers should use ClassifierManageForm to add it
            # before persisting inference settings.
            continue
        if "model_path" in row:
            parser.set(
                "SML settings", f"model_path_{idx}",
                str(row["model_path"]),
            )
        if "threshold" in row:
            parser.set(
                "threshold_settings", f"threshold_{idx}",
                str(row["threshold"]),
            )
        if "min_bout_ms" in row:
            parser.set(
                "Minimum_bout_lengths", f"min_bout_{idx}",
                str(row["min_bout_ms"]),
            )
    with open(config_path, "w") as fp:
        parser.write(fp)


def read_classifier_training_settings(
    config_path,
) -> Dict[str, Any]:
    """Return classifier-training hyperparameters + evaluation
    toggles as a flat dict. Keys match the canonical names the
    trainer's legacy reader expects (``rf_n_estimators``,
    ``rf_max_features``, etc.). Missing keys are absent from the
    result so callers can distinguish 'never set' from a stored
    default.

    Bool-typed values from TOML are returned as bool. Int / float
    values from TOML are returned as those types. Legacy INI
    projects return everything as str (the form casts to the
    appropriate type on its end, same as it did before 122as).

    Patch 122as. Replaces the form-internal ``_load_from_ini`` shim
    from 122aq for the TOML case.
    """
    cp = Path(config_path)
    out: Dict[str, Any] = {}
    if _is_toml_config(cp):
        try:
            data = read_project_toml(cp)
        except Exception:
            return out
        section = data.get("classifier_training", {})
        if not isinstance(section, dict):
            return out
        # Only copy scalar values; nested tables aren't part of
        # the schema for training settings.
        for key, value in section.items():
            if isinstance(value, (str, int, float, bool)):
                out[key] = value
        return out
    # Legacy INI — return string values, caller casts.
    parser = _cp_mod.ConfigParser()
    try:
        parser.read(config_path)
    except _cp_mod.Error:
        return out
    sec = "create_ensemble_settings"
    if not parser.has_section(sec):
        # Old projects spelled it 'create ensemble settings' with
        # spaces; the toml_to_configparser translator also uses that
        # form. Fall back to the spaced section name.
        sec = "create ensemble settings"
    if parser.has_section(sec):
        for key, value in parser.items(sec):
            out[key] = value
    return out


def write_classifier_training_settings(
    config_path,
    settings: Dict[str, Any],
) -> None:
    """Persist classifier-training settings.

    For v1 ``.toml`` projects, writes the flat
    ``[classifier_training]`` section. For legacy ``.ini`` projects,
    writes into ``[create_ensemble_settings]``.

    Pass values in their natural types (bool / int / float / str).
    The legacy INI writer stringifies everything; the TOML writer
    preserves types.

    Patch 122as. Replaces the form-internal ``_write_to_ini``
    shim from 122aq, which raised RuntimeError on TOML projects.
    """
    cp = Path(config_path)
    if _is_toml_config(cp):
        try:
            data = read_project_toml(cp)
        except FileNotFoundError:
            data = {"project_layout_version": PROJECT_LAYOUT_VERSION}
        existing = dict(data.get("classifier_training", {}))
        existing.update(settings)
        data["classifier_training"] = existing
        write_project_toml(cp, data)
        return
    # Legacy INI
    parser = _cp_mod.ConfigParser()
    parser.read(config_path)
    sec = "create_ensemble_settings"
    if not parser.has_section(sec):
        parser.add_section(sec)
    for key, value in settings.items():
        parser.set(sec, key, str(value))
    with open(config_path, "w") as fp:
        parser.write(fp)
