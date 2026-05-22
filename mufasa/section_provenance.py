"""mufasa/section_provenance.py — workflow-section provenance tracking.

Patch 122ds. Records when each workflow section ran so the UI can
surface checkmarks (✓ section is up to date) and stale-circles
(● section needs to re-run because an upstream section ran more
recently). State lives in ``project.toml`` under a ``[provenance]``
table::

    [provenance.import_pose]
    last_run_id = "20260427-184523-a1b2c3"
    last_run_at = "2026-04-27T18:45:23+00:00"

    [provenance.kalman_v2]
    last_run_id = "20260520-233610-6203f1"
    last_run_at = "2026-05-20T23:36:10+00:00"

The section DAG (identity, display, dependencies) is declared
centrally in this module's :data:`SECTIONS` constant — the
authoritative source the staleness check consults.

Module surface
==============

* :class:`SectionSpec` — frozen dataclass describing one section.
* :class:`SectionStatus` — enum of UI states (UNKNOWN/CURRENT/STALE).
* :data:`SECTIONS` — ``dict[str, SectionSpec]``. The DAG.
* :func:`record_run` — backend-side; updates ``[provenance.<id>]`` on
  successful completion of a section.
* :func:`get_status` — workbench-side; returns the current
  :class:`SectionStatus` for one section.
* :func:`get_all_statuses` — convenience helper; returns the full
  mapping for the page-show / project-change refresh path.

Staleness rules
===============

Per-section status is computed as follows:

* No ``last_run_at`` for this section → :data:`SectionStatus.UNKNOWN`.
* ``last_run_at`` present, AND every declared dependency that ALSO
  has a known ``last_run_at`` ran at or before this section →
  :data:`SectionStatus.CURRENT`.
* ``last_run_at`` present, AND at least one declared dependency that
  has a known ``last_run_at`` ran after this section →
  :data:`SectionStatus.STALE`.

Dependencies whose provenance is *unknown* are **ignored** in the
staleness check. This is intentional — during the rollout of the
backend wiring (patch 122dt) most sections will not yet call
:func:`record_run`, so most provenance entries will be missing.
Treating those as "STALE" would be alarmist; treating them as
"freshly run" would be misleading. Ignoring them surfaces only
verifiable staleness signals: a section is STALE only when we
can *prove* a parent ran later.

Migrated projects start with no provenance at all (patch 122dt's
``migrate_project`` retro-fill will populate it). Until that lands,
every section reads UNKNOWN — the UI just shows no badges, which is
the right behavior for "we don't know yet."

DAG validation
==============

:data:`SECTIONS` is validated at module import time:

* Every ``depends_on`` element references a real section_id.
* The dependency graph is acyclic.

Either failure raises ``ProvenanceError`` at import — fail loud and
early rather than surface mysterious behavior at run time.

Threading / concurrency
=======================

:func:`record_run` is a read-modify-write on ``project.toml``. The
workbench is single-threaded UI code, and backends run sequentially
in the form-completion path, so concurrent calls aren't a concern in
the current architecture. If a future architecture introduces
parallel backend execution that touches the same project, this
function will need a file-locking strategy (``fcntl.flock`` on POSIX,
or moving provenance to a sidecar file with atomic-rename writes).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from mufasa.project_layout import read_project_toml, write_project_toml


class ProvenanceError(Exception):
    """Raised for declaration-time or runtime errors in this module."""


class SectionStatus(Enum):
    """UI-facing status for a workflow section."""
    UNKNOWN = "unknown"      # never recorded — no badge
    CURRENT = "current"      # ✓ up-to-date relative to dependencies
    STALE = "stale"          # ● a dependency ran after this section


@dataclass(frozen=True)
class SectionSpec:
    """One workflow section's identity and dependencies.

    Attributes
    ----------
    section_id
        Stable snake_case key. Used as the TOML key under
        ``[provenance.<section_id>]`` and as the lookup handle from
        backend code. Never change an existing ID without a migration
        plan — old project.toml files will reference the old name.
    page
        Display name of the workbench page the section lives on.
        Currently informational; future UI lookup will use it.
    section_title
        Display title of the QGroupBox the badge will attach to.
        Must match exactly for the future UI lookup to find the box.
    depends_on
        Tuple of upstream section_ids. Self-references and cycles are
        rejected at module import time.
    """
    section_id: str
    page: str
    section_title: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# The DAG.
# ---------------------------------------------------------------------------
#
# Sections are listed in roughly the order they appear in the workbench
# sidebar (Data Import → Preprocessing → ROI → Features → Classifier).
# Editing this dict:
#
#   - Adding a section: pick a stable snake_case ID, declare it, list
#     parents. No backend-code change is required just to make it
#     visible — the workbench surfaces UNKNOWN for any declared
#     section whose backend doesn't yet call ``record_run``.
#
#   - Removing a section: prefer to leave the declaration in place
#     for at least one release cycle so old project.toml files'
#     ``[provenance.<id>]`` entries don't cause confusion.
#
#   - Changing dependencies: safe at any time — staleness is recomputed
#     on every page-show / completion event.

SECTIONS: dict[str, SectionSpec] = {
    "import_pose": SectionSpec(
        section_id="import_pose",
        page="Data Import",
        section_title="Import pose data",
        depends_on=(),
    ),
    "pixels_per_mm": SectionSpec(
        section_id="pixels_per_mm",
        page="Pose cleanup",
        section_title="Pixels-per-mm calibration",
        depends_on=(),
    ),
    "interpolate": SectionSpec(
        section_id="interpolate",
        page="Pose cleanup",
        section_title="Interpolate missing frames",
        depends_on=("import_pose",),
    ),
    "kalman_v2": SectionSpec(
        section_id="kalman_v2",
        page="Pose cleanup",
        section_title="Kalman v2 smoother",
        depends_on=("import_pose",),
    ),
    "outlier_correction": SectionSpec(
        section_id="outlier_correction",
        page="Pose cleanup",
        section_title="Run outlier correction",
        depends_on=("import_pose",),
    ),
    "savitzky_golay": SectionSpec(
        section_id="savitzky_golay",
        page="Pose cleanup",
        section_title="Savitzky-Golay smoother (legacy)",
        depends_on=("outlier_correction",),
    ),
    "egocentric": SectionSpec(
        section_id="egocentric",
        page="Pose cleanup",
        section_title="Egocentric alignment",
        depends_on=("outlier_correction",),
    ),
    "drop_body_parts": SectionSpec(
        section_id="drop_body_parts",
        page="Pose cleanup",
        section_title="Drop body parts",
        depends_on=("import_pose",),
    ),
    "roi_definitions": SectionSpec(
        section_id="roi_definitions",
        page="ROI",
        section_title="Definitions",
        depends_on=("pixels_per_mm",),
    ),
    "features_subject": SectionSpec(
        section_id="features_subject",
        page="Features",
        section_title="Subject features",
        depends_on=("outlier_correction",),
    ),
    "features_roi": SectionSpec(
        section_id="features_roi",
        page="Features",
        section_title="ROI features",
        depends_on=("outlier_correction", "roi_definitions"),
    ),
    "annotation": SectionSpec(
        section_id="annotation",
        page="Annotation",
        section_title="Annotate",
        depends_on=("features_subject",),
    ),
    "classifier_train": SectionSpec(
        section_id="classifier_train",
        page="Classifier",
        section_title="Train classifier",
        depends_on=("features_subject", "annotation"),
    ),
    "classifier_run": SectionSpec(
        section_id="classifier_run",
        page="Classifier",
        section_title="Run classifier",
        depends_on=("classifier_train",),
    ),
}


def _validate_dag(sections: Mapping[str, SectionSpec]) -> None:
    """Validate ``sections`` at import time.

    Checks:
      1. Every section_id matches its dict key (no copy-paste typos).
      2. No self-dependencies.
      3. Every depends_on element references a real section_id.
      4. The dependency graph is acyclic (DFS-based cycle detection).
    """
    # 1 & 2 & 3 — local consistency.
    for key, spec in sections.items():
        if spec.section_id != key:
            raise ProvenanceError(
                f"SECTIONS key {key!r} disagrees with "
                f"SectionSpec.section_id {spec.section_id!r}"
            )
        for dep in spec.depends_on:
            if dep == spec.section_id:
                raise ProvenanceError(
                    f"section {spec.section_id!r} depends on itself"
                )
            if dep not in sections:
                raise ProvenanceError(
                    f"section {spec.section_id!r} depends on "
                    f"undeclared section {dep!r}"
                )

    # 4 — DFS cycle detection. Each node has a 3-state color: white
    # (unvisited), gray (on current DFS path), black (fully explored).
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {key: WHITE for key in sections}

    def visit(node: str, stack: tuple[str, ...]) -> None:
        if color[node] == GRAY:
            cycle = " -> ".join((*stack, node))
            raise ProvenanceError(
                f"dependency cycle detected: {cycle}"
            )
        if color[node] == BLACK:
            return
        color[node] = GRAY
        for dep in sections[node].depends_on:
            visit(dep, (*stack, node))
        color[node] = BLACK

    for key in sections:
        visit(key, ())


# Validate at import — fail loud if SECTIONS is malformed.
_validate_dag(SECTIONS)


# ---------------------------------------------------------------------------
# Provenance read / write.
# ---------------------------------------------------------------------------

def record_run(
    config_path: str | Path,
    section_id: str,
    run_id: str | None = None,
    *,
    run_at: datetime | None = None,
) -> None:
    """Record that ``section_id`` just completed successfully.

    Writes ``[provenance.<section_id>]`` to ``project.toml`` with
    ``last_run_at`` (always) and ``last_run_id`` (when supplied).
    Other provenance entries and the rest of ``project.toml`` are
    preserved.

    Parameters
    ----------
    config_path
        Path to the project's ``project.toml``.
    section_id
        Key in :data:`SECTIONS`. ``KeyError`` is raised for unknown
        IDs — this is deliberate; the central declaration is
        authoritative.
    run_id
        Run-id directory name when the section is a producer (e.g.,
        ``"20260520-233610-6203f1"``). ``None`` for settings-only
        sections (pixels-per-mm, outlier settings) that don't have a
        run-id concept; the entry will record only ``last_run_at``.
    run_at
        UTC timestamp for the record. Defaults to ``datetime.now(timezone.utc)``.
        Exposed for testability (deterministic timestamps in tests)
        and for retro-fill paths like :func:`migrate_project` that
        want to use the source file's mtime.
    """
    if section_id not in SECTIONS:
        raise KeyError(
            f"unknown section_id {section_id!r}; not declared in "
            f"mufasa.section_provenance.SECTIONS"
        )

    config_path = Path(config_path)
    data = read_project_toml(config_path)

    prov = data.setdefault("provenance", {})
    entry: dict[str, Any] = {}
    if run_id is not None:
        entry["last_run_id"] = run_id
    if run_at is None:
        run_at = datetime.now(timezone.utc)
    entry["last_run_at"] = run_at.isoformat(timespec="seconds")
    prov[section_id] = entry

    write_project_toml(config_path, data)


def _read_run_at(entry: dict[str, Any]) -> datetime | None:
    """Parse ``last_run_at`` out of a ``[provenance.<id>]`` entry.

    Returns ``None`` if the field is missing or malformed (graceful
    degradation — a corrupted entry shouldn't crash the UI; it
    surfaces as UNKNOWN).
    """
    s = entry.get("last_run_at")
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def get_status(
    config_path: str | Path,
    section_id: str,
) -> SectionStatus:
    """Return the current status of ``section_id``.

    Soft-fails to :data:`SectionStatus.UNKNOWN` if the project.toml
    is missing or the section has no recorded provenance. Use this
    rather than ``record_run``-style strict failure modes — UI code
    needs to be tolerant.

    See module docstring for the staleness-decision rules.
    """
    if section_id not in SECTIONS:
        raise KeyError(
            f"unknown section_id {section_id!r}"
        )

    config_path = Path(config_path)
    try:
        data = read_project_toml(config_path)
    except (FileNotFoundError, OSError):
        return SectionStatus.UNKNOWN

    prov = data.get("provenance", {}) or {}
    my_entry = prov.get(section_id)
    if not isinstance(my_entry, dict):
        return SectionStatus.UNKNOWN
    my_run_at = _read_run_at(my_entry)
    if my_run_at is None:
        return SectionStatus.UNKNOWN

    # Walk declared dependencies, checking only those with known
    # last_run_at. Unknown parents are ignored — see the module
    # docstring for why.
    spec = SECTIONS[section_id]
    for dep_id in spec.depends_on:
        dep_entry = prov.get(dep_id)
        if not isinstance(dep_entry, dict):
            continue
        dep_run_at = _read_run_at(dep_entry)
        if dep_run_at is None:
            continue
        if dep_run_at > my_run_at:
            return SectionStatus.STALE

    return SectionStatus.CURRENT


def get_all_statuses(
    config_path: str | Path,
) -> dict[str, SectionStatus]:
    """Return :class:`SectionStatus` for every section in :data:`SECTIONS`.

    Single-read variant — reads ``project.toml`` once and walks all
    sections from the in-memory dict. Use this rather than calling
    :func:`get_status` in a loop when refreshing every badge on a
    page.
    """
    config_path = Path(config_path)
    try:
        data = read_project_toml(config_path)
    except (FileNotFoundError, OSError):
        return {sid: SectionStatus.UNKNOWN for sid in SECTIONS}

    prov = data.get("provenance", {}) or {}
    out: dict[str, SectionStatus] = {}
    for section_id, spec in SECTIONS.items():
        my_entry = prov.get(section_id)
        if not isinstance(my_entry, dict):
            out[section_id] = SectionStatus.UNKNOWN
            continue
        my_run_at = _read_run_at(my_entry)
        if my_run_at is None:
            out[section_id] = SectionStatus.UNKNOWN
            continue

        status = SectionStatus.CURRENT
        for dep_id in spec.depends_on:
            dep_entry = prov.get(dep_id)
            if not isinstance(dep_entry, dict):
                continue
            dep_run_at = _read_run_at(dep_entry)
            if dep_run_at is None:
                continue
            if dep_run_at > my_run_at:
                status = SectionStatus.STALE
                break
        out[section_id] = status

    return out


def find_section_by_title(
    page: str, section_title: str,
) -> SectionSpec | None:
    """Look up a :class:`SectionSpec` by its (page, section_title) pair.

    Patch 122du — used by the UI to bridge between
    ``WorkflowPage`` (which knows section_title strings) and
    :data:`SECTIONS` (which keys by section_id). Returns ``None``
    if no section matches, which is the right behavior for
    informational-only sections (e.g., "Input source" pickers, the
    "Advanced / legacy" group on Pose cleanup) that aren't tracked
    for provenance.

    Linear scan — :data:`SECTIONS` has ~14 entries today; not worth
    indexing.
    """
    for spec in SECTIONS.values():
        if spec.page == page and spec.section_title == section_title:
            return spec
    return None


__all__ = [
    "ProvenanceError",
    "SECTIONS",
    "SectionSpec",
    "SectionStatus",
    "find_section_by_title",
    "get_all_statuses",
    "get_status",
    "record_run",
]
