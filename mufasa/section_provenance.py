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

Projects predating the section-provenance arc (anything before
122ds) start with no ``[provenance]`` table at all. Every section
reads UNKNOWN — the UI just shows no badges, which is the right
behavior for "we don't know yet." Once a section is run by the user,
its entry is recorded and the badges become meaningful from that
point forward.

(Patch 122dw deleted the legacy → v1 migration tool, so the
retro-fill path that earlier comments mentioned no longer
applies — v1 is the only supported layout, and v1 projects either
have a ``[provenance]`` table already or haven't run anything yet.)

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
from typing import Any, Callable

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
    detect_path
        Patch 122ei — optional callable ``project_root: Path -> Path``
        returning the canonical on-disk location where this section's
        output materializes (a file or non-empty directory). Used by
        :func:`get_status` as an implicit-evidence fallback when no
        ``[provenance.<section_id>]`` entry exists — the path's mtime
        becomes the implicit ``last_run_at``. Without this, projects
        that pre-date provenance wiring (or projects where the user
        manually copied data in) would forever show UNKNOWN even with
        valid output on disk.

        Return value semantics:

        * Returned path is checked via ``Path.exists()``; for
          directories, additionally for any non-hidden entry inside.
        * If the callable raises, ``get_status`` swallows the error
          and behaves as if no fallback existed (UNKNOWN, soft-fail).
        * Sections without a detect_path (typical for pure-settings
          sections like Pixels-per-mm calibration that don't produce
          a file) skip the fallback and remain UNKNOWN until
          explicit provenance lands.
    ui_bound
        Patch 122el — when False, the section is declared in the DAG
        (for dependency tracking and future planning) but no
        QGroupBox in the workbench currently corresponds to it. The
        :func:`smoke_122el_section_binding_audit` smoke test skips
        the resolution check for these entries. Defaults to True
        (the normal case — section has a QGroupBox to attach a
        badge to). Set to False for:

        * Sections whose form is composited inside another
          QGroupBox (e.g. ``savitzky_golay`` lives inside
          "Advanced / legacy").
        * Sections that are aspirationally declared but have no
          implemented form yet (e.g. ``drop_body_parts``,
          ``features_subject``).

        Provenance can still be recorded for ui_bound=False sections
        via explicit ``record_run`` calls from backend code; the
        badge UI just doesn't render for them.
    """
    section_id: str
    page: str
    section_title: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    detect_path: Callable[[Path], Path] | None = None
    ui_bound: bool = True


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
        # Patch 122ei — implicit-evidence fallback for old projects
        # that pre-date 122eb's wiring. Any non-hidden file in
        # sources/pose/ counts as "import has happened"; mtime of
        # the most recently-modified file is used as the implicit
        # last_run_at.
        detect_path=lambda root: root / "sources" / "pose",
    ),
    "pixels_per_mm": SectionSpec(
        section_id="pixels_per_mm",
        page="Preprocessing",
        # Patch 122el — title was "Pixels-per-mm calibration",
        # but the workbench section is registered as "Video
        # Calibration" (which configures pixels-per-mm calibration
        # via a calibration video). Audit caught the silent
        # badge-suppression.
        section_title="Video Calibration",
        depends_on=(),
        # Patch 122es-hotfix — was twice marked "settings-only,
        # no on-disk artifact" (in 122ei and 122ep). Both wrong.
        # The Video Calibration form's Save action writes the
        # whole calibration table (FPS, frame size, distance,
        # pixels-per-mm per video) to sources/video_info.csv
        # — the canonical ``video_info_path`` from
        # v1_project_paths. The CSV mtime IS the natural
        # implicit timestamp for "the user has saved
        # calibration values." Same detection contract as
        # the producer sections; just a flat file instead
        # of a per-run directory.
        #
        # Note: video_info.csv exists from project creation
        # (with header row + one row per video, ppm blank).
        # So an OLD uncalibrated project with the CSV still
        # at creation-time mtime will read CURRENT — false
        # positive. Acceptable because:
        #   (1) the user's explicit save action updates
        #       mtime, which is the case 122es is targeting;
        #   (2) detection of "calibration values actually
        #       present" would require reading the CSV and
        #       checking for non-default ppm values — a
        #       bigger detect_path infrastructure change
        #       (filed as deferred);
        #   (3) the cost of the false-positive is "user sees
        #       green badge on a project that doesn't have
        #       calibration values" — visually misleading
        #       but doesn't block any downstream operation.
        detect_path=lambda root: (
            root / "sources" / "video_info.csv"
        ),
    ),
    "interpolate": SectionSpec(
        section_id="interpolate",
        page="Preprocessing",
        section_title="Interpolate missing frames",
        depends_on=("import_pose",),
        # Patch 122ei — points at derived/interpolated/; if any
        # run subdir exists, the section is treated as completed
        # (with implicit timestamp = mtime of latest run dir).
        detect_path=lambda root: root / "derived" / "interpolated",
    ),
    "kalman_v2": SectionSpec(
        section_id="kalman_v2",
        page="Preprocessing",
        # Patch 122el — was "Kalman v2 smoother"; workbench
        # uses "Kalman v2 smoothing" (audit caught the typo).
        section_title="Kalman v2 smoothing",
        depends_on=("import_pose",),
        # Patch 122ei — derived/smoothed/kalman_v2/ (the
        # source-flavor-prefixed layout that 122dt's
        # publish_source_flavor handles).
        detect_path=lambda root: (
            root / "derived" / "smoothed" / "kalman_v2"
        ),
    ),
    "outlier_correction": SectionSpec(
        section_id="outlier_correction",
        page="Preprocessing",
        section_title="Run outlier correction",
        depends_on=("import_pose",),
        # Patch 122ei — derived/outlier_corrected/. Note this
        # location is shared with kalman_v2 + interpolate +
        # import_pose (they all publish symlinks here). Filesystem
        # detection here is "the canonical 'data ready for
        # features' stage exists," not specifically "outlier
        # correction was run." Acceptable: the staleness rule
        # will downgrade to STALE if an upstream producer
        # (import_pose) has a later mtime — which is the right
        # behaviour for the implicit-detection case as well.
        detect_path=lambda root: (
            root / "derived" / "outlier_corrected"
        ),
    ),
    "savitzky_golay": SectionSpec(
        section_id="savitzky_golay",
        page="Preprocessing",
        section_title="Savitzky-Golay smoother (legacy)",
        depends_on=("outlier_correction",),
        # Patch 122el — savitzky_golay form is composited inside
        # the "Advanced / legacy" QGroupBox on the Preprocessing
        # page, not registered as its own add_section. No badge
        # surface exists; mark unbound so the binding audit
        # skips the resolution check.
        ui_bound=False,
    ),
    "egocentric": SectionSpec(
        section_id="egocentric",
        page="Preprocessing",
        section_title="Egocentric alignment",
        depends_on=("outlier_correction",),
        # Patch 122ez — defensive detect_path. The form's save_dir
        # is user-picked, but the DEFAULT is ``<project>/rotated/``
        # (see EgocentricAlignmentForm.build at line 928 of
        # pose_cleanup.py). 122ep deliberately omitted detect_path
        # here because of the user-picked-dir uncertainty, but the
        # default IS predictable enough that the common case should
        # be detected.
        #
        # User report (May 26, 2026, follow-up to 122ex):
        #   > saved to /data/testing/mufasa/test-20260427/rotated
        #   > and contains the mp4 and parquet files but the badge
        #   > is still white.
        # The user's save dir matches the default. With this
        # detect_path, the badge will go CURRENT via filesystem
        # evidence regardless of whether record_run was called
        # (defends against the running-process-has-cached-old-
        # class-definition failure mode of 122ex).
        #
        # If a user picks a NON-default save_dir, detect_path won't
        # find their output and the badge will rely on record_run
        # via 122ex's section_id wiring. That's the worst case;
        # not strictly worse than no detect_path.
        detect_path=lambda root: root / "rotated",
    ),
    "drop_body_parts": SectionSpec(
        section_id="drop_body_parts",
        page="Preprocessing",
        section_title="Drop body parts",
        depends_on=("import_pose",),
        # Patch 122el — no form for this section currently exists
        # in the workbench. Aspirational placeholder for the
        # workflow DAG; mark unbound until the form is implemented.
        ui_bound=False,
    ),
    "roi_definitions": SectionSpec(
        section_id="roi_definitions",
        page="ROI",
        section_title="Definitions",
        depends_on=("pixels_per_mm",),
        # Patch 122ep — points at logs/measures/ROI_definitions.h5
        # (the SINGLE FILE the ROI definitions live in, per 122eh +
        # 122en's centralized layout). _path_mtime_if_has_content
        # handles file targets via the ``path.is_file()`` branch.
        # Detection signal: "the user has saved any ROIs at all."
        detect_path=lambda root: (
            root / "logs" / "measures" / "ROI_definitions.h5"
        ),
    ),
    "features_subject": SectionSpec(
        section_id="features_subject",
        page="Features",
        section_title="Subject features",
        depends_on=("outlier_correction",),
        # Patch 122el — Features page only has "Compute feature
        # subsets" currently; no separate "Subject features"
        # section. Aspirational; mark unbound.
        ui_bound=False,
    ),
    "features_roi": SectionSpec(
        section_id="features_roi",
        # Patch 122el — ROI features form is on the ROI page,
        # registered as "Features", not on the Features page
        # (which only has "Compute feature subsets"). Audit
        # caught the cross-page misbinding.
        page="ROI",
        section_title="Features",
        depends_on=("outlier_correction", "roi_definitions"),
        # No detect_path — ROI features land in
        # ``derived/features/`` mixed with subject features;
        # filesystem evidence can't distinguish "ROI features
        # appended" from "subject features computed." Skip the
        # implicit detection; rely on explicit ``record_run``
        # if/when this section gets wired to a form's section_id.
    ),
    "annotation": SectionSpec(
        section_id="annotation",
        page="Annotation",
        # Patch 122el — was "Annotate"; the workbench section
        # that hosts the labelling activity is "Frame labelling".
        section_title="Frame labelling",
        depends_on=("features_subject",),
        # Patch 122ep — per-video annotation labels land at
        # derived/labels/<video>.parquet. Any non-hidden entry
        # in the dir counts as "annotation has happened."
        detect_path=lambda root: root / "derived" / "labels",
    ),
    "classifier_train": SectionSpec(
        section_id="classifier_train",
        # Patch 122ey — was page="Classifier"; the Classifier page
        # was split into 6 standalone sidebar pages, one per former
        # section. The page name now matches the section title.
        page="Train classifier",
        section_title="Train classifier",
        depends_on=("features_subject", "annotation"),
        # Patch 122ep — trained classifier .sav files land in
        # the project's models/ directory. NB: this also fires
        # if the user merely COPIED in pre-trained models from
        # another project, which is acceptable — "models exist
        # to run inference with" is the right semantic for a
        # CURRENT badge here, even if not literally "this
        # project trained them."
        detect_path=lambda root: root / "models",
    ),
    "classifier_run": SectionSpec(
        section_id="classifier_run",
        # Patch 122ey — was page="Classifier"; split into 6
        # standalone pages. Page name now matches section title.
        page="Run inference",
        # Patch 122el — was "Run classifier"; workbench section
        # is "Run inference". Same shape of typo as 122ej.
        section_title="Run inference",
        depends_on=("classifier_train",),
        # Patch 122ep — per-video inference outputs land at
        # derived/classifications/<video>.parquet (the post-
        # 122ax v1 location — no run_id subdir under
        # derived/classifications/, the writer is flat).
        detect_path=lambda root: (
            root / "derived" / "classifications"
        ),
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
        Exposed for testability — deterministic timestamps in tests
        are easier to assert on than wall-clock values.
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

    Patch 122ei — added a filesystem-evidence fallback. If no
    ``[provenance.<section_id>]`` entry exists but the section
    declares a ``detect_path`` AND the on-disk location has content,
    the location's mtime is used as the implicit ``last_run_at``.
    This handles the "old project, pre-dates provenance wiring"
    case: a user opens a project where pose data was imported
    before 122eb landed, and the badge should reflect that data
    exists (CURRENT) rather than that no provenance entry was found
    (UNKNOWN). The implicit timestamp also composes correctly with
    the staleness rule: if a parent's later mtime is detected,
    the child still reads STALE.

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
    spec = SECTIONS[section_id]
    project_root = config_path.parent

    my_run_at = _resolve_run_at(
        prov, section_id, spec, project_root,
    )
    if my_run_at is None:
        return SectionStatus.UNKNOWN

    # Walk declared dependencies, checking only those with a known
    # timestamp (either explicit provenance or implicit detect_path).
    # Unknown parents are ignored — see the module docstring for why.
    for dep_id in spec.depends_on:
        dep_spec = SECTIONS.get(dep_id)
        if dep_spec is None:
            # Shouldn't happen given module-init validation, but
            # belt-and-suspenders.
            continue
        dep_run_at = _resolve_run_at(
            prov, dep_id, dep_spec, project_root,
        )
        if dep_run_at is None:
            continue
        if dep_run_at > my_run_at:
            return SectionStatus.STALE

    return SectionStatus.CURRENT


def _resolve_run_at(
    prov: Mapping[str, Any],
    section_id: str,
    spec: SectionSpec,
    project_root: Path,
) -> datetime | None:
    """Return the effective ``last_run_at`` for a section.

    Patch 122ei — composes explicit provenance with the
    filesystem-evidence fallback:

    1. If ``[provenance.<section_id>].last_run_at`` is set, return
       it (explicit wins over implicit — the user / backend
       explicitly recorded a run, that's authoritative).
    2. Else if the section declares a ``detect_path`` AND the
       returned path exists with content, return the mtime of the
       most recently-modified file under that path.
    3. Else return None (UNKNOWN).

    Errors during the filesystem check (permission denied,
    transient races) are swallowed — returns None as if no fallback
    existed. UI code can't afford to crash because of a flaky
    filesystem.
    """
    entry = prov.get(section_id)
    if isinstance(entry, dict):
        explicit = _read_run_at(entry)
        if explicit is not None:
            return explicit

    # Implicit fallback.
    if spec.detect_path is None:
        return None
    try:
        path = spec.detect_path(project_root)
        return _path_mtime_if_has_content(path)
    except Exception:
        return None


def _path_mtime_if_has_content(path: Path) -> datetime | None:
    """Return the mtime of ``path`` (UTC) if it has content, else None.

    "Has content" means:

    * For a regular file: the file exists.
    * For a directory: the directory exists AND contains at least
      one non-hidden entry (file, subdir, or symlink). Hidden
      entries (dotfiles) are ignored so a ``.DS_Store`` doesn't
      trick us into reporting "imported" on a freshly-created
      project.

    For directories with content, returns the MAX mtime across
    all non-hidden entries (recursively, one level deep — enough
    for the v1 layout's shape without doing a full rglob which
    would be slow on large run dirs).

    Returns None if the path doesn't exist or the directory is
    empty.
    """
    if not path.exists():
        return None
    if path.is_file():
        return datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc,
        )
    if not path.is_dir():
        return None
    # Directory: scan one level. Take the max mtime across non-
    # hidden entries.
    latest: float | None = None
    try:
        for entry in path.iterdir():
            if entry.name.startswith("."):
                continue
            try:
                m = entry.stat().st_mtime
            except OSError:
                continue
            if latest is None or m > latest:
                latest = m
    except (OSError, PermissionError):
        return None
    if latest is None:
        return None
    return datetime.fromtimestamp(latest, tz=timezone.utc)


def get_all_statuses(
    config_path: str | Path,
) -> dict[str, SectionStatus]:
    """Return :class:`SectionStatus` for every section in :data:`SECTIONS`.

    Single-read variant — reads ``project.toml`` once and walks all
    sections from the in-memory dict. Use this rather than calling
    :func:`get_status` in a loop when refreshing every badge on a
    page.

    Patch 122ew-hotfix — was reading provenance entries directly via
    ``_read_run_at`` and falling through to UNKNOWN when no entry
    existed, completely bypassing the filesystem-evidence fallback
    that ``get_status`` had via ``_resolve_run_at`` since 122ei.
    The divergence made ``refresh_section_badges`` (post-form
    completion) produce DIFFERENT results from
    ``_paint_initial_badge`` (page open) for the same section state
    — initial paint correctly showed CURRENT via detect_path, but
    refresh wiped that to UNKNOWN. User-visible symptom: badges
    turned white the moment any form completed. Fix:
    ``get_all_statuses`` delegates to the same ``_resolve_run_at``
    helper, restoring contract parity with ``get_status``.
    """
    config_path = Path(config_path)
    try:
        data = read_project_toml(config_path)
    except (FileNotFoundError, OSError):
        # Even with no project.toml, detect_path could still find
        # filesystem evidence (e.g., a partially-set-up project with
        # data files but no provenance yet). Walk SECTIONS via
        # _resolve_run_at with an empty prov dict to handle this.
        prov: Mapping[str, Any] = {}
    else:
        prov = data.get("provenance", {}) or {}

    project_root = config_path.parent
    out: dict[str, SectionStatus] = {}

    for section_id, spec in SECTIONS.items():
        my_run_at = _resolve_run_at(
            prov, section_id, spec, project_root,
        )
        if my_run_at is None:
            out[section_id] = SectionStatus.UNKNOWN
            continue

        status = SectionStatus.CURRENT
        for dep_id in spec.depends_on:
            dep_spec = SECTIONS.get(dep_id)
            if dep_spec is None:
                continue
            dep_run_at = _resolve_run_at(
                prov, dep_id, dep_spec, project_root,
            )
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
