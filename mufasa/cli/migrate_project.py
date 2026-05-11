"""Patch 122a: migrate a legacy SimBA-layout project to v1.

CLI tool::

    python -m mufasa.cli.migrate_project <path-to-project>

By default, runs a dry run that prints every operation it
would perform but doesn't touch the filesystem. Pass
``--commit`` to actually move files.

What it does:

1. Detects the layout (refuses if already v1).
2. Plans the move: walks LEGACY_TO_V1_MAPPING, finds files in
   each legacy subdirectory, and computes destinations under
   the new v1 root.
3. Creates the v1 skeleton (``sources/``, ``derived/``,
   ``models/``, ``logs/``).
4. Moves files. Legacy data is grouped under
   ``imported_<YYYYMMDD>`` runs because we don't know the
   original timing — preserving everything in one labeled bucket
   beats inventing fake per-file run ids.
5. Reads ``project_config.ini`` + body-part-name files and
   writes a ``project.toml`` at the v1 root.
6. Writes a ``MIGRATION.toml`` manifest at the v1 root listing
   every operation, so the migration is auditable and (if
   needed) reversible by hand.

Idempotence: re-running on an already-migrated project is a
no-op (detects v1 layout, returns).

What it does NOT do:

- Re-encode or transform pose data. Files are moved verbatim.
- Touch ``models/`` contents. Trained classifier weights stay
  where they are.
- Delete the original ``project_folder/``. After successful
  migration it's left empty and can be removed manually.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mufasa.legacy_layout import (
    LEGACY_TO_V1_MAPPING,
    LegacyProjectPaths,
    parse_legacy_body_part_names,
    parse_legacy_config,
)
from mufasa.project_layout import (
    PROJECT_LAYOUT_VERSION,
    ProjectPaths,
    detect_layout,
    write_project_toml,
)


# ---------------------------------------------------------------------------
# Plan + execution
# ---------------------------------------------------------------------------

@dataclass
class MigrationOp:
    """One filesystem operation in a migration plan."""

    kind: str             # "move-dir", "move-file", "write-toml", "skip"
    source: Optional[Path]
    destination: Optional[Path]
    description: str

    def render(self) -> str:
        if self.kind == "skip":
            return f"  SKIP    {self.description} ({self.source})"
        if self.kind == "write-toml":
            return f"  WRITE   {self.destination}   ({self.description})"
        return (
            f"  {self.kind.upper():<8s}"
            f" {self.source}  →  {self.destination}"
        )


@dataclass
class MigrationPlan:
    """Total set of operations to migrate one project.

    Computed by :func:`plan_migration`; executed by
    :func:`execute_plan`. Splitting plan from execution lets
    the dry-run mode show exactly what would happen and lets
    tests inspect the plan without touching disk.
    """

    legacy_root: Path
    v1_root: Path
    import_run_label: str
    ops: List[MigrationOp] = field(default_factory=list)
    config_data: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ops)


def _import_run_label() -> str:
    """Stable label for the migrated import. Uses today's
    date rather than a full run-id timestamp because the
    semantics are "one import per project", not "one run per
    invocation."
    """
    return "imported_" + time.strftime("%Y%m%d")


def plan_migration(
    legacy_input: Path,
    v1_root: Optional[Path] = None,
) -> MigrationPlan:
    """Walk the legacy project and produce a plan.

    ``legacy_input`` can be either the parent directory of
    ``project_folder/`` or ``project_folder/`` itself.

    ``v1_root`` defaults to the same directory the legacy
    project lives in: the migration replaces ``project_folder/``
    with the new layout in-place. Pass an explicit path to
    write to a different directory (e.g. for testing or for
    a parallel-tree migration).
    """
    legacy = LegacyProjectPaths.open(legacy_input)
    if v1_root is None:
        # In-place: parent of project_folder/ becomes the new
        # project root. The original project_folder/ stays
        # alongside but empty after migration; the user can
        # rmdir it once happy.
        v1_root = legacy.project_path
    v1_root = Path(v1_root).resolve()

    plan = MigrationPlan(
        legacy_root=legacy.project_path,
        v1_root=v1_root,
        import_run_label=_import_run_label(),
    )

    # ---- Per-directory moves (LEGACY_TO_V1_MAPPING) ----
    for rel_src, dest_template, label in LEGACY_TO_V1_MAPPING:
        src = legacy.stage_path(rel_src)
        if not src.exists():
            continue
        if dest_template is None:
            # Handled by the config-write step
            plan.ops.append(MigrationOp(
                kind="skip", source=src, destination=None,
                description=label,
            ))
            continue
        dest_rel = dest_template.format(
            import_run=plan.import_run_label,
        )
        dest = v1_root / dest_rel
        kind = "move-dir" if src.is_dir() else "move-file"
        plan.ops.append(MigrationOp(
            kind=kind, source=src, destination=dest,
            description=label,
        ))

    # ---- Models: external to project_folder/, preserved in place ----
    if legacy.models_folder.is_dir():
        dest = v1_root / "models"
        if dest.resolve() != legacy.models_folder.resolve():
            plan.ops.append(MigrationOp(
                kind="move-dir",
                source=legacy.models_folder, destination=dest,
                description="trained classifiers",
            ))

    # ---- Parse legacy config + body parts, build v1 dict ----
    cfg = parse_legacy_config(legacy.config_file)
    cfg["project_layout_version"] = PROJECT_LAYOUT_VERSION
    cfg["migrated_from"] = "simba_legacy"
    cfg["migrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    bp = parse_legacy_body_part_names(legacy)
    if bp:
        cfg.setdefault("project", {})["body_parts"] = bp
    plan.config_data = cfg
    plan.ops.append(MigrationOp(
        kind="write-toml",
        source=legacy.config_file,
        destination=v1_root / "project.toml",
        description="project config (transformed)",
    ))

    return plan


def execute_plan(
    plan: MigrationPlan,
    *,
    verbose: bool = True,
) -> Path:
    """Apply ``plan`` to disk. Returns the v1 root.

    Creates the v1 skeleton first, then performs moves in plan
    order, then writes ``project.toml`` and ``MIGRATION.toml``.
    Raises on any I/O error — partial migrations are left
    in-place for the user to inspect (don't auto-rollback, as
    that could destroy data).
    """
    paths = ProjectPaths(plan.v1_root)
    paths.ensure_skeleton()

    moved: List[Tuple[str, str]] = []
    skipped: List[str] = []

    for op in plan.ops:
        if op.kind == "skip":
            skipped.append(op.description)
            if verbose:
                print(op.render())
            continue
        if op.kind == "write-toml":
            continue   # handled below, after moves
        assert op.source is not None and op.destination is not None
        op.destination.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(op.render())
        if op.destination.exists():
            # Refuse to clobber existing dirs/files. If the dest
            # is empty, remove it; otherwise raise.
            if (
                op.destination.is_dir()
                and not any(op.destination.iterdir())
            ):
                op.destination.rmdir()
            else:
                raise FileExistsError(
                    f"destination already exists: "
                    f"{op.destination}"
                )
        shutil.move(str(op.source), str(op.destination))
        moved.append((str(op.source), str(op.destination)))

    # Write project.toml
    write_project_toml(paths.config_file, plan.config_data)
    if verbose:
        print(f"  WROTE   {paths.config_file}")

    # Migration manifest — record every move for auditability
    manifest = {
        "migration_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "legacy_root": str(plan.legacy_root),
        "v1_root": str(plan.v1_root),
        "import_run_label": plan.import_run_label,
        "moves": {
            "count": len(moved),
            "entries": [
                {"src": s, "dst": d} for s, d in moved
            ],
        },
        "skipped": {
            "items": skipped,
        },
    }
    # The manifest is itself TOML, but its "moves.entries" is
    # a list of dicts which our minimal writer doesn't support.
    # Flatten to two parallel lists.
    flat_manifest = {
        "migration_timestamp": manifest["migration_timestamp"],
        "legacy_root": manifest["legacy_root"],
        "v1_root": manifest["v1_root"],
        "import_run_label": manifest["import_run_label"],
        "moves": {
            "count": len(moved),
            "src": [s for s, _ in moved],
            "dst": [d for _, d in moved],
        },
        "skipped": {
            "items": skipped,
        },
    }
    write_project_toml(
        plan.v1_root / "MIGRATION.toml", flat_manifest,
    )
    if verbose:
        print(f"  WROTE   {plan.v1_root / 'MIGRATION.toml'}")

    return plan.v1_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="mufasa-migrate-project",
        description=(
            "Migrate a legacy SimBA-style Mufasa project to "
            "the v1 layout. Defaults to dry-run; pass --commit "
            "to actually move files."
        ),
    )
    p.add_argument(
        "path", type=str,
        help="Path to the legacy project (the parent of "
             "project_folder/, or project_folder/ itself).",
    )
    p.add_argument(
        "--commit", action="store_true",
        help="Actually perform the migration. Without this, "
             "the tool prints the plan and exits.",
    )
    p.add_argument(
        "--v1-root", type=str, default=None,
        help="Where to write the v1 layout. Defaults to the "
             "legacy project's parent directory (in-place "
             "migration).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-operation output.",
    )
    args = p.parse_args(argv)

    path = Path(args.path).resolve()
    layout = detect_layout(path)
    # detect_layout accepts the parent of project_folder/ as
    # "legacy"; it also accepts project_folder/ itself.
    if layout == "v1":
        print(
            f"{path}: already a v1-layout project. Nothing to "
            f"do.",
            file=sys.stderr,
        )
        return 0
    if layout == "unknown":
        print(
            f"{path}: not a recognized Mufasa project layout. "
            f"Expected to find either project.toml (v1) or "
            f"project_folder/project_config.ini (legacy).",
            file=sys.stderr,
        )
        return 2

    v1_root = (
        Path(args.v1_root).resolve()
        if args.v1_root else None
    )
    plan = plan_migration(path, v1_root=v1_root)

    if not args.quiet:
        print(f"Migration plan ({len(plan)} ops):")
        print(f"  legacy: {plan.legacy_root}")
        print(f"  v1:     {plan.v1_root}")
        print(f"  import_run_label: {plan.import_run_label}")
        for op in plan.ops:
            print(op.render())
        print()

    if not args.commit:
        print(
            "Dry run — no files moved. Pass --commit to "
            "actually perform the migration.",
        )
        return 0

    execute_plan(plan, verbose=not args.quiet)
    print(f"\nMigration complete. New layout at: {plan.v1_root}")
    print(
        "Original project_folder/ may now be empty and "
        "removable.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
