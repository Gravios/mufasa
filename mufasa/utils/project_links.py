"""
mufasa.utils.project_links
==========================

Patch 122kk: project links that survive being mounted somewhere else.

Videos were imported as symbolic links holding the **absolute** path of
the original file. Mounting the project at another path (sshfs, NFS, a
different machine) breaks every such link: the absolute path names a
location on the server, not here.

* ``link_file`` (imports): a **relative** link when the target is on the
  same tree, so the pair moves and mounts together.
* ``repair_project_links`` (project load, ``mufasa-relink``): rewrites the
  project's absolute links as relative ones —

  - a link that works (e.g. run on the server) is rewritten to the same
    target, relative;
  - a broken link (e.g. on an sshfs mount) is followed to where its target
    sits on this machine: the mount offset is found by trying the target's
    path, minus leading parts, under each ancestor of the project
    (``/proc/videos/K7.mp4`` on the server, the project at
    ``/cloud/server/proc/mufasa/P`` here → ``/cloud/server/proc/videos/K7.mp4``).
    Only ancestors of the project are tried, so the project lies under the
    same offset and the relative link means the same file on both
    machines. The offset found first is tried first for the rest.

Links whose target cannot be found, and read-only mounts, are reported and
left alone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _rel(target: Path, link: Path) -> str:
    return os.path.relpath(target, link.parent)


def link_file(target, link, *, relative: bool = True) -> Path:
    """Create ``link`` → ``target``; relative when ``relative`` and the two
    share more than the filesystem root (otherwise absolute)."""
    target, link = Path(os.path.abspath(target)), Path(os.path.abspath(link))
    if link.is_symlink() or link.exists():
        link.unlink()
    common = Path(os.path.commonpath([target, link.parent]))
    if relative and common != Path(common.anchor):
        os.symlink(_rel(target, link), link)
    else:
        os.symlink(target, link)
    return link


@dataclass
class RepairReport:
    relinked: list = field(default_factory=list)        # (link, old target, new relative target)
    broken: list = field(default_factory=list)          # (link, target) — not found here
    failed: list = field(default_factory=list)          # (link, error) — e.g. read-only mount
    offset: tuple | None = None                         # (server prefix, local prefix)

    @property
    def changed(self) -> bool:
        return bool(self.relinked)

    def text(self) -> str:
        out = []
        if self.offset:
            out.append(f"mount offset: {self.offset[0]} → {self.offset[1]}")
        if self.relinked:
            out.append(f"{len(self.relinked)} link(s) made relative")
        if self.broken:
            out.append(f"{len(self.broken)} broken link(s) whose target is not reachable here: "
                       + ", ".join(Path(l).name for l, _ in self.broken[:5])
                       + ("…" if len(self.broken) > 5 else "")
                       + " — run `mufasa-relink` on the machine where they work")
        if self.failed:
            out.append(f"{len(self.failed)} link(s) could not be rewritten: {self.failed[0][1]}")
        return "; ".join(out) if out else "all links fine"


def _find_here(target: Path, project_root: Path, hint: tuple | None):
    """Where an absolute ``target`` from another machine is on this one.
    Returns (local path, server prefix, local prefix) or None."""
    parts = target.parts
    if hint is not None:
        sp, lp = hint
        try:
            cand = Path(lp) / target.relative_to(sp)
            if cand.exists():
                return cand, Path(sp), Path(lp)
        except ValueError:
            pass
    for anc in [project_root, *project_root.parents]:
        for i in range(1, len(parts)):
            cand = anc.joinpath(*parts[i:])
            if cand.exists():
                return cand, Path(*parts[:i]), anc
    return None


def repair_project_links(project_root, *, subdirs=("sources",), dry_run: bool = False) -> RepairReport:
    """Make the project's absolute file links relative (see the module
    docstring). Only links under ``subdirs`` are touched."""
    root = Path(project_root).absolute()
    rep = RepairReport()
    hint = None
    for sub in subdirs:
        base = root / sub
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            for name in filenames + dirnames:
                link = Path(dirpath) / name
                if not link.is_symlink():
                    continue
                target = Path(os.readlink(link))
                if not target.is_absolute():
                    continue
                if link.exists():                       # works here: same target, relative
                    new_target = target
                else:
                    found = _find_here(target, root, hint)
                    if found is None:
                        rep.broken.append((str(link), str(target)))
                        continue
                    new_target, sp, lp = found
                    hint = (str(sp), str(lp))
                    rep.offset = rep.offset or hint
                common = Path(os.path.commonpath([new_target, link.parent]))
                if common == Path(common.anchor):
                    continue                            # nothing shared: leave absolute
                rel = _rel(new_target, link)
                if not dry_run:
                    try:
                        tmp = link.with_name(f".{link.name}.relink")
                        if tmp.is_symlink():
                            tmp.unlink()
                        os.symlink(rel, tmp)
                        os.replace(tmp, link)
                    except OSError as e:
                        rep.failed.append((str(link), str(e)))
                        continue
                rep.relinked.append((str(link), str(target), rel))
    return rep


def main(argv: list[str] | None = None) -> int:
    """``mufasa-relink [PROJECT] [--dry-run]``: make a project's file links
    relative (videos imported as absolute links break when the project is
    mounted elsewhere). Run it where the links work — usually the server —
    or on the mount, where it finds the files through the mount offset."""
    import argparse
    ap = argparse.ArgumentParser(prog="mufasa-relink", description=main.__doc__)
    ap.add_argument("project", nargs="?", default=".", help="project folder or project.toml (default: here)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    p = Path(a.project)
    root = p.parent if p.is_file() else p
    rep = repair_project_links(root, dry_run=a.dry_run)
    for link, old, new in rep.relinked:
        print(f"{'would relink' if a.dry_run else 'relinked'}: {Path(link).relative_to(root.absolute())} → {new}")
    print(rep.text())
    return 1 if (rep.broken or rep.failed) else 0


__all__ = ["RepairReport", "link_file", "main", "repair_project_links"]


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
