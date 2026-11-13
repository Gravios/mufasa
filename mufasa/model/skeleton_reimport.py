"""
mufasa.model.skeleton_reimport
==============================

Patch 122km: re-import a project's skeleton model (markers and skeleton)
from a pose file — e.g. after retraining the pose model with new markers
(eyes, shoulders).

``plan_reimport`` compares the project with the file and returns what
would change; ``apply_reimport`` writes it (``project.toml`` backed up
first):

* ``[pose] body_parts`` — the file's markers, in the file's order;
* ``[skeleton]`` — the file's own skeleton (embedded in a FreeDLC/DLC
  parquet) when it has one; otherwise the project's edges between markers
  that remain;
* ``[[pose.segments]]`` (the Kalman v2 kinematic tree; it takes priority
  over ``[skeleton]``) — removed markers leave their segments; each new
  marker joins the segment of its nearest marker on the new skeleton
  (eyes wired to ``head_mid`` join ``head``, shoulders wired to
  ``back_T4`` join ``back``). New markers with no path to a segmented
  marker are reported; a segment left without markers is dropped and its
  children re-attached to its parent.

Removed markers still named elsewhere in ``project.toml`` (``[pose.layout]``
roles, stage settings) are reported, not changed. Pose files already in
``sources/pose/`` are not touched: import the new model's output with
*Import pose data*.
"""
from __future__ import annotations

import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReimportPlan:
    source: str
    old_markers: list[str]
    new_markers: list[str]
    old_edges: list[tuple[str, str]]
    new_edges: list[tuple[str, str]]
    edges_from_file: bool
    segment_joins: dict[str, str] = field(default_factory=dict)       # new marker → segment
    unplaced: list[str] = field(default_factory=list)                 # new, no segment reachable
    emptied_segments: list[str] = field(default_factory=list)
    references: dict[str, list[str]] = field(default_factory=dict)    # removed marker → where named
    has_segments: bool = False

    @property
    def added(self) -> list[str]:
        old = set(self.old_markers)
        return [m for m in self.new_markers if m not in old]

    @property
    def removed(self) -> list[str]:
        new = set(self.new_markers)
        return [m for m in self.old_markers if m not in new]

    @property
    def changed(self) -> bool:
        return (self.new_markers != self.old_markers
                or {frozenset(e) for e in self.new_edges} != {frozenset(e) for e in self.old_edges})

    def text(self) -> str:
        lines = [f"{len(self.old_markers)} → {len(self.new_markers)} markers"]
        if self.added:
            lines.append("added: " + ", ".join(self.added))
        if self.removed:
            lines.append("removed: " + ", ".join(self.removed))
        if not self.added and not self.removed and self.new_markers != self.old_markers:
            lines.append("same markers, new order")
        src = "from the file" if self.edges_from_file else "kept from the project (the file has no skeleton)"
        lines.append(f"skeleton: {len(self.old_edges)} → {len(self.new_edges)} edges, {src}")
        if self.has_segments:
            for m, s in self.segment_joins.items():
                lines.append(f"Kalman segment: {m} joins '{s}'")
            if self.unplaced:
                lines.append("not in any Kalman segment (no skeleton path to one): "
                             + ", ".join(self.unplaced) + " — the smoother will ignore them")
            if self.emptied_segments:
                lines.append("segments left without markers (removed; their children move to "
                             "their parent): " + ", ".join(self.emptied_segments))
        for m, where in self.references.items():
            lines.append(f"'{m}' is removed but still named in: " + ", ".join(where))
        if not self.changed:
            lines.append("nothing to change")
        return "\n".join(lines)


def _neighbours(edges) -> dict[str, set]:
    nb: dict[str, set] = {}
    for a, b in edges:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    return nb


def _segment_of_nearest(marker: str, nb: dict, seg_of: dict) -> str | None:
    """Segment of the nearest segmented marker along the skeleton (BFS;
    ties: the neighbour listed first in the segments)."""
    seen, q = {marker}, deque([marker])
    while q:
        level = []
        for _ in range(len(q)):
            m = q.popleft()
            for n in sorted(nb.get(m, ())):
                if n in seen:
                    continue
                seen.add(n)
                if n in seg_of:
                    level.append(n)
                q.append(n)
        if level:
            order = list(seg_of)
            return seg_of[min(level, key=order.index)]
    return None


def _references(data: dict, markers: list[str]) -> dict[str, list[str]]:
    """Where in project.toml (outside body_parts / skeleton / segments) each
    marker name appears as a value."""
    out: dict[str, list[str]] = {}
    want = set(markers)

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else str(k)
                if p in ("pose.body_parts", "pose.segments", "skeleton"):
                    continue
                walk(v, p)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, path)
        elif isinstance(obj, str) and obj in want:
            out.setdefault(obj, []).append(path)
    walk(data, "")
    return out


def plan_reimport(config_path, pose_path) -> ReimportPlan:
    from mufasa.pose_importers.dlc_autodetect import extract_bodyparts
    from mufasa.pose_importers.pose_parquet import skeleton_edges_for
    from mufasa.project_layout import read_project_toml
    data = read_project_toml(Path(config_path))
    pose = data.get("pose", {}) if isinstance(data.get("pose"), dict) else {}
    old = [str(m) for m in pose.get("body_parts", [])]
    sk = data.get("skeleton") if isinstance(data.get("skeleton"), dict) else {}
    old_edges = [(str(e[0]), str(e[1])) for e in sk.get("edges", []) if len(e) >= 2]
    new = list(dict.fromkeys(extract_bodyparts(str(pose_path))))
    if not new:
        raise ValueError(f"{Path(pose_path).name}: no markers found")
    file_edges = skeleton_edges_for(str(pose_path), new)
    keep = set(new)
    new_edges = file_edges or [(a, b) for a, b in old_edges if a in keep and b in keep]
    plan = ReimportPlan(str(pose_path), old, new, old_edges, new_edges, bool(file_edges))
    segs = pose.get("segments")
    if isinstance(segs, list) and segs:
        plan.has_segments = True
        seg_of = {}
        for s in segs:
            for m in s.get("markers", []):
                if m in keep:
                    seg_of[m] = s.get("name", "?")
        nb = _neighbours(new_edges)
        for m in plan.added:
            s = _segment_of_nearest(m, nb, seg_of)
            if s is None:
                plan.unplaced.append(m)
            else:
                plan.segment_joins[m] = s
        plan.emptied_segments = [s.get("name", "?") for s in segs
                                 if s.get("markers") and not any(m in keep for m in s["markers"])]
    plan.references = _references(data, plan.removed)
    return plan


def apply_reimport(config_path, plan: ReimportPlan, *, extend_segments: bool = True) -> Path:
    """Write the plan into ``project.toml``; returns the backup's path
    (``logs/project.toml.<timestamp>.bak``)."""
    from mufasa.project_layout import read_project_toml, write_project_toml
    cp = Path(config_path)
    logs = cp.parent / "logs"
    logs.mkdir(exist_ok=True)
    backup = logs / f"project.toml.{time.strftime('%Y%m%d-%H%M%S')}.bak"
    shutil.copy2(cp, backup)
    data = read_project_toml(cp)
    pose = data.setdefault("pose", {})
    pose["body_parts"] = list(plan.new_markers)
    keep = set(plan.new_markers)
    segs = pose.get("segments")
    if isinstance(segs, list) and segs:
        for s in segs:
            s["markers"] = [m for m in s.get("markers", []) if m in keep]
            if extend_segments:
                s["markers"] += [m for m, name in plan.segment_joins.items() if name == s.get("name")]
        # a segment left without markers goes; its children move to its parent
        gone = {s.get("name"): s.get("parent") for s in segs if not s["markers"]}
        for s in segs:
            p = s.get("parent")
            while p in gone:
                p = gone[p]
            if p is None:
                s.pop("parent", None)
            else:
                s["parent"] = p
        pose["segments"] = [s for s in segs if s["markers"]]
    data["skeleton"] = {"nodes": list(plan.new_markers),
                        "edges": [[a, b] for a, b in plan.new_edges]}
    write_project_toml(cp, data)
    return backup


__all__ = ["ReimportPlan", "apply_reimport", "plan_reimport"]
