"""
tests/smoke_122gx_layout_general.py
===================================

Patch 122gx — derive the kinematic tree from the project's own skeleton;
stop hard-coding the rat rig.

122gv made marker NAMES configurable but the topology was still the built-in
15-marker rat rig (CANONICAL_LAYOUT_ROLES), so a skeleton with different
connectivity or a different marker count could not be expressed.

Now, layout_from_config resolves most-explicit-first:
  1. [[pose.segments]] — the project's own tree (rigid clusters included).
  2. [skeleton] — BFS spanning tree over the skeleton graph
     (segments_from_skeleton); [pose.kinematics].root picks the anchor,
     else a graph centre. Skeletons have cycles (the rat rig: 24 edges over
     15 nodes), so a spanning tree is required; the extra edges are dropped.
  3. [pose.layout] role map onto the built-in rig (kept for existing users).
  4. The built-in rig unchanged for canonical projects.
  5. Otherwise an actionable error.

Also: the custom TOML writer gained array-of-tables support (without it any
rewrite of a project.toml holding [[pose.segments]] — e.g. a marker rename —
raised and lost the section), and rename_markers renames markers inside
[[pose.segments]] and [pose.layout].
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main():
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        CANONICAL_LAYOUT_ROLES,
        layout_from_config,
        layout_from_segments,
        segments_from_skeleton,
        standard_rat_layout,
    )
    from mufasa.model.marker_rename import rename_markers
    from mufasa.project_layout import (
        PROJECT_LAYOUT_VERSION,
        read_project_toml,
        write_layout_roles,
        write_project_toml,
        write_skeleton,
    )

    def proj(bps, **extra):
        d = Path(tempfile.mkdtemp())
        cp = d / "project.toml"
        body = {"project_layout_version": PROJECT_LAYOUT_VERSION,
                "pose": {"body_parts": list(bps), "file_type": "parquet"}}
        body["pose"].update(extra)
        write_project_toml(cp, body)
        return cp

    # --- arbitrary (non-rat) rig derived straight from its skeleton ---
    S = ["a", "b", "c", "d", "e", "f"]
    SE = [("a", "b"), ("b", "c"), ("c", "d"), ("b", "e"), ("e", "f")]
    cp = proj(S)
    write_skeleton(cp, nodes=S, edges=SE)
    L = layout_from_config(cp)
    check("arbitrary 6-marker skeleton builds with no extra config",
          sorted(L.marker_names) == sorted(S) and len(L.segments) == 6)

    # --- cyclic skeleton (a tree cannot be read off directly) ---
    C = ["p", "q", "r"]
    segs = segments_from_skeleton(C, [("p", "q"), ("q", "r"), ("r", "p")])
    roots = [s for s in segs if s.parent is None]
    check("cyclic skeleton -> spanning tree (one root, no cycles)",
          len(roots) == 1 and len(segs) == 3)

    # --- explicit root override ---
    cp = proj(S, kinematics={"root": "d"})
    write_skeleton(cp, nodes=S, edges=SE)
    check("[pose.kinematics].root anchors the tree",
          [s.name for s in layout_from_config(cp).segments if s.parent is None] == ["d"])

    # --- explicit [[pose.segments]] incl. a rigid cluster ---
    cp = proj(["m1", "m2", "m3"], segments=[
        {"name": "body", "markers": ["m1", "m2"]},
        {"name": "head", "parent": "body", "markers": ["m3"]},
    ])
    L = layout_from_config(cp)
    body_seg = [s for s in L.segments if s.name == "body"][0]
    check("[[pose.segments]] wins and expresses rigid clusters",
          len(L.segments) == 2 and len(body_seg.markers) == 2)

    # --- backward compatibility ---
    ref = standard_rat_layout()
    L = layout_from_config(proj(list(CANONICAL_LAYOUT_ROLES)))
    check("legacy canonical project still gets the built-in rig unchanged",
          L.state_dim == ref.state_dim
          and [s.name for s in L.segments] == [s.name for s in ref.segments])

    RMAP = {r: f"m_{r}" for r in CANONICAL_LAYOUT_ROLES}
    cp = proj(list(RMAP.values()))
    write_layout_roles(cp, RMAP)
    L = layout_from_config(cp)
    check("[pose.layout] role map still works (122gv projects)",
          sorted(L.marker_names) == sorted(RMAP.values())
          and L.state_dim == ref.state_dim)

    # --- validation ---
    bad_cases = [
        ([{"name": "a", "markers": ["x"]}, {"name": "b", "markers": ["y"]}], "two roots"),
        ([{"name": "a", "markers": ["x"]},
          {"name": "b", "parent": "zz", "markers": ["y"]}], "unknown parent"),
        ([{"name": "a", "markers": ["x"]},
          {"name": "b", "parent": "a", "markers": ["x"]}], "duplicate marker"),
    ]
    ok = True
    for spec, _ in bad_cases:
        try:
            layout_from_segments(spec)
            ok = False
        except ValueError:
            pass
    check("[[pose.segments]] validation rejects bad trees", ok)

    # --- writer round-trip + rename carries segment markers ---
    cp = proj(["a", "b", "c"], segments=[
        {"name": "body", "markers": ["a", "c"]},
        {"name": "h", "parent": "body", "markers": ["b"]},
    ])
    rename_markers(cp, {"b": "head_mid", "a": "back_T8"}, snapshot=False)
    segs2 = read_project_toml(cp)["pose"]["segments"]
    check("array-of-tables survives a rewrite AND rename renames its markers",
          len(segs2) == 2 and segs2[0]["markers"] == ["back_T8", "c"]
          and segs2[1]["markers"] == ["head_mid"]
          and sorted(layout_from_config(cp).marker_names)
          == sorted(["back_T8", "c", "head_mid"]))

    print(f"smoke_122gx_layout_general: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
