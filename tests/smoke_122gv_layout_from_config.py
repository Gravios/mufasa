"""
tests/smoke_122gv_layout_from_config.py
=======================================

Patch 122gv — the Kalman layout follows the project's marker names.

Root cause it fixes: smooth_pose_v2 built its kinematic tree from a
HARD-CODED standard_rat_layout() (back1/back2/headmid/...). Session arrays
are pre-filled with NaN and only populated for markers present in BOTH the
layout and the data, so after a marker rename NOTHING matched, every value
stayed NaN, and the EKF "diverged" on an empty array — a confusing error far
from its cause.

Now: structural ROLES are fixed, marker NAMES come from the project.
* CANONICAL_LAYOUT_ROLES — the 15 roles, named by their historical names.
* standard_rat_layout(names={role: marker}) — remaps marker names, topology
  and offsets untouched.
* project.toml [pose.layout] role->name map (write/read_layout_roles).
* layout_from_config(config_path): [pose.layout] -> canonical fallback ->
  loud, actionable error.
* rename_markers seeds/updates [pose.layout] so a rename carries roles.
* smooth_pose_v2 raises when no layout marker matches the data.
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
        standard_rat_layout,
    )
    from mufasa.model.marker_rename import rename_markers
    from mufasa.project_layout import (
        PROJECT_LAYOUT_VERSION,
        read_layout_roles,
        write_layout_roles,
        write_project_toml,
        write_skeleton,
    )

    OLD = list(CANONICAL_LAYOUT_ROLES)
    NEW = {"nose": "head_nose", "headmid": "head_mid", "ear_left": "head_left",
           "ear_right": "head_right", "neck": "head_back", "back1": "back_T4",
           "back2": "back_T8", "center": "back_L2", "back3": "back_L6",
           "back4": "back_V2", "lateral_left": "hip_left",
           "lateral_right": "hip_right", "tailbase": "tail_V6",
           "tailmid": "tail_V18", "tailend": "tail_V32"}

    check("CANONICAL_LAYOUT_ROLES covers the 15 standard markers",
          len(CANONICAL_LAYOUT_ROLES) == 15
          and sorted(standard_rat_layout().marker_names) == sorted(OLD))

    base, renamed = standard_rat_layout(), standard_rat_layout(names=NEW)
    check("names= remaps markers, preserving topology",
          sorted(renamed.marker_names) == sorted(NEW.values())
          and [s.name for s in base.segments] == [s.name for s in renamed.segments])

    def _proj(bps):
        d = Path(tempfile.mkdtemp())
        cp = d / "project.toml"
        write_project_toml(cp, {"project_layout_version": PROJECT_LAYOUT_VERSION,
                                "pose": {"body_parts": list(bps),
                                         "file_type": "parquet"}})
        return cp

    cp_legacy = _proj(OLD)
    check("legacy project (canonical names, no [pose.layout]) still builds",
          sorted(layout_from_config(cp_legacy).marker_names) == sorted(OLD))

    write_skeleton(cp_legacy, nodes=OLD, edges=[("nose", "headmid")])
    rename_markers(cp_legacy, NEW, snapshot=False)
    roles = read_layout_roles(cp_legacy)
    check("rename seeds + updates [pose.layout] roles",
          roles is not None and roles.get("back2") == "back_T8"
          and roles.get("headmid") == "head_mid")
    check("layout follows the renamed project",
          sorted(layout_from_config(cp_legacy).marker_names)
          == sorted(NEW.values()))

    cp_bare = _proj(list(NEW.values()))
    raised = ""
    try:
        layout_from_config(cp_bare)
    except ValueError as e:
        raised = str(e)
    check("renamed project without roles raises an actionable error "
          "(instead of silent all-NaN)",
          "[pose.layout]" in raised and "renamed" in raised)

    write_layout_roles(cp_bare, {**NEW, "back2": "NOPE"})
    raised2 = ""
    try:
        layout_from_config(cp_bare)
    except ValueError as e:
        raised2 = str(e)
    check("roles pointing at a nonexistent marker raise", "don't exist" in raised2)

    src = (REPO / "mufasa" / "data_processors" / "kalman_pose_smoother_v2.py").read_text()
    check("smooth_pose_v2 hard-guards the zero-overlap all-NaN case",
          "matched = set(layout.marker_names) & set(marker_names_data)" in src
          and "No layout markers were found in the pose data" in src)

    cleanup = (REPO / "mufasa" / "ui_qt" / "forms" / "pose_cleanup.py").read_text()
    check("pose_cleanup builds the layout from the project config",
          "layout_from_config(self.config_path)" in cleanup
          and "layout = standard_rat_layout()" not in cleanup)

    print(f"smoke_122gv_layout_from_config: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
