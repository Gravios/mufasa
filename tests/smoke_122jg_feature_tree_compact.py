"""Smoke test for patch 122jg — feature tree compacted by shared name parts.

Inside each family, the wording every feature shares is dropped (shown once
beside the family) and features are grouped by their first differing
component — the first body part — with a second level where a group would
still hold more than 24 features (three-point angles of 15 markers: 455).

Checks ``compact_family`` on the real feature-name shapes Compute Features
writes for the 15-marker rat layout (two-point distances, three-point
angles and hulls, frame-edge distances, movements, head velocity, single
hull area) and then the Feature Visualization tree built from them:
group / sub-group check states, a group tick selecting only its shown
features, search reaching into groups, the flat view toggle, expansion
memory, and that no column is lost or duplicated.
"""
from __future__ import annotations

import itertools
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.tools.feature_plots import compact_family  # noqa: E402
from mufasa.ui_qt.forms.feature_visualization import FeatureVisualizationForm  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

BP = ["head_nose", "head_mid", "head_left", "head_right", "head_back", "back_T4",
      "back_T8", "back_L2", "back_L6", "back_V2", "hip_left", "hip_right", "tail_V6",
      "tail_V18", "tail_V32"]
S = "_FEATURE_SUBSET"
FAM = {
    "two_point_body_part_distances_mm":
        [f"Distance (mm) {a}-{b}{S}" for a, b in itertools.combinations(BP, 2)],
    "within_animal_three_point_body_part_angles_degrees":
        [f"Angle (degrees) {a}-{b}-{c}{S}" for a, b, c in itertools.combinations(BP, 3)],
    "body_part_distances_to_video_frame_edge_mm":
        [f"Animal_1 {a} to {e} video edge distance (mm){S}" for a in BP
         for e in ("bottom", "left", "right", "top")],
    "frame_by_frame_body_part_movements_mm":
        [f"Animal_1 movement {a} (mm){S}" for a in BP],
    "head_velocity_in_body_frame_mm_s":
        ["Animal_1 head velocity forward along body-head axis (mm/s)" + S,
         "Animal_1 head velocity lateral to body-head axis, left positive (mm/s)" + S],
    "entire_animal_convex_hull_area_mm2": ["Animal_1 convex hull area (mm2)" + S],
}

# ---- compact_family -----------------------------------------------------------
t, nodes = compact_family(FAM["two_point_body_part_distances_mm"])
check("distances: shared wording is the template, groups by first body part",
      t == "Distance (mm) …" and [n.label for n in nodes][:3] == BP[:3]
      and [x.label for x in nodes[0].children] == BP[1:])
check("…a body part that is first in only one pair stays a loose leaf "
      "with its full label", nodes[-1].col is not None
      and nodes[-1].label == "tail_V18-tail_V32")
t, nodes = compact_family(FAM["within_animal_three_point_body_part_angles_degrees"])
first = nodes[0]
check("455 angles: 13 first-body-part groups, the big ones split again by the "
      "second body part", t == "Angle (degrees) …" and len(nodes) == 13
      and len(first.leaves()) == 91 and first.children[0].label == "head_mid"
      and [x.label for x in first.children[0].children][:2] == ["head_left", "head_right"])
check("…no group shows more than 24 features at one level",
      all(len(n.children) <= 24 for n in nodes for n in [n] + n.children))
t, nodes = compact_family(FAM["body_part_distances_to_video_frame_edge_mm"])
check("edge distances: template keeps both ends of the shared wording; "
      "leaves are the edges",
      t == "Animal_1 … video edge distance (mm)" and nodes[0].label == "head_nose"
      and [x.label for x in nodes[0].children] == ["to bottom", "to left", "to right", "to top"])
t, nodes = compact_family(FAM["frame_by_frame_body_part_movements_mm"])
check("movements (one body part each): flat list of body parts under the template",
      t == "Animal_1 movement … (mm)" and all(n.col for n in nodes)
      and [n.label for n in nodes] == BP)
t, nodes = compact_family(FAM["head_velocity_in_body_frame_mm_s"])
check("small family stays flat but loses the shared wording",
      t == "Animal_1 head velocity … (mm/s)"
      and [n.label for n in nodes] == ["forward along body-head axis",
                                       "lateral to body-head axis, left positive"])
t, nodes = compact_family(FAM["entire_animal_convex_hull_area_mm2"])
check("single feature keeps its full name", nodes[0].label == "Animal_1 convex hull area (mm2)")
for fam, cols in FAM.items():
    _t, nodes = compact_family(cols)
    got = [leaf.col for n in nodes for leaf in n.leaves()]
    if sorted(got) != sorted(cols) or len(got) != len(set(got)):
        check(f"{fam}: every column exactly once", False)
        break
else:
    check("every family: every column appears exactly once", True)

# ---- the tree in the form -----------------------------------------------------
tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="tree", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=BP).config_path
fd = Path(project_paths_from_config(cfg)["derived_features_dir"])
for fam, cols in FAM.items():
    (fd / fam).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.zeros((3, len(cols)), np.float32), columns=cols
                 ).to_parquet(fd / fam / "v1.parquet")
F = FeatureVisualizationForm(config_path=cfg)
F.resize(1000, 800)
F.show()
app.processEvents()
tops = {F.tree.topLevelItem(i).data(0, Qt.UserRole + 1)[1]: F.tree.topLevelItem(i)
        for i in range(F.tree.topLevelItemCount())}
ang = tops["within_animal_three_point_body_part_angles_degrees"]
check("family rows show the shared wording in the second column",
      ang.text(1) == "Angle (degrees) …" and ang.text(0).endswith("(455)"))
g = ang.child(0)
check("first level under angles = body parts with counts",
      g.text(0) == "head_nose   (91)" and g.child(0).text(0) == "head_mid   (13)")
leaf = g.child(0).child(0)
check("leaves show only what differs; tooltip has the full column",
      leaf.text(0) == "head_left" and leaf.toolTip(0) == f"Angle (degrees) head_nose-head_mid-head_left{S}")
g.child(0).setCheckState(0, Qt.Checked)
check("ticking a sub-group selects its 13 features; parents go partial with counts",
      len(F.selected_features()) == 13 and g.checkState(0) == Qt.PartiallyChecked
      and "13/91 selected" in g.text(0) and "13/455 selected" in ang.text(0))
leaf.setCheckState(0, Qt.Unchecked)
check("unticking one leaf makes its sub-group partial",
      g.child(0).checkState(0) == Qt.PartiallyChecked and len(F.selected_features()) == 12)
def fresh_tops():
    return {F.tree.topLevelItem(i).data(0, Qt.UserRole + 1)[1]: F.tree.topLevelItem(i)
            for i in range(F.tree.topLevelItemCount())}


F._clear_selection()                       # rebuilds the tree
tops = fresh_tops()
g = tops["within_animal_three_point_body_part_angles_degrees"].child(0)
F.search.setText("tail_V32 head_nose")
F._apply_filter()
vis = [it for it in F._walk() if F._is_leaf(it) and not F._hidden_in(it, None)]
check("search reaches inside groups (all words, any position) and opens the path",
      vis and all("tail_V32" in it.toolTip(0) and "head_nose" in it.toolTip(0) for it in vis)
      and g.isExpanded())
dist = tops["two_point_body_part_distances_mm"]
dist.setCheckState(0, Qt.Checked)
F.search.setText("")
F._apply_filter()
check("a family tick while searching takes only the matching features",
      F.selected_features() == [f"Distance (mm) head_nose-tail_V32{S}"])
F._clear_selection()
g = [F.tree.topLevelItem(i) for i in range(F.tree.topLevelItemCount())
     if F.tree.topLevelItem(i).data(0, Qt.UserRole + 1)[1].startswith("within")][0].child(1)
g.setExpanded(True)
F.compact_cb.setChecked(False)
fl = [F.tree.topLevelItem(i) for i in range(F.tree.topLevelItemCount())
      if F.tree.topLevelItem(i).data(0, Qt.UserRole + 1)[1].startswith("within")][0]
check("'Group by name parts' off → flat leaves under each family",
      fl.childCount() == 455 and fl.child(0).text(0) == "head_nose-head_mid-head_left")
F.compact_cb.setChecked(True)
fl = [F.tree.topLevelItem(i) for i in range(F.tree.topLevelItemCount())
      if F.tree.topLevelItem(i).data(0, Qt.UserRole + 1)[1].startswith("within")][0]
check("…and back on, expanded groups are remembered", fl.child(1).isExpanded())
from PySide6.QtTest import QTest  # noqa: E402

F.search.setText("hip_right")
app.processEvents()
before = sum(1 for it in F._walk() if F._is_leaf(it) and not it.isHidden())
QTest.qWait(350)
after = sum(1 for it in F._walk() if F._is_leaf(it) and not it.isHidden())
check(f"typing is debounced: the filter applies ~200 ms after the last key "
      f"({before} → {after} shown)", before > after > 0)
F.search.setText("")
F._apply_filter()
n_leaves = sum(1 for it in F._walk() if F._is_leaf(it))
check(f"the tree holds every column once ({n_leaves})",
      n_leaves == sum(len(c) for c in FAM.values()))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122jg_feature_tree_compact: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
