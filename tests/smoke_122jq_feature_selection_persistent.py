"""Smoke test for patch 122jq — Feature Visualization selection kept
between Mufasa sessions.

User request: "Can you also make the feature selection persistent between
mufasa sessions?"

The selection (in order), video, colour-by, joint style, jitter and the
"Group by name parts" setting are saved (debounced, and on quit) to
logs/measures/feature_view.toml and restored when the form is built again.
Selected features the current video lacks are parked, not dropped.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import tomllib
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="ls", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
P = project_paths_from_config(cfg)
vd, fd = Path(P["video_dir"]), Path(P["derived_features_dir"])
fam = fd / "frame_by_frame_body_part_movements_mm"
for d in (vd, fam):
    d.mkdir(parents=True, exist_ok=True)
A = "Animal_1 movement a (mm)_FEATURE_SUBSET"
B = "Animal_1 movement b (mm)_FEATURE_SUBSET"
C = "Animal_1 movement c (mm)_FEATURE_SUBSET"
ANG = "Animal_1 heading angle (degrees)_FEATURE_SUBSET"
OTHER = "Animal_1 unselected (mm)_FEATURE_SUBSET"
N = 1500
r = np.random.default_rng(3)
for v in ("v1", "v2"):
    w = cv2.VideoWriter(str(vd / f"{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(5):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    g = r.integers(0, 2, N)
    pd.DataFrame({
        A: r.normal(g * 4, 0.6),
        B: r.normal(g * -3, 0.7),
        C: r.normal(0, 1, N),
        ANG: r.uniform(0, 360, N),
        OTHER: r.normal(0, 1, N),
    }).to_parquet(fam / f"{v}.parquet")
# v3 lacks C
df = pd.read_parquet(fam / "v1.parquet").drop(columns=[C])
df.to_parquet(fam / "v3.parquet")
w = cv2.VideoWriter(str(vd / "v3.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
for _ in range(5):
    w.write(np.zeros((48, 64, 3), np.uint8))
w.release()

from mufasa.ui_qt.forms.feature_visualization import FeatureVisualizationForm  # noqa: E402
from mufasa.utils import feature_view_state as st  # noqa: E402


def settle(sec=0.8):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.02)


check("state file is logs/measures/feature_view.toml",
      st.state_path(cfg).parts[-3:] == ("logs", "measures", "feature_view.toml"))
F = FeatureVisualizationForm(config_path=cfg)
check("fresh project: nothing selected, no file yet",
      F.selected_features() == [] and not st.state_path(cfg).exists())
F.video_cb.setCurrentText("v2")
F.set_selected([C, A])
F.joint_style_cb.setCurrentText("scatter")
F.jitter_cb.setChecked(True)
F.compact_cb.setChecked(not F.compact_cb.isChecked())
grouping = F.compact_cb.isChecked()
settle()
with open(st.state_path(cfg), "rb") as fh:
    data = tomllib.load(fh)
check("saved after a short pause (debounced)",
      data["selected"] == [C, A] and data["video"] == "v2" and data["joint_style"] == "scatter"
      and data["jitter"] is True and data["group_by_name_parts"] == grouping)
F.colour_cb.setCurrentText("rear")
F._flush_state()
check("pending save written at once on quit",
      st.load_state(cfg).get("colour_by") == "rear")

G = FeatureVisualizationForm(config_path=cfg)
check("next session: selection restored in order", G.selected_features() == [C, A])
check("next session: video, colour-by, style, jitter, grouping restored",
      G.video_cb.currentText() == "v2" and G.colour_cb.currentText() == "rear"
      and G.joint_style_cb.currentText() == "scatter" and G.jitter_cb.isChecked()
      and G.compact_cb.isChecked() == grouping)
ticked = {it.data(0, Qt.UserRole) for it in G._walk()
          if G._is_leaf(it) and it.checkState(0) == Qt.Checked}
check("restored features ticked in the tree", ticked == {A, C})
check("restoring does not rewrite the file",
      not G._state_timer.isActive())

G.video_cb.setCurrentText("v3")
check("video lacking a selected feature: it is parked, not shown",
      G.selected_features() == [A] and G._parked == [C])
settle()
check("parked feature still saved", set(st.load_state(cfg)["selected"]) == {A, C})
G.video_cb.setCurrentText("v1")
check("back on a video that has it: selected again", set(G.selected_features()) == {A, C})
G._clear_selection()
settle()
check("clear selection is saved too", st.load_state(cfg)["selected"] == [])

st.state_path(cfg).write_text("not = [valid")
H = FeatureVisualizationForm(config_path=cfg)
check("unreadable state file: starts empty, no crash", H.selected_features() == [])
st.save_state(cfg, {"selected": [A, "gone_FEATURE_SUBSET"], "video": "vanished"})
K = FeatureVisualizationForm(config_path=cfg)
check("unknown video / feature ignored on restore",
      K.selected_features() == [A] and K.video_cb.currentText() in ("v1", "v2", "v3"))
st.save_state(cfg, {"custom": 3, "video": None})
check("save merges (other keys kept, None removes)",
      st.load_state(cfg).get("custom") == 3 and "video" not in st.load_state(cfg))

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jq_feature_selection_persistent: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
