"""Smoke test for patch 122ji — ROI features show "modified" when the ROIs change.

User request: "If additional ROI objects are added the ROI features under
Compute Features should show modified status."

ROI-family outputs now carry, in their parquet metadata, a fingerprint of
the ROIs they were computed from (per video). The Compute Features form
compares it with rois.json and badges a ROI family "● … — modified" (amber,
tooltip listing the videos) when ROIs were added, edited or removed, or a
video gained ROIs after the run. It re-checks when rois.json changes on
disk (file watcher), when the form is shown, and after a run.

Runs for real, offscreen: a generated project with two videos and ROIs,
FeatureSubsetsCalculator computing both ROI families, then ROI edits via
ROILogic / RoiStore and the form's badges after each.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import contextlib  # noqa: E402
import io  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

import mufasa.feature_extractors.feature_subsets as FS  # noqa: E402
from mufasa.feature_extractors.feature_subset_orchestration import read_roi_fingerprint  # noqa: E402
from mufasa.mixins.config_reader import ConfigReader  # noqa: E402
from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.roi_tools.roi_logic import ROILogic  # noqa: E402
from mufasa.roi_tools.roi_store import load_rois, update_rois  # noqa: E402
from mufasa.ui_qt.forms.features import FeatureSubsetExtractorForm  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402
from mufasa.utils.feature_io import family_slug, roi_family_status  # noqa: E402

tmp = Path(tempfile.mkdtemp())
bps = ["nose", "center", "tail"]
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="r", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=bps).config_path
P = project_paths_from_config(cfg)
vd = Path(P["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
cr = ConfigReader(config_path=cfg, read_video_info=False)
os.makedirs(cr.outlier_corrected_dir, exist_ok=True)
rng = np.random.default_rng(0)
rows = []
for v in ("v1", "v2"):
    w = cv2.VideoWriter(str(vd / f"{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for _ in range(3):
        w.write(np.zeros((360, 640, 3), np.uint8))
    w.release()
    rows.append([v, 30, 640, 360, 0, 2.0])
    cols = {}
    for i, b in enumerate(bps):
        cols[f"{b}_x"] = rng.uniform(0, 640, 100)
        cols[f"{b}_y"] = rng.uniform(0, 360, 100)
        cols[f"{b}_p"] = 0.99
    pd.DataFrame(cols).to_parquet(os.path.join(cr.outlier_corrected_dir, f"{v}.parquet"))
pd.DataFrame(rows, columns=["Video", "fps", "Resolution_width", "Resolution_height",
                            "Distance_in_mm", "pixels/mm"]).to_csv(P["video_info_path"], index=False)
S = dict(color_name="Red", bgr=(0, 0, 255), thickness=2, ear_tag_size=8)
for v in ("v1", "v2"):
    L = ROILogic(config_path=cfg, video_path=str(vd / f"{v}.mp4"))
    L.add_rectangle(name="left", top_left=(0, 0), bottom_right=(320, 360), **S)
    L.save()
ROI_FAMS = [FS.FRAME_BP_TO_ROI_CENTER, FS.FRAME_BP_INSIDE_ROI]
SLUGS = [family_slug(f) for f in ROI_FAMS]
roi_path = P["roi_definitions_path"]


def compute():
    with contextlib.redirect_stdout(io.StringIO()):
        FS.FeatureSubsetsCalculator(
            config_path=cfg, feature_families=ROI_FAMS, file_checks=True, save_dir=None,
            data_dir=None, derived_features_dir=P["derived_features_dir"], n_workers=1,
            overwrite_existing=True).run()


compute()
fd = Path(P["derived_features_dir"])
f1 = fd / SLUGS[0] / "v1.parquet"
store = load_rois(roi_path)
check("ROI-family outputs carry the fingerprint of the ROIs they used",
      read_roi_fingerprint(str(f1)) == store.fingerprint("v1") is not None)
check("…and still read as ordinary parquet", len(pd.read_parquet(f1)) == 100)
st = roi_family_status(cfg, SLUGS[1])
check("fresh run: current for both videos, not modified",
      st.current == ["v1", "v2"] and not st.modified)

form = FeatureSubsetExtractorForm(config_path=cfg)
form.show()
app.processEvents()


def badge(i: int):
    it = form.roi_families.item(i)
    return it.text(), it.data(Qt.UserRole + 1), it.toolTip()


check("form: both ROI families show ✓ computed",
      all(badge(i)[1] == "computed" and badge(i)[0].startswith("✓")
          for i in range(form.roi_families.count())))

# add a ROI object to v1
L = ROILogic(config_path=cfg, video_path=str(vd / "v1.mp4"))
L.add_circle(name="cup", center=(480, 180), radius=40, **S)
L.save()
st = roi_family_status(cfg, SLUGS[0])
check("adding a ROI to v1 → that video 'changed', v2 still current",
      st.changed == ["v1"] and st.current == ["v2"] and st.modified)
QTest.qWait(700)                     # file watcher + 300 ms debounce
t, state, tip = badge(0)
check("form picks the change up by itself (file watcher) and badges "
      f"'modified' ({t!r})", state == "modified" and t.startswith("●") and t.endswith("— modified"))
check("tooltip says what changed and where",
      "ROIs changed in 1 video(s)" in tip and "• ROIs changed: v1" in tip
      and "run Compute Features to update" in tip)
check("amber text", form.roi_families.item(0).foreground().color().getRgb()[:3] == (178, 106, 0))
check("subject families are unaffected",
      all(form.subject_families.item(i).data(Qt.UserRole + 1) != "modified"
          for i in range(form.subject_families.count())))

compute()
form._refresh_computed_badges()
check("recomputing clears the badge", all(badge(i)[1] == "computed"
                                          for i in range(form.roi_families.count())))

# moving a ROI (same name) is a change too


def _nudge(store):
    roi = store.rois("v2")[0]
    roi.geometry["x"] = roi.geometry["x"] + 10
    store.set_video("v2", [roi] + store.rois("v2")[1:])


update_rois(roi_path, _nudge)
st = roi_family_status(cfg, SLUGS[0])
check("moving / reshaping a ROI counts as changed", st.changed == ["v2"])
compute()

# ROIs removed from a video
update_rois(roi_path, lambda s_: s_.remove_video("v2"))
st = roi_family_status(cfg, SLUGS[0])
check("removing all ROIs of a video → 'removed'", st.removed == ["v2"] and st.modified)
form.hide()
form.show()                           # showEvent re-checks
app.processEvents()
check("form re-checks when shown", "ROIs removed from 1 video(s)" in badge(0)[2])

# output written before 122ji (no fingerprint), older than rois.json
update_rois(roi_path, lambda s_: s_.copy_video("v1", ["v2"]))
compute()
legacy = fd / SLUGS[0] / "v2.parquet"
pd.read_parquet(legacy).to_parquet(legacy)            # rewritten without metadata
old = time.time() - 100
os.utime(legacy, (old, old))
st = roi_family_status(cfg, SLUGS[0])
check("an untracked output older than rois.json is flagged 'unknown'",
      st.unknown == ["v2"] and st.modified)
os.utime(legacy, None)
os.utime(roi_path, (old, old))
st = roi_family_status(cfg, SLUGS[0])
check("…and treated as current when it is newer than rois.json",
      "v2" in st.current and not st.modified)

never = roi_family_status(cfg, "two_point_body_part_distances_mm")
check("a family never computed is not 'modified'", not never.modified and not never.computed)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ji_roi_features_modified_status: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
