"""Smoke test for patch 122iq — ROI-feature "no coverage" error says why.

User report: "NO ROI DATA ERROR: … none of the 39 videos have all required
ROIs defined". That message didn't distinguish the two causes:

* ROIs exist but under video names that no tracking file has;
* names match, but videos carry different ROI names (each ROI is a feature
  column, so every video must have the same set — easy to break with the
  editor's automatic names).

Runs for real: generated projects, ROIs via ROILogic, synthetic tracking,
FeatureSubsetsCalculator.run with the ROI families.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mufasa.feature_extractors.feature_subsets as FS  # noqa: E402
from mufasa.mixins.config_reader import ConfigReader  # noqa: E402
from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.roi_tools.roi_logic import ROILogic  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402
from mufasa.utils.errors import NoROIDataError  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


S = dict(color_name="Red", bgr=(0, 0, 255), thickness=2, ear_tag_size=8)


def project(videos, data_names, rois):
    tmp = Path(tempfile.mkdtemp())
    cfg = ProjectConfigCreator(
        project_path=str(tmp), project_name="c", target_list=[],
        pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
        animal_cnt=1, file_type="parquet", body_parts=["nose", "tail"],
    ).config_path
    P = project_paths_from_config(cfg)
    vd = Path(P["video_dir"])
    vd.mkdir(parents=True, exist_ok=True)
    for v in videos:
        w = cv2.VideoWriter(str(vd / f"{v}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
        for _ in range(3):
            w.write(np.zeros((360, 640, 3), np.uint8))
        w.release()
    names = sorted(set(videos) | set(data_names))
    pd.DataFrame({"Video": names, "fps": 30, "Resolution_width": 640,
                  "Resolution_height": 360, "Distance_in_mm": 100,
                  "pixels/mm": 2.0}).to_csv(P["video_info_path"], index=False)
    for v, roi_names in rois.items():
        L = ROILogic(config_path=cfg, video_path=str(vd / f"{v}.mp4"))
        for i, n in enumerate(roi_names):
            L.add_rectangle(name=n, top_left=(10 + i, 10), bottom_right=(100, 100), **S)
        L.save()
    cr = ConfigReader(config_path=cfg, read_video_info=False)
    os.makedirs(cr.outlier_corrected_dir, exist_ok=True)
    df = pd.DataFrame({"nose_x": np.linspace(0, 300, 50), "nose_y": 50.0,
                       "nose_p": 1.0, "tail_x": 20.0, "tail_y": 50.0,
                       "tail_p": 1.0})
    for d in data_names:
        df.to_parquet(os.path.join(cr.outlier_corrected_dir, f"{d}.parquet"))
    return cfg, P


def run_roi_features(cfg, P) -> str:
    calc = FS.FeatureSubsetsCalculator(
        config_path=cfg,
        feature_families=[FS.FRAME_BP_TO_ROI_CENTER, FS.FRAME_BP_INSIDE_ROI],
        file_checks=False, save_dir=None, data_dir=None,
        derived_features_dir=P["derived_features_dir"], n_workers=1,
        overwrite_existing=False)
    try:
        calc.run()
        return ""
    except NoROIDataError as exc:
        return str(getattr(exc, "msg", exc))


# 1. ROI video names don't match the tracking file names.
cfg, P = project(["camA_session1"], ["session1_pose"], {"camA_session1": ["arena"]})
m = run_roi_features(cfg, P)
check("name mismatch: the error says ROI videos and tracking files differ",
      "no tracking file has one of those names" in m)
check("…and shows both sides' names",
      "camA_session1" in m and "session1_pose" in m)

# 2. Matching names, but inconsistent ROI names across videos.
cfg, P = project(["v1", "v2"], ["v1", "v2"],
                 {"v1": ["arena"], "v2": ["rectangle_1"]})
m = run_roi_features(cfg, P)
check("inconsistent ROI names: the error explains every video needs the same names",
      "same ROI names" in m and "feature column" in m)
check("…lists the names used and what a video lacks",
      "arena" in m and "rectangle_1" in m and "v1 lacks rectangle_1" in m)

# 3. Consistent names: ROI features are computed.
cfg, P = project(["v1", "v2"], ["v1", "v2"],
                 {"v1": ["arena"], "v2": ["arena"]})
m = run_roi_features(cfg, P)
out = list(Path(P["derived_features_dir"]).rglob("*.parquet"))
check("with the same ROI names on every video, ROI features are written",
      m == "" and len(out) == 4)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122iq_roi_coverage_message: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
