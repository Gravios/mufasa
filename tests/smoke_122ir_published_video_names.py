"""Smoke test for patch 122ir — published Kalman runs use video names.

User report: Compute Features → "ROIs are defined for 39 video(s)
(K7_cam_…-triplet, …), but no tracking file has one of those names
(tracking files: K7_cam_…-triplet_smoothed_v2.4d4273a837d0cc40, …)".
Kalman v2 writes ``<video>_smoothed_v2.<hash>.parquet``; publishing linked the
run directory as a whole into derived/outlier_corrected/, and every
downstream tool takes the file name as the video name — so ROIs and the
calibration (keyed by video name) never matched.

Now publish_to_stage presents such runs as a directory of per-file links
named ``<video>.parquet``, and ConfigReader upgrades runs published the old
way when a project is opened.

Runs for real: a generated project, a Kalman-style run on disk published the
old way (directory symlink, exactly as on the user's disk), ConfigReader,
and FeatureSubsetsCalculator with ROI + calibration-dependent families.
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
from mufasa.project_layout import (  # noqa: E402
    project_paths_from_config,
    publish_to_stage,
    refresh_published_runs,
    video_name_from_pose_stem,
)
from mufasa.roi_tools.roi_logic import ROILogic  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


check("video_name_from_pose_stem strips the smoother suffix and hash only",
      video_name_from_pose_stem("K7_cam_A-0-051_640x360-triplet_smoothed_v2.4d4273a837d0cc40")
      == "K7_cam_A-0-051_640x360-triplet"
      and video_name_from_pose_stem("rec_smoothed_v2") == "rec"
      and video_name_from_pose_stem("rec.v2") == "rec.v2"
      and video_name_from_pose_stem("plain") == "plain")

HASH = "4d4273a837d0cc40"
VIDEOS = ["K7_cam_20251124_1510_A10-0-051_640x360-triplet",
          "K7_cam_20251124_1625_A12-0-052_640x360-triplet"]
tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="p", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["nose", "tail"],
).config_path
P = project_paths_from_config(cfg)
root = Path(P["project_root"])
vd = Path(P["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
for v in VIDEOS:
    w = cv2.VideoWriter(str(vd / f"{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                        30, (640, 360))
    for _ in range(3):
        w.write(np.zeros((360, 640, 3), np.uint8))
    w.release()
pd.DataFrame({"Video": VIDEOS, "fps": 30, "Resolution_width": 640,
              "Resolution_height": 360, "Distance_in_mm": 100,
              "pixels/mm": 2.0}).to_csv(P["video_info_path"], index=False)
S = dict(color_name="Red", bgr=(0, 0, 255), thickness=2, ear_tag_size=8)
for v in VIDEOS:
    L = ROILogic(config_path=cfg, video_path=str(vd / f"{v}.mp4"))
    L.add_rectangle(name="left", top_left=(0, 0), bottom_right=(320, 360), **S)
    L.save()

# A Kalman v2 run, published the pre-122ir way (directory symlink).
run_id = "20261105-063000-abcdef"
src = root / "derived" / "smoothed" / "kalman_v2" / run_id
src.mkdir(parents=True)
xs = np.linspace(100, 500, 60)
for v in VIDEOS:
    pd.DataFrame({"nose_x": xs, "nose_y": 180.0, "nose_p": 1.0,
                  "tail_x": xs - 20, "tail_y": 180.0, "tail_p": 1.0}
                 ).to_parquet(src / f"{v}_smoothed_v2.{HASH}.parquet")
(src / "run.toml").write_text("[run]\nproducer = 'kalman_v2'\n")
oc = root / "derived" / "outlier_corrected"
oc.mkdir(parents=True, exist_ok=True)
(oc / run_id).symlink_to(Path("..") / "smoothed" / "kalman_v2" / run_id,
                         target_is_directory=True)
# A real (non-published) run must never be touched.
real = oc / "20250101-000000-111111"
real.mkdir()
(real / "something_smoothed_v2.abc123.parquet").write_bytes(b"data")

cr = ConfigReader(config_path=cfg, read_video_info=False)
pub = oc / run_id
names = sorted(p.name for p in pub.iterdir() if not p.name.startswith("."))
check("opening the project re-publishes the old run under video names",
      not pub.is_symlink()
      and names == sorted([f"{v}.parquet" for v in VIDEOS] + ["run.toml"]))
check("the published files are relative links to the Kalman output (no copies)",
      all((pub / f"{v}.parquet").is_symlink()
          and not os.path.isabs(os.readlink(pub / f"{v}.parquet"))
          and (pub / f"{v}.parquet").resolve()
          == (src / f"{v}_smoothed_v2.{HASH}.parquet").resolve()
          for v in VIDEOS))
check("a real data directory in the stage is left alone",
      (real / "something_smoothed_v2.abc123.parquet").read_bytes() == b"data"
      and not (real / ".published_from").exists())
check("ConfigReader resolves the published run", cr.outlier_corrected_dir == str(pub))
check("refreshing again changes nothing (idempotent)",
      refresh_published_runs(oc) == [])

calc = FS.FeatureSubsetsCalculator(
    config_path=cfg,
    feature_families=[FS.FRAME_BP_INSIDE_ROI, FS.FRAME_BP_TO_ROI_CENTER,
                      FS.FRAME_BP_MOVEMENT],
    file_checks=False, save_dir=None, data_dir=None,
    derived_features_dir=P["derived_features_dir"], n_workers=1,
    overwrite_existing=False)
check("features see the tracking files under their video names",
      sorted(calc.video_names) == sorted(VIDEOS))
calc.run()
out = sorted(Path(P["derived_features_dir"]).rglob("*.parquet"))
check("ROI + calibration features are computed for every video "
      "(the user's error is gone)",
      len(out) == 3 * len(VIDEOS)
      and {p.stem for p in out} == set(VIDEOS))
inside = pd.read_parquet(next(p for p in out if "inside_rois" in str(p)))
col = next(c for c in inside.columns if "nose inside rectangle left" in c)
check("…with correct values", np.array_equal(
    inside[col].to_numpy().astype(bool), xs <= 320))

# A fresh publish of a Kalman-style run goes straight to video names.
run2 = "20261105-070000-fedcba"
src2 = root / "derived" / "smoothed" / "kalman_v2" / run2
src2.mkdir(parents=True)
pd.DataFrame({"a": [1]}).to_parquet(src2 / f"{VIDEOS[0]}_smoothed_v2.{HASH}.parquet")
p2 = publish_to_stage(root, "smoothed", "outlier_corrected", run2,
                      source_flavor="kalman_v2")
check("publish_to_stage publishes new Kalman runs under video names",
      (p2 / f"{VIDEOS[0]}.parquet").is_symlink() and not p2.is_symlink())
check("…and republishing is idempotent",
      publish_to_stage(root, "smoothed", "outlier_corrected", run2,
                       source_flavor="kalman_v2") == p2
      and sorted(x.name for x in p2.iterdir())
      == [".published_from", f"{VIDEOS[0]}.parquet"])
run3 = "20261105-080000-aaaaaa"
src3 = root / "derived" / "interpolated" / run3
src3.mkdir(parents=True)
pd.DataFrame({"a": [1]}).to_parquet(src3 / f"{VIDEOS[0]}.parquet")
p3 = publish_to_stage(root, "interpolated", "outlier_corrected", run3)
check("runs already named by video keep the plain directory link",
      p3.is_symlink())

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ir_published_video_names: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
