"""Smoke test for patch 122id — pair a run's pose.parquet with its video.

The DeepLabCut exporter writes a fixed-name ``pose.parquet`` per analysis run,
with a run TOML beside it (named after the video) recording provenance:

    kind = "analyze"
    inputs = ["/data/.../<video>.mp4"]
    outputs = ["pose.parquet"]

The file name says nothing about the video, so pairing goes through that
record. Covers: read_run_sidecar (matched by ``outputs``, other TOMLs
ignored), pose_parquet_video_name, discovery of run subfolders (one level,
only with a matching sidecar), the importer naming the import after the video
and refusing two runs of one video, Add-sessions finding run subfolders, the
smoother naming its output after the video (writer and incremental skip
agree), and mufasa-preview pairing the run with the recorded video.
"""
from __future__ import annotations

import os
import sys
import tempfile
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mufasa.pose_importers import pose_parquet as PP  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


BPS = ["head_nose", "head_mid", "back_T4", "back_L2", "tail_V6"]
EDGES = [("head_nose", "head_mid"), ("head_mid", "back_T4"),
         ("back_T4", "back_L2"), ("back_L2", "tail_V6")]


def long_df(n=30):
    rows = []
    for f in range(n):
        for i, bp in enumerate(BPS):
            rows.append({"frame": f, "individual": "single", "bodypart": bp,
                         "x": 40.0 + 12.0 * i + f, "y": 60.0 + 3.0 * i,
                         "likelihood": 0.95})
    return pd.DataFrame(rows)


def make_run(parent: Path, video: Path, folder_name: str | None = None) -> Path:
    """A run folder as the exporter writes it: pose.parquet + <video>.toml."""
    run = parent / (folder_name or video.stem)
    run.mkdir(parents=True)
    PP.write_parquet_with_pose_metadata(long_df(), run / "pose.parquet", BPS, EDGES)
    (run / f"{video.stem}.toml").write_text(
        f'run_id = "{video.stem.lower()}"\nkind = "analyze"\n'
        f'status = "finished"\nmodel_id = "20261102-135253-c66d57"\n'
        f'inputs = ["{video}"]\noutputs = ["pose.parquet"]\n'
        f"schema_version = 1\n\n[params]\npcutoff = 0.6\n"
    )
    return run / "pose.parquet"


tmp = Path(tempfile.mkdtemp())
vids = tmp / "videos"
vids.mkdir()
VA = vids / "K7_cam_A_640x360-triplet.mp4"
VB = vids / "K7_cam_B_640x360-triplet.mp4"
runs = tmp / "analyze"
pa_ = make_run(runs, VA)
pb_ = make_run(runs, VB)

# --------------------------------------------------------------- sidecar ---
r = PP.read_run_sidecar(pa_)
check("run sidecar found by its outputs entry", r is not None
      and r["source"].endswith("K7_cam_A_640x360-triplet.toml"))
check("recorded video read from inputs", PP.run_sidecar_video(pa_) == str(VA))
check("pose.parquet named after its run's video",
      PP.pose_parquet_video_name(pa_) == "K7_cam_A_640x360-triplet")
(pa_.parent / "notes.toml").write_text('title = "x"\n')
(pa_.parent / "project.toml").write_text('outputs = ["pose.parquet"]\ninputs = ["/x.mp4"]\n')
check("other TOMLs (no inputs/outputs, project.toml) ignored",
      PP.run_sidecar_video(pa_) == str(VA))
lone = tmp / "lone"
lone.mkdir()
PP.write_parquet_with_pose_metadata(long_df(), lone / "rec9.parquet", BPS, EDGES)
(lone / "other.toml").write_text('inputs = ["/v/zzz.mp4"]\noutputs = ["something_else.parquet"]\n')
check("a sidecar for another output doesn't claim the file",
      PP.read_run_sidecar(lone / "rec9.parquet") is None
      and PP.pose_parquet_video_name(lone / "rec9.parquet") == "rec9")
check("embedded skeleton still read from a run's pose.parquet",
      PP.read_pose_skeleton(pa_)["edges"] == EDGES)

# ------------------------------------------------------------- discovery ---
found = PP.find_long_pose_parquets(runs)
check("discovery descends one level into run folders",
      sorted(found) == sorted([str(pa_), str(pb_)]))
deep = runs / "K7_cam_A_640x360-triplet" / "deeper"
deep.mkdir()
PP.write_parquet_with_pose_metadata(long_df(), deep / "pose.parquet", BPS, EDGES)
plain_sub = runs / "scratch"
plain_sub.mkdir()
PP.write_parquet_with_pose_metadata(long_df(), plain_sub / "x.parquet", BPS, EDGES)
check("no deeper descent; subfolders without a run sidecar ignored",
      sorted(PP.find_long_pose_parquets(runs)) == sorted([str(pa_), str(pb_)]))

# -------------------------------------------------------------- importer ---
from mufasa.pose_importers.fdlc_parquet_importer import (  # noqa: E402
    FDLCParquetImporter,
)
from mufasa.project_layout import read_skeleton  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402


def new_project(name):
    return ProjectConfigCreator(
        project_path=str(tmp), project_name=name, target_list=[],
        pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
        animal_cnt=1, file_type="parquet", body_parts=list(BPS),
    ).config_path


cfg = new_project("p_runs")
FDLCParquetImporter(config_path=cfg, data_folder=str(runs)).run()
pose_dir = Path(cfg).parent / "sources" / "pose"
check("importer names each run's import after its video",
      sorted(os.listdir(pose_dir))
      == ["K7_cam_A_640x360-triplet.parquet", "K7_cam_B_640x360-triplet.parquet"])
check("run's embedded skeleton reaches project.toml",
      read_skeleton(cfg)["edges"] == EDGES)

dup = tmp / "dup"
make_run(dup, VA, folder_name="run1")
make_run(dup, VA, folder_name="run2")
try:
    FDLCParquetImporter(config_path=new_project("p_dup"), data_folder=str(dup))
    check("two runs of the same video are refused", False)
except Exception as exc:
    check("two runs of the same video are refused",
          "K7_cam_A_640x360-triplet" in str(exc) and "overwrite" in str(exc))

# ----------------------------------------------------------- add sessions ---
from mufasa.model.session_ingest import check_pose_compatibility  # noqa: E402

rep = check_pose_compatibility(str(runs), cfg)
check("Add-sessions finds run subfolders",
      sorted(rep["accepted"]) == sorted([str(pa_), str(pb_)]))

# --------------------------------------------------------------- smoother ---
from mufasa.data_processors import kalman_pose_smoother_v2 as K  # noqa: E402

check("smoother output stem = recorded video",
      K._session_output_stem(pa_) == "K7_cam_A_640x360-triplet")
check("smoother output stem unchanged for ordinary files",
      K._session_output_stem(lone / "rec9.parquet") == "rec9"
      and K._session_output_stem(Path("/x/a.fdlc.parquet")) == "a.fdlc")
T = 10
out, _, _ = K._build_and_write_session_output(
    {"path": pa_, "markers": BPS, "likelihoods": np.full((T, len(BPS)), 0.9)},
    np.zeros((T, len(BPS), 2)), np.ones((T, len(BPS), 2)),
    {m: i for i, m in enumerate(BPS)}, pa_.parent, param_hash="ab12",
)
check("smoothed output named <video>_smoothed_v2.<hash>.parquet",
      out is not None and out.name == "K7_cam_A_640x360-triplet_smoothed_v2.ab12.parquet")
src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
check("incremental skip uses the same stem as the writer",
      '_stem = _session_output_stem(raw_sessions[_i]["path"])' in src
      and 'stem = _session_output_stem(s["path"])' in src)

# ----------------------------------------------------------------- preview ---
import cv2  # noqa: E402

w = cv2.VideoWriter(str(VA), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
w.write(np.zeros((48, 64, 3), np.uint8))
w.release()
from mufasa.tools import pose_preview as PV  # noqa: E402

g = PV._discover(str(pa_.parent))
check("preview pairs the run with its recorded video and smoothed output",
      len(g) == 1 and g[0]["video"] == str(VA) and g[0]["raw"] == str(pa_)
      and g[0]["smoothed"] == str(out))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122id_run_sidecar_pairing: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
