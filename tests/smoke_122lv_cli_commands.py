"""Smoke test for patch 122lv — ``mufasa help``, ``mufasa list …``, ``info``, ``project``, ``doctor``,
``version``.

User: "I will need a "mufasa help" command. "mufasa list models" and other useful calls".

A sample project (videos in a subfolder, a 2-D pose file, a 3-D run with a gait file beside it, a
Kalman body model, a classifier, a detector, a calibrated animal, a muscle set, a camera) and a
global model library in a private home folder; each command run as ``mufasa …`` would be, from
inside the project:

* ``mufasa help`` / ``--help``: the commands, ``list``'s subjects, the other tools; ``help list``;
* ``mufasa list models``: every kind, with the project's and the global library's and the built-in
  ones; ``--kind``, ``--json``;
* ``list videos`` (subfolders; ``--details``), ``pose``, ``3d`` (each with its video, not the gait
  file), ``runs``, ``calibrations``, ``steps``, ``tools``, ``projects``;
* ``mufasa info`` on a 3-D file, a video, a model, a camera, the project;
* ``mufasa project``; ``mufasa doctor`` (``--json``); ``mufasa version``; an unknown command;
* the project found from the current folder, or ``--project``, or the last one opened.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


tmp = Path(tempfile.mkdtemp())
home = tmp / "home"
home.mkdir()
ENV = {**os.environ, "HOME": str(home), "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
       "XDG_DATA_HOME": str(home / ".local" / "share"), "XDG_CONFIG_HOME": str(home / ".config")}


def mufasa(*args: str, cwd: Path | None = None) -> tuple[int, str, str]:
    r = subprocess.run([sys.executable, "-m", "mufasa.cli.workbench_launcher", *args], cwd=str(cwd or tmp),
                       env=ENV, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout, r.stderr


# ---- a sample project ------------------------------------------------------------------------
from mufasa.pose3d.camera import Camera  # noqa: E402
from mufasa.pose3d.io import write_pose3d  # noqa: E402
from mufasa.project_layout import (  # noqa: E402
    ProjectPaths,
    generate_run_id,
    project_paths_from_config,
)
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

proj = tmp / "lab"
proj.mkdir()
cfg = Path(ProjectConfigCreator(
    project_path=str(proj), project_name="cli", target_list=["rear", "groom"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0, animal_cnt=1, file_type="parquet",
    body_parts=["nose", "tail"]).config_path)
root = cfg.parent
P = project_paths_from_config(str(cfg))
vd = Path(P["video_dir"]) / "rat6"
vd.mkdir(parents=True)
w = cv2.VideoWriter(str(vd / "K7_cam.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
for k in range(45):
    w.write(np.full((48, 64, 3), k, np.uint8))
w.release()
(root / "sources" / "pose").mkdir(parents=True, exist_ok=True)
pd.DataFrame({"nose_x": np.zeros(45), "nose_y": 0.0, "nose_p": 1.0}).to_parquet(root / "sources" / "pose" / "K7_camDLC.parquet")
pp = ProjectPaths(root)
run = pp.stage_run_dir("pose3d")
write_pose3d(pd.DataFrame({"frame": np.arange(45), "nose_x": 0.0, "nose_y": 0.0, "nose_z": 40.0,
                           "tail_x": -50.0, "tail_y": 0.0, "tail_z": 20.0}),
             run / "K7_camDLC_3d.parquet", {"units": "mm", "fps": 30.0, "box_x": [-200, 200], "box_y": [-150, 150],
                                             "box_z": [0, 300], "box_tilt_deg": 12.0, "edges": []})
pd.DataFrame({"frame": np.arange(45), "speed_mm_s": 1.0}).to_parquet(run / "K7_camDLC_gait.parquet")
sm = pp.stage_run_dir("smoothed", run_id=generate_run_id(), flavor="kalman_v2")
pd.DataFrame({"a": [1]}).to_parquet(sm / "K7_camDLC.parquet")
mdir = Path(P["models_dir"])
m = mdir / "rat_body"
m.mkdir(parents=True)
np.savez(m / "model.npz", version="v2", fps=30.0, layout_segments=np.array([["nose", "tail"]], dtype=object))
(m / "card.toml").write_text('model_name = "rat_body"\nsource_path = "/data/train/rat_body.npz"\n')
(mdir / "generated_models").mkdir()
(mdir / "generated_models" / "rear.sav").write_bytes(b"x" * 100)
(mdir / "yolo_rat.pt").write_bytes(b"x" * 10)
bio = root / "derived" / "biomech"
bio.mkdir(parents=True)
(bio / "K7.animal.json").write_text(json.dumps({"name": "K7", "mass_g": 410}))
(bio / "rat.muscles.json").write_text(json.dumps({"muscles": [{"name": f"m{k}"} for k in range(38)]}))
cal = root / "sources" / "calibration"
cal.mkdir(parents=True)
cam = Camera.straight_down(np.array([32.0, 24.0]), 50.0, 500.0, (64, 48))
cam.box = {"x": [-200.0, 200.0], "y": [-150.0, 150.0], "z": [0.0, 300.0], "tilt_deg": 12.0}
cam.save(cal / "camera.toml")
gl = home / ".config" / "mufasa" / "models"
gl.mkdir(parents=True)
np.savez(gl / "lab_default.npz", version="v2", fps=60.0, layout_segments=np.array([["a"]], dtype=object))
inside = root / "sources" / "pose"                  # run from a folder inside the project

# ---- help -----------------------------------------------------------------------------------
rc, out, _ = mufasa("help")
rc2, out2, _ = mufasa("--help")
check("mufasa help (and --help): the commands, list's subjects, the other tools, the workbench's options",
      rc == 0 and rc2 == 0 and out == out2 and all(f"  {c} " in out for c in ("help", "list", "info", "project",
                                                                              "run", "doctor", "version"))
      and "models" in out and "calibrations" in out and "mufasa-pose3d" in out and "--no-recent" in out)
rc, out, _ = mufasa("help", "list")
check("mufasa help list: its options", rc == 0 and "--kind" in out and "--project" in out and "--json" in out)
rc, out, _ = mufasa("help", "run")
check("mufasa help run: the pipeline's steps", rc == 0 and "gait-extrapolation" in out)

# ---- list models ----------------------------------------------------------------------------
rc, out, err = mufasa("list", "models", cwd=inside)
check("mufasa list models, from inside the project: the project's Kalman body model (and where it came from), "
      "the global library's, the classifier, the detector, the calibrated animal, the muscle set, the built-in "
      "virtual rodent and height prior",
      rc == 0 and "the current folder's" in out and "rat_body" in out and "rat_body.npz" in out
      and "lab_default" in out and "global" in out and "rear" in out and "yolo_rat" in out
      and "K7" in out and "410 g" in out and "38 muscles" in out and "virtual_rodent" in out
      and "rat7m_trunk_prior" in out and "Kalman body model v2, fitted at 30 fps" in out)
rc, out, _ = mufasa("list", "models", "--kind", "smoothing", "--json", "--project", str(root))
js = json.loads(out)
check(f"--kind smoothing --json --project: {[m['name'] for m in js]}",
      rc == 0 and {m["name"] for m in js} == {"rat_body", "lab_default"} and all(m["kind"] == "smoothing" for m in js))
rc, out, _ = mufasa("list", "models", "--project-only", "--json", cwd=inside)
check("--project-only: the global library left out", rc == 0 and "lab_default" not in out and "rat_body" in out)

# ---- the other lists ------------------------------------------------------------------------
rc, out, _ = mufasa("list", "videos", "--details", cwd=inside)
check("list videos --details: in subfolders too, with frames, fps, pixels",
      rc == 0 and "rat6/K7_cam.mp4" in out and " 45 " in out and "64x48" in out)
rc, out, _ = mufasa("list", "pose", cwd=inside)
check("list pose: the 2-D pose files with their frames", rc == 0 and "K7_camDLC.parquet" in out and "45" in out)
rc, out, _ = mufasa("list", "3d", cwd=inside)
check("list 3d: the 3-D file with its video, not the gait file beside it",
      rc == 0 and "K7_camDLC_3d.parquet" in out and "K7_cam.mp4" in out and "_gait" not in out)
rc, out, _ = mufasa("list", "runs", cwd=inside)
rc2, out2, _ = mufasa("list", "runs", "--stage", "pose3d", cwd=inside)
check("list runs: by stage (pose3d, smoothed/kalman_v2); --stage pose3d: its runs",
      rc == 0 and "pose3d" in out and "smoothed/kalman_v2" in out and rc2 == 0 and run.name in out2
      and "kalman_v2" not in out2)
rc, out, _ = mufasa("list", "calibrations", cwd=inside)
check(f"list calibrations: the camera in a line ({out.strip().splitlines()[-1].strip()[:70]}…)",
      rc == 0 and "camera" in out and "500 mm above the floor" in out and "400 × 300 mm" in out and "12" in out)
rc, out, _ = mufasa("list", "steps")
rc2, out2, _ = mufasa("list", "tools")
check("list steps (the pipeline's), list tools (mufasa's commands)",
      rc == 0 and "3d-inference" in out and "dynamics" in out and rc2 == 0 and "mufasa-view3d" in out2)

# ---- the project, found three ways -----------------------------------------------------------
rc, out, err = mufasa("list", "videos")                     # outside, nothing opened yet
check("outside a project, none opened: says how to name one (exit 2)", rc == 2 and "--project" in err)
(home / ".config" / "mufasa").mkdir(parents=True, exist_ok=True)
(home / ".config" / "mufasa" / "recent").write_text(str(cfg) + "\n")
rc, out, _ = mufasa("list", "videos")
rc2, out2, _ = mufasa("list", "projects")
check("…the last one opened in the workbench is used then; list projects shows it",
      rc == 0 and "the last one opened" in out and "K7_cam.mp4" in out and rc2 == 0 and str(cfg) in out2)

# ---- info, project ----------------------------------------------------------------------------
rc, out, _ = mufasa("info", str(run / "K7_camDLC_3d.parquet"), str(vd / "K7_cam.mp4"), str(m / "model.npz"),
                    str(cal / "camera.toml"))
check("mufasa info: a 3-D file (labels, box, flare), a video (frames, fps), a model (and its card), a camera",
      rc == 0 and "3-D pose" in out and "2: nose, tail" in out and "walls flaring 12.0°" in out
      and "45" in out and "64x48" in out and "Kalman body model v2" in out and "/data/train/rat_body.npz" in out
      and "500 mm above the floor" in out)
rc, out, err = mufasa("info", str(tmp / "nope.mp4"))
check("info on a missing file: says so (exit 1)", rc == 1 and "no such file" in err)
rc, out, _ = mufasa("project", cwd=inside)
check("mufasa project: body parts, behaviours, what it holds",
      rc == 0 and "nose, tail" in out and "rear, groom" in out and "videos" in out
      and "1 smoothing" in out and "1 classifier" in out and "2 body" in out and "prior" not in out
      and "pose3d" in out)
rc, out, _ = mufasa("project", "--json", str(root))
js = json.loads(out)
check("mufasa project --json PATH", rc == 0 and js["videos"] == 1 and js["3-D files"] == 1 and js["cameras"] == 1)

# ---- doctor, version, unknown ---------------------------------------------------------------
rc, out, _ = mufasa("doctor", "--json")
js = json.loads(out)
names = [d["check"] for d in js]
check(f"mufasa doctor --json: {len(js)} checks ({', '.join(names[:6])}…)",
      rc in (0, 1) and {"python", "numpy", "PySide6", "torch", "mujoco", "ffmpeg", "GPUs", "display"} <= set(names))
rc, out, _ = mufasa("version")
rc2, out2, _ = mufasa("--version")
check(f"mufasa version / --version: '{out.strip()}'", rc == 0 and out.startswith("mufasa ") and out == out2)
rc, out, err = mufasa("lsit", "models")
check("an unknown command: says so, points to `mufasa help` (exit 2), no window opened",
      rc == 2 and "mufasa help" in err)
out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                      "from mufasa.cli import commands; commands.collect_models(None); "
                      "print('PySide6' in sys.modules)", str(REPO)], capture_output=True, text=True).stdout
from mufasa.cli import commands as C  # noqa: E402
from mufasa.cli import workbench_launcher as WL  # noqa: E402

check("the commands load without Qt; the launcher knows them all",
      out.strip() == "False" and set(WL.COMMANDS) == set(C.COMMANDS))

doc = (REPO / "docs" / "cli.md").read_text()
check("docs/cli.md: the commands", all(s in doc for s in ("mufasa help", "mufasa list models", "mufasa info",
                                                          "mufasa project", "mufasa doctor")))

for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
n_ok = sum(ok for _, ok in checks)
print(f"smoke_122lv_cli_commands: {n_ok}/{len(checks)} checks passed")
sys.exit(0 if n_ok == len(checks) else 1)
