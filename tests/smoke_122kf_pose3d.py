"""Smoke test for patch 122kf — 3-D pose pipeline and the "3-D" tab.

User: "write up a pipeline and tab for 3d inference and kalman filtering.
You can scavange calibration tools from rpimocap and or the tools you
already made, include models as data for upload to the git as well."

* the packaged Rat7M prior loads and predicts trunk heights of an unseen
  rat (leave-one-rat-out, from the packaged table alone);
* cameras: projection/rays/heights, save/load, lens distortion round trip,
  calibration from marked box edges (recovers a known camera), from a
  rendered checkerboard (lens), and DLT from 3-D points;
* single-camera inference on a synthetic rat: head within a few mm, every
  label on its camera ray, tail on/above the floor;
* 3-D Kalman (RTS) smoothing reduces error and gates outliers;
* two-camera triangulation;
* the project pipeline through the tab's forms (calibrate → infer →
  smooth → viewer), the box-marking dialog, the CLI, the entry point and
  the workbench tab.
"""
from __future__ import annotations

import json
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


# --------------------------------------------------------------- synthetic rat
def make(T=1200, seed=0, cam=None, noise=1.0):
    rng=np.random.default_rng(seed); t=np.arange(T)/30
    def sn(sd,cut=40):
        x=np.cumsum(rng.normal(0,1,T)); x=np.convolve(x-x.mean(),np.ones(cut)/cut,"same"); return sd*x/(x.std()+1e-9)
    cx=110*np.sin(t*.31); cy=35*np.sin(t*.47+1)
    yaw=np.unwrap(np.arctan2(np.gradient(cy),np.gradient(cx)))
    rear=np.clip(np.sin(t*.25)*2.2-1.2,0,1)
    R=lambda yw,pt,rl:Rot.from_euler("ZYX",np.c_[yw,-pt,rl]).as_matrix()
    HEAD={"head_back":[0,0,8],"head_mid":[22,0,12],"head_nose":[42,0,0],"head_left":[4,18,6],"head_right":[4,-18,6]}
    hips_c=np.c_[cx,cy,42+15*rear]
    fwd=np.c_[np.cos(yaw),np.sin(yaw),np.zeros(T)]; lat=np.c_[-np.sin(yaw),np.cos(yaw),np.zeros(T)]
    elev=np.deg2rad(8+60*rear)
    tr=np.c_[np.cos(yaw)*np.cos(elev),np.sin(yaw)*np.cos(elev),np.sin(elev)]
    X={}
    X["hip_left"]=hips_c+25*lat; X["hip_right"]=hips_c-25*lat
    L6=hips_c+12*fwd+np.c_[0*t,0*t,18+0*t]; X["back_L6"]=L6
    X["back_V2"]=hips_c-30*fwd
    for name,f in (("back_L2",30),("back_T8",70),("back_T4",105)):
        X[name]=L6+f*tr+np.c_[0*t,0*t,12*np.sin(np.pi*f/120)*(1-rear)]
    neck=X["back_T4"]+35*np.c_[np.cos(yaw)*np.cos(elev*.6),np.sin(yaw)*np.cos(elev*.6),np.sin(elev*.6)]
    Rh=R(yaw+np.deg2rad(sn(20)),np.deg2rad(-15+30*rear+sn(10)),np.deg2rad(sn(8)))
    for m,o in HEAD.items(): X[m]=neck+np.einsum("tij,j->ti",Rh,np.array(o,float))
    tb=X["back_V2"]; X["tail_V6"]=tb-40*fwd; X["tail_V6"][:,2]=8
    X["tail_V18"]=X["tail_V6"]-70*fwd+20*lat; X["tail_V18"][:,2]=5
    X["tail_V32"]=X["tail_V18"]-70*fwd-10*lat; X["tail_V32"][:,2]=5
    cols={}
    for m,P in X.items():
        uv=cam.project(P)+rng.normal(0,noise,(T,2))
        lk=np.full(T,.97); lk[rng.random(T)<.05]=.2
        cols[m+"_x"],cols[m+"_y"],cols[m+"_p"]=uv[:,0],uv[:,1],lk
    return X,pd.DataFrame(cols),rear


from mufasa.pose3d.camera import (  # noqa: E402
    Camera,
    calibrate_checkerboard,
    calibrate_from_box_marks,
    camera_from_projection_matrix,
    dlt_camera_matrix,
)
from mufasa.pose3d.kalman3d import Kalman3DConfig, smooth_markers  # noqa: E402
from mufasa.pose3d.mono import MonoConfig, Rat7MPrior, load_pose2d, reconstruct_mono  # noqa: E402
from mufasa.pose3d.stereo import triangulate  # noqa: E402

tmp = Path(tempfile.mkdtemp())

# ------------------------------------------------------------------ prior
z = np.load(REPO / "mufasa/assets/models/pose3d/rat7m_trunk_prior.npz")
check("prior packaged: 80 000 Rat7M frames, 5 rats, bend tables, attribution",
      len(z["features"]) == 80000 and len(set(z["subject"].tolist())) == 5
      and z["bend_spineM"].shape == (6, 3) and "CC BY 4.0" in str(z["source"])
      and (REPO / "mufasa/assets/models/pose3d/README.md").is_file()
      and (REPO / "scripts/build_rat7m_prior.py").is_file())
test_rat = 5
pr = Rat7MPrior(subjects=[s for s in set(z["subject"].tolist()) if s != test_rat])
te = z["subject"] == test_rat
Y = pr.predict(*z["features"][te].T)
err = np.abs(Y[:, :4] - z["targets"][te])
check(f"unseen rat: trunk heights within {np.median(err):.0f} mm median (< 15)", np.median(err) < 15)
check("lumbar angle predicted (median error < 15 deg)", np.median(np.abs(Y[:, 4] - z["lumbar"][te])) < 15)

# ------------------------------------------------------------------ cameras
cam = Camera.straight_down((330, 190), 243.0, 430.0, (640, 360), center_xy=(-5, 15),
                           box={"x": [-340, 340], "y": [-260, 260], "z": [0, 300]})
P = np.array([[10.0, -40, 0], [100, 80, 120], [-200, 30, 250]])
uv = cam.project(P)
check("straight-down: ray at the right height returns the point",
      np.allclose(cam.point_at_height(uv, P[:, 2]), P, atol=1e-6))
cam.save(tmp / "cam.toml")
c2 = Camera.load(tmp / "cam.toml")
check("camera saved and loaded", np.allclose(c2.P, cam.P) and c2.box == cam.box)
cd = Camera(cam.K, cam.R, cam.t, np.array([-0.18, 0.05, 0.001, -0.001, 0.0]), cam.image_size)
check("lens distortion: undistort(project) = ideal pixels",
      np.allclose(cd.undistort(cd.project(P)), cam.project(P), atol=0.05))
L_, W_, H_ = 554.0, 300.0, 300.0
floor = np.array([[-L_ / 2, W_ / 2, 0], [L_ / 2, W_ / 2, 0], [L_ / 2, -W_ / 2, 0], [-L_ / 2, -W_ / 2, 0]])
edges = [[cam.project(np.array([[x, y, h]]))[0].tolist() for h in (0, 40, 80, 120)] for x, y, _ in floor]
marks = {"image_size": [640, 360], "floor_corners": cam.project(floor).tolist(), "vertical_edges": edges}
cb, rep = calibrate_from_box_marks(marks["floor_corners"], floor_length_mm=L_, floor_width_mm=W_,
                                   wall_height_mm=H_, vertical_edges_px=edges, camera_height_mm=430,
                                   image_size=(640, 360))
check(f"box marks: principal point and scale recovered ({rep.principal_px[0]:.1f}, {rep.principal_px[1]:.1f}; {rep.px_per_mm:.3f} px/mm)",
      np.allclose(rep.principal_px, (330, 190), atol=0.5) and abs(rep.px_per_mm - 243 / 430) < 0.003)
check("box marks: the floor lands where it is", np.allclose(cb.point_at_height(np.array(marks["floor_corners"]), np.zeros(4))[:, :2],
                                                           floor[:, :2], atol=2.0))
_, rep2 = calibrate_from_box_marks(marks["floor_corners"], floor_length_mm=400, floor_width_mm=300,
                                   camera_height_mm=430, image_size=(640, 360))
check("box marks: a size that does not fit the floor is reported (and the width kept)",
      any("as long as wide" in w for w in rep2.warnings) and abs(rep2.floor_mm[1] - 300) < 1)
# checkerboard rendered by a known lens
board = np.full((7 * 40 + 80, 10 * 40 + 80), 255, np.uint8)
for r in range(7):
    for c in range(10):
        if (r + c) % 2 == 0:
            board[40 + r * 40:80 + r * 40, 40 + c * 40:80 + c * 40] = 0
Ktrue = np.array([[520.0, 0, 330], [0, 520.0, 245], [0, 0, 1]])
(tmp / "cb").mkdir()
rng = np.random.default_rng(3)
for i in range(12):
    rv = rng.normal(0, 0.25, 3)
    tv = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(420, 600)])
    Rm = cv2.Rodrigues(rv)[0]
    ob = np.array([[0, 0, 0], [board.shape[1], 0, 0], [board.shape[1], board.shape[0], 0], [0, board.shape[0], 0]], float) * 0.5
    ob -= ob.mean(0)
    im_pts = (Ktrue @ ((ob @ Rm.T) + tv).T).T
    im_pts = im_pts[:, :2] / im_pts[:, 2:]
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [board.shape[1], 0], [board.shape[1], board.shape[0]], [0, board.shape[0]]]),
                                     np.float32(im_pts))
    img = cv2.warpPerspective(board, Hm, (640, 480), borderValue=200)
    cv2.imwrite(str(tmp / "cb" / f"{i:02d}.png"), img)
K, dist, rms, nv, size = calibrate_checkerboard(str(tmp / "cb"), (9, 6), 20.0)
check(f"checkerboard (from rpimocap): lens recovered, focal {K[0, 0]:.0f} px (true 520), rms {rms:.2f} px",
      abs(K[0, 0] - 520) / 520 < 0.03 and nv >= 8)
cam_pnp, rep3 = calibrate_from_box_marks(marks["floor_corners"], floor_length_mm=L_, floor_width_mm=W_,
                                         K=cam.K, dist=np.zeros(5), image_size=(640, 360))
check(f"box + lens: camera height solved ({cam_pnp.height:.0f} mm, true 430)", abs(cam_pnp.height - 430) < 5)
corners3d = np.r_[floor, floor + [0, 0, H_]]
Pd = dlt_camera_matrix(corners3d, cam.project(corners3d))
cdlt = camera_from_projection_matrix(Pd, (640, 360))
check("DLT (from rpimocap): camera recovered from box corners at two heights",
      np.allclose(cdlt.center, cam.center, atol=1.0))

# ------------------------------------------------------------------ single camera
X, df2d, rear = make(900, cam=cam)
# the synthetic head's heights relative to its ear line (back +2, mid +6, nose −6)
SYN_Z = {"head_back": 2.0, "head_mid": 6.0, "head_nose": -6.0}
out, rep = reconstruct_mono(load_pose2d(df2d), cam, MonoConfig(chunk=900, head_template_z=SYN_Z))
e = {m: np.linalg.norm(out[[m + "_x", m + "_y", m + "_z"]].to_numpy() - X[m], axis=1) for m in X}
he = np.concatenate([e[m] for m in ("head_back", "head_nose", "head_left", "head_right")])
check(f"single camera: head {np.median(he):.1f} mm median, {np.percentile(he, 90):.1f} mm p90",
      np.median(he) < 5 and np.percentile(he, 90) < 15)
check("head shape: outline learned, heights fixed from the config",
      abs(rep["head_template_mm"]["head_nose"][2] + 6.0) < 1e-6 and rep["head_template_reproj_px"] < 1.5)
check("steps: rigid head, Rat7M trunk, pelvis, spine chain, tail",
      rep["steps"][:2] == ["rigid head", "Rat7M trunk heights"] and "tail draped" in rep["steps"]
      and any(s.startswith("spine chain") for s in rep["steps"]))
trunk = ["back_T4", "back_T8", "back_L2", "back_L6", "back_V2", "hip_left", "tail_V32"]
check("trunk and tail labels stay on their camera rays (< 0.5 px)",
      all(np.nanmedian(out[m + "_reproj_px"]) < 0.5 for m in trunk))
check("head reprojects within 2 px", np.nanmedian(out["head_back_reproj_px"]) < 2)
check("nothing below the floor; tail lies at or above it",
      (out[[c for c in out.columns if c.endswith("_z")]].to_numpy() >= -1e-6).all()
      and (out["tail_V32_z"] >= 4.9).all())
check("pelvis pitch, lumbar angle and spine length columns",
      {"pelvis_pitch_deg", "lumbar_angle_deg", "spine_length_mm"} <= set(out.columns))

# ------------------------------------------------------------------ 3-D Kalman
T = 3000
tt = np.arange(T) / 30
true = np.stack([np.c_[100 * np.sin(tt * .7 + k), 80 * np.cos(tt * .5 + k), 60 + 30 * np.sin(tt * 1.3 + k)] for k in range(3)], 1)
C = np.array([0, 0, 430.0])
d = true - C
d /= np.linalg.norm(d, axis=2, keepdims=True)
noisy = true + rng.normal(0, 2, true.shape) + d * rng.normal(0, 25, true.shape[:2])[..., None]
noisy[rng.random(true.shape[:2]) < .05] = np.nan
noisy[rng.random(true.shape[:2]) < .005] += 300
S, info = smooth_markers(noisy, Kalman3DConfig(), C)
check("3-D Kalman (RTS): error halved, outliers gated",
      np.nanmedian(np.linalg.norm(S - true, axis=2)) < 0.6 * np.nanmedian(np.linalg.norm(noisy - true, axis=2))
      and info["gated"] > 0)

# ------------------------------------------------------------------ stereo
c0 = Camera.straight_down((320, 240), 600., 1200., (640, 480), center_xy=(-300, 0))
c1 = Camera.straight_down((320, 240), 600., 1200., (640, 480), center_xy=(300, 0))
Pt = np.c_[rng.uniform(-200, 200, 300), rng.uniform(-150, 150, 300), rng.uniform(0, 250, 300)]
Xt = triangulate(c0, c1, c0.project(Pt) + rng.normal(0, .5, (300, 2)), c1.project(Pt) + rng.normal(0, .5, (300, 2)))
check("two cameras: triangulation within 3 mm", np.median(np.linalg.norm(Xt - Pt, axis=1)) < 3)

# ------------------------------------------------------------------ project + tab
from mufasa.project_layout import PROJECT_CONFIG_FILENAME, ProjectPaths, write_project_toml  # noqa: E402

root = tmp / "proj"
root.mkdir()
paths = ProjectPaths(root)
paths.ensure_skeleton()
write_project_toml(root / PROJECT_CONFIG_FILENAME, {"project_layout_version": 1, "project_name": "p3d"})
cfgp = str(root / PROJECT_CONFIG_FILENAME)
stem = "R1_cam_session"
df2d.to_parquet(paths.sources_pose / f"{stem}.parquet")
vw = cv2.VideoWriter(str(paths.sources_videos / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
bg = np.full((360, 640, 3), 90, np.uint8)
cv2.polylines(bg, [cam.project(floor).astype(np.int32)], True, (230, 230, 230), 2)
for _ in range(40):
    vw.write(bg)
vw.release()

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from mufasa.ui_qt.forms.pose3d import (  # noqa: E402
    BoxMarkDialog,
    CameraCalibrationForm,
    Pose3DInferenceForm,
    Pose3DSmoothingForm,
    Pose3DViewerLauncher,
    background_frame,
)

dlg = BoxMarkDialog(background_frame(str(paths.sources_videos / f"{stem}.mp4"), n=5))
s_ = dlg.img.scale
for x, y in marks["floor_corners"]:
    QTest.mouseClick(dlg.img, Qt.LeftButton, Qt.NoModifier, QPoint(int(round(x * s_)), int(round(y * s_))))
dlg.edge_rb.setChecked(True)
for ed in edges[:2]:
    for x, y in ed:
        QTest.mouseClick(dlg.img, Qt.LeftButton, Qt.NoModifier, QPoint(int(round(x * s_)), int(round(y * s_))))
    dlg.next_btn.click()
mk = dlg.marks()
check("box-marking dialog: clicks become 4 floor corners and 2 wall-corner edges",
      len(mk["floor_corners"]) == 4 and len(mk["vertical_edges"]) == 2 and dlg.ok_btn.isEnabled())

F = CameraCalibrationForm(config_path=cfgp)
F._marks = marks
F.name.setText("R1")
F.length.setValue(L_)
F.width.setValue(W_)
F.height.setValue(430)
F.target(**F.collect_args())
camfile = paths.sources_dir / "calibration" / "camera_R1.toml"
check("calibration form: camera_R1.toml and the marks saved, report shown",
      camfile.is_file() and (paths.sources_dir / "calibration" / "R1.marks.json").is_file()
      and "px/mm" in F._report_text)

G = Pose3DInferenceForm(config_path=cfgp)
G.refresh()
check("inference form: wall tolerance (default 20 mm) reaches the settings",
      G.wall_tol.value() == 20 and G.collect_args()["settings"].mono.wall_tolerance_mm == 20)
check("inference form: the video is listed with its camera",
      G.videos.count() == 1 and G.videos.item(0).checkState() == Qt.Checked)
G.kal.setChecked(True)
args = G.collect_args()
args["settings"].mono.chunk = 900
args["settings"].mono.head_template_z = SYN_Z
G.target(**args)
runs = paths.list_runs("pose3d")
outf = runs[-1] / f"{stem}_3d.parquet" if runs else None
check("inference form: derived/pose3d/<run>/<video>_3d.parquet with run.toml, run id recorded",
      outf is not None and outf.is_file() and (runs[-1] / "run.toml").is_file() and G._last_run_id == runs[-1].name)
from mufasa.ui_qt.pose3d_viewer import find_pose3d, load_pose3d  # noqa: E402

P3 = load_pose3d(outf)
check("the viewer reads it: 15 markers, box, edges, camera (P)",
      len(P3.names) == 15 and P3.box is not None and len(P3.edges) >= 15 and P3.projection_matrix() is not None)
check("feature viewer's 3-D… finds it for the video",
      find_pose3d(str(paths.sources_videos / f"{stem}.mp4"), cfgp) == outf)
H = Pose3DSmoothingForm(config_path=cfgp)
H.target(**H.collect_args())
runs2 = paths.list_runs("pose3d")
check("smoothing form: a new run with the smoothed file", len(runs2) == 2 and (runs2[-1] / f"{stem}_3d.parquet").is_file())
Lw = Pose3DViewerLauncher(config_path=cfgp)
Lw.refresh()
v = Lw.open()
check("viewer launcher lists the 3-D files and opens the viewer", Lw.file.count() == 2 and v is not None)
if hasattr(v, "cleanup"):
    v.cleanup()

# ------------------------------------------------------------------ CLI / entry point / tab
from mufasa.tools import pose3d_cli  # noqa: E402

(tmp / "m.json").write_text(json.dumps(marks))
rc = pose3d_cli.main(["calibrate-box", str(tmp / "m.json"), "--width", "300", "--camera-height", "430",
                      "--out", str(tmp / "cli_cam.toml")])
check("CLI calibrate-box", rc == 0 and Camera.load(tmp / "cli_cam.toml").height == 430)
rc = pose3d_cli.main(["smooth", str(outf), "--out", str(tmp / "s_3d.parquet"), "--camera", str(camfile)])
check("CLI smooth", rc == 0 and (tmp / "s_3d.parquet").is_file())
check("entry point mufasa-pose3d", 'mufasa-pose3d    = "mufasa.tools.pose3d_cli:main"' in (REPO / "pyproject.toml").read_text())
from mufasa.ui_qt.workbench_app import build_workbench  # noqa: E402

wb = build_workbench(None)
titles = [wb.sidebar.item(i).text() for i in range(wb.sidebar.count())]
check("workbench: 3-D tab right after Preprocessing", titles.index("3-D") == titles.index("Preprocessing") + 1)

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122kf_pose3d: {len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
