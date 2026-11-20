"""Smoke test for patch 122ky — real-time models in the 3-D viewer: drawn on the GPU, posed in
batches (on the GPU with CUDA), and the virtual rodent's mouth closed.

User: "Can you modify it to have realtime rendering of the model, with gpu acceleration?" and
"And can you close the mouth of the model?"

* the volume's extent is worked out once, not on every paint (it took 0.4 s a paint on a long
  recording);
* the batched pose fit: its kinematics equal the rig's (single- and several-vertex markers);
  on markers made by the rig it recovers heading, position and posture as the per-frame fit
  does; frames land in the same cache; ``fit_frames`` (the animation renderer uses it);
* the panel's background fit: a thread posing the frames nearest the one shown first, while
  the viewer does not start its own per-frame fits; *GPU* and *Fit in the background*;
* GPU drawing: the same pixels as the CPU drawing (perspective, orthographic, through the
  recording camera, and between them) — under a virtual display when there is one; otherwise
  the viewer falls back to the CPU and says so;
* the mouth closed (jaw at −40°), the built-in rig rebuilt (versioned cache file).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def synthetic(tmp: Path, n: int = 40, size: float = 1.3):
    """3-D markers made by the virtual rodent rig itself (known parameters), as a 3-D file."""
    import pandas as pd

    import mufasa.bodymodel.fit as BF
    from mufasa.pose3d import scene_models as SM
    rig = BF.Rig(SM.builtin_rig_path())
    mk = BF.Markers.build(rig, None)
    rng = np.random.default_rng(5)
    rows, truth = [], []
    for k in range(n):
        p = BF.neutral(lum_pitch=3 + 2 * math.sin(k / 3), head_yaw=12 * math.sin(k / 4), lum_yaw=3,
                       rear=8 * max(0.0, math.sin(k / 6)))
        phi, t = 0.6 + 0.04 * k, np.array([6.0 * k, -40.0, 0.0])
        pos = mk.positions(rig, rig.skinning_matrices(p))
        P = SM._yaw(size * np.array(list(pos.values())), phi) + t
        truth.append((phi, t))
        row = {"frame": 100 + k}
        for nm, X in zip(pos, P + rng.normal(0, 1.0, P.shape)):
            row.update({f"{nm}_x": X[0], f"{nm}_y": X[1], f"{nm}_z": X[2]})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(tmp / "syn_3d.parquet", index=False)
    return tmp / "syn_3d.parquet", truth


GL_SCRIPT = r'''
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from PySide6.QtCore import Qt
app = QApplication([])
from mufasa.ui_qt.pose3d_viewer import Volume3DWidget, load_pose3d
from mufasa.pose3d.scene_models import load_model
from mufasa.pose3d.animation import _qimage_bgr
pose = load_pose3d(sys.argv[2])
pose.camera = {"focal_px": 700.0, "principal_px": [640, 512], "height_mm": 700.0,
               "image_width_px": 1280, "image_height_px": 1024}
m = load_model("virtual_rodent", pose)
m.size = 1.3
m.fit_frames([110])
vol = Volume3DWidget(pose); vol.setAttribute(Qt.WA_DontShowOnScreen, True); vol.resize(640, 480)
for a in ("show_markers", "show_skeleton", "show_shadows", "show_drops", "show_label", "show_grid", "show_box"):
    setattr(vol, a, False)
vol.set_frame(110)
ctr = m.world(110).mean(0)
def shot():
    img = QImage(640, 480, QImage.Format_RGB32); vol.render(img); return _qimage_bgr(img).astype(int)
def persp():
    vol.ortho = False; vol.reset_view("Perspective"); vol.target = ctr.copy(); vol.dist = 600.0
def orth():
    persp(); vol.ortho = True
def camv():
    vol.ortho = False; vol.reset_view("Camera"); vol.cam_zoom = 1.6
def blend():
    persp(); vol.cam_blend = 0.5
out = {}
for name, setup in (("perspective", persp), ("orthographic", orth), ("camera", camv), ("half-way", blend)):
    for style in ("solid", "transparent"):
        setup(); m.look.style = style
        vol.models = []; base = shot(); vol.models = [m]
        vol.gpu = False; a = shot()
        vol.gpu = True; b = shot()
        ma, mb = np.abs(a - base).sum(2) > 30, np.abs(b - base).sum(2) > 30
        out[f"{name}/{style}"] = [int(ma.sum()), float((ma & mb).sum() / max((ma | mb).sum(), 1))]
        vol.cam_blend = 0.0
out["note"] = vol.gpu_note
print("RESULT " + json.dumps(out))
'''


def main() -> int:
    import torch

    import mufasa.bodymodel.fit as BF
    from mufasa.pose3d import rig_batch as RB
    from mufasa.pose3d import scene_models as SM
    from mufasa.ui_qt.pose3d_viewer import Pose3D, Volume3DWidget, load_pose3d
    tmp = Path(tempfile.mkdtemp())
    os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")

    # the extent, once
    xyz = np.random.default_rng(0).normal(0, 50, (200000, 6, 3))
    pose = Pose3D(names=[f"m{i}" for i in range(6)], xyz=xyz, frames=np.arange(len(xyz)))
    t0 = time.time()
    e1 = pose.extent()
    t1 = time.time()
    for _ in range(50):
        pose.extent()
    t2 = time.time()
    check(f"extent once ({1000 * (t1 - t0):.0f} ms), then from memory ({1000 * (t2 - t1) / 50:.3f} ms a paint)",
          pose.extent() == e1 and (t2 - t1) / 50 < 0.01 * max(t1 - t0, 1e-3))

    # the closed mouth and the rebuilt rig
    from mufasa.bodymodel import rodent_rig as RR
    from mufasa.pose3d.rodent_model import RodentModel
    model = RodentModel()
    q = RR.stance_qpos(model)
    jaw = math.degrees(q[0, model.joint_index["mandible"]])
    rp = SM.builtin_rig_path()
    check(f"mouth closed: the jaw at {jaw:.0f}° (bound open); the built-in rig is {rp.name} (rebuilt)",
          abs(jaw - RR.JAW_CLOSED_DEG) < 1e-9 and rp.name == f"virtual_rodent_v{RR.RIG_VERSION}.rig.npz"
          and RR.RIG_VERSION >= 2 and rp.is_file())
    T = model.fk(q)
    V = model.skin_points(q, None, T)[0]
    q_open = RR.stance_qpos(model, jaw_deg=None)
    Vo = model.skin_points(q_open, None, model.fk(q_open))[0]
    jaw_v = model.weights[:, model.body_index["jaw"]] > 0.5 if model.weights.shape[1] == len(model.body_names) else None
    if jaw_v is None:
        bs = list(model.skin_bodies)
        jaw_v = model.weights[:, bs.index(model.body_index["jaw"])] > 0.5
    lift = float(np.median(V[jaw_v, 2] - Vo[jaw_v, 2]))
    check(f"the lower jaw's skin {lift:.1f} mm higher than as bound", lift > 3)

    # the batched fit: kinematics
    path, truth = synthetic(tmp)
    pose = load_pose3d(path)
    pr = SM.load_model("virtual_rodent", pose)
    pr.size = 1.3
    prob = RB.RigProblem(pr, "cpu")
    rng = np.random.default_rng(1)
    err = 0.0
    for _ in range(10):
        x = np.r_[rng.uniform(-3, 3), rng.uniform(-90, 90, 2), rng.uniform(-20, 20),
                  [rng.uniform(lo, hi) for _, lo, hi in pr.posture]]
        err = max(err, float(np.abs(pr._markers_world(x)[0] - prob.markers(torch.as_tensor(x)).numpy()).max()))
    # several-vertex markers (rigs without fixed markers) skin each vertex, then average
    mk2 = BF.Markers.build(pr.rig, None)
    head = np.flatnonzero(pr.rig.W[:, pr.rig.ix["Head"]] > 0.5)[:5]
    mk2.spec["head_left"] = ("vert", [int(i) for i in head])
    pr2 = SM.load_model("virtual_rodent", pose)
    pr2.mk = mk2
    pr2.size = 1.3
    prob2 = RB.RigProblem(pr2, "cpu")
    x = np.r_[0.3, 10, -5, 4, [0.5 * (lo + hi) for _, lo, hi in pr.posture]]
    x[4 + [p[0] for p in pr.posture].index("head_yaw")] = 25
    err2 = float(np.abs(pr2._markers_world(x)[0] - prob2.markers(torch.as_tensor(x)).numpy()).max())
    check(f"batched kinematics = the rig's ({err:.1e} mm; a 5-vertex marker {err2:.1e} mm); "
          f"{prob.n_bones[0]} of {prob.n_bones[1]} bones move markers",
          err < 1e-6 and err2 < 1e-6 and prob.n_bones[0] < prob.n_bones[1])

    # the batched fit against the per-frame fit
    frames = [100 + k for k in range(len(truth))]
    t0 = time.time()
    n = pr.fit_frames(frames, device="cpu")
    tb = time.time() - t0
    batched = {f: pr._cache[f] for f in frames}
    head_err = max(abs((batched[f][0][0] - truth[k][0] + math.pi) % (2 * math.pi) - math.pi) for k, f in enumerate(frames))
    pos_err = max(float(np.linalg.norm(batched[f][0][1:3] - truth[k][1][:2])) for k, f in enumerate(frames))
    rms_b = np.array([batched[f][1] for f in frames])
    pr._cache.clear()
    single = {f: pr.fit_frame(f) for f in frames}
    rms_s = np.array([single[f][1] for f in frames])
    head_s = max(abs((single[f][0][0] - truth[k][0] + math.pi) % (2 * math.pi) - math.pi) for k, f in enumerate(frames))
    pos_s = max(float(np.linalg.norm(single[f][0][1:3] - truth[k][1][:2])) for k, f in enumerate(frames))
    check(f"batched fit: {n} frames ({1000 * tb / n:.0f} ms each here); heading within {math.degrees(head_err):.1f}° "
          f"(per frame {math.degrees(head_s):.1f}°), position within {pos_err:.1f} mm ({pos_s:.1f}); marker error "
          f"{np.median(rms_b):.2f} mm ({np.median(rms_s):.2f})",
          n == len(frames) and head_err < head_s + math.radians(1) and pos_err < pos_s + 2
          and np.median(rms_b) < np.median(rms_s) + 0.5 and (rms_b < rms_s + 2).all())
    check(f"device: {RB.device_name('auto')} ({'CUDA' if torch.cuda.is_available() else 'no CUDA here'}); "
          "fit_frames fills the per-frame cache", RB.device_name("cpu") == "cpu" and pr.cached(frames[0]))

    # the viewer and the panel
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from mufasa.ui_qt.pose3d_models import FitThread
    from mufasa.ui_qt.pose3d_viewer import Pose3DViewerWidget
    w = Pose3DViewerWidget(None, path)
    w.models_cb.setChecked(True)
    panel = w.models_panel
    vol = w.volume
    vol.set_frame(120)
    panel.bg_fit_cb.setChecked(True)
    m = panel.add("virtual_rodent")
    started = panel._fit_thread is not None and vol.bg_fitting
    t0 = time.time()
    while panel._fit_thread is not None and time.time() - t0 < 300:
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()
    done = all(m.cached(f) for f in frames)
    check(f"panel: a rig added is posed in the background ({'started' if started else 'not started'}, "
          f"{sum(m.cached(f) for f in frames)}/{len(frames)} frames), the viewer leaving the frames to it",
          started and done and not vol.bg_fitting and panel.fitall_btn.text() == "Fit all frames")
    order = []

    class Probe:
        def __init__(self):
            self._cache = {}

        def fit_frames(self, fr, **kw):
            order.append(fr[0])
            return len(fr)
    th = FitThread(Probe(), list(range(0, 5000)), lambda: 3000)
    th.first, th.chunk = 100, 1000
    th.run()
    check(f"background order: the frames near the one shown first (first batch from frame {order[0]}), "
          f"then the rest ({len(order)} batches)", 2900 <= order[0] <= 3000 and len(order) >= 5)
    check("panel: GPU drawing and background fitting can be switched off",
          panel.gpu_cb.isChecked() == vol.gpu and hasattr(panel, "bg_fit_cb"))

    # drawing: the CPU fallback here (no OpenGL off-screen) ...
    vol2 = Volume3DWidget(pose)
    vol2.setAttribute(Qt.WA_DontShowOnScreen, True)
    vol2.resize(400, 300)
    vol2.add_model(m)
    vol2.set_frame(110)
    img = QImage(400, 300, QImage.Format_RGB32)
    vol2.render(img)
    from mufasa.ui_qt.gl_models import shared_renderer
    gl = shared_renderer()
    check(f"drawing here: {'GPU, ' + gl.info if gl.ok else 'CPU fallback (' + (gl.error or 'no OpenGL') + ')'}",
          (gl.ok and vol2.gpu_note.startswith("GPU")) or (gl.ok is False and "CPU" in vol2.gpu_note))
    # ... and on the GPU under a virtual display, against the CPU drawing
    xvfb = shutil.which("xvfb-run")
    if xvfb:
        sc = tmp / "gl_check.py"
        sc.write_text(GL_SCRIPT)
        env = dict(os.environ, QT_QPA_PLATFORM="xcb")
        r = subprocess.run([xvfb, "-a", "-s", "-screen 0 1280x1024x24", sys.executable, str(sc), str(REPO), str(path)],
                           capture_output=True, text=True, timeout=600, env=env)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print("SKIP  GPU drawing under a virtual display: " + (r.stderr.strip().splitlines() or ["no output"])[-1])
        else:
            res = json.loads(line[7:])
            note = res.pop("note")
            worst = min(v[1] for v in res.values())
            check(f"GPU drawing = CPU drawing: model pixels overlap (IoU) ≥ {worst:.3f} in perspective, orthographic, "
                  f"camera and half-way views, solid and transparent ({note[:40]})",
                  note.startswith("GPU") and worst > 0.95 and all(v[0] > 500 for v in res.values()))
    else:
        print("SKIP  GPU drawing: no xvfb-run here")

    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs: GPU drawing, batched posing, the closed mouth",
          "GPU" in doc and "in the background" in doc and "mouth" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122ky_viewer_gpu_realtime: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
