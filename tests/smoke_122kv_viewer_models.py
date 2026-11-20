"""Smoke test for patch 122kv — models and view options in the 3-D viewer.

User: "Can you add a model loader and view options to the 3d renderer?"

* meshes load from OBJ (quads fanned), STL (ASCII and binary; shared vertices merged) and PLY
  (ASCII and binary little-endian); large meshes are simplified; a mesh is placed with its own
  scale, turn and offset;
* a body-model rig (the virtual rodent) is posed to the 3-D markers: its size from the data,
  heading, position and posture recovered from markers made by the rig itself; frames are
  cached and later frames start from their neighbours; too few shared markers is an error;
* models round-trip as plain values (saved camera animations render with them);
* the volume draws models in every style (solid, transparent, wireframe, points); without
  ``sync_fit`` a rig is posed shortly after the frame is shown; view options: markers, names,
  skeleton, grid, axes, orthographic projection, field of view, background;
* the viewer's *View & models* panel (the built-in virtual rodent, style, opacity, mesh
  placement, fit all frames), a rendered animation with a model, and ``mufasa-view3d --model``.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


CUBE_V = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
CUBE_Q = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]


def write_meshes(d: Path) -> dict:
    tris = [(q[0], q[k], q[k + 1]) for q in CUBE_Q for k in (1, 2)]
    out = {}
    p = d / "cube.obj"
    p.write_text("# cube\n" + "".join(f"v {x} {y} {z}\n" for x, y, z in CUBE_V)
                 + "".join("f " + " ".join(f"{i + 1}/1/1" for i in q) + "\n" for q in CUBE_Q))
    out["obj"] = p
    p = d / "cube_ascii.stl"
    lines = ["solid c"]
    for t in tris:
        lines += ["facet normal 0 0 0", "outer loop"] + [f"vertex {x} {y} {z}" for x, y, z in CUBE_V[list(t)]]
        lines += ["endloop", "endfacet"]
    p.write_text("\n".join(lines + ["endsolid c"]))
    out["stl ascii"] = p
    p = d / "cube_bin.stl"
    b = bytearray(80) + struct.pack("<I", len(tris))
    for t in tris:
        b += struct.pack("<12fH", 0, 0, 0, *CUBE_V[list(t)].ravel(), 0)
    p.write_bytes(bytes(b))
    out["stl binary"] = p
    p = d / "cube_ascii.ply"
    p.write_text("ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\n"
                 "element face 6\nproperty list uchar int vertex_indices\nend_header\n"
                 + "".join(f"{x} {y} {z}\n" for x, y, z in CUBE_V)
                 + "".join("4 " + " ".join(map(str, q)) + "\n" for q in CUBE_Q))
    out["ply ascii"] = p
    p = d / "cube_bin.ply"
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex 8\nproperty float x\nproperty float y\n"
            "property float z\nelement face 6\nproperty list uchar int vertex_indices\nend_header\n").encode()
    body = b"".join(struct.pack("<3f", *v) for v in CUBE_V) + b"".join(struct.pack("<B4i", 4, *q) for q in CUBE_Q)
    p.write_bytes(head + body)
    out["ply binary"] = p
    return out


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")          # the built-in rig is made here
    import pandas as pd

    from mufasa.bodymodel import fit as BF
    from mufasa.pose3d import scene_models as SM

    # ---- meshes
    files = write_meshes(tmp)
    res = {k: SM.load_mesh(p) for k, p in files.items()}
    okm = all(len(V) == 8 and len(F) == 12 and F.max() == 7
              and np.allclose(np.sort(V, axis=0), np.sort(CUBE_V, axis=0)) for V, F in res.values())
    check(f"meshes: OBJ, STL (ASCII, binary), PLY (ASCII, binary) -> 8 vertices, 12 triangles ({', '.join(res)})", okm)
    th, ph = np.meshgrid(np.linspace(0, math.pi, 200), np.linspace(0, 2 * math.pi, 200), indexing="ij")
    Vs = np.c_[np.sin(th).ravel() * np.cos(ph).ravel(), np.sin(th).ravel() * np.sin(ph).ravel(), np.cos(th).ravel()]
    idx = np.arange(200 * 200).reshape(200, 200)
    Fs = np.r_[np.c_[idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()],
               np.c_[idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]]
    Vd, Fd = SM.simplify(Vs, Fs, 5000)
    check(f"large meshes simplified ({len(Fs):,} -> {len(Fd):,} faces)", len(Fd) <= 5000 and len(Fd) > 500)
    m = SM.SceneMesh.load(files["obj"], scale=10.0, rot_z_deg=90.0, offset=(100.0, 0.0, 5.0))
    W = m.world()
    check("mesh placement: scale, turn about the vertical, offset",
          np.allclose(W[1], [100.0, 10.0, 5.0]) and np.allclose(W[0], [100.0, 0.0, 5.0]))
    try:
        SM.load_model(tmp / "x.fbx", None)
        err = ""
    except ValueError as e:
        err = str(e)
    check("unknown model files: a clear error", "not a model" in err)

    # ---- a rig posed to 3-D markers made by the rig itself
    from mufasa.ui_qt.pose3d_viewer import load_pose3d
    rp = SM.builtin_rig_path()
    rig = BF.Rig(rp)
    mk = BF.Markers.build(rig, None)
    SIZE = 1.3
    rng = np.random.default_rng(2)
    truth, rows = [], []
    for k in range(12):
        p = BF.neutral(lum_pitch=3 + 2 * math.sin(k / 3), head_yaw=12 * math.sin(k / 4), lum_yaw=3)
        phi, t = 0.6 + 0.05 * k, np.array([20.0 * k, -40.0, 0.0])
        pos = mk.positions(rig, rig.skinning_matrices(p))
        P = SM._yaw(SIZE * np.array(list(pos.values())), phi) + t
        truth.append((p, phi, t))
        row = {"frame": 100 + k}
        for n, X in zip(pos, P + rng.normal(0, 1.0, P.shape)):
            row.update({f"{n}_x": X[0], f"{n}_y": X[1], f"{n}_z": X[2]})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(tmp / "syn_3d.parquet", index=False)
    pose = load_pose3d(tmp / "syn_3d.parquet")
    pr = SM.load_model("virtual_rodent", pose)
    check(f"rig: size from the markers {pr.size:.3f} (true {SIZE}); {len(pr.names)} shared markers",
          abs(pr.size - SIZE) < 0.04 * SIZE and len(pr.names) == 15 and pr.name == "virtual rodent")
    pr.size = SIZE
    fits = [pr.fit_frame(100 + k) for k in range(12)]
    rms = [f[1] for f in fits]
    verr = []
    for k, (p, phi, t) in enumerate(truth):
        Vt = SM._yaw(SIZE * BF.Rig.lbs(rig.skinning_matrices(p), rig.verts(p, True), rig.Wd), phi) + t
        verr.append(np.median(np.linalg.norm(pr.world(100 + k) - Vt, axis=1)))
    dphi = [abs((f[0][0] - tr[1] + math.pi) % (2 * math.pi) - math.pi) for f, tr in zip(fits, truth)]
    check(f"rig posed to the markers: marker error {np.median(rms):.1f} mm (noise 1 mm), skin {np.median(verr):.1f} mm "
          f"from the truth, heading {math.degrees(max(dphi)):.1f}°",
          np.median(rms) < 3 and np.median(verr) < 5 and math.degrees(max(dphi)) < 5)
    check("fits are cached; the nearest fitted frame is found",
          pr.cached(105) and not pr.cached(99) and pr.nearest_cached(130, within=20) == 111)
    few = pd.DataFrame([{"frame": 0, "a_x": 0, "a_y": 0, "a_z": 0, "head_nose_x": 1, "head_nose_y": 1,
                         "head_nose_z": 1}])
    few.to_parquet(tmp / "few_3d.parquet", index=False)
    try:
        SM.load_model("virtual_rodent", load_pose3d(tmp / "few_3d.parquet"))
        err = ""
    except ValueError as e:
        err = str(e)
    check("rig with fewer than three shared markers: a clear error", "at least three" in err)
    spec = pr.spec()
    back = SM.model_from_spec(spec, pose)
    ms = SM.model_from_spec(m.spec(), pose)
    check("models round-trip as plain values (rig: built-in, size, look; mesh: placement)",
          back.kind == "rig" and back.builtin and abs(back.size - SIZE) < 1e-9 and ms.kind == "mesh"
          and np.allclose(ms.world(), W))

    # ---- the volume
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from mufasa.pose3d.animation import _qimage_bgr
    from mufasa.ui_qt.pose3d_viewer import Pose3DViewerWidget, Volume3DWidget
    vol = Volume3DWidget(pose)
    vol.setAttribute(Qt.WA_DontShowOnScreen, True)
    vol.resize(480, 360)
    vol.set_frame(104)
    vol.target = np.nanmean(pose.xyz[pose.row(104)], 0)
    vol.dist = 500

    def shot():
        img = QImage(480, 360, QImage.Format_RGB32)
        vol.render(img)
        return _qimage_bgr(img).astype(int)
    plain = shot()
    vol.add_model(pr)
    vol.sync_fit = True
    looks = {}
    for st in SM.STYLES:
        pr.look.style = st
        looks[st] = shot()
    diff = {st: int((np.abs(a - plain).sum(2) > 30).sum()) for st, a in looks.items()}
    check(f"the model drawn in every style (pixels changed: {diff})",
          all(v > 300 for v in diff.values()) and diff["solid"] > diff["points"])
    pr.look.visible = False
    check("a hidden model is not drawn", int((np.abs(shot() - plain).sum(2) > 30).sum()) == 0)
    pr.look.visible = True
    pr.look.style = "solid"
    lazy = SM.load_model("virtual_rodent", pose)
    vol.models = [lazy]
    vol.sync_fit = False
    shot()
    pending = not lazy.cached(104) and vol._fit_timer.isActive()
    vol._fit_now()
    check("without sync_fit, a rig is posed shortly after the frame is shown",
          pending and lazy.cached(104))
    vol.models = [pr]
    vol.sync_fit = True
    base = shot()
    vol.show_markers = False
    no_markers = shot()
    vol.show_markers = True
    vol.show_grid = False
    no_grid = shot()
    vol.show_grid = True
    vol.background = "black"
    black = shot()
    vol.background = "theme"
    vol.show_axes = True
    axes = shot()
    vol.show_axes = False
    vol.show_names = True
    names = shot()
    vol.show_names = False
    vol.ortho = True
    (c,), zc = vol.project(vol.target[None])
    ortho = shot()
    vol.ortho = False
    changed = lambda a: int((np.abs(a - base).sum(2) > 30).sum())  # noqa: E731
    check(f"view options change the picture (markers {changed(no_markers)}, grid {changed(no_grid)}, axes "
          f"{changed(axes)}, names {changed(names)}, orthographic {changed(ortho)} px); black background",
          min(changed(no_markers), changed(no_grid), changed(axes), changed(names), changed(ortho)) > 30
          and (black[2, 2] < 10).all())
    check("orthographic: the target in the middle, in front",
          np.allclose(c, [240, 180], atol=1e-6) and zc[0] > 0)

    # ---- the viewer and its panel
    w = Pose3DViewerWidget(None, tmp / "syn_3d.parquet", models=[str(files["obj"])])
    pan = w.models_panel
    check("viewer: 'View & models' opens the panel; models given at start are loaded",
          w.models_cb.isChecked() and pan.isVisibleTo(w) and len(w.volume.models) == 1
          and w.volume.models[0].kind == "mesh")
    pan.rodent_btn.click()
    r2 = pan.selected()
    pan.style.setCurrentIndex(pan.style.findData("wireframe"))
    pan.opacity.setValue(40)
    check("panel: the built-in virtual rodent added and selected; style and opacity reach it; placement only "
          "for meshes", r2.kind == "rig" and r2.look.style == "wireframe" and abs(r2.look.opacity - 0.4) < 1e-9
          and not pan.scale.isEnabled() and pan.fitall_btn.isEnabled())
    pan.list.setCurrentRow(0)
    pan.scale.setValue(25.0)
    pan.offs[2].setValue(7.0)
    mesh = w.volume.models[0]
    check("panel: mesh scale and offset", mesh.scale == 25.0 and mesh.offset[2] == 7.0)
    pan.list.setCurrentRow(1)
    import time

    def wait_fit():                                         # 122ky: rigs are posed in the background
        t_end = time.time() + 300
        while pan._fit_thread is not None and time.time() < t_end:
            QApplication.processEvents()
            time.sleep(0.02)
        QApplication.processEvents()
    wait_fit()                                              # (adding the rig started it)
    pan.fit_all()
    wait_fit()
    check("panel: fit all frames poses the rig to every frame",
          all(r2.cached(int(f)) for f in w.volume.pose.frames) and "marker error" in pan.info.text())
    pan.list.item(1).setCheckState(Qt.Unchecked)
    check("panel: unticking a model hides it", not r2.look.visible)
    pan.size_spin.setValue(150)
    pan.proj.setCurrentIndex(1)
    pan.bg.setCurrentIndex(pan.bg.findData("white"))
    check("panel: marker size, projection and background reach the volume",
          w.volume.marker_scale == 1.5 and w.volume.ortho and w.volume.background == "white")
    st = w.animation.settings()
    check("saved camera animations carry the view options and the models",
          st["style"]["ortho"] is True and len(st["models"]) == 2 and st["models"][1]["kind"] == "rig")
    w.cleanup()

    # ---- a rendered animation with a model
    import cv2

    from mufasa.pose3d.animation import CameraKey, CameraPath, render_animation
    vs = {"az": -1.0, "el": 0.5, "dist": 600.0, "target": list(np.nanmean(pose.xyz[0], 0)), "fov": 0.7}
    cp = CameraPath([CameraKey(100, vs), CameraKey(103, vs)])
    a0 = render_animation(pose, cp, tmp / "a0.avi", size=(320, 240), fps=10)
    a1 = render_animation(pose, cp, tmp / "a1.avi", size=(320, 240), fps=10, models=[spec])

    def first(pth):
        cap = cv2.VideoCapture(str(pth))
        ok, fr = cap.read()
        cap.release()
        return fr.astype(int) if ok else None
    f0, f1 = first(a0), first(a1)
    check("a rendered animation draws the model (from its spec)",
          f0 is not None and f1 is not None and int((np.abs(f1 - f0).sum(2) > 30).sum()) > 300)

    from mufasa.tools import view3d
    a = view3d.build_parser().parse_args(["v.mp4", "p.parquet", "--model", "virtual_rodent", "--model", "a.obj"])
    check("mufasa-view3d --model (repeatable)", a.model == ["virtual_rodent", "a.obj"])
    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs: models and view options in the 3-D viewer", "View & models" in doc and "--model" in doc)
    del app

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kv_viewer_models: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
