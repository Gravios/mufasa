"""Smoke test for patch 122kp — camera animation in the 3-D viewer.

User: "Can you also make an animation editor for the 3d viewer which allows
me to set key points between which I can assign smooth transforms to the
camera rotation and zoom? After which I would like to play through the and
collate the frames into a video."

* CameraPath: a key's own view at the key; the azimuth goes the shorter way
  round; the zoom (distance) is interpolated geometrically; ease in-out
  starts gently; a spline passes through the keys and does not stop at a
  middle key; the data frame advances linearly (real time, slow motion,
  freeze); camera ↔ camera moves stay in the camera view; a move out of
  the camera view starts exactly at it, with no jump in the projected
  markers;
* render_animation: frame count, size, the video below, each output frame
  equals the view drawn directly at that state, the camera moves;
* the editor: add, update, delete and go to keys, real-time lengths, the
  timeline slider drives the viewer, preview plays, save/load round trip,
  render from the panel; the CLI renders a saved path.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def view(az, el=30.0, dist=600.0, target=(0, 0, 50), cam=False, zoom=1.0):
    return {"az": math.radians(az), "el": math.radians(el), "dist": dist, "target": list(target),
            "fov": math.radians(40), "cam_mode": cam, "cam_zoom": zoom, "cam_pan": [0.0, 0.0]}


def main() -> int:
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smoke_122kn_video_in_3d import make_camera, write_pose, write_video

    from mufasa.ui_qt.pose3d_animation import CameraKey, CameraPath, _qimage_bgr, render_animation
    from mufasa.ui_qt.pose3d_viewer import Pose3DViewerWidget, Volume3DWidget, load_pose3d

    # ---- the path ------------------------------------------------------- #
    p = CameraPath([CameraKey(0, view(350, dist=200), 10, "Linear"),
                    CameraKey(10, view(10, dist=800), 20, "Ease in-out"),
                    CameraKey(10, view(90), 30, "Spline"),
                    CameraKey(40, view(180, el=60), 30)])
    check("output frames = sum of lengths + 1", p.n_output() == 10 + 20 + 30 + 1)
    check("at each key: its own view and frame",
          all(abs(p.state_at(s)["az"] % (2 * math.pi) - k.view["az"]) < 1e-9 and p.frame_at(s) == k.frame
              for s, k in zip(p.key_starts(), p.keys)))
    mid = p.state_at(5)
    check(f"azimuth the shorter way round (350° → 10° via 0°: {math.degrees(mid['az']) % 360:.1f}°)",
          abs(math.degrees(mid["az"]) % 360) < 1e-6 or abs(math.degrees(mid["az"]) % 360 - 360) < 1e-6)
    check(f"zoom geometric (200 → 800 mm halfway: {mid['dist']:.1f})", abs(mid["dist"] - 400) < 1e-6)
    check("linear: data frame advances one per output frame (real time)",
          [p.frame_at(n) for n in range(11)] == list(range(11)))
    check("same frame at both keys: the data is frozen while the camera moves",
          {p.frame_at(n) for n in range(10, 31)} == {10})
    check("after the freeze the data runs on (frame 10 → 40 over 30 output frames)", p.frame_at(45) == 25)
    e1 = (p.state_at(11)["az"] - p.keys[1].view["az"] + math.pi) % (2 * math.pi) - math.pi
    e10 = p.state_at(20)["az"] - p.state_at(19)["az"]
    check(f"ease in-out starts gently (first step {math.degrees(e1):.3f}° vs {math.degrees(e10):.2f}° mid)",
          abs(e1) < abs(e10) / 20)
    sp = [p.state_at(n)["az"] for n in range(28, 63)]
    d = (np.diff(sp) + math.pi) % (2 * math.pi) - math.pi
    i_key = 60 - 28                                      # key 3 at output 60
    check(f"spline: no stop at the middle key (step there {math.degrees(d[i_key - 1]):.2f}°)",
          abs(d[i_key - 1]) > 0.3 * np.abs(d).max())
    check("JSON round trip", (lambda q: [k.to_dict() for k in q[0].keys] == [k.to_dict() for k in p.keys]
                              and q[1]["fps"] == 25)(CameraPath.from_json(p.to_json(fps=25))))
    pc = CameraPath([CameraKey(0, view(0, cam=True, zoom=1.0), 10), CameraKey(5, view(0, cam=True, zoom=4.0), 10),
                     CameraKey(5, view(90), 10)])
    s5 = pc.state_at(5)
    check(f"camera → camera: stays in the camera view, zoom ×{s5['cam_zoom']:.2f} halfway",
          s5["cam_mode"] and abs(s5["cam_zoom"] - 2.0) < 1e-6)
    s15 = pc.state_at(15)
    check("camera → orbit: not the camera view, blended", not s15["cam_mode"] and 0 < s15["cam_blend"] < 1)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cam = make_camera()
        pose_path = write_pose(cam, td / "s_3d.parquet", n=60)
        write_video(cam, td / "s.avi", n=60, distort=False)
        v = Pose3DViewerWidget(str(td / "s.avi"), pose_path)
        v.resize(900, 800)
        v.show()
        QApplication.processEvents()
        vol, a = v.volume, v.animation
        check("viewer: 'Animation' shows the editor", not a.isVisible()
              and (v.anim_cb.setChecked(True) or a.isVisible()))

        # leaving the camera view: no jump in the projected markers
        vol.reset_view("Camera")
        cam_key = CameraKey(0, vol.view_state(), 20, "Linear")       # uniform steps: a jump would show
        vol.reset_view("Perspective")
        path_c = CameraPath([cam_key, CameraKey(0, vol.view_state(), 20)])
        pts = np.array([[40.0, -30, 0], [-60, 50, 90]])
        proj = []
        for n in range(4):
            vol.set_view_state(path_c.state_at(n))
            proj.append(vol.project(pts)[0])
        jump0 = np.abs(proj[1] - proj[0]).max()
        step = np.abs(proj[3] - proj[2]).max()
        vol.set_view_state({**path_c.state_at(1), "cam_blend": 0.0})      # what it would be without blending
        unblended = np.abs(vol.project(pts)[0] - proj[0]).max()
        check(f"out of the camera view: no jump ({jump0:.1f} px first step, {step:.1f} px later; "
              f"{unblended:.0f} px without blending)",
              jump0 < 1.5 * step and unblended > 10 * step and path_c.state_at(0)["cam_mode"])
        vol.set_view_state(path_c.state_at(0))
        check("view_state / set_view_state round trip",
              all(np.allclose(vol.view_state()[k], cam_key.view[k]) for k in cam_key.view))

        # ---- the editor ------------------------------------------------- #
        v.scrubber.seek(5)
        vol.reset_view("Perspective")
        a.add_key()
        v.scrubber.seek(35)
        vol.az += math.radians(120)
        vol.dist *= 0.5
        a.add_key()
        check("add key: current frame and view; real-time length to the next key",
              [k.frame for k in a.path.keys] == [5, 35] and a.path.keys[0].length == 30
              and a.table.rowCount() == 2)
        vol.el = math.radians(80)
        a.add_key()                                         # freeze at 35, orbit up
        check("third key on the same frame: default length 2 s",
              a.path.keys[1].length == round(2 * a._video_fps()) and a.path.keys[2].frame == 35)
        a.table.cellWidget(2, 0).setValue(50)
        a.table.cellWidget(1, 2).setCurrentText("Spline")
        check("table edits reach the path", a.path.keys[2].frame == 50 and a.path.keys[1].easing == "Spline")
        a.table.selectRow(0)
        a.go_to_key()
        QApplication.processEvents()
        check("go to key: frame and view", v.scrubber.current_frame == 5 and vol.frame == 5
              and abs(vol.az - a.path.keys[0].view["az"]) < 1e-9)
        a.slider.setValue(15)
        QApplication.processEvents()
        check("timeline slider drives the viewer (frame and camera)",
              vol.frame == a.path.frame_at(15) == 20
              and abs(vol.view_state()["az"] - a.path.state_at(15)["az"]) < 1e-9)
        a.slider.setValue(0)
        a.play_btn.setChecked(True)
        for _ in range(5):
            a._tick()
        check("preview steps through the animation", a.slider.value() == 5)
        a.play_btn.setChecked(False)
        a.table.selectRow(2)
        vol.dist = 123.0
        a.update_key()
        check("update key", abs(a.path.keys[2].view["dist"] - 123.0) < 1e-9)
        a.table.selectRow(2)
        a.delete_key()
        check("delete key", len(a.path.keys) == 2 and a.table.rowCount() == 2)
        saved = a.save(td / "s.camera.json")
        before = [k.to_dict() for k in a.path.keys]
        a.path = CameraPath()
        a._refresh()
        a.load(saved)
        check("save / load", [k.to_dict() for k in a.path.keys] == before)

        # ---- rendering -------------------------------------------------- #
        import cv2
        a.panels_cb.setChecked(False)          # 122lu: the 3-D view alone (All panels: tested in 122lu)
        a.width_spin.setValue(320)
        a.height_spin.setValue(240)
        a.below_cb.setChecked(True)
        out = a.render_to(td / "anim.avi")
        cap = cv2.VideoCapture(str(out))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        check(f"render: {len(frames)} frames ({a.path.n_output()} expected), 320 × (240 + video below)",
              len(frames) == n == a.path.n_output() and frames[0].shape == (240 + 240, 320, 3))
        # frame 10 of the output against the view drawn directly at that state
        ref = Volume3DWidget(load_pose3d(pose_path))
        ref.resize(320, 240)
        ref.show_label = False
        ref.set_frame(a.path.frame_at(10))
        ref.set_view_state(a.path.state_at(10))
        img = QImage(320, 240, QImage.Format_RGB32)
        ref.show_video = False
        vol_style = a.settings()["style"]
        check("video in 3-D carried into the render settings", vol_style["show_video"])
        out2 = render_animation(pose_path, a.path, td / "nv.avi", size=(320, 240), fps=30,
                                style={**vol_style, "show_video": False}, colors=dict(vol.colors))
        cap2 = cv2.VideoCapture(str(out2))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 10)
        ok, f10 = cap2.read()
        ref.colors.update(vol.colors)
        for k in ("show_box", "show_shadows", "show_drops", "trail"):
            setattr(ref, k, vol_style[k])
        ref.render(img)
        diff = np.abs(f10.astype(float) - _qimage_bgr(img).astype(float)).mean()
        check(f"each output frame is the view at that state (mean diff {diff:.1f} grey levels, MJPG)",
              ok and diff < 4)
        moved = np.abs(frames[0][:240].astype(float) - frames[25][:240].astype(float)).mean()
        check(f"the camera moves ({moved:.1f})", moved > 5)
        below = cv2.resize(cv2.VideoCapture(str(td / "s.avi")).read()[1], (320, 240))
        check("video below the 3-D view",
              np.abs(frames[0][240:].astype(float) - below.astype(float)).mean() < 8)

        # ---- CLI -------------------------------------------------------- #
        r = subprocess.run([sys.executable, "-m", "mufasa.tools.pose3d_cli", "render-animation", str(saved),
                            "--pose", str(pose_path), "--video", str(td / "s.avi"), "--out",
                            str(td / "cli.avi"), "--size", "160", "120"],
                           capture_output=True, text=True, cwd=REPO, timeout=600)
        cc = cv2.VideoCapture(str(td / "cli.avi"))
        check(f"CLI render-animation ({r.returncode})",
              r.returncode == 0 and int(cc.get(cv2.CAP_PROP_FRAME_COUNT)) == a.path.n_output()
              and int(cc.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160)
        if r.returncode:
            print(r.stdout[-800:], r.stderr[-1500:])
        v.cleanup()

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kp_camera_animation: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
