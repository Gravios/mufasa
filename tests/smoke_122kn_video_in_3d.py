"""Smoke test for patch 122kn — the video drawn into the 3-D view.

User: "For the 3d view I would like to have the option to place the video
in a view port below the 3d reconstruction so that the markers are aligned
with the video when viewed from above."

A synthetic tilted camera films a synthetic scene: a video whose frames
carry a bright dot at the image of a floor point and of a raised point.
The rendered 3-D view is searched for those dots:

* video on the floor (Top and Perspective views): the floor point's dot
  lands where the 3-D view draws that floor point;
* Camera view: both dots — floor and raised — land on their markers, and
  stay there after zooming and panning;
* the frame is undistorted first when the camera has lens distortion; the
  Reprojection overlay distorts its points to match the raw video;
* the floor is not drawn when it would reach behind the eye;
* viewer controls: "Video in 3-D" (on when a camera and a video exist),
  opacity, "Video panel" (picture hidden, playback controls kept); the
  scrubber's raw frame; viewer_meta carries K, dist and the image width.
"""
from __future__ import annotations

import os
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


W, H = 320, 240
FLOOR = np.array([40.0, -30.0, 0.0])
RAISED = np.array([-60.0, 50.0, 90.0])
EDGE = np.array([-188.5, -142.4, 60.0])     # near the image corner: moved ~21 px by the lens


def make_camera(dist=None):
    from scipy.spatial.transform import Rotation

    from mufasa.pose3d.camera import Camera
    R = Rotation.from_euler("x", 160, degrees=True).as_matrix()          # tilted, looking down
    C = np.array([0.0, -150.0, 450.0])
    K = np.array([[260.0, 0, 165], [0, 260.0, 118], [0, 0, 1]])
    return Camera(K=K, R=R, t=-R @ C, dist=np.zeros(5) if dist is None else np.asarray(dist, float),
                  image_size=(W, H), box={"x": [-200, 200], "y": [-150, 150], "z": [0, 200]})


def write_pose(cam, path: Path, n=20, raised=RAISED):
    import pandas as pd

    from mufasa.pose3d.io import write_pose3d
    df = pd.DataFrame({"frame": np.arange(n)})
    for name, X in (("floor_pt", FLOOR), ("raised_pt", raised)):
        for k, c in enumerate("xyz"):
            df[f"{name}_{c}"] = X[k]
        df[f"{name}_likelihood"] = 1.0
    meta = {"units": "mm", "camera": cam.viewer_meta(), "edges": [],
            "box_x": [-200, 200], "box_y": [-150, 150], "box_z": [0, 200]}
    return write_pose3d(df, path, meta)


def write_video(cam, path: Path, n=20, *, distort: bool, raised=RAISED):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
    uv = cam.project(np.vstack([FLOOR, raised]), distort=distort)
    for _ in range(n):
        img = np.full((H, W, 3), 40, np.uint8)
        cv2.circle(img, tuple(int(round(v)) for v in uv[0]), 4, (0, 0, 255), -1)       # red: floor point
        cv2.circle(img, tuple(int(round(v)) for v in uv[1]), 4, (0, 255, 0), -1)       # green: raised point
        vw.write(img)
    vw.release()
    return uv


def blob(img, rgb) -> np.ndarray | None:
    """Centre of the pixels close to ``rgb`` (QImage → numpy), or None."""
    from PySide6.QtGui import QImage
    q = img.convertToFormat(QImage.Format_RGB888)
    a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, :q.width() * 3]
    a = a.reshape(q.height(), q.width(), 3).astype(int)
    m = (np.abs(a - np.array(rgb)).sum(2) < 90)
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]) if len(xs) > 3 else None


def main() -> int:
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from mufasa.ui_qt.pose3d_viewer import Pose3DViewerWidget, load_pose3d

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cam = make_camera()
        meta = cam.viewer_meta()
        check("viewer_meta: K, dist, image width and height",
              meta["image_width_px"] == W and meta["image_height_px"] == H and len(meta["K"]) == 3
              and len(meta["dist"]) == 5)
        pose_path = write_pose(cam, td / "s_3d.parquet")
        write_video(cam, td / "s.avi", distort=False)
        v = Pose3DViewerWidget(str(td / "s.avi"), pose_path)
        v.resize(900, 800)
        v.show()
        QApplication.processEvents()
        vol = v.volume
        check("'Video in 3-D' on (camera and video present); frame loaded",
              v.video3d_cb.isEnabled() and v.video3d_cb.isChecked() and vol.video_img is not None
              and v.scrubber.raw_frame() is not None)
        # markers drawn tiny and transparent here so the video's dots show
        for c in vol.colors.values():
            c.setAlpha(0)
        vol.show_drops = vol.show_shadows = vol.show_box = False

        def render():
            QApplication.processEvents()
            return vol.grab().toImage()

        for view in ("Top", "Perspective"):
            vol.reset_view(view)
            img = render()
            got = blob(img, (255, 0, 0))
            want = vol.project(FLOOR[None])[0][0]
            err = np.linalg.norm(got - want) if got is not None else np.inf
            check(f"{view} view: the floor point's dot lands on it ({err:.1f} px)", err < 3)
        vol.reset_view("Camera")
        img = render()
        errs = []
        for X, rgb in ((FLOOR, (255, 0, 0)), (RAISED, (0, 255, 0))):
            got = blob(img, rgb)
            errs.append(np.linalg.norm(got - vol.project(X[None])[0][0]) if got is not None else np.inf)
        check(f"Camera view: floor and raised point on their markers ({errs[0]:.1f}, {errs[1]:.1f} px)",
              vol.cam_mode and max(errs) < 3)
        vol.cam_zoom, vol.cam_pan = 1.7, np.array([35.0, -20.0])
        img = render()
        got = blob(img, (0, 255, 0))
        err = np.linalg.norm(got - vol.project(RAISED[None])[0][0]) if got is not None else np.inf
        check(f"Camera view zoomed and panned: still aligned ({err:.1f} px)", err < 3)
        vol.reset_view("Top")
        vol.el = np.radians(-5.0)                 # eye below the floor, looking up and across
        vol.target = np.array([0.0, 0.0, 400.0])
        vol.dist = 50.0
        check("floor reaching behind the eye: not drawn", vol.video_transform() is None)
        v.panel_cb.setChecked(False)
        check("'Video panel' off hides the picture, keeps the controls",
              not v.scrubber._display.isVisible() and v.scrubber._slider.isVisible())
        v.opacity_spin.setValue(40)
        check("opacity", abs(vol.video_opacity - 0.4) < 1e-9)
        v.video3d_cb.setChecked(False)
        check("'Video in 3-D' off: nothing drawn", vol.video_img is None and not vol.show_video)
        v.cleanup()

        # lens distortion: the frame is undistorted before it goes on the floor
        camd = make_camera(dist=[-0.4, 0.12, 0, 0, 0])
        pd_path = write_pose(camd, td / "d_3d.parquet", raised=EDGE)
        write_video(camd, td / "d.avi", distort=True, raised=EDGE)
        pose = load_pose3d(pd_path)
        check("distorted camera: undistorter present", pose.undistorter() is not None
              and load_pose3d(pose_path).undistorter() is None)
        v2 = Pose3DViewerWidget(str(td / "d.avi"), pd_path)
        v2.resize(900, 800)
        v2.show()
        vol2 = v2.volume
        for c in vol2.colors.values():
            c.setAlpha(0)
        vol2.show_drops = vol2.show_shadows = vol2.show_box = False
        vol2.reset_view("Camera")

        def edge_err():
            QApplication.processEvents()
            got = blob(vol2.grab().toImage(), (0, 255, 0))
            return np.linalg.norm(got - vol2.project(EDGE[None])[0][0]) if got is not None else np.inf
        err = edge_err()
        und, vol2._undistort = vol2._undistort, None              # what it would be without
        vol2.set_video_frame(v2.scrubber.raw_frame())
        err_raw = edge_err()
        vol2._undistort = und
        check(f"distorted video, Camera view: point near the corner aligned after undistortion "
              f"({err:.1f} px; {err_raw:.0f} px without)", err < 3 and err_raw > 20)
        # Reprojection onto the (distorted) video panel lands on the dot too
        from PySide6.QtGui import QColor

        from mufasa.ui_qt.pose3d_viewer import reprojection_overlay
        frame = v2.scrubber.raw_frame()
        drawn = reprojection_overlay(pose, {n: QColor(0, 0, 255) for n in pose.names})(np.zeros_like(frame), 0)
        ys, xs = np.nonzero(drawn[..., 0] > 128)
        want = camd.project(EDGE[None], distort=True)[0]
        near = np.hypot(xs - want[0], ys - want[1]).min() if len(xs) else np.inf
        check(f"Reprojection on the distorted video: the edge point drawn where the lens puts it ({near:.1f} px)",
              near < 2)
        v2.cleanup()

    v_src = (REPO / "mufasa" / "ui_qt" / "pose3d_viewer.py").read_text()
    check("viewer module documents the video in the 3-D view", "Patch 122kn" in v_src)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kn_video_in_3d: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
