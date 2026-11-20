"""
mufasa.pose3d.animation
=======================

Patch 122kp: camera paths for the 3-D viewer, and rendering them to a
video. The editor is :class:`mufasa.ui_qt.pose3d_animation.CameraAnimationPanel`;
the description of keys, easing and the camera view is there. Qt is
imported only while rendering, so the ``mufasa-pose3d`` command line stays
free of the GUI until ``render-animation`` is used.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EASINGS = ("Ease in-out", "Linear", "Spline")
STYLE_KEYS = ("show_box", "show_shadows", "show_drops", "trail", "p_threshold", "show_video",
              "video_opacity",
              # 122kv: view options
              "show_markers", "marker_scale", "show_names", "show_skeleton", "show_grid", "show_axes",
              "ortho", "background")


# --------------------------------------------------------------------------- #
# keys and the path through them
# --------------------------------------------------------------------------- #
@dataclass
class CameraKey:
    frame: int
    view: dict                                    # Volume3DWidget.view_state()
    length: int = 60                              # output frames to the next key
    easing: str = "Ease in-out"

    def to_dict(self) -> dict:
        return {"frame": int(self.frame), "length": int(self.length), "easing": self.easing,
                "view": self.view}

    @classmethod
    def from_dict(cls, d: dict) -> CameraKey:
        e = d.get("easing", "Ease in-out")
        return cls(int(d["frame"]), dict(d["view"]), max(1, int(d.get("length", 60))),
                   e if e in EASINGS else "Ease in-out")

    def describe(self) -> str:
        v = self.view
        if v.get("cam_mode"):
            return f"camera view ×{v.get('cam_zoom', 1.0):.2f}"
        return (f"az {math.degrees(v['az']) % 360:.0f}°  el {math.degrees(v['el']):.0f}°  "
                f"dist {v['dist']:.0f}")


def _smooth(u: float) -> float:
    return u * u * u * (u * (6 * u - 15) + 10)           # smootherstep: 0 velocity and acceleration at ends


@dataclass
class CameraPath:
    keys: list[CameraKey] = field(default_factory=list)

    def n_output(self) -> int:
        if not self.keys:
            return 0
        return sum(k.length for k in self.keys[:-1]) + 1

    def key_starts(self) -> list[int]:
        out, n = [], 0
        for k in self.keys:
            out.append(n)
            n += k.length
        return out

    def locate(self, n: int) -> tuple[int, float]:
        """Output frame → (segment i, position u in [0, 1))."""
        if len(self.keys) < 2:
            return 0, 0.0
        n = max(0, min(int(n), self.n_output() - 1))
        for i, s in enumerate(self.key_starts()[:-1]):
            if n < s + self.keys[i].length:
                return i, (n - s) / self.keys[i].length
        return len(self.keys) - 1, 0.0

    def frame_at(self, n: int) -> int:
        i, u = self.locate(n)
        if i >= len(self.keys) - 1:
            return int(self.keys[-1].frame)
        a, b = self.keys[i].frame, self.keys[i + 1].frame
        return int(round(a + u * (b - a)))

    def _params(self) -> np.ndarray:
        """Orbit parameters per key: az (unwrapped: shorter way round), el,
        log dist, target xyz, fov."""
        rows, prev = [], None
        for k in self.keys:
            v = k.view
            az = float(v["az"])
            if prev is not None:
                az = prev + (az - prev + math.pi) % (2 * math.pi) - math.pi
            prev = az
            rows.append([az, v["el"], math.log(max(v["dist"], 1e-9)), *v["target"], v.get("fov", 0.7)])
        return np.asarray(rows, float)

    def state_at(self, n: int) -> dict:
        """The view at output frame ``n``."""
        if not self.keys:
            raise ValueError("no keys")
        i, u = self.locate(n)
        if i >= len(self.keys) - 1 or u == 0.0:
            return dict(self.keys[i].view)
        ka, kb = self.keys[i], self.keys[i + 1]
        P = self._params()
        if ka.easing == "Spline":
            p0, p1, p2, p3 = (P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)])
            u2, u3 = u * u, u * u * u
            p = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
            s = u
        else:
            s = _smooth(u) if ka.easing == "Ease in-out" else u
            p = P[i] + s * (P[i + 1] - P[i])
        lim = math.radians(89.9)
        out = {"az": float(p[0]), "el": float(max(-lim, min(lim, p[1]))), "dist": float(math.exp(p[2])),
               "target": [float(v) for v in p[3:6]], "fov": float(p[6]),
               "cam_mode": False, "cam_zoom": 1.0, "cam_pan": [0.0, 0.0]}
        va, vb = ka.view, kb.view
        if va.get("cam_mode") and vb.get("cam_mode"):        # stay in the camera view
            za, zb = math.log(va.get("cam_zoom", 1.0)), math.log(vb.get("cam_zoom", 1.0))
            pa, pb = np.asarray(va.get("cam_pan", (0, 0)), float), np.asarray(vb.get("cam_pan", (0, 0)), float)
            out.update(cam_mode=True, cam_zoom=float(math.exp(za + s * (zb - za))),
                       cam_pan=[float(v) for v in pa + s * (pb - pa)])
        elif va.get("cam_mode") or vb.get("cam_mode"):        # into / out of the camera view
            cv = va if va.get("cam_mode") else vb
            w = (1.0 - s) if va.get("cam_mode") else s
            out.update(cam_blend=float(min(1.0, max(0.0, w))), cam_zoom=float(cv.get("cam_zoom", 1.0)),
                       cam_pan=[float(v) for v in cv.get("cam_pan", (0, 0))])
        return out

    # -- files --------------------------------------------------------------
    def to_json(self, **settings) -> str:
        return json.dumps({"version": 1, **settings, "keys": [k.to_dict() for k in self.keys]}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> tuple[CameraPath, dict]:
        d = json.loads(text)
        keys = [CameraKey.from_dict(k) for k in d.get("keys", [])]
        settings = {k: v for k, v in d.items() if k not in ("keys", "version")}
        return cls(keys), settings


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
class _VideoFrames:
    """Video frames by index, reading on when the next one is asked for."""

    def __init__(self, path) -> None:
        import cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise OSError(f"cannot open video {path}")
        self.last, self.img = -2, None

    def get(self, i: int):
        import cv2
        if i == self.last:
            return self.img
        if i != self.last + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = self.cap.read()
        self.last, self.img = i, (img if ok else None)
        return self.img

    def close(self) -> None:
        self.cap.release()


def _writer(out_path: Path, fps: float, size: tuple[int, int]):
    import cv2
    four = "MJPG" if out_path.suffix.lower() == ".avi" else "mp4v"
    w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*four), float(fps), size)
    if not w.isOpened():
        raise OSError(f"cannot write {out_path} ({four})")
    return w


def _qimage_bgr(img) -> np.ndarray:
    from PySide6.QtGui import QImage
    q = img.convertToFormat(QImage.Format_BGR888)
    a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())
    return a[:, :q.width() * 3].reshape(q.height(), q.width(), 3).copy()


def render_animation(pose, path: CameraPath, out_path, *, video_path=None, size=(1280, 720),
                     fps: float = 30.0, style: dict | None = None, colors: dict | None = None,
                     video_below: bool = False, frame_label: bool = False, progress=None,
                     models: list | None = None) -> Path:
    """Render ``path`` to a video file; returns its path. ``pose`` is a
    :class:`~mufasa.ui_qt.pose3d_viewer.Pose3D` (or a path to one).
    ``models``: model objects or their specs (``spec()``, 122kv) to draw;
    rigs are fitted to each rendered frame.
    ``progress(done, total)`` returning False stops (the partial file is
    kept)."""
    import cv2
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    from mufasa.ui_qt.pose3d_viewer import Volume3DWidget, load_pose3d  # GUI only while rendering
    if isinstance(pose, (str, Path)):
        pose = load_pose3d(pose)
    if len(path.keys) == 0:
        raise ValueError("the camera path has no keys")
    w, h = (int(size[0]) // 2 * 2, int(size[1]) // 2 * 2)
    vol = Volume3DWidget(pose)
    vol.setAttribute(Qt.WA_DontShowOnScreen, True)
    vol.resize(w, h)
    for k, v in (style or {}).items():
        if k in STYLE_KEYS:
            setattr(vol, k, v)
    if colors:
        vol.colors.update(colors)
    if models:
        from mufasa.pose3d.scene_models import model_from_spec
        for m in models:
            vol.add_model(model_from_spec(m, pose) if isinstance(m, dict) else m)
        vol.sync_fit = True
    vol.show_label = bool(frame_label)
    video = _VideoFrames(video_path) if video_path else None
    if video is None:
        vol.show_video = False
    vh = 0
    if video is not None and video_below:
        iw, ih = pose.image_size() or (int(video.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        vh = int(round(w * ih / max(iw, 1))) // 2 * 2
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _writer(out_path, fps, (w, h + vh))
    total = path.n_output()
    img = QImage(w, h, QImage.Format_RGB32)
    try:
        for n in range(total):
            f = path.frame_at(n)
            vol.set_frame(f)
            raw = video.get(f) if video is not None else None
            if vol.show_video:
                vol.set_video_frame(raw)
            vol.set_view_state(path.state_at(n))
            vol.render(img)
            frame = _qimage_bgr(img)
            if vh:
                below = (cv2.resize(raw, (w, vh), interpolation=cv2.INTER_AREA) if raw is not None
                         else np.zeros((vh, w, 3), np.uint8))
                frame = np.vstack([frame, below])
            writer.write(frame)
            if progress is not None and progress(n + 1, total) is False:
                break
    finally:
        writer.release()
        if video is not None:
            video.close()
        vol.deleteLater()
    return out_path


__all__ = ["EASINGS", "STYLE_KEYS", "CameraKey", "CameraPath", "render_animation"]
