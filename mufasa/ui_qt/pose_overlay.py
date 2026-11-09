"""
mufasa.ui_qt.pose_overlay
=========================

Pose markers drawn on the video of the feature viewer, and recording what
the viewer shows (patch 122ju).

* **FDLC** — the raw tracker output for the video under ``sources/pose/``
  (``<video>.fdlc.parquet``, ``<video>.parquet``, a FreeDLC run's
  ``pose.parquet`` whose run sidecar names the video, or a CSV): hollow
  rings, one colour per marker; points below the likelihood threshold are
  not drawn.
* **Kalman** — the video's file in the latest Kalman v2 run
  (``derived/smoothed/kalman_v2/<run_id>/``): filled dots joined by the
  project / pose-file skeleton.

Pose row *i* is video frame *i*; ``scale`` multiplies the coordinates
(pose estimated on a half-resolution video → 2).

:class:`FrameRecorder` keeps every frame the viewer renders while recording
as numbered PNGs under ``derived/frames/extracted/<video>_rec_<time>/`` (with
``frames.csv`` mapping them to video frames) and, when stopped, encodes them
into ``derived/frames/annotated/<video>_rec_<time>.mp4`` — with ffmpeg
(H.264) when it is installed, else OpenCV (mp4v).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

POSE_EXTS = (".parquet", ".csv")
FDLC_COLOR = (255, 255, 255)


# --------------------------------------------------------------------------- #
# Finding the pose files of a video
# --------------------------------------------------------------------------- #
def _pose_stem(name: str) -> str:
    from mufasa.tools.pose_preview import _pose_stem as ps
    return ps(name)


def find_fdlc_pose(config_path, video: str) -> Path | None:
    """The raw (tracker) pose file of ``video`` under ``sources/pose``
    (searched two levels deep), or None."""
    root = Path(config_path).parent / "sources" / "pose"
    if not root.is_dir():
        return None
    hits: list[Path] = []
    for p in sorted(list(root.glob("*")) + list(root.glob("*/*"))):
        if not p.is_file() or p.suffix.lower() not in POSE_EXTS or "_smoothed_v2" in p.name:
            continue
        if _pose_stem(p.name) == video:
            hits.append(p)
            continue
        if p.suffix.lower() == ".parquet":
            try:
                from mufasa.pose_importers.pose_parquet import run_sidecar_video
                rv = run_sidecar_video(p)
            except Exception:
                rv = None
            if rv and Path(rv).stem == video:
                hits.append(p)
    if not hits:
        return None
    # prefer parquet, then the newest
    return sorted(hits, key=lambda q: (q.suffix.lower() != ".parquet",
                                       -q.stat().st_mtime))[0]


def find_kalman_pose(config_path, video: str) -> Path | None:
    """The video's smoothed file in the latest Kalman v2 run, or None."""
    try:
        from mufasa.project_layout import ProjectPaths
        from mufasa.tools.feature_landscape import latest_kfv2_files
        return latest_kfv2_files(ProjectPaths.open(Path(config_path).parent), None).get(video)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
@dataclass
class PoseOverlay:
    """Draws the enabled pose sources on BGR frames (call as a function)."""

    fdlc: object | None = None          # pose_viewer.PoseFrame
    kalman: object | None = None
    edges: list[tuple[str, str]] = field(default_factory=list)
    show_fdlc: bool = True
    show_kalman: bool = True
    show_skeleton: bool = True
    p_threshold: float = 0.5
    scale: float = 1.0
    offset: int = 0                     # pose frame = video frame + offset
    style: object = None                # overlay_style.OverlayStyle (122jw)

    def __post_init__(self) -> None:
        if self.style is None:
            from mufasa.ui_qt.overlay_style import OverlayStyle
            self.style = OverlayStyle()

    @classmethod
    def for_video(cls, config_path, video: str, video_path: str | None = None
                  ) -> "PoseOverlay":
        from mufasa.tools.pose_viewer import _load_pose_file
        fp, kp = find_fdlc_pose(config_path, video), find_kalman_pose(config_path, video)
        from mufasa.ui_qt.overlay_style import load_style
        ov = cls(style=load_style(config_path))
        ov.fdlc_path, ov.kalman_path = fp, kp
        ov.errors = {}
        for name, p in (("fdlc", fp), ("kalman", kp)):
            if p is None:
                continue
            try:
                setattr(ov, name, _load_pose_file(p))
            except Exception as exc:
                ov.errors[name] = f"{type(exc).__name__}: {exc}"
        try:
            from mufasa.pose_importers.pose_parquet import resolve_skeleton
            sk = resolve_skeleton([fp, kp, video_path], config=config_path)
        except Exception:
            sk = None
        if sk:
            nodes = list(sk.get("nodes") or [])
            for a, b in sk.get("edges", []):
                a = nodes[a] if isinstance(a, int) and a < len(nodes) else a
                b = nodes[b] if isinstance(b, int) and b < len(nodes) else b
                ov.edges.append((str(a), str(b)))
        return ov

    @property
    def available(self) -> dict[str, bool]:
        return {"fdlc": self.fdlc is not None, "kalman": self.kalman is not None}

    def _points(self, pf, i: int, use_p: bool) -> dict[str, tuple[float, float]]:
        i = i + self.offset
        if pf is None or not (0 <= i < pf.n_frames):
            return {}
        xy = pf.positions[i] * self.scale
        ok = np.isfinite(xy).all(axis=1)
        if use_p and pf.likelihoods is not None:
            lk = pf.likelihoods[i]
            ok &= ~(np.isfinite(lk) & (lk < self.p_threshold))
        return {m: (float(xy[k, 0]), float(xy[k, 1]))
                for k, m in enumerate(pf.markers) if ok[k]}

    def __call__(self, frame: np.ndarray, i: int) -> np.ndarray:
        from mufasa.ui_qt.overlay_style import draw_source
        if not ((self.show_fdlc and self.fdlc is not None)
                or (self.show_kalman and self.kalman is not None)):
            return frame
        out = frame.copy()
        h, w = out.shape[:2]
        r = max(3, int(round(min(h, w) / 120)))
        st = self.style
        legend = []
        # Kalman underneath, FDLC on top (patch 122jw: styles per source)
        for name, pf, use_p, sst in (("kalman", self.kalman, False, st.kalman),
                                     ("fdlc", self.fdlc, True, st.fdlc)):
            if pf is None or not getattr(self, f"show_{name}"):
                continue
            pts = {m: _ip(p) for m, p in self._points(pf, i, use_p=use_p).items()}
            draw_source(out, pts, list(pf.markers), self.edges, sst, r,
                        skeleton=self.show_skeleton)
            legend.append("Kalman" if name == "kalman"
                          else f"FDLC (p >= {self.p_threshold:.2f})")
        if st.legend:
            fs = max(0.4, min(h, w) / 900)
            for n, txt in enumerate(reversed(legend)):
                y = int(18 * fs * 1.6 * (n + 1))
                cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235),
                            1, cv2.LINE_AA)
        return out


def _ip(p: tuple[float, float]) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def _even(n: int) -> int:
    return n - (n % 2)


def encode_frames(frames_dir: Path, out_path: Path, fps: float,
                  prefer_ffmpeg: bool = True) -> tuple[bool, str]:
    """Encode ``frames_dir``'s numbered PNGs (``000001.png`` …) into a video.
    ffmpeg (H.264, yuv420p, even size) when available, else OpenCV mp4v."""
    frames = sorted(Path(frames_dir).glob("[0-9]*.png"))
    if not frames:
        return False, "no frames were recorded"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1.0, float(fps))
    ff = shutil.which("ffmpeg") if prefer_ffmpeg else None
    if ff:
        cmd = [ff, "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
               "-i", str(Path(frames_dir) / "%06d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "18", str(out_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and out_path.is_file():
            return True, f"{len(frames)} frames → {out_path} (ffmpeg, H.264)"
    first = cv2.imread(str(frames[0]))
    h, w = _even(first.shape[0]), _even(first.shape[1])
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not wr.isOpened():
        return False, f"could not open a video writer for {out_path}"
    for f in frames:
        img = cv2.imread(str(f))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h)) if abs(img.shape[0] - h) > 1 else img[:h, :w]
        wr.write(img)
    wr.release()
    return True, f"{len(frames)} frames → {out_path} (OpenCV, mp4v)"


class FrameRecorder:
    """Numbered PNGs while recording; ``stop()`` encodes them into a video."""

    def __init__(self, config_path, video: str, fps: float, keep_frames: bool = True,
                 stamp: str | None = None) -> None:
        root = Path(config_path).parent / "derived" / "frames"
        self.stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        self.name = f"{video}_rec_{self.stamp}"
        self.frames_dir = root / "extracted" / self.name
        self.video_path = root / "annotated" / f"{self.name}.mp4"
        self.fps = fps
        self.keep_frames = keep_frames
        self.n = 0
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._index = open(self.frames_dir / "frames.csv", "w")
        self._index.write("image,video_frame\n")
        self._last: int | None = None

    def add(self, frame_bgr: np.ndarray, video_frame: int) -> None:
        """Keep one rendered frame (a repeat of the same video frame, e.g. a
        re-render without moving, is skipped)."""
        if frame_bgr is None or video_frame == self._last:
            return
        self._last = video_frame
        self.n += 1
        name = f"{self.n:06d}.png"
        cv2.imwrite(str(self.frames_dir / name), frame_bgr,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])   # fast; playback keeps up
        self._index.write(f"{name},{video_frame}\n")

    def stop(self) -> tuple[bool, str]:
        self._index.close()
        ok, msg = encode_frames(self.frames_dir, self.video_path, self.fps)
        if ok and not self.keep_frames:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        return ok, msg


__all__ = ["FrameRecorder", "PoseOverlay", "encode_frames", "find_fdlc_pose",
           "find_kalman_pose"]
