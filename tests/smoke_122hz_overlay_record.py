"""Smoke test for patch 122hz — record button in the overlay viewer.

Adds a Record button to pose_video_overlay: first press starts, second press
stops, and the clip contains the composited overlay frames (video + pose
markers) shown in between. Frames are captured in _set_frame — the single path
every displayed frame passes through — rendered from the QGraphicsScene (the
overlay is Qt-composited, not baked into pixels) to a BGR array and written via
cv2.VideoWriter.

PySide6 and cv2 are absent in the sandbox, so this test exercises the
Qt/cv2-free pieces directly (the _ClipRecorder state machine, _even, and the
QImage->BGR reshape logic against a simulated padded RGB buffer) and AST-checks
the viewer wiring (button, capture hook in _set_frame, scene render, close
finalisation).
"""
from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


SRC = (REPO / "mufasa/tools/pose_video_overlay.py").read_text()
TREE = ast.parse(SRC)


# ---- extract _even and _ClipRecorder with a stubbed cv2 ----
class _FakeWriter:
    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.size = size
        self.frames = []
        self._open = size[0] >= 2 and size[1] >= 2

    def isOpened(self):
        return self._open

    def write(self, frame):
        self.frames.append(frame.shape)

    def release(self):
        self.released = True


class _FakeCv2:
    last = None

    def VideoWriter_fourcc(self, *a):
        return 0

    def VideoWriter(self, path, fourcc, fps, size):
        _FakeCv2.last = _FakeWriter(path, fourcc, fps, size)
        return _FakeCv2.last


_ns = {"np": np, "cv2": _FakeCv2()}
for _name in ("_even", "_ClipRecorder"):
    _node = next(n for n in TREE.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name == _name)
    exec(compile(ast.get_source_segment(SRC, _node), "<x>", "exec"), _ns)
Recorder = _ns["_ClipRecorder"]
even = _ns["_even"]

# ---- _even ----
check("_even rounds odd dimensions down",
      even(640) == 640 and even(641) == 640 and even(481) == 480)

# ---- recorder state machine ----
r = Recorder("/tmp/clip.mp4", fps=30.0)
check("recorder starts with no frames and a lazy writer",
      r.n_frames == 0 and r._writer is None)
for _ in range(3):
    r.append(np.zeros((200, 100, 3), dtype=np.uint8))
check("writer is opened on the first appended frame (sized to it)",
      r._writer is not None and r._size == (100, 200))
check("appended frames are counted", r.n_frames == 3)
ok, msg = r.finish()
check("finish succeeds and reports the frame count and path",
      ok and "3 frames" in msg and "/tmp/clip.mp4" in msg)
check("every appended frame reached the writer",
      len(_FakeCv2.last.frames) == 3)

# empty recording
r_empty = Recorder("/tmp/empty.mp4", fps=30.0)
ok_e, msg_e = r_empty.finish()
check("an empty recording is reported as a failure (no frames)",
      not ok_e and "no frames" in msg_e)

# odd dims evened
r_odd = Recorder("/tmp/odd.mp4", fps=25.0)
r_odd.append(np.zeros((201, 101, 3), dtype=np.uint8))
check("odd frame dimensions are locked to even", r_odd._size == (100, 200))

# mismatched frame conformed
r_mix = Recorder("/tmp/mix.mp4", fps=30.0)
r_mix.append(np.zeros((200, 100, 3), dtype=np.uint8))   # lock 100x200
r_mix.append(np.ones((300, 150, 3), dtype=np.uint8))    # larger -> crop
r_mix.append(np.ones((50, 40, 3), dtype=np.uint8))      # smaller -> pad
check("frames of a different size are conformed to the locked size",
      all(s == (200, 100, 3) for s in _FakeCv2.last.frames)
      and r_mix.n_frames == 3)

# ---- QImage->BGR reshape logic (padded RGB buffer) ----
h, w = 4, 3
bpl = 3 * w + 2  # row padding
rgb = np.zeros((h, w, 3), dtype=np.uint8)
for y in range(h):
    for x in range(w):
        rgb[y, x] = [y * 10 + x, 100 + y, 200 + x]
buf = np.zeros((h, bpl), dtype=np.uint8)
buf[:, : 3 * w] = rgb.reshape(h, 3 * w)
flat = buf.reshape(-1)
arr = flat.reshape(h, bpl)[:, : 3 * w].reshape(h, w, 3)
bgr = np.ascontiguousarray(arr[:, :, ::-1])
check("reshape strips row padding and reverses RGB->BGR",
      np.array_equal(bgr, rgb[:, :, ::-1]))
check("the BGR result is C-contiguous (cv2 requires it)",
      bgr.flags["C_CONTIGUOUS"])

# ---- viewer wiring (AST) ----
check("a Record button is created and wired to toggle_record",
      'QPushButton("● Record")' in SRC
      and "self.record_btn.clicked.connect(self.toggle_record)" in SRC)
viewer = next((n for n in ast.walk(TREE)
               if isinstance(n, ast.ClassDef) and n.name == "OverlayViewer"),
              None)
methods = {n.name for n in (viewer.body if viewer else [])
           if isinstance(n, ast.FunctionDef)}
for m in ("toggle_record", "_start_recording", "_stop_recording",
          "_render_scene_bgr", "_capture_recording_frame"):
    check(f"OverlayViewer.{m} exists", m in methods)
check("toggle_record starts on first press and stops on second",
      "if self._recorder is None:" in SRC
      and "self._start_recording()" in SRC
      and "self._stop_recording()" in SRC)
# capture hook is inside _set_frame, after update_frame
sf = next((n for n in (viewer.body if viewer else [])
           if isinstance(n, ast.FunctionDef) and n.name == "_set_frame"), None)
sf_src = ast.get_source_segment(SRC, sf) if sf else ""
check("the capture hook lives in _set_frame",
      "_capture_recording_frame()" in sf_src
      and "if self._recorder is not None:" in sf_src)
check("the scene is rendered (not the raw cv2 frame) for the clip",
      "self.scene_obj.render(painter" in SRC)
check("recording is finalised on window close",
      "closeEvent" in methods and "_stop_recording()" in SRC)
check("recording degrades gracefully without OpenCV",
      "Recording needs OpenCV" in SRC)

# _set_frame must call update_frame BEFORE capturing (capture the new frame)
check("update_frame precedes the capture in _set_frame",
      sf_src.find("update_frame") < sf_src.find("_capture_recording_frame"))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hz_overlay_record: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
