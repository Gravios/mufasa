"""Smoke test for patch 122lb — the posed rig's frame cache, safe to share with the background
fit thread (audit finding 2).

User: "Do it" (the audit's second patch).

Since 122ky the background fit thread writes posed frames into ``PosedRig._cache`` while the
viewer iterates it on every paint; a plain dict raises "dictionary changed size during
iteration" when that happens mid-iteration.

* ``PoseCache`` reads like a dict (get, ``in``, ``[]``, update, clear, len, iteration);
* ``nearest`` gives the same frame as a scan over every key, by binary search;
* a writer thread and reader threads together: no error (the old pattern is run alongside, for
  comparison);
* the viewer painting frame after frame while a real background fit runs: no error;
* nearest on 108,000 posed frames: a binary search, not a scan.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def stress(make_reader, writer_update, seconds=1.5, readers=3):
    """Errors raised by reader threads while one thread writes."""
    errors, stop = [], threading.Event()

    def writer():
        k = 0
        while not stop.is_set():
            writer_update({k + j: (np.zeros(3), 1.0) for j in range(50)})
            k += 50
            time.sleep(0)

    def reader():
        f = make_reader()
        while not stop.is_set():
            try:
                f()
            except Exception as e:  # noqa: BLE001
                errors.append(type(e).__name__ + ": " + str(e))
                return
    th = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(readers)]
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                     # switch threads often, as under load
    try:
        for t in th:
            t.start()
        time.sleep(seconds)
        stop.set()
        for t in th:
            t.join(10)
    finally:
        sys.setswitchinterval(old)
    return errors


def main() -> int:
    from mufasa.pose3d.scene_models import PoseCache

    # dict-like
    c = PoseCache()
    c[5] = ("a", 1.0)
    c.update({2: ("b", 2.0), 9: ("c", 3.0)})
    ok = (5 in c and c[5] == ("a", 1.0) and c.get(7) is None and c.get(2)[1] == 2.0 and len(c) == 3
          and list(c) == [2, 5, 9] and c.keys() == [2, 5, 9])
    c.clear()
    check("reads like a dict: [], get, in, update, len, sorted iteration, clear", ok and len(c) == 0 and list(c) == [])

    # nearest = a scan
    rng = np.random.default_rng(0)
    keys = sorted(set(rng.integers(0, 5000, 800).tolist()))
    c.update({k: (None, 0.0) for k in keys})
    bad = 0
    for q in rng.integers(-50, 5050, 3000):
        for w in (0, 3, 15, 30):
            near = [k for k in keys if abs(k - q) <= w]
            want = min(near, key=lambda k: (abs(k - q), k)) if near else None
            bad += c.nearest(int(q), w) != want
    check(f"nearest = a scan over every frame (12,000 queries, {bad} different)", bad == 0)

    # threads: the old pattern (a dict, a comprehension over it) and the cache
    d = {}

    def old_reader():
        return lambda: [f for f in d if abs(f - 100) <= 15]
    old_err = stress(old_reader, d.update)
    pc = PoseCache()

    def new_reader():
        def r():
            pc.nearest(100, 15)
            list(pc)
            50 in pc
            pc.get(60)
        return r
    new_err = stress(new_reader, pc.update)
    check(f"a writer thread and three readers: no error ({len(new_err)}; a plain dict: "
          f"{old_err[0][:60] if old_err else 'no error this time'})", not new_err)

    # the viewer painting while a real background fit runs
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    from smoke_122ky_viewer_gpu_realtime import synthetic
    QApplication.instance() or QApplication([])
    import tempfile

    from mufasa.ui_qt.pose3d_models import FitThread
    from mufasa.ui_qt.pose3d_viewer import Volume3DWidget, load_pose3d
    tmp = Path(tempfile.mkdtemp())
    os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")
    path, truth = synthetic(tmp, n=60)
    pose = load_pose3d(path)
    from mufasa.pose3d.scene_models import load_model
    m = load_model("virtual_rodent", pose)
    m.size = 1.3
    vol = Volume3DWidget(pose)
    vol.setAttribute(Qt.WA_DontShowOnScreen, True)
    vol.resize(320, 240)
    vol.gpu = False
    vol.add_model(m)
    vol.bg_fitting = True
    frames = [100 + k for k in range(60)]
    th = FitThread(m, frames, lambda: vol.frame)
    th.first, th.chunk = 8, 8                                # many small batches: many writes
    errs = []
    img = QImage(320, 240, QImage.Format_RGB32)
    th.start()
    k, t0 = 0, time.time()
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        while th.isRunning() and time.time() - t0 < 300:
            try:
                vol.set_frame(frames[k % len(frames)])
                vol.render(img)
                m.nearest_cached(frames[(k * 7) % len(frames)])
            except Exception as e:  # noqa: BLE001
                errs.append(f"{type(e).__name__}: {e}")
                break
            k += 1
            QApplication.processEvents()
    finally:
        sys.setswitchinterval(old)
        th.wait(60000)
    check(f"viewer: {k} paints while the background fit posed {len(m._cache)} frames, no error "
          f"({errs[0][:60] if errs else 'none'})", not errs and len(m._cache) == len(frames) and k > 3)
    vol.set_frame(frames[10])
    check("the posed frames show (nearest posed frame within 15)", m.nearest_cached(frames[10]) == frames[10])

    # nearest on a long recording
    big = PoseCache()
    big.update({f: (None, 0.0) for f in range(0, 108000, 2)})
    t0 = time.time()
    for q in range(0, 108000, 97):
        big.nearest(q + 1, 15)
    per = (time.time() - t0) / len(range(0, 108000, 97))
    dd = {f: None for f in range(0, 108000, 2)}
    t0 = time.time()
    for q in range(0, 108000, 9700):
        [f for f in dd if abs(f - q) <= 15]
    scan = (time.time() - t0) / len(range(0, 108000, 9700))
    check(f"nearest on 54,000 posed frames: {per * 1e6:.1f} µs (a scan: {scan * 1e3:.1f} ms)", per < 0.05 * scan)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lb_pose_cache: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
