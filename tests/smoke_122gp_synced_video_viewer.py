"""
tests/smoke_122gp_synced_video_viewer.py
========================================

Patch 122gp — open two or more videos docked together in a separate window
with synchronised frames.

SyncedVideoViewer (QMainWindow) holds each video as a FrameScrubberWidget in
its own QDockWidget (tileable/floatable). Any pane can drive: scrubbing one
seeks the others to the matching TIME (frame/fps), offset by a per-video
reference captured whenever sync is enabled — so different fps / lengths
stay aligned, and manual alignment can be locked in. Launched from the
workbench Tools menu.

Functional checks exec the sync math on stub scrubbers (no Qt/cv2 needed).
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
MOD = REPO / "mufasa" / "ui_qt" / "synced_video_viewer.py"
WB = REPO / "mufasa" / "ui_qt" / "workbench.py"

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main():
    src = MOD.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"FAIL: parse — {e}")
        print("smoke_122gp_synced_video_viewer: 0/7 checks passed")
        return 1
    check("synced_video_viewer.py parses", True)

    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "SyncedVideoViewer"), None)
    check("SyncedVideoViewer(QMainWindow) + open_synced_video_viewer defined",
          cls is not None
          and {b.id for b in cls.bases if isinstance(b, ast.Name)} == {"QMainWindow"}
          and "def open_synced_video_viewer" in src)

    check("each video is a FrameScrubberWidget in its own QDockWidget",
          "FrameScrubberWidget(self)" in src and "QDockWidget(" in src
          and "addDockWidget" in src and "splitDockWidget" in src)

    check("re-entrancy guard + sync toggle present",
          "self._guard" in src and "self._sync_enabled" in src
          and "_on_sync_toggled" in src)

    wb = WB.read_text(encoding="utf-8")
    check("workbench registers Tools action + _launch_synced_viewer",
          "_launch_synced_viewer" in wb
          and "open_synced_video_viewer" in wb
          and "getOpenFileNames" in wb)

    # --- functional: sync math on stubs ---
    m = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    funcs = {"round": round, "max": max, "min": min, "enumerate": enumerate}
    for name in ("_capture_reference", "_on_sync_toggled", "_on_frame"):
        exec(compile(ast.Module([m[name]], []), "<s>", "exec"), funcs)

    class Stub:
        def __init__(s, fps, n, cur=0):
            s.fps = fps
            s.total_frames = n
            s._cur = cur
            s.seeks = []

        @property
        def current_frame(s):
            return s._cur

        def seek(s, f):
            s._cur = f
            s.seeks.append(f)

    class V:
        _capture_reference = funcs["_capture_reference"]
        _on_sync_toggled = funcs["_on_sync_toggled"]
        _on_frame = funcs["_on_frame"]

        def __init__(s, scr):
            s._scrubbers = scr
            s._ref_frames = [x.current_frame for x in scr]
            s._guard = False
            s._sync_enabled = True

    a, b = Stub(30, 1000), Stub(60, 2000)
    V([a, b])._on_frame(0, 60)  # 2s at 30fps -> 120 at 60fps
    check("time-based sync across differing fps", b.current_frame == 120)

    a, b = Stub(30, 1000, 50), Stub(30, 1000, 70)
    V([a, b])._on_frame(0, 80)  # +30 from ref 50 -> 70+30
    guard_a, guard_b = Stub(30, 100), Stub(30, 100)
    vg = V([guard_a, guard_b])
    vg._on_frame(0, 10)
    off = Stub(30, 100), Stub(30, 50)
    voff = V(list(off))
    voff._on_sync_toggled(False)
    voff._on_frame(0, 40)
    voff2 = V([Stub(30, 100), Stub(30, 50)])
    voff2._on_frame(0, 90)  # clamp to 49
    check("reference offset + guard + sync-off + clamp",
          b.current_frame == 100
          and guard_b.seeks == [10]
          and off[1].current_frame == 0
          and voff2._scrubbers[1].current_frame == 49)

    print(f"smoke_122gp_synced_video_viewer: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
