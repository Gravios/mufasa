"""
tests/smoke_122go_scrubber_orientation.py
=========================================

Patch 122go — rotate/flip the video in the annotation GUI.

The annotation GUI (FrameLabellerWidget) embeds FrameScrubberWidget, so
adding a display-only orientation transform to the scrubber gives the
labeller (and the future synced viewer) rotate/flip. Annotation is
frame-level (behaviour labels), so this needs no coordinate remapping.

* Orientation state (rotation in {0,90,180,270} clockwise; flip_h/flip_v),
  applied in _render via _apply_orientation (cv2.flip/rotate), with the raw
  frame cached so changes re-render without re-seeking.
* Public API: rotate_cw / rotate_ccw / flip_horizontal / flip_vertical /
  reset_orientation / set_orientation + an `orientation` property.
* Control-bar buttons wired to those methods.
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SCRUB = REPO / "mufasa" / "ui_qt" / "frame_scrubber.py"

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main():
    src = SCRUB.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"FAIL: parse — {e}")
        print("smoke_122go_scrubber_orientation: 0/6 checks passed")
        return 1
    check("frame_scrubber.py parses", True)

    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "FrameScrubberWidget")
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    need = {"rotate_cw", "rotate_ccw", "flip_horizontal", "flip_vertical",
            "reset_orientation", "set_orientation", "_apply_orientation", "_rerender"}
    check("orientation methods present", need <= set(methods),
          detail=f"missing {need - set(methods)}")

    check("_render caches raw frame + applies orientation",
          "self._last_raw_frame = frame_bgr" in src
          and "_apply_orientation(frame_bgr)" in src)

    check("control-bar rotate/flip buttons wired",
          "self._b_rot_cw" in src and "self._b_rot_ccw" in src
          and "self._b_flip_h" in src and "self._b_flip_v" in src
          and ".clicked.connect(fn)" in src)

    # exec the state-transition methods on a stub (no cv2 needed)
    funcs = {}
    for name in ("rotate_cw", "rotate_ccw", "flip_horizontal", "flip_vertical",
                 "reset_orientation", "set_orientation"):
        exec(compile(ast.Module([methods[name]], []), "<s>", "exec"), funcs)

    class S:
        def __init__(s):
            s._rotation = 0
            s._flip_h = False
            s._flip_v = False

        def _rerender(s):
            pass

    s = S()
    funcs["rotate_cw"](s)
    funcs["rotate_cw"](s)
    cw_ok = s._rotation == 180
    funcs["rotate_ccw"](s)
    funcs["rotate_ccw"](s)
    funcs["rotate_ccw"](s)
    wrap_ok = s._rotation == 270  # 180->90->0->270 (wrap below 0)
    check("rotation cycles clockwise and wraps mod 360", cw_ok and wrap_ok)

    funcs["flip_horizontal"](s)
    fh = s._flip_h is True
    funcs["reset_orientation"](s)
    reset_ok = s._rotation == 0 and not s._flip_h and not s._flip_v
    funcs["set_orientation"](s, 450, True, False)
    set_ok = s._rotation == 90 and s._flip_h and not s._flip_v
    check("flip toggles; reset clears; set_orientation normalises mod 360",
          fh and reset_ok and set_ok)

    print(f"smoke_122go_scrubber_orientation: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
