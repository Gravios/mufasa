"""Smoke test for patch 122ia — scale pose coordinates for reduced-res video.

Pose estimated on a downscaled video is in reduced-resolution coordinates; on a
full-res video the markers cluster in a corner. --pose-scale FACTOR (and a live
Scale selector in the viewer) multiplies pose coordinates by the factor — 2 for
half-res, 4 for quarter-res. Scaling applies to marker positions, skeleton
endpoints, and variance-ellipse *radii* (all pose-pixel space), but NOT the
marker dot size (a fixed on-screen size). Threaded through OverlayScene, the
overlay CLI, and mufasa-preview.

PySide6/cv2 are absent in the sandbox, so the scaling arithmetic and the
scale-label helper are verified in isolation and the wiring is AST-checked.
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


OV = (REPO / "mufasa/tools/pose_video_overlay.py").read_text()
OV_TREE = ast.parse(OV)


# ---- scaling arithmetic (mirrors _update_pose_layer) ----
def dot_pos(x, y, s):
    return (float(x) * s, float(y) * s)


def line(xa, ya, xb, yb, s):
    return (float(xa) * s, float(ya) * s, float(xb) * s, float(yb) * s)


def ellipse(x, y, vx, vy, s):
    sx = 2.0 * float(np.sqrt(vx)) * s
    sy = 2.0 * float(np.sqrt(vy)) * s
    return (float(x) * s - sx, float(y) * s - sy, 2 * sx, 2 * sy)


check("scale=2 doubles a marker position",
      dot_pos(100, 80, 2) == (200.0, 160.0))
check("scale=4 quadruples a marker position",
      dot_pos(100, 80, 4) == (400.0, 320.0))
check("scale=1 leaves a position unchanged",
      dot_pos(100, 80, 1) == (100.0, 80.0))
check("scale doubles both skeleton endpoints",
      line(100, 80, 120, 90, 2) == (200.0, 160.0, 240.0, 180.0))

r2 = ellipse(100, 80, 4, 9, 2)
check("scale moves the ellipse centre with the marker",
      (r2[0] + r2[2] / 2, r2[1] + r2[3] / 2) == (200.0, 160.0))
check("scale multiplies the ellipse radii (uncertainty in pose space)",
      r2[2] / 2 == 8.0 and r2[3] / 2 == 12.0)
r1 = ellipse(100, 80, 4, 9, 1)
check("scale=1 leaves the ellipse radii at their base size",
      r1[2] / 2 == 4.0 and r1[3] / 2 == 6.0)


# ---- _scale_label helper (extracted; it's a Qt-free staticmethod) ----
_viewer = next(n for n in ast.walk(OV_TREE)
               if isinstance(n, ast.ClassDef) and n.name == "OverlayViewer")
_lbl_node = next(n for n in _viewer.body
                 if isinstance(n, ast.FunctionDef) and n.name == "_scale_label")
_code = "\n".join(
    ln for ln in ast.get_source_segment(OV, _lbl_node).splitlines()
    if not ln.strip().startswith("@")
)
_ns: dict = {}
exec(_code, _ns)
_scale_label = _ns["_scale_label"]
check("_scale_label prints integer factors without a decimal",
      _scale_label(2.0) == "2×" and _scale_label(4.0) == "4×")
check("_scale_label prints fractional factors with the value",
      _scale_label(1.5) == "1.5×")
check("_scale_label treats a falsy factor as 1×",
      _scale_label(0) == "1×")


# ---- overlay wiring (AST) ----
check("OverlayScene.__init__ accepts pose_scale",
      "pose_scale: float = 1.0" in OV and "self.pose_scale = " in OV)
check("dots are scaled by pose_scale",
      "dot.setPos(QPointF(float(x) * s, float(y) * s))" in OV)
check("skeleton endpoints are scaled by pose_scale",
      "float(xa) * s, float(ya) * s" in OV
      and "float(xb) * s, float(yb) * s" in OV)
check("ellipse centre and radii are scaled by pose_scale",
      "2.0 * float(np.sqrt(vx)) * s" in OV
      and "float(x) * s - sx" in OV)
# the dot marker RADIUS must NOT be scaled (fixed on-screen size). The radius
# is set once at dot creation; assert those definition lines carry no `* s`.
_radius_lines = [ln for ln in OV.splitlines()
                 if ln.strip().startswith(("radius_s =", "radius_r ="))]
check("the marker dot radius is a fixed size (not multiplied by s)",
      len(_radius_lines) == 2
      and all("* s" not in ln and "*s" not in ln for ln in _radius_lines)
      and "radius_s = 3.5" in OV)
check("the scene exposes set_pose_scale",
      "def set_pose_scale(self, factor" in OV)
check("--pose-scale is a CLI argument",
      '"--pose-scale"' in OV and "pose_scale=args.pose_scale" in OV)
check("OverlayViewer builds a live Scale selector",
      "self.scale_combo = QComboBox()" in OV
      and "self._on_scale_combo_picked" in OV
      and "self._on_scale_combo_edited" in OV)
check("the scale selector applies the factor and redraws",
      "self.scene_obj.set_pose_scale(factor)" in OV
      and "self._set_frame(self.scrubber.value())" in OV)
# a bad typed scale reverts rather than crashing
check("a non-numeric typed scale reverts to the current value",
      "except ValueError:" in OV and "self.scale_combo.setCurrentText(" in OV)


# ---- mufasa-preview passthrough ----
from mufasa.tools.pose_preview import _build_overlay_argv  # noqa: E402

argv = _build_overlay_argv("v.mp4", "s.parquet", "r.parquet", 0.0, 0, 0,
                           pose_scale=2.0)
check("mufasa-preview passes --pose-scale through",
      "--pose-scale" in argv and "2.0" in argv)
argv1 = _build_overlay_argv("v.mp4", "s.parquet", None, 0.0, 0, 0,
                            pose_scale=1.0)
check("mufasa-preview omits --pose-scale when it's 1 (no-op)",
      "--pose-scale" not in argv1)
PV = (REPO / "mufasa/tools/pose_preview.py").read_text()
check("mufasa-preview registers a --pose-scale argument",
      '"--pose-scale"' in PV and "args.pose_scale" in PV)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ia_pose_scale: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
