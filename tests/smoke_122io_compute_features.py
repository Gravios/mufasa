"""Smoke test for patch 122io — Compute Features runs end to end.

User report: Compute Features → "preflight failed … ModuleNotFoundError: No
module named 'shap'". Three bugs on that path:

1. ``train_model_mixin`` imported ``shap`` at module level; shap was dropped
   from the dependencies, and feature computation imports the mixin. It is
   now imported only inside the SHAP helper that uses it.
2. ``perimeter_jit`` imported ``numba.np.extensions.cross2d``, removed in
   recent numba (0.68) — the convex-hull families failed to import.
3. ROI tables carried ``Shape_type`` "Rectangle" / "Circle" / "Polygon" while
   the inside-ROI kernel (and ROI-classifier code) compares to SimBA's
   lower-case values, so "body-parts inside ROIs" silently wrote nothing for
   ROIs drawn in Mufasa.

Runs for real: a generated project with ROIs in rois.json and synthetic
tracking, FeatureSubsetsCalculator preflight + run with every family, and
the outputs read back — all without shap installed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)


class _NoShap:
    """Make ``import shap`` fail as on the user's machine, even where it is
    installed."""

    def find_spec(self, name, path=None, target=None):
        if name == "shap" or name.startswith("shap."):
            raise ModuleNotFoundError("No module named 'shap'", name="shap")
        return None


sys.meta_path.insert(0, _NoShap())

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


try:
    import mufasa.feature_extractors.feature_subsets as FS  # noqa: E402
    imported = True
except ModuleNotFoundError as exc:
    imported = False
    print("import failed:", exc)
check("feature computation imports without shap installed", imported)
if not imported:
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"smoke_122io_compute_features: 0/{len(checks)} checks passed")
    sys.exit(1)

from mufasa.feature_extractors.perimeter_jit import jitted_hull  # noqa: E402
from mufasa.mixins.config_reader import ConfigReader  # noqa: E402
from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.roi_tools.roi_logic import ROILogic  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

# convex hull without numba's cross2d, checked against scipy
from scipy.spatial import ConvexHull  # noqa: E402

pts = np.random.default_rng(1).uniform(0, 100, (40, 8, 2)).astype(np.float32)
area, per = jitted_hull(pts, "area"), jitted_hull(pts, "perimeter")
check("convex hull area / perimeter match scipy",
      np.allclose(area, [ConvexHull(p).volume for p in pts], atol=1e-3)
      and np.allclose(per, [ConvexHull(p).area for p in pts], atol=1e-3))

tmp = Path(tempfile.mkdtemp())
bps = ["nose", "ear_l", "ear_r", "center", "tail"]
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="f", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=bps,
).config_path
P = project_paths_from_config(cfg)
vd = Path(P["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
w = cv2.VideoWriter(str(vd / "v1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                    30, (640, 360))
for _ in range(3):
    w.write(np.zeros((360, 640, 3), np.uint8))
w.release()
pd.DataFrame({"Video": ["v1"], "fps": [30], "Resolution_width": [640],
              "Resolution_height": [360], "Distance_in_mm": [0],
              "pixels/mm": [2.0]}).to_csv(P["video_info_path"], index=False)
S = dict(color_name="Red", bgr=(0, 0, 255), thickness=2, ear_tag_size=8)
L = ROILogic(config_path=cfg, video_path=str(vd / "v1.mp4"))
L.add_rectangle(name="left", top_left=(0, 0), bottom_right=(320, 360), **S)
L.add_circle(name="cup", center=(480, 180), radius=40, **S)
L.add_polygon(name="ring", vertices=[
    (int(480 + 60 * np.cos(t)), int(180 + 60 * np.sin(t)))
    for t in np.linspace(0, 2 * np.pi, 30, endpoint=False)], **S)
L.save()

T = 200
cx = np.linspace(100, 500, T)
cols = {}
for i, b in enumerate(bps):
    cols[f"{b}_x"] = cx + i * 5
    cols[f"{b}_y"] = 180 + ((-1) ** i) * 4
    cols[f"{b}_p"] = 0.99
cr = ConfigReader(config_path=cfg, read_video_info=False)
os.makedirs(cr.outlier_corrected_dir, exist_ok=True)
pd.DataFrame(cols).to_parquet(os.path.join(cr.outlier_corrected_dir, "v1.parquet"))

calc = FS.FeatureSubsetsCalculator(
    config_path=cfg, feature_families=list(FS.FEATURE_FAMILIES),
    file_checks=True, save_dir=None, data_dir=None,
    derived_features_dir=P["derived_features_dir"], n_workers=1,
    overwrite_existing=False)
check("preflight runs (the step that failed in the report)",
      isinstance(calc.preflight_check(), dict))
calc.run()
out = {p.parent.name: p for p in Path(P["derived_features_dir"]).rglob("*.parquet")}
check(f"every feature family is written ({len(out)}/{len(FS.FEATURE_FAMILIES)})",
      len(out) == len(FS.FEATURE_FAMILIES))
inside = pd.read_parquet(next(p for n, p in out.items() if "inside_rois" in n))
col = next(c for c in inside.columns if "nose inside rectangle left" in c)
truth = (cx >= 0) & (cx <= 320)
check("inside-ROI booleans are present and correct for a rectangle",
      np.array_equal(inside[col].to_numpy().astype(bool), truth))
pcol = next(c for c in inside.columns if "nose inside polygon ring" in c)
ccol = next(c for c in inside.columns if "nose inside circle cup" in c)
check("…and for a circle and a 30-vertex polygon",
      np.array_equal(inside[ccol].to_numpy().astype(bool), np.abs(cx - 480) <= 40)
      and inside[pcol].to_numpy().astype(bool)[np.abs(cx - 480) < 50].all()
      and not inside[pcol].to_numpy().astype(bool)[np.abs(cx - 480) > 61].any())
hull = pd.read_parquet(next(p for n, p in out.items() if n == "entire_animal_convex_hull_area_mm2"))
check("convex-hull families produce finite values",
      np.isfinite(hull.to_numpy(dtype=float)).all())
check("shap was never imported", "shap" not in sys.modules)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122io_compute_features: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
