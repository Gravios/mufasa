"""Smoke test for patch 122iz — head velocity in the body frame; landscape
column handling.

User report: mufasa-landscape "only used the angles". All feature families
were loaded, but the angle columns crowded every plot: angles wrap at
0/360°, so an angle hovering near 0° is split into lumps near 0 and 360 and
looks bimodal; and the feature writer stores missing values as -1, a second
spurious mode. Now:

* -1 placeholders in non-negative columns are read as missing;
* angle columns are rotated so the wrap falls at their emptiest direction;
* the log lists, per feature family, how many columns were loaded and kept.

New feature family HEAD VELOCITY IN BODY FRAME (MM/S): the head point's
velocity split into forward (along body → head) and lateral (to the
animal's left) components, in mm/s. Head / body markers come from
project.toml [features], else [[pose.segments]] (segment "head", root
segment), else marker names. Its parquet keeps NaN (no -1: values are
signed).

Runs for real: kernel geometry on constructed motion, marker resolution,
a FeatureSubsetsCalculator run on a project with [[pose.segments]], the
Compute Features form's family list, and the landscape cleaning on data
with planted wrap-around angles and -1 placeholders.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


import mufasa.feature_extractors.feature_subsets as FS  # noqa: E402
from mufasa.feature_extractors.feature_subset_kernels import (  # noqa: E402
    compute_head_velocity_body_frame as K,
    resolve_head_body_markers as R,
)
from mufasa.mixins.config_reader import ConfigReader  # noqa: E402
from mufasa.project_layout import (  # noqa: E402
    project_paths_from_config,
    read_project_toml,
    write_project_toml,
)
from mufasa.tools import feature_landscape as FL  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

FWD = "Animal_1 head velocity forward along body-head axis (mm/s)"
LAT = "Animal_1 head velocity lateral to body-head axis, left positive (mm/s)"

# ---- kernel geometry ---------------------------------------------------------
T = 100
t = np.arange(T, dtype=float)
base = {"b1_x": np.full(T, 100.), "b1_y": np.full(T, 98.), "b2_x": np.full(T, 100.),
        "b2_y": np.full(T, 102.)}
hb = {"A": (["h1", "h2"], ["b1", "b2"], "test")}
# facing +x, head moving +x at 3 px/frame; 2 px/mm, 30 fps → 45 mm/s forward
df = pd.DataFrame({**base, "h1_x": 120 + 3 * t, "h1_y": np.full(T, 99.),
                   "h2_x": 120 + 3 * t, "h2_y": np.full(T, 101.)})
o = K(df, hb, px_per_mm=2.0, fps=30.0)
f, l = o["A head velocity forward along body-head axis (mm/s)"], \
    o["A head velocity lateral to body-head axis, left positive (mm/s)"]
check("moving the way the head points → forward = speed (45 mm/s), lateral 0",
      np.allclose(f, 45) and np.allclose(l, 0, atol=1e-9))
df = pd.DataFrame({**base, "h1_x": np.full(T, 120.), "h1_y": 99 - 3 * t,
                   "h2_x": np.full(T, 120.), "h2_y": 101 - 3 * t})
o = K(df, hb, px_per_mm=2.0, fps=30.0)
f, l = list(o.values())
check("facing +x, head moving up the image (the animal's left) → lateral > 0 "
      "at the start, = 45 mm/s", abs(l[0] - 45) < 1e-9 and abs(f[0]) < 1e-9)
check("forward² + lateral² = speed² everywhere", np.allclose(f ** 2 + l ** 2, 45 ** 2))
rot = 0.7
c, s_ = np.cos(rot), np.sin(rot)
df = pd.DataFrame({**base, "h1_x": 120 + 3 * t, "h1_y": np.full(T, 99.),
                   "h2_x": 120 + 3 * t, "h2_y": np.full(T, 101.)})
dfr = df.copy()
for m in ("b1", "b2", "h1", "h2"):
    x, y = df[f"{m}_x"].to_numpy(), df[f"{m}_y"].to_numpy()
    dfr[f"{m}_x"], dfr[f"{m}_y"] = c * x - s_ * y, s_ * x + c * y
o1, o2 = K(df, hb, 2.0, 30.0), K(dfr, hb, 2.0, 30.0)
check("invariant to rotating the whole scene (it is a body-frame quantity)",
      all(np.allclose(o1[k], o2[k]) for k in o1))
dfn = df.copy()
dfn.loc[10, "h1_x"] = np.nan
dfn.loc[20:21, ["h1_x", "h2_x"]] = np.nan
o = K(dfn, hb, 2.0, 30.0)
f = list(o.values())[0]
check("a missing head marker falls back to the others; a frame with no head is NaN",
      np.isfinite(f[10]) and np.isnan(f[20]) and np.isfinite(f[50]))

# ---- marker resolution --------------------------------------------------------
bps = {"Animal_1": ["nose", "ear_l", "ear_r", "center", "tail"]}
check("marker names: nose/ears = head, center = body",
      R({}, bps)["Animal_1"][:2] == (["nose", "ear_l", "ear_r"], ["center"]))
segs = {"pose": {"segments": [
    {"name": "back", "markers": ["b1", "b2"]},
    {"name": "neck", "parent": "back", "markers": ["n"]},
    {"name": "head", "parent": "neck", "markers": ["hf", "hb"]}]}}
bps2 = {"Animal_1": ["b1", "b2", "n", "hf", "hb"]}
check("[[pose.segments]]: 'head' segment vs the root segment",
      R(segs, bps2)["Animal_1"] == (["hf", "hb"], ["b1", "b2"], "[[pose.segments]]"))
check("[features] head_segment / body_segment pick other segments",
      R({**segs, "features": {"head_segment": "neck", "body_segment": "head"}},
        bps2)["Animal_1"][:2] == (["n"], ["hf", "hb"]))
check("[features] head_markers / body_markers win over everything",
      R({**segs, "features": {"head_markers": ["hf"], "body_markers": ["n"]}},
        bps2)["Animal_1"][:2] == (["hf"], ["n"]))
for bad, why in [({"Animal_1": ["a", "b", "c"]}, "no recognisable names"),
                 ({"Animal_1": ["x"]}, "one marker")]:
    try:
        R({}, bad)
        ok = False
    except ValueError as exc:
        ok = "head_markers" in str(exc)
    check(f"unresolvable markers ({why}) → error saying what to set", ok)

# ---- a real run on a project with [[pose.segments]] ---------------------------
tmp = Path(tempfile.mkdtemp())
names = ["b1", "b2", "n", "hf", "hb", "tail"]
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="hv", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=names).config_path
data = read_project_toml(Path(cfg))
data["pose"]["segments"] = [
    {"name": "back", "markers": ["b1", "b2"]},
    {"name": "neck", "parent": "back", "markers": ["n"]},
    {"name": "head", "parent": "neck", "markers": ["hf", "hb"]},
    {"name": "tail_1", "parent": "back", "markers": ["tail"]}]
write_project_toml(Path(cfg), data)
P = project_paths_from_config(cfg)
vd = Path(P["video_dir"])
vd.mkdir(parents=True, exist_ok=True)
w = cv2.VideoWriter(str(vd / "v1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
for _ in range(3):
    w.write(np.zeros((360, 640, 3), np.uint8))
w.release()
pd.DataFrame({"Video": ["v1"], "fps": [30], "Resolution_width": [640],
              "Resolution_height": [360], "Distance_in_mm": [0],
              "pixels/mm": [2.0]}).to_csv(P["video_info_path"], index=False)
T = 300
t = np.arange(T, dtype=float)
pose = {}
for m, (dx, dy) in {"b1": (0, -2), "b2": (0, 2), "n": (10, 0), "hf": (24, 0),
                    "hb": (16, 0), "tail": (-30, 0)}.items():
    pose[f"{m}_x"] = 100 + 1.5 * t + dx      # whole animal walks +x
    pose[f"{m}_y"] = 180 + dy + 0 * t
    pose[f"{m}_p"] = 0.99
pose["hf_x"][150] = np.nan                   # one missing marker
cr = ConfigReader(config_path=cfg, read_video_info=False)
os.makedirs(cr.outlier_corrected_dir, exist_ok=True)
pd.DataFrame(pose).to_parquet(os.path.join(cr.outlier_corrected_dir, "v1.parquet"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    FS.FeatureSubsetsCalculator(
        config_path=cfg, feature_families=[FS.HEAD_VELOCITY_BODY_FRAME],
        file_checks=True, save_dir=None, data_dir=None,
        derived_features_dir=P["derived_features_dir"], n_workers=1,
        overwrite_existing=False).run()
check("the run reports which markers it used and where from",
      "Head velocity (Animal_1, from [[pose.segments]]): head = ['hf', 'hb']; "
      "body = ['b1', 'b2']" in buf.getvalue())
out = Path(P["derived_features_dir"]) / "head_velocity_in_body_frame_mm_s" / "v1.parquet"
check("written to derived/features/head_velocity_in_body_frame_mm_s/<video>.parquet",
      out.is_file())
hv = pd.read_parquet(out)
check("columns: forward and lateral (with the legacy _FEATURE_SUBSET suffix)",
      sorted(hv.columns) == sorted([FWD + "_FEATURE_SUBSET", LAT + "_FEATURE_SUBSET"]))
fw, la = hv[FWD + "_FEATURE_SUBSET"].to_numpy(), hv[LAT + "_FEATURE_SUBSET"].to_numpy()
check("walking straight at 1.5 px/frame, 2 px/mm, 30 fps → forward 22.5 mm/s, lateral 0",
      np.allclose(fw[5:140], 22.5) and np.allclose(la[5:140], 0, atol=1e-6))
check("the signed family never writes the -1 placeholder",
      not (fw == -1).any() and np.isfinite(fw).all())
check("the family is offered by Compute Features (FEATURE_FAMILIES)",
      FS.HEAD_VELOCITY_BODY_FRAME in FS.FEATURE_FAMILIES)
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from mufasa.ui_qt.forms.features import FeatureSubsetExtractorForm  # noqa: E402

check("…and listed in the form", FS.HEAD_VELOCITY_BODY_FRAME
      in FeatureSubsetExtractorForm._discover_families(None))

# ---- landscape column cleaning ------------------------------------------------
r = np.random.default_rng(2)
n = 20000
ang_true = np.mod(r.normal(0, 15, n), 360)           # hovers around 0° → wraps
dist = r.gamma(4, 5, n)
dist[r.random(n) < 0.1] = -1                         # writer's missing placeholder
X = pd.DataFrame({"Angle (degrees) a-b-c_FEATURE_SUBSET": ang_true,
                  "Animal_1 dist a b (mm)_FEATURE_SUBSET": dist,
                  "Animal_1 head velocity lateral (mm/s)": r.normal(0, 30, n)})
X.loc[:4, "Animal_1 head velocity lateral (mm/s)"] = -1.0
dp = dict(alpha=0.01, min_n=100, min_dip=0.01, max_valley=0.9)
raw_modes = len(FL.dipsplit(X.iloc[:, 0].to_numpy(), **dp))
log = io.StringIO()
with contextlib.redirect_stdout(log):
    Y = FL.clean_columns(X)
a = Y.iloc[:, 0].to_numpy()
check(f"an angle wrapping at 0/360° looks bimodal raw ({raw_modes} modes) …",
      raw_modes == 2)
check("… and unimodal after rotation, values still in degrees and contiguous",
      len(FL.dipsplit(a, **dp)) == 1 and np.ptp(a) < 180)
d = Y.iloc[:, 1].to_numpy()
check("-1 placeholders in a non-negative column become NaN",
      np.isnan(d).sum() == (dist == -1).sum() and np.nanmin(d) > 0)
check("-1 in a signed column is left alone",
      (Y.iloc[:5, 2] == -1).all())
check("both cleanings are reported",
      "-1 placeholders read as missing in 1" in log.getvalue()
      and "1 angle column(s) rotated" in log.getvalue())

# ---- per-family log on a real feature tree ------------------------------------
fam = Path(P["derived_features_dir"]) / "within_animal_three_point_body_part_angles_degrees"
fam.mkdir()
pd.DataFrame({"Angle (degrees) b1-n-hf_FEATURE_SUBSET": np.mod(r.normal(0, 10, T), 360)}
             ).to_parquet(fam / "v1.parquet")
fam2 = Path(P["derived_features_dir"]) / "two_point_body_part_distances_mm"
fam2.mkdir()
pd.DataFrame({"Animal_1 distance (mm) b1 tail_FEATURE_SUBSET": r.gamma(4, 5, T)}
             ).to_parquet(fam2 / "v1.parquet")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    FL.main(["--project", str(cfg), "--sources", "features", "--no-tsne",
             "--n-jobs", "1", "--min-unique", "5"])
txt = buf.getvalue()
check("landscape logs columns per feature family (kept / loaded)",
      "columns screened per family (kept / loaded)" in txt
      and "head_velocity_in_body_frame_mm_s" in txt
      and "within_animal_three_point_body_part_angles_degrees" in txt)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122iz_head_velocity_and_landscape_columns: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
