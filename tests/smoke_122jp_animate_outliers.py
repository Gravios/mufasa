"""Smoke test for patch 122jp — remove outliers from the animation data.

User request: "I would also like an option to remove outliers — from the
animation data".

mufasa-landscape-animate --outliers none | percentile | mad leaves out, before
the points are sampled, frames that are outliers in any animated feature
(--outlier-percentile, default 1 % each tail; --outlier-mad, default 5
scaled MADs; --outliers-tsne also tests the t-SNE coordinates). The landscape
run itself is untouched. The Animate… popup has "Remove outliers".
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
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


app = QApplication.instance() or QApplication([])

from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = ProjectConfigCreator(
    project_path=str(tmp), project_name="ls", target_list=["rear"],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"]).config_path
P = project_paths_from_config(cfg)
vd, fd = Path(P["video_dir"]), Path(P["derived_features_dir"])
fam = fd / "frame_by_frame_body_part_movements_mm"
for d in (vd, fam):
    d.mkdir(parents=True, exist_ok=True)
A = "Animal_1 movement a (mm)_FEATURE_SUBSET"
B = "Animal_1 movement b (mm)_FEATURE_SUBSET"
C = "Animal_1 movement c (mm)_FEATURE_SUBSET"
ANG = "Animal_1 heading angle (degrees)_FEATURE_SUBSET"
OTHER = "Animal_1 unselected (mm)_FEATURE_SUBSET"
N = 1500
r = np.random.default_rng(3)
for v in ("v1", "v2"):
    w = cv2.VideoWriter(str(vd / f"{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(5):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    g = r.integers(0, 2, N)
    pd.DataFrame({
        A: r.normal(g * 4, 0.6),
        B: r.normal(g * -3, 0.7),
        C: r.normal(0, 1, N),
        ANG: r.uniform(0, 360, N),
        OTHER: r.normal(0, 1, N),
    }).to_parquet(fam / f"{v}.parquet")
# a few wild values in B
for v in ("v1", "v2"):
    df = pd.read_parquet(fam / f"{v}.parquet")
    df.loc[::100, B] = 1e4
    df.to_parquet(fam / f"{v}.parquet")

from mufasa.tools import landscape_animate as la  # noqa: E402

# ---------------------------------------------------------------- outlier_mask
rng = np.random.default_rng(0)
x = rng.normal(0, 1, 10000)
x[:20] = 100
y = rng.normal(0, 1, 10000)
m, per = la.outlier_mask({"x": x, "y": y}, "mad", mad=5)
check("mad: the wild values flagged, almost nothing else",
      m[:20].all() and per["x"] == 20 and per["y"] <= 3)
m, per = la.outlier_mask({"x": x}, "percentile", percentile=1.0)
check("percentile: about 1 % per tail", 180 <= per["x"] <= 220 and m[:20].all())
m, per = la.outlier_mask({"x": x, "y": y}, "none")
check("none flags nothing", not m.any())
z = np.r_[np.zeros(800), rng.normal(0, 1, 200), [500.0]]
m, _ = la.outlier_mask({"z": z}, "mad", mad=5)
check("mad with mostly identical values falls back to the SD", m[-1] and m.sum() < 10)
m, _ = la.outlier_mask({"x": np.r_[x[:100], np.nan]}, "mad")
check("NaN is not an outlier", not m[-1])

# ---------------------------------------------------------------- CLI
from mufasa.tools import feature_landscape as fl  # noqa: E402

out = tmp / "run"
cf = tmp / "c.txt"
cf.write_text(f"{A}\n{B}\n{C}\n")
with contextlib.redirect_stdout(io.StringIO()):
    fl.main(["--project", cfg, "--sources", "features", "--columns-file", str(cf),
             "--out", str(out), "--tsne-sample", "800", "--frames-per-session", "1000",
             "--n-jobs", "1", "--perplexity", "20"])
smp = pd.read_parquet(out / "tsne_samples.parquet")
n_wild = int((smp[B] > 1e3).sum())
got = {}
orig = la.feature_space


def spy(a, b, na, nb):  # noqa: ANN001
    got["v"] = (a.copy(), b.copy())
    return orig(a, b, na, nb)


la.feature_space = spy


def run(*extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        la.main([str(out), "--pair", A, B, "--size", "100", "--hold-frames", "1",
                 "--transition-frames", "2", "--intro-frames", "0", "--n-points", "5000",
                 "--out", str(out / "x.gif"), *extra])
    return buf.getvalue(), got["v"]


log0, (a0, b0) = run()
check("without --outliers the wild frames are animated",
      n_wild > 0 and (b0 > 1e3).sum() == n_wild and "outliers" not in log0)
log1, (a1, b1) = run("--outliers", "mad")
check("--outliers mad removes them", (b1 > 1e3).sum() == 0 and len(b1) <= len(b0) - n_wild)
check("…and says how many, per feature",
      f"left out as outliers (mad)" in log1 and "frame(s)" in log1)
log2, (a2, b2) = run("--outliers", "percentile", "--outlier-percentile", "2")
check("--outliers percentile 2 cuts ~4 % of frames",
      0.02 * len(b0) <= len(b0) - len(b2) <= 0.12 * len(b0) and (b2 > 1e3).sum() == 0)
log3, (a3, b3) = run("--outliers", "mad", "--outlier-mad", "3", "--outliers-tsne")
check("--outliers-tsne also tests the map", "t-SNE" in log3 and len(b3) <= len(b1))
check("the landscape run is untouched",
      len(pd.read_parquet(out / "tsne_samples.parquet")) == len(smp))
la.feature_space = orig

# ---------------------------------------------------------------- popup and argv
from mufasa.ui_qt import landscape_runner as lr  # noqa: E402

d = lr.FeatureNamesDialog(None, cfg, [A, B], "animate")
o = d.options()
check("popup: outliers off by default",
      o["outliers"] == "none" and o["outlier_value"] is None
      and not d.outlier_value.isEnabled())
d.outliers.setCurrentText("mad")
check("popup: mad → 5 × MAD", d.outlier_value.value() == 5.0
      and d.outlier_value.suffix().strip() == "× MAD" and d.outlier_value.isEnabled())
d.outliers.setCurrentText("percentile")
d.outlier_value.setValue(0.5)
d.outliers_tsne.setChecked(True)
o = d.options()
argv = lr.animate_argv(out, [(A, B)], out / "x.gif", None, theme="dark", mp4=False,
                       color="tsne", outliers=o["outliers"],
                       outlier_value=o["outlier_value"], outliers_tsne=o["outliers_tsne"])
val = lambda f: argv[argv.index(f) + 1]  # noqa: E731
check("argv carries method, value and t-SNE flag",
      val("--outliers") == "percentile" and val("--outlier-percentile") == "0.5"
      and "--outliers-tsne" in argv)
check("argv without outliers unchanged",
      "--outliers" not in lr.animate_argv(out, [(A, B)], out / "x.gif", None, theme="dark",
                                          mp4=False, color="tsne"))

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jp_animate_outliers: {len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
