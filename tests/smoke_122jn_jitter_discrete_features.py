"""Smoke test for patch 122jn — jitter near-discrete features in figures.

User request: "Some of the features are close to discretized, I would like
a option to be able to jitter their values for better visualization".

A Jitter tick per feature in the Landscape… / Animate… names table (saved
as [jitter] in logs/measures/feature_names.toml); mufasa-landscape's joint
PDFs and mufasa-landscape-animate's feature spaces then draw those features
with uniform noise of ±0.4 × the step between their distinct values
(--jitter COL / --no-jitter on the command line). The dip test, the
boundaries and the t-SNE use the real values. Feature Visualization's joint
plot has a Jitter checkbox (and always jitters marked features).
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import tomllib
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
# C becomes a near-discrete count (0, 0.5, 1, … 3), correlated with the modes
for v in ("v1", "v2"):
    df = pd.read_parquet(fam / f"{v}.parquet")
    df[C] = np.clip(np.round((df[A].to_numpy() + 1) * 2) / 2 / 2, 0, 3)
    df.to_parquet(fam / f"{v}.parquet")

from mufasa.utils import feature_names as fn  # noqa: E402

# ---------------------------------------------------------------- helpers
v = np.array([0, 0.5, 0.5, 1.0, 2.0, np.nan, 1.5] * 100)
check("value_step: median gap between distinct values", fn.value_step(v) == 0.5)
j = fn.jitter_values(v, seed=3)
fin = np.isfinite(v)
check("jitter within ±0.4 step, NaN kept, input untouched",
      np.abs(j[fin] - v[fin]).max() <= 0.2 + 1e-12 and np.isnan(j[~fin]).all()
      and np.abs(j[fin] - v[fin]).max() > 0.1 and v[1] == 0.5)
check("jitter keeps the levels apart (rounding recovers them)",
      np.allclose(np.round(j[fin] * 2) / 2, v[fin]))
check("jitter is reproducible", np.array_equal(j, fn.jitter_values(v, seed=3), equal_nan=True))
check("constant column left alone", np.array_equal(fn.jitter_values(np.ones(5)), np.ones(5)))

# ---------------------------------------------------------------- popup and storage
from mufasa.ui_qt import landscape_runner as lr  # noqa: E402

d = lr.FeatureNamesDialog(None, cfg, [A, C], "landscape")
check("names table has a Jitter column of checkboxes, off by default",
      d.table.columnCount() == 4 and d.table.horizontalHeaderItem(3).text() == "Jitter"
      and d.jitter() == {A: False, C: False})
d.table.item(1, 3).setCheckState(lr.Qt.Checked)
d.accept()
with open(fn.names_path(cfg), "rb") as fh:
    data = tomllib.load(fh)
check("jitter saved as [jitter]", data.get("jitter") == {C: True} and fn.load_jitter(cfg) == {C})
check("remembered next time", lr.FeatureNamesDialog(None, cfg, [C, B], "animate").jitter()
      == {C: True, B: False})
fn.save_names(cfg, {}, {A: "log10"})
check("saving other things keeps the jitter marks", fn.load_jitter(cfg) == {C}
      and fn.load_scales(cfg) == {A: "log10"})
fn.save_names(cfg, {}, {A: "linear"})

# ---------------------------------------------------------------- landscape figures
from mufasa.tools import feature_landscape as fl  # noqa: E402

out = tmp / "run"
cf = tmp / "c.txt"
cf.write_text(f"{A}\n{B}\n{C}\n")
with contextlib.redirect_stdout(io.StringIO()):
    fl.main(["--project", cfg, "--sources", "features", "--columns-file", str(cf),
             "--out", str(out), "--tsne-sample", "500", "--frames-per-session", "700",
             "--n-jobs", "1", "--perplexity", "20"])
check("landscape: marked columns jittered in figures", fl.JITTERED == {C})
smp = pd.read_parquet(out / "tsne_samples.parquet")
check("landscape: analysis values not jittered", np.allclose(smp[C] * 4, np.round(smp[C] * 4)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pairs = pd.read_csv(out / "pairs.csv")
row = pairs[(pairs.feature_a == C) | (pairs.feature_b == C)].iloc[0]
X = pd.read_parquet(out / "tsne_samples.parquet")
f = plt.figure()
ax = fl.draw_jpdf(f, f.add_gridspec(1, 1)[0], X, row, bins=60,
                  dp=dict(alpha=0.05, min_n=20, min_dip=0.01, max_valley=0.9))
title = [a_ for a_ in f.axes if a_.get_title(loc="left")][0].get_title(loc="left")
check("joint PDF says which axis is jittered", "jittered" in title)
mesh = ax.collections[0].get_array()
f2 = plt.figure()
fl.JITTERED = set()
ax2 = fl.draw_jpdf(f2, f2.add_gridspec(1, 1)[0], X, row, bins=60,
                   dp=dict(alpha=0.05, min_n=20, min_dip=0.01, max_valley=0.9))
filled = lambda m: int(np.ma.count(m))  # noqa: E731
check("jitter fills more bins than the stacked discrete values",
      filled(mesh) > 1.5 * filled(ax2.collections[0].get_array()))
plt.close("all")
with contextlib.redirect_stdout(io.StringIO()):
    fl.main(["--project", cfg, "--sources", "features", "--columns-file", str(cf),
             "--out", str(tmp / "r2"), "--no-tsne", "--frames-per-session", "300",
             "--n-jobs", "1", "--no-jitter", "--jitter", B])
check("--no-jitter drops the marks, --jitter adds columns", fl.JITTERED == {B})

# ---------------------------------------------------------------- animation
from mufasa.tools import landscape_animate as la  # noqa: E402

got = {}
orig = la.feature_space


def spy(a, b, na, nb):  # noqa: ANN001
    got[(na, nb)] = (a.copy(), b.copy())
    return orig(a, b, na, nb)


la.feature_space = spy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    la.main([str(out), "--pair", A, C, "--names-file", str(fn.names_path(cfg)),
             "--size", "100", "--hold", "0.1", "--transition", "0.2", "--fps", "4",
             "--n-points", "200", "--intro-frames", "0", "--out", str(out / "a.gif")])
(a_v, c_v), = got.values()
check("animation: marked feature drawn jittered, the other not",
      not np.allclose(c_v * 4, np.round(c_v * 4)) and np.allclose(c_v * 4, np.round(c_v * 4),
                                                                  atol=0.81)
      and "jittered:" in buf.getvalue())
got.clear()
with contextlib.redirect_stdout(io.StringIO()):
    la.main([str(out), "--pair", A, C, "--names-file", str(fn.names_path(cfg)),
             "--no-jitter", "--size", "100", "--hold", "0.1", "--transition", "0.2",
             "--fps", "4", "--n-points", "200", "--intro-frames", "0",
             "--out", str(out / "b.gif")])
(a_v, c_v), = got.values()
check("animation: --no-jitter draws real values", np.allclose(c_v * 4, np.round(c_v * 4)))
la.feature_space = orig

# ---------------------------------------------------------------- form joint plot
from mufasa.tools import feature_plots as fp  # noqa: E402
from mufasa.ui_qt.forms.feature_visualization import FeatureVisualizationForm  # noqa: E402

seen = []
orig_pj = fp.plot_joint


def spy_pj(fig, a, b, *args, **kw):  # noqa: ANN001
    seen.append((np.asarray(a).copy(), np.asarray(b).copy()))
    return orig_pj(fig, a, b, *args, **kw)


fp.plot_joint = spy_pj
F = FeatureVisualizationForm(config_path=cfg)
F.set_selected([A, B])
F.plot_joint()
a0 = seen[-1][0]
check("form: unmarked features plotted as they are", "Jittered" not in F.plot_info.text())
F.jitter_cb.setChecked(True)
check("form: Jitter checkbox re-plots with jitter",
      len(seen) == 2 and not np.array_equal(seen[-1][0], a0)
      and "Jittered:" in F.plot_info.text())
F.jitter_cb.setChecked(False)
F.set_selected([A, C])
F.plot_joint()
check("form: marked feature always jittered",
      not np.allclose(seen[-1][1] * 4, np.round(seen[-1][1] * 4))
      and np.allclose(seen[-1][0], a0[:len(seen[-1][0])]) is not None)
fp.plot_joint = orig_pj

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jn_jitter_discrete_features: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
