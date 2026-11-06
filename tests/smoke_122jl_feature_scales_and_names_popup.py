"""Smoke test for patch 122jl — the names popup opens from the buttons, and
a per-feature linear / log10 scale.

User report / request: "the naming window didn't show up, also in the naming
window i would like an option for each feature to be either linear or
log10, before generating the plots."

Cause of the first: the buttons' clicked(bool) signal was connected straight
to generate_landscape(ask=True), so clicked's False landed in ask and
the popup was skipped. Now the buttons call it with no arguments.

The popup's table gets a Scale column (linear / log10 per feature, plus
"set all"); scales are saved next to the names ([scales] in
logs/measures/feature_names.toml). mufasa-landscape analyses log10
features as log10 (values ≤ 0 → missing), labels them "log10 <name>",
records them in the run's log10_columns.txt; mufasa-landscape-animate
labels them the same; a run made with other scales is not reused.
"""
from __future__ import annotations

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
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

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
# make A positive (log-normal) with a few zeros
for v in ("v1", "v2"):
    df = pd.read_parquet(fam / f"{v}.parquet")
    df[A] = np.exp(df[A].to_numpy())
    df.loc[:9, A] = 0.0
    df.to_parquet(fam / f"{v}.parquet")

from mufasa.ui_qt import landscape_runner as lr  # noqa: E402
from mufasa.ui_qt.forms.feature_visualization import FeatureVisualizationForm  # noqa: E402
from mufasa.utils import feature_names as fn  # noqa: E402

# ---------------------------------------------------------------- popup from the buttons
shown = []


def fake_exec(self):  # noqa: ANN001
    shown.append((self.mode, list(self.cols)))
    return QDialog.Rejected


orig_exec = lr.FeatureNamesDialog.exec
lr.FeatureNamesDialog.exec = fake_exec
F = FeatureVisualizationForm(config_path=cfg)
F.set_selected([A, B])
F.landscape_btn.click()
F.animate_btn.click()
check("Landscape… button opens the names popup", ("landscape", [A, B]) in shown)
check("Animate… button opens the names popup", ("animate", [A, B]) in shown)
check("cancelling the popup starts nothing",
      not lr.landscape_root(cfg).exists() or not any(lr.landscape_root(cfg).iterdir()))
lr.FeatureNamesDialog.exec = orig_exec

# ---------------------------------------------------------------- scale column
d = lr.FeatureNamesDialog(None, cfg, [A, B, C], "landscape")
check("table has a Scale column",
      d.table.columnCount() == 3 and d.table.horizontalHeaderItem(2).text() == "Scale")
check("scales default to linear",
      [cb.currentText() for cb in d.scale_cbs] == ["linear"] * 3
      and [d.scale_cbs[0].itemText(i) for i in range(2)] == ["linear", "log10"])
d.scale_cbs[0].setCurrentText("log10")
d.table.item(0, 1).setText("speed a")
d.accept()
with open(fn.names_path(cfg), "rb") as fh:
    data = tomllib.load(fh)
check("scale saved next to the names",
      data.get("scales") == {A: "log10"} and data["names"] == {A: "speed a"})
d2 = lr.FeatureNamesDialog(None, cfg, [A, B], "animate")
check("scale remembered next time",
      d2.scale_cbs[0].currentText() == "log10" and d2.scale_cbs[1].currentText() == "linear")
d2.reject()
fn.save_names(cfg, {}, {B: "log10"})
fn.save_names(cfg, {}, {B: "linear"})
check("linear forgets the entry; names untouched",
      fn.load_scales(cfg) == {A: "log10"} and fn.load_names(cfg) == {A: "speed a"})
try:
    fn.save_names(cfg, {}, {A: "ln"})
    bad = False
except ValueError:
    bad = True
check("unknown scale rejected", bad)
check("display prefixes log10",
      fn.display(A, fn.load_names(cfg), fn.load_scales(cfg)) == "log10 speed a"
      and fn.display(B, {}, fn.load_scales(cfg)) == fn.default_name(B))
x = pd.DataFrame({A: [100.0, 0.0, -1.0, np.nan], B: [1.0, 2.0, 3.0, 4.0]})
y, done = fn.apply_scales(x, {A: "log10"}, log=lambda *_: None)
check("apply_scales: log10, ≤ 0 → NaN, others untouched, input not modified",
      done == [A] and y[A].iloc[0] == 2.0 and y[A].iloc[1:].isna().all()
      and (y[B] == x[B]).all() and x[A].iloc[0] == 100.0)

# ---------------------------------------------------------------- landscape uses the scale
from mufasa.tools import feature_landscape as fl  # noqa: E402

out = tmp / "run_log"
cf = tmp / "cols.txt"
cf.write_text(f"{A}\n{B}\n{C}\n")
fl.main(["--project", cfg, "--sources", "features", "--columns-file", str(cf),
         "--out", str(out), "--tsne-sample", "300", "--frames-per-session", "500",
         "--n-jobs", "1", "--perplexity", "15"])
smp = pd.read_parquet(out / "tsne_samples.parquet")
raw = pd.concat([pd.read_parquet(fam / f"{v}.parquet") for v in ("v1", "v2")])
check("landscape: log10 column analysed as log10",
      np.nanmax(smp[A]) < np.log10(raw[A].max()) + 1e-6
      and np.nanmin(smp[A]) < 0 < np.nanmax(smp[A]) and np.nanmax(smp[A]) < 3)
check("landscape: linear columns unchanged",
      np.allclose(sorted(smp[B].dropna())[:3], sorted(smp[B].dropna())[:3])
      and np.nanmin(smp[B]) < -3)
check("landscape: log10_columns.txt records it",
      (out / "log10_columns.txt").read_text().split("\n")[0] == A)
check("landscape: label 'log10 speed a'", fl.display(A, fl.DISPLAY) == "log10 speed a")
out2 = tmp / "run_lin"
fl.main(["--project", cfg, "--sources", "features", "--columns-file", str(cf),
         "--out", str(out2), "--tsne-sample", "300", "--frames-per-session", "500",
         "--n-jobs", "1", "--perplexity", "15", "--no-tsne", "--no-scales"])
check("--no-scales analyses everything linear",
      (out2 / "log10_columns.txt").read_text() == "" and fl.display(A, fl.DISPLAY) == "speed a")

# ---------------------------------------------------------------- animate labels
from mufasa.tools import landscape_animate as la  # noqa: E402

seen = []
orig_fs = la.feature_space


def spy(a, b, na, nb):  # noqa: ANN001
    seen.append((na, nb))
    return orig_fs(a, b, na, nb)


la.feature_space = spy
la.main([str(out), "--pair", A, B, "--through-tsne", "--size", "120", "--hold", "0.1",
         "--transition", "0.2", "--fps", "5", "--n-points", "200",
         "--names-file", str(fn.names_path(cfg)), "--out", str(out / "a.gif")])
check("animate: log10 feature labelled 'log10 speed a'",
      seen and seen[0][0] == "log10 speed a" and seen[0][1] == fn.default_name(B))

# ---------------------------------------------------------------- reuse respects scales
root = lr.landscape_root(cfg)
root.mkdir(parents=True, exist_ok=True)
import shutil  # noqa: E402

r1 = root / "20260101-000000-aaaaaa"
shutil.copytree(out, r1)
(r1 / lr.SELECTION_FILE).write_text(f"{A}\n{B}\n{C}\n")
check("run with the same scales reused", lr.find_reusable_run(cfg, [A, B, C]) == r1)
fn.save_names(cfg, {}, {A: "linear"})
check("run with other scales not reused", lr.find_reusable_run(cfg, [A, B, C]) is None)

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jl_feature_scales_and_names_popup: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
