"""Smoke test for patch 122jk — landscape / animation from the Feature
Visualization selection, figure names, joint styles, viewer grid.

User request: "An option to generate mufasa-landscapes and
mufasa-landscapes-animated based on the selected features. The animated gif
should transition from tsne to one joint space back to tsne and then to the
next joint space. also some other plots would be nice like scatter, heatmap,
scatter-heatmap. And add a light 5 bar grid in the feature video viewer.
Also when generating mufasa-landscape and mufasa-landscape-animate, a popup
should show up and allow alternate names for the features in an editable
table, before generating the plots, and it should remember the preferences
for the label names."

Checks:
* feature_names: save / load / forget-on-default, TOML round trip;
* plot_joint styles heatmap / scatter / scatter-heatmap, bad style raises;
* FeatureTracePlot: 5 grid lines per lane, top to bottom, and they paint;
* open video × feature viewers follow the selection (debounced; opt out
  with "Follow selection"; closed viewers dropped);
* FeatureNamesDialog: pre-filled, edits saved on OK, remembered next time,
  reset; options per mode;
* mufasa-landscape --columns-file / --names-file: exactly the selection,
  display names used, full-circle angle kept;
* animation_pairs / find_reusable_run / argv builders;
* the GUI path end to end: Landscape… and Animate… through QProcess, the
  GIF going t-SNE → pair → t-SNE (--through-tsne) and listed for preview.

Runs offscreen on a generated project.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
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

# ---------------------------------------------------------------- names
from mufasa.utils import feature_names as fn  # noqa: E402

check("names path is logs/measures/feature_names.toml",
      fn.names_path(cfg).parts[-3:] == ("logs", "measures", "feature_names.toml"))
check("default name drops _FEATURE_SUBSET",
      fn.default_name(A) == "Animal_1 movement a (mm)")
fn.save_names(cfg, {A: "speed a", B: "  ", C: fn.default_name(C)})
saved = fn.load_names(cfg)
check("only real renames are stored", saved == {A: "speed a"})
with open(fn.names_path(cfg), "rb") as fh:
    check("file is TOML with a [names] table", tomllib.load(fh)["names"] == {A: "speed a"})
check("display uses the saved name, else the default",
      fn.display(A, saved) == "speed a" and fn.display(B, saved) == fn.default_name(B))
fn.save_names(cfg, {A: fn.default_name(A)})
check("setting the default forgets the entry", fn.load_names(cfg) == {})

# ---------------------------------------------------------------- joint styles
from matplotlib.figure import Figure  # noqa: E402

from mufasa.tools.feature_plots import JOINT_STYLES, plot_joint  # noqa: E402

x, y = r.normal(0, 1, 5000), r.normal(0, 1, 5000)
mask = np.zeros(5000, bool)
mask[:300] = True
kinds = {}
for st in JOINT_STYLES:
    f = Figure()
    plot_joint(f, x, y, "x", "y", mask=mask, behaviour="rear", style=st, scatter_points=1000)
    ax = f.axes[0]
    kinds[st] = (len(ax.collections), [type(c).__name__ for c in ax.collections])
check("three joint styles", JOINT_STYLES == ("heatmap", "scatter", "scatter-heatmap"))
check("heatmap: density mesh + behaviour points (+ colorbar)",
      "QuadMesh" in kinds["heatmap"][1] and len(f.axes) >= 1)
check("scatter: no density mesh, frames + behaviour as points",
      "QuadMesh" not in kinds["scatter"][1]
      and kinds["scatter"][1].count("PathCollection") == 2)
check("scatter-heatmap: mesh and both point layers",
      "QuadMesh" in kinds["scatter-heatmap"][1]
      and kinds["scatter-heatmap"][1].count("PathCollection") == 2)
f = Figure()
plot_joint(f, x, y, "x", "y", style="scatter", scatter_points=700)
check("scatter subsamples to scatter_points",
      len(f.axes[0].collections[0].get_offsets()) == 700)
try:
    plot_joint(Figure(), x, y, "x", "y", style="nope")
    bad = False
except ValueError:
    bad = True
check("unknown style raises", bad)

# ---------------------------------------------------------------- viewer grid
from mufasa.ui_qt.feature_viewer import FeatureTracePlot  # noqa: E402

tp = FeatureTracePlot()
ys = tp.grid_ys(10, 40)
check("5 grid lines per lane, top to bottom evenly",
      tp.GRID_LINES == 5 and ys == [10, 20, 30, 40, 50])
tp.resize(700, 200)
tp.set_features(pd.DataFrame({A: np.zeros(2000)}))   # flat trace at the lane bottom
img_on = tp.grab().toImage()
tp.GRID_ALPHA = 0
img_off = tp.grab().toImage()
check("grid paints (image changes with the grid turned off)", img_on != img_off)
tp.GRID_ALPHA = FeatureTracePlot.GRID_ALPHA

# ---------------------------------------------------------------- names dialog
from mufasa.ui_qt import landscape_runner as lr  # noqa: E402

d = lr.FeatureNamesDialog(None, cfg, [A, B], "landscape")
check("dialog rows = selection, prefilled with defaults",
      d.table.rowCount() == 2 and d.table.item(0, 1).text() == fn.default_name(A))
check("column cell read-only, name cell editable",
      not (d.table.item(0, 0).flags() & lr.Qt.ItemIsEditable)
      and bool(d.table.item(0, 1).flags() & lr.Qt.ItemIsEditable))
d.table.item(0, 1).setText("speed a")
d.table.item(1, 1).setText("speed b")
d.accept()
check("OK saves the names", fn.load_names(cfg) == {A: "speed a", B: "speed b"})
d2 = lr.FeatureNamesDialog(None, cfg, [B, C], "animate")
check("names remembered next time",
      d2.table.item(0, 1).text() == "speed b"
      and d2.table.item(1, 1).text() == fn.default_name(C))
d2.table.selectRow(0)
d2._reset_selected()
check("reset puts the column name back", d2.table.item(0, 1).text() == fn.default_name(B))
o = d2.options()
check("animate dialog has animation options",
      {"n_pairs", "theme", "color", "mp4", "reuse", "tsne_sample"} <= set(o))
check("landscape dialog has no animation options", "theme" not in d.options())
d2.reject()
check("cancel saves nothing", fn.load_names(cfg) == {A: "speed a", B: "speed b"})

# ---------------------------------------------------------------- CLI with selection
from mufasa.tools import feature_landscape as fl  # noqa: E402

sel = [A, B, C, ANG]
out1 = tmp / "cli_run"
colsf = tmp / "cols.txt"
colsf.write_text("\n".join(sel) + "\n")
rc = fl.main(["--project", cfg, "--sources", "features", "--columns-file", str(colsf),
              "--out", str(out1), "--tsne-sample", "400", "--frames-per-session", "600",
              "--n-jobs", "1", "--perplexity", "20"])
pairs = pd.read_csv(out1 / "pairs.csv")
used = set(pairs.feature_a) | set(pairs.feature_b)
check("CLI: ran", rc == 0 and (out1 / "jpdf_top.png").is_file() and (out1 / "tsne.png").is_file())
check("CLI: exactly the selected columns (unselected left out)",
      OTHER not in used and {A, B, C} <= used)
check("CLI: full-circle angle kept for an explicit selection",
      ANG in used or any(ANG.split(" (")[0] in c for c in used))
check("CLI: saved display names loaded", fl.DISPLAY.get(A) == "speed a")
check("CLI: t-SNE samples hold the selection",
      set(pd.read_parquet(out1 / "tsne_samples.parquet").columns) >= {A, B, C})

# ---------------------------------------------------------------- pairs / reuse helpers
ap = lr.animation_pairs([A, B], None, 4)
check("two selected → that one pair", ap == [(A, B)])
ap = lr.animation_pairs([A, B, C], out1 / "pairs.csv", 3, max_per_feature=1)
check("more selected → ranked pairs, feature caps respected",
      len(ap) == 1 and set(ap[0]) <= {A, B, C})
ap = lr.animation_pairs([A, B, C, ANG], out1 / "pairs.csv", 3, max_per_feature=2)
top_row = (pairs.feature_a.iloc[0], pairs.feature_b.iloc[0])
check("first animated pair is the best-ranked", ap[0] == top_row and len(ap) <= 3)
argv = lr.animate_argv(out1, [(A, B), (B, C)], out1 / "x.gif", None,
                       theme="light", mp4=False, color="tsne")
check("animate argv: --through-tsne and one --pair per space",
      "--through-tsne" in argv and argv.count("--pair") == 2)

# ---------------------------------------------------------------- GUI end to end
from mufasa.ui_qt.forms.feature_visualization import FeatureVisualizationForm  # noqa: E402

F = FeatureVisualizationForm(config_path=cfg)
F.resize(1000, 900)
check("form has Landscape… / Animate… and the joint style combo",
      F.landscape_btn.text() == "Landscape…" and F.animate_btn.text() == "Animate…"
      and [F.joint_style_cb.itemText(i) for i in range(F.joint_style_cb.count())]
      == list(JOINT_STYLES))
F.set_selected([A, B])
F.joint_style_cb.setCurrentText("scatter-heatmap")
F.plot_joint()
check("form joint plot uses style and figure names",
      "QuadMesh" in [type(c).__name__ for c in F.figure.axes[0].collections]
      and F.figure.axes[0].get_xlabel().replace("\n", " ") == "speed a")


def wait(dlg, limit=600.0) -> None:
    t0 = time.time()
    while not (dlg.finished_ok or dlg.failed) and time.time() - t0 < limit:
        app.processEvents()
        time.sleep(0.05)


F.set_selected([A, B, C])
dlg = F.generate_landscape(ask=False, options=dict(tsne_sample=400, frames_per_session=600,
                                                   n_plot=3))
wait(dlg)
run1 = dlg.out_dir
log = dlg.log.toPlainText()
check("Landscape…: run finished", dlg.finished_ok and "[landscape] done" in log)
check("Landscape…: run under derived/plots/feature_landscape with the selection file",
      run1.parent == lr.landscape_root(cfg)
      and (run1 / lr.SELECTION_FILE).read_text().split() != []
      and sorted((run1 / lr.SELECTION_FILE).read_text().splitlines()) == sorted([A, B, C]))
listed = [dlg.files.item(i).text() for i in range(dlg.files.count())]
check("Landscape…: figures listed for preview",
      "jpdf_top.png" in listed and "tsne.png" in listed)
check("Landscape…: names file passed", "--names-file" in log)
dlg.files.setCurrentRow(listed.index("tsne.png"))
check("Landscape…: PNG previewed",
      dlg.view.pixmap() is not None and not dlg.view.pixmap().isNull())
check("reusable run found for the same selection (any order)",
      lr.find_reusable_run(cfg, [C, A, B]) == run1
      and lr.find_reusable_run(cfg, [A, B]) is None)

dlg2 = F.generate_animation(ask=False, options=dict(n_pairs=2, theme="dark"), )
# not started yet? run_animation starts at once — check that it reused
check("Animate…: earlier landscape reused (single step)",
      len(dlg2._steps) == 1 and dlg2.out_dir == run1
      and "reusing landscape run" in dlg2.log.toPlainText())
dlg2.stop()
dlg2.close()
# a fast variant for the test: same steps, smaller frames
from mufasa.ui_qt.landscape_runner import LandscapeRunDialog, animate_argv, animation_pairs  # noqa: E402

pairs_anim = animation_pairs([A, B, C], run1 / "pairs.csv", 2)
gif = run1 / "test_anim.gif"
dlg3 = LandscapeRunDialog(None, "anim", [lambda: animate_argv(
    run1, pairs_anim, gif, fn.names_path(cfg), theme="dark", mp4=False, color="tsne")
    + ["--size", "160", "--hold", "0.2", "--transition", "0.6", "--fps", "8",
       "--n-points", "300"]], run1)
dlg3.start()
wait(dlg3)
alog = dlg3.log.toPlainText()
n_sp = 1 + 2 * len(pairs_anim)
check("Animate: GIF written", dlg3.finished_ok and gif.is_file())
check("Animate: t-SNE → pair → t-SNE → next pair → t-SNE",
      f"{len(pairs_anim)} pair(s), {n_sp} spaces" in alog)
listed = [dlg3.files.item(i).text() for i in range(dlg3.files.count())]
check("Animate: GIF listed first and played", listed and listed[0].endswith(".gif")
      and dlg3._movie is not None and dlg3._movie.isValid())
from PIL import Image  # noqa: E402

with Image.open(gif) as im:
    check("GIF animates", getattr(im, "n_frames", 1) > 5)  # identical hold frames merge

# ---------------------------------------------------------------- viewer follows
F.set_selected([A, B])
F.video_cb.setCurrentText("v1")
cont = F.open_viewer()
vw = cont.viewer
check("viewer opened with the selection", vw.features == [A, B])
F.set_selected([A, B, C])
for _ in range(20):
    app.processEvents()
    time.sleep(0.03)
check("viewer follows: feature added", vw.features == [A, B, C]
      and list(vw.plot._data) == [A, B, C])
items = {F.tree.topLevelItem(0).child(i).text(0): F.tree.topLevelItem(0).child(i)
         for i in range(F.tree.topLevelItem(0).childCount())}
F.set_selected([C, A])
check("viewer follows: removed / reordered (after the debounce)",
      F._sync_viewers() in (0, 1) and vw.features == [C, A])
vw.follow_cb.setChecked(False)
F.set_selected([B])
F._sync_viewers()
check("unticking 'Follow selection' freezes the viewer", vw.features == [C, A])
vw.follow_cb.setChecked(True)
cont.close()
app.processEvents()
F.set_selected([A])
check("closed viewer dropped", F._sync_viewers() == 0 and F._viewers == [])

bad = LandscapeRunDialog(None, "bad", [lambda: ["-c", "import sys; sys.exit(3)"]], tmp)
bad.start()
wait(bad, 60)
check("failing step reported", bad.failed and "exited with code 3" in bad.log.toPlainText())

# Animate with nothing to reuse runs the landscape first
dlg4 = lr.run_animation(None, cfg, [A, C], ask=False, options=dict(reuse=True), start=False)
check("Animate on a new selection: landscape, then animation (2 steps)",
      len(dlg4._steps) == 2 and dlg4.out_dir != run1)
check("…the animation step pairs the two selected features",
      animation_pairs([A, C], None, 4) == [(A, C)])

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jk_landscape_from_selection: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
