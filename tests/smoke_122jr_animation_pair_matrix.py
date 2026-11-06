"""Smoke test for patch 122jr — choose the animated pairs in a matrix.

User request: "for the animation, I would like a selection matrix where each
row and column are the features and the selected elements will determine
which feature pair are included in the animation".

The Animate… popup gets a "Pairs to animate" tab: a feature × feature grid
of tick boxes (row on x, column on y; the mirror cell swaps the axes; the
diagonal unused). Ticked cells are numbered in the order ticked, which is
the animation order. Pre-filled from the pairs saved last time for these
features, else the best-ranked ones; buttons Best-ranked / All pairs /
Clear. The pairs are passed as --pair to mufasa-landscape-animate and kept
in logs/measures/feature_view.toml.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import time
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
from PySide6.QtCore import Qt  # noqa: E402
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
from mufasa.ui_qt import landscape_runner as lr  # noqa: E402
from mufasa.utils import feature_names as fn  # noqa: E402
from mufasa.utils import feature_view_state as st  # noqa: E402

cols = [A, B, C]
fn.save_names(cfg, {A: "speed a"})
d = lr.FeatureNamesDialog(None, cfg, cols, "animate")
m = d.matrix
check("animate popup has a 'Pairs to animate' tab with an N×N matrix",
      m is not None and d.tabs.count() == 2 and d.tabs.tabText(1).startswith("Pairs to animate")
      and m.grid.rowCount() == 3 and m.grid.columnCount() == 3)
check("landscape popup has no matrix",
      lr.FeatureNamesDialog(None, cfg, cols, "landscape").matrix is None)
check("diagonal not tickable",
      all(not (m.grid.item(i, i).flags() & Qt.ItemIsUserCheckable) for i in range(3)))
check("headers are the custom figure names (122js)",
      m.grid.verticalHeaderItem(0).text() == "speed a"
      and m.grid.horizontalHeaderItem(0).text() == "speed a"
      and m.grid.horizontalHeaderItem(1).text().replace("\n", " ") == fn.default_name(B))
check("pre-filled without a landscape run: features paired in order",
      d.pairs() == [(A, B)])
m.set_pairs([])
check("nothing ticked → Generate disabled", not d._ok.isEnabled())
m.grid.item(2, 0).setCheckState(Qt.Checked)       # C on x, A on y
m.grid.item(0, 1).setCheckState(Qt.Checked)       # A on x, B on y
m.grid.item(1, 0).setCheckState(Qt.Checked)       # B on x, A on y (mirror)
check("ticks become pairs in the order ticked (mirror = swapped axes)",
      d.pairs() == [(C, A), (A, B), (B, A)] and d._ok.isEnabled())
check("cells numbered with their place",
      [m.grid.item(2, 0).text(), m.grid.item(0, 1).text(), m.grid.item(1, 0).text()]
      == ["1", "2", "3"] and m.grid.item(0, 2).text() == "")
m.grid.item(2, 0).setCheckState(Qt.Unchecked)
check("unticking renumbers", d.pairs() == [(A, B), (B, A)] and m.grid.item(0, 1).text() == "1")
check("tab shows the count", d.tabs.tabText(1) == "Pairs to animate (2)"
      and "2 joint space(s) · 5 spaces" in m.count.text())
d.table.item(1, 1).setText("speed b")
check("renaming a feature relabels the matrix",
      m.grid.horizontalHeaderItem(1).text() == "speed b"
      and m.grid.verticalHeaderItem(1).text() == "speed b")
long = "forward head speed relative to body axis"
d.table.item(2, 1).setText(long)
check("long custom names shown in full (columns wrapped, rows on one line)",
      m.grid.verticalHeaderItem(2).text() == long
      and m.grid.horizontalHeaderItem(2).text().replace("\n", " ") == long
      and "\n" in m.grid.horizontalHeaderItem(2).text()
      and C in m.grid.verticalHeaderItem(2).toolTip())
check("options carry the pairs", d.options()["pairs"] == [(A, B), (B, A)])
d.accept()
check("pairs saved for next time", st.load_state(cfg)["animation_pairs"] == [[A, B], [B, A]])
d2 = lr.FeatureNamesDialog(None, cfg, cols, "animate")
check("next popup on these features: saved pairs ticked", d2.pairs() == [(A, B), (B, A)])
d3 = lr.FeatureNamesDialog(None, cfg, [A, C], "animate")
check("saved pairs of other features ignored", d3.pairs() == [(A, C)])
d3.matrix.set_pairs([(C, A)])
d3.accept()
check("saving another selection's pairs keeps the first's",
      st.load_state(cfg)["animation_pairs"] == [[A, B], [B, A], [C, A]])
d2.matrix.findChildren(type(d2.matrix.best_btn))  # buttons exist
[b for b in d2.matrix.findChildren(type(d2.matrix.best_btn))
 if b.text() == "All pairs (upper triangle)"][0].click()
check("All pairs: upper triangle", d2.pairs() == [(A, B), (A, C), (B, C)])
[b for b in d2.matrix.findChildren(type(d2.matrix.best_btn)) if b.text() == "Clear"][0].click()
check("Clear", d2.pairs() == [])

# ---------------------------------------------------------------- best-ranked from a run
from mufasa.ui_qt.forms.feature_visualization import FeatureVisualizationForm  # noqa: E402

F = FeatureVisualizationForm(config_path=cfg)
F.set_selected(cols)
dl = F.generate_landscape(ask=False, options=dict(tsne_sample=400, frames_per_session=600,
                                                  n_plot=3))
t0 = time.time()
while not (dl.finished_ok or dl.failed) and time.time() - t0 < 600:
    app.processEvents()
    time.sleep(0.05)
ranked = pd.read_csv(dl.out_dir / "pairs.csv")
d4 = lr.FeatureNamesDialog(None, cfg, cols, "animate")
d4.n_pairs.setValue(1)
d4.matrix.best_btn.click()
check("Best-ranked ticks the top pair of the landscape run",
      dl.finished_ok and d4.pairs() == [(ranked.feature_a[0], ranked.feature_b[0])])

# ---------------------------------------------------------------- the run uses them
dlg = lr.run_animation(None, cfg, cols, ask=False,
                       options=dict(pairs=[(B, A), (C, B)], reuse=True), start=False)
argv = dlg._steps[-1]()
pv = [(argv[i + 1], argv[i + 2]) for i, x in enumerate(argv) if x == "--pair"]
check("Animate passes the matrix pairs as --pair, in order", pv == [(B, A), (C, B)])
dlg2 = lr.run_animation(None, cfg, cols, ask=False, options=dict(reuse=True, n_pairs=2),
                        start=False)
argv = dlg2._steps[-1]()
check("without matrix pairs: best-ranked as before", argv.count("--pair") >= 1)

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jr_animation_pair_matrix: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
