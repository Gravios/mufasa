"""Smoke test for patch 122jt — JPDF of the last ticked pair beside the
pair matrix.

User request: "to the right of the matrix can you add the jpdf of the last
selected pair?"

The "Pairs to animate" tab splits into the matrix and a joint-PDF preview
of the last ticked pair, pooled over all videos (≤ 3000 frames each, -1
read as missing), drawn with the names, log10 scales and jitter currently
set in the names table, and redrawn when the ticks, a name, a scale or a
jitter tick changes.
"""
from __future__ import annotations

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
# A positive for log10; -1 sentinels in B
for v in ("v1", "v2"):
    df = pd.read_parquet(fam / f"{v}.parquet")
    df[A] = np.exp(df[A])
    df[B] = np.abs(df[B])
    df.loc[:49, B] = -1.0
    df.to_parquet(fam / f"{v}.parquet")

from mufasa.ui_qt import landscape_runner as lr  # noqa: E402
from mufasa.utils import feature_names as fn  # noqa: E402


def settle(sec=0.4):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.02)


fn.save_names(cfg, {A: "speed a", B: "speed b"})
d = lr.FeatureNamesDialog(None, cfg, [A, B, C], "animate")
d.resize(1200, 900)
d.show()
split = d.tabs.widget(1)
check("pairs tab: matrix on the left, preview on the right",
      split.widget(0) is d.matrix and split.widget(1) is d.preview)
d.tabs.setCurrentIndex(1)
settle()
ax_count = len(d.preview.figure.axes)
check("preview drawn for the pre-ticked pair", d.preview.pair == ("speed a", "speed b")
      and ax_count >= 1)
fr = d.preview_frame()
check("data pooled over both videos, -1 read as missing",
      len(fr) == 2 * min(lr.FeatureNamesDialog.PREVIEW_PER_VIDEO, N)
      and fr[B].isna().sum() > 0 and (fr[B] == -1).sum() == 0)
d.matrix.grid.item(2, 0).setCheckState(Qt.Checked)        # C on x, A on y
settle()
check("last ticked pair shown", d.preview.pair == (fn.default_name(C), "speed a")
      and "Pair 2" in d.preview.title.text())
ax = d.preview.figure.axes[0]
check("axes labelled with the figure names",
      ax.get_xlabel().replace("\n", " ") == fn.default_name(C)
      and ax.get_ylabel().replace("\n", " ") == "speed a")
yl0 = ax.get_ylim()
d.scale_cbs[0].setCurrentText("log10")
settle()
ax = d.preview.figure.axes[0]
check("log10 scale applied live", ax.get_ylabel().startswith("log10 speed a")
      and ax.get_ylim()[1] < yl0[1])
d.table.item(2, 1).setText("speed c")
settle()
check("rename redraws", d.preview.figure.axes[0].get_xlabel() == "speed c")
d.table.item(2, 3).setCheckState(Qt.Checked)
settle()
check("jitter tick redraws (still the same pair)", d.preview.pair == ("speed c", "log10 speed a"))
d.matrix.grid.item(2, 0).setCheckState(Qt.Unchecked)
settle()
check("unticking falls back to the previous pair",
      d.preview.pair == ("log10 speed a", "speed b"))
d.matrix.set_pairs([])
settle()
check("nothing ticked: a hint instead", d.preview.pair is None
      and any("Tick a pair" in t.get_text() for t in d.preview.figure.texts))
check("data read once", d.preview_frame() is fr)
d.reject()

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"smoke_122jt_pair_matrix_jpdf_preview: {len(checks) - len(failed)}/{len(checks)} "
      "checks passed")
sys.exit(1 if failed else 0)
