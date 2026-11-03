"""Smoke test for patch 122ic — the data's own skeleton, downstream of import.

122ib made the skeleton travel with a DeepLabCut parquet (embedded ``b"pose"``
metadata). This patch makes processing and viewing use it:

* pose_parquet.resolve_skeleton — explicit project.toml, then the pose files'
  own skeleton (embedded / ``.fdlc.toml``; a folder is probed through its long
  parquets), then the project enclosing the files;
* the smoother's standalone layout (layout_from_fdlc_sidecar) reads the
  embedded skeleton, for a file or a folder of agreeing files;
* a smoothed output keeps the input's skeleton (edges between its markers);
* pose_viewer's loader keeps marker case (it lower-cased every directly-read
  file, so a smoothed back_T4 became back_t4 and lost its skeleton lines);
* the overlay viewer draws the resolved skeleton instead of the legacy
  rat-rig default (whose marker names match nothing in other rigs).

Runs for real: pyarrow round-trips, the smoother's writer, and the overlay's
main() with Qt offscreen (event loop stubbed).
"""
from __future__ import annotations

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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mufasa.pose_importers import pose_parquet as PP  # noqa: E402
from mufasa.project_layout import PROJECT_LAYOUT_VERSION, write_skeleton  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


BPS = ["head_nose", "head_mid", "back_T4", "back_L2", "tail_V6"]
EDGES = [("head_nose", "head_mid"), ("head_mid", "back_T4"),
         ("back_T4", "back_L2"), ("back_L2", "tail_V6")]


def long_df(n=30):
    rows = []
    for f in range(n):
        for i, bp in enumerate(BPS):
            rows.append({"frame": f, "individual": "single", "bodypart": bp,
                         "x": 40.0 + 12.0 * i + f, "y": 60.0 + 3.0 * i,
                         "likelihood": 0.95})
    return pd.DataFrame(rows)


def write_dlc(path, edges=EDGES, n=30):
    PP.write_parquet_with_pose_metadata(long_df(n), path, BPS, edges)
    return Path(path)


tmp = Path(tempfile.mkdtemp())

# ------------------------------------------------------- resolve_skeleton ---
dlc = write_dlc(tmp / "rec.parquet")
r = PP.resolve_skeleton([dlc])
check("resolve: embedded skeleton of the pose file", r and r["edges"] == EDGES)

folder = tmp / "folder"
folder.mkdir()
write_dlc(folder / "a.parquet")
r = PP.resolve_skeleton([folder])
check("resolve: a folder is probed through its long parquets",
      r and r["edges"] == EDGES)

proj = tmp / "proj"
(proj / "derived" / "smoothed" / "run1").mkdir(parents=True)
cfg = proj / "project.toml"
cfg.write_text(f"project_layout_version = {PROJECT_LAYOUT_VERSION}\n")
write_skeleton(cfg, nodes=BPS, edges=EDGES[:2])
inside = proj / "derived" / "smoothed" / "run1" / "x_smoothed_v2.parquet"
pd.DataFrame({"head_nose_x": [1.0], "head_nose_y": [1.0]}).to_parquet(inside)
r = PP.resolve_skeleton([inside])
check("resolve: falls back to the project enclosing the file",
      r and r["edges"] == EDGES[:2] and r["source"] == str(cfg))
r = PP.resolve_skeleton([dlc], config=cfg)
check("resolve: an explicit project.toml wins", r and r["edges"] == EDGES[:2])
loose = tmp / "loose.parquet"
long_df().to_parquet(loose)
check("resolve: nothing found -> None (caller keeps its default)",
      PP.resolve_skeleton([loose, None]) is None)

# --------------------------------------------------- smoother layout ---
from mufasa.data_processors import kalman_pose_smoother_v2 as K  # noqa: E402

lay = K.layout_from_fdlc_sidecar(dlc)
check("smoother layout from the embedded skeleton",
      lay is not None and set(lay.marker_names) == set(BPS))
lay = K.layout_from_fdlc_sidecar(folder)
check("smoother layout from a folder of agreeing files",
      lay is not None and set(lay.marker_names) == set(BPS))
write_dlc(folder / "b.parquet", edges=EDGES[:3])
check("folder whose files disagree on the skeleton -> None",
      K.layout_from_fdlc_sidecar(folder) is None)
check("no skeleton -> None (main falls back as before)",
      K.layout_from_fdlc_sidecar(loose) is None)
edges_only = tmp / "edges_only.parquet"
long_df().to_parquet(edges_only)
PP.embed_pose_metadata(edges_only, EDGES, bodyparts=BPS)
import json  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

t = pq.read_table(edges_only)
meta = dict(t.schema.metadata)
meta[b"pose"] = json.dumps({"skeleton": [list(e) for e in EDGES]}).encode()
pq.write_table(t.replace_schema_metadata(meta), edges_only)
lay = K.layout_from_fdlc_sidecar(edges_only)
check("edges without a bodyparts list: nodes taken from the edges",
      lay is not None and set(lay.marker_names) == set(BPS))

# ------------------------------------------------- smoothed output keeps it ---
T, Kn = 20, len(BPS)
s = {"path": dlc, "markers": BPS, "likelihoods": np.full((T, Kn), 0.9)}
pos = np.random.default_rng(0).normal(50, 5, (T, Kn, 2))
var = np.full((T, Kn, 2), 1.0)
out_dir = tmp / "smoothed"
out_dir.mkdir()
out, was_csv, _ = K._build_and_write_session_output(
    s, pos, var, {m: i for i, m in enumerate(BPS)}, out_dir, param_hash="ab12",
)
osk = PP.read_pose_skeleton(out)
check("smoothed output written as parquet", out is not None and not was_csv)
check("smoothed output carries the input's skeleton",
      osk is not None and osk["edges"] == EDGES and osk["nodes"] == BPS)
out2, _, _ = K._build_and_write_session_output(
    {"path": loose, "markers": BPS, "likelihoods": s["likelihoods"]},
    pos, var, {m: i for i, m in enumerate(BPS)}, out_dir, param_hash="cd34",
)
check("an input without a skeleton gives a plain output (no metadata)",
      PP.read_pose_skeleton(out2) is None and pd.read_parquet(out2).shape[0] == T)

# ------------------------------------------- pose_viewer keeps marker case ---
from mufasa.tools.pose_viewer import _load_pose_file  # noqa: E402

pf = _load_pose_file(out)
check("viewer loader keeps marker case for a directly-read file",
      "back_T4" in pf.markers and "back_t4" not in pf.markers)
check("viewer loader still detects the variance columns",
      pf.variances is not None)

# ----------------------------------------------------- overlay end to end ---
import cv2  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

vid = tmp / "rec.mp4"
w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
for i in range(T):
    w.write(np.full((240, 320, 3), i, np.uint8))
w.release()

QApplication.exec = lambda self: 0
import mufasa.tools.pose_video_overlay as O  # noqa: E402

shown: dict = {}
O.OverlayViewer.show = lambda self: shown.__setitem__("w", self)


def overlay(*extra):
    shown.clear()
    rc = O.main([str(vid), *extra])
    return rc, shown.get("w")


rc, win = overlay("--smoothed", str(out), "--raw", str(dlc))
sc = win.scene_obj if win else None
check("overlay main() runs", rc == 0 and sc is not None)
check("overlay draws the data's skeleton, not the rat-rig default",
      sc is not None and [tuple(e) for e in sc.skeleton_edges] == EDGES)
check("overlay: every edge drawn for raw and for smoothed",
      sc is not None and len(sc._skeleton_lines_raw) == len(EDGES)
      and len(sc._skeleton_lines_smoothed) == len(EDGES))
rc, win = overlay("--smoothed", str(out), "--raw", str(dlc), "--config", str(cfg))
check("overlay --config uses that project's skeleton",
      win is not None and [tuple(e) for e in win.scene_obj.skeleton_edges]
      == EDGES[:2])
rc, win = overlay("--raw", str(loose))
check("overlay without any skeleton keeps the built-in default",
      win is not None and win.scene_obj.skeleton_edges == O.DEFAULT_SKELETON_EDGES)

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ic_skeleton_downstream: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
