"""Smoke test for patch 122ie — FreeDLC parquet without .fdlc naming.

FreeDLC (the modified DeepLabCut) now writes ``pose.parquet`` per analysis run
with a run ``.toml`` beside it and the skeleton embedded in the parquet; files
are no longer ``<video>.fdlc.parquet`` with a ``.fdlc.toml`` sidecar. This
checks the UI paths that set a project up from a pose file actually take such
a file, driven through their real buttons (file dialogs and message boxes
stubbed, Qt offscreen):

* project creation "Auto-detect from pose file…": offers .parquet, reads the
  body parts of a run's pose.parquet (with or without embedded metadata) and
  keeps its skeleton;
* File -> Reconfigure: offers .parquet and reads its body parts;
* one parquet import route, described by the run-folder structure; no file
  filter or ingest suffix list that names .fdlc.parquet;
* Add-sessions accepts a long parquet whatever it is called.
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

import pandas as pd  # noqa: E402

from mufasa.pose_importers import pose_parquet as PP  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


BPS = ["head_nose", "head_mid", "back_T4", "back_L2", "tail_V6"]
EDGES = [("head_nose", "head_mid"), ("head_mid", "back_T4"),
         ("back_T4", "back_L2"), ("back_L2", "tail_V6")]


def long_df(n=20):
    rows = []
    for f in range(n):
        for i, bp in enumerate(BPS):
            rows.append({"frame": f, "individual": "single", "bodypart": bp,
                         "x": 1.0 * i + f, "y": 2.0 * i, "likelihood": 0.9})
    return pd.DataFrame(rows)


tmp = Path(tempfile.mkdtemp())
run = tmp / "runs" / "K7_cam_A"
run.mkdir(parents=True)
pose = run / "pose.parquet"
PP.write_parquet_with_pose_metadata(long_df(), pose, BPS, EDGES)
(run / "K7_cam_A.toml").write_text(
    'kind = "analyze"\ninputs = ["/data/v/K7_cam_A.mp4"]\n'
    'outputs = ["pose.parquet"]\n')
bare = tmp / "bare" / "pose.parquet"
bare.parent.mkdir()
long_df().to_parquet(bare)  # no metadata at all

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QMessageBox,
)

app = QApplication.instance() or QApplication([])
state: dict = {"pick": None, "filters": [], "boxes": []}


def _fake_open(*a, **k):
    state["filters"].append(a[3] if len(a) > 3 else k.get("filter", ""))
    return state["pick"], ""


QFileDialog.getOpenFileName = staticmethod(_fake_open)
for _n in ("warning", "critical", "information"):
    setattr(QMessageBox, _n,
            staticmethod(lambda *a, _n=_n, **k: state["boxes"].append(_n)))
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

# ---------------------------------------------------------- project create ---
from mufasa.ui_qt.forms.project_create import ProjectCreateForm  # noqa: E402

form = ProjectCreateForm(show_create_button=False)
state["pick"] = str(pose)
form._autodetect_from_dlc()
check("create: the file dialog offers .parquet",
      "*.parquet" in state["filters"][-1])
check("create: a run's pose.parquet is read without error",
      form._autodetected_bps == BPS and not state["boxes"])
check("create: the embedded skeleton is kept",
      form._autodetected_skeleton == EDGES
      and "4 skeleton edges" in form._autodetect_label.text())
form._clear_autodetect()
state["pick"] = str(bare)
form._autodetect_from_dlc()
check("create: a pose.parquet without metadata still gives its body parts",
      form._autodetected_bps == BPS and form._autodetected_skeleton == []
      and not state["boxes"])
form._dir_edit.setText(str(tmp))
form._name_edit.setText("proj")
form._autodetected_skeleton = form._skeleton_of(str(pose), BPS)
cfg = form.submit()
from mufasa.project_layout import read_skeleton  # noqa: E402

check("create: project gets the parquet's skeleton",
      cfg and read_skeleton(cfg)["edges"] == EDGES)

# ------------------------------------------------------------- reconfigure ---
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

cfg2 = ProjectConfigCreator(
    project_path=str(tmp), project_name="proj2", target_list=[],
    pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
    animal_cnt=1, file_type="parquet", body_parts=["a", "b"],
).config_path
from mufasa.ui_qt.reconfigure_dialog import ReconfigureProjectDialog  # noqa: E402

dlg = ReconfigureProjectDialog(config_path=cfg2)
state["boxes"].clear()
state["pick"] = str(pose)
dlg._pick_file()
check("reconfigure: the file dialog offers .parquet",
      "*.parquet" in state["filters"][-1])
check("reconfigure: pose.parquet body parts detected",
      dlg._detected_bps == BPS and "warning" not in state["boxes"])

# ------------------------------------------------- no .fdlc naming assumed ---
src_root = REPO / "mufasa"
form_src = (src_root / "ui_qt/forms/pose_import.py").read_text()
import re  # noqa: E402

routes = re.findall(r'^    "([^"]+)":\s*dict\(', form_src, re.MULTILINE)
check("one parquet import route", sum("parquet" in r.lower() for r in routes) == 1)
check("import hint describes run folders, not *.fdlc.parquet",
      "pose.parquet" in form_src and "*.fdlc.parquet files" not in form_src)
check("no file filter names *.fdlc.parquet",
      "*.fdlc.parquet" not in (src_root / "ui_qt/forms/add_sessions.py").read_text())
from mufasa.model import session_ingest as SI  # noqa: E402

check("ingest suffix list has no .fdlc entry",
      all("fdlc" not in s for s in SI._POSE_SUFFIXES))
odd = tmp / "loose" / "session_42_whatever.parquet"
odd.parent.mkdir()
PP.write_parquet_with_pose_metadata(long_df(), odd, BPS, EDGES)
rep = SI.check_pose_compatibility(str(odd.parent), cfg)
check("Add-sessions accepts a long parquet whatever its name",
      rep["accepted"] == [str(odd)])

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ie_freedlc_no_fdlc_names: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
