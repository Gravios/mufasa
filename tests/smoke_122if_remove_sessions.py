"""Smoke test for patch 122if — remove sessions via the Data Import table.

The "Already imported" table on Data Import -> Import video gains a
"Pose data" column, multi-row selection and *Remove selected…* (button,
Delete key, right-click). Removal is a move, not a delete: the video (a
symlink as the link), optionally its imported pose file(s), and its
video_info.csv row go to backups/removed_sessions-<run_id>/ with a manifest.

Runs for real: mufasa.model.session_removal on a ProjectConfigCreator
project, and the VideoImportForm itself with Qt offscreen (only the
confirmation / result message boxes are stubbed).
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
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


from mufasa.project_layout import project_paths_from_config  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402

tmp = Path(tempfile.mkdtemp())


def make_project(name):
    cfg = ProjectConfigCreator(
        project_path=str(tmp), project_name=name, target_list=[],
        pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
        animal_cnt=1, file_type="parquet", body_parts=["a", "b"],
    ).config_path
    p = project_paths_from_config(cfg)
    vd, pd_ = Path(p["video_dir"]), Path(p["input_pose_dir"])
    vd.mkdir(parents=True, exist_ok=True)
    pd_.mkdir(parents=True, exist_ok=True)
    footage = tmp / f"{name}_footage"
    footage.mkdir()
    (footage / "sessB.mp4").write_bytes(b"real footage B")
    (vd / "sessA.mp4").write_bytes(b"copied video A")
    os.symlink(footage / "sessB.mp4", vd / "sessB.mp4")
    (vd / "sessC.mp4").write_bytes(b"video C")
    for stem in ("sessA", "sessB", "sessC"):
        pd.DataFrame({"a_x": [1.0]}).to_parquet(pd_ / f"{stem}.parquet")
    pd.DataFrame({
        "Video": ["sessA", "sessB", "sessC"], "fps": [30, 30, 30],
        "Resolution_width": [640] * 3, "Resolution_height": [360] * 3,
        "Distance_in_mm": [0] * 3, "pixels/mm": [1.0] * 3,
    }).to_csv(p["video_info_path"], index=False)
    return cfg, p, footage


# --------------------------------------------------------------- backend ---
from mufasa.model.session_removal import remove_sessions, session_files  # noqa: E402

cfg, P, footage = make_project("p1")
vd, pdir = Path(P["video_dir"]), Path(P["input_pose_dir"])
plan = session_files(cfg, ["sessA.mp4", "sessB.mp4"])
check("plan finds each video and its same-stem pose file",
      plan["sessA.mp4"]["video"] == str(vd / "sessA.mp4")
      and plan["sessA.mp4"]["pose"] == [str(pdir / "sessA.parquet")])
res = remove_sessions(cfg, ["sessA.mp4", "sessB.mp4"], include_pose=True)
backup = Path(res["backup_dir"])
check("videos left the project", not (vd / "sessA.mp4").exists()
      and not os.path.lexists(vd / "sessB.mp4"))
check("pose files left the project", not (pdir / "sessA.parquet").exists()
      and not (pdir / "sessB.parquet").exists())
check("the untouched session stays", (vd / "sessC.mp4").exists()
      and (pdir / "sessC.parquet").exists())
check("removed files are kept under backups/removed_sessions-*",
      backup.parent == Path(P["project_root"]) / "backups"
      and backup.name.startswith("removed_sessions-")
      and (backup / "videos" / "sessA.mp4").read_bytes() == b"copied video A"
      and (backup / "pose" / "sessA.parquet").exists())
check("a symlinked video moves as the link; the footage is untouched",
      os.path.islink(backup / "videos" / "sessB.mp4")
      and (footage / "sessB.mp4").read_bytes() == b"real footage B")
vi = pd.read_csv(P["video_info_path"])
check("their video_info rows are dropped, others kept",
      list(vi["Video"]) == ["sessC"] and res["video_info_rows"] == 2)
check("a full copy of video_info.csv is kept in the backup",
      len(pd.read_csv(backup / "video_info.csv")) == 3)
man = tomllib.loads((backup / "manifest.toml").read_text())
check("manifest records every move (from -> to)",
      len(man["files"]) == 4
      and {Path(f["from"]).name for f in man["files"]}
      == {"sessA.mp4", "sessB.mp4", "sessA.parquet", "sessB.parquet"}
      and all(Path(f["to"]).exists() or os.path.islink(f["to"])
              for f in man["files"]))

cfg2, P2, _ = make_project("p2")
remove_sessions(cfg2, ["sessA.mp4"], include_pose=False)
check("include_pose=False keeps the pose file",
      (Path(P2["input_pose_dir"]) / "sessA.parquet").exists()
      and not (Path(P2["video_dir"]) / "sessA.mp4").exists())
try:
    remove_sessions(cfg2, ["nope.mp4"])
    check("removing nothing that exists is refused", False)
except ValueError:
    check("removing nothing that exists is refused", True)

# --------------------------------------------------------------- the form ---
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])
answers = {"exec": QMessageBox.Yes, "tick_pose": True}
seen: list[str] = []


def _fake_exec(self):
    seen.append("confirm")
    cb = self.checkBox()
    if cb is not None:
        cb.setChecked(answers["tick_pose"])
    return answers["exec"]


QMessageBox.exec = _fake_exec
QMessageBox.information = staticmethod(lambda *a, **k: seen.append("info"))
QMessageBox.critical = staticmethod(lambda *a, **k: seen.append("critical"))

from mufasa.ui_qt.forms.video_import import VideoImportForm  # noqa: E402

cfg3, P3, _ = make_project("p3")
form = VideoImportForm(config_path=cfg3)
t = form._table
headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
check("table has a Pose data column", "Pose data" in headers)
rows = {t.item(r, 0).text(): t.item(r, 1).text() for r in range(t.rowCount())}
check("Pose data shows each session's imported pose file",
      rows.get("sessA.mp4") == "sessA.parquet")
from PySide6.QtWidgets import QAbstractItemView  # noqa: E402

check("table allows selecting several rows (Ctrl/Shift-click)",
      t.selectionMode() == QAbstractItemView.ExtendedSelection)
check("Remove is disabled with nothing selected",
      not form._remove_btn.isEnabled())


def select(names):
    from PySide6.QtCore import QItemSelectionModel
    t.clearSelection()
    sm = t.selectionModel()
    for r in range(t.rowCount()):
        if t.item(r, 0).text() in names:
            sm.select(t.model().index(r, 0),
                      QItemSelectionModel.Select | QItemSelectionModel.Rows)


select({"sessA.mp4", "sessC.mp4"})
check("multi-row selection enables Remove",
      form._remove_btn.isEnabled()
      and form._selected_video_names() == ["sessA.mp4", "sessC.mp4"])

answers["exec"] = QMessageBox.Cancel
form._remove_selected()
check("cancel leaves everything in place",
      (Path(P3["video_dir"]) / "sessA.mp4").exists() and "info" not in seen)

answers["exec"] = QMessageBox.Yes
answers["tick_pose"] = False
form._remove_selected()
names_after = [t.item(r, 0).text() for r in range(t.rowCount())]
check("confirmed removal refreshes the table",
      names_after == ["sessB.mp4"] and "info" in seen)
check("unticked pose box keeps the pose files",
      (Path(P3["input_pose_dir"]) / "sessA.parquet").exists())

check("Delete key is wired to removal",
      form._del_shortcut.key().toString() == "Del")
cfg4, P4, _ = make_project("p4")
for f in Path(P4["video_dir"]).iterdir():
    f.unlink()
form4 = VideoImportForm(config_path=cfg4)
form4._table.selectAll()
check("the placeholder row can't be removed",
      form4._selected_video_names() == [] and not form4._remove_btn.isEnabled())

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122if_remove_sessions: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
