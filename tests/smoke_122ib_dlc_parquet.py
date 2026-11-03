"""Smoke test for patch 122ib — DeepLabCut long-format parquet + embedded skeleton.

The modified DeepLabCut exporter writes the same long/tidy table as FreeDLC
(frame | individual | bodypart | x | y | likelihood) but names it
``<video>.parquet`` and embeds the skeleton in the parquet key/value metadata
(``b"pose"`` JSON with DLC's own ``bodyparts`` / ``skeleton`` keys) instead of
a ``.fdlc.toml`` sidecar. This covers:

* mufasa.pose_importers.pose_parquet — the dependency-light shared layer:
  schema-based detection, embedded-metadata reading (incl. index-pair edges,
  flat keys, malformed values), embedded-over-sidecar precedence, body-part
  listing, embed_pose_metadata (data and pandas metadata preserved) and its CLI;
* FDLCParquetImporter — finds ``<video>.parquet`` by schema (wide/unrelated
  parquets and subfolders ignored), joins it to the video stem, imports the
  embedded skeleton into project.toml, and honours an explicit file list;
* project creation — extract_bodyparts reads parquet; the create form keeps
  the file's skeleton and writes it to the new project's [skeleton];
* session ingest — routes long parquet through the importer (it used to copy
  a non-``.fdlc`` parquet in untouched) and imports only the accepted files.

Everything runs for real (pyarrow/pandas; the importer and form with their
own dependencies, Qt offscreen).
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import warnings
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
import pyarrow.parquet as pq  # noqa: E402

from mufasa.pose_importers import pose_parquet as PP  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


BPS = ["head_nose", "head_mid", "back_T4", "back_L2", "tail_V6"]
EDGES = [("head_nose", "head_mid"), ("head_mid", "back_T4"),
         ("back_T4", "back_L2"), ("back_L2", "tail_V6")]


def long_df(bps=BPS, n=40, individual="single"):
    rows = []
    for f in range(n):
        for i, bp in enumerate(bps):
            rows.append({"frame": f, "individual": individual, "bodypart": bp,
                         "x": 10.0 * i + f, "y": 5.0 * i, "likelihood": 0.9})
    return pd.DataFrame(rows)


def write_dlc(path, bps=BPS, edges=EDGES, n=40):
    PP.write_parquet_with_pose_metadata(long_df(bps, n), path, bps, edges)
    return path


tmp = Path(tempfile.mkdtemp())

# ---------------------------------------------------------------- format ---
dlc = write_dlc(tmp / "rec1.parquet")
wide = tmp / "wide.parquet"
pd.DataFrame({"head_nose_x": [1.0], "head_nose_y": [2.0],
              "head_nose_p": [0.9]}).to_parquet(wide)
check("long parquet detected by schema", PP.is_long_pose_parquet(dlc))
check("wide pose parquet is not long", not PP.is_long_pose_parquet(wide))
check("a CSV is not a long parquet",
      not PP.is_long_pose_parquet(tmp / "x.csv"))
check("video name: <video>.parquet -> <video>",
      PP.pose_parquet_video_name("/a/rec1.parquet") == "rec1")
check("video name: <video>.fdlc.parquet -> <video>",
      PP.pose_parquet_video_name("/a/rec1.fdlc.parquet") == "rec1")

# ------------------------------------------------------- embedded skeleton ---
sk = PP.read_pose_skeleton(dlc)
check("embedded skeleton read (nodes + name-pair edges)",
      sk is not None and sk["nodes"] == BPS and sk["edges"] == EDGES)
check("embedded skeleton names the parquet as its source",
      sk is not None and sk["source"] == str(dlc))

idx = tmp / "idx.parquet"
PP.write_parquet_with_pose_metadata(long_df(), idx, BPS, [])
import json  # noqa: E402

import pyarrow as pa  # noqa: E402

t = pq.read_table(idx)
meta = dict(t.schema.metadata)
meta[b"pose"] = json.dumps({"bodyparts": BPS, "skeleton": [[0, 1], [1, 2]]}).encode()
pq.write_table(t.replace_schema_metadata(meta), idx)
check("index-pair edges map through bodyparts",
      PP.read_pose_skeleton(idx)["edges"] == [("head_nose", "head_mid"),
                                              ("head_mid", "back_T4")])

flat = tmp / "flat.parquet"
t = pa.Table.from_pandas(long_df(), preserve_index=False)
meta = dict(t.schema.metadata or {})
meta[b"bodyparts"] = json.dumps(BPS).encode()
meta[b"skeleton"] = json.dumps([list(e) for e in EDGES]).encode()
pq.write_table(t.replace_schema_metadata(meta), flat)
check("flat bodyparts/skeleton keys also accepted",
      PP.read_pose_skeleton(flat)["edges"] == EDGES)

bad = tmp / "bad.parquet"
t = pa.Table.from_pandas(long_df(), preserve_index=False)
meta = dict(t.schema.metadata or {})
meta[b"pose"] = b"{not json"
pq.write_table(t.replace_schema_metadata(meta), bad)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r = PP.read_pose_skeleton(bad)
check("malformed embedded metadata -> None + warning (data still usable)",
      r is None and len(w) == 1)

plain = tmp / "plain.parquet"
long_df().to_parquet(plain)
check("no metadata and no sidecar -> None", PP.read_pose_skeleton(plain) is None)

(tmp / "plain.fdlc.toml").write_text(
    'bodyparts=["head_nose","head_mid"]\nskeleton=[["head_nose","head_mid"]]\n')
sc = PP.read_pose_skeleton(plain)
check("FreeDLC .fdlc.toml sidecar still found for a bare <stem>.parquet",
      sc is not None and sc["edges"] == [("head_nose", "head_mid")])
(tmp / "rec1.fdlc.toml").write_text(
    'bodyparts=["head_nose","head_mid"]\nskeleton=[["head_nose","head_mid"]]\n')
check("embedded skeleton takes precedence over a sidecar",
      PP.read_pose_skeleton(dlc)["edges"] == EDGES)

# -------------------------------------------------------------- bodyparts ---
check("bodyparts from embedded metadata (file order)",
      PP.pose_parquet_bodyparts(dlc) == BPS)
check("bodyparts from the bodypart column when no metadata",
      PP.pose_parquet_bodyparts(plain) == BPS)
check("bodyparts from wide columns", PP.pose_parquet_bodyparts(wide) == ["head_nose"])

# ------------------------------------------------------------ embed helper ---
emb = tmp / "emb.parquet"
long_df().to_parquet(emb)
before = pd.read_parquet(emb)
PP.embed_pose_metadata(emb, EDGES)
check("embed_pose_metadata adds the skeleton",
      PP.read_pose_skeleton(emb)["edges"] == EDGES)
check("embed_pose_metadata leaves the data unchanged",
      pd.read_parquet(emb).equals(before))
check("embed_pose_metadata keeps pandas metadata",
      b"pandas" in pq.read_schema(emb).metadata)
try:
    PP.embed_pose_metadata(emb, [("head_nose", "nose_typo")])
    check("embedding an edge with an unknown marker is refused", False)
except ValueError as exc:
    check("embedding an edge with an unknown marker is refused",
          "nose_typo" in str(exc))

cli = tmp / "cli.parquet"
long_df().to_parquet(cli)
rc = PP.main([str(cli), "--skeleton-from", str(dlc)])
check("CLI embeds a skeleton taken from another parquet",
      rc == 0 and PP.read_pose_skeleton(cli)["edges"] == EDGES)

# --------------------------------------------------------------- discovery ---
src = tmp / "dlc_out"
src.mkdir()
write_dlc(src / "sessA.parquet")
write_dlc(src / "sessB.fdlc.parquet")
pd.DataFrame({"head_nose_x": [1.0]}).to_parquet(src / "sessA_smoothed_v2.parquet")
(src / "sub").mkdir()
write_dlc(src / "sub" / "nested.parquet")
found = [os.path.basename(p) for p in PP.find_long_pose_parquets(src)]
check("discovery: DLC and FreeDLC long parquets, wide/smoothed and "
      "subfolders ignored",
      found == ["sessA.parquet", "sessB.fdlc.parquet"])

# ---------------------------------------------------------------- importer ---
from mufasa.project_layout import read_skeleton  # noqa: E402
from mufasa.utils.config_creator import ProjectConfigCreator  # noqa: E402


def new_project(name, bps=BPS):
    return ProjectConfigCreator(
        project_path=str(tmp), project_name=name, target_list=[],
        pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
        animal_cnt=1, file_type="parquet", body_parts=list(bps),
    ).config_path


from mufasa.pose_importers.fdlc_parquet_importer import (  # noqa: E402
    FDLCParquetImporter,
    read_fdlc_skeleton,
)

check("importer re-exports read_fdlc_skeleton from the light module",
      read_fdlc_skeleton is PP.read_fdlc_skeleton)

only_dlc = tmp / "only_dlc"
only_dlc.mkdir()
write_dlc(only_dlc / "videoX.parquet")
pd.DataFrame({"a_x": [1.0]}).to_parquet(only_dlc / "notes.parquet")
cfg = new_project("p_import")
imp = FDLCParquetImporter(config_path=cfg, data_folder=str(only_dlc))
check("importer discovers <video>.parquet by schema, skips other parquets",
      [os.path.basename(p) for p in imp.input_data_paths] == ["videoX.parquet"])
imp.run()
pose_dir = Path(cfg).parent / "sources" / "pose"
out = pose_dir / "videoX.parquet"
check("imported under the video stem", out.is_file())
df = pd.read_parquet(out)
check("imported file is the wide IMPORTED_POSE layout, all frames",
      isinstance(df.columns, pd.MultiIndex) and df.shape == (40, 15))
psk = read_skeleton(cfg)
check("embedded skeleton written to project.toml [skeleton]",
      psk is not None and psk["edges"] == EDGES)

cfg2 = new_project("p_files")
two = tmp / "two"
two.mkdir()
write_dlc(two / "keep.parquet")
write_dlc(two / "skip.parquet")
imp2 = FDLCParquetImporter(config_path=cfg2, data_folder=str(two),
                           file_paths=[str(two / "keep.parquet")])
check("file_paths restricts the import to the given files",
      [os.path.basename(p) for p in imp2.input_data_paths] == ["keep.parquet"])
try:
    FDLCParquetImporter(config_path=cfg2, data_folder=str(two),
                        file_paths=[str(wide)])
    check("file_paths rejects a non-long parquet", False)
except Exception as exc:
    check("file_paths rejects a non-long parquet", "wide.parquet" in str(exc))

# ------------------------------------------------------- project creation ---
from mufasa.pose_importers.dlc_autodetect import extract_bodyparts  # noqa: E402

check("extract_bodyparts reads a DLC parquet", extract_bodyparts(dlc) == BPS)

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from mufasa.ui_qt.forms.project_create import ProjectCreateForm  # noqa: E402

form = ProjectCreateForm(show_create_button=False)
form._autodetected_bps = extract_bodyparts(dlc)
form._autodetected_skeleton = form._skeleton_of(str(dlc), form._autodetected_bps)
check("create form keeps the file's skeleton", form._autodetected_skeleton == EDGES)
partial = form._skeleton_of(str(dlc), BPS[:3])
check("create form drops edges to markers the project won't have",
      partial == EDGES[:2])
form._dir_edit.setText(str(tmp))
form._name_edit.setText("p_created")
created = form.submit()
csk = read_skeleton(created) if created else None
check("created project carries the detected skeleton",
      csk is not None and csk["edges"] == EDGES and csk["nodes"] == BPS)
form._clear_autodetect()
check("clearing autodetect clears the skeleton",
      form._autodetected_skeleton == [])

# --------------------------------------------------------- session ingest ---
from mufasa.model.session_ingest import ingest_sessions  # noqa: E402

cfg3 = new_project("p_ingest")
batch = tmp / "batch"
batch.mkdir()
write_dlc(batch / "newsess.parquet")
write_dlc(batch / "otherpose.parquet", bps=["nose"] + BPS[1:],
          edges=[("nose", "head_mid")])
rep = ingest_sessions(cfg3, str(batch), smooth=False)
pose3 = Path(cfg3).parent / "sources" / "pose"
check("ingest accepts the matching DLC parquet, rejects the other pose",
      [os.path.basename(p) for p in rep["accepted"]] == ["newsess.parquet"]
      and "otherpose.parquet" in {os.path.basename(k) for k in rep["rejected"]})
got = pd.read_parquet(pose3 / "newsess.parquet")
check("ingest imports DLC parquet through the importer (wide, not copied raw)",
      isinstance(got.columns, pd.MultiIndex))
check("ingest does not import the rejected file from the same folder",
      not (pose3 / "otherpose.parquet").exists())

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122ib_dlc_parquet: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
