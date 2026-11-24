#!/usr/bin/env python3
"""Unpack TactileArena runway recordings into the lab's source tree (122lz).

Each recording comes as a zip (``3-23-2026_rat2.zip``) holding

    3-23-2026/runway/rat2/runway_tactile_data_20260323_104422_M200_T700_D100.mat
    3-23-2026/runway/rat2/camera_0_20260323_104251.mp4

This script

1. unzips it to a staging folder beside the destination;
2. converts the ``.mat`` (MATLAB v5: per board, per frame, the M strongest pixels) to one parquet
   table, ``…tactile-grid.parquet``;
3. moves the video, renamed;
4. writes them to

   <tactile-root>/<lab>/<lab>-<who>-<rat>/<lab>-<who>-<rat>-<date>/<lab>-<who>-<rat>-<date>-<time>.M200_T700.tactile-grid.parquet
   <video-root>/<lab>/<lab>-<who>-<rat>/<lab>-<who>-<rat>-<date>/<lab>-<who>-<rat>-<date>-<time>.cam0.mp4

with ``lab`` thakorN, ``who`` wl, ``rat`` the zip's ``ratN`` (``rat2``), ``date`` YYYYMMDD from the
folder (``3-23-2026`` → ``20260323``), and ``time`` HHMMSS. The time is the video's start
(``camera_0_20260323_104251`` → ``104251``), the start of the session, which both files share. The
``.mat``'s own name carries the time it was saved, at the end of the recording (``104422``).

The tactile table (one row per board per frame, as recorded)
------------------------------------------------------------
=============  =================  ==============================================================
column         type               meaning
=============  =================  ==============================================================
board          uint8              the board as named in the file (``Board1`` → 1)
board_id       uint16             the board's hardware id the frame came from (``BoardN_board_ids``)
frame          uint32             the frame's number on its board, from 0
t              float64            seconds on the recording's shared clock (``BoardN_timestamps``)
pixel          list<uint16>       the pixels reported with a value above 0, by their index on the
                                  board's 64 × 64 grid (0–4095), as recorded
value          list<uint16>       their values (≈ 0–3,950; a few 65,280 glitches are kept as they are)
=============  =================  ==============================================================

Pixels reported at 0 are left out (they carry nothing), so each list holds only the pressed and
near-pressed pixels. Frames where nothing was pressed keep their row with empty lists, so every
timestamp is kept and the frame rate and the gaps can be seen.

The file's schema metadata (key ``tactile_grid``, JSON) has everything else in the ``.mat``:

* ``M``, ``threshold``, ``digipot``, ``duration``, ``shared_start_time`` (Unix time), the grid
  (64 × 64), each board's ``buffer_overflows``;
* both cameras' ``frame_timestamps`` (on the same clock), ``frame_indices``, ``total_frames``,
  ``dropped_frames``, ``actual_fps`` and ``filename``, for syncing the video;
* the source zip, ``.mat`` and video, and the session's names.

Read it back with :func:`read_tactile_grid` (``python scripts/import_tactilearena.py --show FILE``).

Usage
-----
::

    python scripts/import_tactilearena.py /data/downloads/*.zip \\
        --tactile-root /cloud/server/source/tactile-grid --video-root /cloud/server/source/video
    python scripts/import_tactilearena.py /data/downloads/ --dry-run      # say what it would do
    python scripts/import_tactilearena.py … --workers 4                   # several zips at once

* A zip whose outputs both exist is skipped (``--overwrite`` redoes it).
* The zips are only read, never changed.
* Outputs are written to a staging name and renamed when complete, so an interrupted run leaves
  no half-written files.
* Each run appends to ``<tactile-root>/<lab>/import_manifest.csv``: the zip, the outputs, sizes,
  frame counts and when.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

META_KEY = b"tactile_grid"
GRID = 64
MAT_RE = re.compile(r"runway_tactile_data_(\d{8})_(\d{6})_M(\d+)_T(\d+)_D(\d+)\.mat$")
CAM_RE = re.compile(r"camera_(\d+)_(\d{8})_(\d{6})\.mp4$")
DATE_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")
RAT_RE = re.compile(r"^rat\d+$", re.I)

__all__ = ["Session", "convert_mat", "plan", "read_tactile_grid", "session_of", "main"]


@dataclass
class Session:
    """One zip: what is in it and where it goes."""
    zip_path: Path
    mat: str                 # member names in the zip
    video: str
    rat: str                 # rat2
    date: str                # 20260323
    time: str                # 104251 (the video's start)
    cam: int                 # 0
    M: int
    T: int
    D: int
    stem: str                # thakorN-wl-rat2-20260323-104251
    tactile_out: Path
    video_out: Path


def session_of(zip_path: Path, tactile_root: Path, video_root: Path, lab: str = "thakorN",
               who: str = "wl") -> Session:
    """Read a zip's listing and work out the session's names and destinations."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
    mats = [n for n in names if MAT_RE.search(Path(n).name)]
    vids = [n for n in names if CAM_RE.search(Path(n).name)]
    if len(mats) != 1 or len(vids) != 1:
        raise ValueError(f"{zip_path.name}: expected one runway_tactile_data_*.mat and one camera_*.mp4, "
                         f"found {len(mats)} and {len(vids)} ({names})")
    parts = Path(mats[0]).parts                       # 3-23-2026/runway/rat2/…mat
    rat = next((p for p in reversed(parts[:-1]) if RAT_RE.match(p)), None)
    datef = next((p for p in parts[:-1] if DATE_RE.match(p)), None)
    if rat is None or datef is None:
        raise ValueError(f"{zip_path.name}: no <M-D-YYYY>/…/ratN/ folders in {mats[0]}")
    mo, dy, yr = DATE_RE.match(datef).groups()
    date = f"{int(yr):04d}{int(mo):02d}{int(dy):02d}"
    cm = CAM_RE.search(Path(vids[0]).name)
    cam, cdate, ctime = int(cm.group(1)), cm.group(2), cm.group(3)
    mm = MAT_RE.search(Path(mats[0]).name)
    if cdate != date:
        raise ValueError(f"{zip_path.name}: the folder's date {date} and the video's {cdate} differ")
    rat = rat.lower()
    animal = f"{lab}-{who}-{rat}"
    day = f"{animal}-{date}"
    stem = f"{day}-{ctime}"
    M, T, D = int(mm.group(3)), int(mm.group(4)), int(mm.group(5))
    return Session(zip_path, mats[0], vids[0], rat, date, ctime, cam, M, T, D, stem,
                   tactile_root / lab / animal / day / f"{stem}.M{M}_T{T}.tactile-grid.parquet",
                   video_root / lab / animal / day / f"{stem}.cam{cam}.mp4")


def _scalar(d, k, cast=float):
    v = d.get(k)
    return None if v is None else cast(np.asarray(v).ravel()[0])


def convert_mat(mat_path: Path, out: Path, meta_extra: dict) -> dict:
    """The ``.mat`` as a tactile-grid parquet (the table in the module docstring). Returns counts."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import scipy.io as sio

    d = sio.loadmat(str(mat_path))
    boards = sorted(int(m.group(1)) for k in d if (m := re.match(r"Board(\d+)_timestamps$", k)))
    tables, counts, overflows, vmax = [], {}, {}, 0
    for b in boards:
        t = np.asarray(d[f"Board{b}_timestamps"], float).ravel()
        idx = d[f"Board{b}_pixel_indices"].ravel()
        val = d[f"Board{b}_pixel_values"].ravel()
        ids = np.asarray(d[f"Board{b}_board_ids"]).ravel().astype(np.uint16)
        n = len(t)
        offsets = np.zeros(n + 1, np.int64)
        pix_parts, val_parts = [], []
        for i in range(n):
            p = np.asarray(idx[i]).ravel()
            v = np.asarray(val[i]).ravel()
            k = v > 0
            p, v = p[k], v[k]
            pix_parts.append(p)
            val_parts.append(v)
            offsets[i + 1] = offsets[i] + len(v)
        pix = np.concatenate(pix_parts) if pix_parts else np.zeros(0)
        vv = np.concatenate(val_parts) if val_parts else np.zeros(0)
        if len(pix) and (pix.min() < 0 or pix.max() >= GRID * GRID):
            raise ValueError(f"Board{b}: pixel index {pix.min()}…{pix.max()} outside the {GRID}×{GRID} grid")
        if len(vv) and vv.max() > np.iinfo(np.uint16).max:
            raise ValueError(f"Board{b}: value {vv.max()} beyond uint16")
        vmax = max(vmax, int(vv.max()) if len(vv) else 0)
        off = pa.array(offsets.astype(np.int32))
        tables.append(pa.table({
            "board": pa.array(np.full(n, b, np.uint8)),
            "board_id": pa.array(ids if len(ids) == n else np.full(n, ids[0] if len(ids) else 0, np.uint16)),
            "frame": pa.array(np.arange(n, dtype=np.uint32)),
            "t": pa.array(t),
            "pixel": pa.ListArray.from_arrays(off, pa.array(pix.astype(np.uint16))),
            "value": pa.ListArray.from_arrays(off, pa.array(vv.astype(np.uint16))),
        }))
        counts[b] = n
        overflows[b] = _scalar(d, f"Board{b}_buffer_overflows", int)
    table = pa.concat_tables(tables)
    cams = {}
    for c in (0, 1):
        if f"camera{c}_frame_timestamps" not in d:
            continue
        fn = d.get(f"camera{c}_filename")
        cams[str(c)] = {
            "frame_timestamps": np.asarray(d[f"camera{c}_frame_timestamps"], float).ravel().tolist(),
            "frame_indices": np.asarray(d.get(f"camera{c}_frame_indices", []), int).ravel().tolist(),
            "total_frames": _scalar(d, f"camera{c}_total_frames", int),
            "dropped_frames": _scalar(d, f"camera{c}_dropped_frames", int),
            "actual_fps": _scalar(d, f"camera{c}_actual_fps"),
            "filename": str(np.asarray(fn).ravel()[0]) if fn is not None and np.size(fn) else None,
        }
    meta = {"format": "tactile-grid", "version": 1, "grid": [GRID, GRID], "pixel_index": "0-4095 on the 64 x 64 grid, as recorded",
            "M": _scalar(d, "M", int), "threshold": _scalar(d, "threshold", int), "digipot": _scalar(d, "digipot", int),
            "duration": _scalar(d, "duration"), "shared_start_time": _scalar(d, "shared_start_time"),
            "boards": boards, "frames": {str(k): v for k, v in counts.items()},
            "buffer_overflows": {str(k): v for k, v in overflows.items()}, "max_value": vmax,
            "cameras": cams, **meta_extra}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), META_KEY: json.dumps(meta).encode()})
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd", row_group_size=50_000)
    return {"frames": counts, "rows": table.num_rows}


def read_tactile_grid(path, boards=None, frames=None) -> tuple:
    """(table, meta): the tactile table as pandas (one row per board-frame, ``pixel``/``value`` as
    arrays) and its metadata. ``boards``: only these; ``frames``: a slice of each board's frames.

    A row as a 64 × 64 image: ``img = np.zeros(4096); img[row.pixel] = row.value; img.reshape(64, 64)``
    (whether the index runs along rows or columns is the board's own convention; TactileArena's
    code and homographies fix it).
    """
    import pyarrow.parquet as pq
    filters = [("board", "in", list(boards))] if boards else None
    t = pq.read_table(path, filters=filters)
    meta = json.loads((t.schema.metadata or {}).get(META_KEY, b"{}"))
    df = t.to_pandas()
    if frames is not None:
        lo, hi = frames.start or 0, frames.stop if frames.stop is not None else 2**32
        df = df[(df["frame"] >= lo) & (df["frame"] < hi)]
    return df, meta


def _process(s: Session, overwrite: bool) -> dict:
    """Unzip, convert, move one session. Returns a manifest row."""
    t0 = time.time()
    row = {"zip": str(s.zip_path), "stem": s.stem, "tactile": str(s.tactile_out), "video": str(s.video_out),
           "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if s.tactile_out.exists() and s.video_out.exists() and not overwrite:
        return {**row, "status": "skipped (exists)"}
    s.tactile_out.parent.mkdir(parents=True, exist_ok=True)
    s.video_out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".staging-{s.stem}-", dir=s.tactile_out.parent))
    try:
        with zipfile.ZipFile(s.zip_path) as z:
            mat = Path(z.extract(s.mat, stage))
            vid = Path(z.extract(s.video, stage))
            crc_ok = z.getinfo(s.video).file_size == vid.stat().st_size
        if not crc_ok:
            raise OSError(f"{s.video}: extracted size differs from the zip's")
        extra = {"session": s.stem, "rat": s.rat, "date": s.date, "time": s.time, "camera": s.cam,
                 "source_zip": s.zip_path.name, "source_mat": Path(s.mat).name, "source_video": Path(s.video).name,
                 "video_file": s.video_out.name}
        tmp_parquet = s.tactile_out.with_name(f".{s.tactile_out.name}.part")
        info = convert_mat(mat, tmp_parquet, extra)
        tmp_video = s.video_out.with_name(f".{s.video_out.name}.part")
        shutil.move(str(vid), tmp_video)               # across filesystems: copies, then removes
        os.replace(tmp_parquet, s.tactile_out)
        os.replace(tmp_video, s.video_out)
        return {**row, "status": "ok", "rows": info["rows"],
                "frames": " ".join(f"B{k}:{v}" for k, v in info["frames"].items()),
                "tactile_mb": round(s.tactile_out.stat().st_size / 2**20, 1),
                "video_mb": round(s.video_out.stat().st_size / 2**20, 1), "seconds": round(time.time() - t0, 1)}
    except Exception as e:                            # report it; the others carry on
        return {**row, "status": f"FAILED: {type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        for p in (s.tactile_out.with_name(f".{s.tactile_out.name}.part"),
                  s.video_out.with_name(f".{s.video_out.name}.part")):
            if p.exists() and not (s.tactile_out.exists() and s.video_out.exists()):
                p.unlink()


def plan(inputs: list[str], tactile_root: Path, video_root: Path, lab: str, who: str) -> tuple[list, list]:
    """(sessions, problems) for the given zips or folders of zips."""
    zips = []
    for x in inputs:
        p = Path(x).expanduser()
        zips += sorted(p.glob("*.zip")) if p.is_dir() else [p]
    sessions, problems, seen = [], [], {}
    for z in zips:
        try:
            s = session_of(z, tactile_root, video_root, lab, who)
        except Exception as e:
            problems.append(f"{z}: {e}")
            continue
        if s.stem in seen:
            problems.append(f"{z}: same session as {seen[s.stem]} ({s.stem}); skipped")
            continue
        seen[s.stem] = z.name
        sessions.append(s)
    return sessions, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unzip TactileArena runway recordings, convert the tactile .mat to "
                                 "parquet, and file them with their videos under the lab's naming.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Usage")[1])
    ap.add_argument("inputs", nargs="*", help="zip files, or folders of them")
    ap.add_argument("--tactile-root", default="/cloud/server/source/tactile-grid")
    ap.add_argument("--video-root", default="/cloud/server/source/video")
    ap.add_argument("--lab", default="thakorN")
    ap.add_argument("--who", default="wl", help="the experimenter's initials (Weiwen Lu: wl)")
    ap.add_argument("--dry-run", action="store_true", help="say what would be done, do nothing")
    ap.add_argument("--overwrite", action="store_true", help="redo sessions already filed")
    ap.add_argument("--workers", type=int, default=1, help="zips at once (each needs ~3 GB of memory)")
    ap.add_argument("--show", metavar="PARQUET", help="summarise a converted file and exit")
    a = ap.parse_args(argv)

    if a.show:
        df, meta = read_tactile_grid(a.show)
        print(f"{a.show}\n  session {meta.get('session')}  rat {meta.get('rat')}  date {meta.get('date')}  "
              f"camera {meta.get('camera')}  M {meta.get('M')}  threshold {meta.get('threshold')}  "
              f"duration {meta.get('duration'):.1f} s")
        for b, g in df.groupby("board"):
            n = g["value"].map(len)
            rate = (len(g) - 1) / (g["t"].iloc[-1] - g["t"].iloc[0])
            print(f"  Board{b} (id {g['board_id'].iloc[0]}): {len(g)} frames, {rate:.0f} Hz, "
                  f"{(n > 0).mean():.0%} with pixels, median {int(n.median())} pixels")
        for c, cam in meta.get("cameras", {}).items():
            ts = cam["frame_timestamps"]
            print(f"  camera{c}: {len(ts)} frames, {(len(ts) - 1) / (ts[-1] - ts[0]):.2f} fps from its timestamps")
        return 0
    if not a.inputs:
        ap.error("give zip files or folders")

    sessions, problems = plan(a.inputs, Path(a.tactile_root), Path(a.video_root), a.lab, a.who)
    for p in problems:
        print(f"! {p}", file=sys.stderr)
    for s in sessions:
        state = "exists" if s.tactile_out.exists() and s.video_out.exists() else "new"
        print(f"{s.zip_path.name}  →  {s.stem}  [{state}]\n    {s.tactile_out}\n    {s.video_out}")
    if a.dry_run or not sessions:
        print(f"{len(sessions)} session(s){' (dry run: nothing done)' if a.dry_run else ''}; {len(problems)} problem(s)")
        return 0 if not problems else 1

    manifest = Path(a.tactile_root) / a.lab / "import_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(_process, s, a.overwrite): s for s in sessions}
            for f in as_completed(futs):
                rows.append(f.result())
                print(f"{rows[-1]['stem']}: {rows[-1]['status']}", flush=True)
    else:
        for s in sessions:
            rows.append(_process(s, a.overwrite))
            print(f"{rows[-1]['stem']}: {rows[-1]['status']}", flush=True)
    fields = ["when", "zip", "stem", "status", "tactile", "video", "rows", "frames", "tactile_mb", "video_mb", "seconds"]
    new = not manifest.exists()
    with manifest.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["status"].startswith("FAILED")]
    print(f"{sum(r['status'] == 'ok' for r in rows)} filed, {sum(r['status'].startswith('skipped') for r in rows)} "
          f"skipped, {len(bad)} failed; manifest: {manifest}")
    return 1 if bad or problems else 0


if __name__ == "__main__":
    sys.exit(main())
