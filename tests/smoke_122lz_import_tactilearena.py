"""Smoke test for patch 122lz — ``scripts/import_tactilearena.py``: TactileArena runway zips
unzipped, the tactile ``.mat`` converted to parquet, both filed under the lab's naming.

User: "I would like to unzip, rename, convert and move the files
/cloud/server/source/tactile-grid/thakorN/thakorN-wl-<ratid>/thakorN-wl-<ratid>-<date>/
thakorN-wl-<ratid>-<date>-<time>.M200_T700.tactile-grid.parquet and
/cloud/server/source/video/thakorN/…/thakorN-wl-<ratid>-<date>-<time>.cam0.mp4".

Synthetic recordings in the TactileArena layout (``3-23-2026/runway/rat2/…``):

* the names: rat, date (YYYYMMDD from the folder), the session's time from the video's start,
  the camera; M and T from the ``.mat``'s name;
* the parquet holds every frame of every board, its time and every non-zero pixel exactly, with
  the cameras' timestamps and the settings in its metadata; the video is moved unchanged;
* a second run skips what is filed; ``--overwrite`` redoes it; ``--dry-run`` does nothing;
* a zip that isn't a recording (Rat7M's) is refused with a reason, the others carry on;
* a manifest row per session; nothing left in staging.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import import_tactilearena as TA  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def recording(td: Path, folder: str, rat: str, cam: int, vtime: str, mtime: str, seed: int) -> tuple[Path, dict]:
    r = np.random.default_rng(seed)
    mo, dy, yr = folder.split("-")
    ymd = f"{yr}{int(mo):02d}{int(dy):02d}"
    d = {"M": np.int64(200), "threshold": np.int64(700), "digipot": np.int64(100), "duration": 3.5,
         "shared_start_time": 1.77e9}
    truth = {}
    for b in (1, 2, 3, 4):
        n = 300 + 10 * b
        t = np.sort(r.uniform(0, 3.5, n))
        idx = np.empty((1, n), object)
        val = np.empty((1, n), object)
        frames = []
        for i in range(n):
            p = r.choice(4096, 200, replace=False).astype(np.int64)
            v = np.where(r.random(200) < 0.15, r.integers(1, 3950, 200), 0).astype(np.int64)
            if i == 5:
                v[:] = 0                                    # a frame with nothing pressed
            if i == 7:
                v[0] = 65280                                # the sensors' glitch value
            idx[0, i], val[0, i] = p[None, :], v[None, :]
            frames.append((p[v > 0], v[v > 0]))
        d[f"Board{b}_timestamps"] = t[None, :]
        d[f"Board{b}_pixel_indices"] = idx
        d[f"Board{b}_pixel_values"] = val
        d[f"Board{b}_board_ids"] = np.full((1, n), 5 - b, np.uint16)
        d[f"Board{b}_buffer_overflows"] = np.int64(0)
        truth[b] = (t, frames)
    for c in (0, 1):
        d[f"camera{c}_frame_timestamps"] = (np.arange(100) / 30.0 + 0.02 * c)[None, :]
        d[f"camera{c}_frame_indices"] = np.arange(100, dtype=np.int32)[None, :]
        d[f"camera{c}_total_frames"] = np.int64(100)
        d[f"camera{c}_dropped_frames"] = np.int64(0)
        d[f"camera{c}_actual_fps"] = 28.5
        d[f"camera{c}_filename"] = f"camera_{c}_{ymd}_{vtime}.mp4"
    mat = td / f"runway_tactile_data_{ymd}_{mtime}_M200_T700_D100.mat"
    sio.savemat(mat, d)
    video = bytes(r.integers(0, 256, 5000, dtype=np.uint8))
    z = td / f"{folder}_{rat}.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(mat, f"{folder}/runway/{rat}/{mat.name}")
        zf.writestr(f"{folder}/runway/{rat}/camera_{cam}_{ymd}_{vtime}.mp4", video)
    mat.unlink()
    return z, {"truth": truth, "video": video}


def run(*args) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = TA.main([str(a) for a in args])
    return rc, out.getvalue() + err.getvalue()


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "downloads"
        src.mkdir()
        z1, k1 = recording(src, "3-23-2026", "rat2", 0, "104251", "104422", 1)
        z2, _ = recording(src, "2-11-2026", "rat1", 1, "200452", "200627", 2)
        with zipfile.ZipFile(src / "13739233.zip", "w") as zf:   # not a recording
            zf.writestr("mocap-s1-d1.mat", b"x")
        troot, vroot = td / "source" / "tactile-grid", td / "source" / "video"
        roots = ["--tactile-root", troot, "--video-root", vroot]

        rc, out = run(src, *roots, "--dry-run")
        check(f"--dry-run: says where each goes, does nothing (exit {rc}: a zip refused)",
              rc == 1 and "thakorN-wl-rat2-20260323-104251" in out and not troot.exists())

        rc, out = run(src, *roots)
        tp = troot / "thakorN/thakorN-wl-rat2/thakorN-wl-rat2-20260323/thakorN-wl-rat2-20260323-104251.M200_T700.tactile-grid.parquet"
        vp = vroot / "thakorN/thakorN-wl-rat2/thakorN-wl-rat2-20260323/thakorN-wl-rat2-20260323-104251.cam0.mp4"
        vp2 = vroot / "thakorN/thakorN-wl-rat1/thakorN-wl-rat1-20260211/thakorN-wl-rat1-20260211-200452.cam1.mp4"
        check("filed under the lab's naming: rat, YYYYMMDD, the video's start time (not the .mat's save time), "
              "the camera, M and T", tp.is_file() and vp.is_file() and vp2.is_file())
        check("the zip that isn't a recording is refused with a reason; the others carry on",
              "13739233.zip" in out and "expected one runway_tactile_data" in out and "2 filed" in out)
        check("the video moved unchanged; the zips untouched", vp.read_bytes() == k1["video"] and z1.is_file()
              and z2.is_file())

        df, meta = TA.read_tactile_grid(tp)
        exact = True
        for b, (t, frames) in k1["truth"].items():
            g = df[df.board == b].sort_values("frame")
            exact &= len(g) == len(t) and np.array_equal(g.t.to_numpy(), t)
            for i, (p, v) in enumerate(frames):
                r = g.iloc[i]
                exact &= np.array_equal(np.asarray(r.pixel), p) and np.array_equal(np.asarray(r.value), v)
        empty = df[(df.board == 1) & (df.frame == 5)]["value"].iloc[0]
        check(f"the parquet: every frame of all {df.board.nunique()} boards ({len(df)}), its time, every non-zero "
              f"pixel exactly; an empty frame kept ({len(empty)} pixels); the 65,280 glitch kept; board ids",
              exact and len(empty) == 0 and meta["max_value"] == 65280
              and sorted(df[df.board == 1].board_id.unique()) == [4])
        check("its metadata: M, threshold, digipot, duration, both cameras' timestamps, the session's names",
              meta["M"] == 200 and meta["threshold"] == 700 and meta["digipot"] == 100
              and len(meta["cameras"]["0"]["frame_timestamps"]) == 100 and meta["camera"] == 0
              and meta["session"] == "thakorN-wl-rat2-20260323-104251" and meta["source_mat"].endswith("104422_M200_T700_D100.mat")
              and json.dumps(meta))

        before = tp.stat().st_mtime_ns
        rc, out = run(z1, *roots)
        rc2, out2 = run(z1, *roots, "--overwrite")
        check("again: skipped (already filed); --overwrite redoes it",
              "skipped" in out and "1 filed" in out2 and tp.stat().st_mtime_ns != before)
        rows = list(csv.DictReader((troot / "thakorN" / "import_manifest.csv").open()))
        leftovers = [p for p in troot.rglob("*") if p.name.startswith(".")]
        check(f"the manifest: a row per session and run ({len(rows)}); nothing left in staging",
              len(rows) == 4 and {r["status"] for r in rows} >= {"ok"} and not leftovers)

        rc, out = run("--show", tp)
        check("--show summarises a converted file", rc == 0 and "Board1 (id 4)" in out and "camera0: 100 frames" in out)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lz_import_tactilearena: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
