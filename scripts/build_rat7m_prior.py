#!/usr/bin/env python3
"""Build ``mufasa/assets/models/pose3d/rat7m_trunk_prior.npz`` from the
Rat7M motion-capture files.

Rat7M (Dunn et al. 2021, Nature Methods 18:564–573, "Geometric deep
learning enables 3D kinematic profiling across species and environments";
CC BY 4.0) — Figshare item 13739233, "Motion Capture Data and Camera
Parameters": ``mocap-s<subject>-d<day>.mat`` with 20 markers in mm,
z up, floor at z = 0.

The prior keeps a random subsample of frames (all subjects) with:

* ``features`` — what a single overhead camera gives reliably:
  head height (HeadB z, mm), head pitch (HeadB→HeadF elevation, deg),
  horizontal HeadB→hip-midpoint distance (mm);
* ``targets`` — heights (mm) of SpineF, SpineM, SpineL and the hip
  midpoint;
* ``lumbar`` — SpineL→SpineM elevation (deg);
* ``roll`` — head roll from the HeadL marker (deg);
* ``subject`` — Rat7M subject id (for leave-one-rat-out checks);
* ``bend_*`` — percentiles of the spine bends at SpineM and SpineF by
  head height (deg; + = bending upward/backward).

    python scripts/build_rat7m_prior.py /path/to/mocap_dir [--rows 80000] [--per-rat] [--workers 8]

122lq: ``--per-rat`` takes an equal share of the rows from every rat (a rat with more sessions
no longer outweighs the others); ``--pattern`` / ``--subject-regex`` read other file names;
``--workers`` reads the files in parallel.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

MARKERS = ['HeadF', 'HeadB', 'HeadL', 'SpineF', 'SpineM', 'SpineL', 'Offset1',
           'Offset2', 'HipL', 'HipR', 'ElbowL', 'ArmL', 'ShoulderL', 'ShoulderR',
           'ElbowR', 'ArmR', 'KneeR', 'KneeL', 'ShinL', 'ShinR']
I = {n: i for i, n in enumerate(MARKERS)}
NEED = [I[n] for n in ('HeadF', 'HeadB', 'SpineF', 'SpineM', 'SpineL', 'HipL', 'HipR')]
BEND_EDGES = np.array([0, 50, 80, 110, 150, 200, 400.0])
BEND_PCTS = np.array([5, 50, 95])


def load_mocap(path: str) -> np.ndarray:
    import scipy.io as sio
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["mocap"]
    return np.stack([np.asarray(getattr(m, n), float) for n in MARKERS], 1)


def elev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = b - a
    return np.degrees(np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1])))


def rows_from(X: np.ndarray) -> dict:
    hb, hf = X[:, I['HeadB']], X[:, I['HeadF']]
    hip = (X[:, I['HipL']] + X[:, I['HipR']]) / 2
    sF, sM, sL = X[:, I['SpineF']], X[:, I['SpineM']], X[:, I['SpineL']]
    feats = np.c_[hb[:, 2], elev(hb, hf), np.hypot(*(hb[:, :2] - hip[:, :2]).T)]
    targ = np.c_[sF[:, 2], sM[:, 2], sL[:, 2], hip[:, 2]]
    hl = X[:, I['HeadL']]
    ax = hf - hb
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    lat = hl - hb
    lat -= np.einsum("ij,ij->i", lat, ax)[:, None] * ax
    with np.errstate(invalid="ignore"):
        lat /= np.linalg.norm(lat, axis=1, keepdims=True)
        roll = np.degrees(np.arcsin(np.clip(lat[:, 2], -1, 1)))
    return dict(features=feats, targets=targ, lumbar=elev(sL, sM), roll=roll,
                bendM=elev(sM, sF) - elev(sL, sM), bendF=elev(sF, hb) - elev(sM, sF))


def _one(args):
    p, rx = args
    X = load_mocap(p)
    ok = np.isfinite(X[:, NEED]).all((1, 2))
    r = rows_from(X[ok])
    m = re.search(rx, os.path.basename(p))
    if m is None:
        raise ValueError(f"{os.path.basename(p)}: no rat number (subject regex {rx!r})")
    r["subject"] = np.full(ok.sum(), int(m.group(1)), np.int16)
    return p, int(ok.sum()), r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mocap_dir")
    ap.add_argument("--rows", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pattern", default="mocap-s*-d*.mat", help="the files to read (glob, in mocap_dir and below)")
    ap.add_argument("--subject-regex", default=r"-s(\d+)-", help="the rat's number in a file's name (group 1)")
    ap.add_argument("--per-rat", action="store_true", help="an equal share of the rows from every rat")
    ap.add_argument("--workers", type=int, default=1, help="files read at once")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "mufasa", "assets", "models", "pose3d",
        "rat7m_trunk_prior.npz"))
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.mocap_dir, "**", a.pattern), recursive=True))
    if not files:
        print(f"no {a.pattern} files found", file=sys.stderr)
        return 2
    jobs = [(p, a.subject_regex) for p in files]
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            done = list(ex.map(_one, jobs))
    else:
        done = [_one(j) for j in jobs]
    parts = []
    for p, n, r in done:
        parts.append(r)
        print(f"{os.path.basename(p)}: {n} usable frames (rat {int(r['subject'][0]) if n else '?'})")
    allr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    hb = allr["features"][:, 0]
    bend = {}
    for key in ("bendM", "bendF"):
        tab = np.array([[np.percentile(v, q) if v.size else np.nan for q in BEND_PCTS]
                        for v in (allr[key][(hb >= BEND_EDGES[i]) & (hb < BEND_EDGES[i + 1])]
                                  for i in range(len(BEND_EDGES) - 1))])
        bend[key] = tab
    rng = np.random.default_rng(a.seed)
    if a.per_rat:
        rats = np.unique(allr["subject"])
        share = a.rows // len(rats)
        pick = np.sort(np.concatenate([rng.choice(ix, min(share, len(ix)), replace=False)
                                       for ix in (np.flatnonzero(allr["subject"] == s) for s in rats)]))
        print(f"{len(rats)} rats, up to {share} rows each")
    else:
        pick = np.sort(rng.choice(len(hb), min(a.rows, len(hb)), replace=False))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out,
        features=allr["features"][pick].astype(np.float32),
        targets=allr["targets"][pick].astype(np.float32),
        lumbar=allr["lumbar"][pick].astype(np.float32),
        roll=allr["roll"][pick].astype(np.float32),
        subject=allr["subject"][pick],
        feature_names=np.array(["head_height_mm", "head_pitch_deg", "head_to_hips_horizontal_mm"]),
        target_names=np.array(["SpineF_z", "SpineM_z", "SpineL_z", "hips_z"]),
        bend_edges=BEND_EDGES, bend_percentiles=BEND_PCTS,
        bend_spineM=bend["bendM"], bend_spineF=bend["bendF"],
        source=np.array("Rat7M motion capture (Dunn et al. 2021, Nature Methods; CC BY 4.0), "
                        "Figshare 13739233; subsampled by scripts/build_rat7m_prior.py"),
    )
    print(f"wrote {a.out}: {len(pick)} frames from {len(files)} sessions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
