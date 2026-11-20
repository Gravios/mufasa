"""Gait bounds from uninjured rats on a CatWalk (Neckel et al. 2013).

Reads ``Neckel_CatWalk_RRMPS.mat`` (Mendeley Data sg3d7dv345; MATLAB 7.3, also as the split
parts ``..._mat_part.aa``, ``.ab``, … read in place), takes the pre-injury runs, and writes
``mufasa/assets/gait/catwalk_rat_reference.json``: the fitted laws of
:data:`mufasa.pose3d.gait.BOUNDS`, medians by speed, percentiles and interlimb phases.

    python scripts/calibrate_gait_catwalk.py PATH_OR_GLOB [-o OUT.json]

The file's ``CW{rat, week, limb, run}`` holds each run's footfalls of a limb (x, y in mm;
initial contact and toe-off in s); limbs right fore, right hind, left fore, left hind; week 1
is before the injury.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

LIMB = {0: "RF", 1: "RH", 2: "LF", 3: "LH"}
SPEED_BINS = [100, 200, 250, 300, 350, 400, 450, 500, 600, 800]
OUT = Path(__file__).resolve().parents[1] / "mufasa/assets/gait/catwalk_rat_reference.json"


class _Parts(io.RawIOBase):
    """Split files read as one."""

    def __init__(self, paths):
        self.fh = [open(p, "rb") for p in paths]
        self.starts = [0]
        for p in paths:
            self.starts.append(self.starts[-1] + os.path.getsize(p))
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.starts[-1] + off
        return self.pos

    def readinto(self, buf):
        mv, got = memoryview(buf), 0
        while got < len(buf) and self.pos < self.starts[-1]:
            i = bisect.bisect_right(self.starts, self.pos) - 1
            self.fh[i].seek(self.pos - self.starts[i])
            d = self.fh[i].read(min(len(buf) - got, self.starts[i + 1] - self.pos))
            mv[got:got + len(d)] = d
            got += len(d)
            self.pos += len(d)
        return got


def footfalls(path: str, week: int = 0) -> pd.DataFrame:
    """Footfalls of ``week`` (0: before the injury): rat, run, paw, x, y, t_on, t_off."""
    import h5py
    paths = sorted(glob.glob(path)) or [path]
    f = h5py.File(_Parts(paths) if len(paths) > 1 else paths[0], "r")
    cw = f["CW"][()]                                   # stored transposed: (run, limb, week, rat)
    rows = []
    for rat in range(cw.shape[3]):
        for limb in range(cw.shape[1]):
            for run in range(cw.shape[0]):
                o = f[cw[run, limb, week, rat]]
                if o.attrs.get("MATLAB_empty", 0) or o.ndim != 2:
                    continue
                for x, y, ton, toff in o[()].T:
                    rows.append((rat, run, LIMB[limb], x, y, ton, toff))
    d = pd.DataFrame(rows, columns=["rat", "run", "paw", "x", "y", "t_on", "t_off"])
    return d.dropna()


def strides(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["rat", "run", "paw", "t_on"]).copy()
    g = d.groupby(["rat", "run", "paw"])
    d["stance_s"] = d.t_off - d.t_on
    d["stride_s"] = g.t_on.shift(-1) - d.t_on
    d["swing_s"] = g.t_on.shift(-1) - d.t_off
    d["stride_mm"] = np.hypot(g.x.shift(-1) - d.x, g.y.shift(-1) - d.y)
    d["speed"] = d.stride_mm / d.stride_s
    d["duty"] = d.stance_s / d.stride_s
    bos = np.full(len(d), np.nan)
    pos = {k: i for i, k in enumerate(d.index)}
    for _, r in d.groupby(["rat", "run"]):
        for a, b in (("RF", "LF"), ("LF", "RF"), ("RH", "LH"), ("LH", "RH")):
            A, B = r[r.paw == a], r[r.paw == b]
            if len(A) and len(B):
                j = np.abs(A.t_on.to_numpy()[:, None] - B.t_on.to_numpy()[None]).argmin(1)
                for i, k in zip(A.index, j):
                    bos[pos[i]] = abs(A.y[i] - B.y.iloc[k])
    d["bos_mm"] = bos
    d = d.dropna(subset=["stride_s"])
    return d[d.stride_mm.between(30, 300) & d.stride_s.between(0.15, 2) & (d.stance_s > 0) & (d.swing_s > 0)]


def phases(d: pd.DataFrame) -> dict:
    out = {}
    for a, b in (("LF", "RF"), ("LH", "RH"), ("LH", "LF"), ("LH", "RF")):
        ph = []
        for _, r in d.groupby(["rat", "run"]):
            B = r[r.paw == b].t_on.to_numpy()
            for _, s in r[r.paw == a].iterrows():
                k = B[(B >= s.t_on) & (B < s.t_on + s.stride_s)]
                if len(k):
                    ph.append((k[0] - s.t_on) / s.stride_s)
        out[f"{a}_{b}"] = [round(float(q), 3) for q in np.percentile(ph, [25, 50, 75])]
    return out


def calibrate(s: pd.DataFrame) -> dict:
    s = s.assign(fh=s.paw.str[1])
    laws, binned, pct = {}, {}, {}
    for fh, x in s.groupby("fh"):
        name = "fore" if fh == "F" else "hind"
        v = x.speed.to_numpy()
        for c in ("stance", "swing"):
            b, a = np.polyfit(np.log(v), np.log(x[f"{c}_s"]), 1)
            laws[f"{c}_law_{name}"] = [round(float(np.exp(a)), 4), round(float(b), 4)]
        x = x.assign(vb=pd.cut(x.speed, SPEED_BINS))
        t = x.groupby("vb", observed=True)[["speed", "stance_s", "swing_s", "stride_mm", "duty", "bos_mm"]].median()
        t["n"] = x.groupby("vb", observed=True).size()
        binned[name] = [{k: round(float(val), 4) for k, val in r.items()} for _, r in t.iterrows()]
        pct[name] = {c: [round(float(q), 3) for q in x[c].quantile([.01, .05, .5, .95, .99])]
                     for c in ("stance_s", "swing_s", "stride_mm", "speed", "duty", "bos_mm")}
    k, c = np.polyfit(s.speed, s.stride_mm, 1)
    laws["stride_at_rest_mm"], laws["stride_per_speed_s"] = round(float(c), 2), round(float(k), 4)
    return {"source": "Neckel et al. 2013, J Neurosci Methods 216:62; Mendeley Data sg3d7dv345 (CatWalk, "
                      "pre-injury runs, young adult female rats)",
            "rats": int(s.rat.nunique()), "strides": int(len(s)),
            "percentiles": "1, 5, 50, 95, 99", "laws": laws, "by_speed": binned, "percentile_table": pct,
            "phases_25_50_75": phases(s)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mat", help="the .mat file, or a glob of its split parts")
    ap.add_argument("-o", "--out", default=str(OUT))
    a = ap.parse_args(argv)
    ref = calibrate(strides(footfalls(a.mat)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(ref, indent=1))
    print(json.dumps(ref["laws"]), ref["rats"], ref["strides"])


if __name__ == "__main__":
    main()
