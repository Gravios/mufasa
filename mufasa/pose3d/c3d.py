"""
mufasa.pose3d.c3d
=================

Patch 122kl: motion-capture files (C3D) into Mufasa's 3-D pose format.

``read_c3d`` is a dependency-free C3D reader (NumPy only): header,
parameter section, 3-D point data in integer or floating-point form, on
Intel, DEC (VAX floats) and MIPS/SGI (big-endian) processors; labels from
``POINT:LABELS`` and its continuations (``LABELS2``…); units from
``POINT:UNITS`` (converted to millimetres); frame count past 65 535 from
``POINT:FRAMES`` / ``TRIAL:ACTUAL_*_FIELD``; points flagged invalid
(negative residual word) become NaN. Analog channels are skipped.

``c3d_to_pose3d`` turns one into a 3-D pose table — ``frame``,
``<marker>_x/_y/_z`` (mm), ``<marker>_likelihood`` (1 measured, 0 missing),
``<marker>_residual_mm`` — with the ``mufasa_pose3d`` metadata the 3-D
viewer and the 3-D Kalman smoothing read. Options: which axis is up,
renaming markers, a skeleton, a frame range.

Checked against ezc3d on vendor files (Vicon, Qualisys, Cortex, BTS,
Kistler, Optotrak, Motion Analysis…): identical points, labels and rates.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

INTEL, DEC, MIPS = 84, 85, 86
UNIT_TO_MM = {"mm": 1.0, "millimeter": 1.0, "millimeters": 1.0, "m": 1000.0, "meter": 1000.0,
              "meters": 1000.0, "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0,
              "in": 25.4, "inch": 25.4, "inches": 25.4, "dm": 100.0}


class C3DError(ValueError):
    pass


@dataclass
class C3D:
    labels: list[str]
    points: np.ndarray          # (T, M, 3) in the file's units, NaN where invalid
    residuals: np.ndarray       # (T, M) in the file's units, NaN where invalid
    rate: float                 # point frames per second
    first_frame: int            # 1-based frame number of points[0]
    units: str                  # POINT:UNITS as written
    params: dict = field(default_factory=dict)   # {GROUP: {PARAM: value}}
    processor: str = "intel"

    @property
    def to_mm(self) -> float:
        return UNIT_TO_MM.get(self.units.strip().lower(), 1.0)


# --------------------------------------------------------------------------- #
# low level
# --------------------------------------------------------------------------- #
def _dec_f32(raw: np.ndarray) -> np.ndarray:
    """DEC (VAX F) 32-bit floats (raw bytes, last axis 4) → IEEE."""
    b = raw.reshape(-1, 4)
    sw = np.ascontiguousarray(b[:, [2, 3, 0, 1]]).view("<f4").ravel() / 4.0
    return sw.reshape(raw.shape[:-1])


class _Reader:
    def __init__(self, buf: bytes, proc: int) -> None:
        self.buf, self.proc = buf, proc
        self.e = ">" if proc == MIPS else "<"

    def i8(self, o):
        return struct.unpack_from("b", self.buf, o)[0]

    def u8(self, o):
        return self.buf[o]

    def i16(self, o):
        return struct.unpack_from(self.e + "h", self.buf, o)[0]

    def u16(self, o):
        return struct.unpack_from(self.e + "H", self.buf, o)[0]

    def f32(self, o):
        if self.proc == DEC:
            return float(_dec_f32(np.frombuffer(self.buf, np.uint8, 4, o).reshape(1, 4))[0])
        return struct.unpack_from(self.e + "f", self.buf, o)[0]

    def array(self, o, dtype: int, dims: list[int]):
        n = int(np.prod(dims)) if dims else 1
        if dtype == -1:                                     # characters
            raw = self.buf[o:o + n]
            if not dims:
                return raw.decode("latin-1").rstrip()
            if len(dims) == 1:
                return raw.decode("latin-1").rstrip()
            w = dims[0]
            if w == 0 or n == 0:
                return []
            k = n // w
            return [raw[i * w:(i + 1) * w].decode("latin-1").strip() for i in range(k)]
        if dtype == 1:
            vals = np.frombuffer(self.buf, np.uint8, n, o).astype(int)
        elif dtype == 2:
            vals = np.frombuffer(self.buf, np.dtype(self.e + "i2"), n, o).astype(int)
        elif dtype == 4:
            if self.proc == DEC:
                vals = _dec_f32(np.frombuffer(self.buf, np.uint8, 4 * n, o).reshape(n, 4))
            else:
                vals = np.frombuffer(self.buf, np.dtype(self.e + "f4"), n, o).astype(float)
        else:
            raise C3DError(f"unknown parameter type {dtype}")
        if not dims:
            return vals[0].item()
        return vals.reshape(dims[::-1]) if len(dims) > 1 else vals


def _parameters(buf: bytes, start: int) -> tuple[dict, int]:
    proc = buf[start + 3]
    if proc not in (INTEL, DEC, MIPS):
        proc = INTEL                                        # some writers leave it 0
    r = _Reader(buf, proc)
    nblocks = buf[start + 2]
    end = start + max(1, nblocks) * 512
    groups: dict[int, str] = {}
    params: dict[int, dict] = {}
    o = start + 4
    while o < min(end, len(buf)) - 4:
        nch = r.i8(o)
        if nch == 0:
            break
        gid = r.i8(o + 1)
        name = buf[o + 2:o + 2 + abs(nch)].decode("latin-1").strip().upper()
        p = o + 2 + abs(nch)
        nxt = r.i16(p)
        body = p + 2
        if gid < 0:
            groups[-gid] = name
        else:
            dtype = r.i8(body)
            nd = r.u8(body + 1)
            dims = [r.u8(body + 2 + i) for i in range(nd)]
            try:
                val = r.array(body + 2 + nd, dtype, dims)
            except Exception:                           # malformed parameter: skip it
                val = None
            params.setdefault(gid, {})[name] = val
        if nxt == 0:
            break
        o = p + nxt
    out = {groups.get(g, f"GROUP{g}"): ps for g, ps in params.items()}
    for g in groups.values():
        out.setdefault(g, {})
    return out, proc


def _scalar(v, default=None):
    if v is None:
        return default
    if isinstance(v, list):                                 # character arrays
        return v[0] if v else default
    if isinstance(v, np.ndarray):
        return v.ravel()[0].item() if v.size else default
    return v


# --------------------------------------------------------------------------- #
# public
# --------------------------------------------------------------------------- #
def read_c3d(path) -> C3D:
    """Read the 3-D points of a C3D file (see the module docstring)."""
    path = Path(path)
    buf = path.read_bytes()
    if len(buf) < 512 or buf[1] != 0x50:
        raise C3DError(f"{path.name}: not a C3D file (header key byte is not 0x50)")
    pstart = (buf[0] - 1) * 512
    params, proc = _parameters(buf, pstart)
    r = _Reader(buf, proc)
    pt = params.get("POINT", {})
    n_pts = r.u16(2)
    analog_per_frame = r.u16(4)
    first = r.u16(6)
    last = r.u16(8)
    scale = r.f32(12)
    data_start = r.u16(16)
    rate = r.f32(20)
    # parameters win over the (16-bit) header where present
    n_pts = int(_scalar(pt.get("USED"), n_pts)) % 65536
    scale = float(_scalar(pt.get("SCALE"), scale))
    rate = float(_scalar(pt.get("RATE"), rate))
    ds = int(_scalar(pt.get("DATA_START"), 0) or 0) % 65536
    data_start = ds if ds > 0 else data_start
    n_frames = last - first + 1
    fr = _scalar(pt.get("FRAMES"))
    if fr is not None:
        fr = int(fr)
        if fr < 0:
            fr += 65536
        if fr > n_frames or n_frames <= 0:
            n_frames = fr
    trial = params.get("TRIAL", {})
    a_s, a_e = trial.get("ACTUAL_START_FIELD"), trial.get("ACTUAL_END_FIELD")
    if a_s is not None and a_e is not None and np.size(a_s) >= 2 and np.size(a_e) >= 2:
        s_ = int(np.ravel(a_s)[0]) % 65536 + (int(np.ravel(a_s)[1]) % 65536) * 65536
        e_ = int(np.ravel(a_e)[0]) % 65536 + (int(np.ravel(a_e)[1]) % 65536) * 65536
        if e_ >= s_ and e_ - s_ + 1 > n_frames:
            first, n_frames = s_, e_ - s_ + 1
    if n_pts == 0 or n_frames <= 0:
        return C3D([], np.zeros((0, 0, 3)), np.zeros((0, 0)), rate, first, "mm", params, _proc_name(proc))
    is_float = scale < 0
    word = 4 if is_float else 2
    per_frame = n_pts * 4 + analog_per_frame
    off = (data_start - 1) * 512
    avail = (len(buf) - off) // (per_frame * word)
    n_frames = min(n_frames, avail)
    raw = np.frombuffer(buf, np.uint8, n_frames * per_frame * word, off).reshape(n_frames, per_frame, word)
    raw = raw[:, :n_pts * 4].reshape(n_frames, n_pts, 4, word)
    if is_float:
        if proc == DEC:
            vals = _dec_f32(raw)
        else:
            vals = np.ascontiguousarray(raw).view(r.e + "f4")[..., 0].astype(float)
        xyz = vals[..., :3]
        resw = np.rint(vals[..., 3]).astype(np.int64)
        res = (resw & 0xFF).astype(float) * abs(scale)
        bad = vals[..., 3] < 0
    else:
        ints = np.ascontiguousarray(raw).view(r.e + "i2")[..., 0].astype(np.int64)
        xyz = ints[..., :3] * scale
        resw = ints[..., 3]
        res = (resw & 0xFF).astype(float) * abs(scale)
        bad = resw < 0
    xyz = np.where(bad[..., None], np.nan, xyz)
    res = np.where(bad, np.nan, res)
    labels = _labels(pt, n_pts)
    units = str(_scalar(pt.get("UNITS"), "mm") or "mm")
    return C3D(labels, xyz, res, rate, int(first), units, params, _proc_name(proc))


def _proc_name(p: int) -> str:
    return {INTEL: "intel", DEC: "dec", MIPS: "mips"}.get(p, "intel")


def _labels(pt: dict, n: int) -> list[str]:
    out: list[str] = []
    for key in ["LABELS"] + [f"LABELS{i}" for i in range(2, 100)]:
        v = pt.get(key)
        if v is None:
            if key != "LABELS":
                break
            continue
        out += [v] if isinstance(v, str) else list(v)
    out = out[:n]
    out += [f"M{i + 1:03d}" for i in range(len(out), n)]
    seen: dict[str, int] = {}
    clean = []
    for i, s in enumerate(out):
        s = s.strip().replace(" ", "_") or f"M{i + 1:03d}"
        if s in seen:
            seen[s] += 1
            s = f"{s}_{seen[s]}"
        else:
            seen[s] = 0
        clean.append(s)
    return clean


UP_AXES = {
    # file axis that points up → rotation taking it to +z (right-handed)
    "z": np.eye(3),
    "y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),     # (x, y, z) → (x, -z, y)
    "x": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float),     # (x, y, z) → (-z, y, x)
}


def c3d_to_pose3d(c: C3D, *, up: str = "z", rename: dict | None = None, edges=None,
                  frames: tuple[int, int] | None = None, source: str = "") -> tuple[pd.DataFrame, dict]:
    """A 3-D pose table (mm, z up) and its metadata from a read C3D."""
    from mufasa.pose3d.io import skeleton_edges
    rename = rename or {}
    names = [rename.get(n, n) for n in c.labels]
    X = c.points * c.to_mm
    R = UP_AXES[up]
    X = X @ R.T
    res = c.residuals * c.to_mm
    T = len(X)
    sl = slice(None)
    fr0 = c.first_frame
    if frames is not None:
        a, b = frames
        sl = slice(max(0, a), min(T, b))
        fr0 += max(0, a)
    X, res = X[sl], res[sl]
    cols: dict = {"frame": np.arange(len(X)), "c3d_frame": fr0 + np.arange(len(X))}
    for j, m in enumerate(names):
        cols[m + "_x"], cols[m + "_y"], cols[m + "_z"] = X[:, j, 0], X[:, j, 1], X[:, j, 2]
        cols[m + "_likelihood"] = np.isfinite(X[:, j, 0]).astype(float)
        cols[m + "_residual_mm"] = res[:, j]
    df = pd.DataFrame(cols)
    ok = np.isfinite(X).all(2)
    meta = {"units": "mm", "fps": c.rate,
            "method": f"motion capture (C3D{', ' + _company(c) if _company(c) else ''}); "
                      f"{'z' if up == 'z' else up + '-up axes turned to z up'}",
            "source_c3d": source, "c3d_units": c.units, "c3d_first_frame": c.first_frame,
            "edges": [list(e) for e in (edges if edges is not None else skeleton_edges(names))]}
    if ok.any():
        P = X[ok]
        lo, hi = np.percentile(P, 0.5, axis=0), np.percentile(P, 99.5, axis=0)
        pad = 0.05 * (hi - lo) + 1.0
        meta.update(box_x=[float(lo[0] - pad[0]), float(hi[0] + pad[0])],
                    box_y=[float(lo[1] - pad[1]), float(hi[1] + pad[1])],
                    box_z=[float(min(0.0, lo[2] - pad[2])), float(hi[2] + pad[2])])
    meta["report"] = {"frames": int(len(X)), "markers": len(names), "rate_hz": c.rate,
                      "visible_fraction": {m: round(float(ok[:, j].mean()), 4) for j, m in enumerate(names)}}
    return df, meta


def _company(c: C3D) -> str:
    v = c.params.get("MANUFACTURER", {}).get("COMPANY")
    return v if isinstance(v, str) else ""


def parse_rename(text: str) -> dict:
    """``"a=b, c=d"`` → ``{"a": "b", "c": "d"}`` (marker renaming)."""
    out = {}
    for item in (text or "").replace(";", ",").split(","):
        if "=" in item:
            a, b = (s.strip() for s in item.split("=", 1))
            if a and b:
                out[a] = b
    return out


def parse_edges(text: str) -> list[list[str]] | None:
    """``"a-b, b-c"`` → ``[["a", "b"], ["b", "c"]]`` (skeleton lines); None
    when empty (then the rat skeleton for the markers present)."""
    out = []
    for item in (text or "").replace(";", ",").split(","):
        if "-" in item:
            a, b = (s.strip() for s in item.split("-", 1))
            if a and b:
                out.append([a, b])
    return out or None


def import_c3d(path, out_dir, *, up: str = "z", rename: dict | None = None, edges=None,
               frames=None) -> Path:
    """Read ``path`` and write ``<out_dir>/<stem>_3d.parquet``."""
    from mufasa.pose3d.io import write_pose3d
    path = Path(path)
    c = read_c3d(path)
    if not c.labels:
        raise C3DError(f"{path.name}: no 3-D points")
    df, meta = c3d_to_pose3d(c, up=up, rename=rename, edges=edges, frames=frames, source=str(path))
    return write_pose3d(df, Path(out_dir) / f"{path.stem}_3d.parquet", meta)


__all__ = ["C3D", "C3DError", "UP_AXES", "c3d_to_pose3d", "import_c3d", "parse_edges", "parse_rename", "read_c3d"]
