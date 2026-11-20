"""
mufasa.pose3d.scene_models
==========================

Patch 122kv: models in the 3-D viewer.

* :class:`SceneMesh` — a static mesh (``.obj``, ``.stl`` ASCII or binary,
  ``.ply`` ASCII or binary little-endian) placed in the volume: an arena,
  a nest, objects. It has its own scale (e.g. 1000 for a mesh in metres),
  rotation about the vertical and offset.
* :class:`PosedRig` — a body-model rig (``.rig.npz`` from the Body model
  tab, or the built-in virtual rodent) posed to the 3-D markers of each
  frame. It shares the 3-D labels' names with the rig's label points (the
  rat's labels for the virtual rodent). The fit is a least-squares fit of
  heading, position and the rig's posture values within their limits. The
  rig is scaled to the animal's size, which is the back's length seen from
  above against the model, as in :func:`mufasa.bodymodel.fit.scale_from_pose3d`.
  Each frame starts from the nearest frame already fitted.

Both give world vertices and faces for a frame; the viewer draws them.
:func:`model_from_spec` and ``spec()`` turn models into plain values (a
saved camera animation renders with them).

No Qt here.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STYLES = ("solid", "transparent", "wireframe", "points")
MESH_SUFFIXES = (".obj", ".stl", ".ply")
MAX_FACES = 20000                     # drawn with QPainter: larger meshes are simplified


# --------------------------------------------------------------------------- #
# mesh files
# --------------------------------------------------------------------------- #
def load_mesh(path) -> tuple[np.ndarray, np.ndarray]:
    """(V (n, 3) float, F (m, 3) int) from an OBJ, STL or PLY file; polygons are fanned into
    triangles."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".obj":
        return _load_obj(path)
    if suf == ".stl":
        return _load_stl(path)
    if suf == ".ply":
        return _load_ply(path)
    raise ValueError(f"{path.name}: unknown mesh format (use .obj, .stl or .ply)")


def _fan(poly):
    return [(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)]


def _load_obj(path):
    V, F = [], []
    for line in path.read_text(errors="replace").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "v" and len(t) >= 4:
            V.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == "f" and len(t) >= 4:
            idx = []
            for s in t[1:]:
                i = int(s.split("/")[0])
                idx.append(i - 1 if i > 0 else len(V) + i)
            F.extend(_fan(idx))
    if not V or not F:
        raise ValueError(f"{path.name}: no vertices or faces")
    return np.array(V, float), np.array(F, int)


def _load_stl(path):
    b = path.read_bytes()
    n = struct.unpack_from("<I", b, 80)[0] if len(b) >= 84 else -1
    if n >= 0 and len(b) == 84 + 50 * n:                         # binary
        rec = np.frombuffer(b, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                            count=n, offset=84)
        P = rec["v"].reshape(-1, 3).astype(float)
    else:                                                       # ASCII
        P = np.array([[float(x) for x in ln.split()[1:4]] for ln in b.decode(errors="replace").splitlines()
                      if ln.strip().startswith("vertex")], float)
    if not len(P):
        raise ValueError(f"{path.name}: no triangles")
    # STL repeats vertices per triangle: merge identical ones
    V, inv = np.unique(P, axis=0, return_inverse=True)
    return V, inv.ravel().reshape(-1, 3)


def _load_ply(path):
    b = path.read_bytes()
    end = b.find(b"end_header")
    if not b.startswith(b"ply") or end < 0:
        raise ValueError(f"{path.name}: not a PLY file")
    header = b[:end].decode(errors="replace").splitlines()
    body = b[b.index(b"\n", end) + 1:]
    fmt = next(ln.split()[1] for ln in header if ln.startswith("format"))
    elements, cur = [], None
    for ln in header:
        t = ln.split()
        if t[:1] == ["element"]:
            cur = {"name": t[1], "n": int(t[2]), "props": []}
            elements.append(cur)
        elif t[:1] == ["property"] and cur is not None:
            cur["props"].append(t[1:])
    types = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
             "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
             "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}
    V, F = None, []
    if fmt == "ascii":
        lines = iter(body.decode(errors="replace").split("\n"))
        for el in elements:
            rows = [next(lines).split() for _ in range(el["n"])]
            if el["name"] == "vertex":
                names = [p[-1] for p in el["props"]]
                ix = [names.index(a) for a in "xyz"]
                V = np.array([[float(r[i]) for i in ix] for r in rows], float)
            elif el["name"] == "face":
                for r in rows:
                    k = int(r[0])
                    F.extend(_fan([int(x) for x in r[1:1 + k]]))
    elif fmt == "binary_little_endian":
        o = 0
        for el in elements:
            if all(p[0] != "list" for p in el["props"]):
                dt = np.dtype([(p[-1], "<" + types[p[0]]) for p in el["props"]])
                arr = np.frombuffer(body, dt, el["n"], o)
                o += dt.itemsize * el["n"]
                if el["name"] == "vertex":
                    V = np.c_[arr["x"], arr["y"], arr["z"]].astype(float)
            else:                                                # faces: one list property
                (_, ct, it, _name), = [p for p in el["props"] if p[0] == "list"]
                cs, isz = np.dtype(types[ct]).itemsize, np.dtype(types[it]).itemsize
                for _ in range(el["n"]):
                    k = int(np.frombuffer(body, "<" + types[ct], 1, o)[0])
                    o += cs
                    F.extend(_fan(np.frombuffer(body, "<" + types[it], k, o).astype(int).tolist()))
                    o += isz * k
    else:
        raise ValueError(f"{path.name}: PLY format {fmt} is not supported (ASCII or binary little-endian)")
    if V is None or not F:
        raise ValueError(f"{path.name}: no vertices or faces")
    return V, np.array(F, int)


def simplify(V, F, max_faces: int = MAX_FACES):
    """Vertex clustering until at most ``max_faces`` faces remain."""
    if len(F) <= max_faces:
        return V, F
    from mufasa.bodymodel.rodent_rig import decimate
    W = np.ones((len(V), 1))
    cell = float(np.ptp(V, axis=0).max()) / 200
    while True:
        Vd, Fd, _ = decimate(V, F, W, cell)
        if len(Fd) <= max_faces:
            return Vd, Fd
        cell *= 1.3


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
@dataclass
class ModelLook:
    style: str = "transparent"          # STYLES
    opacity: float = 0.55
    color: str = "#d9cfc4"
    visible: bool = True


@dataclass
class SceneMesh:
    """A static mesh in the volume."""
    path: str
    V: np.ndarray
    F: np.ndarray
    scale: float = 1.0
    rot_z_deg: float = 0.0
    offset: tuple = (0.0, 0.0, 0.0)
    look: ModelLook = field(default_factory=lambda: ModelLook(style="solid", opacity=1.0, color="#9aa4ae"))
    kind = "mesh"

    @classmethod
    def load(cls, path, **kw) -> SceneMesh:
        V, F = simplify(*load_mesh(path))
        return cls(str(path), V, F, **kw)

    @property
    def name(self) -> str:
        return Path(self.path).name

    def faces(self):
        return self.F

    def world(self, frame: int | None = None) -> np.ndarray:
        a = math.radians(self.rot_z_deg)
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
        return (self.scale * self.V) @ R.T + np.asarray(self.offset, float)

    def info(self, frame: int | None = None) -> str:
        return f"{len(self.V):,} vertices, {len(self.F):,} faces"

    def spec(self) -> dict:
        return {"kind": "mesh", "path": self.path, "scale": self.scale, "rot_z_deg": self.rot_z_deg,
                "offset": list(self.offset), "look": vars(self.look).copy()}


class PoseCache:
    """Patch 122lb: a posed rig's frames — {frame: (parameters, marker RMS mm)} — safe to share
    between the viewer and the background fit thread (122ky), which writes into it while the
    viewer reads. A lock around every access, and the frames kept sorted, so the nearest posed
    frame is a binary search, not a scan of every frame on every paint. Reads like a dict."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._d: dict[int, tuple[np.ndarray, float]] = {}
        self._keys: list[int] = []                 # sorted

    def __contains__(self, frame) -> bool:
        with self._lock:
            return int(frame) in self._d

    def __getitem__(self, frame):
        with self._lock:
            return self._d[int(frame)]

    def get(self, frame, default=None):
        with self._lock:
            return self._d.get(int(frame), default)

    def __setitem__(self, frame, value) -> None:
        self.update({int(frame): value})

    def update(self, items) -> None:
        import bisect
        items = dict(items)
        with self._lock:
            for f, v in items.items():
                f = int(f)
                if f not in self._d:
                    bisect.insort(self._keys, f)
                self._d[f] = v

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._keys.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)

    def __iter__(self):
        with self._lock:
            return iter(list(self._keys))          # a snapshot: safe while the other thread writes

    def keys(self) -> list[int]:
        with self._lock:
            return list(self._keys)

    def nearest(self, frame: int, within: int) -> int | None:
        """The posed frame nearest ``frame``, at most ``within`` away (ties: the earlier one)."""
        import bisect
        frame = int(frame)
        with self._lock:
            k = bisect.bisect_left(self._keys, frame)
            best = None
            for j in (k - 1, k):
                if 0 <= j < len(self._keys):
                    d = abs(self._keys[j] - frame)
                    if d <= within and (best is None or d < abs(best - frame)):
                        best = self._keys[j]
            return best


class PosedRig:
    """A body-model rig posed to the 3-D markers, frame by frame."""
    kind = "rig"

    def __init__(self, rig_path, pose, *, builtin: str | None = None, look: ModelLook | None = None,
                 size: float | None = None) -> None:
        from mufasa.bodymodel.fit import Markers, Rig
        self.path, self.builtin = str(rig_path), builtin
        self.rig = Rig(rig_path)
        self.mk = Markers.build(self.rig, None)
        self.pose = pose
        self.look = look or ModelLook()
        rest = self.mk.positions(self.rig, self.rig.skinning_matrices(_neutral()))
        self.names = [n for n in pose.names if n in rest]
        if len(self.names) < 3:
            raise ValueError(f"{Path(rig_path).name}: needs at least three of the 3-D markers on the model "
                             f"(the model has {', '.join(sorted(rest))})")
        self._col = [pose.names.index(n) for n in self.names]
        self._rest = np.array([rest[n] for n in self.names])
        self.posture = list(self.rig.posture)
        self.size = float(size) if size else self._estimate_size(rest)
        self._cache = PoseCache()                 # 122lb: shared with the background fit thread
        self.good_mm = 12.0                    # a start within this marker error ends the search

    @property
    def name(self) -> str:
        return self.builtin or Path(self.path).name.removesuffix(".rig.npz")

    # -- size ------------------------------------------------------------- #
    def _estimate_size(self, rest) -> float:
        from mufasa.bodymodel.fit import BACK_CHAIN
        X = self.pose.xyz
        chain = [n for n in BACK_CHAIN if n in self.names]
        if len(chain) >= 3:
            ix = [self.pose.names.index(n) for n in chain]
            path = sum(np.linalg.norm((X[:, b, :2] - X[:, a, :2]), axis=1) for a, b in zip(ix[:-1], ix[1:]))
            hz = X[:, ix[0], 2]
            low = np.isfinite(path) & np.isfinite(hz) & (hz <= np.nanmedian(hz))
            model = sum(float(np.linalg.norm((rest[b] - rest[a])[:2])) for a, b in zip(chain[:-1], chain[1:]))
            if low.sum() >= 5 and model > 0:
                return float(np.median(path[low]) / model)
        # otherwise: the median ratio of all marker distances to the model's
        D = X[:, self._col]
        i, j = np.triu_indices(len(self.names), 1)
        d = np.linalg.norm(D[:, i] - D[:, j], axis=2)
        dm = np.linalg.norm(self._rest[i] - self._rest[j], axis=1)
        r = d / np.maximum(dm, 1e-9)
        return float(np.nanmedian(r)) if np.isfinite(r).any() else 1.0

    # -- fit -------------------------------------------------------------- #
    def _x2p(self, x):
        p = _neutral()
        for v, (n, _lo, _hi) in zip(x[4:], self.posture):
            p[n] = float(v)
        return p, x[0], x[1], x[2], x[3]

    def _markers_world(self, x):
        p, phi, tx, ty, tz = self._x2p(x)
        G = self.rig.skinning_matrices(p)
        pos = self.mk.positions(self.rig, G)
        P = self.size * np.array([pos[n] for n in self.names])
        return _yaw(P, phi) + np.array([tx, ty, tz]), G, p

    def _initial(self, Y, ok):
        """Heading and position from a 2-D Procrustes of the rest markers onto the data."""
        A, B = self.size * self._rest[ok, :2], Y[ok, :2]
        a0, b0 = A.mean(0), B.mean(0)
        H = (A - a0).T @ (B - b0)
        phi = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        t = b0 - _yaw(np.r_[a0, 0.0][None], phi)[0, :2]
        return np.r_[phi, t, 0.0, [0.0] * len(self.posture)]

    def fit_frame(self, frame: int):
        """(parameters, marker RMS mm) for a frame, or None without enough markers."""
        frame = int(frame)
        hit = self._cache.get(frame)
        if hit is not None:
            return hit
        i = self.pose.row(frame)
        if i is None:
            return None
        Y = self.pose.xyz[i, self._col]
        ok = np.isfinite(Y).all(1)
        if self.pose.likelihood is not None:
            ok &= ~(self.pose.likelihood[i, self._col] < 0.5)
        if ok.sum() < 3:
            return None
        from scipy.optimize import least_squares
        lo = np.r_[-np.inf, -np.inf, -np.inf, -60.0, [p[1] for p in self.posture]]
        hi = np.r_[np.inf, np.inf, np.inf, 60.0, [p[2] for p in self.posture]]
        half = np.array([(p[2] - p[1]) / 2 or 1.0 for p in self.posture])
        mid = np.array([self.rig.prior_centre.get(p[0], (p[1] + p[2]) / 2) for p in self.posture])

        def res(x):
            P, _, _ = self._markers_world(x)
            r = ((P[ok] - Y[ok]) / 4.0).ravel()                      # 4 mm
            return np.r_[r, 0.3 * (x[4:] - mid) / half, x[3] / 30.0]
        nf = self._cache.nearest(frame, 30)
        x0 = self._initial(Y, ok)
        flip = np.r_[math.pi, 0, 0, 0, [0] * len(self.posture)]
        rear = np.zeros_like(x0)
        if "rear" in [q[0] for q in self.posture]:
            rear[4 + [q[0] for q in self.posture].index("rear")] = 45.0
        cold = [x0, x0 + flip, x0 + rear, x0 + flip + rear]          # both headings, flat or rearing
        nb = self._cache.get(nf) if nf is not None else None
        near = nb is not None
        starts = ([nb[0].copy()] if near else []) + cold
        best = None
        nb_rms = nb[1] if near else np.inf
        for k, s0 in enumerate(starts):
            s0 = np.clip(s0, lo + 1e-9, hi - 1e-9)
            r = least_squares(res, s0, bounds=(lo, hi), max_nfev=25 if (near and k == 0) else 40, x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
            P, _, _ = self._markers_world(best.x)
            err = float(np.sqrt(np.mean(np.sum((P[ok] - Y[ok]) ** 2, 1))))
            # good enough: stop early (from a neighbour: unless clearly worse than the neighbour)
            if err < self.good_mm or (near and k == 0 and err < max(2 * nb_rms, 25.0)):
                break
        P, _, _ = self._markers_world(best.x)
        rms = float(np.sqrt(np.mean(np.sum((P[ok] - Y[ok]) ** 2, 1))))
        x = best.x.copy()
        x[0] = (x[0] + math.pi) % (2 * math.pi) - math.pi
        self._cache[frame] = (x, rms)
        return (x, rms)

    def fit_frames(self, frames, *, device: str = "auto", progress=None, stop=None) -> int:
        """Patch 122ky: pose many frames at once — batched in PyTorch (on a CUDA GPU when there is
        one) — into the same cache :meth:`fit_frame` uses; frame by frame without PyTorch.
        Returns the number of frames posed."""
        todo = [int(f) for f in frames if int(f) not in self._cache]
        if not todo:
            return 0
        from mufasa.pose3d import rig_batch
        if rig_batch.available():
            res = rig_batch.fit_rig_frames(self, todo, device=device, progress=progress, stop=stop)
            self._cache.update(res)
            return len(res)
        n = 0
        for k, f in enumerate(todo):
            if stop is not None and stop():
                break
            n += self.fit_frame(f) is not None
            if progress is not None and k % 20 == 0:
                progress(k + 1, len(todo))
        return n

    def cached(self, frame: int) -> bool:
        return int(frame) in self._cache

    def nearest_cached(self, frame: int, within: int = 15) -> int | None:
        return self._cache.nearest(frame, within)

    def faces(self):
        return self.rig.Fd

    def world(self, frame: int | None = None) -> np.ndarray | None:
        f = self.fit_frame(frame) if frame is not None else None
        if f is None:
            return None
        x = f[0]
        p, phi, tx, ty, tz = self._x2p(x)
        from mufasa.bodymodel.fit import Rig
        V = Rig.lbs(self.rig.skinning_matrices(p), self.rig.verts(p, True), self.rig.Wd)
        return _yaw(self.size * V, phi) + np.array([tx, ty, tz])

    def info(self, frame: int | None = None) -> str:
        f = self.fit_frame(frame) if frame is not None else None
        s = f"{len(self.names)} markers, size {self.size:.2f}x the model"
        if f is not None:
            s += f", marker error {f[1]:.1f} mm"
        return s

    def spec(self) -> dict:
        return {"kind": "rig", "path": self.path, "builtin": self.builtin, "size": self.size,
                "look": vars(self.look).copy()}


def _neutral() -> dict:
    from mufasa.bodymodel.fit import neutral
    return neutral()


def _yaw(P, phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.c_[c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1], P[:, 2]]


def builtin_rig_path() -> Path:
    """The virtual rodent as a rig, made once in the user's cache folder."""
    import os

    from mufasa.bodymodel.rodent_rig import RIG_VERSION
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mufasa"
    p = base / f"virtual_rodent_v{RIG_VERSION}.rig.npz"                # rebuilt when the rig changes
    if not p.is_file():
        from mufasa.bodymodel.rodent_rig import export_rodent_rig
        export_rodent_rig(p, say=lambda s: None)
    return p


class GaitOverlay:
    """Patch 122kw: paws, support polygon, centre of mass and ZMP from a gait file
    (``<video>_gait.parquet``, :mod:`mufasa.pose3d.gait`), drawn on the floor."""
    kind = "gait"

    def __init__(self, path, look: ModelLook | None = None) -> None:
        from mufasa.pose3d.gait import read_gait
        self.path = str(path)
        self.df, self.meta = read_gait(path)
        self.look = look or ModelLook(style="solid", opacity=0.9)
        self._row = {int(f): i for i, f in enumerate(self.df["frame"])}

    @property
    def name(self) -> str:
        return Path(self.path).name

    def at(self, frame: int) -> dict | None:
        """{paws: {paw: (x, y, load, contact)}, com: (3,), zmp: (2,), margin} for a frame."""
        from mufasa.pose3d.gait import PAWS
        i = self._row.get(int(frame))
        if i is None:
            return None
        r = self.df.iloc[i]
        if not bool(r["valid"]):
            return None
        return {"paws": {p: (float(r[f"{p}_x"]), float(r[f"{p}_y"]), float(r[f"{p}_load"]), int(r[f"{p}_contact"]))
                         for p in PAWS},
                "com": np.array([r["com_x"], r["com_y"], r["com_z"]], float),
                "zmp": np.array([r["zmp_x"], r["zmp_y"]], float), "margin": float(r["zmp_margin_mm"])}

    def info(self, frame: int | None = None) -> str:
        s = self.meta.get("summary", {})
        txt = f"{s.get('steps', '?')} steps, {s.get('mass_g', '?')} g"
        g = self.at(frame) if frame is not None else None
        if g is not None:
            on = [p for p, v in g["paws"].items() if v[3] == 1]
            txt += f" · on the floor: {' '.join(on) or 'none'}"
            if np.isfinite(g["margin"]):
                txt += f" · ZMP margin {g['margin']:.0f} mm"
        return txt

    def spec(self) -> dict:
        return {"kind": "gait", "path": self.path, "look": vars(self.look).copy()}


def load_model(path, pose, **kw):
    """A model from a file: a ``.rig.npz`` (posed to ``pose``), a mesh (placed as is) or a gait
    file (``_gait.parquet``: paws, support, centre of mass)."""
    path = str(path)
    if path.endswith("_gait.parquet"):
        return GaitOverlay(path, **{k: v for k, v in kw.items() if k == "look"})
    if path == "virtual_rodent":
        return PosedRig(builtin_rig_path(), pose, builtin="virtual rodent", **kw)
    if path.endswith(".rig.npz"):
        return PosedRig(path, pose, **kw)
    if Path(path).suffix.lower() in MESH_SUFFIXES:
        return SceneMesh.load(path, **kw)
    raise ValueError(f"{Path(path).name}: not a model (.rig.npz, .obj, .stl or .ply)")


def model_from_spec(spec: dict, pose):
    look = ModelLook(**spec.get("look", {}))
    if spec.get("kind") == "gait":
        return GaitOverlay(spec["path"], look=look)
    if spec.get("kind") == "rig":
        src = "virtual_rodent" if spec.get("builtin") else spec["path"]
        return load_model(src, pose, look=look, size=spec.get("size"))
    return SceneMesh.load(spec["path"], scale=spec.get("scale", 1.0), rot_z_deg=spec.get("rot_z_deg", 0.0),
                          offset=tuple(spec.get("offset", (0, 0, 0))), look=look)


__all__ = ["PoseCache", "GaitOverlay", "MESH_SUFFIXES", "STYLES", "ModelLook", "PosedRig", "SceneMesh", "builtin_rig_path", "load_mesh",
           "load_model", "model_from_spec", "simplify"]
