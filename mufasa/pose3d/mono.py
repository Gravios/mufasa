"""
mufasa.pose3d.mono
==================

Patch 122kf: single-camera 3-D pose for rats in a box.

One camera sees each label only as a ray; depth comes from body models:

1. **Head** — a rigid head (``head_back``, ``head_mid``, ``head_nose`` and
   the two ear labels) whose shape is learned from the session itself
   (structure from motion over confident frames); its size is fixed by
   the ear-to-ear distance (``head_width_mm``), so the apparent ear span
   gives the distance to the camera. All frames are fitted jointly with
   the box limits, a modest-roll prior and smooth motion.
2. **Trunk heights** — learned from Rat7M motion capture (see
   ``mufasa/assets/models/pose3d/README.md``): from head height, head
   pitch and the horizontal head→hips distance, the heights of the
   shoulders (SpineF), mid-back (SpineM), lumbar spine (SpineL) and hips
   are looked up (k-nearest neighbours); each project label is placed on
   its camera ray at its mapped height.
3. **Pelvis** — the ``back_V2``→``back_L6`` segment is pitched to the
   Rat7M lumbar angle (labels stay on their rays).
4. **Spine chain** — the back labels between the pelvis and the head are
   refitted as one chain: heights stay near the Rat7M values, bends are
   pulled to the Rat7M hump for that head height, backward bends past the
   Rat7M 95th percentile are penalised, the neck bends within the Rat7M
   range, all smooth over time.
5. **Tail** — each tail label hangs as low as it can on its ray without
   the tail stretching (lengths measured from the session).

Every label stays on its own camera ray, so the 3-D pose reprojects onto
the 2-D labels (rigid head: about a pixel).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot

from mufasa.pose3d.camera import Camera
from mufasa.pose3d.solver import solve_sequence

PRIOR_PATH = Path(__file__).resolve().parents[1] / "assets" / "models" / "pose3d" / "rat7m_trunk_prior.npz"


@dataclass
class MonoConfig:
    # marker names (defaults: the 15-marker rat skeleton)
    head_back: str = "head_back"
    head_mid: str = "head_mid"
    head_nose: str = "head_nose"
    head_left: str = "head_left"
    head_right: str = "head_right"
    spine: tuple = ("back_T4", "back_T8", "back_L2", "back_L6", "back_V2")    # head → tail
    hips: tuple = ("hip_left", "hip_right")
    pelvis: tuple = ("back_V2", "back_L6")                                    # tail end, front end
    tail: tuple = ("tail_V6", "tail_V18", "tail_V32")
    # where each spine label sits on the Rat7M chain HeadB-SpineF-SpineM-SpineL
    # (a, b, f): height = z_a + f (z_b - z_a); b == "SpineL+" extrapolates beyond
    # SpineL along SpineM→SpineL
    spine_map: dict = field(default_factory=lambda: {
        "back_T4": ("HeadB", "SpineF", 0.51), "back_T8": ("SpineF", "SpineM", 0.43),
        "back_L2": ("SpineM", "SpineL", 0.69), "back_L6": ("SpineL", "SpineL+", 0.2),
        "back_V2": ("SpineL", "SpineL+", 0.85)})
    head_width_mm: float = 36.0          # head_left ↔ head_right distance
    # heights (mm) of head_back / head_mid / head_nose relative to the ear line
    # (ears at 0). A camera looking down cannot measure them, so they are fixed
    # (defaults: the K7 head fitted on 2026-11-11, which matched Rat7M); set to
    # None to learn them from the session.
    head_template_z: dict | None = field(default_factory=lambda: {
        "head_back": -0.4, "head_mid": -4.7, "head_nose": -18.4})
    head_height_offset_mm: float = 9.0   # Rat7M HeadB (on an implant) above this project's head_back
    head_pitch_offset_deg: float = 23.0  # Rat7M head pitch minus this project's head_back→head_mid pitch
    body_scale: float = 1.0              # this rat's size relative to Rat7M rats
    likelihood_threshold: float = 0.9
    fps: float = 30.0
    head_roll_sd_deg: float = 20.0
    # the walls are only as accurate as the calibration and the entered box
    # size; labels may sit this far beyond a wall before it pushes back
    # (a hard wall pushes heads up along their rays)
    wall_tolerance_mm: float = 20.0
    use_trunk_prior: bool = True
    pelvis_from_lumbar: bool = True
    spine_chain: bool = True
    chain_passes: int = 3
    drape_tail: bool = True
    tail_floor_mm: float = 5.0
    max_pelvis_rise_mm: float = 90.0
    chunk: int = 6000
    head_iters: int = 150
    # processes for the per-chunk head and spine fits (1: in this process)
    workers: int = 1
    # head and spine solver: "cuda" (PyTorch on the GPU), "torch-cpu"
    # (PyTorch on the CPU) or "cpu" (SciPy); "auto": the first available of
    # those. The PyTorch path is a port of the same trust-region solver
    # (122kj): same results within rounding, faster even on the CPU.
    device: str = "auto"


# --------------------------------------------------------------------------- #
# 2-D input
# --------------------------------------------------------------------------- #
@dataclass
class Pose2D:
    names: list[str]
    uv: dict            # marker -> (T, 2) primary positions (NaN missing)
    lk: dict            # marker -> (T,) likelihood
    filled: dict        # marker -> (T, 2) primary where confident, else fallback / interpolated

    @property
    def n(self) -> int:
        return len(next(iter(self.uv.values())))


def load_pose2d(primary, fallback=None, likelihood_threshold: float = 0.9) -> Pose2D:
    """Load 2-D pose (any Mufasa/FreeDLC/DLC format; ``<m>_x/_y/_p``).
    ``fallback`` (e.g. Kalman v2 smoothed) fills frames where the primary
    labels are below ``likelihood_threshold``; without it gaps are
    interpolated."""
    from mufasa.data_processors.kalman_diagnostic import load_pose_file
    a = primary if isinstance(primary, pd.DataFrame) else load_pose_file(str(primary))[0]
    b = None
    if fallback is not None:
        b = fallback if isinstance(fallback, pd.DataFrame) else load_pose_file(str(fallback))[0]
    names = [c[:-2] for c in a.columns if c.endswith("_x") and c[:-2] + "_y" in a.columns]
    n = len(a) if b is None else min(len(a), len(b))
    uv, lk, filled = {}, {}, {}
    for m in names:
        p = a[[m + "_x", m + "_y"]].to_numpy(float)[:n]
        l = (a[m + "_p"].to_numpy(float)[:n] if m + "_p" in a.columns
             else a[m + "_likelihood"].to_numpy(float)[:n] if m + "_likelihood" in a.columns
             else np.ones(n))
        l = np.nan_to_num(l)
        good = (l >= likelihood_threshold) & np.isfinite(p).all(1)
        if b is not None and m + "_x" in b.columns:
            q = b[[m + "_x", m + "_y"]].to_numpy(float)[:n]
            f = np.where(good[:, None], p, q)
        else:
            f = np.where(good[:, None], p, np.nan)
            f = pd.DataFrame(f).interpolate(limit_direction="both").to_numpy()
        uv[m], lk[m], filled[m] = p, l, f
    return Pose2D(names, uv, lk, filled)


# --------------------------------------------------------------------------- #
# Rat7M prior
# --------------------------------------------------------------------------- #
class Rat7MPrior:
    """k-nearest-neighbour lookup of trunk heights / lumbar angle from the
    packaged Rat7M table."""

    SCALES = np.array([30.0, 15.0, 25.0])

    def __init__(self, path=PRIOR_PATH, k: int = 50, subjects=None) -> None:
        from sklearn.neighbors import KNeighborsRegressor
        z = np.load(path, allow_pickle=False)
        keep = np.ones(len(z["features"]), bool) if subjects is None else np.isin(z["subject"], subjects)
        self.features, self.targets = z["features"][keep], z["targets"][keep]
        self.lumbar, self.roll = z["lumbar"][keep], z["roll"][keep]
        self.subject = z["subject"][keep]
        self.bend_edges = z["bend_edges"]
        self.bend_M, self.bend_F = z["bend_spineM"], z["bend_spineF"]
        Y = np.c_[self.targets, self.lumbar]
        self._knn = KNeighborsRegressor(k, weights="distance").fit(self.features / self.SCALES, Y)

    def predict(self, head_height, head_pitch, head_to_hips) -> np.ndarray:
        """(T, 5): SpineF, SpineM, SpineL, hips heights (mm), lumbar angle (deg)."""
        F = np.c_[head_height, head_pitch, head_to_hips] / self.SCALES
        out = np.full((len(F), 5), np.nan)
        ok = np.isfinite(F).all(1)
        if ok.any():
            out[ok] = self._knn.predict(F[ok])
        return out

    def bend_limits(self, head_height) -> dict:
        cen = (self.bend_edges[:-1] + np.minimum(self.bend_edges[1:], 300)) / 2
        h = np.asarray(head_height, float)
        return {"M50": np.interp(h, cen, self.bend_M[:, 1]), "M95": np.interp(h, cen, self.bend_M[:, 2]),
                "F5": np.interp(h, cen, self.bend_F[:, 0]), "F95": np.interp(h, cen, self.bend_F[:, 2])}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rotm(e: np.ndarray) -> np.ndarray:
    return Rot.from_euler("ZYX", e).as_matrix()


def _huber(e, s):
    q = e / s
    return np.sign(q) * np.sqrt(2 * (np.sqrt(1 + q * q) - 1))


def _smooth(x: np.ndarray, med: int = 7, box: int = 5) -> np.ndarray:
    x = np.asarray(x, float)
    if x.ndim == 1:
        return uniform_filter1d(median_filter(x, med), box)
    return uniform_filter1d(median_filter(x, size=(med, 1)), box, axis=0)


def _on_ray(cam: Camera, P: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Slide points along their camera rays to heights ``z``."""
    C = cam.center
    d = P - C
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (np.asarray(z, float) - C[2]) / d[:, 2]
    return C + s[:, None] * d


def _box_hinge(X: np.ndarray, box: dict | None, margin: float = 0.0) -> list:
    """Penalties for leaving the box; ``margin`` < 0 tolerates that much
    beyond each wall."""
    out = [np.maximum(0, -X[..., 2])]
    if box:
        (x0, x1), (y0, y1), (_, z1) = box["x"], box["y"], box["z"]
        out += [np.maximum(0, X[..., 0] - x1 + margin), np.maximum(0, x0 + margin - X[..., 0]),
                np.maximum(0, X[..., 1] - y1 + margin), np.maximum(0, y0 + margin - X[..., 1]),
                np.maximum(0, X[..., 2] - z1)]
    return out


# --------------------------------------------------------------------------- #
# Rigid head
# --------------------------------------------------------------------------- #
class RigidHead:
    def __init__(self, cam: Camera, names: list[str], ears: tuple[int, int], fwd: tuple[int, int],
                 width_mm: float, fixed_z: dict | None = None) -> None:
        self.cam, self.names, self.ears, self.fwd, self.W = cam, names, ears, fwd, width_mm
        self.fixed_z = fixed_z
        self.T = None

    def _init_pose(self, uv: np.ndarray) -> np.ndarray:
        cam = self.cam
        import warnings
        with warnings.catch_warnings():                 # frames with no head label at all
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmean(uv, 1)
        span = np.linalg.norm(uv[:, self.ears[0]] - uv[:, self.ears[1]], axis=1)
        D = np.clip(cam.focal * self.W / np.maximum(span, 1), 30, 5000)
        axis = cam.R[2]
        rc = cam.rays(c, undistort=False)
        P = cam.center + (D / np.abs(rc @ axis))[:, None] * rc
        rb = cam.rays(uv[:, self.fwd[0]], undistort=False)
        rf = cam.rays(uv[:, self.fwd[1]], undistort=False)
        Pb = cam.center + (D / np.abs(rb @ axis))[:, None] * rb
        Pf = cam.center + (D / np.abs(rf @ axis))[:, None] * rf
        v = Pf - Pb
        yaw = np.arctan2(v[:, 1], v[:, 0])
        return np.c_[P, yaw, np.zeros(len(P)), np.zeros(len(P))]

    def fit_template(self, uv: np.ndarray, lk: np.ndarray, iters: int = 60, sigma: float = 2.0) -> float:
        """Structure from motion over confident frames ``uv`` (n, m, 2).
        With ``fixed_z`` only the outline (x, y) is learned; heights relative
        to the ear line are fixed (unobservable from a camera looking down)."""
        n, m = uv.shape[:2]
        e0, e1 = self.ears
        tpl0 = np.zeros((m, 3))
        tpl0[e0], tpl0[e1] = [0, self.W / 2, 0], [0, -self.W / 2, 0]
        tpl0[self.fwd[1]] = [self.W * 1.4, 0, -12]
        tpl0[self.fwd[0]] = [-self.W * 0.1, 0, 0]
        for j in range(m):
            if j not in (e0, e1, *self.fwd):
                tpl0[j] = [self.W * 0.5, 0, -4]
        zfix = None
        if self.fixed_z:
            zfix = np.array([self.fixed_z.get(nm, tpl0[j, 2]) if j not in (e0, e1) else 0.0
                             for j, nm in enumerate(self.names)])
            tpl0[:, 2] = zfix
        nt = 2 * m if zfix is not None else 3 * m
        p0 = self._init_pose(uv)
        x0 = np.r_[(tpl0[:, :2] if zfix is not None else tpl0).ravel(), p0.ravel()]
        cam = self.cam

        def tpl(x):
            if zfix is not None:
                return np.c_[x[:nt].reshape(m, 2), zfix]
            return x[:nt].reshape(m, 3)

        def res(x):
            T = tpl(x)
            P = x[nt:].reshape(n, 6)
            Xw = P[:, None, :3] + np.einsum("nij,mj->nmi", _rotm(P[:, 3:]), T)
            r = ((cam.project(Xw, distort=False) - uv) / sigma * np.sqrt(np.clip(lk, 0, 1))[..., None]).ravel()
            mid = (T[e0] + T[e1]) / 2
            g = [mid * 10, [(np.linalg.norm(T[e0] - T[e1]) - self.W) * 10],
                 [(T[e0] - T[e1])[0] * 10, (T[e0] - T[e1])[2] * 10, (T[self.fwd[1]] - T[self.fwd[0]])[1] * 10],
                 np.maximum(0, -Xw[..., 2]).ravel() / 2]
            return np.r_[r, np.concatenate([np.ravel(a) for a in g])]
        sp = lil_matrix((len(res(x0)), len(x0)), dtype=np.int8)
        tc = list(range(nt))
        for k in range(n):
            sp[2 * m * k:2 * m * (k + 1), tc + list(range(nt + 6 * k, nt + 6 * k + 6))] = 1
        b = 2 * m * n
        sp[b:b + 7, :nt] = 1
        for k in range(n):
            for j in range(m):
                sp[b + 7 + k * m + j, tc + list(range(nt + 6 * k, nt + 6 * k + 6))] = 1
        out = least_squares(res, x0, jac_sparsity=sp.tocsr(), max_nfev=iters, x_scale="jac")
        T = tpl(out.x)
        self.T = T - (T[e0] + T[e1]) / 2
        err = np.abs(res(out.x)[:2 * m * n]).reshape(n, m, 2) * sigma
        return float(np.median(np.linalg.norm(err, axis=2)))

    def fit_sequence(self, uv: np.ndarray, lk: np.ndarray, box=None, *, sigma: float = 2.5,
                     roll_sd_deg: float = 20.0, iters: int = 150,
                     wall_tolerance_mm: float = 20.0, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
        """All frames jointly: reprojection, box limits, roll prior, smooth
        motion. ``uv`` (n, m, 2) may hold NaN (weight 0)."""
        n, m = uv.shape[:2]
        T, cam = self.T, self.cam
        x0 = self._init_pose(np.where(lk[..., None] > .5, uv, np.nan))
        for j in range(6):
            x0[:, j] = pd.Series(x0[:, j]).interpolate(limit_direction="both").to_numpy()
        x0 = np.nan_to_num(x0, nan=0.0)
        x0[:, 3] = np.unwrap(x0[:, 3])
        wt = np.sqrt(np.clip((np.nan_to_num(lk) - .3) / .6, 0, 1))
        wt[~np.isfinite(uv).all(2)] = 0
        uvf = np.nan_to_num(uv)
        if device != "cpu":
            return self._fit_sequence_torch(x0, uvf, wt, box, sigma=sigma, roll_sd_deg=roll_sd_deg,
                                            iters=iters, wall_tolerance_mm=wall_tolerance_mm, device=device)
        cont = (8.0, np.deg2rad(8))

        def res(x):
            P = x.reshape(n, 6)
            Xw = P[:, None, :3] + np.einsum("nij,mj->nmi", _rotm(P[:, 3:]), T)
            r = [((cam.project(Xw, distort=False) - uvf) / sigma * wt[..., None]).ravel()]
            r.append(np.concatenate([h.ravel() for h in _box_hinge(Xw, box, -wall_tolerance_mm)]) / 2.0)
            dP = np.diff(P, axis=0)
            r.append((dP[:, :3] / cont[0]).ravel())
            r.append((dP[:, 3:] / cont[1]).ravel())
            r.append(P[:, 5] / np.deg2rad(roll_sd_deg))
            return np.concatenate(r)
        nr = len(res(x0.ravel()))
        sp = lil_matrix((nr, 6 * n), dtype=np.int8)
        i = 0
        for k in range(n):
            sp[i:i + 2 * m, 6 * k:6 * k + 6] = 1
            i += 2 * m
        nh = 6 if box else 1
        for _ in range(nh):
            for k in range(n):
                sp[i:i + m, 6 * k:6 * k + 6] = 1
                i += m
        for _ in range(2):
            for k in range(n - 1):
                for c in range(3):
                    sp[i, 6 * k:6 * k + 12] = 1
                    i += 1
        for k in range(n):
            sp[i, 6 * k:6 * k + 6] = 1
            i += 1
        out = solve_sequence(res, x0.ravel(), sp.tocsr(), 6, max_nfev=iters)
        self.last_cost = float(out.cost)
        P = out.x.reshape(n, 6)
        Xw = P[:, None, :3] + np.einsum("nij,mj->nmi", _rotm(P[:, 3:]), T)
        return P, Xw

    def _fit_sequence_torch(self, x0, uvf, wt, box, *, sigma, roll_sd_deg, iters, wall_tolerance_mm, device):
        from mufasa.pose3d import gpu
        f = gpu.head_residual_fn(self.cam, self.T, box, sigma=sigma, roll_sd_deg=roll_sd_deg,
                                 wall_tolerance_mm=wall_tolerance_mm, device=device)
        out = gpu.solve_frames(f, x0, (uvf, wt), gpu.HEAD_SMOOTH_W, device=device, max_nfev=iters)
        self.last_cost = out.cost
        P = out.x
        T = self.T
        Xw = P[:, None, :3] + np.einsum("nij,mj->nmi", _rotm(P[:, 3:]), T)
        return P, Xw


# --------------------------------------------------------------------------- #
# Full single-camera reconstruction
# --------------------------------------------------------------------------- #
def resolve_device(spec: str) -> str:
    """``MonoConfig.device`` → "cuda", "torch-cpu" or "cpu" (SciPy).
    "auto": cuda when PyTorch sees a GPU, else torch-cpu when PyTorch is
    installed, else cpu."""
    if spec in (None, "", "cpu", "scipy"):
        return "cpu"
    from mufasa.pose3d import gpu
    if spec == "auto":
        return "cuda" if gpu.cuda_available() else ("torch-cpu" if gpu.available() else "cpu")
    if spec in ("cuda", "gpu"):
        gpu.resolve_device("cuda")                       # raises when unavailable
        return "cuda"
    if spec == "torch-cpu":
        gpu.resolve_device("cpu")
        return "torch-cpu"
    raise ValueError(f"unknown device {spec!r} (cpu, cuda, torch-cpu or auto)")


def _pool(workers: int, device: str = "cpu"):
    """A process pool for the per-chunk fits, or None for ``workers`` ≤ 1.
    No pool for the GPU: one process drives it (CUDA does not survive a
    fork)."""
    if workers is None or workers <= 1 or resolve_device(device) == "cuda":
        return None
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=int(workers))


def reconstruct_mono(pose: Pose2D, camera: Camera, cfg: MonoConfig | None = None,
                     progress=None) -> tuple[pd.DataFrame, dict]:
    """3-D pose from one camera. Returns (DataFrame, report).
    ``cfg.workers`` > 1 runs the head and spine chunks in parallel."""
    cfg = cfg or MonoConfig()
    pool = _pool(cfg.workers, cfg.device)
    try:
        return _reconstruct_mono(pose, camera, cfg, progress, pool)
    finally:
        if pool is not None:
            pool.shutdown()


def _reconstruct_mono(pose: Pose2D, camera: Camera, cfg: MonoConfig, progress,
                      pool) -> tuple[pd.DataFrame, dict]:
    say = progress or (lambda s: None)
    device = resolve_device(cfg.device)
    t_start = time.time()
    cam0 = replace(camera, dist=np.zeros(5))           # work in undistorted pixels
    N = pose.n
    U = {m: camera.undistort(pose.uv[m]) for m in pose.names}
    Uf = {m: camera.undistort(pose.filled[m]) for m in pose.names}
    L = pose.lk
    has = set(pose.names)
    report: dict = {"frames": N, "steps": []}
    report["solver"] = {"cuda": "PyTorch on the GPU", "torch-cpu": "PyTorch on the CPU"}.get(device, "SciPy")
    X: dict[str, np.ndarray] = {}
    box = camera.box

    # 1. head ---------------------------------------------------------------- #
    head = [cfg.head_back, cfg.head_mid, cfg.head_nose, cfg.head_left, cfg.head_right]
    if not set(head) <= has:
        raise ValueError(f"head markers missing from the 2-D pose: {sorted(set(head) - has)}")
    huv = np.stack([U[m] for m in head], 1)
    hlk = np.stack([L[m] for m in head], 1)
    rh = RigidHead(cam0, head, (3, 4), (0, 2), cfg.head_width_mm, cfg.head_template_z)
    good = np.flatnonzero((hlk > .95).all(1) & np.isfinite(huv).all((1, 2)))
    if len(good) < 50:
        raise ValueError(f"only {len(good)} frames with all five head labels confident (> 0.95)")
    pick = np.sort(np.random.default_rng(0).choice(good, min(1500, len(good)), replace=False))
    say(f"head shape from {len(pick)} confident frames")
    err = rh.fit_template(huv[pick], hlk[pick])
    report["head_template_mm"] = {m: rh.T[j].round(2).tolist() for j, m in enumerate(head)}
    report["head_template_reproj_px"] = err
    Xh = np.full((N, 5, 3), np.nan)
    CH, OV = cfg.chunk, 300
    spans = [(s, max(0, s - OV), min(N, s + CH + OV)) for s in range(0, N, CH)]
    kw = {"roll_sd_deg": cfg.head_roll_sd_deg, "iters": cfg.head_iters,
          "wall_tolerance_mm": cfg.wall_tolerance_mm, "device": device}
    say(f"head: {len(spans)} chunk(s) of {CH} frames" + (f" on {cfg.workers} workers" if pool else ""))
    results = (pool.map if pool else map)(_head_chunk, [(rh, huv[a:b], hlk[a:b], box, kw) for _, a, b in spans])
    for i, ((s, a, _), Xc) in enumerate(zip(spans, results)):
        k = min(CH, N - s)
        Xh[s:s + k] = Xc[s - a:s - a + k]
        say(f"head frames {s}–{s + k} of {N} ({i + 1}/{len(spans)})")
    for j, m in enumerate(head):
        X[m] = Xh[:, j]
    report["steps"].append("rigid head")

    # 2. trunk heights from Rat7M -------------------------------------------- #
    spine = [m for m in cfg.spine if m in has]
    hips = [m for m in cfg.hips if m in has]
    extras: dict[str, np.ndarray] = {}
    prior = Rat7MPrior() if (cfg.use_trunk_prior and (spine or hips)) else None
    hb = X[cfg.head_back]
    s_ = cfg.body_scale

    def place(m, z):
        X[m] = cam0.point_at_height(Uf[m], np.clip(z, 0, (box or {}).get("z", [0, 1e9])[1]), undistort=False)

    if prior is not None and len(hips) == 2:
        hm = X[cfg.head_mid] - hb
        pitch = np.degrees(np.arctan2(hm[:, 2], np.hypot(hm[:, 0], hm[:, 1])))
        hz = np.full(N, 45.0 * s_)
        for _ in range(4):
            for m in hips:
                place(m, hz)
            H = (X[hips[0]] + X[hips[1]]) / 2
            dh = np.hypot(*(hb[:, :2] - H[:, :2]).T)
            Y = prior.predict((hb[:, 2] + cfg.head_height_offset_mm) / s_, pitch - cfg.head_pitch_offset_deg, dh / s_)
            Y = _smooth(pd.DataFrame(Y).interpolate(limit_direction="both").to_numpy())
            Y[:, :4] *= s_
            hz = Y[:, 3]
        for m in hips:
            place(m, hz)
        zHB = hb[:, 2] + cfg.head_height_offset_mm
        node = {"HeadB": zHB, "SpineF": Y[:, 0], "SpineM": Y[:, 1], "SpineL": Y[:, 2]}
        node["SpineL+"] = Y[:, 2] + (Y[:, 2] - Y[:, 1])
        for m in spine:
            if m in cfg.spine_map:
                a_, b_, f_ = cfg.spine_map[m]
                place(m, node[a_] + f_ * (node[b_] - node[a_]))
        extras["lumbar_angle_deg"] = Y[:, 4]
        report["steps"].append("Rat7M trunk heights")
        # 3. pelvis pitched to the lumbar angle ------------------------------ #
        pv = [m for m in cfg.pelvis if m in has]
        if cfg.pelvis_from_lumbar and len(pv) == 2:
            tail_end, front = pv
            zc = (X[tail_end][:, 2] + X[front][:, 2]) / 2
            th = np.deg2rad(Y[:, 4])
            A, B = X[front], X[tail_end]
            for _ in range(4):
                dhh = np.hypot(*(A[:, :2] - B[:, :2]).T)
                dz = np.minimum(dhh * np.tan(np.clip(th, -1.4, 1.4)), cfg.max_pelvis_rise_mm)
                A = _on_ray(cam0, A, np.clip(zc + dz / 2, 3, 1e9))
                B = _on_ray(cam0, B, np.clip(zc - dz / 2, 3, 1e9))
            X[front], X[tail_end] = A, B
            report["steps"].append("pelvis pitched to the lumbar angle")
    else:
        for m in spine + hips:                         # no prior: a plain height guess
            place(m, np.full(N, 50.0 * s_))
        report["steps"].append("trunk at a fixed height (no Rat7M prior)")

    # 4. spine chain -------------------------------------------------------- #
    if cfg.spine_chain and prior is not None and len(spine) >= 3:
        chain = list(reversed(spine)) + [cfg.head_back]
        free = [m for m in reversed(spine) if m not in cfg.pelvis]
        for p in range(cfg.chain_passes):
            say(f"spine chain pass {p + 1}/{cfg.chain_passes}")
            _chain_refit(cam0, X, chain, free, prior, cfg, box, pool, device)
        report["steps"].append(f"spine chain ({cfg.chain_passes} passes)")

    # 5. tail ---------------------------------------------------------------- #
    tail = [m for m in cfg.tail if m in has]
    if cfg.drape_tail and tail:
        prev_name = cfg.pelvis[0] if cfg.pelvis[0] in X else (spine[-1] if spine else None)
        lengths = _tail_lengths(cam0, Uf, L, X, [prev_name] + tail, cfg)
        prev = X[prev_name] if prev_name else None
        for m, Lm in zip(tail, lengths):
            ray0 = cam0.point_at_height(Uf[m], np.full(N, cfg.tail_floor_mm), undistort=False)
            X[m] = ray0 if prev is None else _drape(cam0, ray0, prev, Lm, cfg.tail_floor_mm)
            prev = X[m]
        report["tail_segment_max_mm"] = [round(float(v), 1) for v in lengths]
        report["steps"].append("tail draped")

    # any remaining markers: on their ray at the floor
    for m in pose.names:
        if m not in X:
            X[m] = cam0.point_at_height(Uf[m], np.zeros(N), undistort=False)

    # derived columns ------------------------------------------------------ #
    if len([m for m in cfg.pelvis if m in X]) == 2:
        B, A = X[cfg.pelvis[0]], X[cfg.pelvis[1]]
        extras["pelvis_pitch_deg"] = np.degrees(np.arctan2(A[:, 2] - B[:, 2], np.hypot(*(A[:, :2] - B[:, :2]).T)))
    sp_ = [X[m] for m in spine if m != cfg.pelvis[0]]
    if len(sp_) >= 2:
        extras["spine_length_mm"] = sum(np.linalg.norm(sp_[i + 1] - sp_[i], axis=1) for i in range(len(sp_) - 1))
    cols: dict = {"frame": np.arange(N)}
    for m in pose.names:
        cols[m + "_x"], cols[m + "_y"], cols[m + "_z"] = X[m].T
        cols[m + "_likelihood"] = L[m]
        cols[m + "_reproj_px"] = np.linalg.norm(camera.project(X[m]) - pose.uv[m], axis=1)
    cols.update(extras)
    report["seconds"] = round(time.time() - t_start, 1)
    return pd.DataFrame(cols), report


# --------------------------------------------------------------------------- #
# Spine chain refit
# --------------------------------------------------------------------------- #
def _chain_refit(cam: Camera, X: dict, chain: list[str], free: list[str], prior: Rat7MPrior,
                 cfg: MonoConfig, box, pool=None, device: str = "cpu") -> None:
    """Refit the heights of ``free`` chain labels (tail→head order in
    ``chain``) on their rays; see the module docstring. Chunks run on
    ``pool`` (a process pool) when given."""
    N = len(X[chain[0]])
    C = cam.center
    tail_end, head_end = X[chain[0]], X[chain[-1]]
    hd = head_end[:, :2] - tail_end[:, :2]
    hd /= np.maximum(np.linalg.norm(hd, axis=1, keepdims=True), 1e-6)
    lim = prior.bend_limits((head_end[:, 2] + cfg.head_height_offset_mm) / cfg.body_scale)
    n_mid = max(1, len(chain) - 4)                      # joints between pelvis and neck
    D = {m: (X[m] - C) for m in free}
    CH, OV = cfg.chunk, 100
    spans = [(s, max(0, s - OV), min(N, s + CH + OV)) for s in range(0, N, CH)]
    payloads = [{
        "C": C, "chain": chain, "free": free, "n_mid": n_mid, "device": device,
        "fixed": {m: X[m][a:b] for m in chain if m not in free},
        "D": {m: D[m][a:b] for m in free},
        "z0": np.stack([X[m][a:b, 2] for m in free], 1),
        "hd": hd[a:b], "lim": {k: v[a:b] for k, v in lim.items()},
    } for _, a, b in spans]
    results = (pool.map if pool is not None else map)(_chain_chunk, payloads)
    Z = np.zeros((N, len(free)))
    for (s, a, _), z in zip(spans, results):
        k = min(CH, N - s)
        Z[s:s + k] = z[s - a:s - a + k]
    zmax = (box or {}).get("z", [0, 1e9])[1]
    for i, m in enumerate(free):
        X[m] = _on_ray(cam, X[m], np.clip(Z[:, i], 3, zmax))


def _chain_chunk(p: dict) -> np.ndarray:
    """One chunk of the spine-chain refit (a top-level function so a
    process pool can run it). Returns the free labels' heights (n, nf)."""
    C, chain, free, n_mid = p["C"], p["chain"], p["free"], p["n_mid"]
    D, z0, hdl, fixed = p["D"], p["z0"], p["hd"], p["fixed"]
    n, nf = z0.shape
    if p.get("device", "cpu") != "cpu":
        from mufasa.pose3d import gpu
        f = gpu.chain_residual_fn(C, chain, free, n_mid, p["device"])
        Dk = np.stack([D[m] for m in free], 1)                               # (n, nf, 3)
        Fx = np.stack([fixed[m] for m in chain if m not in free], 1)        # (n, nfix, 3)
        L4 = np.stack([p["lim"][k] for k in ("M50", "M95", "F5", "F95")], 1)
        o = gpu.solve_frames(f, z0, (Dk, Fx, z0, hdl, L4), np.full(nf, 1 / 64.0),
                             device=p["device"], max_nfev=60)
        return o.x
    M50, M95, F5, F95 = (p["lim"][k] for k in ("M50", "M95", "F5", "F95"))
    rad = np.deg2rad

    def elev(a, b):
        return np.arctan2(b[:, 2] - a[:, 2], np.einsum("ij,ij->i", b[:, :2] - a[:, :2], hdl))

    def res(x):
        z = x.reshape(n, nf)
        Q = {}
        for i, m in enumerate(free):
            d = D[m]
            Q[m] = C + ((z[:, i] - C[2]) / d[:, 2])[:, None] * d
        nodes = [Q[m] if m in Q else fixed[m] for m in chain]
        a = [elev(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        bends = [a[i + 1] - a[i] for i in range(len(a) - 1)]   # at chain[1..-2]
        r = [((z - z0) / 20.0).ravel()]
        for j, bd in enumerate(bends):
            node_i = j + 1
            if node_i == 1:                                   # pelvis → lumbar
                r.append(np.maximum(0, bd - rad(10)) / rad(2))
            elif node_i == len(chain) - 2:                    # neck
                r.append(np.maximum(0, bd - rad(F95)) / rad(3))
                r.append(np.maximum(0, rad(F5) - bd) / rad(3))
            else:                                             # mid-back
                r.append((bd - rad(M50) / n_mid) / rad(15))
                r.append(np.maximum(0, bd - rad(M95) / n_mid - rad(3)) / rad(2))
        r.append((np.diff(z, axis=0) / 8.0).ravel())
        r.append(np.maximum(0, 3 - z).ravel())
        return np.concatenate([np.ravel(v) for v in r])
    r0 = res(z0.ravel())
    sp = lil_matrix((len(r0), nf * n), dtype=np.int8)
    i = 0
    for k in range(n):
        sp[i:i + nf, nf * k:nf * k + nf] = 1
        i += nf
    nb = len(r0) - nf * n - (n - 1) * nf - n * nf
    for _ in range(nb // n):
        for k in range(n):
            sp[i, nf * k:nf * k + nf] = 1
            i += 1
    for k in range(n - 1):
        for c in range(nf):
            sp[i, nf * k:nf * k + 2 * nf] = 1
            i += 1
    for k in range(n):
        sp[i:i + nf, nf * k:nf * k + nf] = 1
        i += nf
    o = solve_sequence(res, z0.ravel(), sp.tocsr(), nf, max_nfev=60)
    return o.x.reshape(n, nf)


def _head_chunk(p: tuple) -> np.ndarray:
    """One chunk of the head fit, for a process pool."""
    rh, uv, lk, box, kw = p
    return rh.fit_sequence(uv, lk, box, **kw)[1]


# --------------------------------------------------------------------------- #
# Tail
# --------------------------------------------------------------------------- #
def _drape(cam: Camera, P: np.ndarray, prev: np.ndarray, Lm: float, zmin: float) -> np.ndarray:
    """Lowest point (≥ ``zmin``) on each ray within ``Lm`` of ``prev``;
    the closest approach when none is."""
    C = cam.center
    d = P - C
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    zs = np.linspace(zmin, max(zmin + 1, C[2] * 0.8), 120)
    best = np.full(len(P), np.nan)
    dist_min = np.full(len(P), np.inf)
    z_at_min = np.full(len(P), zmin)
    for z in zs:
        Q = C + ((z - C[2]) / d[:, 2])[:, None] * d
        dd = np.linalg.norm(Q - prev, axis=1)
        ok = (dd <= Lm) & np.isnan(best)
        best[ok] = z
        better = dd < dist_min
        dist_min[better], z_at_min[better] = dd[better], z
    best = np.where(np.isnan(best), z_at_min, best)
    return C + ((best - C[2]) / d[:, 2])[:, None] * d


def _tail_lengths(cam: Camera, Uf: dict, L: dict, X: dict, chain: list, cfg: MonoConfig) -> list[float]:
    """Max length of each tail segment from the session: 90th percentile of
    the distance between confident labels lying on the floor (the first
    segment: from the pelvis marker in frames with the head low)."""
    out = []
    N = len(next(iter(Uf.values())))
    floor = {m: cam.point_at_height(Uf[m], np.full(N, cfg.tail_floor_mm), undistort=False)
             for m in chain[1:]}
    for a, b in zip(chain[:-1], chain[1:]):
        if a is None:
            out.append(60.0)
            continue
        if a in floor:
            ok = (L[a] > .95) & (L[b] > .95)
            dd = np.linalg.norm(floor[a] - floor[b], axis=1)[ok]
        else:
            low = X[cfg.head_back][:, 2] < np.nanpercentile(X[cfg.head_back][:, 2], 40)
            ok = (L[b] > .95) & low
            dd = np.linalg.norm(X[a] - floor[b], axis=1)[ok]
        out.append(float(np.nanpercentile(dd, 90)) if np.isfinite(dd).sum() > 50 else 60.0)
    return out


__all__ = ["MonoConfig", "Pose2D", "Rat7MPrior", "RigidHead", "load_pose2d", "reconstruct_mono"]
