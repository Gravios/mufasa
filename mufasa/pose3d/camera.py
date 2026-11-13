"""
mufasa.pose3d.camera
====================

Patch 122kf: camera models and calibration for 3-D pose.

World frame: millimetres, ``x``/``y`` on the floor (the box floor centre is
the origin when calibrated from the box), ``z`` up, floor at ``z = 0``.

:class:`Camera` is a pinhole camera with optional lens distortion
(OpenCV model): ``K`` (3×3), ``dist`` (k1, k2, p1, p2, k3, …), and the
world→camera pose ``R`` (3×3), ``t`` (3,). It projects world points to
pixels, turns pixels into rays, and finds the point on a pixel's ray at a
given height.

Calibration (the box-, checkerboard- and DLT-based tools are adapted from
rpimocap, MIT, same author):

* :func:`calibrate_from_box_marks` — from marked floor corners and wall
  corner edges in one image, the box size and the camera height (or a
  checkerboard ``K``/``dist``, in which case the height is solved);
* :func:`calibrate_checkerboard` — intrinsics (``K``, ``dist``) from a
  video or image folder of a checkerboard;
* :func:`dlt_camera_matrix` / :func:`camera_from_projection_matrix` —
  a 3×4 projection from ≥6 known 3-D points (e.g. box corners at known
  heights), as rpimocap's ``calibrate_from_corners``.

Cameras are saved as TOML (``Camera.save`` / ``Camera.load``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    image_size: tuple[int, int] = (0, 0)          # (width, height)
    box: dict | None = None                       # {"x": [lo, hi], "y": [..], "z": [..]} mm
    notes: str = ""

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, float).reshape(3, 3)
        self.R = np.asarray(self.R, float).reshape(3, 3)
        self.t = np.asarray(self.t, float).reshape(3)
        self.dist = np.asarray(self.dist, float).ravel()

    # -- geometry --------------------------------------------------------- #
    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates (mm)."""
        return -self.R.T @ self.t

    @property
    def P(self) -> np.ndarray:
        """3×4 projection for undistorted pixels."""
        return self.K @ np.c_[self.R, self.t]

    @property
    def height(self) -> float:
        return float(self.center[2])

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(np.abs(self.dist) > 0))

    def project(self, X: np.ndarray, distort: bool = True) -> np.ndarray:
        """World points ``(..., 3)`` → pixels ``(..., 2)``."""
        X = np.asarray(X, float)
        sh = X.shape[:-1]
        Xc = X.reshape(-1, 3) @ self.R.T + self.t
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = Xc[:, :2] / Xc[:, 2:3]
        if distort and self.has_distortion:
            xn = _distort_normalized(xn, self.dist)
        uv = xn @ self.K[:2, :2].T + self.K[:2, 2]
        return uv.reshape(*sh, 2)

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """Pixels → ideal (undistorted) pixels with the same ``K``."""
        uv = np.asarray(uv, float)
        if not self.has_distortion:
            return uv.copy()
        import cv2
        sh = uv.shape
        flat = uv.reshape(-1, 2)
        ok = np.isfinite(flat).all(1)
        out = np.full_like(flat, np.nan)
        if ok.any():
            u = cv2.undistortPoints(flat[ok].reshape(-1, 1, 2).astype(np.float64),
                                    self.K, self.dist, P=self.K)
            out[ok] = u.reshape(-1, 2)
        return out.reshape(sh)

    def rays(self, uv: np.ndarray, undistort: bool = True) -> np.ndarray:
        """Unit ray directions in world coordinates for pixels ``(..., 2)``."""
        uv = self.undistort(uv) if undistort else np.asarray(uv, float)
        sh = uv.shape[:-1]
        h = np.c_[uv.reshape(-1, 2), np.ones(int(np.prod(sh)) if sh else 1)]
        d = (np.linalg.inv(self.K) @ h.T).T @ self.R          # camera → world (R^T d)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d.reshape(*sh, 3)

    def point_at_height(self, uv: np.ndarray, z, undistort: bool = True) -> np.ndarray:
        """World point on each pixel's ray at height ``z`` (mm)."""
        d = self.rays(uv, undistort=undistort)
        C = self.center
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (np.asarray(z, float) - C[2]) / d[..., 2]
        return C + s[..., None] * d

    def depth(self, X: np.ndarray) -> np.ndarray:
        """Distance along the optical axis (mm)."""
        return (np.asarray(X, float).reshape(-1, 3) @ self.R.T + self.t)[:, 2].reshape(
            np.asarray(X).shape[:-1])

    @property
    def focal(self) -> float:
        return float((self.K[0, 0] + self.K[1, 1]) / 2)

    # -- constructors ----------------------------------------------------- #
    @classmethod
    def straight_down(cls, principal_px, focal_px: float, height_mm: float,
                      image_size=(0, 0), center_xy=(0.0, 0.0), dist=None,
                      box: dict | None = None, notes: str = "") -> Camera:
        """Camera at ``(center_xy, height)`` looking straight down; image x
        runs along world +x, image y along world −y."""
        K = np.array([[focal_px, 0, principal_px[0]], [0, focal_px, principal_px[1]], [0, 0, 1.0]])
        R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        C = np.array([center_xy[0], center_xy[1], height_mm], float)
        return cls(K, R, -R @ C, np.zeros(5) if dist is None else dist,
                   tuple(int(v) for v in image_size), box, notes)

    # -- persistence ------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = {"K": self.K.tolist(), "dist": self.dist.tolist(), "R": self.R.tolist(),
             "t": self.t.tolist(), "image_size": list(self.image_size),
             "center_mm": self.center.round(3).tolist(), "notes": self.notes}
        if self.box:
            d["box"] = {k: [float(x) for x in v] for k, v in self.box.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Camera:
        box = d.get("box")
        return cls(np.array(d["K"]), np.array(d["R"]), np.array(d["t"]),
                   np.array(d.get("dist", [0] * 5)), tuple(d.get("image_size", (0, 0))),
                   {k: list(v) for k, v in box.items()} if box else None, d.get("notes", ""))

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_to_toml({"camera": self.to_dict()}))
        return path

    @classmethod
    def load(cls, path) -> Camera:
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh)["camera"])

    def viewer_meta(self) -> dict:
        """Camera block for the 3-D viewer's metadata (``P`` for undistorted
        pixels, plus the straight-down keys when they apply)."""
        m = {"P": self.P.tolist(), "image_height_px": int(self.image_size[1] or 0)}
        if np.allclose(self.R, np.diag([1.0, -1.0, -1.0]), atol=1e-9):
            C = self.center
            if abs(C[0]) < 1e-9 and abs(C[1]) < 1e-9:
                m.update(height_mm=float(C[2]), focal_px=self.focal,
                         principal_px=[float(self.K[0, 2]), float(self.K[1, 2])])
        return m


def _distort_normalized(xn: np.ndarray, dist: np.ndarray) -> np.ndarray:
    k = np.zeros(8)
    k[:min(8, len(dist))] = dist[:8]
    k1, k2, p1, p2, k3, k4, k5, k6 = k
    x, y = xn[:, 0], xn[:, 1]
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.c_[xd, yd]


def _to_toml(d: dict, prefix: str = "") -> str:
    """Minimal TOML writer for nested dicts of numbers/strings/lists."""
    lines, tables = [], []
    for k, v in d.items():
        if isinstance(v, dict):
            tables.append((k, v))
        else:
            lines.append(f"{k} = {_toml_value(v)}")
    out = "\n".join(lines) + ("\n" if lines else "")
    for k, v in tables:
        name = f"{prefix}.{k}" if prefix else k
        out += f"\n[{name}]\n" + _to_toml(v, name)
    return out


def _toml_value(v) -> str:
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v))


# --------------------------------------------------------------------------- #
# Line / vanishing-point helpers
# --------------------------------------------------------------------------- #
def fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Total-least-squares line: (centre, unit direction, rms distance)."""
    P = np.asarray(points, float).reshape(-1, 2)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    return c, d, float(np.sqrt(np.mean(((P - c) @ n) ** 2)))


def vanishing_point(lines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares point closest to several image lines (each given as
    ≥2 points). Returns (point, per-line distances in px)."""
    A, b = [], []
    for pts in lines:
        c, d, _ = fit_line(pts)
        n = np.array([-d[1], d[0]])
        A.append(n)
        b.append(n @ c)
    A, b = np.array(A), np.array(b)
    vp = np.linalg.lstsq(A, b, rcond=None)[0]
    return vp, A @ vp - b


def order_quad(corners: np.ndarray) -> np.ndarray:
    """Order four image points as top-left, top-right, bottom-right,
    bottom-left (image coordinates)."""
    c = np.asarray(corners, float).reshape(4, 2)
    s, dfn = c.sum(1), np.diff(c, axis=1).ravel()
    return np.array([c[np.argmin(s)], c[np.argmin(dfn)], c[np.argmax(s)], c[np.argmax(dfn)]])


# --------------------------------------------------------------------------- #
# Calibration from a marked box
# --------------------------------------------------------------------------- #
@dataclass
class BoxCalibrationReport:
    principal_px: tuple[float, float]
    px_per_mm: float
    floor_mm: tuple[float, float]
    measured_aspect: float
    given_aspect: float | None
    keystone_pct: float
    vp_residual_px: float | None
    height_mm: float
    method: str
    warnings: list[str] = field(default_factory=list)

    def text(self) -> str:
        s = [f"method: {self.method}",
             f"principal point (below the camera): ({self.principal_px[0]:.1f}, {self.principal_px[1]:.1f}) px",
             f"floor: {self.floor_mm[0]:.0f} × {self.floor_mm[1]:.0f} mm, {self.px_per_mm:.3f} px/mm",
             f"floor aspect in the image {self.measured_aspect:.2f}"
             + (f" (box as entered {self.given_aspect:.2f})" if self.given_aspect else ""),
             f"keystone {self.keystone_pct:.1f}% (0 = straight down)",
             f"camera height {self.height_mm:.0f} mm"]
        if self.vp_residual_px is not None:
            s.append(f"wall-corner edges miss their common point by {self.vp_residual_px:.1f} px (rms)")
        return "\n".join(s + [f"warning: {w}" for w in self.warnings])


def calibrate_from_box_marks(floor_corners_px, *, floor_length_mm: float | None = None,
                             floor_width_mm: float | None = None,
                             wall_height_mm: float | None = None,
                             vertical_edges_px: list | None = None,
                             camera_height_mm: float | None = None,
                             image_size=(0, 0), K=None, dist=None,
                             trust: str = "auto") -> tuple[Camera, BoxCalibrationReport]:
    """Camera from one image of the box.

    ``floor_corners_px`` — the four floor corners (any order); the long
    image side is the box length. ``vertical_edges_px`` — optional point
    lists along the box's vertical corner edges; where they meet is the
    point directly below the camera.

    Without ``K``: a straight-down camera (the floor keystone is reported),
    principal point at the vertical edges' meeting point (else the image
    centre), scale from the floor and ``camera_height_mm`` (required).
    If length and width disagree with the marked floor's shape, ``trust``
    picks one: ``"width"``, ``"length"``, or ``"auto"`` (width).

    With ``K`` (and ``dist``) from a checkerboard: the full pose from the
    four corners (perspective-n-point), including the camera height.
    """
    import cv2
    warnings: list[str] = []
    q = order_quad(floor_corners_px)
    if K is not None:
        K = np.asarray(K, float)
        dist = np.zeros(5) if dist is None else np.asarray(dist, float)
        if not (floor_length_mm and floor_width_mm):
            raise ValueError("a checkerboard calibration needs both floor length and width")
        L, W = floor_length_mm, floor_width_mm
        top, left = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
        if top < left:
            L, W = W, L
        obj = np.array([[-L / 2, W / 2, 0], [L / 2, W / 2, 0], [L / 2, -W / 2, 0], [-L / 2, -W / 2, 0]])
        ok, rvec, tvec = cv2.solvePnP(obj, q.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise RuntimeError("could not solve the camera pose from the floor corners")
        R = cv2.Rodrigues(rvec)[0]
        box = {"x": [-L / 2, L / 2], "y": [-W / 2, W / 2], "z": [0.0, float(wall_height_mm or 0) or 1e9]}
        cam = Camera(K, R, tvec.ravel(), dist, tuple(int(v) for v in image_size), box,
                     "box corners + checkerboard intrinsics (solvePnP)")
        err = np.linalg.norm(cam.project(obj) - q, axis=1)
        rep = BoxCalibrationReport((float(K[0, 2]), float(K[1, 2])), float(cam.focal / max(cam.height, 1)),
                                   (L, W), float(top / left if top > left else left / top), L / W,
                                   0.0, None, cam.height, "checkerboard intrinsics + box corners",
                                   [f"corner reprojection rms {np.sqrt(np.mean(err ** 2)):.1f} px"]
                                   if np.sqrt(np.mean(err ** 2)) > 3 else [])
        return cam, rep
    if not camera_height_mm:
        raise ValueError("camera_height_mm is needed without a checkerboard calibration "
                         "(a straight-down view of the floor fixes the scale, not the height)")
    w_img, h_img = image_size if image_size and image_size[0] else (None, None)
    vp_res = None
    if vertical_edges_px and len(vertical_edges_px) >= 2:
        vp, res = vanishing_point(vertical_edges_px)
        vp_res = float(np.sqrt(np.mean(res ** 2)))
        principal = vp
    else:
        principal = np.array([w_img / 2, h_img / 2]) if w_img else q.mean(0)
        warnings.append("no wall-corner edges: principal point set to the image centre")
    top, bot = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    lef, rig = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    keystone = 100 * max(abs(top - bot) / max(top, bot), abs(lef - rig) / max(lef, rig))
    if keystone > 5:
        warnings.append(f"floor keystone {keystone:.0f}%: the camera is tilted; a straight-down "
                        "model will be off — calibrate the lens with a checkerboard")
    long_px, short_px = max((top + bot) / 2, (lef + rig) / 2), min((top + bot) / 2, (lef + rig) / 2)
    horiz_long = (top + bot) / 2 >= (lef + rig) / 2
    aspect = long_px / short_px
    given = None
    Lg, Wg = floor_length_mm, floor_width_mm
    if Lg and Wg:
        Lg, Wg = max(Lg, Wg), min(Lg, Wg)
        given = Lg / Wg
        if abs(given - aspect) / aspect > 0.08:
            pick = "width" if trust == "auto" else trust
            warnings.append(f"the marked floor is {aspect:.2f}× as long as wide, the box as entered "
                            f"{given:.2f}×; using the {pick}")
            if pick == "length":
                Wg = Lg / aspect
            else:
                Lg = Wg * aspect
    elif Wg:
        Lg = Wg * aspect
    elif Lg:
        Wg = Lg / aspect
    else:
        raise ValueError("give the floor length and/or width")
    s = (long_px / Lg + short_px / Wg) / 2                    # px per mm at the floor
    f = s * camera_height_mm
    # floor centre → world origin; camera above (cx, cy) of the principal point
    fc = q.mean(0)
    off = (fc - principal) / s                                # mm, image axes
    center_xy = (-off[0], off[1])                             # image y down = world −y
    if not horiz_long:
        warnings.append("the box's long side runs vertically in the image (x = image right)")
    box = {"x": [-((top + bot) / 2) / s / 2, ((top + bot) / 2) / s / 2],
           "y": [-((lef + rig) / 2) / s / 2, ((lef + rig) / 2) / s / 2],
           "z": [0.0, float(wall_height_mm or 1e9)]}
    cam = Camera.straight_down(principal, f, camera_height_mm, image_size, center_xy, dist,
                               box, "straight-down camera from marked box edges")
    rep = BoxCalibrationReport((float(principal[0]), float(principal[1])), float(s),
                               (float(Lg), float(Wg)), float(aspect), given, float(keystone),
                               vp_res, float(camera_height_mm),
                               "straight down; principal point from wall-corner edges"
                               if vp_res is not None else "straight down", warnings)
    return cam, rep


# --------------------------------------------------------------------------- #
# Checkerboard intrinsics (adapted from rpimocap.calibration.checkerboard)
# --------------------------------------------------------------------------- #
def _frames(src: str, skip: int = 8):
    import cv2
    p = Path(src)
    if p.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for i, f in enumerate(sorted(x for x in p.iterdir() if x.suffix.lower() in exts)):
            im = cv2.imread(str(f))
            if im is not None:
                yield i, im
        return
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise OSError(f"cannot open {src}")
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % skip == 0:
            yield i, im
        i += 1
    cap.release()


def calibrate_checkerboard(source: str, pattern: tuple[int, int], square_mm: float, *,
                           skip: int = 8, max_views: int = 60, rational: bool = False,
                           progress=None) -> tuple[np.ndarray, np.ndarray, float, int, tuple[int, int]]:
    """Lens calibration from a checkerboard video or image folder.

    ``pattern`` — inner corners (columns, rows). Returns ``(K, dist, rms_px,
    n_views, image_size)``."""
    import cv2
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    img_pts, size = [], None
    for i, im in _frames(source, skip):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        size = (g.shape[1], g.shape[0])
        ok, c = cv2.findChessboardCorners(g, (cols, rows), None)
        if ok:
            img_pts.append(cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit))
            if progress:
                progress(f"checkerboard found in frame {i} ({len(img_pts)} views)")
            if len(img_pts) >= max_views:
                break
    if len(img_pts) < 5:
        raise RuntimeError(f"checkerboard found in only {len(img_pts)} views; need at least 5")
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    rms, K, dist, _, _ = cv2.calibrateCamera([objp] * len(img_pts), img_pts, size, None, None, flags=flags)
    return K, dist.ravel(), float(rms), len(img_pts), size


# --------------------------------------------------------------------------- #
# DLT from known 3-D points (adapted from rpimocap.cli.calibrate_from_corners)
# --------------------------------------------------------------------------- #
def dlt_camera_matrix(pts3d: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
    """3×4 projection from ≥6 non-coplanar 3-D ↔ 2-D correspondences."""
    pts3d, pts2d = np.asarray(pts3d, float), np.asarray(pts2d, float)
    if len(pts3d) < 6:
        raise ValueError("DLT needs at least 6 points")
    A = []
    for (X, Y, Z), (u, v) in zip(pts3d, pts2d):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    _, _, vt = np.linalg.svd(np.array(A))
    P = vt[-1].reshape(3, 4)
    return P / np.linalg.norm(P[2, :3])


def camera_from_projection_matrix(P: np.ndarray, image_size=(0, 0), box=None) -> Camera:
    import cv2
    K, R, Ch = cv2.decomposeProjectionMatrix(np.asarray(P, float))[:3]
    K = K / K[2, 2]
    C = (Ch[:3] / Ch[3]).ravel()
    if np.linalg.det(R) < 0:
        R = -R
    return Camera(K, R, -R @ C, np.zeros(5), tuple(int(v) for v in image_size), box,
                  "from a 3×4 projection (DLT)")


def reprojection_rmse(cam: Camera, pts3d, pts2d) -> float:
    e = cam.project(np.asarray(pts3d, float)) - np.asarray(pts2d, float)
    return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))


def camera_for_video(calibration_dir, video_stem: str) -> Path | None:
    """The camera file for a video: ``<stem>.camera.toml``, else
    ``camera_<first token of the stem>.toml`` (e.g. ``camera_K7.toml`` for
    ``K7_cam_…``), else ``camera.toml``."""
    d = Path(calibration_dir)
    for name in (f"{video_stem}.camera.toml", f"camera_{video_stem.split('_')[0]}.toml", "camera.toml"):
        if (d / name).is_file():
            return d / name
    return None


__all__ = ["BoxCalibrationReport", "Camera", "calibrate_checkerboard", "calibrate_from_box_marks",
           "camera_for_video", "camera_from_projection_matrix", "dlt_camera_matrix", "fit_line",
           "order_quad", "reprojection_rmse", "vanishing_point"]
_ = math
