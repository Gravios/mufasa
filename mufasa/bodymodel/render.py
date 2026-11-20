"""
mufasa.bodymodel.render
=======================

Patch 122kq: draw the fitted model over video frames — shaded with the
fur colours baked into the rig (painter's algorithm, light from above),
optionally the outline, the 2-D labels and the model's label points — and
write an overlay video.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mufasa.bodymodel.fit import Markers, Rig, load_labels, observed, params_at, read_fit, to_image

LIGHT = np.array([0.35, -0.45, 0.82])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


@dataclass
class OverlayStyle:
    alpha: float = 0.8
    outline: bool = True
    labels: bool = True
    info: bool = True
    side_by_side: bool = False       # original | overlay


def draw_model(img, rig: Rig, p: dict, style: OverlayStyle | None = None, *, obs=None, markers=None,
               text: str | None = None):
    """The model at posture ``p`` over ``img`` (BGR); returns a new image."""
    import cv2
    style = style or OverlayStyle()
    G = rig.skinning_matrices(p)
    P3 = Rig.lbs(G, rig.verts(p), rig.W)
    uv = to_image(P3, p)
    tri = P3[rig.F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    up = n[:, 2] > 0                                         # seen from above
    shade = 0.35 + 0.65 * np.clip(n @ LIGHT, 0, 1)
    base = (rig.face_rgb[:, ::-1].astype(float) if rig.face_rgb is not None
            else np.full((len(rig.F), 3), 70.0))
    col = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)
    order = np.argsort(tri[:, :, 2].mean(1))                  # low first, so higher parts are on top
    layer = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    pts = np.round(uv[rig.F] * 4).astype(np.int32)
    for k in order:
        if not up[k] and tri[k, :, 2].mean() > 2.0:          # back faces of raised parts are hidden
            continue
        c = col[k]
        cv2.fillConvexPoly(layer, pts[k], (int(c[0]), int(c[1]), int(c[2])), lineType=cv2.LINE_AA, shift=2)
        cv2.fillConvexPoly(mask, pts[k], 255, shift=2)
    a = (mask.astype(np.float32) / 255.0 * style.alpha)[..., None]
    out = (img.astype(np.float32) * (1 - a) + layer.astype(np.float32) * a).astype(np.uint8)
    if style.outline:
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (0, 165, 255), 1, cv2.LINE_AA)
    if style.labels and obs is not None and markers is not None:
        mp = markers.positions(rig, G)
        for name, X in mp.items():
            if name not in obs or obs[name][1] < 0.5 or not np.isfinite(obs[name][0]).all():
                continue
            m = to_image(X[None], p)[0]
            o = obs[name][0]
            cv2.line(out, tuple(np.round(m).astype(int)), tuple(np.round(o).astype(int)), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(out, tuple(np.round(o).astype(int)), 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.drawMarker(out, tuple(np.round(m).astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 10, 2)
    if style.info and text:
        cv2.putText(out, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    if style.side_by_side:
        out = np.hstack([img, np.full((img.shape[0], 6, 3), 255, np.uint8), out])
    return out


def render_overlay_video(fit_path, out_path, *, rig_path=None, video=None, pose_file=None,
                         style: OverlayStyle | None = None, every_frame: bool = True, progress=None,
                         say=print) -> Path:
    """Overlay video for a fit (frames between fitted ones are interpolated when ``every_frame``)."""
    import cv2
    df, meta = read_fit(fit_path)
    rig = Rig(rig_path or meta["rig"])
    video = video or meta["video"]
    pose_file = pose_file or meta.get("pose")
    labels = load_labels(pose_file) if pose_file and Path(pose_file).is_file() else None
    markers = Markers.build(rig, labels) if labels else None
    style = style or OverlayStyle()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f0, f1 = int(df["frame"].iloc[0]), int(df["frame"].iloc[-1])
    frames = list(range(f0, f1 + 1)) if every_frame else [int(f) for f in df["frame"]]
    step = int(meta.get("config", {}).get("step", 1) or 1)
    out_fps = fps if every_frame else fps / max(1, step)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (2 * W + 6, H) if style.side_by_side else (W, H)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*("MJPG" if out_path.suffix.lower() == ".avi"
                                                                 else "mp4v")), out_fps, size)
    if not wr.isOpened():
        cap.release()
        raise OSError(f"cannot write {out_path}")
    by_frame = df.set_index("frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    pos = f0
    try:
        for k, f in enumerate(frames):
            while pos < f:
                cap.grab()
                pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok:
                break
            p = params_at(df, f)
            if f in by_frame.index:
                r = by_frame.loc[f]
                iou = "" if not np.isfinite(r["iou"]) else f"IoU {r['iou']:.2f}  "
                txt = f"frame {f}  {iou}labels {r['kp_rms_px']:.0f} px"
            else:
                txt = f"frame {f}  (between fits)"
            obs = observed(labels, f) if labels else None
            wr.write(draw_model(img, rig, p, style, obs=obs, markers=markers, text=txt))
            if progress and (k % 10 == 0 or k == len(frames) - 1):
                progress(k + 1, len(frames))
    finally:
        wr.release()
        cap.release()
    say(f"overlay video -> {out_path} ({len(frames)} frames)")
    return out_path


__all__ = ["OverlayStyle", "draw_model", "render_overlay_video"]
