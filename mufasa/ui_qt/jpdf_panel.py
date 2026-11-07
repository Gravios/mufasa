"""
mufasa.ui_qt.jpdf_panel
=======================

A joint-PDF panel shown beside the video in the feature viewer (patch
122jv): the 2-D log density of two features, pooled over every video of the
project, with the current frame's position as a solid red marker and its
recent path as a fading red trail.

The panel is composed onto the right of the displayed frame
(:meth:`FrameScrubberWidget.set_composer`), so what the viewer shows — and
what "● Record" saves — is the video and the JPDF together.

The density is drawn once per (pair, panel height, names, scales) with
matplotlib and cached as an image; each frame only draws the marker and
trail on a copy, mapping feature values to pixels with the axes transform.
Feature values: −1 is read as missing; features marked log10 in
``feature_names.toml`` are shown as log10 (labelled "log10 <name>"); the
background (not the marker) is jittered for features marked for jitter.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

MARKER_BGR = (0, 0, 255)          # solid red
POOL_PER_VIDEO = 5000


def _clean(v) -> np.ndarray:
    v = np.asarray(v, np.float64).copy()
    v[v == -1] = np.nan
    v[~np.isfinite(v)] = np.nan
    return v


def _scaled(v: np.ndarray, log10: bool) -> np.ndarray:
    if not log10:
        return v
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)


def pooled_values(config_path, cols: list[str], per_video: int = POOL_PER_VIDEO,
                  seed: int = 0) -> pd.DataFrame:
    """``cols`` from every video with features, at most ``per_video`` frames
    each (uniform sample)."""
    from mufasa.utils.feature_io import list_video_stems_with_features, load_feature_columns
    parts = []
    for v in list_video_stems_with_features(str(config_path)):
        try:
            df = load_feature_columns(v, config_path, cols)
        except Exception:
            continue
        if not set(cols) <= set(df.columns):
            continue
        if len(df) > per_video:
            df = df.sample(per_video, random_state=seed)
        parts.append(df[cols])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


@dataclass
class _Background:
    image: np.ndarray            # BGR
    to_px: object                # (x, y) arrays -> (px, py) arrays
    xlim: tuple[float, float]
    ylim: tuple[float, float]


class JpdfPanel:
    """Callable composer: ``panel(frame_bgr, frame_idx) -> frame`` with the
    JPDF to the right of the frame."""

    def __init__(self, config_path, x: str, y: str, x_values, y_values,
                 *, trail: int = 30, pooled: pd.DataFrame | None = None) -> None:
        from mufasa.utils.feature_names import (
            display,
            load_jitter,
            load_names,
            load_scales,
        )
        self.config_path, self.x, self.y = config_path, x, y
        self.trail = int(trail)
        names = load_names(config_path) if config_path else {}
        scales = load_scales(config_path) if config_path else {}
        self.jitter = load_jitter(config_path) if config_path else set()
        self.logx, self.logy = scales.get(x) == "log10", scales.get(y) == "log10"
        self.name_x = display(x, names, scales)
        self.name_y = display(y, names, scales)
        # the video's own values (marker positions)
        self.vx = _scaled(_clean(x_values), self.logx)
        self.vy = _scaled(_clean(y_values), self.logy)
        if pooled is None:
            pooled = pooled_values(config_path, [x, y]) if config_path else pd.DataFrame()
        if not len(pooled):
            pooled = pd.DataFrame({x: x_values, y: y_values})
        self.px = _scaled(_clean(pooled[x]), self.logx)
        self.py = _scaled(_clean(pooled[y]), self.logy)
        self._bg: _Background | None = None
        self._bg_h = 0

    # -- background ------------------------------------------------------ #
    def _render_background(self, h: int) -> _Background:
        from matplotlib import colors as mcolors
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from mufasa.utils.feature_names import jitter_values
        ok = np.isfinite(self.px) & np.isfinite(self.py)
        bx, by = self.px[ok], self.py[ok]
        if self.x in self.jitter:
            bx = jitter_values(bx, seed=1)
        if self.y in self.jitter:
            by = jitter_values(by, seed=2)
        xlim = _limits(np.r_[bx, self.vx[np.isfinite(self.vx)]])
        ylim = _limits(np.r_[by, self.vy[np.isfinite(self.vy)]])
        dpi = 100
        w = h
        fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="white")
        FigureCanvasAgg(fig)
        fs = max(6.0, h / 70)
        ax = fig.add_axes([0.17, 0.13, 0.78, 0.80])
        if bx.size:
            H, xe, ye = np.histogram2d(bx, by, bins=90, range=(xlim, ylim))
            Hm = np.ma.masked_less_equal(H.T, 0)
            if Hm.count():
                ax.pcolormesh(xe, ye, Hm, cmap="Greys", shading="auto",
                              norm=mcolors.LogNorm(vmin=1, vmax=max(2, H.max())))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(self.name_x, fontsize=fs)
        ax.set_ylabel(self.name_y, fontsize=fs)
        ax.tick_params(labelsize=fs * 0.85)
        ax.set_title(f"all videos · {int(ok.sum()):,} frames", fontsize=fs * 0.85,
                     loc="left", color="#555555")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = cv2.cvtColor(rgba[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
        trans = ax.transData

        def to_px(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            xs = np.clip(xs, *xlim)
            ys = np.clip(ys, *ylim)
            d = trans.transform(np.c_[xs, ys])
            return d[:, 0], img.shape[0] - d[:, 1]      # display y is bottom-up

        return _Background(img, to_px, xlim, ylim)

    def background(self, h: int) -> _Background:
        if self._bg is None or self._bg_h != h:
            self._bg = self._render_background(h)
            self._bg_h = h
        return self._bg

    # -- per frame -------------------------------------------------------- #
    def panel(self, h: int, i: int) -> np.ndarray:
        bg = self.background(h)
        img = bg.image.copy()
        n = len(self.vx)
        r = max(4, h // 55)
        lo = max(0, i - self.trail)
        seg_x, seg_y = self.vx[lo:i + 1], self.vy[lo:i + 1]
        good = np.isfinite(seg_x) & np.isfinite(seg_y)
        if good.sum() >= 2:
            px, py = bg.to_px(seg_x, seg_y)
            pts = [(int(round(a)), int(round(b))) if g else None
                   for a, b, g in zip(px, py, good)]
            for k in range(1, len(pts)):
                if pts[k - 1] is None or pts[k] is None:
                    continue
                f = k / len(pts)                       # older → fainter
                col = (int(255 * (1 - f)), int(255 * (1 - f)), 255)
                cv2.line(img, pts[k - 1], pts[k], col, max(1, r // 3), cv2.LINE_AA)
        if 0 <= i < n and np.isfinite(self.vx[i]) and np.isfinite(self.vy[i]):
            px, py = bg.to_px(self.vx[i:i + 1], self.vy[i:i + 1])
            c = (int(round(px[0])), int(round(py[0])))
            cv2.circle(img, c, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, MARKER_BGR, -1, cv2.LINE_AA)
        else:
            cv2.putText(img, "no value at this frame", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, h / 900), (0, 0, 200), 1,
                        cv2.LINE_AA)
        return img

    def marker_px(self, h: int, i: int) -> tuple[int, int] | None:
        """Pixel of the current-frame marker in the panel (None if missing)."""
        if not (0 <= i < len(self.vx)) or not (np.isfinite(self.vx[i])
                                               and np.isfinite(self.vy[i])):
            return None
        px, py = self.background(h).to_px(self.vx[i:i + 1], self.vy[i:i + 1])
        return int(round(px[0])), int(round(py[0]))

    def __call__(self, frame: np.ndarray, i: int) -> np.ndarray:
        h = frame.shape[0]
        return np.ascontiguousarray(np.hstack([frame, self.panel(h, i)]))


def _limits(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if not v.size:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.04 * (hi - lo)
    return float(lo - pad), float(hi + pad)


__all__ = ["JpdfPanel", "MARKER_BGR", "pooled_values"]
