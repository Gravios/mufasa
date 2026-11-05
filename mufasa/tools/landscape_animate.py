"""``mufasa-landscape-animate`` — animate frames moving between t-SNE and feature spaces.

Takes a ``mufasa-landscape`` run and renders an animated GIF (and optionally
an MP4) in which the same sampled frames glide from the t-SNE map into the
joint scatter of one feature pair, on to the next pair, and back to t-SNE::

    t-SNE → pair 1 → pair 2 → … → t-SNE          (default)
    t-SNE → pair 1 → t-SNE → pair 2 → … → t-SNE  (--through-tsne)

Each point keeps its colour throughout. By default the colour encodes the
point's position in the t-SNE map (hue = direction from the centre,
saturation = distance), so in every feature space you can see where each
region of the map ended up; ``--color session`` or ``--color <column>``
colour by session or by a column's value instead. Every space is shown in
its own units (percentile-clipped to fill the square); axis labels, ticks,
title and a faint density backdrop cross-fade with the motion.

The pairs come from the run's ``pairs.csv`` (``--top N``: the same ranking
and per-column cap as the JPDF plots) or are named with ``--pair A B``
(repeatable). Positions come from ``tsne.parquet`` and the column values
from ``tsne_samples.parquet``, so nothing is recomputed.

Usage::

    mufasa-landscape-animate                          # latest run of the project in cwd
    mufasa-landscape-animate RUN_DIR --top 4 --mp4
    mufasa-landscape-animate RUN_DIR --pair speed kfv2.nose_x --pair emerg_a emerg_b \\
        --through-tsne --theme light --n-points 5000
"""
from __future__ import annotations

import argparse
import colorsys
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mufasa.project_layout import find_project_config

PAD = 0.04          # data square is [0, 1]; points may sit up to PAD outside it


# --------------------------------------------------------------------------- #
# Spaces
# --------------------------------------------------------------------------- #
@dataclass
class Space:
    title: str
    xlabel: str
    ylabel: str
    xy: np.ndarray                       # (N, 2), normalised to ~[0, 1]
    xticks: list[tuple[float, str]]      # (normalised position, label)
    yticks: list[tuple[float, str]]
    density: np.ndarray                  # (bins, bins) in [0, 1], row = y


def _ticks(lo: float, hi: float) -> list[tuple[float, str]]:
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    vals = [v for v in MaxNLocator(5).tick_values(lo, hi) if lo <= v <= hi]
    fmt = ScalarFormatter(useOffset=False)
    fmt.create_dummy_axis()
    fmt.set_locs(vals)
    return [((v - lo) / (hi - lo), fmt(v)) for v in vals]


def _density(xy: np.ndarray, bins: int = 120) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    H, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=bins,
                             range=((-PAD, 1 + PAD), (-PAD, 1 + PAD)))
    H = gaussian_filter(H, 1.2)
    H = np.log1p(H)
    return H / H.max() if H.max() > 0 else H


def feature_space(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> Space:
    """Joint scatter of two columns, each scaled to its 0.5-99.5 percentile
    range (outliers pinned just outside the square)."""
    cols = []
    ticks = []
    for v in (a, b):
        lo, hi = np.nanpercentile(v, [0.5, 99.5])
        if not hi > lo:
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v)) + 1e-9
        cols.append(np.clip((v - lo) / (hi - lo), -PAD, 1 + PAD))
        ticks.append(_ticks(lo, hi))
    xy = np.stack(cols, 1)
    return Space(f"{name_a}  ×  {name_b}", name_a, name_b, xy, ticks[0], ticks[1],
                 _density(xy))


def tsne_space(E: np.ndarray) -> Space:
    """t-SNE map scaled uniformly (aspect kept) and centred in the square."""
    lo, hi = np.percentile(E, 0.5, axis=0), np.percentile(E, 99.5, axis=0)
    span = float(max(hi - lo))
    xy = (E - (lo + hi) / 2) / span + 0.5
    xy = np.clip(xy, -PAD, 1 + PAD)
    return Space("t-SNE", "t-SNE 1", "t-SNE 2", xy, [], [], _density(xy))


# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #
def tsne_colors(xy: np.ndarray) -> np.ndarray:
    """Colour from position in the t-SNE map: hue = angle about the map's
    centre, saturation = distance from it."""
    c = np.median(xy, axis=0)
    d = xy - c
    ang = (np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi)
    r = np.hypot(d[:, 0], d[:, 1])
    r = np.clip(r / max(np.percentile(r, 95), 1e-9), 0, 1)
    return np.array([colorsys.hsv_to_rgb(h, 0.25 + 0.75 * s, 0.95)
                     for h, s in zip(ang, r)])


def point_colors(mode: str, emb: pd.DataFrame, samples: pd.DataFrame,
                 tsne_xy: np.ndarray) -> np.ndarray:
    import matplotlib
    if mode == "tsne":
        return tsne_colors(tsne_xy)
    if mode == "session":
        cmap = matplotlib.colormaps["tab20"]
        codes = pd.Categorical(emb["session"]).codes
        return np.array([cmap(i % 20)[:3] for i in codes])
    if mode in samples.columns:
        v = samples[mode].to_numpy(np.float64)
        lo, hi = np.nanpercentile(v, [1, 99])
        u = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
        rgb = matplotlib.colormaps["viridis"](np.nan_to_num(u, nan=0.0))[:, :3]
        rgb[~np.isfinite(v)] = 0.5
        return rgb
    raise SystemExit(f"--color: '{mode}' is not tsne, session or a column of the run")


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #
def ease(t: float) -> float:
    """Cubic ease-in-out on [0, 1]."""
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def timeline(n_spaces: int, fps: int, hold: float, transition: float
             ) -> list[tuple[int, int, float]]:
    """Frames as (from-space, to-space, eased progress): hold on each space,
    then move to the next; the last space is held too."""
    nh, nt = max(1, round(hold * fps)), max(2, round(transition * fps))
    out = []
    for k in range(n_spaces):
        out += [(k, k, 0.0)] * nh
        if k < n_spaces - 1:
            out += [(k, k + 1, ease(i / (nt - 1))) for i in range(nt)]
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
THEMES = {
    "dark": dict(bg="#14100c", fg="#efe3cf", dim="#8c7d68", dens="#d9952b"),
    "light": dict(bg="#fbf7ef", fg="#2a1e14", dim="#8a7a66", dens="#a8601a"),
}


class Renderer:
    """One matplotlib figure, updated in place frame by frame."""

    def __init__(self, spaces: list[Space], colors: np.ndarray, *, size: int,
                 theme: str, point_size: float, caption: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        self.plt = plt
        self.spaces = spaces
        T = THEMES[theme]
        self.T = T
        dpi = 100
        self.fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=T["bg"])
        self.ax = self.fig.add_axes((0.15, 0.13, 0.78, 0.74))
        ax = self.ax
        ax.set_facecolor(T["bg"])
        ax.set_xlim(-PAD - 0.02, 1 + PAD + 0.02)
        ax.set_ylim(-PAD - 0.02, 1 + PAD + 0.02)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(T["dim"])
            s.set_linewidth(0.8)
        cm = LinearSegmentedColormap.from_list("d", [T["bg"], T["dens"]])
        ext = (-PAD, 1 + PAD, -PAD, 1 + PAD)
        self.dens = [ax.imshow(np.zeros((2, 2)), extent=ext, origin="lower", cmap=cm,
                               vmin=0, vmax=1, interpolation="bilinear", zorder=0,
                               aspect="auto") for _ in range(2)]
        self.sc = ax.scatter(spaces[0].xy[:, 0], spaces[0].xy[:, 1], s=point_size,
                             c=colors, lw=0, alpha=0.85, zorder=2)
        fs = size / 800
        self.texts: dict[int, list] = {}
        for k, sp in enumerate(spaces):
            t = [self.fig.text(0.5, 0.94, sp.title, ha="center", va="center",
                               color=T["fg"], fontsize=17 * fs, fontweight="bold"),
                 self.fig.text(0.54, 0.045, sp.xlabel, ha="center", color=T["fg"],
                               fontsize=13 * fs),
                 self.fig.text(0.035, 0.5, sp.ylabel, va="center", rotation=90,
                               color=T["fg"], fontsize=13 * fs)]
            for pos, lab in sp.xticks:
                t.append(ax.text(pos, -PAD - 0.035, lab, ha="center", va="top",
                                 color=T["dim"], fontsize=10 * fs, clip_on=False))
            for pos, lab in sp.yticks:
                t.append(ax.text(-PAD - 0.035, pos, lab, ha="right", va="center",
                                 color=T["dim"], fontsize=10 * fs, clip_on=False))
            for o in t:
                o.set_alpha(0.0)
            self.texts[k] = t
        self.fig.text(0.985, 0.012, caption, ha="right", va="bottom",
                      color=T["dim"], fontsize=9 * fs)

    def frame(self, i: int, j: int, t: float) -> np.ndarray:
        a, b = self.spaces[i], self.spaces[j]
        self.sc.set_offsets(a.xy + (b.xy - a.xy) * t)
        self.dens[0].set_data(a.density)
        self.dens[0].set_alpha(0.55 * (1 - t))
        self.dens[1].set_data(b.density)
        self.dens[1].set_alpha(0.55 * t if j != i else 0.0)
        # fade text out over the first half of the move, in over the second
        out_a = 1.0 if i == j else max(0.0, 1 - 2 * t)
        in_b = 0.0 if i == j else max(0.0, 2 * t - 1)
        for k, objs in self.texts.items():
            al = out_a if k == i else (in_b if k == j else 0.0)
            for o in objs:
                o.set_alpha(al)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        self.plt.close(self.fig)


def write_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """GIF with one palette shared by every frame (no colour flicker)."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    picks = imgs[:: max(1, len(imgs) // 12)]
    w, h = picks[0].size
    sheet = Image.new("RGB", (w, h * len(picks)))
    for k, im in enumerate(picks):
        sheet.paste(im, (0, h * k))
    pal = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], loop=0,
              duration=round(1000 / fps), disposal=1, optimize=False)


def write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import cv2
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w2, h2))
    for f in frames:
        vw.write(cv2.cvtColor(f[:h2, :w2], cv2.COLOR_RGB2BGR))
    vw.release()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def find_run(arg: str | None) -> Path:
    """A landscape run directory: the argument itself, or the latest run of
    the project enclosing the argument / the cwd."""
    p = Path(arg or ".").expanduser().resolve()
    if (p / "tsne.parquet").is_file():
        return p
    cfg = p if p.name == "project.toml" else find_project_config(p)
    if cfg is None:
        raise SystemExit(f"{p} is neither a landscape run nor inside a project")
    base = cfg.parent / "derived" / "plots" / "feature_landscape"
    runs = sorted(d for d in base.glob("*") if (d / "tsne.parquet").is_file()) \
        if base.is_dir() else []
    if not runs:
        raise SystemExit(f"no mufasa-landscape run with a t-SNE under {base}")
    return runs[-1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mufasa-landscape-animate",
        description="Animate sampled frames moving between the t-SNE map and "
                    "joint feature spaces of a mufasa-landscape run.")
    p.add_argument("run", nargs="?", default=None,
                   help="landscape run directory, or a path in the project "
                        "(default: latest run of the project in the cwd)")
    p.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                   help="feature pair to visit, in order (repeatable)")
    p.add_argument("--top", type=int, default=4,
                   help="without --pair: visit the N top-ranked pairs of pairs.csv (default 4)")
    p.add_argument("--max-per-feature", type=int, default=1,
                   help="with --top: at most this many pairs per column (default 1)")
    p.add_argument("--max-per-family", type=int, default=2,
                   help="with --top: at most this many pairs touching any one feature "
                        "family (default 2)")
    p.add_argument("--through-tsne", action="store_true",
                   help="return to the t-SNE map between pairs")
    p.add_argument("--color", default="tsne",
                   help="tsne (position in the map, default), session, or a column name")
    p.add_argument("--n-points", type=int, default=4000,
                   help="points animated, a random subset of the t-SNE sample (default 4000)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--hold", type=float, default=1.5, help="seconds on each space")
    p.add_argument("--transition", type=float, default=2.0, help="seconds per move")
    p.add_argument("--size", type=int, default=800, help="square size in pixels")
    p.add_argument("--point-size", type=float, default=None)
    p.add_argument("--theme", choices=tuple(THEMES), default="dark")
    p.add_argument("--mp4", action="store_true", help="also write an MP4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="output file (default <run>/animation[_N].gif)")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    run = find_run(a.run)
    emb = pd.read_parquet(run / "tsne.parquet")
    sp = run / "tsne_samples.parquet"
    if not sp.is_file():
        raise SystemExit(f"{run} has no tsne_samples.parquet (made by mufasa-landscape "
                         "from patch 122iy on) — re-run mufasa-landscape")
    samples = pd.read_parquet(sp)
    if len(samples) != len(emb):
        raise SystemExit("tsne.parquet and tsne_samples.parquet differ in length")

    if a.pair:
        pairs = [tuple(p) for p in a.pair]
    else:
        from mufasa.tools.feature_landscape import pick_pairs
        table = pd.read_csv(run / "pairs.csv")
        top = pick_pairs(table, a.top, a.max_per_feature, a.max_per_family)
        pairs = list(zip(top.feature_a, top.feature_b))
        if not pairs:
            raise SystemExit("pairs.csv has no multimodal pairs; name some with --pair A B")
    missing = sorted({c for p in pairs for c in p} - set(samples.columns))
    if missing:
        raise SystemExit(f"not in the run's samples: {', '.join(missing)}")

    rng = np.random.default_rng(a.seed)
    need = sorted({c for p in pairs for c in p})
    ok = np.isfinite(samples[need].to_numpy(np.float64)).all(1)
    keep = np.flatnonzero(ok)
    if len(keep) < len(samples):
        print(f"[animate] {len(samples) - len(keep)} sample(s) with NaN in the chosen "
              "columns left out")
    keep = np.sort(rng.choice(keep, size=min(a.n_points, len(keep)), replace=False))
    emb, samples = emb.iloc[keep].reset_index(drop=True), samples.iloc[keep].reset_index(drop=True)

    tsne = tsne_space(emb[["tsne_1", "tsne_2"]].to_numpy(np.float64))
    fspaces = [feature_space(samples[x].to_numpy(np.float64),
                             samples[y].to_numpy(np.float64), x, y) for x, y in pairs]
    spaces = [tsne]
    for f in fspaces:
        spaces += [f, tsne] if a.through_tsne else [f]
    if not a.through_tsne:
        spaces.append(tsne)
    colors = point_colors(a.color, emb, samples, tsne.xy)
    psize = a.point_size or float(np.clip(6e4 / len(keep), 2, 30)) * (a.size / 800) ** 2
    caption = (f"{len(keep)} frames · {emb['session'].nunique()} sessions · "
               f"colour: {a.color}")
    R = Renderer(spaces, colors, size=a.size, theme=a.theme, point_size=psize,
                 caption=caption)
    tl = timeline(len(spaces), a.fps, a.hold, a.transition)
    print(f"[animate] {len(pairs)} pair(s), {len(spaces)} spaces, {len(tl)} frames "
          f"({len(tl) / a.fps:.1f} s at {a.fps} fps)")
    t0 = time.time()
    frames = [R.frame(i, j, t) for i, j, t in tl]
    R.close()
    out = Path(a.out) if a.out else None
    if out is None:
        out = run / "animation.gif"
        n = 1
        while out.exists():
            out = run / f"animation_{n}.gif"
            n += 1
    write_gif(frames, out, a.fps)
    print(f"[animate] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"rendered in {time.time() - t0:.1f} s)")
    if a.mp4:
        mp4 = out.with_suffix(".mp4")
        write_mp4(frames, mp4, a.fps)
        print(f"[animate] wrote {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
