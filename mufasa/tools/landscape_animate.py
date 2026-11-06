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
colour by session or by a column's value instead. The animation opens
(``--intro-frames``, default 40) on the t-SNE map coloured by session, then
the colours ease into the main colouring. ``--n-points`` and ``--sampling``
(random / balanced per session / even over the map) choose the points. Every space is shown in
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


def session_colors(sessions) -> np.ndarray:
    """One colour per session: tab20 up to 20 sessions, beyond that hues
    spread by the golden ratio (no repeats)."""
    import matplotlib
    codes = pd.Categorical(sessions).codes
    n = int(codes.max()) + 1 if len(codes) else 0
    if n <= 20:
        cmap = matplotlib.colormaps["tab20"]
        pal = np.array([cmap(i)[:3] for i in range(20)])
    else:
        pal = np.array([colorsys.hsv_to_rgb((i * 0.618034) % 1.0,
                                            0.55 + 0.35 * (i % 3) / 2, 0.92 - 0.12 * (i % 2))
                        for i in range(n)])
    return pal[codes]


# --------------------------------------------------------------------------- #
# Sampling (patch 122jm)
# --------------------------------------------------------------------------- #
SAMPLING = ("random", "balanced", "even")


def sample_points(n: int, candidates: np.ndarray, sessions: np.ndarray,
                  tsne_xy: np.ndarray, method: str, rng: np.random.Generator,
                  grid: int = 40) -> np.ndarray:
    """Pick ``n`` of ``candidates`` (row indices) for the animation.

    * ``random``   — uniformly at random (dense regions of the map dominate,
      as in the data);
    * ``balanced`` — the same number from every session (sessions with too
      few frames give all they have, the rest is shared out);
    * ``even``     — spread over the t-SNE map: the map is cut into a
      ``grid`` × ``grid`` lattice and cells are drawn from in turn, so sparse
      regions are as visible as dense ones.

    Returns sorted row indices."""
    cand = np.asarray(candidates)
    if n >= len(cand):
        return np.sort(cand)
    if method == "random":
        return np.sort(rng.choice(cand, size=n, replace=False))
    if method == "balanced":
        groups = [rng.permutation(cand[sessions[cand] == s])
                  for s in pd.unique(sessions[cand])]
        return np.sort(_round_robin(groups, n))
    if method == "even":
        xy = tsne_xy[cand]
        lo, hi = xy.min(0), xy.max(0)
        ij = np.floor((xy - lo) / np.maximum(hi - lo, 1e-12) * grid).clip(0, grid - 1)
        cell = (ij[:, 0] * grid + ij[:, 1]).astype(int)
        groups = [rng.permutation(cand[cell == c]) for c in rng.permutation(np.unique(cell))]
        return np.sort(_round_robin(groups, n))
    raise SystemExit(f"--sampling must be one of {', '.join(SAMPLING)}")


OUTLIER_METHODS = ("none", "percentile", "mad")


def outlier_mask(values: dict[str, np.ndarray], method: str, *, percentile: float = 1.0,
                 mad: float = 5.0) -> tuple[np.ndarray, dict[str, int]]:
    """Frames that are outliers in any of ``values`` ({name: array}, same
    length): ``percentile`` — below the ``percentile``-th or above the
    (100 − ``percentile``)-th percentile; ``mad`` — more than ``mad`` scaled
    MADs (≈ SDs for normal data) from the median. Returns (mask, {name:
    frames flagged by it})."""
    n = len(next(iter(values.values()))) if values else 0
    bad = np.zeros(n, bool)
    per: dict[str, int] = {}
    for name, v in values.items():
        v = np.asarray(v, np.float64)
        if method == "percentile":
            lo, hi = np.nanpercentile(v, [percentile, 100 - percentile])
            b = (v < lo) | (v > hi)
        elif method == "mad":
            med = np.nanmedian(v)
            s = 1.4826 * np.nanmedian(np.abs(v - med))
            if not s > 0:          # more than half the values identical
                s = np.nanstd(v)
            b = np.abs(v - med) > mad * s if s > 0 else np.zeros(n, bool)
        elif method == "none":
            b = np.zeros(n, bool)
        else:
            raise SystemExit(f"--outliers must be one of {', '.join(OUTLIER_METHODS)}")
        per[name] = int(b.sum())
        bad |= b
    return bad, per


def _round_robin(groups: list[np.ndarray], n: int) -> np.ndarray:
    """Take one from each group in turn until ``n`` are taken."""
    out, k = [], 0
    while len(out) < n:
        took = False
        for g in groups:
            if k < len(g):
                out.append(g[k])
                took = True
                if len(out) == n:
                    break
        if not took:
            break
        k += 1
    return np.asarray(out, dtype=int)


def point_colors(mode: str, emb: pd.DataFrame, samples: pd.DataFrame,
                 tsne_xy: np.ndarray) -> np.ndarray:
    import matplotlib
    if mode == "tsne":
        return tsne_colors(tsne_xy)
    if mode == "session":
        return session_colors(emb["session"])
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


def timeline(n_spaces: int, fps: int, hold: float, transition: float,
             holds: list[int] | None = None, n_transition: int | None = None
             ) -> list[tuple[int, int, float]]:
    """Frames as (from-space, to-space, eased progress): hold on each space,
    then move to the next; the last space is held too.

    Patch 122jo: ``holds`` gives the pause in frames per space (overrides
    ``hold`` seconds; 0 = no pause) and ``n_transition`` the frames per move
    (overrides ``transition`` seconds)."""
    nt = max(2, n_transition if n_transition is not None else round(transition * fps))
    if holds is None:
        holds = [max(1, round(hold * fps))] * n_spaces
    out = []
    for k in range(n_spaces):
        out += [(k, k, 0.0)] * max(0, holds[k])
        if k < n_spaces - 1:
            out += [(k, k + 1, ease(i / (nt - 1))) for i in range(nt)]
    return out


def intro(frames: int, fade: int, start: np.ndarray, end: np.ndarray, label: str
          ) -> list[tuple[np.ndarray, str, float]]:
    """Session intro (patch 122jm): ``frames`` frames on the t-SNE map in
    ``start`` colours, then ``fade`` frames easing to ``end`` colours. Each
    entry is (colours, subtitle, subtitle alpha); the subtitle fades out
    with the colours."""
    out = [(start, label, 1.0)] * max(0, frames)
    for k in range(max(0, fade)):
        t = ease((k + 1) / fade)
        out.append((start + (end - start) * t, label, 1.0 - t))
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
        self.colors = np.asarray(colors, np.float64)
        self.sc = ax.scatter(spaces[0].xy[:, 0], spaces[0].xy[:, 1], s=point_size,
                             c=self.colors, lw=0, alpha=0.85, zorder=2)
        self._cur_colors = self.colors
        fs = size / 800
        self.subtitle = self.fig.text(0.5, 0.895, "", ha="center", va="center",
                                      color=T["dim"], fontsize=12 * fs, alpha=0.0)
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

    def frame(self, i: int, j: int, t: float, colors: np.ndarray | None = None,
              subtitle: str = "", subtitle_alpha: float = 0.0) -> np.ndarray:
        """Draw spaces i → j at progress t. ``colors`` overrides the point
        colours for this frame (the session intro); the subtitle line under
        the title shows ``subtitle`` at ``subtitle_alpha``."""
        cols = self.colors if colors is None else colors
        if cols is not self._cur_colors:
            self.sc.set_facecolors(np.c_[cols, np.full(len(cols), 0.85)])
            self._cur_colors = cols
        self.subtitle.set_text(subtitle)
        self.subtitle.set_alpha(subtitle_alpha)
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
                   help="points animated, a subset of the t-SNE sample (default 4000)")
    p.add_argument("--sampling", choices=SAMPLING, default="random",
                   help="how the points are picked from the t-SNE sample: random "
                        "(default), balanced (same number per session) or even "
                        "(spread over the t-SNE map, sparse regions kept)")
    p.add_argument("--outliers", choices=OUTLIER_METHODS, default="none",
                   help="leave out frames that are outliers in any animated feature: "
                        "percentile (outside --outlier-percentile .. 100 − it) or mad "
                        "(more than --outlier-mad scaled MADs from the median)")
    p.add_argument("--outlier-percentile", type=float, default=1.0,
                   help="with --outliers percentile: tail cut on each side, in %% (default 1)")
    p.add_argument("--outlier-mad", type=float, default=5.0,
                   help="with --outliers mad: threshold in scaled MADs (default 5)")
    p.add_argument("--outliers-tsne", action="store_true",
                   help="also test the t-SNE coordinates for outliers")
    p.add_argument("--intro-frames", type=int, default=40,
                   help="first frames: the t-SNE map coloured by session, before "
                        "--color takes over (0 = no intro; default 40)")
    p.add_argument("--intro-fade", type=int, default=20,
                   help="frames for the colour change after the intro (default 20)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--hold", type=float, default=1.5, help="seconds on each space")
    p.add_argument("--transition", type=float, default=2.0, help="seconds per move")
    p.add_argument("--transition-frames", type=int, default=None,
                   help="frames per move (overrides --transition)")
    p.add_argument("--hold-frames", type=int, default=None,
                   help="frames paused on each feature space (overrides --hold; "
                        "0 = no pause)")
    p.add_argument("--tsne-hold-frames", type=int, default=None,
                   help="frames paused on the t-SNE map (default: as --hold-frames)")
    p.add_argument("--size", type=int, default=800, help="square size in pixels")
    p.add_argument("--point-size", type=float, default=None)
    p.add_argument("--theme", choices=tuple(THEMES), default="dark")
    p.add_argument("--mp4", action="store_true", help="also write an MP4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--names-file", default=None,
                   help="feature display names (TOML [names]); default: the "
                        "project's logs/measures/feature_names.toml")
    p.add_argument("--jitter", action="append", metavar="COLUMN",
                   help="jitter this near-discrete column (repeatable; added to the "
                        "names file's [jitter] marks)")
    p.add_argument("--no-jitter", action="store_true",
                   help="ignore the names file's [jitter] marks")
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

    from mufasa.utils.feature_names import display, load_names, read_names
    cfg = find_project_config(run)
    names = read_names(a.names_file) if a.names_file else (load_names(cfg) if cfg else {})
    rng = np.random.default_rng(a.seed)
    need = sorted({c for p in pairs for c in p})
    ok = np.isfinite(samples[need].to_numpy(np.float64)).all(1)
    keep = np.flatnonzero(ok)
    if len(keep) < len(samples):
        print(f"[animate] {len(samples) - len(keep)} sample(s) with NaN in the chosen "
              "columns left out")
    sess_arr = emb["session"].astype(str).to_numpy()
    raw_xy = emb[["tsne_1", "tsne_2"]].to_numpy(np.float64)
    # Patch 122jp: leave out outlying frames before sampling
    if a.outliers != "none":
        vals = {c: samples[c].to_numpy(np.float64)[keep] for c in need}
        if a.outliers_tsne:
            vals["t-SNE 1"], vals["t-SNE 2"] = raw_xy[keep, 0], raw_xy[keep, 1]
        bad, per = outlier_mask(vals, a.outliers, percentile=a.outlier_percentile,
                                mad=a.outlier_mad)
        if bad.any():
            print(f"[animate] {int(bad.sum())} of {len(keep)} frame(s) left out as outliers "
                  f"({a.outliers}): " + ", ".join(
                      f"{display(c, names) if c in samples.columns else c} {n}"
                      for c, n in per.items() if n))
        keep = keep[~bad]
    keep = sample_points(a.n_points, keep, sess_arr, raw_xy, a.sampling, rng)
    print(f"[animate] {len(keep)} point(s), sampling: {a.sampling} "
          f"({len(pd.unique(sess_arr[keep]))} sessions)")
    emb, samples = emb.iloc[keep].reset_index(drop=True), samples.iloc[keep].reset_index(drop=True)

    tsne = tsne_space(emb[["tsne_1", "tsne_2"]].to_numpy(np.float64))
    # columns the landscape analysed as log10 (samples hold log10 values)
    lf = run / "log10_columns.txt"
    scaled = ({ln.strip(): "log10" for ln in lf.read_text().splitlines() if ln.strip()}
              if lf.is_file() else {})
    # Patch 122jn: near-discrete features jittered (display only)
    from mufasa.utils.feature_names import jitter_values, load_jitter, read_jitter
    jit = set() if a.no_jitter else (read_jitter(a.names_file) if a.names_file
                                     else (load_jitter(cfg) if cfg else set()))
    jit |= set(a.jitter or [])
    if jit & {c for p_ in pairs for c in p_}:
        print("[animate] jittered: " + ", ".join(
            sorted(display(c, names) for c in jit & {c for p_ in pairs for c in p_})))

    def _vals(c: str) -> np.ndarray:
        v = samples[c].to_numpy(np.float64)
        return jitter_values(v, seed=a.seed + 7) if c in jit else v

    fspaces = [feature_space(_vals(x), _vals(y),
                             display(x, names, scaled), display(y, names, scaled))
               for x, y in pairs]
    spaces = [tsne]
    for f in fspaces:
        spaces += [f, tsne] if a.through_tsne else [f]
    if not a.through_tsne:
        spaces.append(tsne)
    colors = point_colors(a.color, emb, samples, tsne.xy)
    psize = a.point_size or float(np.clip(6e4 / len(keep), 2, 30)) * (a.size / 800) ** 2
    caption = (f"{len(keep)} frames ({a.sampling}) · {emb['session'].nunique()} "
               f"sessions · colour: {a.color}")
    R = Renderer(spaces, colors, size=a.size, theme=a.theme, point_size=psize,
                 caption=caption)
    # Patch 122jo: pauses and transitions in frames
    holds = None
    if a.hold_frames is not None or a.tsne_hold_frames is not None:
        fh = a.hold_frames if a.hold_frames is not None else max(1, round(a.hold * a.fps))
        th = a.tsne_hold_frames if a.tsne_hold_frames is not None else fh
        holds = [th if sp is tsne else fh for sp in spaces]
    tl = timeline(len(spaces), a.fps, a.hold, a.transition, holds=holds,
                  n_transition=a.transition_frames)
    if holds is not None or a.transition_frames is not None:
        nt = max(2, a.transition_frames if a.transition_frames is not None
                 else round(a.transition * a.fps))
        hs = holds or [max(1, round(a.hold * a.fps))] * len(spaces)
        print(f"[animate] pause {hs[1] if len(hs) > 1 else hs[0]} frame(s) on each "
              f"feature space, {hs[0]} on t-SNE; {nt} frame(s) per transition")
    print(f"[animate] {len(pairs)} pair(s), {len(spaces)} spaces, {len(tl)} frames "
          f"({len(tl) / a.fps:.1f} s at {a.fps} fps)")
    t0 = time.time()
    frames = []
    if a.intro_frames > 0 and a.color != "session":
        sc = session_colors(emb["session"])
        lead = intro(a.intro_frames, a.intro_fade, sc, np.asarray(colors, np.float64),
                     f"coloured by session ({emb['session'].nunique()})")
        frames += [R.frame(0, 0, 0.0, colors=c, subtitle=lab, subtitle_alpha=al)
                   for c, lab, al in lead]
        print(f"[animate] intro: {a.intro_frames} frame(s) by session, "
              f"{a.intro_fade} to change colour")
    frames += [R.frame(i, j, t) for i, j, t in tl]
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
