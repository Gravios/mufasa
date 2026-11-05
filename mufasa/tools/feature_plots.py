"""
mufasa.tools.feature_plots
==========================

Matplotlib drawing for the Features → Feature Visualization section
(patch 122je). Qt-free: every function draws into a
:class:`matplotlib.figure.Figure` it is given, so the same code serves the
workbench canvas and scripts / tests.

* :func:`plot_distributions` — one histogram per feature; with a behaviour
  mask, frames with and without the behaviour are drawn separately and
  each panel reports the AUC of the feature for that behaviour (0.5 = no
  separation, 1 or 0 = perfect), the quickest read on which features a
  classifier can use.
* :func:`plot_joint` — 2-D log-density of two features, behaviour frames
  overlaid.
* :func:`plot_over_time` — whole-session traces with behaviour bouts
  shaded.
"""
from __future__ import annotations

import math

import numpy as np

BEHAVIOUR_COLOR = "#d9662b"
REST_COLOR = "#4a7fb5"


def short_name(col: str) -> str:
    return col[: -len("_FEATURE_SUBSET")] if col.endswith("_FEATURE_SUBSET") else col


def _wrap(s: str, n: int = 38) -> str:
    s = short_name(s)
    if len(s) <= n:
        return s
    cut = s.rfind(" ", 0, n)
    cut = cut if cut > n // 2 else n
    return s[:cut] + "\n" + _wrap(s[cut:].lstrip(), n)


def auc(values: np.ndarray, mask: np.ndarray) -> float:
    """Area under the ROC curve of ``values`` for separating ``mask``
    frames from the rest (Mann-Whitney U / n1 n0). NaN when either group
    is empty."""
    from scipy.stats import rankdata
    ok = np.isfinite(values)
    v, m = values[ok], mask[ok].astype(bool)
    n1, n0 = int(m.sum()), int((~m).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(v)
    return float((r[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _range(v: np.ndarray) -> tuple[float, float]:
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(fin, [0.5, 99.5])
    if not hi > lo:
        lo, hi = fin.min(), fin.max() + 1e-9
    return float(lo), float(hi)


def plot_distributions(fig, data: dict[str, np.ndarray], *,
                       mask: np.ndarray | None = None, behaviour: str = "",
                       bins: int = 60) -> dict[str, float]:
    """Histogram per feature; returns {feature: AUC} when ``mask`` given."""
    fig.clear()
    cols = list(data)
    if not cols:
        return {}
    ncol = min(3, len(cols))
    nrow = math.ceil(len(cols) / ncol)
    aucs: dict[str, float] = {}
    for k, c in enumerate(cols):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        v = np.asarray(data[c], np.float64)
        lo, hi = _range(v)
        edges = np.linspace(lo, hi, bins + 1)
        fin = np.isfinite(v)
        if mask is not None and mask.any():
            m = mask.astype(bool)
            ax.hist(v[fin & ~m], bins=edges, density=True, color=REST_COLOR,
                    alpha=0.55, label=f"not {behaviour}")
            ax.hist(v[fin & m], bins=edges, density=True, color=BEHAVIOUR_COLOR,
                    alpha=0.6, label=behaviour)
            a = auc(v, m)
            aucs[c] = a
            ax.set_title(f"{_wrap(c)}\nAUC {a:.2f}", fontsize=8)
            if k == 0:
                ax.legend(fontsize=7, frameon=False)
        else:
            ax.hist(v[fin], bins=edges, density=True, color=REST_COLOR, alpha=0.8)
            ax.set_title(_wrap(c), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        n_nan = int((~fin).sum())
        if n_nan:
            ax.text(0.99, 0.97, f"{n_nan:,} NaN", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6, color="#888")
    fig.tight_layout()
    return aucs


def plot_joint(fig, a: np.ndarray, b: np.ndarray, name_a: str, name_b: str, *,
               mask: np.ndarray | None = None, behaviour: str = "",
               bins: int = 90, max_points: int = 4000, seed: int = 0) -> None:
    """2-D log density of two features; behaviour frames overlaid."""
    import matplotlib.colors as mcolors
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    xr, yr = _range(a[ok]), _range(b[ok])
    H, xe, ye = np.histogram2d(a[ok], b[ok], bins=bins, range=(xr, yr))
    Hm = np.ma.masked_less_equal(H.T, 0)
    if Hm.count():
        ax.pcolormesh(xe, ye, Hm, cmap="Greys", shading="auto",
                      norm=mcolors.LogNorm(vmin=1, vmax=max(2, H.max())))
    if mask is not None:
        m = mask.astype(bool) & ok
        idx = np.flatnonzero(m)
        if idx.size:
            if idx.size > max_points:
                idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
            ax.scatter(a[idx], b[idx], s=4, lw=0, color=BEHAVIOUR_COLOR, alpha=0.6,
                       label=f"{behaviour} ({int(m.sum()):,} frames)")
            ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_xlabel(_wrap(name_a, 60), fontsize=8)
    ax.set_ylabel(_wrap(name_b, 60), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{int(ok.sum()):,} frames", fontsize=8)
    fig.tight_layout()


def plot_over_time(fig, data: dict[str, np.ndarray], fps: float, *,
                   labels: dict[str, np.ndarray] | None = None,
                   max_points: int = 6000) -> list:
    """Whole-session traces (min/max decimated) with behaviour bouts
    shaded; returns the axes (time in minutes on x)."""
    fig.clear()
    cols = list(data)
    if not cols:
        return []
    axes = fig.subplots(len(cols), 1, sharex=True, squeeze=False)[:, 0]
    fps = fps if fps and fps > 0 else 30.0
    spans = []
    for k, (name, lab) in enumerate((labels or {}).items()):
        arr = (np.asarray(lab, float) == 1).astype(np.int8)
        d = np.diff(np.r_[0, arr, 0])
        spans.append((name, np.flatnonzero(d == 1), np.flatnonzero(d == -1), k))
    palette = [BEHAVIOUR_COLOR, "#7b5ea7", "#2e9e6b", "#c49a1a", "#c43b6b"]
    for ax, c in zip(axes, cols):
        v = np.asarray(data[c], np.float64)
        n = len(v)
        t = np.arange(n) / fps / 60.0
        if n > max_points:
            step = math.ceil(n / (max_points / 2))
            m = n // step * step
            blk = v[:m].reshape(-1, step)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    lo, hi = np.nanmin(blk, 1), np.nanmax(blk, 1)
            tt = t[:m].reshape(-1, step)[:, 0]
            ax.fill_between(tt, lo, hi, color=REST_COLOR, lw=0.6, alpha=0.9)
        else:
            ax.plot(t, v, color=REST_COLOR, lw=0.7)
        for name, on, off, k in spans:
            for s, e in zip(on, off):
                ax.axvspan(s / fps / 60, e / fps / 60, color=palette[k % len(palette)],
                           alpha=0.25, lw=0)
        ax.set_ylim(*_range(v))
        ax.set_ylabel(_wrap(c, 26), fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (min)", fontsize=8)
    if spans:
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(color=palette[k % len(palette)], alpha=0.4, label=nm)
                                for nm, _, _, k in spans], fontsize=7, frameon=False,
                       loc="upper right", ncol=len(spans))
    fig.tight_layout()
    return list(axes)


__all__ = ["auc", "plot_distributions", "plot_joint", "plot_over_time", "short_name"]
