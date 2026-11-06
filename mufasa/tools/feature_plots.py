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
import re
from dataclasses import dataclass, field

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


def _edges(v: np.ndarray, bins: int, log: bool) -> np.ndarray:
    """Histogram edges over the 0.5–99.5 % range; log-spaced over the
    positive values when ``log``."""
    if log:
        pos = v[np.isfinite(v) & (v > 0)]
        if pos.size == 0:
            return np.logspace(0, 1, bins + 1)
        lo, hi = np.percentile(pos, [0.5, 99.5])
        if not hi > lo:
            lo, hi = pos.min(), pos.max() * 1.0001 + 1e-12
        return np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    lo, hi = _range(v)
    return np.linspace(lo, hi, bins + 1)


def _note(ax, text: str) -> None:
    ax.text(0.99, 0.97, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color="#888")


def plot_distributions(fig, data: dict[str, np.ndarray], *,
                       mask: np.ndarray | None = None, behaviour: str = "",
                       bins: int = 60, swap: bool = False, xlog: bool = False,
                       ylog: bool = False) -> dict[str, float]:
    """Histogram per feature; returns {feature: AUC} when ``mask`` given.

    Feature values run along x and density along y; ``swap`` turns the
    histograms sideways (values on y). ``xlog`` / ``ylog`` put the
    displayed x / y axis on a log10 scale — the value axis gets log-spaced
    bins and values ≤ 0 are left out (and counted in the panel)."""
    fig.clear()
    cols = list(data)
    if not cols:
        return {}
    vlog, dlog = (ylog, xlog) if swap else (xlog, ylog)
    orient = "horizontal" if swap else "vertical"
    ncol = min(3, len(cols))
    nrow = math.ceil(len(cols) / ncol)
    aucs: dict[str, float] = {}
    for k, c in enumerate(cols):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        v = np.asarray(data[c], np.float64)
        edges = _edges(v, bins, vlog)
        fin = np.isfinite(v)
        use = fin & (v > 0) if vlog else fin
        kw = dict(bins=edges, density=True, orientation=orient)
        if mask is not None and mask.any():
            m = mask.astype(bool)
            ax.hist(v[use & ~m], color=REST_COLOR, alpha=0.55,
                    label=f"not {behaviour}", **kw)
            ax.hist(v[use & m], color=BEHAVIOUR_COLOR, alpha=0.6, label=behaviour, **kw)
            a = auc(v, m)
            aucs[c] = a
            ax.set_title(f"{_wrap(c)}\nAUC {a:.2f}", fontsize=8)
            if k == 0:
                ax.legend(fontsize=7, frameon=False)
        else:
            ax.hist(v[use], color=REST_COLOR, alpha=0.8, **kw)
            ax.set_title(_wrap(c), fontsize=8)
        if vlog:
            (ax.set_yscale if swap else ax.set_xscale)("log")
        if dlog:
            (ax.set_xscale if swap else ax.set_yscale)("log")
        else:
            (ax.set_xticks if swap else ax.set_yticks)([])
        ax.tick_params(labelsize=7)
        notes = []
        if int((~fin).sum()):
            notes.append(f"{int((~fin).sum()):,} NaN")
        if vlog and int((fin & (v <= 0)).sum()):
            notes.append(f"{int((fin & (v <= 0)).sum()):,} ≤ 0 not shown")
        if notes:
            _note(ax, " · ".join(notes))
    fig.tight_layout()
    return aucs


def plot_joint(fig, a: np.ndarray, b: np.ndarray, name_a: str, name_b: str, *,
               mask: np.ndarray | None = None, behaviour: str = "",
               bins: int = 90, max_points: int = 4000, seed: int = 0,
               swap: bool = False, xlog: bool = False, ylog: bool = False) -> None:
    """2-D log density of two features (``a`` on x, ``b`` on y; ``swap``
    exchanges them); behaviour frames overlaid. ``xlog`` / ``ylog``: log10
    axes with log-spaced bins; frames with a value ≤ 0 on a log axis are
    left out (and counted)."""
    import matplotlib.colors as mcolors
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    nx, ny = name_a, name_b
    if swap:
        x, y, nx, ny = y, x, ny, nx
    ok = np.isfinite(x) & np.isfinite(y)
    n_fin = int(ok.sum())
    if xlog:
        ok &= x > 0
    if ylog:
        ok &= y > 0
    xe, ye = _edges(x[ok], bins, xlog), _edges(y[ok], bins, ylog)
    H, xe, ye = np.histogram2d(x[ok], y[ok], bins=(xe, ye))
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
            ax.scatter(x[idx], y[idx], s=4, lw=0, color=BEHAVIOUR_COLOR, alpha=0.6,
                       label=f"{behaviour} ({int(m.sum()):,} frames)")
            ax.legend(fontsize=8, frameon=False, loc="upper right")
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(_wrap(nx, 60), fontsize=8)
    ax.set_ylabel(_wrap(ny, 60), fontsize=8)
    ax.tick_params(labelsize=7)
    hidden = n_fin - int(ok.sum())
    ax.set_title(f"{int(ok.sum()):,} frames"
                 + (f" ({hidden:,} with a value ≤ 0 on a log axis not shown)" if hidden else ""),
                 fontsize=8)
    fig.tight_layout()


def plot_over_time(fig, data: dict[str, np.ndarray], fps: float, *,
                   labels: dict[str, np.ndarray] | None = None,
                   max_points: int = 6000, swap: bool = False, xlog: bool = False,
                   ylog: bool = False) -> list:
    """Whole-session traces (min/max decimated) with behaviour bouts
    shaded; returns the axes. Time (minutes) runs along x and values along
    y, one panel per feature stacked vertically; ``swap`` puts time on y
    (downwards) with the panels side by side. ``xlog`` / ``ylog``: log10 on
    the displayed axis (values ≤ 0, or time 0, are not drawn)."""
    import warnings
    fig.clear()
    cols = list(data)
    if not cols:
        return []
    tlog, vlog = (ylog, xlog) if swap else (xlog, ylog)
    if swap:
        axes = fig.subplots(1, len(cols), sharey=True, squeeze=False)[0, :]
    else:
        axes = fig.subplots(len(cols), 1, sharex=True, squeeze=False)[:, 0]
    fps = fps if fps and fps > 0 else 30.0
    spans = []
    for k, (name, lab) in enumerate((labels or {}).items()):
        arr = (np.asarray(lab, float) == 1).astype(np.int8)
        d = np.diff(np.r_[0, arr, 0])
        spans.append((name, np.flatnonzero(d == 1), np.flatnonzero(d == -1), k))
    palette = [BEHAVIOUR_COLOR, "#7b5ea7", "#2e9e6b", "#c49a1a", "#c43b6b"]
    for ax, c in zip(axes, cols):
        v = np.asarray(data[c], np.float64).copy()
        if vlog:
            v[v <= 0] = np.nan
        n = len(v)
        t = np.arange(n) / fps / 60.0
        if n > max_points:
            step = math.ceil(n / (max_points / 2))
            m = n // step * step
            blk = v[:m].reshape(-1, step)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = np.nanmin(blk, 1), np.nanmax(blk, 1)
            tt = t[:m].reshape(-1, step)[:, 0]
            if swap:
                ax.fill_betweenx(tt, lo, hi, color=REST_COLOR, lw=0.6, alpha=0.9)
            else:
                ax.fill_between(tt, lo, hi, color=REST_COLOR, lw=0.6, alpha=0.9)
        elif swap:
            ax.plot(v, t, color=REST_COLOR, lw=0.7)
        else:
            ax.plot(t, v, color=REST_COLOR, lw=0.7)
        for name, on, off, k in spans:
            for s, e in zip(on, off):
                (ax.axhspan if swap else ax.axvspan)(
                    s / fps / 60, e / fps / 60, color=palette[k % len(palette)],
                    alpha=0.25, lw=0)
        e = _edges(v, 1, vlog)
        (ax.set_xlim if swap else ax.set_ylim)(e[0], e[-1])
        if vlog:
            (ax.set_xscale if swap else ax.set_yscale)("log")
        if swap:
            ax.set_xlabel(_wrap(c, 26), fontsize=7)
        else:
            ax.set_ylabel(_wrap(c, 26), fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
    if tlog:
        (axes[0].set_yscale if swap else axes[-1].set_xscale)("log")
    if swap:
        axes[0].set_ylabel("time (min)", fontsize=8)
        axes[0].invert_yaxis()
    else:
        axes[-1].set_xlabel("time (min)", fontsize=8)
    if spans:
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(color=palette[k % len(palette)], alpha=0.4, label=nm)
                                for nm, _, _, k in spans], fontsize=7, frameon=False,
                       loc="upper right", ncol=len(spans))
    fig.tight_layout()
    return list(axes)


# ---------------------------------------------------------------------- #
# Compact naming for the feature tree (patch 122jg)
# ---------------------------------------------------------------------- #
_SEP = re.compile(r"(\s+|-)")


def _tokens(s: str) -> list[str]:
    """Split on spaces and '-', keeping the separators (so the pieces
    re-join to the original string)."""
    return [t for t in _SEP.split(s) if t != ""]


def _common_prefix(seqs: list[list[str]]) -> int:
    n = min(len(x) for x in seqs)
    k = 0
    while k < n and all(x[k] == seqs[0][k] for x in seqs):
        k += 1
    return k


@dataclass
class Node:
    """Feature-tree node: a group (``col`` None, ``children``) or a leaf."""
    label: str
    col: str | None = None
    children: list["Node"] = field(default_factory=list)

    def leaves(self) -> list["Node"]:
        if self.col is not None:
            return [self]
        return [x for c in self.children for x in c.leaves()]


def _group(items: list[tuple[str, str]], *, min_group: int, max_leaves: int,
           depth: int) -> list[Node]:
    """``items`` = [(label core, column)]; group by first component, and
    re-group a group that is still large by its next component."""
    keyed: dict[str, list[tuple[str, str]]] = {}
    for core, col in items:
        parts = _tokens(core)
        keyed.setdefault(parts[0], []).append(
            ("".join(parts[1:]).strip(" -") or core, col))
    if len(keyed) in (1, len(items)):        # would not compact anything
        return [Node(core, col) for core, col in items]
    nodes: list[Node] = []
    loose: list[Node] = []
    by_col = dict((col, core) for core, col in items)
    for key, members in keyed.items():
        if len(members) < min_group:
            loose += [Node(by_col[col], col) for _r, col in members]
            continue
        if len(members) > max_leaves and depth < 2:
            kids = _group(members, min_group=min_group, max_leaves=max_leaves,
                          depth=depth + 1)
        else:
            kids = [Node(r, col) for r, col in members]
        nodes.append(Node(key, None, kids))
    return nodes + loose


def compact_family(cols: list[str], *, min_group: int = 2, min_family: int = 6,
                   max_leaves: int = 24) -> tuple[str, list[Node]]:
    """Group one family's columns by shared name components.

    The words every name in the family shares at the start and end (e.g.
    ``"Animal_1 three-point convex hull perimeter (mm)"``) are dropped, and
    the rest is grouped by its first component — for body-part features
    the first body part — and, where a group still has more than
    ``max_leaves`` members, by the next one (two levels at most):

    ``Distance (mm) center-ear_l``, ``… center-nose``, ``… ear_l-nose`` →
    ``center ▸ ear_l, nose`` · ``ear_l ▸ nose``

    Returns ``(template, nodes)``: ``template`` is the shared wording with
    ``…`` where names differ; ``nodes`` the top level under the family.
    Singletons and families smaller than ``min_family`` stay flat.
    """
    if not cols:
        return "", []
    shorts = [short_name(c) for c in cols]
    if len(cols) == 1:
        return shorts[0], [Node(shorts[0], cols[0])]
    toks = [_tokens(x) for x in shorts]
    pre = _common_prefix(toks)
    suf = min(_common_prefix([t[::-1] for t in toks]),
              min(len(t) for t in toks) - pre)
    cores = ["".join(t[pre: len(t) - suf if suf else None]).strip(" -") for t in toks]
    head = "".join(toks[0][:pre]).strip()
    tail = "".join(toks[0][len(toks[0]) - suf:]).strip() if suf else ""
    template = " ".join(x for x in (head, "…", tail) if x)
    if any(not c for c in cores):
        cores, template = shorts, ""
    items = list(zip(cores, cols))
    if len(cols) < min_family:
        return template, [Node(c, col) for c, col in items]
    return template, _group(items, min_group=min_group, max_leaves=max_leaves, depth=1)


__all__ = ["Node", "auc", "compact_family", "plot_distributions", "plot_joint", "plot_over_time",
           "short_name"]
