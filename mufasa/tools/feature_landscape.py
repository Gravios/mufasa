"""``mufasa-landscape`` — joint PDFs of multimodal feature pairs and a t-SNE map.

Loads, for every session of a project, the Kalman v2 smoothed pose
(``derived/smoothed/kalman_v2/<run_id>/<video>_smoothed_v2.<hash>.parquet``)
and the computed features (``derived/features/<family>/<video>.parquet``),
samples frames from all of them, then:

1. **Screens every pair of columns for multimodality with dipsplit.** Each
   pair is robust-standardised and projected onto six directions: the two
   axes (the columns themselves), the two diagonals and the pair's two
   principal axes. dipsplit cuts each projection into modes: Hartigan's dip
   test (significant at ``--alpha`` and at least ``--min-dip``) decides
   whether a segment is multimodal, the deepest valley of its smoothed
   density is where it is cut, and both halves are tested again. A pair is
   *multimodal* when any projection has two or more modes; it *adds modes*
   when a joint projection has more modes than either column alone, and is
   *emergent* when both columns are unimodal but a joint projection is not.
2. **Plots the joint PDF** (2-D log-density histogram, marginal densities,
   dipsplit mode boundaries) of the top-ranked pairs: emergent first, then
   pairs that add modes, then the rest by dip statistic, with at most
   ``--max-per-feature`` pairs per column and ``--max-per-family`` per
   feature family (so a family with hundreds of columns — three-point
   angles, one per marker triple — cannot take every slot).

Angles are circular; ``--angles`` picks how they enter (rotate the cut to
the emptiest direction and drop full-circle angles, fold mirror images,
or drop them), and ``--include-families`` / ``--exclude-families`` restrict
the screen by family.
3. **Runs t-SNE** on a random sample of frames (robust z-scored, PCA to 50
   dimensions first) and plots it coloured by session and by the features
   that appear most often in the multimodal pairs.

Outputs go to ``derived/plots/feature_landscape/<run_id>/``: ``pairs.csv``
(every screened pair), ``jpdf_top.png`` (grid), ``jpdf_pairs.pdf`` (one page
per plotted pair), ``tsne.png``, ``tsne.parquet`` (embedding with session and
frame index), ``tsne_samples.parquet`` (the embedded frames' column values,
row-aligned, for ``mufasa-landscape-animate``) and ``run.toml``.

Usage::

    mufasa-landscape                       # project found from the cwd
    mufasa-landscape --project /data/proj --frames-per-session 4000 \\
        --tsne-sample 15000 --n-jobs 16

Needs the ``diptest`` package (``pip install mufasa[analysis]``).
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mufasa.project_layout import (
    ProjectPaths,
    SmoothingFlavors,
    Stages,
    find_project_config,
    generate_run_id,
    video_name_from_pose_stem,
    write_run_toml,
)

POSE_PREFIX = "kfv2."


# --------------------------------------------------------------------------- #
# dipsplit
# --------------------------------------------------------------------------- #
def _dip():
    try:
        import diptest
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "feature_landscape needs the 'diptest' package: "
            "pip install diptest  (or: pip install -e '.[analysis]')"
        ) from exc
    return diptest


def dipsplit(x: np.ndarray, alpha: float = 0.01, min_n: int = 100,
             min_dip: float = 0.01, max_valley: float = 0.9,
             _depth: int = 0) -> list[tuple[float, float]]:
    """Split 1-D data into modes; return the mode intervals (low, high).

    Recursive dip splitting: Hartigan's dip test decides whether a segment
    is multimodal (p <= ``alpha`` *and* dip >= ``min_dip``, an effect-size
    floor — with tens of thousands of frames the test alone flags
    negligible departures from unimodality). A multimodal segment is cut
    at the deepest valley of its smoothed density (valley / lower
    neighbouring peak < ``max_valley``) and both sides are tested again.
    A one-sided tail ending at a valley is unimodal to the dip test, so
    tails do not become spurious modes. One interval means unimodal.
    """
    xs = np.sort(np.asarray(x, np.float64)[np.isfinite(x)])
    if xs.size == 0:
        return []
    if not is_multimodal(xs, alpha, min_n, min_dip) or _depth > 8:
        return [(float(xs[0]), float(xs[-1]))]
    cut = deepest_valley(xs, max_valley)
    if cut is None:
        return [(float(xs[0]), float(xs[-1]))]
    k = int(np.searchsorted(xs, cut))
    if k < min_n // 2 or xs.size - k < min_n // 2:
        return [(float(xs[0]), float(xs[-1]))]
    return (dipsplit(xs[:k], alpha, min_n, min_dip, max_valley, _depth + 1)
            + dipsplit(xs[k:], alpha, min_n, min_dip, max_valley, _depth + 1))


def is_multimodal(xs: np.ndarray, alpha: float, min_n: int,
                  min_dip: float) -> bool:
    """Dip test on sorted data: significant and at least ``min_dip``."""
    if xs.size < min_n or xs[0] == xs[-1]:
        return False
    dip, p = _dip().diptest(xs, sort_x=False)
    return p <= alpha and dip >= min_dip


def deepest_valley(xs: np.ndarray, max_valley: float = 0.9,
                   n_bins: int = 256) -> float | None:
    """Location of the deepest density valley between two peaks, or None.

    Density: histogram over the 0.5-99.5 percentile range smoothed with a
    Gaussian of Silverman's bandwidth. Depth of a valley: its density over
    the lower of the highest peaks on either side; the smallest ratio wins
    if below ``max_valley``.
    """
    from scipy.ndimage import gaussian_filter1d
    lo, hi = np.percentile(xs, [0.5, 99.5])
    if not hi > lo:
        return None
    h, e = np.histogram(xs, bins=n_bins, range=(lo, hi))
    q1, q3 = np.percentile(xs, [25, 75])
    bw = 0.9 * min(xs.std(), (q3 - q1) / 1.349) * xs.size ** -0.2
    if not bw > 0:
        return None
    d = gaussian_filter1d(h.astype(np.float64), bw / (e[1] - e[0]), mode="constant")
    best, best_ratio = None, max_valley
    for i in range(1, n_bins - 1):
        if d[i] <= d[i - 1] and d[i] < d[i + 1]:
            ratio = d[i] / max(min(d[:i].max(), d[i + 1:].max()), 1e-12)
            if ratio < best_ratio:
                best, best_ratio = i, ratio
    return None if best is None else float((e[best] + e[best + 1]) / 2)


def split_points(modes: list[tuple[float, float]]) -> list[float]:
    """Boundaries between consecutive modes (gap midpoints)."""
    return [(modes[i][1] + modes[i + 1][0]) / 2 for i in range(len(modes) - 1)]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    name: str
    pose_path: Path | None = None
    n_frames: int = 0
    frames: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    data: pd.DataFrame | None = None


def latest_kfv2_files(paths: ProjectPaths, run: str | None) -> dict[str, Path]:
    """{video: smoothed parquet} from the chosen (default: latest) Kalman v2
    run. Several files for one video (different parameter hashes) → newest."""
    if run:
        run_dir = paths.stage_dir(Stages.SMOOTHED) / SmoothingFlavors.KALMAN_V2 / run
    else:
        run_dir = paths.latest_run(Stages.SMOOTHED, flavor=SmoothingFlavors.KALMAN_V2)
    if run_dir is None or not run_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    for p in sorted(run_dir.glob("*.parquet"), key=lambda q: q.stat().st_mtime):
        out[video_name_from_pose_stem(p.stem)] = p
    return out


def _pose_columns(cols, which: str) -> list[str]:
    keep = ("_x", "_y") if which == "xy" else ("_x", "_y", "_p", "_var_x", "_var_y")
    return [c for c in cols if c.endswith(keep)
            and (which != "xy" or not c.endswith(("_var_x", "_var_y")))]


def load_sessions(config: Path, *, sources: set[str], kfv2_run: str | None,
                  pose_cols: str, frames_per_session: int, seed: int,
                  log=print) -> list[Session]:
    """Load a frame sample from every session that has all requested sources."""
    from mufasa.utils.feature_io import (
        list_video_stems_with_features,
        load_features_for_video,
    )
    paths = ProjectPaths.open(config.parent)
    pose_files = latest_kfv2_files(paths, kfv2_run) if "pose" in sources else {}
    feat_videos = set(list_video_stems_with_features(str(config))) \
        if "features" in sources else set()
    if "pose" in sources and not pose_files:
        raise SystemExit("no Kalman v2 smoothed pose found under "
                         f"{paths.stage_dir(Stages.SMOOTHED) / SmoothingFlavors.KALMAN_V2}")
    if "features" in sources and not feat_videos:
        raise SystemExit("no features found under derived/features/ "
                         "(run Features → Compute Features first)")
    names = set(pose_files) | feat_videos
    if "pose" in sources:
        names &= set(pose_files)
    if "features" in sources:
        names &= feat_videos
    skipped = (set(pose_files) | feat_videos) - names
    if skipped:
        log(f"[landscape] skipping {len(skipped)} session(s) without all of "
            f"{sorted(sources)}: {', '.join(sorted(skipped)[:8])}"
            + (" …" if len(skipped) > 8 else ""))
    rng = np.random.default_rng(seed)
    sessions: list[Session] = []
    global COLUMN_FAMILY
    COLUMN_FAMILY = {}
    for name in sorted(names):
        if not COLUMN_FAMILY:
            COLUMN_FAMILY = column_families(paths, name, pose_files.get(name),
                                            pose_cols, sources)
        parts = []
        if "pose" in sources:
            pose = pd.read_parquet(pose_files[name])
            pose = pose[_pose_columns(pose.columns, pose_cols)]
            parts.append(pose.add_prefix(POSE_PREFIX))
        if "features" in sources:
            parts.append(load_features_for_video(name, str(config)))
        n = min(len(p) for p in parts)
        if len({len(p) for p in parts}) > 1:
            log(f"[landscape] {name}: pose / features lengths differ "
                f"({', '.join(str(len(p)) for p in parts)}); using the first {n} frames")
        df = pd.concat([p.iloc[:n].reset_index(drop=True) for p in parts], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        num = df.select_dtypes(include=[np.number, "bool"]).astype(np.float32)
        take = np.sort(rng.choice(n, size=min(n, frames_per_session), replace=False))
        sessions.append(Session(name, pose_files.get(name), n, take,
                                num.iloc[take].reset_index(drop=True)))
        log(f"[landscape] {name}: {n} frames, {num.shape[1]} columns, "
            f"{take.size} sampled")
    if not sessions:
        raise SystemExit("no session has all requested sources")
    return sessions


COLUMN_FAMILY: dict[str, str] = {}


def column_families(paths: ProjectPaths, name: str, pose_file: Path | None,
                    pose_cols: str, sources: set[str]) -> dict[str, str]:
    """{column: family} for one session, from the per-family parquet
    schemas (no data read), so the log can say what each family
    contributed and what survived the column filters."""
    import pyarrow.parquet as pq
    fam: dict[str, str] = {}
    if pose_file is not None and "pose" in sources:
        for c in _pose_columns(pq.read_schema(pose_file).names, pose_cols):
            fam[POSE_PREFIX + c] = "kfv2 pose"
    root = paths.derived_dir / "features"
    if "features" in sources and root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            f = d / f"{name}.parquet"
            if f.is_file():
                for c in pq.read_schema(f).names:
                    fam.setdefault(c, d.name)
        wide = root / f"{name}.parquet"
        if wide.is_file():
            for c in pq.read_schema(wide).names:
                fam.setdefault(c, "(wide feature file)")
    return fam


def family_counts(cols) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cols:
        f = COLUMN_FAMILY.get(c, "(other)")
        out[f] = out.get(f, 0) + 1
    return out


def log_families(title: str, cols, ref=None, log=print) -> None:
    now = family_counts(cols)
    base = family_counts(ref) if ref is not None else None
    log(f"[landscape] {title}:")
    for f in sorted(set(now) | set(base or {})):
        n = now.get(f, 0)
        log(f"[landscape]   {f:<55} {n:>4}" + (f" / {base[f]}" if base else ""))


ANGLE_MODES = ("unwrap", "fold", "drop")


def is_angle_column(name: str, v: np.ndarray) -> bool:
    w = v[np.isfinite(v)]
    return "angle" in name.lower() and bool(w.size) and w.min() >= -360 and w.max() <= 360


def clean_columns(X: pd.DataFrame, angles: str = "unwrap", log=print) -> pd.DataFrame:
    """Undo artefacts that would otherwise show up as 'modes'.

    * The feature writer stores missing values as -1; in a column whose
      other values are all >= 0 (distances, areas, movements, angles) a
      -1 is a missing value, not a measurement → NaN.
    * Angles are circular, and a dip test on a line cannot read them.
      ``angles`` picks the treatment:

      - ``unwrap`` (default): rotate each angle so its 0/360° cut falls at
        its emptiest direction (values stay in degrees). An angle with no
        empty direction — it covers the whole circle, so any cut splits a
        lump — is dropped;
      - ``fold``: replace a by min(a, 360 - a) (0–180°): mirror-image
        configurations (bent left / bent right, a marker on either side)
        become one value, so left/right symmetry no longer reads as two
        modes;
      - ``drop``: leave angles out.
    """
    if angles not in ANGLE_MODES:
        raise ValueError(f"angles must be one of {ANGLE_MODES}")
    X = X.copy()
    n_sent = 0
    ang: dict[str, int] = {"unwrapped": 0, "folded": 0, "dropped": 0, "full-circle": 0}
    drop = []
    for c in X.columns:
        v = X[c].to_numpy(np.float64, copy=True)
        fin = np.isfinite(v)
        sent = fin & (v == -1)
        if sent.any() and (v[fin & ~sent] >= 0).all():
            v[sent] = np.nan
            n_sent += 1
        if is_angle_column(c, v):
            if angles == "drop":
                drop.append(c)
                ang["dropped"] += 1
                continue
            if angles == "fold":
                a = np.mod(v, 360.0)
                v = np.minimum(a, 360.0 - a)
                ang["folded"] += 1
            else:
                v, gap = unwrap_angle(v, return_gap=True)
                if gap > 0.25:
                    drop.append(c)
                    ang["full-circle"] += 1
                    continue
                ang["unwrapped"] += 1
        X[c] = v.astype(np.float32)
    X = X.drop(columns=drop)
    if n_sent:
        log(f"[landscape] -1 placeholders read as missing in {n_sent} non-negative column(s)")
    if any(ang.values()):
        parts = [f"{n} {k}" for k, n in ang.items() if n]
        log(f"[landscape] angle columns ({angles}): " + ", ".join(parts)
            + (" — full-circle angles have no gap to cut at and are left out"
               if ang["full-circle"] else ""))
    return X


def unwrap_angle(v: np.ndarray, n_bins: int = 72, return_gap: bool = False):
    """Rotate degrees so the cut lies at the least-populated direction.

    With ``return_gap`` also returns the emptiest direction's density
    relative to the mean (0 = a real gap, ~1 = uniform around the circle).
    """
    fin = np.isfinite(v)
    a = np.mod(v[fin], 360.0)
    if a.size == 0:
        return (v, 0.0) if return_gap else v
    h, _ = np.histogram(a, bins=n_bins, range=(0, 360))
    hs = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5) / 5, mode="valid")
    k = int(np.argmin(hs))
    cut = (k + 0.5) * 360.0 / n_bins
    out = np.full_like(v, np.nan, dtype=np.float64)
    out[fin] = cut + np.mod(a - cut, 360.0)
    gap = float(hs[k] / max(hs.mean(), 1e-12))
    return (out, gap) if return_gap else out


def keep_families(X: pd.DataFrame, include: str | None, exclude: str | None,
                  log=print) -> pd.DataFrame:
    """Restrict columns by family (comma-separated substrings of the family
    directory name, e.g. ``angles,edge``; the pose is family ``kfv2 pose``)."""
    def match(c, spec):
        fam = COLUMN_FAMILY.get(c, "(other)")
        return any(t.strip() and t.strip().lower() in fam.lower() for t in spec.split(","))
    cols = list(X.columns)
    if include:
        cols = [c for c in cols if match(c, include)]
    if exclude:
        cols = [c for c in cols if not match(c, exclude)]
    if len(cols) < X.shape[1]:
        log(f"[landscape] family filter: {len(cols)}/{X.shape[1]} columns kept")
    return X[cols]


def stack(sessions: list[Session], log=print) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Concatenate session samples on the columns all sessions share."""
    common = set(sessions[0].data.columns)
    for s in sessions[1:]:
        common &= set(s.data.columns)
    order = [c for c in sessions[0].data.columns if c in common]
    dropped = {c for s in sessions for c in s.data.columns} - common
    if dropped:
        log(f"[landscape] {len(dropped)} column(s) missing in some sessions "
            "were dropped")
    if dropped:
        lost = family_counts(dropped)
        log("[landscape]   by family: " + ", ".join(f"{f}: {n}" for f, n in sorted(lost.items())))
    X = pd.concat([s.data[order] for s in sessions], ignore_index=True)
    sess = np.concatenate([[i] * len(s.data) for i, s in enumerate(sessions)])
    frame = np.concatenate([s.frames for s in sessions])
    return X, sess, frame


def select_columns(X: pd.DataFrame, *, max_nan: float, min_unique: int,
                   max_corr: float, max_features: int | None,
                   log=print) -> list[str]:
    """Usable columns for the screen: mostly finite, not (near-)discrete —
    a 0/1 inside-ROI flag is trivially 'bimodal' — and not duplicates."""
    keep = []
    for c in X.columns:
        v = X[c].to_numpy()
        fin = np.isfinite(v)
        if fin.mean() < 1 - max_nan:
            continue
        if np.unique(v[fin][:20000]).size < min_unique:
            continue
        q1, q3 = np.nanpercentile(v, [25, 75])
        if not q3 > q1:
            continue
        keep.append(c)
    log(f"[landscape] {len(keep)}/{X.shape[1]} columns usable "
        f"(≤{max_nan:.0%} NaN, ≥{min_unique} distinct values, IQR > 0)")
    if max_corr < 1 and len(keep) > 1:
        Z = robust_z(X[keep]).fillna(0.0).to_numpy(np.float64)
        sub = Z[np.random.default_rng(0).choice(len(Z), min(len(Z), 20000), replace=False)]
        C = np.abs(np.corrcoef(sub, rowvar=False))
        drop = set()
        for i in range(len(keep)):
            if i in drop:
                continue
            for j in range(i + 1, len(keep)):
                if j not in drop and C[i, j] > max_corr:
                    drop.add(j)
        if drop:
            log(f"[landscape] {len(drop)} near-duplicate column(s) dropped "
                f"(|r| > {max_corr})")
        keep = [c for i, c in enumerate(keep) if i not in drop]
    if max_features and len(keep) > max_features:
        var = robust_z(X[keep]).abs().mean().sort_values(ascending=False)
        keep = [c for c in keep if c in set(var.index[:max_features])]
        log(f"[landscape] capped to {max_features} columns")
    return keep


def robust_z(X: pd.DataFrame) -> pd.DataFrame:
    med = X.median()
    iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, np.nan)
    return (X - med) / (iqr / 1.349)


# --------------------------------------------------------------------------- #
# Pair screen
# --------------------------------------------------------------------------- #
DIRECTIONS = ("x", "y", "diag+", "diag-", "pc1", "pc2")


def _proj_stats(v: np.ndarray, dp: dict) -> tuple[float, float, int]:
    """(dip, p, number of dipsplit modes) of one projection."""
    dip, p = _dip().diptest(v)
    modes = len(dipsplit(v, **dp)) if p <= dp["alpha"] and dip >= dp["min_dip"] else 1
    return float(dip), float(p), modes


def _pair_stats(a: np.ndarray, b: np.ndarray, dp: dict,
                ma: tuple, mb: tuple) -> dict:
    """Stats of one pair; ``ma`` / ``mb`` are the columns' own (marginal)
    projection stats, computed once per column on the same frames."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok].astype(np.float64), b[ok].astype(np.float64)
    if a.size < max(dp["min_n"], 20):
        return {"n": int(a.size)}
    if ok.all():
        st = {"x": ma, "y": mb}
    else:
        st = {"x": _proj_stats(a, dp), "y": _proj_stats(b, dp)}
    P = np.stack([a, b], 1)
    _, V = np.linalg.eigh(np.cov(P, rowvar=False))
    for k, v in (("diag+", (a + b) / math.sqrt(2)), ("diag-", (a - b) / math.sqrt(2)),
                 ("pc1", P @ V[:, 1]), ("pc2", P @ V[:, 0])):
        st[k] = _proj_stats(v, dp)
    out = {"n": int(a.size)}
    for k in DIRECTIONS:
        out[f"dip_{k}"], out[f"p_{k}"], out[f"modes_{k}"] = st[k]
    joint = ("diag+", "diag-", "pc1", "pc2")
    out["modes_marginal"] = max(out["modes_x"], out["modes_y"])
    out["modes_joint"] = max(out[f"modes_{k}"] for k in DIRECTIONS)
    # most modes; on a tie prefer a column axis (simplest to read), then dip
    out["best_direction"] = max(DIRECTIONS, key=lambda k: (
        out[f"modes_{k}"], k in ("x", "y"), out[f"dip_{k}"]))
    out["dip_max"] = max(out[f"dip_{k}"] for k in DIRECTIONS)
    out["dip_joint_max"] = max(out[f"dip_{k}"] for k in joint)
    out["multimodal"] = out["modes_joint"] >= 2
    out["emergent"] = out["modes_marginal"] < 2 and max(out[f"modes_{k}"] for k in joint) >= 2
    return out


def _marginal_chunk(Z: np.ndarray, idx: list[int], dp: dict) -> list[tuple]:
    out = []
    for i in idx:
        v = Z[:, i].astype(np.float64)
        v = v[np.isfinite(v)]
        out.append(_proj_stats(v, dp) if v.size >= 20 else (np.nan, 1.0, 1))
    return out


def _pair_chunk(Z: np.ndarray, pairs: list[tuple[int, int]], dp: dict,
                marg: list[tuple]) -> list[dict]:
    return [dict(i=i, j=j, **_pair_stats(Z[:, i], Z[:, j], dp, marg[i], marg[j]))
            for i, j in pairs]


def screen_pairs(Z: pd.DataFrame, *, dp: dict, n_jobs: int,
                 log=print) -> pd.DataFrame:
    cols = list(Z.columns)
    pairs = list(itertools.combinations(range(len(cols)), 2))
    log(f"[landscape] dipsplit screen: {len(cols)} columns → {len(pairs)} pairs "
        f"× {len(DIRECTIONS)} projections on {len(Z)} frames ({n_jobs} jobs)")
    A = Z.to_numpy(np.float32)
    t0 = time.time()
    if n_jobs == 1 or len(pairs) < 200:
        marg = _marginal_chunk(A, list(range(len(cols))), dp)
        rows = _pair_chunk(A, pairs, dp, marg)
    else:
        from joblib import Parallel, delayed
        par = Parallel(n_jobs=n_jobs)
        k = max(1, len(cols) // n_jobs)
        marg = [m for part in par(delayed(_marginal_chunk)(A, list(range(i, min(i + k, len(cols)))), dp)
                                  for i in range(0, len(cols), k)) for m in part]
        k = max(1, len(pairs) // (n_jobs * 8))
        chunks = [pairs[i:i + k] for i in range(0, len(pairs), k)]
        rows = [r for part in par(delayed(_pair_chunk)(A, ch, dp, marg) for ch in chunks)
                for r in part]
    log(f"[landscape] screened in {time.time() - t0:.1f} s")
    df = pd.DataFrame(rows)
    df.insert(0, "feature_a", [cols[i] for i in df.pop("i")])
    df.insert(1, "feature_b", [cols[j] for j in df.pop("j")])
    for c in ("multimodal", "emergent", "modes_joint", "modes_marginal",
              "dip_joint_max", "dip_max"):
        if c not in df:
            df[c] = np.nan
    for c in ("multimodal", "emergent"):
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    # a pair "adds modes" when some joint projection has more modes than
    # either marginal: the pair says more than its columns do alone
    df["adds_modes"] = df["modes_joint"].fillna(1) > df["modes_marginal"].fillna(1)
    return df.sort_values(["emergent", "adds_modes", "multimodal", "dip_joint_max",
                           "dip_max"], ascending=False, ignore_index=True)


def pick_pairs(pairs: pd.DataFrame, n: int, max_per_feature: int,
               max_per_family: int | None = None) -> pd.DataFrame:
    """Top-ranked multimodal pairs for plotting, spread out:

    * at most ``max_per_feature`` pairs containing any one column, so one
      bimodal column does not fill the page with its pairings;
    * at most ``max_per_family`` pairs touching any one feature family
      (needs the ``family_a`` / ``family_b`` columns), so a family with
      hundreds of columns (three-point angles: one per marker triple)
      cannot take every slot. Fewer than ``n`` pairs come back when the
      caps run out; raise them to see more.
    """
    has_fam = bool(max_per_family) and {"family_a", "family_b"} <= set(pairs.columns)
    seen: dict[str, int] = {}
    fseen: dict[str, int] = {}
    keep: list = []
    for i, r in pairs[pairs["multimodal"]].iterrows():
        if len(keep) == n:
            break
        if seen.get(r.feature_a, 0) >= max_per_feature or \
                seen.get(r.feature_b, 0) >= max_per_feature:
            continue
        fams = {r.family_a, r.family_b} if has_fam else set()
        if any(fseen.get(f, 0) >= max_per_family for f in fams):
            continue
        keep.append(i)
        seen[r.feature_a] = seen.get(r.feature_a, 0) + 1
        seen[r.feature_b] = seen.get(r.feature_b, 0) + 1
        for f in fams:
            fseen[f] = fseen.get(f, 0) + 1
    return pairs.loc[keep]


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _range(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(v, [0.5, 99.5])
    if not hi > lo:
        lo, hi = np.nanmin(v), np.nanmax(v) + 1e-9
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def _draw_joint_splits(ax, a: np.ndarray, b: np.ndarray, row: pd.Series,
                       dp: dict, xr, yr) -> None:
    """Draw dipsplit boundaries of the best diagonal / principal projection
    as lines in the plot's raw units (the screen works on robust z-scores)."""
    k = row.get("best_direction")
    if k not in ("diag+", "diag-", "pc1", "pc2"):
        return
    def rz(v):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        return med, max((q3 - q1) / 1.349, 1e-12)
    (ma, sa), (mb, sb) = rz(a), rz(b)
    za, zb = (a - ma) / sa, (b - mb) / sb
    if k in ("diag+", "diag-"):
        d = np.array([1.0, 1.0 if k == "diag+" else -1.0]) / math.sqrt(2)
    else:
        _, V = np.linalg.eigh(np.cov(np.stack([za, zb]), rowvar=True))
        d = V[:, 1] if k == "pc1" else V[:, 0]
    for s in split_points(dipsplit(za * d[0] + zb * d[1], **dp)):
        # z-space line d0*za + d1*zb = s, drawn in raw units
        if abs(d[1]) > 1e-9:
            x = np.linspace(*xr, 50)
            y = mb + sb * (s - d[0] * (x - ma) / sa) / d[1]
        else:
            y = np.linspace(*yr, 50)
            x = np.full_like(y, ma + sa * s / d[0])
        ax.plot(x, y, color="#7fd3ff", lw=1.0, ls=":")
    ax.set_xlim(*xr), ax.set_ylim(*yr)


def draw_jpdf(fig, spec, X: pd.DataFrame, row: pd.Series, *, bins: int,
              dp: dict):
    """Joint log-density with marginal densities and dipsplit boundaries."""
    import matplotlib.colors as mcolors
    a = X[row.feature_a].to_numpy(np.float64)
    b = X[row.feature_b].to_numpy(np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    gs = spec.subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)
    xr, yr = _range(a), _range(b)
    H, xe, ye = np.histogram2d(a, b, bins=bins, range=(xr, yr), density=True)
    H = np.ma.masked_less_equal(H.T, 0)
    ax.pcolormesh(xe, ye, H, cmap="magma", shading="auto",
                  norm=mcolors.LogNorm(vmin=max(H.min(), H.max() * 1e-4), vmax=H.max()))
    for v, axm, orient in ((a, axx, "vertical"), (b, axy, "horizontal")):
        rng_ = xr if axm is axx else yr
        h, e = np.histogram(v, bins=bins, range=rng_, density=True)
        c = (e[:-1] + e[1:]) / 2
        if orient == "vertical":
            axm.fill_between(c, h, color="#D9952B", alpha=0.8, lw=0)
        else:
            axm.fill_betweenx(c, h, color="#D9952B", alpha=0.8, lw=0)
        for s in split_points(dipsplit(v, **dp)):
            (ax.axvline if axm is axx else ax.axhline)(s, color="#7fd3ff", lw=0.8, ls="--")
            (axm.axvline if axm is axx else axm.axhline)(s, color="#1f6f99", lw=0.8, ls="--")
        axm.axis("off")
    _draw_joint_splits(ax, a, b, row, dp, xr, yr)
    tag = "emergent" if row.get("emergent") else (
        "joint adds modes" if row.get("adds_modes") else "multimodal")
    axx.set_title(f"{tag}: {int(row.modes_joint)} modes along {row.best_direction}  "
                  f"(dip {row.dip_max:.3f})", fontsize=7, loc="left")
    ax.set_xlabel(_short(row.feature_a), fontsize=7)
    ax.set_ylabel(_short(row.feature_b), fontsize=7)
    ax.tick_params(labelsize=6)
    return ax


def _short(name: str, n: int = 48) -> str:
    return name if len(name) <= n else name[:n - 1] + "…"


def plot_pairs(X: pd.DataFrame, top: pd.DataFrame, out: Path, *, bins: int,
               dp: dict, log=print) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    if top.empty:
        log("[landscape] no multimodal pairs to plot")
        return
    n = len(top)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig = plt.figure(figsize=(4.2 * ncol, 4.2 * nrow), constrained_layout=True)
    gs = fig.add_gridspec(nrow, ncol)
    for k, (_, row) in enumerate(top.iterrows()):
        draw_jpdf(fig, gs[k // ncol, k % ncol], X, row, bins=bins, dp=dp)
    fig.suptitle(f"Joint PDFs of the {n} top-ranked multimodal pairs "
                 f"({len(X)} frames)", fontsize=10)
    fig.savefig(out / "jpdf_top.png", dpi=150)
    plt.close(fig)
    with PdfPages(out / "jpdf_pairs.pdf") as pdf:
        for _, row in top.iterrows():
            fig = plt.figure(figsize=(6.5, 6.5), constrained_layout=True)
            draw_jpdf(fig, fig.add_gridspec(1, 1)[0], X, row, bins=bins, dp=dp)
            pdf.savefig(fig)
            plt.close(fig)
    log(f"[landscape] wrote jpdf_top.png and jpdf_pairs.pdf ({n} pairs)")


def run_tsne(Z: pd.DataFrame, *, n: int, perplexity: float, seed: int,
             pca_dims: int, n_jobs: int, log=print) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(Z), size=min(n, len(Z)), replace=False))
    A = Z.iloc[idx].to_numpy(np.float64)
    A = np.nan_to_num(np.clip(A, -10, 10), nan=0.0)   # robust z: clip, median-impute
    if pca_dims and A.shape[1] > pca_dims:
        A = PCA(n_components=pca_dims, random_state=seed).fit_transform(A)
    log(f"[landscape] t-SNE on {len(idx)} frames × {A.shape[1]} dims "
        f"(perplexity {perplexity})")
    t0 = time.time()
    E = TSNE(n_components=2, perplexity=min(perplexity, (len(idx) - 1) / 3),
             init="pca", learning_rate="auto", random_state=seed,
             n_jobs=n_jobs).fit_transform(A)
    log(f"[landscape] t-SNE done in {time.time() - t0:.1f} s")
    return idx, E


def plot_tsne(E: np.ndarray, sess: np.ndarray, names: list[str], X: pd.DataFrame,
              color_by: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("session", None)] + [(c, c) for c in color_by]
    ncol = min(3, len(panels))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5 * nrow),
                             squeeze=False, constrained_layout=True)
    s = max(0.5, min(6.0, 4e4 / max(len(E), 1)))
    for ax, (title, col) in zip(axes.flat, panels):
        if col is None:
            cmap = plt.get_cmap("tab20")
            ax.scatter(E[:, 0], E[:, 1], c=[cmap(i % 20) for i in sess], s=s,
                       lw=0, rasterized=True)
            ax.set_title(f"by session ({len(names)})", fontsize=9)
        else:
            v = X[col].to_numpy(np.float64)
            lo, hi = np.nanpercentile(v, [1, 99])
            sc = ax.scatter(E[:, 0], E[:, 1], c=np.clip(v, lo, hi), cmap="viridis",
                            s=s, lw=0, rasterized=True)
            fig.colorbar(sc, ax=ax, shrink=0.8)
            ax.set_title(_short(title, 60), fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.savefig(out / "tsne.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mufasa-landscape",
        description="Joint PDFs of multimodal feature pairs (dipsplit) and a "
                    "t-SNE map, from the Kalman v2 pose and features of all "
                    "sessions in a project.")
    p.add_argument("--project", default=".",
                   help="project.toml, or any path inside the project (default: cwd)")
    p.add_argument("--sources", default="pose,features",
                   help="what to load: pose, features or both (default both); "
                        "sessions lacking a requested source are skipped")
    p.add_argument("--kfv2-run", default=None,
                   help="Kalman v2 run id (default: latest run)")
    p.add_argument("--pose-cols", choices=("xy", "all"), default="xy",
                   help="pose columns: marker x/y only (default) or also p and variances")
    p.add_argument("--frames-per-session", type=int, default=3000,
                   help="frames sampled per session for everything below (default 3000)")
    p.add_argument("--screen-sample", type=int, default=20000,
                   help="frames used for the dipsplit pair screen (default 20000)")
    p.add_argument("--alpha", type=float, default=0.01,
                   help="dip-test significance level (default 0.01)")
    p.add_argument("--min-mode-n", type=int, default=100,
                   help="smallest sample dipsplit will split further (default 100)")
    p.add_argument("--min-dip", type=float, default=0.01,
                   help="dip-statistic floor for a split: an effect size, since with "
                        "many frames the test alone flags negligible departures "
                        "(default 0.01)")
    p.add_argument("--max-valley", type=float, default=0.9,
                   help="split only at a density valley below this fraction of the "
                        "lower neighbouring peak (default 0.9)")
    p.add_argument("--max-nan", type=float, default=0.2)
    p.add_argument("--min-unique", type=int, default=20,
                   help="drop columns with fewer distinct values (flags, counts)")
    p.add_argument("--max-corr", type=float, default=0.98,
                   help="drop one of each pair of columns with |r| above this")
    p.add_argument("--max-features", type=int, default=None,
                   help="cap the screen to this many columns")
    p.add_argument("--n-plot", type=int, default=24,
                   help="joint PDFs plotted (top-ranked multimodal pairs)")
    p.add_argument("--angles", choices=ANGLE_MODES, default="unwrap",
                   help="angle columns: unwrap (rotate the 0/360 cut to the emptiest "
                        "direction; full-circle angles dropped), fold (min(a, 360-a): "
                        "left/right mirror images merged) or drop")
    p.add_argument("--include-families", default=None,
                   help="only these families (comma-separated parts of the family "
                        "folder name, e.g. 'distances,movements,head_velocity'; "
                        "'kfv2' = the pose)")
    p.add_argument("--exclude-families", default=None,
                   help="leave these families out (same matching, e.g. 'angles,edge')")
    p.add_argument("--max-per-family", type=int, default=6,
                   help="plot at most this many pairs touching any one feature family "
                        "(default 6)")
    p.add_argument("--max-per-feature", type=int, default=3,
                   help="plot at most this many pairs containing any one column")
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--tsne-sample", type=int, default=10000)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--pca-dims", type=int, default=50)
    p.add_argument("--no-tsne", action="store_true")
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="output directory (default derived/plots/feature_landscape/<run_id>)")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    _dip()
    start = Path(a.project).expanduser().resolve()
    config = start if start.name == "project.toml" else find_project_config(start)
    if config is None:
        raise SystemExit(f"no project.toml found at or above {start}")
    sources = {s.strip() for s in a.sources.split(",") if s.strip()}
    if not sources <= {"pose", "features"} or not sources:
        raise SystemExit("--sources takes pose, features or pose,features")
    run_id = generate_run_id()
    out = Path(a.out) if a.out else (config.parent / "derived" / "plots"
                                     / "feature_landscape" / run_id)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[landscape] project {config.parent}  →  {out}")

    sessions = load_sessions(config, sources=sources, kfv2_run=a.kfv2_run,
                             pose_cols=a.pose_cols,
                             frames_per_session=a.frames_per_session, seed=a.seed)
    X, sess, frame = stack(sessions)
    X = clean_columns(X, angles=a.angles)
    X = keep_families(X, a.include_families, a.exclude_families)
    cols = select_columns(X, max_nan=a.max_nan, min_unique=a.min_unique,
                          max_corr=a.max_corr, max_features=a.max_features)
    log_families("columns screened per family (kept / loaded)", cols, list(X.columns))
    if len(cols) < 2:
        raise SystemExit("fewer than two usable columns")
    Z = robust_z(X[cols])

    rng = np.random.default_rng(a.seed)
    sidx = np.sort(rng.choice(len(Z), size=min(a.screen_sample, len(Z)), replace=False))
    dp = dict(alpha=a.alpha, min_n=a.min_mode_n, min_dip=a.min_dip,
              max_valley=a.max_valley)
    pairs = screen_pairs(Z.iloc[sidx], dp=dp, n_jobs=a.n_jobs)
    pairs.insert(2, "family_a", [COLUMN_FAMILY.get(c, "(other)") for c in pairs.feature_a])
    pairs.insert(3, "family_b", [COLUMN_FAMILY.get(c, "(other)") for c in pairs.feature_b])
    pairs.to_csv(out / "pairs.csv", index=False)
    mm = pairs[pairs["multimodal"]]
    if len(mm):
        fc = pd.concat([mm.family_a, mm.family_b]).value_counts()
        print("[landscape] multimodal pairs per family: "
              + ", ".join(f"{f} {n}" for f, n in fc.items()))
    n_multi = int(pairs["multimodal"].sum())
    n_emerg = int(pairs["emergent"].sum())
    n_adds = int(pairs["adds_modes"].sum())
    print(f"[landscape] {n_multi}/{len(pairs)} pairs multimodal; {n_adds} with more "
          f"modes jointly than in either column, {n_emerg} of them emergent "
          f"(unimodal columns)")
    top = pick_pairs(pairs, a.n_plot, a.max_per_feature, a.max_per_family)
    if 0 < len(top) < min(a.n_plot, n_multi):
        print(f"[landscape] plotting {len(top)} pairs: the per-family "
              f"(--max-per-family {a.max_per_family}) and per-column "
              f"(--max-per-feature {a.max_per_feature}) caps leave no more")
    plot_pairs(X[cols], top, out, bins=a.bins, dp=dp)

    results = {"sessions": len(sessions), "frames": int(len(X)),
               "columns": len(cols), "pairs": int(len(pairs)),
               "multimodal_pairs": n_multi, "emergent_pairs": n_emerg}
    if not a.no_tsne:
        idx, E = run_tsne(Z, n=a.tsne_sample, perplexity=a.perplexity,
                          seed=a.seed, pca_dims=a.pca_dims, n_jobs=a.n_jobs)
        names = [s.name for s in sessions]
        pd.DataFrame({"tsne_1": E[:, 0], "tsne_2": E[:, 1],
                      "session": [names[i] for i in sess[idx]],
                      "frame": frame[idx]}).to_parquet(out / "tsne.parquet")
        # raw values of the embedded frames, row-aligned with tsne.parquet:
        # what mufasa-landscape-animate morphs between
        X[cols].iloc[idx].reset_index(drop=True).to_parquet(out / "tsne_samples.parquet")
        counts = pd.concat([top.feature_a, top.feature_b]).value_counts()
        plot_tsne(E, sess[idx], names, X[cols].iloc[idx].reset_index(drop=True),
                  list(counts.index[:5]), out)
        print("[landscape] wrote tsne.png, tsne.parquet and tsne_samples.parquet")
        results["tsne_frames"] = int(len(idx))

    write_run_toml(out / "run.toml", stage="feature_landscape", run_id=run_id,
                   inputs=[str(s.pose_path) for s in sessions if s.pose_path],
                   params={k: v for k, v in vars(a).items() if v is not None},
                   results=results)
    print(f"[landscape] done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
