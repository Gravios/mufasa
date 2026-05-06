"""Diagnostic tool for the Kalman pose smoother design.

Stage 0 in docs/kalman_smoother_design.md §5.1. Loads a pose
CSV (DLC, SLEAP, or generic) and produces an 8-component report
that confirms whether the data has the characteristics that
justify building the proposed Kalman + velocity-conditional
triplet covariance smoother.

Eight components produced:

  1. Likelihood histograms — one per marker plus aggregated.
     If most p > 0.95, a variance-aware smoother adds little.

  2. Run-length distribution of low-likelihood frames per
     marker. Long runs (>1 second of dropout) are exactly
     where Kalman + spatial prior helps most.

  3. Inter-marker distance distributions for "rigid" pairs.
     Tight distributions confirm the rigidity assumption.

  4. Head-frame egocentric velocity distribution. Multi-modal
     or wide → velocity-conditional Σ justified.

  5. Inter-marker distance vs HEAD forward velocity scatter.
     Tight band varying with velocity → conditioning helps.

  6. Body-frame egocentric velocity distribution. Body frame =
     centroid of body markers + PCA major axis (sign disambiguated
     using head direction).

  7. Inter-marker distance vs BODY forward velocity scatter.

  8. Head-vs-body forward velocity correlation scatter. Tight
     diagonal → 2D conditioning is enough; fan-out → 4D
     conditioning (head + body) is justified.

Components 6-8 require ≥2 non-head markers; otherwise skipped
with a warning, and the report records that.

This is a STANDALONE diagnostic — does not depend on the
full Mufasa runtime (only pandas, numpy, matplotlib). Can run
outside a fully-set-up project environment, e.g. directly
from a CSV in any directory.

Usage:

    python -m mufasa.data_processors.kalman_diagnostic \\
        path/to/pose.csv \\
        --output-dir /tmp/diag/ \\
        --likelihood-threshold 0.95 \\
        --head-markers Nose,Ear_left,Ear_right \\
        --body-markers Center,LeftFlank,RightFlank,TailBase \\
        --rigid-pairs "Ear_left,Ear_right;Nose,Ear_left;Nose,Ear_right"

The input can also be a parquet file or a directory of files:

    # Single parquet file
    python -m mufasa.data_processors.kalman_diagnostic \\
        path/to/pose.parquet --output-dir /tmp/diag/

    # Directory — recursively scans for .parquet files (or .csv
    # if no parquets), treats each as a session, aggregates with
    # proper boundary handling so dropout runs and velocities
    # don't span across files
    python -m mufasa.data_processors.kalman_diagnostic \\
        path/to/csv/input_csv/ --output-dir /tmp/diag/

    # Explicit list of files
    python -m mufasa.data_processors.kalman_diagnostic \\
        session1.parquet session2.parquet session3.parquet \\
        --output-dir /tmp/diag/

Or as a Python API:

    from mufasa.data_processors.kalman_diagnostic import run_diagnostic
    report = run_diagnostic(
        csv_path="path/to/pose.csv",                        # single file
        # OR  csv_path="path/to/csv/input_csv/",            # directory
        # OR  csv_path=["sess1.parquet", "sess2.parquet"],  # list
        output_dir="/tmp/diag/",
    )

The output is a directory containing:
  - 5-8 PNG plots (depending on whether body markers are available)
  - A summary.json with key statistics + per-session breakdown
  - A recommendation.txt with build/don't-build/scope guidance
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Column-naming conventions. DLC/Mufasa use the suffix pattern
# `<bodypart>_x`, `<bodypart>_y`, `<bodypart>_p`. The diagnostic
# auto-detects this pattern.
SUFFIX_X = "_x"
SUFFIX_Y = "_y"
SUFFIX_P = "_p"


@dataclass
class MarkerStats:
    """Per-marker statistics for the diagnostic report."""
    name: str
    n_frames: int
    p_mean: float
    p_median: float
    p_q05: float       # 5th percentile
    p_q95: float       # 95th percentile
    frac_high: float   # fraction with p > likelihood_threshold
    frac_zero: float   # fraction with p == 0 (true dropout)
    longest_low_run: int       # longest run of low-likelihood frames
    n_low_runs: int            # count of low-likelihood runs
    median_low_run: float      # median low-likelihood run length


@dataclass
class DiagnosticReport:
    """Top-level report. Serializable to JSON for persistence."""
    csv_path: str
    n_frames: int
    n_markers: int
    likelihood_threshold: float
    head_markers: List[str]
    body_markers: List[str]
    rigid_pairs: List[Tuple[str, str]]
    per_marker: List[MarkerStats]
    rigid_pair_stats: List[dict]    # mean, std, cv of inter-marker distance
    head_velocity_stats: dict        # mean, std, modality
    body_velocity_stats: dict        # mean, std, modality + sign-disambig info
    velocity_modality: str           # head modality (kept for back-compat)
    head_body_correlation: float     # Pearson r of head_vx vs body_vx
    recommendation: str              # build/static-only/skip/diagnostic-failed
    per_session_summary: List[dict]  # one entry per session: name +
                                      # worst_frac_high + worst_longest_run
                                      # + worst_marker_for_each


# -------------------------------------------------------------------- #
# Loading
# -------------------------------------------------------------------- #

def detect_marker_columns(df: pd.DataFrame) -> List[str]:
    """Find marker base names from DLC/Mufasa-style column suffixes.

    Returns the list of marker names that have all three of
    `<name>_x`, `<name>_y`, `<name>_p`. Markers missing any of
    the three are skipped with a warning.
    """
    cols = [str(c).lower() for c in df.columns]
    candidates = {}
    for c in cols:
        for suffix in (SUFFIX_X, SUFFIX_Y, SUFFIX_P):
            if c.endswith(suffix):
                base = c[: -len(suffix)]
                candidates.setdefault(base, set()).add(suffix)
    complete = sorted(
        base for base, suffixes in candidates.items()
        if {SUFFIX_X, SUFFIX_Y, SUFFIX_P} <= suffixes
    )
    incomplete = sorted(
        base for base, suffixes in candidates.items()
        if not ({SUFFIX_X, SUFFIX_Y, SUFFIX_P} <= suffixes)
    )
    if incomplete:
        print(
            f"[diagnostic] Skipping incomplete markers (missing "
            f"_x/_y/_p): {incomplete}",
            file=sys.stderr,
        )
    return complete


def _load_csv_with_header_detection(csv_path: str) -> pd.DataFrame:
    """Load a CSV, autodetecting flat vs multi-row IMPORTED_POSE
    header. Returns a DataFrame with lowercased column names.

    Multi-row flattening rule: if the LAST header level already
    contains a complete marker name with suffix (e.g. ``nose_x``
    in SimBA's IMPORTED_POSE format where rows 0-1 are both the
    string ``IMPORTED_POSE``), use just the last level. Otherwise
    (DLC standard format where the last level is just ``x`` /
    ``y`` / ``likelihood``), prepend the bodypart from the
    second-to-last level: ``nose`` + ``x`` → ``nose_x``.

    This matches what csv_to_parquet does on the way to parquet
    (it uses ``df.columns.get_level_values(-1)`` plus DLC suffix
    handling), so a CSV and a parquet derived from it produce
    the same column names downstream.
    """
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]
    if detect_marker_columns(df):
        return df

    # No markers found in flat read — try DLC-style 3-row multi-index
    df_multi = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    new_cols = []
    for col in df_multi.columns:
        if isinstance(col, tuple) and len(col) >= 2:
            last = str(col[-1]).lower()
            if last.endswith(("_x", "_y", "_p")):
                # SimBA IMPORTED_POSE: last level already complete
                # (e.g. "nose_x"). Use directly.
                new_cols.append(last)
            else:
                # DLC standard: last level is just "x"/"y"/"likelihood".
                # Prepend bodypart and normalize "likelihood" → "p".
                if last in ("likelihood", "lik"):
                    last = "p"
                new_cols.append(f"{str(col[-2]).lower()}_{last}")
        else:
            new_cols.append(str(col).lower())
    df_multi.columns = new_cols
    df_multi = df_multi.apply(pd.to_numeric, errors="coerce")
    return df_multi


def load_pose_file(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load a single pose file (CSV or parquet) into a flat-column
    DataFrame. Returns (df, list_of_markers).

    Format detection is by file extension. CSV path goes through
    multi-row IMPORTED_POSE header detection; parquet path is a
    direct read (parquet schemas already carry flat column names
    from csv_to_parquet's flattening step).
    """
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
        df.columns = [str(c).lower() for c in df.columns]
    elif ext in (".csv", ".tsv"):
        df = _load_csv_with_header_detection(path)
    else:
        raise ValueError(
            f"Unsupported file extension {ext!r} for {path}; "
            f"expected .csv, .tsv, or .parquet"
        )

    markers = detect_marker_columns(df)
    if not markers:
        raise ValueError(
            f"Could not detect marker columns in {path}. "
            f"Expected pattern <name>_x, <name>_y, <name>_p."
        )
    return df, markers


# Backward-compatible alias (callers used to call load_pose_csv).
def load_pose_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Backward-compatible alias of load_pose_file."""
    return load_pose_file(csv_path)


def load_pose_files(
    paths: List[str],
) -> Tuple[pd.DataFrame, List[str], List[Tuple[str, int, int]]]:
    """Load multiple pose files and concatenate them.

    Files are loaded in the order given. Each file becomes one
    "session" with its own (start_idx, end_idx) range in the
    concatenated DataFrame. All files MUST agree on the marker
    set — any mismatch raises ValueError.

    Parameters
    ----------
    paths : list of str
        Files to load. Each must be readable by load_pose_file.

    Returns
    -------
    df : pd.DataFrame
        Concatenated DataFrame, index reset (0..N-1 across all
        sessions). Column order is taken from the first file.
    markers : list of str
        The shared marker list (validated consistent across files).
    sessions : list of (name, start_idx, end_idx)
        Half-open ranges [start_idx, end_idx) into the concatenated
        df for each session. ``name`` is the file basename without
        extension.
    """
    if not paths:
        raise ValueError("load_pose_files requires at least one path")

    dfs: List[pd.DataFrame] = []
    markers_first: Optional[List[str]] = None
    sessions: List[Tuple[str, int, int]] = []
    cursor = 0
    for p in paths:
        df_one, markers_one = load_pose_file(p)
        if markers_first is None:
            markers_first = markers_one
            # Use the first file's column order as canonical
            canonical_cols = list(df_one.columns)
        else:
            if set(markers_one) != set(markers_first):
                missing = set(markers_first) - set(markers_one)
                extra = set(markers_one) - set(markers_first)
                raise ValueError(
                    f"Marker mismatch in {p}: missing={sorted(missing)}, "
                    f"extra={sorted(extra)}. All files must share the "
                    f"same marker set."
                )
            # Reorder columns to match the canonical layout
            df_one = df_one[canonical_cols]
        n = len(df_one)
        name = Path(p).stem
        sessions.append((name, cursor, cursor + n))
        cursor += n
        dfs.append(df_one)

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined, list(markers_first), sessions


def discover_pose_files(root: str) -> List[str]:
    """Find pose-data files (parquet preferred, then csv) under
    ``root`` recursively. Skips files that look like outputs
    (``.pose.<ext>``) or hidden files."""
    root_path = Path(root)
    if root_path.is_file():
        return [str(root_path)]
    if not root_path.is_dir():
        raise ValueError(f"{root} is not a file or directory")

    parquet_files = sorted(
        str(p) for p in root_path.rglob("*.parquet")
        if not _is_hidden(p) and ".pose." not in p.name
    )
    if parquet_files:
        return parquet_files
    csv_files = sorted(
        str(p) for p in root_path.rglob("*.csv")
        if not _is_hidden(p) and ".pose." not in p.name
    )
    return csv_files


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


# -------------------------------------------------------------------- #
# Component 1: Likelihood histograms
# -------------------------------------------------------------------- #

def compute_marker_stats(
    df: pd.DataFrame,
    marker: str,
    likelihood_threshold: float,
    session_ranges: Optional[List[Tuple[str, int, int]]] = None,
) -> MarkerStats:
    """Compute per-marker likelihood + dropout statistics.

    When ``session_ranges`` is provided, the run-length encoding
    is computed within each session independently — a low-p run
    cannot span across a session boundary. This is important for
    multi-file diagnostic runs where each file is one session.
    """
    p_col = f"{marker}{SUFFIX_P}"
    p = df[p_col].values
    n = len(p)
    is_low = p < likelihood_threshold

    # Run-length encoding of low-likelihood runs, respecting
    # session boundaries.
    if session_ranges is None:
        slice_iter = [(0, n)]
    else:
        slice_iter = [(start, end) for _, start, end in session_ranges]

    runs = []
    for start, end in slice_iter:
        current_run = 0
        for v in is_low[start:end]:
            if v:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)

    return MarkerStats(
        name=marker,
        n_frames=int(n),
        p_mean=float(np.mean(p)),
        p_median=float(np.median(p)),
        p_q05=float(np.quantile(p, 0.05)),
        p_q95=float(np.quantile(p, 0.95)),
        frac_high=float(np.mean(p >= likelihood_threshold)),
        frac_zero=float(np.mean(p == 0.0)),
        longest_low_run=int(max(runs)) if runs else 0,
        n_low_runs=len(runs),
        median_low_run=float(np.median(runs)) if runs else 0.0,
    )


# -------------------------------------------------------------------- #
# Component 3: Inter-marker distance for rigid pairs
# -------------------------------------------------------------------- #

def compute_rigid_pair_stats(
    df: pd.DataFrame,
    marker_a: str,
    marker_b: str,
    likelihood_threshold: float,
) -> dict:
    """For one designated-rigid marker pair, compute inter-marker
    distance distribution restricted to frames where both markers
    are high-likelihood. Returns mean, std, and coefficient of
    variation. CV close to 0 supports the rigidity assumption.
    """
    pa = df[f"{marker_a}{SUFFIX_P}"].values
    pb = df[f"{marker_b}{SUFFIX_P}"].values
    mask = (pa >= likelihood_threshold) & (pb >= likelihood_threshold)
    if mask.sum() < 100:
        return {
            "marker_a": marker_a,
            "marker_b": marker_b,
            "n_high_confidence": int(mask.sum()),
            "warning": "Too few high-confidence frames for stable estimate",
        }
    dx = df[f"{marker_a}{SUFFIX_X}"].values[mask] - df[f"{marker_b}{SUFFIX_X}"].values[mask]
    dy = df[f"{marker_a}{SUFFIX_Y}"].values[mask] - df[f"{marker_b}{SUFFIX_Y}"].values[mask]
    d = np.sqrt(dx * dx + dy * dy)
    mean_d = float(np.mean(d))
    return {
        "marker_a": marker_a,
        "marker_b": marker_b,
        "n_high_confidence": int(mask.sum()),
        "mean_distance": mean_d,
        "std_distance": float(np.std(d)),
        "cv_distance": float(np.std(d) / mean_d) if mean_d > 0 else 0.0,
        "min_distance": float(np.min(d)),
        "max_distance": float(np.max(d)),
        "p05_distance": float(np.quantile(d, 0.05)),
        "p95_distance": float(np.quantile(d, 0.95)),
    }


def auto_detect_rigid_pairs(
    df: pd.DataFrame,
    markers: List[str],
    likelihood_threshold: float,
    cv_threshold: float = 0.20,
    max_pairs: int = 8,
    min_samples: int = 200,
    exclude_markers: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Identify empirically-rigid marker pairs from the data.

    Iterates over all C(n,2) marker pairs (excluding any pair
    that contains a marker in ``exclude_markers``), computes
    the coefficient of variation of inter-marker distance over
    frames where both markers are above the likelihood
    threshold, and returns pairs whose CV is below
    ``cv_threshold`` — sorted ascending by CV — up to
    ``max_pairs``.

    The ``exclude_markers`` parameter exists for the v1 smoother
    architecture: head markers (nose, ear_left, ear_right,
    headmid) are excluded from the BODY-rigid pair detection
    because head pose breaks 2D rigidity assumptions in this
    camera projection (rearing changes ear-distance, nose
    flexes during sniffing). The head is treated separately —
    in v1 with no head triplet, in v2 with a posture-conditional
    head triplet driven by a latent posture variable.

    This is data-driven rigidity detection: it finds pairs
    that ARE empirically tight in the project's data, which
    may or may not match anatomical expectations.

    Pairs with insufficient joint-high-likelihood samples
    (< ``min_samples``) are skipped — the CV would be
    statistically meaningless.

    Returns
    -------
    List of (marker_a, marker_b) tuples, in ascending-CV
    order. Empty list if no pairs qualify.
    """
    excl = set(exclude_markers or [])
    candidates = []
    for i, a in enumerate(markers):
        if a in excl:
            continue
        for b in markers[i + 1:]:
            if b in excl:
                continue
            stats = compute_rigid_pair_stats(
                df, a, b, likelihood_threshold,
            )
            if "warning" in stats:
                continue  # too few samples
            if stats.get("n_high_confidence", 0) < min_samples:
                continue
            cv = stats.get("cv_distance", float("inf"))
            if cv < cv_threshold:
                candidates.append((cv, a, b))

    candidates.sort(key=lambda x: x[0])  # ascending CV
    return [(a, b) for cv, a, b in candidates[:max_pairs]]


def auto_detect_candidate_triplets(
    df: pd.DataFrame,
    markers: List[str],
    likelihood_threshold: float,
    cv_threshold: float = 0.20,
    max_triplets: int = 8,
    min_samples: int = 200,
    exclude_markers: Optional[List[str]] = None,
) -> List[Tuple[Tuple[str, str, str], dict]]:
    """Identify empirically-rigid marker triplets from the data.

    A triplet (a, b, c) is a candidate if all three pairwise
    CVs are below ``cv_threshold``. The triplet is the natural
    unit for the v1 static-Σ triplet prior in the Kalman pose
    smoother — Σ is a 6×6 covariance matrix over the (a, b, c)
    coordinates, which the smoother estimates empirically from
    high-confidence frames.

    Pairwise CV is a screening heuristic: it identifies
    candidate triplets cheaply but the full Σ estimation
    happens at smoother-fit time. We don't require the three
    pairwise distributions to be near-Gaussian; we just want
    the triplet's joint configuration to be sufficiently stable
    that a single Σ describes it well.

    Returns
    -------
    List of ((marker_a, marker_b, marker_c), info_dict) tuples,
    sorted by mean of the three pairwise CVs (ascending).
    info_dict contains:
        - ``cv_ab``, ``cv_ac``, ``cv_bc``: pairwise CVs
        - ``cv_mean``: mean of the three
        - ``cv_max``: worst (loosest) of the three
        - ``n_samples``: minimum joint high-confidence count
          across the three pairs (the binding constraint)

    Capped at ``max_triplets`` for output stability. Empty
    list if no triplet qualifies.
    """
    excl = set(exclude_markers or [])
    eligible = [m for m in markers if m not in excl]
    if len(eligible) < 3:
        return []

    # Cache pairwise CVs so we don't recompute for every triplet
    pair_cv: dict = {}
    pair_n: dict = {}
    for i, a in enumerate(eligible):
        for b in eligible[i + 1:]:
            stats = compute_rigid_pair_stats(
                df, a, b, likelihood_threshold,
            )
            if "warning" in stats:
                pair_cv[(a, b)] = float("inf")
                pair_n[(a, b)] = 0
                continue
            n = stats.get("n_high_confidence", 0)
            if n < min_samples:
                pair_cv[(a, b)] = float("inf")
                pair_n[(a, b)] = n
                continue
            pair_cv[(a, b)] = stats.get("cv_distance", float("inf"))
            pair_n[(a, b)] = n

    def _cv(a, b):
        # Symmetric lookup
        if (a, b) in pair_cv:
            return pair_cv[(a, b)]
        return pair_cv.get((b, a), float("inf"))

    def _n(a, b):
        if (a, b) in pair_n:
            return pair_n[(a, b)]
        return pair_n.get((b, a), 0)

    candidates = []
    for i, a in enumerate(eligible):
        for j in range(i + 1, len(eligible)):
            b = eligible[j]
            for k in range(j + 1, len(eligible)):
                c = eligible[k]
                cv_ab = _cv(a, b)
                cv_ac = _cv(a, c)
                cv_bc = _cv(b, c)
                if max(cv_ab, cv_ac, cv_bc) >= cv_threshold:
                    continue
                cv_mean = (cv_ab + cv_ac + cv_bc) / 3.0
                cv_max = max(cv_ab, cv_ac, cv_bc)
                n_samples = min(_n(a, b), _n(a, c), _n(b, c))
                candidates.append((
                    cv_mean,
                    (a, b, c),
                    {
                        "cv_ab": cv_ab,
                        "cv_ac": cv_ac,
                        "cv_bc": cv_bc,
                        "cv_mean": cv_mean,
                        "cv_max": cv_max,
                        "n_samples": int(n_samples),
                    },
                ))

    candidates.sort(key=lambda x: x[0])  # ascending mean CV
    return [(triplet, info) for _, triplet, info in candidates[:max_triplets]]


def compute_behavioral_signal_pairs(
    df: pd.DataFrame,
    head_markers: List[str],
    likelihood_threshold: float,
) -> List[dict]:
    """Compute distance distributions for head-internal pairs as
    BEHAVIORAL SIGNAL channels (not rigid constraints).

    In a 2D camera projection, head-internal distances vary
    meaningfully with head pose (rearing, head tilt, sniffing).
    These pairs are NOT rigid in the smoother sense — using them
    as rigidity constraints would flatten behaviorally relevant
    signal — but the same distances are *useful features* for
    behavior classification (e.g. ear-distance as a rearing
    proxy: ears farther apart when head is level, closer when
    head is up/down).

    This function computes the distance distribution for each
    head-internal pair, returning the same stats dict shape as
    compute_rigid_pair_stats but with a ``role`` key set to
    ``"behavioral_signal"`` so downstream consumers can route
    appropriately. The diagnostic surfaces these in a separate
    section of summary.json from the rigid pairs.

    Parameters
    ----------
    head_markers : list of str
        Markers comprising the head (typically nose, ear_left,
        ear_right, headmid). All C(n,2) pairs among these are
        characterized.

    Returns
    -------
    List of dicts, one per head-internal pair, with the
    standard rigid-pair-stats fields plus
    ``"role": "behavioral_signal"``. Empty if fewer than 2
    head markers.
    """
    if len(head_markers) < 2:
        return []
    out = []
    for i, a in enumerate(head_markers):
        for b in head_markers[i + 1:]:
            stats = compute_rigid_pair_stats(
                df, a, b, likelihood_threshold,
            )
            stats["role"] = "behavioral_signal"
            out.append(stats)
    return out


# -------------------------------------------------------------------- #
# Component 4: Head-frame velocity
# -------------------------------------------------------------------- #

def compute_head_velocity(
    df: pd.DataFrame,
    head_markers: List[str],
    likelihood_threshold: float,
    fps: float,
    session_ranges: Optional[List[Tuple[str, int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute head-frame egocentric velocity for all frames.

    When ``session_ranges`` is provided, velocity is computed
    per-session and stitched — preventing np.gradient from
    crossing session boundaries (which would produce spurious
    velocities at file joins) and preventing the valid_prev-frame
    check from spanning sessions.

    Returns:
      vx_h, vy_h: velocity in head frame, shape (T,). NaN where
        the head triplet wasn't reliable.
      valid_mask: boolean shape (T,) — True where velocity is
        reliably computable (head triplet high-p at t and t-1).
    """
    if session_ranges is not None:
        vx_h_full = np.full(len(df), np.nan)
        vy_h_full = np.full(len(df), np.nan)
        valid_full = np.zeros(len(df), dtype=bool)
        for _, start, end in session_ranges:
            sub_df = df.iloc[start:end].reset_index(drop=True)
            vx, vy, valid = compute_head_velocity(
                sub_df, head_markers, likelihood_threshold, fps,
                session_ranges=None,
            )
            vx_h_full[start:end] = vx
            vy_h_full[start:end] = vy
            valid_full[start:end] = valid
        return vx_h_full, vy_h_full, valid_full

    if len(head_markers) < 2:
        raise ValueError(
            f"Need ≥2 head markers to define a head frame; "
            f"got {head_markers}"
        )
    # Head centroid per frame (over all head markers)
    cx = np.mean([df[f"{m}{SUFFIX_X}"].values for m in head_markers], axis=0)
    cy = np.mean([df[f"{m}{SUFFIX_Y}"].values for m in head_markers], axis=0)
    # Head heading: from second marker toward first marker
    # (typical: ear_midpoint → nose). Generalizable: use the
    # PCA major axis of all head markers per frame, but the
    # simple two-marker version is adequate for diagnostic.
    if len(head_markers) >= 3:
        # Use first marker (typically nose) minus mean of the rest
        target = np.array([
            df[f"{head_markers[0]}{SUFFIX_X}"].values,
            df[f"{head_markers[0]}{SUFFIX_Y}"].values,
        ])
        rest_x = np.mean(
            [df[f"{m}{SUFFIX_X}"].values for m in head_markers[1:]], axis=0
        )
        rest_y = np.mean(
            [df[f"{m}{SUFFIX_Y}"].values for m in head_markers[1:]], axis=0
        )
        hx = target[0] - rest_x
        hy = target[1] - rest_y
    else:
        # Two markers only: from second to first
        hx = df[f"{head_markers[0]}{SUFFIX_X}"].values - df[f"{head_markers[1]}{SUFFIX_X}"].values
        hy = df[f"{head_markers[0]}{SUFFIX_Y}"].values - df[f"{head_markers[1]}{SUFFIX_Y}"].values

    theta = np.arctan2(hy, hx)

    # World-frame velocity from centroid
    vx_world = np.gradient(cx) * fps
    vy_world = np.gradient(cy) * fps

    # Rotate into head frame: v_head = R(-theta) · v_world
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    vx_h = cos_t * vx_world - sin_t * vy_world
    vy_h = sin_t * vx_world + cos_t * vy_world

    # Valid where all head markers are reliable at t and t-1
    valid_now = np.ones(len(df), dtype=bool)
    for m in head_markers:
        valid_now &= df[f"{m}{SUFFIX_P}"].values >= likelihood_threshold
    valid_prev = np.concatenate([[False], valid_now[:-1]])
    valid_mask = valid_now & valid_prev

    return vx_h, vy_h, valid_mask


# -------------------------------------------------------------------- #
# Component 4b: Body-frame velocity (option b — PCA major axis)
# -------------------------------------------------------------------- #

def _compute_head_direction(
    df: pd.DataFrame,
    head_markers: List[str],
    likelihood_threshold: float,
) -> Optional[np.ndarray]:
    """Compute unit head direction vector per frame.

    Returns shape (T, 2) with NaN rows where head markers are
    unreliable. Returns None if head_markers can't define a
    direction (fewer than 2 markers).
    """
    if len(head_markers) < 2:
        return None
    if len(head_markers) >= 3:
        target_x = df[f"{head_markers[0]}{SUFFIX_X}"].values
        target_y = df[f"{head_markers[0]}{SUFFIX_Y}"].values
        rest_x = np.mean(
            [df[f"{m}{SUFFIX_X}"].values for m in head_markers[1:]], axis=0
        )
        rest_y = np.mean(
            [df[f"{m}{SUFFIX_Y}"].values for m in head_markers[1:]], axis=0
        )
        hx = target_x - rest_x
        hy = target_y - rest_y
    else:
        hx = (df[f"{head_markers[0]}{SUFFIX_X}"].values
              - df[f"{head_markers[1]}{SUFFIX_X}"].values)
        hy = (df[f"{head_markers[0]}{SUFFIX_Y}"].values
              - df[f"{head_markers[1]}{SUFFIX_Y}"].values)
    norms = np.sqrt(hx * hx + hy * hy)
    safe_norms = np.where(norms > 1e-9, norms, 1.0)
    head_dir = np.column_stack([hx / safe_norms, hy / safe_norms])

    # NaN out frames where head markers aren't all high-confidence
    head_p = np.column_stack(
        [df[f"{m}{SUFFIX_P}"].values for m in head_markers]
    )
    head_reliable = np.all(head_p >= likelihood_threshold, axis=1)
    head_dir[~head_reliable] = np.nan
    # Also NaN out frames where the head direction was numerically zero
    head_dir[norms <= 1e-9] = np.nan
    return head_dir


def _compute_body_axis_per_frame(
    body_positions: np.ndarray,        # shape (T, n_body, 2)
    head_directions: Optional[np.ndarray],  # shape (T, 2) or None
    valid_body: np.ndarray,            # shape (T,) bool
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute body major axis + centroid per frame via PCA.

    Body major axis: leading eigenvector of the 2×2 covariance of
    centered body marker positions per frame. Sign disambiguated
    so the axis points toward the head (when head_directions is
    provided and reliable at frame t) or in the direction of the
    previous frame's body axis (continuity fallback).

    Returns
    -------
    body_axis : (T, 2)
        Unit vector of body major axis, NaN where invalid.
    body_centroid : (T, 2)
        Centroid of body markers, NaN where invalid.
    diagnostics : dict
        Counters for sign-disambiguation outcomes (how many frames
        used head-direction reference vs continuity vs neither).
    """
    T = body_positions.shape[0]
    body_axis = np.full((T, 2), np.nan)
    body_centroid = np.full((T, 2), np.nan)
    n_signed_by_head = 0
    n_signed_by_continuity = 0
    n_signed_arbitrary = 0
    n_degenerate = 0  # nearly-isotropic body (eigvals roughly equal)

    last_axis = None  # carry forward for continuity fallback
    for t in range(T):
        if not valid_body[t]:
            last_axis = None  # break continuity if data goes invalid
            continue
        positions = body_positions[t]
        if np.any(np.isnan(positions)):
            last_axis = None
            continue
        c = positions.mean(axis=0)
        body_centroid[t] = c
        centered = positions - c
        cov = centered.T @ centered  # 2x2
        # eigh returns eigenvalues ascending; major axis is the last col
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            last_axis = None
            continue
        # Detect near-isotropic body (eigenvalues roughly equal)
        # Threshold: minor/major < 0.7 means well-defined major axis
        if eigvals[1] < 1e-9:
            n_degenerate += 1
            last_axis = None
            continue
        ratio = eigvals[0] / eigvals[1]
        if ratio > 0.7:
            n_degenerate += 1
        major_axis = eigvecs[:, -1].copy()

        # Sign disambiguation
        signed_by = "arbitrary"
        if head_directions is not None and not np.any(
            np.isnan(head_directions[t])
        ):
            if np.dot(major_axis, head_directions[t]) < 0:
                major_axis = -major_axis
            signed_by = "head"
        elif last_axis is not None:
            if np.dot(major_axis, last_axis) < 0:
                major_axis = -major_axis
            signed_by = "continuity"

        if signed_by == "head":
            n_signed_by_head += 1
        elif signed_by == "continuity":
            n_signed_by_continuity += 1
        else:
            n_signed_arbitrary += 1

        body_axis[t] = major_axis
        last_axis = major_axis

    diagnostics = {
        "n_signed_by_head": n_signed_by_head,
        "n_signed_by_continuity": n_signed_by_continuity,
        "n_signed_arbitrary": n_signed_arbitrary,
        "n_degenerate_body_shape": n_degenerate,
    }
    return body_axis, body_centroid, diagnostics


def compute_body_velocity(
    df: pd.DataFrame,
    body_markers: List[str],
    head_markers: Optional[List[str]],
    likelihood_threshold: float,
    fps: float,
    session_ranges: Optional[List[Tuple[str, int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute body-frame egocentric velocity for all frames.

    Body frame: centroid of body markers + PCA major axis on body
    marker positions (option b in design doc). Sign disambiguated
    using head direction (preferred) or temporal continuity
    (fallback).

    When ``session_ranges`` is provided, velocity is computed
    per-session and stitched. Sign-disambiguation diagnostic
    counters are summed across sessions.

    Parameters
    ----------
    body_markers : list of str
        Markers used for body centroid + PCA. Must have ≥2.
    head_markers : list of str or None
        Markers used to disambiguate body-axis sign. If None or
        fewer than 2, sign is disambiguated by continuity only
        (which can leave the first valid run with arbitrary sign).

    Returns
    -------
    vx_b, vy_b : (T,)
        Velocity in body frame. NaN where unreliable.
    valid_mask : (T,) bool
        True where velocity is reliably computable (body axis
        defined at t and t-1, body centroid valid).
    diagnostics : dict
        Sign-disambiguation counters from
        _compute_body_axis_per_frame (summed across sessions
        when ``session_ranges`` is provided).
    """
    if session_ranges is not None:
        vx_b_full = np.full(len(df), np.nan)
        vy_b_full = np.full(len(df), np.nan)
        valid_full = np.zeros(len(df), dtype=bool)
        agg_diag = {
            "n_signed_by_head": 0,
            "n_signed_by_continuity": 0,
            "n_signed_arbitrary": 0,
            "n_degenerate_body_shape": 0,
        }
        for _, start, end in session_ranges:
            sub_df = df.iloc[start:end].reset_index(drop=True)
            vx, vy, valid, diag = compute_body_velocity(
                sub_df, body_markers, head_markers,
                likelihood_threshold, fps,
                session_ranges=None,
            )
            vx_b_full[start:end] = vx
            vy_b_full[start:end] = vy
            valid_full[start:end] = valid
            for k in agg_diag:
                agg_diag[k] += diag.get(k, 0)
        return vx_b_full, vy_b_full, valid_full, agg_diag

    if len(body_markers) < 2:
        raise ValueError(
            f"Need ≥2 body markers for PCA-based body frame; "
            f"got {body_markers}"
        )

    body_x = np.column_stack(
        [df[f"{m}{SUFFIX_X}"].values for m in body_markers]
    )
    body_y = np.column_stack(
        [df[f"{m}{SUFFIX_Y}"].values for m in body_markers]
    )
    # Stack into (T, n_body, 2)
    body_positions = np.stack([body_x, body_y], axis=2)

    body_p = np.column_stack(
        [df[f"{m}{SUFFIX_P}"].values for m in body_markers]
    )
    valid_body_now = np.all(body_p >= likelihood_threshold, axis=1)

    head_directions = (
        _compute_head_direction(df, head_markers, likelihood_threshold)
        if head_markers else None
    )

    body_axis, body_centroid, diagnostics = _compute_body_axis_per_frame(
        body_positions, head_directions, valid_body_now
    )

    theta = np.arctan2(body_axis[:, 1], body_axis[:, 0])
    cx = body_centroid[:, 0]
    cy = body_centroid[:, 1]
    vx_world = np.gradient(cx) * fps
    vy_world = np.gradient(cy) * fps

    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    vx_b = cos_t * vx_world - sin_t * vy_world
    vy_b = sin_t * vx_world + cos_t * vy_world

    # Reliable velocity needs current and prior frame both valid
    valid_axis = ~np.isnan(theta)
    valid_prev = np.concatenate([[False], valid_axis[:-1]])
    # Also require the computed velocity itself to be finite.
    # np.gradient at frame t looks at t-1 and t+1; if t+1's body
    # is degenerate (NaN centroid), vx_world[t] becomes NaN even
    # though theta[t] and theta[t-1] were both finite. Without
    # this guard, vx_b[t] is NaN but valid_mask[t] is True.
    valid_velocity = np.isfinite(vx_b) & np.isfinite(vy_b)
    valid_mask = valid_body_now & valid_axis & valid_prev & valid_velocity

    return vx_b, vy_b, valid_mask, diagnostics


def head_body_velocity_correlation(
    vx_h: np.ndarray,
    vx_b: np.ndarray,
    head_valid: np.ndarray,
    body_valid: np.ndarray,
) -> float:
    """Pearson correlation between head and body forward velocities
    over frames where both are reliably computable. Returns NaN if
    insufficient overlap.
    """
    joint = head_valid & body_valid
    if joint.sum() < 30:
        return float("nan")
    xh = vx_h[joint]
    xb = vx_b[joint]
    # Defensive: drop any frame where either velocity is non-finite.
    # The valid_mask contract should already exclude these but be
    # robust to unexpected NaN leakage from upstream computation.
    finite = np.isfinite(xh) & np.isfinite(xb)
    if finite.sum() < 30:
        return float("nan")
    xh = xh[finite]
    xb = xb[finite]
    # Avoid divide-by-zero on a degenerate constant series
    if np.std(xh) < 1e-9 or np.std(xb) < 1e-9:
        return float("nan")
    return float(np.corrcoef(xh, xb)[0, 1])


# -------------------------------------------------------------------- #
# Plotting
# -------------------------------------------------------------------- #

def _plot_likelihood_histograms(
    df: pd.DataFrame,
    markers: List[str],
    output_path: Path,
) -> None:
    """Component 1 plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(markers)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = np.atleast_1d(axes).flatten()
    for i, marker in enumerate(markers):
        ax = axes[i]
        p = df[f"{marker}{SUFFIX_P}"].values
        ax.hist(p, bins=50, range=(0, 1), color="steelblue", alpha=0.8)
        ax.set_title(marker, fontsize=9)
        ax.set_xlim(0, 1)
        ax.axvline(0.95, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(
        f"Likelihood histograms (red dashed: 0.95 threshold)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _plot_run_lengths(
    per_marker: List[MarkerStats],
    output_path: Path,
) -> None:
    """Component 2 plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s.name for s in per_marker]
    longest = [s.longest_low_run for s in per_marker]
    n_runs = [s.n_low_runs for s in per_marker]
    median_run = [s.median_low_run for s in per_marker]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, 0.4 * len(names)), 6))
    x = np.arange(len(names))
    ax1.bar(x, longest, color="firebrick", alpha=0.8)
    ax1.set_ylabel("Longest low-p run (frames)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_title("Worst dropout per marker")
    ax2.bar(x, n_runs, color="darkorange", alpha=0.8)
    ax2.set_ylabel("Number of low-p runs")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_title("Total low-p runs per marker")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _plot_rigid_pairs(
    rigid_pair_stats: List[dict],
    df: pd.DataFrame,
    likelihood_threshold: float,
    output_path: Path,
) -> None:
    """Component 3 plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [s for s in rigid_pair_stats if "warning" not in s]
    if not valid:
        # Empty plot with a message
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5,
            "No rigid pairs had enough high-confidence frames",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return

    n = len(valid)
    fig, axes = plt.subplots(
        1, n,
        figsize=(max(4 * n, 6), 3),
        squeeze=False,
    )
    for i, stats in enumerate(valid):
        ax = axes[0, i]
        marker_a, marker_b = stats["marker_a"], stats["marker_b"]
        pa = df[f"{marker_a}{SUFFIX_P}"].values
        pb = df[f"{marker_b}{SUFFIX_P}"].values
        mask = (pa >= likelihood_threshold) & (pb >= likelihood_threshold)
        dx = df[f"{marker_a}{SUFFIX_X}"].values[mask] - df[f"{marker_b}{SUFFIX_X}"].values[mask]
        dy = df[f"{marker_a}{SUFFIX_Y}"].values[mask] - df[f"{marker_b}{SUFFIX_Y}"].values[mask]
        d = np.sqrt(dx * dx + dy * dy)
        ax.hist(d, bins=50, color="seagreen", alpha=0.8)
        ax.set_title(
            f"{marker_a} ↔ {marker_b}\n"
            f"CV={stats['cv_distance']:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("inter-marker distance (px)")
        ax.set_ylabel("count")
    fig.suptitle("Inter-marker distances (rigid-pair check)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _plot_velocity_distribution(
    vx_h: np.ndarray,
    vy_h: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Component 4 plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if valid_mask.sum() < 100:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5,
            "Too few reliable head-velocity samples for diagnostic",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return

    vx = vx_h[valid_mask]
    vy = vy_h[valid_mask]
    speed = np.sqrt(vx * vx + vy * vy)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    # Forward velocity (vx_h > 0 = head-direction motion)
    axes[0].hist(vx, bins=80, color="steelblue", alpha=0.8)
    axes[0].set_title("Forward velocity (head frame)")
    axes[0].set_xlabel("vx_h (px/s)")
    axes[0].axvline(0, color="black", linewidth=0.5)
    # Lateral velocity
    axes[1].hist(vy, bins=80, color="indianred", alpha=0.8)
    axes[1].set_title("Lateral velocity (head frame)")
    axes[1].set_xlabel("vy_h (px/s)")
    axes[1].axvline(0, color="black", linewidth=0.5)
    # Speed
    axes[2].hist(speed, bins=80, color="darkviolet", alpha=0.8)
    axes[2].set_title("Total speed")
    axes[2].set_xlabel("|v| (px/s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _plot_velocity_vs_configuration(
    df: pd.DataFrame,
    rigid_pair_stats: List[dict],
    vx_h: np.ndarray,
    valid_mask: np.ndarray,
    likelihood_threshold: float,
    output_path: Path,
) -> None:
    """Component 5: scatter of inter-marker distance vs forward velocity.

    If the scatter shows a tight band that varies systematically
    with velocity, the velocity-conditional Σ is well-justified.
    A uniform cloud means static Σ is sufficient.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_pairs = [s for s in rigid_pair_stats if "warning" not in s]
    if not valid_pairs or valid_mask.sum() < 100:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5,
            "Too few samples for velocity-vs-configuration analysis",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return

    n = len(valid_pairs)
    fig, axes = plt.subplots(
        1, n,
        figsize=(max(4 * n, 6), 3.5),
        squeeze=False,
    )
    for i, stats in enumerate(valid_pairs):
        ax = axes[0, i]
        marker_a, marker_b = stats["marker_a"], stats["marker_b"]
        pa = df[f"{marker_a}{SUFFIX_P}"].values
        pb = df[f"{marker_b}{SUFFIX_P}"].values
        joint_mask = (
            (pa >= likelihood_threshold)
            & (pb >= likelihood_threshold)
            & valid_mask
        )
        if joint_mask.sum() < 50:
            ax.text(
                0.5, 0.5,
                f"{marker_a}↔{marker_b}\ntoo few joint samples",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        dx = df[f"{marker_a}{SUFFIX_X}"].values[joint_mask] - df[f"{marker_b}{SUFFIX_X}"].values[joint_mask]
        dy = df[f"{marker_a}{SUFFIX_Y}"].values[joint_mask] - df[f"{marker_b}{SUFFIX_Y}"].values[joint_mask]
        d = np.sqrt(dx * dx + dy * dy)
        v = vx_h[joint_mask]
        ax.scatter(v, d, s=2, alpha=0.3, color="purple")
        ax.set_title(f"{marker_a} ↔ {marker_b}", fontsize=9)
        ax.set_xlabel("vx_h (px/s)")
        ax.set_ylabel("inter-marker dist (px)")
    fig.suptitle(
        "Configuration vs forward velocity\n"
        "(systematic variation → velocity-conditional Σ is justified)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _plot_body_velocity_distribution(
    vx_b: np.ndarray,
    vy_b: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Component 4b plot: body-frame velocity distribution.

    Mirror of head velocity plot but for body frame (centroid +
    PCA major axis).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if valid_mask.sum() < 100:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5,
            "Too few reliable body-velocity samples for diagnostic",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return

    vx = vx_b[valid_mask]
    vy = vy_b[valid_mask]
    speed = np.sqrt(vx * vx + vy * vy)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].hist(vx, bins=80, color="steelblue", alpha=0.8)
    axes[0].set_title("Forward velocity (body frame)")
    axes[0].set_xlabel("vx_b (px/s)")
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[1].hist(vy, bins=80, color="indianred", alpha=0.8)
    axes[1].set_title("Lateral velocity (body frame)")
    axes[1].set_xlabel("vy_b (px/s)")
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[2].hist(speed, bins=80, color="darkviolet", alpha=0.8)
    axes[2].set_title("Total body speed")
    axes[2].set_xlabel("|v_b| (px/s)")
    fig.suptitle(
        "Body-frame velocity (PCA major axis on body markers)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _plot_velocity_vs_configuration_body(
    df: pd.DataFrame,
    rigid_pair_stats: List[dict],
    vx_b: np.ndarray,
    valid_mask: np.ndarray,
    likelihood_threshold: float,
    output_path: Path,
) -> None:
    """Component 7: scatter of inter-marker distance vs forward
    BODY velocity. Compare to component 5 (head velocity); if the
    patterns differ markedly, 4D conditioning is well-justified
    over 2D.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_pairs = [s for s in rigid_pair_stats if "warning" not in s]
    if not valid_pairs or valid_mask.sum() < 100:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5,
            "Too few samples for body-velocity vs configuration analysis",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return

    n = len(valid_pairs)
    fig, axes = plt.subplots(
        1, n,
        figsize=(max(4 * n, 6), 3.5),
        squeeze=False,
    )
    for i, stats in enumerate(valid_pairs):
        ax = axes[0, i]
        marker_a, marker_b = stats["marker_a"], stats["marker_b"]
        pa = df[f"{marker_a}{SUFFIX_P}"].values
        pb = df[f"{marker_b}{SUFFIX_P}"].values
        joint_mask = (
            (pa >= likelihood_threshold)
            & (pb >= likelihood_threshold)
            & valid_mask
        )
        if joint_mask.sum() < 50:
            ax.text(
                0.5, 0.5,
                f"{marker_a}↔{marker_b}\ntoo few joint samples",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        dx = (df[f"{marker_a}{SUFFIX_X}"].values[joint_mask]
              - df[f"{marker_b}{SUFFIX_X}"].values[joint_mask])
        dy = (df[f"{marker_a}{SUFFIX_Y}"].values[joint_mask]
              - df[f"{marker_b}{SUFFIX_Y}"].values[joint_mask])
        d = np.sqrt(dx * dx + dy * dy)
        v = vx_b[joint_mask]
        ax.scatter(v, d, s=2, alpha=0.3, color="darkgreen")
        ax.set_title(f"{marker_a} ↔ {marker_b}", fontsize=9)
        ax.set_xlabel("vx_b (px/s)")
        ax.set_ylabel("inter-marker dist (px)")
    fig.suptitle(
        "Configuration vs body forward velocity\n"
        "(compare to head-velocity plot; differing patterns → "
        "4D conditioning is justified)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _plot_head_body_velocity_correlation(
    vx_h: np.ndarray,
    vx_b: np.ndarray,
    head_valid: np.ndarray,
    body_valid: np.ndarray,
    output_path: Path,
) -> None:
    """Component 8: scatter of head_vx vs body_vx for frames where
    both are reliable. If they cluster on the y=x diagonal, head
    and body move together — body velocity adds little. If they
    fan out, the two are decoupled (e.g. scanning behavior) and
    the 4D conditioning captures real additional information.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joint = head_valid & body_valid
    if joint.sum() < 100:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5,
            "Too few joint head+body samples for correlation plot",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return

    xh = vx_h[joint]
    xb = vx_b[joint]
    r = float(np.corrcoef(xh, xb)[0, 1]) if (
        np.std(xh) > 1e-9 and np.std(xb) > 1e-9
    ) else float("nan")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xh, xb, s=2, alpha=0.3, color="navy")
    lim_min = min(np.min(xh), np.min(xb))
    lim_max = max(np.max(xh), np.max(xb))
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        color="red", linestyle="--", linewidth=0.8,
        label="y = x (perfect coupling)",
    )
    ax.set_xlabel("head forward velocity vx_h (px/s)")
    ax.set_ylabel("body forward velocity vx_b (px/s)")
    ax.set_title(
        f"Head vs body forward velocity (Pearson r = {r:.3f})\n"
        f"Tight diagonal → 2D conditioning enough; "
        f"fan-out → 4D justified",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


# -------------------------------------------------------------------- #
# Recommendation logic
# -------------------------------------------------------------------- #

def compute_per_session_summary(
    df: pd.DataFrame,
    markers: List[str],
    sessions: List[Tuple[str, int, int]],
    likelihood_threshold: float,
) -> List[dict]:
    """For each session, compute per-marker frac_high and
    longest_low_run, then summarize the worst marker per session.

    The aggregate worst-marker stats (used by make_recommendation)
    are dominated by whichever session has the worst single
    marker. With per-session breakdowns, users can see the
    distribution: are 1-2 sessions catastrophic and the rest fine,
    or are most sessions equally bad?

    Returns a list (one entry per session), each entry is a dict:
        {
            "name": <session name>,
            "n_frames": int,
            "worst_marker_frac_high": (marker_name, frac_high),
            "worst_marker_longest_run": (marker_name, longest_run),
            "all_markers": {marker_name: {frac_high, longest_low_run}},
        }
    """
    out = []
    for name, start, end in sessions:
        sub_df = df.iloc[start:end].reset_index(drop=True)
        n_frames = end - start
        per_marker = {}
        for m in markers:
            stats = compute_marker_stats(
                sub_df, m, likelihood_threshold,
                session_ranges=None,  # single-session slice
            )
            per_marker[m] = {
                "frac_high": stats.frac_high,
                "longest_low_run": stats.longest_low_run,
                "p_mean": stats.p_mean,
            }
        worst_frac_marker = min(
            per_marker.keys(), key=lambda m: per_marker[m]["frac_high"]
        )
        worst_run_marker = max(
            per_marker.keys(),
            key=lambda m: per_marker[m]["longest_low_run"],
        )
        out.append({
            "name": name,
            "n_frames": int(n_frames),
            "worst_marker_frac_high": [
                worst_frac_marker,
                per_marker[worst_frac_marker]["frac_high"],
            ],
            "worst_marker_longest_run": [
                worst_run_marker,
                per_marker[worst_run_marker]["longest_low_run"],
            ],
            "all_markers": per_marker,
        })
    return out


def _plot_per_session_summary(
    per_session_summary: List[dict],
    likelihood_threshold: float,
    output_path: Path,
) -> None:
    """Component 9: per-session worst-marker bar chart.

    Two side-by-side bar charts, one bar per session:
      - Worst-marker frac_high per session (higher = better)
      - Worst-marker longest dropout run per session (lower = better)

    The aggregate "worst" stats reported by make_recommendation are
    one bar each on these charts (the lowest frac_high bar, the
    tallest longest-run bar). When a few sessions are dramatic
    outliers, this plot shows it immediately.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not per_session_summary:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5, "No per-session data available",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return

    n_sessions = len(per_session_summary)
    names = [s["name"] for s in per_session_summary]
    frac_high = [s["worst_marker_frac_high"][1] for s in per_session_summary]
    longest_run = [s["worst_marker_longest_run"][1] for s in per_session_summary]

    # Color sessions by quality:
    #   green: worst frac_high > 0.5     (this session is fine)
    #   yellow: worst frac_high > 0.2    (marginal)
    #   red: worst frac_high <= 0.2      (one marker is essentially gone)
    colors = []
    for f in frac_high:
        if f > 0.5:
            colors.append("seagreen")
        elif f > 0.2:
            colors.append("goldenrod")
        else:
            colors.append("crimson")

    width = max(12, n_sessions * 0.18)
    fig, axes = plt.subplots(2, 1, figsize=(width, 7))

    axes[0].bar(range(n_sessions), frac_high, color=colors, alpha=0.85)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
    axes[0].axhline(0.2, color="gray", linestyle=":", linewidth=0.7)
    axes[0].set_ylabel(f"frac_high (p ≥ {likelihood_threshold:.2f})")
    axes[0].set_title(
        f"Per-session worst-marker frac_high — green > 0.5, "
        f"yellow > 0.2, red ≤ 0.2 ({n_sessions} sessions)"
    )
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks([])

    axes[1].bar(range(n_sessions), longest_run, color=colors, alpha=0.85)
    axes[1].set_ylabel("longest low-p run (frames)")
    axes[1].set_xlabel("session")
    axes[1].set_title("Per-session worst-marker longest dropout run")
    if n_sessions <= 30:
        axes[1].set_xticks(range(n_sessions))
        axes[1].set_xticklabels(names, rotation=90, fontsize=7)
    else:
        axes[1].set_xticks([])
        axes[1].text(
            0.99, 0.95,
            f"({n_sessions} sessions; "
            "see summary.json per_session_summary for names)",
            ha="right", va="top",
            transform=axes[1].transAxes, fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


# -------------------------------------------------------------------- #
# Recommendation logic
# -------------------------------------------------------------------- #


def make_recommendation(
    per_marker: List[MarkerStats],
    rigid_pair_stats: List[dict],
    velocity_modality: str,
    per_session_summary: Optional[List[dict]] = None,
) -> str:
    """Produce a build/skip/scope recommendation based on stats.

    Uses worst-case per-marker statistics rather than averages
    because dropouts typically hit specific markers (e.g. the
    nose during occlusion) while leaving others clean. Average
    would dilute the dropout signal across clean markers and
    miss the case where the smoother would help.

    When ``per_session_summary`` is provided, the report appends
    a per-session distribution: how many sessions are in each
    quality bucket. This is supplementary; the build decision is
    still gated on aggregate worst-case stats. Users can read the
    per-session distribution to spot whether 1-2 catastrophic
    sessions are inflating the worst-case stats compared to the
    typical session.
    """
    # Worst marker likelihood + worst longest-low-run.
    # The smoother helps when ANY marker has substantial
    # dropouts, not when ALL of them do.
    worst_frac_high = min(m.frac_high for m in per_marker)
    worst_longest_run = max(m.longest_low_run for m in per_marker)
    avg_frac_high = float(np.mean([m.frac_high for m in per_marker]))

    # Aggregate rigidity quality
    valid_pairs = [s for s in rigid_pair_stats if "warning" not in s]
    if valid_pairs:
        avg_cv = float(np.mean([s["cv_distance"] for s in valid_pairs]))
    else:
        avg_cv = float("nan")

    lines = ["DIAGNOSTIC RECOMMENDATION", "=" * 40, ""]
    lines.append(f"worst marker frac high-likelihood: {worst_frac_high:.3f}")
    lines.append(f"avg marker frac high-likelihood:   {avg_frac_high:.3f}")
    lines.append(f"worst longest low-p run:           {worst_longest_run} frames")
    lines.append(f"avg rigidity CV:                   {avg_cv:.4f}")
    lines.append(f"velocity distribution:             {velocity_modality}")
    lines.append("")

    # Per-session distribution (if multi-session)
    if per_session_summary and len(per_session_summary) > 1:
        n_total = len(per_session_summary)
        good = sum(
            1 for s in per_session_summary
            if s["worst_marker_frac_high"][1] > 0.5
        )
        marginal = sum(
            1 for s in per_session_summary
            if 0.2 < s["worst_marker_frac_high"][1] <= 0.5
        )
        bad = sum(
            1 for s in per_session_summary
            if s["worst_marker_frac_high"][1] <= 0.2
        )
        lines.append("PER-SESSION QUALITY (worst-marker frac_high):")
        lines.append(f"  good (>0.5):     {good:3d} of {n_total} sessions")
        lines.append(f"  marginal (>0.2): {marginal:3d} of {n_total} sessions")
        lines.append(f"  bad (≤0.2):      {bad:3d} of {n_total} sessions")
        # Cross-session range of worst longest run
        runs = [s["worst_marker_longest_run"][1] for s in per_session_summary]
        lines.append(
            f"  worst longest run per session: min={min(runs)}, "
            f"median={int(np.median(runs))}, max={max(runs)}"
        )
        # If aggregate worst run is dominated by 1-2 sessions,
        # call that out
        n_sessions_with_huge_runs = sum(
            1 for r in runs
            if r > 0.5 * worst_longest_run
        )
        if n_sessions_with_huge_runs <= 3 and n_total > 10:
            lines.append(
                f"  NOTE: only {n_sessions_with_huge_runs} session(s) "
                f"contribute to the aggregate 'worst longest low-p run' "
                f"figure above. Most sessions have a different worst case."
            )
        lines.append("")

    # Skip-build case: every marker has uniformly high likelihood
    # AND no marker has long dropouts
    if worst_frac_high > 0.97 and worst_longest_run < 10:
        lines.append("RECOMMENDATION: do NOT build.")
        lines.append("  Likelihood is uniformly high and dropouts are short")
        lines.append("  on every marker. Existing simple/advanced smoothers")
        lines.append("  should be fine.")
        return "\n".join(lines)

    # Loose-rigidity case: some markers have dropouts, but
    # rigid-pair CV is too loose for triplet covariance to help
    if not valid_pairs or np.isnan(avg_cv) or avg_cv > 0.15:
        lines.append("RECOMMENDATION: build v1 only (no triplet prior).")
        lines.append("  Likelihood/dropout pattern justifies Kalman, but")
        lines.append("  rigidity CV is too loose for triplet covariance to")
        lines.append("  add value. Per-marker Kalman + RTS is sufficient.")
        return "\n".join(lines)

    # Static-Σ case: dropouts + tight rigidity, but velocity is
    # unimodal so velocity-conditional Σ won't add signal
    if velocity_modality == "unimodal":
        lines.append("RECOMMENDATION: build v1 + static-Σ triplet prior.")
        lines.append("  Skip velocity-conditional Σ — saves ~3 weeks.")
        lines.append("  Velocity distribution is unimodal so velocity")
        lines.append("  conditioning won't add signal.")
        return "\n".join(lines)

    # Full v1+v2 case
    lines.append("RECOMMENDATION: build the full v1 + v2 system.")
    lines.append("  Likelihood, rigidity, and velocity-modality all")
    lines.append("  support the velocity-conditional triplet covariance")
    lines.append("  design. Estimated 4-7 weeks of focused work.")
    return "\n".join(lines)


def classify_velocity_modality(
    vx_h: np.ndarray,
    valid_mask: np.ndarray,
) -> str:
    """Classify the velocity distribution as unimodal / bimodal /
    multimodal based on a heuristic: smooth the histogram and
    count well-separated peaks.

    For production analysis, fit a GMM with model selection. This
    is enough for diagnostic-go/no-go.
    """
    if valid_mask.sum() < 100:
        return "insufficient_samples"
    v = vx_h[valid_mask]
    # Defensive: even with valid_mask filtering, NaN values can leak
    # through if a downstream-of-velocity computation propagated NaN
    # into a frame that valid_mask considers OK (e.g., np.gradient
    # at frame t inheriting NaN from frame t+1 when the t+1 body
    # axis was degenerate). Strip non-finite values explicitly so
    # np.histogram doesn't raise on a non-finite range.
    v = v[np.isfinite(v)]
    if len(v) < 100:
        return "insufficient_samples"
    n_bins = 50
    hist, edges = np.histogram(v, bins=n_bins)
    if hist.max() == 0:
        return "insufficient_samples"
    # Smooth with a 5-bin moving average to reduce histogram noise
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(hist.astype(float), kernel, mode="same")
    threshold = smoothed.max() * 0.3
    # Find candidate peaks (local maxima above threshold)
    candidates = []
    for i in range(2, len(smoothed) - 2):
        if (smoothed[i] > threshold
                and smoothed[i] >= smoothed[i - 1]
                and smoothed[i] >= smoothed[i + 1]
                and smoothed[i] >= smoothed[i - 2]
                and smoothed[i] >= smoothed[i + 2]):
            candidates.append(i)
    # Coalesce nearby candidates into single peaks. Two candidates
    # within min_separation bins of each other → one peak (the
    # taller). This avoids classifying noise on top of a single
    # Gaussian as multiple modes.
    min_separation = max(3, n_bins // 10)
    peaks = []
    for c in candidates:
        merged = False
        for j, p in enumerate(peaks):
            if abs(c - p) < min_separation:
                if smoothed[c] > smoothed[p]:
                    peaks[j] = c
                merged = True
                break
        if not merged:
            peaks.append(c)
    if len(peaks) <= 1:
        return "unimodal"
    elif len(peaks) == 2:
        return "bimodal"
    else:
        return "multimodal"


# -------------------------------------------------------------------- #
# Top-level orchestration
# -------------------------------------------------------------------- #

def run_diagnostic(
    csv_path,  # str | List[str] | Path  — single file, list, or directory
    output_dir: str,
    likelihood_threshold: float = 0.95,
    head_markers: Optional[List[str]] = None,
    body_markers: Optional[List[str]] = None,
    rigid_pairs: Optional[List[Tuple[str, str]]] = None,
    fps: float = 30.0,
    rigid_cv_threshold: float = 0.20,
    rigid_max_pairs: int = 8,
) -> DiagnosticReport:
    """Run the full diagnostic and write outputs to output_dir.

    Parameters
    ----------
    csv_path : str, list of str, or Path
        - A single file path (CSV or parquet) — single-session mode.
        - A directory — recursively discover .parquet files (or
          .csv if no parquets found) and run multi-session mode.
        - A list of file paths — multi-session mode using exactly
          those files in the order given.

        In multi-session mode, dropout run lengths and velocity
        computations are computed PER SESSION and aggregated, so
        no statistic spans a file boundary. Each file becomes one
        "session"; aggregate stats are reported with a per-session
        breakdown in summary.json.
    output_dir : str
        Directory to write 8 PNGs + summary.json + recommendation.txt.
    likelihood_threshold : float
        τ_high in the design doc. Frames below this are "low-p".
    head_markers : list of str
        Markers that define the head frame. If None, tries
        ["nose", "ear_left", "ear_right"] and falls back to the
        first 3 markers.
    body_markers : list of str
        Markers used for body centroid + PCA major axis (the
        body frame). If None, defaults to all markers minus
        head markers. PCA needs ≥2 markers; if fewer, body
        velocity is skipped.
    rigid_pairs : list of (str, str)
        Pairs treated as rigid for the Component 3 check. If None,
        tries default rigid pairs based on detected markers.
    fps : float
        Video frame rate, for converting per-frame velocity to
        per-second units.

    Returns
    -------
    DiagnosticReport
        Structured output. Also written to output_dir/summary.json.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve input(s) into a list of file paths
    paths: List[str]
    if isinstance(csv_path, (list, tuple)):
        paths = [str(p) for p in csv_path]
    elif isinstance(csv_path, (str, Path)):
        path_obj = Path(csv_path)
        if path_obj.is_dir():
            paths = discover_pose_files(str(path_obj))
            if not paths:
                raise ValueError(
                    f"No .parquet or .csv pose files found under "
                    f"{csv_path}"
                )
        else:
            paths = [str(path_obj)]
    else:
        raise TypeError(
            f"csv_path must be str, Path, or list of those; got "
            f"{type(csv_path)}"
        )

    # Load data — single-file or multi-file mode determines
    # whether session_ranges is non-trivial
    if len(paths) == 1:
        df, markers = load_pose_file(paths[0])
        sessions = [(Path(paths[0]).stem, 0, len(df))]
        session_ranges_for_compute = None  # single session = no boundary handling needed
        source_label = paths[0]
        print(
            f"[diagnostic] Loaded {len(df)} frames × {len(markers)} markers "
            f"from 1 file"
        )
    else:
        df, markers, sessions = load_pose_files(paths)
        session_ranges_for_compute = sessions
        source_label = (
            f"<{len(paths)} files concatenated; first={paths[0]}>"
        )
        print(
            f"[diagnostic] Loaded {len(df)} frames × {len(markers)} markers "
            f"from {len(paths)} files (sessions: "
            f"{[s[0] for s in sessions[:3]]}"
            f"{', ...' if len(sessions) > 3 else ''})"
        )

    # Default head markers
    if head_markers is None:
        candidates = [
            ["nose", "ear_left", "ear_right"],
            ["nose", "left_ear", "right_ear"],
            ["nose", "ear_l", "ear_r"],
        ]
        head_markers = None
        for cand in candidates:
            if all(c in markers for c in cand):
                head_markers = cand
                break
        if head_markers is None:
            head_markers = markers[:3]
            print(
                f"[diagnostic] Could not auto-detect head markers; "
                f"using first 3: {head_markers}",
                file=sys.stderr,
            )

    # Default body markers: everything except head markers
    if body_markers is None:
        body_markers = [m for m in markers if m not in head_markers]
        if len(body_markers) < 2:
            print(
                f"[diagnostic] Only {len(body_markers)} non-head markers "
                f"({body_markers}); body velocity needs ≥2. Body-frame "
                f"plots will be skipped.",
                file=sys.stderr,
            )

    # Default rigid pairs — auto-detect from data by CV threshold.
    # The user can still pass an explicit --rigid-pairs list to
    # override this. Auto-detection iterates over all pairs and
    # keeps those with CV < cv_threshold (default 0.20), sorted
    # ascending — picking up to ``rigid_max_pairs`` pairs.
    if rigid_pairs is None:
        # The v1 smoother architecture treats the head as a single
        # posture-coupled unit deferred to v2 (latent posture);
        # consequently, head markers must NOT participate in the
        # body-rigid pair set. Excluding them here keeps the rigid
        # pairs body-only, which is what the v1 triplet prior wants.
        head_excluded = list(head_markers) if head_markers else []
        print(
            f"[diagnostic] Auto-detecting rigid pairs "
            f"(cv_threshold={rigid_cv_threshold}, "
            f"max_pairs={rigid_max_pairs}, "
            f"excluding head markers: {head_excluded})..."
        )
        auto_pairs = auto_detect_rigid_pairs(
            df, markers, likelihood_threshold,
            cv_threshold=rigid_cv_threshold,
            max_pairs=rigid_max_pairs,
            exclude_markers=head_excluded,
        )
        if not auto_pairs:
            print(
                f"[diagnostic] No body-marker pair has CV < "
                f"{rigid_cv_threshold}; rigidity check will be empty. "
                f"This usually means likelihood_threshold is too high "
                f"(too few high-confidence frames) or no body markers "
                f"are actually rigid in this dataset.",
                file=sys.stderr,
            )
            rigid_pairs = []
        else:
            rigid_pairs = auto_pairs
            print(
                f"[diagnostic] Auto-detected {len(rigid_pairs)} rigid "
                f"pair(s):"
            )
            for a, b in rigid_pairs:
                stats = compute_rigid_pair_stats(
                    df, a, b, likelihood_threshold,
                )
                if "cv_distance" in stats:
                    print(
                        f"[diagnostic]   {a} ↔ {b}: "
                        f"CV={stats['cv_distance']:.4f}, "
                        f"mean_dist={stats['mean_distance']:.2f}px, "
                        f"n_samples={stats['n_high_confidence']}"
                    )

    # Candidate triplets — same exclusion as rigid pairs. Used as
    # input to the v1 static-Σ triplet prior in the smoother. We
    # always compute these, even when explicit ``rigid_pairs`` is
    # provided, because the user may want triplet info even with
    # custom pairs.
    head_excluded = list(head_markers) if head_markers else []
    print(
        f"[diagnostic] Detecting candidate triplets "
        f"(cv_threshold={rigid_cv_threshold}, "
        f"max_triplets={rigid_max_pairs})..."
    )
    candidate_triplets = auto_detect_candidate_triplets(
        df, markers, likelihood_threshold,
        cv_threshold=rigid_cv_threshold,
        max_triplets=rigid_max_pairs,
        exclude_markers=head_excluded,
    )
    if candidate_triplets:
        print(
            f"[diagnostic] Found {len(candidate_triplets)} "
            f"candidate triplet(s) for v1 smoother:"
        )
        for triplet, info in candidate_triplets:
            a, b, c = triplet
            print(
                f"[diagnostic]   ({a}, {b}, {c}): "
                f"cv_mean={info['cv_mean']:.4f}, "
                f"cv_max={info['cv_max']:.4f}, "
                f"n_samples={info['n_samples']}"
            )
    else:
        print(
            "[diagnostic] No body-marker triplet has all 3 pairwise "
            f"CVs < {rigid_cv_threshold}. The v1 smoother can still "
            "build with per-marker Kalman + RTS but cannot apply a "
            "triplet prior. Consider raising --rigid-cv-threshold or "
            "looking at the per-marker stats for systemic tracking "
            "issues.",
            file=sys.stderr,
        )

    # Behavioral-signal pairs — head-internal pairs that vary
    # meaningfully with posture (rearing, head tilt). Reported as
    # a separate section in summary.json so downstream consumers
    # can use them for behavior classification (e.g. ear-distance
    # as a rearing proxy) without confusing them with rigidity
    # constraints.
    behavioral_pairs = compute_behavioral_signal_pairs(
        df, head_markers, likelihood_threshold,
    )
    if behavioral_pairs:
        print(
            f"[diagnostic] Computed {len(behavioral_pairs)} "
            f"behavioral-signal pair(s) on head markers:"
        )
        for stats in behavioral_pairs:
            if "cv_distance" in stats:
                print(
                    f"[diagnostic]   {stats['marker_a']} ↔ "
                    f"{stats['marker_b']}: "
                    f"CV={stats['cv_distance']:.4f}, "
                    f"mean_dist={stats['mean_distance']:.2f}px"
                )

    # Component 1: per-marker stats
    print("[diagnostic] Component 1: per-marker likelihood stats...")
    per_marker = [
        compute_marker_stats(
            df, m, likelihood_threshold,
            session_ranges=session_ranges_for_compute,
        )
        for m in markers
    ]
    _plot_likelihood_histograms(df, markers, out / "01_likelihood.png")

    # Component 2: dropout run lengths
    print("[diagnostic] Component 2: dropout run lengths...")
    _plot_run_lengths(per_marker, out / "02_dropouts.png")

    # Component 3: rigid pair distances
    print("[diagnostic] Component 3: rigid pair distances...")
    rigid_pair_stats = [
        compute_rigid_pair_stats(df, a, b, likelihood_threshold)
        for a, b in rigid_pairs
    ]
    _plot_rigid_pairs(
        rigid_pair_stats, df, likelihood_threshold,
        out / "03_rigid_pairs.png",
    )

    # Component 4: head-frame velocity
    print("[diagnostic] Component 4: head-frame velocity distribution...")
    vx_h, vy_h, head_valid = compute_head_velocity(
        df, head_markers, likelihood_threshold, fps,
        session_ranges=session_ranges_for_compute,
    )
    _plot_velocity_distribution(
        vx_h, vy_h, head_valid, out / "04_velocity.png",
    )

    velocity_modality = classify_velocity_modality(vx_h, head_valid)
    head_velocity_stats = {
        "n_valid_samples": int(head_valid.sum()),
        "vx_h_mean": float(np.nanmean(vx_h[head_valid])) if head_valid.sum() > 0 else float("nan"),
        "vx_h_std": float(np.nanstd(vx_h[head_valid])) if head_valid.sum() > 0 else float("nan"),
        "vy_h_mean": float(np.nanmean(vy_h[head_valid])) if head_valid.sum() > 0 else float("nan"),
        "vy_h_std": float(np.nanstd(vy_h[head_valid])) if head_valid.sum() > 0 else float("nan"),
        "modality": velocity_modality,
    }

    # Component 5: head-velocity-vs-configuration scatter
    print("[diagnostic] Component 5: head-velocity-vs-configuration...")
    _plot_velocity_vs_configuration(
        df, rigid_pair_stats, vx_h, head_valid, likelihood_threshold,
        out / "05_velocity_vs_config.png",
    )

    # Component 6: body-frame velocity (if enough body markers)
    body_velocity_stats: dict
    body_modality = "skipped"
    head_body_corr = float("nan")
    if len(body_markers) >= 2:
        print(
            f"[diagnostic] Component 6: body-frame velocity "
            f"({len(body_markers)} markers: {body_markers})..."
        )
        vx_b, vy_b, body_valid, body_diag = compute_body_velocity(
            df, body_markers, head_markers, likelihood_threshold, fps,
            session_ranges=session_ranges_for_compute,
        )
        _plot_body_velocity_distribution(
            vx_b, vy_b, body_valid, out / "06_body_velocity.png",
        )
        body_modality = classify_velocity_modality(vx_b, body_valid)
        body_velocity_stats = {
            "n_valid_samples": int(body_valid.sum()),
            "vx_b_mean": float(np.nanmean(vx_b[body_valid])) if body_valid.sum() > 0 else float("nan"),
            "vx_b_std": float(np.nanstd(vx_b[body_valid])) if body_valid.sum() > 0 else float("nan"),
            "vy_b_mean": float(np.nanmean(vy_b[body_valid])) if body_valid.sum() > 0 else float("nan"),
            "vy_b_std": float(np.nanstd(vy_b[body_valid])) if body_valid.sum() > 0 else float("nan"),
            "modality": body_modality,
            "sign_disambiguation": body_diag,
            "body_markers_used": list(body_markers),
        }

        # Component 7: body-velocity-vs-configuration scatter
        print("[diagnostic] Component 7: body-velocity-vs-configuration...")
        _plot_velocity_vs_configuration_body(
            df, rigid_pair_stats, vx_b, body_valid, likelihood_threshold,
            out / "07_velocity_vs_config_body.png",
        )

        # Component 8: head-body velocity correlation
        print("[diagnostic] Component 8: head-body velocity correlation...")
        head_body_corr = head_body_velocity_correlation(
            vx_h, vx_b, head_valid, body_valid,
        )
        _plot_head_body_velocity_correlation(
            vx_h, vx_b, head_valid, body_valid,
            out / "08_head_body_correlation.png",
        )
    else:
        print(
            "[diagnostic] Components 6-8: SKIPPED "
            "(need ≥2 body markers for PCA-body-frame)"
        )
        body_velocity_stats = {
            "n_valid_samples": 0,
            "skipped_reason": (
                f"only {len(body_markers)} non-head markers; "
                f"need ≥2 for PCA-based body frame"
            ),
            "body_markers_used": list(body_markers),
        }

    # Per-session rollup — computed only in multi-session mode so
    # we don't waste cycles in single-session runs (where the
    # rollup would just duplicate the aggregate stats with one
    # entry).
    if len(sessions) > 1:
        print("[diagnostic] Per-session rollup (worst-marker stats)...")
        per_session_summary = compute_per_session_summary(
            df, markers, sessions, likelihood_threshold,
        )
        _plot_per_session_summary(
            per_session_summary, likelihood_threshold,
            out / "09_per_session.png",
        )
    else:
        per_session_summary = []

    # Recommendation (currently uses head modality only — body
    # information is presented in plots for now; recommendation
    # logic upgrade can come once we see real-data patterns)
    recommendation = make_recommendation(
        per_marker, rigid_pair_stats, velocity_modality,
        per_session_summary=per_session_summary,
    )
    # Append a body-velocity note if available
    if body_modality != "skipped":
        recommendation += (
            f"\n\n[body velocity also available: modality={body_modality}, "
            f"head-body correlation r={head_body_corr:.3f}]"
        )
    (out / "recommendation.txt").write_text(recommendation)

    # Summary
    report = DiagnosticReport(
        csv_path=source_label,
        n_frames=len(df),
        n_markers=len(markers),
        likelihood_threshold=likelihood_threshold,
        head_markers=head_markers,
        body_markers=list(body_markers),
        rigid_pairs=[list(p) for p in rigid_pairs],
        per_marker=per_marker,
        rigid_pair_stats=rigid_pair_stats,
        head_velocity_stats=head_velocity_stats,
        body_velocity_stats=body_velocity_stats,
        velocity_modality=velocity_modality,
        head_body_correlation=head_body_corr,
        recommendation=recommendation,
        per_session_summary=per_session_summary,
    )
    summary_dict = asdict(report)
    # Include the session breakdown (file basenames + frame ranges)
    # so users can see per-file structure even though stats are
    # aggregated.
    summary_dict["sessions"] = [
        {"name": name, "start_idx": start, "end_idx": end,
         "n_frames": end - start}
        for name, start, end in sessions
    ]
    summary_dict["source_files"] = paths
    # Candidate triplets (body-only, head-excluded) for the v1
    # smoother triplet prior. Each entry: [(a, b, c), info_dict].
    summary_dict["candidate_triplets"] = [
        {
            "markers": list(triplet),
            **info,
        }
        for triplet, info in candidate_triplets
    ]
    # Head-internal behavioral-signal pairs (e.g. ear-distance for
    # rearing detection). NOT used as smoother constraints; surfaced
    # for downstream behavior classification.
    summary_dict["behavioral_signal_pairs"] = behavioral_pairs
    (out / "summary.json").write_text(json.dumps(summary_dict, indent=2))

    print("\n" + recommendation + "\n")
    print(f"[diagnostic] Outputs written to: {out}")
    return report


# -------------------------------------------------------------------- #
# CLI
# -------------------------------------------------------------------- #

def _parse_pairs(s: str) -> List[Tuple[str, str]]:
    """Parse 'a,b;c,d;e,f' → [('a','b'),('c','d'),('e','f')]."""
    if not s:
        return []
    out = []
    for piece in s.split(";"):
        parts = [p.strip() for p in piece.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Bad pair {piece!r}; expected 'marker_a,marker_b'"
            )
        out.append((parts[0].lower(), parts[1].lower()))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostic for Kalman pose smoother design",
    )
    parser.add_argument(
        "csv_path", nargs="+",
        help=(
            "Pose data input. Either a single file (CSV or "
            "parquet), a directory (recursively scanned for "
            ".parquet files, falling back to .csv if no parquets), "
            "or multiple file paths separated by spaces. Multiple "
            "files are treated as separate sessions and aggregated "
            "with proper boundary handling."
        ),
    )
    parser.add_argument(
        "--output-dir", default="./kalman_diagnostic_output",
        help="Output directory for plots + reports",
    )
    parser.add_argument(
        "--likelihood-threshold", type=float, default=0.95,
        help="Likelihood threshold for high-confidence frames",
    )
    parser.add_argument(
        "--head-markers", default="",
        help="Comma-separated list of head markers (auto if empty)",
    )
    parser.add_argument(
        "--body-markers", default="",
        help=(
            "Comma-separated list of body markers used for "
            "PCA-based body frame (auto = all markers minus head)"
        ),
    )
    parser.add_argument(
        "--rigid-pairs", default="",
        help=(
            "Semicolon-separated rigid pairs, e.g. 'a,b;c,d'. "
            "If empty, pairs are auto-detected from the data by "
            "the inter-marker distance CV (see --rigid-cv-threshold)."
        ),
    )
    parser.add_argument(
        "--rigid-cv-threshold", type=float, default=0.20,
        help=(
            "Auto-detection threshold: keep marker pairs with "
            "inter-marker-distance CV below this value (default "
            "0.20). Lower = stricter rigidity. Only applies when "
            "--rigid-pairs is not given."
        ),
    )
    parser.add_argument(
        "--rigid-max-pairs", type=int, default=8,
        help=(
            "Maximum number of auto-detected rigid pairs to use "
            "(default 8). The tightest pairs by CV are selected."
        ),
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Video frame rate (for velocity units)",
    )
    args = parser.parse_args(argv)

    head_markers = (
        [m.strip().lower() for m in args.head_markers.split(",") if m.strip()]
        or None
    )
    body_markers = (
        [m.strip().lower() for m in args.body_markers.split(",") if m.strip()]
        or None
    )
    rigid_pairs = _parse_pairs(args.rigid_pairs) or None

    # csv_path is a list (nargs='+'). If exactly one element and
    # it's a directory, run_diagnostic will discover files
    # internally; if it's one file, single-session mode; if
    # multiple files, multi-session mode.
    csv_path = args.csv_path[0] if len(args.csv_path) == 1 else args.csv_path

    run_diagnostic(
        csv_path=csv_path,
        output_dir=args.output_dir,
        likelihood_threshold=args.likelihood_threshold,
        head_markers=head_markers,
        body_markers=body_markers,
        rigid_pairs=rigid_pairs,
        fps=args.fps,
        rigid_cv_threshold=args.rigid_cv_threshold,
        rigid_max_pairs=args.rigid_max_pairs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
