"""Diagnostic tool for the Kalman pose smoother design.

Stage 0 in docs/kalman_smoother_design.md §5.1. Loads a pose
CSV (DLC, SLEAP, or generic) and produces a 5-component report
that confirms whether the data has the characteristics that
justify building the proposed Kalman + velocity-conditional
triplet covariance smoother.

Five components produced:

  1. Likelihood histograms — one per marker plus aggregated.
     If most p > 0.95, a variance-aware smoother adds little.

  2. Run-length distribution of low-likelihood frames per
     marker. Long runs (>1 second of dropout) are exactly
     where Kalman + spatial prior helps most.

  3. Inter-marker distance distributions for "rigid" pairs.
     Tight distributions confirm the rigidity assumption.

  4. Head-frame egocentric velocity distribution. Multi-modal
     or wide → velocity-conditional Σ justified.

  5. Velocity-vs-configuration scatter. Tight band varying
     with velocity → velocity-conditional Σ justified.

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
        --rigid-pairs "Ear_left,Ear_right;Nose,Ear_left;Nose,Ear_right"

Or as a Python API:

    from mufasa.data_processors.kalman_diagnostic import run_diagnostic
    report = run_diagnostic(
        csv_path="path/to/pose.csv",
        output_dir="/tmp/diag/",
    )

The output is a directory containing:
  - 5 PNG plots (one per component)
  - A summary.json with key statistics
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
    rigid_pairs: List[Tuple[str, str]]
    per_marker: List[MarkerStats]
    rigid_pair_stats: List[dict]    # mean, std, cv of inter-marker distance
    velocity_stats: dict             # mean, std of head-frame velocity
    velocity_modality: str           # "unimodal", "bimodal", "multimodal"
    recommendation: str              # build/static-only/skip/diagnostic-failed


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


def load_pose_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load a pose CSV, autodetecting multi-index headers.

    Handles three formats:
      - Mufasa flat-header CSV: row 0 is column names, no index col
      - DLC 3-row multi-index header: rows 0-2 are header levels
      - Single-header CSV with index column 0
    """
    # Read first line to sniff header style
    with open(csv_path, "r") as f:
        first = f.readline().strip()

    # Try flat header first — most Mufasa intermediate files
    df = pd.read_csv(csv_path, index_col=0)
    # Lowercase column names for downstream uniformity
    df.columns = [str(c).lower() for c in df.columns]

    markers = detect_marker_columns(df)
    if markers:
        return df, markers

    # No markers found — could be DLC multi-index
    # Try header=[0,1,2]
    df_multi = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    # Flatten multi-index columns: ('scorer', 'bodypart', 'coord')
    # → 'bodypart_coord' lowercased
    new_cols = []
    for col in df_multi.columns:
        if isinstance(col, tuple) and len(col) >= 2:
            # Use the last two levels: bodypart + coord
            new_cols.append(f"{col[-2]}_{col[-1]}".lower())
        else:
            new_cols.append(str(col).lower())
    df_multi.columns = new_cols
    markers = detect_marker_columns(df_multi)
    if markers:
        return df_multi, markers

    raise ValueError(
        f"Could not detect marker columns in {csv_path}. "
        f"Expected pattern <name>_x, <name>_y, <name>_p."
    )


# -------------------------------------------------------------------- #
# Component 1: Likelihood histograms
# -------------------------------------------------------------------- #

def compute_marker_stats(
    df: pd.DataFrame,
    marker: str,
    likelihood_threshold: float,
) -> MarkerStats:
    """Compute per-marker likelihood + dropout statistics."""
    p_col = f"{marker}{SUFFIX_P}"
    p = df[p_col].values
    n = len(p)
    is_low = p < likelihood_threshold

    # Run-length encoding of low-likelihood runs
    runs = []
    current_run = 0
    for v in is_low:
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


# -------------------------------------------------------------------- #
# Component 4: Head-frame velocity
# -------------------------------------------------------------------- #

def compute_head_velocity(
    df: pd.DataFrame,
    head_markers: List[str],
    likelihood_threshold: float,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute head-frame egocentric velocity for all frames.

    Returns:
      vx_h, vy_h: velocity in head frame, shape (T,). NaN where
        the head triplet wasn't reliable.
      valid_mask: boolean shape (T,) — True where velocity is
        reliably computable (head triplet high-p at t and t-1).
    """
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


# -------------------------------------------------------------------- #
# Recommendation logic
# -------------------------------------------------------------------- #

def make_recommendation(
    per_marker: List[MarkerStats],
    rigid_pair_stats: List[dict],
    velocity_modality: str,
) -> str:
    """Produce a build/skip/scope recommendation based on stats.

    Uses worst-case per-marker statistics rather than averages
    because dropouts typically hit specific markers (e.g. the
    nose during occlusion) while leaving others clean. Average
    would dilute the dropout signal across clean markers and
    miss the case where the smoother would help.
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
    csv_path: str,
    output_dir: str,
    likelihood_threshold: float = 0.95,
    head_markers: Optional[List[str]] = None,
    rigid_pairs: Optional[List[Tuple[str, str]]] = None,
    fps: float = 30.0,
) -> DiagnosticReport:
    """Run the full diagnostic and write outputs to output_dir.

    Parameters
    ----------
    csv_path : str
        Path to a pose CSV (DLC, SLEAP, or Mufasa flat-header).
    output_dir : str
        Directory to write 5 PNGs + summary.json + recommendation.txt.
    likelihood_threshold : float
        τ_high in the design doc. Frames below this are "low-p".
    head_markers : list of str
        Markers that define the head frame (for egocentric velocity).
        If None, tries ["nose", "ear_left", "ear_right"] and falls
        back to the first 3 markers.
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

    df, markers = load_pose_csv(csv_path)
    print(f"[diagnostic] Loaded {len(df)} frames × {len(markers)} markers")

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

    # Default rigid pairs
    if rigid_pairs is None:
        if len(head_markers) >= 3:
            rigid_pairs = [
                (head_markers[1], head_markers[2]),  # ear-ear
                (head_markers[0], head_markers[1]),  # nose-ear
                (head_markers[0], head_markers[2]),  # nose-other-ear
            ]
        else:
            rigid_pairs = []
            print(
                "[diagnostic] No rigid pairs to check (need ≥3 head markers)",
                file=sys.stderr,
            )

    # Component 1: per-marker stats
    print("[diagnostic] Component 1: per-marker likelihood stats...")
    per_marker = [
        compute_marker_stats(df, m, likelihood_threshold)
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
    vx_h, vy_h, valid_mask = compute_head_velocity(
        df, head_markers, likelihood_threshold, fps,
    )
    _plot_velocity_distribution(
        vx_h, vy_h, valid_mask, out / "04_velocity.png",
    )

    velocity_modality = classify_velocity_modality(vx_h, valid_mask)
    velocity_stats = {
        "n_valid_samples": int(valid_mask.sum()),
        "vx_h_mean": float(np.nanmean(vx_h[valid_mask])) if valid_mask.sum() > 0 else float("nan"),
        "vx_h_std": float(np.nanstd(vx_h[valid_mask])) if valid_mask.sum() > 0 else float("nan"),
        "vy_h_mean": float(np.nanmean(vy_h[valid_mask])) if valid_mask.sum() > 0 else float("nan"),
        "vy_h_std": float(np.nanstd(vy_h[valid_mask])) if valid_mask.sum() > 0 else float("nan"),
        "modality": velocity_modality,
    }

    # Component 5: velocity-vs-configuration scatter
    print("[diagnostic] Component 5: velocity-vs-configuration scatter...")
    _plot_velocity_vs_configuration(
        df, rigid_pair_stats, vx_h, valid_mask, likelihood_threshold,
        out / "05_velocity_vs_config.png",
    )

    # Recommendation
    recommendation = make_recommendation(
        per_marker, rigid_pair_stats, velocity_modality,
    )
    (out / "recommendation.txt").write_text(recommendation)

    # Summary
    report = DiagnosticReport(
        csv_path=str(csv_path),
        n_frames=len(df),
        n_markers=len(markers),
        likelihood_threshold=likelihood_threshold,
        head_markers=head_markers,
        rigid_pairs=[list(p) for p in rigid_pairs],
        per_marker=per_marker,
        rigid_pair_stats=rigid_pair_stats,
        velocity_stats=velocity_stats,
        velocity_modality=velocity_modality,
        recommendation=recommendation,
    )
    summary_dict = asdict(report)
    # MarkerStats are dataclasses; asdict already converted them
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
    parser.add_argument("csv_path", help="Path to pose CSV file")
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
        "--rigid-pairs", default="",
        help="Semicolon-separated rigid pairs, e.g. 'a,b;c,d'",
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
    rigid_pairs = _parse_pairs(args.rigid_pairs) or None

    run_diagnostic(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        likelihood_threshold=args.likelihood_threshold,
        head_markers=head_markers,
        rigid_pairs=rigid_pairs,
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
