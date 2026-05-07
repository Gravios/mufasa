#!/usr/bin/env python3
"""Per-session marker offset diagnostic for Mufasa v2 outputs.

Hypothesis: the v2 smoother's q_root_pos hits the ceiling
because the rigid (single global) marker_offsets don't fit
the data — different sessions/rats have systematically
different marker placement, and the smoother absorbs this
by inflating root motion noise.

This script tests that hypothesis. For each session:
  1. Load raw observations (parquet input) and smoothed
     output (smoothed_v2.parquet).
  2. From the smoothed pose, compute per-frame body-frame
     coordinates for each marker.
  3. Compute the median (length, angle) offset per marker
     per session.
  4. Compare across sessions: if the inter-session spread
     of offsets is large compared to within-session
     scatter, per-session fitting is justified.

Usage
-----

  python per_session_offset_diagnostic.py \\
      --raw-dir   /data/.../input_csv/ \\
      --smoothed-dir /tmp/smoothed_v6/ \\
      --model     /tmp/smoothed_v6/v2_model.npz \\
      --output    /tmp/offset_diagnostic.txt

Output is a text report:
  - Per-marker table: median length, IQR length, median
    angle, IQR angle (in degrees) across sessions
  - "Disagreement score" per marker: ratio of inter-session
    IQR to within-session IQR
  - Sessions ranked by total disagreement (which sessions
    are most "off model")

A high disagreement score (>2-3) for any marker is evidence
that per-session offset fitting would help.

Patch 119-pre: file loading
---------------------------
Previously this script assumed both raw and smoothed parquets
used a DLC MultiIndex (scorer, bodyparts, coords). The v2
smoother actually writes flat columns (``<marker>_x``,
``<marker>_y``, ``<marker>_p``, ``<marker>_var_x``,
``<marker>_var_y``), so ``load_session`` raised KeyError on
every smoothed file, exceptions were caught silently, and
the recommendation block fell through to a default "Phase 2"
string with zero data behind it. Loading now goes through
``mufasa.data_processors.kalman_diagnostic.load_pose_file``,
which handles both formats. The recommendation block also
no longer manufactures a recommendation when no data made
it through.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_session(
    raw_path: Path, smoothed_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load raw + smoothed marker positions for one session.

    Both inputs are routed through
    ``mufasa.data_processors.kalman_diagnostic.load_pose_file``
    which handles flat-column parquet/CSV (the v2 smoother's
    output format) and DLC multi-row-header CSV (the typical
    raw input format). Previously this routine assumed both
    files were MultiIndex DLC parquets and raised KeyError on
    every smoothed file.

    Returns
    -------
    raw : dict[marker -> (T, 2) array]
    smooth : dict[marker -> (T, 2) array of smoothed positions]
    """
    try:
        from mufasa.data_processors.kalman_diagnostic import (
            load_pose_file,
        )
    except ImportError as e:
        raise RuntimeError(
            "Could not import load_pose_file from "
            "mufasa.data_processors.kalman_diagnostic. Make "
            "sure mufasa is on PYTHONPATH."
        ) from e

    raw_df, raw_markers_list = load_pose_file(str(raw_path))
    smooth_df, smooth_markers_list = load_pose_file(str(smoothed_path))

    raw_markers: Dict[str, np.ndarray] = {}
    for m in raw_markers_list:
        # load_pose_file lowercases columns; markers list is
        # already lowercase. Marker names from the model file
        # may be mixed-case, callers normalize as needed.
        x = raw_df[f"{m}_x"].to_numpy(dtype=float)
        y = raw_df[f"{m}_y"].to_numpy(dtype=float)
        raw_markers[m] = np.stack([x, y], axis=1)

    smooth_markers: Dict[str, np.ndarray] = {}
    for m in smooth_markers_list:
        x = smooth_df[f"{m}_x"].to_numpy(dtype=float)
        y = smooth_df[f"{m}_y"].to_numpy(dtype=float)
        smooth_markers[m] = np.stack([x, y], axis=1)

    return raw_markers, smooth_markers


def compute_body_frame_offset(
    raw_markers: Dict[str, np.ndarray],
    smooth_markers: Dict[str, np.ndarray],
    model_data,
    marker: str,
) -> np.ndarray:
    """For a given marker, compute the (length, angle) of the
    raw observation in the body frame across all valid frames.

    The body frame is defined by the smoothed body's root
    position and orientation. We need to know:
      - The marker's parent segment
      - The smoothed segment world position and orientation

    For simplicity, this script uses the SMOOTHED MARKER
    as a proxy for the segment frame. This is a reasonable
    approximation when there's a marker at each segment's
    distal end (the standard rat layout has back2 = root,
    headmid = head distal, tailbase = tail_1 distal).

    Returns
    -------
    (T, 2) array of (length_t, angle_t) for valid frames
    only (filtered for likelihood). NaN for invalid frames.
    """
    # Map marker → parent segment marker (for body-frame ref)
    layout_segments = model_data["layout_segments"]
    parent_marker_map: Dict[str, str] = {}
    for seg_dict in layout_segments:
        seg_name = seg_dict["name"]
        seg_distal_marker = None
        # Find the marker with offset (0, 0) — that's the
        # segment's distal endpoint
        for mk, (l, a) in seg_dict["markers"].items():
            if abs(l) < 1e-6 and abs(a) < 1e-6:
                seg_distal_marker = mk
                break
        if seg_distal_marker is not None:
            for mk in seg_dict["markers"]:
                if mk != seg_distal_marker:
                    parent_marker_map[mk] = seg_distal_marker

    if marker not in parent_marker_map:
        return np.full((1, 2), np.nan)

    parent_marker = parent_marker_map[marker]
    if (
        parent_marker not in smooth_markers
        or marker not in raw_markers
    ):
        return np.full((1, 2), np.nan)

    raw_pos = raw_markers[marker]
    parent_pos = smooth_markers[parent_marker]

    if raw_pos.shape != parent_pos.shape:
        T = min(raw_pos.shape[0], parent_pos.shape[0])
        raw_pos = raw_pos[:T]
        parent_pos = parent_pos[:T]

    # Vector from parent marker to this marker, in WORLD frame
    rel = raw_pos - parent_pos

    # Crude body orientation: vector from root to head
    # (back2 → headmid). If both available, use this.
    if "back2" in smooth_markers and "headmid" in smooth_markers:
        ref = smooth_markers["headmid"] - smooth_markers["back2"]
        T = min(rel.shape[0], ref.shape[0])
        rel = rel[:T]
        ref = ref[:T]
        ref_angle = np.arctan2(ref[:, 1], ref[:, 0])
    else:
        ref_angle = np.zeros(rel.shape[0])

    # Rotate rel into body frame
    cos_phi = np.cos(-ref_angle)
    sin_phi = np.sin(-ref_angle)
    x_body = cos_phi * rel[:, 0] - sin_phi * rel[:, 1]
    y_body = sin_phi * rel[:, 0] + cos_phi * rel[:, 1]

    length = np.sqrt(x_body ** 2 + y_body ** 2)
    angle = np.arctan2(y_body, x_body)

    # Mask out NaN and absurd values
    mask = (
        np.isfinite(length)
        & np.isfinite(angle)
        & (length < 200)  # 200 px = sanity cap
    )
    out = np.full((rel.shape[0], 2), np.nan)
    out[mask, 0] = length[mask]
    out[mask, 1] = angle[mask]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir", required=True, type=Path,
        help="Directory containing input parquet files",
    )
    parser.add_argument(
        "--smoothed-dir", required=True, type=Path,
        help=(
            "Directory containing _smoothed_v2.parquet "
            "files (output of v2 smoother)"
        ),
    )
    parser.add_argument(
        "--model", required=True, type=Path,
        help="v2_model.npz file from the v2 smoother run",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("offset_diagnostic.txt"),
        help="Output report path (default: offset_diagnostic.txt)",
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"ERROR: raw-dir not found: {args.raw_dir}")
        return 1
    if not args.smoothed_dir.exists():
        print(f"ERROR: smoothed-dir not found: {args.smoothed_dir}")
        return 1
    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}")
        return 1

    print(f"Loading model from {args.model}...")
    model_data = np.load(args.model, allow_pickle=True)

    # Pair up raw + smoothed files
    raw_files = sorted(args.raw_dir.glob("*.parquet"))
    print(f"Found {len(raw_files)} raw parquet files")

    pairs: List[Tuple[Path, Path]] = []
    for raw_p in raw_files:
        smooth_name = raw_p.stem + "_smoothed_v2.parquet"
        smooth_p = args.smoothed_dir / smooth_name
        if smooth_p.exists():
            pairs.append((raw_p, smooth_p))
        else:
            print(f"  warn: no smoothed file for {raw_p.name}")
    print(f"Paired {len(pairs)} sessions")

    # Get list of markers from model
    sigma_marker_items = model_data["params_sigma_marker"]
    all_markers = [m for m, _ in sigma_marker_items]
    print(f"Markers: {all_markers}")

    # Per-session, per-marker median (length, angle)
    # offsets_table[marker][session_name] = (median_l, median_a, iqr_l, iqr_a)
    offsets_table: Dict[
        str, Dict[str, Tuple[float, float, float, float]],
    ] = {m: {} for m in all_markers}

    # Track load failures so we don't fall through to a
    # bogus recommendation when most sessions failed to
    # load. Previously these were printed once each and
    # then forgotten.
    load_errors: List[Tuple[str, str]] = []

    for i, (raw_p, smooth_p) in enumerate(pairs):
        sess_name = raw_p.stem
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(pairs)}] {sess_name}")
        try:
            raw_markers, smooth_markers = load_session(
                raw_p, smooth_p,
            )
        except Exception as e:
            load_errors.append((sess_name, f"{type(e).__name__}: {e}"))
            print(f"    ERROR: {e}")
            continue

        for marker in all_markers:
            offset_series = compute_body_frame_offset(
                raw_markers, smooth_markers, model_data, marker,
            )
            valid = ~np.isnan(offset_series[:, 0])
            if valid.sum() < 100:
                continue
            ls = offset_series[valid, 0]
            angs = offset_series[valid, 1]
            # Robust stats: median + IQR
            med_l = float(np.median(ls))
            iqr_l = float(np.percentile(ls, 75) - np.percentile(ls, 25))
            med_a_deg = float(np.degrees(np.median(angs)))
            iqr_a_deg = float(np.degrees(
                np.percentile(angs, 75) - np.percentile(angs, 25)
            ))
            offsets_table[marker][sess_name] = (
                med_l, med_a_deg, iqr_l, iqr_a_deg,
            )

    # Build report
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("Per-session marker offset diagnostic")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"Analyzed {len(pairs)} sessions × {len(all_markers)} markers"
    )
    if load_errors:
        lines.append(
            f"WARNING: {len(load_errors)} session(s) failed "
            f"to load and were skipped"
        )
    lines.append("")
    lines.append("Per-marker summary:")
    lines.append("-" * 78)
    lines.append(
        f"{'marker':<20s}  "
        f"{'inter_l_iqr':>11s}  {'med_within_l_iqr':>16s}  "
        f"{'disagree_l':>10s}  "
        f"{'inter_a_iqr':>11s}  {'med_within_a_iqr':>16s}  "
        f"{'disagree_a':>10s}"
    )
    lines.append("-" * 78)

    overall_disagree: List[Tuple[str, float]] = []
    session_disagreement: Dict[str, float] = {}

    for marker in all_markers:
        sess_data = offsets_table[marker]
        if not sess_data:
            lines.append(f"{marker:<20s}  (no data)")
            continue
        med_ls = np.array([v[0] for v in sess_data.values()])
        med_as = np.array([v[1] for v in sess_data.values()])
        iqr_ls = np.array([v[2] for v in sess_data.values()])
        iqr_as = np.array([v[3] for v in sess_data.values()])

        inter_l_iqr = float(
            np.percentile(med_ls, 75) - np.percentile(med_ls, 25)
        )
        within_l_iqr = float(np.median(iqr_ls))
        inter_a_iqr = float(
            np.percentile(med_as, 75) - np.percentile(med_as, 25)
        )
        within_a_iqr = float(np.median(iqr_as))

        disagree_l = (
            inter_l_iqr / max(within_l_iqr, 0.1)
        )
        disagree_a = (
            inter_a_iqr / max(within_a_iqr, 0.1)
        )
        overall_disagree.append(
            (marker, max(disagree_l, disagree_a))
        )
        lines.append(
            f"{marker:<20s}  "
            f"{inter_l_iqr:>10.2f}px {within_l_iqr:>15.2f}px "
            f"{disagree_l:>10.2f} "
            f"{inter_a_iqr:>10.2f}° {within_a_iqr:>15.2f}° "
            f"{disagree_a:>10.2f}"
        )

        # Per-session disagreement: |session_offset - global_median|
        # in IQR units
        global_med_l = np.median(med_ls)
        global_med_a = np.median(med_as)
        for sess, (ml, ma, _, _) in sess_data.items():
            d_l = abs(ml - global_med_l) / max(within_l_iqr, 0.1)
            d_a = abs(ma - global_med_a) / max(within_a_iqr, 0.1)
            session_disagreement[sess] = (
                session_disagreement.get(sess, 0.0)
                + max(d_l, d_a)
            )

    lines.append("-" * 78)
    lines.append("")
    lines.append("Interpretation of disagree score (inter / within IQR ratio):")
    lines.append("  < 1.0:  inter-session variation is smaller than within;")
    lines.append("          per-session offsets won't help much.")
    lines.append("  1-2:    moderate variation; per-session offsets useful.")
    lines.append("  > 2.0:  inter-session dominates; per-session essential.")
    lines.append("")
    lines.append("Markers ranked by max disagreement:")
    overall_disagree.sort(key=lambda x: x[1], reverse=True)
    for i, (m, d) in enumerate(overall_disagree[:15]):
        lines.append(f"  {i+1:2d}. {m:<20s} {d:.2f}")
    lines.append("")
    lines.append("Sessions ranked by total disagreement (most outlier first):")
    sess_rank = sorted(
        session_disagreement.items(),
        key=lambda x: x[1], reverse=True,
    )
    for i, (s, d) in enumerate(sess_rank[:15]):
        lines.append(f"  {i+1:2d}. {s:<40s} {d:.1f}")
    lines.append("")
    lines.append("Recommendation:")
    n_with_data = sum(1 for m in all_markers if offsets_table[m])
    max_disagree = (
        max(d for _, d in overall_disagree)
        if overall_disagree else 0.0
    )
    if n_with_data == 0:
        lines.append(
            "  NO DATA — every (session, marker) pair was "
            "either dropped at load time or had < 100 valid "
            "frames. Cannot recommend a phase from this run."
        )
        if load_errors:
            n_err = len(load_errors)
            lines.append(
                f"  {n_err} session(s) failed to load. First "
                f"3 errors:"
            )
            for sess, msg in load_errors[:3]:
                lines.append(f"    {sess}: {msg}")
            if n_err > 3:
                lines.append(f"    ...and {n_err - 3} more.")
        else:
            lines.append(
                "  No load errors — failure is downstream "
                "(e.g. parent_marker_map empty, or all "
                "frames filtered by the body-frame mask). "
                "Inspect compute_body_frame_offset by hand "
                "on a single session."
            )
    elif max_disagree > 2.0:
        lines.append(
            "  Strong evidence for per-session offsets. "
            "Implement Phase 1 of the v2 next-steps plan."
        )
    elif max_disagree > 1.0:
        lines.append(
            "  Moderate evidence for per-session offsets. "
            "Consider implementing Phase 1; could help."
        )
    else:
        lines.append(
            "  Weak evidence for per-session offsets. "
            "Inter-session variation is small relative to "
            "within-session scatter. The q_root_pos ceiling "
            "issue is likely caused by within-session marker "
            "drift instead — go to Phase 2 (latent drift state)."
        )

    report = "\n".join(lines)
    args.output.write_text(report)
    print()
    print(report)
    print()
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
