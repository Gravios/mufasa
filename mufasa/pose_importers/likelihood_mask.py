"""
mufasa.pose_importers.likelihood_mask
=====================================

Likelihood-threshold masking for DeepLabCut pose data.

Context: DLC outputs a (x, y, likelihood) triplet per body-part per
frame. Likelihood is the network's confidence that the predicted
(x, y) is correct. For fast-moving or occluded body-parts, DLC will
still emit an (x, y) guess — often wildly wrong — with low likelihood.

Downstream analyses (distance, velocity, bout detection, etc.)
degrade badly when low-confidence coordinates are treated as real.
The existing cure — Mufasa's ``body_part_interpolator`` — only fires
for points that are already at the (0, 0) sentinel position.
Low-confidence DLC points never reach (0, 0); they just carry a
wrong-ish prediction plus a low likelihood that's silently ignored.

This module bridges the gap by zeroing out (x, y) pairs whose
likelihood falls below a user-set threshold, which plugs them into
the existing (0, 0) → interpolate-or-clip machinery without changes
to downstream code. The likelihood column itself is preserved for
traceability — a downstream analysis can still consult it if it
wants to (and it tells you, after the fact, how aggressive the mask
was).

Public API:

    apply_likelihood_threshold(df, threshold)
        -> (masked_df, counts_per_bp)

The input df is expected to have DLC's flat bp_header layout, i.e.
columns named ``<bp>_x``, ``<bp>_y``, ``<bp>_likelihood`` (in any
order, though DLC always writes them grouped by bp). No column-level
validation is performed here — the caller is responsible for having
flattened the DLC MultiIndex before calling this.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def apply_likelihood_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Zero out (x, y) pairs where ``<bp>_likelihood < threshold``.

    :param df: DataFrame with DLC bp_header columns
        (``<bp>_x``, ``<bp>_y``, ``<bp>_likelihood`` for each bp).
        The column set needs to include all three of x/y/likelihood
        per bp; bodyparts with any of the three missing are silently
        skipped (caller's responsibility to pass a valid frame).
    :param threshold: Likelihood floor in ``[0.0, 1.0]``. Points with
        likelihood **strictly less** than this are masked. A threshold
        of ``0.0`` is a no-op.
    :returns: ``(masked_df, counts)``. ``masked_df`` is a copy of the
        input with masked x/y set to 0.0. ``counts[bp]`` is how many
        frames were masked for each body-part — useful for telling
        the user "bp N dropped 12% of frames" so they can spot
        failing body-parts.

    The likelihood column itself is **not** modified — downstream
    code can still inspect it. Only x and y are zeroed.

    This function does not mutate its input.
    """
    if threshold <= 0.0:
        return df.copy(), {}
    if threshold > 1.0:
        # Defensive: a threshold above 1.0 masks everything. That's
        # almost certainly a user error (slider precision issue etc.)
        # so we cap rather than quietly wiping the data.
        threshold = 1.0

    out = df.copy()
    counts: Dict[str, int] = {}

    # Group columns by body-part stem. A column's body-part stem is
    # its name with the trailing _x / _y / _likelihood removed.
    bps: Dict[str, Dict[str, str]] = {}
    for col in out.columns:
        for suffix in ("_likelihood", "_x", "_y"):
            if col.endswith(suffix):
                stem = col[: -len(suffix)]
                bps.setdefault(stem, {})[suffix[1:]] = col
                break

    for stem, parts in bps.items():
        if not {"x", "y", "likelihood"}.issubset(parts.keys()):
            # Incomplete triplet — skip rather than corrupt.
            continue
        x_col, y_col, p_col = parts["x"], parts["y"], parts["likelihood"]
        mask = out[p_col] < threshold
        n = int(mask.sum())
        if n:
            out.loc[mask, x_col] = 0.0
            out.loc[mask, y_col] = 0.0
            counts[stem] = n

    return out, counts


def summarize_mask_counts(counts: Dict[str, int], n_frames: int) -> str:
    """Format a per-bp masking summary for stdout logging.

    Returns an empty string when ``counts`` is empty (no masking
    applied). Otherwise one line per body-part with a mask, ordered
    by descending count so the worst-tracked parts come first."""
    if not counts or n_frames <= 0:
        return ""
    lines = ["  Likelihood mask summary:"]
    for stem, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / n_frames
        lines.append(f"    {stem:<24s} {n:>6d} frames ({pct:5.1f}%)")
    return "\n".join(lines)


__all__ = [
    "apply_likelihood_threshold",
    "summarize_mask_counts",
]
