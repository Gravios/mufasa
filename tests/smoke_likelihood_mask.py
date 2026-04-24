"""Smoke-test for mufasa.pose_importers.likelihood_mask.

Covers the pure-Python masking logic that the DLC H5 importer uses
to zero out low-confidence (x, y) pairs.

Run headless:

    PYTHONPATH=. python tests/smoke_likelihood_mask.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def build_frame(n_frames: int = 10) -> pd.DataFrame:
    """2-bp fake DLC frame: nose + tail, flat bp_header layout."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "nose_x": rng.uniform(10, 500, n_frames),
        "nose_y": rng.uniform(10, 500, n_frames),
        "nose_likelihood": rng.uniform(0.0, 1.0, n_frames),
        "tail_x": rng.uniform(10, 500, n_frames),
        "tail_y": rng.uniform(10, 500, n_frames),
        "tail_likelihood": rng.uniform(0.0, 1.0, n_frames),
    })


def main() -> int:
    from mufasa.pose_importers.likelihood_mask import (
        apply_likelihood_threshold, summarize_mask_counts,
    )

    # ------------------------------------------------------------------ #
    # Case 1: threshold=0.0 is a no-op
    # ------------------------------------------------------------------ #
    df = build_frame(100)
    out, counts = apply_likelihood_threshold(df, threshold=0.0)
    assert out.equals(df), "case 1: no-op threshold mutated the frame"
    assert counts == {}, f"case 1: unexpected counts {counts}"

    # ------------------------------------------------------------------ #
    # Case 2: threshold=1.0 (hard ceiling) masks everything except
    # points with likelihood exactly 1.0. Use deterministic data.
    # ------------------------------------------------------------------ #
    df2 = pd.DataFrame({
        "a_x": [1.0, 2.0, 3.0],
        "a_y": [10.0, 20.0, 30.0],
        "a_likelihood": [0.5, 0.9, 1.0],
    })
    out2, counts2 = apply_likelihood_threshold(df2, threshold=1.0)
    assert out2["a_x"].tolist() == [0.0, 0.0, 3.0]
    assert out2["a_y"].tolist() == [0.0, 0.0, 30.0]
    assert out2["a_likelihood"].tolist() == [0.5, 0.9, 1.0], (
        "case 2: likelihood column must not be modified"
    )
    assert counts2 == {"a": 2}, f"case 2: {counts2}"

    # ------------------------------------------------------------------ #
    # Case 3: partial mask at threshold=0.7
    # ------------------------------------------------------------------ #
    df3 = pd.DataFrame({
        "b_x": [100.0, 200.0, 300.0, 400.0],
        "b_y": [10.0, 20.0, 30.0, 40.0],
        "b_likelihood": [0.1, 0.5, 0.7, 0.99],
    })
    out3, counts3 = apply_likelihood_threshold(df3, threshold=0.7)
    # strictly-less-than: 0.7 is kept, 0.1 and 0.5 are zeroed
    assert out3["b_x"].tolist() == [0.0, 0.0, 300.0, 400.0]
    assert out3["b_y"].tolist() == [0.0, 0.0, 30.0, 40.0]
    assert counts3 == {"b": 2}, f"case 3: {counts3}"

    # ------------------------------------------------------------------ #
    # Case 4: multi-bp behaves independently
    # ------------------------------------------------------------------ #
    df4 = pd.DataFrame({
        "nose_x": [1.0, 2.0, 3.0],
        "nose_y": [1.0, 2.0, 3.0],
        "nose_likelihood": [0.1, 0.9, 0.9],
        "tail_x": [10.0, 20.0, 30.0],
        "tail_y": [10.0, 20.0, 30.0],
        "tail_likelihood": [0.9, 0.9, 0.1],
    })
    out4, counts4 = apply_likelihood_threshold(df4, threshold=0.5)
    assert out4["nose_x"].tolist() == [0.0, 2.0, 3.0]
    assert out4["tail_x"].tolist() == [10.0, 20.0, 0.0]
    assert counts4 == {"nose": 1, "tail": 1}, f"case 4: {counts4}"

    # ------------------------------------------------------------------ #
    # Case 5: input dict is not mutated
    # ------------------------------------------------------------------ #
    df5 = pd.DataFrame({
        "c_x": [1.0, 2.0],
        "c_y": [3.0, 4.0],
        "c_likelihood": [0.1, 0.9],
    })
    before_x = df5["c_x"].tolist()
    _ = apply_likelihood_threshold(df5, threshold=0.5)
    assert df5["c_x"].tolist() == before_x, (
        "case 5: input df was mutated"
    )

    # ------------------------------------------------------------------ #
    # Case 6: incomplete triplet silently skipped (no crash)
    # ------------------------------------------------------------------ #
    df6 = pd.DataFrame({
        "a_x": [1.0, 2.0],
        "a_y": [3.0, 4.0],
        # no a_likelihood — triplet incomplete
    })
    out6, counts6 = apply_likelihood_threshold(df6, threshold=0.5)
    assert out6.equals(df6), "case 6: incomplete triplet should be left alone"
    assert counts6 == {}, f"case 6: {counts6}"

    # ------------------------------------------------------------------ #
    # Case 7: summarize_mask_counts formatting
    # ------------------------------------------------------------------ #
    summary = summarize_mask_counts({"nose": 5, "tail": 1}, n_frames=100)
    assert "nose" in summary and "tail" in summary
    # Descending order: nose (5) before tail (1)
    assert summary.index("nose") < summary.index("tail")
    assert "5.0%" in summary  # 5/100 formatted
    assert summarize_mask_counts({}, 100) == ""

    # ------------------------------------------------------------------ #
    # Case 8: out-of-range threshold clamp
    # ------------------------------------------------------------------ #
    df8 = pd.DataFrame({
        "a_x": [1.0, 2.0],
        "a_y": [3.0, 4.0],
        "a_likelihood": [0.5, 0.99],
    })
    # threshold > 1.0 should be capped to 1.0 rather than masking
    # everything (defensive against slider precision bugs)
    out8, counts8 = apply_likelihood_threshold(df8, threshold=1.5)
    # Capped to 1.0 -> only points with likelihood < 1.0 masked
    assert out8["a_x"].tolist() == [0.0, 0.0]  # both < 1.0
    assert counts8 == {"a": 2}

    # ------------------------------------------------------------------ #
    # Case 9: Mufasa _p suffix convention works identically to DLC
    # _likelihood. This is the regression case that actually shipped
    # broken in the first likelihood_threshold_fix iteration — the
    # DLC H5 importer applies df.columns = self.bp_headers BEFORE
    # calling the mask, and bp_headers uses _p not _likelihood. The
    # mask silently produced zero counts because no triplet matched.
    # ------------------------------------------------------------------ #
    df9 = pd.DataFrame({
        "nose_x": [100.0, 200.0, 300.0],
        "nose_y": [10.0, 20.0, 30.0],
        "nose_p": [0.1, 0.5, 0.99],      # <-- _p, not _likelihood
        "tail_x": [1.0, 2.0, 3.0],
        "tail_y": [4.0, 5.0, 6.0],
        "tail_p": [0.99, 0.99, 0.1],
    })
    out9, counts9 = apply_likelihood_threshold(df9, threshold=0.5)
    # strict less-than: nose[0] (p=0.1) masked, nose[1] (p=0.5) kept
    assert out9["nose_x"].tolist() == [0.0, 200.0, 300.0]
    assert out9["nose_y"].tolist() == [0.0, 20.0, 30.0]
    assert out9["nose_p"].tolist() == [0.1, 0.5, 0.99], (
        "likelihood (_p) column must not be modified"
    )
    assert out9["tail_x"].tolist() == [1.0, 2.0, 0.0]
    assert counts9 == {"nose": 1, "tail": 1}, f"case 9: {counts9}"

    # ------------------------------------------------------------------ #
    # Case 10: mixed conventions in the same frame are supported.
    # Not expected in practice but the suffix ambiguity shouldn't
    # cause silent data loss if it ever happens.
    # ------------------------------------------------------------------ #
    df10 = pd.DataFrame({
        "a_x": [1.0, 2.0],
        "a_y": [1.0, 2.0],
        "a_likelihood": [0.1, 0.9],
        "b_x": [1.0, 2.0],
        "b_y": [1.0, 2.0],
        "b_p": [0.1, 0.9],
    })
    out10, counts10 = apply_likelihood_threshold(df10, threshold=0.5)
    assert counts10 == {"a": 1, "b": 1}, f"case 10: {counts10}"

    print("smoke_likelihood_mask: 10/10 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
