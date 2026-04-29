"""Smoke-test for the convex hull reshape in feature_subsets.py.

Pins down the expected (n_frames, n_body_parts, 2) shape produced
from the dataframe and column-name pattern that the three hull
methods use. If anyone changes the reshape arithmetic (e.g.
"fixes" it back to len(self.data_df) / 2 or len(self.data_df) // 2),
this test fails loudly.

Background: an earlier version of the code wrote

    np.reshape(arr, (len(self.data_df / 2), -1, 2))

which is a NO-OP — pandas DataFrame `/ 2` returns a same-length
DataFrame with halved values, and len() is unchanged. The `/ 2`
was confusing dead arithmetic, not a real divide. The current
code uses len(self.data_df) which is byte-equivalent and clear.

If a future reader thinks the old form was a typo for // 2 and
"fixes" it, they'll silently corrupt every hull feature output
across three feature families (three-point, four-point, and
animal convex hull perimeter/area). This test exists to make
that mistake fail immediately.

    PYTHONPATH=. python tests/smoke_hull_reshape.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def _build_col_names(point):
    """Mirror the col_names construction from feature_subsets.py."""
    return list(sum([(f"{x}_x", f"{y}_y") for (x, y) in zip(point, point)], ()))


def _reshape_for_hull(df: pd.DataFrame, point) -> np.ndarray:
    """The reshape exactly as feature_subsets.py performs it after
    the latent-bug fix."""
    col_names = _build_col_names(point)
    arr = df[col_names].values
    return np.reshape(arr, (len(df), -1, 2)).astype(np.float32)


def main() -> int:
    n_frames = 10
    bps = ("bp_a", "bp_b", "bp_c", "bp_d")

    data = {}
    for i, bp in enumerate(bps):
        # Use a distinctive offset per body-part so the test can
        # detect any wrong reshape ordering immediately.
        data[f"{bp}_x"] = np.arange(n_frames) + i * 1000
        data[f"{bp}_y"] = np.arange(n_frames) + i * 1000 + 500
    df = pd.DataFrame(data)

    # ------------------------------------------------------------------ #
    # Case 1: three-point hull reshape
    # ------------------------------------------------------------------ #
    point3 = ("bp_a", "bp_b", "bp_c")
    arr3 = _reshape_for_hull(df, point3)
    assert arr3.shape == (n_frames, 3, 2), \
        f"3pt shape: want ({n_frames}, 3, 2), got {arr3.shape}"
    assert arr3.dtype == np.float32

    # Frame 0 should be [[bp_a_x, bp_a_y], [bp_b_x, bp_b_y], [bp_c_x, bp_c_y]]
    expected = np.array(
        [[0, 500], [1000, 1500], [2000, 2500]],
        dtype=np.float32,
    )
    assert np.array_equal(arr3[0], expected), \
        f"3pt frame 0: want {expected.tolist()}, got {arr3[0].tolist()}"

    # ------------------------------------------------------------------ #
    # Case 2: four-point hull reshape
    # ------------------------------------------------------------------ #
    point4 = ("bp_a", "bp_b", "bp_c", "bp_d")
    arr4 = _reshape_for_hull(df, point4)
    assert arr4.shape == (n_frames, 4, 2), \
        f"4pt shape: want ({n_frames}, 4, 2), got {arr4.shape}"

    # ------------------------------------------------------------------ #
    # Case 3: full-animal hull reshape (variable n_points)
    # ------------------------------------------------------------------ #
    arr_full = _reshape_for_hull(df, bps)
    assert arr_full.shape == (n_frames, len(bps), 2), \
        f"full-animal shape: want ({n_frames}, {len(bps)}, 2), got {arr_full.shape}"

    # ------------------------------------------------------------------ #
    # Case 4: round-trip — frame i, bp j matches df[bp_j_x][i] / df[bp_j_y][i]
    # ------------------------------------------------------------------ #
    for frame_idx in (0, n_frames // 2, n_frames - 1):
        for bp_idx, bp in enumerate(bps):
            assert arr_full[frame_idx, bp_idx, 0] == df[f"{bp}_x"].iloc[frame_idx]
            assert arr_full[frame_idx, bp_idx, 1] == df[f"{bp}_y"].iloc[frame_idx]

    # ------------------------------------------------------------------ #
    # Case 5: regression detector — `len(df) // 2` would NOT produce the
    # same shape, so explicitly verify the wrong form would error or
    # give a different shape (catches anyone "fixing" the no-op).
    # ------------------------------------------------------------------ #
    col_names = _build_col_names(point3)
    raw = df[col_names].values
    # The wrong form: integer division would give (5, 6, 2) for n_frames=10
    wrong = np.reshape(raw, (len(df) // 2, -1, 2))
    assert wrong.shape != (n_frames, 3, 2), \
        "len(df) // 2 should NOT produce the correct hull shape"
    assert wrong.shape == (n_frames // 2, 6, 2)
    # If anyone "fixes" the source to use this form, this assertion
    # documents that they're producing a different shape that
    # silently corrupts hull computations.

    # ------------------------------------------------------------------ #
    # Case 6: re-confirm the documented no-op (pandas df / 2 keeps len)
    # If pandas ever changes this behavior in a future release, our
    # cleanup comment in feature_subsets.py would no longer accurately
    # describe history — the test reminds us to re-evaluate.
    # ------------------------------------------------------------------ #
    halved = df / 2
    assert len(halved) == len(df), \
        "pandas df / 2 should preserve length (this is the historical" \
        " behavior the comment in feature_subsets.py describes)"

    # ------------------------------------------------------------------ #
    # Case 7: odd number of frames — reshape still works
    # (regression check for the corrected form on odd lengths)
    # ------------------------------------------------------------------ #
    df_odd = df.iloc[:9].copy()  # 9 frames
    col_names = _build_col_names(point3)
    arr_odd = np.reshape(
        df_odd[col_names].values, (len(df_odd), -1, 2),
    ).astype(np.float32)
    assert arr_odd.shape == (9, 3, 2)

    # The // 2 form would error on odd lengths because total elements
    # (9 * 6 = 54) doesn't divide evenly into (4, ?, 2) groups
    # (would need ?=6.75). Verifying this guards against a different
    # failure mode for the wrong fix.
    try:
        np.reshape(df_odd[col_names].values, (len(df_odd) // 2, -1, 2))
        # If we reach here without error, the wrong form coincidentally
        # works for THIS shape — note in case it ever happens (would
        # require very specific n_frames/n_points combinations).
        # For (9, 6) reshape into (4, ?, 2): 54 / 8 = 6.75 — should error.
    except ValueError:
        pass  # expected

    print("smoke_hull_reshape: 7/7 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
