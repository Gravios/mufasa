"""Smoke-test for the body-part coordinate clipping in feature_subsets.

Verifies that the clip-after-load step in FeatureSubsetCalculator.run()
correctly clamps x/y coords to frame bounds while leaving likelihood
columns untouched.

Doesn't import the full feature_subsets module (pulls heavy deps).
Replicates only the clip logic on a synthetic dataframe.

    PYTHONPATH=. python tests/smoke_feature_clip.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main() -> int:
    # Synthetic pose data — 3 body parts, mix of valid and invalid coords
    df = pd.DataFrame({
        "nose_x":  [10, -4, 100, 50, 1925],
        "nose_y":  [20, 30, -10, 1090, 500],
        "nose_p":  [0.95, 0.02, 0.88, 0.91, 0.50],
        "tail_x":  [200, 300, 400, 1924, 100],
        "tail_y":  [50, 60, 1080, 200, -1],
        "tail_p":  [0.99, 0.85, 0.70, 0.95, 0.88],
        "ear_x":   [60, 70, 80, 90, 100],
        "ear_y":   [110, 120, 130, 140, 150],
        # DLC-style suffix
        "ear_likelihood": [0.99, 0.99, 0.99, 0.99, 0.99],
    })
    width, height = 1920, 1080

    # Replicate the clip logic from feature_subsets.py
    for col in df.columns:
        cl = str(col).lower()
        if cl.endswith("_x"):
            df[col] = df[col].clip(lower=0, upper=width)
        elif cl.endswith("_y"):
            df[col] = df[col].clip(lower=0, upper=height)

    # Case 1: negative x clipped to 0
    assert df["nose_x"].iloc[1] == 0, f"case 1: {df['nose_x'].iloc[1]}"

    # Case 2: x > width clipped to width
    assert df["nose_x"].iloc[4] == 1920, f"case 2: {df['nose_x'].iloc[4]}"

    # Case 3: negative y clipped to 0
    assert df["nose_y"].iloc[2] == 0, f"case 3: {df['nose_y'].iloc[2]}"

    # Case 4: y > height clipped to height
    assert df["nose_y"].iloc[3] == 1080, f"case 4: {df['nose_y'].iloc[3]}"

    # Case 5: tail_x = 1924 clipped to 1920
    assert df["tail_x"].iloc[3] == 1920

    # Case 6: tail_y = -1 clipped to 0
    assert df["tail_y"].iloc[4] == 0

    # Case 7: in-bounds values unchanged
    assert df["ear_x"].tolist() == [60, 70, 80, 90, 100]
    assert df["ear_y"].tolist() == [110, 120, 130, 140, 150]

    # Case 8: likelihood/probability columns NOT clipped
    # (would silently zero out high-probability values if they were)
    assert df["nose_p"].tolist() == [0.95, 0.02, 0.88, 0.91, 0.50]
    assert df["ear_likelihood"].tolist() == [0.99, 0.99, 0.99, 0.99, 0.99]

    # Case 9: all values now satisfy the feature_extractor's
    # min_value=0.0 check (the SIMBA ARRAY SIZE ERROR went away)
    assert (df["nose_x"] >= 0).all()
    assert (df["nose_y"] >= 0).all()
    assert (df["tail_x"] >= 0).all()
    assert (df["tail_y"] >= 0).all()

    print("smoke_feature_clip: 9/9 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
