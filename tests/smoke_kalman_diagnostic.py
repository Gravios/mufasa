"""Smoke test for the Kalman pose smoother diagnostic tool.

Builds a synthetic pose CSV, runs the diagnostic on it, and
verifies:
- All 5 plots are produced
- summary.json is valid + has expected keys
- recommendation.txt has one of the expected outcomes
- Per-marker stats are reasonable for the synthetic data
- The recommendation logic correctly classifies known cases

Tests three synthetic regimes designed to exercise different
branches of the recommendation logic:

- "clean": all markers have p > 0.99 always — should recommend
  NOT building.
- "dropouts": markers occasionally drop to p ≈ 0 for ~30 frames
  — should recommend building Kalman.
- "loose-rigidity": rigid-pair distances vary widely — should
  recommend v1-only (no triplet prior).

    PYTHONPATH=. python tests/smoke_kalman_diagnostic.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Headless matplotlib for tests
os.environ.setdefault("MPLBACKEND", "Agg")

from mufasa.data_processors.kalman_diagnostic import (
    classify_velocity_modality,
    compute_head_velocity,
    compute_marker_stats,
    compute_rigid_pair_stats,
    detect_marker_columns,
    load_pose_csv,
    make_recommendation,
    run_diagnostic,
)


def _build_synthetic_csv(
    path: Path,
    n_frames: int = 5000,
    likelihood_pattern: str = "clean",
    rigidity_cv: float = 0.02,
) -> None:
    """Build a synthetic pose CSV with known characteristics.

    likelihood_pattern:
      - "clean": all p ≈ 0.99
      - "dropouts": p drops to 0 for ~30-frame runs every ~500 frames
      - "noisy": p uniform in [0.5, 1.0]

    rigidity_cv controls the std of inter-marker distance as a
    fraction of mean distance. Low → tight rigidity (CV ~0.02).
    High → loose rigidity (CV ~0.20).
    """
    rng = np.random.default_rng(42)
    t = np.arange(n_frames)

    # Animal trajectory: slow drift in a square arena
    cx = 200 + 100 * np.sin(t / 500)
    cy = 200 + 100 * np.cos(t / 500)

    # Mean inter-marker offsets (defines anatomy)
    nose_offset = (15.0, 0.0)
    ear_l_offset = (-5.0, -8.0)
    ear_r_offset = (-5.0, 8.0)
    center_offset = (0.0, 0.0)

    def jitter(mean_d):
        return rng.normal(0, mean_d * rigidity_cv, n_frames)

    nose_x = cx + nose_offset[0] + jitter(15.0)
    nose_y = cy + nose_offset[1] + jitter(15.0)
    ear_l_x = cx + ear_l_offset[0] + jitter(8.0)
    ear_l_y = cy + ear_l_offset[1] + jitter(8.0)
    ear_r_x = cx + ear_r_offset[0] + jitter(8.0)
    ear_r_y = cy + ear_r_offset[1] + jitter(8.0)
    center_x = cx + center_offset[0] + jitter(2.0)
    center_y = cy + center_offset[1] + jitter(2.0)

    # Likelihoods
    if likelihood_pattern == "clean":
        nose_p = np.full(n_frames, 0.99)
        ear_l_p = np.full(n_frames, 0.99)
        ear_r_p = np.full(n_frames, 0.99)
        center_p = np.full(n_frames, 0.99)
    elif likelihood_pattern == "dropouts":
        nose_p = np.full(n_frames, 0.95)
        for start in range(500, n_frames, 500):
            end = min(start + 30, n_frames)
            nose_p[start:end] = 0.0
        # Other markers cleaner
        ear_l_p = np.full(n_frames, 0.97)
        ear_r_p = np.full(n_frames, 0.97)
        center_p = np.full(n_frames, 0.99)
    elif likelihood_pattern == "noisy":
        nose_p = rng.uniform(0.5, 1.0, n_frames)
        ear_l_p = rng.uniform(0.5, 1.0, n_frames)
        ear_r_p = rng.uniform(0.5, 1.0, n_frames)
        center_p = rng.uniform(0.5, 1.0, n_frames)
    else:
        raise ValueError(f"Unknown pattern {likelihood_pattern}")

    df = pd.DataFrame({
        "Nose_x": nose_x, "Nose_y": nose_y, "Nose_p": nose_p,
        "Ear_left_x": ear_l_x, "Ear_left_y": ear_l_y, "Ear_left_p": ear_l_p,
        "Ear_right_x": ear_r_x, "Ear_right_y": ear_r_y,
        "Ear_right_p": ear_r_p,
        "Center_x": center_x, "Center_y": center_y, "Center_p": center_p,
    })
    df.to_csv(path, index=True)


def main() -> int:
    # ---------------------------------------------------------- #
    # Case 1: marker column detection
    # ---------------------------------------------------------- #
    df = pd.DataFrame({
        "Nose_x": [1, 2], "Nose_y": [3, 4], "Nose_p": [0.9, 0.95],
        "Tail_x": [5, 6], "Tail_y": [7, 8], "Tail_p": [0.8, 0.85],
        "Bad_x": [0, 0], "Bad_y": [0, 0],   # missing _p — should skip
    })
    df.columns = [c.lower() for c in df.columns]
    markers = detect_marker_columns(df)
    assert "nose" in markers and "tail" in markers, (
        f"Should detect nose+tail; got {markers}"
    )
    assert "bad" not in markers, "Should skip incomplete markers"

    # ---------------------------------------------------------- #
    # Case 2: load_pose_csv with Mufasa flat-header CSV
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "test.csv"
        _build_synthetic_csv(csv, n_frames=200, likelihood_pattern="clean")
        loaded_df, markers = load_pose_csv(str(csv))
        assert len(loaded_df) == 200
        assert "nose" in markers
        assert "ear_left" in markers
        assert "ear_right" in markers

    # ---------------------------------------------------------- #
    # Case 3: per-marker stats reasonable for clean data
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "clean.csv"
        _build_synthetic_csv(csv, n_frames=1000, likelihood_pattern="clean")
        df, _ = load_pose_csv(str(csv))
        stats = compute_marker_stats(df, "nose", likelihood_threshold=0.95)
        assert stats.frac_high > 0.99, (
            f"Clean data should have ~all frames high-p; "
            f"got frac_high={stats.frac_high}"
        )
        assert stats.longest_low_run == 0, (
            f"Clean data should have no low-p runs; "
            f"got longest={stats.longest_low_run}"
        )

    # ---------------------------------------------------------- #
    # Case 4: per-marker stats catch dropouts
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "dropouts.csv"
        _build_synthetic_csv(
            csv, n_frames=2000, likelihood_pattern="dropouts",
        )
        df, _ = load_pose_csv(str(csv))
        stats = compute_marker_stats(df, "nose", likelihood_threshold=0.95)
        assert stats.frac_zero > 0.03, (
            f"Dropout pattern should have non-trivial frac_zero; "
            f"got {stats.frac_zero}"
        )
        assert stats.longest_low_run >= 25, (
            f"Dropout pattern should have ~30-frame runs; "
            f"got longest={stats.longest_low_run}"
        )

    # ---------------------------------------------------------- #
    # Case 5: rigid pair stats reasonable for tight rigidity
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "tight.csv"
        _build_synthetic_csv(
            csv, n_frames=2000, likelihood_pattern="clean",
            rigidity_cv=0.02,
        )
        df, _ = load_pose_csv(str(csv))
        stats = compute_rigid_pair_stats(
            df, "ear_left", "ear_right", likelihood_threshold=0.95,
        )
        assert "warning" not in stats, (
            f"Should have enough samples; got {stats}"
        )
        assert stats["cv_distance"] < 0.10, (
            f"Tight rigidity should have low CV; got {stats['cv_distance']}"
        )

    # ---------------------------------------------------------- #
    # Case 6: head velocity computation
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "vel.csv"
        _build_synthetic_csv(csv, n_frames=2000, likelihood_pattern="clean")
        df, _ = load_pose_csv(str(csv))
        vx_h, vy_h, valid_mask = compute_head_velocity(
            df, ["nose", "ear_left", "ear_right"],
            likelihood_threshold=0.95, fps=30.0,
        )
        assert vx_h.shape == (2000,)
        assert valid_mask.sum() > 100, (
            f"Should have many valid velocity samples; "
            f"got {valid_mask.sum()}"
        )

    # ---------------------------------------------------------- #
    # Case 7: velocity modality classification
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(0)
    # Unimodal: single Gaussian
    v_unimodal = rng.normal(0, 5, 5000)
    mask = np.ones(len(v_unimodal), dtype=bool)
    assert classify_velocity_modality(v_unimodal, mask) == "unimodal"

    # ---------------------------------------------------------- #
    # Case 8: full diagnostic on clean data → recommends NOT build
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "clean.csv"
        _build_synthetic_csv(csv, n_frames=3000, likelihood_pattern="clean")
        out_dir = Path(tmp) / "out"
        report = run_diagnostic(
            csv_path=str(csv),
            output_dir=str(out_dir),
            fps=30.0,
        )
        # All 5 plots produced
        for fname in [
            "01_likelihood.png",
            "02_dropouts.png",
            "03_rigid_pairs.png",
            "04_velocity.png",
            "05_velocity_vs_config.png",
        ]:
            assert (out_dir / fname).is_file(), f"Missing plot: {fname}"
        # summary.json valid
        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["n_frames"] == 3000
        assert summary["n_markers"] == 4
        # Recommendation: clean data → don't build
        rec = (out_dir / "recommendation.txt").read_text()
        assert "do NOT build" in rec, (
            f"Clean data should recommend NOT building; got:\n{rec}"
        )

    # ---------------------------------------------------------- #
    # Case 9: full diagnostic on dropout data → recommends building
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "drop.csv"
        _build_synthetic_csv(
            csv, n_frames=5000, likelihood_pattern="dropouts",
            rigidity_cv=0.03,
        )
        out_dir = Path(tmp) / "out"
        report = run_diagnostic(
            csv_path=str(csv),
            output_dir=str(out_dir),
            fps=30.0,
        )
        rec = (out_dir / "recommendation.txt").read_text()
        # With dropouts + tight rigidity, should recommend building
        # something (either v1-only or full v1+v2). Don't pin
        # exactly which because velocity-modality classification
        # on synthetic data is a heuristic.
        assert "do NOT build" not in rec, (
            f"Dropout data should NOT recommend skipping; got:\n{rec}"
        )
        assert "build" in rec.lower(), (
            f"Dropout data should recommend building; got:\n{rec}"
        )

    # ---------------------------------------------------------- #
    # Case 10: full diagnostic on loose-rigidity → v1 only
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "loose.csv"
        _build_synthetic_csv(
            csv, n_frames=5000, likelihood_pattern="dropouts",
            rigidity_cv=0.30,  # very loose — CV will be ~0.30
        )
        out_dir = Path(tmp) / "out"
        report = run_diagnostic(
            csv_path=str(csv),
            output_dir=str(out_dir),
            fps=30.0,
        )
        rec = (out_dir / "recommendation.txt").read_text()
        # Loose rigidity → v1 only (no triplet prior)
        assert "v1 only" in rec or "triplet covariance" in rec.lower(), (
            f"Loose rigidity should recommend v1-only or "
            f"discuss triplet limitations; got:\n{rec}"
        )

    print("smoke_kalman_diagnostic: 10/10 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
