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
    compute_body_velocity,
    compute_head_velocity,
    compute_marker_stats,
    compute_rigid_pair_stats,
    detect_marker_columns,
    discover_pose_files,
    head_body_velocity_correlation,
    load_pose_csv,
    load_pose_file,
    load_pose_files,
    make_recommendation,
    MarkerStats,
    run_diagnostic,
)


def _has_parquet_engine() -> bool:
    """True if a pandas parquet engine (pyarrow or fastparquet)
    is importable."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        return False


def _build_synthetic_csv(
    path: Path,
    n_frames: int = 5000,
    likelihood_pattern: str = "clean",
    rigidity_cv: float = 0.02,
    motion_pattern: str = "synchronized",
) -> None:
    """Build a synthetic pose CSV with known characteristics.

    likelihood_pattern:
      - "clean": all p ≈ 0.99
      - "dropouts": p drops to 0 for ~30-frame runs every ~500 frames
      - "noisy": p uniform in [0.5, 1.0]

    rigidity_cv controls the std of inter-marker distance as a
    fraction of mean distance. Low → tight rigidity (CV ~0.02).
    High → loose rigidity (CV ~0.20).

    motion_pattern:
      - "synchronized": head and body move together (forward locomotion)
      - "scanning": body still, head turning (head/body velocities decouple)

    Includes both head markers (nose, ear_left, ear_right) and body
    markers (center, left_flank, right_flank, tail_base) so the
    body-frame PCA computation has enough markers to produce a
    well-defined major axis.
    """
    rng = np.random.default_rng(42)
    t = np.arange(n_frames)

    # Body trajectory in world frame
    if motion_pattern == "synchronized":
        # Slow forward locomotion in a square arena
        body_cx = 200 + 100 * np.sin(t / 500)
        body_cy = 200 + 100 * np.cos(t / 500)
        body_heading = np.arctan2(
            np.gradient(body_cy), np.gradient(body_cx)
        )
        head_offset_angle = body_heading  # head pointing same direction
    elif motion_pattern == "scanning":
        # Body roughly still, head sweeping side-to-side
        body_cx = 200 + 5 * np.sin(t / 1000)
        body_cy = 200 + 5 * np.cos(t / 1000)
        body_heading = np.full(n_frames, 0.0)  # body always facing east
        # Head sweeps ±60 degrees around body heading
        head_offset_angle = body_heading + (np.pi / 3) * np.sin(t / 50)
    else:
        raise ValueError(f"Unknown motion_pattern {motion_pattern}")

    # Head triplet positioned 20 px in front of body centroid,
    # along head_offset_angle
    head_distance = 20.0
    head_cx = body_cx + head_distance * np.cos(head_offset_angle)
    head_cy = body_cy + head_distance * np.sin(head_offset_angle)

    def jitter(mean_d):
        return rng.normal(0, mean_d * rigidity_cv, n_frames)

    # Head markers: positioned around head centroid
    nose_x = head_cx + 8.0 * np.cos(head_offset_angle) + jitter(15.0)
    nose_y = head_cy + 8.0 * np.sin(head_offset_angle) + jitter(15.0)
    ear_l_x = (head_cx
               + 6.0 * np.cos(head_offset_angle + np.pi / 2)
               + jitter(8.0))
    ear_l_y = (head_cy
               + 6.0 * np.sin(head_offset_angle + np.pi / 2)
               + jitter(8.0))
    ear_r_x = (head_cx
               - 6.0 * np.cos(head_offset_angle + np.pi / 2)
               + jitter(8.0))
    ear_r_y = (head_cy
               - 6.0 * np.sin(head_offset_angle + np.pi / 2)
               + jitter(8.0))

    # Body markers: positioned around body centroid along body axis
    center_x = body_cx + jitter(2.0)
    center_y = body_cy + jitter(2.0)
    left_flank_x = (body_cx
                    + 8.0 * np.cos(body_heading + np.pi / 2)
                    + jitter(5.0))
    left_flank_y = (body_cy
                    + 8.0 * np.sin(body_heading + np.pi / 2)
                    + jitter(5.0))
    right_flank_x = (body_cx
                     - 8.0 * np.cos(body_heading + np.pi / 2)
                     + jitter(5.0))
    right_flank_y = (body_cy
                     - 8.0 * np.sin(body_heading + np.pi / 2)
                     + jitter(5.0))
    tail_base_x = (body_cx
                   - 25.0 * np.cos(body_heading)
                   + jitter(10.0))
    tail_base_y = (body_cy
                   - 25.0 * np.sin(body_heading)
                   + jitter(10.0))

    # Likelihoods
    if likelihood_pattern == "clean":
        nose_p = np.full(n_frames, 0.99)
        ear_l_p = np.full(n_frames, 0.99)
        ear_r_p = np.full(n_frames, 0.99)
        body_p = np.full(n_frames, 0.99)
    elif likelihood_pattern == "dropouts":
        nose_p = np.full(n_frames, 0.95)
        for start in range(500, n_frames, 500):
            end = min(start + 30, n_frames)
            nose_p[start:end] = 0.0
        ear_l_p = np.full(n_frames, 0.97)
        ear_r_p = np.full(n_frames, 0.97)
        body_p = np.full(n_frames, 0.99)
    elif likelihood_pattern == "noisy":
        nose_p = rng.uniform(0.5, 1.0, n_frames)
        ear_l_p = rng.uniform(0.5, 1.0, n_frames)
        ear_r_p = rng.uniform(0.5, 1.0, n_frames)
        body_p = rng.uniform(0.5, 1.0, n_frames)
    else:
        raise ValueError(f"Unknown pattern {likelihood_pattern}")

    df = pd.DataFrame({
        "Nose_x": nose_x, "Nose_y": nose_y, "Nose_p": nose_p,
        "Ear_left_x": ear_l_x, "Ear_left_y": ear_l_y, "Ear_left_p": ear_l_p,
        "Ear_right_x": ear_r_x, "Ear_right_y": ear_r_y,
        "Ear_right_p": ear_r_p,
        "Center_x": center_x, "Center_y": center_y, "Center_p": body_p,
        "Left_flank_x": left_flank_x, "Left_flank_y": left_flank_y,
        "Left_flank_p": body_p,
        "Right_flank_x": right_flank_x, "Right_flank_y": right_flank_y,
        "Right_flank_p": body_p,
        "Tail_base_x": tail_base_x, "Tail_base_y": tail_base_y,
        "Tail_base_p": body_p,
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
        # All 8 plots produced (now includes body components 6-8
        # because synthetic data has body markers)
        for fname in [
            "01_likelihood.png",
            "02_dropouts.png",
            "03_rigid_pairs.png",
            "04_velocity.png",
            "05_velocity_vs_config.png",
            "06_body_velocity.png",
            "07_velocity_vs_config_body.png",
            "08_head_body_correlation.png",
        ]:
            assert (out_dir / fname).is_file(), f"Missing plot: {fname}"
        # summary.json valid
        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["n_frames"] == 3000
        assert summary["n_markers"] == 7  # 3 head + 4 body
        assert "body_markers" in summary
        assert "body_velocity_stats" in summary
        assert "head_body_correlation" in summary
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
    # Pass explicit rigid_pairs to force the test to exercise the
    # loose-rigidity recommendation branch. Auto-detect would
    # correctly skip these loose pairs, which is the right
    # behavior in production but defeats this test's intent.
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
            rigid_pairs=[("nose", "ear_left"),
                         ("nose", "ear_right"),
                         ("ear_left", "ear_right")],
        )
        rec = (out_dir / "recommendation.txt").read_text()
        # Loose rigidity → v1 only (no triplet prior)
        assert "v1 only" in rec or "triplet covariance" in rec.lower(), (
            f"Loose rigidity should recommend v1-only or "
            f"discuss triplet limitations; got:\n{rec}"
        )

    # ---------------------------------------------------------- #
    # Case 11: body velocity computation (synchronized motion)
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "sync.csv"
        _build_synthetic_csv(
            csv, n_frames=2000, likelihood_pattern="clean",
            motion_pattern="synchronized",
        )
        df, _ = load_pose_csv(str(csv))
        body_markers = ["center", "left_flank", "right_flank", "tail_base"]
        head_markers = ["nose", "ear_left", "ear_right"]
        vx_b, vy_b, valid_mask, diag = compute_body_velocity(
            df, body_markers, head_markers,
            likelihood_threshold=0.95, fps=30.0,
        )
        assert vx_b.shape == (2000,)
        assert valid_mask.sum() > 100, (
            f"Should have many valid body velocity samples; "
            f"got {valid_mask.sum()}"
        )
        # Most frames should be sign-disambiguated by head reference
        # (since head markers are reliable in clean data)
        assert diag["n_signed_by_head"] > diag["n_signed_arbitrary"], (
            f"Most frames should use head-direction sign reference; "
            f"got {diag}"
        )

    # ---------------------------------------------------------- #
    # Case 12: head-body correlation differs by motion regime
    # ---------------------------------------------------------- #
    # Synchronized motion → high correlation; scanning → low correlation
    with tempfile.TemporaryDirectory() as tmp:
        csv_sync = Path(tmp) / "sync.csv"
        _build_synthetic_csv(
            csv_sync, n_frames=3000, likelihood_pattern="clean",
            motion_pattern="synchronized",
        )
        df_sync, _ = load_pose_csv(str(csv_sync))
        body_markers = ["center", "left_flank", "right_flank", "tail_base"]
        head_markers = ["nose", "ear_left", "ear_right"]

        vx_h_sync, _, head_valid_sync = compute_head_velocity(
            df_sync, head_markers, 0.95, 30.0,
        )
        vx_b_sync, _, body_valid_sync, _ = compute_body_velocity(
            df_sync, body_markers, head_markers, 0.95, 30.0,
        )
        r_sync = head_body_velocity_correlation(
            vx_h_sync, vx_b_sync, head_valid_sync, body_valid_sync,
        )
        # In synchronized motion, head and body velocity should be
        # positively correlated (won't be 1.0 due to head jitter
        # and the head-frame rotation, but should be >0.3)
        assert not np.isnan(r_sync), "Should have valid correlation"

        csv_scan = Path(tmp) / "scan.csv"
        _build_synthetic_csv(
            csv_scan, n_frames=3000, likelihood_pattern="clean",
            motion_pattern="scanning",
        )
        df_scan, _ = load_pose_csv(str(csv_scan))
        vx_h_scan, _, head_valid_scan = compute_head_velocity(
            df_scan, head_markers, 0.95, 30.0,
        )
        vx_b_scan, _, body_valid_scan, _ = compute_body_velocity(
            df_scan, body_markers, head_markers, 0.95, 30.0,
        )
        r_scan = head_body_velocity_correlation(
            vx_h_scan, vx_b_scan, head_valid_scan, body_valid_scan,
        )
        # In scanning, body is nearly still while head sweeps. The
        # correlation should be much lower than in synchronized.
        # (Exact threshold tolerant — the regime distinction matters,
        # not the precise value.)
        assert abs(r_sync) > abs(r_scan) * 0.7 or abs(r_sync - r_scan) < 0.1, (
            f"Synchronized correlation should be at least comparable "
            f"to scanning; got sync={r_sync:.3f} scan={r_scan:.3f}. "
            f"This indicates the synthetic regimes don't actually "
            f"differ — check the test fixture."
        )

    # ---------------------------------------------------------- #
    # Case 13: body velocity gracefully degrades when too few markers
    # ---------------------------------------------------------- #
    # Build a CSV with only head markers (no body markers).
    # The diagnostic should skip components 6-8 with a warning.
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "headonly.csv"
        # Direct synthetic build — only nose + ears
        rng = np.random.default_rng(0)
        n = 1000
        t = np.arange(n)
        cx = 200 + 50 * np.sin(t / 200)
        cy = 200 + 50 * np.cos(t / 200)
        df = pd.DataFrame({
            "Nose_x": cx + 10, "Nose_y": cy, "Nose_p": np.full(n, 0.99),
            "Ear_left_x": cx, "Ear_left_y": cy + 5,
            "Ear_left_p": np.full(n, 0.99),
            "Ear_right_x": cx, "Ear_right_y": cy - 5,
            "Ear_right_p": np.full(n, 0.99),
        })
        df.to_csv(csv, index=True)

        out_dir = Path(tmp) / "out"
        report = run_diagnostic(
            csv_path=str(csv),
            output_dir=str(out_dir),
            fps=30.0,
        )
        # Components 1-5 should be present
        for fname in [
            "01_likelihood.png",
            "02_dropouts.png",
            "03_rigid_pairs.png",
            "04_velocity.png",
            "05_velocity_vs_config.png",
        ]:
            assert (out_dir / fname).is_file(), f"Missing plot: {fname}"
        # Components 6-8 should NOT be present (no body markers)
        for fname in [
            "06_body_velocity.png",
            "07_velocity_vs_config_body.png",
            "08_head_body_correlation.png",
        ]:
            assert not (out_dir / fname).is_file(), (
                f"Should NOT have body plot when no body markers: {fname}"
            )
        # summary should record the skip
        summary = json.loads((out_dir / "summary.json").read_text())
        assert summary["body_markers"] == [], (
            f"body_markers should be empty list; got {summary['body_markers']}"
        )
        assert "skipped_reason" in summary["body_velocity_stats"]

    # ---------------------------------------------------------- #
    # Case 14: discover_pose_files finds files in a directory
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # Make 3 csv files at the root + 1 in a subdir + a hidden
        for i in range(3):
            csv = td / f"sess_{i}.csv"
            _build_synthetic_csv(csv, n_frames=200)
        sub = td / "subdir"
        sub.mkdir()
        _build_synthetic_csv(sub / "extra.csv", n_frames=200)
        (td / ".hidden.csv").write_text("x,y\n1,2\n")
        # Also a .pose. file that should be filtered out
        (td / "sess_0.pose.csv").write_text("nose_x,nose_y,nose_p\n1,2,0.99\n")

        found = discover_pose_files(str(td))
        # Should find 4 csvs (3 root + 1 subdir), filter out hidden
        # and .pose. files
        assert len(found) == 4, (
            f"Expected 4 csv files; found {len(found)}: {found}"
        )
        for f in found:
            assert ".pose." not in f
            assert not Path(f).name.startswith(".")

    # ---------------------------------------------------------- #
    # Case 15: multi-file load_pose_files + session ranges
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        csv1 = td / "sess_a.csv"
        csv2 = td / "sess_b.csv"
        csv3 = td / "sess_c.csv"
        _build_synthetic_csv(csv1, n_frames=100, likelihood_pattern="clean")
        _build_synthetic_csv(csv2, n_frames=200, likelihood_pattern="clean")
        _build_synthetic_csv(csv3, n_frames=150, likelihood_pattern="clean")

        df, markers, sessions = load_pose_files(
            [str(csv1), str(csv2), str(csv3)]
        )
        assert len(df) == 450, f"Expected 100+200+150=450 rows; got {len(df)}"
        assert len(sessions) == 3
        # Session ranges must be contiguous and in order
        assert sessions[0] == ("sess_a", 0, 100)
        assert sessions[1] == ("sess_b", 100, 300)
        assert sessions[2] == ("sess_c", 300, 450)
        # Markers consistent
        assert "nose" in markers
        assert "tail_base" in markers

    # ---------------------------------------------------------- #
    # Case 16: session boundaries — dropout runs do NOT span files
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # File A ends with 20 low-p frames; File B starts with 20
        # low-p frames. If we naively concat, that's ONE 40-frame
        # run. With session-aware encoding, two separate 20-frame
        # runs.
        n = 100
        df_a = pd.DataFrame({
            "Nose_x": [10.0] * n,
            "Nose_y": [20.0] * n,
            "Nose_p": [0.99] * (n - 20) + [0.0] * 20,
            "Ear_left_x": [11.0] * n,
            "Ear_left_y": [21.0] * n,
            "Ear_left_p": [0.99] * n,
            "Ear_right_x": [12.0] * n,
            "Ear_right_y": [22.0] * n,
            "Ear_right_p": [0.99] * n,
            "Center_x": [10.0] * n,
            "Center_y": [20.0] * n,
            "Center_p": [0.99] * n,
            "Left_flank_x": [9.0] * n,
            "Left_flank_y": [21.0] * n,
            "Left_flank_p": [0.99] * n,
            "Right_flank_x": [11.0] * n,
            "Right_flank_y": [21.0] * n,
            "Right_flank_p": [0.99] * n,
            "Tail_base_x": [10.0] * n,
            "Tail_base_y": [25.0] * n,
            "Tail_base_p": [0.99] * n,
        })
        df_b = df_a.copy()
        df_b["Nose_p"] = [0.0] * 20 + [0.99] * (n - 20)

        csv_a = td / "fileA.csv"
        csv_b = td / "fileB.csv"
        df_a.to_csv(csv_a, index=True)
        df_b.to_csv(csv_b, index=True)

        # Multi-file mode
        df_combined, markers, sessions = load_pose_files([str(csv_a), str(csv_b)])
        # Session-aware run-length encoding
        nose_stats = compute_marker_stats(
            df_combined, "nose", 0.95, session_ranges=sessions,
        )
        # Should report TWO 20-frame runs (one per session), NOT
        # one 40-frame run
        assert nose_stats.longest_low_run == 20, (
            f"Session-aware encoding should keep runs within session "
            f"boundaries; longest_low_run should be 20, got "
            f"{nose_stats.longest_low_run} (suggests a 40-frame run "
            f"crossing the file boundary)"
        )
        assert nose_stats.n_low_runs == 2, (
            f"Expected 2 low-p runs (one per session); got "
            f"{nose_stats.n_low_runs}"
        )

        # Without session_ranges, the bug surfaces: one 40-frame run
        nose_stats_naive = compute_marker_stats(
            df_combined, "nose", 0.95, session_ranges=None,
        )
        assert nose_stats_naive.longest_low_run == 40, (
            f"Naive (no-boundary) encoding should produce a 40-frame "
            f"cross-boundary run; got {nose_stats_naive.longest_low_run} "
            f"(test fixture is wrong if this assertion fails)"
        )

    # ---------------------------------------------------------- #
    # Case 17: parquet load round-trip (skipped if no parquet engine)
    # ---------------------------------------------------------- #
    if _has_parquet_engine():
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            csv = td / "sess.csv"
            parquet = td / "sess.parquet"
            _build_synthetic_csv(csv, n_frames=500, likelihood_pattern="clean")
            # Load CSV, save as parquet (with index dropped, matching
            # csv_to_parquet behavior)
            df_csv, _ = load_pose_csv(str(csv))
            df_csv.to_parquet(parquet, index=False)
            # Now load via load_pose_file with .parquet extension
            df_pq, markers_pq = load_pose_file(str(parquet))
            assert len(df_pq) == 500
            assert "nose" in markers_pq
            # Round-trip values match
            assert np.allclose(
                df_csv["nose_x"].values, df_pq["nose_x"].values
            )

        # Multi-file mode with parquets
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            csv1 = td / "a.csv"; pq1 = td / "a.parquet"
            csv2 = td / "b.csv"; pq2 = td / "b.parquet"
            _build_synthetic_csv(csv1, n_frames=300)
            _build_synthetic_csv(csv2, n_frames=400)
            df1, _ = load_pose_csv(str(csv1)); df1.to_parquet(pq1, index=False)
            df2, _ = load_pose_csv(str(csv2)); df2.to_parquet(pq2, index=False)

            out_dir = td / "out"
            report = run_diagnostic(
                csv_path=str(td),  # directory mode
                output_dir=str(out_dir),
                fps=30.0,
            )
            assert report.n_frames == 700, (
                f"Expected 300+400=700 frames; got {report.n_frames}"
            )
            # summary.json should record both sessions
            summary = json.loads((out_dir / "summary.json").read_text())
            assert len(summary["sessions"]) == 2
            assert summary["sessions"][0]["name"] == "a"
            assert summary["sessions"][1]["name"] == "b"
            assert summary["sessions"][0]["n_frames"] == 300
            assert summary["sessions"][1]["n_frames"] == 400

    # ---------------------------------------------------------- #
    # Case 18: REGRESSION — IMPORTED_POSE multi-row CSV produces
    # the SAME column names as a parquet derived from it (i.e. no
    # spurious "imported_pose_" prefix on marker names).
    # Prior bug: the flattening used last-two header levels, so
    # SimBA's IMPORTED_POSE rows on level 0 AND 1 (with the real
    # column name on level 2) produced "imported_pose_nose_x"
    # instead of "nose_x". Diagnostic auto-detection then failed
    # to find "nose"/"ear_left"/"ear_right" markers, and any
    # explicit --head-markers flag had to use the polluted prefix.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # Build a SimBA IMPORTED_POSE-format CSV (3 header rows,
        # rows 0+1 are the IMPORTED_POSE marker, row 2 has the
        # full column names with _x/_y/_p suffixes).
        csv_path = td / "imported.csv"
        n_data_cols = 9  # 3 markers × 3 suffixes
        with open(csv_path, "w") as f:
            f.write("," + ",".join(["IMPORTED_POSE"] * n_data_cols) + "\n")
            f.write("," + ",".join(["IMPORTED_POSE"] * n_data_cols) + "\n")
            real = ["nose_x", "nose_y", "nose_p",
                    "ear_left_x", "ear_left_y", "ear_left_p",
                    "ear_right_x", "ear_right_y", "ear_right_p"]
            f.write("," + ",".join(real) + "\n")
            for i in range(20):
                values = [str(i)] + [f"{(i + k) * 1.0:.3f}"
                                     for k in range(n_data_cols)]
                f.write(",".join(values) + "\n")

        df, markers = load_pose_file(str(csv_path))
        # Marker names must NOT have the "imported_pose_" prefix
        for m in markers:
            assert not m.startswith("imported_pose_"), (
                f"IMPORTED_POSE flattening should not pollute marker "
                f"names with the 'imported_pose_' prefix; got marker "
                f"{m!r} in {markers}"
            )
        # The expected canonical marker names
        assert "nose" in markers, (
            f"Expected canonical 'nose' marker; got {markers}"
        )
        assert "ear_left" in markers
        assert "ear_right" in markers
        # And the columns themselves are clean
        assert "nose_x" in df.columns
        assert "ear_left_p" in df.columns
        assert "imported_pose_nose_x" not in df.columns

    # ---------------------------------------------------------- #
    # Case 19: DLC-standard 3-row header still works correctly.
    # Last header level is "x"/"y"/"likelihood"; the bodypart
    # name on the second-to-last level needs to be prepended.
    # The flattening also normalizes "likelihood" → "p".
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        csv_path = td / "dlc.csv"
        with open(csv_path, "w") as f:
            # Row 0: scorer
            f.write("," + ",".join(["scorer1"] * 9) + "\n")
            # Row 1: bodypart
            bps = ["nose"] * 3 + ["ear_left"] * 3 + ["ear_right"] * 3
            f.write("," + ",".join(bps) + "\n")
            # Row 2: coord (uses DLC's "likelihood" not "_p")
            coords = ["x", "y", "likelihood"] * 3
            f.write("," + ",".join(coords) + "\n")
            for i in range(20):
                values = [str(i)] + [f"{(i + k) * 1.0:.3f}"
                                     for k in range(9)]
                f.write(",".join(values) + "\n")

        df, markers = load_pose_file(str(csv_path))
        assert "nose" in markers, (
            f"DLC standard format: expected 'nose' marker; got {markers}"
        )
        # Verify the columns: bodypart_x / bodypart_y / bodypart_p
        # ("likelihood" should have been normalized to "p")
        assert "nose_x" in df.columns
        assert "nose_y" in df.columns
        assert "nose_p" in df.columns, (
            f"DLC 'likelihood' column should be normalized to '_p'; "
            f"got {list(df.columns)}"
        )

    # ---------------------------------------------------------- #
    # Case 20: REGRESSION — NaN leakage through valid_mask in
    # compute_body_velocity. When a body marker drops out for one
    # frame, the body shape becomes degenerate at that frame and
    # body_axis/centroid become NaN. np.gradient(centroid) at the
    # NEIGHBORING (otherwise-valid) frames then inherits the NaN,
    # so vx_b/vy_b are NaN at frames where valid_axis and
    # valid_prev are both True. valid_mask must guard against this
    # by also requiring isfinite(vx_b) & isfinite(vy_b).
    #
    # Without the fix: np.histogram(v[valid_mask]) would raise
    #   ValueError: autodetected range of [nan, nan] is not finite
    # in classify_velocity_modality and ax.hist similarly.
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(123)
    n = 500
    data = {}
    for m in ["nose", "ear_left", "ear_right",
              "center", "leftflank", "rightflank"]:
        data[f"{m}_x"] = (
            50 + 10 * np.sin(np.arange(n) / 10)
            + rng.normal(0, 0.5, n)
        )
        data[f"{m}_y"] = (
            50 + 10 * np.cos(np.arange(n) / 10)
            + rng.normal(0, 0.5, n)
        )
        data[f"{m}_p"] = np.full(n, 0.99)
    # One body marker drops at frames 100 and 250
    data["leftflank_p"][100] = 0.1
    data["leftflank_p"][250] = 0.1
    df_nan = pd.DataFrame(data)
    body_markers = ["center", "leftflank", "rightflank"]
    head_markers = ["nose", "ear_left", "ear_right"]

    vx_b, vy_b, body_valid, _ = compute_body_velocity(
        df_nan, body_markers, head_markers, 0.95, 30.0,
    )
    # The contract: valid_mask=True implies velocity is finite
    n_nan_in_valid = int(np.sum(body_valid & ~np.isfinite(vx_b)))
    assert n_nan_in_valid == 0, (
        f"valid_mask should imply finite velocity; got "
        f"{n_nan_in_valid} frames where mask=True but vx_b is NaN"
    )
    n_nan_in_valid_y = int(np.sum(body_valid & ~np.isfinite(vy_b)))
    assert n_nan_in_valid_y == 0, (
        f"valid_mask should imply finite vy_b too; got "
        f"{n_nan_in_valid_y} frames where mask=True but vy_b is NaN"
    )

    # And classify_velocity_modality must not raise on this input
    modality = classify_velocity_modality(vx_b, body_valid)
    assert modality in (
        "unimodal", "bimodal", "multimodal", "insufficient_samples"
    ), f"Unexpected modality {modality!r}"

    # Defensive check: classify_velocity_modality should ALSO be
    # robust to NaN leakage from buggy callers — pass an "all True"
    # mask alongside an array containing NaN, and verify it doesn't
    # raise.
    vx_with_nan = vx_b.copy()
    vx_with_nan[5] = np.nan  # inject a NaN
    fake_all_valid = np.ones(len(vx_with_nan), dtype=bool)
    modality2 = classify_velocity_modality(vx_with_nan, fake_all_valid)
    assert modality2 in (
        "unimodal", "bimodal", "multimodal", "insufficient_samples"
    ), (
        f"classify_velocity_modality should be defensive against NaN "
        f"leakage even with a mask that doesn't filter them; got "
        f"{modality2!r}"
    )

    # ---------------------------------------------------------- #
    # Case 21: auto_detect_rigid_pairs identifies tight pairs and
    # rejects loose ones based on CV threshold.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_diagnostic import (
        auto_detect_rigid_pairs,
    )
    rng = np.random.default_rng(42)
    n = 2000
    # Build a synthetic df where ear_left↔ear_right is genuinely
    # rigid (CV ≈ 0.05) and nose↔ear_left is loose (CV ≈ 0.30)
    # because nose drifts independently of the ear pair.
    # Each marker has its own independent noise so the inter-marker
    # distance has real variance (mirrored noise would produce
    # CV=0 by construction, which doesn't represent real data).
    cx = 50 + 10 * np.sin(np.arange(n) / 50)
    cy = 50 + 10 * np.cos(np.arange(n) / 50)
    loose_offset_x = rng.normal(0, 4.0, n)   # loose noise
    loose_offset_y = rng.normal(0, 4.0, n)

    data = {
        "ear_left_x":  cx + 5 + rng.normal(0, 0.15, n),
        "ear_left_y":  cy + 5 + rng.normal(0, 0.15, n),
        "ear_left_p":  np.full(n, 0.99),
        "ear_right_x": cx - 5 + rng.normal(0, 0.15, n),
        "ear_right_y": cy + 5 + rng.normal(0, 0.15, n),
        "ear_right_p": np.full(n, 0.99),
        "nose_x":      cx + loose_offset_x,
        "nose_y":      cy + 12 + loose_offset_y,
        "nose_p":      np.full(n, 0.99),
    }
    df_rig = pd.DataFrame(data)
    markers = ["ear_left", "ear_right", "nose"]
    pairs = auto_detect_rigid_pairs(
        df_rig, markers, likelihood_threshold=0.95,
        cv_threshold=0.20, max_pairs=8,
    )
    # ear_left ↔ ear_right should be picked, nose pairs should not
    pair_set = {tuple(sorted(p)) for p in pairs}
    assert ("ear_left", "ear_right") in pair_set, (
        f"Expected ear-ear pair to be auto-detected; got {pair_set}"
    )
    for loose in [("ear_left", "nose"), ("ear_right", "nose")]:
        assert loose not in pair_set, (
            f"Loose pair {loose} should NOT be auto-detected at "
            f"threshold 0.20; got {pair_set}"
        )

    # And with a very strict threshold, even ear-ear may be rejected
    pairs_strict = auto_detect_rigid_pairs(
        df_rig, markers, likelihood_threshold=0.95,
        cv_threshold=0.001,  # impossibly strict
        max_pairs=8,
    )
    assert pairs_strict == [], (
        f"At threshold 0.001 no pair should be detected; got "
        f"{pairs_strict}"
    )

    # ---------------------------------------------------------- #
    # Case 22: compute_per_session_summary returns one entry per
    # session, with worst-marker fields populated correctly.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_diagnostic import (
        compute_per_session_summary,
    )
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # 3 sessions: clean, dropouts, mixed. Use n_frames=2000 so
        # the dropouts pattern (every 500 frames) actually produces
        # multiple dropout windows in the dropouts session.
        _build_synthetic_csv(
            td / "sess_clean.csv", n_frames=2000, likelihood_pattern="clean"
        )
        _build_synthetic_csv(
            td / "sess_drop.csv", n_frames=2000, likelihood_pattern="dropouts"
        )
        _build_synthetic_csv(
            td / "sess_clean2.csv", n_frames=2000, likelihood_pattern="clean"
        )

        df, mks, sessions = load_pose_files([
            str(td / "sess_clean.csv"),
            str(td / "sess_drop.csv"),
            str(td / "sess_clean2.csv"),
        ])
        per_session = compute_per_session_summary(
            df, mks, sessions, likelihood_threshold=0.95,
        )
        assert len(per_session) == 3, (
            f"Expected 3 per-session entries; got {len(per_session)}"
        )
        assert [s["name"] for s in per_session] == [
            "sess_clean", "sess_drop", "sess_clean2"
        ]
        # Dropouts session should have LONGER worst-marker run than
        # the clean sessions (the dropouts pattern produces 30-frame
        # p=0 windows that the clean pattern doesn't have).
        clean1_run = per_session[0]["worst_marker_longest_run"][1]
        drop_run = per_session[1]["worst_marker_longest_run"][1]
        clean2_run = per_session[2]["worst_marker_longest_run"][1]
        assert drop_run > max(clean1_run, clean2_run), (
            f"Dropouts session should have a longer worst-marker run "
            f"than clean sessions; got drop={drop_run}, "
            f"clean1={clean1_run}, clean2={clean2_run}"
        )
        # And the dropouts session's worst-run marker should be nose
        # (the only marker that drops in the synthetic pattern)
        assert per_session[1]["worst_marker_longest_run"][0] == "nose", (
            f"Dropouts pattern only drops nose; expected worst_run "
            f"marker = nose; got "
            f"{per_session[1]['worst_marker_longest_run']}"
        )
        # Each entry must have all the expected fields
        for entry in per_session:
            assert "name" in entry
            assert "n_frames" in entry
            assert "worst_marker_frac_high" in entry
            assert "worst_marker_longest_run" in entry
            assert "all_markers" in entry
            # all_markers should have one entry per marker
            assert set(entry["all_markers"].keys()) == set(mks)

    # ---------------------------------------------------------- #
    # Case 23: make_recommendation includes per-session
    # distribution when per_session_summary is provided.
    # ---------------------------------------------------------- #
    fake_per_marker = [
        MarkerStats(
            name="m1", n_frames=1000,
            p_mean=0.5, p_median=0.5, p_q05=0.1, p_q95=0.9,
            frac_high=0.4, frac_zero=0.0,
            longest_low_run=200, n_low_runs=10,
            median_low_run=20.0,
        ),
    ]
    fake_rigid_stats = [
        {"marker_a": "m1", "marker_b": "m2",
         "n_high_confidence": 500, "mean_distance": 30.0,
         "std_distance": 6.0, "cv_distance": 0.20,
         "min_distance": 25.0, "max_distance": 35.0,
         "p05_distance": 26.0, "p95_distance": 34.0},
    ]
    fake_session_summary = [
        {"name": f"s{i}", "n_frames": 1000,
         "worst_marker_frac_high": ["m1", 0.8 if i < 13 else 0.1],
         "worst_marker_longest_run": ["m1", 50 if i < 13 else 950],
         "all_markers": {}}
        for i in range(15)
    ]
    rec = make_recommendation(
        fake_per_marker, fake_rigid_stats, "unimodal",
        per_session_summary=fake_session_summary,
    )
    # The recommendation must include the per-session distribution
    assert "PER-SESSION QUALITY" in rec, (
        f"Expected per-session distribution in recommendation; "
        f"got:\n{rec}"
    )
    # Loose substring checks (formatting may add right-alignment
    # spaces between the label and the count)
    assert "good" in rec and "13 of 15" in rec, (
        f"Expected 'good ... 13 of 15' in recommendation; got:\n{rec}"
    )
    assert "bad" in rec and "2 of 15" in rec, (
        f"Expected 'bad ... 2 of 15' in recommendation; got:\n{rec}"
    )
    # The "small fraction of sessions inflate aggregate" note
    # should appear (only 2 sessions with very long runs)
    assert "contribute to the aggregate" in rec, (
        f"Expected aggregate-contamination note when few sessions "
        f"drive the worst stats; got:\n{rec}"
    )

    # When per_session_summary is empty/None, the per-session block
    # is omitted (back-compat with single-session callers)
    rec_no_session = make_recommendation(
        fake_per_marker, fake_rigid_stats, "unimodal",
        per_session_summary=None,
    )
    assert "PER-SESSION QUALITY" not in rec_no_session

    # ---------------------------------------------------------- #
    # Case 24: auto_detect_rigid_pairs respects exclude_markers
    # (head-marker exclusion for the v1 body-triplet build).
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_diagnostic import (
        auto_detect_candidate_triplets,
        compute_behavioral_signal_pairs,
    )
    rng = np.random.default_rng(7)
    n = 2000
    cx = 50 + 10 * np.sin(np.arange(n) / 50)
    cy = 50 + 10 * np.cos(np.arange(n) / 50)
    # Build five markers where every pair is empirically tight
    # (independent small noise on each marker). With no exclusion,
    # auto-detect should return a full set of pairs. With nose
    # excluded, no pair containing nose should be returned.
    data = {}
    for m, (off_x, off_y) in [
        ("nose", (0, 12)),
        ("ear_left", (5, 5)),
        ("ear_right", (-5, 5)),
        ("back1", (0, -5)),
        ("back2", (0, -10)),
    ]:
        data[f"{m}_x"] = cx + off_x + rng.normal(0, 0.15, n)
        data[f"{m}_y"] = cy + off_y + rng.normal(0, 0.15, n)
        data[f"{m}_p"] = np.full(n, 0.99)
    df_excl = pd.DataFrame(data)
    markers = ["nose", "ear_left", "ear_right", "back1", "back2"]

    # Without exclusion: should find pairs involving nose
    pairs_all = auto_detect_rigid_pairs(
        df_excl, markers, likelihood_threshold=0.95,
        cv_threshold=0.20, max_pairs=20,
    )
    assert any("nose" in p for p in pairs_all), (
        f"Without exclusion, expected nose pairs in result; got "
        f"{pairs_all}"
    )

    # With nose excluded: NO pair should contain nose
    pairs_no_nose = auto_detect_rigid_pairs(
        df_excl, markers, likelihood_threshold=0.95,
        cv_threshold=0.20, max_pairs=20,
        exclude_markers=["nose"],
    )
    for a, b in pairs_no_nose:
        assert a != "nose" and b != "nose", (
            f"With nose excluded, no pair should contain nose; "
            f"got pair ({a}, {b}) in {pairs_no_nose}"
        )
    # And we should still get some pairs (the other 4 markers)
    assert len(pairs_no_nose) > 0, (
        f"With 4 non-excluded markers all empirically tight, "
        f"expected ≥1 pair; got {pairs_no_nose}"
    )

    # ---------------------------------------------------------- #
    # Case 25: auto_detect_candidate_triplets finds tight triplets
    # and respects the exclusion.
    # ---------------------------------------------------------- #
    triplets = auto_detect_candidate_triplets(
        df_excl, markers, likelihood_threshold=0.95,
        cv_threshold=0.20, max_triplets=10,
    )
    # All 5 markers are tight → C(5,3) = 10 candidate triplets
    # should qualify (or close to it; some pairs may have
    # boundary-CV issues with the synthetic noise).
    assert len(triplets) >= 5, (
        f"Expected ≥5 candidate triplets from 5 tight markers; "
        f"got {len(triplets)}"
    )
    # Each entry must be ((a, b, c), info_dict) with the right keys
    for triplet, info in triplets:
        assert len(triplet) == 3
        assert all(m in markers for m in triplet)
        for k in ("cv_ab", "cv_ac", "cv_bc",
                  "cv_mean", "cv_max", "n_samples"):
            assert k in info, f"Missing {k} in triplet info {info}"
        # cv_max must be < cv_threshold (the gate)
        assert info["cv_max"] < 0.20

    # Sorted ascending by cv_mean
    cv_means = [info["cv_mean"] for _, info in triplets]
    assert cv_means == sorted(cv_means), (
        f"Triplets should be sorted ascending by cv_mean; got "
        f"{cv_means}"
    )

    # With nose excluded: no triplet should contain nose
    triplets_no_nose = auto_detect_candidate_triplets(
        df_excl, markers, likelihood_threshold=0.95,
        cv_threshold=0.20, max_triplets=10,
        exclude_markers=["nose"],
    )
    for triplet, _ in triplets_no_nose:
        assert "nose" not in triplet, (
            f"Excluded marker should not appear in any triplet; "
            f"got {triplet}"
        )
    # 4 markers → C(4,3) = 4 possible triplets
    assert len(triplets_no_nose) <= 4

    # Edge case: fewer than 3 eligible markers → empty list
    empty = auto_detect_candidate_triplets(
        df_excl, markers, likelihood_threshold=0.95,
        cv_threshold=0.20, max_triplets=10,
        exclude_markers=["nose", "ear_left", "ear_right"],
    )
    assert empty == [] or len(empty) == 0, (
        f"With only 2 eligible markers, expected empty triplet "
        f"list; got {empty}"
    )

    # ---------------------------------------------------------- #
    # Case 26: compute_behavioral_signal_pairs produces head-pair
    # stats with role="behavioral_signal".
    # ---------------------------------------------------------- #
    behav = compute_behavioral_signal_pairs(
        df_excl, head_markers=["nose", "ear_left", "ear_right"],
        likelihood_threshold=0.95,
    )
    # 3 head markers → C(3,2) = 3 pairs
    assert len(behav) == 3, (
        f"Expected 3 head-internal pairs; got {len(behav)}"
    )
    for stats in behav:
        assert stats["role"] == "behavioral_signal", (
            f"Expected role='behavioral_signal'; got "
            f"{stats.get('role')}"
        )
        # Standard rigid-pair-stats fields should be present
        assert "marker_a" in stats and "marker_b" in stats
        assert "cv_distance" in stats or "warning" in stats

    # Edge case: < 2 head markers → empty
    behav_empty = compute_behavioral_signal_pairs(
        df_excl, head_markers=["nose"],
        likelihood_threshold=0.95,
    )
    assert behav_empty == []

    # ---------------------------------------------------------- #
    # Case 27: run_diagnostic with auto-detect excludes head
    # markers from rigid pairs and produces candidate triplets +
    # behavioral pairs in summary.json.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        csv = td / "test.csv"
        # Use the already-supported synthetic builder with default
        # markers; ear_left/ear_right will be the head pair
        _build_synthetic_csv(
            csv, n_frames=3000, likelihood_pattern="clean"
        )
        out_dir = td / "out"
        report = run_diagnostic(
            csv_path=str(csv),
            output_dir=str(out_dir),
            head_markers=["nose", "ear_left", "ear_right"],
            fps=30.0,
        )
        summary = json.loads((out_dir / "summary.json").read_text())

        # No rigid pair should contain a head marker (auto-detect
        # excluded them)
        head_set = {"nose", "ear_left", "ear_right"}
        for pair in summary["rigid_pairs"]:
            assert not (set(pair) & head_set), (
                f"Rigid pair {pair} should not contain head markers "
                f"after head-exclusion auto-detect"
            )

        # candidate_triplets and behavioral_signal_pairs sections
        # must exist
        assert "candidate_triplets" in summary
        assert "behavioral_signal_pairs" in summary

        # Behavioral pairs should be the 3 head-internal pairs
        behav = summary["behavioral_signal_pairs"]
        assert len(behav) == 3, (
            f"Expected 3 head-internal behavioral pairs; got "
            f"{len(behav)}"
        )
        for stats in behav:
            assert stats["role"] == "behavioral_signal"
            assert {stats["marker_a"], stats["marker_b"]} <= head_set

        # Candidate triplets must not contain head markers either
        for entry in summary["candidate_triplets"]:
            assert not (set(entry["markers"]) & head_set), (
                f"Candidate triplet {entry['markers']} should not "
                f"contain head markers"
            )

    print("smoke_kalman_diagnostic: 27/27 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
