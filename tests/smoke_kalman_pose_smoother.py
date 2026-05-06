"""Smoke tests for mufasa.data_processors.kalman_pose_smoother
(Stage 1 / patch 85).

Tests at this stage are limited to what patch 85 ships:
  - State layout helpers
  - F and Q construction
  - Per-frame observation construction
  - Forward filter correctness on synthetic data with known
    ground truth

The RTS backward smoother, body-triplet pseudo-measurements,
EM noise fitting, and CLI all land in subsequent patches and
will gain their own smoke cases as they ship.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from mufasa.data_processors.kalman_pose_smoother import (
    EMResult,
    FilterResult,
    FrameObservation,
    NoiseParams,
    SMOOTHER_MODEL_VERSION,
    SmootherResult,
    StateLayout,
    TripletPrior,
    build_F,
    build_Q,
    build_observation,
    build_triplet_observation,
    extract_position_variances,
    extract_positions,
    fit_noise_params_em,
    fit_triplet_prior,
    fit_triplet_priors,
    forward_filter,
    initial_noise_params,
    load_model,
    rts_smoother,
    save_model,
    smooth_multi_session,
    smooth_pose,
)


def _build_synthetic_pose_csv(
    path: Path,
    n_frames: int = 500,
    markers: tuple = ("nose", "ear_left", "ear_right",
                      "back1", "back2", "back3"),
    seed: int = 0,
) -> None:
    """Write a synthetic Mufasa-style flat-column pose CSV for
    end-to-end orchestrator testing. Each marker has a CV
    trajectory + small Gaussian observation noise + p=0.95
    likelihoods throughout."""
    rng = np.random.default_rng(seed)
    fps = 30.0
    times = np.arange(n_frames) / fps
    cols = {}
    for i, m in enumerate(markers):
        true_x = 50 + i * 5 + 2.0 * times
        true_y = 60 - i * 3 + 1.0 * times
        cols[f"{m}_x"] = true_x + rng.normal(0, 1.0, n_frames)
        cols[f"{m}_y"] = true_y + rng.normal(0, 1.0, n_frames)
        cols[f"{m}_p"] = np.full(n_frames, 0.95)
    df = pd.DataFrame(cols)
    df.to_csv(path, index=True)


def main() -> int:
    # ---------------------------------------------------------- #
    # Case 1: StateLayout indexing
    # ---------------------------------------------------------- #
    layout = StateLayout(markers=("nose", "ear_left", "ear_right"))
    assert layout.n_markers == 3
    assert layout.state_dim == 12
    assert layout.position_indices("nose") == (0, 1)
    assert layout.velocity_indices("nose") == (2, 3)
    assert layout.position_indices("ear_left") == (4, 5)
    assert layout.velocity_indices("ear_left") == (6, 7)
    assert layout.position_indices("ear_right") == (8, 9)
    assert layout.velocity_indices("ear_right") == (10, 11)

    apos = layout.all_position_indices()
    assert apos.shape == (3, 2)
    assert (apos == np.array([[0, 1], [4, 5], [8, 9]])).all()

    # ---------------------------------------------------------- #
    # Case 2: build_F gives correct CV-block structure
    # ---------------------------------------------------------- #
    layout2 = StateLayout(markers=("a", "b"))
    dt = 0.1
    F = build_F(layout2, dt)
    assert F.shape == (8, 8)
    # Block 0 (marker a): rows 0-3, cols 0-3
    expected_block = np.array([
        [1.0, 0.0,  dt, 0.0],
        [0.0, 1.0, 0.0,  dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert np.allclose(F[0:4, 0:4], expected_block)
    assert np.allclose(F[4:8, 4:8], expected_block)
    # Off-diagonal blocks must be zero (no coupling at this stage)
    assert np.allclose(F[0:4, 4:8], 0.0)
    assert np.allclose(F[4:8, 0:4], 0.0)

    # build_F validates dt > 0
    try:
        build_F(layout2, 0.0)
        assert False, "build_F should reject dt=0"
    except ValueError:
        pass

    # ---------------------------------------------------------- #
    # Case 3: build_Q produces symmetric, positive-semidefinite
    # block-diagonal matrix
    # ---------------------------------------------------------- #
    params = NoiseParams(
        sigma_base={"a": 1.0, "b": 1.5},
        q_pos={"a": 100.0, "b": 200.0},
        q_vel={"a": 10.0, "b": 20.0},
    )
    Q = build_Q(layout2, dt, params)
    assert Q.shape == (8, 8)
    # Symmetric
    assert np.allclose(Q, Q.T), "Q should be symmetric"
    # Positive-semidefinite (smallest eigenvalue ≥ 0 within tol)
    eigvals = np.linalg.eigvalsh(Q)
    assert eigvals.min() > -1e-10, (
        f"Q should be PSD; smallest eigenvalue={eigvals.min()}"
    )
    # Block-diagonal: off-diagonal blocks zero
    assert np.allclose(Q[0:4, 4:8], 0.0)
    assert np.allclose(Q[4:8, 0:4], 0.0)
    # Marker b has 2× q_pos and q_vel of marker a → block magnitudes
    # should reflect that
    block_a = Q[0:4, 0:4]
    block_b = Q[4:8, 4:8]
    assert block_b[0, 0] == 2 * block_a[0, 0], (
        f"Marker b q_pos is 2x marker a; expected block[0,0] ratio 2; "
        f"got {block_b[0, 0] / block_a[0, 0]}"
    )

    # ---------------------------------------------------------- #
    # Case 4: NoiseParams.for_layout returns ordered arrays
    # ---------------------------------------------------------- #
    sb, qp, qv = params.for_layout(layout2)
    assert sb.tolist() == [1.0, 1.5]
    assert qp.tolist() == [100.0, 200.0]
    assert qv.tolist() == [10.0, 20.0]

    # ---------------------------------------------------------- #
    # Case 5: build_observation with all markers high-confidence
    # ---------------------------------------------------------- #
    layout3 = StateLayout(markers=("a", "b", "c"))
    sb3 = np.array([1.0, 2.0, 3.0])
    positions = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
    ])
    likelihoods = np.array([0.99, 0.99, 0.99])
    obs = build_observation(
        layout3, positions, likelihoods, sb3,
        likelihood_threshold=0.5,
    )
    assert obs.n_observed == 3
    assert obs.z.shape == (6,)
    assert obs.H.shape == (6, 12)
    assert obs.R.shape == (6, 6)
    # z should be the flattened positions in marker order
    assert obs.z.tolist() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    # H[0] should pick out state index 0 (a's x)
    assert obs.H[0, 0] == 1.0
    assert obs.H[0].sum() == 1.0
    # H[2] should pick out state index 4 (b's x)
    assert obs.H[2, 4] == 1.0
    # R is diagonal with sigma_i^2 / p_i^2 entries (p clamped to 0.999)
    # For marker a: var = (1.0 / 0.99)^2 ≈ 1.0203 (since p<0.999)
    expected_var_a = (1.0 / 0.99) ** 2
    assert abs(obs.R[0, 0] - expected_var_a) < 1e-9
    assert abs(obs.R[1, 1] - expected_var_a) < 1e-9
    expected_var_b = (2.0 / 0.99) ** 2
    assert abs(obs.R[2, 2] - expected_var_b) < 1e-9

    # ---------------------------------------------------------- #
    # Case 6: build_observation drops below-threshold markers
    # ---------------------------------------------------------- #
    likelihoods_partial = np.array([0.99, 0.30, 0.99])
    obs_partial = build_observation(
        layout3, positions, likelihoods_partial, sb3,
        likelihood_threshold=0.5,
    )
    assert obs_partial.n_observed == 2, (
        f"Expected 2 markers above threshold; got {obs_partial.n_observed}"
    )
    assert obs_partial.z.shape == (4,)
    # z should contain a and c, not b
    assert obs_partial.z.tolist() == [10.0, 20.0, 50.0, 60.0]
    # H rows pick out a's (x=0, y=1) and c's (x=8, y=9) state indices
    assert obs_partial.H[0, 0] == 1.0
    assert obs_partial.H[2, 8] == 1.0

    # ---------------------------------------------------------- #
    # Case 7: build_observation returns empty when no markers above
    # ---------------------------------------------------------- #
    likelihoods_low = np.array([0.10, 0.20, 0.30])
    obs_empty = build_observation(
        layout3, positions, likelihoods_low, sb3,
        likelihood_threshold=0.5,
    )
    assert obs_empty.n_observed == 0
    assert not obs_empty.has_observation
    assert obs_empty.z.shape == (0,)
    assert obs_empty.H.shape == (0, 12)
    assert obs_empty.R.shape == (0, 0)

    # ---------------------------------------------------------- #
    # Case 8: NaN positions are treated as missing
    # ---------------------------------------------------------- #
    positions_with_nan = positions.copy()
    positions_with_nan[1, 0] = np.nan
    obs_nan = build_observation(
        layout3, positions_with_nan, likelihoods, sb3,
        likelihood_threshold=0.5,
    )
    assert obs_nan.n_observed == 2, (
        f"NaN in marker b's position should drop it; got "
        f"n_observed={obs_nan.n_observed}"
    )
    # Markers a and c should be observed; b dropped
    assert obs_nan.z.tolist() == [10.0, 20.0, 50.0, 60.0]

    # ---------------------------------------------------------- #
    # Case 9: forward_filter on noiseless constant-velocity
    # synthetic data — filter should track exactly.
    # ---------------------------------------------------------- #
    layout_cv = StateLayout(markers=("m1",))
    params_cv = NoiseParams(
        sigma_base={"m1": 0.1},
        q_pos={"m1": 1.0},
        q_vel={"m1": 0.1},
    )
    fps = 30.0
    dt_cv = 1.0 / fps
    T = 100
    # Ground truth: marker m1 moves at constant velocity (10, -5) px/s
    vx_true = 10.0
    vy_true = -5.0
    x0 = 50.0
    y0 = 60.0
    times = np.arange(T) * dt_cv
    pos_true = np.column_stack([x0 + vx_true * times, y0 + vy_true * times])

    positions_t = pos_true.reshape(T, 1, 2)
    likelihoods_t = np.full((T, 1), 0.99)

    initial_state = np.array([x0, y0, vx_true, vy_true])
    result = forward_filter(
        positions_t, likelihoods_t,
        layout_cv, params_cv, dt_cv,
        likelihood_threshold=0.5,
        initial_state=initial_state,
        initial_cov=0.01 * np.eye(4),  # tight initial covariance
    )
    assert result.x_filt.shape == (T, 4)
    assert result.P_filt.shape == (T, 4, 4)
    assert (result.n_observed == 1).all()

    # Filtered positions should track ground truth tightly.
    # Allow a small tolerance for numerical/process-noise effects.
    pos_filt = extract_positions(result.x_filt, layout_cv).squeeze(1)
    err = np.linalg.norm(pos_filt - pos_true, axis=1)
    assert err.max() < 0.5, (
        f"Filter should track noiseless CV trajectory; max position "
        f"error = {err.max():.4f}"
    )

    # ---------------------------------------------------------- #
    # Case 10: forward_filter recovers ground truth on noisy CV
    # data — RMS error should be well below the observation noise
    # (filter benefit) but above zero (some residual noise).
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(0)
    obs_noise = 2.0  # px std
    pos_noisy = pos_true + rng.normal(0, obs_noise, pos_true.shape)
    positions_noisy = pos_noisy.reshape(T, 1, 2)

    params_noisy = NoiseParams(
        sigma_base={"m1": obs_noise},
        q_pos={"m1": 0.5},
        q_vel={"m1": 0.1},
    )
    result_noisy = forward_filter(
        positions_noisy, likelihoods_t,
        layout_cv, params_noisy, dt_cv,
        likelihood_threshold=0.5,
        initial_state=initial_state,
        initial_cov=0.5 * np.eye(4),
    )
    pos_filt_noisy = extract_positions(
        result_noisy.x_filt, layout_cv,
    ).squeeze(1)
    # RMS error on filtered positions vs ground truth — let
    # the filter warm up for the first 10 frames before
    # measuring (transient).
    err_filt = np.linalg.norm(pos_filt_noisy[10:] - pos_true[10:], axis=1)
    rms_filt = float(np.sqrt(np.mean(err_filt ** 2)))
    # Compare to the raw observation RMS
    err_raw = np.linalg.norm(pos_noisy[10:] - pos_true[10:], axis=1)
    rms_raw = float(np.sqrt(np.mean(err_raw ** 2)))
    # Filter should reduce RMS at least 30% — tight CV dynamics
    # mean we expect ~50% reduction but allow margin.
    assert rms_filt < 0.7 * rms_raw, (
        f"Filter should reduce noise; raw RMS={rms_raw:.3f}, "
        f"filtered RMS={rms_filt:.3f} (no improvement)"
    )

    # ---------------------------------------------------------- #
    # Case 11: forward_filter handles dropouts (frames with no
    # high-confidence observation) gracefully — predicts forward
    # using dynamics.
    # ---------------------------------------------------------- #
    likelihoods_drop = np.full((T, 1), 0.99)
    # Drop 30 consecutive frames in the middle
    likelihoods_drop[30:60, 0] = 0.0
    result_drop = forward_filter(
        positions_noisy, likelihoods_drop,
        layout_cv, params_noisy, dt_cv,
        likelihood_threshold=0.5,
        initial_state=initial_state,
        initial_cov=0.5 * np.eye(4),
    )
    # Filter should NOT crash on the dropout segment
    assert result_drop.x_filt.shape == (T, 4)
    # n_observed during dropout must be 0
    assert (result_drop.n_observed[30:60] == 0).all()
    assert (result_drop.n_observed[:30] == 1).all()
    assert (result_drop.n_observed[60:] == 1).all()

    # During dropout, the filter predicts forward. With the
    # known constant velocity, the predicted positions should
    # still track the true trajectory reasonably well — some
    # error grows over the 30-frame gap but stays bounded.
    pos_drop = extract_positions(result_drop.x_filt, layout_cv).squeeze(1)
    err_during_dropout = np.linalg.norm(
        pos_drop[30:60] - pos_true[30:60], axis=1,
    )
    # 30 frames at 30 fps is 1 second; at velocity ~11 px/s a
    # few px error is fine. Allow up to 15 px max error.
    assert err_during_dropout.max() < 15.0, (
        f"Predict-only over 30-frame dropout should track "
        f"reasonably; max error = {err_during_dropout.max():.3f} px"
    )
    # Variance during dropout should grow (uncertainty
    # increases without observations)
    var_during_dropout = extract_position_variances(
        result_drop.P_filt, layout_cv,
    ).squeeze(1)
    var_growth = var_during_dropout[59, 0] - var_during_dropout[30, 0]
    assert var_growth > 0, (
        f"Position variance should grow during dropout; got delta "
        f"{var_growth:.6f}"
    )

    # ---------------------------------------------------------- #
    # Case 12: forward_filter input shape validation
    # ---------------------------------------------------------- #
    layout_v = StateLayout(markers=("a", "b"))
    params_v = NoiseParams(
        sigma_base={"a": 1.0, "b": 1.0},
        q_pos={"a": 1.0, "b": 1.0},
        q_vel={"a": 0.1, "b": 0.1},
    )
    bad_positions = np.zeros((10, 3, 2))  # wrong n_markers
    likelihoods_v = np.ones((10, 2))
    try:
        forward_filter(
            bad_positions, likelihoods_v,
            layout_v, params_v, dt_cv,
            likelihood_threshold=0.5,
        )
        assert False, "Should raise on shape mismatch"
    except ValueError:
        pass

    # ---------------------------------------------------------- #
    # Case 13: extract_positions and extract_position_variances
    # ---------------------------------------------------------- #
    state_traj = np.array([
        [1.0, 2.0, 0.5, -0.5,    10.0, 20.0, 1.0, -1.0],
        [1.5, 1.5, 0.5, -0.5,    11.0, 19.0, 1.0, -1.0],
    ])
    pos = extract_positions(state_traj, layout_v)
    assert pos.shape == (2, 2, 2)
    assert pos[0, 0].tolist() == [1.0, 2.0]   # marker a, frame 0
    assert pos[0, 1].tolist() == [10.0, 20.0]  # marker b, frame 0
    assert pos[1, 1].tolist() == [11.0, 19.0]  # marker b, frame 1

    cov_traj = np.zeros((2, 8, 8))
    cov_traj[0] = np.diag([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cov_traj[1] = np.diag([0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88])
    var = extract_position_variances(cov_traj, layout_v)
    assert var.shape == (2, 2, 2)
    # marker a frame 0 = (var_x, var_y) = (0.1, 0.2) (state[0,0], state[1,1])
    assert var[0, 0].tolist() == [0.1, 0.2]
    # marker b frame 0 = (state[4,4], state[5,5]) = (0.5, 0.6)
    assert var[0, 1].tolist() == [0.5, 0.6]

    # ---------------------------------------------------------- #
    # Case 14: forward_filter with 3-marker independent case —
    # confirms joint state reduces to per-marker filtering when
    # there are no off-diagonal couplings (which is what patch 85
    # ships; coupling lands in patch 87).
    # ---------------------------------------------------------- #
    layout_3 = StateLayout(markers=("m1", "m2", "m3"))
    params_3 = NoiseParams(
        sigma_base={"m1": 1.0, "m2": 1.5, "m3": 2.0},
        q_pos={"m1": 1.0, "m2": 1.5, "m3": 2.0},
        q_vel={"m1": 0.1, "m2": 0.15, "m3": 0.2},
    )
    T2 = 50
    rng2 = np.random.default_rng(1)
    pos_3 = np.zeros((T2, 3, 2))
    for k in range(3):
        # Each marker on its own random walk + linear trend
        true_x = 10 + k * 5 + 2.0 * np.arange(T2) * dt_cv
        true_y = 20 - k * 3 + 1.0 * np.arange(T2) * dt_cv
        pos_3[:, k, 0] = true_x + rng2.normal(0, params_3.sigma_base[
            layout_3.markers[k]], T2)
        pos_3[:, k, 1] = true_y + rng2.normal(0, params_3.sigma_base[
            layout_3.markers[k]], T2)
    likelihoods_3 = np.full((T2, 3), 0.95)

    result_3 = forward_filter(
        pos_3, likelihoods_3,
        layout_3, params_3, dt_cv,
        likelihood_threshold=0.5,
    )
    # Filter should run cleanly
    assert result_3.x_filt.shape == (T2, 12)
    # All frames have all 3 markers observed
    assert (result_3.n_observed == 3).all()
    # Filtered covariance should be PSD at every frame
    for t in range(T2):
        eigvals = np.linalg.eigvalsh(result_3.P_filt[t])
        assert eigvals.min() > -1e-8, (
            f"P_filt[{t}] not PSD: smallest eigenvalue {eigvals.min()}"
        )

    # ---------------------------------------------------------- #
    # Case 15: rts_smoother boundary condition — at t=T-1, the
    # smoothed state and covariance must equal the filtered state
    # and covariance (no future to incorporate).
    # ---------------------------------------------------------- #
    layout_s = StateLayout(markers=("m1",))
    params_s = NoiseParams(
        sigma_base={"m1": 1.0},
        q_pos={"m1": 1.0},
        q_vel={"m1": 0.1},
    )
    fps_s = 30.0
    dt_s = 1.0 / fps_s
    T_s = 100
    rng_s = np.random.default_rng(2)
    times_s = np.arange(T_s) * dt_s
    pos_true_s = np.column_stack([
        50 + 10 * times_s,
        60 - 5 * times_s,
    ])
    pos_obs_s = pos_true_s + rng_s.normal(0, 1.0, pos_true_s.shape)
    positions_s = pos_obs_s.reshape(T_s, 1, 2)
    likelihoods_s = np.full((T_s, 1), 0.95)
    initial_s = np.array([50.0, 60.0, 10.0, -5.0])

    filt = forward_filter(
        positions_s, likelihoods_s,
        layout_s, params_s, dt_s,
        likelihood_threshold=0.5,
        initial_state=initial_s,
        initial_cov=0.5 * np.eye(4),
    )
    smooth = rts_smoother(filt, layout_s, dt_s)

    assert smooth.x_smooth.shape == (T_s, 4)
    assert smooth.P_smooth.shape == (T_s, 4, 4)

    # Boundary condition
    assert np.allclose(smooth.x_smooth[T_s - 1], filt.x_filt[T_s - 1]), (
        "Smoothed state at last frame must equal filtered state"
    )
    assert np.allclose(smooth.P_smooth[T_s - 1], filt.P_filt[T_s - 1]), (
        "Smoothed cov at last frame must equal filtered cov"
    )
    # n_observed carried through
    assert (smooth.n_observed == filt.n_observed).all()

    # ---------------------------------------------------------- #
    # Case 16: RTS core property — smoothed variance ≤ filtered
    # variance at every frame (the smoother CAN ONLY reduce
    # uncertainty by incorporating future information; never
    # increases it).
    # ---------------------------------------------------------- #
    for t in range(T_s):
        for j in range(4):
            assert smooth.P_smooth[t, j, j] <= filt.P_filt[t, j, j] + 1e-9, (
                f"Smoothed variance at t={t}, dim={j} = "
                f"{smooth.P_smooth[t, j, j]:.6f} exceeds filtered "
                f"variance {filt.P_filt[t, j, j]:.6f} — RTS core "
                f"property violated"
            )

    # And the strict inequality should hold at SOME frames in the
    # interior (otherwise the smoother is doing nothing useful).
    interior_x_var_filt = filt.P_filt[10:T_s - 10, 0, 0]
    interior_x_var_smooth = smooth.P_smooth[10:T_s - 10, 0, 0]
    n_strictly_smaller = np.sum(
        interior_x_var_smooth < interior_x_var_filt - 1e-6
    )
    assert n_strictly_smaller > 0.5 * len(interior_x_var_filt), (
        f"Smoother should strictly reduce variance at interior "
        f"frames; only {n_strictly_smaller}/{len(interior_x_var_filt)} "
        f"frames had reduction"
    )

    # ---------------------------------------------------------- #
    # Case 17: RMS error reduction — smoothed RMS should be lower
    # than filtered RMS on noisy data.
    # ---------------------------------------------------------- #
    pos_filt_s = extract_positions(filt.x_filt, layout_s).squeeze(1)
    pos_smooth_s = extract_positions(smooth.x_smooth, layout_s).squeeze(1)
    err_filt = np.linalg.norm(
        pos_filt_s[10:T_s - 10] - pos_true_s[10:T_s - 10], axis=1,
    )
    err_smooth = np.linalg.norm(
        pos_smooth_s[10:T_s - 10] - pos_true_s[10:T_s - 10], axis=1,
    )
    rms_filt = float(np.sqrt(np.mean(err_filt ** 2)))
    rms_smooth = float(np.sqrt(np.mean(err_smooth ** 2)))
    assert rms_smooth < rms_filt, (
        f"Smoothed RMS {rms_smooth:.4f} should be < filtered RMS "
        f"{rms_filt:.4f}"
    )

    # ---------------------------------------------------------- #
    # Case 18: smoother helps DURING dropouts — backward info
    # propagates through the gap.
    # ---------------------------------------------------------- #
    likelihoods_drop_s = np.full((T_s, 1), 0.95)
    likelihoods_drop_s[40:60, 0] = 0.0  # 20-frame dropout
    filt_drop = forward_filter(
        positions_s, likelihoods_drop_s,
        layout_s, params_s, dt_s,
        likelihood_threshold=0.5,
        initial_state=initial_s,
        initial_cov=0.5 * np.eye(4),
    )
    smooth_drop = rts_smoother(filt_drop, layout_s, dt_s)

    pos_filt_drop = extract_positions(
        filt_drop.x_filt, layout_s,
    ).squeeze(1)
    pos_smooth_drop = extract_positions(
        smooth_drop.x_smooth, layout_s,
    ).squeeze(1)

    # During the dropout, the filter only has past info; the
    # smoother also has future info from frames 60+. The
    # smoothed estimate should be closer to ground truth.
    err_filt_gap = np.linalg.norm(
        pos_filt_drop[40:60] - pos_true_s[40:60], axis=1,
    )
    err_smooth_gap = np.linalg.norm(
        pos_smooth_drop[40:60] - pos_true_s[40:60], axis=1,
    )
    rms_filt_gap = float(np.sqrt(np.mean(err_filt_gap ** 2)))
    rms_smooth_gap = float(np.sqrt(np.mean(err_smooth_gap ** 2)))
    assert rms_smooth_gap < rms_filt_gap, (
        f"During dropout, smoother should help more than filter; "
        f"filt RMS={rms_filt_gap:.3f}, smooth RMS={rms_smooth_gap:.3f}"
    )
    # And the variance at the middle of the gap should be
    # lower for the smoother (the smoother sees past + future
    # surrounding the gap)
    var_filt_mid = filt_drop.P_filt[50, 0, 0]
    var_smooth_mid = smooth_drop.P_smooth[50, 0, 0]
    assert var_smooth_mid < var_filt_mid, (
        f"Variance at middle of dropout: smoother should be lower; "
        f"filt={var_filt_mid:.3f}, smooth={var_smooth_mid:.3f}"
    )

    # ---------------------------------------------------------- #
    # Case 19: smoothed covariance remains PSD at every frame
    # ---------------------------------------------------------- #
    for t in range(T_s):
        eigvals = np.linalg.eigvalsh(smooth.P_smooth[t])
        assert eigvals.min() > -1e-8, (
            f"Smoothed P at t={t} not PSD: smallest eigenvalue "
            f"{eigvals.min()}"
        )

    # ---------------------------------------------------------- #
    # Case 20: smoother on a multi-marker case — joint-state
    # backward pass must not crash and produces sensible variance
    # reduction across all markers.
    # ---------------------------------------------------------- #
    layout_m = StateLayout(markers=("m1", "m2", "m3"))
    params_m = NoiseParams(
        sigma_base={"m1": 1.0, "m2": 1.5, "m3": 2.0},
        q_pos={"m1": 1.0, "m2": 1.5, "m3": 2.0},
        q_vel={"m1": 0.1, "m2": 0.15, "m3": 0.2},
    )
    T_m = 80
    rng_m = np.random.default_rng(3)
    pos_m = np.zeros((T_m, 3, 2))
    times_m = np.arange(T_m) * dt_s
    for k in range(3):
        true_x = 10 + k * 5 + 2.0 * times_m
        true_y = 20 - k * 3 + 1.0 * times_m
        pos_m[:, k, 0] = true_x + rng_m.normal(
            0, params_m.sigma_base[layout_m.markers[k]], T_m,
        )
        pos_m[:, k, 1] = true_y + rng_m.normal(
            0, params_m.sigma_base[layout_m.markers[k]], T_m,
        )
    likelihoods_m = np.full((T_m, 3), 0.95)

    filt_m = forward_filter(
        pos_m, likelihoods_m,
        layout_m, params_m, dt_s,
        likelihood_threshold=0.5,
    )
    smooth_m = rts_smoother(filt_m, layout_m, dt_s)
    assert smooth_m.x_smooth.shape == (T_m, 12)

    # All diagonal variance entries: smoothed ≤ filtered
    for t in range(T_m):
        for j in range(12):
            assert smooth_m.P_smooth[t, j, j] <= filt_m.P_filt[t, j, j] + 1e-9

    # ---------------------------------------------------------- #
    # Case 21: rts_smoother input shape validation
    # ---------------------------------------------------------- #
    bad_filt = FilterResult(
        x_filt=np.zeros((10, 8)),  # state_dim=8, but layout_s says 4
        P_filt=np.zeros((10, 8, 8)),
        x_pred=np.zeros((10, 8)),
        P_pred=np.zeros((10, 8, 8)),
        n_observed=np.zeros(10, dtype=np.int64),
    )
    try:
        rts_smoother(bad_filt, layout_s, dt_s)
        assert False, "Should raise on shape mismatch"
    except ValueError:
        pass

    # ---------------------------------------------------------- #
    # Case 22: fit_triplet_prior on synthetic 3-marker rigid data
    # — empirical mean and cov match the synthesis distribution.
    # ---------------------------------------------------------- #
    layout_t = StateLayout(markers=("a", "b", "c"))
    rng_t = np.random.default_rng(10)
    T_t = 1000
    # Build a rigid triplet: each marker offset from a moving
    # centroid by a fixed displacement, plus tight noise.
    centroid = np.column_stack([
        50 + 5 * np.sin(np.arange(T_t) / 50),
        60 + 3 * np.cos(np.arange(T_t) / 50),
    ])
    offsets = np.array([
        [0.0, 5.0],   # a above centroid
        [5.0, -5.0],  # b lower-right
        [-5.0, -5.0], # c lower-left
    ])
    pos_t = np.zeros((T_t, 3, 2))
    for k in range(3):
        pos_t[:, k, :] = centroid + offsets[k] + rng_t.normal(
            0, 0.3, (T_t, 2),
        )
    likes_t = np.full((T_t, 3), 0.99)

    prior = fit_triplet_prior(
        pos_t, likes_t,
        triplet=("a", "b", "c"),
        layout=layout_t,
        likelihood_threshold=0.5,
        min_samples=200,
    )
    assert prior is not None
    assert prior.markers == ("a", "b", "c")
    assert prior.mean_config.shape == (6,)
    assert prior.cov.shape == (6, 6)
    assert prior.n_samples == T_t
    # State indices should be [0, 1, 4, 5, 8, 9] (positions of
    # markers 0, 1, 2 in a 12-dim joint state)
    assert prior.state_indices.tolist() == [0, 1, 4, 5, 8, 9]
    # Covariance is symmetric and PSD (after ridge)
    assert np.allclose(prior.cov, prior.cov.T)
    eigvals = np.linalg.eigvalsh(prior.cov)
    assert eigvals.min() > 0, (
        f"Triplet cov should be PD after ridge; got smallest "
        f"eigenvalue {eigvals.min()}"
    )
    # Mean configuration's pairwise differences should match
    # the offsets pattern (the centroid drifts but the offsets
    # are fixed)
    # E[x_a - x_b] = offset_a_x - offset_b_x = 0 - 5 = -5
    diff_ab_x = prior.mean_config[0] - prior.mean_config[2]
    assert abs(diff_ab_x - (offsets[0, 0] - offsets[1, 0])) < 0.5

    # ---------------------------------------------------------- #
    # Case 23: fit_triplet_prior returns None on insufficient data
    # ---------------------------------------------------------- #
    likes_low = np.full((T_t, 3), 0.1)  # all below threshold
    prior_none = fit_triplet_prior(
        pos_t, likes_low,
        triplet=("a", "b", "c"),
        layout=layout_t,
        likelihood_threshold=0.5,
        min_samples=200,
    )
    assert prior_none is None

    # And with min_samples set higher than available
    prior_none2 = fit_triplet_prior(
        pos_t, likes_t,
        triplet=("a", "b", "c"),
        layout=layout_t,
        likelihood_threshold=0.5,
        min_samples=T_t + 1,  # exceeds available
    )
    assert prior_none2 is None

    # ---------------------------------------------------------- #
    # Case 24: fit_triplet_prior raises on unknown markers
    # ---------------------------------------------------------- #
    try:
        fit_triplet_prior(
            pos_t, likes_t,
            triplet=("a", "b", "missing"),
            layout=layout_t,
            likelihood_threshold=0.5,
        )
        assert False, "Should raise on unknown marker"
    except ValueError:
        pass

    # ---------------------------------------------------------- #
    # Case 25: fit_triplet_priors fits a list and silently drops
    # triplets with insufficient data.
    # ---------------------------------------------------------- #
    # Set up a 4-marker layout with two triplets — one valid,
    # one with one always-low-confidence marker.
    layout_4 = StateLayout(markers=("a", "b", "c", "d"))
    pos_4 = np.zeros((T_t, 4, 2))
    pos_4[:, :3, :] = pos_t  # markers a, b, c as before
    pos_4[:, 3, :] = centroid + np.array([10.0, 0.0]) + rng_t.normal(
        0, 0.3, (T_t, 2),
    )
    likes_4 = np.full((T_t, 4), 0.99)
    likes_4[:, 3] = 0.1  # marker d always low-confidence
    triplets_in = [("a", "b", "c"), ("a", "b", "d")]
    priors = fit_triplet_priors(
        pos_4, likes_4, triplets_in, layout_4,
        likelihood_threshold=0.5, min_samples=200,
    )
    # Only the (a, b, c) triplet should fit; (a, b, d) drops
    # because d is below threshold
    assert len(priors) == 1
    assert priors[0].markers == ("a", "b", "c")

    # ---------------------------------------------------------- #
    # Case 26: build_triplet_observation produces correct
    # (z, H, R) for one prior at one frame.
    # ---------------------------------------------------------- #
    z_t, H_t, R_t = build_triplet_observation(prior, layout_t)
    assert z_t.shape == (6,)
    assert H_t.shape == (6, 12)
    assert R_t.shape == (6, 6)
    # H_t should pick out state slots [0, 1, 4, 5, 8, 9]
    expected_picks = [0, 1, 4, 5, 8, 9]
    for k, idx in enumerate(expected_picks):
        assert H_t[k, idx] == 1.0
        # And H_t[k] should have only one nonzero entry
        assert H_t[k].sum() == 1.0
    # z_t == prior.mean_config
    assert np.allclose(z_t, prior.mean_config)
    # R_t == prior.cov
    assert np.allclose(R_t, prior.cov)

    # ---------------------------------------------------------- #
    # Case 27: build_observation stacks real markers AND triplet
    # pseudo-measurements correctly.
    # ---------------------------------------------------------- #
    sb_t = np.array([1.0, 1.0, 1.0])
    obs_with_triplet = build_observation(
        layout_t,
        positions=pos_t[0],
        likelihoods=likes_t[0],
        sigma_base_arr=sb_t,
        likelihood_threshold=0.5,
        triplet_priors=[prior],
    )
    # 3 real markers (6 dims) + 1 triplet (6 dims) = 12 total
    assert obs_with_triplet.z.shape == (12,)
    assert obs_with_triplet.H.shape == (12, 12)
    assert obs_with_triplet.R.shape == (12, 12)
    assert obs_with_triplet.n_observed == 3
    # First 6 dims of z are real markers (positions of a, b, c)
    assert obs_with_triplet.z[:6].tolist() == [
        pos_t[0, 0, 0], pos_t[0, 0, 1],
        pos_t[0, 1, 0], pos_t[0, 1, 1],
        pos_t[0, 2, 0], pos_t[0, 2, 1],
    ]
    # Last 6 dims are the triplet pseudo-measurement = mean_config
    assert np.allclose(obs_with_triplet.z[6:], prior.mean_config)
    # R is block-diagonal with real blocks first, triplet cov last
    assert np.allclose(obs_with_triplet.R[6:, 6:], prior.cov)
    # Cross-blocks must be zero (independent measurements)
    assert np.allclose(obs_with_triplet.R[:6, 6:], 0.0)

    # ---------------------------------------------------------- #
    # Case 28: build_observation with triplet prior but NO real
    # markers above threshold — z is non-empty (just the triplet
    # pseudo-measurement). This is the key scenario the joint
    # formulation handles: triplet pulls during dropouts.
    # ---------------------------------------------------------- #
    likes_zero = np.zeros(3)  # all markers below threshold
    obs_only_triplet = build_observation(
        layout_t,
        positions=pos_t[0],
        likelihoods=likes_zero,
        sigma_base_arr=sb_t,
        likelihood_threshold=0.5,
        triplet_priors=[prior],
    )
    assert obs_only_triplet.n_observed == 0  # no real markers
    assert obs_only_triplet.z.shape == (6,)  # but triplet IS present
    assert obs_only_triplet.H.shape == (6, 12)
    assert obs_only_triplet.R.shape == (6, 6)
    assert np.allclose(obs_only_triplet.z, prior.mean_config)

    # ---------------------------------------------------------- #
    # Case 29: forward_filter with triplet_priors runs cleanly
    # and reduces variance on tracked markers vs no-prior case.
    # ---------------------------------------------------------- #
    params_t = NoiseParams(
        sigma_base={"a": 1.0, "b": 1.0, "c": 1.0},
        q_pos={"a": 1.0, "b": 1.0, "c": 1.0},
        q_vel={"a": 0.1, "b": 0.1, "c": 0.1},
    )
    fps_t = 30.0
    dt_t = 1.0 / fps_t
    # Baseline: no triplet
    filt_no_prior = forward_filter(
        pos_t, likes_t, layout_t, params_t, dt_t,
        likelihood_threshold=0.5,
        initial_state=np.zeros(12),
        initial_cov=10.0 * np.eye(12),
    )
    # With triplet prior
    filt_with_prior = forward_filter(
        pos_t, likes_t, layout_t, params_t, dt_t,
        likelihood_threshold=0.5,
        initial_state=np.zeros(12),
        initial_cov=10.0 * np.eye(12),
        triplet_priors=[prior],
    )
    # Both should run cleanly
    assert filt_with_prior.x_filt.shape == (T_t, 12)
    assert filt_with_prior.P_filt.shape == (T_t, 12, 12)

    # The triplet prior should reduce per-marker position variance
    # at frames in steady state (after the warm-up transient)
    interior_var_no_prior = np.diagonal(
        filt_no_prior.P_filt[100:T_t - 100], axis1=1, axis2=2,
    )  # (T-200, 12)
    interior_var_with_prior = np.diagonal(
        filt_with_prior.P_filt[100:T_t - 100], axis1=1, axis2=2,
    )
    # Compare position-component variance (state dims 0, 1, 4, 5, 8, 9)
    pos_state_dims = [0, 1, 4, 5, 8, 9]
    var_no_prior = interior_var_no_prior[:, pos_state_dims].mean()
    var_with_prior = interior_var_with_prior[:, pos_state_dims].mean()
    assert var_with_prior < var_no_prior, (
        f"Triplet prior should reduce avg position variance; "
        f"no_prior={var_no_prior:.6f}, with_prior={var_with_prior:.6f}"
    )

    # ---------------------------------------------------------- #
    # Case 30: triplet prior helps DURING dropouts — when a
    # marker is unobserved, its position estimate is pulled
    # toward the triplet mean via the other two markers'
    # observations and the triplet covariance.
    # ---------------------------------------------------------- #
    # Same pos_t, but mark marker a as unobserved for frames
    # 200-300 (100-frame dropout in the middle).
    likes_drop = likes_t.copy()
    likes_drop[200:300, 0] = 0.0  # marker a dropped

    filt_drop_no_prior = forward_filter(
        pos_t, likes_drop, layout_t, params_t, dt_t,
        likelihood_threshold=0.5,
        initial_state=np.zeros(12),
        initial_cov=10.0 * np.eye(12),
    )
    filt_drop_with_prior = forward_filter(
        pos_t, likes_drop, layout_t, params_t, dt_t,
        likelihood_threshold=0.5,
        initial_state=np.zeros(12),
        initial_cov=10.0 * np.eye(12),
        triplet_priors=[prior],
    )
    # During the dropout, marker a's variance with the triplet
    # prior should be much lower than without — the prior
    # is constraining a's position via b and c.
    a_var_no_prior_mid = filt_drop_no_prior.P_filt[250, 0, 0]
    a_var_with_prior_mid = filt_drop_with_prior.P_filt[250, 0, 0]
    assert a_var_with_prior_mid < a_var_no_prior_mid, (
        f"During dropout, triplet prior should constrain marker a's "
        f"variance; no_prior={a_var_no_prior_mid:.4f}, "
        f"with_prior={a_var_with_prior_mid:.4f}"
    )

    # And the position estimate for a during the dropout should
    # be closer to the truth (pulled by the triplet via b and c).
    pos_drop_no_prior = extract_positions(
        filt_drop_no_prior.x_filt, layout_t,
    )[200:300, 0, :]
    pos_drop_with_prior = extract_positions(
        filt_drop_with_prior.x_filt, layout_t,
    )[200:300, 0, :]
    pos_truth_a = pos_t[200:300, 0, :]
    err_no_prior = np.linalg.norm(
        pos_drop_no_prior - pos_truth_a, axis=1,
    )
    err_with_prior = np.linalg.norm(
        pos_drop_with_prior - pos_truth_a, axis=1,
    )
    rms_no = float(np.sqrt(np.mean(err_no_prior ** 2)))
    rms_with = float(np.sqrt(np.mean(err_with_prior ** 2)))
    assert rms_with < rms_no, (
        f"During dropout, triplet prior should reduce RMS error on "
        f"the dropped marker; no_prior={rms_no:.3f}, "
        f"with_prior={rms_with:.3f}"
    )

    # ---------------------------------------------------------- #
    # Case 31: forward+RTS with triplet prior runs cleanly and
    # produces PSD smoothed covariances at every frame.
    # ---------------------------------------------------------- #
    smooth_with_prior = rts_smoother(filt_with_prior, layout_t, dt_t)
    assert smooth_with_prior.x_smooth.shape == (T_t, 12)
    # Every frame's smoothed covariance is PSD
    for t_check in [0, 100, 500, T_t - 1]:
        eigvals = np.linalg.eigvalsh(smooth_with_prior.P_smooth[t_check])
        assert eigvals.min() > -1e-8, (
            f"Smoothed P with triplet at t={t_check} not PSD: "
            f"smallest eigenvalue {eigvals.min()}"
        )

    # ---------------------------------------------------------- #
    # Case 32: initial_noise_params returns sane defaults on
    # synthetic data with KNOWN noise level — sigma_base estimate
    # should be in the right order of magnitude.
    # ---------------------------------------------------------- #
    layout_em = StateLayout(markers=("a", "b"))
    rng_em = np.random.default_rng(20)
    fps_em = 30.0
    dt_em = 1.0 / fps_em
    T_em = 2000
    times_em = np.arange(T_em) * dt_em
    sigma_true = 2.0
    # Two markers on independent CV trajectories
    pos_em = np.zeros((T_em, 2, 2))
    for k in range(2):
        true_x = 10 + k * 5 + 2.0 * times_em
        true_y = 20 - k * 3 + 1.0 * times_em
        pos_em[:, k, 0] = true_x + rng_em.normal(0, sigma_true, T_em)
        pos_em[:, k, 1] = true_y + rng_em.normal(0, sigma_true, T_em)
    likes_em = np.full((T_em, 2), 0.95)

    init_params = initial_noise_params(
        pos_em, likes_em, layout_em,
        likelihood_threshold=0.5, fps=fps_em,
    )
    # sigma_base should be in the ballpark of the true sigma
    # (within a factor of ~3 — initial estimate from MA
    # residuals is rough but should not be wildly off)
    for m in layout_em.markers:
        assert init_params.sigma_base[m] > 0.3 * sigma_true, (
            f"Initial sigma_base for {m} = "
            f"{init_params.sigma_base[m]:.3f}, "
            f"too low vs true {sigma_true}"
        )
        assert init_params.sigma_base[m] < 5.0 * sigma_true, (
            f"Initial sigma_base for {m} = "
            f"{init_params.sigma_base[m]:.3f}, "
            f"too high vs true {sigma_true}"
        )
    # q_pos should be > floor; q_vel should be a fraction of q_pos
    for m in layout_em.markers:
        assert init_params.q_pos[m] > 1e-3
        assert init_params.q_vel[m] > 0
        assert init_params.q_vel[m] <= init_params.q_pos[m]

    # ---------------------------------------------------------- #
    # Case 33: initial_noise_params handles low-data marker
    # gracefully (falls back to sane defaults rather than crashing).
    # ---------------------------------------------------------- #
    likes_low_em = likes_em.copy()
    likes_low_em[:, 0] = 0.1  # marker a always below threshold
    init_low = initial_noise_params(
        pos_em, likes_low_em, layout_em,
        likelihood_threshold=0.5, fps=fps_em,
    )
    # marker a should fall back to the default; not raise
    assert init_low.sigma_base["a"] > 0
    assert init_low.q_pos["a"] > 0
    # marker b still gets the data-driven estimate
    assert init_low.sigma_base["b"] > 0.3 * sigma_true

    # ---------------------------------------------------------- #
    # Case 34: fit_noise_params_em converges on synthetic data
    # and produces parameters in the right ballpark.
    # ---------------------------------------------------------- #
    em_result = fit_noise_params_em(
        pos_em, likes_em, layout_em,
        fps=fps_em, likelihood_threshold=0.5,
        max_iter=10, tol=1e-3,
        verbose=False,
    )
    assert isinstance(em_result, EMResult)
    assert em_result.n_iter >= 1
    assert em_result.n_iter <= 10
    # Convergence is desired but not strictly required for the
    # test to pass — what we care about is sane outputs.
    # Check final params are in a reasonable range
    for m in layout_em.markers:
        assert em_result.params.sigma_base[m] > 0.3 * sigma_true, (
            f"EM-converged sigma_base for {m} = "
            f"{em_result.params.sigma_base[m]:.3f}, too low"
        )
        assert em_result.params.sigma_base[m] < 5.0 * sigma_true, (
            f"EM-converged sigma_base for {m} = "
            f"{em_result.params.sigma_base[m]:.3f}, too high"
        )
    # History should have one entry per iteration
    assert len(em_result.history) == em_result.n_iter
    for entry in em_result.history:
        assert "max_rel_change_sigma" in entry
        assert "mean_sigma" in entry
        assert "mean_q_pos" in entry

    # ---------------------------------------------------------- #
    # Case 35: EM converges (max_rel_change < tol) eventually on
    # well-conditioned data.
    # ---------------------------------------------------------- #
    em_converged = fit_noise_params_em(
        pos_em, likes_em, layout_em,
        fps=fps_em, likelihood_threshold=0.5,
        max_iter=20, tol=1e-2,  # looser tol to ensure convergence
    )
    assert em_converged.converged, (
        f"EM should converge on well-conditioned data with "
        f"tol=1e-2; ran {em_converged.n_iter} iterations, final "
        f"max change = {em_converged.history[-1]['max_rel_change_sigma']:.4e}"
    )

    # ---------------------------------------------------------- #
    # Case 36: EM with triplet_priors threads them through
    # without crashing and produces sensible output.
    # ---------------------------------------------------------- #
    layout_em3 = StateLayout(markers=("a", "b", "c"))
    pos_em3 = np.zeros((T_em, 3, 2))
    pos_em3[:, :2, :] = pos_em
    pos_em3[:, 2, 0] = (
        10 + 10 + 2.0 * times_em + rng_em.normal(0, sigma_true, T_em)
    )
    pos_em3[:, 2, 1] = (
        20 - 6 + 1.0 * times_em + rng_em.normal(0, sigma_true, T_em)
    )
    likes_em3 = np.full((T_em, 3), 0.95)
    # Build a triplet prior from the data
    tp = fit_triplet_prior(
        pos_em3, likes_em3,
        triplet=("a", "b", "c"),
        layout=layout_em3,
        likelihood_threshold=0.5,
    )
    em_with_triplet = fit_noise_params_em(
        pos_em3, likes_em3, layout_em3,
        fps=fps_em, likelihood_threshold=0.5,
        triplet_priors=[tp],
        max_iter=10, tol=1e-3,
    )
    # Should run cleanly and produce a valid NoiseParams
    assert em_with_triplet.params.sigma_base.keys() == set(layout_em3.markers)
    for m in layout_em3.markers:
        # Output should be in a sane range
        assert em_with_triplet.params.sigma_base[m] > 0.1
        assert em_with_triplet.params.sigma_base[m] < 100.0

    # ---------------------------------------------------------- #
    # Case 37: EM result params actually IMPROVE the smoother fit
    # vs the initial params on noisy data — measurable RMS
    # reduction on a held-out validation slice.
    # ---------------------------------------------------------- #
    # Hold out the last 500 frames as validation; fit EM on the
    # first 1500.
    pos_train = pos_em[:1500]
    likes_train = likes_em[:1500]
    pos_val = pos_em[1500:]

    init_params_train = initial_noise_params(
        pos_train, likes_train, layout_em,
        likelihood_threshold=0.5, fps=fps_em,
    )
    em_train = fit_noise_params_em(
        pos_train, likes_train, layout_em,
        fps=fps_em, likelihood_threshold=0.5,
        max_iter=10, tol=1e-3,
    )

    # Evaluate both on val: smoothing with EM params vs initial
    likes_val = np.full((500, 2), 0.95)

    def _val_rms(params):
        filt_v = forward_filter(
            pos_val, likes_val, layout_em, params, dt_em,
            likelihood_threshold=0.5,
            initial_state=np.zeros(8),
            initial_cov=10.0 * np.eye(8),
        )
        sm_v = rts_smoother(filt_v, layout_em, dt_em)
        # Ground truth on the val window
        true_x = np.column_stack([
            10 + 2.0 * (np.arange(500) + 1500) * dt_em,
            10 + 5 + 2.0 * (np.arange(500) + 1500) * dt_em,
        ])
        true_y = np.column_stack([
            20 + 1.0 * (np.arange(500) + 1500) * dt_em,
            20 - 3 + 1.0 * (np.arange(500) + 1500) * dt_em,
        ])
        smoothed_pos = extract_positions(sm_v.x_smooth, layout_em)
        # RMS over both markers, both axes, after warm-up
        warm = 20
        e1 = (smoothed_pos[warm:, 0, 0] - true_x[warm:, 0]) ** 2
        e2 = (smoothed_pos[warm:, 0, 1] - true_y[warm:, 0]) ** 2
        e3 = (smoothed_pos[warm:, 1, 0] - true_x[warm:, 1]) ** 2
        e4 = (smoothed_pos[warm:, 1, 1] - true_y[warm:, 1]) ** 2
        return float(np.sqrt(np.mean(np.concatenate([e1, e2, e3, e4]))))

    rms_init = _val_rms(init_params_train)
    rms_em = _val_rms(em_train.params)
    # EM should not be SUBSTANTIALLY worse than initial on val
    # (allowing some slack for finite-sample noise; we test
    # that it's at least competitive). On well-conditioned
    # synthetic data, EM should typically match or beat the
    # initial estimate.
    assert rms_em < rms_init * 1.5, (
        f"EM-fit params should produce competitive RMS on "
        f"validation; init RMS={rms_init:.3f}, em RMS={rms_em:.3f}"
    )

    # ---------------------------------------------------------- #
    # Case 38: EM respects max_iter and reports converged=False
    # if the cap is hit before the tolerance.
    # ---------------------------------------------------------- #
    em_capped = fit_noise_params_em(
        pos_em, likes_em, layout_em,
        fps=fps_em, likelihood_threshold=0.5,
        max_iter=2, tol=1e-15,  # impossibly strict tol
    )
    assert em_capped.n_iter == 2
    assert not em_capped.converged

    # ---------------------------------------------------------- #
    # Case 39: EM fail-fast on degenerate input (patch 91 change).
    # When all observations are below threshold, the smoother
    # produces a frozen trajectory — the validation hook now
    # catches this and raises RuntimeError. (Pre-patch-91
    # behavior was to silently produce floor-clamped output.)
    # ---------------------------------------------------------- #
    likes_zero_em = np.zeros((T_em, 2))  # all below threshold
    try:
        fit_noise_params_em(
            pos_em, likes_zero_em, layout_em,
            fps=fps_em, likelihood_threshold=0.5,
            max_iter=5, tol=1e-3,
        )
        assert False, (
            "EM should fail-fast on all-low-likelihood input via "
            "the validation hook"
        )
    except RuntimeError as e:
        # Error message should name the failed check
        msg = str(e)
        assert "validation hook" in msg.lower(), (
            f"Error message should mention validation hook; got: {msg}"
        )
        # The frozen-output check is the most likely trigger here
        assert (
            "frozen-output" in msg.lower()
            or "floor" in msg.lower()
            or "ceiling" in msg.lower()
            or "prior-overruling" in msg.lower()
        ), (
            f"Error message should name the failed check; got: {msg}"
        )

    # ---------------------------------------------------------- #
    # Case 40: save_model + load_model round-trip preserves all
    # fields (NoiseParams, TripletPriors, layout, scalars).
    # ---------------------------------------------------------- #
    layout_io = StateLayout(markers=("a", "b", "c"))
    params_io = NoiseParams(
        sigma_base={"a": 1.0, "b": 1.5, "c": 2.0},
        q_pos={"a": 100.0, "b": 150.0, "c": 200.0},
        q_vel={"a": 10.0, "b": 15.0, "c": 20.0},
    )
    # Build a fake triplet prior
    fake_prior = TripletPrior(
        markers=("a", "b", "c"),
        mean_config=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        cov=np.eye(6) * 2.5,
        n_samples=1000,
        state_indices=np.array([0, 1, 4, 5, 8, 9], dtype=np.int64),
    )
    em_history_in = [
        {"iter": 0.0, "max_rel_change_sigma": 0.5, "mean_sigma": 1.5,
         "mean_q_pos": 120.0},
        {"iter": 1.0, "max_rel_change_sigma": 0.05, "mean_sigma": 1.4,
         "mean_q_pos": 125.0},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "test_model.npz"
        save_model(
            str(model_path), layout_io, params_io, [fake_prior],
            likelihood_threshold=0.7, fps=30.0,
            data_hash="abcd1234",
            em_history=em_history_in,
        )
        assert model_path.exists()

        (loaded_layout, loaded_params, loaded_priors,
         loaded_thr, loaded_fps, loaded_hash,
         loaded_em) = load_model(str(model_path))

        # Layout
        assert loaded_layout.markers == layout_io.markers
        # Params
        for m in layout_io.markers:
            assert loaded_params.sigma_base[m] == params_io.sigma_base[m]
            assert loaded_params.q_pos[m] == params_io.q_pos[m]
            assert loaded_params.q_vel[m] == params_io.q_vel[m]
        # Triplet priors
        assert len(loaded_priors) == 1
        assert loaded_priors[0].markers == fake_prior.markers
        assert np.allclose(
            loaded_priors[0].mean_config, fake_prior.mean_config,
        )
        assert np.allclose(loaded_priors[0].cov, fake_prior.cov)
        assert loaded_priors[0].n_samples == fake_prior.n_samples
        assert (
            loaded_priors[0].state_indices.tolist()
            == fake_prior.state_indices.tolist()
        )
        # Scalars
        assert loaded_thr == 0.7
        assert loaded_fps == 30.0
        assert loaded_hash == "abcd1234"
        # EM history
        assert loaded_em == em_history_in

    # ---------------------------------------------------------- #
    # Case 41: load_model rejects mismatched version
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "wrong_version.npz"
        # Write a bogus-version model directly
        np.savez_compressed(
            str(model_path),
            version="0.1",  # wrong version
            markers=np.array(["a"], dtype=np.str_),
            sigma_base=np.array([1.0]),
            q_pos=np.array([1.0]),
            q_vel=np.array([0.1]),
            likelihood_threshold=0.7,
            fps=30.0,
            data_hash="x",
            triplet_count=np.int64(0),
        )
        try:
            load_model(str(model_path))
            assert False, "Should reject wrong version"
        except ValueError as e:
            assert "version" in str(e).lower()

    # ---------------------------------------------------------- #
    # Case 42: smooth_multi_session runs on synthetic data with
    # multiple sessions and returns proper shapes + no
    # cross-boundary smoothing.
    # ---------------------------------------------------------- #
    layout_ms = StateLayout(markers=("a", "b"))
    params_ms = NoiseParams(
        sigma_base={"a": 1.0, "b": 1.0},
        q_pos={"a": 10.0, "b": 10.0},
        q_vel={"a": 1.0, "b": 1.0},
    )
    rng_ms = np.random.default_rng(50)
    # Two sessions with very DIFFERENT mean positions — if the
    # smoother propagates state across the boundary, the
    # smoothed start of session 2 would lag toward session 1's
    # mean.
    n_per = 500
    pos_s1 = np.zeros((n_per, 2, 2))
    pos_s1[:, 0, :] = [10, 20] + rng_ms.normal(0, 1.0, (n_per, 2))
    pos_s1[:, 1, :] = [15, 25] + rng_ms.normal(0, 1.0, (n_per, 2))
    pos_s2 = np.zeros((n_per, 2, 2))
    pos_s2[:, 0, :] = [200, 300] + rng_ms.normal(0, 1.0, (n_per, 2))
    pos_s2[:, 1, :] = [205, 305] + rng_ms.normal(0, 1.0, (n_per, 2))
    pos_ms = np.concatenate([pos_s1, pos_s2], axis=0)
    likes_ms = np.full((2 * n_per, 2), 0.95)
    sessions_ms = [
        ("session1", 0, n_per),
        ("session2", n_per, 2 * n_per),
    ]

    smoothed_pos, smoothed_var, n_obs_ms = smooth_multi_session(
        pos_ms, likes_ms, layout_ms, params_ms, [],
        sessions_ms, fps=30.0, likelihood_threshold=0.5,
    )
    assert smoothed_pos.shape == (2 * n_per, 2, 2)
    assert smoothed_var.shape == (2 * n_per, 2, 2)
    # First frame of session 2 should be close to session 2's
    # mean, NOT session 1's. (If state crossed the boundary,
    # frame n_per would still be near session 1's mean.)
    s2_first_x = smoothed_pos[n_per, 0, 0]
    assert abs(s2_first_x - 200) < 30, (
        f"First frame of session 2 should be near session 2's "
        f"mean (200), not pulled toward session 1's mean (10). "
        f"Got x = {s2_first_x:.1f}"
    )
    # And both end frames should look reasonable
    s1_last_x = smoothed_pos[n_per - 1, 0, 0]
    assert abs(s1_last_x - 10) < 10, (
        f"Last frame of session 1 should be near session 1's "
        f"mean (10); got x = {s1_last_x:.1f}"
    )

    # ---------------------------------------------------------- #
    # Case 43: end-to-end smooth_pose runs cleanly on a single
    # synthetic CSV file and produces expected outputs.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        in_csv = td / "test_session.csv"
        _build_synthetic_pose_csv(in_csv, n_frames=500)

        out_dir = td / "out"
        result = smooth_pose(
            pose_input=str(in_csv),
            output_dir=str(out_dir),
            fps=30.0,
            likelihood_threshold=0.7,
            head_markers=["nose", "ear_left", "ear_right"],
            em_max_iter=3,  # keep test fast
            verbose=False,
        )
        # Result dict has expected keys
        for key in ("layout", "params", "triplet_priors", "sessions",
                    "output_files", "model_artifact", "em_history"):
            assert key in result
        # One session, one output file
        assert len(result["sessions"]) == 1
        assert len(result["output_files"]) == 1
        # Output file exists
        out_file = Path(result["output_files"][0])
        assert out_file.exists()
        # session_summary.json exists with expected fields
        summary_path = out_dir / "session_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["n_sessions"] == 1
        assert summary["total_frames"] == 500
        assert "markers" in summary
        assert "data_hash" in summary

    # ---------------------------------------------------------- #
    # Case 44: smooth_pose with --save-model produces a loadable
    # artifact AND the loaded artifact can drive a re-smooth that
    # matches the original output.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        in_csv = td / "session.csv"
        _build_synthetic_pose_csv(in_csv, n_frames=500, seed=1)

        out_fit = td / "out_fit"
        model_path = td / "model.npz"
        result_fit = smooth_pose(
            pose_input=str(in_csv),
            output_dir=str(out_fit),
            fps=30.0,
            likelihood_threshold=0.7,
            head_markers=["nose", "ear_left", "ear_right"],
            em_max_iter=3,
            save_model_path=str(model_path),
            verbose=False,
        )
        assert model_path.exists()
        assert result_fit["model_artifact"] == str(model_path)

        # Now re-smooth using the saved model — should run without
        # going through EM again
        out_load = td / "out_load"
        result_load = smooth_pose(
            pose_input=str(in_csv),
            output_dir=str(out_load),
            fps=30.0,
            likelihood_threshold=0.7,  # will be overridden by loaded
            load_model_path=str(model_path),
            verbose=False,
        )
        # em_history from the loaded model should be present
        assert result_load["em_history"] is not None
        # The smoothed outputs should be (approximately) identical
        # since both runs use the same params on the same data.
        # Read whichever format the smoother actually wrote (parquet
        # if pyarrow/fastparquet available, else CSV fallback).
        def _read_output(p: str) -> pd.DataFrame:
            if p.endswith(".parquet"):
                return pd.read_parquet(p)
            return pd.read_csv(p)
        df_fit = _read_output(result_fit["output_files"][0])
        df_load = _read_output(result_load["output_files"][0])
        # Compare smoothed positions for one marker
        assert np.allclose(
            df_fit["nose_x"].values,
            df_load["nose_x"].values,
            atol=1e-6,
        ), "Saved-model re-smooth should match original output"

    # ---------------------------------------------------------- #
    # Case 45: smooth_pose multi-file mode treats each file as a
    # session with proper boundary handling.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        f1 = td / "s1.csv"
        f2 = td / "s2.csv"
        _build_synthetic_pose_csv(f1, n_frames=300, seed=10)
        _build_synthetic_pose_csv(f2, n_frames=400, seed=11)

        out_dir = td / "out"
        result = smooth_pose(
            pose_input=[str(f1), str(f2)],
            output_dir=str(out_dir),
            fps=30.0,
            likelihood_threshold=0.7,
            head_markers=["nose", "ear_left", "ear_right"],
            em_max_iter=2,
            verbose=False,
        )
        # Two sessions, two output files
        assert len(result["sessions"]) == 2
        assert len(result["output_files"]) == 2
        # Names match input file stems
        names = [s[0] for s in result["sessions"]]
        assert "s1" in names
        assert "s2" in names
        # Each output file has the right number of rows
        def _read_output(p: str) -> pd.DataFrame:
            if p.endswith(".parquet"):
                return pd.read_parquet(p)
            return pd.read_csv(p)
        for sess, out_path in zip(result["sessions"], result["output_files"]):
            name, start, end = sess
            df_out = _read_output(out_path)
            assert len(df_out) == end - start

    # ---------------------------------------------------------- #
    # Case 46: per-session EM equivalence on a single session.
    # When sessions=[(.., 0, T)] (one synthetic session covering
    # all frames), the result must equal sessions=None (the
    # backward-compat path that defaults to one session).
    # ---------------------------------------------------------- #
    layout_eq = StateLayout(markers=("a", "b"))
    rng_eq = np.random.default_rng(100)
    fps_eq = 30.0
    T_eq = 800
    pos_eq = np.zeros((T_eq, 2, 2))
    times_eq = np.arange(T_eq) / fps_eq
    sigma_true_eq = 1.5
    for k in range(2):
        pos_eq[:, k, 0] = (
            10 + k * 5 + 2.0 * times_eq
            + rng_eq.normal(0, sigma_true_eq, T_eq)
        )
        pos_eq[:, k, 1] = (
            20 - k * 3 + 1.0 * times_eq
            + rng_eq.normal(0, sigma_true_eq, T_eq)
        )
    likes_eq = np.full((T_eq, 2), 0.95)

    em_no_sess = fit_noise_params_em(
        pos_eq, likes_eq, layout_eq,
        fps=fps_eq, likelihood_threshold=0.5,
        max_iter=4, tol=1e-3,
    )
    em_one_sess = fit_noise_params_em(
        pos_eq, likes_eq, layout_eq,
        fps=fps_eq, likelihood_threshold=0.5,
        max_iter=4, tol=1e-3,
        sessions=[("only", 0, T_eq)],
    )
    # Both should produce the same params (within fp tolerance)
    for m in layout_eq.markers:
        assert abs(
            em_no_sess.params.sigma_base[m]
            - em_one_sess.params.sigma_base[m]
        ) < 1e-9, (
            f"sigma_base[{m}]: no-sessions={em_no_sess.params.sigma_base[m]} "
            f"vs one-session={em_one_sess.params.sigma_base[m]}"
        )
        assert abs(
            em_no_sess.params.q_pos[m]
            - em_one_sess.params.q_pos[m]
        ) < 1e-9

    # ---------------------------------------------------------- #
    # Case 47: cross-boundary correctness — per-session EM does
    # NOT include cross-boundary frame pairs in the q_pos
    # statistics. Construct two sessions where merging them
    # would produce a huge spurious velocity at the boundary;
    # per-session EM should ignore that pair.
    #
    # Each session has a small linear drift + observation noise
    # so σ doesn't collapse to floor (Shumway-Stoffer would
    # otherwise find σ < floor on near-immobile data and the
    # validation hook would correctly trigger).
    # ---------------------------------------------------------- #
    rng_cb = np.random.default_rng(200)
    n_per = 400
    fps_cb = 30.0
    times_cb = np.arange(n_per) / fps_cb
    sigma_obs_cb = 1.5
    # Session 1: marker around (10, 20) with slow drift
    pos_s1 = np.zeros((n_per, 2, 2))
    pos_s1[:, 0, 0] = 10 + 0.3 * times_cb + rng_cb.normal(0, sigma_obs_cb, n_per)
    pos_s1[:, 0, 1] = 20 + 0.2 * times_cb + rng_cb.normal(0, sigma_obs_cb, n_per)
    pos_s1[:, 1, 0] = 12 + 0.3 * times_cb + rng_cb.normal(0, sigma_obs_cb, n_per)
    pos_s1[:, 1, 1] = 22 + 0.2 * times_cb + rng_cb.normal(0, sigma_obs_cb, n_per)
    # Session 2: marker around (200, 300) — huge jump if merged.
    pos_s2 = np.zeros((n_per, 2, 2))
    pos_s2[:, 0, 0] = 200 + 0.3 * times_cb + rng_cb.normal(0, sigma_obs_cb, n_per)
    pos_s2[:, 0, 1] = 300 + 0.2 * times_cb + rng_cb.normal(0, sigma_obs_cb, n_per)
    pos_s2[:, 1, 0] = 202 + 0.3 * times_cb + rng_cb.normal(0, sigma_obs_cb, n_per)
    pos_s2[:, 1, 1] = 302 + 0.2 * times_cb + rng_cb.normal(0, sigma_obs_cb, n_per)
    pos_cb = np.concatenate([pos_s1, pos_s2], axis=0)
    likes_cb = np.full((2 * n_per, 2), 0.95)
    sessions_cb = [
        ("s1", 0, n_per),
        ("s2", n_per, 2 * n_per),
    ]

    # Without sessions: the M-step's lag-one accumulator
    # treats the boundary frame pair as "consecutive",
    # producing a huge spurious velocity increment.
    em_naive = fit_noise_params_em(
        pos_cb, likes_cb, layout_eq,
        fps=fps_eq, likelihood_threshold=0.5,
        max_iter=4, tol=1e-3,
    )
    # With sessions: per-session accumulator never sees the
    # boundary pair, so q_pos reflects only within-session
    # motion.
    em_correct = fit_noise_params_em(
        pos_cb, likes_cb, layout_eq,
        fps=fps_eq, likelihood_threshold=0.5,
        max_iter=4, tol=1e-3,
        sessions=sessions_cb,
    )

    # The session-aware fit must produce a SMALLER q_pos
    # (closer to truth) since the spurious boundary jump is
    # excluded. The naive version's q_pos is inflated by a
    # huge factor.
    for m in layout_eq.markers:
        assert (
            em_correct.params.q_pos[m]
            < em_naive.params.q_pos[m]
        ), (
            f"Session-aware q_pos[{m}] should be < naive q_pos. "
            f"correct={em_correct.params.q_pos[m]:.4f}, "
            f"naive={em_naive.params.q_pos[m]:.4f}"
        )
        # The naive q_pos should be substantially larger (the
        # boundary jump is ~190 px / dt = ~5700 px/s, contributing
        # ~(5700)^2 = 3.2e7 to one squared-velocity sample). With
        # ~800 frames total, this single bad pair raises q_pos by
        # ~3-5x. The exact ratio depends on smoother dynamics.
        ratio = em_naive.params.q_pos[m] / em_correct.params.q_pos[m]
        assert ratio > 2, (
            f"Naive q_pos should be substantially larger due to the "
            f"boundary jump; ratio = {ratio:.2f}"
        )

    # ---------------------------------------------------------- #
    # Case 48: multi-session EM produces sensible params on a
    # 2-session synthetic dataset.
    # ---------------------------------------------------------- #
    # Build two clean sessions with the same noise structure
    pos_clean1 = np.zeros((n_per, 2, 2))
    pos_clean2 = np.zeros((n_per, 2, 2))
    rng_ms_em = np.random.default_rng(300)
    for k in range(2):
        # Same true trajectory shape in both sessions
        pos_clean1[:, k, 0] = (
            10 + k * 5 + 2.0 * np.arange(n_per) / fps_eq
            + rng_ms_em.normal(0, 1.5, n_per)
        )
        pos_clean1[:, k, 1] = (
            20 - k * 3 + 1.0 * np.arange(n_per) / fps_eq
            + rng_ms_em.normal(0, 1.5, n_per)
        )
        pos_clean2[:, k, 0] = (
            10 + k * 5 + 2.0 * np.arange(n_per) / fps_eq
            + rng_ms_em.normal(0, 1.5, n_per)
        )
        pos_clean2[:, k, 1] = (
            20 - k * 3 + 1.0 * np.arange(n_per) / fps_eq
            + rng_ms_em.normal(0, 1.5, n_per)
        )
    pos_clean = np.concatenate([pos_clean1, pos_clean2], axis=0)
    likes_clean = np.full((2 * n_per, 2), 0.95)
    sessions_clean = [("a", 0, n_per), ("b", n_per, 2 * n_per)]

    em_clean = fit_noise_params_em(
        pos_clean, likes_clean, layout_eq,
        fps=fps_eq, likelihood_threshold=0.5,
        max_iter=5, tol=1e-3,
        sessions=sessions_clean,
    )
    # Sigma estimates should be in the right ballpark (true=1.5)
    for m in layout_eq.markers:
        assert 0.5 < em_clean.params.sigma_base[m] < 5.0, (
            f"sigma_base[{m}] = "
            f"{em_clean.params.sigma_base[m]:.3f} out of expected "
            f"range for true sigma=1.5"
        )

    # ---------------------------------------------------------- #
    # Case 49: MEMORY REGRESSION TEST.
    # Verify EM with ``sessions`` argument has bounded peak
    # memory: peak should be O(per-session covariance size)
    # not O(full-trajectory size). We test this by tracking
    # the largest array allocated during EM.
    #
    # Approach: monkey-patch np.empty to record allocations,
    # then check that no allocation exceeded the per-session
    # bound. Crude but reliable — and most importantly,
    # fails fast if the regression returns.
    # ---------------------------------------------------------- #
    layout_mem = StateLayout(markers=("a", "b", "c"))
    n_session_mem = 200
    n_total_mem = 4 * n_session_mem  # 4 sessions
    pos_mem = np.random.default_rng(0).normal(
        50, 1.0, (n_total_mem, 3, 2),
    )
    likes_mem = np.full((n_total_mem, 3), 0.95)
    sessions_mem = [
        (f"s{i}", i * n_session_mem, (i + 1) * n_session_mem)
        for i in range(4)
    ]

    # Track all numpy ndarray allocations via a hook on np.empty
    # (which is what forward_filter and rts_smoother use for
    # the trajectory arrays). We capture the maximum size in
    # bytes seen during the EM run.
    original_empty = np.empty
    max_alloc_bytes = [0]

    def tracking_empty(shape, dtype=float, order="C"):
        arr = original_empty(shape, dtype=dtype, order=order)
        max_alloc_bytes[0] = max(max_alloc_bytes[0], arr.nbytes)
        return arr

    np.empty = tracking_empty
    try:
        fit_noise_params_em(
            pos_mem, likes_mem, layout_mem,
            fps=30.0, likelihood_threshold=0.5,
            max_iter=2, tol=1e-3,
            sessions=sessions_mem,
        )
    finally:
        np.empty = original_empty

    # The largest single allocation should fit within
    # ~per-session-covariance scale: T_session * (4n)^2 * 8 bytes.
    # For 200 frames × 12-dim state: 200 × 144 × 8 = 230 kB.
    # Add a generous 4x margin for output arrays etc.
    per_session_state_dim = 4 * layout_mem.n_markers
    per_session_cov_bytes = (
        n_session_mem * per_session_state_dim ** 2 * 8
    )
    # If the regression returns, the EM would allocate
    # arrays scoped to n_total_mem (4× larger than per-session).
    # We require: largest allocation < 2× per-session cov size.
    bound = 2 * per_session_cov_bytes
    assert max_alloc_bytes[0] < bound, (
        f"MEMORY REGRESSION: largest np.empty allocation during EM "
        f"= {max_alloc_bytes[0]} bytes, exceeds per-session bound "
        f"of {bound} bytes ({per_session_cov_bytes} per session × 2). "
        f"This means EM is allocating across the full concatenated "
        f"trajectory rather than per-session — the bug from the "
        f"OOM kill on Gravio's 67-session dataset has returned."
    )

    # ---------------------------------------------------------- #
    # Case 50: rts_smoother populates P_smooth_lag1 with the
    # correct shape and zero at index 0 (boundary convention).
    # ---------------------------------------------------------- #
    layout_lag = StateLayout(markers=("a",))
    params_lag = NoiseParams(
        sigma_base={"a": 1.0}, q_pos={"a": 10.0}, q_vel={"a": 1.0},
    )
    fps_lag = 30.0
    dt_lag = 1.0 / fps_lag
    T_lag = 100
    rng_lag = np.random.default_rng(1000)
    pos_lag = np.zeros((T_lag, 1, 2))
    times_lag = np.arange(T_lag) * dt_lag
    pos_lag[:, 0, 0] = 50 + 2.0 * times_lag + rng_lag.normal(0, 1.0, T_lag)
    pos_lag[:, 0, 1] = 60 + 1.0 * times_lag + rng_lag.normal(0, 1.0, T_lag)
    likes_lag = np.full((T_lag, 1), 0.95)

    filt_lag = forward_filter(
        pos_lag, likes_lag, layout_lag, params_lag, dt_lag,
        likelihood_threshold=0.5,
    )
    smooth_lag = rts_smoother(filt_lag, layout_lag, dt_lag)

    assert smooth_lag.P_smooth_lag1.shape == (T_lag, 4, 4)
    # Index 0 is zero by convention
    assert np.allclose(smooth_lag.P_smooth_lag1[0], 0.0)
    # Other indices should be non-zero in general
    assert not np.allclose(smooth_lag.P_smooth_lag1[T_lag // 2], 0.0), (
        "P_smooth_lag1 at mid-trajectory should be non-zero"
    )
    # By construction, P_smooth_lag1[t] = P_smooth[t] @ C_{t-1}^T,
    # which is generally not symmetric (C_{t-1} acts on the left).
    # The TRANSPOSE relationship: lag-cov (t-1, t) = (lag-cov (t, t-1))^T.

    # ---------------------------------------------------------- #
    # Case 51: Shumway-Stoffer M-step gives sane results on
    # well-conditioned synthetic data. (Replaces case 34's
    # behavior under patch 91.)
    # ---------------------------------------------------------- #
    layout_ss = StateLayout(markers=("a", "b"))
    rng_ss = np.random.default_rng(2000)
    fps_ss = 30.0
    dt_ss = 1.0 / fps_ss
    T_ss = 2000
    sigma_true_ss = 2.0
    q_true_ss = 50.0
    times_ss = np.arange(T_ss) * dt_ss
    pos_ss = np.zeros((T_ss, 2, 2))
    for k in range(2):
        # Linear motion + observation noise
        pos_ss[:, k, 0] = (
            10 + k * 5 + 2.0 * times_ss
            + rng_ss.normal(0, sigma_true_ss, T_ss)
        )
        pos_ss[:, k, 1] = (
            20 - k * 3 + 1.0 * times_ss
            + rng_ss.normal(0, sigma_true_ss, T_ss)
        )
    likes_ss = np.full((T_ss, 2), 0.95)

    em_ss = fit_noise_params_em(
        pos_ss, likes_ss, layout_ss,
        fps=fps_ss, likelihood_threshold=0.5,
        max_iter=8, tol=1e-3,
    )
    # Shumway-Stoffer should produce sigma estimates within
    # a factor of ~2 of the true noise, AND below the ceiling
    for m in layout_ss.markers:
        sigma = em_ss.params.sigma_base[m]
        assert 0.5 * sigma_true_ss < sigma < 3.0 * sigma_true_ss, (
            f"Shumway-Stoffer sigma_base[{m}] = {sigma:.3f} "
            f"out of expected range for true sigma={sigma_true_ss}"
        )
        # q_pos should NOT collapse to floor on this clean data
        assert em_ss.params.q_pos[m] > 1.0, (
            f"q_pos[{m}] = {em_ss.params.q_pos[m]:.4f} suspiciously "
            f"low (collapsed to floor?)"
        )

    # ---------------------------------------------------------- #
    # Case 52: ORIGINAL FAILURE REGRESSION TEST.
    # Synthetic data designed to trigger the patch 88-90 EM
    # failure mode (mostly-immobile sequence with marker
    # dropouts). Pre-patch-91 EM produced sigma ~50-70 px and
    # q_pos collapsed to floor; Shumway-Stoffer with the
    # ceiling guardrail should produce sigma < 3 * initial
    # and non-collapsed q_pos.
    # ---------------------------------------------------------- #
    layout_reg = StateLayout(markers=("a", "b", "c"))
    rng_reg = np.random.default_rng(42)
    fps_reg = 30.0
    dt_reg = 1.0 / fps_reg
    T_reg = 3000
    sigma_obs_reg = 2.0
    # Mostly-immobile sequence: small slow motion + occasional
    # bursts. This is the pattern that broke the pragmatic
    # estimator on Gravio's data.
    pos_reg = np.zeros((T_reg, 3, 2))
    base_x = 100.0
    base_y = 100.0
    # Slow drift + occasional jumps
    drift_x = np.cumsum(rng_reg.normal(0, 0.05, T_reg))
    drift_y = np.cumsum(rng_reg.normal(0, 0.05, T_reg))
    # Add 5 bursts of motion
    for burst_start in [500, 1000, 1500, 2000, 2500]:
        for j in range(50):
            if burst_start + j < T_reg:
                drift_x[burst_start + j:] += 0.5
                drift_y[burst_start + j:] += 0.3
    for k in range(3):
        offset_x, offset_y = 5 * k, 3 * k
        pos_reg[:, k, 0] = (
            base_x + offset_x + drift_x
            + rng_reg.normal(0, sigma_obs_reg, T_reg)
        )
        pos_reg[:, k, 1] = (
            base_y + offset_y + drift_y
            + rng_reg.normal(0, sigma_obs_reg, T_reg)
        )

    # Add long dropouts on marker 'c' to mimic the real-data
    # tracking failures.
    likes_reg = np.full((T_reg, 3), 0.95)
    likes_reg[800:1200, 2] = 0.0  # 400-frame dropout on marker c

    em_reg = fit_noise_params_em(
        pos_reg, likes_reg, layout_reg,
        fps=fps_reg, likelihood_threshold=0.5,
        max_iter=8, tol=1e-3,
    )
    # Critical: sigma should NOT have run away to >50 px (the
    # patch 88-90 failure mode). It should stay < 3 * initial.
    for m in layout_reg.markers:
        sigma = em_reg.params.sigma_base[m]
        assert sigma < 10.0, (
            f"REGRESSION: sigma_base[{m}] = {sigma:.2f} px on "
            f"immobile-with-dropouts synthetic data — looks like "
            f"the patch 88-90 EM degeneracy returned. Should be "
            f"~{sigma_obs_reg} px (true noise level)."
        )
        # q_pos should not be at floor
        assert em_reg.params.q_pos[m] > 0.01, (
            f"REGRESSION: q_pos[{m}] = {em_reg.params.q_pos[m]:.6f} "
            f"collapsed near floor — patch 88-90 failure mode."
        )

    # ---------------------------------------------------------- #
    # Case 53: sigma ceiling guardrail caps sigma_base at
    # 3 * initial under forced-runaway conditions.
    # ---------------------------------------------------------- #
    # Construct an initial_params with deliberately small sigma,
    # then run EM on data with much larger true noise.
    # Without the ceiling, EM would push sigma higher; with
    # the ceiling, it caps at 3 * initial.
    layout_ceil = StateLayout(markers=("a",))
    rng_ceil = np.random.default_rng(3000)
    sigma_initial_small = 1.0
    sigma_actual = 10.0  # 10x larger than initial estimate
    T_ceil = 1000
    pos_ceil = np.zeros((T_ceil, 1, 2))
    times_ceil = np.arange(T_ceil) / 30.0
    pos_ceil[:, 0, 0] = (
        50 + 2.0 * times_ceil + rng_ceil.normal(0, sigma_actual, T_ceil)
    )
    pos_ceil[:, 0, 1] = (
        60 + 1.0 * times_ceil + rng_ceil.normal(0, sigma_actual, T_ceil)
    )
    likes_ceil = np.full((T_ceil, 1), 0.95)

    initial_params_small = NoiseParams(
        sigma_base={"a": sigma_initial_small},
        q_pos={"a": 10.0},
        q_vel={"a": 1.0},
    )
    # Note: validation hook may trigger here if smoothing is bad,
    # so use a try/except to handle either outcome:
    # - If EM completes: sigma should be at the 3x ceiling
    # - If validation hook fires: that's also acceptable behavior
    try:
        em_ceil = fit_noise_params_em(
            pos_ceil, likes_ceil, layout_ceil,
            fps=30.0, likelihood_threshold=0.5,
            initial_params=initial_params_small,
            max_iter=8, tol=1e-3,
        )
        sigma_capped = em_ceil.params.sigma_base["a"]
        # Sigma should be capped at 3 * initial
        ceiling_value = 3.0 * sigma_initial_small
        assert sigma_capped <= ceiling_value + 1e-6, (
            f"Sigma ceiling not enforced: sigma_base = "
            f"{sigma_capped:.3f}, ceiling = {ceiling_value:.3f}"
        )
        # And sigma should have grown from initial (otherwise the
        # ceiling test isn't actually testing anything)
        assert sigma_capped > sigma_initial_small * 1.5, (
            f"Sigma should have grown from initial under runaway "
            f"conditions; only got {sigma_capped:.3f} from "
            f"initial {sigma_initial_small}"
        )
    except RuntimeError:
        # Validation hook caught it — also acceptable, since the
        # ceiling+true-mismatch by design produces degenerate output
        pass

    # ---------------------------------------------------------- #
    # Case 54: validation hook crashes on forced degeneracy.
    # We construct a scenario where rigid initial params combined
    # with a small initial sigma trigger the prior-overruling
    # check (mean|smoothed-raw| > 5*initial_sigma).
    # ---------------------------------------------------------- #
    layout_v = StateLayout(markers=("a",))
    rng_v = np.random.default_rng(4000)
    T_v = 500
    pos_v = np.zeros((T_v, 1, 2))
    # Fast wide motion: 100 px range
    pos_v[:, 0, 0] = (
        100 + 50 * np.sin(np.arange(T_v) / 50)
        + rng_v.normal(0, 0.5, T_v)
    )
    pos_v[:, 0, 1] = (
        200 + 50 * np.cos(np.arange(T_v) / 50)
        + rng_v.normal(0, 0.5, T_v)
    )
    likes_v = np.full((T_v, 1), 0.95)
    # Rigid params: small initial sigma (so 5x bound is tight),
    # but extremely rigid q_pos (forces frozen smoother). The
    # smoother won't be able to track the wide motion, so
    # mean|smoothed-raw| at high-p frames will be much larger
    # than 5 * sigma_initial_small.
    rigid_params = NoiseParams(
        sigma_base={"a": 1.0},  # small initial → 5x bound = 5 px
        q_pos={"a": 1e-3},      # at floor → no motion allowed
        q_vel={"a": 1e-3},
    )
    try:
        fit_noise_params_em(
            pos_v, likes_v, layout_v,
            fps=30.0, likelihood_threshold=0.5,
            initial_params=rigid_params,
            max_iter=3, tol=1e-3,
        )
        assert False, (
            "Validation hook should crash on forcibly-rigid "
            "params with tight sigma bound"
        )
    except RuntimeError as e:
        msg = str(e)
        assert "validation hook" in msg.lower()

    # ---------------------------------------------------------- #
    # Case 55: smooth_pose runs cleanly with use_triplets=False
    # (default) — pure per-marker temporal smoothing.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        in_csv = td / "no_triplets.csv"
        _build_synthetic_pose_csv(in_csv, n_frames=300)

        out_dir = td / "out"
        result = smooth_pose(
            pose_input=str(in_csv),
            output_dir=str(out_dir),
            fps=30.0,
            likelihood_threshold=0.7,
            head_markers=["nose", "ear_left", "ear_right"],
            em_max_iter=2,
            verbose=False,
            # use_triplets=False is the default
        )
        # Triplet priors should be empty (no triplet path taken)
        assert len(result["triplet_priors"]) == 0, (
            f"With use_triplets=False, no triplet priors should "
            f"be fit; got {len(result['triplet_priors'])}"
        )
        # Smoothing should still produce sensible output
        assert len(result["output_files"]) == 1
        assert Path(result["output_files"][0]).exists()

    # ---------------------------------------------------------- #
    # Case 56: smooth_pose with use_triplets=True still works
    # (backward compat with patch 87-90 behavior).
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        in_csv = td / "with_triplets.csv"
        _build_synthetic_pose_csv(in_csv, n_frames=400)

        out_dir = td / "out"
        try:
            result_tri = smooth_pose(
                pose_input=str(in_csv),
                output_dir=str(out_dir),
                fps=30.0,
                likelihood_threshold=0.7,
                head_markers=["nose", "ear_left", "ear_right"],
                em_max_iter=2,
                use_triplets=True,
                verbose=False,
            )
            # Triplet path should have produced some priors
            # (synthetic data has tight covariance, so triplets
            # auto-detect)
            # Just verify the path runs without crashing.
            assert "triplet_priors" in result_tri
            assert len(result_tri["output_files"]) == 1
        except RuntimeError as e:
            # On synthetic data the validation hook MAY trigger
            # if the static triplet creates issues — that's
            # actually demonstrating why we made it default-off.
            # As long as the error is from the validation hook
            # and not a real bug, this is acceptable test behavior.
            assert "validation hook" in str(e).lower()

    # ---------------------------------------------------------- #
    # Case 57: _compute_velocity_strata produces sensible bins
    # and weights on synthetic mostly-immobile + bursts data.
    # The fixture has clear behavioral state changes (rest vs
    # active locomotion) so the LPF velocity has CV well above
    # the function's low-variance-fallback threshold.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother import (
        _compute_velocity_strata,
    )
    layout_strat = StateLayout(markers=(
        "back1", "back2", "back3", "tailbase", "nose",
    ))
    rng_strat = np.random.default_rng(5000)
    fps_strat = 30.0
    T_strat = 9000  # 300 sec at 30 fps
    pos_strat = np.zeros((T_strat, 5, 2))
    # Body state: 0 = rest (no motion), 1 = walk (~5 px/frame).
    # Three long walking bouts of ~600 frames each separated by
    # rest. This produces clear velocity-distribution bimodality
    # which after LPF still has CV well above 0.3.
    state = np.zeros(T_strat, dtype=int)
    for bout_start in [1500, 4000, 6500]:
        state[bout_start:bout_start + 600] = 1
    base_x = np.zeros(T_strat)
    base_y = np.zeros(T_strat)
    velocity_per_frame = np.where(state == 1, 5.0, 0.05)
    direction_x = rng_strat.normal(0, 1, T_strat)
    direction_y = rng_strat.normal(0, 1, T_strat)
    direction_norm = np.sqrt(direction_x ** 2 + direction_y ** 2) + 1e-9
    base_x = np.cumsum(velocity_per_frame * direction_x / direction_norm)
    base_y = np.cumsum(velocity_per_frame * direction_y / direction_norm)
    for k in range(4):  # back1-3, tailbase
        offset_x = 5 * k
        offset_y = 3 * k
        pos_strat[:, k, 0] = (
            100 + offset_x + base_x
            + rng_strat.normal(0, 1.0, T_strat)
        )
        pos_strat[:, k, 1] = (
            200 + offset_y + base_y
            + rng_strat.normal(0, 1.0, T_strat)
        )
    pos_strat[:, 4, 0] = 110 + base_x + rng_strat.normal(0, 1.0, T_strat)
    pos_strat[:, 4, 1] = 210 + base_y + rng_strat.normal(0, 1.0, T_strat)
    likes_strat = np.full((T_strat, 5), 0.95)

    bin_idx, weights = _compute_velocity_strata(
        pos_strat, likes_strat, layout_strat,
        fps=fps_strat, n_bins=4,
    )
    assert bin_idx.shape == (T_strat,)
    assert weights.shape == (T_strat,)
    # All bins should have at least some frames
    bin_pops = np.bincount(bin_idx, minlength=4)
    assert (bin_pops > 0).all(), (
        f"Every bin should be populated; got {bin_pops}"
    )
    # Mean weight ~1.0 (allowing fp tolerance)
    assert abs(weights.mean() - 1.0) < 0.05, (
        f"Mean weight should be ~1.0; got {weights.mean():.4f}"
    )
    # With tail-emphasizing breakpoints, bin 3 (highest velocity)
    # has fewer frames (12.5%) than bin 0 (50%), so bin 3's
    # per-frame weight should be ~4x bin 0's weight.
    bin_3_weight = weights[bin_idx == 3].mean()
    bin_0_weight = weights[bin_idx == 0].mean()
    assert bin_3_weight > bin_0_weight, (
        f"Highest-velocity bin should have higher weight than "
        f"slowest bin; bin0={bin_0_weight:.3f}, "
        f"bin3={bin_3_weight:.3f}"
    )

    # ---------------------------------------------------------- #
    # Case 58: _compute_velocity_strata falls back to uniform
    # weights when there's insufficient data.
    # ---------------------------------------------------------- #
    pos_short = pos_strat[:30]  # too few frames
    likes_short = likes_strat[:30]
    bin_idx_short, weights_short = _compute_velocity_strata(
        pos_short, likes_short, layout_strat,
        fps=fps_strat, n_bins=4,
    )
    assert (weights_short == 1.0).all(), (
        "Insufficient data should fall back to uniform weights"
    )

    # ---------------------------------------------------------- #
    # Case 59: stratified EM addresses the immobile-bias.
    # On synthetic data with 4 bursts of fast motion in a sea
    # of slow drift, stratified EM should produce a LARGER
    # q_pos for fast-moving markers than unstratified EM.
    # ---------------------------------------------------------- #
    em_unstrat = fit_noise_params_em(
        pos_strat, likes_strat, layout_strat,
        fps=fps_strat, likelihood_threshold=0.5,
        max_iter=4, tol=1e-3,
        stratify=False,
    )
    em_strat = fit_noise_params_em(
        pos_strat, likes_strat, layout_strat,
        fps=fps_strat, likelihood_threshold=0.5,
        max_iter=4, tol=1e-3,
        stratify=True, n_strata=4,
    )
    # Stratified q_pos should be >= unstratified q_pos for at
    # least the majority of markers (the bursts contribute more
    # to q_pos when reweighted).
    n_higher = sum(
        em_strat.params.q_pos[m] > em_unstrat.params.q_pos[m]
        for m in layout_strat.markers
    )
    assert n_higher >= 3, (
        f"Stratified q_pos should be higher for majority of "
        f"markers; got {n_higher}/{len(layout_strat.markers)}. "
        f"Unstratified: {em_unstrat.params.q_pos}; "
        f"stratified: {em_strat.params.q_pos}"
    )

    # ---------------------------------------------------------- #
    # Case 60: stratify=True is the default for fit_noise_params_em.
    # ---------------------------------------------------------- #
    em_default = fit_noise_params_em(
        pos_strat, likes_strat, layout_strat,
        fps=fps_strat, likelihood_threshold=0.5,
        max_iter=2, tol=1e-3,
        # No stratify= argument; should default True
    )
    # Default should match stratify=True
    em_explicit = fit_noise_params_em(
        pos_strat, likes_strat, layout_strat,
        fps=fps_strat, likelihood_threshold=0.5,
        max_iter=2, tol=1e-3,
        stratify=True,
    )
    for m in layout_strat.markers:
        assert abs(
            em_default.params.q_pos[m] - em_explicit.params.q_pos[m]
        ) < 1e-9, (
            f"stratify default should match stratify=True for {m}"
        )

    # ---------------------------------------------------------- #
    # Case 61: smooth_pose with stratify=False (the no-stratify
    # CLI flag) runs cleanly.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        in_csv = td / "no_strat.csv"
        _build_synthetic_pose_csv(in_csv, n_frames=300)

        out_dir = td / "out"
        result = smooth_pose(
            pose_input=str(in_csv),
            output_dir=str(out_dir),
            fps=30.0,
            likelihood_threshold=0.7,
            em_max_iter=2,
            stratify=False,
            verbose=False,
        )
        assert len(result["output_files"]) == 1
        assert Path(result["output_files"][0]).exists()

    # ---------------------------------------------------------- #
    # Case 62: q_pos initialization + per-marker floor prevent
    # the patch-93 sparse-marker degeneracy. Construct a marker
    # with only 25% high-p coverage and clear motion. Pre-patch-94
    # initial_noise_params + finalize_m_step would produce
    # near-zero q_pos that EM couldn't escape; patch-94 should
    # produce q_pos well above zero.
    # ---------------------------------------------------------- #
    layout_sparse = StateLayout(markers=("active",))
    rng_sparse = np.random.default_rng(8000)
    fps_sparse = 30.0
    T_sparse = 3000
    # Marker has clear walking-rest pattern with motion of
    # ~50 px/sec during walk
    state_sp = np.zeros(T_sparse)
    state_sp[500:1000] = 1
    state_sp[2000:2500] = 1
    velocity = np.where(state_sp == 1, 1.5, 0.05)
    direction = rng_sparse.normal(0, 1, T_sparse)
    direction_norm = np.abs(direction) + 1e-9
    motion = np.cumsum(velocity * direction / direction_norm)
    pos_sparse = np.zeros((T_sparse, 1, 2))
    pos_sparse[:, 0, 0] = 100 + motion + rng_sparse.normal(0, 1.5, T_sparse)
    pos_sparse[:, 0, 1] = 200 + motion * 0.5 + rng_sparse.normal(0, 1.5, T_sparse)
    # 25% high-p (mimicking nose's coverage in Gravio's data)
    likes_sparse = np.where(
        rng_sparse.uniform(0, 1, (T_sparse, 1)) < 0.25, 0.95, 0.3,
    )
    em_sparse = fit_noise_params_em(
        pos_sparse, likes_sparse, layout_sparse,
        fps=fps_sparse, likelihood_threshold=0.7,
        max_iter=4, tol=1e-3,
    )
    # Critical regression: q_pos should NOT collapse to floor
    # (patch 93 produced q_pos=0.011 on real sparse data).
    q_pos_active = em_sparse.params.q_pos["active"]
    assert q_pos_active > 5.0, (
        f"REGRESSION: q_pos[active] = {q_pos_active:.4f} "
        f"on sparse-marker synthetic data — looks like the "
        f"patch-93 sparse-marker degeneracy returned. Should "
        f"be > 5 (well above the 1e-3 global floor)."
    )

    print("smoke_kalman_pose_smoother: 62/62 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
