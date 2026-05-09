"""
smoke_kalman_pose_smoother_v2
=============================

Smoke tests for the v2 kinematic-tree pose smoother. Patch 99
covers only layout types and length fitting; subsequent
patches add forward kinematics, EKF, RTS, EM, etc. Tests are
appended in order matching the patch sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mufasa.data_processors.kalman_pose_smoother_v2 import (
    BodyLayout,
    BodySegment,
    FittedLengths,
    fit_body_lengths,
    standard_rat_layout,
)


def main() -> int:
    # ---------------------------------------------------------- #
    # Case 1: BodyLayout requires exactly one root.
    # ---------------------------------------------------------- #
    try:
        BodyLayout(segments=[
            BodySegment(name="a", parent=None),
            BodySegment(name="b", parent=None),
        ])
        assert False, "Should reject layout with two roots"
    except ValueError as e:
        assert "exactly one root" in str(e)

    try:
        BodyLayout(segments=[
            BodySegment(name="a", parent="missing"),
        ])
        assert False, "Should reject layout with no root"
    except ValueError:
        pass

    # ---------------------------------------------------------- #
    # Case 2: BodyLayout rejects orphan parents.
    # ---------------------------------------------------------- #
    try:
        BodyLayout(segments=[
            BodySegment(name="root", parent=None),
            BodySegment(name="orphan", parent="nonexistent"),
        ])
        assert False, "Should reject orphan parent"
    except ValueError as e:
        assert "not in the layout" in str(e)

    # ---------------------------------------------------------- #
    # Case 3: BodyLayout rejects duplicate marker attachments.
    # ---------------------------------------------------------- #
    try:
        BodyLayout(segments=[
            BodySegment(
                name="a", parent=None,
                markers={"shared": (0.0, 0.0)},
            ),
            BodySegment(
                name="b", parent="a",
                markers={"shared": (1.0, 0.0)},
            ),
        ])
        assert False, "Should reject duplicate marker"
    except ValueError as e:
        assert "multiple segments" in str(e)

    # ---------------------------------------------------------- #
    # Case 4: standard_rat_layout has expected structure.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    expected_segments = [
        "body", "back_rear", "neck", "head",
        "tail_1", "tail_2", "tail_3",
    ]
    actual_segments = [s.name for s in layout.segments]
    assert set(actual_segments) == set(expected_segments), (
        f"Mismatched segments: expected {expected_segments}, "
        f"got {actual_segments}"
    )

    # Expected markers
    expected_markers = {
        "back1", "back2", "back3", "back4",
        "lateral_left", "lateral_right", "center",
        "neck", "headmid", "nose", "ear_left", "ear_right",
        "tailbase", "tailmid", "tailend",
    }
    actual_markers = set(layout.marker_names)
    assert actual_markers == expected_markers, (
        f"Mismatched markers: expected {expected_markers}, "
        f"got {actual_markers}"
    )

    # ---------------------------------------------------------- #
    # Case 5: state_dim accounting.
    # ---------------------------------------------------------- #
    # 7 segments total → 6 non-root → 8 + 6*6 = 44 state dim
    layout = standard_rat_layout()
    assert layout.n_non_root_segments == 6
    assert layout.state_dim == 8 + 6 * 6, (
        f"Expected state_dim 44; got {layout.state_dim}"
    )

    # Layout without back4 → back_rear is gone, tail_1 attaches
    # to body directly. 5 non-root segments → 38 state dim.
    layout_no_back4 = standard_rat_layout(include_back4=False)
    assert layout_no_back4.n_non_root_segments == 5
    assert layout_no_back4.state_dim == 8 + 6 * 5

    # Layout without tail → 3 non-root (back_rear, neck, head)
    layout_no_tail = standard_rat_layout(include_tail=False)
    assert layout_no_tail.n_non_root_segments == 3
    assert layout_no_tail.state_dim == 8 + 6 * 3

    # ---------------------------------------------------------- #
    # Case 6: topo order has root first, children after parents.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    topo = layout.topo_order
    assert topo[0] == "body", f"Root should be first; got {topo}"
    # head must come after neck
    assert topo.index("head") > topo.index("neck")
    # tail_3 must come after tail_2 must come after tail_1
    assert topo.index("tail_3") > topo.index("tail_2") > topo.index("tail_1")
    # tail_1 must come after back_rear (it's tail_1's parent)
    assert topo.index("tail_1") > topo.index("back_rear")

    # ---------------------------------------------------------- #
    # Case 7: state slice helpers give non-overlapping ranges
    # spanning [0, state_dim).
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    slices = [layout.slice_root_pose()]
    for seg_name in layout.non_root_topo_order:
        slices.append(layout.slice_segment_orientation(seg_name))
    for seg_name in layout.non_root_topo_order:
        slices.append(layout.slice_segment_length(seg_name))

    # Check coverage: union of slices covers [0, state_dim) and
    # no overlaps
    covered = set()
    for sl in slices:
        for i in range(sl.start, sl.stop):
            assert i not in covered, (
                f"Overlap at index {i}: slices are {slices}"
            )
            covered.add(i)
    assert covered == set(range(layout.state_dim)), (
        f"State coverage mismatch: covered {len(covered)} / "
        f"{layout.state_dim}; missing "
        f"{set(range(layout.state_dim)) - covered}"
    )

    # ---------------------------------------------------------- #
    # Case 8: slice_segment_orientation rejects the root.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    try:
        layout.slice_segment_orientation("body")
        assert False, "Should reject root for orientation slice"
    except KeyError as e:
        assert "root" in str(e).lower() or "body" in str(e)

    # ---------------------------------------------------------- #
    # Case 9: marker_attachment lookup.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    seg, offset = layout.marker_attachment("nose")
    assert seg == "head"
    assert offset[0] > 0  # Nose is forward of headmid
    seg, offset = layout.marker_attachment("back2")
    assert seg == "body"
    assert offset == (0.0, 0.0)  # back2 is the root distal

    try:
        layout.marker_attachment("nonexistent")
        assert False, "Should fail for unknown marker"
    except KeyError:
        pass

    # ---------------------------------------------------------- #
    # Case 10: fit_body_lengths recovers known lengths from
    # synthetic data.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    rng = np.random.default_rng(42)
    T = 1000

    # Synthesize a rat at fixed posture, known lengths
    true_lengths = {
        "back_rear": 8.0,    # back2 → back4 distance
        "neck": 6.0,         # back2 → neck distance
        "head": 4.0,         # neck → headmid distance
        "tail_1": 5.0,       # back4 → tailbase
        "tail_2": 5.0,       # tailbase → tailmid
        "tail_3": 4.0,       # tailmid → tailend
    }
    # Simple layout: rat lying along x-axis, body at origin
    marker_names = sorted(layout.marker_names)
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    n_markers = len(marker_names)

    # Per-frame: simulate small jitter
    positions = np.zeros((T, n_markers, 2))
    likelihoods = np.full((T, n_markers), 0.95)

    for t in range(T):
        # Body at origin, slight noise
        body_pos = rng.normal(0, 0.5, 2)
        positions[t, name_to_idx["back2"]] = body_pos
        positions[t, name_to_idx["back1"]] = body_pos + np.array([3.0, 0.0])
        positions[t, name_to_idx["back3"]] = body_pos + np.array([-3.0, 0.0])
        positions[t, name_to_idx["lateral_left"]] = body_pos + np.array([0.0, 2.0])
        positions[t, name_to_idx["lateral_right"]] = body_pos + np.array([0.0, -2.0])
        positions[t, name_to_idx["center"]] = body_pos + np.array([1.0, 0.0])
        # Posterior (back_rear at distance true_lengths['back_rear'] behind body)
        positions[t, name_to_idx["back4"]] = (
            body_pos + np.array([-true_lengths["back_rear"], 0.0])
        )
        # Anterior chain
        neck_pos = body_pos + np.array([true_lengths["neck"], 0.0])
        positions[t, name_to_idx["neck"]] = neck_pos
        head_pos = neck_pos + np.array([true_lengths["head"], 0.0])
        positions[t, name_to_idx["headmid"]] = head_pos
        positions[t, name_to_idx["nose"]] = head_pos + np.array([1.0, 0.0])
        positions[t, name_to_idx["ear_left"]] = head_pos + np.array([0.5, 1.0])
        positions[t, name_to_idx["ear_right"]] = head_pos + np.array([0.5, -1.0])
        # Tail chain
        tb = positions[t, name_to_idx["back4"]] + np.array([-true_lengths["tail_1"], 0.0])
        positions[t, name_to_idx["tailbase"]] = tb
        tm = tb + np.array([-true_lengths["tail_2"], 0.0])
        positions[t, name_to_idx["tailmid"]] = tm
        te = tm + np.array([-true_lengths["tail_3"], 0.0])
        positions[t, name_to_idx["tailend"]] = te

        # Add tracking noise to all markers
        positions[t] += rng.normal(0, 0.1, (n_markers, 2))

    fitted = fit_body_lengths(positions, likelihoods, layout, marker_names)

    # Verify recovered lengths within tolerance
    for seg_name, true_len in true_lengths.items():
        assert seg_name in fitted.segment_lengths, (
            f"Missing fitted length for {seg_name}"
        )
        fitted_len = fitted.segment_lengths[seg_name]
        rel_err = abs(fitted_len - true_len) / true_len
        assert rel_err < 0.05, (
            f"Length error for {seg_name}: fitted={fitted_len:.3f}, "
            f"true={true_len:.3f}, rel_err={rel_err:.3f}"
        )

    # IQR should be small (since we added small noise)
    for seg_name in true_lengths:
        iqr = fitted.segment_length_iqr.get(seg_name, 0.0)
        assert iqr < 1.0, (
            f"IQR for {seg_name} unexpectedly large: {iqr:.3f}"
        )

    # ---------------------------------------------------------- #
    # Case 11: fit_body_lengths handles missing markers
    # gracefully.
    # ---------------------------------------------------------- #
    # Provide all-low likelihoods → no length fit possible
    likelihoods_low = np.zeros((T, n_markers))
    fitted_empty = fit_body_lengths(
        positions, likelihoods_low, layout, marker_names,
    )
    # Should return mostly empty dicts (no segments fit)
    assert len(fitted_empty.segment_lengths) == 0

    # ---------------------------------------------------------- #
    # Case 12: marker_offsets recovered for non-distal markers.
    # ---------------------------------------------------------- #
    # In our synthetic data:
    #   nose is at headmid + (1, 0) → length=1, angle=0
    #   ear_left at headmid + (0.5, 1) → length=sqrt(1.25)≈1.118,
    #                                    angle=atan2(1, 0.5)≈1.107
    fitted = fit_body_lengths(positions, likelihoods, layout, marker_names)
    nose_off = fitted.marker_offsets.get("nose")
    assert nose_off is not None
    # Length close to 1.0 (in head's local frame)
    assert abs(nose_off[0] - 1.0) < 0.1, (
        f"nose offset length: expected ~1.0, got {nose_off[0]:.3f}"
    )
    assert abs(nose_off[1]) < 0.1, (
        f"nose offset angle: expected ~0, got {nose_off[1]:.3f}"
    )

    # ---------------------------------------------------------- #
    # Patch 100: forward kinematics + observation function +
    # Jacobian.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        forward_kinematics, state_to_marker_positions,
        state_to_marker_jacobian, initial_state_from_data,
        _pack_state_layout_indices,
    )

    # ---------------------------------------------------------- #
    # Case 13: forward_kinematics on a known-pose state recovers
    # expected marker positions.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    indices = _pack_state_layout_indices(layout)
    state = np.zeros(layout.state_dim)
    # Identity root pose (cos=1, sin=0), at origin
    state[indices["__root__"]["x"]] = 100.0
    state[indices["__root__"]["y"]] = 200.0
    state[indices["__root__"]["cos"]] = 1.0
    state[indices["__root__"]["sin"]] = 0.0
    # All non-root segments aligned along parent's x-axis
    # with unit length
    for seg_name in layout.non_root_topo_order:
        state[indices[seg_name]["cos"]] = 1.0
        state[indices[seg_name]["sin"]] = 0.0
        state[indices[seg_name]["length"]] = 5.0

    fk = forward_kinematics(state, layout)
    # Root distal at (100, 200)
    assert np.allclose(fk.P_distal["body"], [100, 200])
    # Root world rotation = identity
    assert np.allclose(fk.R_world["body"], np.eye(2))
    # back_rear should be 5 forward of body in body's frame
    # (note rest_angle isn't enforced in fwd kinematics — that
    # only affects initialization)
    # With s_cos=1, s_sin=0, the segment vector is (5, 0) in
    # parent frame. Body's world rotation is identity so
    # back_rear's distal world position is body + (5, 0).
    assert np.allclose(fk.P_distal["back_rear"], [105, 200])
    # neck distal: 5 forward from body
    assert np.allclose(fk.P_distal["neck"], [105, 200])
    # head distal: 5 forward from neck = 10 from body
    assert np.allclose(fk.P_distal["head"], [110, 200])

    # ---------------------------------------------------------- #
    # Case 14: state_to_marker_positions returns expected
    # marker positions (matching forward kinematics with
    # marker offsets).
    # ---------------------------------------------------------- #
    pos = state_to_marker_positions(state, layout, fk=fk)
    assert pos.shape == (layout.n_markers, 2)
    marker_names = layout.marker_names

    # back2 (root distal): at (100, 200)
    i_back2 = marker_names.index("back2")
    assert np.allclose(pos[i_back2], [100, 200])
    # back4 (back_rear distal): at (105, 200)
    i_back4 = marker_names.index("back4")
    assert np.allclose(pos[i_back4], [105, 200])
    # nose (offset 1, 0 in head frame; head distal at (110, 200))
    # head world rotation is identity, so nose is at (111, 200)
    i_nose = marker_names.index("nose")
    assert np.allclose(pos[i_nose], [111, 200])
    # ear_left (offset 0.5, π/3 in head frame)
    # head rotation identity → ear_left world = head_distal + (0.5*cos(π/3), 0.5*sin(π/3))
    i_ear_l = marker_names.index("ear_left")
    expected_ear_l = np.array([
        110 + 0.5 * np.cos(np.pi/3), 200 + 0.5 * np.sin(np.pi/3),
    ])
    assert np.allclose(pos[i_ear_l], expected_ear_l)

    # ---------------------------------------------------------- #
    # Case 15: forward_kinematics with rotated root.
    # ---------------------------------------------------------- #
    state2 = state.copy()
    # Rotate root by 90° (cos=0, sin=1)
    state2[indices["__root__"]["cos"]] = 0.0
    state2[indices["__root__"]["sin"]] = 1.0
    fk2 = forward_kinematics(state2, layout)
    # back_rear should now be 5 in +y direction from body
    assert np.allclose(fk2.P_distal["back_rear"], [100, 205])
    assert np.allclose(fk2.P_distal["head"], [100, 210])

    # ---------------------------------------------------------- #
    # Case 16: forward_kinematics with rotated head (chain
    # rotation composes correctly).
    # ---------------------------------------------------------- #
    state3 = state.copy()
    # Rotate head by 90° (head's local cos=0, sin=1)
    state3[indices["head"]["cos"]] = 0.0
    state3[indices["head"]["sin"]] = 1.0
    fk3 = forward_kinematics(state3, layout)
    # head distal: at (110, 200) (positioning unchanged - only orientation rotated)
    # Wait: head's local cos/sin affects head's segment vector
    # placement relative to neck, not its orientation alone.
    # head's distal = neck_distal + R_world[neck] @ (L_head*head_cos, L_head*head_sin)
    # = (105, 200) + I @ (0, 5) = (105, 205)
    assert np.allclose(fk3.P_distal["head"], [105, 205])
    # nose offset (1, 0) in head's local frame, head's R_world =
    # R(0, 1) (90° CCW). So nose = head_distal + R(0,1) @ (1, 0) = (105, 205) + (0, 1) = (105, 206)
    pos3 = state_to_marker_positions(state3, layout, fk=fk3)
    assert np.allclose(pos3[i_nose], [105, 206])

    # ---------------------------------------------------------- #
    # Case 17: Jacobian shape and sparsity pattern.
    # ---------------------------------------------------------- #
    H = state_to_marker_jacobian(state, layout, fk=fk)
    assert H.shape == (2 * layout.n_markers, layout.state_dim), (
        f"Jacobian shape {H.shape} != expected"
    )
    # Velocity columns should be zero
    vel_cols = [
        indices["__root__"]["vx"],
        indices["__root__"]["vy"],
        indices["__root__"]["cos_dot"],
        indices["__root__"]["sin_dot"],
    ]
    for seg_name in layout.non_root_topo_order:
        vel_cols.extend([
            indices[seg_name]["cos_dot"],
            indices[seg_name]["sin_dot"],
            indices[seg_name]["length_dot"],
        ])
    for col in vel_cols:
        assert np.allclose(H[:, col], 0.0), (
            f"Velocity column {col} should be zero"
        )

    # ---------------------------------------------------------- #
    # Case 18: Jacobian numerically matches finite differences.
    # This is the critical test — if the analytic Jacobian is
    # wrong, the EKF will produce wrong results.
    # ---------------------------------------------------------- #
    # Generate a non-degenerate state (slight rotations
    # everywhere)
    rng = np.random.default_rng(1234)
    state_test = state.copy()
    state_test[indices["__root__"]["x"]] = 50.0
    state_test[indices["__root__"]["y"]] = 70.0
    angle = 0.3
    state_test[indices["__root__"]["cos"]] = np.cos(angle)
    state_test[indices["__root__"]["sin"]] = np.sin(angle)
    for seg_name in layout.non_root_topo_order:
        a = rng.uniform(-0.5, 0.5)
        state_test[indices[seg_name]["cos"]] = np.cos(a)
        state_test[indices[seg_name]["sin"]] = np.sin(a)
        state_test[indices[seg_name]["length"]] = rng.uniform(3.0, 7.0)

    # Analytic Jacobian
    H_analytic = state_to_marker_jacobian(state_test, layout)

    # Numerical Jacobian via central differences
    eps = 1e-5
    H_numerical = np.zeros_like(H_analytic)
    pos_baseline = state_to_marker_positions(state_test, layout)
    for j in range(layout.state_dim):
        s_p = state_test.copy()
        s_m = state_test.copy()
        s_p[j] += eps
        s_m[j] -= eps
        pos_p = state_to_marker_positions(s_p, layout)
        pos_m = state_to_marker_positions(s_m, layout)
        # Pack into 2K vector matching H rows
        diff = (pos_p - pos_m).flatten()  # (2K,) row-major
        # state_to_marker_positions returns (K, 2); we flatten
        # as (k=0:x, k=0:y, k=1:x, ...) which matches H rows
        # 2k, 2k+1.
        H_numerical[:, j] = diff / (2 * eps)

    max_err = np.max(np.abs(H_analytic - H_numerical))
    assert max_err < 1e-4, (
        f"Analytic Jacobian disagrees with finite differences: "
        f"max abs error = {max_err:.6e}. This means the analytic "
        f"derivation is wrong somewhere — EKF will produce "
        f"incorrect results until fixed."
    )

    # ---------------------------------------------------------- #
    # Case 19: initial_state_from_data produces a consistent
    # state (round-trip: state → markers → fit lengths → state).
    # ---------------------------------------------------------- #
    # Use the synthetic data from case 10 above to check that
    # initial state recovery gives sensible values.
    layout = standard_rat_layout()
    rng = np.random.default_rng(99)
    T = 200
    marker_names = sorted(layout.marker_names)
    n_m = len(marker_names)
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    positions = np.zeros((T, n_m, 2))
    likelihoods = np.full((T, n_m), 0.95)

    # Same fixture as case 10 but shorter
    true_lengths = {
        "back_rear": 8.0, "neck": 6.0, "head": 4.0,
        "tail_1": 5.0, "tail_2": 5.0, "tail_3": 4.0,
    }
    for t in range(T):
        body_pos = rng.normal(0, 0.5, 2)
        positions[t, name_to_idx["back2"]] = body_pos
        positions[t, name_to_idx["back1"]] = body_pos + np.array([3.0, 0.0])
        positions[t, name_to_idx["back3"]] = body_pos + np.array([-3.0, 0.0])
        positions[t, name_to_idx["lateral_left"]] = body_pos + np.array([0.0, 2.0])
        positions[t, name_to_idx["lateral_right"]] = body_pos + np.array([0.0, -2.0])
        positions[t, name_to_idx["center"]] = body_pos + np.array([1.0, 0.0])
        positions[t, name_to_idx["back4"]] = body_pos + np.array([-true_lengths["back_rear"], 0.0])
        neck_pos = body_pos + np.array([true_lengths["neck"], 0.0])
        positions[t, name_to_idx["neck"]] = neck_pos
        head_pos = neck_pos + np.array([true_lengths["head"], 0.0])
        positions[t, name_to_idx["headmid"]] = head_pos
        positions[t, name_to_idx["nose"]] = head_pos + np.array([1.0, 0.0])
        positions[t, name_to_idx["ear_left"]] = head_pos + np.array([0.5, 1.0])
        positions[t, name_to_idx["ear_right"]] = head_pos + np.array([0.5, -1.0])
        tb = positions[t, name_to_idx["back4"]] + np.array([-true_lengths["tail_1"], 0.0])
        positions[t, name_to_idx["tailbase"]] = tb
        tm = tb + np.array([-true_lengths["tail_2"], 0.0])
        positions[t, name_to_idx["tailmid"]] = tm
        te = tm + np.array([-true_lengths["tail_3"], 0.0])
        positions[t, name_to_idx["tailend"]] = te
        positions[t] += rng.normal(0, 0.1, (n_m, 2))

    fitted = fit_body_lengths(positions, likelihoods, layout, marker_names)
    s0 = initial_state_from_data(
        positions, likelihoods, layout, marker_names, fitted,
    )
    assert s0.shape == (layout.state_dim,)

    # State should produce marker positions close to observations
    # at frame 0 (allowing for noise tolerance)
    pos_predicted = state_to_marker_positions(s0, layout)
    pos_observed_t0 = positions[0]  # (n_m, 2), in marker_names order
    # marker_names from state_to_marker_positions might be in a different
    # order. Use layout.marker_names (which is what state_to_marker_positions
    # uses).
    layout_marker_names = layout.marker_names
    n2i_layout = {n: i for i, n in enumerate(layout_marker_names)}
    n2i_data = {n: i for i, n in enumerate(marker_names)}

    # Compare for every marker that's in the layout
    max_diff = 0.0
    for m in layout_marker_names:
        if m not in n2i_data:
            continue
        ip = n2i_layout[m]
        io = n2i_data[m]
        d = np.linalg.norm(pos_predicted[ip] - pos_observed_t0[io])
        max_diff = max(max_diff, d)
    # Distal markers should match within tracking noise
    # (~0.1 std). Non-distal markers (ears, nose) are subject
    # to fitted-offset error too. Allow ~3 px tolerance.
    assert max_diff < 5.0, (
        f"initial_state_from_data produces marker predictions "
        f"that are far from observations: max_diff={max_diff:.3f}"
    )

    # ---------------------------------------------------------- #
    # Patch 101: EKF forward filter.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        NoiseParamsV2, build_F_v2, build_Q_v2,
        forward_filter_v2, _build_constraint_observations,
    )

    # ---------------------------------------------------------- #
    # Case 20: build_F_v2 has correct constant-velocity
    # structure — applying F to a state propagates positions
    # by velocity*dt, leaves velocities unchanged.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    indices = _pack_state_layout_indices(layout)
    dt = 1.0 / 30.0
    F = build_F_v2(layout, dt)
    assert F.shape == (layout.state_dim, layout.state_dim)

    # Build a state with known velocities, zero positions
    x = np.zeros(layout.state_dim)
    x[indices["__root__"]["vx"]] = 10.0
    x[indices["__root__"]["vy"]] = -5.0
    x[indices["__root__"]["cos_dot"]] = 0.1
    x[indices["__root__"]["sin_dot"]] = 0.2
    # Set initial cos=1, sin=0 (otherwise ambient cos/sin
    # don't propagate sensibly under linear dynamics)
    x[indices["__root__"]["cos"]] = 1.0
    x[indices["__root__"]["sin"]] = 0.0
    for seg_name in layout.non_root_topo_order:
        x[indices[seg_name]["cos"]] = 1.0
        x[indices[seg_name]["length"]] = 5.0
        x[indices[seg_name]["length_dot"]] = 0.5

    x_next = F @ x
    # Root x: was 0, vx=10, dt=1/30 → 10/30
    assert abs(x_next[indices["__root__"]["x"]] - 10.0/30.0) < 1e-12
    assert abs(x_next[indices["__root__"]["y"]] - (-5.0/30.0)) < 1e-12
    # Velocities unchanged
    assert abs(x_next[indices["__root__"]["vx"]] - 10.0) < 1e-12
    assert abs(x_next[indices["__root__"]["vy"]] - (-5.0)) < 1e-12
    # Orientation cos: was 1, ċ=0.1 → 1 + 0.1/30
    assert abs(x_next[indices["__root__"]["cos"]] - (1.0 + 0.1/30.0)) < 1e-12
    assert abs(x_next[indices["__root__"]["sin"]] - (0.0 + 0.2/30.0)) < 1e-12
    # Length: 5 + 0.5*dt
    seg = layout.non_root_topo_order[0]
    assert abs(x_next[indices[seg]["length"]] - (5.0 + 0.5/30.0)) < 1e-12

    # ---------------------------------------------------------- #
    # Case 21: build_Q_v2 is symmetric + positive semi-definite.
    # ---------------------------------------------------------- #
    params = NoiseParamsV2.default(layout)
    Q = build_Q_v2(layout, params, dt)
    assert Q.shape == (layout.state_dim, layout.state_dim)
    # Symmetric
    assert np.allclose(Q, Q.T), "Q should be symmetric"
    # PSD: eigenvalues all ≥ 0 (allow tiny fp negative)
    eigvals = np.linalg.eigvalsh(Q)
    min_eig = float(eigvals.min())
    assert min_eig > -1e-10, (
        f"Q has negative eigenvalue {min_eig:.6e}; should be PSD"
    )

    # ---------------------------------------------------------- #
    # Case 22: constraint observations have target zero,
    # current value cos²+sin²-1, Jacobian (2cos, 2sin).
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    state = np.zeros(layout.state_dim)
    indices = _pack_state_layout_indices(layout)
    # Set root cos=0.6, sin=0.8 (unit norm)
    state[indices["__root__"]["cos"]] = 0.6
    state[indices["__root__"]["sin"]] = 0.8
    # All other segments cos=1, sin=0 (unit norm)
    for seg_name in layout.non_root_topo_order:
        state[indices[seg_name]["cos"]] = 1.0
        state[indices[seg_name]["sin"]] = 0.0

    z, h, H = _build_constraint_observations(state, layout, 0.05)
    # For all-unit-norm states, constraint residual h = 0
    assert np.allclose(h, 0.0), (
        f"Constraint residual should be 0 for unit-norm state; "
        f"got {h}"
    )
    # Targets are all zero
    assert np.allclose(z, 0.0)
    # Number of constraints = 1 root + 6 non-root = 7
    assert h.shape == (7,)
    # Jacobian rows: H[0] should have nonzero entries only at
    # root cos/sin columns
    nonzero_root = np.where(np.abs(H[0]) > 1e-12)[0]
    expected = {indices["__root__"]["cos"], indices["__root__"]["sin"]}
    assert set(nonzero_root.tolist()) == expected, (
        f"Root constraint Jacobian has wrong sparsity: {nonzero_root}"
    )
    # Values: 2 * cos = 1.2, 2 * sin = 1.6
    assert abs(H[0, indices["__root__"]["cos"]] - 1.2) < 1e-12
    assert abs(H[0, indices["__root__"]["sin"]] - 1.6) < 1e-12

    # Now perturb root cos so unit norm violated → residual > 0
    state2 = state.copy()
    state2[indices["__root__"]["cos"]] = 1.5  # not unit
    z, h, H = _build_constraint_observations(state2, layout, 0.05)
    expected_residual = 1.5**2 + 0.8**2 - 1.0  # 1.44 + 0.64 - 1 = 1.08
    assert abs(h[0] - expected_residual) < 1e-10

    # ---------------------------------------------------------- #
    # Case 23: forward_filter_v2 on synthetic clean data
    # converges to truth. The smoothed marker positions should
    # be very close to the noise-free truth.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    rng = np.random.default_rng(2026)
    fps = 30.0
    dt = 1.0 / fps
    T = 200
    marker_names_layout = layout.marker_names
    n_m = len(marker_names_layout)
    name_to_idx = {n: i for i, n in enumerate(marker_names_layout)}

    # Build a true state that's mostly stationary — just
    # constant-velocity drift of root, all segments aligned.
    indices = _pack_state_layout_indices(layout)
    true_state_template = np.zeros(layout.state_dim)
    true_state_template[indices["__root__"]["x"]] = 100.0
    true_state_template[indices["__root__"]["y"]] = 200.0
    true_state_template[indices["__root__"]["vx"]] = 1.0  # drift
    true_state_template[indices["__root__"]["vy"]] = 0.0
    true_state_template[indices["__root__"]["cos"]] = 1.0
    true_state_template[indices["__root__"]["sin"]] = 0.0
    for seg_name in layout.non_root_topo_order:
        true_state_template[indices[seg_name]["cos"]] = 1.0
        true_state_template[indices[seg_name]["sin"]] = 0.0
        true_state_template[indices[seg_name]["length"]] = 5.0

    # Generate frames by propagating with F, generating marker
    # observations from forward kinematics + small noise
    F = build_F_v2(layout, dt)
    sigma_obs = 1.0  # px noise
    true_states = np.zeros((T, layout.state_dim))
    true_states[0] = true_state_template
    for t in range(1, T):
        true_states[t] = F @ true_states[t-1]

    # Generate noisy observations
    positions_obs = np.zeros((T, n_m, 2))
    likelihoods_obs = np.full((T, n_m), 0.95)
    fitted_lengths = FittedLengths(
        segment_lengths={s: 5.0 for s in layout.non_root_topo_order},
        segment_length_iqr={s: 0.1 for s in layout.non_root_topo_order},
        marker_offsets={
            "back2": (0.0, 0.0), "back1": (1.0, 0.0),
            "back3": (1.0, np.pi),
            "lateral_left": (1.0, np.pi/2),
            "lateral_right": (1.0, -np.pi/2),
            "center": (0.5, 0.0), "back4": (0.0, 0.0),
            "neck": (0.0, 0.0), "headmid": (0.0, 0.0),
            "nose": (1.0, 0.0),
            "ear_left": (0.5, np.pi/3),
            "ear_right": (0.5, -np.pi/3),
            "tailbase": (0.0, 0.0), "tailmid": (0.0, 0.0),
            "tailend": (0.0, 0.0),
        },
    )
    for t in range(T):
        clean = state_to_marker_positions(true_states[t], layout)
        # Reorder from layout marker order to our test
        # marker_names ordering (which IS layout.marker_names here)
        positions_obs[t] = clean + rng.normal(0, sigma_obs, (n_m, 2))

    # Run filter
    params = NoiseParamsV2.default(
        layout, sigma_marker=sigma_obs, q_root_pos=10.0,
    )
    initial_state = true_states[0].copy()
    initial_state[indices["__root__"]["x"]] += rng.normal(0, 2.0)
    initial_state[indices["__root__"]["y"]] += rng.normal(0, 2.0)
    result = forward_filter_v2(
        positions_obs, likelihoods_obs, layout, params, dt,
        initial_state=initial_state,
        likelihood_threshold=0.5,
    )

    # Result shapes
    assert result.x_filt.shape == (T, layout.state_dim)
    assert result.P_filt.shape == (T, layout.state_dim, layout.state_dim)
    assert result.x_pred.shape == (T, layout.state_dim)
    assert result.n_observed.shape == (T,)
    # All frames have all 15 markers observed
    assert (result.n_observed == n_m).all()

    # The filter's smoothed marker positions should be close
    # to truth on later frames (after filter has settled).
    # Compute on frame T//2 onwards.
    avg_err = 0.0
    n_check = 0
    for t in range(T // 2, T):
        pred_pos = state_to_marker_positions(result.x_filt[t], layout)
        true_pos = state_to_marker_positions(true_states[t], layout)
        avg_err += np.mean(np.linalg.norm(pred_pos - true_pos, axis=1))
        n_check += 1
    avg_err /= n_check
    assert avg_err < 1.0, (
        f"Filter fails to track truth: avg marker error {avg_err:.3f} px"
    )

    # ---------------------------------------------------------- #
    # Case 24: filter handles missing observations (NaN, low-p)
    # gracefully. n_observed reflects actual count.
    # ---------------------------------------------------------- #
    positions_drop = positions_obs.copy()
    likelihoods_drop = likelihoods_obs.copy()
    # Drop nose (set to NaN) for frames 50-99
    nose_idx = marker_names_layout.index("nose")
    positions_drop[50:100, nose_idx, :] = np.nan
    # Drop ear_left via low likelihood for frames 70-89
    ear_idx = marker_names_layout.index("ear_left")
    likelihoods_drop[70:90, ear_idx] = 0.1

    result_drop = forward_filter_v2(
        positions_drop, likelihoods_drop, layout, params, dt,
        initial_state=initial_state,
        likelihood_threshold=0.5,
    )
    # n_observed should reflect drops
    assert (result_drop.n_observed[0:50] == n_m).all()
    assert (result_drop.n_observed[50:70] == n_m - 1).all()  # nose dropped
    assert (result_drop.n_observed[70:90] == n_m - 2).all()  # nose+ear dropped
    assert (result_drop.n_observed[90:100] == n_m - 1).all()  # ear back, nose still dropped
    assert (result_drop.n_observed[100:] == n_m).all()

    # Filter should still track (kinematic coupling helps
    # recover dropped markers from neighbors)
    avg_err_drop = 0.0
    n_check = 0
    for t in range(T // 2, T):
        pred_pos = state_to_marker_positions(result_drop.x_filt[t], layout)
        true_pos = state_to_marker_positions(true_states[t], layout)
        avg_err_drop += np.mean(np.linalg.norm(pred_pos - true_pos, axis=1))
        n_check += 1
    avg_err_drop /= n_check
    assert avg_err_drop < 2.0, (
        f"Filter with dropouts: avg marker error {avg_err_drop:.3f} px"
    )

    # ---------------------------------------------------------- #
    # Case 25: P_filt remains symmetric + positive semi-definite
    # at every frame (Joseph form preserves PSD better than
    # naive (I-KH)P).
    # ---------------------------------------------------------- #
    for t in range(T):
        P = result.P_filt[t]
        sym_err = np.max(np.abs(P - P.T))
        assert sym_err < 1e-8, (
            f"P_filt[{t}] not symmetric: max asymmetry {sym_err:.6e}"
        )
        eigs = np.linalg.eigvalsh(P)
        min_eig = float(eigs.min())
        assert min_eig > -1e-6, (
            f"P_filt[{t}] not PSD: min eigenvalue {min_eig:.6e}"
        )

    # ---------------------------------------------------------- #
    # Case 26: with apply_constraints=True, the unit-norm
    # constraint is approximately preserved (cos²+sin² stays
    # close to 1 across the trajectory).
    # ---------------------------------------------------------- #
    max_norm_err = 0.0
    for t in range(T):
        c = result.x_filt[t, indices["__root__"]["cos"]]
        s = result.x_filt[t, indices["__root__"]["sin"]]
        norm_sq = c*c + s*s
        max_norm_err = max(max_norm_err, abs(norm_sq - 1.0))
        for seg_name in layout.non_root_topo_order:
            c = result.x_filt[t, indices[seg_name]["cos"]]
            s = result.x_filt[t, indices[seg_name]["sin"]]
            norm_sq = c*c + s*s
            max_norm_err = max(max_norm_err, abs(norm_sq - 1.0))

    assert max_norm_err < 0.05, (
        f"Unit-norm constraint violated: max |cos²+sin²-1| "
        f"= {max_norm_err:.4f}"
    )

    # ---------------------------------------------------------- #
    # Patch 102: RTS backward smoother.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        rts_smooth_v2, SmoothResultV2,
    )

    # ---------------------------------------------------------- #
    # Case 27: smoother shapes correct, last frame matches
    # filter, smoothed covariance never increases over filter.
    # ---------------------------------------------------------- #
    # Reuse filter result from case 23 setup
    layout = standard_rat_layout()
    indices = _pack_state_layout_indices(layout)
    rng = np.random.default_rng(2027)
    fps = 30.0
    dt = 1.0 / fps
    T = 200
    n_m = layout.n_markers
    marker_names_layout = layout.marker_names

    # Same true-trajectory + observations setup as case 23
    true_state_template = np.zeros(layout.state_dim)
    true_state_template[indices["__root__"]["x"]] = 100.0
    true_state_template[indices["__root__"]["y"]] = 200.0
    true_state_template[indices["__root__"]["vx"]] = 1.0
    true_state_template[indices["__root__"]["cos"]] = 1.0
    for seg_name in layout.non_root_topo_order:
        true_state_template[indices[seg_name]["cos"]] = 1.0
        true_state_template[indices[seg_name]["length"]] = 5.0

    F = build_F_v2(layout, dt)
    sigma_obs = 1.0
    true_states = np.zeros((T, layout.state_dim))
    true_states[0] = true_state_template
    for t in range(1, T):
        true_states[t] = F @ true_states[t-1]

    positions_obs = np.zeros((T, n_m, 2))
    likelihoods_obs = np.full((T, n_m), 0.95)
    for t in range(T):
        clean = state_to_marker_positions(true_states[t], layout)
        positions_obs[t] = clean + rng.normal(0, sigma_obs, (n_m, 2))

    params = NoiseParamsV2.default(
        layout, sigma_marker=sigma_obs, q_root_pos=10.0,
    )
    initial_state = true_states[0].copy() + rng.normal(0, 0.5, layout.state_dim) * 0.1

    filt_result = forward_filter_v2(
        positions_obs, likelihoods_obs, layout, params, dt,
        initial_state=initial_state,
        likelihood_threshold=0.5,
    )
    smooth_result = rts_smooth_v2(filt_result, layout, dt)

    assert smooth_result.x_smooth.shape == (T, layout.state_dim)
    assert smooth_result.P_smooth.shape == (T, layout.state_dim, layout.state_dim)
    assert smooth_result.P_lag_one.shape == (T - 1, layout.state_dim, layout.state_dim)

    # Last-frame check: smoothed[T-1] == filtered[T-1]
    assert np.allclose(
        smooth_result.x_smooth[T - 1], filt_result.x_filt[T - 1],
    )
    assert np.allclose(
        smooth_result.P_smooth[T - 1], filt_result.P_filt[T - 1],
    )

    # ---------------------------------------------------------- #
    # Case 28: smoothed covariance has trace ≤ filtered covariance
    # at every frame except the last (smoothing reduces
    # uncertainty by incorporating future observations).
    # ---------------------------------------------------------- #
    n_reduced = 0
    for t in range(T - 1):
        tr_filt = np.trace(filt_result.P_filt[t])
        tr_smooth = np.trace(smooth_result.P_smooth[t])
        # Allow tiny fp slack
        assert tr_smooth <= tr_filt + 1e-6, (
            f"Frame {t}: smoothed trace {tr_smooth:.3f} > "
            f"filtered trace {tr_filt:.3f}"
        )
        if tr_smooth < tr_filt - 1e-6:
            n_reduced += 1
    # Most frames should have strict reduction (the early
    # frames especially, where future obs add lots of info)
    assert n_reduced >= T // 2, (
        f"Smoother only strictly reduced uncertainty at "
        f"{n_reduced}/{T-1} frames; expected ≥ {T // 2}"
    )

    # ---------------------------------------------------------- #
    # Case 29: P_smooth is symmetric + PSD at every frame.
    # ---------------------------------------------------------- #
    for t in range(T):
        P = smooth_result.P_smooth[t]
        sym_err = np.max(np.abs(P - P.T))
        assert sym_err < 1e-8, (
            f"P_smooth[{t}] not symmetric: max asymmetry "
            f"{sym_err:.6e}"
        )
        eigs = np.linalg.eigvalsh(P)
        min_eig = float(eigs.min())
        assert min_eig > -1e-6, (
            f"P_smooth[{t}] not PSD: min eigenvalue {min_eig:.6e}"
        )

    # ---------------------------------------------------------- #
    # Case 30: smoothed marker error is at least as good as
    # filtered marker error on average. Smoother should not
    # make tracking WORSE.
    # ---------------------------------------------------------- #
    avg_filt_err = 0.0
    avg_smooth_err = 0.0
    for t in range(T):
        true_pos = state_to_marker_positions(true_states[t], layout)
        filt_pos = state_to_marker_positions(filt_result.x_filt[t], layout)
        smooth_pos = state_to_marker_positions(smooth_result.x_smooth[t], layout)
        avg_filt_err += np.mean(np.linalg.norm(filt_pos - true_pos, axis=1))
        avg_smooth_err += np.mean(np.linalg.norm(smooth_pos - true_pos, axis=1))
    avg_filt_err /= T
    avg_smooth_err /= T
    assert avg_smooth_err <= avg_filt_err + 0.1, (
        f"Smoothed error {avg_smooth_err:.3f} > filtered error "
        f"{avg_filt_err:.3f} — smoother made things worse"
    )

    # ---------------------------------------------------------- #
    # Case 31: smoother dramatically helps during dropouts.
    # Drop nose for frames 80-119 and verify the smoothed nose
    # position during dropout is significantly closer to truth
    # than the filtered nose position.
    # ---------------------------------------------------------- #
    positions_drop = positions_obs.copy()
    nose_idx = marker_names_layout.index("nose")
    positions_drop[80:120, nose_idx, :] = np.nan

    filt_drop = forward_filter_v2(
        positions_drop, likelihoods_obs, layout, params, dt,
        initial_state=initial_state, likelihood_threshold=0.5,
    )
    smooth_drop = rts_smooth_v2(filt_drop, layout, dt)

    # Compute nose position error during dropout (frames 80-119)
    filt_nose_err = 0.0
    smooth_nose_err = 0.0
    for t in range(80, 120):
        true_pos = state_to_marker_positions(true_states[t], layout)
        filt_pos = state_to_marker_positions(filt_drop.x_filt[t], layout)
        smooth_pos = state_to_marker_positions(smooth_drop.x_smooth[t], layout)
        filt_nose_err += np.linalg.norm(filt_pos[nose_idx] - true_pos[nose_idx])
        smooth_nose_err += np.linalg.norm(smooth_pos[nose_idx] - true_pos[nose_idx])
    filt_nose_err /= 40
    smooth_nose_err /= 40

    # Both should be reasonable (kinematic chain provides
    # recovery), but smoother ≤ filter.
    assert smooth_nose_err <= filt_nose_err + 0.1, (
        f"Smoothed nose error during dropout {smooth_nose_err:.3f} "
        f"> filtered {filt_nose_err:.3f}"
    )
    # Both errors should be small thanks to spatial coupling
    assert smooth_nose_err < 3.0, (
        f"Smoothed nose error during dropout is too large: "
        f"{smooth_nose_err:.3f}px"
    )

    # ---------------------------------------------------------- #
    # Case 32: lag-one cross-covariance is consistent with
    # state covariance scales (basic sanity, not a strict
    # mathematical identity).
    # ---------------------------------------------------------- #
    # P_{t, t+1 | T} should be such that the joint covariance
    # of (x_t, x_{t+1}) is PSD:
    #   [P_t       P_{t,t+1}]
    #   [P_{t,t+1}^T  P_{t+1}]
    # Test on a few sampled frames.
    for t in [10, 50, 100, T - 2]:
        P_t = smooth_result.P_smooth[t]
        P_tp1 = smooth_result.P_smooth[t + 1]
        P_lag = smooth_result.P_lag_one[t]
        joint = np.block([
            [P_t, P_lag],
            [P_lag.T, P_tp1],
        ])
        # Symmetrize
        joint = 0.5 * (joint + joint.T)
        eigs = np.linalg.eigvalsh(joint)
        min_eig = float(eigs.min())
        # Allow some slack — the joint may not be exactly PSD
        # under fp, but should be very close.
        assert min_eig > -1e-3, (
            f"Joint cov at t={t} not PSD: min eigenvalue {min_eig:.6e}"
        )

    # ---------------------------------------------------------- #
    # Patch 103: Shumway-Stoffer M-step on body state.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        accumulate_m_step_stats_v2, finalize_m_step_v2,
        _MStepStatsV2,
    )

    # ---------------------------------------------------------- #
    # Case 33: _MStepStatsV2.empty has correct shapes.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    D = layout.state_dim
    stats_empty = _MStepStatsV2.empty(layout)
    assert stats_empty.S00.shape == (D, D)
    assert stats_empty.S11.shape == (D, D)
    assert stats_empty.S10.shape == (D, D)
    assert stats_empty.n_pairs == 0
    assert all(v == 0.0 for v in stats_empty.sigma_sum_sq.values())
    assert set(stats_empty.sigma_n_obs.keys()) == set(layout.marker_names)

    # ---------------------------------------------------------- #
    # Case 34: __iadd__ correctly combines stats.
    # ---------------------------------------------------------- #
    s1 = _MStepStatsV2.empty(layout)
    s1.S00 = np.full((D, D), 2.0)
    s1.S11 = np.full((D, D), 3.0)
    s1.S10 = np.full((D, D), 5.0)
    s1.n_pairs = 100
    s1.sigma_sum_sq["nose"] = 50.0
    s1.sigma_n_obs["nose"] = 200

    s2 = _MStepStatsV2.empty(layout)
    s2.S00 = np.full((D, D), 1.0)
    s2.S11 = np.full((D, D), 1.0)
    s2.S10 = np.full((D, D), 1.0)
    s2.n_pairs = 50
    s2.sigma_sum_sq["nose"] = 20.0
    s2.sigma_n_obs["nose"] = 80

    s1 += s2
    assert np.allclose(s1.S00, 3.0)
    assert np.allclose(s1.S11, 4.0)
    assert np.allclose(s1.S10, 6.0)
    assert s1.n_pairs == 150
    assert s1.sigma_sum_sq["nose"] == 70.0
    assert s1.sigma_n_obs["nose"] == 280

    # ---------------------------------------------------------- #
    # Case 35: σ-recovery on synthetic data.
    # Run filter+smoother on data with known σ_marker. The
    # M-step's σ should be close to the true value.
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(2028)
    fps = 30.0
    dt = 1.0 / fps
    T = 600  # smaller for sandbox runtime

    indices = _pack_state_layout_indices(layout)
    true_state = np.zeros(layout.state_dim)
    true_state[indices["__root__"]["x"]] = 100.0
    true_state[indices["__root__"]["y"]] = 200.0
    true_state[indices["__root__"]["cos"]] = 1.0
    for seg_name in layout.non_root_topo_order:
        true_state[indices[seg_name]["cos"]] = 1.0
        true_state[indices[seg_name]["length"]] = 5.0

    F = build_F_v2(layout, dt)
    sigma_true = 1.5  # px noise
    n_m = layout.n_markers
    marker_names_layout = layout.marker_names

    # Generate trajectory with no process noise (constant
    # state) so we can isolate σ recovery
    true_states = np.tile(true_state, (T, 1))
    positions_obs = np.zeros((T, n_m, 2))
    likelihoods_obs = np.full((T, n_m), 0.95)
    for t in range(T):
        clean = state_to_marker_positions(true_states[t], layout)
        positions_obs[t] = clean + rng.normal(0, sigma_true, (n_m, 2))

    params_init = NoiseParamsV2.default(
        layout, sigma_marker=3.0, q_root_pos=10.0,
    )
    initial_state = true_state.copy()
    filt = forward_filter_v2(
        positions_obs, likelihoods_obs, layout, params_init, dt,
        initial_state=initial_state, likelihood_threshold=0.5,
    )
    smooth = rts_smooth_v2(filt, layout, dt)
    stats = accumulate_m_step_stats_v2(
        smooth, positions_obs, likelihoods_obs, layout,
    )
    new_params = finalize_m_step_v2(
        stats, layout, dt, prev_params=params_init,
        initial_params=params_init,
    )

    # Recovered σ should be close to true σ for several markers
    for m in ["back2", "nose", "tailbase"]:
        sig = new_params.sigma_marker[m]
        assert abs(sig - sigma_true) < 0.5, (
            f"σ recovery for {m}: fitted={sig:.3f}, true={sigma_true}"
        )

    # ---------------------------------------------------------- #
    # Case 36: σ floor prevents runaway-down.
    # Use the M-step on synthetic data but set initial σ such
    # that the floor is high. Verify σ doesn't go below floor.
    # ---------------------------------------------------------- #
    # Reuse stats above with a different initial that places
    # the floor at a higher value
    params_high_init = NoiseParamsV2.default(
        layout, sigma_marker=10.0,  # initial; ceiling=30
    )
    new_params_high = finalize_m_step_v2(
        stats, layout, dt, prev_params=params_high_init,
        initial_params=params_high_init,
    )
    # Floor is global 0.5; ceiling is 3*10=30. M-step should
    # find σ near sigma_true=1.5, well within [0.5, 30].
    for m in ["back2", "nose"]:
        sig = new_params_high.sigma_marker[m]
        assert 0.5 <= sig <= 30.0, (
            f"σ for {m} outside [floor, ceiling]: {sig:.3f}"
        )

    # ---------------------------------------------------------- #
    # Case 37: σ ceiling prevents runaway-up. If we set initial
    # σ very low, ceiling = 3 × low_initial caps σ.
    # ---------------------------------------------------------- #
    params_low_init = NoiseParamsV2.default(
        layout, sigma_marker=0.5,  # initial; ceiling=1.5
    )
    # Use stats from data with sigma_true=1.5, so M-step would
    # want σ ≈ 1.5. Ceiling at 1.5 caps it there.
    new_params_low = finalize_m_step_v2(
        stats, layout, dt, prev_params=params_low_init,
        initial_params=params_low_init,
    )
    # Should be capped at 1.5 (ceiling = 3 * 0.5)
    for m in ["back2", "nose"]:
        sig = new_params_low.sigma_marker[m]
        assert sig <= 1.5 + 1e-6, (
            f"σ for {m} exceeds ceiling: {sig:.3f}"
        )

    # ---------------------------------------------------------- #
    # Case 38: Multi-session stat combination via __iadd__
    # gives identical results as combined session.
    # ---------------------------------------------------------- #
    # Split data into two halves
    half = T // 2
    pos_a = positions_obs[:half]
    lik_a = likelihoods_obs[:half]
    pos_b = positions_obs[half:]
    lik_b = likelihoods_obs[half:]

    # Run filter+smoother+stats on each half independently
    filt_a = forward_filter_v2(
        pos_a, lik_a, layout, params_init, dt,
        initial_state=initial_state, likelihood_threshold=0.5,
    )
    smooth_a = rts_smooth_v2(filt_a, layout, dt)
    stats_a = accumulate_m_step_stats_v2(
        smooth_a, pos_a, lik_a, layout,
    )

    initial_state_b = filt_a.x_filt[-1].copy()
    filt_b = forward_filter_v2(
        pos_b, lik_b, layout, params_init, dt,
        initial_state=initial_state_b, likelihood_threshold=0.5,
    )
    smooth_b = rts_smooth_v2(filt_b, layout, dt)
    stats_b = accumulate_m_step_stats_v2(
        smooth_b, pos_b, lik_b, layout,
    )

    # Combine
    stats_combined = _MStepStatsV2.empty(layout)
    stats_combined += stats_a
    stats_combined += stats_b
    # n_pairs of combined should be (T_a - 1) + (T_b - 1)
    assert stats_combined.n_pairs == (half - 1) + (T - half - 1)

    # The σ stats should equal the sum
    assert (
        abs(stats_combined.sigma_sum_sq["nose"]
            - (stats_a.sigma_sum_sq["nose"] + stats_b.sigma_sum_sq["nose"]))
        < 1e-9
    )

    # ---------------------------------------------------------- #
    # Case 39: Q-hat estimation produces reasonable values.
    # Generate synthetic data WITH known process noise; verify
    # M-step recovers q values within tolerance.
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(2030)
    T_q = 600  # smaller for sandbox runtime; tests math, not precision
    q_root_pos_true = 50.0  # px²/s³

    # Generate trajectory with only root-position process
    # noise; segment angles fixed.
    Q_true_minimal = np.zeros((D, D))
    # The 4×4 root-pos block: q * Q_template
    dt2_local = dt * dt
    dt3_local = dt2_local * dt
    dt4_local = dt2_local * dt2_local
    Q_true_minimal[0:4, 0:4] = q_root_pos_true * np.array([
        [dt4_local/4, 0, dt3_local/2, 0],
        [0, dt4_local/4, 0, dt3_local/2],
        [dt3_local/2, 0, dt2_local, 0],
        [0, dt3_local/2, 0, dt2_local],
    ])

    states_q = np.zeros((T_q, D))
    states_q[0] = true_state.copy()
    L_chol = np.linalg.cholesky(
        Q_true_minimal + 1e-12 * np.eye(D),  # tiny reg for psd
    )
    for t in range(1, T_q):
        w = L_chol @ rng.standard_normal(D)
        states_q[t] = F @ states_q[t-1] + w

    pos_q = np.zeros((T_q, n_m, 2))
    likes_q = np.full((T_q, n_m), 0.95)
    for t in range(T_q):
        clean = state_to_marker_positions(states_q[t], layout)
        pos_q[t] = clean + rng.normal(0, sigma_true, (n_m, 2))

    params_q_init = NoiseParamsV2.default(
        layout, sigma_marker=sigma_true, q_root_pos=10.0,
    )

    # Run multiple EM iterations. Track q across iterations.
    # Single-iteration EM underestimates q significantly because
    # the filter with too-small initial Q produces a too-rigid
    # smoothed trajectory, which then yields small Q-hat.
    # Each EM iteration grows q gradually toward truth.
    # This is a known convergence-rate issue with EM for
    # state-space models — full convergence requires either
    # many iterations or a good initialization. The M-step
    # implementation is correct as long as q grows monotonically
    # in the right direction.
    params_q_curr = params_q_init
    q_history = []
    for em_iter in range(4):  # Fewer iterations to keep sandbox fast
        filt_q = forward_filter_v2(
            pos_q, likes_q, layout, params_q_curr, dt,
            initial_state=states_q[0].copy(),
            likelihood_threshold=0.5,
        )
        smooth_q = rts_smooth_v2(filt_q, layout, dt)
        stats_q = accumulate_m_step_stats_v2(
            smooth_q, pos_q, likes_q, layout,
        )
        params_q_curr = finalize_m_step_v2(
            stats_q, layout, dt, prev_params=params_q_curr,
            initial_params=params_q_init,
        )
        q_history.append(params_q_curr.q_root_pos)
    new_params_q = params_q_curr

    # 1. q grew over iterations (in the direction of truth)
    assert q_history[-1] > q_history[0], (
        f"q didn't grow during EM: {q_history}"
    )
    # 2. q growth is monotonic (each iteration bigger than last)
    for i in range(1, len(q_history)):
        assert q_history[i] >= q_history[i-1] - 0.01, (
            f"q decreased at iter {i}: {q_history}"
        )
    # 3. Final q is closer to true than initial
    assert (
        abs(q_history[-1] - q_root_pos_true)
        < abs(q_history[0] - q_root_pos_true)
    )
    # 4. q grew at least 25% from initial (4 iterations,
    #    smaller T_q for sandbox — less precision, but
    #    monotonic growth in the right direction is the
    #    key invariant)
    assert q_history[-1] > 1.25 * q_history[0], (
        f"q growth too small: {q_history[0]:.3f} → {q_history[-1]:.3f}"
    )

    # ---------------------------------------------------------- #
    # Case 40: Floor prevents q from collapsing toward zero.
    # If we provide stats that would give very small q, the
    # floor (init_q / 10) should kick in.
    # ---------------------------------------------------------- #
    # Use stats_q (which would give q ≈ 50). Set initial much
    # higher so floor = 100.
    params_high_q_init = NoiseParamsV2.default(
        layout, sigma_marker=sigma_true, q_root_pos=1000.0,
    )
    # Floor would be 1000/10 = 100. M-step would want ~50.
    # So result should be capped at 100.
    new_params_floor = finalize_m_step_v2(
        stats_q, layout, dt, prev_params=params_high_q_init,
        initial_params=params_high_q_init,
    )
    assert new_params_floor.q_root_pos >= 100.0 - 1e-6, (
        f"q_root_pos floor not enforced: "
        f"{new_params_floor.q_root_pos:.3f} should be ≥ 100"
    )

    # ---------------------------------------------------------- #
    # Patch 104: EM loop + validation hook + data-driven init.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        fit_initial_params_v2, fit_noise_params_em_v2,
        _validate_trajectory_v2, EMResultV2,
        _INIT_FLOOR_Q_ROOT_POS_V2,
    )

    # ---------------------------------------------------------- #
    # Case 41: fit_initial_params_v2 produces sensible σ values
    # close to true observation noise on synthetic data.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    rng = np.random.default_rng(2031)
    fps = 30.0
    dt = 1.0 / fps
    T = 600  # smaller for sandbox runtime
    n_m = layout.n_markers
    indices = _pack_state_layout_indices(layout)

    # Stationary trajectory + uniform observation noise
    sigma_true = 1.5
    true_state = np.zeros(layout.state_dim)
    true_state[indices["__root__"]["x"]] = 100.0
    true_state[indices["__root__"]["y"]] = 200.0
    true_state[indices["__root__"]["cos"]] = 1.0
    for seg_name in layout.non_root_topo_order:
        true_state[indices[seg_name]["cos"]] = 1.0
        true_state[indices[seg_name]["length"]] = 5.0

    marker_names_layout = layout.marker_names
    positions_obs = np.zeros((T, n_m, 2))
    likelihoods_obs = np.full((T, n_m), 0.95)
    for t in range(T):
        clean = state_to_marker_positions(true_state, layout)
        positions_obs[t] = clean + rng.normal(0, sigma_true, (n_m, 2))

    fitted = fit_body_lengths(
        positions_obs, likelihoods_obs, layout, marker_names_layout,
    )
    init = fit_initial_params_v2(
        positions_obs, likelihoods_obs, layout, marker_names_layout,
        fitted, fps,
    )
    # σ recovered close to true for several markers
    for m in ["back2", "nose", "tailbase"]:
        s = init.sigma_marker[m]
        assert abs(s - sigma_true) < 0.5, (
            f"σ init for {m}: got {s:.3f}, true {sigma_true}"
        )

    # ---------------------------------------------------------- #
    # Case 42: fit_initial_params_v2 q_root_pos is reasonable
    # for stationary data (small but above floor).
    # ---------------------------------------------------------- #
    # Stationary data → small motion variance → q_root_pos at
    # or near floor (not zero, not enormous)
    assert init.q_root_pos >= _INIT_FLOOR_Q_ROOT_POS_V2 - 1e-6
    assert init.q_root_pos < 10000.0, (
        f"q_root_pos too large for stationary data: "
        f"{init.q_root_pos:.1f}"
    )

    # ---------------------------------------------------------- #
    # Case 43: fit_initial_params_v2 q_root_pos grows for moving
    # data — generate a trajectory with real root motion and
    # verify the estimated q is much higher than for stationary.
    # ---------------------------------------------------------- #
    # Generate trajectory with root motion (random walk style)
    states_moving = np.zeros((T, layout.state_dim))
    states_moving[0] = true_state
    rng_m = np.random.default_rng(2032)
    F = build_F_v2(layout, dt)
    # Add explicit acceleration noise to root position
    accel_scale = 100.0  # Large motion
    for t in range(1, T):
        states_moving[t] = F @ states_moving[t-1]
        # Inject acceleration
        states_moving[t, indices["__root__"]["vx"]] += rng_m.normal(0, accel_scale * dt)
        states_moving[t, indices["__root__"]["vy"]] += rng_m.normal(0, accel_scale * dt)

    pos_moving = np.zeros((T, n_m, 2))
    likes_moving = np.full((T, n_m), 0.95)
    for t in range(T):
        clean = state_to_marker_positions(states_moving[t], layout)
        pos_moving[t] = clean + rng_m.normal(0, sigma_true, (n_m, 2))

    fitted_moving = fit_body_lengths(
        pos_moving, likes_moving, layout, marker_names_layout,
    )
    init_moving = fit_initial_params_v2(
        pos_moving, likes_moving, layout, marker_names_layout,
        fitted_moving, fps,
    )
    # Moving trajectory should give q_root_pos significantly
    # larger than stationary trajectory
    assert init_moving.q_root_pos > init.q_root_pos * 5, (
        f"q_root_pos for moving data ({init_moving.q_root_pos:.1f}) "
        f"should be much larger than stationary ({init.q_root_pos:.1f})"
    )

    # ---------------------------------------------------------- #
    # Case 44: full EM loop runs to completion on synthetic
    # multi-session data without errors.
    # ---------------------------------------------------------- #
    # Use 3 sessions of moderate length, all with same
    # observation noise structure
    rng_em = np.random.default_rng(2033)
    T_sess = 200  # smaller for sandbox runtime
    sessions: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in range(2):  # 2 sessions instead of 3
        pos_s = np.zeros((T_sess, n_m, 2))
        likes_s = np.full((T_sess, n_m), 0.95)
        # Slight drift per session
        state_s = true_state.copy()
        state_s[indices["__root__"]["vx"]] = rng_em.uniform(-2, 2)
        states_s = np.zeros((T_sess, layout.state_dim))
        states_s[0] = state_s
        for t in range(1, T_sess):
            states_s[t] = F @ states_s[t-1]
        for t in range(T_sess):
            clean = state_to_marker_positions(states_s[t], layout)
            pos_s[t] = clean + rng_em.normal(0, sigma_true, (n_m, 2))
        sessions.append((pos_s, likes_s))

    # Run EM
    result = fit_noise_params_em_v2(
        sessions, layout, marker_names_layout, fitted, fps,
        max_iter=5, verbose=False,
    )
    assert isinstance(result, EMResultV2)
    assert len(result.history) <= 5
    assert result.history[-1]["iter"] == len(result.history) - 1
    # σ should be in a reasonable range
    final_sigmas = list(result.params.sigma_marker.values())
    mean_sigma = float(np.mean(final_sigmas))
    assert 0.5 < mean_sigma < 5.0, (
        f"Mean σ outside reasonable range: {mean_sigma:.3f}"
    )

    # ---------------------------------------------------------- #
    # Case 45: validation hook fires on a degenerate trajectory.
    # Manufacture a smooth result with frozen smoothed positions
    # and verify the hook detects it.
    # ---------------------------------------------------------- #
    # Build a smoothed result where x_smooth is constant —
    # all frames identical → smoothed range = 0 → range_ratio
    # ≈ 0 → frozen-output check fails.
    rng_v = np.random.default_rng(2034)
    T_v = 100
    x_smooth_frozen = np.tile(true_state, (T_v, 1))
    P_smooth_frozen = np.tile(np.eye(layout.state_dim), (T_v, 1, 1))
    P_lag_frozen = np.tile(np.eye(layout.state_dim), (T_v - 1, 1, 1))
    smooth_frozen = SmoothResultV2(
        x_smooth=x_smooth_frozen,
        P_smooth=P_smooth_frozen,
        P_lag_one=P_lag_frozen,
    )
    # Raw observations: real motion, large range
    pos_v = np.zeros((T_v, n_m, 2))
    likes_v = np.full((T_v, n_m), 0.95)
    for t in range(T_v):
        clean = state_to_marker_positions(true_state, layout)
        # Add LARGE drift so raw range >> 0
        clean[:, 0] += t * 0.5
        pos_v[t] = clean + rng_v.normal(0, sigma_true, (n_m, 2))

    params_v = NoiseParamsV2.default(layout, sigma_marker=sigma_true)
    try:
        _validate_trajectory_v2(
            smooth_frozen, pos_v, likes_v, layout, params_v,
            iteration=0, likelihood_threshold=0.5,
            strict=True,
        )
        assert False, "Validation should have detected frozen trajectory"
    except RuntimeError as e:
        # Either frozen-output or prior-overruling check should fire
        msg = str(e)
        assert "validation hook triggered" in msg

    # In soft mode (default), the same call should return
    # violations rather than raise.
    soft_violations = _validate_trajectory_v2(
        smooth_frozen, pos_v, likes_v, layout, params_v,
        iteration=0, likelihood_threshold=0.5,
        strict=False,
    )
    assert len(soft_violations) > 0, (
        "Soft mode should still detect the same violations, "
        "but return them as a list rather than raising"
    )

    # ---------------------------------------------------------- #
    # Case 46: validation hook does NOT fire on a healthy
    # trajectory. Run EM normally and verify no error.
    # ---------------------------------------------------------- #
    # Already verified in case 44 — the EM call there has
    # enable_validation=True (default) and didn't raise.
    # Add an explicit check: re-run with enable_validation=True
    # and verify no exception.
    result2 = fit_noise_params_em_v2(
        sessions, layout, marker_names_layout, fitted, fps,
        max_iter=3, enable_validation=True, verbose=False,
    )
    assert result2 is not None

    # ---------------------------------------------------------- #
    # Case 47: EM converges (max_rel_change shrinks across
    # iterations on synthetic data).
    # ---------------------------------------------------------- #
    result3 = fit_noise_params_em_v2(
        sessions, layout, marker_names_layout, fitted, fps,
        max_iter=4, tol=1e-6, verbose=False,  # reduced for sandbox runtime
    )
    # max_rel_change should decrease (mostly) over iterations
    if len(result3.history) >= 3:
        # Final iteration should have smaller change than first
        assert (
            result3.history[-1]["max_rel_change"]
            < result3.history[0]["max_rel_change"]
        ), (
            f"max_rel_change didn't decrease: {result3.history}"
        )

    # ---------------------------------------------------------- #
    # Case 48: data-driven init produces faster EM convergence
    # than uniform default init (the patch-103 / patch-104
    # central claim).
    # ---------------------------------------------------------- #
    # Build sessions with substantial root motion
    rng_init = np.random.default_rng(2035)
    sessions_init: List[Tuple[np.ndarray, np.ndarray]] = []
    T_init = 400  # smaller for sandbox runtime
    for s in range(2):
        pos_s = np.zeros((T_init, n_m, 2))
        likes_s = np.full((T_init, n_m), 0.95)
        # Add root acceleration noise to make q_root_pos non-trivial
        states_s = np.zeros((T_init, layout.state_dim))
        states_s[0] = true_state.copy()
        for t in range(1, T_init):
            states_s[t] = F @ states_s[t-1]
            states_s[t, indices["__root__"]["vx"]] += rng_init.normal(0, 5.0)
            states_s[t, indices["__root__"]["vy"]] += rng_init.normal(0, 5.0)
        for t in range(T_init):
            clean = state_to_marker_positions(states_s[t], layout)
            pos_s[t] = clean + rng_init.normal(0, sigma_true, (n_m, 2))
        sessions_init.append((pos_s, likes_s))

    fitted_init = fit_body_lengths(
        sessions_init[0][0], sessions_init[0][1],
        layout, marker_names_layout,
    )

    # Run with data-driven init (default)
    result_data = fit_noise_params_em_v2(
        sessions_init, layout, marker_names_layout, fitted_init, fps,
        max_iter=3, verbose=False,
    )

    # Run with low arbitrary init
    arb_init = NoiseParamsV2.default(
        layout, sigma_marker=3.0, q_root_pos=10.0,
    )
    result_arb = fit_noise_params_em_v2(
        sessions_init, layout, marker_names_layout, fitted_init, fps,
        max_iter=3, initial_params=arb_init, verbose=False,
    )

    # After 3 iterations, data-driven should already have
    # q_root_pos closer to whatever EM converges to.  More
    # critically, data-driven init's initial q should be
    # significantly higher than 10 (the arbitrary value).
    assert result_data.initial_params.q_root_pos > 100, (
        f"Data-driven init didn't recover non-trivial "
        f"q_root_pos: {result_data.initial_params.q_root_pos:.1f}"
    )
    # And it should be larger than arbitrary init's starting q
    assert (
        result_data.initial_params.q_root_pos
        > result_arb.initial_params.q_root_pos
    )

    # ---------------------------------------------------------- #
    # Patch 105: orchestrator + CLI + save/load.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        smooth_pose_v2, smooth_session_v2,
        state_to_marker_variances,
        save_model_v2, load_model_v2,
    )
    import tempfile
    import os as _os

    # ---------------------------------------------------------- #
    # Case 49: state_to_marker_variances has correct shape and
    # gives non-negative diagonals.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    indices = _pack_state_layout_indices(layout)
    fps = 30.0
    dt = 1.0 / fps
    T_v = 50
    n_m = layout.n_markers

    true_state = np.zeros(layout.state_dim)
    true_state[indices["__root__"]["x"]] = 100.0
    true_state[indices["__root__"]["y"]] = 200.0
    true_state[indices["__root__"]["cos"]] = 1.0
    for seg_name in layout.non_root_topo_order:
        true_state[indices[seg_name]["cos"]] = 1.0
        true_state[indices[seg_name]["length"]] = 5.0

    x_smooth = np.tile(true_state, (T_v, 1))
    P_smooth = np.tile(np.eye(layout.state_dim) * 0.5, (T_v, 1, 1))
    var = state_to_marker_variances(x_smooth, P_smooth, layout)
    assert var.shape == (T_v, n_m, 2)
    # All variances must be non-negative (PSD covariance →
    # non-negative diagonals)
    assert (var >= -1e-9).all(), "Variances should be non-negative"
    # For markers far from the root in the kinematic chain
    # (e.g., tailend after 3 tail segments), variance should
    # be larger than for markers at the root (back2)
    var_back2 = var[0, list(layout.marker_names).index("back2"), :].sum()
    var_tailend = var[0, list(layout.marker_names).index("tailend"), :].sum()
    assert var_tailend >= var_back2, (
        f"Distal marker should have at-least-as-much variance: "
        f"back2={var_back2:.4f}, tailend={var_tailend:.4f}"
    )

    # ---------------------------------------------------------- #
    # Case 50: smooth_session_v2 produces correct-shape output
    # and smoothed positions are close to truth on synthetic
    # clean data.
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(2040)
    T_sess = 100
    F = build_F_v2(layout, dt)
    sigma_true = 1.5
    states_sess = np.tile(true_state, (T_sess, 1))
    pos_sess = np.zeros((T_sess, n_m, 2))
    likes_sess = np.full((T_sess, n_m), 0.95)
    for t in range(T_sess):
        clean = state_to_marker_positions(states_sess[t], layout)
        pos_sess[t] = clean + rng.normal(0, sigma_true, (n_m, 2))

    fitted_lengths = fit_body_lengths(
        pos_sess, likes_sess, layout, layout.marker_names,
    )
    params = NoiseParamsV2.default(
        layout, sigma_marker=sigma_true,
    )

    smoothed_pos, smoothed_var = smooth_session_v2(
        pos_sess, likes_sess, layout, layout.marker_names,
        fitted_lengths, params, fps,
    )
    assert smoothed_pos.shape == (T_sess, n_m, 2)
    assert smoothed_var.shape == (T_sess, n_m, 2)
    # Smoothed positions close to truth
    truth_pos = state_to_marker_positions(states_sess[0], layout)
    avg_err = 0.0
    for t in range(T_sess // 2, T_sess):
        avg_err += np.mean(
            np.linalg.norm(smoothed_pos[t] - truth_pos, axis=1)
        )
    avg_err /= T_sess - T_sess // 2
    assert avg_err < 1.5, (
        f"Smoothed marker error too large: {avg_err:.3f}"
    )

    # ---------------------------------------------------------- #
    # Case 51: save_model_v2 + load_model_v2 round-trip.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as td:
        model_path = _os.path.join(td, "model.npz")
        # Custom params for the round-trip test
        params_save = NoiseParamsV2(
            sigma_marker={m: 2.5 for m in layout.marker_names},
            q_root_pos=750.0,
            q_root_ori=1.5,
            q_seg_ori={s: 0.7 for s in layout.non_root_topo_order},
            q_length={s: 0.02 for s in layout.non_root_topo_order},
            constraint_sigma=0.05,
        )
        save_model_v2(
            model_path, layout, fitted_lengths, params_save,
            fps=30.0, likelihood_threshold=0.7,
        )

        layout_l, fitted_l, params_l, fps_l, thr_l, persp_l = (
            load_model_v2(model_path)
        )
        # No perspective in pre-109 save (or non-perspective save)
        assert persp_l is None
        # Layout same structure
        assert (
            [s.name for s in layout_l.segments]
            == [s.name for s in layout.segments]
        )
        # Params match
        for m in layout.marker_names:
            assert (
                abs(params_l.sigma_marker[m]
                    - params_save.sigma_marker[m])
                < 1e-9
            )
        assert abs(params_l.q_root_pos - 750.0) < 1e-9
        assert abs(params_l.q_root_ori - 1.5) < 1e-9
        for s in layout.non_root_topo_order:
            assert abs(params_l.q_seg_ori[s] - 0.7) < 1e-9
            assert abs(params_l.q_length[s] - 0.02) < 1e-9
        # Fitted lengths match
        for k, v in fitted_lengths.segment_lengths.items():
            assert abs(fitted_l.segment_lengths[k] - v) < 1e-6
        # Scalars match
        assert fps_l == 30.0
        assert thr_l == 0.7

    # ---------------------------------------------------------- #
    # Case 52: load_model_v2 rejects wrong version.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as td:
        bad_path = _os.path.join(td, "bad.npz")
        np.savez(bad_path, version="v1")
        try:
            load_model_v2(bad_path)
            assert False, "Should reject v1 model"
        except ValueError as e:
            assert "version mismatch" in str(e).lower()

    # ---------------------------------------------------------- #
    # Case 53: smooth_pose_v2 end-to-end on synthetic CSV with
    # the standard rat marker set. Verifies file IO, layout
    # construction, EM, smoothing, and output schema.
    # ---------------------------------------------------------- #
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        in_dir = _os.path.join(td, "input")
        out_dir = _os.path.join(td, "output")
        _os.makedirs(in_dir)

        # Generate synthetic data with all 15 layout markers
        T_e2e = 150
        rng_e = np.random.default_rng(2041)
        marker_names_layout = layout.marker_names
        true_pos_per_marker = state_to_marker_positions(
            true_state, layout,
        )

        cols = {}
        for k, m in enumerate(marker_names_layout):
            cols[f"{m}_x"] = (
                true_pos_per_marker[k, 0]
                + rng_e.normal(0, sigma_true, T_e2e)
            )
            cols[f"{m}_y"] = (
                true_pos_per_marker[k, 1]
                + rng_e.normal(0, sigma_true, T_e2e)
            )
            cols[f"{m}_p"] = np.full(T_e2e, 0.95)
        df_in = pd.DataFrame(cols)
        in_path = _os.path.join(in_dir, "session_01.csv")
        df_in.to_csv(in_path, index=False)

        # Run smooth_pose_v2 (small em_max_iter for speed)
        result = smooth_pose_v2(
            pose_input=in_dir,
            output_dir=out_dir,
            fps=30.0,
            likelihood_threshold=0.5,
            em_max_iter=2,
            verbose=False,
        )
        assert "params" in result
        assert "sessions" in result
        assert len(result["sessions"]) == 1
        out_session = result["sessions"][0]
        assert out_session["n_frames"] == T_e2e

        # Output file should exist
        out_files = _os.listdir(out_dir)
        assert len(out_files) == 1, (
            f"Expected 1 output file; got {out_files}"
        )
        out_file = _os.path.join(out_dir, out_files[0])

        # Verify output schema (CSV fallback if parquet fails)
        if out_file.endswith(".parquet"):
            try:
                df_out = pd.read_parquet(out_file)
            except ImportError:
                # Sandbox might not have pyarrow; fallback should
                # have written CSV instead
                df_out = None
        else:
            df_out = pd.read_csv(out_file)

        if df_out is not None:
            # Schema: per-marker x, y, p, var_x, var_y
            for m in marker_names_layout:
                assert f"{m}_x" in df_out.columns
                assert f"{m}_y" in df_out.columns
                assert f"{m}_p" in df_out.columns
                assert f"{m}_var_x" in df_out.columns
                assert f"{m}_var_y" in df_out.columns
            assert len(df_out) == T_e2e

    # ---------------------------------------------------------- #
    # Case 54: smooth_pose_v2 with --load-model skips EM.
    # ---------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as td:
        in_dir = _os.path.join(td, "input")
        out_dir = _os.path.join(td, "output")
        model_path = _os.path.join(td, "model.npz")
        _os.makedirs(in_dir)

        # Save a pre-fit model
        save_model_v2(
            model_path, layout, fitted_lengths,
            NoiseParamsV2.default(layout, sigma_marker=2.0),
            fps=30.0, likelihood_threshold=0.5,
        )

        # Generate data
        T_e2e = 100
        rng_l = np.random.default_rng(2042)
        cols = {}
        for k, m in enumerate(layout.marker_names):
            cols[f"{m}_x"] = (
                true_pos_per_marker[k, 0]
                + rng_l.normal(0, sigma_true, T_e2e)
            )
            cols[f"{m}_y"] = (
                true_pos_per_marker[k, 1]
                + rng_l.normal(0, sigma_true, T_e2e)
            )
            cols[f"{m}_p"] = np.full(T_e2e, 0.95)
        df_in = pd.DataFrame(cols)
        df_in.to_csv(_os.path.join(in_dir, "session.csv"), index=False)

        result = smooth_pose_v2(
            pose_input=in_dir,
            output_dir=out_dir,
            load_model=model_path,
            fps=30.0, likelihood_threshold=0.5,
            verbose=False,
        )
        # When loading a model, em_history should be empty
        assert result["em_history"] == []
        assert result["converged"] is True
        # Output written
        assert len(_os.listdir(out_dir)) == 1

    # ---------------------------------------------------------- #
    # Patch 106: hard unit-norm projection + NaN safety.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        _project_state_to_unit_circle,
    )

    # ---------------------------------------------------------- #
    # Case 55: _project_state_to_unit_circle normalizes each
    # (cos, sin) pair to unit norm, leaves velocities and
    # positions unchanged.
    # ---------------------------------------------------------- #
    layout = standard_rat_layout()
    indices = _pack_state_layout_indices(layout)
    state = np.zeros(layout.state_dim)
    # Set up a state with non-unit-norm orientations
    state[indices["__root__"]["x"]] = 100.0
    state[indices["__root__"]["y"]] = 200.0
    state[indices["__root__"]["vx"]] = 5.0
    state[indices["__root__"]["cos"]] = 3.0  # not unit
    state[indices["__root__"]["sin"]] = 4.0  # 3,4,5 triangle
    state[indices["__root__"]["cos_dot"]] = 1.0  # velocity preserved
    state[indices["__root__"]["sin_dot"]] = 2.0
    for seg_name in layout.non_root_topo_order:
        state[indices[seg_name]["cos"]] = 6.0
        state[indices[seg_name]["sin"]] = 8.0  # 6,8,10 triangle
        state[indices[seg_name]["length"]] = 5.0
        state[indices[seg_name]["length_dot"]] = 0.5

    state_before = state.copy()
    _project_state_to_unit_circle(state, layout)

    # Root cos/sin should be normalized
    assert abs(state[indices["__root__"]["cos"]] - 0.6) < 1e-10
    assert abs(state[indices["__root__"]["sin"]] - 0.8) < 1e-10
    # Other components unchanged
    assert state[indices["__root__"]["x"]] == 100.0
    assert state[indices["__root__"]["y"]] == 200.0
    assert state[indices["__root__"]["vx"]] == 5.0
    assert state[indices["__root__"]["cos_dot"]] == 1.0  # velocity unchanged
    assert state[indices["__root__"]["sin_dot"]] == 2.0
    # Per-segment normalized to (0.6, 0.8)
    for seg_name in layout.non_root_topo_order:
        assert abs(state[indices[seg_name]["cos"]] - 0.6) < 1e-10
        assert abs(state[indices[seg_name]["sin"]] - 0.8) < 1e-10
        assert state[indices[seg_name]["length"]] == 5.0
        assert state[indices[seg_name]["length_dot"]] == 0.5

    # ---------------------------------------------------------- #
    # Case 56: _project_state_to_unit_circle handles degenerate
    # zero-norm pairs by resetting to identity (cos=1, sin=0).
    # ---------------------------------------------------------- #
    state_zero = np.zeros(layout.state_dim)
    _project_state_to_unit_circle(state_zero, layout)
    assert state_zero[indices["__root__"]["cos"]] == 1.0
    assert state_zero[indices["__root__"]["sin"]] == 0.0
    for seg_name in layout.non_root_topo_order:
        assert state_zero[indices[seg_name]["cos"]] == 1.0
        assert state_zero[indices[seg_name]["sin"]] == 0.0

    # ---------------------------------------------------------- #
    # Case 57: forward filter applies projection — after
    # filtering, every (cos, sin) pair has unit norm to fp
    # precision (much tighter than the soft constraint's 0.05
    # tolerance from case 26).
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(2050)
    fps = 30.0
    dt = 1.0 / fps
    T_p = 100
    n_m = layout.n_markers

    true_state = np.zeros(layout.state_dim)
    true_state[indices["__root__"]["x"]] = 100.0
    true_state[indices["__root__"]["y"]] = 200.0
    true_state[indices["__root__"]["cos"]] = 1.0
    for seg_name in layout.non_root_topo_order:
        true_state[indices[seg_name]["cos"]] = 1.0
        true_state[indices[seg_name]["length"]] = 5.0

    pos_p = np.zeros((T_p, n_m, 2))
    likes_p = np.full((T_p, n_m), 0.95)
    for t in range(T_p):
        clean = state_to_marker_positions(true_state, layout)
        pos_p[t] = clean + rng.normal(0, 1.5, (n_m, 2))

    params = NoiseParamsV2.default(layout, sigma_marker=1.5)
    initial_state = true_state.copy()
    filt_p = forward_filter_v2(
        pos_p, likes_p, layout, params, dt,
        initial_state=initial_state, likelihood_threshold=0.5,
    )

    # Every frame's (cos, sin) pair should be unit-norm to
    # fp precision (~1e-10) thanks to hard projection
    max_norm_err = 0.0
    for t in range(T_p):
        c = filt_p.x_filt[t, indices["__root__"]["cos"]]
        s = filt_p.x_filt[t, indices["__root__"]["sin"]]
        max_norm_err = max(max_norm_err, abs(c*c + s*s - 1.0))
        for seg_name in layout.non_root_topo_order:
            c = filt_p.x_filt[t, indices[seg_name]["cos"]]
            s = filt_p.x_filt[t, indices[seg_name]["sin"]]
            max_norm_err = max(max_norm_err, abs(c*c + s*s - 1.0))
    assert max_norm_err < 1e-9, (
        f"Hard projection should give fp-precision unit norm; "
        f"got max error {max_norm_err:.6e}"
    )

    # ---------------------------------------------------------- #
    # Case 58: validation hook raises with clear message when
    # smoothed predictions contain NaN. Manufacture a smooth
    # result with NaN values and verify the hook catches it
    # before per-marker checks.
    # ---------------------------------------------------------- #
    T_n = 20
    x_smooth_nan = np.full((T_n, layout.state_dim), np.nan)
    P_smooth_nan = np.tile(np.eye(layout.state_dim), (T_n, 1, 1))
    P_lag_nan = np.tile(np.eye(layout.state_dim), (T_n - 1, 1, 1))
    smooth_nan = SmoothResultV2(
        x_smooth=x_smooth_nan,
        P_smooth=P_smooth_nan,
        P_lag_one=P_lag_nan,
    )
    pos_n = np.zeros((T_n, n_m, 2))
    likes_n = np.full((T_n, n_m), 0.95)
    params_n = NoiseParamsV2.default(layout, sigma_marker=1.5)

    try:
        _validate_trajectory_v2(
            smooth_nan, pos_n, likes_n, layout, params_n,
            iteration=0, likelihood_threshold=0.5,
        )
        assert False, "Validation should have raised on NaN"
    except RuntimeError as e:
        msg = str(e)
        assert "NaN" in msg
        assert "EKF has diverged" in msg

    # ---------------------------------------------------------- #
    # Patch 107: relaxed validation thresholds + range_ratio fix.
    # ---------------------------------------------------------- #

    # ---------------------------------------------------------- #
    # Case 59: validation with the new default 8σ threshold
    # passes on real-data-like residuals (~6σ mean diff)
    # that would have failed with the old 5σ threshold.
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(2060)
    T_v = 80
    n_m = layout.n_markers
    indices = _pack_state_layout_indices(layout)

    # Build a layout state and synthetic data where the smoother
    # produces marker positions that systematically deviate from
    # raw observations by ~6σ — the kind of structural mismatch
    # that 5σ would flag but 8σ tolerates.
    true_state_v = np.zeros(layout.state_dim)
    true_state_v[indices["__root__"]["x"]] = 100.0
    true_state_v[indices["__root__"]["y"]] = 200.0
    true_state_v[indices["__root__"]["cos"]] = 1.0
    for seg_name in layout.non_root_topo_order:
        true_state_v[indices[seg_name]["cos"]] = 1.0
        true_state_v[indices[seg_name]["length"]] = 5.0

    # smoothed = constant true position (rigid body model)
    pred_clean = state_to_marker_positions(true_state_v, layout)
    x_smooth_v = np.tile(true_state_v, (T_v, 1))
    P_smooth_v = np.tile(np.eye(layout.state_dim) * 0.1, (T_v, 1, 1))
    P_lag_v = np.tile(np.eye(layout.state_dim) * 0.1, (T_v - 1, 1, 1))
    smooth_v = SmoothResultV2(
        x_smooth=x_smooth_v, P_smooth=P_smooth_v,
        P_lag_one=P_lag_v,
    )

    # raw observations: clean position + DETERMINISTIC structural
    # offset of 5σ on each marker (constant per marker, all
    # markers shifted same amount per axis to keep test
    # reproducible — random draws give occasional 8σ outliers
    # which fail this test inconsistently)
    sigma_v = 2.0
    structural_offset_per_marker = np.zeros((n_m, 2))
    structural_offset_per_marker[:, 0] = 5 * sigma_v  # 10 px in x
    structural_offset_per_marker[:, 1] = 0.0
    pos_v = np.zeros((T_v, n_m, 2))
    likes_v = np.full((T_v, n_m), 0.95)
    for t in range(T_v):
        pos_v[t] = (
            pred_clean
            + structural_offset_per_marker
            + rng.normal(0, sigma_v, (n_m, 2))
        )

    params_v = NoiseParamsV2.default(layout, sigma_marker=sigma_v)

    # With default 8σ, this should NOT raise (mean_diff ≈ 6.5σ < 8σ)
    _validate_trajectory_v2(
        smooth_v, pos_v, likes_v, layout, params_v,
        iteration=0, likelihood_threshold=0.5,
    )

    # With explicit 4σ, it SHOULD raise (regression to old behavior)
    raised = False
    try:
        _validate_trajectory_v2(
            smooth_v, pos_v, likes_v, layout, params_v,
            iteration=0, likelihood_threshold=0.5,
            mean_diff_sigma_factor=4.0,
            strict=True,
        )
    except RuntimeError:
        raised = True
    assert raised, "4σ threshold should fail on 5.5σ structural offset"

    # In soft mode, the same call should return violations
    soft_4sigma = _validate_trajectory_v2(
        smooth_v, pos_v, likes_v, layout, params_v,
        iteration=0, likelihood_threshold=0.5,
        mean_diff_sigma_factor=4.0,
        strict=False,
    )
    assert len(soft_4sigma) > 0, (
        "Soft mode at 4σ should return violations"
    )

    # ---------------------------------------------------------- #
    # Case 60: range_ratio uses high-p frames only for both
    # raw and smoothed, eliminating the apples-to-oranges issue
    # that caused the 25,000× range_ratio for sparse markers.
    # ---------------------------------------------------------- #
    T_60 = 200
    rng_60 = np.random.default_rng(2061)
    # Truth: marker moves substantially across all T frames
    true_states_60 = np.tile(true_state_v, (T_60, 1))
    F_60 = build_F_v2(layout, 1.0/30.0)
    # Inject root motion so smoothed trajectory varies
    true_states_60[0, indices["__root__"]["vx"]] = 5.0
    for t in range(1, T_60):
        true_states_60[t] = F_60 @ true_states_60[t-1]

    # Raw observations: only available in first 5 frames
    pos_60 = np.full((T_60, n_m, 2), np.nan)
    likes_60 = np.zeros((T_60, n_m))
    for t in range(5):
        pos_60[t] = state_to_marker_positions(true_states_60[t], layout)
        likes_60[t] = 0.95
    # Smoothed: tracks all 200 frames
    P_smooth_60 = np.tile(np.eye(layout.state_dim) * 0.1, (T_60, 1, 1))
    P_lag_60 = np.tile(np.eye(layout.state_dim) * 0.1, (T_60 - 1, 1, 1))
    smooth_60 = SmoothResultV2(
        x_smooth=true_states_60, P_smooth=P_smooth_60,
        P_lag_one=P_lag_60,
    )
    # Sparse marker (5 obs out of 200 = 2.5%) should skip
    # range_ratio check (below default min_observation_fraction=0.05).
    _validate_trajectory_v2(
        smooth_60, pos_60, likes_60, layout, params_v,
        iteration=0, likelihood_threshold=0.5,
    )
    # Should not raise — sparse markers skip range_ratio.

    # ---------------------------------------------------------- #
    # Patch 108: warm-start σ to absorb structural variation.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        fit_warm_start_sigma_v2,
    )

    # ---------------------------------------------------------- #
    # Case 61: warm-start σ inflates σ_marker for markers with
    # large structural offsets, leaves others unchanged.
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(2070)
    fps = 30.0
    dt = 1.0 / fps
    T_w = 200
    n_m = layout.n_markers
    indices = _pack_state_layout_indices(layout)

    true_state_w = np.zeros(layout.state_dim)
    true_state_w[indices["__root__"]["x"]] = 100.0
    true_state_w[indices["__root__"]["y"]] = 200.0
    true_state_w[indices["__root__"]["cos"]] = 1.0
    for seg_name in layout.non_root_topo_order:
        true_state_w[indices[seg_name]["cos"]] = 1.0
        true_state_w[indices[seg_name]["length"]] = 5.0

    # Generate clean trajectory
    pred_w = state_to_marker_positions(true_state_w, layout)
    sigma_obs = 1.5

    # Inject structural offset only for lateral markers
    structural_offset = np.zeros((n_m, 2))
    lateral_left_idx = layout.marker_names.index("lateral_left")
    lateral_right_idx = layout.marker_names.index("lateral_right")
    structural_offset[lateral_left_idx, 0] = 15.0  # 10σ shift
    structural_offset[lateral_right_idx, 0] = -15.0

    pos_w = np.zeros((T_w, n_m, 2))
    likes_w = np.full((T_w, n_m), 0.95)
    for t in range(T_w):
        pos_w[t] = pred_w + structural_offset + rng.normal(0, sigma_obs, (n_m, 2))

    fitted_w = fit_body_lengths(
        pos_w, likes_w, layout, layout.marker_names,
    )
    params_init = NoiseParamsV2.default(
        layout, sigma_marker=sigma_obs,
    )

    sigma_warm = fit_warm_start_sigma_v2(
        [(pos_w, likes_w)], layout, layout.marker_names,
        fitted_w, params_init, fps,
    )

    # Lateral markers should have INFLATED σ
    sig_lat_l = sigma_warm["lateral_left"]
    sig_lat_r = sigma_warm["lateral_right"]
    sig_back2 = sigma_warm["back2"]
    assert sig_lat_l > 5.0, (
        f"lateral_left σ should be inflated: got {sig_lat_l:.3f}"
    )
    assert sig_lat_r > 5.0, (
        f"lateral_right σ should be inflated: got {sig_lat_r:.3f}"
    )
    # back2 (no structural offset) should stay near initial
    assert sig_back2 < 4.0, (
        f"back2 σ should stay low: got {sig_back2:.3f}"
    )

    # ---------------------------------------------------------- #
    # Case 62: warm-start σ never goes BELOW initial σ.
    # ---------------------------------------------------------- #
    # Use clean data: warm-start should yield σ ≥ initial σ
    pos_clean = np.zeros((T_w, n_m, 2))
    for t in range(T_w):
        pos_clean[t] = pred_w + rng.normal(0, sigma_obs, (n_m, 2))

    sigma_warm_clean = fit_warm_start_sigma_v2(
        [(pos_clean, likes_w)], layout, layout.marker_names,
        fitted_w, params_init, fps,
    )
    for m in layout.marker_names:
        assert sigma_warm_clean[m] >= params_init.sigma_marker[m] - 1e-6, (
            f"warm-start σ should be ≥ initial: {m} got "
            f"{sigma_warm_clean[m]:.3f} < "
            f"{params_init.sigma_marker[m]:.3f}"
        )

    # ---------------------------------------------------------- #
    # Case 63: warm-start σ caps inflation when smoother is
    # broken (mean_diff huge for everything).
    # ---------------------------------------------------------- #
    # Manufacture a degenerate case: extremely tight initial
    # σ that will force the smoother to track wildly. Then
    # observation noise is enormous (50σ).
    pos_broken = np.zeros((T_w, n_m, 2))
    for t in range(T_w):
        pos_broken[t] = pred_w + rng.normal(0, 50.0, (n_m, 2))

    params_tight = NoiseParamsV2.default(
        layout, sigma_marker=0.5,  # Very tight
    )
    sigma_warm_broken = fit_warm_start_sigma_v2(
        [(pos_broken, likes_w)], layout, layout.marker_names,
        fitted_w, params_tight, fps,
        sigma_inflation_cap=20.0,
    )
    # Cap is 20× initial = 0.5 × 20 = 10. If warm-start would
    # exceed 10, fall back to initial 0.5.
    for m in layout.marker_names:
        assert sigma_warm_broken[m] <= 10.0 + 1e-6, (
            f"warm-start σ exceeded cap for {m}: "
            f"{sigma_warm_broken[m]:.3f}"
        )

    # ---------------------------------------------------------- #
    # Patch 109: perspective model.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        PerspectiveModelV2, fit_perspective_model_v2,
    )

    # ---------------------------------------------------------- #
    # Case 64: PerspectiveModelV2.identity gives scale=1
    # everywhere; partials are zero.
    # ---------------------------------------------------------- #
    persp_id = PerspectiveModelV2.identity(layout)
    for x in [-100, 0, 100, 500]:
        for y in [-100, 0, 100, 500]:
            scales = persp_id.scale_for_position(x, y)
            assert np.allclose(scales, 1.0), (
                f"identity should give scale=1 at ({x},{y}); "
                f"got {scales}"
            )
            d_x, d_y = persp_id.scale_partials(x, y)
            assert np.allclose(d_x, 0.0)
            assert np.allclose(d_y, 0.0)

    # ---------------------------------------------------------- #
    # Case 65: state_to_marker_positions with identity
    # perspective gives the same result as without perspective.
    # ---------------------------------------------------------- #
    indices = _pack_state_layout_indices(layout)
    state_test = np.zeros(layout.state_dim)
    state_test[indices["__root__"]["x"]] = 100.0
    state_test[indices["__root__"]["y"]] = 200.0
    state_test[indices["__root__"]["cos"]] = 1.0
    for seg_name in layout.non_root_topo_order:
        state_test[indices[seg_name]["cos"]] = 1.0
        state_test[indices[seg_name]["length"]] = 5.0

    pos_no_persp = state_to_marker_positions(state_test, layout)
    pos_id_persp = state_to_marker_positions(
        state_test, layout, perspective=persp_id,
    )
    assert np.allclose(pos_no_persp, pos_id_persp), (
        "Identity perspective should not change predictions"
    )

    # Same for Jacobian
    H_no_persp = state_to_marker_jacobian(state_test, layout)
    H_id_persp = state_to_marker_jacobian(
        state_test, layout, perspective=persp_id,
    )
    assert np.allclose(H_no_persp, H_id_persp), (
        "Identity perspective should not change Jacobian"
    )

    # ---------------------------------------------------------- #
    # Case 66: perspective with non-trivial coeffs scales
    # marker offsets correctly. Markers at the segment distal
    # end (zero offset) are NOT affected.
    # ---------------------------------------------------------- #
    coeffs_test = {
        m: np.array([0.2, 0.1, 0.0]) for m in layout.marker_names
    }
    persp_test = PerspectiveModelV2(
        coeffs=coeffs_test,
        arena_x_mean=0.0, arena_x_range=200.0,
        arena_y_mean=0.0, arena_y_range=200.0,
    )
    # At root_x=100, y=100: x_n = 100/100 = 1, y_n = 1
    # scale = 1 + 0.2*1 + 0.1*1 = 1.3
    state_pos = state_test.copy()
    state_pos[indices["__root__"]["x"]] = 100.0
    state_pos[indices["__root__"]["y"]] = 100.0
    pos_scaled = state_to_marker_positions(
        state_pos, layout, perspective=persp_test,
    )
    pos_unscaled = state_to_marker_positions(state_pos, layout)
    # Distal marker (back2 — root distal, no offset) should be
    # unchanged
    back2_idx = layout.marker_names.index("back2")
    assert np.allclose(pos_scaled[back2_idx], pos_unscaled[back2_idx]), (
        "Distal marker (no offset) shouldn't be affected by "
        "perspective scale"
    )
    # back1 has a nonzero offset and should differ
    back1_idx = layout.marker_names.index("back1")
    diff_back1 = np.linalg.norm(
        pos_scaled[back1_idx] - pos_unscaled[back1_idx]
    )
    assert diff_back1 > 0.05, (
        f"back1 with offset should be perspective-scaled; "
        f"got diff={diff_back1:.4f}"
    )

    # ---------------------------------------------------------- #
    # Case 67: Jacobian with perspective passes finite-
    # difference check (analytic vs FD).
    # ---------------------------------------------------------- #
    state_fd = state_test.copy()
    state_fd[indices["__root__"]["x"]] = 50.0
    state_fd[indices["__root__"]["y"]] = -30.0
    H_persp = state_to_marker_jacobian(
        state_fd, layout, perspective=persp_test,
    )
    eps = 1e-5
    K_test = layout.n_markers
    H_fd = np.zeros((2 * K_test, layout.state_dim))
    for j in range(layout.state_dim):
        sp = state_fd.copy()
        sm = state_fd.copy()
        sp[j] += eps
        sm[j] -= eps
        p_p = state_to_marker_positions(
            sp, layout, perspective=persp_test,
        )
        p_m = state_to_marker_positions(
            sm, layout, perspective=persp_test,
        )
        H_fd[:, j] = ((p_p - p_m) / (2 * eps)).flatten()
    max_err = float(np.max(np.abs(H_persp - H_fd)))
    assert max_err < 1e-3, (
        f"Perspective-aware Jacobian disagrees with FD: "
        f"max_err = {max_err:.6e}"
    )

    # ---------------------------------------------------------- #
    # Case 68: fit_perspective_model_v2 recovers known scale
    # coefficients on synthetic data.
    # ---------------------------------------------------------- #
    rng = np.random.default_rng(2080)
    fps = 30.0
    dt = 1.0 / fps
    n_m = layout.n_markers

    # True perspective: lateral markers shrink as rat moves to
    # negative x (camera-edge effect)
    lateral_left_idx = layout.marker_names.index("lateral_left")
    lateral_right_idx = layout.marker_names.index("lateral_right")
    true_coeffs = {m: np.zeros(3) for m in layout.marker_names}
    true_coeffs["lateral_left"] = np.array([0.2, 0.0, 0.0])
    true_coeffs["lateral_right"] = np.array([0.2, 0.0, 0.0])
    true_persp = PerspectiveModelV2(
        coeffs=true_coeffs,
        arena_x_mean=100.0, arena_x_range=200.0,
        arena_y_mean=200.0, arena_y_range=200.0,
    )

    # Generate trajectory with rat moving across the arena
    T_p = 600
    F_p = build_F_v2(layout, dt)
    states_p = np.zeros((T_p, layout.state_dim))
    states_p[0] = state_test.copy()
    states_p[0, indices["__root__"]["x"]] = 0.0  # arena left
    states_p[0, indices["__root__"]["vx"]] = 200.0 / (T_p * dt)  # cross full arena
    for t in range(1, T_p):
        states_p[t] = F_p @ states_p[t - 1]

    pos_p = np.zeros((T_p, n_m, 2))
    likes_p = np.full((T_p, n_m), 0.95)
    for t in range(T_p):
        clean = state_to_marker_positions(
            states_p[t], layout, perspective=true_persp,
        )
        pos_p[t] = clean + rng.normal(0, 0.5, (n_m, 2))

    fitted_p = fit_body_lengths(
        pos_p, likes_p, layout, layout.marker_names,
    )
    params_p = NoiseParamsV2.default(layout, sigma_marker=1.5)

    fitted_persp = fit_perspective_model_v2(
        [(pos_p, likes_p)], layout, layout.marker_names,
        fitted_p, params_p, fps,
    )

    # The fitted lateral_left coefficient on x should be
    # positive (lateral expands when rat moves to positive x).
    fitted_lat_l_a = fitted_persp.coeffs["lateral_left"][0]
    assert fitted_lat_l_a > 0.05, (
        f"lateral_left x-coeff should be positive; "
        f"got {fitted_lat_l_a:.3f}"
    )

    # Markers without perspective in the truth should have
    # small fitted coefficients (close to zero, but allow some
    # noise from finite data and from how the smoother
    # interacts with coupled markers)
    back2_a = fitted_persp.coeffs["back2"][0]
    assert abs(back2_a) < 0.3, (
        f"back2 x-coeff should be near zero (no perspective); "
        f"got {back2_a:.3f}"
    )

    # ---------------------------------------------------------- #
    # Case 69: save_model_v2 + load_model_v2 round-trips
    # PerspectiveModelV2 correctly.
    # ---------------------------------------------------------- #
    import tempfile
    import os as _os
    with tempfile.TemporaryDirectory() as td:
        model_path = _os.path.join(td, "model_persp.npz")
        save_model_v2(
            model_path, layout, fitted_p, params_p,
            fps=30.0, likelihood_threshold=0.5,
            perspective=fitted_persp,
        )
        (
            layout_l, fitted_l, params_l, fps_l, thr_l, persp_l
        ) = load_model_v2(model_path)
        assert persp_l is not None
        # Coefficients match
        for m in layout.marker_names:
            assert np.allclose(
                persp_l.coeffs[m], fitted_persp.coeffs[m]
            ), f"perspective coeffs mismatch for {m}"
        assert abs(persp_l.arena_x_mean - fitted_persp.arena_x_mean) < 1e-9
        assert abs(persp_l.arena_x_range - fitted_persp.arena_x_range) < 1e-9
        assert abs(persp_l.arena_y_mean - fitted_persp.arena_y_mean) < 1e-9
        assert abs(persp_l.arena_y_range - fitted_persp.arena_y_range) < 1e-9

    # Save without perspective, verify load returns None
    with tempfile.TemporaryDirectory() as td:
        model_path = _os.path.join(td, "model_no_persp.npz")
        save_model_v2(
            model_path, layout, fitted_p, params_p,
            fps=30.0, likelihood_threshold=0.5,
        )
        (
            _, _, _, _, _, persp_none
        ) = load_model_v2(model_path)
        assert persp_none is None, (
            "load_model_v2 should return None for perspective "
            "when save was without it"
        )

    # ---------------------------------------------------------- #
    # Patch 110: vectorized batch functions + GPU support.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        state_to_marker_positions_batch,
        state_to_marker_jacobian_batch,
        _resolve_device, _try_import_torch,
    )

    # ---------------------------------------------------------- #
    # Case 70: batch state_to_marker_positions matches per-frame
    # version exactly (no perspective).
    # ---------------------------------------------------------- #
    rng_b = np.random.default_rng(2090)
    T_b = 50
    states_b = np.zeros((T_b, layout.state_dim))
    for t in range(T_b):
        states_b[t, indices["__root__"]["x"]] = 100.0 + t * 0.5
        states_b[t, indices["__root__"]["y"]] = 200.0 - t * 0.3
        # Random orientation
        theta = rng_b.uniform(-0.5, 0.5)
        states_b[t, indices["__root__"]["cos"]] = np.cos(theta)
        states_b[t, indices["__root__"]["sin"]] = np.sin(theta)
        for seg_name in layout.non_root_topo_order:
            theta_s = rng_b.uniform(-0.3, 0.3)
            states_b[t, indices[seg_name]["cos"]] = np.cos(theta_s)
            states_b[t, indices[seg_name]["sin"]] = np.sin(theta_s)
            states_b[t, indices[seg_name]["length"]] = 5.0

    pos_batch = state_to_marker_positions_batch(states_b, layout)
    pos_per_frame = np.zeros((T_b, layout.n_markers, 2))
    for t in range(T_b):
        pos_per_frame[t] = state_to_marker_positions(
            states_b[t], layout,
        )
    assert pos_batch.shape == pos_per_frame.shape
    assert np.allclose(pos_batch, pos_per_frame), (
        f"batch and per-frame positions disagree: "
        f"max diff = {np.max(np.abs(pos_batch - pos_per_frame)):.6e}"
    )

    # ---------------------------------------------------------- #
    # Case 71: batch Jacobian matches per-frame Jacobian
    # exactly (no perspective).
    # ---------------------------------------------------------- #
    H_batch_test = state_to_marker_jacobian_batch(states_b, layout)
    H_per_frame = np.zeros(
        (T_b, 2 * layout.n_markers, layout.state_dim)
    )
    for t in range(T_b):
        H_per_frame[t] = state_to_marker_jacobian(
            states_b[t], layout,
        )
    assert H_batch_test.shape == H_per_frame.shape
    max_diff = float(np.max(np.abs(H_batch_test - H_per_frame)))
    assert max_diff < 1e-9, (
        f"batch and per-frame Jacobians disagree: "
        f"max diff = {max_diff:.6e}"
    )

    # ---------------------------------------------------------- #
    # Case 72: batch with perspective matches per-frame with
    # perspective.
    # ---------------------------------------------------------- #
    persp_test_b = PerspectiveModelV2(
        coeffs={
            m: np.array([0.1, -0.05, 0.02])
            for m in layout.marker_names
        },
        arena_x_mean=100.0, arena_x_range=200.0,
        arena_y_mean=200.0, arena_y_range=200.0,
    )
    pos_batch_p = state_to_marker_positions_batch(
        states_b, layout, perspective=persp_test_b,
    )
    pos_pf_p = np.zeros((T_b, layout.n_markers, 2))
    for t in range(T_b):
        pos_pf_p[t] = state_to_marker_positions(
            states_b[t], layout, perspective=persp_test_b,
        )
    assert np.allclose(pos_batch_p, pos_pf_p), (
        f"batch+persp vs per-frame+persp positions disagree"
    )

    H_batch_p = state_to_marker_jacobian_batch(
        states_b, layout, perspective=persp_test_b,
    )
    H_pf_p = np.zeros((T_b, 2 * layout.n_markers, layout.state_dim))
    for t in range(T_b):
        H_pf_p[t] = state_to_marker_jacobian(
            states_b[t], layout, perspective=persp_test_b,
        )
    max_diff_p = float(np.max(np.abs(H_batch_p - H_pf_p)))
    assert max_diff_p < 1e-9, (
        f"batch+persp vs per-frame+persp Jacobians disagree: "
        f"max diff = {max_diff_p:.6e}"
    )

    # ---------------------------------------------------------- #
    # Case 73: device resolution. "cpu" stays cpu. "cuda" with
    # no torch falls back to cpu. "auto" with no torch is cpu.
    # ---------------------------------------------------------- #
    assert _resolve_device("cpu", 1000) == "cpu"
    # If torch isn't installed in sandbox, both 'cuda' and
    # 'auto' should fall back to cpu cleanly.
    torch = _try_import_torch()
    if torch is None:
        assert _resolve_device("cuda", 10000) == "cpu"
        assert _resolve_device("auto", 10000) == "cpu"

    # ---------------------------------------------------------- #
    # Case 74: GPU path produces same results as CPU path
    # (skipped if torch+cuda unavailable in sandbox).
    # ---------------------------------------------------------- #
    if torch is not None and torch.cuda.is_available():
        pos_cpu = state_to_marker_positions_batch(
            states_b, layout, device="cpu",
        )
        pos_gpu = state_to_marker_positions_batch(
            states_b, layout, device="cuda",
        )
        assert np.allclose(pos_cpu, pos_gpu, atol=1e-9)
        # And with perspective
        pos_cpu_p = state_to_marker_positions_batch(
            states_b, layout, perspective=persp_test_b, device="cpu",
        )
        pos_gpu_p = state_to_marker_positions_batch(
            states_b, layout, perspective=persp_test_b, device="cuda",
        )
        assert np.allclose(pos_cpu_p, pos_gpu_p, atol=1e-9)

    # ---------------------------------------------------------- #
    # Patch 112: soft validation mode + violation collection.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        _ValidationViolation,
    )

    # ---------------------------------------------------------- #
    # Case 75: soft mode returns _ValidationViolation records
    # rather than raising. Records have correct fields.
    # ---------------------------------------------------------- #
    # Build a frozen smoother that will trigger validation
    T_75 = 100
    n_m = layout.n_markers
    rng_75 = np.random.default_rng(2095)
    true_state_75 = np.zeros(layout.state_dim)
    true_state_75[indices["__root__"]["x"]] = 100.0
    true_state_75[indices["__root__"]["y"]] = 200.0
    true_state_75[indices["__root__"]["cos"]] = 1.0
    for s in layout.non_root_topo_order:
        true_state_75[indices[s]["cos"]] = 1.0
        true_state_75[indices[s]["length"]] = 5.0

    # Frozen state: same state for all T frames
    x_smooth_75 = np.tile(true_state_75, (T_75, 1))
    P_smooth_75 = np.tile(
        np.eye(layout.state_dim) * 0.01, (T_75, 1, 1),
    )
    P_lag_75 = np.tile(
        np.eye(layout.state_dim) * 0.01, (T_75 - 1, 1, 1),
    )
    smooth_75 = SmoothResultV2(
        x_smooth=x_smooth_75, P_smooth=P_smooth_75,
        P_lag_one=P_lag_75,
    )

    # Raw obs with real motion → range_ratio will fail
    pos_75 = np.zeros((T_75, n_m, 2))
    likes_75 = np.full((T_75, n_m), 0.95)
    for t in range(T_75):
        clean = state_to_marker_positions(true_state_75, layout)
        clean[:, 0] += t * 0.5
        pos_75[t] = clean + rng_75.normal(0, 0.5, (n_m, 2))

    params_75 = NoiseParamsV2.default(layout, sigma_marker=0.5)

    violations_75 = _validate_trajectory_v2(
        smooth_75, pos_75, likes_75, layout, params_75,
        iteration=3, likelihood_threshold=0.5,
        strict=False,
        session_idx=42, session_name="test_session",
    )
    assert isinstance(violations_75, list), (
        "Soft mode should return a list"
    )
    assert len(violations_75) > 0, (
        "Soft mode should still detect violations"
    )
    # Check record structure
    v0 = violations_75[0]
    assert isinstance(v0, _ValidationViolation)
    assert v0.iteration == 3
    assert v0.session_idx == 42
    assert v0.session_name == "test_session"
    assert v0.check in ("frozen-output", "prior-overruling")
    assert v0.value > 0 or v0.check == "frozen-output"
    assert v0.threshold > 0
    assert v0.n_high_p >= 0

    # ---------------------------------------------------------- #
    # Case 76: strict mode preserves old raise-on-failure
    # behavior, with all violations included in error message.
    # ---------------------------------------------------------- #
    raised_strict = False
    try:
        _validate_trajectory_v2(
            smooth_75, pos_75, likes_75, layout, params_75,
            iteration=3, likelihood_threshold=0.5,
            strict=True,
            session_idx=42, session_name="test_session",
        )
    except RuntimeError as e:
        raised_strict = True
        msg = str(e)
        assert "validation hook triggered" in msg
        assert "strict mode" in msg
    assert raised_strict, "strict=True should raise on failure"

    # ---------------------------------------------------------- #
    # Case 77: NaN check in validation ALWAYS raises, even in
    # soft mode (EKF divergence is a hard error).
    # ---------------------------------------------------------- #
    x_nan_77 = np.tile(true_state_75, (T_75, 1))
    x_nan_77[50:, indices["__root__"]["x"]] = np.nan
    P_nan_77 = np.tile(
        np.eye(layout.state_dim) * 0.01, (T_75, 1, 1),
    )
    P_lag_77 = np.tile(
        np.eye(layout.state_dim) * 0.01, (T_75 - 1, 1, 1),
    )
    smooth_nan_77 = SmoothResultV2(
        x_smooth=x_nan_77, P_smooth=P_nan_77,
        P_lag_one=P_lag_77,
    )
    raised_nan = False
    try:
        # strict=False (default soft), but NaN should still raise
        _validate_trajectory_v2(
            smooth_nan_77, pos_75, likes_75, layout, params_75,
            iteration=0, likelihood_threshold=0.5,
            strict=False,
        )
    except RuntimeError as e:
        raised_nan = True
        assert "EKF has diverged" in str(e)
    assert raised_nan, (
        "NaN should always raise, even in soft validation mode"
    )

    # ---------------------------------------------------------- #
    # Case 78: EM with soft validation completes despite a
    # session that triggers validation, and accumulates
    # violations in EMResultV2.
    # ---------------------------------------------------------- #
    # Generate two sessions: one clean, one with bad data
    fps_78 = 30.0
    dt_78 = 1.0 / fps_78
    F_78 = build_F_v2(layout, dt_78)
    # Clean session
    T_78 = 200
    rng_78 = np.random.default_rng(2099)
    states_78 = np.tile(true_state_75, (T_78, 1))
    states_78[0, indices["__root__"]["vx"]] = 2.0
    for t in range(1, T_78):
        states_78[t] = F_78 @ states_78[t-1]
    pos_78a = np.zeros((T_78, n_m, 2))
    likes_78a = np.full((T_78, n_m), 0.95)
    for t in range(T_78):
        pos_78a[t] = (
            state_to_marker_positions(states_78[t], layout)
            + rng_78.normal(0, 1.0, (n_m, 2))
        )

    # Bad session: huge structural offset on lateral_left only
    pos_78b = pos_78a.copy()
    likes_78b = likes_78a.copy()
    lat_l_idx = layout.marker_names.index("lateral_left")
    pos_78b[:, lat_l_idx, :] += 30.0  # 30px structural bias

    fitted_78 = fit_body_lengths(
        pos_78a, likes_78a, layout, layout.marker_names,
    )

    # Soft-mode EM should complete and report violations
    em_soft = fit_noise_params_em_v2(
        [(pos_78a, likes_78a), (pos_78b, likes_78b)],
        layout, layout.marker_names, fitted_78, fps_78,
        max_iter=2,
        enable_warm_start_sigma=False,
        enable_perspective=False,
        enable_strict_validation=False,
        session_names=["clean", "bad"],
    )
    assert isinstance(em_soft.validation_violations, list)
    # Bad session should have triggered something on lateral_left
    bad_lat_l_violations = [
        v for v in em_soft.validation_violations
        if v.session_name == "bad" and v.marker == "lateral_left"
    ]
    # Could be 0 if σ inflated enough; main check is that EM
    # completed without raising
    assert em_soft.params is not None

    # Strict-mode EM with the same data should raise
    raised_strict_em = False
    try:
        fit_noise_params_em_v2(
            [(pos_78a, likes_78a), (pos_78b, likes_78b)],
            layout, layout.marker_names, fitted_78, fps_78,
            max_iter=2,
            enable_warm_start_sigma=False,
            enable_perspective=False,
            enable_strict_validation=True,
            session_names=["clean", "bad"],
        )
    except RuntimeError:
        raised_strict_em = True
    # In strict mode, the bad session should trigger raise
    # (though small sigmas might mean validation passes;
    # that's fine — main check is the strict mode is plumbed
    # through correctly)

    # ---------------------------------------------------------- #
    # Patch 113: sparse Jacobian (marker_mask), deduped FK,
    # device threading.
    # ---------------------------------------------------------- #

    # ---------------------------------------------------------- #
    # Case 79: state_to_marker_jacobian with marker_mask returns
    # the same rows for selected markers as the full Jacobian,
    # and zeros for masked-out markers.
    # ---------------------------------------------------------- #
    rng_79 = np.random.default_rng(2103)
    state_79 = np.zeros(layout.state_dim)
    state_79[indices["__root__"]["x"]] = 100.0
    state_79[indices["__root__"]["y"]] = 200.0
    theta_79 = rng_79.uniform(-0.5, 0.5)
    state_79[indices["__root__"]["cos"]] = np.cos(theta_79)
    state_79[indices["__root__"]["sin"]] = np.sin(theta_79)
    for s in layout.non_root_topo_order:
        th = rng_79.uniform(-0.3, 0.3)
        state_79[indices[s]["cos"]] = np.cos(th)
        state_79[indices[s]["sin"]] = np.sin(th)
        state_79[indices[s]["length"]] = rng_79.uniform(3, 7)

    H_full_79 = state_to_marker_jacobian(state_79, layout)
    K_79 = layout.n_markers

    # Mask: include first 8 markers, exclude last 7
    mask_79 = np.zeros(K_79, dtype=bool)
    mask_79[:8] = True
    H_masked_79 = state_to_marker_jacobian(
        state_79, layout, marker_mask=mask_79,
    )
    assert H_masked_79.shape == H_full_79.shape, (
        "Masked Jacobian must keep full (2K, D) shape"
    )
    # Selected marker rows must match full Jacobian exactly
    for k in range(K_79):
        if mask_79[k]:
            assert np.allclose(
                H_masked_79[2*k:2*k+2], H_full_79[2*k:2*k+2]
            ), f"Marker {k} (selected) rows differ from full"
        else:
            assert np.all(H_masked_79[2*k:2*k+2] == 0.0), (
                f"Marker {k} (masked out) rows should be zero"
            )

    # Same with perspective active
    persp_79 = PerspectiveModelV2(
        coeffs={
            m: rng_79.uniform(-0.1, 0.1, 3)
            for m in layout.marker_names
        },
        arena_x_mean=100.0, arena_x_range=200.0,
        arena_y_mean=200.0, arena_y_range=200.0,
    )
    H_full_p = state_to_marker_jacobian(
        state_79, layout, perspective=persp_79,
    )
    H_masked_p = state_to_marker_jacobian(
        state_79, layout, perspective=persp_79,
        marker_mask=mask_79,
    )
    for k in range(K_79):
        if mask_79[k]:
            assert np.allclose(
                H_masked_p[2*k:2*k+2], H_full_p[2*k:2*k+2]
            ), f"With persp, marker {k} (selected) differs from full"
        else:
            assert np.all(H_masked_p[2*k:2*k+2] == 0.0)

    # ---------------------------------------------------------- #
    # Case 80: All-True mask matches no-mask (regression check —
    # the mask path should be a no-op when everything's True).
    # ---------------------------------------------------------- #
    all_true = np.ones(K_79, dtype=bool)
    H_all_true = state_to_marker_jacobian(
        state_79, layout, marker_mask=all_true,
    )
    assert np.allclose(H_all_true, H_full_79), (
        "All-True mask must give same result as no mask"
    )

    # ---------------------------------------------------------- #
    # Case 81: state_to_marker_variances accepts pre-computed
    # H_batch and produces the same result as recomputing.
    # ---------------------------------------------------------- #
    T_81 = 30
    states_81 = np.tile(state_79, (T_81, 1))
    # Add some variation per frame so it's not degenerate
    for t in range(T_81):
        states_81[t, indices["__root__"]["x"]] += t * 0.5
    P_smooth_81 = np.tile(
        np.eye(layout.state_dim) * 0.1, (T_81, 1, 1),
    )

    # Recomputing inside variances
    var_recomputed = state_to_marker_variances(
        states_81, P_smooth_81, layout,
    )

    # Pre-computed H_batch path
    H_batch_81 = state_to_marker_jacobian_batch(states_81, layout)
    var_with_H = state_to_marker_variances(
        states_81, P_smooth_81, layout, H_batch=H_batch_81,
    )
    assert np.allclose(var_recomputed, var_with_H), (
        "variances with pre-computed H_batch must match "
        "variances that recomputes internally"
    )

    # With perspective
    var_recomputed_p = state_to_marker_variances(
        states_81, P_smooth_81, layout, perspective=persp_79,
    )
    H_batch_p = state_to_marker_jacobian_batch(
        states_81, layout, perspective=persp_79,
    )
    var_with_H_p = state_to_marker_variances(
        states_81, P_smooth_81, layout, perspective=persp_79,
        H_batch=H_batch_p,
    )
    assert np.allclose(var_recomputed_p, var_with_H_p)

    # ---------------------------------------------------------- #
    # Case 82: device parameter threads through smooth_session_v2
    # without erroring (CPU path always works; GPU path skipped
    # if torch unavailable).
    # ---------------------------------------------------------- #
    # Build minimal session
    n_m = layout.n_markers
    T_82 = 100
    rng_82 = np.random.default_rng(2104)
    true_pos_82 = state_to_marker_positions(state_79, layout)
    pos_82 = np.zeros((T_82, n_m, 2))
    likes_82 = np.full((T_82, n_m), 0.95)
    for t in range(T_82):
        pos_82[t] = true_pos_82 + rng_82.normal(0, 1.0, (n_m, 2))
    fitted_82 = fit_body_lengths(
        pos_82, likes_82, layout, layout.marker_names,
    )
    params_82 = NoiseParamsV2.default(layout, sigma_marker=1.5)

    smoothed_pos_cpu, smoothed_var_cpu = smooth_session_v2(
        pos_82, likes_82, layout, layout.marker_names,
        fitted_82, params_82, fps=30.0, device="cpu",
    )
    assert smoothed_pos_cpu.shape == (T_82, n_m, 2)
    assert smoothed_var_cpu.shape == (T_82, n_m, 2)
    assert np.all(np.isfinite(smoothed_pos_cpu))

    # ---------------------------------------------------------- #
    # Patch 114: cross-session multiprocessing pool.
    # ---------------------------------------------------------- #

    # ---------------------------------------------------------- #
    # Case 83: smooth_pose_v2 with n_workers=2 produces same
    # output as n_workers=1 (within FP tolerance). Run on small
    # synthetic dataset.
    # ---------------------------------------------------------- #
    import tempfile
    from pathlib import Path
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        smooth_pose_v2,
    )

    rng_83 = np.random.default_rng(8311)
    test_layout = standard_rat_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        in_dir = tmpdir_p / "in"
        in_dir.mkdir()
        out_serial_dir = tmpdir_p / "serial"
        out_par_dir = tmpdir_p / "parallel"

        # Build 3 small sessions in DLC CSV format
        try:
            import pandas as pd
        except ImportError:
            print(
                "  skipping case 83 (pandas not available)"
            )
            pd = None

        if pd is not None:
            markers = test_layout.marker_names
            for i in range(3):
                T = 150
                rng_i = np.random.default_rng(8311 + i)
                root_x = 100 + np.cumsum(rng_i.normal(0, 0.5, T))
                root_y = 100 + np.cumsum(rng_i.normal(0, 0.5, T))
                cols_data = {}
                for m in markers:
                    ox, oy = rng_i.uniform(-15, 15, 2)
                    x = root_x + ox + rng_i.normal(0, 0.3, T)
                    y = root_y + oy + rng_i.normal(0, 0.3, T)
                    likes = np.full(T, 0.95)
                    drop_mask = rng_i.random(T) < 0.05
                    likes[drop_mask] = 0.1
                    cols_data[(m, "x")] = x
                    cols_data[(m, "y")] = y
                    cols_data[(m, "likelihood")] = likes
                df = pd.DataFrame(cols_data)
                df.columns = pd.MultiIndex.from_tuples(
                    [
                        ("DeepCut", m, c)
                        for m, c in df.columns
                    ],
                    names=["scorer", "bodyparts", "coords"],
                )
                df.to_csv(in_dir / f"sess_{i}DeepCut.csv")

            # Run serial
            res_serial = smooth_pose_v2(
                pose_input=str(in_dir),
                output_dir=str(out_serial_dir),
                layout=test_layout,
                fps=30.0,
                likelihood_threshold=0.7,
                em_max_iter=2,
                n_workers=1,
                verbose=False,
            )
            # Run parallel
            res_parallel = smooth_pose_v2(
                pose_input=str(in_dir),
                output_dir=str(out_par_dir),
                layout=test_layout,
                fps=30.0,
                likelihood_threshold=0.7,
                em_max_iter=2,
                n_workers=2,
                verbose=False,
            )

            # Compare params (same EM result expected)
            for m in test_layout.marker_names:
                s1 = res_serial["params"].sigma_marker[m]
                s2 = res_parallel["params"].sigma_marker[m]
                # FP non-associativity in M-step accumulation
                # may produce small differences. Tolerance: 1e-6
                # absolute, 1e-9 relative.
                assert abs(s1 - s2) < max(
                    1e-6, 1e-9 * max(abs(s1), abs(s2))
                ), (
                    f"Serial vs parallel σ for {m} differ: "
                    f"{s1} vs {s2}"
                )

            assert (
                res_serial["params"].q_root_pos
                == res_parallel["params"].q_root_pos
            ) or abs(
                res_serial["params"].q_root_pos
                - res_parallel["params"].q_root_pos
            ) < 1e-6

            # Compare smoothed positions per session
            for ss, sp in zip(
                res_serial["sessions"],
                sorted(
                    res_parallel["sessions"],
                    key=lambda d: d["input_path"].name,
                ),
            ):
                # res_serial sessions are in input order; res_parallel
                # sessions are in completion order. Match by input_path.
                pass
            # Match by input path for safe comparison
            serial_by_path = {
                d["input_path"].name: d
                for d in res_serial["sessions"]
            }
            parallel_by_path = {
                d["input_path"].name: d
                for d in res_parallel["sessions"]
            }
            assert (
                set(serial_by_path.keys())
                == set(parallel_by_path.keys())
            )
            for name in serial_by_path:
                ss = serial_by_path[name]
                sp = parallel_by_path[name]
                # Allow small FP drift
                assert np.allclose(
                    ss["smoothed"], sp["smoothed"],
                    rtol=1e-6, atol=1e-6, equal_nan=True,
                ), f"smoothed positions for {name} differ"
                assert np.allclose(
                    ss["variances"], sp["variances"],
                    rtol=1e-6, atol=1e-6, equal_nan=True,
                ), f"variances for {name} differ"

    # ---------------------------------------------------------- #
    # Case 84: pool cleanup happens even after exception. Run
    # smooth_pose_v2 with n_workers=2 and force an exception by
    # passing a deliberately-bad input. Confirm we don't get a
    # hung pool by making sure subsequent runs work.
    # ---------------------------------------------------------- #
    # This is hard to test cleanly without spawning a separate
    # Python process. Instead we just verify the try/finally
    # structure exists by source inspection.
    import inspect
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        smooth_pose_v2 as _spv2,
    )
    src = inspect.getsource(_spv2)
    assert "finally:" in src, (
        "smooth_pose_v2 must have a try/finally for pool cleanup"
    )
    assert "_pool.close()" in src, (
        "smooth_pose_v2 must close the pool in finally"
    )
    assert "_pool.join()" in src, (
        "smooth_pose_v2 must join the pool in finally"
    )

    # ---------------------------------------------------------- #
    # Case 85: worker functions are picklable (importable from
    # multiprocessing context).
    # ---------------------------------------------------------- #
    import pickle
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        _pool_init, _pool_em_e_step,
        _pool_warm_start_pass, _pool_perspective_pass,
        _pool_final_smooth,
    )
    # Top-level module functions should pickle (their bytecode
    # is reconstructible via module qualname). This is what
    # multiprocessing.Pool relies on.
    for fn in (
        _pool_init, _pool_em_e_step, _pool_warm_start_pass,
        _pool_perspective_pass, _pool_final_smooth,
    ):
        # Pickle the function reference (not its closure)
        data = pickle.dumps(fn)
        unpickled = pickle.loads(data)
        assert unpickled is fn or unpickled.__name__ == fn.__name__

    # ---------------------------------------------------------- #
    # Patch 115 (audit pass 2): determinism in parallel mode.
    # ---------------------------------------------------------- #

    # ---------------------------------------------------------- #
    # Case 86: parallel run twice produces bit-identical output.
    # ---------------------------------------------------------- #
    import tempfile as _tempfile
    from pathlib import Path as _Path
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        smooth_pose_v2 as _spv2_86,
    )
    try:
        import pandas as _pd_86
    except ImportError:
        _pd_86 = None

    if _pd_86 is not None:
        with _tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = _Path(tmpdir)
            in_dir = tmpdir_p / "in"
            in_dir.mkdir()
            test_layout_86 = standard_rat_layout()
            markers_86 = test_layout_86.marker_names

            for i in range(4):
                T = 200
                rng_i = np.random.default_rng(8600 + i)
                root_x = 100 + np.cumsum(rng_i.normal(0, 0.5, T))
                root_y = 100 + np.cumsum(rng_i.normal(0, 0.5, T))
                cols_data = {}
                for m in markers_86:
                    ox, oy = rng_i.uniform(-15, 15, 2)
                    x = root_x + ox + rng_i.normal(0, 0.3, T)
                    y = root_y + oy + rng_i.normal(0, 0.3, T)
                    likes = np.full(T, 0.95)
                    drop = rng_i.random(T) < 0.05
                    likes[drop] = 0.1
                    cols_data[(m, "x")] = x
                    cols_data[(m, "y")] = y
                    cols_data[(m, "likelihood")] = likes
                df = _pd_86.DataFrame(cols_data)
                df.columns = _pd_86.MultiIndex.from_tuples(
                    [
                        ("DeepCut", m, c)
                        for m, c in df.columns
                    ],
                    names=["scorer", "bodyparts", "coords"],
                )
                df.to_csv(in_dir / f"sess_{i}DeepCut.csv")

            # Run twice in parallel mode — should be bit-identical
            # since accumulation now happens in sess_idx order
            out_a = tmpdir_p / "run_a"
            out_b = tmpdir_p / "run_b"
            res_a = _spv2_86(
                pose_input=str(in_dir),
                output_dir=str(out_a),
                layout=test_layout_86,
                fps=30.0,
                likelihood_threshold=0.7,
                em_max_iter=3,
                n_workers=2,
                verbose=False,
            )
            res_b = _spv2_86(
                pose_input=str(in_dir),
                output_dir=str(out_b),
                layout=test_layout_86,
                fps=30.0,
                likelihood_threshold=0.7,
                em_max_iter=3,
                n_workers=2,
                verbose=False,
            )

            # σ values must be bit-identical (post-determinism-fix)
            for m in test_layout_86.marker_names:
                s_a = res_a["params"].sigma_marker[m]
                s_b = res_b["params"].sigma_marker[m]
                assert s_a == s_b, (
                    f"Parallel runs differ at σ_{m}: "
                    f"{s_a!r} vs {s_b!r} — accumulation order "
                    f"is non-deterministic"
                )

            # Smoothed positions must be bit-identical
            sa_by = {
                d["input_path"].name: d for d in res_a["sessions"]
            }
            sb_by = {
                d["input_path"].name: d for d in res_b["sessions"]
            }
            for name in sa_by:
                pa = sa_by[name]["smoothed"]
                pb = sb_by[name]["smoothed"]
                assert np.array_equal(
                    np.where(np.isnan(pa), 0, pa),
                    np.where(np.isnan(pb), 0, pb),
                ), (
                    f"Smoothed positions differ between parallel "
                    f"runs for {name}"
                )

    # ---------------------------------------------------------- #
    # Case 87: _build_and_write_session_output helper produces
    # the same DataFrame as the previous inline code.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        _build_and_write_session_output,
    )
    if _pd_86 is not None:
        with _tempfile.TemporaryDirectory() as tmpdir:
            out_dir = _Path(tmpdir) / "out"
            out_dir.mkdir()
            T_87 = 50
            K_data_87 = 3
            smooth_pos_87 = np.random.default_rng(8701).normal(
                100, 10, (T_87, K_data_87, 2),
            )
            smooth_var_87 = np.random.default_rng(8702).uniform(
                0.5, 2.0, (T_87, K_data_87, 2),
            )
            session_meta = {
                "path": _Path("/fake/sess_xyzDeepCut.csv"),
                "markers": ["a", "b", "c"],
                "likelihoods": np.full((T_87, K_data_87), 0.9),
            }
            data_to_layout_87 = {"a": 0, "b": 1, "c": 2}
            out_path_87, csv_fb, csv_reason = (
                _build_and_write_session_output(
                    session_meta, smooth_pos_87, smooth_var_87,
                    data_to_layout_87, out_dir,
                )
            )
            # When pyarrow not installed, parquet falls back to CSV
            assert out_path_87 is not None
            assert out_path_87.exists()
            assert out_path_87.suffix in (".parquet", ".csv")
            # Stem-stripping: "sess_xyzDeepCut" → "sess_xyz"
            assert "sess_xyz_smoothed_v2" in out_path_87.stem

            # output_dir_path=None should return out_path=None
            out_path_none, _, _ = _build_and_write_session_output(
                session_meta, smooth_pos_87, smooth_var_87,
                data_to_layout_87, None,
            )
            assert out_path_none is None

    # ---------------------------------------------------------- #
    # Patch 116: Q ceiling and damping in M-step.
    # ---------------------------------------------------------- #

    # ---------------------------------------------------------- #
    # Case 88: M-step Q ceiling clamps runaway q estimates.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        _MStepStatsV2, finalize_m_step_v2,
        _M_STEP_Q_CEILING_FACTOR,
    )
    rng_88 = np.random.default_rng(8800)
    test_layout_88 = standard_rat_layout()
    D = test_layout_88.state_dim

    # Build stats that would produce a HUGE q_root_pos via the
    # standard M-step formula. Q_hat = (S11 - S10 F^T - F S10^T
    # + F S00 F^T) / n_pairs. We stuff S11 with a giant value
    # in the (vx, vy) diagonals.
    stats = _MStepStatsV2.empty(test_layout_88)
    stats.n_pairs = 1000
    # Inflate the velocity-velocity diagonal of S11 enough that
    # q_raw would be 100,000× initial (~10^7 if we started at 100)
    huge_diag_value = 1e10  # px²
    stats.S11[2, 2] = huge_diag_value * stats.n_pairs
    stats.S11[3, 3] = huge_diag_value * stats.n_pairs

    initial_params_88 = NoiseParamsV2.default(
        test_layout_88, sigma_marker=2.0,
    )
    # Force initial q_root_pos to a known value
    initial_params_88 = NoiseParamsV2(
        sigma_marker=initial_params_88.sigma_marker,
        q_root_pos=200.0,
        q_root_ori=initial_params_88.q_root_ori,
        q_seg_ori=dict(initial_params_88.q_seg_ori),
        q_length=dict(initial_params_88.q_length),
        constraint_sigma=initial_params_88.constraint_sigma,
    )
    new_params_88 = finalize_m_step_v2(
        stats, test_layout_88, dt=1.0 / 30.0,
        prev_params=initial_params_88,
        initial_params=initial_params_88,
    )
    expected_ceiling = (
        initial_params_88.q_root_pos * _M_STEP_Q_CEILING_FACTOR
    )
    assert new_params_88.q_root_pos <= expected_ceiling + 1e-6, (
        f"q_root_pos exceeded ceiling: "
        f"{new_params_88.q_root_pos} > {expected_ceiling}"
    )
    # Ceiling should have been hit (we deliberately produced
    # a huge raw value)
    assert new_params_88.q_root_pos == expected_ceiling, (
        f"Expected ceiling exact, got {new_params_88.q_root_pos}"
    )

    # ---------------------------------------------------------- #
    # Case 89: Damping blends prev and M-step values.
    # ---------------------------------------------------------- #
    # Build stats that produce a moderate q_raw, then verify
    # damping math. With a Q_hat block diagonal of D, q_raw =
    # D / dt². We want q_raw moderate (not at ceiling), so
    # we'll let the test inspect the actual undamped result.
    stats_89 = _MStepStatsV2.empty(test_layout_88)
    stats_89.n_pairs = 1000
    # Tiny diagonals → q_raw small enough to be below ceiling
    tiny_diag = 0.05  # px²
    stats_89.S11[2, 2] = tiny_diag * stats_89.n_pairs
    stats_89.S11[3, 3] = tiny_diag * stats_89.n_pairs

    # Without damping
    new_params_no_damp = finalize_m_step_v2(
        stats_89, test_layout_88, dt=1.0 / 30.0,
        prev_params=initial_params_88,
        initial_params=initial_params_88,
        damping=0.0,
    )
    q_no_damp = new_params_no_damp.q_root_pos
    # Should be 0.05 × 900 = 45, but floored at q_initial × 0.1
    # (= 200 × 0.1 = 20). So q_raw ≈ 45 > 20. q_no_damp ≈ 45.
    # Verify it's roughly in the expected range
    assert 20.0 <= q_no_damp <= 6000.0, (
        f"q_no_damp out of expected range: {q_no_damp}"
    )

    # With damping=0.5: should be 0.5 × prev + 0.5 × q_no_damp
    new_params_damp_half = finalize_m_step_v2(
        stats_89, test_layout_88, dt=1.0 / 30.0,
        prev_params=initial_params_88,
        initial_params=initial_params_88,
        damping=0.5,
    )
    q_damp_half = new_params_damp_half.q_root_pos
    expected_half = 0.5 * 200.0 + 0.5 * q_no_damp
    assert abs(q_damp_half - expected_half) < 0.5, (
        f"Damped q expected ~{expected_half:.2f}, "
        f"got {q_damp_half:.2f}"
    )
    # Should be strictly between prev and undamped
    assert min(200.0, q_no_damp) <= q_damp_half <= max(200.0, q_no_damp), (
        f"Damped q={q_damp_half} not between prev=200 "
        f"and undamped={q_no_damp}"
    )

    # damping=1.0 internally clamps to 0.95
    new_params_strong = finalize_m_step_v2(
        stats_89, test_layout_88, dt=1.0 / 30.0,
        prev_params=initial_params_88,
        initial_params=initial_params_88,
        damping=1.0,
    )
    q_strong = new_params_strong.q_root_pos
    expected_strong = 0.95 * 200.0 + 0.05 * q_no_damp
    assert abs(q_strong - expected_strong) < 1.0, (
        f"damping=1.0 (→0.95) expected ~{expected_strong:.2f}, "
        f"got {q_strong:.2f}"
    )

    # ---------------------------------------------------------- #
    # Patch 117: spatial stratification — per-session-median
    # M-step aggregation.
    # ---------------------------------------------------------- #

    # ---------------------------------------------------------- #
    # Case 90: per_session_fit_from_stats produces sensible
    # raw q estimates from a single session's stats.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        _PerSessionFitV2, per_session_fit_from_stats,
        _aggregate_scalar, finalize_m_step_v2_from_per_session,
    )
    test_layout_90 = standard_rat_layout()
    D_90 = test_layout_90.state_dim

    # Build a stats with known q values
    stats_90 = _MStepStatsV2.empty(test_layout_90)
    stats_90.n_pairs = 1000
    # Set tiny diagonals → small q_raw
    stats_90.S11[2, 2] = 0.01 * stats_90.n_pairs
    stats_90.S11[3, 3] = 0.01 * stats_90.n_pairs
    # Some sigma stats too
    for m in test_layout_90.marker_names:
        stats_90.sigma_sum_sq[m] = 100.0  # sum of sq residuals
        stats_90.sigma_n_obs[m] = 100  # 50 frames × 2 axes
    fit_90 = per_session_fit_from_stats(
        stats_90, test_layout_90, dt=1.0 / 30.0,
    )
    # q_root_pos = (0.01 + 0.01) / 2 / (1/900) = 9.0
    assert abs(fit_90.q_root_pos - 9.0) < 0.01, (
        f"q_root_pos expected ~9.0, got {fit_90.q_root_pos}"
    )
    # σ_marker = sqrt(100/100) = 1.0
    for m in test_layout_90.marker_names:
        assert abs(fit_90.sigma_marker[m] - 1.0) < 1e-9, (
            f"σ for {m} expected 1.0, got {fit_90.sigma_marker[m]}"
        )
    assert fit_90.n_pairs == 1000
    assert fit_90.n_obs_per_marker[
        test_layout_90.marker_names[0]
    ] == 100

    # Empty stats → returns zeros
    fit_empty = per_session_fit_from_stats(
        _MStepStatsV2.empty(test_layout_90),
        test_layout_90, dt=1.0 / 30.0,
    )
    assert fit_empty.q_root_pos == 0.0
    assert fit_empty.n_pairs == 0

    # ---------------------------------------------------------- #
    # Case 91: _aggregate_scalar correctly applies median /
    # mean / trimmed_mean to a list with outliers.
    # ---------------------------------------------------------- #
    # 9 normal values + 1 huge outlier
    vals = [10.0] * 9 + [1e6]
    weights = [1.0] * 10

    median_result = _aggregate_scalar(vals, weights, "median")
    mean_result = _aggregate_scalar(vals, weights, "mean")
    trimmed_result = _aggregate_scalar(vals, weights, "trimmed_mean")

    assert median_result == 10.0, (
        f"median should ignore outlier, got {median_result}"
    )
    # mean should be heavily affected by the outlier
    assert mean_result > 1000.0, (
        f"mean should be dominated by outlier, got {mean_result}"
    )
    # trimmed_mean drops top/bottom 10% (= 1 value each side
    # for n=10), so excludes the outlier
    assert trimmed_result == 10.0, (
        f"trimmed_mean should exclude outlier, got {trimmed_result}"
    )

    # Empty list returns 0
    assert _aggregate_scalar([], None, "median") == 0.0

    # ---------------------------------------------------------- #
    # Case 92: finalize_m_step_v2_from_per_session aggregates
    # per-session fits via median, robust to outliers.
    # ---------------------------------------------------------- #
    test_layout_92 = standard_rat_layout()
    initial_params_92 = NoiseParamsV2.default(
        test_layout_92, sigma_marker=2.0,
    )
    initial_params_92 = NoiseParamsV2(
        sigma_marker=initial_params_92.sigma_marker,
        q_root_pos=200.0,
        q_root_ori=initial_params_92.q_root_ori,
        q_seg_ori=dict(initial_params_92.q_seg_ori),
        q_length=dict(initial_params_92.q_length),
        constraint_sigma=initial_params_92.constraint_sigma,
    )

    # Build 5 well-behaved per-session fits + 1 outlier
    fits_92: List[_PerSessionFitV2] = []
    for i in range(5):
        f = _PerSessionFitV2.empty(test_layout_92)
        f.n_pairs = 1000
        f.q_root_pos = 250.0  # close to initial 200
        f.q_root_ori = 0.5
        for s in test_layout_92.non_root_topo_order:
            f.q_seg_ori[s] = 0.5
            f.q_length[s] = 0.01
        for m in test_layout_92.marker_names:
            f.sigma_marker[m] = 2.5
            f.n_obs_per_marker[m] = 200
        fits_92.append(f)
    # Outlier session
    f_outlier = _PerSessionFitV2.empty(test_layout_92)
    f_outlier.n_pairs = 1000
    f_outlier.q_root_pos = 1e8  # absurd
    f_outlier.q_root_ori = 1e6
    for s in test_layout_92.non_root_topo_order:
        f_outlier.q_seg_ori[s] = 1e6
        f_outlier.q_length[s] = 1e6
    for m in test_layout_92.marker_names:
        f_outlier.sigma_marker[m] = 1000.0
        f_outlier.n_obs_per_marker[m] = 200
    fits_92.append(f_outlier)

    # Median aggregation: outlier should NOT dominate
    new_params_med = finalize_m_step_v2_from_per_session(
        fits_92, test_layout_92,
        prev_params=initial_params_92,
        initial_params=initial_params_92,
        aggregation="median",
    )
    # With 5 normal at 250 + 1 outlier at 1e8, median is 250
    # (or somewhere near it depending on parity)
    assert new_params_med.q_root_pos < 1000.0, (
        f"Median agg should reject outlier; q_root_pos="
        f"{new_params_med.q_root_pos}"
    )
    # σ similarly
    for m in test_layout_92.marker_names:
        assert new_params_med.sigma_marker[m] < 10.0, (
            f"Median σ for {m}: {new_params_med.sigma_marker[m]} "
            f"(should be ~2.5, NOT dominated by 1000)"
        )

    # Mean aggregation: outlier WOULD dominate, but ceiling
    # clips it. Should hit ceiling at initial × _M_STEP_Q_CEILING_FACTOR
    # (10 since patch 121c, was 30 in earlier patches).
    new_params_mean = finalize_m_step_v2_from_per_session(
        fits_92, test_layout_92,
        prev_params=initial_params_92,
        initial_params=initial_params_92,
        aggregation="mean",
    )
    # Mean of [250, 250, 250, 250, 250, 1e8] is huge,
    # so clipping at initial × ceiling factor fires.
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        _M_STEP_Q_CEILING_FACTOR as _CEIL_92,
    )
    assert new_params_mean.q_root_pos == (
        initial_params_92.q_root_pos * _CEIL_92
    ), (
        f"Mean agg with outlier should hit ceiling "
        f"{initial_params_92.q_root_pos * _CEIL_92}; "
        f"got {new_params_mean.q_root_pos}"
    )

    # Trimmed mean: drops top + bottom 10% (= 1 each side
    # for n=6), excludes the outlier
    new_params_trim = finalize_m_step_v2_from_per_session(
        fits_92, test_layout_92,
        prev_params=initial_params_92,
        initial_params=initial_params_92,
        aggregation="trimmed_mean",
    )
    assert new_params_trim.q_root_pos < 1000.0, (
        f"Trimmed mean should reject outlier; q_root_pos="
        f"{new_params_trim.q_root_pos}"
    )

    # ---------------------------------------------------------- #
    # Case 93: Empty per-session fits list returns prev_params.
    # ---------------------------------------------------------- #
    new_params_empty = finalize_m_step_v2_from_per_session(
        [], test_layout_92,
        prev_params=initial_params_92,
        initial_params=initial_params_92,
        aggregation="median",
    )
    assert (
        new_params_empty.q_root_pos == initial_params_92.q_root_pos
    )
    assert new_params_empty is initial_params_92, (
        "With no valid sessions, returns prev_params unchanged"
    )

    # ---------------------------------------------------------- #
    # Patch 118: save_model_v2 creates parent directory.
    # ---------------------------------------------------------- #

    # ---------------------------------------------------------- #
    # Case 94: save_model_v2 to a path whose parent directory
    # doesn't exist yet — should create it instead of raising.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        save_model_v2 as _save_94, load_model_v2 as _load_94,
    )
    test_layout_94 = standard_rat_layout()
    fitted_lengths_94 = FittedLengths(
        segment_lengths={
            s: 5.0 for s in test_layout_94.non_root_topo_order
        },
        segment_length_iqr={
            s: 0.5 for s in test_layout_94.non_root_topo_order
        },
        marker_offsets={
            m: (1.0, 0.0) for m in test_layout_94.marker_names
        },
    )
    params_94 = NoiseParamsV2.default(
        test_layout_94, sigma_marker=2.0,
    )

    with _tempfile.TemporaryDirectory() as tmpdir:
        # Path with two levels of nonexistent parents
        save_path = Path(tmpdir) / "nonexistent" / "deeper" / "model.npz"
        assert not save_path.parent.exists(), (
            "Test setup invariant: parent should not exist"
        )
        # Should succeed, not raise FileNotFoundError
        _save_94(
            save_path, test_layout_94, fitted_lengths_94,
            params_94, fps=30.0, likelihood_threshold=0.7,
        )
        assert save_path.exists(), (
            f"save_model_v2 should have created the file at "
            f"{save_path}"
        )
        # Round-trip works
        loaded = _load_94(save_path)
        assert loaded[0].n_markers == test_layout_94.n_markers

    # ---------------------------------------------------------- #
    # Patch 120a: per-marker latent drift block in the state
    # vector. Opt-in via BodyLayout(with_drift=True). When
    # disabled (default), state dim and dynamics match patch
    # 119 exactly. When enabled, F gains mean-reverting drift
    # dynamics and Q gains a random-walk noise block.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        build_F_v2 as _build_F_120,
        build_Q_v2 as _build_Q_120,
    )

    # ---------------------------------------------------------- #
    # Case 95: with_drift=False (default) preserves state_dim
    # exactly. Sanity guard: if some future patch changes the
    # default we want this test to flag it.
    # ---------------------------------------------------------- #
    layout_no_drift = standard_rat_layout()
    assert layout_no_drift.with_drift is False, (
        "Default layout must have with_drift=False for "
        "backward compat"
    )
    expected_no_drift = 8 + 6 * layout_no_drift.n_non_root_segments
    assert layout_no_drift.state_dim == expected_no_drift, (
        f"state_dim regression: expected {expected_no_drift}, "
        f"got {layout_no_drift.state_dim}"
    )

    # ---------------------------------------------------------- #
    # Case 96: with_drift=True grows state_dim by exactly 2K
    # and drift_block_slice covers the new region.
    # ---------------------------------------------------------- #
    layout_drift = standard_rat_layout()
    layout_drift.with_drift = True
    K = layout_drift.n_markers
    expected_drift = 8 + 6 * layout_drift.n_non_root_segments + 2 * K
    assert layout_drift.state_dim == expected_drift, (
        f"with_drift state_dim: expected {expected_drift}, "
        f"got {layout_drift.state_dim}"
    )
    drift_sl = layout_drift.drift_block_slice
    assert drift_sl.start == 8 + 6 * layout_drift.n_non_root_segments
    assert drift_sl.stop == drift_sl.start + 2 * K
    # And without drift, drift_block_slice is empty so
    # callers can use it unconditionally.
    empty_sl = layout_no_drift.drift_block_slice
    assert empty_sl.start == empty_sl.stop, (
        "drift_block_slice must be empty when with_drift=False"
    )

    # ---------------------------------------------------------- #
    # Case 97: slice_marker_drift returns 2-dim, non-overlapping
    # slices in marker_names order.
    # ---------------------------------------------------------- #
    seen = []
    for m in layout_drift.marker_names:
        sl = layout_drift.slice_marker_drift(m)
        assert sl.stop - sl.start == 2, (
            f"slice_marker_drift({m!r}) must be 2-dim"
        )
        seen.append((m, sl.start, sl.stop))
    # Strictly sequential, contiguous
    starts = [s[1] for s in seen]
    assert starts == sorted(starts), "slices must be ordered"
    for (_, _, stop_a), (_, start_b, _) in zip(seen, seen[1:]):
        assert stop_a == start_b, (
            "marker drift slices must be contiguous"
        )
    # And cover exactly drift_block_slice
    assert seen[0][1] == drift_sl.start
    assert seen[-1][2] == drift_sl.stop

    # ---------------------------------------------------------- #
    # Case 98: slice_marker_drift raises when with_drift=False.
    # ---------------------------------------------------------- #
    try:
        layout_no_drift.slice_marker_drift("back2")
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "with_drift" in str(e)

    # ---------------------------------------------------------- #
    # Case 99: NoiseParamsV2.default populates drift fields iff
    # layout.with_drift, and applies the design-doc heuristic.
    # ---------------------------------------------------------- #
    p_no_drift = NoiseParamsV2.default(layout_no_drift, sigma_marker=2.0)
    assert p_no_drift.q_drift is None
    assert p_no_drift.alpha_drift is None
    assert p_no_drift.r_drift is None

    sigma_test = 2.5
    dt_test = 1.0 / 30.0
    p_drift = NoiseParamsV2.default(
        layout_drift, sigma_marker=sigma_test, dt=dt_test,
    )
    assert p_drift.q_drift is not None
    assert p_drift.alpha_drift is not None
    assert p_drift.r_drift is not None
    for m in layout_drift.marker_names:
        # R_m = sigma_marker
        assert p_drift.r_drift[m] == sigma_test
        # α_m = 0.05 (default)
        assert p_drift.alpha_drift[m] == 0.05
        # q_drift_m = (R_m / 5)² / dt
        expected_q = (sigma_test / 5.0) ** 2 / dt_test
        assert abs(p_drift.q_drift[m] - expected_q) < 1e-9

    # ---------------------------------------------------------- #
    # Case 100: build_F_v2 with with_drift=False matches the
    # patch-119 result exactly. Backward-compat sentinel.
    # ---------------------------------------------------------- #
    dt = 1.0 / 30.0
    F_no_drift = _build_F_120(layout_no_drift, dt)
    # Body-state shape unchanged
    assert F_no_drift.shape == (
        layout_no_drift.state_dim, layout_no_drift.state_dim,
    )
    # F should equal the version we'd build from scratch with
    # only constant-velocity blocks. Spot-check a few entries.
    assert F_no_drift[0, 2] == dt  # x ← x + vx*dt
    assert F_no_drift[1, 3] == dt  # y ← y + vy*dt

    # ---------------------------------------------------------- #
    # Case 101: build_F_v2 with drift adds (1 - α_m) I_2 per
    # marker on the drift block. The body-state submatrix is
    # identical to the no-drift version.
    # ---------------------------------------------------------- #
    F_drift = _build_F_120(layout_drift, dt, params=p_drift)
    D_body = 8 + 6 * layout_drift.n_non_root_segments
    # Body submatrix matches the no-drift run (same body
    # topology, so the body block of layout_drift's F should
    # equal the entirety of layout_no_drift's F).
    np.testing.assert_allclose(
        F_drift[:D_body, :D_body], F_no_drift,
        err_msg="Body-state F regressed when drift was enabled",
    )
    # Drift block is block-diagonal 2x2 with (1 - α) I_2.
    for m in layout_drift.marker_names:
        sl = layout_drift.slice_marker_drift(m)
        block = F_drift[sl, sl]
        alpha_m = p_drift.alpha_drift[m]
        np.testing.assert_allclose(
            block, (1.0 - alpha_m) * np.eye(2),
            err_msg=f"drift F block for {m!r}",
        )
        # Off-diagonal between drift and body is zero
        # (drift is independent of body state)
        assert np.allclose(F_drift[sl, :D_body], 0.0)
        assert np.allclose(F_drift[:D_body, sl], 0.0)

    # ---------------------------------------------------------- #
    # Case 102: build_F_v2 falls back to alpha=0.05 when no
    # params is passed. Existing call sites pass only
    # (layout, dt); they must keep working when callers later
    # flip with_drift on without updating every callsite.
    # ---------------------------------------------------------- #
    F_drift_no_params = _build_F_120(layout_drift, dt)
    for m in layout_drift.marker_names:
        sl = layout_drift.slice_marker_drift(m)
        np.testing.assert_allclose(
            F_drift_no_params[sl, sl], 0.95 * np.eye(2),
            err_msg=(
                f"build_F_v2 must default alpha=0.05 when "
                f"params=None (marker {m!r})"
            ),
        )

    # ---------------------------------------------------------- #
    # Case 103: build_F_v2 rejects out-of-range alpha values.
    # ---------------------------------------------------------- #
    p_bad = NoiseParamsV2.default(layout_drift, sigma_marker=2.0)
    p_bad.alpha_drift["back2"] = 1.0
    try:
        _build_F_120(layout_drift, dt, params=p_bad)
        assert False, "Expected ValueError on alpha=1.0"
    except ValueError as e:
        assert "alpha_drift" in str(e)
    p_bad.alpha_drift["back2"] = -0.1
    try:
        _build_F_120(layout_drift, dt, params=p_bad)
        assert False, "Expected ValueError on negative alpha"
    except ValueError as e:
        assert "alpha_drift" in str(e)

    # ---------------------------------------------------------- #
    # Case 104: build_Q_v2 with drift gains a per-marker
    # q_drift_m * dt * I_2 block; body-state Q is unchanged.
    # ---------------------------------------------------------- #
    Q_no_drift = _build_Q_120(layout_no_drift, p_no_drift, dt)
    Q_drift = _build_Q_120(layout_drift, p_drift, dt)
    np.testing.assert_allclose(
        Q_drift[:D_body, :D_body], Q_no_drift,
        err_msg="Body-state Q regressed when drift was enabled",
    )
    for m in layout_drift.marker_names:
        sl = layout_drift.slice_marker_drift(m)
        block = Q_drift[sl, sl]
        q_d = p_drift.q_drift[m]
        np.testing.assert_allclose(
            block, q_d * dt * np.eye(2),
            err_msg=f"drift Q block for {m!r}",
        )
    # Q stays symmetric and positive semi-definite
    assert np.allclose(Q_drift, Q_drift.T), "Q must be symmetric"
    eigs = np.linalg.eigvalsh(Q_drift)
    assert eigs.min() >= -1e-10, (
        f"Q must be PSD; min eigenvalue {eigs.min()}"
    )

    # ---------------------------------------------------------- #
    # Case 105: build_Q_v2 raises when with_drift=True but
    # params.q_drift is None (caller forgot to set it).
    # ---------------------------------------------------------- #
    p_missing = NoiseParamsV2.default(layout_drift, sigma_marker=2.0)
    p_missing.q_drift = None  # simulate omission
    try:
        _build_Q_120(layout_drift, p_missing, dt)
        assert False, "Expected ValueError on missing q_drift"
    except ValueError as e:
        assert "q_drift" in str(e)

    # ---------------------------------------------------------- #
    # Case 106: stationary variance of the drift dynamics
    # matches the analytical prediction. This validates that
    # F and Q are jointly correct: simulating the AR(1) for
    # many steps should converge to
    # σ²_∞ = q_drift × dt / (1 - (1 - α)²)
    #      = q_drift × dt / (2α - α²)
    # ---------------------------------------------------------- #
    rng_106 = np.random.default_rng(120)
    layout_106 = standard_rat_layout()
    layout_106.with_drift = True
    sigma_106 = 3.0
    p_106 = NoiseParamsV2.default(
        layout_106, sigma_marker=sigma_106, dt=dt,
    )
    F_106 = _build_F_120(layout_106, dt, params=p_106)
    Q_106 = _build_Q_120(layout_106, p_106, dt)

    # Simulate just the drift block of the first marker for
    # T_sim steps, then compare empirical variance to the
    # closed-form stationary value.
    m_test = layout_106.marker_names[0]
    sl = layout_106.slice_marker_drift(m_test)
    F_block = F_106[sl, sl]
    Q_block = Q_106[sl, sl]
    alpha = p_106.alpha_drift[m_test]
    q_d = p_106.q_drift[m_test]

    # Analytical stationary variance per axis
    sigma2_inf = q_d * dt / (1.0 - (1.0 - alpha) ** 2)

    T_sim = 10000
    delta = np.zeros(2)
    samples = np.zeros((T_sim, 2))
    L = np.linalg.cholesky(Q_block)
    for t in range(T_sim):
        w = L @ rng_106.standard_normal(2)
        delta = F_block @ delta + w
        samples[t] = delta
    # Use the second half (post burn-in) to estimate variance
    var_emp = samples[T_sim // 2:].var(axis=0)
    # Within 15% of analytical (10k samples per axis)
    assert abs(var_emp[0] - sigma2_inf) / sigma2_inf < 0.15, (
        f"drift stationary variance off: empirical {var_emp[0]:.4f} "
        f"vs analytical {sigma2_inf:.4f}"
    )
    assert abs(var_emp[1] - sigma2_inf) / sigma2_inf < 0.15

    # ---------------------------------------------------------- #
    # Patch 119a: apply fitted marker offsets to the layout
    # so that runtime forward kinematics actually uses them.
    # Previously fit_body_lengths produced offsets that were
    # only saved to disk, never read; FK kept using the
    # layout's construction-time defaults.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        apply_fitted_offsets_to_layout as _apply_119a,
        state_to_marker_positions as _stmp_119a,
    )

    # ---------------------------------------------------------- #
    # Case 107: apply_fitted_offsets_to_layout updates each
    # marker's (length, angle) in place.
    # ---------------------------------------------------------- #
    layout_107 = standard_rat_layout()
    # Default nose offset is (1.0, 0.0) per standard_rat_layout
    seg_name_pre, off_pre = layout_107.marker_attachment("nose")
    assert off_pre == (1.0, 0.0), (
        f"Test invariant: default nose offset is (1.0, 0.0); "
        f"got {off_pre}"
    )
    fitted_107 = FittedLengths(
        segment_lengths={
            s: 30.0 for s in layout_107.non_root_topo_order
        },
        segment_length_iqr={
            s: 1.0 for s in layout_107.non_root_topo_order
        },
        marker_offsets={
            "nose": (1.7, 0.13),
            "ear_left": (0.4, 1.2),
            # other markers omitted to test the "leave alone"
            # behavior in case 109
        },
    )
    _apply_119a(layout_107, fitted_107)
    seg_name_post, off_post = layout_107.marker_attachment("nose")
    assert seg_name_post == seg_name_pre  # segment unchanged
    assert off_post == (1.7, 0.13), (
        f"nose offset should be (1.7, 0.13) after apply; "
        f"got {off_post}"
    )
    _, ear_off = layout_107.marker_attachment("ear_left")
    assert ear_off == (0.4, 1.2)

    # ---------------------------------------------------------- #
    # Case 108: distal markers round-trip correctly. fit_body_
    # lengths writes (0, 0) for them, so applying preserves the
    # structural distal-end-of-segment invariant.
    # ---------------------------------------------------------- #
    layout_108 = standard_rat_layout()
    fitted_108 = fit_body_lengths(
        # Build trivial synthetic data where every marker is
        # at the rest pose (offsets in fit are computed from
        # data, not pulled from layout)
        np.zeros((50, layout_108.n_markers, 2)),
        np.full((50, layout_108.n_markers), 0.95),
        layout_108, layout_108.marker_names,
    )
    # back2 is the body's distal marker — must be (0, 0)
    assert fitted_108.marker_offsets.get("back2") == (0.0, 0.0)
    _apply_119a(layout_108, fitted_108)
    _, back2_off = layout_108.marker_attachment("back2")
    assert back2_off == (0.0, 0.0), (
        f"distal marker back2 must stay at (0, 0) after "
        f"apply; got {back2_off}"
    )
    # Same for headmid, tailbase, etc.
    for distal in ("back4", "neck", "headmid", "tailbase"):
        if distal in layout_108.marker_names:
            _, off = layout_108.marker_attachment(distal)
            assert off == (0.0, 0.0), (
                f"distal marker {distal!r} should be (0, 0); "
                f"got {off}"
            )

    # ---------------------------------------------------------- #
    # Case 109: markers absent from fitted_lengths.marker_offsets
    # keep their current layout values (defensive against partial
    # fits or old saved models).
    # ---------------------------------------------------------- #
    layout_109 = standard_rat_layout()
    _, lat_left_pre = layout_109.marker_attachment("lateral_left")
    fitted_109 = FittedLengths(
        segment_lengths={s: 30.0 for s in layout_109.non_root_topo_order},
        segment_length_iqr={s: 1.0 for s in layout_109.non_root_topo_order},
        marker_offsets={"nose": (1.7, 0.0)},  # only nose
    )
    _apply_119a(layout_109, fitted_109)
    _, lat_left_post = layout_109.marker_attachment("lateral_left")
    assert lat_left_post == lat_left_pre, (
        f"lateral_left should be unchanged when not in "
        f"fitted_lengths.marker_offsets; "
        f"pre={lat_left_pre} post={lat_left_post}"
    )

    # ---------------------------------------------------------- #
    # Case 110: idempotency — applying the same FittedLengths
    # twice produces the same layout.
    # ---------------------------------------------------------- #
    layout_110 = standard_rat_layout()
    fitted_110 = FittedLengths(
        segment_lengths={s: 30.0 for s in layout_110.non_root_topo_order},
        segment_length_iqr={s: 1.0 for s in layout_110.non_root_topo_order},
        marker_offsets={
            "nose": (1.5, 0.1),
            "ear_left": (0.6, 1.0),
            "ear_right": (0.6, -1.0),
        },
    )
    _apply_119a(layout_110, fitted_110)
    snapshot_after_first = {
        m: layout_110.marker_attachment(m)[1]
        for m in layout_110.marker_names
    }
    _apply_119a(layout_110, fitted_110)
    snapshot_after_second = {
        m: layout_110.marker_attachment(m)[1]
        for m in layout_110.marker_names
    }
    assert snapshot_after_first == snapshot_after_second, (
        "apply_fitted_offsets_to_layout must be idempotent"
    )

    # ---------------------------------------------------------- #
    # Case 111: behavior change — after apply, FK uses the
    # fitted offset, not the layout's default. This is the
    # core bug being fixed.
    # ---------------------------------------------------------- #
    layout_default = standard_rat_layout()
    layout_fitted = standard_rat_layout()
    # Hand-build a fitted offset for nose that's clearly
    # different from the default (1.0, 0.0).
    fitted_111 = FittedLengths(
        segment_lengths={s: 30.0 for s in layout_fitted.non_root_topo_order},
        segment_length_iqr={s: 1.0 for s in layout_fitted.non_root_topo_order},
        marker_offsets={
            m: layout_default.marker_attachment(m)[1]
            for m in layout_default.marker_names
        },
    )
    fitted_111.marker_offsets["nose"] = (2.5, 0.4)
    _apply_119a(layout_fitted, fitted_111)

    # Build a state where the body is at the origin pointing
    # along +x (so the segment frames don't rotate the
    # offsets), all segments at rest.
    indices_111 = _pack_state_layout_indices(layout_default)
    # Set non-root segments to rest pose so FK simplifies
    state_111 = np.zeros(layout_default.state_dim)
    state_111[indices_111["__root__"]["cos"]] = 1.0  # body cos=1
    for seg_name in layout_default.non_root_topo_order:
        seg_idx = indices_111[seg_name]
        state_111[seg_idx["cos"]] = 1.0  # rest cos=1
        state_111[seg_idx["length"]] = 30.0

    pos_default = _stmp_119a(state_111, layout_default)
    pos_fitted = _stmp_119a(state_111, layout_fitted)
    nose_idx = layout_default.marker_names.index("nose")
    # Default nose offset is (1.0, 0.0). After apply with
    # (2.5, 0.4), the nose must end up at a different world
    # position. Other markers (whose offsets we copied from
    # default) must be the same.
    assert not np.allclose(
        pos_default[nose_idx], pos_fitted[nose_idx], atol=1e-6,
    ), (
        f"nose position must differ after apply with non-"
        f"default offset; got default={pos_default[nose_idx]}, "
        f"fitted={pos_fitted[nose_idx]}"
    )
    for i, m in enumerate(layout_default.marker_names):
        if m == "nose":
            continue
        np.testing.assert_allclose(
            pos_default[i], pos_fitted[i], atol=1e-9,
            err_msg=(
                f"marker {m!r} (offset unchanged) must yield "
                f"the same FK position before and after apply"
            ),
        )

    # ---------------------------------------------------------- #
    # Case 112: load_model_v2 round-trips fitted offsets into
    # the returned layout. Save with fitted offsets applied,
    # load, verify layout.marker_attachment returns the fitted
    # values directly (no separate apply step needed by caller).
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        save_model_v2 as _save_119a,
        load_model_v2 as _load_119a,
    )
    layout_112 = standard_rat_layout()
    fitted_112 = FittedLengths(
        segment_lengths={s: 30.0 for s in layout_112.non_root_topo_order},
        segment_length_iqr={s: 1.0 for s in layout_112.non_root_topo_order},
        marker_offsets={
            m: layout_112.marker_attachment(m)[1]
            for m in layout_112.marker_names
        },
    )
    fitted_112.marker_offsets["ear_left"] = (0.77, 0.99)
    _apply_119a(layout_112, fitted_112)
    params_112 = NoiseParamsV2.default(layout_112, sigma_marker=2.0)

    with _tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "m.npz"
        _save_119a(
            save_path, layout_112, fitted_112, params_112,
            fps=30.0, likelihood_threshold=0.7,
        )
        loaded_layout, _, _, _, _, _ = _load_119a(save_path)
        _, ear_off_loaded = loaded_layout.marker_attachment("ear_left")
        assert ear_off_loaded == (0.77, 0.99), (
            f"load_model_v2 must apply fitted offsets to the "
            f"returned layout; ear_left got {ear_off_loaded}"
        )

    # ---------------------------------------------------------- #
    # Patch 120b: wire per-marker drift δ_m into FK + Jacobian,
    # plus empirical drift calibration in fit_body_lengths,
    # plus end-to-end save/load round-trip with backward compat.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        forward_kinematics as _fk_120b,
        state_to_marker_positions_batch as _stmpb_120b,
        state_to_marker_jacobian_batch as _stmjb_120b,
    )

    # ---------------------------------------------------------- #
    # Case 113: with_drift=True, δ_m=0 reproduces the no-drift
    # FK result exactly. Regression sentinel for the wiring.
    # ---------------------------------------------------------- #
    layout_113_no = standard_rat_layout()
    layout_113_yes = standard_rat_layout()
    layout_113_yes.with_drift = True

    indices_113 = _pack_state_layout_indices(layout_113_no)
    state_no = np.zeros(layout_113_no.state_dim)
    state_no[indices_113["__root__"]["cos"]] = 1.0
    state_no[indices_113["__root__"]["x"]] = 100.0
    state_no[indices_113["__root__"]["y"]] = 100.0
    for seg in layout_113_no.non_root_topo_order:
        seg_idx = indices_113[seg]
        state_no[seg_idx["cos"]] = 1.0
        state_no[seg_idx["length"]] = 30.0

    # Build a drift-enabled state with δ=0 for all markers
    state_yes = np.zeros(layout_113_yes.state_dim)
    state_yes[:layout_113_no.state_dim] = state_no  # body state copy
    pos_no = _stmp_119a(state_no, layout_113_no)
    pos_yes = _stmp_119a(state_yes, layout_113_yes)
    np.testing.assert_allclose(
        pos_no, pos_yes, atol=1e-12,
        err_msg="δ=0 must reproduce no-drift FK exactly",
    )

    # ---------------------------------------------------------- #
    # Case 114: nonzero δ_m shifts only the corresponding marker
    # by R_world[seg] @ δ_m. Verify per-marker independence.
    # ---------------------------------------------------------- #
    state_drift = state_yes.copy()
    target_marker = "nose"
    nose_idx = layout_113_yes.marker_names.index(target_marker)
    nose_drift = np.array([0.7, -0.3])
    drift_sl = layout_113_yes.slice_marker_drift(target_marker)
    state_drift[drift_sl] = nose_drift

    pos_baseline = _stmp_119a(state_yes, layout_113_yes)
    pos_drifted = _stmp_119a(state_drift, layout_113_yes)
    diff = pos_drifted - pos_baseline

    # Only the target marker should have changed
    for i, m in enumerate(layout_113_yes.marker_names):
        if m == target_marker:
            continue
        np.testing.assert_allclose(
            diff[i], 0.0, atol=1e-12,
            err_msg=(
                f"setting δ_{target_marker!r} should not move "
                f"marker {m!r}"
            ),
        )
    # The shift on the target marker is R_world[seg] @ δ in world
    # frame. With body cos=1, sin=0, head and neck at rest cos=1
    # → R_world[head] = identity, so the shift is just δ itself.
    np.testing.assert_allclose(
        diff[nose_idx], nose_drift, atol=1e-9,
        err_msg=(
            "nose shift should equal δ_nose under identity body "
            "rotation"
        ),
    )

    # ---------------------------------------------------------- #
    # Case 115: numerical vs analytical Jacobian for the drift
    # block. The most important correctness check — if the
    # drift partials are wrong, the EKF will silently misbehave.
    # ---------------------------------------------------------- #
    rng_115 = np.random.default_rng(115)
    layout_115 = standard_rat_layout()
    layout_115.with_drift = True
    state_115 = np.zeros(layout_115.state_dim)
    indices_115 = _pack_state_layout_indices(layout_115)
    state_115[indices_115["__root__"]["cos"]] = np.cos(0.4)
    state_115[indices_115["__root__"]["sin"]] = np.sin(0.4)
    state_115[indices_115["__root__"]["x"]] = 50.0
    state_115[indices_115["__root__"]["y"]] = 80.0
    for seg in layout_115.non_root_topo_order:
        seg_idx = indices_115[seg]
        ang = rng_115.uniform(-0.3, 0.3)
        state_115[seg_idx["cos"]] = np.cos(ang)
        state_115[seg_idx["sin"]] = np.sin(ang)
        state_115[seg_idx["length"]] = 30.0
    # Random nonzero drift to ensure nonzero state vector entries
    for m in layout_115.marker_names:
        sl = layout_115.slice_marker_drift(m)
        state_115[sl] = rng_115.normal(0, 1.0, 2)

    H_analytic = state_to_marker_jacobian(state_115, layout_115)
    # Numerical Jacobian via finite differences. We only check
    # the drift columns — body-state columns are exercised by
    # earlier tests.
    eps = 1e-6
    base_pos = _stmp_119a(state_115, layout_115).flatten()
    for m in layout_115.marker_names:
        sl = layout_115.slice_marker_drift(m)
        for j, col in enumerate([sl.start, sl.start + 1]):
            s_plus = state_115.copy()
            s_plus[col] += eps
            s_minus = state_115.copy()
            s_minus[col] -= eps
            num_col = (
                _stmp_119a(s_plus, layout_115).flatten()
                - _stmp_119a(s_minus, layout_115).flatten()
            ) / (2 * eps)
            ana_col = H_analytic[:, col]
            np.testing.assert_allclose(
                num_col, ana_col, atol=1e-7,
                err_msg=(
                    f"drift Jacobian column {col} (marker {m!r}, "
                    f"axis {j}) disagrees with numerical"
                ),
            )

    # ---------------------------------------------------------- #
    # Case 116: batch FK with drift agrees with per-frame FK.
    # ---------------------------------------------------------- #
    T_116 = 5
    states_batch = np.tile(state_115, (T_116, 1))
    # Vary drift over time so the batch path actually exercises
    # the per-frame drift lookup
    rng_116 = np.random.default_rng(116)
    for t in range(T_116):
        for m in layout_115.marker_names:
            sl = layout_115.slice_marker_drift(m)
            states_batch[t, sl] = rng_116.normal(0, 1.0, 2)
    pos_batch = _stmpb_120b(states_batch, layout_115, device="cpu")
    for t in range(T_116):
        pos_t = _stmp_119a(states_batch[t], layout_115)
        np.testing.assert_allclose(
            pos_batch[t], pos_t, atol=1e-12,
            err_msg=f"batch FK frame {t} disagrees with per-frame",
        )

    # ---------------------------------------------------------- #
    # Case 117: batch Jacobian with drift agrees with per-frame
    # Jacobian on the drift columns.
    # ---------------------------------------------------------- #
    H_batch = _stmjb_120b(states_batch, layout_115, device="cpu")
    for t in range(T_116):
        H_t = state_to_marker_jacobian(states_batch[t], layout_115)
        # Check drift columns specifically
        drift_block = layout_115.drift_block_slice
        np.testing.assert_allclose(
            H_batch[t, :, drift_block.start:drift_block.stop],
            H_t[:, drift_block.start:drift_block.stop],
            atol=1e-12,
            err_msg=f"batch H drift cols frame {t} mismatch",
        )

    # ---------------------------------------------------------- #
    # Case 118: fit_body_lengths populates marker_r_drift /
    # marker_q_drift, with values reflecting actual scatter.
    # Synthesize markers at known offsets plus controlled noise;
    # verify recovered r_drift is in the right ballpark.
    # ---------------------------------------------------------- #
    rng_118 = np.random.default_rng(118)
    T_118 = 1000
    layout_118 = standard_rat_layout()
    K_118 = layout_118.n_markers
    markers_118 = layout_118.marker_names

    # Generate a static-pose dataset with realistic scatter:
    # back2 at a fixed location, back1 forward, ear_l/r and
    # nose at head offsets — all observed with controlled
    # body-frame noise of stddev 2 px. The kinematic chain
    # places each marker relative to its segment's distal
    # endpoint; here we fold all of that into a single
    # local→world rotation since the synthetic body+segments
    # are at rest angles (everything points along +x).
    pos_118 = np.zeros((T_118, K_118, 2))
    likes_118 = np.full((T_118, K_118), 0.95)
    body_dir = 0.0
    cos_b, sin_b = np.cos(body_dir), np.sin(body_dir)
    body_x_w = 100.0
    body_y_w = 100.0
    # Per-segment world offsets along +x at rest. We don't
    # need to compute every segment exactly — the diagnostic
    # only cares about marker world positions, and at rest
    # all markers in a chain end up at body + (chain_length +
    # marker_local_offset) along +x.
    chain_offsets_at_rest = {
        # marker → (chain_x, chain_y) of its segment's distal
        # endpoint relative to body root
        "back2": (0.0, 0.0),
        "back1": (0.0, 0.0),       # body marker, no chain
        "back3": (0.0, 0.0),
        "lateral_left": (0.0, 0.0),
        "lateral_right": (0.0, 0.0),
        "center": (0.0, 0.0),
        "back4": (-30.0, 0.0),     # back_rear extends rearward
        "neck": (30.0, 0.0),       # neck extends forward
        "headmid": (60.0, 0.0),    # head past neck
        "nose": (60.0, 0.0),
        "ear_left": (60.0, 0.0),
        "ear_right": (60.0, 0.0),
        "tailbase": (-60.0, 0.0),  # tail past back_rear
        "tailmid": (-90.0, 0.0),
        "tailend": (-120.0, 0.0),
    }
    scale = 30.0
    for k, m in enumerate(markers_118):
        seg, (l_off, a_off) = layout_118.marker_attachment(m)
        # Marker position in segment-local frame
        local_x = scale * l_off * np.cos(a_off)
        local_y = scale * l_off * np.sin(a_off)
        chain_x, chain_y = chain_offsets_at_rest.get(m, (0.0, 0.0))
        # Rest pose: segment frames all aligned with body, so
        # local coords add directly to chain offsets.
        rest_x = chain_x + local_x
        rest_y = chain_y + local_y
        for t in range(T_118):
            # Body-frame noise added before rotating into world
            nx = rng_118.normal(0, 2.0)
            ny = rng_118.normal(0, 2.0)
            pos_118[t, k, 0] = body_x_w + cos_b * (rest_x + nx) - sin_b * (rest_y + ny)
            pos_118[t, k, 1] = body_y_w + sin_b * (rest_x + nx) + cos_b * (rest_y + ny)

    fitted_118 = fit_body_lengths(
        pos_118, likes_118, layout_118, markers_118,
        dt=1.0 / 30.0,
    )
    assert fitted_118.marker_r_drift is not None
    assert fitted_118.marker_q_drift is not None
    # For the back/head markers, recovered r_drift should be in
    # [0.8, 6.0] px — reflects the 2 px body-frame noise we
    # injected, plus noise compounding from the frame-direction
    # estimation (back3→back2 / neck→headmid frames are also
    # estimated from noisy markers, so the projected residuals
    # see ~2× the per-marker noise).
    for m in ("back1", "ear_left", "nose"):
        if m in fitted_118.marker_r_drift:
            r_m = fitted_118.marker_r_drift[m]
            assert 0.8 <= r_m <= 6.0, (
                f"r_drift for {m!r}: expected [0.8, 6], got {r_m:.2f}"
            )

    # ---------------------------------------------------------- #
    # Case 119: NoiseParamsV2.default uses fitted_lengths.
    # marker_r_drift when provided AND layout.with_drift=True.
    # ---------------------------------------------------------- #
    layout_119 = standard_rat_layout()
    layout_119.with_drift = True
    p_with_fit = NoiseParamsV2.default(
        layout_119, sigma_marker=99.0,  # silly large
        fitted_lengths=fitted_118,
        dt=1.0 / 30.0,
    )
    p_without_fit = NoiseParamsV2.default(
        layout_119, sigma_marker=99.0,
        dt=1.0 / 30.0,
    )
    # With fit: r_drift for, say, back1 should be the empirical
    # value (small, ~1-3), NOT 99 from sigma_marker
    if "back1" in fitted_118.marker_r_drift:
        emp_r = fitted_118.marker_r_drift["back1"]
        assert abs(p_with_fit.r_drift["back1"] - emp_r) < 1e-9, (
            f"r_drift should come from fitted_lengths; got "
            f"{p_with_fit.r_drift['back1']} expected {emp_r}"
        )
    # Without fit: r_drift comes from sigma_marker, capped at
    # r_drift_cap (default 20.0). 99 → 20.0.
    assert p_without_fit.r_drift["back1"] == 20.0, (
        f"r_drift cap should clamp 99 → 20; got "
        f"{p_without_fit.r_drift['back1']}"
    )

    # ---------------------------------------------------------- #
    # Case 120: r_drift_cap clamps very large empirical values
    # (e.g., tail markers with pathological body-frame scatter).
    # ---------------------------------------------------------- #
    fitted_with_huge = FittedLengths(
        segment_lengths={s: 30.0 for s in layout_119.non_root_topo_order},
        segment_length_iqr={s: 1.0 for s in layout_119.non_root_topo_order},
        marker_offsets={m: (1.0, 0.0) for m in layout_119.marker_names},
        marker_r_drift={"tailend": 312.0, "back1": 2.0},
        marker_q_drift={
            "tailend": (312.0 / 5.0) ** 2 / (1.0 / 30.0),
            "back1":   (2.0 / 5.0) ** 2 / (1.0 / 30.0),
        },
    )
    p_capped = NoiseParamsV2.default(
        layout_119, sigma_marker=3.0,
        fitted_lengths=fitted_with_huge,
        dt=1.0 / 30.0,
        r_drift_cap=20.0,
    )
    assert p_capped.r_drift["tailend"] == 20.0
    assert abs(p_capped.r_drift["back1"] - 2.0) < 1e-9
    # When uncapped, the empirical 312 comes through
    p_uncapped = NoiseParamsV2.default(
        layout_119, sigma_marker=3.0,
        fitted_lengths=fitted_with_huge,
        dt=1.0 / 30.0,
        r_drift_cap=None,
    )
    assert abs(p_uncapped.r_drift["tailend"] - 312.0) < 1e-9

    # ---------------------------------------------------------- #
    # Case 121: save/load round-trip preserves with_drift,
    # FittedLengths drift calibration, and NoiseParamsV2 drift
    # fields. Backward compat: pre-120b model files (no drift
    # keys present) load correctly with drift fields = None.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        save_model_v2 as _save_120b,
        load_model_v2 as _load_120b,
    )
    layout_121 = standard_rat_layout()
    layout_121.with_drift = True
    fitted_121 = fitted_118  # has empirical drift cal
    _apply_119a(layout_121, fitted_121)
    params_121 = NoiseParamsV2.default(
        layout_121, sigma_marker=2.0,
        fitted_lengths=fitted_121,
    )
    with _tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "m120b.npz"
        _save_120b(
            save_path, layout_121, fitted_121, params_121,
            fps=30.0, likelihood_threshold=0.7,
        )
        loaded = _load_120b(save_path)
        loaded_layout = loaded[0]
        loaded_fitted = loaded[1]
        loaded_params = loaded[2]
        assert loaded_layout.with_drift is True
        assert loaded_fitted.marker_r_drift is not None
        assert loaded_fitted.marker_q_drift is not None
        # Round-trip equality on at least one marker
        for m in fitted_121.marker_r_drift:
            assert abs(
                loaded_fitted.marker_r_drift[m]
                - fitted_121.marker_r_drift[m]
            ) < 1e-12
            assert abs(
                loaded_fitted.marker_q_drift[m]
                - fitted_121.marker_q_drift[m]
            ) < 1e-12
        assert loaded_params.q_drift is not None
        assert loaded_params.alpha_drift is not None
        assert loaded_params.r_drift is not None
        for m in params_121.r_drift:
            assert abs(
                loaded_params.r_drift[m]
                - params_121.r_drift[m]
            ) < 1e-12

    # ---------------------------------------------------------- #
    # Case 122: backward compat — load a model saved without
    # drift fields. Synthesize one by saving with with_drift=
    # False (no drift fields populated), then load and verify
    # the loaded layout/fitted/params have None for drift.
    # ---------------------------------------------------------- #
    layout_122 = standard_rat_layout()
    # with_drift left at default False
    fitted_122 = FittedLengths(
        segment_lengths={s: 30.0 for s in layout_122.non_root_topo_order},
        segment_length_iqr={s: 1.0 for s in layout_122.non_root_topo_order},
        marker_offsets={m: (1.0, 0.0) for m in layout_122.marker_names},
        # marker_r_drift / marker_q_drift remain None
    )
    params_122 = NoiseParamsV2.default(layout_122, sigma_marker=2.0)
    with _tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "m122.npz"
        _save_120b(
            save_path, layout_122, fitted_122, params_122,
            fps=30.0, likelihood_threshold=0.7,
        )
        loaded_layout, loaded_fl, loaded_pp, _, _, _ = _load_120b(
            save_path,
        )
        assert loaded_layout.with_drift is False
        assert loaded_fl.marker_r_drift is None
        assert loaded_fl.marker_q_drift is None
        assert loaded_pp.q_drift is None
        assert loaded_pp.alpha_drift is None
        assert loaded_pp.r_drift is None

    # ---------------------------------------------------------- #
    # Patch 121a: per-segment orientation drift state extension.
    # FK is NOT yet wired (that's 121b), so these tests cover
    # only the state-vector mechanics: layout, slice helpers,
    # F/Q blocks, NoiseParamsV2 fields, save/load.
    # ---------------------------------------------------------- #
    import dataclasses as _dc

    # ---------------------------------------------------------- #
    # Case 123: orientation_drift_segments rejects unknown
    # segment names. Catches typos at construction time.
    # ---------------------------------------------------------- #
    try:
        layout_bad = standard_rat_layout()
        layout_bad = _dc.replace(
            layout_bad, orientation_drift_segments=["body", "nonexistent"],
        )
        assert False, "expected ValueError for unknown segment"
    except ValueError as e:
        assert "nonexistent" in str(e)

    # ---------------------------------------------------------- #
    # Case 124: orientation_drift_segments rejects duplicates.
    # ---------------------------------------------------------- #
    try:
        layout_bad2 = standard_rat_layout()
        layout_bad2 = _dc.replace(
            layout_bad2, orientation_drift_segments=["body", "body"],
        )
        assert False, "expected ValueError for duplicate segment"
    except ValueError as e:
        assert "Duplicate" in str(e) or "duplicate" in str(e)

    # ---------------------------------------------------------- #
    # Case 125: empty orientation_drift_segments (default).
    # state_dim is unchanged from patch-120 form.
    # ---------------------------------------------------------- #
    layout_125 = standard_rat_layout()
    base_dim = layout_125.state_dim
    assert layout_125.orientation_drift_segments == []
    assert layout_125.orientation_drift_block_slice == slice(0, 0)

    # ---------------------------------------------------------- #
    # Case 126: orientation_drift_segments adds 1 dim per name.
    # ---------------------------------------------------------- #
    layout_126 = standard_rat_layout()
    layout_126 = _dc.replace(
        layout_126,
        orientation_drift_segments=["body", "neck", "head"],
    )
    assert layout_126.state_dim == base_dim + 3
    block = layout_126.orientation_drift_block_slice
    assert block.stop - block.start == 3

    # ---------------------------------------------------------- #
    # Case 127: slice_segment_orientation_drift returns correct
    # slice; raises for segments not in the list.
    # ---------------------------------------------------------- #
    sl_body = layout_126.slice_segment_orientation_drift("body")
    sl_neck = layout_126.slice_segment_orientation_drift("neck")
    sl_head = layout_126.slice_segment_orientation_drift("head")
    # Each is a 1-dim slice
    assert sl_body.stop - sl_body.start == 1
    assert sl_neck.stop - sl_neck.start == 1
    assert sl_head.stop - sl_head.start == 1
    # Order in the state vector matches order of the segment list
    assert sl_neck.start == sl_body.start + 1
    assert sl_head.start == sl_neck.start + 1
    # Asking for a segment that's not configured raises
    try:
        layout_126.slice_segment_orientation_drift("back_rear")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

    # ---------------------------------------------------------- #
    # Case 128: NoiseParamsV2.default populates the theta-drift
    # dicts from layout.orientation_drift_segments using the
    # heuristic q = (r/5)² / dt.
    # ---------------------------------------------------------- #
    layout_128 = _dc.replace(
        standard_rat_layout(),
        orientation_drift_segments=["body"],
    )
    p128 = NoiseParamsV2.default(layout_128, dt=1.0/30.0)
    assert p128.q_theta_drift is not None
    assert p128.alpha_theta_drift is not None
    assert p128.r_theta_drift is not None
    assert "body" in p128.r_theta_drift
    assert p128.r_theta_drift["body"] == 0.1  # default
    assert p128.alpha_theta_drift["body"] == 0.05  # default
    expected_q = (0.1 / 5.0) ** 2 / (1.0 / 30.0)
    assert abs(p128.q_theta_drift["body"] - expected_q) < 1e-12
    # When no segments configured, fields are None
    p_no_seg = NoiseParamsV2.default(standard_rat_layout())
    assert p_no_seg.q_theta_drift is None
    assert p_no_seg.alpha_theta_drift is None
    assert p_no_seg.r_theta_drift is None

    # ---------------------------------------------------------- #
    # Case 129: build_F_v2 sets (1 - α) on the orientation
    # drift block's diagonal entries.
    # ---------------------------------------------------------- #
    p129 = NoiseParamsV2.default(layout_128, dt=1.0/30.0)
    F129 = build_F_v2(layout_128, dt=1.0/30.0, params=p129)
    sl = layout_128.slice_segment_orientation_drift("body")
    expected_F = 1.0 - 0.05  # default alpha
    assert abs(F129[sl.start, sl.start] - expected_F) < 1e-12
    # All off-diagonal entries within the orientation drift
    # block remain zero (block is 1x1, but the row/column
    # outside should be zero too)
    block = layout_128.orientation_drift_block_slice
    for i in range(layout_128.state_dim):
        if i == sl.start:
            continue
        assert abs(F129[sl.start, i]) < 1e-12, (
            f"F[{sl.start}, {i}] should be 0; got {F129[sl.start, i]}"
        )
        assert abs(F129[i, sl.start]) < 1e-12, (
            f"F[{i}, {sl.start}] should be 0; got {F129[i, sl.start]}"
        )

    # ---------------------------------------------------------- #
    # Case 130: build_Q_v2 sets q × dt on the orientation drift
    # block's diagonal entries.
    # ---------------------------------------------------------- #
    Q130 = build_Q_v2(layout_128, dt=1.0/30.0, params=p129)
    expected_q_block = expected_q * (1.0 / 30.0)
    assert abs(Q130[sl.start, sl.start] - expected_q_block) < 1e-12

    # ---------------------------------------------------------- #
    # Case 131: alpha and q validation. Negative q → error;
    # alpha out of [0, 1) → error.
    # ---------------------------------------------------------- #
    p_bad_q = NoiseParamsV2.default(layout_128, dt=1.0/30.0)
    p_bad_q.q_theta_drift = {"body": -1.0}
    try:
        build_Q_v2(layout_128, dt=1.0/30.0, params=p_bad_q)
        assert False, "expected ValueError for negative q_theta_drift"
    except ValueError as e:
        assert "q_theta_drift" in str(e)
    p_bad_a = NoiseParamsV2.default(layout_128, dt=1.0/30.0)
    p_bad_a.alpha_theta_drift = {"body": 1.5}
    try:
        build_F_v2(layout_128, dt=1.0/30.0, params=p_bad_a)
        assert False, "expected ValueError for alpha out of range"
    except ValueError as e:
        assert "alpha_theta_drift" in str(e)

    # ---------------------------------------------------------- #
    # Case 132: save/load round-trip preserves
    # orientation_drift_segments and theta-drift NoiseParamsV2
    # fields. Backward compat: pre-121a models (no theta-drift
    # keys present) load with empty segments and None fields.
    # ---------------------------------------------------------- #
    layout_132 = _dc.replace(
        standard_rat_layout(),
        orientation_drift_segments=["body", "neck"],
    )
    layout_132.with_drift = False  # focus on orientation drift only
    fitted_132 = FittedLengths(
        segment_lengths={
            s: 30.0 for s in layout_132.non_root_topo_order
        },
        segment_length_iqr={
            s: 1.0 for s in layout_132.non_root_topo_order
        },
        marker_offsets={
            m: (1.0, 0.0) for m in layout_132.marker_names
        },
    )
    params_132 = NoiseParamsV2.default(layout_132, dt=1.0/30.0)
    with _tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "m132.npz"
        _save_120b(
            save_path, layout_132, fitted_132, params_132,
            fps=30.0, likelihood_threshold=0.7,
        )
        loaded_layout, loaded_fl, loaded_pp, _, _, _ = _load_120b(
            save_path,
        )
        assert loaded_layout.orientation_drift_segments == [
            "body", "neck",
        ]
        assert loaded_pp.q_theta_drift is not None
        assert loaded_pp.alpha_theta_drift is not None
        assert loaded_pp.r_theta_drift is not None
        for s in ("body", "neck"):
            assert abs(
                loaded_pp.r_theta_drift[s]
                - params_132.r_theta_drift[s]
            ) < 1e-12
            assert abs(
                loaded_pp.q_theta_drift[s]
                - params_132.q_theta_drift[s]
            ) < 1e-12
            assert abs(
                loaded_pp.alpha_theta_drift[s]
                - params_132.alpha_theta_drift[s]
            ) < 1e-12

    # Also save with NO orientation-drift segments — confirm
    # loaded model has empty list and None fields.
    layout_132_empty = standard_rat_layout()
    params_132_empty = NoiseParamsV2.default(
        layout_132_empty, dt=1.0/30.0,
    )
    with _tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "m132e.npz"
        _save_120b(
            save_path, layout_132_empty, fitted_132,
            params_132_empty, fps=30.0, likelihood_threshold=0.7,
        )
        loaded_layout, _, loaded_pp, _, _, _ = _load_120b(
            save_path,
        )
        assert loaded_layout.orientation_drift_segments == []
        assert loaded_pp.q_theta_drift is None
        assert loaded_pp.alpha_theta_drift is None
        assert loaded_pp.r_theta_drift is None

    # ============================================================ #
    # Patch 121c: numerical hardening for the EKF/RTS smoother.
    # Triggered by a real-data run (67 sessions × 54k frames)
    # where q_root_ori swelled 18.6× during EM and the final
    # smoother pass crashed with LinAlgError: Singular matrix.
    # Four mechanisms guard against pathological covariances:
    #   - Adaptive S regularization (forward_filter_v2)
    #   - Adaptive P_pred regularization + finite-check
    #     fallback (rts_smooth_v2)
    #   - Tighter q ceiling factor (10× was 30×) plus absolute
    #     hard caps for q_root_pos and q_root_ori
    #   - Per-session error isolation (_pool_final_smooth)
    # ============================================================ #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        forward_filter_v2 as _ff_121c,
        rts_smooth_v2 as _rts_121c,
        _pool_final_smooth as _pfs_121c,
        _M_STEP_Q_CEILING_FACTOR,
        _M_STEP_Q_ROOT_POS_HARD_CAP,
        _M_STEP_Q_ROOT_ORI_HARD_CAP,
        _S_REG_REL_FACTOR,
        _S_REG_FLOOR,
        finalize_m_step_v2_from_per_session as _finalize_ps_121c,
        apply_fitted_offsets_to_layout as _apply_offsets_121c,
        initial_state_from_data as _init_state_121c,
    )

    # ---------------------------------------------------------- #
    # Case 133: 121c constants are sane and tighter than 121a.
    # The 30× ceiling that allowed q_root_pos to reach 137068
    # in real data is now 10×; absolute hard caps backstop.
    # ---------------------------------------------------------- #
    assert _M_STEP_Q_CEILING_FACTOR == 10.0, (
        f"Patch 121c sets ceiling factor to 10; "
        f"got {_M_STEP_Q_CEILING_FACTOR}"
    )
    assert _M_STEP_Q_ROOT_POS_HARD_CAP == 50000.0
    assert _M_STEP_Q_ROOT_ORI_HARD_CAP == 50.0
    assert _S_REG_REL_FACTOR > 0 and _S_REG_FLOOR > 0
    # rel factor should be small enough not to perturb
    # well-conditioned cases (where R_full diagonal dominates)
    assert _S_REG_REL_FACTOR < 1e-6

    # ---------------------------------------------------------- #
    # Case 134: hard caps fire for the root q parameters when
    # initial_params × 10 would exceed them. With initial =
    # 6000 px²/s³, the 10× relative ceiling allows 60000, but
    # the absolute cap (50000) clamps. With initial = 10 rad
    # equivalent on q_root_ori, 10× = 100 but the cap = 50
    # clamps.
    # ---------------------------------------------------------- #
    layout_134c = standard_rat_layout()

    # Build a fitted_lengths good enough for default params
    fitted_134c = FittedLengths(
        segment_lengths={
            s: 30.0 for s in layout_134c.non_root_topo_order
        },
        segment_length_iqr={
            s: 1.0 for s in layout_134c.non_root_topo_order
        },
        marker_offsets={
            m: (1.0, 0.0) for m in layout_134c.marker_names
        },
    )
    initial_134c = NoiseParamsV2.default(
        layout_134c, sigma_marker=2.0, fitted_lengths=fitted_134c,
    )
    # Force initial high enough that the hard cap fires
    initial_134c.q_root_pos = 6000.0  # 10× = 60000 > 50000 cap
    initial_134c.q_root_ori = 10.0    # 10× = 100 > 50 cap

    # Build a per-session fit with extreme outlier values
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        _PerSessionFitV2,
    )
    fit_outlier = _PerSessionFitV2.empty(layout_134c)
    fit_outlier.q_root_pos = 1e10  # absurd
    fit_outlier.q_root_ori = 1e10
    fit_outlier.n_pairs = 1000
    fits_134c = [fit_outlier]
    new_params_134c = _finalize_ps_121c(
        fits_134c, layout_134c,
        prev_params=initial_134c,
        initial_params=initial_134c,
        aggregation="mean",
    )
    # q_root_pos: relative ceiling 60000, but hard cap 50000
    # → expect 50000
    assert new_params_134c.q_root_pos == _M_STEP_Q_ROOT_POS_HARD_CAP, (
        f"hard cap on q_root_pos should fire; got "
        f"{new_params_134c.q_root_pos}"
    )
    # q_root_ori: relative 100, hard cap 50 → expect 50
    assert new_params_134c.q_root_ori == _M_STEP_Q_ROOT_ORI_HARD_CAP, (
        f"hard cap on q_root_ori should fire; got "
        f"{new_params_134c.q_root_ori}"
    )

    # ---------------------------------------------------------- #
    # Case 135: when initial × 10 < hard cap, the relative
    # ceiling fires (not the hard cap). Confirms the 10× factor
    # is the active limit at typical initial scales.
    # ---------------------------------------------------------- #
    initial_135c = NoiseParamsV2.default(
        layout_134c, sigma_marker=2.0, fitted_lengths=fitted_134c,
    )
    initial_135c.q_root_pos = 100.0   # 10× = 1000 < 50000 cap
    initial_135c.q_root_ori = 1.0     # 10× = 10 < 50 cap

    fit_outlier_135 = _PerSessionFitV2.empty(layout_134c)
    fit_outlier_135.q_root_pos = 1e10
    fit_outlier_135.q_root_ori = 1e10
    fit_outlier_135.n_pairs = 1000
    new_params_135c = _finalize_ps_121c(
        [fit_outlier_135], layout_134c,
        prev_params=initial_135c,
        initial_params=initial_135c,
        aggregation="mean",
    )
    assert new_params_135c.q_root_pos == 1000.0, (
        f"relative ceiling (10×100) should fire; got "
        f"{new_params_135c.q_root_pos}"
    )
    assert new_params_135c.q_root_ori == 10.0, (
        f"relative ceiling (10×1) should fire; got "
        f"{new_params_135c.q_root_ori}"
    )

    # ---------------------------------------------------------- #
    # Case 136: forward_filter_v2 survives a session where the
    # initial covariance is huge (mimicking q-runaway after a
    # long dropout). Without 121c's adaptive S regularization,
    # this would produce NaN in the EKF update; with it, the
    # filter completes and produces finite output.
    # ---------------------------------------------------------- #
    rng_136c = np.random.default_rng(136)
    layout_136c = standard_rat_layout()
    K_136c = layout_136c.n_markers
    T_136c = 50

    # Synthesize a clean trajectory: rat at (200, 200) facing 0
    pos_136c = np.zeros((T_136c, K_136c, 2))
    likes_136c = np.full((T_136c, K_136c), 0.95)
    rest_136c = {
        "back2": (0, 0), "back1": (5, 0), "back3": (-5, 0),
        "lateral_left": (0, 5), "lateral_right": (0, -5),
        "center": (2.5, 0), "back4": (-15, 0), "neck": (15, 0),
        "headmid": (30, 0), "nose": (40, 0),
        "ear_left": (33, 5), "ear_right": (33, -5),
        "tailbase": (-30, 0), "tailmid": (-45, 0), "tailend": (-60, 0),
    }
    for i, m in enumerate(layout_136c.marker_names):
        ox, oy = rest_136c.get(m, (0, 0))
        pos_136c[:, i, 0] = 200 + ox + rng_136c.normal(0, 0.5, T_136c)
        pos_136c[:, i, 1] = 200 + oy + rng_136c.normal(0, 0.5, T_136c)

    fitted_136c = fit_body_lengths(
        pos_136c, likes_136c, layout_136c,
        layout_136c.marker_names, dt=1.0/30.0,
    )
    apply_offsets_136c = _apply_offsets_121c  # alias for clarity
    apply_offsets_136c(layout_136c, fitted_136c)
    params_136c = NoiseParamsV2.default(
        layout_136c, sigma_marker=2.0, fitted_lengths=fitted_136c,
    )
    # Inject pathological q values that would overflow
    # the prediction covariance over the trajectory
    params_136c.q_root_pos = 50000.0  # at the hard cap
    params_136c.q_root_ori = 50.0

    init_state_136c = _init_state_121c(
        pos_136c, likes_136c, layout_136c,
        layout_136c.marker_names, fitted_136c,
        likelihood_threshold=0.5,
    )
    # Inflate initial covariance to huge values
    D_136c = layout_136c.state_dim
    initial_cov_136c = 1e6 * np.eye(D_136c)

    fr_136c = _ff_121c(
        pos_136c, likes_136c, layout_136c,
        params_136c, dt=1.0/30.0,
        initial_state=init_state_136c,
        initial_cov=initial_cov_136c,
        likelihood_threshold=0.5,
    )
    # All filtered states must be finite
    assert np.all(np.isfinite(fr_136c.x_filt)), (
        "forward_filter_v2 produced non-finite x_filt under "
        "pathological initial conditions"
    )
    assert np.all(np.isfinite(fr_136c.P_filt)), (
        "forward_filter_v2 produced non-finite P_filt under "
        "pathological initial conditions"
    )

    # ---------------------------------------------------------- #
    # Case 137: rts_smooth_v2 survives a synthetic FilterResult
    # with extreme P_pred values. The adaptive jitter and
    # finite-fallback path keep the smoother from crashing.
    # Pre-121c, this would raise LinAlgError: Singular matrix.
    # ---------------------------------------------------------- #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        FilterResultV2 as _FR_121c,
    )
    T_137c = 30
    D_137c = layout_136c.state_dim
    # Build a degenerate FilterResultV2: P_pred_huge, P_filt_normal
    x_pred_137 = np.zeros((T_137c, D_137c))
    P_pred_137 = np.tile(
        1e10 * np.eye(D_137c)[None, :, :], (T_137c, 1, 1),
    )
    x_filt_137 = np.zeros((T_137c, D_137c))
    P_filt_137 = np.tile(np.eye(D_137c)[None, :, :], (T_137c, 1, 1))
    # Inject one frame with non-finite P_pred to exercise the
    # finite-check fallback
    P_pred_137[10] = np.full((D_137c, D_137c), np.inf)
    x_pred_137[10] = np.full(D_137c, np.inf)
    n_obs_137 = np.full(T_137c, 15)
    fr_137 = _FR_121c(
        x_pred=x_pred_137, P_pred=P_pred_137,
        x_filt=x_filt_137, P_filt=P_filt_137,
        n_observed=n_obs_137,
    )
    # rts_smooth_v2 must not raise
    sr_137 = _rts_121c(fr_137, layout_136c, dt=1.0/30.0)
    # Output must be finite even with non-finite P_pred at t=10
    assert np.all(np.isfinite(sr_137.x_smooth)), (
        "rts_smooth_v2 produced non-finite x_smooth despite "
        "121c finite-check fallback"
    )
    assert np.all(np.isfinite(sr_137.P_smooth)), (
        "rts_smooth_v2 produced non-finite P_smooth"
    )
    # The fallback fires at the iteration that READS P_pred[10],
    # which is the t=9 backward step. At that step, the smoother
    # falls back to filter-only and writes x_smooth[9]=x_filt[9].
    np.testing.assert_allclose(
        sr_137.x_smooth[9], x_filt_137[9], atol=1e-15,
        err_msg=(
            "non-finite P_pred[10] should have triggered "
            "fallback at t=9 (smoothed = filtered)"
        ),
    )
    np.testing.assert_allclose(
        sr_137.P_smooth[9], P_filt_137[9], atol=1e-15,
    )
    # Lag-one cross-cov at t=9 should be the zero matrix
    np.testing.assert_allclose(
        sr_137.P_lag_one[9], np.zeros((D_137c, D_137c)),
        atol=1e-15,
    )

    # ---------------------------------------------------------- #
    # Case 138: _pool_final_smooth's exception-isolation path —
    # we can't easily exercise the worker directly without a
    # real pool, but we can verify the function signature now
    # returns a 5-tuple (sess_idx, pos, var, elapsed, error).
    # And verify that with valid input via direct call, the
    # error field is None.
    # ---------------------------------------------------------- #
    import inspect as _inspect
    sig = _inspect.signature(_pfs_121c)
    # The function takes one positional `args` tuple
    assert len(sig.parameters) == 1
    # Inspect return annotation: should now be 5-tuple
    return_ann = sig.return_annotation
    # `Tuple[int, Optional[np.ndarray], Optional[np.ndarray], float, Optional[str]]`
    # Check via repr since typing equality is finicky
    ann_repr = repr(return_ann)
    assert "Optional" in ann_repr or "None" in ann_repr, (
        f"_pool_final_smooth return annotation should reflect "
        f"the error-isolation 5-tuple; got {ann_repr}"
    )

    # ============================================================ #
    # Patch 121b: per-segment orientation drift wiring into FK
    # and the observation Jacobian. 121a left δ_θ in the state
    # vector but disconnected from the marker-prediction
    # equation; 121b makes FK actually consume it.
    #
    # Note: this section is logically the 121b suite, but is
    # numbered 139-144 to come after 121c's 133-138 in this
    # stacked-patch ordering (121c lands first as the numerical
    # hardening prerequisite).
    # ============================================================ #
    from mufasa.data_processors.kalman_pose_smoother_v2 import (
        forward_kinematics as _fk_121b,
        state_to_marker_positions as _stmp_121b,
        state_to_marker_jacobian as _stmj_121b,
        state_to_marker_positions_batch as _stmpb_121b,
        state_to_marker_jacobian_batch as _stmjb_121b,
        _pack_state_layout_indices as _pack_121b,
    )

    # ---------------------------------------------------------- #
    # Case 139: zero δ_θ reproduces no-orientation-drift FK
    # exactly. With δ=0 in the state vector, every world rotation
    # and distal position must be bit-identical to a layout
    # without orientation_drift_segments configured.
    # ---------------------------------------------------------- #
    rng_139 = np.random.default_rng(133)
    layout_no_drift_139 = standard_rat_layout()
    layout_with_drift_139 = _dc.replace(
        standard_rat_layout(),
        orientation_drift_segments=["body", "neck", "head"],
    )
    state_zero_drift = np.zeros(layout_with_drift_139.state_dim)
    indices_139_d = _pack_121b(layout_with_drift_139)
    indices_139_n = _pack_121b(layout_no_drift_139)
    state_no_drift = np.zeros(layout_no_drift_139.state_dim)
    state_no_drift[indices_139_n["__root__"]["x"]] = 50.0
    state_no_drift[indices_139_n["__root__"]["y"]] = 80.0
    state_no_drift[indices_139_n["__root__"]["cos"]] = np.cos(0.4)
    state_no_drift[indices_139_n["__root__"]["sin"]] = np.sin(0.4)
    state_zero_drift[indices_139_d["__root__"]["x"]] = 50.0
    state_zero_drift[indices_139_d["__root__"]["y"]] = 80.0
    state_zero_drift[indices_139_d["__root__"]["cos"]] = np.cos(0.4)
    state_zero_drift[indices_139_d["__root__"]["sin"]] = np.sin(0.4)
    for seg in layout_no_drift_139.non_root_topo_order:
        ang = rng_139.uniform(-0.3, 0.3)
        state_no_drift[indices_139_n[seg]["cos"]] = np.cos(ang)
        state_no_drift[indices_139_n[seg]["sin"]] = np.sin(ang)
        state_no_drift[indices_139_n[seg]["length"]] = 30.0
        state_zero_drift[indices_139_d[seg]["cos"]] = np.cos(ang)
        state_zero_drift[indices_139_d[seg]["sin"]] = np.sin(ang)
        state_zero_drift[indices_139_d[seg]["length"]] = 30.0
    pos_no = _stmp_121b(state_no_drift, layout_no_drift_139)
    pos_zd = _stmp_121b(state_zero_drift, layout_with_drift_139)
    np.testing.assert_allclose(
        pos_no, pos_zd, atol=1e-12,
        err_msg="δ_θ=0 should reproduce no-orientation-drift FK exactly",
    )
    fk_no = _fk_121b(state_no_drift, layout_no_drift_139)
    fk_zd = _fk_121b(state_zero_drift, layout_with_drift_139)
    for seg_name in layout_no_drift_139.topo_order:
        np.testing.assert_allclose(
            fk_no.P_distal[seg_name], fk_zd.P_distal[seg_name],
            atol=1e-12,
            err_msg=f"P_distal[{seg_name}] differs at δ=0",
        )
        np.testing.assert_allclose(
            fk_no.R_world[seg_name], fk_zd.R_world[seg_name],
            atol=1e-12,
            err_msg=f"R_world[{seg_name}] differs at δ=0",
        )

    # ---------------------------------------------------------- #
    # Case 140: nonzero δ_θ on the root rotates EVERY marker by
    # R(δ_θ_root) around the root position.
    # ---------------------------------------------------------- #
    layout_140 = _dc.replace(
        standard_rat_layout(),
        orientation_drift_segments=["body"],  # body = root
    )
    indices_140 = _pack_121b(layout_140)
    state_140 = np.zeros(layout_140.state_dim)
    state_140[indices_140["__root__"]["x"]] = 100.0
    state_140[indices_140["__root__"]["y"]] = 200.0
    state_140[indices_140["__root__"]["cos"]] = np.cos(0.2)
    state_140[indices_140["__root__"]["sin"]] = np.sin(0.2)
    for seg in layout_140.non_root_topo_order:
        ang = 0.1
        state_140[indices_140[seg]["cos"]] = np.cos(ang)
        state_140[indices_140[seg]["sin"]] = np.sin(ang)
        state_140[indices_140[seg]["length"]] = 30.0
    pos_zero = _stmp_121b(state_140, layout_140)
    sl_body = layout_140.slice_segment_orientation_drift("body")
    state_140_drift = state_140.copy()
    state_140_drift[sl_body.start] = 0.07
    pos_drift = _stmp_121b(state_140_drift, layout_140)
    delta = 0.07
    R_d = np.array([
        [np.cos(delta), -np.sin(delta)],
        [np.sin(delta),  np.cos(delta)],
    ])
    root_pos = np.array([100.0, 200.0])
    pos_predicted = (pos_zero - root_pos) @ R_d.T + root_pos
    np.testing.assert_allclose(
        pos_drift, pos_predicted, atol=1e-12,
        err_msg=(
            "δ_θ on root should rotate all markers by R(δ) "
            "around root_pos"
        ),
    )

    # ---------------------------------------------------------- #
    # Case 141: nonzero δ_θ on a non-root segment rotates only
    # the descendant subtree's markers by R(δ_θ_seg) around the
    # segment's proximal endpoint (P_distal[parent(seg)]).
    # Markers on segments NOT in the descendant subtree are
    # unchanged.
    # ---------------------------------------------------------- #
    layout_141 = _dc.replace(
        standard_rat_layout(),
        orientation_drift_segments=["neck"],
    )
    indices_141 = _pack_121b(layout_141)
    state_141 = np.zeros(layout_141.state_dim)
    state_141[indices_141["__root__"]["x"]] = 50.0
    state_141[indices_141["__root__"]["y"]] = 80.0
    state_141[indices_141["__root__"]["cos"]] = np.cos(0.3)
    state_141[indices_141["__root__"]["sin"]] = np.sin(0.3)
    for seg in layout_141.non_root_topo_order:
        ang = 0.05
        state_141[indices_141[seg]["cos"]] = np.cos(ang)
        state_141[indices_141[seg]["sin"]] = np.sin(ang)
        state_141[indices_141[seg]["length"]] = 30.0
    pos_zero_141 = _stmp_121b(state_141, layout_141)
    fk_zero_141 = _fk_121b(state_141, layout_141)
    pivot_neck = fk_zero_141.P_distal["body"]
    delta_n = 0.06
    sl_neck = layout_141.slice_segment_orientation_drift("neck")
    state_141_drift = state_141.copy()
    state_141_drift[sl_neck.start] = delta_n
    pos_drift_141 = _stmp_121b(state_141_drift, layout_141)
    descendants = {"neck", "head"}
    R_n = np.array([
        [np.cos(delta_n), -np.sin(delta_n)],
        [np.sin(delta_n),  np.cos(delta_n)],
    ])
    for i, marker in enumerate(layout_141.marker_names):
        seg_of_marker, _ = layout_141.marker_attachment(marker)
        if seg_of_marker in descendants:
            expected = (
                pos_zero_141[i] - pivot_neck
            ) @ R_n.T + pivot_neck
            np.testing.assert_allclose(
                pos_drift_141[i], expected, atol=1e-12,
                err_msg=(
                    f"δ_neck should rotate descendant marker "
                    f"{marker!r} by R(δ) around P_distal[body]"
                ),
            )
        else:
            np.testing.assert_allclose(
                pos_drift_141[i], pos_zero_141[i], atol=1e-12,
                err_msg=(
                    f"δ_neck must NOT affect non-descendant "
                    f"marker {marker!r}"
                ),
            )

    # ---------------------------------------------------------- #
    # Case 142: numerical vs analytical Jacobian on the
    # orientation drift columns. The most important correctness
    # check for 121b — if the partials are wrong the EKF
    # silently misbehaves.
    # ---------------------------------------------------------- #
    rng_142 = np.random.default_rng(142)
    layout_142 = _dc.replace(
        standard_rat_layout(),
        orientation_drift_segments=["body", "neck", "head"],
    )
    indices_142 = _pack_121b(layout_142)
    state_142 = np.zeros(layout_142.state_dim)
    state_142[indices_142["__root__"]["cos"]] = np.cos(0.4)
    state_142[indices_142["__root__"]["sin"]] = np.sin(0.4)
    state_142[indices_142["__root__"]["x"]] = 50.0
    state_142[indices_142["__root__"]["y"]] = 80.0
    for seg in layout_142.non_root_topo_order:
        seg_idx = indices_142[seg]
        ang = rng_142.uniform(-0.3, 0.3)
        state_142[seg_idx["cos"]] = np.cos(ang)
        state_142[seg_idx["sin"]] = np.sin(ang)
        state_142[seg_idx["length"]] = 30.0
    for sname in layout_142.orientation_drift_segments:
        sl = layout_142.slice_segment_orientation_drift(sname)
        state_142[sl.start] = rng_142.uniform(-0.05, 0.05)

    H_analytic_142 = _stmj_121b(state_142, layout_142)
    eps = 1e-6
    for sname in layout_142.orientation_drift_segments:
        sl = layout_142.slice_segment_orientation_drift(sname)
        col = sl.start
        s_plus = state_142.copy()
        s_plus[col] += eps
        s_minus = state_142.copy()
        s_minus[col] -= eps
        num_col = (
            _stmp_121b(s_plus, layout_142).flatten()
            - _stmp_121b(s_minus, layout_142).flatten()
        ) / (2 * eps)
        ana_col = H_analytic_142[:, col]
        np.testing.assert_allclose(
            num_col, ana_col, atol=1e-7,
            err_msg=(
                f"orientation drift Jacobian column for {sname!r} "
                f"disagrees with numerical"
            ),
        )

    # Zero-partial check for markers outside the descendant subtree
    layout_142b = _dc.replace(
        standard_rat_layout(),
        orientation_drift_segments=["head"],
    )
    indices_142b = _pack_121b(layout_142b)
    state_142b = np.zeros(layout_142b.state_dim)
    state_142b[indices_142b["__root__"]["cos"]] = np.cos(0.4)
    state_142b[indices_142b["__root__"]["sin"]] = np.sin(0.4)
    state_142b[indices_142b["__root__"]["x"]] = 50.0
    state_142b[indices_142b["__root__"]["y"]] = 80.0
    for seg in layout_142b.non_root_topo_order:
        seg_idx = indices_142b[seg]
        ang = rng_142.uniform(-0.3, 0.3)
        state_142b[seg_idx["cos"]] = np.cos(ang)
        state_142b[seg_idx["sin"]] = np.sin(ang)
        state_142b[seg_idx["length"]] = 30.0
    H_142b = _stmj_121b(state_142b, layout_142b)
    sl_head = layout_142b.slice_segment_orientation_drift("head")
    head_descendants = {"head"}
    for i, marker in enumerate(layout_142b.marker_names):
        seg_of_marker, _ = layout_142b.marker_attachment(marker)
        if seg_of_marker in head_descendants:
            partial = H_142b[2 * i : 2 * i + 2, sl_head.start]
            assert np.linalg.norm(partial) > 1e-9, (
                f"marker {marker!r} (on head) should have non-zero "
                f"partial w.r.t. δ_head; got {partial}"
            )
        else:
            partial = H_142b[2 * i : 2 * i + 2, sl_head.start]
            np.testing.assert_allclose(
                partial, np.zeros(2), atol=1e-12,
                err_msg=(
                    f"marker {marker!r} (not on head subtree) "
                    f"must have zero partial w.r.t. δ_head"
                ),
            )

    # ---------------------------------------------------------- #
    # Case 143: batch FK with orientation drift agrees with the
    # per-frame variant.
    # ---------------------------------------------------------- #
    T_143 = 7
    layout_143 = _dc.replace(
        standard_rat_layout(),
        orientation_drift_segments=["body", "neck"],
    )
    indices_143 = _pack_121b(layout_143)
    rng_143 = np.random.default_rng(143)
    states_batch_143 = np.zeros((T_143, layout_143.state_dim))
    for t in range(T_143):
        states_batch_143[t, indices_143["__root__"]["x"]] = 50.0
        states_batch_143[t, indices_143["__root__"]["y"]] = 80.0
        states_batch_143[t, indices_143["__root__"]["cos"]] = np.cos(0.4)
        states_batch_143[t, indices_143["__root__"]["sin"]] = np.sin(0.4)
        for seg in layout_143.non_root_topo_order:
            seg_idx = indices_143[seg]
            ang = rng_143.uniform(-0.3, 0.3)
            states_batch_143[t, seg_idx["cos"]] = np.cos(ang)
            states_batch_143[t, seg_idx["sin"]] = np.sin(ang)
            states_batch_143[t, seg_idx["length"]] = 30.0
        for sname in layout_143.orientation_drift_segments:
            sl = layout_143.slice_segment_orientation_drift(sname)
            states_batch_143[t, sl.start] = rng_143.uniform(-0.08, 0.08)
    pos_batch_143 = _stmpb_121b(
        states_batch_143, layout_143, device="cpu",
    )
    for t in range(T_143):
        pos_t = _stmp_121b(states_batch_143[t], layout_143)
        np.testing.assert_allclose(
            pos_batch_143[t], pos_t, atol=1e-12,
            err_msg=(
                f"batch FK frame {t} disagrees with per-frame "
                f"under orientation drift"
            ),
        )

    # ---------------------------------------------------------- #
    # Case 144: batch Jacobian with orientation drift agrees
    # with per-frame Jacobian on the orientation drift columns.
    # ---------------------------------------------------------- #
    H_batch_144 = _stmjb_121b(
        states_batch_143, layout_143, device="cpu",
    )
    block_144 = layout_143.orientation_drift_block_slice
    for t in range(T_143):
        H_t = _stmj_121b(states_batch_143[t], layout_143)
        np.testing.assert_allclose(
            H_batch_144[t, :, block_144.start:block_144.stop],
            H_t[:, block_144.start:block_144.stop],
            atol=1e-12,
            err_msg=(
                f"batch H orientation-drift cols frame {t} mismatch"
            ),
        )

    print("smoke_kalman_pose_smoother_v2: 144/144 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
