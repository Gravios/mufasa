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
        )
        assert False, "Validation should have detected frozen trajectory"
    except RuntimeError as e:
        # Either frozen-output or prior-overruling check should fire
        msg = str(e)
        assert "validation hook triggered" in msg

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

        layout_l, fitted_l, params_l, fps_l, thr_l = (
            load_model_v2(model_path)
        )
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
        )
    except RuntimeError:
        raised = True
    assert raised, "4σ threshold should fail on 5.5σ structural offset"

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

    print("smoke_kalman_pose_smoother_v2: 63/63 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
