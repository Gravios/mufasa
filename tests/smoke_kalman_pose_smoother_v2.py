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

    print("smoke_kalman_pose_smoother_v2: 19/19 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
