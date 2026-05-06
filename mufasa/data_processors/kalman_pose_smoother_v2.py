"""
mufasa.data_processors.kalman_pose_smoother_v2
==============================================

Joint-state Kalman smoother for pose data with kinematic
constraints (v2). Sister module to ``kalman_pose_smoother``
(v1) which treats markers as independent. v2 treats markers
as observations of a kinematic tree where segments share
state through forward kinematics, so spatial coupling
constrains the per-frame marginals — addressing the
"markers wander during low-p periods" failure mode of v1.

This file (patch 99, the foundation) defines:

  - ``BodySegment``: one segment of the kinematic tree
  - ``BodyLayout``: the full topology + state packing convention
  - ``standard_rat_layout``: a default mufasa rat skeleton
  - ``FittedLengths``: per-segment fitted lengths + length-σ
  - ``fit_body_lengths``: estimate segment lengths from data

Subsequent patches add:
  - patch 100 — forward kinematics and observation function
  - patch 101 — EKF forward filter
  - patch 102 — RTS smoother backward pass
  - patch 103 — Shumway-Stoffer M-step on body state
  - patch 104 — EM loop + validation hook
  - patch 105 — orchestrator + CLI + save/load
  - patch 106 — end-to-end tests + comparison vs v1

Design choices (locked in patch 99, see commit message)
-------------------------------------------------------

1. Body-attached kinematic chain: each non-root segment's
   orientation is parameterized as a unit vector in its
   parent's frame. A "head turn" is a small angle relative
   to neck; a "rat changes facing direction" is a rotation
   of the root frame. This encodes the biomechanically
   correct prior that head-vs-body angles are tightly bounded
   while body-vs-world is unconstrained.

2. Unit-vector orientation parameterization: each angle is
   stored as (cos θ, sin θ). Costs +1 state dim per angle
   vs. pure-angle but eliminates wraparound bugs and avoids
   special-casing in the EKF math. Norm-1 constraint enforced
   softly via observation-noise scaling (for now); future
   patches may add a hard projection step in the M-step.

3. Slow-varying segment lengths: each length is an additional
   state with a small process noise q, allowing it to drift
   slowly across frames to absorb animal-state variation
   (rearing changes apparent length under top-down projection,
   breathing slightly changes some distances). Within a session
   the EM-fit q is tiny so lengths are nearly constant; across
   sessions they re-fit to the new animal.

4. Tree topology: the standard mufasa rat layout assumes a
   rigid trunk (back1 / back2 / back3 / back4 / lateral_left /
   lateral_right / center are all rigidly attached to the body
   root) with pivoting neck → head → nose chain forward and
   tailbase → tailmid → tailend chain backward. Body flexion
   could be added in a later refinement if the rigid-trunk
   assumption proves too restrictive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# Default likelihood threshold for length fitting. Lower than
# the smoother's working threshold (typically 0.7) because we
# want enough samples for robust median estimation.
_LENGTH_FIT_THRESHOLD = 0.5


@dataclass
class BodySegment:
    """One segment of the kinematic tree.

    A segment has a name, a parent (None if root), a default
    rest-angle in the parent's frame (only meaningful for
    non-root segments), and a dict of attached markers.

    Each attached marker has an offset ``(length, angle)`` in
    THIS segment's distal-end frame. So if the segment has
    length L_seg and the marker's offset is (L_marker, α), the
    marker's position is:

        marker_world = segment_distal_end +
            R(segment_world_orientation) @ (L_marker * (cos α, sin α))

    For the simplest case where a single marker IS the segment's
    distal endpoint, the marker offset is (0, 0): the marker
    coincides with the segment's distal end.

    For multiple markers attached to one segment (e.g., head with
    headmid + ear_left + ear_right + nose all pointing in
    different directions from the head joint), each marker's
    offset is its position in the head's local frame.

    Fields
    ------
    name : str
        Unique identifier within a layout.
    parent : str or None
        Name of parent segment. None marks this as the root.
    rest_angle : float
        Default angle (radians) in parent's frame. Used as
        initial guess for state initialization.
    markers : dict[str, (float, float)]
        Maps attached marker name to (length, angle) offset
        in this segment's distal-end frame.
    """
    name: str
    parent: Optional[str]
    rest_angle: float = 0.0
    markers: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class BodyLayout:
    """Kinematic-tree topology with state-packing convention.

    Stores the list of segments + their parent relationships,
    plus a stable index of marker → (segment, offset). Once
    built, the layout is immutable; all subsequent patches use
    ``state_dim`` and the index helpers to pack/unpack state
    vectors.

    State vector layout
    -------------------

    For a layout with M segments (1 root + N non-root) and
    K markers, the state vector has the following blocks:

        Block 1 (root pose, 8 dims):
          [root_x, root_y,
           root_vx, root_vy,
           root_cos, root_sin,
           root_cos_dot, root_sin_dot]

        Block 2 (per-non-root-segment orientation, 4 dims each):
          for each non-root segment s:
            [s_cos, s_sin, s_cos_dot, s_sin_dot]
          where (s_cos, s_sin) is the unit vector in parent's frame.

        Block 3 (segment lengths, 2 dims each):
          for each non-root segment s (root has no length):
            [s_length, s_length_dot]

    Total state dim = 8 + 4*N + 2*N = 8 + 6*N where N is the
    number of non-root segments.

    For the standard rat layout with 9 non-root segments
    (back1, back3, back4, lateral_l/r, neck, head, tail1/2/3),
    that's 8 + 6*9 = 62 state dims.

    The full state dim is comparable to v1 (60), but the
    coupling structure is fundamentally different — v2's
    observations couple all markers through the shared
    body state, so dropouts on any individual marker don't
    cause that marker's state to drift away from the others.
    """
    segments: List[BodySegment]

    def __post_init__(self) -> None:
        # Validate exactly one root
        roots = [s for s in self.segments if s.parent is None]
        if len(roots) != 1:
            raise ValueError(
                f"Layout must have exactly one root segment "
                f"(parent=None); got {len(roots)}: "
                f"{[s.name for s in roots]}"
            )

        # Validate all non-root parents exist
        seg_names = {s.name for s in self.segments}
        for s in self.segments:
            if s.parent is not None and s.parent not in seg_names:
                raise ValueError(
                    f"Segment {s.name!r} has parent {s.parent!r} "
                    f"which is not in the layout"
                )

        # Validate no cycles (DFS from root, every segment
        # reachable in topo order)
        self._topo_order = self._compute_topo_order()
        if len(self._topo_order) != len(self.segments):
            raise ValueError(
                f"Layout has unreachable or cyclic segments. "
                f"Reachable: {self._topo_order}; "
                f"all: {[s.name for s in self.segments]}"
            )

        # Validate no duplicate marker attachments
        seen_markers: Dict[str, str] = {}
        for s in self.segments:
            for m in s.markers:
                if m in seen_markers:
                    raise ValueError(
                        f"Marker {m!r} attached to multiple "
                        f"segments: {seen_markers[m]!r} and "
                        f"{s.name!r}"
                    )
                seen_markers[m] = s.name

    def _compute_topo_order(self) -> List[str]:
        """Topological sort: root first, children after parents.

        Used by forward kinematics (compute root → children) and
        by state-packing index conventions.
        """
        result: List[str] = []
        children_of: Dict[Optional[str], List[str]] = {}
        for s in self.segments:
            children_of.setdefault(s.parent, []).append(s.name)

        # BFS from root
        queue = [s.name for s in self.segments if s.parent is None]
        while queue:
            current = queue.pop(0)
            result.append(current)
            queue.extend(children_of.get(current, []))
        return result

    # ----- structural queries -----

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_non_root_segments(self) -> int:
        return sum(1 for s in self.segments if s.parent is not None)

    @property
    def n_markers(self) -> int:
        return sum(len(s.markers) for s in self.segments)

    @property
    def root_segment(self) -> BodySegment:
        for s in self.segments:
            if s.parent is None:
                return s
        raise RuntimeError("No root segment (validated in __post_init__)")

    @property
    def topo_order(self) -> List[str]:
        """Segment names in topological order (root first)."""
        return list(self._topo_order)

    @property
    def non_root_topo_order(self) -> List[str]:
        """Non-root segments in topological order. The
        canonical order for state-packing.
        """
        return [n for n in self._topo_order if n != self.root_segment.name]

    @property
    def marker_names(self) -> List[str]:
        """Markers in topological order of their attached
        segments (stable; used for observation vector
        packing).
        """
        names: List[str] = []
        for seg_name in self._topo_order:
            seg = self.segment_by_name(seg_name)
            names.extend(seg.markers.keys())
        return names

    def segment_by_name(self, name: str) -> BodySegment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(f"No segment named {name!r}")

    def marker_attachment(self, marker: str) -> Tuple[str, Tuple[float, float]]:
        """Returns (segment_name, (length, angle)) for a marker."""
        for s in self.segments:
            if marker in s.markers:
                return s.name, s.markers[marker]
        raise KeyError(f"No segment owns marker {marker!r}")

    # ----- state-packing index helpers -----
    #
    # The state vector layout (see class docstring) is a series
    # of contiguous blocks. These helpers return the (start, end)
    # slice for each named block, so subsequent patches can
    # extract or update blocks without hardcoding indices.

    @property
    def state_dim(self) -> int:
        """Total state dimension."""
        return 8 + 6 * self.n_non_root_segments

    def slice_root_pose(self) -> slice:
        """Slice for [root_x, root_y, root_vx, root_vy,
        root_cos, root_sin, root_cos_dot, root_sin_dot].
        """
        return slice(0, 8)

    def slice_segment_orientation(self, segment_name: str) -> slice:
        """Slice for [s_cos, s_sin, s_cos_dot, s_sin_dot]
        of a non-root segment.
        """
        if segment_name not in self.non_root_topo_order:
            raise KeyError(
                f"slice_segment_orientation called for non-existent "
                f"or root segment: {segment_name!r}"
            )
        idx = self.non_root_topo_order.index(segment_name)
        start = 8 + 4 * idx
        return slice(start, start + 4)

    def slice_segment_length(self, segment_name: str) -> slice:
        """Slice for [s_length, s_length_dot]."""
        if segment_name not in self.non_root_topo_order:
            raise KeyError(
                f"slice_segment_length called for non-existent "
                f"or root segment: {segment_name!r}"
            )
        idx = self.non_root_topo_order.index(segment_name)
        # All segment-length pairs come after all orientation blocks
        start = 8 + 4 * self.n_non_root_segments + 2 * idx
        return slice(start, start + 2)


def standard_rat_layout(
    include_lateral: bool = True,
    include_center: bool = True,
    include_back4: bool = True,
    include_tail: bool = True,
) -> BodyLayout:
    """Default mufasa rat skeleton.

    Topology:
      body (root, marker=back2, with back1/back3/lateral_l/r/center
                                rigidly attached as offsets)
        ├─ back_rear (marker=back4 if included)
        ├─ neck (marker=neck) → head (marker=headmid)
        │                              ├─ nose
        │                              ├─ ear_left
        │                              └─ ear_right
        └─ tail_1 (marker=tailbase) → tail_2 (tailmid) → tail_3 (tailend)

    The body is treated as locally rigid (no spine flexion DOF
    in v2-MVP). All trunk markers (back1, back3, lateral_left,
    lateral_right, center, plus the back2 root) are attached to
    the body segment with fixed offsets — fit at calibration
    time.

    The neck and head segments each have one orientation DOF.
    The three tail segments each have one. Total non-root
    segments: 7 (back_rear, neck, head, tail_1, tail_2, tail_3,
    plus the body itself isn't counted since it's the root).
    Wait — let me recount. Non-root: back_rear (if include_back4),
    neck, head, tail_1, tail_2, tail_3. That's 6 if include_back4
    is True, 5 if False.

    Note that head holds nose + ear_left + ear_right with
    offsets in the head's distal frame; these markers don't get
    their own orientation DOF (they're rigid attachments).

    Parameters
    ----------
    include_lateral : bool
        Include lateral_left and lateral_right as body
        attachments.
    include_center : bool
        Include the 'center' marker as a body attachment.
    include_back4 : bool
        Include back4 as a separate posterior segment. If
        False, back4 is omitted entirely (some experiments
        only track back1-back3).
    include_tail : bool
        Include tail segments. Some sessions don't track tail.

    Returns
    -------
    BodyLayout
    """
    segments: List[BodySegment] = []

    # Root: 'body' segment, with back2 as the canonical position
    # marker. Other trunk markers attach as fixed offsets.
    body_markers: Dict[str, Tuple[float, float]] = {
        "back2": (0.0, 0.0),  # root; body's distal-end IS back2
    }
    # Other trunk markers will get their offsets fit from data
    # by fit_body_lengths. Initial values are placeholders.
    body_markers["back1"] = (1.0, 0.0)        # forward of back2
    body_markers["back3"] = (1.0, np.pi)      # behind back2
    if include_lateral:
        body_markers["lateral_left"] = (1.0, np.pi / 2)
        body_markers["lateral_right"] = (1.0, -np.pi / 2)
    if include_center:
        body_markers["center"] = (0.5, 0.0)
    segments.append(BodySegment(
        name="body",
        parent=None,
        rest_angle=0.0,
        markers=body_markers,
    ))

    # Posterior — back4
    if include_back4:
        segments.append(BodySegment(
            name="back_rear",
            parent="body",
            rest_angle=np.pi,  # extends behind body
            markers={"back4": (0.0, 0.0)},
        ))

    # Neck → head → (nose, ears)
    segments.append(BodySegment(
        name="neck",
        parent="body",
        rest_angle=0.0,  # extends forward from body
        markers={"neck": (0.0, 0.0)},
    ))
    segments.append(BodySegment(
        name="head",
        parent="neck",
        rest_angle=0.0,
        markers={
            "headmid": (0.0, 0.0),
            "nose": (1.0, 0.0),               # forward of headmid
            "ear_left": (0.5, np.pi / 3),     # left of headmid
            "ear_right": (0.5, -np.pi / 3),   # right of headmid
        },
    ))

    # Tail
    if include_tail:
        segments.append(BodySegment(
            name="tail_1",
            parent="back_rear" if include_back4 else "body",
            rest_angle=np.pi if not include_back4 else 0.0,
            markers={"tailbase": (0.0, 0.0)},
        ))
        segments.append(BodySegment(
            name="tail_2",
            parent="tail_1",
            rest_angle=0.0,
            markers={"tailmid": (0.0, 0.0)},
        ))
        segments.append(BodySegment(
            name="tail_3",
            parent="tail_2",
            rest_angle=0.0,
            markers={"tailend": (0.0, 0.0)},
        ))

    return BodyLayout(segments=segments)


@dataclass
class FittedLengths:
    """Per-segment lengths and offsets fitted from observation
    data, plus dispersion estimates used to set process noise
    on the slow-varying length state.

    Fields
    ------
    segment_lengths : dict[str, float]
        Length of each non-root segment (median observed
        distance from parent's distal-end-attached marker
        to this segment's distal-end-attached marker).
    segment_length_iqr : dict[str, float]
        IQR of those distances. Used as a proxy for typical
        length variation; sets the magnitude of the slow-
        varying length state's process noise.
    marker_offsets : dict[str, (float, float)]
        Updated marker-in-segment-frame offsets (length, angle)
        for markers that are NOT the segment's distal-end
        marker. These are the rigid-attachment offsets fit
        from data.
    """
    segment_lengths: Dict[str, float]
    segment_length_iqr: Dict[str, float]
    marker_offsets: Dict[str, Tuple[float, float]]


def _segment_distal_marker(layout: BodyLayout, segment_name: str) -> Optional[str]:
    """Return the marker attached at the segment's distal end
    (i.e., offset (0, 0) in segment frame), or None if no
    marker is at the distal end. There can be at most one such
    marker per segment by convention; for the root, this is
    the marker that defines the root position.
    """
    seg = layout.segment_by_name(segment_name)
    for marker, (length, angle) in seg.markers.items():
        if abs(length) < 1e-9 and abs(angle) < 1e-9:
            return marker
    return None


def fit_body_lengths(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: BodyLayout,
    marker_names: List[str],
    likelihood_threshold: float = _LENGTH_FIT_THRESHOLD,
) -> FittedLengths:
    """Estimate per-segment lengths and per-marker offsets
    from observation data.

    Method
    ------

    For each non-root segment with a distal marker, length is
    the median of pairwise distances ``||distal_marker -
    parent_distal_marker||`` across high-confidence frames
    where both are observed. IQR captures typical variation,
    used to set process-noise scale on the slow-varying
    length state.

    For markers attached to a segment with non-zero offsets
    (e.g., ear_left attached to head), the offset is fit by
    median position-relative-to-distal in the segment's frame.
    Frame transforms require the segment orientation, which we
    don't have at this point; we approximate by aligning each
    frame's orientation along the segment's parent direction
    and computing the offset in that approximate frame, then
    median-aggregating.

    Marker → segment mapping is taken from the layout.

    Parameters
    ----------
    positions : (T, n_markers, 2)
    likelihoods : (T, n_markers)
    layout : BodyLayout
    marker_names : list[str]
        Names matching the columns of positions/likelihoods.
        Must include all markers that the layout references;
        markers in the layout that are missing from the data
        produce a warning but don't fail (their lengths/offsets
        revert to layout defaults).
    likelihood_threshold : float

    Returns
    -------
    FittedLengths
    """
    name_to_idx = {n: i for i, n in enumerate(marker_names)}

    segment_lengths: Dict[str, float] = {}
    segment_length_iqr: Dict[str, float] = {}
    marker_offsets: Dict[str, Tuple[float, float]] = {}

    # Per non-root segment: fit length from distal-marker pair
    for seg_name in layout.non_root_topo_order:
        seg = layout.segment_by_name(seg_name)
        parent_seg = layout.segment_by_name(seg.parent)
        distal_marker = _segment_distal_marker(layout, seg_name)
        parent_distal = _segment_distal_marker(layout, parent_seg.name)

        if distal_marker is None or parent_distal is None:
            # Can't fit length without endpoint markers
            continue
        if distal_marker not in name_to_idx or parent_distal not in name_to_idx:
            continue

        i_dist = name_to_idx[distal_marker]
        i_par = name_to_idx[parent_distal]
        p_dist = positions[:, i_dist, :]
        p_par = positions[:, i_par, :]
        l_dist = likelihoods[:, i_dist]
        l_par = likelihoods[:, i_par]
        mask = (
            (l_dist >= likelihood_threshold)
            & (l_par >= likelihood_threshold)
            & np.all(np.isfinite(p_dist), axis=1)
            & np.all(np.isfinite(p_par), axis=1)
        )
        if mask.sum() < 20:
            continue
        diffs = p_dist[mask] - p_par[mask]
        dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        segment_lengths[seg_name] = float(np.median(dists))
        # IQR (q75 - q25), used to set process noise scale
        q25, q75 = float(np.percentile(dists, 25)), float(np.percentile(dists, 75))
        segment_length_iqr[seg_name] = q75 - q25

    # Per non-distal-end marker: fit offset (length, angle) in
    # the attached segment's distal-end frame. The "frame" here
    # is approximated by the direction from parent_distal to
    # this_distal (i.e., the segment's instantaneous direction).
    for seg in layout.segments:
        seg_distal = _segment_distal_marker(layout, seg.name)
        if seg_distal is None:
            continue

        # Need a frame direction. For non-root, this is
        # parent_distal → seg_distal. For root, we don't have
        # a frame direction from data alone, so we use the
        # body's direction proxy: distance from one trunk
        # marker pair (e.g., back3 → back1 if both present, or
        # back2 → neck-attachment if neck-marker is present).
        if seg.parent is not None:
            parent_distal = _segment_distal_marker(
                layout, seg.parent,
            )
        else:
            # Root: use a trunk-aligned proxy. Pick the most
            # forward trunk marker available to define a body
            # direction. For mufasa rat, the canonical front-
            # back proxy is back1 (front) vs. back3 (back) if
            # both are root attachments.
            parent_distal = None
            if "back1" in seg.markers and "back3" in seg.markers:
                # Use back3 (rear) → back1 (forward) as the body
                # direction proxy. seg_distal is back2 (the root
                # marker), and we treat the parent_distal proxy
                # as back3 for the purpose of frame definition.
                parent_distal = "back3"
                # We'll override below to compute the body
                # direction directly rather than using
                # parent_distal as a true parent.

        if parent_distal is None:
            # Can't establish a frame direction — leave default
            # offsets in the layout
            continue
        if parent_distal not in name_to_idx or seg_distal not in name_to_idx:
            continue

        i_seg = name_to_idx[seg_distal]
        i_par = name_to_idx[parent_distal]
        p_seg = positions[:, i_seg, :]
        p_par = positions[:, i_par, :]
        l_seg = likelihoods[:, i_seg]
        l_par = likelihoods[:, i_par]

        for marker, (default_len, default_angle) in seg.markers.items():
            if marker == seg_distal:
                continue  # Distal marker has offset (0, 0) by definition
            if marker not in name_to_idx:
                # Marker not in data — keep default
                marker_offsets[marker] = (default_len, default_angle)
                continue

            i_m = name_to_idx[marker]
            p_m = positions[:, i_m, :]
            l_m = likelihoods[:, i_m]
            mask = (
                (l_seg >= likelihood_threshold)
                & (l_par >= likelihood_threshold)
                & (l_m >= likelihood_threshold)
                & np.all(np.isfinite(p_seg), axis=1)
                & np.all(np.isfinite(p_par), axis=1)
                & np.all(np.isfinite(p_m), axis=1)
            )
            if mask.sum() < 20:
                marker_offsets[marker] = (default_len, default_angle)
                continue

            # Frame: x-axis points from parent_distal to seg_distal.
            # In that frame, marker offset is computed by rotating
            # (marker - seg_distal) by minus-frame-angle and finding
            # the polar coords.
            frame_dx = p_seg[mask] - p_par[mask]
            frame_dist = np.sqrt(frame_dx[:, 0] ** 2 + frame_dx[:, 1] ** 2)
            # Avoid division by zero
            ok_frame = frame_dist > 1e-6
            if ok_frame.sum() < 20:
                marker_offsets[marker] = (default_len, default_angle)
                continue
            frame_dx = frame_dx[ok_frame]
            frame_dist = frame_dist[ok_frame]
            cos_phi = frame_dx[:, 0] / frame_dist
            sin_phi = frame_dx[:, 1] / frame_dist

            # Marker relative to seg_distal, in world frame
            rel = (p_m[mask] - p_seg[mask])[ok_frame]
            # Rotate into segment frame: x' = cos*x + sin*y,
            # y' = -sin*x + cos*y
            x_local = cos_phi * rel[:, 0] + sin_phi * rel[:, 1]
            y_local = -sin_phi * rel[:, 0] + cos_phi * rel[:, 1]

            length = np.sqrt(x_local ** 2 + y_local ** 2)
            angle = np.arctan2(y_local, x_local)

            # Median for robustness
            marker_offsets[marker] = (
                float(np.median(length)),
                float(np.median(angle)),
            )

    # For markers that are distal-end markers, offset is (0, 0)
    for seg in layout.segments:
        seg_distal = _segment_distal_marker(layout, seg.name)
        if seg_distal is not None and seg_distal not in marker_offsets:
            marker_offsets[seg_distal] = (0.0, 0.0)

    return FittedLengths(
        segment_lengths=segment_lengths,
        segment_length_iqr=segment_length_iqr,
        marker_offsets=marker_offsets,
    )


# ============================================================
# Forward kinematics + observation function + Jacobian
# (patch 100)
# ============================================================
#
# Given a body state vector and a layout, compute the predicted
# 2D position of every marker (forward kinematics) and the
# Jacobian of those positions with respect to state. The Jacobian
# is the linearization needed by the EKF.
#
# Math (see commit message for full derivation)
# ----------------------------------------------
#
# State packing (per BodyLayout):
#
#   [root_x, root_y, root_vx, root_vy,
#    root_cos, root_sin, root_cos_dot, root_sin_dot,
#    for each non-root segment in topo order:
#      s_cos, s_sin, s_cos_dot, s_sin_dot,
#    for each non-root segment in topo order:
#      s_length, s_length_dot]
#
# Forward kinematics (depth-first by topo order):
#
#   P_root = (root_x, root_y)
#   R_root = R(root_cos, root_sin)
#
#   for each non-root segment s in topo order:
#     v_s_local = (s_length * s_cos, s_length * s_sin)
#       in parent(s)'s frame
#     P_s_distal = P_parent_distal + R_world[parent(s)] @ v_s_local
#     R_world[s] = R_world[parent(s)] @ R_local(s_cos, s_sin)
#
# Marker position (for marker on segment s with offset (l, α)
# in s's frame):
#
#   v_marker_local = (l * cos α, l * sin α)
#   p_marker = P_s_distal + R_world[s] @ v_marker_local
#
# Jacobian (∂p_marker / ∂state):
#
# For state components NOT in {positions, orientations, lengths}
# (i.e., velocities), ∂ = 0 (velocities don't affect positions).
#
# Position components (root_x, root_y):
#   ∂p_marker / ∂root_x = (1, 0)
#   ∂p_marker / ∂root_y = (0, 1)
#
# Orientation components — use 2-column ambient-space Jacobian
# (treats cos and sin as independent R^2 coords, NOT
# constrained-tangent).  For each ancestor a in the chain from
# root to marker's segment, with cumulative_offset_at_a being
# the marker position in a's parent's frame minus P_a_proximal:
#
#   ∂p_marker / ∂a_cos =
#     R_world[parent(a)] @ ((L_a, 0) + cumulative_offset_past_a)
#   ∂p_marker / ∂a_sin =
#     R_world[parent(a)] @ ((0, L_a) + J @ cumulative_offset_past_a)
#
# where for root a: R_world[parent(root)] := I, and L_root := 0
# (root has no length state — its position state covers
# translation directly).  cumulative_offset_past_root = (p_m -
# P_root) expressed in root's frame.
#
# Length components:
#   ∂p_marker / ∂s_length = R_world[parent(s)] @ (s_cos, s_sin)
#
# Note: for ancestors of the marker's segment that are NOT in
# the chain from root to that segment, partials are zero
# (different branches of the tree don't affect each other).


def _R(cos_a: float, sin_a: float) -> np.ndarray:
    """2x2 rotation matrix from (cos, sin)."""
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])


# 2x2 90-degree rotation matrix used in Jacobian computation
_J90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def _pack_state_layout_indices(layout: BodyLayout) -> Dict[str, Dict[str, int]]:
    """Convenience: pre-compute named indices into the state
    vector for fast access during forward kinematics.

    Returns a dict ``{block_name: {field_name: index}}`` where:
      block_name = 'root' or non-root segment name
      field_name in {'x', 'y', 'vx', 'vy', 'cos', 'sin',
                     'cos_dot', 'sin_dot', 'length',
                     'length_dot'} (root has no length;
                     non-root has all)
    """
    indices: Dict[str, Dict[str, int]] = {}
    rs = layout.slice_root_pose()
    indices["__root__"] = {
        "x": rs.start + 0,
        "y": rs.start + 1,
        "vx": rs.start + 2,
        "vy": rs.start + 3,
        "cos": rs.start + 4,
        "sin": rs.start + 5,
        "cos_dot": rs.start + 6,
        "sin_dot": rs.start + 7,
    }
    for seg_name in layout.non_root_topo_order:
        os_ = layout.slice_segment_orientation(seg_name)
        ls = layout.slice_segment_length(seg_name)
        indices[seg_name] = {
            "cos": os_.start + 0,
            "sin": os_.start + 1,
            "cos_dot": os_.start + 2,
            "sin_dot": os_.start + 3,
            "length": ls.start + 0,
            "length_dot": ls.start + 1,
        }
    return indices


@dataclass
class ForwardKinematicsResult:
    """Per-segment world poses computed by forward kinematics.

    For segment s:
      P_distal[s]: world position of s's distal end (where
                   markers attached at offset (0, 0) live)
      R_world[s]:  world-frame rotation matrix of s

    For the root, P_distal is the root position (root_x, root_y)
    and R_world is R(root_cos, root_sin).

    These are the inputs to marker-position and Jacobian
    computation.
    """
    P_distal: Dict[str, np.ndarray]   # segment name -> (2,) array
    R_world: Dict[str, np.ndarray]    # segment name -> (2, 2) array


def forward_kinematics(
    state: np.ndarray,
    layout: BodyLayout,
) -> ForwardKinematicsResult:
    """Compute per-segment world poses from a body state.

    Walks the kinematic tree in topological order (root first,
    children after parents), accumulating world rotations and
    positions through the chain. O(M) where M is number of
    segments.

    Parameters
    ----------
    state : (state_dim,) array
        Body state vector packed per ``BodyLayout`` convention.
    layout : BodyLayout

    Returns
    -------
    ForwardKinematicsResult
    """
    if state.shape != (layout.state_dim,):
        raise ValueError(
            f"state shape {state.shape} != ({layout.state_dim},)"
        )

    idx = _pack_state_layout_indices(layout)
    P_distal: Dict[str, np.ndarray] = {}
    R_world: Dict[str, np.ndarray] = {}

    # Root
    root = layout.root_segment
    P_distal[root.name] = np.array([
        state[idx["__root__"]["x"]],
        state[idx["__root__"]["y"]],
    ])
    R_world[root.name] = _R(
        state[idx["__root__"]["cos"]],
        state[idx["__root__"]["sin"]],
    )

    # Non-root in topo order
    for seg_name in layout.non_root_topo_order:
        seg = layout.segment_by_name(seg_name)
        parent_name = seg.parent
        # State for s
        s_cos = state[idx[seg_name]["cos"]]
        s_sin = state[idx[seg_name]["sin"]]
        s_length = state[idx[seg_name]["length"]]
        # Segment vector in parent frame
        v_s_local = np.array([s_length * s_cos, s_length * s_sin])
        # World position
        P_distal[seg_name] = (
            P_distal[parent_name] + R_world[parent_name] @ v_s_local
        )
        # World rotation
        R_local_s = _R(s_cos, s_sin)
        R_world[seg_name] = R_world[parent_name] @ R_local_s

    return ForwardKinematicsResult(P_distal=P_distal, R_world=R_world)


def state_to_marker_positions(
    state: np.ndarray,
    layout: BodyLayout,
    fk: Optional[ForwardKinematicsResult] = None,
) -> np.ndarray:
    """Compute predicted 2D positions for all markers.

    Order matches ``layout.marker_names`` (topological by
    segment, then alphabetical within segment as Python dict
    iteration preserves insertion order).

    Parameters
    ----------
    state : (state_dim,) array
    layout : BodyLayout
    fk : ForwardKinematicsResult, optional
        If provided, skip recomputing forward kinematics
        (useful when computing both positions and Jacobian).

    Returns
    -------
    (n_markers, 2) array
        Predicted x, y per marker.
    """
    if fk is None:
        fk = forward_kinematics(state, layout)

    marker_names = layout.marker_names
    positions = np.zeros((len(marker_names), 2))

    for i, marker in enumerate(marker_names):
        seg_name, (l_off, a_off) = layout.marker_attachment(marker)
        v_marker_local = np.array([
            l_off * np.cos(a_off),
            l_off * np.sin(a_off),
        ])
        positions[i] = (
            fk.P_distal[seg_name] + fk.R_world[seg_name] @ v_marker_local
        )
    return positions


def state_to_marker_jacobian(
    state: np.ndarray,
    layout: BodyLayout,
    fk: Optional[ForwardKinematicsResult] = None,
) -> np.ndarray:
    """Compute the Jacobian of marker positions w.r.t. state.

    For K markers and state dimension D, returns a (2K, D)
    matrix where row 2i, 2i+1 are ∂x_marker_i/∂state and
    ∂y_marker_i/∂state respectively.

    Velocity components have zero columns (don't affect
    positions). Orientation columns use the 2-coordinate
    ambient-space Jacobian (cos and sin treated as independent
    coordinates, not constrained tangent space) — this is the
    correct form for an EKF that tracks (cos, sin) as state
    components.

    Parameters
    ----------
    state : (state_dim,) array
    layout : BodyLayout
    fk : ForwardKinematicsResult, optional
        If provided, skip recomputing forward kinematics.

    Returns
    -------
    (2 * n_markers, state_dim) array
    """
    if fk is None:
        fk = forward_kinematics(state, layout)

    idx = _pack_state_layout_indices(layout)
    marker_names = layout.marker_names
    K = len(marker_names)
    D = layout.state_dim
    H = np.zeros((2 * K, D))

    # Pre-compute segment ancestor chains (root → marker's
    # segment, in topo order). Used to know which a's affect
    # which marker.
    parent_of: Dict[str, Optional[str]] = {
        s.name: s.parent for s in layout.segments
    }

    def chain_to_root(seg_name: str) -> List[str]:
        """List of segments from root to seg_name (inclusive),
        in proximal-to-distal order.
        """
        chain: List[str] = []
        cur: Optional[str] = seg_name
        while cur is not None:
            chain.append(cur)
            cur = parent_of[cur]
        return list(reversed(chain))

    root_name = layout.root_segment.name

    for i, marker in enumerate(marker_names):
        seg_name, (l_off, a_off) = layout.marker_attachment(marker)
        v_marker_local = np.array([
            l_off * np.cos(a_off),
            l_off * np.sin(a_off),
        ])
        # World marker position
        p_m = fk.P_distal[seg_name] + fk.R_world[seg_name] @ v_marker_local

        row_x = 2 * i
        row_y = 2 * i + 1

        # Position-of-root partials are constant
        H[row_x, idx["__root__"]["x"]] = 1.0
        H[row_y, idx["__root__"]["y"]] = 1.0

        # Walk the chain from root to seg_name, computing
        # partials at each ancestor.
        chain = chain_to_root(seg_name)
        for a_name in chain:
            a_seg = layout.segment_by_name(a_name)

            # Determine R_world[parent(a)]: identity for root,
            # R_world[parent] otherwise.
            if a_seg.parent is None:
                R_pa = np.eye(2)
            else:
                R_pa = fk.R_world[a_seg.parent]

            # Cumulative offset past a, in a's frame:
            # (p_m - P_a_distal) expressed in a's local frame.
            # For root, P_a_distal = P_root and R_world[a]^T
            # transforms p_m - P_root into root's frame.
            P_a_distal = fk.P_distal[a_name]
            R_a_world = fk.R_world[a_name]
            offset_past_a_in_a_frame = R_a_world.T @ (p_m - P_a_distal)

            # Length state — only for non-root segments.
            if a_seg.parent is not None:
                a_cos = state[idx[a_name]["cos"]]
                a_sin = state[idx[a_name]["sin"]]
                a_length = state[idx[a_name]["length"]]
                # ∂p_m / ∂a_length = R_world[parent(a)] @ (a_cos, a_sin)
                # (changing length scales the segment vector,
                # which shifts everything past a by that
                # direction in parent(a)'s frame.)
                d_p_d_length = R_pa @ np.array([a_cos, a_sin])
                col = idx[a_name]["length"]
                H[row_x, col] = d_p_d_length[0]
                H[row_y, col] = d_p_d_length[1]
            else:
                a_length = 0.0

            # Orientation state:
            #   ∂p_m / ∂a_cos = R_world[parent(a)] @ (
            #     (L_a, 0) + offset_past_a_in_a_frame)
            #   ∂p_m / ∂a_sin = R_world[parent(a)] @ (
            #     (0, L_a) + J @ offset_past_a_in_a_frame)
            #
            # The (L_a, 0)/(0, L_a) terms come from
            # differentiating the segment vector (a_length *
            # a_cos, a_length * a_sin).  For root, L_a = 0 so
            # those terms vanish.

            d_offset_d_cos = (
                np.array([a_length, 0.0]) + offset_past_a_in_a_frame
            )
            d_offset_d_sin = (
                np.array([0.0, a_length]) + _J90 @ offset_past_a_in_a_frame
            )

            d_p_d_cos = R_pa @ d_offset_d_cos
            d_p_d_sin = R_pa @ d_offset_d_sin

            if a_seg.parent is None:
                col_cos = idx["__root__"]["cos"]
                col_sin = idx["__root__"]["sin"]
            else:
                col_cos = idx[a_name]["cos"]
                col_sin = idx[a_name]["sin"]
            H[row_x, col_cos] = d_p_d_cos[0]
            H[row_y, col_cos] = d_p_d_cos[1]
            H[row_x, col_sin] = d_p_d_sin[0]
            H[row_y, col_sin] = d_p_d_sin[1]

    return H


def initial_state_from_data(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: BodyLayout,
    marker_names: List[str],
    fitted: FittedLengths,
    likelihood_threshold: float = 0.5,
) -> np.ndarray:
    """Build an initial state vector by inverting the kinematic
    chain on the first well-observed frame.

    For each segment, compute its orientation and length from
    the observed marker positions. Velocities are initialized
    to zero. This gives EM a sensible starting point that's
    consistent with the data.

    Returns
    -------
    state : (state_dim,) array
    """
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    state = np.zeros(layout.state_dim)
    indices = _pack_state_layout_indices(layout)

    # Find a frame where many key markers are observed
    distal_markers = [
        m for s in layout.segments
        for m in [_segment_distal_marker(layout, s.name)]
        if m is not None and m in name_to_idx
    ]
    if not distal_markers:
        # Can't initialize meaningfully — use zeros
        # (root cos = 1.0 to avoid degenerate identity)
        state[indices["__root__"]["cos"]] = 1.0
        for seg_name in layout.non_root_topo_order:
            state[indices[seg_name]["cos"]] = 1.0
            state[indices[seg_name]["length"]] = (
                fitted.segment_lengths.get(seg_name, 1.0)
            )
        return state

    distal_idx = [name_to_idx[m] for m in distal_markers]
    finite_per_frame = np.all(
        np.isfinite(positions[:, distal_idx, :]), axis=(1, 2),
    )
    high_p_per_frame = np.all(
        likelihoods[:, distal_idx] >= likelihood_threshold, axis=1,
    )
    valid = finite_per_frame & high_p_per_frame
    if not valid.any():
        # Fallback to first frame's data
        t0 = 0
    else:
        t0 = int(np.argmax(valid))

    # Root position from root's distal marker
    root = layout.root_segment
    root_distal = _segment_distal_marker(layout, root.name)
    if root_distal in name_to_idx:
        root_pos = positions[t0, name_to_idx[root_distal], :]
        state[indices["__root__"]["x"]] = root_pos[0]
        state[indices["__root__"]["y"]] = root_pos[1]

    # Root orientation: align with body direction (back3 → back1
    # if available, otherwise default to identity).
    root_cos = 1.0
    root_sin = 0.0
    if "back1" in name_to_idx and "back3" in name_to_idx:
        front = positions[t0, name_to_idx["back1"], :]
        rear = positions[t0, name_to_idx["back3"], :]
        if np.all(np.isfinite(front)) and np.all(np.isfinite(rear)):
            d = front - rear
            n = np.linalg.norm(d)
            if n > 1e-6:
                root_cos = float(d[0] / n)
                root_sin = float(d[1] / n)
    state[indices["__root__"]["cos"]] = root_cos
    state[indices["__root__"]["sin"]] = root_sin

    # Per non-root segment: fit cos/sin and length from data
    # Build forward kinematics incrementally.  At each segment
    # in topo order, we know parent's world pose; compute s's
    # local angle from observed (P_s_distal - P_parent_distal)
    # in parent's frame.
    P_world: Dict[str, np.ndarray] = {root.name: np.array([
        state[indices["__root__"]["x"]],
        state[indices["__root__"]["y"]],
    ])}
    R_world: Dict[str, np.ndarray] = {root.name: _R(root_cos, root_sin)}

    for seg_name in layout.non_root_topo_order:
        seg = layout.segment_by_name(seg_name)
        parent_distal = _segment_distal_marker(layout, seg.parent)
        seg_distal = _segment_distal_marker(layout, seg_name)

        s_cos = 1.0
        s_sin = 0.0
        s_length = fitted.segment_lengths.get(seg_name, 1.0)

        if (
            parent_distal in name_to_idx and seg_distal in name_to_idx
        ):
            p_par = positions[t0, name_to_idx[parent_distal], :]
            p_seg = positions[t0, name_to_idx[seg_distal], :]
            if np.all(np.isfinite(p_par)) and np.all(np.isfinite(p_seg)):
                world_offset = p_seg - p_par
                # In parent's frame
                R_par = R_world[seg.parent]
                local_offset = R_par.T @ world_offset
                length = float(np.linalg.norm(local_offset))
                if length > 1e-6:
                    s_cos = float(local_offset[0] / length)
                    s_sin = float(local_offset[1] / length)
                    s_length = length

        state[indices[seg_name]["cos"]] = s_cos
        state[indices[seg_name]["sin"]] = s_sin
        state[indices[seg_name]["length"]] = s_length

        # Update P_world / R_world for children to use
        v_local = np.array([s_length * s_cos, s_length * s_sin])
        P_world[seg_name] = P_world[seg.parent] + R_world[seg.parent] @ v_local
        R_world[seg_name] = R_world[seg.parent] @ _R(s_cos, s_sin)

    return state
