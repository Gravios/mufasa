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
    perspective: Optional["PerspectiveModelV2"] = None,
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
    perspective : PerspectiveModelV2, optional
        If provided, scale each marker's body-frame offset by
        a position-dependent factor. Captures perspective /
        radial-distortion effects where apparent body width
        depends on the rat's location in the arena. When None,
        behavior is identical to previous patches (rigid
        offsets).

    Returns
    -------
    (n_markers, 2) array
        Predicted x, y per marker.
    """
    if fk is None:
        fk = forward_kinematics(state, layout)

    marker_names = layout.marker_names
    positions = np.zeros((len(marker_names), 2))

    # Pre-compute scale factors per marker if perspective active
    if perspective is not None:
        idx = _pack_state_layout_indices(layout)
        root_x = state[idx["__root__"]["x"]]
        root_y = state[idx["__root__"]["y"]]
        scales = perspective.scale_for_position(root_x, root_y)
    else:
        scales = None

    for i, marker in enumerate(marker_names):
        seg_name, (l_off, a_off) = layout.marker_attachment(marker)
        v_marker_local = np.array([
            l_off * np.cos(a_off),
            l_off * np.sin(a_off),
        ])
        if scales is not None:
            v_marker_local = v_marker_local * scales[i]
        positions[i] = (
            fk.P_distal[seg_name] + fk.R_world[seg_name] @ v_marker_local
        )
    return positions


def state_to_marker_jacobian(
    state: np.ndarray,
    layout: BodyLayout,
    fk: Optional[ForwardKinematicsResult] = None,
    perspective: Optional["PerspectiveModelV2"] = None,
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
    perspective : PerspectiveModelV2, optional
        When provided, the Jacobian includes terms from the
        position-dependent scale function:
          ∂p_m/∂root_x picks up R_world @ (offset_m * ∂scale/∂x)
          ∂p_m/∂root_y picks up R_world @ (offset_m * ∂scale/∂y)
        and existing offset terms get multiplied by scale.

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

    # Perspective scales and partials
    if perspective is not None:
        root_x = state[idx["__root__"]["x"]]
        root_y = state[idx["__root__"]["y"]]
        scales = perspective.scale_for_position(root_x, root_y)
        d_scales_d_x, d_scales_d_y = (
            perspective.scale_partials(root_x, root_y)
        )
    else:
        scales = None
        d_scales_d_x = None
        d_scales_d_y = None

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
        scale_m = 1.0 if scales is None else scales[i]
        v_marker_local = np.array([
            l_off * np.cos(a_off),
            l_off * np.sin(a_off),
        ]) * scale_m
        # World marker position (using scaled offset)
        p_m = fk.P_distal[seg_name] + fk.R_world[seg_name] @ v_marker_local

        row_x = 2 * i
        row_y = 2 * i + 1

        # Position-of-root partials are constant 1
        H[row_x, idx["__root__"]["x"]] = 1.0
        H[row_y, idx["__root__"]["y"]] = 1.0

        # Perspective contribution to root-position partials:
        # ∂(R_world @ (offset_m * scale)) / ∂root_x
        #   = R_world @ (offset_m * ∂scale/∂root_x)
        # (R_world doesn't depend on root_x, only on
        # orientation states.)
        if perspective is not None:
            R_seg = fk.R_world[seg_name]
            offset_unit = np.array([
                l_off * np.cos(a_off),
                l_off * np.sin(a_off),
            ])
            d_p_d_rx_persp = R_seg @ (offset_unit * d_scales_d_x[i])
            d_p_d_ry_persp = R_seg @ (offset_unit * d_scales_d_y[i])
            H[row_x, idx["__root__"]["x"]] += d_p_d_rx_persp[0]
            H[row_y, idx["__root__"]["x"]] += d_p_d_rx_persp[1]
            H[row_x, idx["__root__"]["y"]] += d_p_d_ry_persp[0]
            H[row_y, idx["__root__"]["y"]] += d_p_d_ry_persp[1]

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
            # Note p_m already includes the perspective scale.
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


# ============================================================
# EKF forward filter — patch 101
# ============================================================

# Floor on per-frame likelihood used in observation-noise
# scaling. p == 0 means "definitely missed" (excluded from
# observations entirely); but for borderline-low p we use the
# inverse-likelihood-squared scaling and need to clip away
# from zero to avoid numerical issues.
_P_FLOOR_V2 = 0.05

# Default noise on the unit-norm constraint observations.
# Small enough to pull (cos, sin) toward the unit circle each
# frame; large enough not to dominate over real marker
# observations during normal operation.
_DEFAULT_CONSTRAINT_SIGMA = 0.05

# Floors and conservative defaults for noise params. Set with
# v2's smaller-state-space EM degeneracy bounds in mind.
_FLOOR_SIGMA_MARKER_V2 = 0.5      # px
_FLOOR_Q_ROOT_POS = 1.0           # px²/s³ (continuous-time q for accel noise)
_FLOOR_Q_ROOT_ORI = 0.01
_FLOOR_Q_SEG_ORI = 0.001
_FLOOR_Q_LENGTH = 1e-6


@dataclass
class NoiseParamsV2:
    """EM-fittable noise parameters for the v2 smoother.

    Compared to v1, the parameter set is much smaller —
    ~5+M scalars instead of 3M (where M = number of markers).
    The smaller parameter space leaves less room for the
    degenerate EM fixed points that plagued v1.

    Fields
    ------
    sigma_marker : dict[str, float]
        Per-marker observation noise (px) at p=1. Effective
        noise at frame t with likelihood p_t is
        ``sigma_marker / max(p_t, p_floor)``.
    q_root_pos : float
        Continuous-time process noise on root position
        acceleration, px²/s³. Drives root translation
        randomness.
    q_root_ori : float
        Process noise on root orientation (in ambient
        cos/sin coordinates) angular acceleration.
    q_seg_ori : dict[str, float]
        Per non-root segment process noise on relative
        orientation angular acceleration. Smaller than root
        because segment angles are bounded (head can't
        rotate as fast as the body).
    q_length : dict[str, float]
        Per-segment length process noise. Very small (lengths
        nearly constant within session).
    constraint_sigma : float
        Observation noise on the unit-norm constraint
        observations. Held fixed (not EM-fit); acts as
        regularization toward (cos² + sin²) = 1.
    """
    sigma_marker: Dict[str, float]
    q_root_pos: float
    q_root_ori: float
    q_seg_ori: Dict[str, float]
    q_length: Dict[str, float]
    constraint_sigma: float = _DEFAULT_CONSTRAINT_SIGMA

    @classmethod
    def default(
        cls,
        layout: BodyLayout,
        sigma_marker: float = 3.0,
        q_root_pos: float = 200.0,
        q_root_ori: float = 1.0,
        q_seg_ori: float = 0.5,
        q_length: float = 0.01,
        constraint_sigma: float = _DEFAULT_CONSTRAINT_SIGMA,
    ) -> "NoiseParamsV2":
        """Build default noise params with uniform per-marker
        and per-segment values. Useful as initial values for
        EM.
        """
        return cls(
            sigma_marker={m: sigma_marker for m in layout.marker_names},
            q_root_pos=q_root_pos,
            q_root_ori=q_root_ori,
            q_seg_ori={s: q_seg_ori for s in layout.non_root_topo_order},
            q_length={s: q_length for s in layout.non_root_topo_order},
            constraint_sigma=constraint_sigma,
        )


def build_F_v2(layout: BodyLayout, dt: float) -> np.ndarray:
    """Build the state transition matrix F.

    F is block-diagonal:
      - Root pose block (8×8): standard 2D constant-velocity
        for (x, y, vx, vy) plus same form for ambient
        orientation (cos, sin, ċ, ṡ).
      - Per non-root segment orientation block (4×4):
        constant-velocity for (cos, sin) treated as ambient
        coordinates.
      - Per non-root segment length block (2×2):
        constant-velocity for (length, length_dot).

    All velocities propagate by their corresponding rate ×
    dt; rates themselves are constant under this dynamics
    model (with process noise injected via Q).

    Parameters
    ----------
    layout : BodyLayout
    dt : float

    Returns
    -------
    (state_dim, state_dim) array
    """
    D = layout.state_dim
    F = np.eye(D)

    # Helper: 2x2 constant-velocity block [[1, dt], [0, 1]]
    cv = np.array([[1.0, dt], [0.0, 1.0]])

    # Root: pose (x, y, vx, vy) at indices 0..3
    # The convention is [x, y, vx, vy], NOT [x, vx, y, vy].
    # F maps x → x + vx*dt, y → y + vy*dt.
    F[0, 2] = dt
    F[1, 3] = dt
    # Root orientation (cos, sin, ċ, ṡ) at indices 4..7
    F[4, 6] = dt
    F[5, 7] = dt

    # Per-segment orientation blocks
    for seg_name in layout.non_root_topo_order:
        sl = layout.slice_segment_orientation(seg_name)
        # [cos, sin, cos_dot, sin_dot]
        F[sl.start, sl.start + 2] = dt
        F[sl.start + 1, sl.start + 3] = dt

    # Per-segment length blocks
    for seg_name in layout.non_root_topo_order:
        sl = layout.slice_segment_length(seg_name)
        F[sl.start, sl.start + 1] = dt

    return F


def build_Q_v2(
    layout: BodyLayout, params: NoiseParamsV2, dt: float,
) -> np.ndarray:
    """Build the process noise covariance Q.

    Q is block-diagonal with each block parameterized by a
    scalar continuous-time process-noise intensity q. For a
    2D constant-velocity (pos, vel) block with process noise q
    on acceleration:

      Q_block = q * [dt^4/4   dt^3/2]
                    [dt^3/2   dt^2  ]

    For 4D blocks (pos, vel) × 2 axes (e.g., x and y of root,
    or cos and sin of orientation), Q is the kron product of
    the above 2×2 with I_2. Equivalently, the 4×4 block's
    diagonal has [dt^4/4, dt^4/4, dt^2, dt^2] and the
    cross-correlation between position and velocity is
    [dt^3/2, dt^3/2, 0, 0] in appropriate places.

    Note we store state as [x, y, vx, vy], not [x, vx, y, vy],
    so positions are at the front of the block and velocities
    at the back. The Q block accordingly is:

      Q_4d = q * [dt^4/4    0     dt^3/2  0     ]
                 [0      dt^4/4   0       dt^3/2]
                 [dt^3/2   0      dt^2    0     ]
                 [0      dt^3/2   0       dt^2  ]

    Parameters
    ----------
    layout : BodyLayout
    params : NoiseParamsV2
    dt : float

    Returns
    -------
    (state_dim, state_dim) array
    """
    D = layout.state_dim
    Q = np.zeros((D, D))

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    def _q4_block(q: float) -> np.ndarray:
        """4x4 block for [pos_x, pos_y, vel_x, vel_y]."""
        b = np.zeros((4, 4))
        b[0, 0] = q * dt4 / 4
        b[1, 1] = q * dt4 / 4
        b[2, 2] = q * dt2
        b[3, 3] = q * dt2
        b[0, 2] = q * dt3 / 2
        b[2, 0] = q * dt3 / 2
        b[1, 3] = q * dt3 / 2
        b[3, 1] = q * dt3 / 2
        return b

    def _q2_block(q: float) -> np.ndarray:
        """2x2 block for [pos, vel]."""
        b = np.zeros((2, 2))
        b[0, 0] = q * dt4 / 4
        b[1, 1] = q * dt2
        b[0, 1] = q * dt3 / 2
        b[1, 0] = q * dt3 / 2
        return b

    # Root pose: (x, y, vx, vy) at 0..3
    Q[0:4, 0:4] = _q4_block(params.q_root_pos)
    # Root orientation: (cos, sin, ċ, ṡ) at 4..7
    Q[4:8, 4:8] = _q4_block(params.q_root_ori)

    # Per-segment orientation blocks
    for seg_name in layout.non_root_topo_order:
        sl = layout.slice_segment_orientation(seg_name)
        q_s = params.q_seg_ori.get(seg_name, _FLOOR_Q_SEG_ORI)
        Q[sl, sl] = _q4_block(q_s)

    # Per-segment length blocks
    for seg_name in layout.non_root_topo_order:
        sl = layout.slice_segment_length(seg_name)
        q_l = params.q_length.get(seg_name, _FLOOR_Q_LENGTH)
        Q[sl, sl] = _q2_block(q_l)

    return Q


@dataclass
class FilterResultV2:
    """Per-frame filtered state estimate.

    Stores predicted and filtered (after observation update)
    means and covariances at every frame, plus per-frame
    metadata used by the smoother and EM.

    Fields
    ------
    x_pred : (T, D)
        Predicted state mean (before observation update),
        x̂_{t|t-1}.
    P_pred : (T, D, D)
        Predicted state covariance.
    x_filt : (T, D)
        Filtered state mean (after observation update),
        x̂_{t|t}.
    P_filt : (T, D, D)
        Filtered state covariance.
    n_observed : (T,) int
        Number of markers contributing observations per frame
        (excluding constraint observations).
    """
    x_pred: np.ndarray
    P_pred: np.ndarray
    x_filt: np.ndarray
    P_filt: np.ndarray
    n_observed: np.ndarray


def _build_constraint_observations(
    state: np.ndarray,
    layout: BodyLayout,
    constraint_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the unit-norm constraint pseudo-observations.

    For each (cos, sin) pair in the state, the constraint is
    cos² + sin² = 1. As a soft observation:
      h_constraint(x) = cos² + sin² - 1
      target z_constraint = 0
      noise variance = constraint_sigma²

    Linearization at the current state gives Jacobian row
    H_row = [..., 2*cos, 2*sin, ...] (zeros elsewhere).

    Returns
    -------
    z : (n_constraints,) — target values, all zero
    h : (n_constraints,) — current values of cos²+sin²-1
    H : (n_constraints, state_dim) — Jacobian
    """
    indices = _pack_state_layout_indices(layout)
    rows: List[np.ndarray] = []
    h_vals: List[float] = []

    # Root constraint
    c = state[indices["__root__"]["cos"]]
    s = state[indices["__root__"]["sin"]]
    h_vals.append(c * c + s * s - 1.0)
    row = np.zeros(layout.state_dim)
    row[indices["__root__"]["cos"]] = 2.0 * c
    row[indices["__root__"]["sin"]] = 2.0 * s
    rows.append(row)

    # Per-segment constraints
    for seg_name in layout.non_root_topo_order:
        c = state[indices[seg_name]["cos"]]
        s = state[indices[seg_name]["sin"]]
        h_vals.append(c * c + s * s - 1.0)
        row = np.zeros(layout.state_dim)
        row[indices[seg_name]["cos"]] = 2.0 * c
        row[indices[seg_name]["sin"]] = 2.0 * s
        rows.append(row)

    n_constraints = len(rows)
    z = np.zeros(n_constraints)
    h = np.array(h_vals)
    H = np.array(rows)
    return z, h, H


def _build_marker_observations(
    state: np.ndarray,
    obs: np.ndarray,            # (K, 2) per-frame marker positions
    likes: np.ndarray,          # (K,) per-frame marker likelihoods
    layout: BodyLayout,
    params: NoiseParamsV2,
    likelihood_threshold: float,
    perspective: Optional["PerspectiveModelV2"] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build the observation vector, predicted observation,
    Jacobian, and observation noise R for one frame's marker
    observations.

    Excludes markers below the likelihood threshold or with
    NaN positions. The remaining markers are stacked into z,
    with corresponding rows of H and diagonal entries of R.

    Returns
    -------
    z : (n_obs,) observation vector
    h : (n_obs,) predicted observation at current state
    H : (n_obs, state_dim) Jacobian
    R_diag : (n_obs,) diagonal of observation noise covariance
    n_observed_markers : int
    """
    fk = forward_kinematics(state, layout)
    pred = state_to_marker_positions(
        state, layout, fk=fk, perspective=perspective,
    )
    H_full = state_to_marker_jacobian(
        state, layout, fk=fk, perspective=perspective,
    )
    marker_names = layout.marker_names

    z_list: List[np.ndarray] = []
    h_list: List[np.ndarray] = []
    H_list: List[np.ndarray] = []
    R_list: List[float] = []
    n_obs = 0

    for k, m in enumerate(marker_names):
        x_obs = obs[k, 0]
        y_obs = obs[k, 1]
        p = likes[k]
        if (
            not (np.isfinite(x_obs) and np.isfinite(y_obs))
            or p < likelihood_threshold
        ):
            continue
        n_obs += 1
        z_list.append(np.array([x_obs, y_obs]))
        h_list.append(pred[k])
        # Two rows of H per marker (x and y)
        H_list.append(H_full[2 * k:2 * k + 2, :])
        # Likelihood-modulated noise
        sigma = params.sigma_marker.get(m, 3.0)
        p_eff = max(p, _P_FLOOR_V2)
        sigma_eff = sigma / p_eff
        R_list.append(sigma_eff ** 2)
        R_list.append(sigma_eff ** 2)

    if n_obs == 0:
        return (
            np.zeros(0), np.zeros(0),
            np.zeros((0, layout.state_dim)),
            np.zeros(0), 0,
        )

    z = np.concatenate(z_list)
    h = np.concatenate(h_list)
    H = np.vstack(H_list)
    R_diag = np.array(R_list)
    return z, h, H, R_diag, n_obs


def forward_filter_v2(
    positions: np.ndarray,        # (T, K, 2)
    likelihoods: np.ndarray,      # (T, K)
    layout: BodyLayout,
    params: NoiseParamsV2,
    dt: float,
    initial_state: np.ndarray,
    initial_cov: Optional[np.ndarray] = None,
    likelihood_threshold: float = 0.5,
    apply_constraints: bool = True,
    perspective: Optional["PerspectiveModelV2"] = None,
) -> FilterResultV2:
    """EKF forward pass on the v2 body-state representation.

    For each frame:
      1. Predict: x_{t|t-1} = F x_{t-1|t-1};
                  P_{t|t-1} = F P_{t-1|t-1} F^T + Q
      2. Build per-frame observations from observed markers
         (excluding low-likelihood / NaN) plus optional unit-
         norm constraint pseudo-observations.
      3. Update: standard EKF update with linearized H from
         the v2 forward kinematics Jacobian.

    Parameters
    ----------
    positions : (T, K, 2)
        Marker observations per frame.
    likelihoods : (T, K)
        Per-marker likelihoods.
    layout : BodyLayout
    params : NoiseParamsV2
    dt : float
        Time step (1 / fps).
    initial_state : (state_dim,)
        Starting state estimate.
    initial_cov : (state_dim, state_dim), optional
        Starting state covariance. Defaults to a moderately
        large diagonal (100 px² for positions, 1.0 for
        unit-vector orientations, length scale² for lengths).
    likelihood_threshold : float
        Markers with p below this are excluded from
        observations.
    apply_constraints : bool
        If True (default), include unit-norm constraint
        observations. Disable for debugging or testing the
        filter without regularization.

    Returns
    -------
    FilterResultV2
    """
    T, K, _ = positions.shape
    D = layout.state_dim
    if initial_state.shape != (D,):
        raise ValueError(
            f"initial_state shape {initial_state.shape} != ({D},)"
        )

    F = build_F_v2(layout, dt)
    Q = build_Q_v2(layout, params, dt)

    if initial_cov is None:
        # Sensible defaults — moderate uncertainty
        P0 = np.eye(D)
        indices = _pack_state_layout_indices(layout)
        # Position uncertainty: 100 px²
        for k in ("x", "y"):
            P0[indices["__root__"][k], indices["__root__"][k]] = 100.0
        # Velocity uncertainty: 1000 px²/s² (the rat could be
        # moving fast or slow at start)
        for k in ("vx", "vy"):
            P0[indices["__root__"][k], indices["__root__"][k]] = 1000.0
        # Orientation uncertainty: small (we initialized from data)
        for k in ("cos", "sin"):
            P0[indices["__root__"][k], indices["__root__"][k]] = 0.1
        for k in ("cos_dot", "sin_dot"):
            P0[indices["__root__"][k], indices["__root__"][k]] = 1.0
        for seg_name in layout.non_root_topo_order:
            for k in ("cos", "sin"):
                P0[indices[seg_name][k], indices[seg_name][k]] = 0.1
            for k in ("cos_dot", "sin_dot"):
                P0[indices[seg_name][k], indices[seg_name][k]] = 1.0
            # Length uncertainty: 1.0 px²
            P0[indices[seg_name]["length"], indices[seg_name]["length"]] = 1.0
            P0[indices[seg_name]["length_dot"], indices[seg_name]["length_dot"]] = 0.01
    else:
        P0 = initial_cov

    x_pred = np.empty((T, D))
    P_pred = np.empty((T, D, D))
    x_filt = np.empty((T, D))
    P_filt = np.empty((T, D, D))
    n_observed = np.zeros(T, dtype=np.int64)

    # Initialize: no predict step at t=0; treat initial as the
    # filter's starting belief.
    x_pred[0] = initial_state
    P_pred[0] = P0
    x_prev = initial_state
    P_prev = P0

    for t in range(T):
        # Predict (skip at t=0; pred[0] is the initial)
        if t > 0:
            x_p = F @ x_prev
            P_p = F @ P_prev @ F.T + Q
            x_pred[t] = x_p
            P_pred[t] = P_p
        else:
            x_p = x_pred[0]
            P_p = P_pred[0]

        # Build observation
        z_m, h_m, H_m, R_diag_m, n_m = _build_marker_observations(
            x_p, positions[t], likelihoods[t], layout, params,
            likelihood_threshold, perspective=perspective,
        )
        n_observed[t] = n_m

        if apply_constraints:
            z_c, h_c, H_c = _build_constraint_observations(
                x_p, layout, params.constraint_sigma,
            )
            z_full = np.concatenate([z_m, z_c])
            h_full = np.concatenate([h_m, h_c])
            H_full = np.vstack([H_m, H_c]) if n_m > 0 else H_c
            R_diag_full = np.concatenate([
                R_diag_m,
                np.full(z_c.shape[0], params.constraint_sigma ** 2),
            ])
        else:
            z_full = z_m
            h_full = h_m
            H_full = H_m
            R_diag_full = R_diag_m

        # EKF update — only if we have any observations
        if z_full.shape[0] > 0:
            innovation = z_full - h_full
            R_full = np.diag(R_diag_full)
            S = H_full @ P_p @ H_full.T + R_full
            # Solve K = P_p @ H^T @ inv(S) via solve for stability
            try:
                K = np.linalg.solve(S.T, H_full @ P_p.T).T
            except np.linalg.LinAlgError:
                # Singular S — skip update for this frame.
                # Shouldn't happen with constraint regularization.
                x_filt[t] = x_p
                P_filt[t] = P_p
                x_prev = x_p
                P_prev = P_p
                continue

            x_f = x_p + K @ innovation
            # Joseph form for numerical stability:
            # P_filt = (I - K H) P_p (I - K H)^T + K R K^T
            I_KH = np.eye(D) - K @ H_full
            P_f = I_KH @ P_p @ I_KH.T + K @ R_full @ K.T
            # Symmetrize against fp drift
            P_f = 0.5 * (P_f + P_f.T)
        else:
            x_f = x_p
            P_f = P_p

        # Hard unit-norm re-projection on each (cos, sin) pair.
        # The soft constraint observations should hold the
        # state near the unit circle, but on real data with
        # long trajectories they aren't reliable enough — once
        # the state drifts even moderately, the linearization
        # at a non-unit-norm point amplifies the drift. Hard
        # projection is the robust fix: after each EKF update,
        # snap each (cos, sin) pair onto the unit circle.
        # This is a standard manifold-EKF trick (see e.g.,
        # Hertzberg et al. 2013, "Integrating Generic Sensor
        # Fusion Algorithms with Sound State Representations
        # through Encapsulation of Manifolds").
        _project_state_to_unit_circle(x_f, layout)

        # NaN/inf detection. If the filter has gone pathological
        # (overflow, division by zero in S inverse), x_f or P_f
        # contain NaN/inf. Recovery: fall back to predict-only
        # (skip the update, propagate P_p with a slight
        # inflation to model the missed observation). If the
        # bad state was introduced in the predict step itself
        # (e.g., from prior NaN), reset to last known-good state.
        if not np.all(np.isfinite(x_f)) or not np.all(np.isfinite(P_f)):
            # Fall back to predict-only at this frame
            if np.all(np.isfinite(x_p)) and np.all(np.isfinite(P_p)):
                x_f = x_p
                P_f = P_p
                _project_state_to_unit_circle(x_f, layout)
            elif t > 0 and np.all(np.isfinite(x_filt[t - 1])):
                # Reset to last known-good filtered state
                x_f = x_filt[t - 1].copy()
                P_f = P_filt[t - 1].copy()
            else:
                # No fallback available — re-initialize from
                # initial state (rare; only first frame and
                # initial state was bad)
                x_f = initial_state.copy()
                P_f = P0.copy()

        x_filt[t] = x_f
        P_filt[t] = P_f
        x_prev = x_f
        P_prev = P_f

    return FilterResultV2(
        x_pred=x_pred, P_pred=P_pred,
        x_filt=x_filt, P_filt=P_filt,
        n_observed=n_observed,
    )


def _project_state_to_unit_circle(
    state: np.ndarray, layout: BodyLayout,
) -> None:
    """Project each (cos, sin) pair in the state onto the unit
    circle, IN PLACE. This is the hard counterpart to the soft
    constraint observations — guarantees |R(cos, sin)| = 1
    after each filter step regardless of how much the state
    drifted.

    The (cos_dot, sin_dot) velocity components are NOT
    projected — they're tangent-space velocities and can have
    any magnitude. This means after projection, the velocity
    might point off-tangent, but Q-noise + the next predict
    step will mix this back into a valid trajectory.

    Why this matters: the EKF tracks (cos, sin) as 2 ambient
    coordinates. Without hard projection, the linearization at
    a state off the unit circle amplifies subsequent drift —
    R = [[c, -s], [s, c]] with c² + s² = 4 has determinant 4,
    and applying it to a vector scales it by 2. A few
    iterations of this and overflow happens.

    Run AFTER each filter update (and could also be run after
    the predict step, but predict alone shouldn't cause
    significant drift).
    """
    indices = _pack_state_layout_indices(layout)

    # Root
    c_idx = indices["__root__"]["cos"]
    s_idx = indices["__root__"]["sin"]
    norm = np.sqrt(state[c_idx] ** 2 + state[s_idx] ** 2)
    if norm > 1e-9:
        state[c_idx] /= norm
        state[s_idx] /= norm
    else:
        # Degenerate — reset to identity
        state[c_idx] = 1.0
        state[s_idx] = 0.0

    # Per-segment
    for seg_name in layout.non_root_topo_order:
        c_idx = indices[seg_name]["cos"]
        s_idx = indices[seg_name]["sin"]
        norm = np.sqrt(state[c_idx] ** 2 + state[s_idx] ** 2)
        if norm > 1e-9:
            state[c_idx] /= norm
            state[s_idx] /= norm
        else:
            state[c_idx] = 1.0
            state[s_idx] = 0.0


# ============================================================
# RTS backward smoother — patch 102
# ============================================================
#
# Given the forward filter's pred and filt arrays (x̂_{t|t-1},
# P_{t|t-1}, x̂_{t|t}, P_{t|t}), produce smoothed estimates
# x̂_{t|T}, P_{t|T} that use ALL observations, not just past.
#
# Standard RTS recursion (linear dynamics, F constant):
#
#   x̂_{T-1|T} = x̂_{T-1|T-1}
#   P_{T-1|T} = P_{T-1|T-1}
#
#   For t = T-2 down to 0:
#     G_t = P_{t|t} F^T inv(P_{t+1|t})
#     x̂_{t|T} = x̂_{t|t} + G_t (x̂_{t+1|T} - x̂_{t+1|t})
#     P_{t|T} = P_{t|t} + G_t (P_{t+1|T} - P_{t+1|t}) G_t^T
#
# Lag-one cross-covariance for the M-step (Shumway-Stoffer):
#
#   P_{t, t+1 | T} = G_t P_{t+1|T}
#
# These are computed during the same backward pass.
#
# Numerical stability: use solve() instead of explicit inverse;
# symmetrize covariances each step.


@dataclass
class SmoothResultV2:
    """RTS smoother output.

    Fields
    ------
    x_smooth : (T, D)
        Smoothed state means x̂_{t|T} using all observations.
    P_smooth : (T, D, D)
        Smoothed state covariances.
    P_lag_one : (T-1, D, D)
        Lag-one cross-covariance: P_lag_one[t] = Cov(x_t,
        x_{t+1} | y_{1:T}). Used by the Shumway-Stoffer
        M-step (patch 103).
    """
    x_smooth: np.ndarray
    P_smooth: np.ndarray
    P_lag_one: np.ndarray


def rts_smooth_v2(
    filter_result: FilterResultV2,
    layout: BodyLayout,
    dt: float,
) -> SmoothResultV2:
    """RTS backward smoother for v2.

    Refines the forward filter's per-frame estimates using
    future observations. For frames where the forward filter
    was uncertain (typically dropout intervals), the smoother
    can reduce uncertainty significantly using the smoothed
    estimate at t+1 to constrain t.

    Also computes the lag-one cross-covariance needed by
    Shumway-Stoffer EM in patch 103.

    Parameters
    ----------
    filter_result : FilterResultV2
        Output of forward_filter_v2.
    layout : BodyLayout
        Used to rebuild F (must match what was used in the
        forward pass).
    dt : float
        Same dt used in the forward pass.

    Returns
    -------
    SmoothResultV2
    """
    T = filter_result.x_filt.shape[0]
    D = layout.state_dim
    F = build_F_v2(layout, dt)

    x_smooth = np.empty_like(filter_result.x_filt)
    P_smooth = np.empty_like(filter_result.P_filt)
    P_lag_one = np.empty((T - 1, D, D))

    # Initialize at the last frame: smoothed = filtered
    x_smooth[T - 1] = filter_result.x_filt[T - 1]
    P_smooth[T - 1] = filter_result.P_filt[T - 1]

    # Backward pass
    for t in range(T - 2, -1, -1):
        P_filt_t = filter_result.P_filt[t]
        x_filt_t = filter_result.x_filt[t]
        P_pred_tp1 = filter_result.P_pred[t + 1]
        x_pred_tp1 = filter_result.x_pred[t + 1]

        # Symmetrize P_pred to guard against fp drift
        P_pred_tp1_sym = 0.5 * (P_pred_tp1 + P_pred_tp1.T)

        # Smoother gain: G_t = P_filt_t @ F^T @ inv(P_pred_tp1)
        # Solve via:
        #   G_t @ P_pred_tp1 = P_filt_t @ F^T
        # i.e., G_t.T = solve(P_pred_tp1.T, F @ P_filt_t.T)
        # (using solve A^T G^T = (P_filt @ F^T)^T = F @ P_filt^T)
        try:
            G_t = np.linalg.solve(
                P_pred_tp1_sym.T, F @ P_filt_t.T,
            ).T
        except np.linalg.LinAlgError:
            # Singular P_pred — add small regularizer and retry
            P_pred_reg = P_pred_tp1_sym + 1e-9 * np.eye(D)
            G_t = np.linalg.solve(
                P_pred_reg.T, F @ P_filt_t.T,
            ).T

        # Smoothed mean and covariance
        x_smooth[t] = x_filt_t + G_t @ (x_smooth[t + 1] - x_pred_tp1)
        P_smooth_t = (
            P_filt_t + G_t @ (P_smooth[t + 1] - P_pred_tp1_sym) @ G_t.T
        )
        # Symmetrize against fp drift
        P_smooth[t] = 0.5 * (P_smooth_t + P_smooth_t.T)

        # Lag-one cross-covariance: P_{t, t+1 | T} = G_t @ P_smooth[t+1]
        # (Shumway-Stoffer Property 6.3)
        P_lag_one[t] = G_t @ P_smooth[t + 1]

    return SmoothResultV2(
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        P_lag_one=P_lag_one,
    )


# ============================================================
# Shumway-Stoffer M-step on body state — patch 103
# ============================================================
#
# After the E-step (forward filter + RTS smoother), the M-step
# fits noise parameters that maximize the expected complete-
# data log-likelihood given the smoothed posterior estimates.
#
# For our linear-dynamics model x_{t+1} = F x_t + w_t with
# w_t ~ N(0, Q):
#
#   Q-hat = (1/(T-1)) (S11 - S10 F^T - F S10^T + F S00 F^T)
#
# where:
#   S00 = Σ_{t=0..T-2} (x̂_t x̂_t^T + P_t)
#   S11 = Σ_{t=0..T-2} (x̂_{t+1} x̂_{t+1}^T + P_{t+1})
#   S10 = Σ_{t=0..T-2} (x̂_{t+1} x̂_t^T + P_{t,t+1|T})
#                                         ↑ lag-one cross-cov
#
# These are the sufficient statistics (E-step output, used in
# M-step). Each is a (D, D) matrix.
#
# σ_marker is fit per-marker from observation residuals:
#
#   σ_m²_hat = (1/(2 N_m)) Σ_t p_t² (||z_t - h(x̂_t)||²
#                                    + trace(H_t P_{t|T} H_t^T))
#
# where 2 N_m is the total observation count (2 axes × N_m
# observation frames) and the trace correction accounts for
# posterior uncertainty in h(x).
#
# Per-block q values are recovered from the diagonal-block
# of Q-hat using the q-scaling template:
#
#   For 4×4 (pos_x, pos_y, vel_x, vel_y) block:
#     Q-hat block diagonals: dt^4/4 (pos), dt² (vel)
#     q-recovered = average(vel-vel diagonals) / dt²
#
#   For 2×2 (pos, vel) block:
#     Q-hat block diagonals: dt^4/4, dt²
#     q-recovered = vel-vel diagonal / dt²
#
# Floors and ceilings (from v1 patches 91-94 lessons):
#   - σ floor: 0.5 px global; ceiling: 3× initial estimate
#   - q floor: max(global_floor, q_initial / 10)
#
# These prevent the degenerate fixed points that plagued v1's
# EM — σ→0 / q→∞ runaway and σ→large / q→0 collapse.


# Per-axis floor for σ. Below this, observations are essentially
# noise-free; setting σ here prevents σ from going to zero.
_M_STEP_FLOOR_SIGMA = _FLOOR_SIGMA_MARKER_V2  # 0.5 from patch 101

# Per-axis ceiling factor: σ ≤ ceiling_factor × σ_initial.
# Prevents σ from running away upward.
_M_STEP_SIGMA_CEILING_FACTOR = 3.0

# Per-axis q floor factor: q ≥ floor_factor × q_initial. Below
# the global floor, we use the global floor.
_M_STEP_Q_FLOOR_FACTOR = 0.1


@dataclass
class _MStepStatsV2:
    """Sufficient statistics accumulated from one or more
    sessions during the E-step. Used by ``finalize_m_step_v2``
    to fit Q and σ.

    Q stats (D, D):
      S00, S11, S10 — sums over transition pairs.
    σ stats (per marker):
      sigma_sum_sq[m] — Σ p_t² (||residual||² + trace_correction)
      sigma_n_obs[m]  — total number of observation axes for m
                        (2 × frames where m was observed)
    n_pairs: total number of transition pairs (T-1 per session,
             summed across sessions).
    """
    S00: np.ndarray
    S11: np.ndarray
    S10: np.ndarray
    n_pairs: int
    sigma_sum_sq: Dict[str, float]
    sigma_n_obs: Dict[str, int]

    @classmethod
    def empty(cls, layout: BodyLayout) -> "_MStepStatsV2":
        D = layout.state_dim
        return cls(
            S00=np.zeros((D, D)),
            S11=np.zeros((D, D)),
            S10=np.zeros((D, D)),
            n_pairs=0,
            sigma_sum_sq={m: 0.0 for m in layout.marker_names},
            sigma_n_obs={m: 0 for m in layout.marker_names},
        )

    def __iadd__(self, other: "_MStepStatsV2") -> "_MStepStatsV2":
        """Combine stats from two sessions (used for multi-
        session EM).
        """
        self.S00 += other.S00
        self.S11 += other.S11
        self.S10 += other.S10
        self.n_pairs += other.n_pairs
        for m in self.sigma_sum_sq:
            self.sigma_sum_sq[m] += other.sigma_sum_sq.get(m, 0.0)
            self.sigma_n_obs[m] += other.sigma_n_obs.get(m, 0)
        return self


def accumulate_m_step_stats_v2(
    smooth_result: SmoothResultV2,
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: BodyLayout,
    likelihood_threshold: float = 0.5,
    perspective: Optional["PerspectiveModelV2"] = None,
) -> _MStepStatsV2:
    """Accumulate sufficient statistics for the M-step from
    one session's smoother output.

    Computes:
      - Q stats (S00, S11, S10) over all transition pairs
        t=0..T-2.
      - σ stats per marker, including trace correction for
        posterior uncertainty in predicted observations.

    Parameters
    ----------
    smooth_result : SmoothResultV2
        Output of rts_smooth_v2.
    positions : (T, K, 2)
    likelihoods : (T, K)
    layout : BodyLayout
    likelihood_threshold : float
        Frames with marker likelihood below this are excluded
        from σ stats for that marker.

    Returns
    -------
    _MStepStatsV2
    """
    T = smooth_result.x_smooth.shape[0]
    D = layout.state_dim
    K = positions.shape[1]
    marker_names = layout.marker_names

    stats = _MStepStatsV2.empty(layout)

    # Q stats: sum over t=0..T-2 of (x_t x_t^T + P_t),
    # (x_{t+1} x_{t+1}^T + P_{t+1}), (x_{t+1} x_t^T + P_{t,t+1}).
    # Vectorize for speed.
    x = smooth_result.x_smooth  # (T, D)
    P = smooth_result.P_smooth  # (T, D, D)
    P_lag = smooth_result.P_lag_one  # (T-1, D, D)

    # S00: sum over t=0..T-2 of x_t x_t^T + P_t
    # → outer products of x[0:T-1] + sum of P[0:T-1]
    x_lo = x[:-1]  # (T-1, D)
    x_hi = x[1:]   # (T-1, D)
    # einsum for outer product sum
    S00 = np.einsum("ti,tj->ij", x_lo, x_lo) + P[:-1].sum(axis=0)
    S11 = np.einsum("ti,tj->ij", x_hi, x_hi) + P[1:].sum(axis=0)
    S10 = np.einsum("ti,tj->ij", x_hi, x_lo) + P_lag.sum(axis=0)
    stats.S00 = S00
    stats.S11 = S11
    stats.S10 = S10
    stats.n_pairs = T - 1

    # σ stats: per-marker.
    # For each frame and each observed marker, compute:
    #   residual_sq = ||z_t - h(x̂_t)||²
    #   trace_corr = trace(H_t,m P_{t|T} H_t,m^T)
    #   weighted contribution = p_t² * (residual_sq + trace_corr)
    for t in range(T):
        # Build h(x̂_t) and H at x̂_t
        fk = forward_kinematics(x[t], layout)
        pred = state_to_marker_positions(
            x[t], layout, fk=fk, perspective=perspective,
        )
        H_full = state_to_marker_jacobian(
            x[t], layout, fk=fk, perspective=perspective,
        )
        # H_full is (2K, D)

        for k, m in enumerate(marker_names):
            x_obs = positions[t, k, 0]
            y_obs = positions[t, k, 1]
            p = likelihoods[t, k]
            if (
                not (np.isfinite(x_obs) and np.isfinite(y_obs))
                or p < likelihood_threshold
            ):
                continue

            residual = np.array([
                x_obs - pred[k, 0],
                y_obs - pred[k, 1],
            ])
            residual_sq = float(residual @ residual)
            # trace(H_m P H_m^T) = sum over rows of H_m of
            # H_row @ P @ H_row^T.  H_m is (2, D).
            H_m = H_full[2 * k:2 * k + 2, :]
            trace_corr = float(np.trace(H_m @ P[t] @ H_m.T))

            w = p * p
            stats.sigma_sum_sq[m] += w * (residual_sq + trace_corr)
            stats.sigma_n_obs[m] += 2  # x and y axes

    return stats


def finalize_m_step_v2(
    stats: _MStepStatsV2,
    layout: BodyLayout,
    dt: float,
    prev_params: NoiseParamsV2,
    initial_params: Optional[NoiseParamsV2] = None,
) -> NoiseParamsV2:
    """Convert accumulated sufficient statistics into refined
    NoiseParamsV2.

    Computes Q-hat from S00/S11/S10/F, then extracts per-block
    q values. Computes σ-hat from per-marker stats. Applies
    floors and ceilings to prevent EM from settling on the
    degenerate fixed points that plagued v1.

    Parameters
    ----------
    stats : _MStepStatsV2
        Accumulated sufficient statistics.
    layout : BodyLayout
    dt : float
    prev_params : NoiseParamsV2
        Previous iteration's params, used as fallback for
        markers / segments with insufficient data.
    initial_params : NoiseParamsV2, optional
        Initial (iteration-0) params, used for floor/ceiling
        computation. If None, uses prev_params.

    Returns
    -------
    NoiseParamsV2
    """
    if initial_params is None:
        initial_params = prev_params

    F = build_F_v2(layout, dt)
    n_pairs = max(stats.n_pairs, 1)

    # Q-hat = (1/n_pairs) * (S11 - S10 F^T - F S10^T + F S00 F^T)
    Q_hat = (
        stats.S11
        - stats.S10 @ F.T
        - F @ stats.S10.T
        + F @ stats.S00 @ F.T
    ) / n_pairs
    # Symmetrize against fp drift
    Q_hat = 0.5 * (Q_hat + Q_hat.T)

    indices = _pack_state_layout_indices(layout)
    dt2 = dt * dt

    # Helper: extract q from a 4×4 (pos_x, pos_y, vel_x, vel_y)
    # block. Use velocity-velocity diagonals (Q[2,2] = Q[3,3]
    # = q * dt²).
    def _q_from_4block(block: np.ndarray) -> float:
        return float((block[2, 2] + block[3, 3]) / 2.0 / dt2)

    # Helper: extract q from a 2×2 (pos, vel) block.
    def _q_from_2block(block: np.ndarray) -> float:
        return float(block[1, 1] / dt2)

    # ---------- Q components ----------

    # Root translation: indices 0:4 (x, y, vx, vy)
    q_root_pos_raw = _q_from_4block(Q_hat[0:4, 0:4])
    q_root_pos_floor = max(
        _FLOOR_Q_ROOT_POS,
        initial_params.q_root_pos * _M_STEP_Q_FLOOR_FACTOR,
    )
    q_root_pos = max(q_root_pos_raw, q_root_pos_floor)

    # Root orientation: indices 4:8
    q_root_ori_raw = _q_from_4block(Q_hat[4:8, 4:8])
    q_root_ori_floor = max(
        _FLOOR_Q_ROOT_ORI,
        initial_params.q_root_ori * _M_STEP_Q_FLOOR_FACTOR,
    )
    q_root_ori = max(q_root_ori_raw, q_root_ori_floor)

    # Per-segment orientation
    q_seg_ori: Dict[str, float] = {}
    for seg_name in layout.non_root_topo_order:
        sl = layout.slice_segment_orientation(seg_name)
        block = Q_hat[sl, sl]
        q_raw = _q_from_4block(block)
        q_initial = initial_params.q_seg_ori.get(
            seg_name, _FLOOR_Q_SEG_ORI,
        )
        q_floor = max(
            _FLOOR_Q_SEG_ORI, q_initial * _M_STEP_Q_FLOOR_FACTOR,
        )
        q_seg_ori[seg_name] = max(q_raw, q_floor)

    # Per-segment length
    q_length: Dict[str, float] = {}
    for seg_name in layout.non_root_topo_order:
        sl = layout.slice_segment_length(seg_name)
        block = Q_hat[sl, sl]
        q_raw = _q_from_2block(block)
        q_initial = initial_params.q_length.get(
            seg_name, _FLOOR_Q_LENGTH,
        )
        q_floor = max(
            _FLOOR_Q_LENGTH, q_initial * _M_STEP_Q_FLOOR_FACTOR,
        )
        q_length[seg_name] = max(q_raw, q_floor)

    # ---------- σ components ----------

    sigma_marker: Dict[str, float] = {}
    for m in layout.marker_names:
        n_obs = stats.sigma_n_obs.get(m, 0)
        if n_obs < 4:
            # Not enough observations — keep previous value
            sigma_marker[m] = prev_params.sigma_marker.get(m, 3.0)
            continue
        # σ²_hat = (1/n_obs) * sum_sq
        # n_obs already counts 2 axes per frame
        sigma_sq = stats.sigma_sum_sq[m] / n_obs
        sigma_raw = float(np.sqrt(max(sigma_sq, 0.0)))
        # Floor and ceiling
        sigma_floor = _M_STEP_FLOOR_SIGMA
        sigma_initial = initial_params.sigma_marker.get(m, 3.0)
        sigma_ceiling = _M_STEP_SIGMA_CEILING_FACTOR * sigma_initial
        sigma_marker[m] = float(
            max(sigma_floor, min(sigma_ceiling, sigma_raw))
        )

    return NoiseParamsV2(
        sigma_marker=sigma_marker,
        q_root_pos=q_root_pos,
        q_root_ori=q_root_ori,
        q_seg_ori=q_seg_ori,
        q_length=q_length,
        constraint_sigma=prev_params.constraint_sigma,
    )


# ============================================================
# Data-driven parameter initialization + EM loop +
# validation hook — patch 104
# ============================================================
#
# This patch ties together patches 99-103 into a working
# multi-iteration EM that fits NoiseParamsV2 to one or more
# sessions. Three components:
#
# 1. fit_initial_params_v2: estimates σ_marker from MA-residuals
#    and q_root_pos from window-variance of the root marker
#    (the v1 patch-94 approach ported to v2). This is the
#    fix for the slow-EM-convergence issue documented in
#    patch 103.
#
# 2. fit_noise_params_em_v2: the EM loop. Multi-session aware
#    via per-session E-step + combined M-step.
#
# 3. _validate_trajectory_v2: post-E-step hook that catches
#    degenerate trajectories (frozen smoothed, prior overruling
#    data) and raises RuntimeError with per-marker breakdown.


_INIT_FLOOR_Q_ROOT_POS_V2 = 100.0
_INIT_DEFAULT_Q_ROOT_ORI = 1.0
_INIT_DEFAULT_Q_SEG_ORI = 0.5
_INIT_DEFAULT_Q_LENGTH = 0.01


def fit_initial_params_v2(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: BodyLayout,
    marker_names: List[str],
    fitted_lengths: FittedLengths,
    fps: float,
    likelihood_threshold: float = 0.5,
) -> NoiseParamsV2:
    """Estimate initial NoiseParamsV2 from data.

    Calibrates EM's starting point so it doesn't have to
    grow q from arbitrary defaults across many iterations.

    σ_marker estimation
    -------------------

    Per marker, residuals from a 5-frame moving average over
    high-confidence frames give a rough observation noise
    estimate. Falls back to 2.0 px if insufficient data.

    q_root_pos estimation
    ---------------------

    Window-variance estimator (patch 94 from v1, ported here):

      For sliding 1-second windows, compute Var(positions in
      window). Under a CV model:
        Var(x in window) ≈ σ² + (1/12) q T_w³
      → q ≈ 12 * (mean_window_var - σ²) / T_w³

    Uses all observations in each window (not just consecutive
    high-p pairs), so it's robust to sparse coverage.  Mean
    across windows captures motion across all behavioral
    regimes.

    q_root_ori estimation
    ---------------------

    Compute the body-orientation proxy per frame from
    (back1 - back3) — direction the rat is facing. The
    angle θ_t = atan2(dy, dx). Window-variance of θ →
    q_root_ori. Uses unwrapped angle to avoid wraparound.

    Other q values
    --------------

    q_seg_ori per segment: default value (refined by EM).
    q_length per segment: default value calibrated by IQR
                          if available.

    Parameters
    ----------
    positions : (T, K, 2)
    likelihoods : (T, K)
    layout : BodyLayout
    marker_names : list[str]
        Marker names matching the columns of positions /
        likelihoods.
    fitted_lengths : FittedLengths
        From fit_body_lengths; provides marker offsets and
        length IQR.
    fps : float
    likelihood_threshold : float

    Returns
    -------
    NoiseParamsV2
    """
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    dt = 1.0 / fps
    T = positions.shape[0]
    window_frames = min(int(round(fps)), max(T // 4, 5))
    T_w = window_frames * dt

    # ---------- σ_marker ----------
    sigma_marker: Dict[str, float] = {}
    for m in layout.marker_names:
        if m not in name_to_idx:
            sigma_marker[m] = 2.0
            continue
        i = name_to_idx[m]
        x = positions[:, i, 0]
        y = positions[:, i, 1]
        p = likelihoods[:, i]
        mask = (
            (p >= likelihood_threshold)
            & np.isfinite(x) & np.isfinite(y)
        )
        if mask.sum() < 20:
            sigma_marker[m] = 2.0
            continue
        x_clean = x[mask]
        y_clean = y[mask]
        ma_w = 5
        kernel = np.ones(ma_w) / ma_w
        x_ma = np.convolve(x_clean, kernel, mode="valid")
        y_ma = np.convolve(y_clean, kernel, mode="valid")
        x_resid = x_clean[ma_w // 2: ma_w // 2 + len(x_ma)] - x_ma
        y_resid = y_clean[ma_w // 2: ma_w // 2 + len(y_ma)] - y_ma
        sigma_est = float(
            np.sqrt(0.5 * (np.var(x_resid) + np.var(y_resid)))
        )
        sigma_marker[m] = max(sigma_est, _FLOOR_SIGMA_MARKER_V2)

    # ---------- q_root_pos ----------
    # Use root distal marker (back2 in standard rat). Estimate
    # from window-variance of position.
    root_distal = _segment_distal_marker(layout, layout.root_segment.name)
    q_root_pos = _INIT_FLOOR_Q_ROOT_POS_V2
    if root_distal is not None and root_distal in name_to_idx:
        i = name_to_idx[root_distal]
        x = positions[:, i, 0]
        y = positions[:, i, 1]
        p = likelihoods[:, i]
        mask = (
            (p >= likelihood_threshold)
            & np.isfinite(x) & np.isfinite(y)
        )
        if mask.sum() >= 20:
            n_windows = T // window_frames
            window_vars: List[float] = []
            for w in range(n_windows):
                start = w * window_frames
                end = start + window_frames
                wm = mask[start:end]
                if wm.sum() < 5:
                    continue
                wx = x[start:end][wm]
                wy = y[start:end][wm]
                window_vars.append(0.5 * (np.var(wx) + np.var(wy)))
            if len(window_vars) >= 3:
                mean_var = float(np.mean(window_vars))
                sigma_root = sigma_marker.get(root_distal, 2.0)
                motion_var = max(mean_var - sigma_root ** 2, 0.0)
                q_est = 12.0 * motion_var / (T_w ** 3)
                q_root_pos = max(q_est, _INIT_FLOOR_Q_ROOT_POS_V2)

    # ---------- q_root_ori ----------
    # Body orientation proxy: angle of (back1 - back3) per
    # frame. Use unwrapped angle for variance computation.
    q_root_ori = _INIT_DEFAULT_Q_ROOT_ORI
    if (
        "back1" in name_to_idx and "back3" in name_to_idx
    ):
        i1 = name_to_idx["back1"]
        i3 = name_to_idx["back3"]
        p1 = likelihoods[:, i1]
        p3 = likelihoods[:, i3]
        x1 = positions[:, i1, 0]
        y1 = positions[:, i1, 1]
        x3 = positions[:, i3, 0]
        y3 = positions[:, i3, 1]
        mask = (
            (p1 >= likelihood_threshold) & (p3 >= likelihood_threshold)
            & np.isfinite(x1) & np.isfinite(y1)
            & np.isfinite(x3) & np.isfinite(y3)
        )
        if mask.sum() >= 50:
            dx = x1[mask] - x3[mask]
            dy = y1[mask] - y3[mask]
            theta = np.arctan2(dy, dx)
            theta_unwrapped = np.unwrap(theta)
            # Window-variance of orientation. Convert angular
            # variance per second² to ambient (cos, sin)
            # acceleration variance using small-angle
            # approximation: d/dt(cos) ≈ -sin * ω, so for
            # tracking purposes the ambient q ≈ angular q.
            n_w = len(theta_unwrapped) // window_frames
            wv: List[float] = []
            for w in range(n_w):
                start = w * window_frames
                end = start + window_frames
                if end > len(theta_unwrapped):
                    break
                wv.append(np.var(theta_unwrapped[start:end]))
            if len(wv) >= 3:
                mean_var = float(np.mean(wv))
                # Constraint variance ≈ obs noise on θ from
                # σ_root / d_root3-to-1. Small if back markers
                # are several pixels apart.
                q_est = 12.0 * mean_var / (T_w ** 3)
                q_root_ori = max(q_est, _FLOOR_Q_ROOT_ORI)

    # ---------- q_seg_ori, q_length ----------
    q_seg_ori: Dict[str, float] = {}
    q_length: Dict[str, float] = {}
    for seg_name in layout.non_root_topo_order:
        # Default per category — head/tail are more mobile
        if seg_name in ("head", "neck"):
            q_seg_ori[seg_name] = 2.0
        elif seg_name.startswith("tail"):
            q_seg_ori[seg_name] = 5.0
        else:
            q_seg_ori[seg_name] = _INIT_DEFAULT_Q_SEG_ORI

        # q_length from IQR if available, else default
        iqr = fitted_lengths.segment_length_iqr.get(seg_name, 0.0)
        if iqr > 0:
            # Treat IQR as a 1-second standard deviation proxy
            q_length[seg_name] = max(
                (iqr ** 2) / (T_w ** 2), _FLOOR_Q_LENGTH,
            )
        else:
            q_length[seg_name] = _INIT_DEFAULT_Q_LENGTH

    return NoiseParamsV2(
        sigma_marker=sigma_marker,
        q_root_pos=q_root_pos,
        q_root_ori=q_root_ori,
        q_seg_ori=q_seg_ori,
        q_length=q_length,
    )


def fit_warm_start_sigma_v2(
    sessions: List[Tuple[np.ndarray, np.ndarray]],
    layout: BodyLayout,
    marker_names: List[str],
    fitted_lengths: FittedLengths,
    params: NoiseParamsV2,
    fps: float,
    likelihood_threshold: float = 0.5,
    sigma_inflation_cap: float = 20.0,
    apply_constraints: bool = True,
    perspective: Optional["PerspectiveModelV2"] = None,
) -> Dict[str, float]:
    """Warm-start σ_marker estimation: run filter+smoother
    once with current params, measure mean |smoothed - raw|
    per marker, use that as a *lower bound* for σ_marker.

    Why this matters
    ----------------

    The MA-residual estimator in fit_initial_params_v2 captures
    only frame-to-frame jitter (typically 1-3 px). For body
    markers at the widest part of the rat (lateral_left/
    lateral_right) and other markers where rigid-offset
    prediction has structural mismatch (ear positions varying
    with head orientation), the actual σ should also account
    for postural variation — typically 5-30 px depending on
    the marker.

    The MA-residual approach systematically underestimates σ
    for these markers, causing the validation hook to fire
    even though the smoother is doing its job correctly.

    Solution: a single warm-up pass to MEASURE structural
    residuals from the smoother's predictions, use that as a
    lower bound for σ_marker before EM.

    For Gaussian residuals: mean(|residual|) ≈ σ × √(2/π) ≈ 0.8σ
    So σ ≈ mean_diff × 1.25.

    sigma_inflation_cap protects against the degenerate case
    where the smoother itself is broken (mean_diff huge for
    every marker). If warm-start σ is more than
    sigma_inflation_cap × initial σ, that's a sign the warm-
    start is unreliable and we keep the initial σ.

    Parameters
    ----------
    sessions : list of (positions, likelihoods)
    layout : BodyLayout
    marker_names : list[str]
    fitted_lengths : FittedLengths
    params : NoiseParamsV2
        Current params; used to run the warm-up smoother and
        as the floor for the new σ.
    fps : float
    likelihood_threshold : float
    sigma_inflation_cap : float, default 20
        Maximum factor by which warm-start σ can exceed
        initial σ. Above this, fall back to initial σ
        (warm-start unreliable).
    apply_constraints : bool

    Returns
    -------
    sigma_warm : Dict[str, float]
        Per-marker warm-started σ, ≥ initial σ.
    """
    dt = 1.0 / fps
    # Per-marker accumulators across all sessions
    sum_abs_diff: Dict[str, float] = {m: 0.0 for m in layout.marker_names}
    n_obs: Dict[str, int] = {m: 0 for m in layout.marker_names}

    for pos, likes in sessions:
        # Run a forward filter + smoother with current params.
        x0 = initial_state_from_data(
            pos, likes, layout, marker_names,
            fitted_lengths, likelihood_threshold,
        )
        filt = forward_filter_v2(
            pos, likes, layout, params, dt,
            initial_state=x0,
            likelihood_threshold=likelihood_threshold,
            apply_constraints=apply_constraints,
            perspective=perspective,
        )
        smooth = rts_smooth_v2(filt, layout, dt)

        T = smooth.x_smooth.shape[0]
        # Predicted marker positions per frame
        for t in range(T):
            pred = state_to_marker_positions(
                smooth.x_smooth[t], layout,
                perspective=perspective,
            )
            for k, m in enumerate(layout.marker_names):
                x_obs = pos[t, k, 0]
                y_obs = pos[t, k, 1]
                p = likes[t, k]
                if (
                    not (np.isfinite(x_obs) and np.isfinite(y_obs))
                    or p < likelihood_threshold
                    or not np.all(np.isfinite(pred[k]))
                ):
                    continue
                diff = np.sqrt(
                    (pred[k, 0] - x_obs) ** 2
                    + (pred[k, 1] - y_obs) ** 2
                )
                sum_abs_diff[m] += float(diff)
                n_obs[m] += 1

    sigma_warm: Dict[str, float] = {}
    for m in layout.marker_names:
        sigma_init = params.sigma_marker.get(m, 3.0)
        if n_obs[m] < 50:
            # Insufficient data for warm-start; keep initial
            sigma_warm[m] = sigma_init
            continue
        mean_diff = sum_abs_diff[m] / n_obs[m]
        # Convert mean abs diff to σ estimate (Gaussian: σ ≈
        # mean_diff / 0.8). Also accounts for the smoother
        # already absorbing some noise — the residuals here
        # are after the smoother, so they reflect what the
        # smoother CAN'T explain (mostly structural).
        sigma_residual = mean_diff / 0.8
        # Lower bound at initial σ
        sigma_candidate = max(sigma_init, sigma_residual)
        # Cap inflation factor
        sigma_max = sigma_init * sigma_inflation_cap
        if sigma_candidate > sigma_max:
            # Warm-start unreliable (smoother probably broken).
            # Stay at initial σ.
            sigma_warm[m] = sigma_init
        else:
            sigma_warm[m] = sigma_candidate

    return sigma_warm


def _validate_trajectory_v2(
    smooth: SmoothResultV2,
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: BodyLayout,
    params: NoiseParamsV2,
    iteration: int,
    likelihood_threshold: float,
    range_ratio_floor: float = 0.1,
    mean_diff_sigma_factor: float = 8.0,
    min_observation_fraction: float = 0.05,
    verbose: bool = False,
) -> None:
    """Validation hook: catches degenerate trajectories.

    Two checks per marker (port from v1 patch 91-93):

    1. range_ratio: range of smoothed marker positions
       should be at least range_ratio_floor (default 0.1) of
       the range of raw observations on high-confidence
       frames. A smoothed range much smaller than raw means
       the smoother is freezing the trajectory — typically
       a sign that q has collapsed too far.

    2. mean_diff: mean |predicted - observed| on high-p frames
       should be ≤ mean_diff_sigma_factor (default 5) × σ_marker.
       If predicted deviates strongly from data on confident
       frames, the prior is overruling observations — typically
       a sign that σ has collapsed too far down.

    Raises RuntimeError on the first marker that fails either
    check, with diagnostic info (per-marker breakdown when
    verbose).

    Parameters
    ----------
    smooth : SmoothResultV2
    positions : (T, K, 2)
    likelihoods : (T, K)
    layout : BodyLayout
    params : NoiseParamsV2
    iteration : int
        EM iteration number (for error message).
    likelihood_threshold : float
    range_ratio_floor : float
    mean_diff_sigma_factor : float
    verbose : bool
        If True, print per-marker check results before any
        failure.
    """
    T = smooth.x_smooth.shape[0]
    K = positions.shape[1]
    marker_names = layout.marker_names

    # Compute predicted marker positions from smoothed states
    # for every frame.
    pred = np.zeros((T, K, 2))
    for t in range(T):
        pred[t] = state_to_marker_positions(smooth.x_smooth[t], layout)

    # For each marker, compute range_ratio and mean_diff
    if verbose:
        print(
            f"[v2-em-val] iter {iteration}: per-marker stats "
            f"(range_ratio>{range_ratio_floor:.2f} ok; "
            f"mean_diff<{mean_diff_sigma_factor:.0f}σ ok):"
        )

    # Global NaN check — if any predicted positions are NaN,
    # the EKF has gone pathological. Raise with a clear
    # message rather than producing all-NaN per-marker stats.
    n_nan = int(np.isnan(pred).sum())
    if n_nan > 0:
        raise RuntimeError(
            f"v2 EM validation hook triggered at iteration "
            f"{iteration}: smoothed marker predictions contain "
            f"{n_nan} NaN values out of {pred.size}. The EKF has "
            f"diverged — likely causes: orientation state drifted "
            f"off the unit circle (should be prevented by hard "
            f"projection in patch 106), initial state was "
            f"inconsistent with observations, or process noise Q "
            f"is too large. Inspect the early frames of x_smooth "
            f"to diagnose."
        )

    failures: List[str] = []
    T = positions.shape[0]
    for k, m in enumerate(marker_names):
        sigma_m = params.sigma_marker.get(m, 3.0)

        x_obs = positions[:, k, 0]
        y_obs = positions[:, k, 1]
        p = likelihoods[:, k]
        high_p_mask = (
            (p >= likelihood_threshold)
            & np.isfinite(x_obs) & np.isfinite(y_obs)
        )
        n_hp = int(high_p_mask.sum())

        # range_ratio: compute range of smoothed and raw on
        # high-p frames. If too few high-p frames, skip.
        if n_hp < 5:
            if verbose:
                print(
                    f"[v2-em-val]   {m:<14s}  skipped "
                    f"(only n_high_p={n_hp})"
                )
            continue

        # Skip range_ratio when the marker is rarely observed.
        # In sparse-observation regimes (e.g., tailend with
        # n_high_p=196 in a 53k-frame session), the raw range
        # is computed over a small subset of frames while the
        # smoothed range covers the whole trajectory — these
        # aren't comparable. The smoother is filling in long
        # gaps via spatial coupling and naturally explores
        # more space than the sparse observations.
        obs_fraction = n_hp / max(T, 1)
        skip_range_check = obs_fraction < min_observation_fraction

        raw_x_range = float(x_obs[high_p_mask].max() - x_obs[high_p_mask].min())
        raw_y_range = float(y_obs[high_p_mask].max() - y_obs[high_p_mask].min())
        raw_range = max(raw_x_range, raw_y_range)
        sm_x = pred[:, k, 0]
        sm_y = pred[:, k, 1]
        # Compare smoothed range over the SAME frames where
        # raw observations exist (high-p frames). Apples-to-
        # apples comparison.
        sm_x_range = float(sm_x[high_p_mask].max() - sm_x[high_p_mask].min())
        sm_y_range = float(sm_y[high_p_mask].max() - sm_y[high_p_mask].min())
        sm_range = max(sm_x_range, sm_y_range)
        # range_ratio is only meaningful when raw_range
        # substantially exceeds observation noise. For
        # nearly-stationary markers, raw_range is dominated by
        # observation noise (~6σ peak-to-peak), and the
        # smoother's job is to suppress that noise — so
        # smoothed range << raw range is CORRECT, not frozen.
        # Skip the check when raw range is below ~10× σ
        # (equivalent to the marker being effectively
        # stationary).
        raw_motion_threshold = 10.0 * sigma_m
        if skip_range_check or raw_range <= raw_motion_threshold:
            range_ratio = 1.0  # Skipped; pass
        else:
            range_ratio = sm_range / raw_range

        # mean_diff: mean Euclidean distance smoothed vs raw
        # on high-p frames
        diff = np.sqrt(
            (sm_x[high_p_mask] - x_obs[high_p_mask]) ** 2
            + (sm_y[high_p_mask] - y_obs[high_p_mask]) ** 2
        )
        mean_diff = float(np.mean(diff))
        sigma_5 = mean_diff_sigma_factor * sigma_m

        range_ok = range_ratio >= range_ratio_floor
        mean_ok = mean_diff <= sigma_5

        if verbose:
            print(
                f"[v2-em-val]   {m:<14s}  "
                f"range_ratio={range_ratio:5.2f}  "
                f"({'ok' if range_ok else 'FROZEN'})  "
                f"mean_diff={mean_diff:6.2f}px  "
                f"{int(mean_diff_sigma_factor)}σ={sigma_5:6.2f}px  "
                f"({'ok' if mean_ok else 'OVERRULE'})  "
                f"n_high_p={n_hp}"
            )

        if not range_ok:
            failures.append(
                f"frozen-output check failed for marker {m!r}: "
                f"smoothed range {sm_range:.2f}px is "
                f"{range_ratio:.4f}x raw range {raw_range:.2f}px "
                f"(threshold: {range_ratio_floor:.2f})"
            )
        if not mean_ok:
            failures.append(
                f"prior-overruling-data check failed for marker "
                f"{m!r}: mean |smoothed - raw| at high-p frames "
                f"= {mean_diff:.2f}px, exceeds "
                f"{mean_diff_sigma_factor:.0f}x current sigma_marker "
                f"({sigma_5:.2f}px). This indicates the prior is "
                f"systematically pulling away from trusted "
                f"observations."
            )

    if failures:
        raise RuntimeError(
            f"v2 EM validation hook triggered at iteration "
            f"{iteration}: " + "; ".join(failures)
        )


@dataclass
class EMResultV2:
    """Output of fit_noise_params_em_v2.

    Fields
    ------
    params : NoiseParamsV2
        Final fitted noise parameters.
    history : list of dicts
        Per-iteration diagnostics: iter number, max change,
        mean σ across markers, mean q values.
    initial_params : NoiseParamsV2
        Initial parameters (data-driven estimate).
    converged : bool
    perspective : PerspectiveModelV2 or None
        Fitted perspective correction model, or None if
        perspective fitting was disabled.
    """
    params: NoiseParamsV2
    history: List[Dict[str, float]]
    initial_params: NoiseParamsV2
    converged: bool
    perspective: Optional["PerspectiveModelV2"] = None


def _max_param_change(
    new: NoiseParamsV2, old: NoiseParamsV2,
) -> float:
    """Compute max relative change in any parameter."""
    rels: List[float] = []
    for m, sigma_new in new.sigma_marker.items():
        sigma_old = old.sigma_marker.get(m, sigma_new)
        if sigma_old > 1e-9:
            rels.append(abs(sigma_new - sigma_old) / sigma_old)
    if old.q_root_pos > 1e-9:
        rels.append(abs(new.q_root_pos - old.q_root_pos) / old.q_root_pos)
    if old.q_root_ori > 1e-9:
        rels.append(abs(new.q_root_ori - old.q_root_ori) / old.q_root_ori)
    for s, q_new in new.q_seg_ori.items():
        q_old = old.q_seg_ori.get(s, q_new)
        if q_old > 1e-9:
            rels.append(abs(q_new - q_old) / q_old)
    for s, q_new in new.q_length.items():
        q_old = old.q_length.get(s, q_new)
        if q_old > 1e-9:
            rels.append(abs(q_new - q_old) / q_old)
    return max(rels) if rels else 0.0


def fit_noise_params_em_v2(
    sessions: List[Tuple[np.ndarray, np.ndarray]],
    layout: BodyLayout,
    marker_names: List[str],
    fitted_lengths: FittedLengths,
    fps: float,
    likelihood_threshold: float = 0.5,
    max_iter: int = 10,
    tol: float = 1e-3,
    initial_params: Optional[NoiseParamsV2] = None,
    apply_constraints: bool = True,
    enable_validation: bool = True,
    enable_warm_start_sigma: bool = True,
    enable_perspective: bool = True,
    verbose: bool = False,
) -> EMResultV2:
    """Multi-session EM for v2 noise parameters.

    Each iteration:
      For each session:
        - Build initial state from data
        - Forward filter
        - RTS smooth
        - Accumulate M-step stats (per session, then combined)
        - Validation hook (warns/raises if degenerate)
      Combined M-step finalizes new params.

    Parameters
    ----------
    sessions : list of (positions, likelihoods)
        Each session's (T, K, 2) positions and (T, K)
        likelihoods. Sessions can have different T values.
    layout : BodyLayout
    marker_names : list[str]
        Column names for the positions / likelihoods arrays.
    fitted_lengths : FittedLengths
        Fit body lengths from data once before EM.
    fps : float
    likelihood_threshold : float
    max_iter : int
    tol : float
        Convergence threshold on max relative parameter change.
    initial_params : NoiseParamsV2, optional
        If None, computed from data via fit_initial_params_v2.
    apply_constraints : bool
        Pass through to forward filter.
    enable_validation : bool
        If True, run validation hook each iteration.
    verbose : bool

    Returns
    -------
    EMResultV2
    """
    dt = 1.0 / fps

    # Data-driven initial params (uses first session — assumed
    # representative; for multi-session, all sessions have
    # similar tracking quality)
    if initial_params is None:
        first_pos, first_likes = sessions[0]
        initial_params = fit_initial_params_v2(
            first_pos, first_likes, layout, marker_names,
            fitted_lengths, fps, likelihood_threshold,
        )

    if verbose:
        print(
            f"[v2-em] initial: σ̄ = "
            f"{np.mean(list(initial_params.sigma_marker.values())):.3f}px, "
            f"q_root_pos = {initial_params.q_root_pos:.1f}, "
            f"q_root_ori = {initial_params.q_root_ori:.3f}"
        )

    # Warm-start σ_marker: run a single filter+smoother pass
    # with current params, measure mean |smoothed - raw| per
    # marker, use that as a lower bound for σ_marker. This
    # absorbs structural variation (postural sway in body
    # markers like lateral_left/right) that the raw MA-residual
    # estimator misses. Without this, EM struggles to grow σ
    # for these markers and validation may false-positive.
    if enable_warm_start_sigma:
        if verbose:
            print(
                "[v2-em] Running warm-start σ pass to absorb "
                "structural variation..."
            )
        sigma_warm = fit_warm_start_sigma_v2(
            sessions, layout, marker_names, fitted_lengths,
            initial_params, fps,
            likelihood_threshold=likelihood_threshold,
            apply_constraints=apply_constraints,
        )
        # Update initial_params with warm-started σ
        initial_params = NoiseParamsV2(
            sigma_marker=sigma_warm,
            q_root_pos=initial_params.q_root_pos,
            q_root_ori=initial_params.q_root_ori,
            q_seg_ori=dict(initial_params.q_seg_ori),
            q_length=dict(initial_params.q_length),
            constraint_sigma=initial_params.constraint_sigma,
        )
        if verbose:
            sigma_changes = []
            for m in layout.marker_names:
                sigma_changes.append(
                    f"{m}={sigma_warm[m]:.2f}"
                )
            print(
                f"[v2-em] warm-start σ: "
                f"σ̄ = {np.mean(list(sigma_warm.values())):.3f}px, "
                f"per-marker: {', '.join(sigma_changes)}"
            )

    # Fit perspective model: a per-marker bilinear scale
    # function that captures lens distortion / camera-tilt
    # effects where apparent body width depends on the rat's
    # location in the arena. Fit ONCE before EM iterations
    # begin; not refit during EM (would be expensive and
    # interacts badly with σ EM step).
    perspective: Optional[PerspectiveModelV2] = None
    if enable_perspective:
        if verbose:
            print(
                "[v2-em] Fitting perspective model "
                "(per-marker bilinear scale)..."
            )
        perspective = fit_perspective_model_v2(
            sessions, layout, marker_names, fitted_lengths,
            initial_params, fps,
            likelihood_threshold=likelihood_threshold,
            apply_constraints=apply_constraints,
        )
        if verbose:
            # Report per-marker max |scale - 1| at arena edges
            max_corrections = []
            for m, c in perspective.coeffs.items():
                # Max correction at corner: 1 + a + b + c
                # Bounds: |a|+|b|+|c| (worst case)
                max_corr = float(abs(c[0]) + abs(c[1]) + abs(c[2]))
                if max_corr > 0.01:
                    max_corrections.append(
                        f"{m}={max_corr:.2f}"
                    )
            print(
                f"[v2-em] perspective: max |Δscale| at corners "
                f"per marker (>1% only): "
                f"{', '.join(max_corrections) if max_corrections else 'all <1%'}"
            )

    params = initial_params
    history: List[Dict[str, float]] = []
    converged = False

    for iteration in range(max_iter):
        # E-step + stat accumulation across sessions
        combined_stats = _MStepStatsV2.empty(layout)

        for sess_idx, (pos, likes) in enumerate(sessions):
            # Build initial state for this session
            x0 = initial_state_from_data(
                pos, likes, layout, marker_names,
                fitted_lengths, likelihood_threshold,
            )
            # Forward filter + smoother
            filt = forward_filter_v2(
                pos, likes, layout, params, dt,
                initial_state=x0,
                likelihood_threshold=likelihood_threshold,
                apply_constraints=apply_constraints,
                perspective=perspective,
            )
            smooth = rts_smooth_v2(filt, layout, dt)

            # Validation hook
            if enable_validation:
                _validate_trajectory_v2(
                    smooth, pos, likes, layout, params,
                    iteration=iteration,
                    likelihood_threshold=likelihood_threshold,
                    verbose=verbose and sess_idx == 0,
                )

            # Accumulate M-step stats
            sess_stats = accumulate_m_step_stats_v2(
                smooth, pos, likes, layout, likelihood_threshold,
                perspective=perspective,
            )
            combined_stats += sess_stats

        # M-step finalization
        new_params = finalize_m_step_v2(
            combined_stats, layout, dt,
            prev_params=params, initial_params=initial_params,
        )

        max_rel = _max_param_change(new_params, params)
        sigmas = list(new_params.sigma_marker.values())
        q_segs = list(new_params.q_seg_ori.values())
        history.append({
            "iter": iteration,
            "max_rel_change": max_rel,
            "mean_sigma": float(np.mean(sigmas)),
            "q_root_pos": float(new_params.q_root_pos),
            "q_root_ori": float(new_params.q_root_ori),
            "mean_q_seg_ori": float(np.mean(q_segs)),
        })

        if verbose:
            print(
                f"[v2-em] iter {iteration}: max Δ/x = "
                f"{max_rel:.4e}, σ̄ = "
                f"{np.mean(sigmas):.3f}px, "
                f"q_root_pos = {new_params.q_root_pos:.1f}"
            )

        params = new_params
        if max_rel < tol:
            converged = True
            break

    return EMResultV2(
        params=params, history=history,
        initial_params=initial_params, converged=converged,
        perspective=perspective,
    )


# ============================================================
# Orchestrator + CLI + save/load — patch 105
# ============================================================
#
# This is the user-facing entry point. ``smooth_pose_v2`` and
# its CLI wrapper ``main`` glue patches 99-104 into a complete
# pipeline:
#
#   1. Discover input files
#   2. Load each session (DLC CSV / parquet, via the
#      diagnostic loader for full format compatibility)
#   3. Fit body lengths once across sessions
#   4. Compute initial NoiseParamsV2 from data
#   5. Run multi-session EM
#   6. Apply final smoother to each session, write smoothed
#      output (parquet by default; CSV fallback)
#   7. Save model artifact for re-use
#
# Output schema matches v1: per-marker [_x, _y, _p, _var_x,
# _var_y]. The Qt viewer (patches 95-98) auto-detects smoothed
# output via _var_x columns and works on either v1 or v2
# output without changes.


# Lazy imports — pulled in only when running the CLI / loader,
# avoids importing pandas/argparse just to use the math layer.
def _import_io_helpers():
    """Lazy import of pandas-dependent utilities. Returns a
    namespace dict.
    """
    import argparse
    import hashlib
    import os
    import sys as _sys
    from pathlib import Path
    import pandas as pd
    return {
        "argparse": argparse,
        "hashlib": hashlib,
        "os": os,
        "sys": _sys,
        "Path": Path,
        "pd": pd,
    }


def _arrays_to_df_v2(
    positions: np.ndarray,    # (T, K, 2)
    variances: np.ndarray,    # (T, K, 2) — diagonal of marker covariance
    likelihoods: np.ndarray,  # (T, K) — original raw likelihoods
    marker_names: List[str],
):
    """Convert smoothed (T, K, 2) marker positions + variances
    + raw likelihoods into a flat-column DataFrame.

    Schema matches v1's ``_arrays_to_df`` for downstream-tool
    and viewer compatibility:

      <marker>_x, <marker>_y, <marker>_p,
      <marker>_var_x, <marker>_var_y

    The Qt viewer (mufasa.tools.pose_viewer) auto-detects this
    schema as smoothed output.
    """
    io = _import_io_helpers()
    pd = io["pd"]
    T = positions.shape[0]
    cols: Dict[str, np.ndarray] = {}
    for i, m in enumerate(marker_names):
        cols[f"{m}_x"] = positions[:, i, 0]
        cols[f"{m}_y"] = positions[:, i, 1]
        cols[f"{m}_p"] = likelihoods[:, i]
        cols[f"{m}_var_x"] = variances[:, i, 0]
        cols[f"{m}_var_y"] = variances[:, i, 1]
    return pd.DataFrame(cols, index=np.arange(T))


def state_to_marker_variances(
    x_smooth: np.ndarray,   # (T, D)
    P_smooth: np.ndarray,   # (T, D, D)
    layout: BodyLayout,
    perspective: Optional["PerspectiveModelV2"] = None,
) -> np.ndarray:
    """Compute per-marker per-frame variance from the smoothed
    state covariance.

    For marker m at frame t with Jacobian H_t,m (2 × D):
      Cov(marker_m, t) = H_t,m @ P_smooth[t] @ H_t,m^T

    Returns the diagonal (var_x, var_y) per marker per frame.

    Cross-covariance (off-diagonal of the 2x2 marker
    covariance) is discarded — v1's output schema doesn't
    store it, and the viewer draws axis-aligned ellipses.

    Returns
    -------
    (T, K, 2) array
        variances[t, k, 0] = var_x for marker k at frame t
        variances[t, k, 1] = var_y for marker k at frame t
    """
    T = x_smooth.shape[0]
    K = layout.n_markers
    variances = np.zeros((T, K, 2))
    for t in range(T):
        fk = forward_kinematics(x_smooth[t], layout)
        H = state_to_marker_jacobian(
            x_smooth[t], layout, fk=fk, perspective=perspective,
        )
        # H shape: (2K, D)
        # For each marker k, rows 2k and 2k+1 are H_t,m
        for k in range(K):
            H_m = H[2 * k:2 * k + 2, :]
            cov_m = H_m @ P_smooth[t] @ H_m.T
            variances[t, k, 0] = cov_m[0, 0]
            variances[t, k, 1] = cov_m[1, 1]
    return variances


def smooth_session_v2(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: BodyLayout,
    marker_names: List[str],
    fitted_lengths: FittedLengths,
    params: NoiseParamsV2,
    fps: float,
    likelihood_threshold: float = 0.7,
    apply_constraints: bool = True,
    perspective: Optional["PerspectiveModelV2"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run forward filter + RTS smoother on one session, return
    smoothed marker positions and variances.

    Parameters
    ----------
    positions : (T, K, 2)
    likelihoods : (T, K)
    layout : BodyLayout
    marker_names : list[str]
    fitted_lengths : FittedLengths
    params : NoiseParamsV2
        Fitted noise params from EM.
    fps : float
    likelihood_threshold : float
    apply_constraints : bool
    perspective : PerspectiveModelV2, optional
        Fitted perspective model from EM.

    Returns
    -------
    smoothed_positions : (T, K, 2)
    smoothed_variances : (T, K, 2)
    """
    dt = 1.0 / fps
    initial_state = initial_state_from_data(
        positions, likelihoods, layout, marker_names,
        fitted_lengths, likelihood_threshold,
    )
    filt = forward_filter_v2(
        positions, likelihoods, layout, params, dt,
        initial_state=initial_state,
        likelihood_threshold=likelihood_threshold,
        apply_constraints=apply_constraints,
        perspective=perspective,
    )
    smooth = rts_smooth_v2(filt, layout, dt)

    # Compute marker positions from smoothed body state
    T = smooth.x_smooth.shape[0]
    K = layout.n_markers
    smoothed_positions = np.zeros((T, K, 2))
    for t in range(T):
        smoothed_positions[t] = state_to_marker_positions(
            smooth.x_smooth[t], layout,
            perspective=perspective,
        )
    smoothed_variances = state_to_marker_variances(
        smooth.x_smooth, smooth.P_smooth, layout,
        perspective=perspective,
    )
    return smoothed_positions, smoothed_variances


def save_model_v2(
    path,
    layout: BodyLayout,
    fitted_lengths: FittedLengths,
    params: NoiseParamsV2,
    fps: float,
    likelihood_threshold: float,
    perspective: Optional["PerspectiveModelV2"] = None,
) -> None:
    """Serialize a fitted v2 model to .npz.

    Stored fields include layout, fitted_lengths, params,
    scalars, and (since patch 109) optional perspective
    correction.

    Reload via ``load_model_v2``.
    """
    io = _import_io_helpers()
    Path = io["Path"]
    path = Path(path)

    # Serialize layout — npz can't store dataclasses directly,
    # so flatten to plain types.
    seg_data = []
    for seg in layout.segments:
        seg_data.append({
            "name": seg.name,
            "parent": seg.parent if seg.parent is not None else "__root__",
            "rest_angle": seg.rest_angle,
            "markers": dict(seg.markers),
        })

    save_kwargs = dict(
        version="v2",
        layout_segments=np.array(seg_data, dtype=object),
        fitted_segment_lengths=np.array(
            list(fitted_lengths.segment_lengths.items()),
            dtype=object,
        ),
        fitted_segment_length_iqr=np.array(
            list(fitted_lengths.segment_length_iqr.items()),
            dtype=object,
        ),
        fitted_marker_offsets=np.array(
            list(fitted_lengths.marker_offsets.items()),
            dtype=object,
        ),
        params_sigma_marker=np.array(
            list(params.sigma_marker.items()), dtype=object,
        ),
        params_q_root_pos=params.q_root_pos,
        params_q_root_ori=params.q_root_ori,
        params_q_seg_ori=np.array(
            list(params.q_seg_ori.items()), dtype=object,
        ),
        params_q_length=np.array(
            list(params.q_length.items()), dtype=object,
        ),
        params_constraint_sigma=params.constraint_sigma,
        fps=fps,
        likelihood_threshold=likelihood_threshold,
        has_perspective=(perspective is not None),
    )

    if perspective is not None:
        save_kwargs["persp_coeffs"] = np.array(
            list(perspective.coeffs.items()), dtype=object,
        )
        save_kwargs["persp_arena_x_mean"] = perspective.arena_x_mean
        save_kwargs["persp_arena_x_range"] = perspective.arena_x_range
        save_kwargs["persp_arena_y_mean"] = perspective.arena_y_mean
        save_kwargs["persp_arena_y_range"] = perspective.arena_y_range

    np.savez(path, **save_kwargs)


def load_model_v2(
    path,
) -> Tuple[
    BodyLayout, FittedLengths, NoiseParamsV2, float, float,
    Optional["PerspectiveModelV2"],
]:
    """Load a v2 model from .npz.

    Returns
    -------
    layout : BodyLayout
    fitted_lengths : FittedLengths
    params : NoiseParamsV2
    fps : float
    likelihood_threshold : float
    perspective : PerspectiveModelV2 or None
        Returned None if the saved model didn't include a
        perspective correction (pre-patch-109 models).
    """
    io = _import_io_helpers()
    Path = io["Path"]
    path = Path(path)

    data = np.load(path, allow_pickle=True)
    version = str(data["version"])
    if version != "v2":
        raise ValueError(
            f"Model version mismatch: expected 'v2', got "
            f"{version!r}. Did you mean to use the v1 loader?"
        )

    # Reconstruct BodyLayout
    seg_data = data["layout_segments"]
    segments = []
    for sd in seg_data:
        parent = sd["parent"]
        if parent == "__root__":
            parent = None
        segments.append(BodySegment(
            name=sd["name"],
            parent=parent,
            rest_angle=float(sd["rest_angle"]),
            markers=dict(sd["markers"]),
        ))
    layout = BodyLayout(segments=segments)

    # FittedLengths
    fitted_lengths = FittedLengths(
        segment_lengths={
            k: float(v) for k, v in data["fitted_segment_lengths"]
        },
        segment_length_iqr={
            k: float(v) for k, v in data["fitted_segment_length_iqr"]
        },
        marker_offsets={
            k: tuple(v) for k, v in data["fitted_marker_offsets"]
        },
    )

    # NoiseParamsV2
    params = NoiseParamsV2(
        sigma_marker={
            k: float(v) for k, v in data["params_sigma_marker"]
        },
        q_root_pos=float(data["params_q_root_pos"]),
        q_root_ori=float(data["params_q_root_ori"]),
        q_seg_ori={
            k: float(v) for k, v in data["params_q_seg_ori"]
        },
        q_length={
            k: float(v) for k, v in data["params_q_length"]
        },
        constraint_sigma=float(data["params_constraint_sigma"]),
    )

    fps = float(data["fps"])
    likelihood_threshold = float(data["likelihood_threshold"])

    # Perspective model (optional, only present in patch-109+
    # saved models)
    perspective: Optional["PerspectiveModelV2"] = None
    has_perspective = (
        "has_perspective" in data.files
        and bool(data["has_perspective"])
    )
    if has_perspective:
        coeffs_data = data["persp_coeffs"]
        coeffs = {
            k: np.asarray(v, dtype=float) for k, v in coeffs_data
        }
        perspective = PerspectiveModelV2(
            coeffs=coeffs,
            arena_x_mean=float(data["persp_arena_x_mean"]),
            arena_x_range=float(data["persp_arena_x_range"]),
            arena_y_mean=float(data["persp_arena_y_mean"]),
            arena_y_range=float(data["persp_arena_y_range"]),
        )

    return (
        layout, fitted_lengths, params, fps,
        likelihood_threshold, perspective,
    )


def smooth_pose_v2(
    pose_input,
    output_dir=None,
    layout: Optional[BodyLayout] = None,
    fps: float = 30.0,
    likelihood_threshold: float = 0.7,
    em_max_iter: int = 10,
    em_tol: float = 1e-3,
    save_model: Optional[str] = None,
    load_model: Optional[str] = None,
    apply_constraints: bool = True,
    enable_validation: bool = True,
    enable_warm_start_sigma: bool = True,
    enable_perspective: bool = True,
    verbose: bool = False,
) -> Dict:
    """Top-level user-facing function for v2 pose smoothing.

    Discovers input files, loads each as a session, fits body
    lengths + noise params via EM (or loads from a saved
    model), runs the final smoother on each session, writes
    smoothed output to disk.

    Output schema is identical to v1's: per-marker [_x, _y,
    _p, _var_x, _var_y] columns. The Qt viewer (mufasa.tools.
    pose_viewer) handles either smoothly.

    Parameters
    ----------
    pose_input : str or list[str] or Path
        File or directory of pose data. If directory, scanned
        for parquet (preferred) or CSV files.
    output_dir : str or Path, optional
        Output directory. Created if missing. If None, no
        output written (useful for in-process use).
    layout : BodyLayout, optional
        Custom body layout. Defaults to standard_rat_layout().
    fps : float
    likelihood_threshold : float
    em_max_iter : int
    em_tol : float
    save_model : str, optional
        If given, save fitted model to this path after EM.
    load_model : str, optional
        If given, load model and skip EM. Mutually exclusive
        with save_model? No — save_model just re-saves what
        was loaded; that's fine.
    apply_constraints : bool
        Pass through to forward filter.
    enable_validation : bool
        Run validation hook each EM iteration.
    verbose : bool

    Returns
    -------
    dict with:
      params: NoiseParamsV2
      fitted_lengths: FittedLengths
      layout: BodyLayout
      em_history: list of dicts
      sessions: list of {input_path, output_path, smoothed,
                          variances, n_frames}
      converged: bool
    """
    io = _import_io_helpers()
    Path = io["Path"]
    pd = io["pd"]
    os_ = io["os"]

    # ---------- Discover input files ----------
    if isinstance(pose_input, (str, type(Path()))):
        pose_input = [pose_input]
    paths: List = []
    for p in pose_input:
        path_obj = Path(p)
        if path_obj.is_dir():
            try:
                from mufasa.data_processors.kalman_diagnostic import (
                    discover_pose_files,
                )
                discovered = discover_pose_files(str(path_obj))
                paths.extend(Path(d) for d in discovered)
            except ImportError:
                # Fallback: glob for csv / parquet
                paths.extend(path_obj.glob("**/*.parquet"))
                if not paths:
                    paths.extend(path_obj.glob("**/*.csv"))
        else:
            paths.append(path_obj)

    if not paths:
        raise FileNotFoundError(
            f"No pose files found in input: {pose_input}"
        )

    if verbose:
        print(f"[smoother-v2] Discovered {len(paths)} file(s)")

    # ---------- Load each session ----------
    # Inline loader — direct read first, fall back to
    # diagnostic loader for DLC multi-row headers. Same
    # logic as mufasa.tools.pose_viewer._load_pose_file but
    # without the PySide6 dependency.
    raw_sessions: List[Dict] = []
    for path in paths:
        if verbose:
            print(f"[smoother-v2]   loading {path}")
        path = Path(path)
        suffix = path.suffix.lower()
        df = None
        # Try direct read first (works for smoothed-flat
        # parquets and pre-flattened CSVs)
        try:
            if suffix == ".parquet":
                df_direct = pd.read_parquet(path)
            elif suffix in ("", ".csv", ".tsv"):
                df_direct = pd.read_csv(path, low_memory=False)
            else:
                try:
                    df_direct = pd.read_parquet(path)
                except Exception:
                    df_direct = pd.read_csv(path, low_memory=False)
            df_direct.columns = [
                str(c).lower() for c in df_direct.columns
            ]
            # Check for marker columns
            markers_found = set()
            for col in df_direct.columns:
                if col.endswith("_x"):
                    base = col[:-2]
                    if (
                        f"{base}_y" in df_direct.columns
                        and not base.endswith("_var")
                    ):
                        markers_found.add(base)
            if markers_found:
                df = df_direct
        except Exception:
            df = None

        # Fall back to diagnostic loader (DLC multi-row header)
        if df is None:
            try:
                from mufasa.data_processors.kalman_diagnostic import (
                    load_pose_file as _diag_load_pose_file,
                )
                df, _ = _diag_load_pose_file(str(path))
            except (ImportError, ValueError):
                pass

        if df is None:
            raise RuntimeError(
                f"Could not load {path}. Tried direct read and "
                f"DLC multi-row header parsing."
            )

        # Detect markers
        markers = sorted({
            col[:-2] for col in df.columns
            if col.endswith("_x")
            and f"{col[:-2]}_y" in df.columns
            and not col[:-2].endswith("_var")
        })
        if not markers:
            raise RuntimeError(
                f"No marker columns detected in {path}"
            )
        T = len(df)
        K = len(markers)
        positions = np.full((T, K, 2), np.nan)
        likelihoods = np.zeros((T, K))
        for k, m in enumerate(markers):
            positions[:, k, 0] = pd.to_numeric(
                df[f"{m}_x"], errors="coerce",
            ).to_numpy()
            positions[:, k, 1] = pd.to_numeric(
                df[f"{m}_y"], errors="coerce",
            ).to_numpy()
            if f"{m}_p" in df.columns:
                likelihoods[:, k] = pd.to_numeric(
                    df[f"{m}_p"], errors="coerce",
                ).fillna(0.0).to_numpy()
            else:
                likelihoods[:, k] = 1.0
        raw_sessions.append({
            "path": Path(path),
            "markers": markers,
            "positions": positions,
            "likelihoods": likelihoods,
            "n_frames": T,
        })

    if verbose:
        total_frames = sum(s["n_frames"] for s in raw_sessions)
        n_markers_first = len(raw_sessions[0]["markers"])
        print(
            f"[smoother-v2] Total: {total_frames} frames × "
            f"{n_markers_first} markers from "
            f"{len(raw_sessions)} session(s)"
        )

    # Use the marker list from the first session as canonical.
    # Sessions with different markers will have NaN-padded
    # rows for missing markers — but for now, require all
    # sessions to have the same markers.
    first_markers = sorted(raw_sessions[0]["markers"])
    for s in raw_sessions[1:]:
        if sorted(s["markers"]) != first_markers:
            raise ValueError(
                f"Inconsistent marker sets across sessions. "
                f"First session ({raw_sessions[0]['path']}): "
                f"{first_markers}. "
                f"This session ({s['path']}): "
                f"{sorted(s['markers'])}"
            )
    marker_names_data = first_markers

    # ---------- Layout ----------
    if layout is None:
        layout = standard_rat_layout()

    # Verify layout marker names are subset of data marker names
    missing = set(layout.marker_names) - set(marker_names_data)
    if missing:
        if verbose:
            print(
                f"[smoother-v2] WARNING: layout markers not in "
                f"data: {sorted(missing)}. These markers will "
                f"have default offsets."
            )

    # Build session arrays in layout marker order — that's what
    # the EM/filter expect. We keep a mapping data_idx → layout_idx.
    layout_marker_names = layout.marker_names
    data_to_layout: Dict[str, int] = {
        m: layout_marker_names.index(m)
        for m in marker_names_data
        if m in layout_marker_names
    }

    sessions_arr: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in raw_sessions:
        K_layout = layout.n_markers
        T = s["n_frames"]
        pos = np.full((T, K_layout, 2), np.nan)
        likes = np.zeros((T, K_layout))
        for k_data, m in enumerate(s["markers"]):
            if m in data_to_layout:
                k_layout = data_to_layout[m]
                pos[:, k_layout, :] = s["positions"][:, k_data, :]
                likes[:, k_layout] = s["likelihoods"][:, k_data]
        sessions_arr.append((pos, likes))

    # ---------- Fit lengths + EM (or load) ----------
    if load_model is not None:
        if verbose:
            print(f"[smoother-v2] Loading model from {load_model}")
        (
            layout, fitted_lengths, params, _, _, perspective
        ) = load_model_v2(load_model)
        em_history: List[Dict] = []
        converged = True
    else:
        # Fit lengths from first session (could be improved to
        # aggregate across all)
        first_pos, first_likes = sessions_arr[0]
        fitted_lengths = fit_body_lengths(
            first_pos, first_likes, layout, layout_marker_names,
            likelihood_threshold,
        )
        if verbose:
            print(
                f"[smoother-v2] Fitted body lengths: "
                f"{ {k: f'{v:.2f}' for k, v in fitted_lengths.segment_lengths.items()} }"
            )

        if verbose:
            print(
                f"[smoother-v2] Fitting noise params via EM "
                f"(max_iter={em_max_iter}, tol={em_tol:.4f})..."
            )

        em_result = fit_noise_params_em_v2(
            sessions_arr, layout, layout_marker_names,
            fitted_lengths, fps,
            likelihood_threshold=likelihood_threshold,
            max_iter=em_max_iter, tol=em_tol,
            apply_constraints=apply_constraints,
            enable_validation=enable_validation,
            enable_warm_start_sigma=enable_warm_start_sigma,
            enable_perspective=enable_perspective,
            verbose=verbose,
        )
        params = em_result.params
        em_history = em_result.history
        converged = em_result.converged
        perspective = em_result.perspective

        if verbose:
            print(
                f"[smoother-v2] EM finished after "
                f"{len(em_history)} iterations "
                f"(converged={converged})"
            )

    # Save model if requested
    if save_model is not None:
        if verbose:
            print(f"[smoother-v2] Saving model to {save_model}")
        save_model_v2(
            save_model, layout, fitted_lengths, params,
            fps, likelihood_threshold,
            perspective=perspective,
        )

    # ---------- Final smoother pass per session ----------
    output_sessions: List[Dict] = []
    if verbose:
        print(
            f"[smoother-v2] Smoothing {len(sessions_arr)} "
            f"session(s)..."
        )

    if output_dir is not None:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    for s, (pos, likes) in zip(raw_sessions, sessions_arr):
        smooth_pos, smooth_var = smooth_session_v2(
            pos, likes, layout, layout_marker_names,
            fitted_lengths, params, fps,
            likelihood_threshold=likelihood_threshold,
            apply_constraints=apply_constraints,
            perspective=perspective,
        )

        # Build output DataFrame in DATA marker order (so it
        # matches the original input file's marker set, even
        # if layout has more)
        T = smooth_pos.shape[0]
        K_data = len(s["markers"])
        out_pos = np.full((T, K_data, 2), np.nan)
        out_var = np.full((T, K_data, 2), np.nan)
        out_likes = s["likelihoods"].copy()
        for k_data, m in enumerate(s["markers"]):
            if m in data_to_layout:
                k_layout = data_to_layout[m]
                out_pos[:, k_data, :] = smooth_pos[:, k_layout, :]
                out_var[:, k_data, :] = smooth_var[:, k_layout, :]

        df_out = _arrays_to_df_v2(
            out_pos, out_var, out_likes, list(s["markers"]),
        )

        out_path = None
        if output_dir is not None:
            stem = s["path"].stem
            # Strip common DLC suffix patterns
            if stem.endswith("DeepCut"):
                stem = stem[: -len("DeepCut")]
            out_name = f"{stem}_smoothed_v2.parquet"
            out_path = output_dir_path / out_name
            try:
                df_out.to_parquet(out_path, index=False)
            except (ImportError, Exception) as e:
                # Fallback to CSV
                out_path = output_dir_path / f"{stem}_smoothed_v2.csv"
                df_out.to_csv(out_path, index=False)
                if verbose:
                    print(
                        f"[smoother-v2]   {out_path.name} "
                        f"(parquet failed: {type(e).__name__}, "
                        f"used CSV)"
                    )
            else:
                if verbose:
                    print(f"[smoother-v2]   wrote {out_path.name}")

        output_sessions.append({
            "input_path": s["path"],
            "output_path": out_path,
            "smoothed": smooth_pos,
            "variances": smooth_var,
            "n_frames": T,
        })

    return {
        "params": params,
        "fitted_lengths": fitted_lengths,
        "layout": layout,
        "em_history": em_history,
        "sessions": output_sessions,
        "converged": converged,
    }


def main(argv=None) -> int:
    """CLI entry point. ``python -m mufasa.data_processors.
    kalman_pose_smoother_v2 ...``.
    """
    io = _import_io_helpers()
    argparse = io["argparse"]
    sys = io["sys"]

    parser = argparse.ArgumentParser(
        description=(
            "Joint-state Kalman pose smoother v2 with kinematic-"
            "tree spatial coupling. Stage 2 of the Mufasa Kalman "
            "smoother. v1 is in mufasa.data_processors."
            "kalman_pose_smoother."
        ),
    )
    parser.add_argument(
        "pose_input", nargs="+",
        help=(
            "Pose data input. Single file (CSV or parquet), "
            "directory (recursively scanned), or multiple "
            "file paths. Multiple files are smoothed as "
            "separate sessions with proper boundary handling."
        ),
    )
    parser.add_argument(
        "--output-dir", default="./kalman_smoother_v2_output",
        help=(
            "Output directory for smoothed parquets. Files "
            "are named <stem>_smoothed_v2.parquet."
        ),
    )
    parser.add_argument(
        "--likelihood-threshold", type=float, default=0.7,
        help="Likelihood threshold for high-confidence frames",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Frame rate of the input data (default 30)",
    )
    parser.add_argument(
        "--em-max-iter", type=int, default=10,
        help="Max EM iterations (default 10)",
    )
    parser.add_argument(
        "--em-tol", type=float, default=1e-3,
        help=(
            "EM convergence threshold on max relative param "
            "change (default 1e-3)"
        ),
    )
    parser.add_argument(
        "--save-model", default=None,
        help="Save fitted model artifact to this path",
    )
    parser.add_argument(
        "--load-model", default=None,
        help="Load model from this path; skip EM",
    )
    parser.add_argument(
        "--no-back4", action="store_true",
        help="Layout: omit back4 / back_rear segment",
    )
    parser.add_argument(
        "--no-tail", action="store_true",
        help="Layout: omit all tail segments",
    )
    parser.add_argument(
        "--no-lateral", action="store_true",
        help="Layout: omit lateral_left / lateral_right markers",
    )
    parser.add_argument(
        "--no-center", action="store_true",
        help="Layout: omit center marker",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Disable EM validation hook (not recommended)",
    )
    parser.add_argument(
        "--no-constraints", action="store_true",
        help=(
            "Disable unit-norm constraint observations "
            "(not recommended; can cause orientation drift)"
        ),
    )
    parser.add_argument(
        "--no-warm-start-sigma", action="store_true",
        help=(
            "Disable warm-start σ pass. By default, an extra "
            "filter+smoother pass runs before EM to inflate "
            "σ_marker for body markers with structural "
            "variation (lateral_left/right, etc.). Disable to "
            "use raw MA-residual σ only."
        ),
    )
    parser.add_argument(
        "--no-perspective", action="store_true",
        help=(
            "Disable per-marker bilinear perspective "
            "correction. By default, an extra fit pass after "
            "warm-start σ measures position-dependent scale "
            "factors (camera lens distortion, slight tilt) "
            "and applies them in the observation function. "
            "Disable to use rigid-offset prediction only."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args(argv)

    layout = standard_rat_layout(
        include_back4=not args.no_back4,
        include_tail=not args.no_tail,
        include_lateral=not args.no_lateral,
        include_center=not args.no_center,
    )

    try:
        smooth_pose_v2(
            pose_input=args.pose_input,
            output_dir=args.output_dir,
            layout=layout,
            fps=args.fps,
            likelihood_threshold=args.likelihood_threshold,
            em_max_iter=args.em_max_iter,
            em_tol=args.em_tol,
            save_model=args.save_model,
            load_model=args.load_model,
            apply_constraints=not args.no_constraints,
            enable_validation=not args.no_validate,
            enable_warm_start_sigma=not args.no_warm_start_sigma,
            enable_perspective=not args.no_perspective,
            verbose=args.verbose,
        )
    except Exception as e:
        print(
            f"[smoother-v2] ERROR: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1
    return 0


# ============================================================
# Perspective model — patch 109
# ============================================================
#
# Captures the position-dependent scaling of marker offsets
# caused by camera lens distortion, slight camera tilt, or
# non-perpendicular projection. Without this, the rigid-offset
# observation model has systematic residuals of 5-30 px for
# markers at the rat's widest body points (lateral_left,
# lateral_right) when the rat moves toward the arena edges.
#
# Model: per-marker bilinear scale function
#
#   scale_m(x, y) = 1 + a_m * x_n + b_m * y_n + c_m * x_n * y_n
#
# where (x_n, y_n) are the rat's root position normalized to
# the arena range (roughly [-1, 1]). At the arena center
# scale = 1 (rigid model is correct); at the edges scale
# can vary by ±0.3 or so. Scale is applied in the body frame:
#
#   v_marker_local_corrected = scale_m * (l_off * cos(a_off),
#                                          l_off * sin(a_off))
#
# This is then transformed by R_world[seg] and added to
# P_distal[seg] as before.
#
# Markers AT the segment distal end have l_off=0 so they're
# unaffected (scale times zero is zero). Markers far from
# the segment center get more correction.
#
# Parameters fit from warm-start smoother residuals via
# closed-form OLS per marker. 3 params per marker.


@dataclass
class PerspectiveModelV2:
    """Per-marker bilinear perspective correction.

    Fields
    ------
    coeffs : Dict[str, np.ndarray]
        Per-marker (a, b, c) coefficients for
        scale_m(x, y) = 1 + a x_n + b y_n + c x_n y_n.
    arena_x_mean, arena_x_range : float
        Used to normalize: x_n = (root_x - arena_x_mean) /
        (arena_x_range / 2). At arena center, x_n = 0.
    arena_y_mean, arena_y_range : float
    """
    coeffs: Dict[str, np.ndarray]
    arena_x_mean: float
    arena_x_range: float
    arena_y_mean: float
    arena_y_range: float

    @classmethod
    def identity(cls, layout: BodyLayout) -> "PerspectiveModelV2":
        """Identity perspective model (scale = 1 everywhere).
        Useful for testing / when perspective fitting is
        disabled.
        """
        return cls(
            coeffs={
                m: np.zeros(3) for m in layout.marker_names
            },
            arena_x_mean=0.0, arena_x_range=2.0,
            arena_y_mean=0.0, arena_y_range=2.0,
        )

    def _normalize(self, x: float, y: float) -> Tuple[float, float]:
        """Convert root_x, root_y to arena-normalized x_n, y_n
        in roughly [-1, 1].
        """
        x_n = (x - self.arena_x_mean) / max(
            self.arena_x_range / 2.0, 1.0,
        )
        y_n = (y - self.arena_y_mean) / max(
            self.arena_y_range / 2.0, 1.0,
        )
        return x_n, y_n

    def scale_for_position(
        self, root_x: float, root_y: float,
    ) -> np.ndarray:
        """Compute scale factor per marker at the given root
        position. Returns (n_markers,) array in the same
        order as the layout's marker_names attribute (use
        the layout that was used during fitting).

        Note: relies on ordering of self.coeffs being the
        same as layout.marker_names. coeffs is built via dict
        comprehension preserving insertion order, and we use
        layout.marker_names to insert.
        """
        x_n, y_n = self._normalize(root_x, root_y)
        scales = np.empty(len(self.coeffs))
        for i, m in enumerate(self.coeffs.keys()):
            a, b, c = self.coeffs[m]
            scales[i] = 1.0 + a * x_n + b * y_n + c * x_n * y_n
        return scales

    def scale_partials(
        self, root_x: float, root_y: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """∂scale/∂root_x and ∂scale/∂root_y per marker.
        Used in the observation Jacobian.

        scale_m = 1 + a x_n + b y_n + c x_n y_n
        ∂scale/∂x = (a + c y_n) / (arena_x_range / 2)
        ∂scale/∂y = (b + c x_n) / (arena_y_range / 2)
        """
        x_n, y_n = self._normalize(root_x, root_y)
        x_factor = 1.0 / max(self.arena_x_range / 2.0, 1.0)
        y_factor = 1.0 / max(self.arena_y_range / 2.0, 1.0)
        d_x = np.empty(len(self.coeffs))
        d_y = np.empty(len(self.coeffs))
        for i, m in enumerate(self.coeffs.keys()):
            a, b, c = self.coeffs[m]
            d_x[i] = (a + c * y_n) * x_factor
            d_y[i] = (b + c * x_n) * y_factor
        return d_x, d_y


def fit_perspective_model_v2(
    sessions: List[Tuple[np.ndarray, np.ndarray]],
    layout: BodyLayout,
    marker_names: List[str],
    fitted_lengths: FittedLengths,
    params: NoiseParamsV2,
    fps: float,
    likelihood_threshold: float = 0.5,
    apply_constraints: bool = True,
    min_offset_magnitude: float = 1.0,
    max_abs_coeff: float = 0.4,
) -> PerspectiveModelV2:
    """Fit a per-marker bilinear perspective correction.

    Workflow:
      1. Run a filter+smoother pass with current params (no
         perspective) on each session.
      2. For each frame, transform raw marker observations
         into the body frame using the smoothed body pose.
         Compute the ratio of observed-offset-magnitude to
         predicted-offset-magnitude per marker — this is the
         "scale" the rat needs at that frame for the rigid
         model to match.
      3. Bin scales by (root_x, root_y), fit
         scale_m(x_n, y_n) = 1 + a x_n + b y_n + c x_n y_n
         via OLS per marker.
      4. Clip coefficients to [-max_abs_coeff, max_abs_coeff]
         to prevent extreme corrections.

    Parameters
    ----------
    sessions : list of (positions, likelihoods)
    layout : BodyLayout
    marker_names : list[str]
    fitted_lengths : FittedLengths
    params : NoiseParamsV2
    fps : float
    likelihood_threshold : float
    apply_constraints : bool
    min_offset_magnitude : float
        Markers with predicted offset magnitude below this
        (in body frame, px) are skipped — their scale is
        ill-defined. Distal markers like back2 (which IS
        the segment endpoint) have zero offset and get
        identity coefficients.
    max_abs_coeff : float
        Hard cap on |a|, |b|, |c|. Default 0.4 means scale
        can range from 0.6 to 1.4 across the arena. Prevents
        runaway when fits are unreliable.

    Returns
    -------
    PerspectiveModelV2
    """
    dt = 1.0 / fps

    # First pass: collect arena bounds from root positions
    # (back2 marker, the root segment's distal point) over
    # all sessions.
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    root_distal = _segment_distal_marker(layout, layout.root_segment.name)
    if root_distal is None or root_distal not in name_to_idx:
        # Can't fit perspective without root marker; return identity
        return PerspectiveModelV2.identity(layout)

    root_idx = name_to_idx[root_distal]
    all_root_x: List[float] = []
    all_root_y: List[float] = []
    for pos, likes in sessions:
        m_x = pos[:, root_idx, 0]
        m_y = pos[:, root_idx, 1]
        m_p = likes[:, root_idx]
        mask = (
            (m_p >= likelihood_threshold)
            & np.isfinite(m_x) & np.isfinite(m_y)
        )
        all_root_x.extend(m_x[mask].tolist())
        all_root_y.extend(m_y[mask].tolist())
    if len(all_root_x) < 100:
        # Not enough data to fit perspective
        return PerspectiveModelV2.identity(layout)

    arena_x_mean = float(np.mean(all_root_x))
    arena_y_mean = float(np.mean(all_root_y))
    # Use 95th percentile spread to avoid outlier sensitivity
    x_lo, x_hi = float(np.percentile(all_root_x, 2.5)), float(np.percentile(all_root_x, 97.5))
    y_lo, y_hi = float(np.percentile(all_root_y, 2.5)), float(np.percentile(all_root_y, 97.5))
    arena_x_range = max(x_hi - x_lo, 10.0)
    arena_y_range = max(y_hi - y_lo, 10.0)

    # Per-marker accumulators for OLS fit
    # scale_m(x, y) - 1 = a*x_n + b*y_n + c*x_n*y_n
    # → for each marker, accumulate design matrix X^T X and X^T y
    # and solve at the end.
    XtX_per_marker: Dict[str, np.ndarray] = {
        m: np.zeros((3, 3)) for m in layout.marker_names
    }
    Xty_per_marker: Dict[str, np.ndarray] = {
        m: np.zeros(3) for m in layout.marker_names
    }
    n_obs_per_marker: Dict[str, int] = {
        m: 0 for m in layout.marker_names
    }

    # Pre-compute layout marker → data marker mapping
    layout_to_data: Dict[str, int] = {}
    for m in layout.marker_names:
        if m in name_to_idx:
            layout_to_data[m] = name_to_idx[m]

    for pos, likes in sessions:
        # Run smoother
        x0 = initial_state_from_data(
            pos, likes, layout, marker_names,
            fitted_lengths, likelihood_threshold,
        )
        filt = forward_filter_v2(
            pos, likes, layout, params, dt,
            initial_state=x0,
            likelihood_threshold=likelihood_threshold,
            apply_constraints=apply_constraints,
        )
        smooth = rts_smooth_v2(filt, layout, dt)
        T = smooth.x_smooth.shape[0]
        idx_pack = _pack_state_layout_indices(layout)

        for t in range(T):
            state_t = smooth.x_smooth[t]
            if not np.all(np.isfinite(state_t)):
                continue
            # Body pose at frame t
            fk = forward_kinematics(state_t, layout)
            root_x = state_t[idx_pack["__root__"]["x"]]
            root_y = state_t[idx_pack["__root__"]["y"]]
            x_n = (root_x - arena_x_mean) / max(arena_x_range / 2.0, 1.0)
            y_n = (root_y - arena_y_mean) / max(arena_y_range / 2.0, 1.0)

            for m, k_data in layout_to_data.items():
                seg_name, (l_off, a_off) = layout.marker_attachment(m)
                if l_off < min_offset_magnitude:
                    # Distal marker — no offset to scale
                    continue

                # Predicted offset in body frame (segment frame)
                offset_pred_local = np.array([
                    l_off * np.cos(a_off),
                    l_off * np.sin(a_off),
                ])

                # Observed offset in body frame: take raw obs,
                # subtract distal point, rotate into segment
                # frame.
                x_obs = pos[t, k_data, 0]
                y_obs = pos[t, k_data, 1]
                p = likes[t, k_data]
                if (
                    not (np.isfinite(x_obs) and np.isfinite(y_obs))
                    or p < likelihood_threshold
                ):
                    continue
                P_distal = fk.P_distal[seg_name]
                R_seg = fk.R_world[seg_name]
                obs_world = np.array([x_obs, y_obs])
                obs_local = R_seg.T @ (obs_world - P_distal)

                # Scale = projection of observed offset onto
                # predicted offset direction divided by
                # predicted offset magnitude
                pred_norm_sq = float(offset_pred_local @ offset_pred_local)
                if pred_norm_sq < 1e-9:
                    continue
                scale_t = float(
                    offset_pred_local @ obs_local
                ) / pred_norm_sq

                # OLS: y = scale_t - 1, x = [x_n, y_n, x_n*y_n]
                y_obs_ols = scale_t - 1.0
                x_design = np.array([x_n, y_n, x_n * y_n])
                XtX_per_marker[m] += np.outer(x_design, x_design)
                Xty_per_marker[m] += y_obs_ols * x_design
                n_obs_per_marker[m] += 1

    # Solve OLS per marker
    coeffs: Dict[str, np.ndarray] = {}
    for m in layout.marker_names:
        if n_obs_per_marker[m] < 50:
            # Insufficient observations
            coeffs[m] = np.zeros(3)
            continue
        XtX = XtX_per_marker[m]
        Xty = Xty_per_marker[m]
        # Add small regularization for numerical stability
        XtX_reg = XtX + 1e-3 * np.eye(3)
        try:
            coeff = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            coeff = np.zeros(3)
        # Clip
        coeff = np.clip(coeff, -max_abs_coeff, max_abs_coeff)
        coeffs[m] = coeff

    return PerspectiveModelV2(
        coeffs=coeffs,
        arena_x_mean=arena_x_mean,
        arena_x_range=arena_x_range,
        arena_y_mean=arena_y_mean,
        arena_y_range=arena_y_range,
    )


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
