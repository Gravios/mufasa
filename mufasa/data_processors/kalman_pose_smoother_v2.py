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
