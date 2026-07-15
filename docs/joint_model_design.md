# Design: joint model for the v2 pose smoother

Status: proposal. Supersedes nothing; extends the layout work in patches
122gv (marker names from the project) and 122gx (kinematic tree derived
from `[skeleton]`).

## 1. Motivation

Since 122gx, `layout_from_config` derives the kinematic tree from the
project's `[skeleton]` graph, so any rig works with no configuration. The
derivation costs modelling power, and the cost is measurable on the
Cacna1Train rig (15 markers, 24 skeleton edges):

| layout source | segments | non-root | `state_dim` |
| --- | --- | --- | --- |
| derived spanning tree (122gx default) | 15 | 14 | **92** |
| `[[pose.segments]]` with rigid clusters | 7 | 6 | **44** |
| built-in `standard_rat_layout()` | 7 | 6 | 44 |

The derived tree gives every marker its own orientation and length DOF
because a graph cannot say which markers move rigidly together. Nothing in
`back_T4 — back_T8 — back_L2` distinguishes a stiff spine from three
hinges. The built-in rig knew, because a human wrote it down.

So the question this spec answers: **how should a project declare movement
class?** Two shapes were proposed — a `[pose.hierarchy]` table connecting
body parts, and a `[[pose.joints]]` table with member segments and a joint
type.

## 2. What the layout already encodes

Before adding anything, three facts about `BodyLayout` (see
`kalman_pose_smoother_v2.py`):

1. **Hierarchy is already inline.** `BodySegment.parent` names the parent
   segment. In a tree this is exactly 1:1 with non-root segments.
2. **Rigidity is already expressible.** A segment's `markers` dict maps
   marker to a `(length, angle)` offset in the segment's distal frame.
   Markers sharing a segment are rigidly attached and carry **no DOF** —
   this is precisely how the built-in rig models the trunk (`body` holds
   `back1`, `back2`, `back3`, `lateral_*`, `center`).
3. **Per-segment behaviour is already parametric.** `NoiseParamsV2` carries
   `q_seg_ori: dict[str, float]` and `q_length: dict[str, float]` — process
   noise keyed by segment name.

State packing is uniform and positional:

```
state_dim = 8 + 6*N          (N = non-root segments)
  root pose            8 dims
  per non-root segment 4 dims orientation (cos, sin, cos_dot, sin_dot)
                     + 2 dims length      (length, length_dot)
```

with slices computed as `8 + 4*N + 2*idx`. Optional blocks (`with_drift`,
`orientation_drift_segments`, `const_accel_segments`) are **appended at the
end** so the base packing is never disturbed.

## 3. Decision

**Add an optional per-segment `joint` field. Do not add a hierarchy table.
Do not add a joints table.**

### 3.1 Rejected: `[pose.hierarchy]`

`parent` already is the hierarchy, 1:1 with segments. A second table adds no
expressive power and introduces a sync hazard: two places that can disagree
about the same edge, with no single source of truth. It would earn its place
only for a reusable segment *library* — segments defined once, parented
differently per rig — which is not a current requirement.

### 3.2 Rejected: `[[pose.joints]]` with member segments

In a tree, a joint **is** the edge from a segment to its parent: exactly one
per non-root segment, and its members are always `(this segment, its
parent)`. A joints table would normalise a relation that is already 1:1 and
force every segment to be named twice.

First-class joints earn their place only when joints can outnumber
`segments - 1` — i.e. **kinematic loops**. See §8.

### 3.3 Rejected: variable state dims per joint type

The natural reading of "joint type restricts the movement class" is fewer
dims for more restrictive joints. That breaks the uniform `8 + 6*N` packing:
every slice helper, the forward kinematics, and the observation Jacobian
index by `idx * const`, and would need cumulative per-segment offsets. The
codebase's own extensions deliberately append at the end to avoid exactly
this. Adding dims is cheap; removing them is invasive.

The same restriction is available **without touching packing** (§5.2).

## 4. Joint classes

| `joint` | Meaning | Mechanism | Dims |
| --- | --- | --- | --- |
| `rigid` | Welded to parent; no relative motion | merge markers into the parent segment | **removes 6** |
| `revolute` | Hinge; constant bone length | pin `q_length[seg] = 0` | 6 (2 frozen) |
| `revolute_prismatic` | Hinge + slow-varying length (**default; today's behaviour**) | unchanged | 6 |

`rigid` is deliberately *not* a new mechanism — it resolves to segment
membership, which already means zero DOF. Declaring
`joint = "rigid"` on a segment merges its markers into the parent's
`markers` dict with fitted offsets. This is the lever that moves 92 → 44.

`revolute` is the genuinely missing class: a bone whose length is constant.
Pinning `q_length = 0` makes the length a **random constant** — estimated
from data once, then not permitted to wander. That is the correct semantics
for a rigid limb of unknown size.

## 5. Implementation

### 5.1 Structural (`rigid`) — reduces dims

Resolved in `layout_from_segments` / `segments_from_skeleton` before
`BodyLayout` is constructed. A segment marked `rigid` contributes its
markers to its parent with placeholder offsets, which `fit_body_lengths`
replaces with medians from data. No change to state packing, FK, or the
Jacobian: the segment simply ceases to exist.

### 5.2 Parametric (`revolute`) — reduces freedom, not dims

`BodyLayout` grows a `pinned_q: dict[str, set[str]]` (segment → which of
`{"ori", "length"}` are frozen). `NoiseParamsV2.default` sets pinned entries
to `0.0`.

**The M-step must skip pinned parameters**, or EM will re-inflate them on
the first iteration. The refit sites are:

* `fit.q_seg_ori[seg_name] = _q_from_4block(...)` and
  `fit.q_length[seg_name] = _q_from_2block(...)`
* the two aggregation variants (`pooled`, `median`) that write
  `q_seg_ori[seg_name]` / `q_length[seg_name]` via `_damp(...)`

Each becomes conditional on the segment not being pinned for that block.

### 5.3 Honest cost of pinning

`q = 0` does not delete a dimension — the state stays in the vector and
still costs compute in `F`, `Q`, and the covariance update. It becomes a
constant to be estimated rather than a random walk. Pinning is a *modelling*
restriction, not a performance win. Only `rigid` (§5.1) reduces `state_dim`.

## 6. TOML schema

```toml
# Rigid trunk: one segment, six markers, zero relative DOF.
[[pose.segments]]
name    = "body"
markers = ["back_T8", "back_T4", "back_L2", "back_L6", "hip_left", "hip_right"]

[[pose.segments]]
name    = "back_rear"
parent  = "body"
markers = ["back_V2"]
joint   = "revolute"          # hinge, constant length

[[pose.segments]]
name    = "neck"
parent  = "body"
markers = ["head_back"]
joint   = "revolute"

[[pose.segments]]
name    = "head"
parent  = "neck"
markers = ["head_mid", "head_nose", "head_left", "head_right"]  # rigid cluster
joint   = "revolute"

[[pose.segments]]
name    = "tail_1"
parent  = "back_rear"
markers = ["tail_V6"]
# joint omitted -> revolute_prismatic (default, current behaviour)

[[pose.segments]]
name    = "tail_2"
parent  = "tail_1"
markers = ["tail_V18"]

[[pose.segments]]
name    = "tail_3"
parent  = "tail_2"
markers = ["tail_V32"]
```

That is the Cacna1Train rig in full: 15 markers, 7 segments, 6 non-root,
`state_dim = 44` — the built-in rig's number, reached from config alone.

`joint` is optional and defaults to `revolute_prismatic`, so every existing
`[[pose.segments]]` spec keeps its current meaning.

Derived layouts (§`[skeleton]` path) may opt in globally:

```toml
[pose.kinematics]
root  = "back_T8"
joint = "revolute"            # default class for every derived segment
rigid = [["back_T8", "back_T4", "back_L2"]]   # optional rigid groups
```

## 7. Backward compatibility

* `joint` absent → `revolute_prismatic` → byte-identical behaviour to today.
* The built-in rig and the 122gv `[pose.layout]` role map are untouched.
* `pinned_q` defaults to empty, so `NoiseParamsV2.default` and the M-step
  behave exactly as now.
* Reciprocal tripwire: `state_dim` assertions in existing smokes must be
  re-checked, since `rigid` changes segment counts.

## 8. Open question: loop closure

The Cacna1Train skeleton has 24 edges over 15 nodes; a spanning tree uses
14, so **10 edges are discarded**. Those are real anatomical constraints
(`hip_left — hip_right`, `head_mid — tail_V6`, the head triangle) that a
tree cannot express.

Using them would require first-class joints — the one case that justifies
§3.2's rejected table — plus a constrained EKF (projection onto the
constraint manifold, or pseudo-measurements enforcing constant distance).
That is a research-grade change, not a config change, and is explicitly out
of scope here. Worth noting that rigid clusters (§5.1) already absorb most
of the discarded edges: every edge *within* a rigid cluster becomes a fitted
offset rather than a dropped constraint.

## 9. Open question: joint limits

`joint = "revolute"` says nothing about range. Anatomical limits (a neck
does not bend 180°) would need `limits = [min, max]` and constraint
projection in the update step. The machinery exists in spirit —
`apply_constraints`, and the unit-circle projection from patch 106 — but
angle-range projection on a `(cos, sin)` parameterisation needs its own
design. Deferred.

## 10. Implementation plan

1. **Schema + resolution.** `joint` field parsed in `layout_from_segments`;
   `rigid` merges into parent. `BodyLayout.pinned_q` populated. Smoke:
   rigid cluster reduces `state_dim`; default unchanged.
2. **Noise pinning.** `NoiseParamsV2.default` honours `pinned_q`; both
   M-step aggregation paths skip pinned blocks. Smoke: pinned `q` stays 0
   across EM iterations.
3. **Derived-layout opt-in.** `[pose.kinematics].joint` / `.rigid` for the
   `[skeleton]` path.
4. **UI (optional).** Segment/joint editor in the Model modifications tab.

Phases 1 and 2 are independent and separately shippable; phase 1 alone
delivers the 92 → 44 reduction.
