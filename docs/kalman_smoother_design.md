# Design: Triplet-egocentric covariance Kalman smoother for pose data

**Status**: design document, no implementation yet.
**Scope**: a kinematic-model-aware smoother that uses
empirically-fit egocentric angular covariance from
marker triplets to inform a Kalman filter / RTS smoother.
**Author**: collaborative design between Gravio and the patch
assistant, May 2026.
**Companion to**: `docs/workflows.md` (Smooth pose section,
which documents the limitations of the current SavGol/Gaussian
smoothers).

---

## 1. Motivation

### What's wrong with current smoothers

Mufasa's existing pose smoothers (`Smoothing.savgol_smoother`,
`Smoothing.df_smoother`, plus the orphaned `AdvancedSmoother`)
all share three structural limitations:

1. **Likelihood-blind.** Every frame contributes equally to the
   smoothed output regardless of detector confidence. A
   p=0.99 frame and a p=0.10 frame are pulled into the same
   moving average. The simple Smoother actually smooths the
   `_p` values themselves (already documented as a bug); the
   AdvancedSmoother accidentally avoids that by per-bodypart
   slicing.

2. **Per-bodypart independent.** Each marker's trajectory is
   smoothed in isolation. No information transfer between
   markers that are anatomically related. If the nose detector
   fails for 30 frames but the left ear and right ear are
   confidently detected, the nose's smoothed position is
   inferred only from its own (failed) trajectory, not from
   the geometric relationship to the ears.

3. **Time-stationary.** The smoothing window is fixed; there's
   no mechanism to "soften" the prior during high-confidence
   segments and "tighten" it during dropouts.

### What "empirically-fit triplet-egocentric Kalman smoother" means

Three components, each addressing one of the limitations above:

1. **Kalman filter / RTS smoother** with per-frame measurement
   variance derived from the likelihood — directly addresses
   limitation (1). High-likelihood frames anchor the trajectory;
   low-likelihood frames defer to the dynamics prior.

2. **Triplet structure** — pick three anatomically-rigid
   markers (e.g., nose, left ear, right ear), express their
   relationships in an egocentric frame, fit an empirical
   covariance over the angular configuration, and use that as
   a spatial prior. This addresses limitation (2): when one
   marker fails, the filter has a structured prior from the
   triplet to fall back on.

3. **Empirical fit from high-likelihood segments** —
   automatically extract the segments where all three triplet
   markers have p > τ simultaneously, fit the covariance from
   those, and use it as the prior. No magic numbers, no
   per-video tuning. The model self-calibrates to whatever
   anatomy/posture range your data contains.

The result is a smoother whose behavior adapts to the data:
on a video with a good DLC model and few dropouts, it acts
similarly to existing smoothers (because the likelihood-derived
variance is uniformly low and the spatial prior is rarely
needed). On a video with frequent dropouts or detector
failures, it draws on the triplet anatomical prior to
reconstruct trajectories that the per-bodypart smoothers
couldn't.

---

## 2. Prior art (what we're NOT inventing)

This idea sits at the intersection of three published lines of
work. The design is novel as an *integration* but each
component has prior art.

### 2.1 Kalman smoothing with likelihood-derived variance

Standard Kalman filtering with per-frame measurement variance
is textbook (Anderson & Moore, *Optimal Filtering*, 1979). For
pose data specifically, this approach is implemented in:

- **DeepLabCut's `filterpredictions`** has a Kalman filter
  option that uses likelihood as inverse-variance, but
  **per-bodypart only** — no spatial structure.
- **KeySORT** (Vandenhof et al. 2025, arXiv:2503.10450) — uses
  an adaptive Kalman filter for cattle pose with skeleton
  structure for tracklet construction. Different application
  (multi-animal tracking) but similar variance-aware filtering
  conceptually.

### 2.2 Anatomical priors on pose

Limb-length priors and rigid-body assumptions are widely used:

- **Anipose** (Karashchuk et al. 2021, *Cell Reports*) — uses a
  Viterbi filter analogous to motion-based pictorial
  structures. Their docs explicitly mention this could be
  "extended to handle priors on limb lengths." Anipose's
  triangulation uses `scale_smooth` (smoothness) and
  `scale_length` (limb-length consistency) as separate weights
  in the optimization. Not a Kalman smoother, but uses the
  same family of structural priors.

- **EKF for rigid body pose** (multiple biomechanics papers,
  e.g., the foundational Söderkvist & Wedin 1993) — the
  classic approach for marker-based motion capture. Works in
  3D with explicitly rigid bodies. Our case is 2D
  pseudo-rigid (animal anatomy isn't strictly rigid; head can
  turn relative to body, etc.).

- **Pictorial structures** (Yang et al. 2016, Amin et al. 2013,
  Günel et al. 2019) — older but related: encode anatomical
  constraints as priors over keypoint configurations.

### 2.3 Closest direct precedent — Lightning Pose's EKS

The single most relevant work is **Lightning Pose's Ensemble
Kalman Smoother** (Biderman et al. 2024, *Nature Methods*).
It does almost exactly what we're proposing, with these
differences:

- **EKS uses ensemble variance, not likelihood**, as the
  measurement variance source. They train multiple DLC
  networks with different random seeds, and the variance of
  their predictions across the ensemble is the per-frame R_t.
  This requires training N models. Our approach uses the
  single-model likelihood, which is what most users actually
  have.

- **EKS uses Pose PCA loss for spatial constraint**, not
  triplet covariance. They project pose into a low-dimensional
  PCA subspace fit from training labels. Frames whose poses lie
  far from the PCA subspace are penalized. This is a *holistic*
  constraint over all keypoints simultaneously.

- **Triplet structure is more local** than PCA, which has
  pros and cons:
  - Pro: doesn't require labeled training data, only requires
    high-likelihood inference output. So you can run on any
    project regardless of label set.
  - Pro: more interpretable — "this triplet of markers is
    rigid" maps to anatomical reasoning.
  - Pro: failure of one triplet doesn't contaminate the others.
  - Con: doesn't capture larger-scale postural correlations
    (e.g., "rearing pose" affects multiple triplets).
  - Con: requires a choice of which triplets to use.

The Lightning Pose EKS code is open source
(github.com/paninski-lab/lightning-pose, file
`eks/singleview_smoother.py`). We could read it for
implementation reference and structure our Kalman/RTS
implementation similarly. But the spatial-prior component is
a different design choice — PCA vs triplet.

### 2.4 Bottom line on prior art

Our proposed approach is **novel as a specific design**
(triplet egocentric covariance + likelihood-derived variance
in a single Kalman smoother), but every component has
established mathematical foundation. Risk is in
implementation correctness and engineering, not in the
fundamental approach.

---

## 3. Mathematical formulation

This section defines the model precisely. Read it as the
specification of what the implementation has to do, not as a
final commitment to every detail.

### 3.1 Notation

- N markers per animal (e.g., N=8 for the 8bp scheme)
- T frames in the video
- For frame t and marker i:
  - `(x_t^i, y_t^i)` — measured position
  - `p_t^i` ∈ [0, 1] — likelihood
- Triplets: a set 𝒯 of 3-tuples (i, j, k) of marker indices.
  K = |𝒯| triplets.

### 3.2 State space (per-marker dynamics)

For each marker i, define the per-marker state
```
s_t^i = [x_t^i, y_t^i, vx_t^i, vy_t^i]ᵀ ∈ ℝ⁴
```
with constant-velocity dynamics
```
s_{t+1}^i = F · s_t^i + w_t^i,    w_t^i ~ N(0, Q^i)
```
where
```
F = [[1, 0, Δt, 0],
     [0, 1,  0, Δt],
     [0, 0,  1,  0],
     [0, 0,  0,  1]]
```
and Δt = 1/fps. Q^i is the per-marker process noise:
```
Q^i = diag(q_pos², q_pos², q_vel², q_vel²)
```
fit empirically (see §3.5).

This is identical per-bodypart to the classical Kalman pose
smoother. The triplet structure enters through the measurement
model.

### 3.3 Measurement model — temporal part

The position observation per marker:
```
z_t^i = [x_t^i, y_t^i]ᵀ = H · s_t^i + v_t^i,
v_t^i ~ N(0, R_t^i)
```
where H = [[1,0,0,0],[0,1,0,0]] and the measurement variance is
likelihood-derived:
```
R_t^i = (σ_base^i)² / max(p_t^i, p_floor) · I_2
```
- σ_base^i is the baseline measurement noise for marker i,
  empirically fit (see §3.5)
- p_floor is a small constant (e.g., 0.01) preventing division
  by zero
- For frames where p_t^i = 0 (true dropout): treat as missing
  observation (skip the update step for marker i at frame t,
  but still propagate the dynamics)

This is the standard part. A bare Kalman smoother with this
measurement model would already improve on the existing
smoothers for limitation (1).

### 3.4 Spatial prior — triplet egocentric covariance

This is the structurally novel part. **Updated 2026-05-01**:
the design has converged on the full feature vector (option d
in the original enumeration below), and the covariance is
velocity-conditional rather than static. See §3.4.5 for the
velocity-conditional treatment.

**Egocentric frame definition**: for triplet τ = (i, j, k), at
frame t, define the egocentric coordinate frame:
- Origin: centroid of the three markers
  `c_t^τ = ((x_t^i + x_t^j + x_t^k)/3, (y_t^i + y_t^j + y_t^k)/3)`
- Orientation: along the principal axis of the triangle (PCA
  on the centered marker positions; this is the most stable
  axis choice across postures and avoids the
  i→j-axis-degenerates-when-i=j edge case)

**Egocentric configuration vector — the committed choice**:

The full feature vector per triplet per frame combines four
quantities, giving the model the maximum information to fit
against. The dimensionality is larger than any single
sub-option, but rank-deficient (constraints among the entries),
which the empirical covariance handles naturally via small
eigenvalues:

```
e_t^τ = [
    α_1, α_2,                    # 2 internal angles (3rd is determined)
    d_12, d_13, d_23,            # 3 pairwise distances
    x_1, y_1, x_2, y_2, x_3, y_3, # 6 egocentric coordinates
    vx_h, vy_h,                  # 2 head-frame egocentric velocities
    vx_b, vy_b                   # 2 body-frame egocentric velocities
] ∈ ℝ¹⁵
```

The egocentric (x,y) of each marker plus the angles plus the
distances are partly redundant — knowing the egocentric
positions determines the distances and angles. Including all
of them is technically redundant, but:

1. **The redundancy doesn't hurt** — the covariance estimator
   will see the constraints as small eigenvalues. The
   Mahalanobis prior properly handles rank-deficient covariance
   via pseudo-inverse (Moore-Penrose) or regularization.

2. **The redundancy helps numerical conditioning** —
   different feature subsets are sensitive to different
   sources of noise. Angles are scale-invariant; distances
   are translation-invariant; egocentric coords are not.
   Combining gives the filter multiple ways to constrain the
   same physical relationship.

3. **The velocity component is genuinely additional** — it's
   not redundant with the static configuration. It's what
   enables the velocity-conditional covariance described in
   §3.4.5.

**Head-frame egocentric velocity**: `(vx_h, vy_h)` is the
animal's velocity expressed in the head's frame of reference,
not the world frame. Computation:

1. From frame t-1 to t, compute the head's world-frame
   position from the head triplet (typically nose + leftEar +
   rightEar centroid, or the triplet itself if it includes
   head markers).
2. Compute the head's heading angle `θ_t` from the same triplet
   (e.g., direction from ear-midpoint to nose).
3. World-frame velocity: `v_world = (head_t - head_{t-1}) / Δt`
4. Rotate into head frame: `(vx_h, vy_h) = R(-θ_t) · v_world`

This means `vx_h > 0` corresponds to "moving in the direction
the head is pointing" (forward locomotion) regardless of the
world-frame orientation. `vy_h` is lateral motion in the
animal's own frame.

**Body-frame egocentric velocity**: `(vx_b, vy_b)` is the
animal's body-centroid velocity expressed in the body's frame
of reference. Body frame definition (committed: option (b),
PCA major axis):

1. Body markers: all markers minus head markers (configurable
   per project; default = automatic).
2. Body centroid `c_b^t`: mean of body marker positions at
   frame t.
3. Body major axis `â_b^t`: leading eigenvector of the 2×2
   covariance matrix of centered body marker positions at
   frame t (PCA per frame).
4. Sign disambiguation: PCA returns an unsigned axis (v and
   -v are both eigenvectors). The sign is chosen so that
   `dot(â_b^t, head_direction_t) > 0` when head direction is
   reliable, falling back to continuity with the previous
   frame's body axis when head direction is unreliable.
5. Body heading angle: `θ_b^t = atan2(â_b^t,y, â_b^t,x)`
6. World-frame body velocity: `v_b_world = (c_b^t - c_b^{t-1}) / Δt`
7. Rotate into body frame: `(vx_b, vy_b) = R(-θ_b^t) · v_b_world`

This means `vx_b > 0` corresponds to "moving in the direction
the body is oriented" (forward body locomotion). For typical
rodent postures, the body major axis is approximately the
spine direction, so vx_b roughly tracks "rate of forward
spine progress."

The reason for using BOTH velocities (rather than head only):
head and body don't always move together. The clearest case
is **scanning**: animal stationary, head sweeping side to
side. Head velocity is high; body velocity is zero. With
head-only conditioning, ALL triplets would condition on a
"high-velocity" regime, but the body triplet's anatomical
configuration is in the "stationary" regime. Including body
velocity in the conditioning vector lets the empirical
fitting separate these regimes.

For triplets that don't include head markers (most body
triplets), the head-frame velocity is still computed from
the head triplet (special role at config time) and the
body-frame velocity is computed from the body markers
(special role too). Both velocities are global per-frame
quantities used as the conditioning vector for ALL triplets.
Every triplet's configuration is conditioned on the *animal's*
overall motion state — head and body — not on the triplet's
own local velocity. This preserves the "behavioral state"
intuition: when the animal is locomoting forward, ALL body
parts have correlated configurations; when the animal is
scanning, head triplets and body triplets have very different
anatomical regimes.

**The empirical sample set**:
```
S_high^τ = { t : p_t^i > τ_high
              AND p_t^j > τ_high
              AND p_t^k > τ_high
              AND head triplet is high-p at t and t-1
              AND body markers are high-p at t and t-1 }
```
The extra "head triplet AND body markers high-p at t and t-1"
requirement is necessary because both velocity components
need reliable markers in two consecutive frames. Frames
where either head or body was poorly tracked are excluded
from fitting.

`τ_high` is a threshold (default 0.95). |S_high^τ| should be
several thousand frames minimum for stable estimation in the
15-dim feature space; fewer than ~500 → fall back to lower-dim
feature vector (drop body velocity → 13-dim → drop head
velocity → 11-dim → drop coords → 5-dim) and warn.

### 3.4.5 Velocity-conditional covariance — manifold approach

**The committed approach**: instead of fitting a single static
covariance Σ^τ from S_high^τ, fit a **velocity-conditional**
covariance Σ^τ(v) where v = (vx_h, vy_h) is the head-frame
velocity at the query frame. Different velocity regimes have
different anatomical configurations:
- Stationary (|v_h| ≈ 0, |v_b| ≈ 0): head can turn freely,
  wide spatial covariance over nose-ear angles
- Forward locomotion (vx_h > 0, vx_b > 0, both aligned): head
  locked toward motion direction, narrow covariance
- Scanning (|v_h| > 0, |v_b| ≈ 0): head sweeping, body still.
  Head triplets in "scanning" regime; body triplets in
  "stationary" regime. ONLY 4D conditioning captures this.
- Lateral motion (vy_h ≠ 0): different distribution again
- Reverse motion (vx_h < 0): rare in rats, distinct
  configuration if it occurs

A static Σ^τ averages over all regimes and fits none well.
2D conditioning (head only) misses the scanning regime and
treats body triplets as if they shared the head's motion
state, which they don't. 4D conditioning (head + body)
captures all four regimes natively. Velocity-conditional
Σ^τ(v) with v = (vx_h, vy_h, vx_b, vy_b) lets the smoother
specialize per regime.

**Method (committed: KD-tree + local kernel covariance, 4D)**:

Pre-fit phase, once per project:
1. Compute the full feature vector e_t^τ ∈ ℝ¹⁵ for every
   triplet τ and every frame t ∈ S_high^τ.
2. Extract the 4D velocity component
   v_t = (vx_h, vy_h, vx_b, vy_b) from each.
3. Build a KD-tree indexed on the 4D velocity space over
   S_high^τ.
4. Persist the (KD-tree, features, velocities) tuple to disk
   as the saved model artifact (see §3.4.6 for storage spec).

Inference phase, per query frame t:
1. Estimate or read the 4D velocity v_t at the query frame.
   (State-dependent; see §3.4.7 for bootstrapping treatment.)
2. KD-tree query: find the K nearest neighbors of v_t in the
   stored velocity set.
3. Compute the local empirical covariance with Gaussian kernel
   weighting:
   ```
   w_s = exp(-||v_t - v_s||² / (2h²))
   μ^τ(v_t) = Σ_s w_s · e_s^τ / Σ_s w_s
   Σ^τ(v_t) = Σ_s w_s · (e_s^τ - μ^τ(v_t))(e_s^τ - μ^τ(v_t))ᵀ / Σ_s w_s
   ```
4. Use μ^τ(v_t) and Σ^τ(v_t) as the spatial prior in the
   Mahalanobis loss for the Kalman update:
   ```
   L_spatial^τ(e_t, v_t) = (e_t - μ^τ(v_t))ᵀ Σ^τ(v_t)⁻¹ (e_t - μ^τ(v_t))
   ```

**K-tuning for 4D vs 2D**: in 4D space, the typical neighbor
distance grows as N^{-1/4} versus N^{-1/2} in 2D. For typical
N = 30K high-confidence frames, the 4D ball containing K=200
neighbors is ~3.5× larger than the equivalent 2D ball with
the same K. To keep the local-covariance neighborhood at
comparable specificity:

| Conditioning | Default K | Bandwidth h |
|--------------|-----------|-------------|
| 2D (head only)   | 200       | Silverman 2D     |
| 4D (head + body) | 50        | Silverman 4D     |

The 4D defaults (K=50, Silverman 4D bandwidth) keep ball
volume comparable. Both are tunable per project. Cross-
validation for K and h is a v3 enhancement; for v2 we use
defaults and let users adjust if results are unsatisfactory.

**Pros of this approach**:
- No parametric form to commit to (no "covariance is
  quadratic in velocity" assumption that could be wrong)
- Naturally adapts to whatever velocity-pose coupling exists
  in the data
- Maps cleanly to the "manifold for sampling" framing: the
  high-confidence frames define a manifold in
  (velocity, configuration) space; local covariance is the
  manifold's tangent structure
- Adds new high-confidence data incrementally (just append
  to the saved features array)
- 4D conditioning captures head/body decoupled regimes
  (scanning, twisting) that 2D conditioning misses

**Cons / costs**:
- Storage: O(|S_high^τ| × dim) per triplet — for a 50K-frame
  video with ~30K high-p frames per triplet and 15-dim
  features, ~3.6MB per triplet. With ~5 triplets per
  body-part scheme, ~18MB per project. Trivial.
- KD-tree query is O(log N) per query frame regardless of
  ambient dimension. Sub-millisecond.
- O(K × dim²) per query for the covariance computation.
  K=50, dim=15: ~11K ops per query frame. Faster than the
  2D K=200 case despite larger feature dim. Per-video cost:
  ~5s for a 50K-frame video on one core. Acceptable,
  parallelizable across triplets.
- Bandwidth h is a hyperparameter. The Silverman default is
  reasonable but not optimal; could tune via cross-validation
  if it matters.

**Numerical edge cases**:
- Query velocity in a region with few/no neighbors (animal
  doing something rare). Mitigation: when the KNN distances
  are all > some_threshold, fall back to global covariance
  (the static Σ^τ from §3.4) for that frame and emit a
  diagnostic count.
- Σ^τ(v_t) is singular or near-singular (can happen if K
  is small or the local samples are colinear). Mitigation:
  add a small regularizer εI before inversion (ε = 1e-6 ×
  trace(Σ)/dim by default).
- Head velocity can't be reliably computed at query time
  (head triplet has low-p at t or t-1). Mitigation: use the
  smoother's own predicted head velocity from the dynamics
  model. This creates a circular dependency that resolves
  via §3.4.7.

### 3.4.6 Saved model artifact

The fitted velocity-conditional covariance model is persisted
per project as a `.npz` file:

```
project_folder/logs/kalman_smoother_model.npz
├── version          : str (e.g., "v2.0")
├── created_at       : ISO datetime
├── source_data_hash : str (hash of the input pose CSVs that
                            were used for fitting)
├── triplets         : list of triplet definitions
                       [(name, marker_i, marker_j, marker_k), ...]
├── head_triplet     : reference to which triplet defines the
                       head frame
├── body_markers     : list of marker names defining the body
                       frame (centroid + PCA major axis); empty
                       if body conditioning unavailable
├── per_triplet:
│   ├── <triplet_name>_features    : (N_high, 15) float32 array
│   ├── <triplet_name>_velocities  : (N_high, 4)  float32 array
                                      (vx_h, vy_h, vx_b, vy_b)
│   ├── <triplet_name>_kdtree_data : pickled scipy.spatial.cKDTree
│   ├── <triplet_name>_static_mu   : (15,) float32 — fallback mean
│   ├── <triplet_name>_static_cov  : (15, 15) float32 — fallback covariance
│   └── <triplet_name>_n_samples   : int
├── per_marker:
│   ├── <marker>_sigma_base : float32 — measurement noise baseline
│   ├── <marker>_q_pos      : float32 — position process noise
│   ├── <marker>_q_vel      : float32 — velocity process noise
│   └── <marker>_n_samples  : int
└── config:
    ├── tau_high            : float — likelihood threshold for fitting
    ├── p_floor             : float — likelihood floor for variance map
    ├── kdtree_k            : int   — K for KNN (default 50 in 4D)
    ├── kdtree_bandwidth    : float — kernel bandwidth h
    ├── spatial_weight      : float — λ_spatial multiplier
    ├── reg_epsilon         : float — covariance regularizer
    ├── conditioning_dim    : int   — 2 (head only) or 4 (head + body)
    └── body_axis_method    : str   — "pca_major" | "spine_direction"
                                       | "designated_triplet"
```

For body_axis_method = "pca_major" (the committed default),
the body axis is recomputed at fit and inference time by
running PCA on the body markers per frame. The body markers
list is recorded in the artifact so inference uses the same
markers. PCA itself is reproducible (deterministic given
inputs) so the body axis is reconstructible without storing
per-frame axis arrays.

**Versioning**: any change to the feature vector definition,
the number of dimensions, the conditioning scheme, or the
fitting protocol increments the version string. Loaders
check version compatibility and refuse to load mismatches.

**Reuse across projects**: if `source_data_hash` matches the
input data hash at load time, the fitted model is used as-is.
If it doesn't match (data has changed), the user can choose
to: (a) refit, (b) load the old model and apply to new data
unchanged (with a warning), (c) abort.

**Cross-project sharing**: a model fit on Project A can be
loaded by Project B if the body-part schemes match exactly.
This is useful when fitting on a large project and applying
to a smaller / newer one.

### 3.4.7 The state-velocity circular dependency

The velocity-conditional covariance creates a circular
dependency: Σ^τ(v_t) depends on v_t, but v_t is what we're
trying to estimate via the smoother. Standard linear Kalman
assumes R_t and Q are constant, not state-dependent.

Three resolutions, in order of complexity:

**(a) Two-pass approach** (simplest, recommended for v1):
- Pass 1: run plain Kalman with static Σ^τ (no velocity
  conditioning). Get a first estimate of all states including
  velocities.
- Pass 2: re-run Kalman with velocity-conditional Σ^τ(v_t)
  using the velocities from Pass 1.

This is approximate but converges quickly in practice (the
Pass-1 velocities are usually accurate enough that Pass-2
covariances are stable). Iterate if Pass-2 velocities differ
significantly from Pass-1.

**(b) Iterated EKF** (cleanest mathematically):
- At each Kalman update step, use the predicted state
  ŝ_{t|t-1} to compute v̂_t, then use Σ^τ(v̂_t) for the update.
- The state estimate appears in the covariance through the
  velocity, making this a state-dependent measurement noise.
- Convergence is guaranteed under mild regularity conditions
  (Jazwinski 1970, Ch. 8).

**(c) Particle filter / Unscented Kalman**: most flexible but
much heavier compute. Not justified here.

**Recommendation**: implement (a) for v1; upgrade to (b) if
empirical results show Pass-1/Pass-2 differing materially.



### 3.5 Empirical fit of process noise and measurement noise

For each marker i, take the high-confidence segments
S_high^i = { t : p_t^i > τ_high }. From these:

```
σ_base^i² = Var(residuals after detrending position with savgol)
            — captures within-segment measurement noise
q_vel^i² = Var(velocity differences across consecutive
                high-confidence frames)
            — captures dynamic process noise
q_pos^i² ≈ q_vel^i² · Δt²
            — derived consistency constraint
```

This is a one-pass fit per video. Cheap, and self-calibrating.

### 3.6 Augmented measurement update

The Kalman filter update step is normally:
```
y_t = z_t - H · ŝ_t|t-1                    # innovation
S_t = H · P_t|t-1 · Hᵀ + R_t              # innovation covariance
K_t = P_t|t-1 · Hᵀ · S_t⁻¹                # Kalman gain
ŝ_t|t = ŝ_t|t-1 + K_t · y_t               # state update
P_t|t = (I - K_t · H) · P_t|t-1          # covariance update
```

To incorporate the triplet spatial prior, we add a "virtual
measurement" per triplet at each frame:
```
z_t^τ_virtual = μ^τ                                     # the empirical mean
R_t^τ_virtual = Σ^τ                                     # the empirical covariance
H_t^τ_virtual: ℝ^(4N) → ℝ⁶, computed from the predicted
positions and the egocentric transform at the predicted
configuration
```

This is standard "constrained Kalman filter" territory.
Because the egocentric transform is *nonlinear* (it depends on
the predicted positions), we'd actually need an Extended
Kalman filter (EKF) here — linearize the transform at each
prediction step. The math is well-documented (Simon, *Optimal
State Estimation*, 2006, Ch. 13) and not novel to us.

Alternatively: do the linear Kalman without spatial prior in
the forward pass, then use the spatial prior as a one-shot
constraint pass after RTS smoothing. Less optimal but much
simpler to implement.

### 3.7 RTS backward smoother

Standard Rauch-Tung-Striebel smoother:
```
ŝ_t|T = ŝ_t|t + C_t · (ŝ_{t+1}|T - ŝ_{t+1}|t)
P_t|T = P_t|t + C_t · (P_{t+1}|T - P_{t+1}|t) · C_tᵀ
C_t = P_t|t · Fᵀ · P_{t+1}|t⁻¹
```
Done after the forward pass completes. Gives the optimal
posterior estimate using all T frames of data, both past and
future.

For offline pose smoothing (which is our use case — we have
the whole video), the smoother is the right choice over the
plain filter.

---

## 4. Design decisions — committed answers

These were design choices when the doc was first drafted.
Gravio committed answers on 2026-05-01. The committed answers
are recorded here, with the original options preserved as
context for future reference.

### 4.1 What goes in the egocentric covariance ✅ COMMITTED
**Committed: full feature vector (option d), extended to 15-dim
in 2026-05-01 update.** Internal angles + inter-marker distances
+ egocentric (x,y) of each marker + head-frame egocentric
velocities + body-frame egocentric velocities. 15-dim per
triplet. See §3.4 for the full vector definition. Rank-deficient
covariance handled via small-eigenvalue regularization or
pseudo-inverse.

Original options kept for reference:
- (a) Internal angles only — 2-dim
- (b) Pairwise distances + angles — 5-dim
- (c) Egocentric (x,y) of each marker — 6-dim
- (d) **All combined + head velocity — 13-dim** (initial commit)
- (e) **All combined + head velocity + body velocity — 15-dim**
      ← extended commit (2026-05-01)

### 4.2 How are triplets identified ✅ COMMITTED
**Committed: use ALL angles and ALL marker distances**
(combined-feature approach), and **persist the fitted model**
to disk so it can be reloaded across sessions and reused
across projects with the same body-part scheme. See §3.4.6 for
the saved-model artifact spec.

This is a hybrid of options (b) and (c) in the original
enumeration: there's still a default selection per body-part
scheme that determines which 3-marker groups get treated as
"triplets" (you wouldn't compute angle-distance features for
every possible (i, j, k) combination — that scales as N choose
3, prohibitive for 16bp schemes), but **within each chosen
triplet**, all features are used. The triplet selection
defaults are listed in `defaults.py`; users can override per
project.

Original options kept for reference:
- (a) User picks manually
- (b) Hardcoded per body-part scheme — defaults
- (c) Auto-detected from data
- **Committed: defaults from (b) + manual override from (a),
  with full-feature treatment per triplet, plus saved model
  artifact**

### 4.3 How rigid is the constraint ✅ COMMITTED + EXTENDED
**Committed: option (b) — soft Mahalanobis with tunable
weight** AS THE BASE, BUT EXTENDED with velocity-conditional
covariance. The covariance Σ^τ is not static; it's a function
Σ^τ(v) of the head-frame egocentric velocity at the query
frame. See §3.4.5 for the full velocity-conditional
treatment.

This is a meaningful upgrade over the original option (b). The
Mahalanobis loss still applies:
```
L_spatial^τ(e_t, v_t) = (e_t - μ^τ(v_t))ᵀ Σ^τ(v_t)⁻¹ (e_t - μ^τ(v_t))
```
But both the mean μ and covariance Σ are now velocity-
dependent, fit locally via KD-tree + kernel weighting.

The spatial weight `λ_spatial` (default 1.0) still applies as
a multiplicative scalar, so the user can tune the relative
strength of the spatial vs temporal priors:
```
total_loss = L_temporal + λ_spatial · L_spatial
```

Original options kept for reference:
- (a) Strict (Mahalanobis as exact)
- (b) Soft empirical with tunable weight ← base of committed
- (c) Multimodal (GMM) — superseded by velocity-conditional
- **Committed: (b) + velocity-conditional Σ via KD-tree
  (committed approach is option C in the velocity-conditional
  section: KD-tree + local kernel covariance)**

The velocity-conditional Σ is a strict generalization of (c):
where (c) would identify discrete postural modes, the
manifold approach captures continuous variation. If the
empirical data has discrete clusters in velocity space, the
KD-tree naturally captures them; if the variation is smooth,
the kernel weighting handles that too. (c) is therefore a
special case of the committed approach, not an alternative.

### 4.4 Linear vs extended Kalman ✅ COMMITTED FOR v1
**Committed for v1: option (a) — linear KF + two-pass
treatment of the velocity-state circular dependency.** See
§3.4.7. Upgrade to (b) iterated EKF in v2 if Pass-1/Pass-2
velocities differ materially in empirical testing.

### 4.5 Per-marker vs joint state ✅ COMMITTED
**Committed: per-marker.** Each marker has its own 4-dim
Kalman filter; triplet spatial coupling enters via post-hoc
constraint pass. See §3.6 for the augmented-measurement
formulation.

### 4.6 Validation strategy ✅ COMMITTED
**Committed: combined approach.** Synthetic data validation +
hand-curated frames + downstream-task improvement. Skipping
ground-truth video segments (option c original) since not all
behavioral paradigms have natural anchor events. Specifics:

1. Synthetic: generate trajectories with known noise + known
   dropouts of varying length. Verify recovery error scales
   correctly with input noise and is bounded.
2. Hand-curated: ~500 frames manually labeled across 3 videos
   with diverse behaviors. Compare smoothed positions to
   labels via mean Euclidean error and 95th-percentile error.
3. Velocity manifold validation: hold out a velocity bin
   (e.g., the fastest 10% of frames), fit on the rest, verify
   the smoother performs well in the held-out regime.
4. Downstream: train a behavior classifier on features
   derived from smoothed trajectories; compare F1 score to
   baseline (savgol-smoothed input).

---

## 5. Implementation plan (revised after committed answers + Gravio data review)

After the committed answers in §4 AND the Stage 0 diagnostic
results on Gravio's 67-session dataset (see §5.0), the plan
has been refined further. The big change vs the original
sketch: the head-marker triplet is deferred from v1 to v2 as
a full latent-posture model, because in 2D camera projection
inter-head-marker distances vary meaningfully with posture
(rearing, head pitch) and a static-Σ triplet prior would
flatten that signal.

### 5.0 Stage 0 results summary (April-May 2026)

Run on `/data/testing/mufasa/test-20260427/project_folder/csv/input_csv/`:
- 67 sessions × ~54k frames = 3.6M frames
- Per-marker median p around 0.5-0.87; threshold 0.95 produced
  almost no high-confidence frames → ran at threshold 0.70.
- Aggregate worst frac_high = 0.302; avg = 0.561
- Per-session quality: 0/67 "good" (>0.5), 34 marginal, 33 bad
- 5-7 sessions have full-session-length dropouts on back3 or
  back4 (catastrophic — flagged for separate handling)
- Auto-detected rigid pairs (CV < 0.20) are dominated by
  back↔headmid and back↔lateral_* with avg CV = 0.131
- Velocity unimodal (head and body); head-body Pearson r = 0.82
- ear_left↔ear_right CV = 0.154 (just outside the rigid-pair
  threshold) — but this is genuine 3D-rigid pair, with apparent
  variance driven by 2D projection (rearing changes projected
  inter-ear distance). Treated as BEHAVIORAL SIGNAL, not
  rigidity constraint.

Recommendation produced: **build v1 + static-Σ triplet prior**
(skip velocity-conditional Σ for v1; defer to v2 as full
latent-posture extension).

### 5.1 Stages

**Stage 0: Diagnostic** — DONE. See `kalman_diagnostic.py` and
patches 75-84. The diagnostic now produces:
- Per-marker likelihood + dropout stats (per-session and
  aggregate)
- Auto-detected rigid pairs (excluding head markers)
- Auto-detected candidate triplets (body-only)
- Behavioral-signal pairs (head-internal, e.g. ear-distance)
- Per-session quality bar chart
- Build/scope recommendation

**Stage 1: v1 — joint-state Kalman + RTS with body-only static-Σ
triplet prior** ~4 weeks

After review of the per-marker-only v1 sketch, we committed
to a stronger v1: a joint-state Kalman filter where the body
triplet prior enters as a *pseudo-measurement* applied
jointly with real per-marker observations, rather than as a
sequential post-hoc correction (which would have been the
original v1's "Option A" approximation). This is "Option B"
in the build review notes — a principled implementation
rather than a faster approximation that v2 would have to
replace.

**State vector**: 4n-dim where n is the number of markers.
For each marker: [x, y, vx, vy]. For Gravio's 15-marker
project: 60-dim state.

**Dynamics**: F is block-diagonal with 15 copies of the
standard CV block, dt=1/fps. Q has per-marker dynamics
noise on the diagonal blocks.

**Per-frame observations**: variable-dimensional per frame.
For each marker i with p_{i,t} ≥ τ: a 2-dim measurement
(x_i, y_i) with R_{i,t} = σ_base,i² / p_{i,t}². For each
candidate triplet (a, b, c) where global Σ exists: a 6-dim
pseudo-measurement [x_a, y_a, x_b, y_b, x_c, y_c] = mean
configuration with covariance Σ_abc.

**Triplet Σ scope**: GLOBAL fit. One Σ_abc per triplet,
estimated from high-confidence frames across all sessions
in the input collection. Sessions where a triplet's markers
have insufficient coverage still get the global prior
applied (at the cost of confident-looking interpolation
that's not directly supported by that session's
observations — flagged in output for the user to decide).

**HEAD MARKERS get NO triplet prior in v1.** Each head
marker (nose, ear_left, ear_right, headmid) is smoothed
through the same joint-state filter as part of the 60-dim
state but with NO pseudo-measurements — only their per-marker
observations and their dynamics couple them. This preserves
posture-driven signal (rearing, head pitch) for downstream
behavioral feature extraction. Coupled handling deferred to
v2 with full latent posture.

**EM noise fitting**: σ_base, q_pos, q_vel per marker fit
via EM iterations (typically 3-5):
  1. Initialize from p>0.5 finite-difference velocity stats
  2. Run forward+backward smoother
  3. Re-estimate noise from smoothed residuals
  4. Iterate until parameters stabilize
Σ_abc is fit ONCE (not EM-updated) — the rigid configuration
prior shouldn't depend on noise estimates.

**Numerical implementation**: dense NumPy for the v1 build.
60×60 matrices are tractable (~10ms/frame, ~10 hours for
3.6M frames; acceptable for one-time runs). Sparse / info
form deferred unless real-data runs prove unacceptably slow.

**Saved-model artifact** (.npz): per-marker noise
parameters, triplet definitions, triplet Σ matrices, source
data hash, version tag.

**Patch breakdown** (patches 85-90, sequential ship):
  85. Joint-state forward filter only, fixed noise params,
      no triplet. Synthetic smoke tests with known ground
      truth.
  86. RTS backward smoother. Synthetic: smoother variance
      < filter variance.
  87. Triplet pseudo-measurement infrastructure + global
      Σ fitting.
  88. EM noise parameter fitting wrapped around forward+
      backward.
  89. Multi-file orchestrator, parquet support, CLI,
      save/load. Matches diagnostic's interface.
  90. Validation on real data + iteration.

Each patch lands cleanly without breaking the others.
Synthetic smoke tests run after every patch.

**Stage 1.5: Rearing feature module** ~1 day (after Stage 1
ships)
- New module `mufasa/features/rearing.py`
- Takes smoothed pose DataFrame, returns continuous `d_ears(t)`
  and discrete rearing state via 2-state lognormal HMM (EM-fit)
- Global HMM fit across all sessions concatenated; applied per
  session for inference
- Lognormal emission distribution (right choice for
  non-negative distances; +5 lines vs Gaussian)
- Missing-observation handling for ear-dropout frames
- Exposes both posterior `p_rearing(t)` and Viterbi
  `is_rearing(t)`

**Stage 2: v2 — full latent posture model for the head** ~3-4
weeks
- Augment v1's joint-state with a 1D head-pitch latent variable
- Posture-conditional triplet Σ for the head triplet
  (nose, ear_left, ear_right) — distinct Σ per posture bin,
  or a continuous Σ(pitch)
- Body remains v1 (static Σ). Head gets posture-aware Σ.
- Validation: synthetic + hand-curated rearing frames + held-out
  posture-distinguishability test

**Stage 3: v3 — Qt UI** ~1 week
- Per-bodypart and per-triplet config table form
- Triplet selection UI (defaults from auto-detect + manual
  override)
- Likelihood threshold + spatial weight controls
- Diagnostic plots in-app (per-session quality, candidate
  triplet visualization, before/after trajectory comparison)
- Integration with existing data import workflow

**Total estimate: 6-8 weeks** for v1 + v1.5 + v2 + Qt UI.
v1 + v1.5 alone is ~4-4.5 weeks and produces principled
joint smoothed pose + rearing feature.

### 5.2 Module structure (revised)
```
mufasa/data_processors/kalman_smoother/
├── __init__.py
├── empirical_fit.py          # σ_base, q_pos, q_vel fitting from S_high
├── kalman_filter.py          # per-marker forward filter + RTS smoother
├── triplet_features.py       # egocentric frame + 15-dim feature extraction
├── velocity_frame.py         # head- and body-frame velocity computation
├── conditional_covariance.py # KD-tree + local kernel covariance
├── spatial_prior.py          # Mahalanobis loss in measurement update
├── two_pass.py               # bootstrap velocity → re-smooth orchestration
├── model_io.py               # save/load .npz artifact with versioning
├── orchestrator.py           # KalmanPoseSmoother — main user-facing class
└── defaults.py               # default triplet definitions per body-part scheme
```

### 5.3 Public API (revised)
```python
class KalmanPoseSmoother(ConfigReader):
    def __init__(
        self,
        config_path: str,
        data_path: str | list[str],
        # Triplet configuration
        triplet_strategy: Literal['default', 'manual'] = 'default',
        manual_triplets: list[tuple[str, str, str]] | None = None,
        head_triplet: tuple[str, str, str] | None = None,
            # If None, picked from defaults; required for vel-conditional
        body_markers: list[str] | None = None,
            # Markers used for body centroid + PCA major axis.
            # If None, defaults to all markers minus head triplet markers.
            # Pass empty list to disable body conditioning (2D fallback).
        # Hyperparameters
        likelihood_threshold: float = 0.95,  # τ_high
        likelihood_floor: float = 0.01,      # p_floor
        spatial_weight: float = 1.0,         # λ_spatial
        kdtree_k: int | None = None,
            # KNN for vel-conditional Σ. Default: 200 if 2D, 50 if 4D.
        kdtree_bandwidth: float | None = None,  # None = Silverman
        spatial_reg_epsilon: float = 1e-6,   # covariance regularizer
        # Mode flags
        use_velocity_conditional: bool = True,
            # Set False to use static Σ (Stage 1 behavior)
        use_body_velocity: bool = True,
            # Set False to drop body velocity from conditioning vector
            # (2D conditioning even when body markers are available).
            # Useful for ablation testing or when body markers are
            # too unreliable to anchor the body frame.
        # Model persistence
        model_path: str | None = None,
            # If set + file exists, load that model (skip refit)
            # If set + file doesn't exist, fit and save to that path
            # Default: <project>/logs/kalman_smoother_model.npz
        force_refit: bool = False,
        # Output
        save_originals: bool = True,
        copy_to_dir: str | None = None,
    ) -> None: ...

    def fit(self) -> None:
        """Compute or load the saved model. Idempotent."""
    def run(self) -> None:
        """Apply the fitted smoother to all input files."""
    def fit_and_run(self) -> None:
        """Convenience: fit() then run()."""
```

### 5.4 Performance budget (revised)
For a typical project (50K frames × 8 markers × 67 videos):

**Fitting (one-time per project)**:
- Empirical fit per marker (σ_base, q_pos, q_vel):
  O(T × N) per video. ~50K × 8 = 400K ops. Negligible.
- Egocentric feature extraction:
  O(T × K_triplets × 13) per video. ~50K × 5 × 13 = 3.3M ops.
  Negligible.
- KD-tree build: O(N_high × log(N_high)) per triplet.
  N_high ~ 30K, ~15K log(30K) ≈ 220K ops per triplet × 5
  triplets = 1M ops. Negligible.
- Total fit time per video: ~50ms on one core, dominated by
  I/O. Project total: ~3-5 seconds.

**Inference**:
- Per-marker Kalman + RTS: O(T × N × state_dim²) per video
  = 50K × 8 × 16 = 6.4M ops. ~100ms.
- Velocity-conditional covariance lookups: O(T × K × log N + T × K × dim²)
  per video. With K=200, dim=13, T=50K: ~50K × 200 × 13² × 5
  = ~85M ops. ~3-5 seconds. **Dominant cost**.
- Spatial prior measurement update: O(T × K_triplets × dim³)
  for matrix inverse. ~50K × 5 × 13³ = 550M ops. ~10-15s.
  **THIS is the actual dominant cost**.

**Per-video total: ~15-20 seconds. Project total (67 videos):
~15-20 minutes** on one core. Parallelizable across videos
trivially. Acceptable but no longer "negligible" — the
velocity-conditional path is genuinely substantive compute.

If this turns out too slow on real data, optimization
options (in order of leverage):
1. Reuse covariance across nearby frames if velocity hasn't
   moved much (avoid recomputing Σ^τ(v) at every frame)
2. Cache pseudo-inverse of Σ^τ(v) via the matrix-inversion
   lemma when Σ changes incrementally
3. Numba @jit on the inner KD-tree-query + weighted-covariance
   loop (~5-10× speedup)
4. Move to Cython or NumPy-broadcast-batched if the above
   isn't enough

We'd profile after Stage 1 finishes and decide based on
empirical timings.

### 5.5 Testing strategy (revised)
Beyond the v1-original list:

- **Saved-model round-trip**: fit, save, load, run; verify
  identical outputs to the unsaved path
- **Velocity-conditional sanity**: synthetic data with known
  velocity-dependent posture distributions; verify the local
  covariance correctly identifies the regimes
- **Bandwidth sensitivity**: vary h; verify outputs are stable
  across reasonable choices
- **Sparse-velocity-region fallback**: simulate a query frame
  with no nearby high-confidence neighbors; verify the static
  fallback kicks in cleanly
- **Two-pass convergence**: verify Pass-1/Pass-2 velocities
  converge in <3 iterations on real data
- **Cross-version refusal**: try to load a v1.0 model with v2.0
  code; verify clean error message rather than silent corruption

### 5.6 What's NOT in the v1+v2 implementation
- Multi-animal handling — single animal only initially. Multi-
  animal would require extending the triplet system to handle
  per-animal triplet sets.
- Joint pose state — still per-marker. Joint state could
  improve the spatial coupling but adds state-dimension
  explosion.
- Online (real-time) mode — strictly offline batch processing
  matches the existing smoother UX.
- Adaptive bandwidth selection (cross-validation for h) —
  defaults to Silverman; user can override; cross-validation
  is a v3 feature if needed.
- Multi-modal explicit fit (option (c) in original §4.3) —
  superseded by velocity-conditional manifold approach.

These are natural future extensions if the core system proves
valuable.

---

## 6. Risks and unknowns

### What I'm certain of
- The math is correct (for the static-Σ case). Linear-Gaussian
  state-space models with per-frame measurement variance are
  textbook (Kalman 1960, Rauch-Tung-Striebel 1965). Empirical
  fitting of process noise from high-confidence segments is
  sensible. Adding a Mahalanobis-distance spatial prior from
  triplet covariance is well-defined.
- Kernel density estimation and KD-tree-based local statistics
  are textbook (Silverman 1986, Friedman et al. 1977). Using
  them for state-conditional covariance is a sensible extension.
- The implementation is tractable. Pure NumPy + scipy.linalg
  + scipy.spatial.cKDTree. No new dependencies.
- Prior art exists. Lightning Pose's EKS does almost this
  exact thing with PCA spatial prior instead of triplets.
  We're not inventing fundamentally new math for the static-Σ
  case.

### What I'm asserting but not certain of
- That this will *visibly improve* the smoothed trajectories
  on Gravio's specific videos. The improvement size depends
  on the likelihood distribution: if your DLC model produces
  consistent p>0.95 detections on all bodyparts, the
  Kalman smoother behaves similarly to a low-pass filter and
  the marginal benefit is modest. If you have frequent
  dropouts or bimodal likelihood distributions, the benefit
  is large.
- That triplets are the "right" structural prior for
  rodent behavior. Lightning Pose chose PCA. Anipose chose
  limb-length constraints. Triplets are appealing (local,
  interpretable) but not obviously better. We may discover
  that PCA-style holistic priors work better in practice.
- **That velocity-conditional Σ buys substantial benefit over
  static Σ.** The velocity conditioning is a clean
  generalization, but it adds compute (~150-200x more
  inference cost — see §5.4 performance budget) and
  implementation complexity. If the empirical
  velocity-vs-configuration coupling in real rodent data is
  weak, the static Σ is good enough and the velocity
  conditioning is wasted work. **The diagnostic step
  specifically tests this.**

### What I haven't investigated
- Whether your test-project video has a likelihood
  distribution that would benefit from this. The diagnostic
  is now a committed first step (Stage 0 in §5.1).
- Whether the existing Lightning Pose EKS can be wrapped
  rather than reimplemented. They're open source. If the API
  is reasonable, "use Lightning Pose's EKS" might be a faster
  path than "build our own triplet variant." (Caveat: their
  EKS uses static PCA prior, not velocity-conditional triplet
  covariance, so wrapping wouldn't give us the velocity-
  conditional behavior.)
- Whether the rigid-triangle assumption holds for typical
  rat/mouse behavioral postures. Head-turning alone breaks the
  nose+leftEar+rightEar triplet's rigidity. **However**,
  velocity-conditional Σ partially addresses this: when the
  head turns sharply, the velocity changes, and the
  conditional covariance for that velocity regime captures
  whatever the empirical postural correlations are.
- Whether the two-pass approach (§3.4.7) converges in 2-3
  iterations or needs more. If many, we'd want iterated EKF.
- Whether bandwidth h selection is critical. Default
  Silverman is reasonable but not always optimal. Cross-
  validation might be needed.

### What could go wrong
- **Over-smoothing.** If σ_base is mis-estimated low, the
  filter trusts the measurements too much and the output
  isn't smooth enough. If too high, the filter trusts the
  dynamics too much and the output drifts away from real
  data.
- **Velocity-conditional Σ ill-conditioned in sparse regions.**
  If the animal rarely visits certain velocity regimes, the
  KNN-based local covariance has high variance / few samples.
  The fallback-to-static-Σ mechanism should catch this but
  could be brittle if the threshold is mis-tuned. The 4D
  conditioning makes this risk worse than 2D, since 4D space
  has many more "sparse corners."
- **Body-axis sign flip from PCA.** PCA returns an unsigned
  eigenvector (v and -v are both valid). Sign disambiguation
  uses head direction when available, with continuity fallback.
  Risks: if head markers fail at the start of a long valid
  body run, the sign is arbitrary for that whole run; if
  head markers are unreliable for an extended segment, the
  body axis could drift via continuity errors (each frame's
  sign is correct relative to the previous, but the chain can
  walk away from the head-anchored definition). The diagnostic
  reports n_signed_arbitrary / n_signed_by_continuity counts
  so users can detect this.
- **Body shape near-isotropic (PCA degenerate).** When body
  markers form a roughly circular pattern around the body
  centroid (eigenvalue ratio close to 1), the PCA major axis
  is poorly determined and small noise can flip its
  orientation. Common in body-marker schemes that lack a
  strong anterior-posterior arrangement. Diagnostic detects
  these frames; the fitting code should warn if their
  fraction is non-trivial.
- **Two-pass divergence.** If Pass-1 velocities are very wrong
  (because the animal moves erratically and the static-Σ
  initial smoother gets it wrong), Pass-2 conditional Σ might
  be worse than static. Iterated EKF would be more robust but
  more complex.
- **Mode collapse on multi-postural data.** Less of a concern
  with velocity conditioning than static (different postures
  often have different velocities, so velocity conditioning
  separates them naturally), but not impossible if two
  postures share a velocity profile.
- **Triplet selection brittleness.** A "wrong" triplet (one
  whose markers aren't actually rigid) gives a covariance
  that constrains the smoothed output toward an artificial
  configuration. Defaults need to be carefully chosen per
  body-part scheme. Velocity conditioning doesn't fix this —
  if the triplet is wrong, it's wrong in every velocity
  regime.
- **Implementation bugs.** Kalman filter implementations are
  notoriously bug-prone — sign errors in covariance updates,
  forgetting to symmetrize, numerical ill-conditioning. The
  velocity-conditional KD-tree path adds new bug surfaces:
  off-by-one in feature indexing, wrong rotation matrix in
  egocentric transform, KD-tree query returning wrong-frame
  velocities. Careful tests required.

---

## 7. Recommendation

Before any v1+v2 implementation: **run the diagnostic.** This
has been committed (Stage 0 in §5.1). The diagnostic produces
a 5-component report for one or more of Gravio's videos:

1. **Likelihood histograms** per marker, all-marker
   aggregated. If mostly p > 0.95, the Kalman smoother's
   variance-aware behavior buys little; existing smoothers
   suffice.

2. **Run-length distribution of consecutive low-p frames.**
   Long runs (>30 frames at 30 fps = >1 second of dropout)
   are exactly where Kalman + spatial prior helps most.

3. **Inter-marker distance distributions** for canonical
   "rigid" pairs (e.g., nose↔leftEar in 8bp scheme). Tight
   distributions confirm the rigidity assumption that triplet
   covariance relies on.

4. **Head-frame egocentric velocity distribution.** If
   uniform/single-mode, velocity-conditional Σ adds compute
   without adding signal — fall back to static Σ. If
   multi-modal or has wide variance, velocity-conditional Σ
   is well-justified.

5. **Velocity-vs-configuration scatter** for one canonical
   triplet. Plot e.g. nose-ear angle vs. forward velocity for
   high-confidence frames. Tight band that varies with
   velocity → velocity-conditional Σ is well-justified.
   Uniform cloud → static Σ is enough.

If components (1) and (2) show your data is clean, **don't
build this. Existing smoothers are fine.**

If they show real dropout / bimodal-likelihood patterns, and
(3) shows tight rigidity, build v1.

If additionally (4) and (5) show velocity coupling, build v2.

If components (4) or (5) are weak: build v1 + a static-Σ
spatial prior, skip the velocity-conditional machinery.
Saves ~3 weeks of work.

The diagnostic is shipping in a separate patch alongside this
doc update. Run it on a representative subset of your videos,
share the plots, and we decide v1-only vs v1+v2 based on
empirical evidence rather than speculation.

### Build order if diagnostic green-lights v1+v2

1. **Stage 1 — v1 KalmanPoseSmoother**: per-marker filter +
   RTS smoother with likelihood-derived variance. Saved
   per-marker noise parameters. ~1-2 weeks.
2. **Stage 2 — v2 velocity-conditional triplet covariance**:
   add the full feature vector + KD-tree + local kernel
   covariance. Saved-model artifact. Two-pass orchestration.
   ~3-4 weeks.
3. **Stage 3 — Qt UI**: per-bodypart and per-triplet config
   table. Diagnostic plots in-app. ~1 week.

**Honest scope quote**: 4-7 weeks of focused work for the
full system. This is a real feature, not a side project. It's
publishable methodologically (velocity-conditional anatomical-
prior Kalman smoother for behavioral pose data is, to my
knowledge, novel). Worth building only if the diagnostic
confirms the data warrants it.

---

## 8. Decisions committed (2026-05-01)

The five original questions have been answered, plus a sixth
about body-frame velocity:

1. **Triplet content (§4.1)**: ✅ Full 15-dim feature vector
   (angles + distances + egocentric coords + head-frame
   velocity + body-frame velocity). Was option (d) — "all of
   the above" — committed; subsequently extended in #6 below.

2. **Triplet selection (§4.2)**: ✅ Defaults per body-part
   scheme + manual override. All angles + all marker
   distances per chosen triplet. **Saved model artifact** so
   fitted models persist across sessions and can be reused
   across projects. Hybrid of (a) and (b) committed.

3. **Constraint rigidity (§4.3)**: ✅ Soft Mahalanobis with
   tunable λ_spatial weight (option b) AS THE BASE,
   **EXTENDED with velocity-conditional Σ via KD-tree + local
   kernel covariance** (committed approach: Option C in the
   velocity-conditional section). The covariance is no longer
   static — it's a function Σ^τ(v) of velocity. Initially v
   was 2D (head only); subsequently extended to 4D in #6.

4. **Run diagnostic before committing? (§7)**: ✅ Yes.
   Diagnostic is Stage 0 in §5.1 and ships as a separate
   patch alongside this doc update.

5. **Wrap Lightning Pose's EKS instead of building? (§7)**:
   Open question — wrapping LP-EKS would save weeks but
   gives static PCA prior, not velocity-conditional triplet
   covariance. Decision deferred to post-diagnostic: if
   diagnostic shows velocity conditioning isn't needed,
   wrapping LP-EKS becomes much more attractive.

6. **Body-frame velocity? (added 2026-05-01)**: ✅ Yes — use
   both head-frame and body-frame velocities as 4D
   conditioning. Body frame defined by **option (b): body
   centroid + PCA major axis** on body marker positions per
   frame. Sign disambiguated using head direction (preferred)
   or temporal continuity (fallback). The 4D conditioning
   captures decoupled head/body motion regimes (scanning,
   twisting) that 2D conditioning misses. Approach 1 (4D
   shared conditioning across all triplets) committed over
   Approach 2 (per-triplet head-or-body conditioning) for
   simplicity — the empirical fitting will learn which
   dimensions matter for each triplet without us having to
   label them.

### Remaining open decisions

These haven't been decided yet, but are second-order — we can
make these calls during implementation rather than as a
prerequisite:

- **Bandwidth h selection method**: Silverman default vs
  cross-validated. Decide empirically after seeing the
  diagnostic data.
- **K (KNN size for local covariance)**: default K=50 in 4D
  conditioning (was K=200 in 2D). Tunable per project.
- **Two-pass vs iterated EKF for the circular dependency**:
  start with two-pass; upgrade to iterated only if convergence
  problems show in v1.
- **Fallback threshold** for sparse-velocity-region (when to
  use static Σ instead of conditional). Default: average
  KNN distance > 2× median in training set.
- **Multi-animal extension**: deferred to post-v2.
- **Body-frame fallback** when body shape is near-isotropic
  or body markers are scarce. Current plan: detect at fit
  time, warn, fall back to 2D head-only conditioning if
  fraction of degenerate frames exceeds threshold (default
  20%). Threshold is tunable.

### Decisions committed but worth revisiting if v1 results disappoint

- Per-marker state vs joint state — if the post-hoc spatial
  prior pass proves insufficient to couple markers, joint
  state may be needed.
- Linear vs Extended KF — same principle.
- Triplet-based vs PCA-based spatial prior — if triplet
  selection proves brittle in practice, switching to
  PCA-derived priors (Lightning Pose-style) is a meaningful
  fallback.

---

## 9. Status

**Design committed; no implementation yet beyond Stage 0
(diagnostic, shipped separately).**

Design choices have been made (§4, §8). Mathematical model is
specified (§3). Implementation plan is laid out (§5). The
diagnostic patch (`kalman_smoother_diagnostic`) ships
alongside this doc revision; once Gravio runs it on
representative data, we decide whether to proceed with v1+v2
or to scope back / reconsider Lightning Pose EKS as an
alternative.

This file is a living document. Future work that implements
any of this should reference and extend it rather than
duplicate the design conversation.
