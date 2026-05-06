"""Joint-state Kalman pose smoother for Mufasa.

Stage 1 (v1) of the Kalman smoother build. This module
implements the joint-state forward filter only; the RTS
backward smoother lands in patch 86, the body-triplet
pseudo-measurement prior in patch 87, EM noise fitting in
patch 88, the multi-file orchestrator + CLI in patch 89,
and real-data validation in patch 90.

State and dynamics
------------------

For a project with n markers, the joint state at time t is:

    x_t = [x_1, y_1, vx_1, vy_1,
           x_2, y_2, vx_2, vy_2,
           ...
           x_n, y_n, vx_n, vy_n]

a 4n-dimensional vector. F is block-diagonal with n copies
of the standard constant-velocity block:

    [1, 0, dt, 0]
    [0, 1, 0, dt]
    [0, 0, 1,  0]
    [0, 0, 0,  1]

Q has per-marker process-noise blocks on its diagonal at
this stage; off-diagonal triplet coupling lands in patch 87.

Per-marker process noise (the standard CV-model spec):

    Q_i = [dt^4/4 * q_pos,  0,                dt^3/2 * q_pos,  0]
          [0,                dt^4/4 * q_pos,  0,                dt^3/2 * q_pos]
          [dt^3/2 * q_pos,  0,                dt^2 * q_pos,    0]
          [0,                dt^3/2 * q_pos,  0,                dt^2 * q_pos]

with an additional q_vel term added to the velocity-velocity
block to capture velocity drift independent of position.

Observation model
-----------------

Variable-dimensional per frame. For each marker i with
p_{i,t} >= likelihood_threshold:

    z_{i,t} = [x_{i,t}, y_{i,t}]
    R_{i,t} = (sigma_base_i^2 / p_{i,t}^2) * I_2

with a floor on R to avoid R=0 when p->1 (we use p=0.999 as
the practical maximum for noise scaling). H_i extracts the
two position rows for marker i from the 4n state vector.

The full observation at frame t is the stack of all
high-confidence markers' measurements; if no markers are
high-confidence at frame t, the filter does prediction-only
for that step.

What this stage does NOT do
---------------------------

- No real-data validation (patch 90)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------- #
# State vector layout helpers
# -------------------------------------------------------------------- #

@dataclass(frozen=True)
class StateLayout:
    """Mapping between marker names and joint-state indices.

    For n markers, the state has dimension 4n with each marker
    occupying a contiguous 4-slot block: [x, y, vx, vy].
    """
    markers: Tuple[str, ...]

    @property
    def n_markers(self) -> int:
        return len(self.markers)

    @property
    def state_dim(self) -> int:
        return 4 * len(self.markers)

    def position_indices(self, marker: str) -> Tuple[int, int]:
        """Return (x_idx, y_idx) in the joint state for ``marker``."""
        i = self.markers.index(marker)
        return (4 * i, 4 * i + 1)

    def velocity_indices(self, marker: str) -> Tuple[int, int]:
        """Return (vx_idx, vy_idx) in the joint state for ``marker``."""
        i = self.markers.index(marker)
        return (4 * i + 2, 4 * i + 3)

    def all_position_indices(self) -> np.ndarray:
        """Return shape (n_markers, 2) array: row i is (x_idx, y_idx)
        for the i-th marker, in the same order as ``markers``."""
        out = np.empty((self.n_markers, 2), dtype=np.int64)
        for i in range(self.n_markers):
            out[i, 0] = 4 * i
            out[i, 1] = 4 * i + 1
        return out


# -------------------------------------------------------------------- #
# Noise parameters
# -------------------------------------------------------------------- #

@dataclass(frozen=True)
class NoiseParams:
    """Per-marker noise parameters for the Kalman filter.

    These are passed as fixed inputs in patch 85; patch 88
    will add EM-based fitting of these from data. All values
    are stored in *world* units (typically pixels for x/y and
    pixels-per-second² for the q_pos drive) — the caller is
    responsible for unit consistency.

    Fields
    ------
    sigma_base : Dict[marker_name, float]
        Base observation noise per marker (pixels). Used to
        construct R_i = sigma_base^2 / p_t^2 in the
        likelihood-modulated observation covariance.
    q_pos : Dict[marker_name, float]
        Position-noise drive per marker (pixels²/s²). Couples
        into Q via the standard CV-model formula.
    q_vel : Dict[marker_name, float]
        Velocity-noise drive per marker (pixels²/s⁴).
        Independent velocity-component drift added to Q.
    """
    sigma_base: Dict[str, float]
    q_pos: Dict[str, float]
    q_vel: Dict[str, float]

    def for_layout(self, layout: StateLayout) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Return (sigma_base, q_pos, q_vel) as 1D np.ndarrays
        ordered to match ``layout.markers``. Useful when the
        downstream code wants positional access rather than
        dict lookup."""
        sb = np.array(
            [self.sigma_base[m] for m in layout.markers], dtype=np.float64,
        )
        qp = np.array(
            [self.q_pos[m] for m in layout.markers], dtype=np.float64,
        )
        qv = np.array(
            [self.q_vel[m] for m in layout.markers], dtype=np.float64,
        )
        return sb, qp, qv


# -------------------------------------------------------------------- #
# Triplet prior (joint-state coupling)
# -------------------------------------------------------------------- #

# Default Tikhonov regularization added to triplet covariance
# diagonals to handle rank deficiency. The empirical Σ from
# high-confidence frames can be near-singular if the triplet
# moves rigidly with no rotational variation in the joint
# configuration; a small ridge keeps the matrix invertible
# without distorting the prior in any meaningful way.
_TRIPLET_RIDGE = 1e-6


@dataclass(frozen=True)
class TripletPrior:
    """Per-triplet static prior used as a pseudo-measurement.

    A TripletPrior encodes the empirical *configuration*
    distribution of three markers — their joint (x_a, y_a,
    x_b, y_b, x_c, y_c) configuration is approximated as a
    6-dim Gaussian with mean ``mean_config`` and covariance
    ``cov``. At smoothing time, this is applied as a 6-dim
    pseudo-measurement at every frame: the joint state
    estimate is pulled toward the empirical mean
    configuration with strength inversely proportional to
    ``cov``. This couples the three markers' estimates
    jointly with their per-marker observations.

    Fields
    ------
    markers : Tuple[str, str, str]
        The three markers participating in the triplet, in
        the order their coordinates appear in mean_config and
        cov.
    mean_config : (6,) np.ndarray
        Empirical mean of [x_a, y_a, x_b, y_b, x_c, y_c]
        across high-confidence frames.
    cov : (6, 6) np.ndarray
        Empirical covariance, regularized for invertibility.
        Symmetric positive definite.
    n_samples : int
        Number of high-confidence frames used in the fit.
        Reported for diagnostics; not used at smoothing time.
    state_indices : (6,) np.ndarray
        Indices into the joint state vector for the six
        position slots [x_a, y_a, x_b, y_b, x_c, y_c]. Used
        to construct the 6×(4n) observation matrix H_triplet.
    """
    markers: Tuple[str, str, str]
    mean_config: np.ndarray
    cov: np.ndarray
    n_samples: int
    state_indices: np.ndarray


def fit_triplet_prior(
    positions: np.ndarray,    # (T, n_markers, 2)
    likelihoods: np.ndarray,  # (T, n_markers)
    triplet: Tuple[str, str, str],
    layout: StateLayout,
    likelihood_threshold: float,
    min_samples: int = 200,
    ridge: float = _TRIPLET_RIDGE,
) -> Optional[TripletPrior]:
    """Fit one TripletPrior from data.

    Selects frames where all three markers in ``triplet``
    have likelihood >= ``likelihood_threshold`` AND finite
    positions, then computes the sample mean and covariance
    of the joint 6-dim configuration. Adds ``ridge * I`` to
    the covariance to avoid rank deficiency when the
    configuration moves rigidly with insufficient rotational
    variation in 2D.

    Returns None if fewer than ``min_samples`` joint
    high-confidence frames exist (CV would be statistically
    meaningless; downstream code should skip this triplet).

    Parameters
    ----------
    positions : (T, n_markers, 2)
        Per-frame (x, y) for each marker, in layout order.
    likelihoods : (T, n_markers)
        Per-frame likelihoods, in layout order.
    triplet : (str, str, str)
        Marker names. Must all be in layout.markers.
    layout : StateLayout
    likelihood_threshold : float
        Minimum p for a frame to contribute to the fit.
    min_samples : int
        Minimum number of contributing frames; below this
        the function returns None.
    ridge : float
        Tikhonov regularization scalar added to the diagonal
        of the empirical covariance.

    Returns
    -------
    TripletPrior, or None if insufficient data.
    """
    a, b, c = triplet
    for m in triplet:
        if m not in layout.markers:
            raise ValueError(
                f"Triplet marker {m!r} not in layout markers "
                f"{layout.markers}"
            )

    ia = layout.markers.index(a)
    ib = layout.markers.index(b)
    ic = layout.markers.index(c)

    # Joint high-confidence mask
    above = (
        (likelihoods[:, ia] >= likelihood_threshold)
        & (likelihoods[:, ib] >= likelihood_threshold)
        & (likelihoods[:, ic] >= likelihood_threshold)
    )
    finite = (
        np.all(np.isfinite(positions[:, ia, :]), axis=1)
        & np.all(np.isfinite(positions[:, ib, :]), axis=1)
        & np.all(np.isfinite(positions[:, ic, :]), axis=1)
    )
    mask = above & finite
    n = int(mask.sum())
    if n < min_samples:
        return None

    # Stack the 6-dim configuration vector across selected frames
    config = np.column_stack([
        positions[mask, ia, 0], positions[mask, ia, 1],
        positions[mask, ib, 0], positions[mask, ib, 1],
        positions[mask, ic, 0], positions[mask, ic, 1],
    ])  # shape (n, 6)

    mean_config = config.mean(axis=0)
    # Sample covariance with ddof=1 (unbiased)
    cov = np.cov(config, rowvar=False, ddof=1)
    # Ridge for invertibility
    cov = cov + ridge * np.eye(6)
    # Symmetrize (np.cov should already be symmetric within fp)
    cov = 0.5 * (cov + cov.T)

    state_indices = np.array([
        4 * ia, 4 * ia + 1,
        4 * ib, 4 * ib + 1,
        4 * ic, 4 * ic + 1,
    ], dtype=np.int64)

    return TripletPrior(
        markers=tuple(triplet),
        mean_config=mean_config,
        cov=cov,
        n_samples=n,
        state_indices=state_indices,
    )


def fit_triplet_priors(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    triplets: List[Tuple[str, str, str]],
    layout: StateLayout,
    likelihood_threshold: float,
    min_samples: int = 200,
    ridge: float = _TRIPLET_RIDGE,
) -> List[TripletPrior]:
    """Fit a TripletPrior for each requested triplet.

    Triplets with insufficient joint-high-confidence samples
    are silently dropped from the output (a logged warning
    can be added by the caller if needed).

    Returns a list of TripletPriors in the same order as the
    input triplets, EXCLUDING those that returned None from
    the per-triplet fit.
    """
    out: List[TripletPrior] = []
    for triplet in triplets:
        prior = fit_triplet_prior(
            positions, likelihoods, triplet, layout,
            likelihood_threshold, min_samples, ridge,
        )
        if prior is not None:
            out.append(prior)
    return out


def build_triplet_observation(
    prior: TripletPrior, layout: StateLayout,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (z, H, R) pseudo-measurement triple for one
    triplet at one frame.

    The pseudo-measurement is constant across frames at this
    stage (Stage 1: static Σ): ``z`` is always the empirical
    mean configuration, ``R`` is always the empirical
    covariance. The H matrix picks out the 6 position slots
    for the triplet from the 4n joint state. Stage 2 will
    add posture-conditional Σ for the head triplet but the
    body triplets remain static.

    Returns
    -------
    z : (6,) np.ndarray
        Mean configuration.
    H : (6, 4n) np.ndarray
        Picks out the 6 position slots for the triplet.
    R : (6, 6) np.ndarray
        Empirical covariance.
    """
    z = prior.mean_config.copy()
    H = np.zeros((6, layout.state_dim))
    for k, idx in enumerate(prior.state_indices):
        H[k, idx] = 1.0
    R = prior.cov.copy()
    return z, H, R


# -------------------------------------------------------------------- #
# Transition and process-noise matrices
# -------------------------------------------------------------------- #

def build_F(layout: StateLayout, dt: float) -> np.ndarray:
    """Build the joint-state transition matrix F.

    Block-diagonal with n copies of the standard CV-model
    block. Independent of which markers are observed.

    Parameters
    ----------
    layout : StateLayout
        Marker layout. ``F`` shape will be
        ``(layout.state_dim, layout.state_dim)``.
    dt : float
        Time step (seconds), typically 1/fps.

    Returns
    -------
    F : (4n, 4n) np.ndarray
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive; got {dt}")
    block = np.array([
        [1.0, 0.0,  dt, 0.0],
        [0.0, 1.0, 0.0,  dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    n = layout.n_markers
    F = np.zeros((4 * n, 4 * n))
    for i in range(n):
        F[4 * i:4 * (i + 1), 4 * i:4 * (i + 1)] = block
    return F


def build_Q(
    layout: StateLayout, dt: float, params: NoiseParams,
) -> np.ndarray:
    """Build the joint-state process-noise covariance Q.

    Block-diagonal with one 4×4 block per marker. Each block
    follows the standard CV-model derivation (white-noise
    acceleration q_pos integrated over a step of length dt),
    with an additional q_vel term added to the velocity
    diagonal to model velocity drift independent of position.

    Patch 85 only constructs the diagonal blocks. Off-diagonal
    triplet coupling lands in patch 87.

    Parameters
    ----------
    layout : StateLayout
    dt : float
        Time step (seconds).
    params : NoiseParams
        Provides q_pos and q_vel per marker.

    Returns
    -------
    Q : (4n, 4n) np.ndarray
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive; got {dt}")
    n = layout.n_markers
    Q = np.zeros((4 * n, 4 * n))
    _, q_pos_arr, q_vel_arr = params.for_layout(layout)

    # Standard CV-model block elements (Bar-Shalom et al.,
    # "Estimation with Applications to Tracking and Navigation").
    # For 1D case with white-noise acceleration of intensity q:
    #   [[dt^4/4, dt^3/2],
    #    [dt^3/2, dt^2  ]] * q
    # The 2D version (x and y independent) has this block on
    # both the (x, vx) and (y, vy) sub-blocks.
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    for i in range(n):
        qp = q_pos_arr[i]
        qv = q_vel_arr[i]
        block = np.zeros((4, 4))
        # (x, vx) sub-block driven by q_pos
        block[0, 0] = dt4 / 4.0 * qp
        block[0, 2] = dt3 / 2.0 * qp
        block[2, 0] = dt3 / 2.0 * qp
        block[2, 2] = dt2 * qp + dt2 * qv  # base + extra q_vel drive
        # (y, vy) sub-block — same structure
        block[1, 1] = dt4 / 4.0 * qp
        block[1, 3] = dt3 / 2.0 * qp
        block[3, 1] = dt3 / 2.0 * qp
        block[3, 3] = dt2 * qp + dt2 * qv
        Q[4 * i:4 * (i + 1), 4 * i:4 * (i + 1)] = block

    return Q


# -------------------------------------------------------------------- #
# Per-frame observation construction
# -------------------------------------------------------------------- #

# Floor on per-frame likelihood used for noise scaling, to
# avoid division by very-small p when constructing
# R = sigma^2 / p^2. A p of 0.001 → R inflated by 1e6, which
# is plenty to "ignore" the observation in practice without
# producing infinities. We do NOT skip likelihood-weighted
# observations entirely — that's what the threshold check is
# for.
_P_FLOOR = 1e-3


@dataclass
class FrameObservation:
    """Bundle the variable-dim observation pieces for one frame.

    z : (m,) — observation vector, m = 2 * n_observed_markers
    H : (m, 4n) — observation matrix
    R : (m, m) — observation covariance (block-diagonal,
                  one 2×2 R_i per observed marker)
    n_observed : int — number of markers contributing
    """
    z: np.ndarray
    H: np.ndarray
    R: np.ndarray
    n_observed: int

    @property
    def has_observation(self) -> bool:
        return self.n_observed > 0


def build_observation(
    layout: StateLayout,
    positions: np.ndarray,    # (n_markers, 2)
    likelihoods: np.ndarray,  # (n_markers,)
    sigma_base_arr: np.ndarray,
    likelihood_threshold: float,
    triplet_priors: Optional[List[TripletPrior]] = None,
) -> FrameObservation:
    """Build (z, H, R) for the markers above ``likelihood_threshold``,
    plus optional triplet pseudo-measurements stacked on top.

    Markers below the threshold contribute NO real measurement —
    they are propagated by dynamics (and triplet priors, if any)
    only.

    Triplet priors contribute a 6-dim pseudo-measurement EACH at
    EVERY frame (regardless of whether the triplet's markers are
    observed at this frame). This is what makes the joint-state
    formulation different from a sequential post-hoc correction:
    the triplet prior pulls the joint configuration toward the
    empirical mean even during dropouts, simultaneously with the
    real per-marker observations rather than after them.

    R for each contributing real marker:
        R_i = (sigma_base_i / p_i)^2 * I_2
    with p clamped to [P_FLOOR, 0.999].

    R for each triplet is the prior's covariance (constant
    across frames at this stage; Stage 2 will add posture-
    conditional Σ for the head triplet).

    Parameters
    ----------
    layout : StateLayout
    positions : (n_markers, 2) array of (x, y)
        Order matches ``layout.markers``.
    likelihoods : (n_markers,) array
        Order matches ``layout.markers``.
    sigma_base_arr : (n_markers,) array
        Base observation noise per marker, ordered to match
        layout.
    likelihood_threshold : float
        Markers with p < this value are dropped.
    triplet_priors : list of TripletPrior, optional
        If provided, each contributes a 6-dim pseudo-
        measurement to the stacked observation.

    Returns
    -------
    FrameObservation with combined real + pseudo measurements.
    """
    if positions.shape != (layout.n_markers, 2):
        raise ValueError(
            f"positions shape {positions.shape} != "
            f"({layout.n_markers}, 2)"
        )
    if likelihoods.shape != (layout.n_markers,):
        raise ValueError(
            f"likelihoods shape {likelihoods.shape} != "
            f"({layout.n_markers},)"
        )
    if sigma_base_arr.shape != (layout.n_markers,):
        raise ValueError(
            f"sigma_base_arr shape {sigma_base_arr.shape} != "
            f"({layout.n_markers},)"
        )

    above = likelihoods >= likelihood_threshold
    above &= np.all(np.isfinite(positions), axis=1)
    observed_idx = np.where(above)[0]
    n_obs = int(observed_idx.size)
    n_triplets = len(triplet_priors) if triplet_priors else 0

    # Total observation dimension: 2 per real marker + 6 per triplet
    real_dim = 2 * n_obs
    triplet_dim = 6 * n_triplets
    total_dim = real_dim + triplet_dim

    if total_dim == 0:
        return FrameObservation(
            z=np.zeros(0),
            H=np.zeros((0, layout.state_dim)),
            R=np.zeros((0, 0)),
            n_observed=0,
        )

    z = np.zeros(total_dim)
    H = np.zeros((total_dim, layout.state_dim))
    R = np.zeros((total_dim, total_dim))

    # Real marker observations
    for k, i in enumerate(observed_idx):
        z[2 * k] = positions[i, 0]
        z[2 * k + 1] = positions[i, 1]
        H[2 * k, 4 * i] = 1.0
        H[2 * k + 1, 4 * i + 1] = 1.0
        p = float(np.clip(likelihoods[i], _P_FLOOR, 0.999))
        var = (sigma_base_arr[i] / p) ** 2
        R[2 * k, 2 * k] = var
        R[2 * k + 1, 2 * k + 1] = var

    # Triplet pseudo-measurements stacked after the real markers
    for t_idx, prior in enumerate(triplet_priors or []):
        z_t, H_t, R_t = build_triplet_observation(prior, layout)
        row_offset = real_dim + 6 * t_idx
        z[row_offset:row_offset + 6] = z_t
        H[row_offset:row_offset + 6] = H_t
        R[row_offset:row_offset + 6, row_offset:row_offset + 6] = R_t

    return FrameObservation(z=z, H=H, R=R, n_observed=n_obs)


# -------------------------------------------------------------------- #
# Joint-state forward Kalman filter
# -------------------------------------------------------------------- #

@dataclass
class FilterResult:
    """Output of the forward Kalman filter pass.

    Stores per-frame state mean and covariance, plus the
    one-step-ahead predicted mean and covariance (needed by
    the RTS backward smoother in patch 86).

    Fields
    ------
    x_filt : (T, 4n) array
        Filtered state mean at each frame (after the
        observation update).
    P_filt : (T, 4n, 4n) array
        Filtered state covariance.
    x_pred : (T, 4n) array
        Predicted state mean (before the observation update).
        x_pred[0] is the initial state; x_pred[t>0] is the
        prediction from frame t-1.
    P_pred : (T, 4n, 4n) array
        Predicted state covariance.
    n_observed : (T,) int array
        Number of markers contributing observations per frame.
    """
    x_filt: np.ndarray
    P_filt: np.ndarray
    x_pred: np.ndarray
    P_pred: np.ndarray
    n_observed: np.ndarray


def forward_filter(
    positions: np.ndarray,    # (T, n_markers, 2)
    likelihoods: np.ndarray,  # (T, n_markers)
    layout: StateLayout,
    params: NoiseParams,
    dt: float,
    likelihood_threshold: float,
    initial_state: Optional[np.ndarray] = None,
    initial_cov: Optional[np.ndarray] = None,
    triplet_priors: Optional[List[TripletPrior]] = None,
) -> FilterResult:
    """Run the joint-state Kalman forward filter.

    Variable-dim observations per frame; markers below
    ``likelihood_threshold`` (or with NaN positions) don't
    contribute. If ``triplet_priors`` is provided, each one
    contributes a 6-dim pseudo-measurement at every frame
    that pulls the joint configuration toward its empirical
    mean — the Stage 1 body-triplet prior.

    Predict-update structure with standard Joseph-form
    covariance update for numerical stability.

    Parameters
    ----------
    positions : (T, n_markers, 2)
        Per-frame (x, y) for each marker. NaN positions are
        treated as missing observations.
    likelihoods : (T, n_markers)
        Per-frame likelihoods. Below ``likelihood_threshold``
        the corresponding marker contributes no observation.
    layout : StateLayout
    params : NoiseParams
    dt : float
        Time step (seconds), typically 1/fps.
    likelihood_threshold : float
        Threshold for treating a marker as observed.
    initial_state : (4n,) optional
        Initial state mean. Defaults to zero positions and
        zero velocities; in practice the caller should pass
        an estimate from the first observed frame to avoid
        a long convergence transient.
    initial_cov : (4n, 4n) optional
        Initial state covariance. Defaults to a large
        diagonal (1e6) so the filter quickly forgets the
        initial guess.
    triplet_priors : list of TripletPrior, optional
        Pre-fit body-triplet priors. Each contributes a 6-dim
        pseudo-measurement at every frame, applied jointly
        with real marker observations.

    Returns
    -------
    FilterResult
    """
    if positions.ndim != 3 or positions.shape[1:] != (
        layout.n_markers, 2
    ):
        raise ValueError(
            f"positions shape {positions.shape} != "
            f"(T, {layout.n_markers}, 2)"
        )
    if likelihoods.shape != (positions.shape[0], layout.n_markers):
        raise ValueError(
            f"likelihoods shape {likelihoods.shape} != "
            f"({positions.shape[0]}, {layout.n_markers})"
        )
    T = positions.shape[0]
    d = layout.state_dim

    # Pre-build F, Q (constant across frames in this stage)
    F = build_F(layout, dt)
    Q = build_Q(layout, dt, params)

    sigma_base_arr, _, _ = params.for_layout(layout)

    # Initial state and covariance
    if initial_state is None:
        x = np.zeros(d)
    else:
        if initial_state.shape != (d,):
            raise ValueError(
                f"initial_state shape {initial_state.shape} != ({d},)"
            )
        x = initial_state.astype(np.float64).copy()
    if initial_cov is None:
        P = 1e6 * np.eye(d)
    else:
        if initial_cov.shape != (d, d):
            raise ValueError(
                f"initial_cov shape {initial_cov.shape} != ({d}, {d})"
            )
        P = initial_cov.astype(np.float64).copy()

    # Outputs
    x_pred = np.empty((T, d))
    P_pred = np.empty((T, d, d))
    x_filt = np.empty((T, d))
    P_filt = np.empty((T, d, d))
    n_obs_arr = np.zeros(T, dtype=np.int64)

    I = np.eye(d)

    for t in range(T):
        # Predict step. For t=0 the "prediction" is the initial
        # state — the filter then immediately observes-and-updates.
        if t == 0:
            x_p = x.copy()
            P_p = P.copy()
        else:
            x_p = F @ x_filt[t - 1]
            P_p = F @ P_filt[t - 1] @ F.T + Q
        x_pred[t] = x_p
        P_pred[t] = P_p

        # Observation step (real markers + triplet pseudo-measurements
        # if any). When triplet_priors is non-empty, z always has
        # entries even if no real markers are observed at this frame
        # — that's the whole point of the triplet prior.
        obs = build_observation(
            layout, positions[t], likelihoods[t],
            sigma_base_arr, likelihood_threshold,
            triplet_priors=triplet_priors,
        )
        n_obs_arr[t] = obs.n_observed

        # Predict-only path: no real markers AND no triplet priors
        # (i.e. the stacked observation vector is empty).
        if obs.z.size == 0:
            x_filt[t] = x_p
            P_filt[t] = P_p
            continue

        # Standard Kalman update with Joseph form for the
        # covariance update (numerically stabler than the
        # textbook P = (I - KH)P form for ill-conditioned
        # cases).
        H = obs.H
        R = obs.R
        z = obs.z
        # Innovation
        y = z - H @ x_p
        # Innovation covariance
        S = H @ P_p @ H.T + R
        # Kalman gain
        try:
            K = P_p @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Kalman gain inversion failed at t={t} "
                f"(n_observed={obs.n_observed}): {e}"
            )
        # State update
        x_n = x_p + K @ y
        # Joseph-form covariance update:
        #   P = (I - KH) P (I - KH)^T + K R K^T
        IKH = I - K @ H
        P_n = IKH @ P_p @ IKH.T + K @ R @ K.T
        # Symmetrize to absorb tiny numerical asymmetries
        P_n = 0.5 * (P_n + P_n.T)

        x_filt[t] = x_n
        P_filt[t] = P_n

    return FilterResult(
        x_filt=x_filt,
        P_filt=P_filt,
        x_pred=x_pred,
        P_pred=P_pred,
        n_observed=n_obs_arr,
    )


# -------------------------------------------------------------------- #
# RTS backward smoother
# -------------------------------------------------------------------- #

@dataclass
class SmootherResult:
    """Output of the RTS backward smoother pass.

    Stores per-frame smoothed state mean and covariance.
    Compared to the FilterResult, the smoothed estimates use
    information from FUTURE frames (the backward pass), so
    the smoothed variance is always ≤ the filtered variance
    at every frame.

    Fields
    ------
    x_smooth : (T, 4n) array
        Smoothed state mean at each frame.
    P_smooth : (T, 4n, 4n) array
        Smoothed state covariance.
    P_smooth_lag1 : (T, 4n, 4n) array
        Smoothed lag-one cross-covariance:
            P_smooth_lag1[t] = Cov(x_t, x_{t-1} | z_{1:T})
        For t = 0 (no t=-1 to lag against), this is the zero
        matrix by convention. Required by the Shumway-Stoffer
        M-step's Q estimator. Computed essentially for free
        during the backward pass since it reuses the smoother
        gain C_{t-1}: P_smooth_lag1[t] = (C_{t-1} P_smooth[t])^T.
    n_observed : (T,) int array
        Number of markers contributing observations per frame
        (carried through from the FilterResult so downstream
        consumers don't need both objects).
    """
    x_smooth: np.ndarray
    P_smooth: np.ndarray
    P_smooth_lag1: np.ndarray
    n_observed: np.ndarray


def rts_smoother(
    filter_result: FilterResult,
    layout: StateLayout,
    dt: float,
) -> SmootherResult:
    """Run the Rauch-Tung-Striebel backward smoother.

    Standard RTS recursion: starting from the final filtered
    state (which is also the smoothed state, since there's no
    future to look forward to), propagate backward through the
    trajectory using the smoother gain:

        C_t = P_t F^T (P^-_{t+1})^{-1}
        x^s_t = x_t + C_t (x^s_{t+1} - x^-_{t+1})
        P^s_t = P_t + C_t (P^s_{t+1} - P^-_{t+1}) C_t^T

    where x_t, P_t are the filtered estimates and x^-_{t+1},
    P^-_{t+1} are the one-step-ahead predicted estimates from
    the forward pass.

    Numerical stability: instead of computing (P^-_{t+1})^{-1}
    explicitly, we solve the linear system

        P^-_{t+1} X^T = (P_t F^T)^T

    via np.linalg.solve, which is faster and more stable for
    moderately ill-conditioned predicted covariances.

    The output covariance is symmetrized at each step to
    absorb tiny numerical asymmetries from matrix arithmetic
    (same pattern as forward_filter).

    Parameters
    ----------
    filter_result : FilterResult
        Output from forward_filter — provides the filtered
        ``(x_filt, P_filt)`` and predicted ``(x_pred, P_pred)``
        trajectories needed by the backward pass.
    layout : StateLayout
    dt : float
        Time step (seconds), typically 1/fps. Used to rebuild
        the transition matrix F (kept identical to the forward
        pass — caller MUST pass the same dt that was used in
        forward_filter).

    Returns
    -------
    SmootherResult
    """
    T = filter_result.x_filt.shape[0]
    d = layout.state_dim
    if filter_result.x_filt.shape != (T, d):
        raise ValueError(
            f"filter_result.x_filt shape {filter_result.x_filt.shape} "
            f"!= ({T}, {d})"
        )
    if filter_result.P_filt.shape != (T, d, d):
        raise ValueError(
            f"filter_result.P_filt shape {filter_result.P_filt.shape} "
            f"!= ({T}, {d}, {d})"
        )

    F = build_F(layout, dt)

    x_smooth = np.empty((T, d))
    P_smooth = np.empty((T, d, d))
    P_smooth_lag1 = np.zeros((T, d, d))  # P_smooth_lag1[0] = 0 by convention

    # Boundary condition: smoothed state at t=T-1 IS the
    # filtered state — there's no future to incorporate.
    x_smooth[T - 1] = filter_result.x_filt[T - 1]
    P_smooth[T - 1] = filter_result.P_filt[T - 1]

    # Backward recursion: t from T-2 down to 0. At each step
    # we compute the smoother gain C_t which gives us both
    # x_smooth[t]/P_smooth[t] AND P_smooth_lag1[t+1] (since
    # the lag-one cross-cov at frame t+1 references frame t).
    for t in range(T - 2, -1, -1):
        x_t = filter_result.x_filt[t]
        P_t = filter_result.P_filt[t]
        x_pred_next = filter_result.x_pred[t + 1]
        P_pred_next = filter_result.P_pred[t + 1]

        # Smoother gain: C = P_t F^T (P^-_{t+1})^{-1}
        # Computed via solve: P^-_{t+1} X = F P_t^T  →  X = (P^-)^{-1} F P_t^T
        # Then C = X^T = P_t F^T (P^-)^{-T} = P_t F^T (P^-)^{-1}
        # (P^- is symmetric so (P^-)^{-T} = (P^-)^{-1}.)
        try:
            X = np.linalg.solve(P_pred_next, F @ P_t.T)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"RTS smoother gain solve failed at t={t}: {e}"
            )
        C = X.T

        # Smoothed state update
        x_smooth[t] = x_t + C @ (x_smooth[t + 1] - x_pred_next)

        # Smoothed covariance update
        P_s = P_t + C @ (P_smooth[t + 1] - P_pred_next) @ C.T
        # Symmetrize to absorb numerical asymmetries
        P_smooth[t] = 0.5 * (P_s + P_s.T)

        # Lag-one cross-covariance:
        #   P_smooth_lag1[t+1] = Cov(x_{t+1}, x_t | z_{1:T})
        #                      = (C_t P_smooth[t+1])^T
        #                      = P_smooth[t+1] C_t^T
        # Standard result; see Shumway-Stoffer 1982 eq. (4.3).
        P_smooth_lag1[t + 1] = P_smooth[t + 1] @ C.T

    return SmootherResult(
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        P_smooth_lag1=P_smooth_lag1,
        n_observed=filter_result.n_observed.copy(),
    )


# -------------------------------------------------------------------- #
# EM noise parameter fitting
# -------------------------------------------------------------------- #

# Default convergence tolerance: relative change in sigma_base
# across markers must drop below this for EM to stop early.
_EM_DEFAULT_TOL = 1e-3

# Default maximum EM iterations. Real data typically converges
# in 3-5; the cap prevents runaway in pathological cases.
_EM_DEFAULT_MAX_ITER = 10

# Floor values to keep noise parameters in a sane range across
# iterations. Without these, a degenerate marker (no observations
# above threshold) could collapse sigma_base to zero or negative
# in the M-step.
_EM_FLOOR_SIGMA_BASE = 0.1
_EM_FLOOR_Q_POS = 1e-3
_EM_FLOOR_Q_VEL = 1e-3

# Initial q_pos floor (used during initial_noise_params).
# Corresponds to ~3 px/sec velocity std at 30 fps. Above this
# floor, EM has a non-degenerate starting point and can refine
# downward (or upward) from observations. Below it, the
# Shumway-Stoffer M-step can settle into a self-reinforcing
# zero fixed point on sparse markers (frozen smoothed → tiny
# smoothed-state velocity variance → q_pos confirmed near zero
# → repeat).
_INIT_FLOOR_Q_POS = 100.0

# Hard ceiling on sigma_base updates: cap at this multiple of
# the initial estimate. The initial estimate (5-frame moving-
# average residuals on high-confidence frames) is empirically
# accurate to within ~50% of true noise level on clean data,
# so capping at 3x leaves substantial headroom for legitimate
# growth while preventing the runaway-to-infinity degeneracy
# observed on Gravio's full-data run before patch 91.
_EM_SIGMA_CEILING_FACTOR = 3.0


@dataclass
class EMResult:
    """Output of EM noise parameter fitting.

    Fields
    ------
    params : NoiseParams
        Final fit parameters.
    n_iter : int
        Number of iterations actually run.
    converged : bool
        True if the convergence criterion was met before the
        max iteration cap.
    history : List[Dict[str, float]]
        Per-iteration diagnostic data: max relative change in
        sigma_base, mean log-likelihood (proxy: filtered
        innovation magnitude), per-iteration timestamp.
        Useful for inspecting convergence behavior.
    """
    params: NoiseParams
    n_iter: int
    converged: bool
    history: List[Dict[str, float]] = field(default_factory=list)


def initial_noise_params(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: StateLayout,
    likelihood_threshold: float = 0.5,
    fps: float = 30.0,
) -> NoiseParams:
    """Robust initial noise parameter estimate from data.

    For each marker, looks at high-confidence (p ≥ threshold)
    frames and computes:

    - ``sigma_base``: rough observation noise estimate from
      the residual between raw positions and a 5-frame moving
      average. With insufficient data, falls back to a
      reasonable default.

    - ``q_pos``: estimated from the variance of positions
      within sliding 1-second windows (CV-model formula:
      Var(x in window) ≈ σ²_obs + (1/12) q_pos T_w³). Robust
      against sparse high-confidence coverage because it uses
      ALL observations in each window, not just consecutive
      high-p pairs. A reasonable floor (100, ≈3 px/sec velocity
      std) prevents degenerate-zero initialization that EM
      can't escape from.

    - ``q_vel``: a small fraction of q_pos as a default; EM
      will refine.

    The q_pos initialization fix above is essential for sparse
    markers (e.g., nose at <25% high-p coverage). The previous
    consecutive-high-p-pair estimator with observation-noise
    correction systematically underestimated q_pos for sparse
    markers, sometimes producing near-zero q_pos that became a
    self-reinforcing degenerate fixed point in EM (the smoother
    propagated a frozen trajectory → tiny smoothed-velocity
    variance → Shumway-Stoffer's M-step confirmed q_pos near
    zero → repeat). Window-variance estimation sidesteps this
    by measuring actual positional spread.

    Parameters
    ----------
    positions : (T, n_markers, 2)
    likelihoods : (T, n_markers)
    layout : StateLayout
    likelihood_threshold : float
        Minimum p for a frame to contribute to initial estimate.
    fps : float
        Used to convert window variance to per-second² noise.

    Returns
    -------
    NoiseParams
    """
    dt = 1.0 / fps
    T = positions.shape[0]
    # Window for q_pos estimation: 1 second, but no more than
    # T/4 (so we have at least 4 windows worth of data).
    window_frames = min(int(round(fps)), max(T // 4, 5))
    T_w = window_frames * dt

    sigma_base: Dict[str, float] = {}
    q_pos: Dict[str, float] = {}
    q_vel: Dict[str, float] = {}

    for i, marker in enumerate(layout.markers):
        p = likelihoods[:, i]
        x = positions[:, i, 0]
        y = positions[:, i, 1]
        mask = (p >= likelihood_threshold) & np.isfinite(x) & np.isfinite(y)

        # sigma_base: residuals from a 5-frame moving average
        if mask.sum() >= 20:
            x_clean = x[mask]
            y_clean = y[mask]
            ma_window = 5
            kernel = np.ones(ma_window) / ma_window
            x_ma = np.convolve(x_clean, kernel, mode="valid")
            y_ma = np.convolve(y_clean, kernel, mode="valid")
            x_resid = (
                x_clean[ma_window // 2: ma_window // 2 + len(x_ma)] - x_ma
            )
            y_resid = (
                y_clean[ma_window // 2: ma_window // 2 + len(y_ma)] - y_ma
            )
            sigma_est = float(
                np.sqrt(0.5 * (np.var(x_resid) + np.var(y_resid)))
            )
            sigma_base[marker] = max(sigma_est, _EM_FLOOR_SIGMA_BASE)
        else:
            sigma_base[marker] = 2.0

        # q_pos: window-variance estimator. For each non-overlapping
        # window of `window_frames`, compute variance of high-p
        # positions within. CV model:
        #   Var(x in window) ≈ σ²_obs + (1/12) q_pos * T_w³
        # → q_pos ≈ 12 * (Var(x) - σ²_obs) / T_w³
        # We aggregate across windows by taking the median window
        # variance (robust against outlier windows with bursts).
        n_windows = T // window_frames
        if n_windows >= 4 and mask.sum() >= 20:
            window_vars = []
            for w in range(n_windows):
                start = w * window_frames
                end = start + window_frames
                w_mask = mask[start:end]
                if w_mask.sum() < 5:
                    continue
                # Variance across high-p positions in this window
                wx = x[start:end][w_mask]
                wy = y[start:end][w_mask]
                window_vars.append(0.5 * (np.var(wx) + np.var(wy)))
            if len(window_vars) >= 3:
                # Use mean window variance — this captures motion
                # across all behavioral regimes. Median would be
                # dominated by rest-period windows in datasets
                # where the animal is mostly immobile (typical
                # rodent chamber data: 80%+ rest), giving an
                # underestimate of q_pos for active regimes.
                # Mean weighted by window count gives a balanced
                # estimate that ensures the smoother is calibrated
                # for the motion regimes it will need to track.
                mean_var = float(np.mean(window_vars))
                # Subtract observation-noise contribution
                motion_var = max(mean_var - sigma_base[marker] ** 2, 0.0)
                q_pos_est = 12.0 * motion_var / (T_w ** 3)
                q_pos[marker] = float(max(q_pos_est, _INIT_FLOOR_Q_POS))
            else:
                q_pos[marker] = _INIT_FLOOR_Q_POS
        else:
            q_pos[marker] = _INIT_FLOOR_Q_POS

        # q_vel: small fraction of q_pos by default
        q_vel[marker] = float(max(0.1 * q_pos[marker], _EM_FLOOR_Q_VEL))

    return NoiseParams(
        sigma_base=sigma_base,
        q_pos=q_pos,
        q_vel=q_vel,
    )


# Default number of body-velocity strata for the M-step.
# 4 bins with tail-emphasizing breakpoints (cumulative
# quantiles [0, 0.5, 0.75, 0.875, 1.0]) — slow regime gets
# 50%, fastest gets 12.5%. Inverse-frequency reweighting then
# gives the fast bin ~4x weight, addressing the immobile-bias
# observed on Gravio's full-data run (May 2026, patches 91-93).
_EM_DEFAULT_N_STRATA = 4

# Default low-pass-filter window for the body-velocity signal
# used to stratify the M-step. ~1 second smooths out per-frame
# tracking noise without blurring out behavioral state changes
# (rest → walk transitions take 100s of ms).
_EM_DEFAULT_VELOCITY_LPF_SEC = 1.0


def _compute_velocity_strata(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: StateLayout,
    fps: float,
    n_bins: int = _EM_DEFAULT_N_STRATA,
    lpf_window_sec: float = _EM_DEFAULT_VELOCITY_LPF_SEC,
    body_markers: Optional[List[str]] = None,
    likelihood_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute body-velocity strata for the M-step's stratified
    sampling.

    The Shumway-Stoffer M-step's q_pos estimator is dominated
    by frames where the animal is immobile (typical in rodent
    chamber data: ~80% of frames are sitting). This biases
    q_pos low for fast-moving markers (head, ears) and the
    smoother then can't accelerate fast enough during burst
    motions, producing large smoother-vs-observation deviations.

    Stratifying frames by body velocity and reweighting each
    frame's M-step contribution by the inverse of its bin's
    population gives every velocity regime equal weight,
    addressing the immobile-bias.

    Body centroid: mean of (back1, back2, back3, back4,
    lateral_left, lateral_right, tailbase) where observed.
    Falls back to mean of all observed markers if none of
    those are present in the layout (which would be unusual
    for mufasa data).

    Velocity is smoothed with a centered moving-average filter
    of ``lpf_window_sec`` to reduce per-frame tracking noise
    without blurring behavioral state changes.

    Returns
    -------
    (bin_idx, weights) : (np.ndarray, np.ndarray)
        bin_idx (T,) int — bin assignment per frame in
        [0, n_bins-1]
        weights (T,) float — per-frame weight, normalized so
        the mean weight (across valid frames) is 1.0. Frames
        in a bin with population N_b get weight scaled inversely
        to N_b — bins with fewer frames get higher per-frame
        weight.

    For frames where body velocity can't be computed (no
    observed body markers), bin_idx is set to 0 and weight to
    0.0 — those frames don't contribute to stratified
    statistics.
    """
    T, n_markers, _ = positions.shape

    # Identify body markers from the layout
    default_body = [
        "back1", "back2", "back3", "back4",
        "lateral_left", "lateral_right", "tailbase",
    ]
    if body_markers is None:
        body_markers = [m for m in default_body if m in layout.markers]
    if not body_markers:
        # Fallback: use all markers
        body_markers = list(layout.markers)
    body_idx = np.array(
        [layout.markers.index(m) for m in body_markers], dtype=np.int64,
    )

    # Per-frame body centroid — mean across observed body
    # markers. NaN-safe via the high-confidence + finite mask.
    high_p = (
        (likelihoods[:, body_idx] >= likelihood_threshold)
        & np.all(np.isfinite(positions[:, body_idx, :]), axis=2)
    )  # (T, n_body)
    n_body_obs = high_p.sum(axis=1)  # (T,)

    centroid = np.full((T, 2), np.nan)
    valid_centroid = n_body_obs > 0
    if valid_centroid.any():
        # Mask positions; sum and divide by count
        masked_pos = np.where(
            high_p[..., None],
            positions[:, body_idx, :],
            0.0,
        )
        sum_pos = masked_pos.sum(axis=1)  # (T, 2)
        denom = np.where(n_body_obs > 0, n_body_obs, 1)[..., None]
        centroid[valid_centroid] = (sum_pos / denom)[valid_centroid]

    # Per-frame body velocity (px/sec). Only valid where both
    # this frame's centroid and the previous frame's centroid
    # are valid.
    dt = 1.0 / fps
    velocity = np.zeros(T)
    if T >= 2:
        delta = centroid[1:] - centroid[:-1]  # (T-1, 2)
        speed = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2) / dt
        valid_pair = valid_centroid[1:] & valid_centroid[:-1]
        velocity[1:] = np.where(valid_pair, speed, 0.0)
        velocity[0] = velocity[1] if T > 1 else 0.0

    # Low-pass filter via centered moving average.
    window = max(3, int(round(lpf_window_sec * fps)))
    if window % 2 == 0:
        window += 1  # Make odd for centered filter
    if window > T:
        window = T if T % 2 == 1 else T - 1
        window = max(1, window)

    if window > 1:
        kernel = np.ones(window) / window
        padded = np.concatenate([
            np.full(window // 2, velocity[0]),
            velocity,
            np.full(window // 2, velocity[-1]),
        ])
        velocity_lpf = np.convolve(padded, kernel, mode="valid")
    else:
        velocity_lpf = velocity.copy()

    # Quartile binning. Use frames with valid centroid only
    # for quantile computation.
    valid_for_bins = valid_centroid.copy()
    if valid_for_bins.sum() < n_bins * 10:
        # Too few valid frames to stratify — return uniform.
        bin_idx = np.zeros(T, dtype=np.int64)
        weights = np.ones(T)
        return bin_idx, weights

    # Low-variance fallback. If the velocity distribution is
    # nearly constant (synthetic or pathological data), tail-
    # bin reweighting adds noise without addressing real motion
    # variation. Detect this via the coefficient of variation:
    # if velocity_std < 0.3 * velocity_mean (or velocity_mean is
    # itself near zero), the bins capture only noise — return
    # uniform weights.
    v_valid = velocity_lpf[valid_for_bins]
    v_mean = float(v_valid.mean())
    v_std = float(v_valid.std())
    cv = v_std / max(v_mean, 1e-9)
    if cv < 0.3:
        # Velocities too uniform for stratification to help
        bin_idx = np.zeros(T, dtype=np.int64)
        weights = np.where(valid_for_bins, 1.0, 0.0)
        return bin_idx, weights

    # Tail-emphasizing quantile breakpoints. Equal-population
    # quantiles (np.linspace) give every bin the same weight
    # under inverse-frequency reweighting, which is a no-op.
    # We want to amplify rare fast-motion frames, so we use
    # non-uniform breakpoints that put more bins in the upper
    # tail. Pattern: each successive bin captures half of the
    # remaining frames. For n_bins=4 this gives cumulative
    # quantiles [0, 0.5, 0.75, 0.875, 1.0] → bin populations
    # 50/25/12.5/12.5 → tail bin gets ~4x weight relative to
    # slowest. The last bin always picks up the residual to
    # reach 1.0.
    if n_bins == 1:
        breakpoints = np.array([0.0, 1.0])
    else:
        cum = [0.0]
        remaining = 1.0
        for k in range(n_bins - 1):
            step = remaining * 0.5
            cum.append(cum[-1] + step)
            remaining -= step
        cum.append(1.0)
        breakpoints = np.array(cum)

    edges = np.quantile(velocity_lpf[valid_for_bins], breakpoints)
    # Make sure edges are strictly increasing (defensive
    # against degenerate cases where multiple quantiles
    # coincide because of repeated values, e.g. lots of zeros)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    bin_idx = np.clip(
        np.searchsorted(edges[1:-1], velocity_lpf, side="right"),
        0, n_bins - 1,
    ).astype(np.int64)

    # Per-frame weight: bin with population N_b gets weight
    # (T_valid / n_bins) / N_b, so each bin's TOTAL weight
    # contribution equals (T_valid / n_bins). With
    # tail-emphasizing breakpoints, bins with smaller
    # populations get proportionally larger weights.
    weights = np.zeros(T)
    bin_pops = np.bincount(
        bin_idx[valid_for_bins], minlength=n_bins,
    ).astype(np.float64)
    target_per_bin = float(valid_for_bins.sum()) / n_bins
    bin_weights = np.where(bin_pops > 0, target_per_bin / bin_pops, 0.0)
    weights[valid_for_bins] = bin_weights[bin_idx[valid_for_bins]]

    # Normalize so mean weight (across valid frames) is 1.0
    mean_w = weights[valid_for_bins].mean() if valid_for_bins.any() else 1.0
    if mean_w > 0:
        weights[valid_for_bins] /= mean_w

    return bin_idx, weights


@dataclass
class _MStepStats:
    """Sufficient statistics for the Shumway-Stoffer M-step,
    accumulated across one or more session slices.

    The Shumway-Stoffer (1982) ML estimators for linear
    Gaussian state-space models give:

      Q-hat = (1/(T-1)) * (S11 - S10 F^T - F S10^T + F S00 F^T)

      sigma_i^2-hat = (1/N_i) * sum_{t: i obs} [
          p_t^2 * ((z_{i,t} - H_i x^s_t)^2 - H_i P^s_t H_i^T)
      ]

    where the smoothed sufficient statistics are:
      S00 = sum_{t=0}^{T-2} (P^s_t + x^s_t (x^s_t)^T)
      S11 = sum_{t=1}^{T-1} (P^s_t + x^s_t (x^s_t)^T)
      S10 = sum_{t=1}^{T-1} (P^s_{t,t-1} + x^s_t (x^s_{t-1})^T)

    The H_i P^s_t H_i^T subtraction in the sigma estimator is
    the structural protection against the runaway degeneracy
    observed in patches 88-90's pragmatic estimator: when the
    smoother's posterior variance already explains observed
    residual variance, the contribution to sigma^2 is small
    (clamped at zero rather than growing unbounded).

    The Q estimator uses the smoothed lag-one cross-covariance
    P^s_{t,t-1}, which incorporates the model's belief about
    one-step state evolution and prevents the q_pos collapse
    seen with the velocity-increment estimator.

    Per-session statistics are additive in (S00, S11, S10) and
    in the per-marker sigma terms, so cross-session aggregation
    is straightforward.

    Fields
    ------
    S00, S11, S10 : (4n, 4n) arrays
        Smoothed second-moment accumulators (see formulas
        above). S00 ranges over t = 0..T-2; S11 and S10 range
        over t = 1..T-1. Per-session memory: 4n*4n*8 ~ 28 kB
        for n=15, trivial vs the per-session covariance
        trajectory.
    n_pairs : int
        Number of (t, t-1) pairs accumulated across all
        sessions. Used as the divisor for Q-hat.
    sigma_sq_sum : (n_markers,)
        Per-marker sum of p_t^2 * [(z - Hx^s)^2 - HP^s H^T]+,
        with the [...]_+ clamping at zero.
    sigma_n : (n_markers,) int
        Per-marker count of observation samples (each frame
        with the marker observed contributes 2 samples — x
        and y).
    """
    S00: np.ndarray
    S11: np.ndarray
    S10: np.ndarray
    n_pairs: float
    sigma_sq_sum: np.ndarray
    sigma_n: np.ndarray

    @classmethod
    def zeros(cls, n_markers: int, state_dim: int) -> "_MStepStats":
        return cls(
            S00=np.zeros((state_dim, state_dim)),
            S11=np.zeros((state_dim, state_dim)),
            S10=np.zeros((state_dim, state_dim)),
            n_pairs=0.0,
            sigma_sq_sum=np.zeros(n_markers),
            sigma_n=np.zeros(n_markers, dtype=np.int64),
        )

    def add(self, other: "_MStepStats") -> None:
        self.S00 += other.S00
        self.S11 += other.S11
        self.S10 += other.S10
        self.n_pairs += other.n_pairs
        self.sigma_sq_sum += other.sigma_sq_sum
        self.sigma_n += other.sigma_n


def _accumulate_m_step_stats(
    positions: np.ndarray,    # one session slice (T, n, 2)
    likelihoods: np.ndarray,  # one session slice (T, n)
    smoothed: SmootherResult, # one session's smoother result
    layout: StateLayout,
    likelihood_threshold: float,
    stats: _MStepStats,
    weights: Optional[np.ndarray] = None,
) -> None:
    """Accumulate one session's contribution to the
    Shumway-Stoffer sufficient statistics.

    Operates on a single session slice — within-session frame
    pairs are correctly identified without crossing session
    boundaries. The caller is responsible for slicing the
    inputs to match the smoother result.

    For each frame t:
      x^s_t (x^s_t)^T   — outer product of smoothed mean
      P^s_t             — smoothed covariance (already symmetric)
      P^s_{t,t-1}       — smoothed lag-one cross-covariance

    These accumulate into S00, S11, S10 according to the
    Shumway-Stoffer formulas in the dataclass docstring.

    For the per-marker sigma update, the residual
    (z_{i,t} - H_i x^s_t) and the subtraction term
    H_i P^s_t H_i^T are computed for each marker that's
    observed at this frame.

    Stratified weights
    ------------------

    When ``weights`` is provided (one scalar per frame, mean
    1.0 over valid frames), the Q-hat sufficient statistics
    (S00, S11, S10, n_pairs) are weighted by these per-frame
    factors. This implements stratified estimation: bins with
    fewer frames get higher per-frame weight, balancing the
    M-step across motion regimes and addressing the
    immobile-bias on rodent data.

    The σ_base sufficient statistics (sigma_sq_sum, sigma_n)
    are NOT weighted. σ_base is observation noise — it doesn't
    depend on motion regime — so estimating it from all frames
    equally is correct.

    The lag-one weights for S10 use the average of the two
    frame weights: w_pair[t] = 0.5 * (weights[t] + weights[t-1]).
    """
    x_smooth = smoothed.x_smooth
    P_smooth = smoothed.P_smooth
    P_lag1 = smoothed.P_smooth_lag1
    T = x_smooth.shape[0]
    if T < 2:
        # Need at least 2 frames for the lag-one statistics.
        return

    # Q-hat sufficient statistics (state-only, marker-agnostic).
    if weights is None:
        # Uniform-weight path (faster vectorized form, matches
        # original implementation).
        xx_full = np.einsum("ti,tj->ij", x_smooth, x_smooth)
        xx_first = np.outer(x_smooth[0], x_smooth[0])
        xx_last = np.outer(x_smooth[T - 1], x_smooth[T - 1])
        P_sum = P_smooth.sum(axis=0)

        stats.S00 += (xx_full - xx_last) + (P_sum - P_smooth[T - 1])
        stats.S11 += (xx_full - xx_first) + (P_sum - P_smooth[0])

        cross_xx = np.einsum("ti,tj->ij", x_smooth[1:], x_smooth[:-1])
        P_lag1_sum = P_lag1[1:].sum(axis=0)
        stats.S10 += cross_xx + P_lag1_sum

        stats.n_pairs += T - 1
    else:
        # Weighted path. Frame weights w[t]. Lag-one pairs use
        # w_pair[t] = 0.5 * (w[t] + w[t-1]).
        if weights.shape != (T,):
            raise ValueError(
                f"weights shape {weights.shape} != ({T},)"
            )
        w = weights.astype(np.float64)
        # For S00 (range t=0..T-2)
        w_S00 = w[:-1]
        # For S11 (range t=1..T-1)
        w_S11 = w[1:]
        # For S10 (range t=1..T-1, pairs (t, t-1))
        w_pair = 0.5 * (w[1:] + w[:-1])

        # Weighted outer products: einsum with broadcast of w
        xx_S00 = np.einsum(
            "t,ti,tj->ij", w_S00, x_smooth[:-1], x_smooth[:-1],
        )
        xx_S11 = np.einsum(
            "t,ti,tj->ij", w_S11, x_smooth[1:], x_smooth[1:],
        )
        # Weighted P sums
        P_S00 = np.einsum("t,tij->ij", w_S00, P_smooth[:-1])
        P_S11 = np.einsum("t,tij->ij", w_S11, P_smooth[1:])

        stats.S00 += xx_S00 + P_S00
        stats.S11 += xx_S11 + P_S11

        cross_xx = np.einsum(
            "t,ti,tj->ij", w_pair, x_smooth[1:], x_smooth[:-1],
        )
        P_lag1_w = np.einsum("t,tij->ij", w_pair, P_lag1[1:])
        stats.S10 += cross_xx + P_lag1_w

        # Effective n_pairs: sum of the pair weights. With
        # mean weight 1 over valid frames, this is approximately
        # T-1 minus the contribution from invalid (zero-weight)
        # frames.
        stats.n_pairs += float(w_pair.sum())

    # Per-marker sigma sufficient statistics. NOT weighted —
    # σ_base estimates observation noise, which doesn't depend
    # on motion regime.
    pos_smooth = extract_positions(x_smooth, layout)
    for i in range(layout.n_markers):
        p = likelihoods[:, i]
        x_obs = positions[:, i, 0]
        y_obs = positions[:, i, 1]
        x_sm = pos_smooth[:, i, 0]
        y_sm = pos_smooth[:, i, 1]
        mask = (
            (p >= likelihood_threshold)
            & np.isfinite(x_obs) & np.isfinite(y_obs)
        )
        n_obs = int(mask.sum())
        if n_obs == 0:
            continue

        p_clipped = np.clip(p[mask], _P_FLOOR, 0.999)
        resid_x = x_obs[mask] - x_sm[mask]
        resid_y = y_obs[mask] - y_sm[mask]

        # H_i P^s_t H_i^T for marker i is the (4i, 4i) and
        # (4i+1, 4i+1) diagonal entries of P^s_t — i.e. the
        # x and y position posterior variances.
        var_x_smooth = P_smooth[:, 4 * i, 4 * i][mask]
        var_y_smooth = P_smooth[:, 4 * i + 1, 4 * i + 1][mask]

        # Shumway-Stoffer per-axis innovation contribution:
        #   p_t^2 * [(z_t - x^s_t)^2 - var_smooth]
        # with the [...]_+ clamping at zero (the smoother
        # posterior is already explaining the residual).
        contrib_x = p_clipped ** 2 * np.maximum(
            resid_x ** 2 - var_x_smooth, 0.0,
        )
        contrib_y = p_clipped ** 2 * np.maximum(
            resid_y ** 2 - var_y_smooth, 0.0,
        )
        stats.sigma_sq_sum[i] += float(np.sum(contrib_x) + np.sum(contrib_y))
        # Each frame contributes two samples (x and y residuals)
        stats.sigma_n[i] += 2 * n_obs


def _finalize_m_step(
    stats: _MStepStats,
    layout: StateLayout,
    dt: float,
    prev_params: NoiseParams,
    sigma_ceilings: Optional[Dict[str, float]] = None,
    q_pos_floors: Optional[Dict[str, float]] = None,
) -> NoiseParams:
    """Convert accumulated sufficient statistics into a final
    NoiseParams via the Shumway-Stoffer ML formulas.

    For Q:
      Q-hat = (1 / n_pairs) * (S11 - S10 F^T - F S10^T + F S00 F^T)

    Per marker, q_pos and q_vel are recovered from the
    diagonal elements of Q-hat's per-marker 4x4 block:
      Q[0,0] = Q[1,1] = (dt^4 / 4) * q_pos
      Q[2,2] = Q[3,3] = dt^2 * (q_pos + q_vel)

    For sigma:
      sigma_i^2-hat = sigma_sq_sum[i] / sigma_n[i]
      sigma_base[i] = sqrt(sigma_i^2-hat), clamped to
                      [floor, ceiling] where ceiling is
                      sigma_ceilings[i] if provided.

    Markers with insufficient samples (< 20 in either Q
    pair-count or sigma observation count) keep their
    previous-iteration value.

    Parameters
    ----------
    stats : _MStepStats
        Accumulated sufficient statistics from one or more
        sessions.
    layout : StateLayout
    dt : float
    prev_params : NoiseParams
        Used as fallback for markers with insufficient
        samples.
    sigma_ceilings : dict, optional
        Per-marker ceiling values for sigma_base. If None,
        no ceiling is applied (only the global floor).
    q_pos_floors : dict, optional
        Per-marker floor values for q_pos. If None, only the
        global floor (1e-3) applies. Typically passed by the
        EM loop as ``initial_q_pos / 10`` — prevents the M-step
        from collapsing q_pos to near-zero for sparse markers,
        which would create a self-reinforcing degenerate fixed
        point (frozen smoothed trajectory → tiny smoothed-state
        velocity variance → q_pos confirmed near zero → repeat).

    Returns
    -------
    NoiseParams
    """
    sigma_base: Dict[str, float] = {}
    q_pos: Dict[str, float] = {}
    q_vel: Dict[str, float] = {}

    # Compute Q-hat once (same matrix is used for all marker
    # decompositions). Falls back to previous params if no
    # frame pairs were accumulated.
    if stats.n_pairs >= 20:
        F = build_F(layout, dt)
        # Shumway-Stoffer formula:
        #   Q = (1 / n_pairs) * (S11 - S10 F^T - F S10^T + F S00 F^T)
        Q_hat = (
            stats.S11
            - stats.S10 @ F.T
            - F @ stats.S10.T
            + F @ stats.S00 @ F.T
        ) / stats.n_pairs
        # Symmetrize to absorb fp asymmetries
        Q_hat = 0.5 * (Q_hat + Q_hat.T)
        have_q = True
    else:
        Q_hat = None
        have_q = False

    dt2 = dt * dt
    dt4 = dt2 * dt2

    for i, marker in enumerate(layout.markers):
        # sigma_base
        if stats.sigma_n[i] >= 20:
            sigma_sq = stats.sigma_sq_sum[i] / stats.sigma_n[i]
            sigma_est = float(np.sqrt(max(sigma_sq, 0.0)))
            sigma_est = max(sigma_est, _EM_FLOOR_SIGMA_BASE)
            if sigma_ceilings is not None and marker in sigma_ceilings:
                sigma_est = min(sigma_est, sigma_ceilings[marker])
            sigma_base[marker] = sigma_est
        else:
            sigma_base[marker] = prev_params.sigma_base[marker]

        # q_pos and q_vel from per-marker 4x4 block of Q-hat
        if have_q:
            blk = Q_hat[4 * i:4 * (i + 1), 4 * i:4 * (i + 1)]
            # q_pos from position-position diagonals (avg of x and y)
            q_pos_est = (blk[0, 0] + blk[1, 1]) / 2.0 / (dt4 / 4.0)
            # q_vel from velocity-velocity diagonals minus q_pos
            qv_total = (blk[2, 2] + blk[3, 3]) / 2.0 / dt2
            q_vel_est = qv_total - q_pos_est
            # Apply floors. Per-marker floor (if provided) takes
            # precedence over the global floor.
            marker_q_pos_floor = _EM_FLOOR_Q_POS
            if q_pos_floors is not None and marker in q_pos_floors:
                marker_q_pos_floor = max(
                    marker_q_pos_floor, q_pos_floors[marker],
                )
            q_pos[marker] = float(max(q_pos_est, marker_q_pos_floor))
            q_vel[marker] = float(max(q_vel_est, _EM_FLOOR_Q_VEL))
        else:
            q_pos[marker] = prev_params.q_pos[marker]
            q_vel[marker] = prev_params.q_vel[marker]

    return NoiseParams(sigma_base=sigma_base, q_pos=q_pos, q_vel=q_vel)


def _m_step_update(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    smoothed: SmootherResult,
    layout: StateLayout,
    likelihood_threshold: float,
    dt: float,
    prev_params: NoiseParams,
) -> NoiseParams:
    """Single-session M-step using the Shumway-Stoffer
    estimator. Thin wrapper around the accumulator + finalizer
    that lets external callers (and existing patch-88 tests)
    invoke the M-step directly on full-trajectory inputs.

    Note: callers wanting the sigma ceiling protection should
    use ``fit_noise_params_em`` instead, which threads the
    ceiling through automatically. This wrapper applies only
    the floor.
    """
    stats = _MStepStats.zeros(layout.n_markers, layout.state_dim)
    _accumulate_m_step_stats(
        positions, likelihoods, smoothed, layout,
        likelihood_threshold, stats,
    )
    return _finalize_m_step(stats, layout, dt, prev_params)


def _em_validation_check(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    smoothed: SmootherResult,
    layout: StateLayout,
    likelihood_threshold: float,
    initial_sigma: Dict[str, float],
    sigma_base: Dict[str, float],
    q_pos: Dict[str, float],
    sample_size: int = 1000,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """Sanity-check a single session's smoother output for
    signs of EM degeneracy.

    Three checks (chosen based on the failure mode observed
    on Gravio's full-data run before patch 91):

      1. Trajectory range bound — for each marker, the
         smoothed trajectory range across the sample must
         exceed 0.1 * the raw observation range. Catches
         the "frozen output" symptom where the smoother
         compresses everything to a tiny range.

      2. Smoothed-vs-raw at high-confidence frames — for
         each marker, mean |x_smooth - x_raw| at frames
         where p >= threshold must be < 5 * sigma_base.
         Catches "prior overruling data even at trusted
         observation frames". The bound uses the CURRENT
         M-step's sigma_base, not the pre-EM initial — the
         initial estimator (5-frame MA residuals) is biased
         low on actively-moving markers, and the current σ
         is the smoother's own noise model.

      3. Floor/ceiling fraction — fraction of markers with
         sigma_base AT the floor or ceiling, plus q_pos at
         floor, must be < 50%. Catches "everything
         degenerated to bounds".

    Per-marker statistics are computed for all three checks
    before any check is evaluated. This lets us print a
    unified per-marker breakdown when ``verbose`` is True,
    which is essential for diagnosing failures (the failed
    marker's stats are visible alongside the rest).

    Returns
    -------
    (ok, reason) : (bool, str)
        ok = True if all checks pass; otherwise False with
        a human-readable reason naming the failed check and
        the marker(s) implicated.
    """
    # Sample frames uniformly to keep the check fast even on
    # long sessions.
    T = positions.shape[0]
    if T == 0:
        return True, "empty session, skipped"
    if T <= sample_size:
        idx = np.arange(T)
    else:
        idx = np.linspace(0, T - 1, sample_size, dtype=np.int64)

    pos_smooth = extract_positions(smoothed.x_smooth, layout)
    pos_sample = positions[idx]
    likes_sample = likelihoods[idx]
    smooth_sample = pos_smooth[idx]

    # Phase 1: compute per-marker statistics for all checks.
    # Stored so we can print a unified report below.
    per_marker_stats: List[Dict[str, object]] = []
    for i, marker in enumerate(layout.markers):
        x_obs = pos_sample[:, i, 0]
        y_obs = pos_sample[:, i, 1]
        x_sm = smooth_sample[:, i, 0]
        y_sm = smooth_sample[:, i, 1]
        p = likes_sample[:, i]

        finite_obs = np.isfinite(x_obs) & np.isfinite(y_obs)
        n_finite = int(finite_obs.sum())

        # Range stats (Check 1)
        if n_finite >= 10:
            raw_range = max(
                float(x_obs[finite_obs].max() - x_obs[finite_obs].min()),
                float(y_obs[finite_obs].max() - y_obs[finite_obs].min()),
            )
            smooth_range = max(
                float(x_sm.max() - x_sm.min()),
                float(y_sm.max() - y_sm.min()),
            )
            range_ratio = (
                smooth_range / raw_range if raw_range > 1.0 else 1.0
            )
        else:
            raw_range = 0.0
            smooth_range = 0.0
            range_ratio = 1.0

        # High-p diff stats (Check 2)
        mask = (
            (p >= likelihood_threshold)
            & np.isfinite(x_obs) & np.isfinite(y_obs)
        )
        n_high_p = int(mask.sum())
        if n_high_p >= 10:
            diff = np.sqrt(
                (x_obs[mask] - x_sm[mask]) ** 2
                + (y_obs[mask] - y_sm[mask]) ** 2,
            )
            mean_diff = float(diff.mean())
        else:
            mean_diff = 0.0

        sigma_current = float(sigma_base.get(marker, 5.0))
        diff_bound = 5.0 * sigma_current

        per_marker_stats.append({
            "marker": marker,
            "n_finite": n_finite,
            "raw_range": raw_range,
            "smooth_range": smooth_range,
            "range_ratio": range_ratio,
            "n_high_p": n_high_p,
            "mean_diff": mean_diff,
            "sigma_current": sigma_current,
            "diff_bound": diff_bound,
        })

    # Phase 2: print per-marker breakdown if verbose. Print
    # before evaluating checks so the values are visible even
    # when a check fails (and the function then raises).
    if verbose:
        print(
            "[em-val] per-marker stats (range_ratio>0.1 ok; "
            "mean_diff<5σ ok):"
        )
        for s in per_marker_stats:
            flag_range = (
                "FROZEN" if s["raw_range"] > 1.0 and s["range_ratio"] < 0.1
                else "ok"
            )
            flag_diff = (
                "OVERRULE" if s["mean_diff"] > s["diff_bound"]
                else "ok"
            )
            print(
                f"[em-val]   {s['marker']:<14s}  "
                f"range_ratio={s['range_ratio']:>5.2f}  ({flag_range})  "
                f"mean_diff={s['mean_diff']:>6.2f}px  "
                f"5σ={s['diff_bound']:>6.2f}px  ({flag_diff})  "
                f"n_high_p={s['n_high_p']}"
            )

    # Phase 3: evaluate checks. First failure wins.
    for s in per_marker_stats:
        if s["n_finite"] < 10:
            continue
        if s["raw_range"] > 1.0 and s["range_ratio"] < 0.1:
            return False, (
                f"frozen-output check failed for marker "
                f"{s['marker']!r}: smoothed range = "
                f"{s['smooth_range']:.2f} px, raw range = "
                f"{s['raw_range']:.2f} px (ratio "
                f"{s['range_ratio']:.3f} < 0.1). This indicates "
                f"the smoother has collapsed motion to a tiny "
                f"range — the prior or noise parameters are "
                f"dominating real observations."
            )

    for s in per_marker_stats:
        if s["n_high_p"] < 10:
            continue
        if s["mean_diff"] > s["diff_bound"]:
            return False, (
                f"prior-overruling-data check failed for marker "
                f"{s['marker']!r}: mean |smoothed - raw| at "
                f"high-p frames = {s['mean_diff']:.2f} px, "
                f"exceeds 5x current sigma_base "
                f"({s['diff_bound']:.2f} px). This indicates the "
                f"prior or smoothed dynamics are systematically "
                f"pulling away from trusted observations."
            )

    # Check 3: floor/ceiling fraction
    n_at_bound = 0
    for marker in layout.markers:
        sb = sigma_base[marker]
        sigma_init = initial_sigma.get(marker, 5.0)
        ceiling = _EM_SIGMA_CEILING_FACTOR * sigma_init
        if sb <= 1.01 * _EM_FLOOR_SIGMA_BASE or sb >= 0.99 * ceiling:
            n_at_bound += 1
        if q_pos[marker] <= 1.01 * _EM_FLOOR_Q_POS:
            n_at_bound += 1
    n_total = 2 * len(layout.markers)  # sigma + q_pos per marker
    if n_at_bound > 0.5 * n_total:
        return False, (
            f"floor/ceiling-fraction check failed: "
            f"{n_at_bound}/{n_total} parameters "
            f"({n_at_bound / n_total:.0%}) are at their floor or "
            f"ceiling. This indicates pervasive EM degeneracy."
        )

    return True, "ok"


def fit_noise_params_em(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: StateLayout,
    fps: float,
    likelihood_threshold: float,
    triplet_priors: Optional[List[TripletPrior]] = None,
    initial_params: Optional[NoiseParams] = None,
    max_iter: int = _EM_DEFAULT_MAX_ITER,
    tol: float = _EM_DEFAULT_TOL,
    verbose: bool = False,
    sessions: Optional[List[Tuple[str, int, int]]] = None,
    stratify: bool = True,
    n_strata: int = _EM_DEFAULT_N_STRATA,
) -> EMResult:
    """Fit per-marker noise parameters via EM iterations using
    the Shumway-Stoffer (1982) closed-form ML estimators.

    EM loop:
        1. Initialize params (passed-in or auto from data) and
           lock per-marker sigma ceilings at
           ``_EM_SIGMA_CEILING_FACTOR * initial_sigma``.
        2. E-step: forward filter + RTS PER SESSION; M-step
           sufficient statistics accumulated across sessions.
        3. M-step: finalize one global NoiseParams from the
           accumulated stats using the closed-form
           Shumway-Stoffer estimators (Q from smoothed lag-one
           cross-covariances; sigma from
           innovation-corrected residuals).
        4. Validation hook: sanity-check the smoothed output
           on the first session. If any check fails, raise
           RuntimeError with a clear message naming the failed
           bound. This aborts the whole EM run by design —
           fail fast rather than silently produce bad output.
        5. Convergence check: max relative change in sigma_base
           across markers < tol.
        6. Repeat to max_iter.

    Why Shumway-Stoffer (and not the patches 88-90 pragmatic
    estimator)
    -------------------------------------------------------

    The pragmatic residual-based estimator shipped in patches
    88-90 fails on real data with mostly-immobile sequences
    and dropout-heavy markers (Gravio's full-data run, May
    2026): q_pos collapses to its floor, then sigma blows up
    to compensate for the resulting too-rigid trajectories.
    Output is unusably "frozen" (smoothed range << raw range,
    smoothed-raw differences >> initial sigma even at
    high-confidence frames).

    The Shumway-Stoffer estimator has structural protections:

      - The sigma update subtracts H P^s H^T from the residual
        variance — when the smoother is uncertain (large
        posterior), the apparent residual is corrected, so
        runaway is bounded.

      - The Q update uses smoothed lag-one cross-covariances
        which capture the model's belief about one-step state
        evolution, not just point-estimate velocity increments.
        This prevents the q_pos collapse seen on immobile data.

    A hard ceiling on sigma_base (3x the initial estimate)
    gives belt-and-suspenders protection against any residual
    runaway.

    Memory and correctness
    ----------------------

    When ``sessions`` is provided, the E-step runs forward+RTS
    independently per session — bounding peak memory at one
    session's covariance trajectory size (4n * 4n * T_session)
    rather than the full concatenated dataset's size.

    Per-session E-step also avoids cross-session-boundary
    contamination of the M-step statistics.

    Triplet priors (if provided) are applied at every E-step
    iteration but are NOT updated themselves — their Σ_abc
    is fit once from the data and held constant.

    Note that since patch 91, ``smooth_pose`` defaults to
    NOT using triplet priors (the static rigid-body assumption
    causes systematic bias in posture-variable real-data
    behavior). The triplet machinery remains available as
    explicit API but is no longer the default path. See the
    patch 91 commit message for the empirical rationale.

    Parameters
    ----------
    positions : (T, n_markers, 2)
    likelihoods : (T, n_markers)
    layout : StateLayout
    fps : float
    likelihood_threshold : float
    triplet_priors : list of TripletPrior, optional
        Pass None (the default) for pure per-marker temporal
        smoothing.
    initial_params : NoiseParams, optional
        If None, ``initial_noise_params`` is called with
        ``likelihood_threshold * 0.5`` (a more permissive
        threshold for initialization).
    max_iter : int
    tol : float
        Convergence tolerance on max-relative-change-in-sigma.
    verbose : bool
        If True, print per-iteration parameter summaries
        including per-marker sigma values (not just the
        average).
    sessions : list of (name, start_idx, end_idx), optional
        If provided, the E-step runs per session with covariance
        trajectories scoped to each session. If None, the entire
        input is treated as a single session.

    Raises
    ------
    RuntimeError
        If the validation hook detects degeneracy in the
        smoothed output. The error message names the failed
        check and implicated marker(s). This aborts the whole
        EM run rather than reverting (per design choice in
        patch 91 review — fail fast on bad data is more
        informative than silent recovery).

    Returns
    -------
    EMResult
    """
    dt = 1.0 / fps
    if initial_params is None:
        # Use a more permissive threshold for initial estimation
        # — we want enough samples even if the smoother proper
        # uses a stricter threshold.
        init_threshold = max(0.3, likelihood_threshold * 0.5)
        params = initial_noise_params(
            positions, likelihoods, layout,
            likelihood_threshold=init_threshold,
            fps=fps,
        )
    else:
        params = initial_params

    # Lock in the initial sigma estimates for use as ceiling
    # references and for the validation hook. These are
    # computed from data BEFORE EM starts and remain fixed
    # across iterations — they're a proxy for "the true noise
    # level we expect."
    initial_sigma = dict(params.sigma_base)
    initial_q_pos = dict(params.q_pos)
    sigma_ceilings = {
        m: _EM_SIGMA_CEILING_FACTOR * initial_sigma[m]
        for m in layout.markers
    }
    # Per-marker q_pos floors at 1/10 of initial estimate.
    # Prevents M-step from collapsing q_pos to near-zero for
    # sparse markers, which would create a self-reinforcing
    # degenerate fixed point in EM (frozen smoothed → tiny
    # smoothed-state velocity variance → q_pos confirmed near
    # zero → repeat). 1/10 leaves substantial room for EM to
    # legitimately reduce q_pos from a generous initial estimate
    # while preventing collapse.
    q_pos_floors = {
        m: max(initial_q_pos[m] / 10.0, _EM_FLOOR_Q_POS)
        for m in layout.markers
    }

    # Default to one synthetic session covering all frames, for
    # backward compat with single-session callers.
    if sessions is None:
        sessions = [("__all__", 0, positions.shape[0])]

    # Compute per-frame stratification weights once before the
    # iteration loop. The bins/weights are based on body
    # velocity computed from raw observations, so they don't
    # depend on the current EM state and are fixed across
    # iterations.
    if stratify:
        bin_idx, frame_weights = _compute_velocity_strata(
            positions, likelihoods, layout, fps,
            n_bins=n_strata,
            likelihood_threshold=max(0.3, likelihood_threshold * 0.5),
        )
        if verbose:
            valid_mask = frame_weights > 0
            n_valid = int(valid_mask.sum())
            if n_valid >= n_strata * 10:
                # Report bin populations
                bin_pops = np.bincount(bin_idx[valid_mask], minlength=n_strata)
                bin_w_means = np.array([
                    frame_weights[valid_mask & (bin_idx == b)].mean()
                    if (valid_mask & (bin_idx == b)).any() else 0.0
                    for b in range(n_strata)
                ])
                print(
                    f"[em] stratification ON: {n_strata} body-velocity "
                    f"bins, {n_valid} valid frames"
                )
                for b in range(n_strata):
                    print(
                        f"[em]   bin {b}: pop={bin_pops[b]:>8d}  "
                        f"weight={bin_w_means[b]:.3f}"
                    )
            else:
                print(
                    f"[em] stratification: insufficient valid frames "
                    f"({n_valid}); falling back to uniform weights"
                )
    else:
        frame_weights = None
        if verbose:
            print("[em] stratification OFF — uniform M-step weights")

    history: List[Dict[str, float]] = []
    converged = False
    n_iter = 0

    for it in range(max_iter):
        n_iter = it + 1

        # E-step: forward+RTS PER SESSION; M-step sufficient
        # statistics accumulated across sessions. Save the
        # FIRST session's smoother output for the post-M-step
        # validation hook (cheap, and that session is
        # representative of the rest if anything's going wrong).
        stats = _MStepStats.zeros(layout.n_markers, layout.state_dim)
        validation_smoothed: Optional[SmootherResult] = None
        validation_pos: Optional[np.ndarray] = None
        validation_likes: Optional[np.ndarray] = None

        for sess_idx, (sess_name, start, end) in enumerate(sessions):
            sub_pos = positions[start:end]
            sub_likes = likelihoods[start:end]
            sub_weights = (
                frame_weights[start:end] if frame_weights is not None else None
            )
            filt = forward_filter(
                sub_pos, sub_likes, layout, params, dt,
                likelihood_threshold=likelihood_threshold,
                triplet_priors=triplet_priors,
            )
            smoothed = rts_smoother(filt, layout, dt)
            _accumulate_m_step_stats(
                sub_pos, sub_likes, smoothed, layout,
                likelihood_threshold, stats,
                weights=sub_weights,
            )
            if sess_idx == 0:
                validation_smoothed = smoothed
                validation_pos = sub_pos
                validation_likes = sub_likes
                # Don't delete smoothed yet — we need it for
                # the validation check after the M-step finalizes.
                del filt
            else:
                # Free the per-session covariance trajectories
                # before moving on. CPython will collect these
                # on next gc cycle anyway, but for very long
                # sessions we want the memory back NOW.
                del filt, smoothed

        # M-step: finalize from accumulated stats with sigma
        # ceiling protection.
        new_params = _finalize_m_step(
            stats, layout, dt, params,
            sigma_ceilings=sigma_ceilings,
            q_pos_floors=q_pos_floors,
        )

        # Convergence: max relative change in sigma_base across markers
        rel_changes = []
        for m in layout.markers:
            old = params.sigma_base[m]
            new = new_params.sigma_base[m]
            rel_changes.append(abs(new - old) / max(old, 1e-9))
        max_rel_change = float(max(rel_changes))

        history.append({
            "iter": float(it),
            "max_rel_change_sigma": max_rel_change,
            "mean_sigma": float(np.mean(list(new_params.sigma_base.values()))),
            "mean_q_pos": float(np.mean(list(new_params.q_pos.values()))),
        })

        if verbose:
            print(
                f"[em] iter {it}: "
                f"max Δσ/σ = {max_rel_change:.4e}, "
                f"⟨σ⟩ = {history[-1]['mean_sigma']:.3f}px, "
                f"⟨q_pos⟩ = {history[-1]['mean_q_pos']:.3f}"
            )
            # Per-marker breakdown — useful for spotting which
            # markers are dominating the average.
            for m in layout.markers:
                print(
                    f"[em]   {m:<14s}  σ={new_params.sigma_base[m]:>7.2f}px"
                    f"  q_pos={new_params.q_pos[m]:>10.3f}"
                    f"  q_vel={new_params.q_vel[m]:>10.3f}"
                )

        # Validation hook: check the first session's smoothed
        # output for signs of degeneracy. Crashes the run with
        # a clear error message if any check fails — by design
        # we fail fast rather than silently produce bad output
        # (failure mode observed on Gravio's full-data run
        # before patch 91). Logging runs FIRST so the per-marker
        # parameter values are visible before the crash, which
        # is essential for diagnosing the failure.
        if validation_smoothed is not None:
            ok, reason = _em_validation_check(
                validation_pos, validation_likes,
                validation_smoothed, layout,
                likelihood_threshold,
                initial_sigma=initial_sigma,
                sigma_base=new_params.sigma_base,
                q_pos=new_params.q_pos,
                verbose=verbose,
            )
            del validation_smoothed, validation_pos, validation_likes
            if not ok:
                raise RuntimeError(
                    f"EM validation hook triggered at iteration "
                    f"{it}: {reason}"
                )

        params = new_params

        if max_rel_change < tol:
            converged = True
            break

    return EMResult(
        params=params,
        n_iter=n_iter,
        converged=converged,
        history=history,
    )


# -------------------------------------------------------------------- #
# Convenience: extract per-marker positions from the joint state
# -------------------------------------------------------------------- #

def extract_positions(
    state_trajectory: np.ndarray,  # (T, 4n)
    layout: StateLayout,
) -> np.ndarray:
    """Extract per-marker positions (T, n_markers, 2) from a
    joint-state trajectory."""
    if state_trajectory.ndim != 2 or state_trajectory.shape[1] != layout.state_dim:
        raise ValueError(
            f"state_trajectory shape {state_trajectory.shape} != "
            f"(T, {layout.state_dim})"
        )
    T = state_trajectory.shape[0]
    out = np.empty((T, layout.n_markers, 2))
    for i in range(layout.n_markers):
        out[:, i, 0] = state_trajectory[:, 4 * i]
        out[:, i, 1] = state_trajectory[:, 4 * i + 1]
    return out


def extract_position_variances(
    cov_trajectory: np.ndarray,    # (T, 4n, 4n)
    layout: StateLayout,
) -> np.ndarray:
    """Extract per-marker position variances (T, n_markers, 2)
    from the diagonal of the joint-state covariance trajectory.

    Returns the diagonal (var_x, var_y) per marker per frame.
    Off-diagonal coupling between markers is discarded — the
    caller wanting full uncertainty information should keep
    the original covariance trajectory.
    """
    if cov_trajectory.ndim != 3 or cov_trajectory.shape[1:] != (
        layout.state_dim, layout.state_dim
    ):
        raise ValueError(
            f"cov_trajectory shape {cov_trajectory.shape} != "
            f"(T, {layout.state_dim}, {layout.state_dim})"
        )
    T = cov_trajectory.shape[0]
    out = np.empty((T, layout.n_markers, 2))
    for i in range(layout.n_markers):
        out[:, i, 0] = cov_trajectory[:, 4 * i, 4 * i]
        out[:, i, 1] = cov_trajectory[:, 4 * i + 1, 4 * i + 1]
    return out


# -------------------------------------------------------------------- #
# Multi-file ingestion helpers
# -------------------------------------------------------------------- #
#
# Reuses the existing diagnostic loader rather than duplicating
# the logic. The diagnostic produces a flat-column DataFrame
# with markers detected from the column-name pattern. The
# smoother needs the same shape — (T, n_markers, 2) for
# positions and (T, n_markers) for likelihoods — plus the list
# of session boundaries so it can reset the joint-state
# covariance at session transitions.

# Module version, written into the saved-model artifact. Bump
# when the saved-model format changes incompatibly.
SMOOTHER_MODEL_VERSION = "1.0"


def _df_to_arrays(
    df: pd.DataFrame, markers: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a flat-column pose DataFrame into (positions,
    likelihoods) arrays of shape (T, n_markers, 2) and
    (T, n_markers) respectively.

    Column names are matched lowercased; ``markers`` should
    already be lowercase.
    """
    T = len(df)
    n = len(markers)
    positions = np.empty((T, n, 2), dtype=np.float64)
    likelihoods = np.empty((T, n), dtype=np.float64)
    for i, m in enumerate(markers):
        positions[:, i, 0] = df[f"{m}_x"].values
        positions[:, i, 1] = df[f"{m}_y"].values
        likelihoods[:, i] = df[f"{m}_p"].values
    return positions, likelihoods


def _arrays_to_df(
    positions: np.ndarray,
    variances: np.ndarray,
    likelihoods: np.ndarray,
    markers: List[str],
) -> pd.DataFrame:
    """Convert smoothed (T, n, 2) positions + variances + raw
    likelihoods into a flat-column DataFrame for output.

    Column layout (per marker m):
        m_x        — smoothed x position
        m_y        — smoothed y position
        m_p        — original likelihood (preserved for
                     downstream consumers like the rearing HMM)
        m_var_x    — smoothed x variance (uncertainty channel)
        m_var_y    — smoothed y variance
    """
    T = positions.shape[0]
    cols: Dict[str, np.ndarray] = {}
    for i, m in enumerate(markers):
        cols[f"{m}_x"] = positions[:, i, 0]
        cols[f"{m}_y"] = positions[:, i, 1]
        cols[f"{m}_p"] = likelihoods[:, i]
        cols[f"{m}_var_x"] = variances[:, i, 0]
        cols[f"{m}_var_y"] = variances[:, i, 1]
    return pd.DataFrame(cols, index=np.arange(T))


def _hash_data(
    positions: np.ndarray, likelihoods: np.ndarray,
) -> str:
    """Compute a content hash of the input pose data, used in
    the saved-model artifact for provenance tracking.
    Includes shape and a sample-based digest (full-array
    hashing would be slow for 3.6M-frame datasets)."""
    h = hashlib.sha256()
    h.update(str(positions.shape).encode())
    h.update(str(likelihoods.shape).encode())
    # Sample 1000 evenly-spaced rows for the digest — full
    # array would be slow for large inputs and the goal is
    # provenance, not cryptographic uniqueness.
    n_sample = min(1000, positions.shape[0])
    idx = np.linspace(0, positions.shape[0] - 1, n_sample, dtype=int)
    h.update(positions[idx].tobytes())
    h.update(likelihoods[idx].tobytes())
    return h.hexdigest()[:16]  # truncate to 16 chars


# -------------------------------------------------------------------- #
# Save/load model artifact
# -------------------------------------------------------------------- #

def save_model(
    path: str,
    layout: StateLayout,
    params: NoiseParams,
    triplet_priors: List[TripletPrior],
    likelihood_threshold: float,
    fps: float,
    data_hash: str,
    em_history: Optional[List[Dict[str, float]]] = None,
) -> None:
    """Save a fit smoother model to a .npz file.

    Persisted fields:
      - version: SMOOTHER_MODEL_VERSION (string)
      - markers: layout's marker tuple (numpy unicode array)
      - sigma_base, q_pos, q_vel: per-marker noise params,
        as 1D arrays in marker order
      - triplet_count: int (so load can iterate)
      - triplet_<i>_markers, _mean, _cov, _n_samples,
        _state_indices: one set per triplet
      - likelihood_threshold, fps: scalar floats
      - data_hash: 16-char content hash of the source data
      - em_history (if provided): JSON-encoded list of dicts

    Loaded by ``load_model``.
    """
    sb_arr, qp_arr, qv_arr = params.for_layout(layout)
    payload = {
        "version": SMOOTHER_MODEL_VERSION,
        "markers": np.array(layout.markers, dtype=np.str_),
        "sigma_base": sb_arr,
        "q_pos": qp_arr,
        "q_vel": qv_arr,
        "likelihood_threshold": float(likelihood_threshold),
        "fps": float(fps),
        "data_hash": str(data_hash),
        "triplet_count": np.int64(len(triplet_priors)),
    }
    for i, tp in enumerate(triplet_priors):
        payload[f"triplet_{i}_markers"] = np.array(tp.markers, dtype=np.str_)
        payload[f"triplet_{i}_mean"] = tp.mean_config
        payload[f"triplet_{i}_cov"] = tp.cov
        payload[f"triplet_{i}_n_samples"] = np.int64(tp.n_samples)
        payload[f"triplet_{i}_state_indices"] = tp.state_indices
    if em_history is not None:
        payload["em_history"] = json.dumps(em_history)

    np.savez_compressed(path, **payload)


def load_model(
    path: str,
) -> Tuple[
    StateLayout, NoiseParams, List[TripletPrior],
    float, float, str, Optional[List[Dict[str, float]]],
]:
    """Load a fit smoother model from a .npz file.

    Returns
    -------
    (layout, params, triplet_priors, likelihood_threshold,
    fps, data_hash, em_history)
    """
    data = np.load(path, allow_pickle=False)
    version = str(data["version"])
    if version != SMOOTHER_MODEL_VERSION:
        raise ValueError(
            f"Saved model version {version!r} != current "
            f"{SMOOTHER_MODEL_VERSION!r}. Cross-version loading "
            f"is not supported."
        )
    markers = tuple(str(m) for m in data["markers"])
    layout = StateLayout(markers=markers)
    sb_arr = data["sigma_base"]
    qp_arr = data["q_pos"]
    qv_arr = data["q_vel"]
    params = NoiseParams(
        sigma_base={m: float(sb_arr[i]) for i, m in enumerate(markers)},
        q_pos={m: float(qp_arr[i]) for i, m in enumerate(markers)},
        q_vel={m: float(qv_arr[i]) for i, m in enumerate(markers)},
    )
    n_triplets = int(data["triplet_count"])
    triplet_priors: List[TripletPrior] = []
    for i in range(n_triplets):
        tp_markers = tuple(
            str(m) for m in data[f"triplet_{i}_markers"]
        )
        triplet_priors.append(TripletPrior(
            markers=tp_markers,
            mean_config=data[f"triplet_{i}_mean"],
            cov=data[f"triplet_{i}_cov"],
            n_samples=int(data[f"triplet_{i}_n_samples"]),
            state_indices=data[f"triplet_{i}_state_indices"],
        ))
    likelihood_threshold = float(data["likelihood_threshold"])
    fps = float(data["fps"])
    data_hash = str(data["data_hash"])
    em_history = None
    if "em_history" in data.files:
        em_history = json.loads(str(data["em_history"]))
    return (
        layout, params, triplet_priors,
        likelihood_threshold, fps, data_hash, em_history,
    )


# -------------------------------------------------------------------- #
# Multi-session smoothing with boundary resets
# -------------------------------------------------------------------- #

def smooth_multi_session(
    positions: np.ndarray,
    likelihoods: np.ndarray,
    layout: StateLayout,
    params: NoiseParams,
    triplet_priors: List[TripletPrior],
    sessions: List[Tuple[str, int, int]],
    fps: float,
    likelihood_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run forward+RTS smoothing per session with boundary resets.

    Each session is filtered + smoothed independently; the
    joint-state covariance is reset at session boundaries so
    the last frame of session N doesn't anchor the first frame
    of session N+1. The triplet priors are GLOBAL — same Σ
    used in every session, since that's the v1 commitment.

    Parameters
    ----------
    positions, likelihoods : full concatenated arrays
    sessions : list of (name, start_idx, end_idx) tuples
        Half-open ranges into the concatenated data for each
        session.

    Returns
    -------
    smoothed_positions : (T, n_markers, 2)
    smoothed_variances : (T, n_markers, 2)
        Diagonal of the smoothed covariance, per-marker.
    n_observed : (T,) int array
        Per-frame count of observed (high-confidence) markers.
    """
    dt = 1.0 / fps
    T = positions.shape[0]
    smoothed_positions = np.empty((T, layout.n_markers, 2))
    smoothed_variances = np.empty((T, layout.n_markers, 2))
    n_obs_arr = np.zeros(T, dtype=np.int64)

    for name, start, end in sessions:
        sub_pos = positions[start:end]
        sub_likes = likelihoods[start:end]
        # Each session starts fresh — no cross-session anchoring.
        # Initial state guess: zeros (filter quickly recovers).
        # Initial covariance: large diagonal (uninformative prior).
        filt = forward_filter(
            sub_pos, sub_likes, layout, params, dt,
            likelihood_threshold=likelihood_threshold,
            triplet_priors=triplet_priors or None,
        )
        smoothed = rts_smoother(filt, layout, dt)
        sub_pos_smooth = extract_positions(smoothed.x_smooth, layout)
        sub_var_smooth = extract_position_variances(
            smoothed.P_smooth, layout,
        )
        smoothed_positions[start:end] = sub_pos_smooth
        smoothed_variances[start:end] = sub_var_smooth
        n_obs_arr[start:end] = smoothed.n_observed

    return smoothed_positions, smoothed_variances, n_obs_arr


# -------------------------------------------------------------------- #
# Top-level orchestrator
# -------------------------------------------------------------------- #

def smooth_pose(
    pose_input,  # str | List[str] | Path — single file, list, or directory
    output_dir: str,
    fps: float = 30.0,
    likelihood_threshold: float = 0.7,
    head_markers: Optional[List[str]] = None,
    candidate_triplets: Optional[List[Tuple[str, str, str]]] = None,
    rigid_cv_threshold: float = 0.20,
    rigid_max_pairs: int = 8,
    em_max_iter: int = 10,
    em_tol: float = 1e-3,
    load_model_path: Optional[str] = None,
    save_model_path: Optional[str] = None,
    use_triplets: bool = False,
    stratify: bool = True,
    n_strata: int = _EM_DEFAULT_N_STRATA,
    verbose: bool = False,
) -> dict:
    """Top-level smoother pipeline.

    Loads pose data (single file / dir / list — same shapes as
    the diagnostic), fits noise params (and optionally triplet
    priors), runs forward+RTS smoothing per session, and
    writes one smoothed parquet per input session into
    ``output_dir``.

    Triplet priors and the rigid-body assumption
    --------------------------------------------

    Default ``use_triplets=False``: the v1 smoother runs as
    pure per-marker temporal smoothing (joint state, no
    spatial pseudo-measurements). This is the recommended
    default since patch 91 — the static rigid-body triplet
    prior was found to systematically bias smoothed output
    on posture-variable real-data behavior, producing
    "frozen" trajectories that contradict observations even
    at high-confidence frames.

    Setting ``use_triplets=True`` re-enables the triplet
    auto-detection + fitting pipeline from patches 87-90.
    Use only if you have data where the static-rigid
    assumption is genuinely justified (animal in fixed
    posture, e.g. a head-fixed prep), or for direct
    comparison with the patch-90 behavior. The triplet API
    (``TripletPrior``, ``fit_triplet_prior``,
    ``fit_triplet_priors``, ``build_triplet_observation``)
    remains available regardless of this flag for direct
    use by callers who want fine-grained control.

    Parameters mirror the diagnostic where applicable.

    Returns
    -------
    dict with keys:
        - ``layout``: StateLayout
        - ``params``: NoiseParams (final)
        - ``triplet_priors``: List[TripletPrior] (empty if
          use_triplets=False)
        - ``sessions``: list of (name, start_idx, end_idx)
        - ``output_files``: List[str], one per session
        - ``model_artifact``: str | None, path if save_model_path given
        - ``em_history``: List[Dict] | None
    """
    # Lazy import of the diagnostic's loaders to avoid pulling
    # matplotlib (which the diagnostic uses for plots) into the
    # smoother's dependency chain unnecessarily. The triplet
    # auto-detector is also imported lazily only if needed.
    from mufasa.data_processors.kalman_diagnostic import (
        discover_pose_files,
        load_pose_files,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve input(s) into a list of file paths
    if isinstance(pose_input, (list, tuple)):
        paths = [str(p) for p in pose_input]
    elif isinstance(pose_input, (str, Path)):
        path_obj = Path(pose_input)
        if path_obj.is_dir():
            paths = discover_pose_files(str(path_obj))
            if not paths:
                raise ValueError(
                    f"No pose files found under {pose_input}"
                )
        else:
            paths = [str(path_obj)]
    else:
        raise TypeError(
            f"pose_input must be str, Path, or list; got {type(pose_input)}"
        )

    # Load all sessions
    df, markers, sessions = load_pose_files(paths)
    if verbose:
        print(
            f"[smoother] Loaded {len(df)} frames × {len(markers)} markers "
            f"from {len(paths)} file(s)"
        )

    layout = StateLayout(markers=tuple(markers))
    positions, likelihoods = _df_to_arrays(df, markers)
    data_hash = _hash_data(positions, likelihoods)

    # Two paths: load from model OR fit fresh
    em_history = None
    if load_model_path is not None:
        if verbose:
            print(f"[smoother] Loading model from {load_model_path}")
        (loaded_layout, params, triplet_priors,
         loaded_threshold, loaded_fps, loaded_hash,
         em_history) = load_model(load_model_path)
        # Sanity checks on the loaded model
        if loaded_layout.markers != layout.markers:
            raise ValueError(
                f"Loaded model markers {loaded_layout.markers} != "
                f"data markers {layout.markers}"
            )
        if abs(loaded_fps - fps) > 1e-6:
            print(
                f"[smoother] WARNING: loaded model fps={loaded_fps} "
                f"!= requested fps={fps}; using model fps",
                file=sys.stderr,
            )
            fps = loaded_fps
        if loaded_threshold != likelihood_threshold:
            print(
                f"[smoother] WARNING: loaded model threshold="
                f"{loaded_threshold} != requested {likelihood_threshold}; "
                f"using model threshold",
                file=sys.stderr,
            )
            likelihood_threshold = loaded_threshold
        if loaded_hash != data_hash and verbose:
            print(
                f"[smoother] NOTE: loaded model was fit on data with "
                f"hash {loaded_hash}, current data has hash {data_hash}. "
                f"Different data may give different results."
            )
    else:
        # Fit fresh: optionally triplet priors, then EM noise params.
        # Resolve head markers (default = mufasa convention if
        # the markers exist, else empty)
        if head_markers is None:
            default_head = ["nose", "ear_left", "ear_right", "headmid"]
            head_markers = [m for m in default_head if m in markers]
        head_set = set(head_markers)

        triplet_priors: List[TripletPrior] = []
        if use_triplets:
            # Triplet path (patch 87-90 behavior). Off by default
            # since patch 91 — see smooth_pose docstring for
            # rationale.
            from mufasa.data_processors.kalman_diagnostic import (
                auto_detect_candidate_triplets,
            )
            if candidate_triplets is None:
                if verbose:
                    print(
                        f"[smoother] use_triplets=True. "
                        f"Auto-detecting candidate triplets "
                        f"(cv_threshold={rigid_cv_threshold}, "
                        f"excluding head: {head_markers})..."
                    )
                triplet_results = auto_detect_candidate_triplets(
                    df, markers, likelihood_threshold,
                    cv_threshold=rigid_cv_threshold,
                    max_triplets=rigid_max_pairs,
                    exclude_markers=head_markers,
                )
                candidate_triplets = [tp for tp, _ in triplet_results]
                if verbose:
                    print(
                        f"[smoother] Found {len(candidate_triplets)} "
                        f"candidate triplets"
                    )
                    for tp in candidate_triplets:
                        print(f"[smoother]   {tp}")

            triplet_priors = fit_triplet_priors(
                positions, likelihoods, candidate_triplets, layout,
                likelihood_threshold,
            )
            if verbose:
                print(
                    f"[smoother] Fit {len(triplet_priors)}/"
                    f"{len(candidate_triplets)} triplet priors "
                    f"(others had insufficient data)"
                )
        else:
            if verbose:
                print(
                    f"[smoother] use_triplets=False (default). "
                    f"Pure per-marker temporal smoothing — no "
                    f"spatial pseudo-measurements. Pass "
                    f"--use-triplets to re-enable the patch-87 "
                    f"static rigid-body prior."
                )

        # Fit noise params via EM
        if verbose:
            print(
                f"[smoother] Fitting noise params via EM "
                f"(max_iter={em_max_iter}, tol={em_tol})..."
            )
        em_result = fit_noise_params_em(
            positions, likelihoods, layout,
            fps=fps, likelihood_threshold=likelihood_threshold,
            triplet_priors=triplet_priors,
            max_iter=em_max_iter, tol=em_tol,
            verbose=verbose,
            sessions=sessions,
            stratify=stratify,
            n_strata=n_strata,
        )
        params = em_result.params
        em_history = em_result.history
        if verbose:
            print(
                f"[smoother] EM finished after {em_result.n_iter} "
                f"iterations (converged={em_result.converged})"
            )

    # Save model artifact if requested
    model_artifact_path = None
    if save_model_path is not None:
        save_model(
            save_model_path, layout, params, triplet_priors,
            likelihood_threshold, fps, data_hash,
            em_history=em_history,
        )
        model_artifact_path = save_model_path
        if verbose:
            print(f"[smoother] Saved model to {save_model_path}")

    # Multi-session smoothing pass
    if verbose:
        print(
            f"[smoother] Smoothing {len(sessions)} session(s) "
            f"with boundary resets..."
        )
    smoothed_positions, smoothed_variances, n_obs = smooth_multi_session(
        positions, likelihoods, layout, params, triplet_priors,
        sessions, fps, likelihood_threshold,
    )

    # Write one parquet per session, plus a session_summary.json
    output_files: List[str] = []
    session_summary = []
    # Track whether we've already warned about parquet
    # unavailability — avoids one warning per session in
    # multi-session runs.
    parquet_warned = False
    for name, start, end in sessions:
        df_out = _arrays_to_df(
            smoothed_positions[start:end],
            smoothed_variances[start:end],
            likelihoods[start:end],
            markers,
        )
        out_path = out / f"{name}_smoothed.parquet"
        try:
            df_out.to_parquet(out_path, index=False)
        except (ImportError, ValueError) as e:
            # No parquet engine — fall back to CSV. The user
            # gets a working output even without pyarrow.
            if verbose and not parquet_warned:
                print(
                    f"[smoother] parquet write failed ({e}); "
                    f"falling back to CSV for all sessions"
                )
                parquet_warned = True
            out_path = out / f"{name}_smoothed.csv"
            df_out.to_csv(out_path, index=False)
        output_files.append(str(out_path))
        session_summary.append({
            "name": name,
            "n_frames": end - start,
            "output": str(out_path),
            "n_obs_mean": float(np.mean(n_obs[start:end])),
        })

    (out / "session_summary.json").write_text(
        json.dumps({
            "n_sessions": len(sessions),
            "total_frames": int(positions.shape[0]),
            "n_markers": layout.n_markers,
            "markers": list(layout.markers),
            "fps": fps,
            "likelihood_threshold": likelihood_threshold,
            "n_triplet_priors": len(triplet_priors),
            "triplet_priors": [
                {
                    "markers": list(tp.markers),
                    "n_samples": tp.n_samples,
                }
                for tp in triplet_priors
            ],
            "data_hash": data_hash,
            "model_artifact": model_artifact_path,
            "sessions": session_summary,
        }, indent=2),
    )

    if verbose:
        print(
            f"[smoother] Done. {len(output_files)} session(s) "
            f"written to {out}"
        )

    return {
        "layout": layout,
        "params": params,
        "triplet_priors": triplet_priors,
        "sessions": sessions,
        "output_files": output_files,
        "model_artifact": model_artifact_path,
        "em_history": em_history,
    }


# -------------------------------------------------------------------- #
# CLI
# -------------------------------------------------------------------- #

def _parse_triplets(s: str) -> Optional[List[Tuple[str, str, str]]]:
    """Parse semicolon-separated triplets, e.g. 'a,b,c;d,e,f'."""
    if not s:
        return None
    out: List[Tuple[str, str, str]] = []
    for chunk in s.split(";"):
        parts = [p.strip().lower() for p in chunk.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(
                f"Each triplet must have 3 markers; got {parts!r}"
            )
        out.append(tuple(parts))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Joint-state Kalman pose smoother with body-triplet "
            "prior. Stage 1 (v1) of the Mufasa Kalman smoother."
        ),
    )
    parser.add_argument(
        "pose_input", nargs="+",
        help=(
            "Pose data input. Single file (CSV or parquet), "
            "directory (recursively scanned for parquet, falling "
            "back to CSV), or multiple file paths. Multiple files "
            "are smoothed as separate sessions with proper "
            "boundary handling."
        ),
    )
    parser.add_argument(
        "--output-dir", default="./kalman_smoother_output",
        help="Output directory for smoothed parquets + summary",
    )
    parser.add_argument(
        "--likelihood-threshold", type=float, default=0.7,
        help=(
            "Likelihood threshold for high-confidence frames "
            "(matches diagnostic's --likelihood-threshold; default "
            "0.7 from Gravio's data review)"
        ),
    )
    parser.add_argument(
        "--head-markers", default="",
        help=(
            "Comma-separated list of head markers (excluded from "
            "body-triplet auto-detect). Default: nose,ear_left,"
            "ear_right,headmid where applicable."
        ),
    )
    parser.add_argument(
        "--triplets", default="",
        help=(
            "Semicolon-separated triplets, e.g. "
            "'back4,lateral_left,lateral_right;...'. If empty, "
            "auto-detected via the diagnostic's logic."
        ),
    )
    parser.add_argument(
        "--rigid-cv-threshold", type=float, default=0.20,
        help="CV threshold for triplet auto-detect",
    )
    parser.add_argument(
        "--rigid-max-pairs", type=int, default=8,
        help="Max number of auto-detected triplets",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Video frame rate",
    )
    parser.add_argument(
        "--em-max-iter", type=int, default=10,
        help="Max EM iterations for noise param fitting",
    )
    parser.add_argument(
        "--em-tol", type=float, default=1e-3,
        help="EM convergence tolerance (max relative sigma change)",
    )
    parser.add_argument(
        "--load-model", default="",
        help=(
            "Load a previously saved model (.npz) and skip fitting. "
            "Useful for applying a fit from one project to a new "
            "session of similar data."
        ),
    )
    parser.add_argument(
        "--save-model", default="",
        help="Save fit model to this .npz path",
    )
    parser.add_argument(
        "--use-triplets", action="store_true",
        help=(
            "Enable static rigid-body triplet priors (patch 87-90 "
            "behavior). OFF by default since patch 91 — the static "
            "rigid assumption was found to systematically bias "
            "smoothed output on posture-variable real-data behavior. "
            "Pass this flag only if you have data where the "
            "rigid-body assumption is genuinely justified."
        ),
    )
    parser.add_argument(
        "--no-stratify", action="store_true",
        help=(
            "Disable body-velocity stratification of the M-step. "
            "Stratification is ON by default since patch 94 — it "
            "addresses immobile-bias on rodent data by reweighting "
            "the M-step's q_pos sufficient statistics across "
            "velocity bins."
        ),
    )
    parser.add_argument(
        "--n-strata", type=int, default=_EM_DEFAULT_N_STRATA,
        help="Number of body-velocity bins for stratification",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress messages and EM iteration summaries",
    )
    args = parser.parse_args(argv)

    head_markers = (
        [m.strip().lower() for m in args.head_markers.split(",") if m.strip()]
        or None
    )
    triplets = _parse_triplets(args.triplets) or None

    pose_input = (
        args.pose_input[0] if len(args.pose_input) == 1
        else args.pose_input
    )

    smooth_pose(
        pose_input=pose_input,
        output_dir=args.output_dir,
        fps=args.fps,
        likelihood_threshold=args.likelihood_threshold,
        head_markers=head_markers,
        candidate_triplets=triplets,
        rigid_cv_threshold=args.rigid_cv_threshold,
        rigid_max_pairs=args.rigid_max_pairs,
        em_max_iter=args.em_max_iter,
        em_tol=args.em_tol,
        load_model_path=args.load_model or None,
        save_model_path=args.save_model or None,
        use_triplets=args.use_triplets,
        stratify=not args.no_stratify,
        n_strata=args.n_strata,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
