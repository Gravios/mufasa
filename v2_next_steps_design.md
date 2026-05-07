v2 next-steps design — addressing q_root_pos hitting the ceiling
================================================================

Date: 2026-05-07
Author: Claude (Opus 4.7)
Context: Real-data run with 67 sessions, patches 113-118
applied, completed without crashes but produced
q_root_pos = 99.7% of ceiling (136699 vs 137070 cap).
Smoothed pose in viewer shows skeleton displaced from raw
markers with huge variance ellipse (~120 px radius for
root). The smoother is honestly reporting "I can't fit
this", but the model is too rigid.

Diagnosis recap
===============

The current v2 model says marker m is at exactly:
  world_pos_m(t) = parent_segment.transform(offset_m) + ε(t)

where offset_m is a rigid (length, angle) polar offset and
ε is iid Gaussian observation noise σ_marker[m].

But real DLC markers move SYSTEMATICALLY within the body
frame:
  - nose: oscillates with sniffing (~7-10 Hz), whisking
  - ears: rotate, flick, can flatten
  - lateral markers: sit on fur that compresses/extends
    during rearing, crouching
  - back markers: spine flexes during locomotion
  - tail: undulates laterally — modeled as 1-DOF length
    per segment, doesn't capture lateral motion

The current model has only ONE outlet for this systematic
within-body motion: σ_marker[m]. But Gaussian iid noise
can't fit autocorrelated drift. So EM inflates q_root_pos
instead, letting the ROOT wander to absorb residuals that
the marker offsets can't explain. q_root_pos → ceiling.

Three possible solutions, in order of complexity
==================================================

Solution A: per-session marker offsets (smallest)
--------------------------------------------------

Each session fits its own marker_offsets, while
segment_lengths stay global. Handles inter-session
variation only:
  - Different rats have slightly different proportions
  - Camera position differs between recording days
  - Marker placement on the rat may shift over time

Code change scope: ~300 lines + tests
Where: fit_body_lengths becomes per-session call;
       FittedLengths gains per-session offsets dict;
       FK/observation looks up offsets by session_idx
EM impact: marker_offsets become per-session E-step
           parameters; M-step computes per-session offsets
           by trimmed mean of (obs - parent.transform)
           in the body frame
Limitation: doesn't handle within-session marker drift
            (sniffing oscillation, ear flicks)

Solution B: per-marker latent drift state (deeper)
---------------------------------------------------

Add δ_m(t) ∈ ℝ² to the state vector for each marker, with
mean-reverting random walk dynamics:
  δ_m(t+1) = (1 - α_m) δ_m(t) + w_m(t)
  w_m ~ N(0, q_drift_m × dt × I_2)

Observation:
  y_m(t) = parent.transform(offset_m + δ_m(t)) + ε_m(t)

Mean reversion (α_m > 0) gives a soft radius constraint:
stationary variance is q_drift_m × dt / (2α_m - α_m²).
Choosing α_m so stationary stddev ≈ R_m / 2 caps drift
at ~R_m.

This is the "unconstrained marker-pair distance within
max radius" interpretation: each marker can wander within
a ball relative to its rigid attachment point.

Code change scope: ~800-1000 lines + tests
State dim grows: D = 8 + 6S → D + 2K = 8 + 6S + 2K
For standard rat layout: 44 → 74 dimensions
EKF cost: O(D²) per frame → ~3× slower
Where: BodyLayout slicers, F matrix, H matrix, FK,
       observation, Jacobian, M-step blocks for q_drift,
       save/load
EM impact: M-step gains q_drift_m per marker (from
           S00/S11/S10 blocks for δ rows). α_m and R_m
           either fixed config or learned by EM.

Initial param values:
  R_m = sigma_marker[m] (use existing σ as drift bound)
  α_m = 0.05 (slow mean reversion)
  q_drift_m = (R_m / 5)² / dt (initial estimate)

Solution C: multi-taper FFT features + GHMM (research)
-------------------------------------------------------

Once Solution B is in place, δ_m(t) is an explicit clean
signal. Apply post-hoc analysis:

1. Sliding-window multi-taper spectrogram of δ_m(t) per
   marker, per session. Window ~1-2 sec, 5-7 tapers.
   Output: (T_windows, K_markers, F_bins) tensor of
   spectral power.

2. Stack per-marker spectra into a feature vector per
   window. PCA-reduce to ~10-20 dims if needed.

3. Fit a Gaussian HMM on the feature sequence. Choose
   N=4-8 states. Each state is a behavioral regime
   (walking, grooming, resting, sniffing, rearing).

4. (Optional) Use HMM state inferences as side info in
   re-fitting the smoother: per-state q_drift_m, σ_marker.
   Becomes a switching state-space model.

Code change scope: ~500 lines + tests, but separate module
(not part of smoother core)
Where: new mufasa.analysis.behavior_states module
       depends on scipy.signal.windows for tapers,
       pomegranate or hmmlearn for HMM
Use case: behavioral analysis output, not smoother
          improvement (that's Solution B)

Note on multi-taper for the smoother itself
============================================

In principle, knowing the marker's recent oscillation
characteristics could inform the smoother's prediction
of where it will be next. E.g., if nose has been
oscillating at 8 Hz with amplitude 5 px, predict next
position with much lower variance than assuming Gaussian
random walk.

But: the EKF already does this implicitly via the
predict-update cycle. The "spectral information" lives
in the state covariance P_t and the dynamics F. If the
dynamics model includes mode-switching (Solution C),
the EKF naturally adapts to oscillation regimes via the
state probabilities.

So multi-taper FFT is BEST USED OFFLINE on smoothed δ_m
trajectories to discover regimes, then fed back into
the smoother as a switching parameter set. NOT computed
online inside the EKF inner loop.

Decision: phased implementation
================================

Phase 1 (next patch, 119): per-session marker offsets
  - Smallest change, addresses inter-session variation
  - Test on Gravio's data: does q_root_pos drop below
    ceiling? If yes, we may not need Solutions B/C at all.
  - If still hits ceiling, continue to phase 2.

Phase 2 (later patch, 120): latent drift state
  - Larger change, addresses within-session drift
  - Adds 30 state dims for K=15 markers
  - Generates clean δ_m(t) signals as output
  - This is where the "radius constraint" really lives

Phase 3 (later, separate module): multi-taper + GHMM
  - Behavior analysis on δ_m trajectories
  - NOT a smoother modification
  - Outputs behavior state sequences, can be used as
    covariates in downstream calcium imaging analysis

Sanity check first
==================

BEFORE Phase 1, run v1 smoother on the same dataset.
v1 has different (and arguably less restrictive)
modeling assumptions. If v1 produces visually plausible
output, v2 has a regression we should fix in v2 directly.
If v1 also struggles, the dataset itself stresses both
models — Phase 1 work is justified.

Quick v1 invocation:

  python -m mufasa.data_processors.kalman_pose_smoother \\
      /data/testing/mufasa/test-20260427/.../input_csv/ \\
      --output-dir /tmp/smoothed_v1/ \\
      --likelihood-threshold 0.7 --fps 30 \\
      --verbose

Open the equivalent session in pose_viewer. Compare:
  - Skeleton location vs raw markers
  - Variance ellipse size
  - Smoothness vs jitter trade-off

If v1 looks correct, the v2 model is too constrained for
this data and we definitely need Phase 1.
If v1 also looks wrong, the dataset has structure neither
model captures — Phase 2 is justified.
