# The biomechanical model

A model of the rat's body, built on DeepMind's virtual rodent in MuJoCo, to go with the marker
and silhouette fits. It works in stages; each is useful on its own.

| stage | what | where | patch |
|---|---|---|---|
| 1 | the legs bend to the gait's paws | `mufasa/pose3d/leg_ik.py`, the 3-D viewer | 122lh |
| 2 | joint torques and ground forces from the observed motion | `mufasa/biomech`, `mufasa-pose3d dynamics` | 122li |
| 3 | muscles: paths, moment arms, loads shared out to give the torques (hindlimbs first) | planned | |
| 4 | trajectories optimised for tracking and effort (iterative LQR, forward and backward sweeps) | planned | |

Install: `pip install "mufasa[biomech]"` (MuJoCo 3.2+).

## Stage 2: joint torques

```
mufasa-pose3d dynamics K7_3d.parquet --gait K7_gait.parquet [--frames 30000:31000] [--mass-g 450]
```

writes `K7_dynamics.parquet`: per frame the fit's error, each paw's force (N, `<paw>_grf_x/y/z_N`)
and whether it bore weight, the residual at the root, and per joint its angle (`q_<joint>_deg`)
and net torque (`tau_<joint>_mNm`), under the model's joint names (`hip_L_extend`, `knee_L`,
`ankle_L`, …). The summary and the calibrated label sites are in the file's metadata
(`mufasa.biomech.pipeline.read_dynamics`).

1. **The model** (`mjmodel.py`). The virtual rodent (`rodent.xml`, 67 hinge joints) gets a free
   root, so the whole body moves and the ground must hold it up. It is scaled to the animal as the
   gait scales it (nose to tail base): lengths × size, masses × size³, inertias × size⁵, or
   rescaled to `--mass-g`.
   * A site per label on its body. A skin label goes on the vertex's main body, at most 2.5 mm
     off where the skin would carry it.
   * Two sole sites per paw.
   * dm_control's joint springs, damping and armature are off. So are MuJoCo's constraints, which
     inverse dynamics would read as forces. The torques are each joint's **net** torque: muscles
     and passive tissue together.
   * MuJoCo's kinematics equal the repo's own reader's (`rodent_model.py`) to 1e-13 mm.
2. **Calibration** (STAC; Wu et al. 2013). Where a label sits on its body is the labeller's, not
   the model's.
   * K7's back labels are evenly spaced, 40–45 mm apart, where the model's named vertebrae are
     21–36 mm apart. Its `back_V2` sits about 20 mm further back.
   * On 400 frames spread over the recording, poses are fitted, then each label's place on its
     body is re-solved (the median of its observed place in that body's frame), a few times.
     The round that fits best is kept.
   * On K7 the labels moved 6–35 mm, and the fit on frames left out went from 31.6 to 24.2 mm.
3. **The fit** (`fit.py`). Every joint, frame by frame, within its range (Levenberg–Marquardt on
   MuJoCo's Jacobians, from the frame before):
   * the labels (σ 4 mm);
   * the gait's paws: stance on the floor at the gait's position, swing on its arc, as in
     stage 1;
   * a weak pull towards the stance.
4. **Dynamics** (`dynamics.py`).
   * Velocities and accelerations come from MuJoCo's position differences, smoothed
     (Savitzky–Golay over 0.15 s).
   * `mj_inverse` gives the generalised forces.
   * The ground forces at the planted paws are solved to supply the root's six:
     * never pulling, within friction (μ 0.8, assumed);
     * torque rows weighted by a 5 cm lever;
     * the split of the weight among the paws as the gait splits it (four paws down leave more
       unknowns than equations).
   * Joint torques are what is left at the hinges.

### The residual

Whatever force and torque at the root the paws cannot supply is the **residual**. It checks
everything before it: the kinematics, the masses, the contacts.

* Standing still on four paws, the support equals the weight and the residual is zero
  (2e-6 N).
* Falling freely, no joint needs a torque.
* On K7, walking (500 frames from frame 30037, label fit 21.5 mm):
  * the paws carry the weight (support 1.00 × weight);
  * the residual is 41 % of the weight (median);
  * the hips carry up to about 125 mN·m, knees and ankles 85–130 mN·m, the lumbar spine
    up to 240 mN·m (90th percentiles).
* The residual is large because **the gait's paws do not keep the body balanced**. In about
  half these frames the fitted body's centre of mass is not over the paws planted under it. The
  gait analysis found the same independently: its ZMP is outside the support in 46 % of frames.
* A real rat's paws do keep it balanced, so the inferred placements are what is off. Labelled
  paws would settle it, and stage 4's optimisation would place paws that balance.
* While rearing, the forepaws cannot reach the floor (43 % of K7's frames). The residual is
  then the support from walls, which nothing models.

Speed: about 11 ms per frame for the fit, 1 ms for the dynamics (CPU).
