# The biomechanical model

A model of the rat's body, built on DeepMind's virtual rodent in MuJoCo, to go with the marker
and silhouette fits. It works in stages; each is useful on its own.

| stage | what | where | patch |
|---|---|---|---|
| 1 | the legs bend to the gait's paws | `mufasa/pose3d/leg_ik.py`, the 3-D viewer | 122lh |
| 2 | joint torques and ground forces from the observed motion | `mufasa/biomech`, `mufasa-pose3d dynamics` | 122li |
| 3 | hindlimb muscles: paths, moment arms, forces shared out to give the torques | `mufasa/biomech/osim.py`, `muscle.py`, `--muscles` | 122lj |
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

## Stage 3: the hindlimbs' muscles

```
mufasa-pose3d dynamics K7_3d.parquet --gait K7_gait.parquet --muscles rat_hindlimb_bilateral.osim \
    [--architecture eng2008] [--save-muscles K7_muscles.json]
```

adds, per frame, each muscle's activation (`<muscle>_act`, 0–1) and force (`<muscle>_N`), and
the **reserve** at each leg joint (`reserve_<joint>_mNm`).

**Where the muscles come from.** Mufasa does not ship a muscle model of its own.

* It reads an OpenSim model of the rat hindlimb. The tested one is the UVA MAMP Lab's
  `rat_hindlimb_bilateral.osim` (github.com/UVA-MAMP-Lab/RatHindlimb):
  * Johnson et al. 2008's model;
  * with Young et al. 2019's attachments;
  * Johnson et al. 2011's muscle parameters;
  * 38 muscles a side.
* It is distributed under its own terms (SimTK `rat_hlimb_model`), so download it yourself and
  pass the `.osim` file.
* `--save-muscles` keeps the muscles once carried onto the virtual rodent (a `.json`), which
  `--muscles` also takes.

**Carrying them over** (`osim.py`).

1. Each bone's attachment points are moved by a rotation, one scale and a translation. These
   are fitted to landmarks both models have:
   * the hip, knee and ankle centres;
   * the knee's and the ankle's axes, pointing outwards;
   * the pelvis's long axis;
   * the foot's tip.
   The bones come out within 2 % of the virtual rodent's lengths.
2. **The knee.** OpenSim's knee slides as it bends (its translations follow the angle), while
   the virtual rodent's is a hinge. Carried over as they were, the quadriceps lost their lever at
   a bent knee: 0 mm against OpenSim's 3.5–4.6 mm.
   * So the 17 muscles crossing the knee have their points near it (within 12 mm) refitted, until
     their length follows OpenSim's over the angles both knees reach.
   * Their length curves went from 0.6–1.7 mm rms off to 0.02–0.26 mm.
3. **Tendon slack lengths.** Each muscle's tendon slack length is shifted by its length on the
   virtual rodent less OpenSim's, with both models' bones posed alike (OpenSim's default
   standing pose). Its fibres are then as stretched as in OpenSim there.
   * The quadriceps' slack lengths moved 8–9 mm, most others under 3 mm.
   * Without this the quadriceps sat at 1.3–1.7 optimal lengths in the stance, and their passive
     pull took most of the knee's torque.
4. Forces scale with size², lengths with size; a weighed mass rescales forces by mass^(2/3).

The moment arms come out as in Johnson et al. 2008, signs and sizes (mm):

| muscles | hip | knee | ankle |
|---|---|---|---|
| hamstrings | 7–10 (extend) | 2–11 (flex) | |
| hip flexors: iliopsoas, TFL | 3–12 (flex) | | |
| quadriceps | | 3–4 (extend) | |
| gastrocnemius, plantaris | | 1–5 (flex) | 1.4–2.1 (extend) |
| tibialis anterior, EDL | | | 1.7–2.3 (flex) |

**The muscles' forces** (`muscle.py`).

* Each muscle is a spatial tendon in MuJoCo, so its length and moment arms come from MuJoCo.
* It is a Hill-type muscle with a rigid tendon (Zajac 1989):
  * active force–length, a Gaussian of width 0.45;
  * passive, exponential (Thelen 2003);
  * force–velocity, Hill's hyperbola shortening, up to 1.8× lengthening.
* **Static optimisation** (Crowninshield & Brand 1981): per frame and hindlimb, the activations
  with the least Σa² that give the hip's three torques, the knee's and the ankle's.
* Where the muscles cannot give them, a **reserve** torque makes up the rest, at a cost of 100
  muscles fully on per 10 mN·m.
* The toes have no muscle in the model (the digit muscles end on the foot), so their torque is
  left as it is.

**Checks.**

* Standing in OpenSim's pose on the hindpaws (196 mN·m at the hip, 122 at the ankle), the
  muscles supply everything (reserves under 0.05 mN·m).
* Standing in the virtual rodent's tucked stance, they supply 99 %. The small hip rotation
  torques need the few muscles that rotate the hip outwards (obturator internus, gemelli),
  fully on.

**Architecture** (`--architecture eng2008`).

* Peak forces come from Eng et al. 2008's PCSAs × 22.5 N/cm² (Powell et al. 1984); fibre
  lengths and pennation from their Table 1.
* Both are scaled from their 323 g rats. A muscle they measured whole (biceps femoris,
  semitendinosus) is shared out as the OpenSim model shares it.
* The tendon slack lengths are re-solved to keep the fibres' stretch in the reference pose.
* The 27 muscles both cover total 196 N (Johnson 2011) against 194 N (Eng 2008) a leg. They
  differ most at the adductors:
  * adductor brevis 12.3 → 1.4 N;
  * adductor longus 1.1 → 4.9 N;
  * adductor magnus 3.0 → 6.5 N (Eng's adductor tertius is not in the model);
  * peronei 3.9 → 7.7 N.

### On K7

500 frames of walking from frame 30037, the muscles taking 14 % of the leg torques' size as
reserve (median), 15 % with Eng's architecture.

* The leg joints are at their limits in nearly every frame. The knee is at the virtual rodent's
  flexion limit (−70°, 33° between thigh and shank) in 57 % (left) and 83 % (right) of frames,
  well past OpenSim's range (49°).
* This is the gait's inferred paws again (stage 2): they sit too close under the hips, so the
  legs fold to their stops.
* In the 6 frames with no leg joint at a limit, the reserve is 3.5 % of the torque.
* The most active muscles: rectus femoris (median 0.71), obturator internus, the gastrocnemii.
  The highest forces: lateral gastrocnemius and rectus femoris, about 25 N (90th percentile).

So the muscle model is consistent, and the legs' kinematics are what limit it on K7. Labelled
paws, or stage 4's optimisation (which keeps joints in range and the body balanced), are the
remedy.
