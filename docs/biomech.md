# The biomechanical model

A model of the rat's body, built on DeepMind's virtual rodent in MuJoCo, to go with the marker
and silhouette fits. It works in stages; each is useful on its own.

| stage | what | where | patch |
|---|---|---|---|
| 1 | the legs bend to the gait's paws | `mufasa/pose3d/leg_ik.py`, the 3-D viewer | 122lh |
| 2 | joint torques and ground forces from the observed motion | `mufasa/biomech`, `mufasa-pose3d dynamics` | 122li |
| 3 | hindlimb muscles: paths, moment arms, forces shared out to give the torques | `mufasa/biomech/osim.py`, `muscle.py`, `--muscles` | 122lj |
| 4 | whole movements re-solved for tracking, balance and effort; muscle activations smooth in time | `mufasa/biomech/optimise.py`, `--optimise` | 122lk |

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

## Stage 4: the movement re-solved

```
mufasa-pose3d dynamics K7_3d.parquet --gait K7_gait.parquet --muscles rat_hindlimb_bilateral.osim \
    --optimise [--iters 40] --frames 30137:30256
```

Stages 2 and 3 take the motion as fitted and ask what forces it needs, so a fit that is off
gives impossible forces. Here the motion is changed as well (`optimise.py`). Over a window of
frames, these are solved together:

* every joint of every frame;
* the paws' forces;
* with muscles, every hindlimb muscle's activation.

The costs:

| cost | σ / weight | |
|---|---|---|
| labels | 4 mm | tracking, as in the fit |
| stance paws on the floor | 1 mm | on the gait's stance/swing timing |
| stance paws not sliding | 1 mm a frame | *where* a paw is planted is free: it goes where the body balances |
| swing paws above the floor | 1 mm | and bearing nothing |
| the root's residual | 2 % of the weight | the motion's accelerations must be supplied by the ground (pushing, μ 0.8) |
| joint torques | 50 mN·m, weight 0.3 | effort |
| muscle torques = leg torques | 2 mN·m | with muscles |
| activations | Σa² | with muscles |
| activation rates | 0.1 a frame | with muscles: loads stay correlated in time |
| smooth joints | 0.1 rad a frame² | the model's redundant joints (the neck's twists) jitter otherwise |
| posture | as the fit | a weak pull to the stance; every joint within its range (bounds) |

**Forward and backward sweeps.**

* Velocities and accelerations are central differences, so each frame is coupled to its
  neighbours. The planted paws and the activation rates couple them too.
* Each step is Levenberg–Marquardt on the whole window. It solves one sparse linear system
  (joint Jacobians from MuJoCo's `mj_jacSite` and `mjd_inverseFD`; forces and activations enter
  linearly).
* Each solve passes information both ways through the window. That is the work iterated
  forward and backward sweeps (iLQR, DDP) would do, done at once.
* Windows are 40 frames, 30 apart. Each holds the previous window's last 2 solved frames, so
  velocities, accelerations and activations carry on across the joins.

Speed: about 1 s a frame with 76 muscles, 0.4 s without (one CPU core). That suits walking
bouts, not whole recordings.

**Checks** (`tests/smoke_122lk_optimise.py`).

* A disturbed stance (a leg folded, the body 3 mm up, the neck's twists jittering ±17°) is
  re-solved:
  * soles on the floor within 0.1 mm;
  * the ground carries the weight;
  * residual 3 % of the weight;
  * neck still;
  * labels within 1.3 mm.
* Gliding on planted paws, stitched windows match one window over the whole.

### On K7

120 frames of walking from frame 30137, with muscles (2 min):

| | stages 2 + 3 | stage 4 |
|---|---|---|
| label fit (median) | 20.4 mm | 19.6 mm |
| residual at the root (median) | 0.37 × weight | 0.08 × weight |
| reserve beyond the muscles | 21 % of the leg torques | 1.8 % |
| activation change a frame (mean) | 0.104 | 0.021 |
| knees at the flexion stop (L, R) | 54 %, 92 % | 23 %, 57 % |
| leg torques, 90th percentile | 104 mN·m | 75 mN·m |
| most active muscles (median) | tibialis anterior 0.82, obturator internus 0.66 | lateral gastrocnemius 0.57, rectus femoris 0.55, medial gastrocnemius 0.50, vastus lateralis 0.49 |

* The stance paws moved from the gait's placements by 6–18 mm (median), up to about 60 mm.
* The muscle pattern becomes a stance pattern: ankle extensors and knee extensors carry the
  body, where frame by frame the solver had leaned on hip rotators and the ankle's flexor.
* The knees still reach their stop in part of the stance. The virtual rodent's knee bends
  further than the rat's: to 33° between thigh and shank, against 38° in Johnson et al.'s
  model's range. See *The rat's leg ranges* below.

## The animal file

```
mufasa-pose3d animal K7_day1_3d.parquet K7_day2_3d.parquet --out K7.animal.json [--mass-g 450] [--limits rat]
mufasa run animal ...                               (the same)
mufasa-pose3d dynamics K7_day3_3d.parquet --gait K7_day3_gait.parquet --animal K7.animal.json
mufasa-pose3d smooth K7_day3_3d.parquet --out K7_day3_body.parquet --model --animal K7.animal.json
```

An animal's size and where its labels sit on its body do not change between sessions. The
animal file (`animal.py`) holds them:

* The virtual rodent is calibrated once (STAC) on frames drawn evenly from every session given.
  More sessions bring more behaviour, and a better-placed set of labels.
* The size is the median of the sessions' nose-to-tail estimates, or `--size`.
* The file is JSON:
  * each label's body and place on it (at size 1, so it scales with the model);
  * the size, and the weighed mass if given;
  * the joint ranges used;
  * the sessions, and the calibration's fit.
* `dynamics --animal` and `smooth --model --animal` use the same body and labels in every
  session, and skip their own calibration.
* K7: 600 frames, the fit from 27.2 to 23.7 mm, in 18 s.

## The rat's leg ranges (`--limits rat`)

The virtual rodent's knees and ankles reach further than a rat's. `--limits rat` (`dynamics`,
`smooth --model`, `animal`) narrows them:

* the source is Johnson et al. 2008's model: knee −150…−20°, ankle −30…60° (its flexion
  coordinates);
* they are carried onto the virtual rodent through stage 3's bone registration, posing both
  models' bones alike at each end;
* in the virtual rodent's angles: knee −50…79° (from −70…88°), ankle 18…80° (from −8…80°).

The ankle's mapping is the rougher one. The two ankles are not the same joint (OpenSim's has
three axes, the virtual rodent's one), and the posed bones differ by 11–29° beyond the hinge.

On K7 (the stage 4 window above), the rat's ranges make things worse, not better:

* residual 0.08 → 0.16 × weight;
* support 0.98 → 0.89 × weight;
* the legs sit at the new stops much of the time (ankles 38–47 % of frames).

K7's labelled trunk is low (60 mm) for the virtual rodent's legs at its size. Either K7's legs
are shorter than the scaled model's, or the trunk labels sit lower than the model's spine.
Until limb lengths are calibrated per animal, the virtual rodent's own ranges stay the default.
