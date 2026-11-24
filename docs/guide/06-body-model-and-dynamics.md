# 6. Body model and dynamics

Two things in Mufasa use a model of the whole body rather than separate markers:

* **The Body model page** fits a rigged 3-D body (from Blender) to top-down video. It uses the
  animal's silhouette and its 2-D labels, within joint limits.
* **The biomechanical model** (from the command line) uses DeepMind's virtual rodent in MuJoCo. It
  smooths 3-D labels on a body whose bones keep their lengths, and gives joint torques, ground
  forces and hindlimb muscle forces. It can also re-solve the movement for tracking, balance and
  effort.

In depth: [`../bodymodel.md`](../bodymodel.md) and [`../biomech.md`](../biomech.md).

## The Body model page

### Import rig (.blend)

**Body model › Import rig (.blend)** turns a rigged Blender model into a file Mufasa can pose:

| Field | What to put |
|---|---|
| .blend file | the Blender file (e.g. the HBP mouse, `full_mouse_blender5.blend`) |
| Armature object · Skin mesh object | the names of the skeleton and the skin in that file (`Armature` and `Skin` in the HBP mouse) |
| Rig name | the saved rig's name |
| Blender | Blender's Python module (`pip install bpy`) or a Blender executable, which reads the file |

The rig is saved as `sources/bodymodel/<name>.rig.npz`. It holds the bones, the skin (full and
simplified), the skinning weights and the fur colour.

### Fit body model

**Body model › Fit body model** fits the rig's posture to each frame:

| Field | What it does |
|---|---|
| Rig · Video · 2-D labels | the rig, the video, and its labels (an import or a Kalman v2 run) |
| Frames | from, to, and every n-th |
| *Forwards and backwards* | use the neighbouring frames too (smoother, slower) |
| Smoothness · Scale · Scale from · Animal | how smooth over time; the animal's size, and what it is measured from |
| Workers | frames fitted in parallel |
| *Render the model over the video* · *side by side with the original* | write an overlay video |

Each frame's posture includes:

* position and heading on the floor;
* rearing and roll about the hind paws;
* the bend of the lumbar, thoracic and cervical spine;
* the head's yaw and pitch;
* the tail.

Each is kept within published joint limits. The result opens in the 3-D viewer (*View & models* ›
*Load…* the rig).

## The biomechanical model (command line)

These steps run from the terminal (`mufasa run <step>` or `mufasa-pose3d`). They need
`pip install -e ".[biomech]"` (MuJoCo). Each step is useful on its own:

| Stage | What | Command |
|---|---|---|
| Smooth with the skeleton | 3-D labels explained by the virtual rodent: bones keep their lengths, gaps are filled, swaps are flagged | `mufasa run smooth X_3d.parquet --model` |
| 1 | the legs bend to the gait's paws | the 3-D viewer (with the gait file) |
| 2 | joint torques and ground forces from the motion | `mufasa run dynamics` |
| 3 | hindlimb muscles: forces shared out to give the torques | `dynamics --muscles rat_hindlimb.osim` |
| 4 | the movement re-solved for tracking, balance and effort | `dynamics --optimise` |

### Smoothing with the skeleton

```bash
mufasa run smooth K7_3d.parquet --out K7_3d_body.parquet --model --joints-out K7_joints.parquet
```

The virtual rodent's label sites are first calibrated to this animal. Then stretches of frames are
solved together: the labels are fitted robustly (a swapped or wild label barely pulls on the body),
the joints move smoothly within their ranges, and the paws stay above the floor. Out come:

* the labels on a body whose bones keep their lengths, gaps filled;
* `<label>_model_mm`: each label's distance from the body;
* `<label>_flag`: frames where a label disagreed with the body (an outlier or a left/right swap);
* with `--joints-out`, every joint's angle per frame.

### Joint torques, ground forces and muscles

```bash
mufasa run dynamics K7_3d.parquet --gait K7_gait.parquet --mass-g 410
mufasa run dynamics K7_3d.parquet --gait K7_gait.parquet --mass-g 410 --muscles rat_hindlimb_bilateral.osim
mufasa run dynamics K7_3d.parquet --gait K7_gait.parquet --mass-g 410 --muscles rat_hindlimb_bilateral.osim \
    --optimise --frames 30137:30256
```

* **Stage 2:** the forces at the paws and every joint's torque that the observed motion needs
  (inverse dynamics). The *residual* (the force no paw can explain) says how far the motion is
  from physically possible.
* **Stage 3:** `--muscles` adds the hindlimb muscles from an OpenSim model (e.g. Johnson et al.
  2008's rat hindlimb). Each muscle's path and moment arms are registered onto the virtual rodent,
  and the torques are shared out between them. `--save-muscles` keeps the registered set as
  `.muscles.json` for next time. `--architecture eng2008` uses Eng et al. 2008's muscle
  architecture.
* **Stage 4:** `--optimise` changes the motion as well, over windows of frames. It solves the
  joints, the paw forces and the muscle activations together, for tracking, balance and smooth
  effort. Labelled paws stay where they were seen planted. `--iters` sets how hard it works.

The output is `<video>_dynamics.parquet`: per frame, the forces, the torques, and (with muscles)
each muscle's force and activation. The 3-D viewer's *Features* lists them.

### One animal, many sessions

An animal's size and where its labels sit on its body don't change between sessions. Calibrate them
once and reuse them:

```bash
mufasa run animal K7_day1_3d.parquet K7_day2_3d.parquet --name K7 --mass-g 410 --out K7.animal.json
mufasa run dynamics K7_day3_3d.parquet --gait K7_day3_gait.parquet --animal K7.animal.json
mufasa run smooth K7_day3_3d.parquet --out K7_day3_body.parquet --model --animal K7.animal.json
mufasa run all --pose sources/pose/K7_day4DLC.parquet --camera sources/calibration/camera_K7.toml \
    --animal K7.animal.json
```

More sessions give the calibration more behaviour to work from. `mufasa list models --kind body`
lists the calibrated animals; `mufasa info K7.animal.json` shows one.

### The rat's leg ranges

The virtual rodent's knees and ankles reach further than a rat's. `--limits rat` (on `dynamics`,
`smooth --model` and `animal`) narrows them to Johnson et al. 2008's ranges, carried onto the
virtual rodent: knee −50…79°, ankle 18…80°.
