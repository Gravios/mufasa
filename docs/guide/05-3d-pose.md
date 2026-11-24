# 5. 3-D pose

The **3-D** page turns 2-D labels into 3-D marker positions in millimetres. It works from one
overhead camera, using models of the rat's body for depth, or from two synchronised cameras by
triangulation. From the 3-D pose it estimates gait: where the paws are, the steps, and how the
weight is shared.

```text
camera calibration ─┐
2-D labels ─────────┼─► 3-D inference ─► 3-D smoothing ─► heights ─► gait ─► (dynamics, ch. 6)
Kalman v2 run ──────┘          │
                               └─► the 3-D viewer
```

The world frame is in millimetres, with the floor at `z = 0` and `z` up. The origin is the centre
of the box's floor.

This chapter walks through the page. [`../pose3d.md`](../pose3d.md) explains each step's method,
options and output in depth.

## Calibrate the camera

**3-D › Camera calibration**. Calibrate each camera set-up once. The calibration is saved in
`sources/calibration/`.

### From the box (usually enough)

1. Choose a *Video* filmed with the camera, and give the camera a name (e.g. `K7`).
2. Press **Mark box…**. It shows the median of many frames, so the animal disappears.
   * Click the box's **four floor corners**.
   * Then click several points up each **vertical edge** where two walls meet, pressing *Next
     edge* between edges. Right-click undoes a point.
3. Enter the box's size **measured at the floor**: *Floor length*, *Floor width*, *Wall height*,
   and the *Camera height (above floor)*. The camera height must be measured: the marks fix the
   scale at the floor but not how high the camera is.
4. If the walls lean outward (many cages are wider at the top), set *Walls flare outward* to the
   angle, or tick *fit from the corner edges* to estimate it from three or more marked edges.
5. **Run**. The report gives the point below the camera, the focal length, how well the marks fit,
   and the fitted flare.

The vertical edges meet below the camera, which fixes where the camera is. When the length and
width you entered don't match the marked floor's shape, *If they disagree, trust* decides which is
kept, and the report says so.

### With a checkerboard (lens distortion)

For a wide-angle lens, also calibrate the lens: *Lens (checkerboard)* takes a video or a folder of
images of a checkerboard, with its inner-corner count and square size. The camera's position is
then solved from the floor corners.

### Which camera goes with which video

A video uses `<video>.camera.toml` if there is one. Otherwise it uses `camera_<first part of its
name>.toml` (`camera_K7.toml` for `K7_cam_2026…mp4`), else `camera.toml`. `mufasa list
calibrations` lists the cameras.

## Infer 3-D pose

**3-D › 3-D inference**:

| Option | What it does |
|---|---|
| 2-D labels | the pose to lift into 3-D: an import, or a Kalman v2 run |
| Gaps | low-confidence labels (below 0.9) are filled from a Kalman v2 run (*fill from Kalman v2 run*) or interpolated |
| Cameras · Videos | the calibration(s), and the videos to do |
| Mode | one camera, or two cameras (files whose names differ only by a token such as `camA`/`camB` are paired) |
| Ear-to-ear distance | the head's size (default 36 mm), which sets the scale of the head |
| Wall tolerance | how far beyond the box's walls a label may go (default 20 mm), allowing for calibration error |
| Body size vs Rat7M · Body size vs virtual rodent | the animal's size against the models; *auto* measures it |
| Rat7M trunk heights | look up the trunk's heights in Rat7M motion capture from the head's height and pitch and the head-to-hips distance |
| spine within bend limits · anatomical limits (virtual rodent) | keep the spine and each label within what a rat's body can do |
| 3-D Kalman smoothing | smooth the result in the same run |
| Frame rate · Workers · Solver | the videos' fps; how many videos at once; the solver (GPU, PyTorch on the CPU, or SciPy) |

**How one camera gives depth.** A single camera gives each label only a line of sight, and the
body models choose where along it the label sits:

* a rigid head, learned from the session;
* the trunk's heights from Rat7M;
* the spine as one chain within its bend limits;
* every joint within the virtual rodent's range.

Every label stays on its own line of sight, so the 3-D pose still lands on the 2-D labels.

Output: `derived/pose3d/<run>/<video>_3d.parquet`, with each marker's `_x/_y/_z` in mm. The file's
metadata holds the camera, the box (and its flare) and a report.

### Heights and the floor

Every run ends by keeping each label within the height range the body allows, and never below the
floor. A single-camera label moves along its line of sight, so it still matches its 2-D label.
`<marker>_z_fixed` marks the frames moved. For older files, run `mufasa run heights X_3d.parquet
--out X_3d_heights.parquet`.

## Smooth in 3-D

**3-D › 3-D Kalman smoothing** smooths an existing 3-D run into a new one: each marker with a
constant-velocity Kalman filter and a backward pass, with outlier gating.

* For one camera, *Noise along the ray* is larger than *Noise across the ray*: depth is the
  uncertain direction, so the smoothing evens out depth while each point stays on its label.
* *Outlier gate* sets how far from the prediction a measurement may be before it is ignored.

To smooth with the whole body instead, so bones can't stretch, use `mufasa run smooth X_3d.parquet
--model` ([chapter 6](06-body-model-and-dynamics.md#smoothing-with-the-skeleton)).

## Import motion capture (C3D)

**3-D › Import motion capture (C3D)** reads `.c3d` files from motion-capture systems (Vicon,
Qualisys, OptiTrack, Cortex, …) into `sources/pose3d/<trial>_3d.parquet`. They open in the 3-D
viewer like any 3-D run, for example as ground truth for the single-camera estimate.

* *Up axis in the file*: most systems use z up; OptiTrack and some Vicon set-ups use y up.
* *Floor height in the file*: where the lab's floor is, which is moved to z = 0. Gait needs that.
* *Rename markers* (`HeadF=head_nose, SpineM=back_T8`) maps the system's names onto the project's.
  *Skeleton* (`a-b, b-c`) gives the lines between markers.

## Gait

**3-D › Gait (paws from a body model)** estimates from the 3-D body where the paws are, when each
is planted, how the weight is shared between them, and the centre of mass:

| Option | What it does |
|---|---|
| 3-D run | the 3-D pose |
| Body mass | the animal's weighed mass, in grams, for the loads. *From the size* guesses it from the body's size, roughly |
| Frame rate | the fps (default: the run's) |
| Bounds | the normal ranges of stance, swing, stride, duty factor and base of support that the gait is kept within: *CatWalk rats, by speed* (108 uninjured rats, 13–71 cm/s; Neckel et al. 2013) or *TactileArena* (one rat, fixed) |

The paws needn't be labelled: they are inferred from what the body can do. Where they are labelled
(`paw_LF`, `paw_RF`, `paw_LH`, `paw_RH`, or `forepaw_left` and the like), the seen paws anchor the
gait. The output is `<video>_gait.parquet`, with each paw's position, contact and load per frame,
and `<video>_steps.csv`, with each step. To see the paws, open the 3-D viewer › *View & models* ›
*Load…* › the gait file. The bounds are adult-rat values; for a juvenile or a mouse, see
[`../pose3d.md`](../pose3d.md#gait-paws-weight-distribution-and-the-centre-of-mass).

## The 3-D viewer

**3-D › 3-D viewer**: choose a 3-D file and *Open 3-D viewer*.

* **Video.** The default, *(the 3-D file's own video)*, lists every 3-D file in the project and
  finds each one's video by name, in subfolders too; the section says which it found. Choose a
  video to list only its 3-D files, or *(no video)* for motion capture alone. In the viewer,
  **Video…** picks another video.
* **The view.** Left-drag orbits, right-drag pans, the wheel zooms. *Top* looks straight down.
  *Camera* looks through the calibrated camera, so the markers sit on the animal in the video.
* **Video in 3-D** draws the current frame on the floor of the box. **Reprojection** draws the 3-D
  markers back onto the video, to check the fit.
* **View & models.** Show markers, names, the skeleton, the grid. Add the virtual rodent or a
  rigged body, posed to the markers of each frame, or an arena mesh (`.obj`, `.stl`, `.ply`).
* **Features** lists the 3-D file's numbers and those of the gait and dynamics files beside it, as
  traces with histograms that follow the playback.
* **JPDF** shows the joint distribution of two of those numbers beside the 3-D view, with the
  current frame and its recent trail.
* **Animation** builds a camera move from key views, previews it, and **Render video…** writes it.
  With *All panels*, every panel shown is in the video.
* **● Record** records what is shown while it plays: every panel, laid out as on the screen, at
  the video's resolution. It is saved under `derived/frames/annotated/`.

The window can be made narrow: its toolbars wrap onto more rows.

From a terminal, `mufasa-view3d --pose X_3d.parquet` opens the viewer; it finds the video the same
way.

## The 3-D pipeline from the command line

Every step on this page also runs as `mufasa run <step>` (and `mufasa-pose3d`), so a session can be
done start to finish without the workbench:

```bash
mufasa run calibrate box sources/calibration/camera.marks.json --length 400 --width 300 \
    --wall-height 250 --camera-height 500 --wall-tilt auto --out sources/calibration/camera_K7.toml
mufasa run all --pose sources/pose/K7_camDLC.parquet --camera sources/calibration/camera_K7.toml \
    --mass-g 410
```

See [chapter 13](13-command-line.md) and [`../cli.md`](../cli.md#mufasa-run).
