# 3-D pose (patch 122kf)

The **3-D** tab (after Preprocessing) turns 2-D labels into 3-D marker
positions in millimetres, from one overhead camera or two synchronised
cameras, and smooths them with a 3-D Kalman filter. The same steps are
available as `mufasa-pose3d` on the command line and as
`mufasa.pose3d` in Python. Results open in the 3-D viewer
(`mufasa-view3d`, the feature viewer's **3-D…** button, or the tab's
**3-D viewer** section).

```
2-D labels (sources/pose/) ─┐
Kalman v2 run (fills gaps) ─┼─► 3-D inference ─► 3-D Kalman smoothing ─► derived/pose3d/<run>/<video>_3d.parquet
camera (sources/calibration/)┘
```

World frame: millimetres, floor at `z = 0`, `z` up, origin at the box
floor centre (when calibrated from the box).

## 1. Camera calibration

**From the box (one image).** *Mark box…* shows the median of frames from
the chosen video (the animal disappears). Click the four floor corners,
then several points along each vertical edge where two walls meet
(*Next edge* between edges; right-click undoes). Enter the floor length
and width, the wall height and the camera height above the floor.

* The wall-corner edges meet directly below the camera; that point is
  the principal point. Without them the image centre is used.
* A straight-down camera sees the floor's true shape. If the entered
  length and width do not match the marked floor's shape, the one chosen
  under *If they disagree, trust* is kept and the report says so.
* Marks and the floor fix the scale at the floor but **not the camera
  height** — measure it.

**With a checkerboard (lens).** Give a video or image folder of a
checkerboard (inner corners, square size). The lens (focal length,
distortion) is calibrated (adapted from rpimocap), then the camera pose —
including its height — is solved from the four floor corners.

Cameras are saved as `sources/calibration/camera_<name>.toml`; a video
uses `<video>.camera.toml`, else `camera_<first part of its name>.toml`
(`camera_K7.toml` for `K7_cam_…`), else `camera.toml`, in any letter case.
A video with none shows *(no camera)*. The line under the list names the files
it looked for and the camera files the folder has (patch 122la).

## 2. 3-D inference — one camera

One camera gives each label only a line of sight. Depth comes from body
models; every label stays on its own line of sight, so the 3-D pose
lands on the 2-D labels.

1. **Rigid head.** The outline of the five head labels is learned from
   the session (structure from motion over confident frames); its size
   is the ear-to-ear distance (*Ear-to-ear distance*, default 36 mm).
   How far the nose and the top of the head sit below the ear line
   cannot be seen from above, so it is fixed
   (`MonoConfig.head_template_z`; default nose −18.4 mm, top of head
   −4.7 mm, from the K7 head that matched Rat7M) — learning it freely
   lets the fit trade head pitch against height. All frames are fitted
   together with the box limits, a modest head-roll prior and smooth
   motion. The walls tolerate labels up to *Wall tolerance* (20 mm)
   beyond them: the box is only as accurate as the calibration and the
   entered size, and a hard wall pushes heads up along their lines of
   sight (on K7 by about 20 mm at the median).
2. **Trunk heights from Rat7M.** From head height, head pitch and the
   overhead head-to-hips distance, the heights of the shoulders, mid-back,
   lumbar spine and hips are looked up in Rat7M motion capture
   (k-nearest neighbours; 5–9 mm median error on an unseen rat). *Body
   size vs Rat7M* scales the lookup for smaller or larger animals.
3. **Pelvis.** `back_V2`→`back_L6` is pitched to the Rat7M lumbar angle
   (about 25° walking, about 60° rearing).
4. **Spine chain.** The back labels between pelvis and head are refitted
   as one chain: heights stay near the Rat7M values, bends are pulled to
   the Rat7M hump for that head height, backward bends beyond the Rat7M
   95th percentile are penalised, the neck bends within the Rat7M range.
5. **Tail.** Each tail label hangs as low as possible on its line of
   sight without the tail stretching (lengths measured from the session).

**Anatomical limits** (*anatomical limits (virtual rodent)*, on by
default; patch 122kr). DeepMind's virtual rodent — a rat skeleton with
every joint's range and a skin, from dm_control (Merel et al. 2020;
Apache 2.0) — sets what the body can do:

* **Heights.** For each head and trunk label, the highest it can be (the
  rat rearing on its hind paws, every joint within range) and the lowest
  (no part of the body below the floor), scaled to this rat with 10%
  slack. The head is moved along its lines of sight into its range — it
  is scaled about the camera, so it still lies on its labels — and the
  trunk labels are kept within theirs, before and during the spine chain.
  A head placed far too high usually means the ear labels came too close
  together (the head turned or seen end-on).
* **Spine bends.** The up/down bend of the lumbar region (`back_V2`→`back_L6`
  against `back_T8`→`back_T4`) and of the neck (`back_T8`→`back_T4`
  against `back_T4`→`head_back`) stay within the model's range (5° slack),
  as hard limits in the spine chain, in frames where every spine segment
  points forward from overhead. Where one points backwards (labels stacked
  above each other, as when rearing or with the head over the shoulders —
  about a third of K7's frames) an up/down bend cannot be measured from one
  camera and is left to Rat7M, which still sets the typical shape.
* **Size.** *Body size vs virtual rodent* (auto): the length of the back
  (`head_back` → `back_V2`, overhead) in confident frames with the head
  low, against the model's.

The report (`anatomy`) gives the size, the frames whose head was lowered or
raised, the trunk labels moved per label, and frames still outside the
bend ranges. The limits are worked out once by
`scripts/build_virtual_rodent_limits.py` into
`mufasa/assets/models/pose3d/virtual_rodent/virtual_rodent_limits.json`;
`mufasa.pose3d.rodent_model` reads the model (forward kinematics and
skinning, NumPy only).

Low-confidence 2-D labels (below 0.9) are replaced by a Kalman v2 run
(*fill from Kalman v2 run*) or interpolated.

Marker names and the mapping of a project's spine labels onto the Rat7M
chain are in `mufasa.pose3d.mono.MonoConfig` (defaults: the 15-marker rat
skeleton).

## 3. 3-D inference — two cameras

Files whose names differ only by a token (e.g. `camA` / `camB`) are
paired; each camera needs its own camera file. Labels are triangulated
(linear DLT on undistorted pixels) and gated by likelihood and
reprojection error.

## 4. 3-D Kalman smoothing

Constant-velocity Kalman filter with Mahalanobis outlier gating and a
backward (Rauch–Tung–Striebel) pass, per marker. For single-camera runs
the measurement noise lies mostly along the line of sight (*Noise along
the ray* vs *across the ray*), so smoothing evens out depth while points
stay on their labels. Runs inside 3-D inference (checkbox) or on an
existing run (*3-D Kalman smoothing* section, new run).

## 5. Heights and the floor

A single camera sees heights worst near the floor, where a few pixels
along the line of sight are many millimetres of height. Patch 122le adds a
last step to every 3-D run, after the Kalman smoothing: each label is kept
within the height range the virtual rodent allows it (`head_nose`, the
ears, the back and the hips, scaled to the animal, plus 10 %) and **never
below the floor (z = 0)**. Labels the model has no range for, and
two-camera runs, are kept above the floor only.

* A single-camera label moves **along its camera ray**, so it still
  reprojects onto its 2-D label. Without a camera it moves straight up or
  down.
* The head moves as one: if any head label would go under the floor or out
  of its range, the whole head slides along the rays. The floor wins when
  the ranges disagree. Before, only the ears' mean height was checked, and
  K7's nose went under the floor in 5 % of frames.
* `<marker>_z_fixed` marks the frames moved; the report's `heights_fixed`
  counts them per marker.
* The form's 3-D smoothing applies it after smoothing too.

Files made before this need no rerun of 3-D inference:

```
mufasa-pose3d heights K7_3d.parquet --out K7_3d_h.parquet [--scale 1.17] [--model none] [--camera camera.toml]
```

It takes the camera and the scale from the file (the scale otherwise from
nose to tail base). On K7 it moved the nose in 4.7 % of frames, `head_back`
in 1.6 % and `back_T4` in 0.8 %.

## The 3-D viewer: video in the 3-D view

With a camera in the file (every 3-D inference run has one) and a video,
*Video in 3-D* draws the current frame on the floor of the volume, mapped
through the calibration — undistorted first when the lens has distortion.
Markers on the floor sit on their image in every view; raised markers
(head, back) show their height as parallax away from the camera.

**Camera** renders through the calibration itself: the frame fills the
view as recorded and every marker sits on the animal whatever its height.
Wheel zooms and right-drag pans without losing the alignment; left-drag
orbits away from it (press *Camera* to return). *Top* looks straight down
on the floor; the video under it is then a map of the box.

The opacity box sets how strongly the video shows; *Video panel* hides the
separate picture below (its playback controls stay). *Reprojection* still
draws the 3-D markers on that picture — now with the lens distortion
applied, so they also line up on distorted videos.

## The 3-D viewer: view options and models

*View & models* (patch 122kv) opens a panel below the volume.

**View.**
* Markers on or off, and their size.
* Marker names, the skeleton, the floor grid and an x/y/z gizmo.
* Perspective or orthographic projection, and the field of view.
* The background: the theme's, dark, light, black or white.

**Models.** Models are drawn in the volume. Each has a style (solid,
transparent, wireframe or points), an opacity and a colour, and can be
hidden by unticking it.

* *Virtual rodent* adds the built-in rat (DeepMind's virtual rodent; see
  *Anatomical limits*). *Load…* takes a body-model rig (`.rig.npz` from the
  Body model tab). Either is **posed to the 3-D markers** of each frame,
  using the markers it shares with the model (the rat's labels):
  * heading and position, and the rig's posture values within their
    limits, by least squares;
  * scaled to the animal's size: the back's length seen from above,
    against the model's.

  A frame is posed shortly after it is shown, starting from the nearest
  posed frame. *Fit all frames* poses all of them up front, for smooth
  playback. The panel shows the marker error. A rat rearing upright goes
  past the rig's 60° rearing limit and fits less well.
* *Load…* also takes a mesh (`.obj`, `.stl` ASCII or binary, `.ply` ASCII
  or binary): an arena, a nest, objects. It is placed with a scale (e.g.
  1000 for a mesh in metres), a turn about the vertical and an offset in
  x, y and z. Meshes over 20,000 faces are simplified.

**Real time (patch 122ky).**
* *GPU* (View): the models are drawn with OpenGL 3.3, into an off-screen
  picture with a depth buffer and smoothed edges, laid under the markers.
  The shader projects exactly as the viewer does, in every view, so models
  stay on the markers. Solid models hide what is behind them, faces and
  models alike. Without OpenGL (no driver, a remote session), the models
  are drawn on the CPU as before, and the panel says why.
* Posing: *Fit all frames*, and *Fit in the background* when a rig is
  added, pose the rig to every frame in a background thread. A small batch
  around the frame shown comes first, then batches of 2048 frames, always
  the batch nearest the frame shown, so playback catches up. The batches
  run in PyTorch, on a CUDA GPU when there is one:
  * the same unknowns, terms and cold starts as the per-frame fit;
  * every 8th frame from all cold starts, the frames between from the
    nearest solved frame;
  * a last pass from neighbouring frames.

  On K7 this gives the same marker error as the per-frame fit, at 28 ms a
  frame on a 2-core CPU against 220 ms. While it runs, the viewer shows the
  nearest posed frame instead of fitting frames itself. *Fit all frames*
  again stops it.
* The rig stays on top of the floor (122le). Before, the fitted mesh went
  under the floor in 97 % of K7's frames, by up to 10 cm. Mostly that was
  the legs and the tail. The rig's legs cannot bend (they keep the tucked
  stance), so when the spine flexes 3–5° a joint the forelimbs swing
  30–40 mm into the floor. Now:
  * **the trunk** (pelvis, spine, neck, head) never goes below the floor.
    The fit has a penalty for 37 *floor probes*, the lowest trunk vertices
    below z = 0, and after the solve the body is lifted by whatever trunk is
    still under the floor;
  * **the limbs and the tail** that reach the floor are drawn lying on it,
    as a flexed leg or a dragged tail would. They do not lift the body off
    its markers;
  * marker heights are clipped to the body's range before fitting, so old
    files pose sensibly too.

  The per-frame and batched fits do the same (`mufasa/pose3d/rig_floor.py`).
  On K7 the trunk never goes under the floor, and the marker error is
  28.7 mm (median; 27.1 before). Lifting the whole body instead cost 37 mm.
  The limbs or the tail reach the floor in 98 % of frames, 25 mm deep at
  the median. Legs that bend to the floor would need leg inverse
  kinematics, which the rig does not have yet.
* The viewer works out the volume's extent once. It was recomputed over
  every frame on each paint (0.4 s a paint on a 108,000-frame recording).
* The virtual rodent's mouth is closed. The skin was bound with the jaw
  open, wider than the jaw joint's range closes, so the stance pose sets
  the jaw at −40°, where the lips meet. Rigs exported from the Body model
  tab have it too. The cached built-in rig is rebuilt
  (`virtual_rodent_v2.rig.npz`).

A saved camera animation keeps the view options and the models, and
renders with them. On the command line:

```
mufasa-view3d VIDEO X_3d.parquet --model virtual_rodent --model arena.obj
```

## Gait: paws, weight distribution and the centre of mass

*Gait (paws from a body model)* on the 3-D tab (patch 122kw;
`mufasa-pose3d gait X_3d.parquet --out X_gait.parquet`) estimates where
the paws are, how the weight is shared between them, and the gait. It
works from the 3-D body markers alone: the paws are not labelled, so they
are inferred from what the body can do.

**The physics model** is DeepMind's virtual rodent:
* Each segment's mass comes from its collision shapes (density × volume):
  338 g in all, heaviest at the pelvis and abdomen. Each segment's centre
  of mass is tied to its nearest markers on the standing model, so the
  whole-body centre of mass follows the markers when the animal crouches,
  rears, turns its head or swings its tail. Over sampled spine postures it
  is within about 3 mm of the model's own.
* The hip and shoulder joints (the limb girdles) are tied to the markers
  in the same way.
* Each leg's reach, from girdle to paw over the joint ranges, is 86 mm
  for the hindlimbs and 65 mm for the forelimbs, at the model's size.

**The body in three sections** (patch 122lf): lower, mid and upper.

| section | segments | mass (model) | heading from |
|---|---|---|---|
| lower | pelvis, hind legs, tail | 212 g | `back_V2` → `back_L2` |
| mid | thorax, lumbar spine | 74 g | `back_L2` → `back_T4` |
| upper | shoulders, forelegs, neck, head | 52 g | `back_T8` → `head_back` |

* Each section has its own centre of mass, and its velocity and momentum
  split along its heading (forward), across it (sideways, + to the
  animal's left) and up.
* A heading is taken only where that stretch of the back is level enough
  to give one from above (within 60° of level). When the animal rears, the
  heading is held from the frames on either side.
* The hips move with the lower section and half the trunk. The shoulders
  move with the thorax: the head and neck swing on their own while the
  animal sniffs. Taking them in, or using one momentum for the whole body,
  put the forepaws' left–right phase further from the CatWalk's 0.5.
* Stance, swing and stride follow each girdle's **forward** speed, and the
  stride clock the body's. Sideways drift, swaying and turning on the spot
  no longer count as walking speed.

New columns in `<video>_gait.parquet`, all in mm/s, mN·s (kg × mm/s), °
and °/s:

* the body: `v_fwd_mm_s`, `v_lat_mm_s`, `yaw_rate_deg_s`, and the flags
  `heading_ok`, `turning` (over 90°/s) and `sidestep` (sideways speed over
  half the forward speed);
* per section: `<s>_com_x/y/z`, `<s>_heading_deg`, `<s>_yaw_rate_deg_s`,
  `<s>_v_fwd_mm_s`, `<s>_v_lat_mm_s`, `<s>_v_z_mm_s` and
  `<s>_p_fwd_mNs`, `<s>_p_lat_mNs`, `<s>_p_z_mNs`.

Steps get `v_fwd_mm_s` and `straight`. The walking strides in the summary
(stride, stance, duty factor, phases) are straight ones only: level, not
turning, not stepping sideways.

**Paw placement.** Paws are placed as rarely as the body allows, in the
order and at the times rats use (patch 122kx):
* A stride clock turns once per stride, at speed ÷ stride length, and
  stands still when the body does. Each paw's stance takes the first
  *duty factor* of its turn. The paws are offset as uninjured rats walk:
  hind left 0, hind right ½, fore left 0.39, fore right 0.89 (a lateral
  sequence: each forepaw lands about 0.39 of a stride after its hindpaw).
* A planted paw lifts when its turn's stance is over, once the paw on the
  other side has been down for (duty − ½) of a stride. It then swings
  until its next turn comes round (0.08–0.24 s), so a late paw catches up.
* **Scanning forward**, it lands where its girdle's place will be at
  mid-stance, half a stance later, read from the body's actual path rather
  than extrapolated from its speed. The place is beside the midline
  between the girdle pair, half the base of support out across that
  girdle's heading. A still body's paws land under the girdles.
* **Scanning backward**, once every stance's start and end are known, each
  paw is moved to where it stays closest to its place over the whole
  stance (the mean of its places), within its leg's reach at every frame.
  If no such point exists, the forward pass's place stays.
* Steps off the clock: a paw also lifts once its leg cannot reach it, or
  once it has fallen half a stance (at least a quarter of a stride) behind
  its place along its girdle's heading. Such a step must keep the body balanced (Hof 2008): the
  extrapolated centre of mass (position + velocity / √(g/h), which takes
  in momentum) must lie over the other planted paws and the place the paw
  will land, as it will be when it lands.
* No paw lifts before a minimum stance.
* A girdle too high for its leg to reach the floor (forelimbs when
  rearing) keeps that paw in the air.

On K7 (108 000 frames, mostly slow exploration), measured in each girdle
pair's own frame at mid-stance:

| | before (122kx) | 122lf |
|---|---|---|
| steps | 3138 | 3024 |
| fore–aft offset from the girdle, spread (MAD) | 7.7 mm | 3.9 mm |
| sideways offset from its place, 90th percentile | 28.4 mm | 12.8 mm |
| paw past the midline | 21 % | 8.5 % |
| hind → fore phase (CatWalk 0.39) | 0.33 | 0.39 |
| fore left → right phase (CatWalk 0.5) | 0.32 | 0.40 |
| ZMP outside the support | 72 % | 46 % |
| time | 264 s | 116 s |

What is left past the midline is mostly long, slow stances (1.1 s median)
in which the body pivots over a planted paw. A lift whenever a paw drifted
across was tried: it added 1300 short stances and left almost as many
paws past the midline (221 against 259), so it was dropped.

**Weight distribution.** The vertical load m(g + a_z) is shared among the
planted paws:
* Their centre of pressure sits at the zero-moment point,
  ZMP = CoM − h·a/(g + a_z), which accounts for acceleration.
* The sharing is as even as that allows, with no negative loads.
* If the ZMP falls outside the support (e.g. a rat rearing against a wall,
  whose forepaws rest on the wall, which the model does not know about),
  the stability margin is negative.

**Bounds** come from 108 uninjured young adult female rats crossing a
CatWalk: the pre-injury runs of Neckel et al. (2013), 5780 strides at
13–71 cm/s. They depend on speed v (mm/s, held at 100–710):

| Bound | Value | Seen (5–95 %) |
|---|---|---|
| Stance, forelimbs | 13.9 · v^−0.723 s (0.41 s at 13 cm/s, 0.20 s at 35, 0.12 s at 70) | 0.12–0.35 s |
| Stance, hindlimbs | 26.3 · v^−0.830 s | 0.12–0.36 s |
| Swing | 0.366 · v^−0.164 s (fore), 0.151 · v^−0.019 s (hind): 0.13–0.16 s | 0.10–0.20 s |
| Duty factor | stance ÷ (stance + swing): 0.7–0.75 at 13 cm/s, 0.6 at 35, 0.5 at 60 | 0.45–0.75 |
| Stride | 65 mm + 0.156 s × v, ≤ 187 mm | 82–163 mm |
| Interlimb phase | left–right ½; LH→LF 0.39; LH→RF 0.85 | 0.47–0.54; 0.33–0.45 |
| Base of support | 23 mm fore, 29 mm hind | 12–34, 18–38 mm |
| Shortest stance | 0.10 s | 2nd percentile |

TactileArena's demo recording (Lu et al. 2026; one rat, 58 stances,
13–25 cm/s) agrees within the CatWalk's scatter: strides within 10 mm and
duty factors within 0.05 of the CatWalk laws at the same speed. Its fixed
bounds remain available (*Bounds: TactileArena*; `--bounds tactilearena`).
Mouse values (Mendes et al. 2015) are kept for reference.

The bounds are adult-rat values. For a juvenile, a mouse or an unusually
large rat, `--gait-scale k` scales lengths by k and times by √k (dynamic
similarity). `scripts/calibrate_gait_catwalk.py` re-fits the laws from the
Mendeley file (`Neckel_CatWalk_RRMPS.mat`, or its split parts) and writes
`mufasa/assets/gait/catwalk_rat_reference.json`: the laws, medians by
speed, percentiles and phases.

On synthetic walks at 60 fps, the estimated gait reproduces the laws:
strides within 8 mm and phases within 0.03 at 15, 30 and 50 cm/s. At
30 fps, walking faster than about 45 cm/s is near the frame rate's
limit: a 0.15 s stance is 4–5 frames.

**Size and mass.** The animal's size against the model comes from the
straight distance from the nose to the tail's base, with the head low.
(Patch 122kw used the path along the back. A dataset's back labels need
not sit where the model's do, and noise lengthens a path through them;
on K7 that gave 1.4× the model, 916 g. Nose to tail base gives 1.2×.)
Size sets the body's geometry: the girdles, the legs' reach and the mass. Its mass is the model's × size³ unless you
give the weighed mass, which gives forces in grams-force. Size³ is only a
guess: weigh the animal.

**Output** in `derived/gait/<run>/`:
* `<video>_gait.parquet`, one row per frame: each paw's position,
  contact (1 down, 0 swinging, −1 its leg cannot reach the floor), load
  share and force; the centre of mass, its speed, the ZMP, the
  extrapolated centre of mass, the ZMP's margin inside the support, and
  the girdles. The summary is in the file's metadata.
* `<video>_steps.csv`, one row per stance: paw, touch-down and lift-off,
  position, stance and swing time, stride and duty factor.

The summary gives the median speed, the share of frames on 0–4 paws, each
paw's mean load share, strides, stance times and duty factors, and the
phases (LF→RF, LH→RH, LH→LF, LH→RF). In the 3-D viewer, *View & models →
Load… → `<video>_gait.parquet`* draws the planted paws (sized by load),
the support polygon, the centre of mass over the floor and the ZMP (green
inside, red outside). It is an overlay next to the virtual rodent. It does
not move the model's legs, which stay in the rig's tucked stance.

**Limits.**
* A single camera's heights are uncertain, and the centre of mass and the
  girdles inherit that.
* The model knows only the floor: walls and climbing are unsupported.
* Footfalls are inferred, not seen. Check them against a pressure floor or
  labelled paws where you can (TactileArena's data has both).
* The timing follows rats walking straight along a runway at 13–71 cm/s.
  Slower than that the duty factor is held at its slowest value and the
  stride grows longer in time. Under about 2 cm/s (a stride longer than
  4 s) the clock stops, and turning on the spot or shifting the weight is
  stepped off the clock, by reach and balance alone.

## Camera animation

Tick **Animation** in the 3-D viewer to open the editor (patch 122kp).

1. Go to a frame and set up a view (orbit, zoom, pan, or *Camera*), then
   press **Add key**. Repeat for each key. The key is inserted after the
   selected one. The move from the previous key gets a real-time length:
   the difference in frames. If two keys are on the same frame, the move
   gets 2 s, and the data freezes while the camera moves.
2. In the table, set each key's *Video frame*, its *To next* length (in
   output frames, with seconds shown) and its *Easing*:
   * *Ease in-out* starts and stops gently at each key;
   * *Linear* moves at a constant rate;
   * *Spline* is a smooth curve through the keys that does not stop at
     them.
   **Update key** sets the selected key to the current view.
   Double-clicking a row or pressing **Go to** shows that key.
3. Drag the timeline or press **Preview** to play the move in the viewer.
4. Press **Render video…**. You can set the size, the fps, *Video below*
   (the video under the 3-D view) and *Frame label*. The 3-D view's own
   settings are used: box, shadows, height lines, trail, video in 3-D and
   its opacity. The output is .mp4 (mp4v) or .avi (MJPG). The camera path
   is saved beside it as `.camera.json`. **Save…**/**Load…** keep paths
   for later.

The data frame moves linearly between keys, so a length longer than the
frame difference plays in slow motion. The azimuth turns the short way
round, and the zoom changes at an even rate. A move between two *Camera*
keys stays in the recording camera's view and only zooms and pans. A move
into or out of it blends the two projections, so there is no jump. Render
a saved path without the GUI (e.g. on the server):

```
mufasa-pose3d render-animation K7.camera.json --pose K7_3d.parquet --video K7.mp4 --out K7_anim.mp4 [--size 1920 1080] [--fps 30]
```

## Motion capture (C3D)

*Import motion capture (C3D)* (tab) / `mufasa-pose3d import-c3d` reads
`.c3d` files from motion-capture systems (Vicon, Qualisys, OptiTrack,
Cortex, …) into `sources/pose3d/<trial>_3d.parquet`: millimetres, z up,
one row per mocap frame, `_likelihood` 1 where the marker was measured and
0 where it was missing, the residual in `_residual_mm`. They open in the 3-D
viewer (choose *(no video)*, or a video with the same name) like 3-D
inference runs — e.g. as ground truth for the single-camera estimate.

* *Up axis in the file*: most systems use z up; OptiTrack and some Vicon
  setups y up — that axis becomes z.
* *Rename markers* `HeadF=head_nose, SpineM=back_T8`: project names give the
  rat skeleton in the viewer; otherwise give *Skeleton* lines `a-b, b-c`.
* *Floor height in the file* (`--floor-z`): where the lab's floor is, after
  turning to z up. It is moved to z = 0, which gait needs. The import says
  so when markers lie below the floor (patch 122kz).
* The reader needs nothing beyond NumPy: integer and floating-point files,
  Intel, DEC and big-endian (MIPS/SGI) processors, labels beyond 255
  (`LABELS2`…), units in m / cm / mm, more than 65 535 frames. Analog
  channels (force plates, EMG) are not imported.

```
mufasa-pose3d import-c3d trial01.c3d trial02.c3d --project project.toml [--up y] [--rename HeadF=head_nose] [--edges a-b,b-c]
```

## Output

`derived/pose3d/<run_id>/<video>_3d.parquet`: `frame`, per marker
`_x/_y/_z` (mm), `_likelihood`, `_reproj_px`; single-camera runs add
`pelvis_pitch_deg`, `lumbar_angle_deg` and `spine_length_mm`. Metadata
(`mufasa_pose3d`): camera (3×4 `P`), box, skeleton edges, method, a
per-video report. `run.toml` and `report.json` sit beside them.

## The 3-D pose file

Patch 122kz writes down the contract every 3-D stage relies on
(`mufasa/pose3d/io.py`):

| Column or key | Meaning |
|---|---|
| `<marker>_x`, `_y`, `_z` | millimetres; z up; **the floor at z = 0** |
| `frame` | the video frame of each row, in order, each once |
| `<marker>_likelihood`, `<marker>_reproj_px` | optional, per marker |
| other columns (`lumbar_angle_deg`…) | optional, per frame |
| metadata `mufasa_pose3d` | `units`, `fps`, `method`, `camera`, `edges`, `box_x/_y/_z`, `report` |

Every reader finds markers with one rule (`pose_markers`): a name with `_x`,
`_y` and `_z` columns. A name `<m>_var` is skipped when `<m>` is a marker. These
are the per-axis variances the Kalman v2 smoother writes into 2-D pose files.
Single-camera runs made from a smoothed 2-D file once reconstructed them as
markers of their own (K7: 15 such, up to 2,000 km away, which set the 3-D
viewer's volume). Such files now read as their real markers without being
rebuilt.

`check_pose3d` lists what a file breaks: no frame rate, frames out of order
or missing, markers below the floor, units other than mm, derived columns.
Gait reports it with its results, and says when it assumed 30 fps.

## 3-D features

*Compute Features* has a **3-D features** group (patch 122ko). Each family
reads the video's 3-D pose file: by default from the newest
`derived/pose3d/<run>` that has the video, then `sources/pose3d/` (motion
capture). You can also pick one run, or motion capture, under *3-D pose
from*. The output goes to `derived/features/<family>/<video>.parquet`, like
the 2-D families, with one row per video frame:

| Family | Columns |
|---|---|
| 3-D BODY-PART HEIGHTS (MM) | `Height (mm) <marker>`: height above the floor |
| 3-D TWO-POINT BODY-PART DISTANCES (MM) | `3-D distance (mm) a-b` for every marker pair |
| 3-D FRAME-BY-FRAME BODY-PART MOVEMENTS (MM) | `3-D movement <marker> (mm)`: distance moved since the previous frame |
| 3-D BODY POSTURE ANGLES (DEGREES) | head pitch, head roll (right side up is positive), body pitch (`tail_V6` → `head_back`), head pitch relative to the body, `Spine bend at <joint>`, pelvis pitch, lumbar angle |

`Spine bend at <joint>` is the change in elevation, in the body's heading
plane, from the segment behind the joint to the segment in front of it. It
is positive when the back is concave (bending up toward the head) and
negative when it is hunched. An angle is left out when its markers are
missing.

Frames that are missing from the 3-D file are written as NaN for heights
and angles, and as -1 for distances and movements (the 2-D convention).
Videos without a 3-D file get no 3-D features, and this is reported. If
you ask only for 3-D families and no video has a 3-D file, the run stops
with an error.

## Command line

```
mufasa-pose3d calibrate-box marks.json --width 300 --camera-height 500 --out sources/calibration/camera_K7.toml
mufasa-pose3d calibrate-checkerboard board.mp4 --pattern 9 6 --square-mm 20 --out lens.toml
mufasa-pose3d calibrate-box marks.json --length 554 --width 300 --intrinsics lens.toml --out camera_K7.toml
mufasa-pose3d infer --project project.toml [--video STEM ...] [--no-kalman] [--stereo camA camB] [--workers N] [--device cuda]
mufasa-pose3d gpu-check
mufasa-pose3d infer --pose labels.parquet --fallback kv2.parquet --camera camera_K7.toml --out K7_3d.parquet
mufasa-pose3d smooth K7_3d.parquet --camera camera_K7.toml --out K7_3d_smooth.parquet
mufasa-pose3d heights K7_3d.parquet --out K7_3d_h.parquet
```

## Solver: GPU, PyTorch or SciPy

The head and spine-chain fits (most of the time) run on

* **GPU** — PyTorch with CUDA; several videos share the GPU (up to 4 at
  once, in fresh processes);
* **CPU (PyTorch)** — the same code on the CPU, about 2× faster than SciPy
  per core;
* **CPU (SciPy)** — the original.

The two PyTorch solvers need PyTorch 2.7 or later in Mufasa's environment
(`pip install "mufasa[pose3d]"`, or `".[qt,pose3d]"` from a checkout).
Without it only SciPy is offered, and the greyed-out choices say why. On
Linux, PyTorch 2.11+ from PyPI is a CUDA 13 build, which needs a driver
whose `nvidia-smi` shows CUDA 13. With an older driver, first install
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.
RTX 50xx (Blackwell) cards need 2.7+ built for CUDA 12.8 or later.
`mufasa-pose3d gpu-check` names what is missing: PyTorch, a CUDA build,
a hidden GPU (`CUDA_VISIBLE_DEVICES`), the driver, or kernels for the card.

*Solver* (tab) / `--device auto|cuda|torch-cpu|cpu` (CLI) /
`MonoConfig.device`; *auto* (default) takes the first available. The
PyTorch solver is a port of SciPy's trust-region solver (same steps, exact
derivatives), so results match within rounding (K7, 13 000 frames: median
difference 2·10⁻⁵ mm, 99th percentile 4–5 mm in poorly constrained
frames). PyTorch comes with the `gpu` extra (`pip install -e '.[gpu]'`);
an RTX 50xx (Blackwell) needs a CUDA ≥ 12.8 build. Check a machine with

```
mufasa-pose3d gpu-check
```

(a synthetic run on CPU and GPU: time and agreement).

## Workers

*Workers* (tab) / `--workers` (CLI; default: all cores but one) /
`RunSettings.workers`: several videos run at once, one process each; a
single video splits its 6000-frame head and spine chunks over the
workers instead. Each worker needs about 1.5 GB for a 30-minute video.

Each worker runs NumPy, SciPy and PyTorch with **one thread** (patch 122lc).
Before 122lc each worker's libraries started a thread per core. 8 workers on
a 16-thread machine reached a load average of 118, with the CPU busy but
spinning: 10 videos took 14 hours instead of minutes. Launching Mufasa with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` works around it
on older versions. A worker that dies (out of memory, stopped from outside)
ends the run with an error that says how many videos are done and where.

## Model data

`mufasa/assets/models/pose3d/rat7m_trunk_prior.npz` — 80 000 frames of
Rat7M motion capture (Dunn et al. 2021, *Nature Methods*; CC BY 4.0),
rebuilt by `scripts/build_rat7m_prior.py`.
`mufasa/assets/models/pose3d/virtual_rodent/` — DeepMind's virtual rodent
(dm_control `rodent.xml` and skin, Apache 2.0) and the limits worked out
from it (`scripts/build_virtual_rodent_limits.py`). See
`mufasa/assets/models/pose3d/README.md`.

## Limits

* One camera cannot see roll or depth directly; heights between the head
  and the floor come from Rat7M posture statistics, not from the image.
* When a rat rears, its back is nearly vertical and spine labels stack
  above each other from overhead; a few pixels of label noise can still
  bend the back backwards in some frames.
* The camera height must be measured (or solved with a checkerboard).
