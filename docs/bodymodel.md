# Body model

The **Body model** tab (patch 122kq) fits a rigged 3-D model (a Blender
armature with the mesh it deforms) to a top-down video, frame by frame. It
uses the animal's silhouette and its 2-D labels, and keeps every joint
within its limits. Optionally it renders the model over the video.

## 1. Import rig (.blend)

Choose the `.blend` file and the names of the armature and the skin mesh
(`Armature` and `Skin` in the HBP mouse). The rig is saved as arrays in
`sources/bodymodel/<name>.rig.npz`. This step contains:

* the bones, parents first, with their rest matrices;
* the skin, both full and decimated (the decimated copy is used for
  fitting);
* the skinning weights;
* the fur colour of each face, taken from the skin's image texture.

Reading a `.blend` needs Blender's Python module (`pip install bpy`, which
requires a matching Python version) or a Blender executable. Either runs in
a separate process. On the command line:

```
mufasa-pose3d body-rig full_mouse_blender5.blend sources/bodymodel/mouse.rig.npz [--blender /path/to/blender]
```

The fitter expects the bone names of the HBP mouse after its update to
Blender 5: `Pelvis`, `Spine1`–`Spine17`, `Neck1`–`Neck8`, `Head`,
`Tail1`–`Tail23`, `Femur.L/.R`, `BPhalange1–5.L/.R`. Bones it does not know
keep their rest pose.

### The rat: DeepMind's virtual rodent

*Model: Virtual rodent (rat, built in)* (patch 122ks) makes a rig from the
virtual rodent that ships with Mufasa (dm_control's `rodent.xml` and skin,
Apache 2.0; see `docs/pose3d.md`). It needs no Blender:

```
mufasa-pose3d body-rig --virtual-rodent sources/bodymodel/rat.rig.npz
```

The rig carries its own *profile*: the regions of its spine, its joint
limits, and where each label sits on it. The fitter uses those in place
of the mouse's:

| Joint | Limit | Source |
|---|---|---|
| Lumbar (6 joints) | sideways ±10°, up/down ±15° per joint (+ flexes) | the model's two bend joints (±30°) and two extension joints (−30…45°, + flexing): 90° of flexion in all. Extension goes past the model's 60° to 90°, for its rigid thorax (patch 122lg) |
| Thorax | rigid | the model |
| Cervical (7 bones) | sideways ±10°, up/down −10…+15° | the model's three extension joints (−23…35°) and two bend joints (±35°) |
| Head (atlas) | up/down −45…+5°; sideways ±30° | the model; sideways assumed |
| Tail (29 joints) | ±8° per joint (the model allows ±15°); the first fixed | the model |
| Body | rearing −10…60°, roll ±15° about the hind paws | assumed |
| Width | `widen` 0…0.5 per frame (the trunk up to 1.5× wider) | for heavy rats; see *Width* |
| Length | `stretch` −0.3…+0.4 per frame (the spine and neck 30 % shorter … 40 % longer) | hunched to stretched; see *Length* |

The rig stands with its limbs tucked under the body (patch 122ku;
`rodent_rig.STANCE`). In the model's own zero pose the legs splay sideways
and stick out of a rat's outline seen from above. The angles are within
every joint's range, with all four paws on the floor; the fitter keeps
the limbs in this pose.

The labels sit where `mufasa.pose3d.rodent_model` puts them (the skin over
the vertebrae, the crown for the ears, the hip joints, caudal vertebrae).
So the scale does not depend on the head's size; this matters because a
large rat's head is small for its body. The model is a slim rat standing
on all fours. A rat curled up or lying flat covers more floor than the
model, and the silhouette then overlaps less.

**Width** (patch 122kt). The virtual rodent is a slim rat standing on all
fours; a heavy rat sitting or curled up covers more floor. Two settings
handle this:

* *Girth* (Import rig; `body-rig --virtual-rodent --girth 1.3`) gives a
  heavier base shape. The trunk is widened sideways from the thorax to the
  pelvis, tapering at the neck and the tail base. The head, the tail, the
  paws and the back's midline (where the labels sit) keep their size.
* Each frame also has a posture value `widen` (0…0.5): the trunk up to
  1.5× wider than the rig, with a weak pull back to the rig's width. A rat
  that curls up can then widen without the whole video taking that width.

**Length** (patch 122lg). The model's thorax is one rigid body and its
bones keep their lengths, so bending alone barely changes how long it is.
A rat's length changes a lot between hunched and stretched. On K7 the
back's path from `head_back` to `back_V2` spans 162–291 mm (5–95 %), while
the fitted model stayed at 173–184 mm. Its marker error grew with the
stretch: 34 mm on the longest tenth, against 25 mm on the shortest.

* Each frame has a posture value `stretch` (−0.3…+0.4), pulled weakly to
  0. Each lumbar and cervical joint and the atlas slides along its bone
  from its parent's joint, so the spine and neck are 30 % shorter … 40 %
  longer. Nose to tail base goes from 0.8× to 1.27× the model's. The
  pelvis, hind legs and tail stay where they are, and the forelegs move
  with the thorax.
* It stands in for what the model cannot do: bend its thorax, slide its
  scapulae, straighten the curve of its neck.
* Before this patch, the lumbar limits had the model's flexion and
  extension the wrong way round (+10° flexion, −15° extension per joint).
  Hunched rats sat at the flexion limit in 27 % of K7's frames.

With both changes, the model's length on K7 follows the animal's (148 mm
on the shortest tenth, 226 mm on the longest). Marker error is now
25–28 mm whatever the length: 26 mm on the longest tenth (34 before), 25
on the next 15 % (29 before).

A rig opts in by listing the bones whose joints slide as `stretch` in its
profile (the virtual rodent: `Spine1`–`Spine6`, `Neck1`–`Neck7`, `Head`)
and `stretch` among its `posture` values. The HBP mouse has none.

After a fit, the log gives the mask's area against the fitted body's, and
for the virtual rodent a girth to try when the animal covers more than
1.15× the model. The value is also stored in the fit file's metadata as
`area_ratio`. Width and scale trade off against each other. The labels fix
the length, so they keep the two apart, but only as well as the labels
are placed.

Any rig can carry a profile: a `profile` entry in the `.rig.npz` (JSON)
with `groups`, `head`, `trunk`, `tail`, `tail_hidden`, `paws`, `hips`,
`posture` (name, low, high; include `widen` if the rig stores a widening
shape `D`/`Dd`, and `stretch` if it lists `stretch` bones), `prior_centre`,
`limits` and, optionally, `stretch` and `markers`. See
`mufasa.bodymodel.fit.HBP_PROFILE`. Rigs without one are taken to be the
HBP mouse.

## 2. Fit body model

Pick the rig, the video, and the 2-D labels: the imported ones, or a
Kalman v2 run. Then set the frames: *from*, *to*, and *every* n-th frame.

**Posture.** Each frame has these numbers:

* position and heading on the floor;
* the scale, which is one value per video;
* rearing and roll about the hind paws, which stay on the floor;
* sideways and up/down bend of the lumbar, thoracic and cervical spine
  (the same angle at every joint of a region);
* head yaw and pitch;
* the tail: sideways for the near half and for the far half, plus up/down;
* for rigs that have them, `widen` and `stretch`.

The limbs keep their rest pose.

| Joint | Limit per joint | Source |
|---|---|---|
| Lumbar (Spine1–6) | sideways ±5°, up/down ±6° | Dewan et al. 2010 (HBP mouse model) |
| Thoracic (Spine7–17) | ±5° | HBP mouse model |
| Cervical (Neck1–8) | sideways ±5°, up/down ±9° | Shahrokni et al. 2012 (HBP mouse model) |
| Head | yaw ±30°, pitch ±30° | assumed |
| Tail (Tail4–23) | ±20°; Tail1–3 fixed | HBP mouse model |
| Body | rearing −10…60°, roll ±15° | assumed |
| Floor | nothing below it | — |

**What a frame is fitted to.**

* **Silhouette.** The animal's mask is the pixels much darker than the
  median background (dark animals, e.g. under IR) or much lighter (a white
  rat on darker ground). *Animal* chooses; *auto* compares the pixels under
  the confident back labels with the background. For a light animal, only
  the region around the confident labels counts, and of that the part that
  holds most of them: moved bedding also differs from the background. The
  model's body is drawn without the tail and compared to the mask by IoU.
  A mask is used only when it is plausible: not touching the frame edge,
  and 0.45–3 times the model's area.
* **Labels.** Likelihood-weighted, robust distance between each 2-D label
  and the same point on the model. The chain labels (`head_nose`,
  `head_mid`, `head_back`, `back_T4` … `tail_V32`) are placed along the
  model at the spacing the labels themselves show over the video. Their
  scale is anchored on nose → back of the head. The ears give
  `head_left`/`head_right`, and the knees give the hips.
* A weak pull towards a neutral posture, for joints that can't be seen.

**Scale.** By default it is estimated per video: a first value from the
labels (corrected by the animal's area in the masks when the two disagree
by more than 15%), then a fit with the scale free on a few frames that
have a usable mask, taking the median. That estimate cannot tell a wider
animal from a bigger one, and it varies with what the animal does in the
frames it uses.

*Scale from: 3-D run …* (patch 122ku; `--scale-from VIDEO_3d.parquet`)
takes it from a single-camera 3-D run of the same video instead. It
multiplies two numbers:

* the camera's pixels per mm at the back's median height and place;
* the animal's size against the model: the back's length seen from above
  (`head_back` → `back_V2`, median over the frames with the head low)
  over the same labels on the model.

The 3-D run's camera file must still be where the run found it. The
numbers used are stored in the fit file's metadata (`scale_from`). You can also set it in px per model unit (the
mouse model is in mm).

**Over time** (*Forwards and backwards*, *Smoothness*):

* **Forwards:** each frame starts from the previous frame's posture and
  is pulled towards it. If it loses track (silhouette or labels far off),
  it is searched again from scratch, with both headings.
* **Backwards:** each frame is refined again, pulled towards both of its
  neighbours. The new posture is kept only if it is better.
* *Smoothness* sets the strength of the pull. 0 fits frames on their own.

This helps most where labels are briefly wrong, or the animal is blurred
or partly hidden.

**Speed.** A fitted frame takes a few seconds of CPU. The run is split into
chunks of 300 frames (overlapping by 8) on the given number of workers; each
overlapping frame is taken from the chunk where it is furthest from an edge.
Use *every* n-th frame for long stretches. The overlay interpolates between
fitted frames.

**Output.** `derived/bodymodel/<run>/` contains:

* `<video>_fit.parquet`: one row per fitted frame, with the 17 posture
  values, `iou`, `kp_rms_px` (label error), `mask_used`, the loss terms,
  and `restarted`. The rig, video, labels, scale, limits and settings are
  in the file's metadata.
* `<video>_overlay.mp4` (with *Render the model over the video*): the
  model shaded with its fur colours, its outline, and the labels (green)
  against the model's points (magenta). Optionally side by side with the
  original.
* `run.toml`.

On the command line:

```
mufasa-pose3d body-fit mouse.rig.npz VIDEO.mp4 LABELS.parquet --out fit.parquet \
    [--start 0 --stop 9000 --step 3 --workers 8] [--no-refine] [--smoothness 0.5] [--scale 4.9] \
    [--overlay overlay.mp4 [--side-by-side]] [--animal auto|dark|light] [--scale-from VIDEO_3d.parquet]
```

In Python, `mufasa.bodymodel.fit.Rig.local_quaternions(params)` gives each
bone's rotation as Blender pose bones use it, to drive the rig in Blender.

**Limits of the method.**

* The model's shape is fixed (a slim mouse in a walking stance), so a
  crouched, fluffed-up animal overlaps it at an IoU of about 0.5.
* Rearing toward the camera is hard to tell apart from crouching when seen
  from above.
* Limbs are not fitted.
