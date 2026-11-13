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
(`camera_K7.toml` for `K7_cam_…`), else `camera.toml`.

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

## Command line

```
mufasa-pose3d calibrate-box marks.json --width 300 --camera-height 500 --out sources/calibration/camera_K7.toml
mufasa-pose3d calibrate-checkerboard board.mp4 --pattern 9 6 --square-mm 20 --out lens.toml
mufasa-pose3d calibrate-box marks.json --length 554 --width 300 --intrinsics lens.toml --out camera_K7.toml
mufasa-pose3d infer --project project.toml [--video STEM ...] [--no-kalman] [--stereo camA camB] [--workers N] [--device cuda]
mufasa-pose3d gpu-check
mufasa-pose3d infer --pose labels.parquet --fallback kv2.parquet --camera camera_K7.toml --out K7_3d.parquet
mufasa-pose3d smooth K7_3d.parquet --camera camera_K7.toml --out K7_3d_smooth.parquet
```

## Solver: GPU, PyTorch or SciPy

The head and spine-chain fits (most of the time) run on

* **GPU** — PyTorch with CUDA; several videos share the GPU (up to 4 at
  once, in fresh processes);
* **CPU (PyTorch)** — the same code on the CPU, about 2× faster than SciPy
  per core;
* **CPU (SciPy)** — the original.

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
workers instead. Each worker needs a few hundred MB.

## Model data

`mufasa/assets/models/pose3d/rat7m_trunk_prior.npz` — 80 000 frames of
Rat7M motion capture (Dunn et al. 2021, *Nature Methods*; CC BY 4.0),
rebuilt by `scripts/build_rat7m_prior.py`. See
`mufasa/assets/models/pose3d/README.md`.

## Limits

* One camera cannot see roll or depth directly; heights between the head
  and the floor come from Rat7M posture statistics, not from the image.
* When a rat rears, its back is nearly vertical and spine labels stack
  above each other from overhead; a few pixels of label noise can still
  bend the back backwards in some frames.
* The camera height must be measured (or solved with a checkerboard).
