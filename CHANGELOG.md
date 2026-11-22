# CHANGELOG

Working notes on what landed in each session, what's deferred, and what
to pick up next time. Keep entries dated and grouped by patch series so
"where did this behavior come from" is answerable from git log alone.

---

## Session 2026-07-17 — marker names are case-sensitive; the model's layout wins

**122lg — A spine that goes from hunched to stretched.** On K7 the back's path from head to
tail base spans 162–291 mm (5–95 %), but the fitted virtual rodent stayed at 173–184 mm. Its thorax
is one rigid body and its bones keep their lengths, so bending barely changes its length. Marker
error grew with the stretch: 34 mm on the longest tenth.
* Lumbar limits corrected. The model's extension joints flex +45° and extend −30° (90° and 60°
  in all), but the rig had them the wrong way round (+10° / −15° per joint), and hunched rats
  sat at the flexion limit in 27 % of frames. Now ±15° per joint. Extension goes past the model's
  60° for its rigid thorax: stretched rats held 60° in 28 % of frames.
* `stretch` (−0.3…+0.4 per frame, pulled weakly to 0) slides each lumbar and cervical joint and
  the atlas along its bone from its parent's, so the spine and neck are 30 % shorter … 40 %
  longer. Nose to tail base goes from 0.8× to 1.27×; the pelvis, hind legs and tail stay put.
  `Rig.skinning_matrices` and the batched kinematics (`rig_batch`) move the joints alike. A rig
  opts in by listing `stretch` bones in its profile.
* Rig version 3: the built-in rig is rebuilt the first time it is used.
* K7: the model's length follows the animal's (148 → 226 mm across the length tenths). Marker
  error is 25–28 mm at every length: the longest tenth 34 → 26 mm, the next 15 % 29 → 25 mm.
* docs/bodymodel.md: *Length*, and the limits table. docs/pose3d.md.
* `tests/smoke_122lg_spine_stretch.py` (9 checks): with physically valid hunched and stretched
  rats, both fits pose them within 0.1 mm and recover stretch within 0.002. With the length
  fixed, as before, they are off by 19 mm.

**122lf — Gait from the momentum of the lower, mid and upper body; paws placed scanning
forward and backward.** Before this, gait used one speed in any direction, so sideways drift and
turning on the spot counted as walking speed, and paws were placed by one whole-back heading and
the current velocity.
* `BodyPhysics` splits the virtual rodent into three sections:
  * lower: pelvis, hind legs and tail (212 g);
  * mid: thorax and lumbar spine (74 g);
  * upper: shoulders, forelegs, neck and head (52 g).
* Each section has its centre of mass, a heading from its stretch of the back seen from above,
  its turning rate, and velocity and momentum along, across and up. A heading is taken only
  within 60° of level; when the animal rears, it is held from the frames on either side.
* The hips move with the lower section and half the trunk, the shoulders with the thorax
  (`LIMB_DRIVE`). Stance, swing and stride follow each girdle's forward speed; the stride
  clock follows the body's.
* Scanning forward: a paw lands where its girdle's place will be at mid-stance, read from the
  body's actual path.
* Scanning backward: once the stances are known, each paw stands where it stays closest to its
  place over the whole stance, within reach at every frame (`_place_stance`).
* New columns: `v_fwd_mm_s`, `v_lat_mm_s`, `yaw_rate_deg_s`, the flags `heading_ok`, `turning`
  and `sidestep`, and per section `<s>_com_*`, `<s>_heading_deg`, `<s>_yaw_rate_deg_s`,
  `<s>_v_{fwd,lat,z}_mm_s` and `<s>_p_{fwd,lat,z}_mNs`. Steps get `v_fwd_mm_s` and `straight`.
  Walking strides are straight ones only. The summary records section masses, speeds, the
  drive and how often the animal turns or sidesteps (flat, for run.toml).
* `predict_all` and `body_frames` compute the body for every frame at once, 75× faster than
  frame by frame.
* K7, before → after:
  * steps 3138 → 3024;
  * fore–aft spread at mid-stance 7.7 → 3.9 mm;
  * sideways error 28.4 → 12.8 mm (90th percentile);
  * paws past the midline 21 % → 8.5 %;
  * phases hind→fore 0.33 → 0.39 (CatWalk 0.39) and fore left→right 0.32 → 0.40 (0.5);
  * ZMP outside the support 72 % → 46 %;
  * run time 264 → 116 s.
* Tried and dropped:
  * driving the forepaws with the head and neck, or one momentum for the whole body: worse
    phases;
  * lifting a paw whenever it drifted across the body: about 1300 extra stances, with almost
    as many paws still past the midline.
* `tests/smoke_122lf_gait_sections.py` (12 checks).

**122le — Heights the body can have, and the model on top of the floor.**
The virtual rodent clipped through the floor in the 3-D viewer, and single-camera heights near
the ground were sometimes impossible: K7's nose was under the floor in 5 % of frames, and
`head_back` and `back_T4` were out of range in 2–3 %.
* `mono.enforce_heights` is the last step of every 3-D run, after the Kalman smoothing. Each
  label stays within the virtual rodent's height range for it, scaled to the animal plus 10 %,
  and above the floor (z = 0).
  * Single-camera labels move along their camera ray, so they still reproject onto the 2-D
    label; without a camera they move straight up or down.
  * Two-camera runs, and labels the model has no range for, are kept above the floor only.
  * `<m>_z_fixed` flags the frames moved; the report's `heights_fixed` counts them.
  * Project labels map to the model's through the run's anatomy report (now `labels`).
* `Anatomy.limit_head` keeps every head label in range and above the floor, sliding the rigid
  head along the rays. Before, it checked only the ears' mean. The floor wins a conflict.
* The form's 3-D smoothing applies it too. `mono.camera_center_from_meta` replaces the form's
  own copy of the computation.
* `mufasa-pose3d heights IN --out OUT` fixes files already made, without rerunning 3-D
  inference.
* The viewer's rig fit (`pose3d/rig_floor.py`, shared by the per-frame and batched fits):
  * the trunk (the profile's `trunk` bones) never goes below the floor: a penalty for 37 floor
    probes (each trunk bone's lowest vertex plus the 32 lowest), and an exact lift after the
    solve, measured on every trunk vertex;
  * the limbs and the tail are drawn lying on the floor where they reach it. The rig's legs
    cannot bend, and the forelimbs swing 30–40 mm into the floor at 3–5° of lumbar flexion
    per joint; lifting the whole body for them cost 10 mm of marker error;
  * marker targets clipped to the body's heights.

  On K7 the mesh went under the floor in 97 % of frames, by up to 10 cm. Now the trunk never
  does, and the marker error is 28.7 mm (median; 27.1 before). The limbs or the tail lie on the
  floor in 98 % of frames, 25 mm deep at the median: leg IK is the real fix. Only the trunk's
  bones are kept in the batched kinematics, as before.
* docs/pose3d.md: *5. Heights and the floor*; the gait overlay does not move the model's legs.
* `tests/smoke_122le_heights_floor.py` (13 checks).

**122ld — Install dependencies for 3-D pose; saying what is missing.**
Mufasa said PyTorch CUDA was not available. PyTorch was not installed in the user's environment
at all (an RTX 5070 Ti, driver showing CUDA 13.2), so 3-D inference had run the SciPy solver.
* `pyproject.toml`: a `pose3d` extra, `torch>=2.7`, for 3-D inference's PyTorch solvers (CPU and
  GPU), the viewer's batched posing and `gpu-check`. 2.7 is the first release with Blackwell
  (RTX 50xx, sm_120) kernels, in its CUDA 12.8 builds. `gpu` and `all` move from `torch>=2.4`
  to 2.7. On Linux, PyTorch 2.11+ from PyPI is a CUDA 13.0 build; with an older driver, install
  from the cu128 index.
* `gpu.install_hint()` names what is missing, with the fix:
  * no PyTorch;
  * a CPU-only build;
  * the GPU hidden by `CUDA_VISIBLE_DEVICES`;
  * no GPU seen (the driver);
  * a build without kernels for the card.

  `mufasa-pose3d gpu-check` prints it, and the 3-D form's greyed-out solver choices carry it as
  a tooltip.
* README (install line and extras table) and docs/pose3d.md (Solver).
* `tests/smoke_122ld_install_deps.py` (10 checks).

**122lc — One thread per 3-D inference worker.**
The user's 39 videos on 8 workers ran for 14 hours: 10 done, "2500 minutes left". top showed
load average 118, each worker at 100–270 % CPU, the CPU 98.6 % busy. The solver held PyTorch to
one thread, but every other step let each worker's OpenBLAS, MKL, OpenMP and PyTorch pools start
a thread per core, so the threads spun against each other.
* `mufasa/pose3d/threads.py`: `one_thread()` sets the environment, `threadpoolctl` limits for
  the pools already loaded, and PyTorch's threads. `worker_pool(n, ctx)` is a process pool with
  it as the initializer. It serves the per-video pool (`pipeline.run_pose3d`) and the per-chunk
  pool (`mono._pool`), forked or spawned. A single process keeps the defaults.
* A worker that dies ends the run with an error: how many videos are done, where, and to try
  fewer workers.
* The *Workers* tooltip and docs give the real memory (about 1.5 GB per worker for a
  30-minute video, measured) and the workaround for older versions.
* `tests/smoke_122lc_worker_threads.py` (7 checks: every pool's workers at one thread in every
  library, the parent unchanged, a dying worker).

**122lb — The posed rig's frame cache, safe with the background fit (audit finding 2).**
Second patch of the 3-D pipeline audit.
* Since 122ky the background fit thread wrote posed frames into `PosedRig._cache`, a plain
  dict, while the viewer iterated it in `fit_frame` and `nearest_cached` on every paint. That
  can raise "dictionary changed size during iteration" mid-paint.
* `scene_models.PoseCache` takes its place: a lock around every access, and the posed frames
  kept sorted. It reads like a dict (`[]`, `get`, `in`, `update`, `clear`, `len`, sorted
  iteration over a snapshot). `nearest(frame, within)` is a binary search: microseconds on
  54,000 posed frames instead of a scan of every frame on each paint.
* `tests/smoke_122lb_pose_cache.py`:
  * the cache's nearest frame equals a scan (12,000 queries);
  * a writer thread against three readers;
  * the viewer painting frame after frame while a real background fit runs;
  * the speed of the lookup.

**122la — 3-D inference says why a video has no camera.**
The user's screenshot showed every video as "(no camera)": the project had no camera file
under the names looked for, and the form kept the reason to itself.
* `camera.camera_names(stem)`: the names a video looks for (`<stem>.camera.toml`,
  `camera_<prefix>.toml`, `camera.toml`). `camera_for_video` matches them in any letter case.
* The form gets a line under the video list: what it looked for, the camera files the folder
  has, and what to do (*Camera calibration* with the video's prefix). A missing folder is said to
  be missing. Each unmatched video's tooltip names its files.
* *Mark the box…* failed with "TypeError: 'bool' object is not subscriptable": the button's
  `clicked(checked)` went in as the image. The skeleton re-import's *Apply* had the same wiring,
  and `clicked(False)` went in as `confirm`, so it rewrote project.toml without asking. Both are
  wired through a lambda now. A scan of the whole UI finds no other button whose `checked` flag
  lands in an optional parameter.
* `tests/smoke_122la_camera_pairing.py` (7 checks, including that scan).

**122kz — The pose file contract, in one place (audit findings 1, 6, 10).**
First patch of the 3-D pipeline audit.
* `pose3d/io.py`: `pose_markers(columns, dims)` is the one rule for finding markers. A name
  `<m>_var` is skipped when `<m>` is a marker: the per-axis variances the Kalman v2 smoother
  writes into 2-D pose files. From a smoothed 2-D file, the single-camera reconstruction made
  each one a marker of its own (K7: 15, up to 2,000 km away, setting the 3-D viewer's volume
  and adding 3-D features for markers that do not exist). A real marker whose name ends in
  `_var` is kept.
* The rule replaces six copies: the single-camera 2-D loader, `pipeline.py`, the viewer's
  `load_pose3d`, `features_3d.load_pose3d_aligned`, gait (via `marker_names`) and the body
  model's `load_labels`. Existing files read as their real markers without being rebuilt; K7's
  volume goes from ±2,000 km to ±0.3 m.
* The contract is written in `io.py` and docs/pose3d.md: millimetres, z up, the floor at
  z = 0, `frame`, `fps` in the metadata. `check_pose3d` lists breaks: no frame rate, gaps or
  disorder, markers below the floor, units, derived columns. Gait reports them in its summary
  (`contract`), and says when it assumed 30 fps. The gait form and CLI print them.
* C3D import: `floor_z` (form *Floor height in the file*, CLI `--floor-z`) moves the lab's
  floor to z = 0. The import reports markers below the floor.
* `tests/smoke_122kz_pose_file_contract.py` (12 checks).

**122ky — Real-time models in the 3-D viewer: GPU drawing, batched posing; the mouth closed.**
The user asked for real-time rendering of the model with GPU acceleration, and for the
virtual rodent's mouth to be closed.
* `Pose3D.extent()` is worked out once. It was recomputed over every frame on each paint:
  0.4 s a paint on K7's 108,000 frames, the viewer's main cost.
* `mufasa/ui_qt/gl_models.py`: the models drawn with OpenGL 3.3 into an off-screen,
  multisampled picture with a depth buffer, laid under the markers. The vertex shader
  projects exactly as `Volume3DWidget.project` does (perspective, orthographic, the
  recording camera, the animation mix); flat shading from screen-space derivatives;
  premultiplied blending for transparent models. It falls back to QPainter without OpenGL.
  Against the CPU drawing, model pixels overlap (IoU) ≥ 0.99 in every view.
* `mufasa/pose3d/rig_batch.py`: the rig's posing batched in PyTorch (CUDA when present):
  * kinematics of the bones that move markers (49 of 63), exact to 1e-13 mm;
  * `vmap(jacfwd)` Jacobians; bounded Levenberg–Marquardt with damping per frame;
  * the per-frame fit's cold starts on every 8th frame, the frames between from the
    nearest solved frame, a neighbour pass.

  K7: the same marker error as the per-frame fit at 28 ms a frame on a 2-core CPU (it
  was 220).
* `PosedRig.fit_frames`; animations pose every rendered frame up front. The *View & models*
  panel fits in a background thread, nearest the frame shown first, while the viewer shows
  the nearest posed frame instead of fitting on its own. New controls: *Fit in the
  background* (on adding a rig) and *GPU*. Closing the viewer stops the thread.
* The mouth: the skin was bound with the jaw open, wider than the joint's range closes. The
  stance sets the jaw at −40° (`JAW_CLOSED_DEG`), where the lips meet. The built-in rig cache
  is versioned (`virtual_rodent_v2.rig.npz`).
* The viewer's rig keeps its size from the back's labels: on K7 that fits the markers
  better (20.7 vs 23.9 mm) than nose to tail base, which the gait's physics uses.
* `tests/smoke_122ky_viewer_gpu_realtime.py` (12 checks; the GPU comparison runs under
  `xvfb-run` when present). 122kv and 122ku were updated: background fitting, and the jaw
  set past its range.

**122kx — Gait bounds from uninjured rats on a CatWalk; a stride clock.**
The user attached Neckel et al.'s CatWalk data (Mendeley sg3d7dv345, MATLAB 7.3, in nine
parts) to calibrate 122kw's bounds.
* `scripts/calibrate_gait_catwalk.py` reads the file, or its split parts in place, takes the
  pre-injury runs (108 rats, 5780 strides, 13–71 cm/s) and writes
  `mufasa/assets/gait/catwalk_rat_reference.json`: laws, medians by speed, percentiles,
  phases.
* `BOUNDS` are now speed-dependent (`gait_laws`):
  * stance a·v^b (0.41 s at 13 cm/s, 0.12 s at 70);
  * swing about 0.14 s;
  * stride 65 mm + 0.156 s × v;
  * duty factor 0.75 → 0.5;
  * base of support 23 / 29 mm.

  TactileArena's demo agrees within the scatter; its fixed bounds stay available
  (`bounds_preset("tactilearena")`). The bounds are adult-rat values; `scale` gives dynamic
  similarity.
* `estimate_gait`:
  * a stride clock (one turn per stride, at speed ÷ stride) times lift-offs with the rats'
    offsets (LH 0, RH ½, LF 0.39, RF 0.89), and touch-downs so that late paws catch up;
  * paws land half a stance's travel ahead of the girdle (it was half a stride, beyond the
    foreleg's reach at speed);
  * steps off the clock must keep the extrapolated centre of mass over the support as it
    will be when the paw lands (Hof 2008);
  * the other side's paw must be down for (duty − ½) of a stride, not a whole minimum
    stance; that wait had lengthened every stride;
  * the summary's medians and phases use walking strides only (≥ 10 cm/s, ≤ 1 s).
* Size from nose to tail base. The path along the back gave K7 1.4× the model (916 g): its
  back labels sit unlike the model's, and noise lengthens the path. The new estimate is 1.2×.
* Synthetic walks at 15, 30 and 50 cm/s (60 fps): strides 85/89, 110/112, 142/143 mm
  (estimated/rats); left–right phases 0.5; hind→fore 0.39–0.43 (rats 0.39).
* CLI `mufasa-pose3d gait --bounds {catwalk,tactilearena} --gait-scale K`; a *Bounds* choice on
  the 3-D tab's gait form (recorded in run.toml); docs/pose3d.md's gait section.
* `tests/smoke_122kx_gait_catwalk.py` (12 checks).

**122kw — Gait: paw placement, weight distribution and the centre of mass from a physics model.**
The user asked for a physics model of the body and limbs to estimate paw placement and gait,
using weight distribution and momentum to minimise paw placements within physiological bounds
(sources: Mendes et al. 2015; Neckel et al. 2013 and their CatWalk data; TactileArena).
* `RodentModel.body_masses()`: each body's mass and centre of mass from its collision shapes,
  with geom default classes inherited (338 g in all).
* `mufasa/pose3d/gait.py`:
  * `BodyPhysics.virtual_rodent()`: each segment's centre of mass and each limb girdle is
    tied to its nearest markers (head on head, neck between head and shoulders, tail on tail,
    the rest on the back) as a convex combination plus an offset in a back frame that pitches
    with the body. Over sampled spine postures it is within about 3 mm. Legs' reach comes
    from the joint ranges.
  * `estimate_gait`: a smoothed centre of mass, velocity and acceleration; the extrapolated
    centre of mass (Hof 2005) and the ZMP. Paw placement keeps the fewest placements: a paw
    stays planted within reach and half a stride; it lifts after a minimum stance, only if
    the rest keep the body balanced and the other side's paw is down; left and right
    alternate (duty factor); it swings for a physiological time and lands half a stride
    ahead of its girdle, beside the midline at half the base of support. Forelimbs whose
    girdle is out of reach stay in the air.
  * Loads: the centre of pressure at the ZMP, the most even non-negative sharing
    (`share_load`), and `support_margin`.
  * Size from the back's length; mass = model × size³ or weighed.
  * Steps table (stride, swing, duty factor) and summary (support pattern, load shares,
    phases).
  * `BOUNDS` from TactileArena's demo recording (58 labelled rat stances); `MOUSE_BOUNDS` from
    Mendes et al. Lengths scale with size, times with √size.
  * `gait_from_pose3d`, `write_gait` / `read_gait`.
* 3-D tab: *Gait (paws from a body model)* (`GaitForm`, section `pose3d_gait`, run →
  `derived/gait/<run>/`). CLI: `mufasa-pose3d gait`.
* 3-D viewer: `GaitOverlay` (load a `_gait.parquet`) draws the planted paws (sized by load),
  the support polygon, the centre of mass over the floor and the ZMP (green inside, red
  outside).
* Pins: 17 of 20 ui_bound sections with detect_path (122ep, 122eq, 122er, 122es); 14 wired
  forms (122ex, 122em, 122fj).
* Docs: `docs/pose3d.md` (*Gait*). Test: `tests/smoke_122kw_gait_physics.py` (18 checks).

**122kv — 3-D viewer: a model loader and view options.**
The user asked for a model loader and view options in the 3-D renderer.
* `mufasa/pose3d/scene_models.py` (no Qt):
  * `load_mesh` reads OBJ (polygons fanned), STL (ASCII or binary; shared vertices merged) and
    PLY (ASCII or binary little-endian). `simplify` reduces meshes above 20,000 faces by vertex
    clustering.
  * `SceneMesh` has a scale, a turn about the vertical and an offset.
  * `PosedRig` poses a body-model rig to the 3-D markers it shares with the rig's label points.
    It fits heading, position (small height offset) and the rig's posture within its limits by
    least squares, with a weak pull to the rest pose. Its size comes from the back's length seen
    from above (median over frames with the head low), or from the median ratio of marker
    distances. It starts from the nearest posed frame, else from a 2-D Procrustes start with
    both headings, flat or rearing. Fits are cached per frame.
  * `ModelLook` holds style (solid / transparent / wireframe / points), opacity, colour and
    visibility.
  * `load_model`, `model_from_spec` and `spec()` handle loading and saving; `builtin_rig_path()`
    makes the virtual rodent once in the user cache.
* `Volume3DWidget` draws models: back to front, flat-shaded towards the eye, as a wireframe or
  as points. Rigs are posed 60 ms after a frame is shown (the nearest posed frame is drawn
  meanwhile), or while painting with `sync_fit` (rendering). New view options:
  `show_markers`, `marker_scale`, `show_names`, `show_skeleton`, `show_grid`, `show_axes`
  (gizmo), `ortho` (orthographic, scaled to look as the target plane does in perspective) and
  `background`. They are added to `STYLE_KEYS`, so saved animations keep them.
* `mufasa/ui_qt/pose3d_models.py`: the *View & models* panel. *View*: markers and size, names,
  skeleton, grid, axes, projection, field of view, background. *Models*: Load…, Virtual
  rodent, Remove, Fit all frames; style, opacity and colour; scale, turn and offset for meshes;
  the marker error of the frame.
* Camera animations save the models (`settings()["models"]`). `render_animation(models=…)`
  draws them (rigs posed per rendered frame); the CLI `render-animation` passes them.
  `mufasa-view3d --model PATH` (repeatable; `virtual_rodent`) loads models at start.
* K7: the virtual rodent posed to the 3-D markers at about 0.2 s per frame (median); marker
  error 17–30 mm on walking and rearing frames.
* Docs: `docs/pose3d.md`. Test: `tests/smoke_122kv_viewer_models.py`.

**122ku — Body model: the scale from a single-camera 3-D run; the rat stands with tucked limbs.**
In 122kt, the scale estimated from each stretch of K7 ranged 0.59–0.81 px per model mm, and
width and scale traded off.
* `fit.scale_from_pose3d(rig, VIDEO_3d.parquet)` multiplies two numbers: the camera's pixels
  per mm at the back's median height and place (the camera file named in the 3-D run's
  metadata, or one given), and the animal's size against the model (the back's length from
  above, `head_back` → `back_V2`, median over frames with the head low, over the same labels
  on the model at rest).
* `RunConfig.scale_from` is used when no scale is set. The log reports it, and the metadata
  stores `scale_from` (px/mm, height, size, labels).
* Fit form: *Scale from* (estimated from the video, or one of the project's 3-D runs, using
  its file for this video). CLI: `body-fit --scale-from`.
* The virtual rodent's rest pose tucks the limbs under the body (`rodent_rig.STANCE`,
  `stance_qpos`; `export_rodent_rig(stance=False)` keeps the model's zero pose). In the zero
  pose the legs splay 45 mm from the midline against the trunk's 33 mm. The angles came from a
  seeded search (limbs inside the trunk's outline from above and under the back, fore and hind
  paws on the floor, small angles) and lie within every range.
* Docs: `docs/bodymodel.md`. Test: `tests/smoke_122ku_scale_from_pose3d.py` (11 checks).

**122kt — The rat's width: girth and per-frame widening.**
In 122ks's K7 fits, rats that were curled up or sitting covered more floor than the slim virtual
rodent.
* `rodent_rig.widen`: the trunk is widened sideways about the midline, by the girth from the
  thorax to the pelvis, tapering to 1 at the atlas and 15 mm behind the tail base. The head,
  tail, paws (lowest 8 mm) and the midline keep their size. It is linear in the girth.
* `export_rodent_rig(girth=…)` gives a heavier base shape; the profile name and the metadata
  record it. The rig also stores the displacement per unit of widening (`D`, and `Dd` for the
  fitting mesh).
* A new posture value `widen` (in `PARAMS`; the rat's profile allows 0…0.5 and pulls weakly
  towards 0): `Rig.verts(p)` applies it, and the silhouette and the renderer use it. Rigs
  without `D` ignore it, the mouse's posture leaves it out, and `params_at` reads old fit files
  as unwidened.
* After a fit, `body_area_ratio` (the mask's area against the fitted body's, over up to 40
  frames) is logged and stored as `area_ratio` in the metadata. For the virtual rodent, the
  log suggests a girth (`suggest_girth`) when the ratio is above 1.15.
* Import rig: *Girth* (virtual rodent only); CLI `body-rig --virtual-rodent --girth`.
* Docs: `docs/bodymodel.md` (*Width*). Test: `tests/smoke_122kt_rat_width.py` (14 checks).

**122ks — The rat in the Body model tab: the virtual rodent as a rig, rig profiles.**
The plan's next step after 122kr ("looks good, continue").
* `mufasa/bodymodel/rodent_rig.py`: `export_rodent_rig` builds a `.rig.npz`
  from the packaged virtual rodent, no Blender. HBP conventions (facing −y,
  left +x, z up, floor 0, mm), re-rooted at the pelvis: `Pelvis`,
  `Spine1`–`Spine6` (lumbar vertebrae, the rigid thorax last), `Neck1`–`Neck7`,
  `Head`, `Tail1`–`Tail30`, hind and fore legs. Fitting mesh by vertex
  clustering (`decimate`, 3 mm). Skin drawn light grey.
* Rig profiles (`profile` JSON in the `.rig.npz`; `fit.HBP_PROFILE` for rigs
  without one): spine regions, head, trunk and tail chains, hidden tail,
  paws, hips, posture limits, limit texts, prior centres, and optionally
  fixed label positions. `Rig`, `Markers`, the search, the free-scale fit,
  the temporal term and the prior all read the rig's profile. The rat's limits
  come from the model's joint ranges spread over each region (lumbar ±10°
  sideways, −15…+10° up/down per joint; cervical ±10°, −10…+15°; atlas
  −45…+5°; tail ±8°; rigid thorax), its prior pulls to the rest pose, and its
  labels sit where `rodent_model` puts them (the scale does not hinge on the
  head's size, which is small for a large rat).
* Masks for light animals: `MaskConfig.polarity` (auto/dark/light; auto from
  the pixels under the confident back labels against the background),
  `min_grey_light`, `near_labels` (light animals by default: only the region
  around the confident labels, and of its parts the one holding most of
  them, since moved bedding also differs from the background). The scale's
  first guess is corrected by the animal's mask area when they disagree by
  more than 15%, and the free-scale fit starts from that guess.
* Forms: *Import rig* → *Model: Virtual rodent (rat, built in)*; *Fit body
  model* → *Animal* (auto / darker / lighter than the background). CLI:
  `body-rig --virtual-rodent OUT`, `body-fit --animal`.
* K7 (white rat on bedding): scale 0.85 px per model mm, consistent with the
  camera and the 122kr size estimate (0.66 px/mm × 1.34); see the session
  notes for IoU and label error by behaviour.
* Docs: `docs/bodymodel.md` (the rat, profiles, light animals).
* Test: `tests/smoke_122ks_rat_rig.py` (20 checks; a synthetic white rat on
  darker bedding with a bright patch that is not the rat).

**122kr — Anatomical limits for single-camera 3-D pose (virtual rodent).**
The user has one camera and DeepLabCut labels, and asked for the height
estimates to be held to what a body can do; for a rat, the reference is
DeepMind's virtual rodent (dm_control, Apache 2.0), now packaged in
`mufasa/assets/models/pose3d/virtual_rodent/` with its licence and notice.
* `mufasa/pose3d/rodent_model.py`: reads the MJCF (bodies, hinge joints,
  default-class inheritance) and the skin (scaled ×0.82 to the skeleton it
  was bound to); forward kinematics and skinning in NumPy; the project's
  labels placed on the model; `build_limits` (a seeded random search per
  label for its highest height on the hind paws and its lowest; sampled
  lumbar and neck bend ranges, measured as the spine chain measures them).
  `scripts/build_virtual_rodent_limits.py` writes
  `virtual_rodent_limits.json`.
* `mono.MonoConfig.anatomy` ("virtual_rodent", default; None to switch off),
  `anatomy_scale` (None: from the back's overhead length in the session),
  `anatomy_margin` (10%), `anatomy_bend_margin_deg` (5°). `mono.Anatomy`:
  * the rigid head is scaled about the camera centre so the ears are within
    their height range (it still reprojects);
  * trunk labels and hips are kept in their ranges along their rays, before
    and after the spine chain, and inside it as hinge terms;
  * the chain keeps the lumbar and neck bends within range (hard hinges; the
    pelvis segment was only limited upwards before) in frames where every
    spine segment points forward from overhead (`Anatomy.measurable`);
    elsewhere the bend is not measurable from one camera. SciPy and PyTorch
    (`gpu.chain_residual_fn(anatomy=…)`) alike.
  * report `anatomy`: size and how it was found, head frames lowered/raised,
    trunk labels moved, frames still outside the bend ranges.
* 3-D inference form: *anatomical limits (virtual rodent)* and *Body size vs
  virtual rodent* (auto); CLI `--no-anatomy`, `--anatomy-scale`.
* Docs: `docs/pose3d.md`, the models README.
* Test: `tests/smoke_122kr_anatomical_limits.py`; `smoke_122kf` lets the new
  step sit after the rigid head.

**122kq — Body model tab: fit a rigged 3-D model to video, over time.**
The new package `mufasa/bodymodel/` and the tab `pages/bodymodel_page.py`
(after 3-D) are documented in `docs/bodymodel.md`.
* *Import rig (.blend)*: an armature and the skin it deforms are saved as
  `sources/bodymodel/<name>.rig.npz`:
  * the bones (parents first, rest matrices);
  * the skin, full and decimated;
  * the weights;
  * the fur colour per face, from the texture.
  It runs in a separate process with Blender's Python module (`bpy`) or a
  Blender executable.
* *Fit body model*: rig, video, 2-D labels (imported or Kalman v2), frames
  from/to/every, *Forwards and backwards*, *Smoothness*, *Scale* (auto),
  *Workers*, *Render the model over the video* (optionally side by side).
* Posture: 17 values per frame. Per-joint limits are from the HBP mouse
  model's table (Dewan 2010, Shahrokni 2012); head and body limits are
  assumed. The body rears about the hind paws, and nothing may go below the
  floor.
* Per-frame loss:
  * silhouette IoU of the body without the tail, against the animal's mask
    (much darker than a median background taken over the whole video),
    used when the mask is plausible;
  * robust, likelihood-weighted label distance. The chain labels are placed
    along the model at the labels' own spacing, with the scale anchored on
    nose to back of the head;
  * a floor penalty and a weak neutral prior.
  CMA-ES within the limits.
* Over time (the user's frame-history request):
  * forwards, each frame starts from the previous one and is pulled towards
    it, and is searched again (both headings) if it loses track;
  * backwards, each frame is refined towards both neighbours, keeping the
    better result.
  On the synthetic test this reduces the head error on frames with wrong
  head labels from 4.8° to about 2°, and the frame-to-frame jitter by a
  third.
* Runs in overlapping chunks on a process pool. Output:
  `derived/bodymodel/<run>/<video>_fit.parquet` (posture, IoU, label error,
  loss terms; rig, scale, limits and settings in the metadata), an optional
  `<video>_overlay.mp4` (shaded with the fur colours, outline, labels
  against model points; every frame, interpolated between fits), and
  `run.toml`.
* `Rig.local_quaternions` gives each bone's rotation in Blender's
  convention. The CLI has `mufasa-pose3d body-rig` and `mufasa-pose3d
  body-fit`.
* The model silhouette is filled one triangle at a time: `cv2.fillPoly`
  with many overlapping triangles cancels where they overlap (the top and
  underside of the body), which had left holes in it.
* Test: `smoke_122kq_body_model_fit` uses a synthetic rig with the mouse's
  bone names, walking, turning and bending through a synthetic video with
  noisy labels, dropouts and wrong head labels. The section-count pins are
  now 16 of 19 ui_bound sections with detect_path, and 13 wired forms.

**122kp — camera animation in the 3-D viewer, rendered to video.**
`mufasa/pose3d/animation.py` (paths, rendering) and
`mufasa/ui_qt/pose3d_animation.py` (the editor). Tick *Animation* in the viewer to open
the editor:
* keys of the view at video frames, in a table: frame, length to the next
  key (output frames), easing (*Ease in-out*, *Linear*, *Spline*);
* *Add key*, *Update key*, *Delete* and *Go to*;
* a timeline slider and *Preview*;
* *Save…* and *Load…* (`.camera.json`);
* *Render video…* with size, fps, *Video below* and *Frame label*.
How it works:
* `CameraPath`: the azimuth takes the shorter way round, the distance is
  interpolated on a log scale, and the data frame moves linearly between
  keys (real time, slow motion, or frozen on equal frames). Camera →
  camera moves zoom and pan in the recording camera's view.
* Moves into or out of the camera view blend the calibrated and the orbit
  projection, both for the markers and for the video homography.
  `Volume3DWidget.cam_blend`. On K7 the first step changed 40 grey levels
  on average before blending and 2.6 after.
* `Volume3DWidget.view_state()` / `set_view_state()`, plus `show_label`.
* `render_animation` draws offscreen at any size. On K7, 960×1080 with the
  video below rendered at about 35 frames/s.
* CLI: `mufasa-pose3d render-animation`.
* Test: `smoke_122kp_camera_animation` (30 checks). It covers
  interpolation, freeze, the spline through keys, no jump out of the
  camera view (6 px without blending), every editor action, the rendered
  frames against the view drawn directly, the video below, and the CLI.

**122ko — 3-D features in Compute Features.**
`mufasa/feature_extractors/features_3d.py` adds four families. They are in
`FEATURE_FAMILIES` and in the form's new **3-D features** group:
* heights above the floor;
* 3-D distances for every marker pair;
* 3-D frame-to-frame movements;
* posture angles: head pitch and roll, body pitch, head pitch relative to
  the body, and the signed spine bend at each joint (concave back is
  positive, hunched is negative). The run's own pelvis pitch and lumbar
  angle are passed through.
How it works:
* Source: the video's 3-D pose file from the newest `derived/pose3d` run
  that has it, then `sources/pose3d` (motion capture). The form's *3-D
  pose from* combo can pick one run or motion capture instead
  (`FeatureSubsetsCalculator(pose3d_dir=...)`).
* Rows follow the video's frames through the file's `frame` column.
  Missing frames are NaN for heights and angles (added to the signed
  families) and -1 for distances and movements.
* Videos without a 3-D file are reported and get no 3-D features. If only
  3-D families are asked for and no video has a 3-D file, the run stops
  with an error. If 2-D families are asked for too, the 3-D ones are
  skipped and the 2-D ones are still computed.
* Checked on K7: 15 markers, 108,030 frames. Body pitch has a median of
  31° (5–95 %: 7–72°), head pitch relative to the body a median of −47°.
* Test: `smoke_122ko_features_3d` uses a synthetic walking-then-rearing
  rat with a gap in the 3-D file, and checks values, signs, the gap,
  source choice, errors and the form. `smoke_122io_compute_features` now
  runs the 2-D families only.

**122kn — the video in the 3-D view, aligned with the markers.**
3-D viewer: *Video in 3-D* (on when the file has a camera and a video is
open) draws the current frame on the floor (`z = 0`) of the volume. The
frame is mapped through the calibration — image → floor homography from
`P`, undistorted first (`cv2.remap` with the file's `K` and `dist`) when the
lens has distortion — and through the view (floor → screen, a second
homography), as one projective `QTransform`; no OpenGL. It is not drawn
when the floor it covers would reach behind the eye.
* **Camera** view now renders through the calibration (`P`) instead of an
  approximate orbit camera with a guessed field of view: the frame fills the
  view as recorded and every marker sits on the animal at any height; wheel
  zoom and right-drag pan keep the alignment, left-drag orbits away. In
  other views floor-level markers sit on their image, raised ones show
  parallax.
* Opacity of the video; *Video panel* hides the picture below (playback
  controls stay). `FrameScrubberWidget.raw_frame()` and
  `set_picture_visible()`.
* `Camera.viewer_meta` also stores `K`, `dist` and the image width (new 3-D
  runs); without them the frame is taken as undistorted.
* Fix: *Reprojection* drew undistorted points on the raw (distorted) video;
  it now applies the lens distortion (off by up to 41 px near the corners of
  the test lens; K7 has no distortion).
* Checked on K7 (frames 3700 and 3775, rat on the wall and on the bars): box
  floor on the bedding edges, markers on the rat in the Camera view. Test:
  `smoke_122kn_video_in_3d` (a synthetic tilted, distorting camera filming
  dots at a floor and a raised point; their rendered positions within 3 px
  of the markers in Top, Perspective and Camera views, zoomed and panned).

**122km — re-import the skeleton model (markers and skeleton) on the Data Import page.**
For a retrained pose model with new markers (e.g. eyes, shoulders). New
`mufasa/model/skeleton_reimport.py` and the Data Import section *Re-import
skeleton model*: pick the new model's pose file (FreeDLC / DeepLabCut
parquet, .h5, .csv), read what would change, *Apply*.
* `[pose] body_parts` ← the file's markers (its order); `[skeleton]` ← the
  file's embedded skeleton, or the project's edges between markers that
  remain when the file has none.
* `[[pose.segments]]` (Kalman v2, takes priority over `[skeleton]`): removed
  markers leave their segments; each new marker joins the segment of its
  nearest marker on the new skeleton (eyes → head, shoulders → back); a
  segment left without markers is dropped and its children re-attached to
  its parent. New markers with no skeleton path to a segment are reported
  (the smoother ignores them); optional (*Add new markers to the Kalman v2
  segments*).
* Removed markers still named elsewhere in project.toml (`[pose.layout]`,
  stage settings) are reported, not changed. project.toml is backed up to
  `logs/project.toml.<timestamp>.bak` first (comments in project.toml are
  not kept by the rewrite; the backup has them). Pose files already
  imported are untouched. After applying, the form offers to reopen the
  project.
* On the strohA-kj project.toml with eyes and shoulders added and tail_V32
  dropped: eyes in `head`, shoulders in `back`, `tail_3` removed; the Kalman
  v2 layout builds with 18 markers. The single-camera 3-D solver does not
  use eyes or shoulders yet. Test: `smoke_122km_skeleton_reimport`.

**122kl — import motion capture (C3D) as 3-D pose.**
New `mufasa/pose3d/c3d.py`: a C3D reader needing only NumPy — header and
parameter section; integer and floating-point point data; Intel, DEC (VAX
floats) and MIPS/SGI (big-endian) processors; labels from `POINT:LABELS` and
its continuations; units (m, cm, mm, in) to millimetres; frame counts beyond
the 16-bit header (`POINT:FRAMES`, `TRIAL:ACTUAL_*_FIELD`); invalid points
(negative residual word) → NaN; analog channels skipped. Tolerates
`POINT:DATA_START = 0`, empty character parameters, a processor byte of 0.
Checked against ezc3d on 13 vendor files (Vicon, Qualisys, Cortex, BTS,
Kistler, Optotrak, Motion Analysis, one DEC): identical points, invalid
markers, labels, rates; residuals as the C3D specification and the `c3d`
package (low byte of the fourth word × |scale|; ezc3d reads float-file
residuals differently).
`c3d_to_pose3d` / `import_c3d` write `sources/pose3d/<trial>_3d.parquet`
(mm, z up; `_likelihood` 1 measured / 0 missing; `_residual_mm`;
`c3d_frame`) with the `mufasa_pose3d` metadata (box from the data, rat
skeleton for known names). Options: the file's up axis (y-up files turned),
renaming markers, skeleton lines, a frame range.
* 3-D tab: *Import motion capture (C3D)* section (`pose3d_c3d`, detect path
  `sources/pose3d`); the 3-D viewer section has *(no video)* and lists
  `sources/pose3d/`; `find_pose3d` also looks there.
* CLI: `mufasa-pose3d import-c3d FILES --project P | --out DIR [--up]
  [--rename] [--edges] [--frames]`.
* Section-count pins updated (14 of 17 with a detect path, 11 wired forms).
  Test: `smoke_122kl_c3d_import` (writes C3D files in every storage form).

**122kk — project links survive an sshfs mount; classifier table warning.**
Imported videos were symbolic links holding the absolute path of the
original; opening the project on an sshfs mount (another path) broke every
one ("NO FILES FOUND ERROR: …/sources/videos/K7_….mp4 is not a valid file
path"). New `mufasa/utils/project_links.py`:
* imports (single and many videos, frame folders) create **relative**
  links (`link_file`);
* `repair_project_links` rewrites absolute links under `sources/` as
  relative: a working link to the same target; a broken one after finding
  its target through the mount offset — the target path minus leading parts
  under each ancestor of the project (`/proc/videos/K7.mp4` on the server,
  the project at `/cloud/server/proc/mufasa/P` here → `/cloud/server/proc/
  videos/K7.mp4`); only the project's ancestors are tried, so the relative
  link names the same file on the server. Targets not reachable here are
  reported and left alone;
* the workbench runs it when a project opens (one line on the console when
  anything changed); `mufasa-relink [PROJECT] [--dry-run]` runs it by hand.
Also: the classifier table connected its double-click handler after a
blind `disconnect()`, which made PySide6 print a RuntimeWarning on every
refresh; it now connects once. Test: `smoke_122kk_project_links`.

**122kj — 3-D pose on the GPU (PyTorch).**
New `mufasa/pose3d/gpu.py`: the single-camera head and spine-chain fits in
PyTorch, on CUDA or the CPU. *Solver* on the 3-D tab, `--device
auto|cuda|torch-cpu|cpu` on `mufasa-pose3d infer`, `MonoConfig.device`
(default *auto*: GPU, else PyTorch on the CPU, else SciPy).
* **Solver.** A port of SciPy's `least_squares(method="trf",
  x_scale="jac", tr_solver="lsmr")` — the CPU path's solver — with the
  same trust-region loop, regularised LSMR, 2-D subspace step, radius update
  and stopping tests. Residuals and per-frame Jacobians come from
  `torch.func` (exact); the Jacobian is applied as batched per-frame blocks
  plus the smoothness rows. Float64 (the work is launch-bound, so slow
  consumer-GPU FP64 does not matter). Same 6000-frame chunks as the CPU.
* **Same results.** The PyTorch residuals equal the NumPy ones to the last
  digit. K7 head chunk 5700–12 300: cost 19 015.3 against SciPy's 19 014.8,
  identical 2-D error, no yaw flips. K7, 13 000 frames, whole
  reconstruction: median difference 2·10⁻⁵ mm, 95th percentile 0.3 mm, 99th
  4–5 mm (head); trunk heights, pelvis pitch, spine length unchanged.
  Exact Levenberg–Marquardt steps (one damping, per-frame damping, frame
  rollback) were tried first and settled in clearly worse minima on K7
  (33–37k against 19k): the step rule matters, so SciPy's was ported.
* **Speed.** On this 2-core cloud machine (no GPU): PyTorch on the CPU
  128 s for 13 000 K7 frames against 383 s for SciPy; about 2× per core
  single-threaded. PyTorch on the CPU solves with one thread so a video
  gives the same result whatever the number of workers. The GPU path could
  not be timed here (no GPU); `mufasa-pose3d gpu-check` times CPU vs CUDA on
  a synthetic sequence and checks they agree.
* **Several videos on the GPU** run in spawned processes (CUDA does not
  survive a fork), at most 4 at once; no chunk pool on the GPU.
* `torch>=2.4` added to the `gpu` extra. Test: `smoke_122kj_pose3d_gpu`
  (CUDA agreement is checked when the machine has a GPU).

**122ki — 3-D inference faster: parallel workers and a cheaper Jacobian.**
* **Workers.** *Workers* on the 3-D tab, `--workers` on `mufasa-pose3d infer`,
  `RunSettings.workers` (default: all cores but one). Several videos run at
  once, one process each, with progress per finished video and the time
  left; a single video splits its 6000-frame head and spine-chain chunks
  over the workers instead (`MonoConfig.workers`). Same results as one
  process. K7, 13 000 frames (3 chunks): 1.9× faster on 2 workers than on 1.
* **Jacobian** (`mufasa/pose3d/solver.py`). The head and spine fits used
  SciPy's sparse finite differences (17 column groups, the matrix rebuilt
  each time: 726 ms per evaluation on a K7 head chunk, ~130 per chunk). The
  columns are now grouped by frame parity (12 groups, exact because no
  residual spans two frames that far apart) and the matrix filled from a
  fixed index: 198 ms. Same solver, same steps: about 1.3× faster per
  worker (K7, 13 000 frames: 292 s against 383 s).
  Results: on the K7 head chunk 5700–12 300 the cost is identical and
  positions agree within 0.0055 mm (99th percentile) — closer than the old
  solver agrees with itself after a 10⁻⁶ px change of its input (0.045 mm).
  Over a whole reconstruction rounding still steers a few poorly
  constrained frames elsewhere (13 000 K7 frames: median difference
  10⁻⁵ mm; 3% of frames over 1 mm, 99th percentile 5–10 mm).
  Rejected because they changed results: capping LSMR at 20 iterations per
  step (2.3× faster; lower cost on one chunk, 10% higher on another, heads
  up to 50 mm lower) and Levenberg–Marquardt with direct banded solves (6×
  per chunk, stuck in higher local minima at the box-wall hinges).
* Expected: about 2–3 hours for 67 sessions on an 8-core / 16-thread
  workstation, instead of about 1.5 days. Test: `smoke_122ki_pose3d_speed`.

**122kh — smoke suite green: three real fixes, 19 obsolete tests retired, the rest updated.**
Fixes found by the suite (regression test `smoke_122kh_settings_and_tk_free_imports`):
* **mufasa imports without tkinter.** `utils.errors`, `utils.lookups`,
  `roi_tools.roi_utils` and `video_processors.video_processing` imported
  tkinter (and `PIL.ImageTk`) at module level, so on a Python without Tk
  most of mufasa — ConfigReader, checks, every backend — failed to import.
  Tk is now imported only where a Tk window is opened.
* **Classifier model paths.** project.toml stores them relative to the
  project (`relativize_project_path`), but the TOML→INI translator passed
  them on unresolved, so inference read `model_path_N` as
  `../../…/model.sav`. They are resolved against the project again.
* **Directing settings.** The add-ons "Directing settings" form read and
  wrote project.toml with configparser — saving replaced the TOML with INI
  text — and the translator never passed `bodypart_direction` on. It is now
  `[directionality] bodypart_direction` in project.toml and reaches
  ConfigReader as `[Directionality settings]`.
* Lint: UP037 and I001 in the 3-D modules and a few others; two grey text
  colours in the classifier form now use `palette(placeholder-text)`.

Retired (their subject is gone or they only pinned a past patch's
counts): the Tk-migration and Tk-deletion acceptance tests
`smoke_122ae5`, `122ae5d`, `122aw`, `122bn`, `122br`, `122ck`, `122cm`,
`122cr`, `122cx`, `122cy`, `122d5`, `dead_code_sweep`, `pyglet_removal`;
the rename/cleanup count floors `122bl`, `122bm`, `122bq`; `v1_configreader`
(the `_is_v1` branch went in 122dy); `pose_import_rename` (122w, superseded
by 122eb); `hull_area_diagnostic` (a one-off diagnostic). Updated: checks
on deleted Tk files dropped from `122ae5c`, `122ak`, `122an`, `122ax`,
`frame_ops_cluster`, `image_format_converter`, `join_transition_cluster`,
`pose_tools_port`, `video_utilities_cluster`; legacy-INI cases dropped or
moved to project.toml (`edit_project_metadata`, `model_dual_save`,
`input_source_picker`, `annotation_polish`, `122bc`, `project_layout`,
`workbench_auto_discover`, `roi_video_table_helpers`); `Optional[X]`
spellings accept `X | None` (`122ae3`, `122ae5e`, `122az`, `122be`, `122bi`);
current section names and order (`122aq`, `122ar`, `projects_page_restructure`,
`video_import`); `data_import_minimal` knows `animal_ids`;
`docs/README.md` indexes every doc (`122cn`). Eleven tests put the
repository root on the import path, so the suite runs without installing
mufasa. Needs the `analysis` extra (`diptest`) and built Cython kernels
(`python setup.py build_ext --inplace`) for the landscape and native-kernel
tests. `smoke_feature_kernels_realdata` and `smoke_feature_parallel_verify`
remain manual checks on a real project (`--config`).

**122kg — tests brought up to date; the feature viewer's 3-D… button no longer crashes.**
Fix: `feature_viewer.open_3d` (122ke) used `Path` without importing it, so the
3-D… button raised NameError when no 3-D file was found next to the session.
Tests updated for intended changes: `smoke_kalman_diagnostic` writes lowercase
marker names (names are case-sensitive since 122hb) and runs from any
directory; `smoke_122ed` accepts the FreeDLC parquet importer (122ie), whose
Interpolate step is guarded and off for both callers (13 importers);
`smoke_122bj` no longer spot-checks `labelling_interface.py` (deleted in
122d5); the section-count pins in `smoke_122ep/eq/er/es/ex/fj` include the
three 3-D sections (13 of 16 ui_bound sections with a detect path, 10 wired
forms); the README lists `mufasa-pose3d`, `mufasa-view3d` and the 3-D page
(`smoke_122iw`). Lint regressions from 122ke/122kf fixed: a lambda shadowing
its loop variable in `pose3d_viewer` (`smoke_122fs`), a `.format` call in
`pose3d.pipeline` (`smoke_122fw`), and four unused imports
(`landscape_animate`, `forms/pose3d`, `csv_export`, `feature_viewer`;
`smoke_122fy`). Full smoke suite: every test that passed before 122kd passes;
the 71 that already failed (mostly tests that pin files deleted with the Tk
surface, lint counts and legacy-INI paths) are unchanged.

**122kf — 3-D pose: calibration, single-camera and stereo inference, 3-D Kalman smoothing, and a "3-D" tab.**
New package `mufasa/pose3d/`: `camera` (pinhole + lens distortion; calibration
from marked box edges — principal point from the wall-corner edges, scale from
the floor, warnings when the entered size does not fit the marked floor — from a
checkerboard (lens, adapted from rpimocap; with the box corners it also solves
the camera height) and DLT from 3-D points (rpimocap)); `mono` (single camera:
rigid head learned from the session with the ear distance as scale; spine and hip
heights and the lumbar angle from Rat7M motion capture by k-nearest neighbours;
pelvis pitched to the lumbar angle; back refitted as one chain within Rat7M bend
limits; tail draped without stretching; every label stays on its camera ray; the
head's heights below the ear line are fixed (unobservable from above) and walls
tolerate 20 mm so an imprecise box does not push heads up);
`stereo` (triangulation with likelihood/reprojection gates); `kalman3d`
(constant-velocity Kalman + RTS with gating, noise along the camera ray for
single-camera runs); `io`; `pipeline` (project: `sources/pose/` + newest Kalman v2
run to fill gaps + `sources/calibration/camera_<name>.toml` →
`derived/pose3d/<run>/<video>_3d.parquet` with `run.toml`). Model data:
`mufasa/assets/models/pose3d/rat7m_trunk_prior.npz` (80 000 Rat7M frames, Dunn et
al. 2021, CC BY 4.0; README with attribution; rebuilt by
`scripts/build_rat7m_prior.py`). Workbench: "3-D" tab after Preprocessing with
Camera calibration (box-marking dialog on a median frame), 3-D inference, 3-D
Kalman smoothing and a viewer launcher; sections registered for status badges.
CLI `mufasa-pose3d` (calibrate-box, calibrate-checkerboard, infer, smooth). Docs:
`docs/pose3d.md`. Tests: `smoke_122kf_pose3d`; `smoke_122em_section_id_audit` now
pins 10 wired sections.

**122ke — 3-D marker viewer with the video below.**
New `mufasa/ui_qt/pose3d_viewer.py`: the reconstruction volume on top and the
video as a viewport below (one splitter), kept in step — scrub or play the
video and the 3-D pose follows. The volume is drawn with QPainter (no OpenGL
or new dependencies): box, floor grid, depth-sorted markers, skeleton,
shadows on the floor, height lines and an optional trail; drag to orbit,
right/middle-drag to pan, wheel to zoom, double-click to reset; presets
Perspective / Top / Front / Side / Camera (the recording camera's own view).
"Reprojection" draws the 3-D markers on the video as the camera sees them.
Input: `<marker>_x/_y/_z` parquet/CSV with optional `frame`,
`<marker>_likelihood` and `mufasa_pose3d` metadata (box, edges, units,
camera as height/focal/principal point or a 3×4 `P`). Open it from the
feature viewer ("3-D…": a `<video stem>*3d*.parquet` beside the video or
under `derived/pose3d/`, else choose a file) or `mufasa-view3d VIDEO [POSE3D]`.
Test: `smoke_122ke_pose3d_viewer`.

**122kd — Kalman v2: confirmed prior settings are the defaults; serial and parallel runs now agree.**
The settings confirmed on the K7 data ("It works very well") are now the
defaults: robust outlier gate 25, segment-length prior 1, speed prior 1, and
the joint angle prior on (panel box ticked; CLI `--joint-prior` on by
default, `--no-joint-prior` turns it off; `--tune outlier_gate=0`,
`length_prior=0`, `speed_prior=0` turn the others off). Fixes two plumbing
bugs: the pooled EM E-step (`n_workers > 1`, what the GUI uses) did not pass
the joint/kinematic priors to the filter, and the serial final pass
(`n_workers = 1`) did not either, so serial and parallel runs fitted or
smoothed differently whenever a prior was on. Re-run on K7, results are
unchanged (hidden-head error median / p90 / p99 14.6 / 35.9 / 71.5 px).
Tests: `smoke_122kd_prior_defaults_and_pool_parity`; older tests pin the
previous values where they check other behaviour.

**122kc — Kalman v2: stretchier segment lengths by default.**
`length_q_scale` default 1 → 50 and `length_q_ceiling` 10 → 500 — the
setting recommended on 2026-11-11 and confirmed by the user ("That looks
much better"). On Cacna_87 (frames 20000–29000) it cut the 90th-percentile
error on confident frames for tail_V32 14.2 → 8.1 px, tail_V18 12.2 → 5.7,
back_V2 6.3 → 2.6, head_mid 7.0 → 4.1, and fixed the floating skeleton
seen in that session. Only training is affected (initial length noise and
its EM ceiling); loading a saved model is unchanged. The other new options
(outlier gate, length / speed priors) stay off by default. To get the old
behaviour: `--tune length_q_scale=1 --tune length_q_ceiling=10` or the
advanced view. smoke_122jy / smoke_122kb follow the new defaults;
smoke_122hf pins the old values for its synthetic joint-prior checks.

**122kb — Kalman v2 panel remembers its settings and the last model.**
Starting a run writes every option of the panel — mode, fps, likelihood
threshold, workers, overwrite, EM settings, model extensions, advanced fine
tuning (and whether that view is open), input folder — to
`logs/measures/kalman_v2_settings.toml` (the user's mufasa config folder
outside a project), together with the model the run trains or loads as
`last_model`. Building the panel restores them; the output folder keeps its
fresh run id and the save path stays inside it. The last model (if it still
exists) is put into "Load saved model", and that mode is kept when it was
used last. "Load saved model" now shows what the chosen model was trained
with (fps, threshold, extensions, joint prior, perspective, tuning, date),
and "Train a new model with these settings" copies them into the training
options. smoke_pose_cleanup_v2_wiring reads `collect_args` and its new
`_collect_args`.
Test: `tests/smoke_122kb_kalman_panel_remembers.py` (16 checks).

**122ka — Kalman v2: segment-length and spatial speed priors.**
`fit_kinematic_priors_v2` fits, once from confident frames (like the joint
prior), each segment's typical length and spread (IQR / 1.349) and the
root's typical (RMS) speed — overall and on a grid over the arena, so the
prior is tight where the animal rests and loose where it runs (cells with
< 30 samples fall back to the overall value; speeds floored at 5 px/s).
New tuning knobs `length_prior` and `speed_prior` (0 = off, the default;
otherwise the prior SD as a multiple of the observed spread / local typical
speed; 1 = as wide as the data) add pseudo-observations: each segment length
→ its typical length, the root velocity → rest, with the variance looked up
at the current position. Carried on `JointPriorV2` (new fields, saved with
the model; older models load without them) and in the advanced view.
Measured on K7 frames 1500–7500 with 30 head windows of 45 frames hidden on
purpose: hidden-head error median 15.0 → 15.3 px, p90 38.8 → 37.1, p99
93.5 → 71.0 (with the joint prior too: 14.6 / 35.9 / 71.5); all markers
hidden for 30 frames: unchanged. The RTS smoother already interpolates
across gaps, so the priors mainly trim the worst cases.
Test: `tests/smoke_122ka_kinematic_priors.py` (16 checks); smoke_122jy
skips the new zero-means-off knobs in its defaults check.

**122jz — Kalman v2: robust outlier gate (confident-but-wrong detections).**
On K7_cam_20251124_1510_A10-0-051_640x360-triplet the body "twisted" during
long low-confidence stretches (e.g. around frame 3767). The RTS smoother
already uses the whole future, so look-ahead was not the problem: while the
head is hidden FDLC still returns head markers at p ≈ 0.5–0.7 hundreds of px
away, which pass the likelihood threshold and pull the head (and body)
around. New tuning knob `outlier_gate` (χ², 2 dof; 0 = off, the default;
25 recommended; `--tune outlier_gate=25`, "Outlier gate" in the advanced
view): each marker's innovation is tested against its own predicted
covariance and, beyond the gate, its noise is inflated by d²/gate — a wild
detection barely pulls, a genuine fast move is still followed (a hard gate
that dropped detections lost lock on the tail). The test widens by itself
while the state is uncertain. Applied in EM and the final pass alike;
`FilterResultV2.n_gated` counts down-weighted detections and the run log
reports them. On frames 1500–7500 of that session (length ×50): head flips
> 60° within 3 frames 7 → 3, the 3720 / 3767 head twists gone, tail frames
> 30 px off a confident detection 82 → 31 (tail_V32), 52 → 15 (tail_V18).
Test: `tests/smoke_122jz_kalman_outlier_gate.py` (12 checks).

**122jy — Kalman v2: fine-tuning knobs and an advanced view.**
`BodyLayout.tuning` (saved with the model as `layout_tuning`, so it reaches
the worker processes and survives `--load-model`; recorded in the run
provenance) carries knobs that were hard-coded constants:
`drift_radius_cap` (max per-marker drift radius, 20 px), `length_q_scale`
(× initial segment-length noise, 1), `length_q_ceiling` (EM ceiling of the
length noise, 10×), `seg_q_ceiling` (EM ceiling of segment-angle noise,
10×; high-angular-noise segments keep max(40×, it)), `q_floor_factor`
(EM floor of all process noises, 0.1×), `sigma_ceiling` (EM ceiling of
marker σ, 3×) and `constraint_sigma` (unit-norm constraint softness,
0.05). Defaults reproduce the previous behaviour. CLI: `--tune KEY=VALUE`
(repeatable / comma-separated; ignored with `--load-model`). The Kalman v2
page gets "Show advanced fine tuning": a spin box per knob, unit-norm
constraints on/off, strict validation, device, Reset, and a summary of
changed knobs. The M-step aggregation combo now offers the smoother's
choices (pooled / median / trimmed_mean; "per-session" silently fell
through to a weighted mean). smoke_122hj accepts the tunable length ceiling.
Test: `tests/smoke_122jy_kalman_advanced_tuning.py` (30 checks).

**122jx — Viewer JPDF: bins and outliers.**
The "JPDF beside the video" row gets "Bins" (per axis, default 90) and
"Outliers": none, percentile tails (default 0.5 %, as before) or MAD
(default 5 ×), with the threshold. Outlying background frames (in either
feature) are left out and the axes fitted to the frames kept (previously
fixed 0.5–99.5 % of background + video); a current-frame marker outside is
pinned to the edge. The panel title gives frames kept, outliers left out
and bins. Changes redraw at once and carry over to a new pair
(`JpdfPanel.set_options`, `JpdfPanel.outlier_mask`).
Test: `tests/smoke_122jx_viewer_jpdf_bins_outliers.py` (13 checks).

**122jw — Style / visibility popup for the FDLC and Kalman overlays.**
"Style…" in the viewer's overlay row opens a non-modal popup with a tab per
source. Markers: show, shape (filled circle / ring / cross / square /
diamond), size (px or auto), line width, opacity, black outline, palette or
single colour, marker names; a table to hide single markers or give them
their own colour (double-click; Shift+double-click resets). Skeleton: show,
colour, width, opacity; a table to hide single segments or colour them.
Plus a legend switch and Reset to defaults. Edits redraw the video at once
(and so go into recordings); Close saves to
`logs/measures/pose_overlay_style.toml`, used by every viewer of the
project. Defaults keep the 122ju look (FDLC rings without skeleton, Kalman
dots with skeleton); FDLC can now have its own skeleton. The row's
"Skeleton" box stays a master switch. New module
`mufasa.ui_qt.overlay_style`.
Test: `tests/smoke_122jw_overlay_style_popup.py` (27 checks).

**122jv — Feature viewer: JPDF beside the video, recorded with it.**
"JPDF beside the video" with x / y feature combos (the viewer's features,
by their figure names) and a Trail length composes, to the right of the
displayed frame, the 2-D log density of the pair pooled over all videos
(≤ 5000 frames each; −1 as missing; log10 scales and labels from
`feature_names.toml`; background jittered for jitter-marked features) with
the current frame's position as a solid red marker (white rim) and a red
trail fading over the last N frames. The panel is square at the frame
height; the density is drawn once per pair / height and cached, each frame
only draws marker and trail. It is composed after overlay and orientation
(`FrameScrubberWidget.set_composer`), so "● Record" saves video + JPDF side
by side. New module `mufasa.ui_qt.jpdf_panel`.
Test: `tests/smoke_122jv_viewer_jpdf_beside_video.py` (19 checks).

**122ju — Feature viewer: FDLC / Kalman marker overlays and recording.**
The video × feature viewer finds the video's tracker pose under
`sources/pose` (`<video>.fdlc.parquet`, `<video>.parquet`, a FreeDLC run's
`pose.parquet` named by its run sidecar, or CSV) and its file in the latest
Kalman v2 run. "FDLC markers" draws hollow rings (hidden below the "p ≥"
likelihood threshold), "Kalman markers" filled dots joined by the skeleton
("Skeleton"; project / pose-file skeleton); "Pose scale" multiplies the
coordinates. Overlays are drawn in video coordinates before the display
rotation (`FrameScrubberWidget.set_overlay`); boxes are disabled with the
reason when a source is missing. "● Record" keeps every frame shown from
then on — playing or stepping, overlays included — as numbered PNGs in
`derived/frames/extracted/<video>_rec_<time>/` with `frames.csv` (image →
video frame); pressing it again encodes them at the video's frame rate into
`derived/frames/annotated/<video>_rec_<time>.mp4` (ffmpeg H.264 when
installed, else OpenCV mp4v; in the background). "Keep frames" off deletes
the PNGs afterwards. New module `mufasa.ui_qt.pose_overlay`.
Test: `tests/smoke_122ju_viewer_pose_overlay_and_record.py` (25 checks).

**122jt — JPDF of the last ticked pair beside the pair matrix.**
The "Pairs to animate" tab is split: the matrix on the left, on the right a
joint PDF (2-D log density, `plot_joint` heatmap) of the last ticked pair,
pooled over all videos (≤ 3000 frames each, read once; -1 as missing). It
uses the names, log10 scales and jitter currently set in the names table
and redraws when ticks, names, scales or jitter change; unticking shows the
previous pair, nothing ticked shows a hint.
Test: `tests/smoke_122jt_pair_matrix_jpdf_preview.py` (11 checks).

**122js — Pair matrix headed by the custom feature names.**
The matrix rows and columns now show exactly the names typed in the names
table (no "x:"/"y:" prefix, no truncation; column headers wrap onto
several lines, rows stay on one), updated as you rename; the tooltip gives
the column name and its axis. The hint above says rows are x, columns y.
smoke_122jr updated (22 checks).

**122jr — Animation pairs chosen in a feature × feature matrix.**
The Animate… popup gets a "Pairs to animate" tab: a grid of tick boxes with
the selected features as rows (x axis) and columns (y axis); the mirror
cell swaps the axes, the diagonal is unused. Ticked cells are numbered in
the order ticked — the animation order (t-SNE → 1 → t-SNE → 2 → …).
Pre-filled with the pairs used last time for these features, else the
best-ranked; buttons Best-ranked (from an earlier landscape of the
selection, up to "Best-ranked joint spaces"), All pairs (upper triangle),
Clear. Generate is disabled with nothing ticked. The pairs go to
mufasa-landscape-animate as `--pair`s and are kept per selection in
`logs/measures/feature_view.toml` (`animation_pairs`); renaming a feature
relabels the matrix.
Test: `tests/smoke_122jr_animation_pair_matrix.py` (21 checks).

**122jq — Feature selection kept between sessions.**
Feature Visualization saves its selection (in order), video, colour-by,
joint style, jitter and "Group by name parts" to
`logs/measures/feature_view.toml` (`mufasa.utils.feature_view_state`;
debounced 0.5 s, and flushed on quit) and restores them when the form is
built. Selected features the current video lacks are parked rather than
dropped, and come back on a video that has them; unknown videos / features
in the file are ignored, an unreadable file starts empty.
Test: `tests/smoke_122jq_feature_selection_persistent.py` (15 checks).

**122jp — Animation: remove outliers.**
mufasa-landscape-animate `--outliers none | percentile | mad` leaves out,
before the points are sampled, frames that are outliers in any animated
feature: outside `--outlier-percentile` (default 1 %) on either tail, or
more than `--outlier-mad` (default 5) scaled MADs from the median (SD when
over half the values are identical); `--outliers-tsne` tests the t-SNE
coordinates too. The log says how many frames each feature flagged. The
landscape run is untouched. The Animate… popup has "Remove outliers"
(method, threshold, "also t-SNE coordinates").
Test: `tests/smoke_122jp_animate_outliers.py` (15 checks).

**122jo — Animation: pause and transition in frames.**
mufasa-landscape-animate takes `--transition-frames` (frames per move),
`--hold-frames` (pause on each feature space, 0 = none) and
`--tsne-hold-frames` (pause on the t-SNE map, default as `--hold-frames`),
overriding the seconds options; `timeline()` takes per-space holds. The
Animate… popup has the three spins (defaults 30 / 30 / 40 frames — the
previous 1.5 s / 2 s at 20 fps).
Test: `tests/smoke_122jo_animate_pause_and_transition_frames.py` (12 checks).

**122jn — Jitter near-discrete features in figures.**
The Landscape… / Animate… names table gets a Jitter tick per feature, saved
as `[jitter]` in `logs/measures/feature_names.toml`. Marked features are
drawn with uniform noise of ±0.4 × the median step between their distinct
values (`feature_names.jitter_values`; levels stay separable, NaN stays
NaN, reproducible): mufasa-landscape's joint PDFs (titled "x/y jittered")
and mufasa-landscape-animate's feature spaces. The dip test, its boundaries,
`tsne_samples.parquet` and the t-SNE keep the real values. Both tools take
`--jitter COL` (repeatable) and `--no-jitter`. Feature Visualization's joint
plot gets a Jitter checkbox and always jitters marked features.
Test: `tests/smoke_122jn_jitter_discrete_features.py` (19 checks);
smoke_122jl allows the extra table column.

**122jm — Animation: number of points, sampling method, session intro.**
mufasa-landscape-animate gets `--sampling random | balanced | even` next to
`--n-points`: random as before (dense regions dominate, as in the data),
balanced takes the same number from every session, even spreads the points
over the t-SNE map (a 40 × 40 lattice drawn from in turn) so sparse regions
stay visible. The GIF now opens with `--intro-frames` (default 40) frames of
the t-SNE map coloured by session ("coloured by session (N)" under the
title), then eases over `--intro-fade` (20) frames into the `--color`
colouring; skipped when colouring by session anyway. Session colours stay
distinct beyond 20 sessions. The Animate… popup has Points in animation,
Sampling and Intro frames. smoke_122iy runs its timing check with
`--intro-frames 0`.
Test: `tests/smoke_122jm_animate_sampling_and_session_intro.py` (19 checks).

**122jl — Names popup now opens; per-feature linear / log10 scale.**
The Landscape… / Animate… buttons connected clicked(bool) straight to
`generate_*(ask=True)`, so clicked's `False` skipped the names popup; they
now call with no arguments. The popup's table gains a Scale column
(linear / log10 per feature, "Set all scales" buttons), saved as `[scales]`
next to the names in `logs/measures/feature_names.toml`. mufasa-landscape
analyses log10 features as log10 (values ≤ 0 → missing), labels them
"log10 <name>", and lists them in the run's `log10_columns.txt`
(`--no-scales` to ignore the marks); mufasa-landscape-animate labels them the
same way, and a landscape made with other scales is not reused.
Test: `tests/smoke_122jl_feature_scales_and_names_popup.py` (19 checks).

**122jk — Landscape / animation from the selection; figure names; joint
styles; viewer grid and follow.**
Feature Visualization gets *Landscape…* and *Animate…*: the ticked features
go through mufasa-landscape (`--sources features --columns-file`, pooled over
all sessions; exactly the selection — no family filter, no near-duplicate
pruning, full-circle angles kept — and the best-ranked pairs plotted whether
or not they are multimodal) and mufasa-landscape-animate (`--through-tsne`
with explicit `--pair`s: t-SNE → pair 1 → t-SNE → pair 2 → … → t-SNE; an
earlier landscape of the same selection is reused). Both run as child
processes with the log streaming into a dialog that then previews the PNGs
and plays the GIF. Before either runs, a table of the selected features
(column | name in figures) is shown; edits are saved to
`logs/measures/feature_names.toml` (`mufasa.utils.feature_names`) and used by
both tools (`--names-file`, default the project's file) and the form's joint
plot. Joint plots take a style: heatmap (now with a colour bar), scatter or
scatter-heatmap. The video × feature viewer draws a light 5-line grid per
lane (0/25/50/75/100 %) and follows the selection: ticking / unticking
features updates open viewers (debounced; "Follow selection" to opt out).
Test: `tests/smoke_122jk_landscape_from_selection.py` (52 checks);
smoke_122je allows the heatmap colour bar axes.

**122jj — Feature Visualization: swap axes, linear / log10.**
New "Axes:" row under the plot buttons: *Swap X ↔ Y* and an X and a Y scale
(linear / log10) for the displayed axes, applied to the plot on screen at
once from cached data (no re-read; cache cleared on Refresh). Swapping also
exchanges the two scale settings, so each quantity keeps its scale.
Distributions: values on x, density on y (swap → sideways histograms); a
log value axis uses log-spaced bins and leaves out values ≤ 0, counted in
the panel; AUC unaffected. Joint: the features exchange axes; log axes use
log-spaced bins, frames ≤ 0 on a log axis are counted in the title. Over
time: time on x (swap → time running down y, panels side by side);
double-click reads time from whichever axis holds it.
`plot_distributions` / `plot_joint` / `plot_over_time` take `swap`, `xlog`,
`ylog`.
smoke_122jj_feature_view_axes: 19/19. 122je 38/38, 122jg 20/20.

**122ji — Compute Features: ROI families show "modified" when the ROIs change.**
ROI-family outputs (distances to ROI centres, body parts inside ROIs) now
carry in their parquet metadata a fingerprint of the ROIs each video was
computed from (`RoiStore.fingerprint`: hash of names, shapes, geometry,
order; `write_with_roi_fingerprint` / `read_roi_fingerprint`). New
`feature_io.roi_family_status` compares outputs with rois.json: *changed*
(a ROI added, removed, renamed, moved or reshaped), *missing* (a video
gained ROIs after the run), *removed* (a video lost its ROIs), *unknown*
(output from before 122ji and older than rois.json). The Compute Features
form badges such a family "● … — modified" in amber with a tooltip listing
the videos per reason; it re-checks when rois.json changes on disk (file
watcher on the file and its folder, 300 ms debounce), when shown, and after
a run. Recomputing clears it. Subject families unaffected.
smoke_122ji_roi_features_modified_status: 16/16 (real ROI-family run;
add / move / remove ROIs; untracked old outputs; watcher; show; recompute).
Badge, Compute Features, ROI and feature-pool suites as before.

**122jh — ROI "Apply to all" copies only the selected ROI; Manage classifiers above Features.**
ROI editor: the button is now "Apply '<roi>' to all…", enabled only with a
ROI selected. It copies that one ROI to every other video, rescaled to each
video's frame size; a ROI with the same name there is replaced in place,
their other ROIs are kept, the source video is unchanged. The confirmation
names the ROI and lists videos that already have one of that name. New
`RoiStore.copy_roi` and `roi_utils.copy_roi_to_all`. The subset button is
relabelled "Copy all ROIs to chosen videos…" (behaviour unchanged).
Workbench: the Manage classifiers page is built right after ROI, above
Features (Annotation / Train / Validate / Inference still follow Features).
smoke_122jh_roi_apply_one_and_page_order: 18/18 (copy_roi in-place replace,
keep, add, rescale, error; editor: disabled without selection, question
text, decline, apply across a mixed-resolution project, status; page order
by AST). 122ij and 122dl updated to the new button behaviour / label; ROI
and page-order suites otherwise as before.

**122jg — feature tree compacted by shared name parts.**
Inside each family the wording all its features share is dropped from the
leaves and shown once beside the family (`Angle (degrees) …`,
`Animal_1 … video edge distance (mm)`), and features are grouped by their
first differing component — the first body part — with a second level when
a group would still hold more than 24 (455 three-point angles of 15 markers
→ 13 body-part groups → sub-groups by the second body part; 1365 four-point
hulls likewise). Singletons and small families stay flat. Groups and
sub-groups have their own none / some / all checkbox with counts; a tick
selects their shown features. Search matches feature, group and family
names. "Group by name parts" switches to flat leaves (still without the
shared wording). Expansion is remembered by node path across rebuilds and
view toggles. Search is debounced (200 ms); building / ticking a 2455-leaf
tree takes < 1 s / ~0.1 s.
Qt-free grouping in `feature_plots.compact_family` / `Node`.
smoke_122jg_feature_tree_compact: 20/20 (real feature-name shapes of the
15-marker layout: templates, grouping levels, loose leaves, every column
once; tree states, counts, group / family ticks under search, flat toggle,
expansion memory, debounce). smoke_122je_feature_visualization 38/38.

**122jf — Feature Visualization: features as a tree by family.**
The family drop-down and flat list are replaced by a tree: one bold branch
per feature family, its features as checkable leaves. A family's checkbox
shows none / some / all selected and its title the counts ("1/3
selected", "2/455 shown" while searching); ticking a family takes all its
*shown* features, so search + family tick selects a subset. Search matches
every word against feature and family names, hides families with no
match and expands the rest. Expand all / Collapse all; expansion and
selection survive rebuilds (video change, Clear selection).
smoke_122je_feature_visualization: 38/38 (tree structure, counts,
partial / full family states, family tick under a search, search over
family names, rebuild persistence).

**122je — Features → Feature Visualization.**
New section on the Features page (`FeatureVisualizationForm`). Pick a video,
filter features by family and by search words, tick any number (the
selection survives filtering), then:
*Distributions* — histogram per feature (≤ 12), this video or all videos
pooled; with *Colour by* a behaviour, labelled frames vs the rest with the
feature's AUC for that behaviour in each panel and the best-separating
features summarised (videos with no labels for it are left out of the split
and reported). *Joint* — two features as a 2-D log density with behaviour
frames overlaid. *Over time* — whole session per feature (min/max decimated)
with behaviour bouts shaded; double-click opens the video there. *Open with
video* — `FeatureViewerWidget` in a floating dock: the scrubber with one
trace lane per feature under it (each scaled to its 0.5–99.5 % range over
the session, value at the cursor beside the name), a behaviour strip,
click / drag to seek, wheel or the ±window box to zoom.
Only the selected columns are read: `feature_io.feature_catalog` (schemas
only) and `feature_io.load_feature_columns` (parquet column projection).
Plot drawing is Qt-free in `mufasa/tools/feature_plots.py` (`auc`,
`plot_distributions`, `plot_joint`, `plot_over_time`).
smoke_122je_feature_visualization: 31/31 (offscreen on a generated project
with real videos, two feature families and labels: catalogue / projection,
AUC vs sklearn, filtering and selection, every plot, scope and the
unlabelled-video rule, guards, viewer seek / drag / wheel / spin, page
registration). Page / section / feature-I/O suites as before.

**122jd — Frame Labeling table: "Frames labelled" column.**
New column after Status: frames carrying at least one behaviour, out of the
video's frames, with the percentage (a frame with two behaviours counts
once). The per-behaviour breakdown moves to "Per behaviour". Sorts by count.
`VideoAnnotation.any_labelled` in `mufasa.utils.annotation_status`.
smoke_122jc_labeling_status_complete_plot_seek: 32/32 (column position,
union counting, sorting, live refresh).

**122jc — Frame Labeling: annotation status table, "Mark video complete", click-to-seek in the label plot.**
Frame Labeling page: a table of every project video with its status
(✓ complete / ● in progress / ○ saved, empty / — not started), labelled
frames per behaviour (count and %), and last save; a summary line; numeric
sorting; Refresh, plus live refresh whenever a labeller saves or changes a
complete mark (`annotation_events()` signal). **Label** (or double-click)
opens the selected video directly; with no selection it still asks for a
file. Choosing *New labeling* for a video that already has labels asks
first and offers Continue (New would start empty and overwrite on save).
Labeller: **Mark video complete** checkbox next to Save, initialised from
and written to `derived/labels/annotation_status.toml` (new
`mufasa.utils.annotation_status`); checking it saves unsaved labels first.
Label plot: left-click (or drag) inside the lanes moves the video to that
frame, using the same window maths as the drawing; clicks on the name
column are ignored; seeks clamp to the video.
Two fixes found on the way: the TOML writer emitted every key bare, so a
key with a dot or space (a dotted video name, a classifier called "head
dip" in `[classifiers.keys]`) produced a project.toml that could not be read
back — keys are now quoted when needed (`_toml_key`); and the label /
feature / classification / CSV-export helpers stripped "extensions" with
`Path.stem`, so labels for a video named `rec.v2` were saved as `rec` — they
now strip only real video / table extensions (`strip_media_ext`).
smoke_122jc_labeling_status_complete_plot_seek: 30/30 (offscreen form +
labeller on real video files: table, sorting, launch / guard / double-click,
complete mark incl. save-first and live refresh, plot click / name column /
edge / drag / clamp, TOML keys, extension stripping). Labelling, label /
feature / classification I/O and project-layout suites as before.

**122jb — mufasa-landscape: angles stop taking over; family filters and caps.**
User report (after 122iz): still only angle pairs. With 15 markers the
three-point-angle family has one column per marker triple (455), and much
of their structure is circular in ways rotating the 0/360° cut cannot fix:
angles covering the whole circle (no empty direction) and mirror images
(bent left / bent right). Now:
`--angles unwrap` (default) also drops full-circle angles (emptiest
direction > 25 % of the mean density); `--angles fold` maps a → min(a,
360 − a) so mirror images coincide; `--angles drop`.
`--include-families` / `--exclude-families` (substrings of the family folder
name; `kfv2` = pose) restrict the screen.
Plot selection caps pairs touching any one family (`--max-per-family`,
default 6; animate default 2) on top of the per-column cap; the run says
when the caps leave slots empty.
`pairs.csv` gains `family_a` / `family_b`; the log counts multimodal pairs
per family.
smoke_122jb_landscape_angles_and_families: 14/14 (unwrap / fold / drop on
mirror, full-circle and wrap-around angles; a project with a 40-column
circular angle family beside distance / movement / head-velocity families
sharing a two-state structure: families spread in the plots, head velocity
among them, fold cuts angle×angle multimodal pairs 231 → 0, include /
exclude / drop). 122ix / 122iy / 122iz pass.

**122ja — fix: HEAD VELOCITY IN BODY FRAME failed with NameError 'warnings'.**
User report: every video failed with "NameError: name 'warnings' is not
defined". `feature_subset_kernels` imported `warnings` only inside the
fallback branch taken when the compiled (Cython) kernels are missing; the
122iz kernel used it at module scope. Without compiled kernels (the
development sandbox) the fallback ran and the name existed, so 122iz's tests
passed; on an installed copy the compiled kernels load and the name was
never bound. `warnings` is now imported at top level.
smoke_122ja_kernels_module_imports: 4/4 — AST check that every module global
a kernel function uses is bound on both import paths (would have caught
this), plus a run-time import with a stub `mufasa._native` (the compiled
path) and a head-velocity call; fails 1/4 on the 122iz file. 122iz, kernel
and Compute Features suites as before.

**122iz — head velocity in the body frame; landscape: angles, -1 placeholders, per-family log.**
User report: mufasa-landscape "only used the angles". All families were
loaded (checked on a real Compute Features output), but angle columns
crowded every plot for two artefacts the screen took for modes: angles wrap
at 0/360° (an angle near 0° splits into lumps at 0 and 360), and the feature
writer stores missing values as -1. Now `clean_columns` reads -1 in
non-negative columns as missing and rotates each angle column so its wrap
falls at its emptiest direction (still degrees); both are logged, and the
log lists columns kept / loaded per feature family (from the per-family
parquet schemas), plus which families lost columns missing in some sessions.
New feature family **HEAD VELOCITY IN BODY FRAME (MM/S)**
(`compute_head_velocity_body_frame`): head point = mean of the head markers,
body point = mean of the body markers, axis = body → head; the head point's
velocity (central differences, mm/s from px/mm and fps) split into
*forward* (along the axis) and *lateral* (to the animal's left, image y
down). Markers from project.toml `[features] head_markers/body_markers`,
else `[[pose.segments]]` (segment `head` / `[features] head_segment`, root
segment / `body_segment`), else marker names; the run prints which and from
where, and an unresolvable set raises saying what to add. Signed, so its
parquet keeps NaN instead of the -1 placeholder (`SIGNED_FAMILIES`). Listed
in Compute Features automatically. `[features]` documented in
docs/v1_project_layout.md.
smoke_122iz_head_velocity_and_landscape_columns: 24/24 (kernel geometry:
forward/lateral sign and magnitude, |v|² = f² + l², rotation invariance,
missing markers; resolution order and errors; a real FeatureSubsetsCalculator
run on a [[pose.segments]] project; the form's family list; angle unwrap,
-1 handling, family log). smoke_feature_kernels' documented-kernel list
gains the new kernel. Feature/landscape suites as before.

**122iy — `mufasa-landscape-animate`: frames gliding between t-SNE and feature spaces.**
New `mufasa/tools/landscape_animate.py` (console script
`mufasa-landscape-animate`) renders a presentation GIF (`--mp4` adds an MP4)
from a `mufasa-landscape` run: the sampled frames move from the t-SNE map
into the joint scatter of each chosen pair and back (t-SNE → p1 → p2 → … →
t-SNE, or `--through-tsne` to return between pairs), with a hold on each
space and cubic ease-in-out moves. Colour stays fixed per point — by default
its position in the t-SNE map (hue = direction, saturation = distance from
the centre), so each map region can be followed into every feature space;
`--color session|<column>` otherwise. Each space is shown in its own units
(0.5–99.5 percentile range; t-SNE keeps its aspect); title, axis labels,
ticks and a faint density backdrop cross-fade with the motion. Pairs: `--top
N` from `pairs.csv` (same ranking, `--max-per-feature` default 1) or
`--pair A B` (repeatable). One shared palette for all GIF frames (no
flicker); `--theme dark|light`, `--size`, `--n-points`, `--fps`, `--hold`,
`--transition`. Run found from a run dir or the project (latest run).
`mufasa-landscape` now also writes `tsne_samples.parquet` (the embedded
frames' column values, row-aligned with `tsne.parquet`), which the
animation reads — no recomputation; runs from 122ix lack it and get a
clear "re-run mufasa-landscape" error.
smoke_122iy_landscape_animate: 20/20 (timeline and easing, GIF size / loop
/ duration, start = end, exact point positions at the end and middle of a
move, aspect, colour modes, --top / --through-tsne / project discovery,
MP4, error paths). 122ix, 122iw, 122gs pass.

**122ix — `mufasa-landscape`: dipsplit joint PDFs and t-SNE across sessions.**
New `mufasa/tools/feature_landscape.py` (console script `mufasa-landscape`,
needs the new `analysis` extra = `diptest`). Loads, for every session, the
latest Kalman v2 run's smoothed pose (x/y, prefixed `kfv2.`; `--pose-cols all`
adds p and variances) and all feature families, samples
`--frames-per-session` frames from each, and keeps the columns all sessions
share. Sessions lacking a requested source (`--sources`) are skipped and
named; pose/feature length mismatches are reported and truncated.
Screen: columns that are mostly NaN, near-discrete (flags such as inside-ROI
0/1, which are trivially "bimodal") or near-duplicates (|r| > 0.98) are
dropped; every remaining pair is robust-z-scored and projected on the two
axes, both diagonals and both principal axes. dipsplit counts modes per
projection: Hartigan's dip test (p ≤ `--alpha` and dip ≥ `--min-dip`, an
effect-size floor because at tens of thousands of frames the test alone
flags skewed/heavy-tailed unimodal data) decides, the deepest valley of the
smoothed density is the cut, both halves are re-tested. Pairs are ranked
emergent (both columns unimodal, a joint projection not) → adds modes (more
modes jointly than in either column) → by dip. Plots: joint log-density with
marginals and split boundaries (joint splits drawn as lines in raw units) for
the top `--n-plot` pairs, at most `--max-per-feature` per column
(`jpdf_top.png`, `jpdf_pairs.pdf`); t-SNE (PCA to 50, perplexity 30) on
`--tsne-sample` frames coloured by session and by the columns most frequent
among the plotted pairs (`tsne.png`, `tsne.parquet`). All screened pairs in
`pairs.csv`; provenance in `run.toml`; output under
`derived/plots/feature_landscape/<run_id>/`. About 3 ms per pair on 20k
frames (300 columns ≈ 2.5 CPU-minutes, parallel over `--n-jobs`).
smoke_122ix_feature_landscape: 24/24 (dipsplit mode counts on normal,
lognormal, t3, uniform, 2- and 3-component mixtures; a generated project
with a planted emergent pair, a bimodal column, a 0/1 flag, an orphan
session, a stale run and a length mismatch; every output). README command
and extras tables updated (122iw tripwire passes).

**122iw — README brought up to date with the project.**
The README described an older Mufasa. Corrected against the code:
license (said GPL v3; LICENSE is SimBA's BSD 3-Clause, modified, academic
use only — badge and section fixed); install (`pip install -e .` gave no Qt —
PySide6 is in the `qt` extra — now `.[qt]`, with the other extras listed and
the Cython build noted; clone URL Gravios/mufasa; TensorFlow mention
dropped); commands (`mufasa-preview` added, removed-command history cut to
one line); a table of the workbench pages; the standalone Kalman v2 command;
project tree (`derived/pose|interpolated|smoothed/kalman_v2/<run_id>`,
publishing into `outlier_corrected`, per-family features, flat labels /
classifications, `models/` for Kalman models + classifiers,
`logs/measures/rois.json`, `backups/`); importers (FreeDLC first); docs
table (Kalman design docs, testing workflow, CHANGELOG); "Relationship to
SimBA" now says how Mufasa diverged (own formats, no SimBA projects) instead
of "core algorithms are identical"; citation asks for SimBA for the
classifier pipeline. `docs/v1_project_layout.md` got the same tree, a
corrected `derived/` table and `models/` section, rois.json, and lost
"Both layouts remain supported".
smoke_122iw_readme_current: 22/22 — each README fact checked against its
source (pyproject scripts and extras, page titles, import form, layout code,
LICENSE, smoother flags, links). 122df's GPL check now checks the README
names LICENSE's license. README/doc tripwires pass; 122cn as before.

**122iv — banner tool icons head their doc sections.**
The banner's three tool tiles are standalone SVGs: `docs/images/icon_smoothing.svg`,
`icon_roi.svg`, `icon_classifiers.svg`. As in neurosuite-3's README, a new
"The tools" section has one collapsible `<details>` block per tool — icon and
bold name in the `<summary>`, four bullets, doc links. The icons also head
the matching sections of `docs/workflows.md` (Smooth pose; Stages 4–8) and
the Kalman design track in `docs/README.md`. Links into workflows.md point at
the file, not at heading anchors, since an image in a heading changes
GitHub's anchor.
smoke_122iv_docs_section_icons: 9/9. Doc tripwires unchanged (122cn fails
as before).

**122iu — README / docs banner (wide + mobile, dark + light).**
New `docs/images/`: the Mufasa banner in the neurosuite-3 layout and the lion
palette — `mufasa_banner_{dark,light}` (2560x800), a stacked narrow-screen
version `mufasa_banner_mobile_{dark,light}` (1200x820) and a GitHub social
preview (1280x640), each as PNG plus SVG source. The README and the docs
index open with a `<picture>` that picks mobile vs wide by viewport width
(`max-width: 700px`, listed first because the first matching source wins)
and light vs dark by `prefers-color-scheme`; the dark wide banner is the
fallback `<img>`. Tagline: "Kalman pose smoothing and animal behavior
analysis from video pose-tracking data".
smoke_122iu_readme_banner: 11/11. 122cn (docs index lists 5 unlisted docs)
and 122br (SimBA.py gone) fail as before.

**122it — Mufasa lion logo: app icon and startup splash.**
New `mufasa/assets/branding/`: the lion-head symbol (`mufasa_icon_<16…512>.png`,
`.svg`, `.ico`, transparent background) and the symbol + MUFASA wordmark
(`mufasa_splash.png`, `mufasa_wordmark.svg`). New `mufasa.ui_qt.branding`:
`app_icon()` (all sizes in one QIcon), `splash_pixmap()` (opaque dark card —
no reliance on window translucency, which remote X11 often lacks),
`MufasaSplash`, `show_splash()` (skipped with MUFASA_NO_SPLASH=1 and on
offscreen/minimal platforms). The workbench entry point sets the app icon
and desktop file name, shows the splash with "Opening <project>…" while the
workbench is built and closes it when the window appears; the main window
icon is the lion instead of the SimBA logo; `Paths.LOGO_ICON_*` point at the
new files.
smoke_122it_branding: 10/10 (assets + transparency, icon sizes, splash,
a real workbench window's icon, entry-point wiring). 122d5 module ceiling
432 → 433. Workbench suites unchanged.

**122is — parallel feature extraction: spawn workers, not fork.**
User report: Compute Features with 16 workers → every video "BrokenProcessPool:
A process in the process pool was terminated abruptly". Reproduced here: the
workers print "Terminating: fork() called from a process already using GNU
OpenMP, this is unsafe." `FeatureSubsetsCalculator._run_parallel` used a
plain ProcessPoolExecutor (Linux default start method: fork) from a process
that had already run numba-parallel kernels (OpenMP threading layer), and a
forked child of a process with live GNU OpenMP threads aborts; one dead
worker breaks the whole pool. It also forked the Qt GUI process from its
worker thread. The pool now uses `mp_context=get_context("spawn")`; the
console-script launchers guard `main()`, so spawned workers don't re-run the
app. Not a leftover pool from an earlier attempt — each run makes a new one.
smoke_122is_feature_pool_spawn: 4/4 — OpenMP threads started first, then a
2-worker run, output identical to sequential (~16 s). On the fork pool:
0/4 with the same "fork() … GNU OpenMP" abort. Feature suites unchanged.

**122ir — published Kalman runs appear under video names.**
User report (via 122iq's new message): ROIs defined for 39 videos
(`K7_cam_…-triplet`), tracking files named `K7_cam_…-triplet_smoothed_v2.
4d4273a837d0cc40`. Kalman v2 writes `<video>_smoothed_v2.<hash>.parquet`;
`publish_to_stage` linked the whole run directory into
derived/outlier_corrected/, and every downstream tool (features, ROI
analysis, classifiers, plots) takes the file name as the video name — so
nothing keyed by video (ROIs, video_info calibration) matched. Only the
egocentric aligner had a private suffix-stripping helper.
* `project_layout.video_name_from_pose_stem` (moved from the egocentric
  aligner, which now uses it).
* `publish_to_stage`: when a run's pose files carry the smoother suffix, the
  published run is a real directory of relative per-file links named
  `<video>.parquet` (+ a `.published_from` marker), built in a temp dir and
  swapped in; idempotent; other runs keep the plain directory link; real
  data directories are still never clobbered.
* `refresh_published_runs`, called by ConfigReader when a project is
  opened, upgrades runs published the old way (links only, never data).
smoke_122ir_published_video_names: 12/12 — the user's layout (directory
symlink to a run of `_smoothed_v2.<hash>` files), ROI + calibration features
computed for every video with correct values. Tripwire: without the refresh
on open the publish checks fail. Full sweep: no regression.

**122iq — ROI-feature "no coverage" error says why.**
User report: "NO ROI DATA ERROR: … none of the 39 videos have all required
ROIs defined." ROI features need, for every tracking file, a video with the
same name in rois.json carrying every ROI name used in the project (each ROI
is a feature column). The message didn't say which condition failed. Now
`_explain_no_roi_coverage` reports either that no ROI video name matches a
tracking file (showing both sides' names), or the project's ROI names and
what each video lacks (with the fix: rename in the ROI table, or Apply to
all / selected) — the latter is easy to hit with the editor's automatic
names.
smoke_122iq_roi_coverage_message: 5/5 (real projects, FeatureSubsetsCalculator
with the ROI families: name mismatch, inconsistent names, consistent names →
features written). On the previous code 1/5.

**122ip — the calibration is actually saved.**
User report: calibrated in Preprocessing → Video Calibration, yet Compute
Features said video_info.csv was missing. `VideoInfoForm` inherited the
standard Run button; Run called its no-op `target()` and reported "Done.",
while only a separate "Save to video_info.csv" button in the action row wrote
the file (the form's comments claimed there was no Run button and that
on_run was overridden — neither was true). Now:
* Run is labelled "Save calibration" and saves (`on_run` → `_save`).
* Edits and per-row calibrations flag "● Unsaved changes" with the target
  path until saved; a fresh load is not an edit.
* The missing-file error (read_video_info_csv / ConfigReader) names
  Preprocessing → Video Calibration instead of SimBA's "[Video parameters]
  tab"; stale logs/video_info.csv wording in the form fixed.
smoke_122ip_calibration_save: 12/12 (form offscreen on a generated project,
saved CSV read by read_video_info_csv and ConfigReader). Tripwire: removing
only the Run→save override reproduces the report (file never written).
smoke_video_info_form, 122ag, 122er unchanged.

**122io — Compute Features runs again (shap, numba cross2d, ROI shape types).**
User report: Compute Features → "preflight failed … No module named 'shap'".
* `train_model_mixin` imported `shap` at module level; shap was dropped from
  the dependencies, and feature computation (`FeatureSubsetsCalculator`)
  inherits the mixin. shap is now imported inside the SHAP helper only
  (annotations via `TYPE_CHECKING`). Feature computation, training,
  inference and result plotting all import without it.
* `perimeter_jit` imported `numba.np.extensions.cross2d`, gone in numba 0.68
  (pyproject allows any numba ≥ 0.60) — the convex-hull families failed to
  import. The 2-D cross product is written out; results match scipy's
  ConvexHull to 1e-11.
* ROI tables said `Shape_type` "Rectangle" / "Circle" / "Polygon" (inherited
  from the old ROILogic writer); the inside-ROI feature kernel and the ROI
  classifier code compare to SimBA's lower-case values, so "body-parts
  inside ROIs" silently wrote no file for Mufasa-drawn ROIs. `roi_store` now
  emits lower-case; `ROI_directing_analyzer` (the one consumer expecting
  "Circle") compares case-insensitively.
smoke_122io_compute_features: 8/8 — shap blocked via a meta-path finder,
a generated project with ROIs, preflight + every family, inside-ROI booleans
checked against the geometry. Tripwires: top-level `import shap` and
capitalised shape types each fail it (cross2d is absent from this numba, so
the old perimeter_jit fails too). Full sweep: no regression.

**122in — a click is not a drag (ROI cycling with a real mouse).**
User report after 122im: the ROI under another still couldn't be selected.
Cause: no drag threshold. A real click moves the cursor a pixel or two
between press and release; the Select tool treated that as a drag, nudged
the selected ROI by that amount and, since it "moved", never cycled to the
ROI beneath. 122im's test used QTest.mouseClick, which never moves, so it
passed. Moving / reshaping now starts only once the cursor has travelled
`QApplication.startDragDistance()` (default 10 px); below that the press is
a click.
smoke_122im_roi_overlap: +4 checks with realistic clicks (2 px jitter):
cycling works, ROIs and sides are not nudged. On the 122im canvas those
fail (12/17); now 17/17. 122ij, 122dm, 122fd, smoke_roi_canvas unchanged.

**122im — overlapping ROIs stay reachable.**
The Select tool resolved a click to the top-most (most recently drawn) ROI
whose area contained it, so an ROI under a newer one could not be selected,
and its sides / polygon vertices could not be dragged where the newer ROI
covered them. `ROICanvas._hit_test` priority is now:
1. handles of every ROI (polygon vertices, rectangle corners / side
   midpoints, circle rim handle), the selected ROI's first;
2. the nearest outline of any ROI — rectangle sides (resize that side),
   circle rims (radius), polygon edges (new: drag moves both vertices);
3. interiors, smallest area first; clicking the same spot again (without
   dragging) cycles through the overlapping ROIs.
smoke_122im_roi_overlap: 13/13 (editor offscreen via QTest: small rectangle,
polygon and circle under a newer large rectangle). Tripwires (top-most
interior wins, no outline priority, no cycling) each fail it. 122ij, 122dm,
122fd and smoke_roi_canvas unchanged.

**122il — ROIs stored in `logs/measures/rois.json`; SimBA H5 dropped.**
The project's ROIs move from SimBA's `ROI_definitions.h5` (three pickled HDF5
tables: Python/pandas-version fragile, needs pytables, opaque, stores
derived values) to one JSON file. No legacy compatibility, by request: the H5
is neither read nor written.
* New `mufasa/roi_tools/roi_store.py`: `Roi` / `VideoRois` / `RoiStore`;
  per video `frame_size` + ROIs in drawing order; geometry + style only
  (rectangle x/y/w/h, circle center/radius, polygon vertices — any number —,
  `#rrggbb` colour, line width, tag size). `load_rois` (invalid file →
  `RoiFileError`, never "empty"), `save_rois`, `update_rois` (locked
  read-modify-write, atomic temp+rename), `rois_to_frames` (the three
  analysis tables with every derived column computed on read, cm sizes from
  the CURRENT video_info pixels/mm), `frames_to_store` / `write_frames`.
* `project_paths_from_config()["roi_definitions_path"]` → `rois.json` (v1
  and legacy layouts); section provenance and `Paths.ROI_DEFINITIONS` too.
* Readers: `read_write.read_roi_data` and `ConfigReader.read_roi_data` read
  rois.json, so every analysis / plotting / feature module keeps working
  unchanged (verified: ROIAggregateStatisticsAnalyzer on a 30-vertex polygon,
  a circle and a rectangle gives the right times).
* Writers: `ROILogic.save` (editor) is a locked update of its own video —
  two editors open at once can't drop each other's work; `multiply_ROIs`
  now rescales copied ROIs to each target's frame size; `reset_video_ROIs`,
  the duplicate dialog, CSV import (append keeps other videos) and the size
  standardizer all write through `write_frames`.
* Removed the 122ik SimBA-row normalisation (and its test) — moot without H5.
smoke_122il_roi_store: 21/21 (real project with two resolutions, two
editors at once, ConfigReader, multiply with rescale, reset, write_frames,
corrupt file, a real ROI aggregate analysis). Tripwires (save without
re-read, no rescale, bad file read as empty) each fail it. Path pins updated
in 122ai/122eh/122en/122ep/layout_paths_extension/roi_definitions_description/
roi_video_table_helpers; 122ek/122er now pin the store-based multiply/reset;
smoke_roi_logic loads roi_store and replaces its H5-specific cases; 122d5
module ceiling 431 → 432. Full sweep: no other change.

**122ik — saving ROIs no longer loses other videos' ROIs.**
`ROILogic.save` rewrites the whole ROI_definitions.h5. It read the existing
file through `read_roi_data` (drops every rectangle / circle row with a NaN),
swallowed read errors and then wrote a file holding only the current video,
and wrote its own rows in a private format: `Color BGR` / `Tags` / `vertices`
as strings, circles with extra `Center_X` / `Center_Y` columns. Mixed with
SimBA-format rows that left NaN cells, so the next read dropped rows —
measured: saving one circle on video A deleted video B's SimBA circle on the
first save. String vertices / tags also broke ROI analysis
(`vertices.astype`) and visualisation (`Tags.items()`).
* Save reads the file raw; an unreadable file aborts the save
  (`RuntimeError`) instead of being replaced.
* Every row — new and existing — is written SimBA-native (tuple colour, dict
  tags incl. the full rectangle tag set and circle 'Border tag', ndarray
  vertices) with SimBA's column set; missing standard values are filled;
  an extra column is kept only if no row lacks it. Old-Mufasa rows are
  repaired on the next save.
* Load reads raw as well and normalises each row, so mixed files show every
  ROI; an unreadable file sets `ROILogic.load_error` and the panel warns.
smoke_122ik_roi_save_preserves: 15/15 (real H5 files in SimBA and old-Mufasa
format, ROILogic load/save, read_roi_data, multiply_ROIs, corrupt file).
Tripwires (swallowed read error, extra NaN columns, string vertices) each
fail it. smoke_roi_logic: added its missing `import os`; its cross-video save
case (19), which had only been skipping for lack of pytables, fails on the
old code and passes now (20/20).

**122ij — ROI editor redesign; polygons have no vertex limit.**
* Bug: `ROICanvas._commit_polygon` ran every polygon through
  `shapely.simplify(tolerance=20)`, silently deleting each vertex within 20
  frame px of the line through its neighbours. On a 640×360 frame that capped
  polygons at ~4–8 vertices however many were clicked (measured: a 30- or
  60-vertex ring of radius 80 px → 8). Vertices are now kept as placed; only
  exact consecutive duplicates are dropped.
* Logic: Esc now abandons a polygon in progress (it used to commit it with
  ≥3 vertices). Finish with a click on the first vertex, double-click or
  Enter; Backspace / right-click removes the last vertex.
* Workflow: no more "type a name → pick a tool → press Draw → draw one shape".
  Tools (Select V / Rectangle R / Circle C / Polygon P) are one exclusive group
  and drawing tools are sticky; new shapes are named from an optional Name box
  or automatically (`polygon_1`…). The separate Edit toggle that disabled Draw
  is replaced by the Select tool.
* Editing: rectangles get edge-midpoint handles too; polygon vertices can be
  dragged, inserted (double-click an edge) and removed (right-click); arrow
  keys nudge (Shift: 10 px). Name (double-click), colour and line width are
  editable per row in the table (`ROILogic.update_style`). Duplicate / Delete
  buttons; undo / redo (Ctrl+Z / Ctrl+Shift+Z, `ROILogic.snapshot/restore`).
* View: wheel zoom about the cursor, middle-drag pan, F to fit; ROI names drawn
  on the frame (toggle); live size / radius / vertex-count readouts; a hint
  line says what the current tool expects.
* Layout: three panes — videos (with filter) | tools + canvas + frame nav |
  ROI table, new-shape style, copy-to-videos, import, clear; unsaved-changes
  indicator, Save (Ctrl+S). `,` / `.` step frames.
* "Apply to all…" now confirms and names how many other videos' existing ROIs
  `multiply_ROIs` will replace (it rewrites the whole H5). Import /
  standardize now reloads the current video so imported ROIs appear.
smoke_122ij_roi_editor: 32/32 — real project, generated videos, the widget
offscreen driven by QTest mouse/key events, saved H5 read back by a fresh
ROILogic. Tripwires (simplify restored, Esc commits, no undo, no apply-all
confirmation, non-sticky tool) each fail it. smoke_122dm updated: the Edit
toggle checks now pin the Select tool; move/resize bodies moved into
`_move_selected_to` / `_drag_handle_to`. Other ROI suites unchanged.
Note: smoke_roi_logic case 19 had been skipping for lack of pytables in the
sandbox; with pytables installed it fails (missing `import os`, then
`other_video_rois` empty in its stub harness) on the old roi_logic as well —
pre-existing, not touched here.

**122ii — the joint-prior fit uses the worker pool.**
`fit_joint_priors_v2` accepted `pool` and ignored it: its pass (a filter +
smoother per session, the cost of one EM E-step) ran every session serially in
the main process on one core, while the perspective pass just before it used
all workers. EM also called it with `verbose=False`, so it sat silent. The
per-session work is now `_joint_prior_session_samples`, shared by the serial
path and a new `_pool_joint_prior_pass` worker; results are accumulated in
session order, so the pooled fit is identical to the serial one. Progress
prints per session (`[v2-jprior] n/N … ETA`).
smoke_122ii_joint_prior_pool: 5/5 (real multiprocessing pool via
`_pool_init`; pooled == serial exactly; nothing runs in the main process).
Tripwire: forcing the serial path fails it. smoke_122hf's source check now
reads the helper. Every Kalman v2 suite passes.

**122ih — no blocking progress dialog.**
Running an operation opened a window-modal QProgressDialog that froze the whole
workbench — including the console dock showing that run's own output (e.g. the
Kalman v2 EM iterations), which couldn't be scrolled until the run ended.
`run_with_progress` now shows progress inline in the main window's status bar
(label, bar, Cancel; busy until the first progress report) and opens no dialog.
Parents outside a QMainWindow get a NON-modal dialog. Since nothing is modal,
`OperationForm._lock_run_button()` disables Run while the form's operation is
in flight (base `on_run`, Features, Video filters); the heatmap popup disables
its two run buttons the same way. New `on_finished` callback on
`run_with_progress`.
smoke_122ih_inline_progress: 18/18 (real OperationForm in a QMainWindow,
offscreen, blocking worker). Tripwires: modal fallback, no status-bar path,
no Run lock, indicator left behind — each fails. Suites touching the
workbench/progress/forms: no change vs the previous sweep.

**122ig — root orientation from the trunk; root-noise ceiling as a setting.**
* Bug: the root's initial orientation and the initial q_root_ori estimate read
  the legacy rig's `back1`/`back3` by name. Any other rig started at identity
  (up to 180° off the frame its root offsets are fitted in) with q_root_ori at
  the default 1.0 — and the M-step ceiling is 10× the initial value, so the
  fitted q_root_ori pinned at exactly 10 (seen in every strohA run). Both now
  come from the root cluster's own body axis (`_root_axis_per_frame`, the same
  widest-chord axis the root offsets use); back1/back3 only as fallback.
* All whole-body motion runs through the root's constant-velocity model, so on
  bursty movement EM pins q_root_pos / q_root_ori at the 10× ceiling within a
  few iterations. New `BodyLayout.root_q_ceiling_factor` (default 10 = prior
  behaviour), `--root-q-ceiling`, and "Root noise ceiling" in the Kalman v2
  form; absolute hard caps unchanged.
Measured on 6,000 frames of the user's strohA-xw session (drift on, back,head
orientation drift, head high-angular, joint prior, threshold 0.5, 15 EM
iterations): 10× -> median offset 2.36 px, worst marker 6.7 px; 50× -> 2.09 px,
6.7 px, q_root_pos at the 50,000 hard cap, no instability. EM still does not
converge (q_root_ori keeps rising) — a model mismatch with bursty motion, not
fixed here.
smoke_122ig_root_axis_and_ceiling: 19/19 (real functions, real median M-step,
form offscreen). Tripwires: identity start, default q_root_ori, ceiling
ignoring the layout, hard cap removed, GUI default changed — each fails.
Canonical rig 152/152.
smoke_122hf_joint_prior: the near-uniform-tail check compared the full prior
set on vs off and passed at <2° only because both runs sat at chance (~105°).
With q_root_ori now data-estimated (~9 vs 1.0 in that rig) the ACTIVE
back_rear/neck priors steer the unobservable 20 s tail_3 walk onto another
chance-level path (57° vs 114°; hold-last scores 104°, circular mean 109°). The
check now tests what it meant — tail_3's own inert prior alone changes nothing
(bit-identical) — plus a sanity bound (<130°). 43/43.

**122if — remove sessions via the Data Import table.**
The "Already imported" table (Data Import -> Import video) gains a "Pose data"
column (the imported pose file(s) with the video's stem), multi-row selection,
and *Remove selected…* — button, Delete key, or right-click. Confirmation lists
the sessions, with "Also remove their imported pose data" (ticked when there is
any). New Qt-free `mufasa/model/session_removal.py` (`session_files`,
`remove_sessions`): nothing is deleted — the video (a symlink as the link; the
footage is never touched), the pose file(s) if chosen, and the session's
`video_info.csv` row go to `backups/removed_sessions-<run_id>/` (rename on the
same disk, so instant for large videos) with a full `video_info.csv` copy and
a `manifest.toml` of from -> to. Derived outputs are left alone.
smoke_122if_remove_sessions: 21/21 (real backend on a ProjectConfigCreator
project; the real form offscreen, message boxes stubbed). Tripwires: delete
instead of move; following the symlink; video_info not cleaned; pose box
ignored; single selection; placeholder row removable — each fails.
mufasa/**/*.py 430 -> 431 (smoke_122d5 ceiling bumped).

**122ie — FreeDLC parquet without `.fdlc` naming; pose-file autodetect says parquet.**
FreeDLC (the modified DeepLabCut) now writes `pose.parquet` per run + a run
`.toml`, skeleton embedded — not `<video>.fdlc.parquet` + `.fdlc.toml`.
Nothing requires the old naming any more (it is still read, as legacy):
* one parquet import route ("FreeDLC parquet (single animal)"; 122ib's
  separate "DLC parquet" entry folded back), hint describes run folders;
* Add-sessions file filter and ingest suffix list drop `*.fdlc.parquet`
  (any long parquet is accepted by schema); importer/preview docs and the
  "nothing found" message describe the run-folder layout.
* Create-project "Auto-detect from pose file…" (renamed from "…DLC file"):
  dialog lists `Pose files (*.parquet *.h5 *.csv)` with parquet first; the
  skeleton filter is the shared `pose_parquet.skeleton_edges_for`.
* File -> Reconfigure: its text and placeholder still said ".h5 or .csv";
  now parquet too, filter `Pose files (*.parquet *.h5 *.csv)`. (Its apply step
  is the legacy INI-only `project_reconfigure` — unchanged, pre-existing.)
* Edit-metadata autodetect error text mentions parquet.
smoke_122ie_freedlc_no_fdlc_names: 12/12, driving the real create-form and
reconfigure buttons (dialogs stubbed, Qt offscreen) with a run's pose.parquet,
with and without embedded metadata. smoke_122dj route count back to 12.

**122id — pair a run's pose.parquet with its video through the run sidecar.**
The DeepLabCut exporter writes a fixed-name `pose.parquet` per analysis run
with a run TOML beside it (named after the video): `kind = "analyze"`,
`inputs = ["…/<video>.mp4"]`, `outputs = ["pose.parquet"]`, model/snapshot
provenance. No skeleton in it. `pose_parquet.read_run_sidecar` finds the
record whose `outputs` names the file (other TOMLs ignored);
`pose_parquet_video_name` returns the recorded video's stem first.
* Discovery descends one level into run folders (only where a run sidecar
  names a long parquet) — importer and Add-sessions alike.
* Importer names each import after the run's video and refuses two files that
  map to one video (they would overwrite each other).
* Smoother names output `<video>_smoothed_v2.<hash>.parquet` for run files
  (`_session_output_stem`, shared by writer and incremental skip); other
  files keep their names.
* mufasa-preview pairs the run with the recorded video even outside the folder.
smoke_122id_run_sidecar_pairing: 17/17 (real importer, writer, preview).
Tripwires: name ignores sidecar, no run-folder descent, no clash check,
smoother stem from file name, preview ignores sidecar — each fails. Real data:
the user's run TOML beside a slice as `pose.parquet` -> named
`K7_cam_20251124_1510_A10-0-051_640x360-triplet`, found from the parent folder,
smoothed standalone to `K7_cam_…_640x360-triplet_smoothed_v2.<hash>.parquet`.

**122ic — processing and viewing use the data's own skeleton.**
`pose_parquet.resolve_skeleton(paths, config=None)`: explicit project.toml, then
the pose files' own skeleton (embedded / `.fdlc.toml`; a folder is probed
through its long parquets), then the project enclosing the files.
* Smoother: `layout_from_fdlc_sidecar` reads the embedded skeleton too (and a
  folder of agreeing files; nodes taken from the edges if `bodyparts` is
  absent), so `--beside-input` on a standalone DeepLabCut parquet builds its
  tree with no project. Imports the light module, not the importer.
* Smoothed output keeps the input's skeleton (`b"pose"` metadata, edges
  between its markers), so a standalone smoothed file stays self-describing.
* Overlay viewer: draws the resolved skeleton (new `--config`) instead of
  `DEFAULT_SKELETON_EDGES`, the legacy rat-rig names, which drew nothing for
  any other rig. pose_viewer auto-resolves when `--config` isn't given.
* Fix: pose_viewer's direct-read loader blanket-lower-cased columns (the
  122hb bug, flagged in 122hx), so every smoothed file's back_T4 became
  back_t4 — smoothed markers stopped matching the raw file and the skeleton
  (10 of 24 edges drawn on the real rig). Now suffix-only normalisation.
smoke_122ic_skeleton_downstream: 20/20 (real writer, real overlay main()
offscreen). Tripwires: layout ignores embedded, writer drops skeleton, loader
lower-cases, overlay ignores resolution, no project walk-up — each fails.
Real data: a 1,500-frame slice of the user's export smoothed standalone
(`--beside-input`, no project) — layout "from the pose files' own skeleton
(15 markers)", output carries 24 edges; overlay draws 24/24 raw and 24/24
smoothed lines (10/24 smoothed before the case fix).

**122ib — DeepLabCut long-format parquet with an embedded skeleton.**
The modified DeepLabCut exporter writes the same long table as FreeDLC
(frame | individual | bodypart | x | y | likelihood) but names it
`<video>.parquet` and carries the skeleton inside the file: parquet key/value
metadata `b"pose"` = JSON `{"version", "bodyparts", "skeleton"}` (DLC's own
config.yaml keys; edges as name or index pairs). New dependency-light module
`mufasa/pose_importers/pose_parquet.py` is the shared layer: schema-based
detection (`is_long_pose_parquet`, non-recursive `find_long_pose_parquets`),
`read_pose_skeleton` (embedded first, then the FreeDLC `.fdlc.toml` sidecar),
`pose_parquet_bodyparts`, `embed_pose_metadata` + a CLI
(`python -m mufasa.pose_importers.pose_parquet FILE... --skeleton-from
project.toml|config.yaml|x.fdlc.toml|x.parquet`) to stamp existing files.
`read_fdlc_skeleton` moved here (re-exported by the importer) so the smoother
and viewers no longer drag in the importer's h5py chain.

* Import: FDLCParquetImporter finds long parquets by schema, so `<video>.parquet`
  is picked up beside `*.fdlc.parquet` while wide/smoothed/unrelated parquets
  are ignored; the video stem works for both names; the embedded skeleton is
  written to project.toml [skeleton]; new `file_paths=` imports exactly the
  given files. "DLC parquet (single animal)" added to the pose-import list.
* Project creation: "Auto-detect from DLC file" accepts .parquet
  (`extract_bodyparts_from_parquet`), keeps the file's skeleton (edges between
  its own markers) and writes it to the new project's [skeleton].
* Add sessions: long parquet routed by format, not by the `.fdlc.parquet`
  suffix — a `<video>.parquet` used to be copied in raw as if already wide.
  The importer now receives the accepted files only, so a rejected file in the
  same folder is no longer imported with them (that aborted the whole ingest).

smoke_122ib_dlc_parquet: 37/37, run for real (pyarrow, the importer, the create
form offscreen, session ingest). Tripwires: suffix-only discovery, embedded
skeleton ignored, suffix-based ingest routing, whole-folder ingest import,
form not writing the skeleton, parquet missing from autodetect — each fails the
suite. Real data: the user's 108,030-frame × 15-marker export, with the
project's 24-edge skeleton embedded (0.3 s, data byte-identical), imports to a
108,030 × 45 wide file and a 24-edge project [skeleton]. Test harnesses of
smoke_122gm / smoke_122gn now take `_LONG_COLUMNS` / `read_fdlc_skeleton` from
the new module. mufasa/**/*.py 429 -> 430 (smoke_122d5 ceiling bumped).

**122ia — scale pose coordinates for reduced-resolution video.**
Pose estimated on a downscaled video (e.g. *_reduced_negate.mp4) is in
reduced-resolution coordinates; overlaid on the full-res video the markers
cluster in a corner. A --pose-scale FACTOR multiplies pose coordinates by the
factor — 2 for a half-resolution pose, 4 for quarter — so they line up. The
factor is also a live Scale selector in the viewer (presets 1×/2×/4×, editable
for any value), so which factor aligns can be eyeballed without a restart.

Scaling applies to marker positions, skeleton endpoints, and the
variance-ellipse *centre and radii* — all in pose-pixel space, so a coordinate
rescale scales the ellipse's std-dev linearly too. It does NOT scale the marker
dot size, which is a fixed on-screen size (the dot's own geometry, set once;
only setPos is scaled). Threaded through OverlayScene (pose_scale + set_pose_scale),
the overlay CLI, and mufasa-preview (--pose-scale passthrough, omitted when 1).

smoke_122ia_pose_scale: 23/23 — the scaling arithmetic (positions ×s, endpoints
×s, ellipse centre+radii ×s, s=1 a no-op) and the scale-label helper run in
isolation; AST confirms the dots/skeleton/ellipse scale by s while the dot
radius definitions carry no ×s, the set_pose_scale method, the CLI arg, the live
Scale selector (with a non-numeric entry reverting rather than crashing), and
the mufasa-preview passthrough. Tripwires: unscale the dot positions -> 22/23;
scale the dot radius -> 22/23 (the check pins the radius lines); unscale the
ellipse radii -> 22/23; drop the preview passthrough -> 22/23. 122hy 26/26 and
122hz 23/23 unchanged. Both sweeps vs 5671eed: 210 smoke_122 + 73 others, no
regressions. ruff 1 (overlay, pre-existing) / 0 (preview), F821 = 7, I001 = 1,
mufasa/**/*.py = 429.

**122hz — record button in the overlay viewer.**
Adds a Record button to pose_video_overlay: first press starts, second press
stops, and the saved clip contains the composited overlay frames (video + pose
markers, exactly as shown) for every frame displayed in between — whether
advanced by play, the scrubber, or stepping.

Because the overlay is Qt-composited (markers are QGraphicsItems over the video
pixmap, not painted into the frame pixels), the clip is captured by rendering
the QGraphicsScene to a QImage and converting to BGR for cv2.VideoWriter — not
from the raw cv2 frame. Capture is hooked in _set_frame, the single path every
displayed frame passes through, so any on-screen frame during recording is in
the clip. A _ClipRecorder (Qt-free, takes plain BGR arrays) owns the writer:
created lazily on the first frame so it sizes to the actual render, dimensions
forced even for codec compatibility, later frames conformed to the locked size.
The clip is saved beside the video by default (a Save-As dialog offers a
different path); an in-progress recording is finalised on window close so the
file isn't truncated. Degrades gracefully (status message) when cv2 is absent.

smoke_122hz_overlay_record: 23/23 — the _ClipRecorder state machine (lazy
writer, frame count, finish message, empty-recording failure, even-dimension
lock, size conforming) and the QImage->BGR reshape (row-padding stripped,
channels reversed, C-contiguous) run directly against a stubbed cv2 / simulated
buffer; AST confirms the button, the toggle_record start/stop, the capture hook
inside _set_frame after update_frame, the scene render, the close-finalisation,
and the no-cv2 fallback. Tripwires: drop the capture hook -> 21/23; break the
empty-recording check -> 22/23; break the even-dimension rounding -> 21/23.
Both sweeps vs 5671eed: 209 smoke_122 + 73 others, no regressions. ruff 1
(overlay — one better than the pre-existing baseline of 2, a nested-if collapse
cleared it), F821 = 7, I001 = 1, mufasa/**/*.py = 429 (edits only, no new
module). GUI feature; the Qt-free logic is tested directly, the rest AST-checked
(PySide6/cv2 absent in the sandbox).

**122hy — mufasa-preview: open the overlay viewer by paths or folder search.**
Adds a mufasa-preview console command (new module mufasa/tools/pose_preview.py,
entry point in pyproject.toml) that pairs a video with its pose files and opens
pose_video_overlay — so the long, hash-tagged pose paths don't have to be typed
by hand. Two modes: explicit (mufasa-preview --video rec.mp4 [--smoothed ...]
[--raw ...]; with only --video, pose files beside it are auto-found) and folder
search (mufasa-preview --folder .; videos paired with their pose files by name).

Pairing reduces a video and its pose files to a common session stem, so rec.mp4
matches rec.fdlc.parquet (raw) and rec.fdlc_smoothed_v2.<hash>.parquet
(smoothed) — the parameter-hash tag (122hm/hn) and the .fdlc/_smoothed_v2
suffixes are stripped; among several smoothed hashes the newest wins. The viewer
runs in-process via pose_video_overlay.main (same package, no subprocess);
--print shows the equivalent command instead of opening it; --all previews every
match in a multi-video folder (ambiguity otherwise errors with the match list).

smoke_122hy_mufasa_preview: 26/26 — stem reduction (incl. the reported real
filename), folder discovery (pairing, no-pose skipped, newest-hash wins), argv
construction (present/omitted opts), both --print modes, the error/ambiguity
paths (ambiguous folder, no video, missing pose, missing video, --video/--folder
mutual exclusion), the in-process overlay invocation with the built argv, and
the registered entry point. Tripwires: break the hash stripping (smoothed won't
pair); disable the ambiguity guard -> 25/26; skip the overlay call -> 24/26.
Backend logic, tested directly (real temp folders + a stubbed overlay main).

Adds a source module, so mufasa/**/*.py 428 -> 429; the smoke_122d5_stage_b
py-count ceiling is bumped 428 -> 429 to match (26/26). Both sweeps vs 5671eed:
208 smoke_122 + 73 others, no regressions. F821 = 7, I001 = 1, ruff 0 (module).

**122hx — shared pose loader handles all three FreeDLC formats (dedup).**
Fixes pose_video_overlay / pose_viewer._load_pose_file failing on a raw FreeDLC
long-format parquet — the same shape 122hw fixed, but in a *separate* loader
copy (the tools loader, distinct from the smoother's). Rather than copy the fix
a third time, the two non-flat FreeDLC parquet shapes are now handled in the one
place both loaders already fall back to: kalman_diagnostic.load_pose_file. Its
parquet branch now flattens a 3-level MultiIndex header (imported form, 122ht)
and pivots the long/tidy schema (raw form, 122hw) before marker detection.
pose_viewer inherits the fix by falling through to it; the smoother keeps its
inline copy for now.

The canonical _pivot_fdlc_long_to_wide lives here now (kalman_diagnostic is the
dependency-light shared module); the smoother's 122hw copy is an equivalent that
a later pass can collapse onto this one. That fuller consolidation — the
smoother and pose_viewer both delegating entirely to this loader instead of
carrying near-identical direct-read blocks (the smoother's own code comments
already note it duplicates pose_viewer._load_pose_file) — is a reasonable
follow-up; this patch fixes the immediate failure and removes the *format*
duplication without touching the smoother's hot path.

smoke_122hx_shared_loader_formats: 12/12 — the shared loader reads all three
shapes (raw long, imported MultiIndex, wide flat) with case preserved
(back_T4); pose_viewer's direct read finds no markers on a long file (so it
falls back) and the shared loader then loads it; and the pivot helper's
unit behaviour (sentinel clip, multi-animal rejection). Tripwires: drop the long
handling / drop the MultiIndex flatten -> the loader raises on that shape.
Canonical rig 152/152; 122ht 6/6, 122hw 14/14, 122hb 29/29 unchanged. Both
sweeps vs 5671eed: 207 smoke_122 + 73 others, no regressions (the two
pre-existing diagnostic-suite import failures are unchanged). ruff 1 (diagnostic,
baseline), F821 = 7, I001 = 1, mufasa/**/*.py = 428.

**122hw — smoother reads a raw FreeDLC long-format parquet.**
Fixes "Could not load .../K7_..._reduced_negate.fdlc.parquet. Tried direct read
and DLC multi-row header parsing." when smoothing a *raw* FreeDLC file
standalone (--load-model --beside-input). FreeDLC's native .fdlc.parquet is
LONG/tidy — (frame, individual, bodypart, x, y, likelihood), one row per
bodypart per frame — not the wide <bp>_x/_y/_p the loader expects. 122ht handled
the *imported* wide MultiIndex form; a raw file that never went through the
importer is long, has no _x columns, and fell through to the DLC parser and
failed.

Adds _pivot_fdlc_long_to_wide (mirrors the importer's long_to_wide but
layout-independent: every bodypart -> <bp>_x/_y/_p, no project order imposed,
-1 no-detection sentinel clipped to 0, frame index made contiguous, multi-animal
rejected with an actionable message). The loader detects the long schema
(non-MultiIndex columns containing frame/bodypart/x/y) and pivots before the
marker check; wide and imported-MultiIndex inputs are untouched.

smoke_122hw_fdlc_long_format: 14/14 (pivot recovers all bodyparts with case
preserved, emits _x/_y/_p, preserves or reindexes the frame count, clips the -1
sentinel, rejects multi-animal with the explicit message; AST confirms the long
detection is gated to non-MultiIndex and precedes the marker check). Tripwires:
remove the long detection -> 9/14; disable multi-animal rejection -> 13/14 (the
strengthened check pins the explicit message, not pandas' incidental reshape
error); break the sentinel clip -> 12/14. Canonical rig 152/152; 122ht 6/6,
122hu 18/18, 122gm 8/8 unchanged. Both sweeps vs 5671eed: 206 smoke_122 + 73
others, no regressions. F821 = 7, I001 = 1, ruff 22 (kalman), mufasa/**/*.py =
428.

**122hv — --load-model needs no project, --config, or skeleton.**
Answers "do I still need the inputs / a project to smooth an external file once
the model is built?" — no. A saved v2 model already stores its own layout
(save_model_v2 serializes every segment's name/parent/markers; load_model_v2
rebuilds the BodyLayout), and smooth_pose_v2 uses the loaded layout and ignores
any passed one. So with --load-model the only things still needed are the input
files and where to write.

main() had been running layout resolution (project search -> FreeDLC sidecar ->
rig fallback) unconditionally, before smooth_pose_v2, even with --load-model —
wasted work whose result the model overrides, and for a standalone file it would
emit the rig-fallback warning needlessly and the multi-project error could even
block a legitimate load-model run. This guards the whole layout-resolution block
behind `if args.load_model is None:`. With a model, config and layout are left
None (smooth_pose_v2 supplies the model's own layout) and any layout-shaping
flags passed anyway (--config, --with-drift, --*-segments, --no-*) are reported
as ignored. Without a model, resolution runs exactly as before (project /
sidecar / rig), so the 122hu standalone-sidecar path is unchanged.

Net: `... --load-model model.npz some_file.parquet --beside-input` smooths an
external file with no project and no skeleton, output beside it. Nothing needs
to track the smoother's inputs — the model is the complete spec.

smoke_122hv_load_model_skips_layout: 12/12 (the model serializes/rebuilds the
layout and smooth_pose_v2 prefers it; the guard exists and its else-branch holds
all of find_project_config / sidecar / rig; find_project_config runs only under
the no-model branch; config/layout default to None; the ignored-flag reporting).
Tripwires: disable the guard -> 8/12; drop the `config = None` default -> 11/12.
Canonical rig 152/152 and smoke_122hu 18/18 both survive the 217-line
re-indentation. Both sweeps vs 5671eed: 205 smoke_122 + 73 others, no
regressions. F821 = 7, I001 = 1, ruff 22 (kalman), mufasa/**/*.py = 428.

**122hu — smooth a standalone pose file in place, outside a project.**
Two CLI additions let the smoother process a parquet that isn't in a Mufasa
project, with output beside the input:

* ``layout_from_fdlc_sidecar(pose_path, root=None, **flags)`` builds a
  BodyLayout from a FreeDLC ``<stem>.fdlc.toml`` skeleton sidecar
  (read_fdlc_skeleton -> segments_from_skeleton -> BodyLayout), so the
  kinematic tree travels with the FreeDLC export and no project.toml is needed.
  main() tries this before the built-in-rig fallback when no project is found
  above the input and no --config is given — but only when every input resolves
  to a sidecar and they agree on the marker set; otherwise it falls through to
  the existing rig-fallback warning.
* ``--beside-input`` writes each smoothed file into its input's own directory
  instead of --output-dir. Inputs are grouped by parent directory and the
  smoother is run once per group with that directory as the output; a directory
  argument resolves to itself. Without the flag it is a single group -> a
  single --output-dir call, exactly as before.

smoke_122hu_standalone_beside_input: 18/18 — layout_from_fdlc_sidecar (defined;
builds a full spanning-tree layout from a sidecar; flags pass through; resolves
from either the .parquet or .toml path; None when absent so main() can fall
back); the CLI wiring by AST including a parsed check that the beside-branch
really builds batches from the per-directory groups (not a collapsed single
output); and the batching algorithm across single-file/multi-folder/directory/
no-flag cases. Tripwires: sidecar helper -> None breaks the standalone layout;
beside-branch collapsed to one output dir -> 17/18; main() not trying the
sidecar -> 14/15 (pre-strengthening) / caught. Canonical rig 152/152. Both
sweeps vs 5671eed: 204 smoke_122 + 73 others, no regressions. F821 = 7, I001 =
1, ruff 22 (kalman), mufasa/**/*.py = 428.

Note: this is the smoother step in place; extending --beside-input to the later
pipeline stages (outlier, egocentric) is a follow-up. The sidecar read uses the
importer's read_fdlc_skeleton (works where h5py is installed — the user's
environment); the sandbox can't exercise that import, so the layout half is
verified with a stubbed reader mirroring the real TOML parse.

**122ht — smoother reads the imported MultiIndex pose parquet.**
Fixes "Could not load .../sources/pose/<session>.parquet. Tried direct read and
DLC multi-row header parsing." hit when adding sessions with smoothing. The
FreeDLC/SimBA importer writes sources/pose parquets with a 3-level MultiIndex
column header (scorer/bodypart/coords = IMPORTED_POSE/IMPORTED_POSE/<bp>_x), via
write_df's multi_idx_header path. The smoother's inline loader read the parquet
then looked for columns ending in "_x"; on a MultiIndex the columns come back as
tuples, so no markers matched, the load fell through to the DLC-CSV diagnostic
parser, and raised.

Fix: flatten a MultiIndex column header to its last level — which holds the flat
<bp>_x/_y/_p names — immediately after the direct read and before the marker
detection. Plain (already-flat) columns are untouched by the isinstance guard,
so smoothed-flat parquets and pre-flattened CSVs load exactly as before; a
header whose last level isn't <bp>_x (e.g. a 2-level DLC header) still falls
through to the diagnostic loader as before.

smoke_122ht_imported_multiindex_parquet: 6/6 — against a parquet written exactly
as the importer writes it: all markers recovered, case preserved (back_T4 not
back_t4), plain-column parquets still load, _var columns not mistaken for
markers, and AST confirms the flatten sits in the loader ahead of the marker
check. Tripwire: remove the flatten -> 4/6. Canonical rig 152/152; FDLC importer
and pose-import suites unchanged (122gm 8/8, 122gn 9/9, 122dh 27/27, 122di
12/12, 122dj 14/14). Both sweeps vs 5671eed: 203 smoke_122 + 73 others, no
regressions. F821 = 7, I001 = 1, ruff 22 (kalman), mufasa/**/*.py = 428.

**122hs — matplotlib 3.11 compat: replace removed cm.get_cmap.**
Fixes "module 'matplotlib.cm' has no attribute 'get_cmap'" hit when adding
sessions with smoothing on a recent matplotlib. cm.get_cmap was deprecated in
3.7 and removed in 3.11; the colour-palette helpers still called it. Replaced
every cm.get_cmap(name, N) with the version-safe
matplotlib.colormaps[name].resampled(N) across plotting_mixin.py,
utils/data.py (create_color_palette / create_color_palettes) and
utils/lookups.py, updating imports (colormaps added; cm dropped where no longer
used). In lookups this also repaired a latent bug — the old branch tested an
int index for membership in the colormap dict (never matched, always fell to
"spring"); the new code tests the actual colormap name, so a requested map is
used when it exists and falls back to spring otherwise.

Verified the modern call is numerically identical to the old API (same N, same
sampled colours for viridis/jet/spring/tab20) and that the fixed helpers run
even with cm.get_cmap deleted (simulating 3.11), reproducing the documented
create_color_palette('jet',3,hex) output.

smoke_122hs_get_cmap_removal: 12/12 (no cm.get_cmap( call remains in mufasa/;
the three files use resampled(); old/new equivalence where the old API is still
present; resampled() palette shape; the spring fallback). Tripwires: reintro
cm.get_cmap in data.py -> 10/12; drop the colormaps import in plotting_mixin ->
11/12. Both sweeps vs 5671eed: 202 smoke_122 + 73 others, no regressions.
ruff at baseline (40 plotting_mixin / 35 data / 12 lookups), F821 = 7, I001 = 1,
mufasa/**/*.py = 428.

**122hr — hotfix: restore _launch_synced_viewer swallowed by 122hp's closeEvent.**
Patch 122hp added a closeEvent method to MufasaWorkbench by inserting it before
_launch_synced_viewer, but the edit consumed that method's `def` line — so
_launch_synced_viewer's body was absorbed into closeEvent and the method
vanished. The workbench then crashed on construction: _build_menus wires
"Synced video viewer…" to self._launch_synced_viewer, which no longer existed
(AttributeError, window won't open at all).

Fix restores the `def _launch_synced_viewer(self) -> None:` line and its
docstring; closeEvent and _launch_synced_viewer are again separate methods.

Also hardens smoke_122hp_console_dock so this class of edit error can't recur
silently: it now parses MufasaWorkbench and asserts closeEvent and
_launch_synced_viewer are both real methods and that every menu handler
(_launch*/_on*/_open*/_switch*/_show*) resolves to an existing method — i.e. no
def was swallowed. 33/33 (was 29). Confirmed the guard catches the bug:
reintroducing the swallowed def drops the suite to 31/33 on exactly the two new
checks. F821 = 7, I001 = 1, ruff 0 (workbench), mufasa/**/*.py = 428.

**122hq — egocentric alignment: strip smoother suffixes before the video lookup.**
Fixes "NO FILES FOUND ERROR: could not find a video file representing
Cacna_..._smoothed_v2.<hash>" from egocentric alignment. The aligner finds each
pose file's source video by name, but derived the name straight from the pose
stem — which carries _smoothed_v2 and (since the parameter hash, 122hm/hn) a
trailing .<hash> that the source video filename does not. The _smoothed_v2
suffix alone would already have broken this; the hash made it visible.

Adds _video_base_from_pose_name(), which strips a trailing hex .<hash> (only
when what remains still ends in _smoothed_v2, so a legitimately dotted video
name is never truncated) and then the _smoothed_v2 suffix, returning anything
unrecognised unchanged. Used only for the three find_video_of_file lookups (the
pre-flight check and the two in run()); the output data/video keep the full
pose-derived video_name, so a rotated output still traces to the specific
smoothed input (important now that different parameter hashes are distinct
inputs).

smoke_122hq_egocentric_video_base: 16/16 (the reported 16-char-hash case, the
8-char and no-hash cases, plain stems unchanged, and the false-positive guards:
a dotted video name without the marker, a hex extension without _smoothed_v2, a
non-hex suffix, and a hash-like token are all left alone; idempotent). Also
verifies the wiring — lookups use the stripped base, outputs keep video_name,
and no lookup still passes the raw name. Tripwires: don't strip _smoothed_v2 ->
12/16; drop the hex-guard so it over-strips -> 15/16; use the raw video_name in
the run() lookups -> 14/16. Existing egocentric suites pass:
smoke_122ex_egocentric_section_id 10/10, smoke_122ez_egocentric_detect_path
10/10. Both sweeps vs 5671eed: 201 smoke_122 + 73 others, no regressions.
F821 = 7, I001 = 1, ruff 2 (aligner, baseline), mufasa/**/*.py = 428.

**122hp — dockable console for verbose output.**
Adds a read-only console docked at the bottom of the workbench that mirrors
stdout/stderr live, so GUI users see the verbose progress (the smoother's
[smoother-v2] lines, per-EM-iteration logs, provenance notes) that previously
only reached the terminal.

New module mufasa/ui_qt/console_dock.py: a _StreamRedirector that stands in for
sys.stdout/stderr, tees every write to the original stream (terminal users lose
nothing) and emits it on a Qt signal; a ConsoleDockWidget (bounded, read-only,
with Clear / Copy / Autoscroll / Wrap); and attach/detach helpers. Because
operations print from a ProcessorRunner worker thread and Qt widgets are
GUI-thread-only, the signal->slot uses a queued connection so the append lands
on the GUI thread — this is the thread-safety crux. The redirector implements
enough of the file protocol (write/flush/isatty/writable/fileno) to be a safe
stand-in, and a broken original stream can't lose console output or crash.

Wired into MufasaWorkbench: attach_console_dock in __init__ (guarded so a
console failure can't stop the window opening), detach in closeEvent to restore
the streams on teardown. Project switches rebuild the workbench; attach unwraps
any existing mufasa redirector (_base_stream) so it tees to the real terminal
rather than chaining redirector->redirector, and detach only restores when
still the active stream, so the rebuild doesn't duplicate output or clobber the
new window.

smoke_122hp_console_dock: 29/29. AST-checks the module structure and workbench
wiring (queued connection, bottom dock, idempotent attach, read-only + bounded
view, attach/detach), and exercises the pure logic in isolation (a stub
signal): tee to both streams, empty write emits nothing, write returns length,
_base_stream unwraps a chain to the real stream and handles None, and a broken
original neither loses output nor crashes. Tripwires: break the tee -> 28/29;
drop the queued connection -> 28/29; break the unwrap -> 28/29; remove the
workbench attach -> 28/29. GUI-only, AST/pure-logic verified (PySide6 absent in
the sandbox); the Qt widget runtime is validated on hardware.

Note: this ADDS a source module, so mufasa/**/*.py goes 427 -> 428. The
smoke_122d5_stage_b py-count ceiling is bumped 427 -> 428 to match (it is an
explicit loose ceiling that already moved 425 -> 427 in 122gz). Sweeps vs
5671eed: 200 smoke_122 + 73 others = no regressions after the ceiling bump.
F821 = 7, I001 = 1, ruff 0 (new module) / 0 (workbench).

**122ho — project-aware segment fields in the Kalman v2 GUI.**
Fixes the confusing failure where the form's orientation-drift / const-accel /
high-angular-noise fields carried a stale 'body,head' example (and persisted
value), and submitting it died deep in the smoother with a raw dataclasses
ValueError stack trace: "orientation_drift_segments references unknown segment
'body'". The rig was renamed, so this project's segments are back / back_rear /
head / neck / tail_1.. — there is no 'body' (it became 'back').

Three parts: (1) a _project_segment_names(config_path) helper resolves the
project's actual segment names via layout_from_config, degrading to [] on any
failure so the form never crashes when no project is loaded; (2) the form
validates the three segment fields against those real names BEFORE
dataclasses.replace hits the layout's own validator, raising a clear message
that lists the available segments and notes the body->back rename — the
smoother's __post_init__ stays as the backstop; (3) the stale 'body,head'
examples are replaced with project-aware placeholders (the real segment names)
and tooltips, and the stale docstrings are corrected.

smoke_122ho_segment_validation: 16/16 (helper definition + safe runtime
degradation on bad/empty paths; validation present and ahead of _dc.replace;
correct accept/reject behaviour incl. the exact 'body' case, a typo, and empty
lists; no 'body,head' example remains; placeholders are project-aware).
Tripwires: remove the validation -> 13/16; hardcode the placeholders -> 15/16;
narrow the helper's except so it no longer degrades -> the runtime check
raises. GUI suites unchanged: smoke_pose_cleanup_v2_wiring 2/2,
smoke_122hk_gui_and_model_io 21/21, smoke_122hd_segment_flag_errors 27/27. Both
sweeps vs 5671eed: 271 suites, no regressions. F821 = 7, I001 = 1, ruff 12
(form), mufasa/**/*.py = 427.

**122hn — longer parameter hash.**
Widened the smoother parameter hash (122hm) from a 4-byte digest (8 hex chars)
to an 8-byte digest (16 hex chars), for stronger collision resistance across
parameter variants while still keeping the output filename readable. One-line
change (blake2b digest_size 4 -> 8); everything downstream is length-agnostic
(the writer and the incremental skip both build "." + param_hash). The 122hm
test's length assertion is updated 8 -> 16 accordingly.

smoke_122hn_hash_length: 8/8 (asserts the 16-char length, determinism,
parameter-sensitivity at the new length, filename propagation, and digest_size
== 8). Tripwire: revert digest_size to 4 -> 122hn 3/8 and 122hm 21/22.
smoke_122hm 22/22 with the updated assertion, 122hl 20/20, 122hk 21/21,
canonical 152/152. smoke_122 sweep vs 5671eed: 198 suites, no regressions.
(The 73 non-smoke_122 suites don't touch the hash helper or the output
filename, so a digest-width change can't affect them.) F821 = 7, I001 = 1,
ruff 22, mufasa/**/*.py = 427.

**122hm — parameter hash in the smoothed-output filename.**
Smoothed output is now written as ``<stem>_smoothed_v2.<hash>.parquet`` (and
the ``.csv`` fallback likewise), where ``<hash>`` is a short blake2b digest (8
hex chars) of the smoother parameters. Two effects: parameter variants coexist
in one output dir instead of clobbering each other, and the incremental skip
(122hl) becomes parameter-aware — a changed setting yields a different
filename, so the skip re-smooths rather than leaving the stale file (the exact
gap flagged when 122hl landed).

The digest is built from the effective smoothing configuration as it exists at
write time: the fitted noise params (all numeric NoiseParamsV2 fields), the
layout flags and tree, the joint prior (mu/kappa) and perspective presence, and
the smoothing-time knobs (likelihood_threshold, fps, apply_constraints).
Reading from the effective objects makes it mode-independent — a model trained
here and the same model reloaded later produce the same digest (their params
came off the same fit), so a file produced in a training run and one produced
in a later load run of that model share a digest and the skip treats them as
the same. Floats are rounded to 6 significant figures before hashing, so
meaningless last-bit differences don't perturb the digest while a genuinely
different fit (different data, EM settings, or config, all of which move the
fitted params) does.

Computed once per run (the config is identical across sessions), threaded into
both the incremental skip's existence check and both writer call sites
(parallel and serial). Legacy behaviour is preserved when no hash is supplied
(in-memory callers with output_dir=None get the un-suffixed name). Blast radius
checked: nothing globs or reconstructs the ``_smoothed_v2.parquet`` name;
consumers use the returned output_path.

smoke_122hm_param_hash_filename: 22/22. Tripwires flip: drop
likelihood_threshold from the payload -> 21/22; make the writer ignore
param_hash -> 20/22; remove the float rounding -> 21/22. 122hl still 20/20 (its
skip is now hash-aware), canonical 152/152. Both sweeps baseline-diffed against
5671eed: 270 suites, no regressions. F821 = 7, I001 = 1, ruff 22 (kalman),
mufasa/**/*.py = 427.

**122hl — incremental smoothing / overwrite option.**
smooth_pose_v2 gains an ``overwrite`` parameter (default True — unchanged
behaviour, re-smooth every input). With overwrite=False, any session whose
smoothed output already exists (``<stem>_smoothed_v2.parquet`` or the ``.csv``
fallback in the output dir) is skipped, so a re-run only processes files that
have no output yet — for resuming an interrupted batch or adding new sessions
without recomputing the finished ones.

The skip is computed once, before the smoothing dispatch, and applied in both
the parallel path (filtering task_args) and the serial path (continue in the
loop), so skipped sessions never enter the per-session smoother — it saves
compute, not just the write. The skip is gated on output_dir being set and
overwrite being False; with no output_dir (in-memory use) nothing is skipped.
skip_session indexes raw_sessions / sessions_arr, which are built in lockstep,
so the index is stable across both paths.

CLI: a mutually-exclusive ``--skip-existing`` (overwrite off) / ``--overwrite``
(the default), passed through main. GUI: an "Overwrite existing smoothed
output" checkbox in the Kalman v2 form's common options, threaded through both
collect_args returns into the three output-writing smooth_pose_v2 calls
(two-pass pass-2, single-pass, load); pass-1 writes no output so it is left
alone.

smoke_122hl_incremental_overwrite: 20/20. Tripwires flip: remove the parallel
task_args filter -> 19/20; drop the "not overwrite" gate (skip even when
overwriting) -> 19/20; drop overwrite from a GUI output-writing call -> 19/20.
Related suites pass: smoke_122hk 21/21, smoke_pose_cleanup_v2_wiring 2/2,
canonical 152/152. Both sweeps baseline-diffed against 5671eed: 269 suites, no
regressions. F821 = 7, I001 = 1, ruff 22 (kalman) / 12 (form),
mufasa/**/*.py = 427.

**122hk — GUI wiring for the new smoother options + model save/load fix.**
Exposes the three smoother options added this session in the Kalman v2 form,
and fixes a silent correctness bug in model persistence.

The save/load bug (the important half). Training a model with the joint prior
(122hf), saving it, and loading it to smooth other sessions silently DROPPED
the prior: the save_model_v2 call in smooth_pose_v2 passed perspective but not
joint_prior, so save_model_v2 took its default None and wrote
has_joint_prior=False. A loaded model then smoothed new sessions WITHOUT the
prior — reintroducing the covariance divergence (tail tip and rear back
markers flying off to thousands of px) that the prior is there to prevent. The
save/load path is otherwise correct: load_model_v2 already reads the prior back
and returns it, and the final smoothing pool already threads it to the filter;
only the save call omitted it. One line: pass joint_prior=joint_prior. Verified
by a round-trip (kappa/mu/n_obs preserved) and by reproducing the old loss.
tau_accel (122hg) already round-tripped; high_angular_noise_segments (122hj)
round-trips via the layout save/load added in that patch.

GUI wiring. KalmanV2SmoothingForm exposed with-drift / orientation-drift /
const-accel but not the joint prior, accel-tau, or high-angular-noise
segments. Added: a joint-prior checkbox, an accel-tau spinbox (0 shows
"off (random walk)" and maps to None, not an invalid 0.0), and a
high-angular-noise-segments field. Threaded through collect_args into both
train smooth_pose_v2 calls (single-pass and two-pass pass-1) and, for
high-angular-noise, into the layout replacements so it is saved with the model
and restored on load like the other segment flags. Load mode is unchanged —
those settings come from the saved model, which now includes them.

smoke_122hk_gui_and_model_io: 21/21. Tripwires flip: drop joint_prior from the
save call -> 20/21; drop the enable_joint_prior collect_args key -> 20/21; drop
accel_tau from a train smooth call -> 20/21. Directly-affected suites pass:
smoke_pose_cleanup_v2_wiring 2/2, smoke_122hf_joint_prior 42/42,
smoke_save_destination_validation 8/8 (smoke_model_dual_save's 22/23 is a
pre-existing legacy-INI failure, unrelated). Canonical rig 152/152. Both sweeps
baseline-diffed against 5671eed: 268 suites, no regressions. F821 = 7, I001 =
1, ruff 22 (kalman) / 12 (form), mufasa/**/*.py = 427.

**122hj — per-segment orientation-noise ceiling, for a fast head.**
The head impulse-tracking failure, root-caused from a real 67-session EM log.
The M-step clips each segment's fitted orientation process noise q_seg_ori to
q_initial * _M_STEP_Q_CEILING_FACTOR (10x). The log shows the head's q_seg_ori
pinned at exactly 20.0 (its init is 2.0, so 2.0*10) across every iteration
while the mean rises — a segment starved by its cap. The head turns ~3x faster
than the slowest segment (measured 6.2 vs 1.9 rad/s), so a single global
ceiling factor can't serve both: at the value that keeps the trunk stable, the
head is denied the process noise it needs, over-trusts its smooth prediction,
and rejects fast high-likelihood observations at acceleration impulses — the
head "not keeping up" with sharp movements, and exactly the user's report.

Confirmed the mechanism against the data: at high-acceleration impulses where
FreeDLC likelihood is high (p>=0.7, marker clearly visible), the smoothed head
still misses by ~9.5px, and the miss is not a lag (min error at time-shift -1)
and not damping (speed ratio ~1.0). The filter is rejecting good, visible, fast
observations because its dynamics prior is too tight.

The fix adds a layout field high_angular_noise_segments and a
--high-angular-noise-segments CLI flag (mirroring --const-accel-segments):
listed segments use a raised ceiling factor (_M_STEP_Q_CEILING_FACTOR_HIGH,
40x), all others stay at 10x. For the head that lifts the cap from 20 to 80.
Low-likelihood observations are still hard-gated out in the filter update
(markers below --likelihood-threshold are dropped, not down-weighted), so a
higher ceiling changes responsiveness only to observations that already pass
the gate — it does NOT make the head chase spurious detections (verified on a
real explosion frame: 11/15 markers below 0.5 there are correctly excluded).

Field-whitelist-rot guard: the model save/load path hand-lists layout fields,
so high_angular_noise_segments is persisted and restored symmetrically; pre-
122hj model files lack the key and load empty. Only q_seg_ori uses the per-
segment ceiling; q_length and the root ceilings keep the global factor.

smoke_122hj_per_segment_ceiling: 18/18. Tripwires flip: high factor == default
-> 16/18; selector ignores the flag -> 17/18; validation dropped -> 16/18.
Canonical rig unchanged (152/152, it sets no flag). Both sweeps baseline-diffed
against 5671eed: 267 suites, no regressions. F821 = 7, I001 = 1, ruff 22,
mufasa/**/*.py = 427.

**122hi — root-cluster frame from the widest chord, not PCA.**
122hh fixed the tail but not the trunk, and testing on the user's real 54k-frame
session showed why. 122hh derived the root cluster's body frame from per-frame
PCA and gated it on anisotropy, assuming a trunk is elongated like a spine.
Measured on real top-down mouse data it is not: the six dorsal markers project
into a roughly ROUND 2D patch — ~40px across but eigenvalue ratio ~1.75, no
dominant axis. PCA of a round patch returns a noise direction; the anisotropy
gate then rejected ~89% of frames; the cluster fell back to the placeholder
ring; the trunk kept bunching exactly as before.

A round patch still has a well-defined frame: the two markers farthest apart on
average define its widest chord, whose per-frame direction is stable even when
the cluster as a whole is round. 122hi uses that chord as the body axis — no
marker names, no elongation assumption, works for an elongated spine and a
round patch alike. The chord is inherently signed, so unlike a PCA eigenvector
there is no sign-flip ambiguity to repair. The anisotropy gate is removed. The
back1/back3 fallback stays for the canonical round cluster.

Verified on the real session: fitted trunk offsets match observed inter-marker
distances to within 1px (back_T4-back_L6 fitted 39.8 vs observed 38.6; the hip
pair 46.9 vs 47.4), and zero markers remain on the ring. The canonical rig is
unchanged (152/152). smoke_122hi_root_pair_axis (renamed from _pca_axis): 18/18,
including a controlled orthogonal-pair case that detects a wrong (closest-pair)
selection. Both sweeps baseline-diffed against 5671eed: 266 suites, no
regressions.

**122hh — root-cluster offsets from a PCA body axis; the bunched-back-markers fix.**
The reported bug: back markers bunched together even when FreeDLC saw them
clearly, and it started around 121. Traced to 122gx. Every non-root segment
gets its body frame from `parent_distal -> seg_distal`; the root has no parent,
so the code filled the gap with two hardcoded rig marker names (`back3 ->
back1`). 122gx generalized the tree to any skeleton but left that proxy
rig-specific — so `strohA-kj`, whose trunk markers are `back_T8..hip_right`,
matched nothing, `parent_distal` stayed None, and `fit_body_lengths` skipped
offset-fitting for the WHOLE root cluster. Every non-distal trunk marker kept
its placeholder ring offset `(1.0, angle)`: six markers pinned at unit radius
around a circle instead of along the spine. The forward model then placed them
on that ring no matter what the data said — hence "bunched even when plainly
visible." Not perspective, not convergence, not this session's patches; a
rig-name assumption left behind by a generalization three patches back, which
is exactly why it worked before 121.

Trunk markers lie along the body, so the first principal component of the
cluster's point cloud IS the body axis — for any rig, no names. Computed per
frame (the animal rotates). On strohA-kj this turns the ring into real
geometry: spine collinear (max |y| < 0.2px), lengths increasing down the back
(0 -> 8 -> 16px), hips straddling symmetrically (+-10px). PCA's sign is
arbitrary and can flip frame to frame, which would scatter each marker's angle
across two clusters 180 deg apart; a mean-axis fold fixes it. Only per-frame
consistency matters, not which end is "forward" — a globally tail-forward axis
just rotates the whole offset set 180 deg, which the filter absorbs into the
segment orientation. (An earlier attempt oriented toward a child marker; for
this tree that marker sits almost on the trunk line, making the sign test
degenerate — dropped.)

Gated on cluster anisotropy. PCA needs an elongated cluster; the canonical
rig's root markers are nearly coincident (eigenvalue ratio ~1.1, a round
blob), where the principal axis is just noise. Below a 4:1 ratio the axis is
refused and the code falls back to the old back3->back1 proxy, so the built-in
rig is bit-unchanged and 122hh strictly ADDS the elongated-cluster case. The
guard is load-bearing: disabling it reproduces the exact `r_drift` regression
(12.19 vs [0.8,6]) it was added to prevent.

**122hg — Singer acceleration: mean-reverting instead of a random walk.**
Answers "could we use the jerk if it was conditioned on velocity and/or
acceleration?" Conditioning Q on the state is mechanically fine — condition on
the *previous EM iteration's* smoothed state and Q_t is a known sequence, so
the filter stays linear-Gaussian. But conditioning on velocity is backwards
during a dropout: under const-accel with no data `v += a*dt` grows while `a`
holds, so q_jerk(|v|) rises as the extrapolation runs away. Positive feedback
on exactly the divergence we hit.

The Singer model gets both halves without state dependence. 121d's comment
says it plainly — "ax_new = ax (random walk)", the accel diagonal left at the
1.0 np.eye(D) put there. Setting it to phi = exp(-dt/tau_accel) makes
acceleration Ornstein-Uhlenbeck: still a tracked state, so fast motion is
captured exactly as before, but unobserved it relaxes to zero and the
extrapolation degrades CA -> CV -> CP.

Measured growth exponents: **t^5.01 -> t^3.17**. At a 1s dropout the two agree
within 1.5x (nothing given up where const-accel earns its keep); at 20s they
differ by 15x. `tau_accel=None` is the default and reproduces 121d bit-for-bit
— the OU noise `q*T/2*(1-e^(-2dt/T))` tends to `q*dt` as T -> inf, so this is
an identity, not an approximation. Fourth mean-reverting block in the file
after alpha_drift (120a), orientation drift (121a) and the joint prior (122hf).

**Two latent bugs surfaced doing it.**

`build_F_v2` takes a `params` argument to read `alpha_drift`. All four call
sites — forward_filter_v2, rts_smooth_v2, per_session_fit_from_stats,
finalize_m_step_v2 — omitted it, so it has *always* been None. `alpha_drift`
has therefore been inert since 120a: the block silently used build_F_v2's
hardcoded 0.05 fallback, which happens to equal the default, which is why no
one noticed. `--accel-tau` was silently ignored the same way until this was
fixed. rts_smooth_v2 mattered most: the RTS gain uses F, so a backward pass
building a different F than the forward pass is wrong, not merely inert.

And the whitelist, for the third time. `finalize_m_step_v2` and
`finalize_m_step_v2_from_per_session` each hand-listed 15 of NoiseParamsV2's
16 fields. 122hc fixed the warm-start site and flagged these two for audit;
adding `tau_accel` is what called it in. Both now `dataclasses.replace` from
prev_params. `fit_initial_params_v2` stays a constructor — it has no prev to
copy from. The next field cannot be forgotten because there is nothing left
to forget.

**122hf — von Mises joint prior on segment angles.** Answers "for longer
stretches could we constrain the markers with the prior given the current
inter-marker relationships?" Yes, and the mechanism was already in the file.

Five of the seven segments in this tree carry exactly one marker. Two-plus
markers means losing one is free — the survivors pin the segment and FK places
the missing marker. One marker means no redundancy: when it drops, nothing
observes the segment's angle, and nothing constrains it either.
`build_F_v2` makes orientation a random walk; `cos²+sin²=1` fixes the radius
and says nothing about the angle. At the fitted `q_seg_ori` ≈ 6, the (cos,sin)
sd passes 1.4 after one second on a circle of radius 1 — the norm constraint is
just projecting a wild vector back at an arbitrary angle.

The prior: two pseudo-observation rows per segment on the same channel that
already carries the norm constraint, pulling `(cos_s, sin_s)` toward
`(cos mu_s, sin mu_s)` with variance `1/kappa_s`. Linear, so the Jacobian is
constant. The vector form rather than the tidier one-row `sin(theta-mu)=0`,
because that is satisfied at both `mu` and `mu+pi` and would settle the tail
folded back on itself. `mu`/`kappa` from a von Mises MLE fitted once before EM
(same lifecycle as the perspective model) and **only from frames where the
segment's marker was observed**. Zero extra state dims.

Measured, tail tip blinded for 20 s: **115° RMSE (worst 180°) -> 18° (worst
46°)**.

The kappa floor turned out to matter as much as the prior. During a dropout
nothing competes with it, so even a weak prior fully determines the angle —
on a near-uniform synthetic the fitted kappas were 0.2-0.8 and applying them
made the worst case *worse* (151° -> 180°), because mu was meaningless. At
`kappa >= 1.0` (sigma_theta <= 57°) a diffuse joint is skipped entirely and
the output is identical to before. Failure mode is "no help", never "wrong
help".

### Found while building it: the (L, theta) sign gauge

A segment's length is a free state with process noise and **nothing holds it
positive**. `(L, theta)` and `(-L, theta+pi)` place the marker at exactly the
same point, so the filter is free to sit in either. Measured, tail tip blinded
20 s: L walks through zero during the dropout (47% of frames negative),
re-acquires the marker in the mirrored gauge, and **stays there — 100%
negative afterwards, theta reported 180° from truth**. The user's run has
`q_length` ≈ 105 against fitted lengths of 10-20 px, which is ample to cross
zero.

Pre-existing, not introduced here. Marker *positions* stay correct (the
degeneracy is exact), so output isn't directly corrupted — but the RTS
smoother can smear between the two modes through `L ≈ 0`, which puts the
marker at the parent joint, and any consumer reading segment angles gets
garbage. 122hf only folds the mirrored frames back before averaging, because a
circular mean across both gauges is meaningless (measured: mu off by 177°,
kappa collapsed 13.3 -> 1.1). **Holding L > 0 in the filter is the real fix
and is not done.**

### Correction to the record

122hc and 122he both assert that `--const-accel-segments` had "never run".
That is wrong. The broken whitelist sits inside `if enable_warm_start_sigma:`;
verified on `5671eed` with rig-named data, `enable_warm_start_sigma=False`
runs const-accel end-to-end through EM. The accurate claim is **unusable in
the default configuration** — the CLI has `--no-warm-start-sigma` and the GUI
a checkbox (`pose_cleanup.py:1259`, default checked). Every log the user
showed has warm-start on, so it never ran *for them*. 121d wired q_jerk into
four of the five reconstruction sites and missed only the warm-start one.

### And a hole in my own verification

The sweep glob is `tests/smoke_122*.py`, which misses **73 test files**.
122he (pushed as `2859314`) rephrased the divergence message and broke
`smoke_kalman_pose_smoother_v2.py` — 152/152 on `5671eed`, failing on
upstream. Fixed here, and both pinned-wording assertions relaxed to pin
behaviour. **Sweeps should cover `tests/*.py`, not `tests/smoke_122*.py`.**

### Next
* Hold `L > 0` in the filter — the sign gauge above.
* GUI checkbox for the joint prior (CLI only for now).
* Fit `q_length` against the gauge too; it is likely inflated by sign flips.



**122he — cap iteration 0.** With the layout finally correct, the user's run
reached the filter and the EKF diverged: "validation hook triggered at
iteration 0", 96% NaN.

Not reproduced in-sandbox (3 GB RAM caps synthetic sessions far below their
54k frames), so this ships a structural fix and a diagnostic, not a
confirmed repair. **The user should drop `--const-accel-segments` first.**

The structural finding: *every* q hard cap in this module was applied only
inside `finalize_m_step_v2`. They guarded iterations 1..N and left iteration 0
running on whatever `fit_initial_params_v2` returned. The caps encode a
numerical stability boundary for the EKF, and the EKF does not care which
iteration it is on. Patch 121c's own comment says the absolute caps exist to
backstop "cases where the initial q is already large (heavy-noise sessions
push initial fit upward)" — it saw the initial fit going large, capped the
M-step's *use* of it, and never capped the value itself.

It is not a corner case. On ordinary synthetic mouse data the initial fit
returns `q_root_pos` ≈ 78,000–86,000 against 121c's documented boundary of
50,000 — past the point where, in its words, "even a moderate dropout (1
second) produces prediction covariances ... where the EKF S = HPH^T + R loses
conditioning". `_cap_params_to_hard_limits` is now the single place that
enforces the caps, applied to the initial fit and reported under `--verbose`.

Why const-accel is the prime suspect: `q_jerk_* = q_root_* × 10` at init
(uncapped), and jerk drives prediction variance as τ⁵ rather than τ³ across a
dropout. A tail marker occluded for seconds costs quadratically more with
const-accel on. And per 122hc, `--const-accel-segments` had *never run* — the
121d whitelist bug killed it in `build_Q_v2` before any EM. 121d and 121e are
both untested code that 122hc unblocked yesterday; 121e's jerk caps were
sized as "roughly 10× the q_root caps", which is dimensionally naive (px²/s⁵
vs px²/s³) and, at 500,000, sit above what the initial fit produces anyway.

The divergence message also stopped speculating. It listed three candidate
causes and told the reader to "inspect the early frames of x_smooth" — an
array they have no access to. It now reports NaN fraction, first non-finite
frame and its timestamp, which parameters sit on their cap, and whether
const-accel is active.

### Next
* Ask the user for a `--verbose` run without `--const-accel-segments`. If it
  survives, 121d's const-accel needs a properly calibrated q_jerk cap
  (measured against the τ⁵ boundary, not scaled off the q_root cap).
* CLI `--em-aggregation` still defaults to `pooled`; the handoff's own lesson
  is that median is strongly preferred across many sessions. Changing a
  default needs a decision, so it's flagged rather than done.



**122hd** fixes the error message 122hc introduced. It was readable and wrong.
Against a project that declared its own `[[pose.segments]]` and typed
`--orient-drift-segments body,head`, it asserted the layout came from
`[skeleton]`, that there was "no 'body' or 'head' to point at" (`head` was in
the list the message itself had just printed), and that the fix was to add a
`[[pose.segments]]` block the user already had — while printing the flag as
`--orientation-drift-segments`, which is not the flag. The one real fault,
`body` should be `back`, was the one thing it never said.

The lesson is the session's lesson again, one level up: a canned narrative
that asserts a cause it hasn't checked. An error that guesses is worse than one
that only lists facts, because the whole value of an error is that it can be
trusted. 122hd states what is missing, lists what exists, names the root, and
explains only where there is a signal checkable against the object in hand:
the name belongs to the built-in rig (compare against `standard_rat_layout()`),
or every segment is named after a marker and therefore the tree really was
derived from `[skeleton]` (a property of the layout, not a guess about
provenance). Near-match suggestions use a strict cutoff, since a loose one
invents advice (`trunk` -> `neck`) that is worse than silence.



**122hc follows 122hb the same day**, because 122hb's fix was incomplete in a
way worth recording: it made `smooth_pose_v2` stop assuming the built-in rig
when `layout is None`, and added `--config` to the CLI so the layout could come
from the project — but `main()` still *eagerly built the rig* when `--config`
wasn't passed, so it never reached the new guard. The user's next run failed
identically. A flag you have to know about is still a default for everyone who
doesn't. `find_project_config` now walks up from the input path, and the CLI
looks for the project before assuming anything.

Two more bugs surfaced underneath, both previously masked by the layout failure:

* `--orient-drift-segments body,head` cannot work against a `[skeleton]`-derived
  tree, which names each segment after its distal marker (`head_mid`, `back_L2`)
  and therefore has no `body` or `head`. That is not a bug — it's the reason
  `[[pose.segments]]` exists — but the failure was a raw `__post_init__`
  traceback. It now names the available segments and says to declare the tree.
* The warm-start rebuild of `initial_params` in `fit_noise_params_em_v2` was a
  hand-written whitelist of fields to preserve. It rotted three times: patch
  120b added the drift trio after `with_drift` runs crashed having lost them,
  patch 121a added the orientation-drift trio for the same reason, and patch
  121d added const-accel and never came back — so `--const-accel-segments`
  dropped `q_jerk_*` to None and died in `build_Q_v2`. Now a copy-every-field
  `dataclasses.replace`, which cannot rot again.



One patch, **122hb**, closing two bugs that both presented as "the smoother
can't find my markers" and were each misdiagnosed once.

**Bug 1 — the loaders renamed the markers.** Every pose reader did
`[str(c).lower() for c in df.columns]`. The intent was case-insensitive
*suffix* matching (`_X` vs `_x`); the effect was that any marker with an
upper-case letter came back renamed. `back_T4` → `back_t4`, `tail_V6` →
`tail_v6`. A layout built from `project.toml` (which keeps the case the user
typed) then matched nothing for those markers.

The tell was the split: on the strohA-kj project, 7 of 15 markers matched and
8 did not, and the 8 were **exactly** the 8 whose names contain an upper-case
letter. This was previously read as evidence of a partial marker rename, and
cost a session hunting for a second generation of files under `backups/` and
`derived/` that never existed. The recursive-`rglob` discovery scope is still
worth tightening, but it was not this bug and never was. The user's own
report — "the csv files show the old labels and the parquet files show the new
labels" — was correct and should have been believed.

Marker names are identifiers. `normalize_pose_columns` now folds the suffix
only and returns the base verbatim. No-op for the all-lower-case marker sets
of legacy projects.

**Bug 2 — the layout was resolved before the model that carries it.**
`smooth_pose_v2` defaulted `layout` to `standard_rat_layout()`, validated and
packed the data against it, and *then* — fifty lines later — opened the `.npz`
whose layout it should have used. Both load-mode paths in the GUI (the "load
model" mode, and pass 2 of the train-on-a-subset two-pass workflow) pass no
layout, so both validated project data against the built-in 2019 rat rig and
died citing `nose`/`back1`/`tailbase`. Load now happens first; the model's
layout is authoritative because its parameters are dimensioned to it.

A default that names markers the project has never heard of is not a default.
`layout=None` with no model now resolves from `config_path`, and falls back to
the built-in rig only when the data genuinely is the built-in rig — otherwise
it raises and says what to add.

**The 122gv hole is closed.** That guard only fired on *zero* overlap; a 7/15
mismatch merely warned that the missing markers "will have default offsets",
which undersells "eight segments are unobservable, the covariances will blow
up, and this will look like a hang". Any missing layout marker is now a hard
error, and the error names case-only differences explicitly.

### Shipped

* `normalize_pose_columns` in `kalman_diagnostic.py`; all four loaders there
  plus the smoother's own direct reader routed through it.
* `_resolve_layout` / `_validate_layout_against_data` in
  `kalman_pose_smoother_v2.py`; `load_model_v2` hoisted above both.
* `config_path=` on `smooth_pose_v2`, threaded through all four GUI call sites
  and a new `--config` on the CLI (which also stopped assuming the rat rig).
* Pre-flight block: layout, `state_dim`, markers matched, sessions, workers,
  projected GiB/worker, and which stage is about to run — printed before the
  first frame is filtered.
* `find_project_config` in `project_layout.py`; the CLI uses it to locate the
  project from the input path, warns when it falls back to the rig, and refuses
  inputs spanning two projects rather than guessing between them.
* `fit_noise_params_em_v2`'s warm-start rebuild: whitelist -> copy-everything.
* `tests/smoke_122hb_marker_case_and_layout.py` (29 checks) and
  `tests/smoke_122hc_project_discovery.py` (32 checks). Real tests: actual
  parquet/CSV/DLC-header files through the actual loaders, and real subprocess
  CLI runs against real project trees — not AST-only.

### Next

* Discovery scope: `discover_pose_files` and the `smooth_pose_v2` fallback
  glob are still recursive and still sweep `backups/` and `derived/`. Now that
  the guard is strict this surfaces as a loud error rather than silent garbage,
  but it should be scoped or excluded outright.
* Audit the other `NoiseParamsV2` rebuild sites (`finalize_m_step_v2`,
  `finalize_m_step_v2_from_per_session`) for the same whitelist rot.
* Heartbeat inside warm-start / EM — long phases are still silent between
  iterations.
* `[pose.skeleton]` schema migration (drop `nodes`, move under `[pose]`).
* Joint model phase 1: `joint = "rigid"` → merge into parent.



Two patches: **122e** makes `ConfigReader` (959 lines, inherited
by 268+ files across the codebase) read v1 `project.toml` while
preserving its legacy attribute surface, so every backend
continues to work transparently against v1 projects without
per-file modification. **122f** removes direct `configparser`
usage from every UI form method, encapsulating the legacy INI
branch in a handful of layout-agnostic helpers.

Directional commitment: as of this session, Mufasa no longer
depends on `project_config.ini` for any production code path.
INI parsing is kept alive in three places only — the migration
tool (`legacy_layout.py` + `migrate_project.py`), the
`read_config_file` fallback for legacy projects opened post-
migration, and the encapsulated legacy branches of the new
read/write helpers.

### Shipped

#### TOML-aware ConfigReader (patch 122e)

New module **`mufasa/utils/toml_to_configparser.py`** translates
a parsed v1 `project.toml` into a synthetic
`configparser.ConfigParser` populated with the legacy section
and key names. Coverage:

- `[General settings]` — `project_path` (= v1 root, not a
  `project_folder/` subdir), `project_name`, `file_type`,
  `workflow_file_type`, `animal_no`, `os_system`
- `[SML settings]` — `model_dir`, `no_targets`, `target_name_N`,
  `model_path_N` for each classifier
- `[threshold_settings]`, `[Minimum_bout_lengths]` — per-
  classifier placeholders (`"NaN"`)
- `[create ensemble settings]` — pose preset code +
  ML-training placeholders (`"NaN"` for unset)
- `[Multi animal IDs]` — comma-joined `id_list`
- `[Outlier settings]` — movement / location criteria
- Plus `[Frame settings]`, `[Line plot settings]`,
  `[Path plot settings]`, `[ROI settings]`, `[Directionality
  settings]`, `[process movement settings]` (empty placeholders
  the legacy creator wrote)

Missing TOML values default to `"NaN"` (the legacy
`Dtypes.NONE` sentinel) so existing
`read_config_entry(..., default_value=...)` calls behave
identically.

**`mufasa/utils/read_write.py`** — `read_config_file` now
branches on the suffix: `.toml` → delegates to the shim;
anything else → the original INI parser. Case-insensitive
(`.TOML` works too). Legacy projects opened after a migration
keep their existing path verbatim.

**`mufasa/mixins/config_reader.py`** — `ConfigReader.__init__`
detects v1 by suffix, pre-loads the parsed TOML on
`self._v1_toml_data`, and routes the body-parts handling
through TOML rather than the `project_bp_names.csv` file. A
new method `_apply_v1_path_overrides` runs early in `__init__`
(before any filesystem read) and rewrites the 30+ path
attributes — `input_csv_dir`, `outlier_corrected_dir`,
`features_dir`, `targets_folder`, `machine_results_dir`,
`video_dir`, `video_info_path`, all the plot dirs, the SHAP /
directionality / ROI / clf-validation dirs, plus `logs_path`
and `roi_coordinates_path` — to v1 equivalents under
`sources/`, `derived/`, `models/`, `logs/`. Multi-run stages
(outlier_corrected, features, classifications) resolve to the
**latest run** subdir (`derived/<stage>/<run_id>/`) when any
exist, falling back to the stage parent. File lists are re-
globbed against the new paths.

#### configparser removal from ui_qt forms (patch 122f)

Two new helpers in **`mufasa/project_layout.py`**:

- `project_paths_from_config(config_path)` → dict mapping
  `project_root`, `video_dir`, `input_pose_dir`, `logs_dir`,
  `video_info_path`, `models_dir` to v1 or legacy equivalents
  depending on the config suffix.
- `project_metadata_from_config(config_path)` → dict with
  `animal_count`, `file_type`, `body_parts`, `animal_ids`,
  `classifier_targets`, `pose_config_code`. v1 reads from
  `project.toml` directly; legacy parses the INI plus the
  `project_bp_names.csv` file.

Forms migrated (every direct `configparser.ConfigParser()` call
in form methods replaced):

- **pose_cleanup.py** — `SmoothingForm.target`,
  `InterpolateForm.target`, `OutlierSettingsForm.build/target`,
  `EgocentricAlignmentForm.build/collect_args`,
  `KalmanV2SmoothingForm.build` (legacy fallback),
  `RunOutlierCorrectionForm.build/target`. The
  read-modify-write of outlier settings is encapsulated in two
  module-level helpers `_read_outlier_settings` and
  `_write_outlier_settings`. v1's TOML schema gains a nested
  `[outlier_settings.references]` table
  (`Animal_X = ["bp1", "bp2"]`) mirroring the legacy
  per-animal reference keys.
- **classifier.py** — `_refresh_remove_options`,
  `_add_classifier`, `_remove_classifier` now use two
  module-level helpers `_read_classifiers` and
  `_write_classifiers`. v1 read-modify-writes `project.toml`'s
  `[classifiers].targets`.
- **analysis.py** — `_load_classifier_names` delegates to
  `project_metadata_from_config`. Top-level `import
  configparser` removed.
- **visualizations.py** — the data-paths auto-population path
  uses `project_paths_from_config` + `project_metadata_from_config`.
  Inline `import configparser` inside `target` removed.
- **addons.py** — `_load_cue_light_names` branches on suffix.
  v1 returns `[]` (cue-light ROI names aren't in the v1 schema
  yet; form falls back to free-text entry).
- **annotation.py**, **roi.py** — dead `import configparser`
  lines removed (no other use).

Net effect: every form *method* is `configparser`-free.
`configparser` references that remain are in three places, all
encapsulated:

1. `pose_cleanup.py` — legacy branches of `_read_outlier_settings`
   and `_write_outlier_settings`
2. `classifier.py` — legacy branch of `_write_classifiers`
3. `addons.py` — legacy branch of `_load_cue_light_names`

### Test counts at session end

| suite                                    | count    |
|------------------------------------------|----------|
| smoke_project_layout                     |   9/9    |
| smoke_migrate_project                    |   5/5    |
| smoke_recent_project                     |   6/6    |
| smoke_pose_cleanup_v2_wiring             |   2/2    |
| smoke_model_dual_save                    |  23/23   |
| smoke_input_source_picker                |  29/29   |
| smoke_outlier_forms_wiring               |  40/40   |
| smoke_empty_classifier                   |   1/1    |
| smoke_config_creator_v1                  |  38/38   |
| smoke_v1_configreader (NEW, 122e)        |  69/69   |
| smoke_v1_form_configparser_removal       |  37/37   |
| (NEW, 122f)                              |          |

### Sandbox testing limitation

ConfigReader itself can't be instantiated in this sandbox —
its import chain pulls cv2, h5py, trafaret, and tkinter, none
of which are available. The 122e test exercises the three
layers it can reach (the TOML shim behaviorally, the
`read_config_file` routing via AST, the ConfigReader v1 logic
via AST). Full behavioral verification — actually constructing
a `ConfigReader(config_path=project.toml)` and inspecting its
attribute surface against the expected v1 paths — is the
user's to confirm on a real install. The wiring is testable
even if the runtime isn't.

### Deliberately deferred

**Backend writes leak provenance.** ConfigReader's
`_apply_v1_path_overrides` resolves multi-run stage attributes
to the **latest run** subdir. That's correct for reads but wrong
for writes — a backend that does `os.makedirs(self.outlier_corrected_dir,
exist_ok=True)` and then writes files there will mutate the
prior run's directory in place, corrupting provenance. Each
writing backend needs `ProjectPaths.stage_run_dir(...)`
allocation at run time instead. Track as the "v1-aware backend
writes" thread.

**Configs directory.** `configs_meta_dir` falls back to
`<root>/configs/` for v1; the v1 layout doesn't model a configs
dir yet.

**Body-parts path in v1.** `body_parts_path` is set to the
project.toml path itself (informational only — v1 reads body
parts from TOML data directly). Any legacy code that tries to
parse this file as a CSV will fail loudly.

**reconfigure_project_user_defined.** Still INI-only.
Currently unreachable from CreateProjectDialog. Kept for the
File → Reconfigure menu action; v1 rewrite is a small patch
when needed.

**Cue-light ROI metadata in v1.** `addons.py._load_cue_light_names`
returns `[]` for v1. If cue-lights matter going forward, the v1
schema needs a `[roi.cue_lights]` table plus form-side writers.

### Pickup checklist for next session

1. Confirm tests green:
   ```bash
   for t in tests/smoke_*.py; do PYTHONPATH=. python "$t"; done
   ```
2. **Full ConfigReader behavioral verification** on a real
   install — instantiate against a v1 project.toml and walk
   the attribute surface. Anything still under `csv/` instead
   of `sources/`/`derived/` is a missed override.
3. **v1-aware backend writes** — the highest-priority deferred
   thread. Each writing backend (outlier correction, feature
   extraction, classifier training, visualization) needs
   `ProjectPaths.stage_run_dir(...)` allocation at run time
   instead of reading a "latest" path from ConfigReader.
4. **Batch pipeline through inference** — still the original
   ask from session 2026-05-11; never started.
5. **Migration prompt on legacy projects** — when
   `detect_layout(project) == 'legacy'`, dialog at workbench
   launch offering migration. Small patch.

**Also fixed here: a regression this session shipped.** 122he rephrased the
divergence message and broke `tests/smoke_kalman_pose_smoother_v2.py`, which
pinned the string "EKF has diverged". It went unnoticed because the sweep glob
is `smoke_122*.py` and **73 test files sit outside it**, that one included. The
suite was 152/152 before this session and 122he is already pushed. Both sweeps
are now baseline-diffed against 5671eed (pre-session): 264 suites, no
regressions.

### Next after 122hf
* The sweep glob is a verification hole. `smoke_122*.py` is what the handoff
  prescribes, but it misses 73 files; the loop should cover `tests/*.py`.
* `--joint-prior` is opt-in and unproven on real data. On strohA-kj the fit
  would learn mu/kappa for `tail_V32` from a state its sigma=225px marker
  barely influences; kappa may come back under the floor and disable itself.
  Honest, but it means the sigma=225 problem has to be solved, not routed
  around.
* A 20 s reconstruction at 18 deg RMSE is a prior talking, not a measurement.
  Anything scoring tail kinematics needs to know which it is looking at.

---

## Session 2026-05-11 — Dual-save model store + InputSourcePicker + pose-cleanup redesign + v1-only new projects

Three patch series landed: **122b** factors model provenance and
input-source selection into reusable primitives; **122c** uses
them to redesign the pose-cleanup page with explicit Run / Skip
outlier-correction forms and the legacy forms folded into an
"Advanced / legacy" section; **122d** cuts new-project creation
over to v1-only and makes the Kalman v2 form v1-aware so the
fresh-project pose-cleanup workflow runs end-to-end without
touching legacy paths.

### Shipped

#### Dual-save model store (patch 122b)

Trained models now live in two places. The **global cache** at
`~/.config/mufasa/models/<name>.npz` is a cross-project library
— a model trained once is discoverable from any project on the
same machine. The **project store** at `<project>/models/<name>/`
holds `model.npz` plus a `card.toml` carrying provenance
(`source_path`, `sha256`, `copied_at`, `mufasa_version`). Both
get a copy on every save / load crossing.

- New `project_layout.py` helpers: `import_model_into_project`,
  `mirror_model_to_global_cache`, `file_sha256`,
  `global_model_cache_dir`, `resolve_v1_project_root`.
  Content-hash idempotence: re-importing/mirroring the same
  model is a no-op. Different-content collision raises
  `FileExistsError` rather than silently overwriting (training
  output passes `overwrite=True`; load-side imports use the
  default no-overwrite to protect known-good models).
- `KalmanV2SmoothingForm.target`: a `_post_train_dual_save()`
  closure runs after every `smooth_pose_v2(save_model=...)`
  call (both single-pass and two-pass training paths); the
  load branch imports the chosen model into the project's
  `models/` before passing it to the smoother. Failures in
  either are soft — logged, but the original `.npz` is left
  intact.

#### InputSourcePicker (patch 122b)

New module `mufasa/ui_qt/input_source_picker.py`.

- `discover_input_sources(config_path, project_root)` — pure
  function. Walks both layouts: legacy `csv/{input_csv,
  outlier_corrected_movement, outlier_corrected_movement_location,
  smoothed_v2}/` and v1 `sources/pose/` + `derived/smoothed/
  <flavor>/<run>/` + `derived/outlier_corrected/<run>/`. v1
  candidates come first when both are present (post-migration
  transient). Multi-run stages list one entry per run, newest
  first. Empty / nonexistent dirs are skipped silently.
- `_DEFAULT_PREFER_ORDER`: kalman_v2 → outlier_corrected →
  savitzky → raw. Exactly one source is marked `is_default`.
  Overridable per call (the new `RunOutlierCorrectionForm`
  prefers raw, since outlier correction usually precedes
  smoothing).
- `InputSourcePicker` Qt widget: combobox + path field + browse
  + refresh (`↻`). Custom-path sentinel un-greys the path field.
  PySide6 imported lazily so the pure discovery function tests
  headless.
- **First consumer**: `EgocentricAlignmentForm`. The previous
  auto-detect of `csv/outlier_corrected_movement_location/`-vs-
  `csv/input_csv/` is gone; users can now feed egocentric
  alignment any prior stage's output, including a specific
  Kalman v2 run.

#### Pose cleanup redesign (patch 122c)

- New `RunOutlierCorrectionForm`. Chains
  `OutlierCorrecterMovement` → `OutlierCorrecterLocation`. Input
  via `InputSourcePicker` (preference RAW). Per-stage checkboxes
  let users disable movement or location independently. Output
  dir auto-generates `derived/outlier_corrected/<run_id>/` on v1
  projects (with `run.toml` provenance written after) and falls
  back to legacy `csv/outlier_corrected_movement_location/`
  otherwise. When both stages run, movement writes to a
  `_movement_intermediate/` sibling that location reads from.
- New `SkipOutlierCorrectionForm`. Thin wrapper around
  `OutlierCorrectionSkipper` — no fields, just a Run button.
  Writes `run.toml` with `params.skipped = true` and a
  `SKIPPED` marker file when run inside a v1 project so future
  layout-aware tooling can distinguish "skipped" from "missing"
  runs.
- Pose cleanup page sections reordered to match the conceptual
  pipeline:

  ```
  1. Interpolate missing frames
  2. Kalman v2 smoothing
  3. Run outlier correction         ← NEW
  4. Skip outlier correction        ← NEW
  5. Egocentric alignment           (now picker-driven)
  6. Advanced / legacy              (stacks SmoothingForm,
                                     OutlierSettingsForm,
                                     DropBodypartsForm)
  ```

  Legacy `Smoothing` / `Outlier correction settings` /
  `Drop body-parts` top-level sections are gone; their forms
  still exist (settings is still consumed by the new outlier
  forms — it writes the thresholds they read) but they're
  folded into one Advanced section using
  `WorkflowPage._instantiate`'s existing form-stacking.

#### v1-only new projects + Kalman v2 v1-awareness (patch 122d)

`ProjectConfigCreator` was rewritten end-to-end. The legacy
SimBA tree generator (the `__create_directories` /
`__create_configparser_config` pair that built `project_folder/
csv/{input_csv, outlier_corrected_*, features_extracted, ...}`
+ `project_config.ini`) is gone. New projects land as v1:

```
<parent>/<project_name>/
├── project.toml                       (new)
├── sources/{videos,pose,annotations}/  (via ProjectPaths.ensure_skeleton)
├── derived/
├── models/
└── logs/
```

The new `project.toml` schema is small and explicit:

- top-level: `project_layout_version`, `project_name`,
  `created`, `mufasa_version`, `os_platform`
- `[pose]`: `animal_count`, `file_type`, `body_parts` (list,
  in order), `pose_config_code`, `pose_config_idx`, `animal_ids`
- `[classifiers]`: `targets` (may be empty)
- `[outlier_settings]`: `movement_criterion`, `location_criterion`
  (initial `"NaN"` matches the legacy `Dtypes.NONE` sentinel
  so existing settings code can read it verbatim once it's TOML-
  aware)

The Qt create-project dialog (`CreateProjectDialog`) was
simplified accordingly: the previous "create with placeholder
preset, then call `reconfigure_project_user_defined` to patch
in the autodetected body parts" two-step is gone. Both the
preset and autodetect paths now funnel body parts directly into
`ProjectConfigCreator` via the new `body_parts=` parameter, so
the project is fully configured on first write. The dialog's
docstring and the project-setup-page banner were updated to
say `project.toml` instead of `project_config.ini` (with the
banner still acknowledging legacy `project_config.ini` for
projects opened from before this cutover).

`KalmanV2SmoothingForm` was made v1-aware so the fresh-project
pose-cleanup workflow runs end-to-end on v1 without any legacy
path resolution:

- **Input dir** defaults to `<project>/sources/pose/`.
- **Output dir** defaults to
  `<project>/derived/smoothed/kalman_v2/<run_id>/`. A fresh
  `run_id` is allocated at form-build time (cached on
  `self._v1_run_id`) so the path the user sees in the field is
  exactly what gets written on Run. Closing and reopening the
  form gives a fresh `run_id`; the form never overwrites a
  prior run.
- **Save-model path** defaults to `<run_dir>/model.npz`,
  co-located with the smoothed pose data. The 122b dual-save
  flow still mirrors to `~/.config/mufasa/models/model.npz` and
  imports into `<project>/models/model/` on top.
- After smoothing completes, the form writes a
  `<run_dir>/run.toml` provenance file tagged
  `stage = "smoothed.kalman_v2"`, with `params` capturing every
  scalar/list kwarg passed to `smooth_pose_v2` plus the mode
  (`"train"` or `"load"`). Soft-fails (logged) if anything goes
  wrong — pose data already on disk is left intact.

Legacy projects still work via the existing `configparser`
fallback in `build()`. `resolve_v1_project_root` distinguishes
the two cases.

Net effect: the data flow for the v1 pose-cleanup pipeline is
now self-consistent — raw pose lives at `sources/pose/`, the
Kalman v2 smoother writes to
`derived/smoothed/kalman_v2/<run_id>/`, and the egocentric
alignment form's `InputSourcePicker` surfaces both raw and any
prior smoothed run as candidates (with the most-processed run
as the default). Subsequent pose-cleanup operations read from
raw or smoothed exactly as intended.

### Test counts at session end

| suite                          | count    |
|--------------------------------|----------|
| smoke_kalman_pose_smoother_v2  | 152/152  |
| smoke_project_layout           |   9/9    |
| smoke_migrate_project          |   5/5    |
| smoke_recent_project           |   6/6    |
| smoke_pose_cleanup_v2_wiring   |   2/2    |
| smoke_model_dual_save          |  23/23   |
| smoke_input_source_picker      |  29/29   |
| smoke_outlier_forms_wiring     |  40/40   |
| smoke_empty_classifier         |   1/1    |
| smoke_config_creator_v1        |  38/38   |

`smoke_model_dual_save` redirects `HOME` so it doesn't pollute
the real `~/.config/mufasa/models/`. `smoke_input_source_picker`
covers the pure discovery function behaviorally and the Qt
widget at AST level (PySide6-free sandbox). The wiring tests
are AST-only. `smoke_empty_classifier` was rewritten from its
legacy INI shape into a v1-`project.toml` shape; the original
contract (empty `target_list` is accepted) is preserved.

### Deliberately deferred

**Model store**

- Trained-model auto-naming. The form's default save path is
  now `<run_dir>/model.npz` for v1 projects (one model per run,
  collision-free by construction); the global cache still flat-
  names them as `model.npz` which IS collision-prone. Probably
  want to use the run_id as the global-cache name too.
- Card.toml schema is minimal — `model_name`, `source_path`,
  `sha256`, `copied_at`, `mufasa_version`. No training params,
  no eval metrics, no fingerprint of the training data. The
  changelog's earlier note on `card.toml` for classifiers still
  applies; this is the v0.

**InputSourcePicker**

- Picker doesn't show file counts or last-modified time per
  source. For a project with many runs it'd be useful to know
  which has data and how fresh it is. Two lines per item in the
  combobox would do it.
- No source preview / inspect button. Picking a run and seeing
  its `run.toml` parameters before committing would help.

**Pose cleanup**

- "Drop body-parts" still lives in Advanced / legacy. It's
  really a project-setup decision, not a per-run cleanup step.
  Move it to the project-setup page once that's the right
  surface.
- `RunOutlierCorrectionForm` writes `run.toml` for v1 but
  doesn't yet enumerate the actual input file list (params has
  the `data_dir`, not the per-file paths). The `write_run_toml`
  `inputs` field accepts a list; populating it would make
  full reproducibility checkable.
- `SkipOutlierCorrectionForm` writes a v1 `run.toml` next to a
  `SKIPPED` marker file but doesn't actually populate the v1
  `derived/outlier_corrected/<run>/` with the skip-copied
  files. That's because `OutlierCorrectionSkipper` writes to
  the legacy `csv/outlier_corrected_movement_location/` and
  isn't yet v1-aware. Fixing this is part of the broader
  "make each backend v1-aware" thread.
- The legacy `SmoothingForm` (Savitzky-Golay) is still mounted
  in Advanced. Once we're confident Kalman v2 covers every
  pre-Kalman use case the dataset hit, this form can be
  removed entirely.

**v1 cutover (122d) — the iceberg below the waterline**

The 122d patch makes new projects v1, but the downstream forms
that consume those projects are still mostly INI-driven. The
**only** form that fully works in a fresh v1 project right now
is `KalmanV2SmoothingForm` (because its target reads pose
directly from `input_dir` without ever touching
`config_path`). Every other form — `InterpolateForm`,
`RunOutlierCorrectionForm`, `SkipOutlierCorrectionForm`,
`EgocentricAlignmentForm`, the legacy three, plus every form
on every other page — calls into a backend that uses
`ConfigReader` to parse `config_path` as an INI. In a v1
project `config_path` points to `project.toml`, and those
backends will fail to read it.

The migration thread that fixes this:

1. Add a `ConfigReader` subclass (or a `read_project_toml`
   helper) that reads `project.toml` and exposes the same
   attributes the legacy ConfigReader does (`animal_cnt`,
   `body_parts`, `file_type`, `input_csv_dir`, etc.). For v1,
   `input_csv_dir` resolves to `sources/pose/`,
   `outlier_corrected_dir` to a latest-run path under
   `derived/outlier_corrected/`, and so on.
2. Route every `ConfigReader`-using backend through the
   subclass.
3. Make each form's `target()` pass the right v1 paths down
   so its backend never has to guess.

That's a multi-session thread; the priority order matches the
pose-cleanup page's section order (Interpolate, Outlier-Run,
Outlier-Skip, Egocentric backend, Advanced/legacy).

Until that's done, the practical v1 workflow is:

- Drop raw pose into `<project>/sources/pose/`.
- Run **Kalman v2 smoothing** — produces
  `derived/smoothed/kalman_v2/<run>/`.
- (Other pose-cleanup forms will fail with a config error.)
- Egocentric alignment partially works — the form constructs
  fine (no INI needed in the form itself, since the picker
  drives the data path), but its backend
  (`EgocentricalAligner`) reads `config_path` and may fall
  over depending on how deeply it relies on the INI.

In short: **122d unlocks the smoother in v1 mode; the rest of
the pipeline catches up patch by patch.**

### Open questions

- Should `RunOutlierCorrectionForm`'s save_dir picker accept a
  v1 run directory the user already created (rather than always
  auto-generating)? Useful for re-running with tweaked
  thresholds in the same provenance dir. Currently auto-gen
  only.
- The InputSourcePicker shows v1 runs newest-first. For a
  project with dozens of runs this could get unwieldy. A
  "latest only" toggle (showing just the most recent run per
  flavor) would help; the full list could be reachable via an
  "All runs…" entry. Not urgent.
- The legacy `reconfigure_project_user_defined` helper still
  exists but isn't called by `CreateProjectDialog` anymore.
  Keep it for the File → Reconfigure project from DLC file…
  menu action, or retire it entirely? Probably keep until the
  v1-aware reconfigure equivalent exists.

### Pickup checklist for next session

1. Confirm tests still green:
   ```bash
   PYTHONPATH=. python tests/smoke_project_layout.py && \
   PYTHONPATH=. python tests/smoke_migrate_project.py && \
   PYTHONPATH=. python tests/smoke_recent_project.py && \
   PYTHONPATH=. python tests/smoke_pose_cleanup_v2_wiring.py && \
   PYTHONPATH=. python tests/smoke_model_dual_save.py && \
   PYTHONPATH=. python tests/smoke_input_source_picker.py && \
   PYTHONPATH=. python tests/smoke_outlier_forms_wiring.py && \
   PYTHONPATH=. python tests/smoke_empty_classifier.py && \
   PYTHONPATH=. python tests/smoke_config_creator_v1.py
   ```
2. Three threads ready to pick up:
   - **v1-aware ConfigReader.** Highest priority — every form
     beyond Kalman v2 fails in a fresh v1 project until this
     exists. See "iceberg below the waterline" section above.
   - **Batch pipeline through inference** (still the original
     ask from this session — scoped out, never started).
     Schema for `pipeline.toml`, CLI scaffold, ephemeral-project
     synthesizer, four stage adapters. Estimated 1500–2500 LOC
     across one or two sessions.
   - **Migration prompt on legacy projects.** Original
     deferred item from 122a: when `detect_layout(project) ==
     'legacy'`, surface a dialog at workbench launch offering
     migration. Small patch.

---

## Session 2026-05-08 to 2026-05-11 — Kalman v2 smoother feature completion + workbench wiring + project-layout redesign

### Shipped

#### Smoother backend (patch 121d–e)

- **121d — constant-acceleration extension** for selected segments.
  Listed segments gain an acceleration block at the end of the state
  vector (+4 dims for root translation + orientation, +2 dims per
  non-root segment). F gets integrator cross-terms (`v_new = v + a·dt`,
  `x_new = x + v·dt + a·dt²/2`); Q gets diagonal jerk noise on accel
  slots. FK and Jacobian unchanged. New CLI flag
  `--const-accel-segments`. Backward compatible (empty list = no-op).
  Tests 145-148 (4 new on top of 121b/c's 144).

- **121e — M-step learning of q_jerk_***. EM now adapts `q_jerk_root_pos`,
  `q_jerk_root_ori`, `q_jerk_seg_ori` from Q_hat's accel-block
  diagonals instead of holding them at the initial 10×-q-root default.
  Floor/ceiling/hard-cap pattern from 121c
  (`_M_STEP_Q_JERK_*_HARD_CAP = 500000 / 500 / 500`). Per-session
  aggregator still passes through prev values — deferred. Tests 149-152.

#### Workbench wiring (patch 121f–i)

- **121f — Kalman v2 smoother in Pose Cleanup page**. New
  `KalmanV2SmoothingForm` section sitting below the legacy Savitzky-
  Golay smoother. Calls `smooth_pose_v2` via Python API (not
  subprocess) so errors surface in the progress dialog. Fields:
  input/output dirs, fps, likelihood threshold, EM iters, workers,
  per-marker drift, orientation drift segments, const-accel segments,
  save-model checkbox. PySide6-independent AST-level wiring smoke
  test.

- **121g — train/load mode + training subset + EM parameters**. Mode
  selector at top of form (radio buttons: train new / load saved).
  Training-mode group adds a file-list picker for subset selection
  (auto-refreshes on input-dir change, with select-all/clear/refresh
  buttons), full EM parameter surface (tolerance, damping, pooled-vs-
  per-session aggregation, warm-start σ, perspective, validation),
  save-model path with browse. Load-mode group surfaces the model
  file picker. Two-pass workflow runs when training on a subset: fit
  on subset → load → smooth all.

- **121h — standard model directory at `~/.config/mufasa/models/`**.
  Module-level helper `_default_model_dir()`. Five touchpoints
  (placeholder, save-browse, load-browse, collect_args, two-pass
  workflow) route through it. Models trained once are discoverable
  from any project on the same machine.

- **121i — persistent recent-project at `~/.config/mufasa/recent`**.
  New `mufasa/ui_qt/recent_project.py` module (PySide6-free, testable
  headless). Launch priority: `--project` → CWD auto-discover →
  recent file → none. `--no-recent` flag disables fallback.
  `_switch_to_project` saves on every project switch. Stale/empty/
  unwriteable entries fail silently.

#### Project layout redesign (patch 122a)

Major restructure — patch number stepped to 122 to signal the break.

- **122a — full project-layout v1 + migration tool**. New layout:

  ```
  <project_root>/
  ├── project.toml                     # was project_config.ini
  ├── sources/{videos,pose,annotations}/   # read-only inputs
  ├── derived/                         # generated; safe to delete
  │   ├── smoothed/<flavor>/<run_id>/
  │   ├── outlier_corrected/<run_id>/
  │   ├── features/<run_id>/
  │   ├── classifications/<run_id>/
  │   └── frames/{extracted,annotated}/
  ├── models/<model_name>/
  └── logs/<run_id>/
  ```

  Three new modules: `mufasa/project_layout.py` (`ProjectPaths`,
  `Stages`, `SmoothingFlavors`, `generate_run_id`, TOML I/O, version
  validation), `mufasa/legacy_layout.py` (SimBA-to-v1 mapping table,
  INI parser preserving unknown sections under `[legacy.*]`,
  body-part-name file reader), and `mufasa/cli/migrate_project.py`
  (`MigrationPlan` separates planning from execution, dry-run default,
  `--commit` to actually move, idempotent on already-migrated
  projects, `MIGRATION.toml` manifest written for auditability).

  Run-id format: `YYYYMMDD-HHMMSS-<6hex>` — sortable lexically =
  sortable chronologically. Per-run `run.toml` carries stage,
  parameters, input file list, mufasa version, results.

  Tests: `tests/smoke_project_layout.py` (9 checks: constants, run-id
  format, ProjectPaths skeleton, stage_run_dir, list_runs/latest_run,
  TOML roundtrip, version guard, run.toml roundtrip, detect_layout)
  and `tests/smoke_migrate_project.py` (5 checks: synthetic SimBA
  project → migration → verify all files at expected v1 locations +
  project.toml content + MIGRATION.toml manifest + idempotence + dry-
  run + unknown-path exit code + project_folder-pointed call shape).

### Test counts at session end

| suite                              | count    |
|------------------------------------|----------|
| smoke_kalman_pose_smoother_v2      | 152/152  |
| smoke_workbench_auto_discover      |  10/10   |
| smoke_workbench_launcher           |  10/10   |
| smoke_pose_cleanup_v2_wiring       |   2/2    |
| smoke_recent_project               |   6/6    |
| smoke_project_layout               |   9/9    |
| smoke_migrate_project              |   5/5    |

### Real-data observations worth preserving

Confirmed during 121bc real-data runs (67-session dataset, 30 fps):

- `--orient-drift-segments body` made the tail look better and gave
  honest larger variance circles (the over-confidence from before was
  hiding real uncertainty), but head markers became too constrained —
  root angular residual no longer absorbing.
- Adding `head` to drift loosened it. `body,head` worked.
- With drift wired and `--const-accel-segments body,head`, motion was
  slightly over-smoothed during fast turns. Hypothesis: the constant-
  velocity predictor lags on accelerating motion; const-accel should
  help. 121d/e land that.
- Useful knobs found during tuning: `--likelihood-threshold` (lower
  includes more obs, less prior reliance), q-scaling via save/load
  model.
- Recommended config for this dataset:
  `--with-drift --orient-drift-segments body,head --const-accel-segments body,head --fps 30 --em-max-iter 20 --workers 12`.

### Deliberately deferred

Notes on shape, not commitment to implementation — pick whichever
matters next.

**Smoother backend**

- M-step learning of q_jerk in the per-session aggregator. Currently
  only the pooled path (the default) learns; the per-session path
  passes through prev_params. Pattern would mirror what's in
  `finalize_m_step_v2` already — same sufficient stats, same caps,
  per-session aggregation via the existing scalar/median machinery.

**Workbench**

- Live progress reporting from `smooth_pose_v2`. The form currently
  shows an indeterminate progress dialog; the `[smoother-v2] ...`
  lines still go to stdout. A Qt-friendly progress-callback parameter
  on `smooth_pose_v2` plus a signal-based bridge into the dialog
  would surface them. Worth doing but not blocking.
- Cancel button. Dialog has one but `smooth_pose_v2`'s worker pool
  doesn't poll for cancellation between sessions.
- Multi-segment pickers. Orientation-drift and const-accel segments
  are typed as comma-separated strings. Checkbox grid against the
  layout's actual segment list would be a UX upgrade.
- Model inspection. Loading a model just feeds it to the smoother;
  the form doesn't surface the layout extensions or fitted q values
  from the loaded file. An "Inspect" button could read the npz and
  show a summary.

**Project layout / consumer migration**

The 122a patch lands the layout primitives but NOT the form
integration. Recommended order for the next session:

1. **Launcher UX**: when `detect_layout(project_root) == 'legacy'`,
   pop a dialog offering to run `migrate_project`. Small patch
   touching `workbench_app.py`'s startup path.
2. **`KalmanV2SmoothingForm` v1-aware**: when `config_path` resolves
   to a v1 project, default output_dir to
   `ProjectPaths.smoothed_run_dir(SmoothingFlavors.KALMAN_V2)` (auto-
   generated run id) and default save_model to
   `<run_dir>/model.npz`. Falls back to legacy defaults for legacy
   projects. Write `run.toml` after each successful run.
3. **Each other form** (outlier correction, features, classification)
   gets the same one-form-at-a-time treatment. Stop once the user's
   workflow doesn't hit a legacy path anymore.
4. **`ConfigReader` subclass** that reads `project.toml`. Existing
   consumers route through it transparently.
5. **Run-id selector** in the v2 form: optional "Run name" field
   (default: auto), and the "Load saved model" picker enumerates
   `derived/smoothed/kalman_v2/*/model.npz` from the current project
   when one is loaded.
6. Once all forms migrated: deprecate the legacy INI reader. The
   legacy-layout reader stays as long as we want migrate-project to
   work.

**Schemas to design later**

- `models/<model_name>/card.toml` for trained classifiers — what they
  were trained on, when, what params. Migration just moves
  `models/` as-is right now; the format for new training runs is open.
- Anything that wants a stable schema in `project.toml` beyond what
  `parse_legacy_config` produces (animal_count, body_parts, stage
  defaults). Worth letting it grow organically as consumer code
  migrates.

### Open questions

- Is `~/.config/mufasa/models/` the right place for the global model
  cache once projects have their own `<project>/models/`? Two options:
  keep both (global cache for cross-project reuse, project-local for
  reproducibility) or drop the global one and require explicit copy.
  No urgency — current 121h behavior is fine for the immediate need.
- Migration of `logs/` is bucketed under `logs/imported_<date>/`
  alongside per-run dirs. If existing log-parsing code expects flat
  `logs/`, will need a small compat shim — but nothing observed yet
  to suggest the migration would break anything.

### Pickup checklist for next session

1. `cd` into the project, confirm tests still green:
   `python tests/smoke_kalman_pose_smoother_v2.py && \
    PYTHONPATH=. python tests/smoke_project_layout.py && \
    PYTHONPATH=. python tests/smoke_migrate_project.py`
2. Decide between the layout-consumer work (item 1 above) vs. the
   smoother polish work (live progress, cancel) vs. continuing
   real-data tuning of the 121d/e behavior.
3. If continuing real-data tuning: run with the recommended config
   above, compare against the 121bc baseline you have. q_jerk values
   are now in the saved model; check whether EM is adapting them
   reasonably or hitting caps. If hitting caps, the cap values may
   need rethinking.
