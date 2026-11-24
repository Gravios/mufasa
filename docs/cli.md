# The `mufasa` command line

`mufasa` on its own opens the workbench. Followed by a command, it works in the terminal and opens
no window. That makes it the quickest way to see what a project holds, to check a new machine, or
to run the 3-D pipeline over SSH or from a script.

```text
mufasa help [command]        the commands; a command's options
mufasa list <what>           models, videos, pose, 3d, runs, calibrations, steps, tools, projects
mufasa info FILE…            what a file holds
mufasa project [PATH]        a project's summary
mufasa run <step> …          the 3-D pipeline, a step at a time or all of it
mufasa doctor                check the environment
mufasa version               mufasa's version
```

* `mufasa --help` is the same as `mufasa help`, and `mufasa --version` the same as `mufasa version`.
* A misspelt command is refused, rather than opening the workbench:

```text
$ mufasa lsit models
mufasa: no command 'lsit' (`mufasa help` lists them)
```

Contents:

* [Quick start](#quick-start)
* [Which project a command uses](#which-project-a-command-uses)
* [`mufasa help`](#mufasa-help)
* [`mufasa list`](#mufasa-list): [models](#mufasa-list-models), [videos](#mufasa-list-videos),
  [pose](#mufasa-list-pose), [3d](#mufasa-list-3d), [runs](#mufasa-list-runs),
  [calibrations](#mufasa-list-calibrations), [steps, tools, projects](#mufasa-list-steps-tools-projects)
* [`mufasa info`](#mufasa-info)
* [`mufasa project`](#mufasa-project)
* [`mufasa run`](#mufasa-run)
* [`mufasa doctor`](#mufasa-doctor)
* [Using the output in scripts (`--json`)](#using-the-output-in-scripts---json)
* [Recipes](#recipes)
* [Exit codes](#exit-codes)
* [Troubleshooting](#troubleshooting)

The examples below come from a small sample project at `/data/K7_project`: one video, its 2-D
labels, a 3-D run, a camera, a few models. Your dates, sizes and paths will differ.

## Quick start

```bash
cd /data/K7_project             # or any folder inside the project
mufasa project                  # what is in it
mufasa list models              # which models it can use
mufasa list 3d                  # its 3-D files, each with its video
mufasa info sources/videos/rat6/K7_cam.mp4
mufasa doctor                   # is this machine set up (GPU, FFmpeg, packages)?
```

## Which project a command uses

Commands that look into a project (`list`, `project`, and `doctor`'s last line) use the first of
these that applies:

1. `--project` (or `-p`): the project's `project.toml`, an old-layout `.ini` config, or any folder
   inside the project;
2. the project the current folder is in, looking up to six folders up;
3. the last project opened in the workbench.

Each list's first line says which project it used and how it was found:

```text
Project: /data/K7_project (the current folder's)
Project: /data/K7_project (given)
Project: /data/K7_project (the last one opened)
```

With no project, the command says how to name one and exits with code 2:

```text
$ cd / && mufasa list videos
mufasa: no project: pass --project, or run this inside a project's folder
```

These all give the same list:

```bash
cd /data/K7_project/sources/pose && mufasa list videos
mufasa list videos --project /data/K7_project
mufasa list videos -p /data/K7_project/project.toml
```

## `mufasa help`

`mufasa help` lists the commands, what `mufasa list` can list, and mufasa's other tools.
`mufasa help <command>` shows that command's options:

```text
$ mufasa help list
usage: mufasa list [-h] [-p PROJECT] [--json]
                   [--kind {smoothing,classifier,detector,body,prior}]
                   [--project-only] [--details] [--stage STAGE]
                   what
…
```

* `mufasa help run` lists the pipeline steps.
* `mufasa help pose3d` (or `mufasa help mufasa-pose3d`) points to another tool and its `--help`.

## `mufasa list`

`mufasa list <what>` lists one kind of thing. Every list takes `--project` and `--json`.

| what | lists | options |
|---|---|---|
| `models` | every model the project can use | `--kind`, `--project-only` |
| `videos` | the project's videos, in subfolders too | `--details` |
| `pose` | the 2-D pose files in `sources/pose` | |
| `3d` | the 3-D files, each with its video | |
| `runs` | the derived runs, by stage | `--stage` |
| `calibrations` | the cameras in `sources/calibration` | |
| `steps` | the steps of `mufasa run` | |
| `tools` | mufasa's command-line tools | |
| `projects` | the recently opened projects | |

### `mufasa list models`

This lists every model the project can use, grouped by kind. **where** is `project` (in the
project's folder), `global` (your own library, shared by all your projects) or `built in`
(shipped with mufasa).

```text
$ mufasa list models
Project: /data/K7_project (the current folder's)

Kalman body models (smoothing):
  name         where    what                                                       modified          path
  -----------  -------  ---------------------------------------------------------  ----------------  ---------------------------------------
  rat_body     project  Kalman body model v2, fitted at 30 fps; from rat_body.npz  2026-11-24 07:15  models/rat_body/model.npz
  lab_default  global   Kalman body model v2, fitted at 60 fps                     2026-11-24 07:15  ~/.config/mufasa/models/lab_default.npz

Behaviour classifiers:
  name  where    what  modified          path
  ----  -------  ----  ----------------  --------------------------------
  rear  project  sav   2026-11-24 07:15  models/generated_models/rear.sav

Detectors (YOLO):
  name      where    what     modified          path
  --------  -------  -------  ----------------  ------------------
  yolo_rat  project  YOLO pt  2026-11-24 07:15  models/yolo_rat.pt

3-D body models:
  name            where     what                          modified          path
  --------------  --------  ----------------------------  ----------------  -----------------------------------------------------
  K7              project   calibrated animal: K7, 410 g  2026-11-24 07:15  derived/biomech/K7.animal.json
  rat             project   38 muscles                    2026-11-24 07:15  derived/biomech/rat.muscles.json
  virtual_rodent  built in  the virtual rodent (MuJoCo)   2026-11-19 21:04  mufasa/assets/models/pose3d/virtual_rodent/rodent.xml

Height priors:
  name               where     what                                   modified          path
  -----------------  --------  -------------------------------------  ----------------  -------------------------------------------------
  rat7m_trunk_prior  built in  height prior: 80000 frames, 5 animals  2026-11-12 20:06  mufasa/assets/models/pose3d/rat7m_trunk_prior.npz
```

The kinds, and where each is looked for:

| kind (`--kind`) | what it is | where |
|---|---|---|
| `smoothing` | Kalman body models: the body's segment lengths and noise, used by the pose smoother. A project's model shows the file it was trained as (from its `card.toml`) | `models/<name>/model.npz`; the global library `~/.config/mufasa/models/*.npz` |
| `classifier` | behaviour classifiers | `.sav`, `.onnx`, `.pkl`, `.joblib` under `models/` |
| `detector` | YOLO detectors and pose models | `.pt`, `.engine` under `models/` |
| `body` | 3-D bodies: rigs (`*.rig.npz`, from the Body model tab), calibrated animals (`*.animal.json`, from `mufasa run animal`, with name and mass), muscle sets (`*.muscles.json`, from `dynamics --save-muscles`), and the built-in virtual rodent | anywhere in the project, except extracted frames |
| `prior` | the height prior used for single-camera 3-D (built from Rat7M) | mufasa's own files |

Options:

```bash
mufasa list models --kind smoothing     # one kind only
mufasa list models --project-only       # leave out your global library
mufasa list models --json               # for scripts (below)
```

The workbench puts a newly trained Kalman model in both places: the project's `models/<name>/` and
your global library. This lets a model trained in one project be found from another.

### `mufasa list videos`

This lists the project's videos, including those in subfolders of `sources/videos` (old projects:
`videos/`). `--details` opens each one for its frames, frame rate, size and length; on a large
project that takes a moment.

```text
$ mufasa list videos --details
Project: /data/K7_project (the current folder's)

1 video(s):
  video                           size    modified          frames  fps   pixels  length
  ------------------------------  ------  ----------------  ------  ----  ------  --------
  sources/videos/rat6/K7_cam.mp4  3.1 kB  2026-11-24 07:15  45      30.0  64x48   00:00:01
```

### `mufasa list pose`

This lists the 2-D pose files in `sources/pose` (DeepLabCut, SLEAP and the like) with their frame
counts. 3-D files written beside them are left out; see `list 3d`.

```text
$ mufasa list pose
1 2-D pose file(s):
  file                            frames  modified
  ------------------------------  ------  ----------------
  sources/pose/K7_camDLC.parquet  45      2026-11-24 07:15
```

### `mufasa list 3d`

This lists the 3-D pose files, newest run first:

* the 3-D runs (`derived/pose3d/<run>/`);
* imported motion capture (`sources/pose3d/`);
* those `mufasa run` wrote beside the 2-D labels.

Each is shown with the video it was made from, found by name the way the 3-D viewer finds it. The
gait, dynamics and joint files written beside a 3-D file are left out.

```text
$ mufasa list 3d
1 3-D file(s), newest run first:
  file                                                        frames  video       modified
  ----------------------------------------------------------  ------  ----------  ----------------
  derived/pose3d/20261124-071559-0b565c/K7_camDLC_3d.parquet  45      K7_cam.mp4  2026-11-24 07:15
```

`(none found)` in the video column means no video's name begins the 3-D file's name. The viewer's
**Video…** button then picks one by hand.

### `mufasa list runs`

Each processing stage (smoothing, features, 3-D, …) writes into a new run folder,
`derived/<stage>/<run id>/`, so earlier results are kept. `list runs` gives each stage's count and
latest run:

```text
$ mufasa list runs
Runs by stage (`mufasa list runs --stage S` for one stage's):
  stage               runs  latest                  files in latest
  ------------------  ----  ----------------------  ---------------
  pose3d              1     20261124-071559-0b565c  2
  smoothed/kalman_v2  1     20261124-071559-5a0cca  1
```

`--stage` lists one stage's runs, oldest first:

```text
$ mufasa list runs --stage pose3d
  stage   run                     files  path
  ------  ----------------------  -----  -------------------------------------
  pose3d  20261124-071559-0b565c  2      derived/pose3d/20261124-071559-0b565c
```

### `mufasa list calibrations`

This lists the cameras in `sources/calibration`, each in a line: the image size, the focal length,
the height above the floor, and the box with its wall flare (if it was calibrated from box marks).

```text
$ mufasa list calibrations
1 camera(s):
  camera  what                                                                                  modified          path
  ------  ------------------------------------------------------------------------------------  ----------------  -------------------------------
  camera  64x48 px, focal 50 px, 500 mm above the floor, box 400 × 300 mm, walls flaring 12.0°  2026-11-24 07:15  sources/calibration/camera.toml
```

A wrong height or box size here is the first thing to check when 3-D heights look off.

### `mufasa list steps`, `tools`, `projects`

* `mufasa list steps`: the steps of `mufasa run` and what each does.
* `mufasa list tools`: mufasa's command-line tools (`mufasa-pose3d`, `mufasa-view3d`,
  `mufasa-preview`, …). Each takes `--help`.
* `mufasa list projects`: the projects recently opened in the workbench, marking any that have
  moved or gone.

These three need no project.

## `mufasa info`

`mufasa info FILE…` says what a file holds. Give it one file or several.

A 3-D pose file shows its labels, units, frame rate, the box, and the video it was made from:

```text
$ mufasa info derived/pose3d/20261124-071559-0b565c/K7_camDLC_3d.parquet
file      /data/K7_project/derived/pose3d/20261124-071559-0b565c/K7_camDLC_3d.parquet
size      5.0 kB
modified  2026-11-24 07:15
rows      45
columns   7
kind      3-D pose
units     mm
fps       30.0
labels    2: nose, tail
box (mm)  x [-200, 200], y [-150, 150], z [0, 300], walls flaring 12.0°
video     /data/K7_project/sources/videos/rat6/K7_cam.mp4
```

A video shows its frames, frame rate, size and length:

```text
$ mufasa info sources/videos/rat6/K7_cam.mp4
kind       video
frames     45
fps        30.0
size (px)  64x48
length     00:00:01
```

A model shows what it is and where it came from:

```text
$ mufasa info models/rat_body/model.npz
kind         model
holds        Kalman body model v2, fitted at 30 fps
source path  /data/train/rat_body.npz
```

A camera, or a calibrated animal:

```text
$ mufasa info sources/calibration/camera.toml
kind      camera
camera    64x48 px, focal 50 px, 500 mm above the floor, box 400 × 300 mm, walls flaring 12.0°

$ mufasa info derived/biomech/K7.animal.json
kind      calibrated animal
name      K7
mass_g    410
```

(The `file`, `size` and `modified` lines are left out of the last four examples.)

| file | shows |
|---|---|
| video (`.mp4`, `.avi`, `.mov`, …) | frames, fps, pixels, length |
| 3-D pose (`*_3d*.parquet`, motion capture) | rows, labels, units, fps, the box and its flare, the camera, the 2-D labels it came from, its video |
| gait (`*_gait.parquet`), dynamics (`*_dynamics.parquet`) | its columns |
| other `.parquet` | rows and columns |
| model (`.npz`) | its kind (Kalman body model, height prior, arrays) and its `card.toml` |
| camera (`.toml`) | image size, focal length, height, box |
| animal, muscle set, camera path (`.json`) | its fields; lists and tables as counts |
| `project.toml` | as `mufasa project` |

`--json` gives the same fields as JSON. A missing file is reported and the exit code is 1; the
other files are still described.

## `mufasa project`

`mufasa project [PATH]` gives a project's summary: where it is, its animals, body parts and
behaviours, and how much of each thing it holds.

```text
$ mufasa project
project         /data/K7_project/project.toml (the current folder's)
root            /data/K7_project
layout          project.toml
animals         1 (Animal_1)
body parts      2: nose, tail
behaviours      rear, groom
import format   parquet
videos          1
2-D pose files  1
3-D files       1
cameras         1
models          1 smoothing, 1 classifier, 1 detector, 2 body
runs            1 pose3d, 1 smoothed
```

The model count covers the project's own models; the built-in ones and your global library are in
`mufasa list models`. `--json` gives the same as JSON.

## `mufasa run`

`mufasa run` runs the 3-D pipeline from the terminal. Each step is a `mufasa-pose3d` command under
a pipeline name, with the same options (`mufasa run <step> --help`). docs/pose3d.md explains what
each step does; this section shows how to drive them.

```text
$ mufasa list steps
  step                what
  calibrate           the camera: `calibrate box` (marked box edges) or `calibrate checkerboard` (lens)
  3d-inference        3-D pose from 2-D labels and the camera, Kalman-smoothed
  smooth              3-D Kalman smoothing of a 3-D pose file
  heights             labels within the body's height range and above the floor
  gait-extrapolation  paws, steps and loads from the body's motion
  body-fit            a rigged body fitted to a top-down video
  dynamics            joint torques, ground forces, muscle forces (--muscles), re-solved motion (--optimise)
  animal              calibrate the body model to an animal once, from its sessions (--animal for the rest)
```

**One step at a time.** First calibrate the camera from the box edges marked in the 3-D tab. The
box is measured at the floor, and the walls' flare is fitted from the marked corner edges:

```bash
mufasa run calibrate box sources/calibration/camera.marks.json \
    --length 400 --width 300 --wall-height 250 --camera-height 500 \
    --wall-tilt auto --out sources/calibration/camera.toml
```

Then run 3-D pose, its heights, and the gait:

```bash
mufasa run 3d-inference --pose sources/pose/K7_camDLC.parquet \
    --camera sources/calibration/camera.toml --out K7_3d.parquet
mufasa run heights K7_3d.parquet --out K7_3d_heights.parquet
mufasa run gait-extrapolation K7_3d_heights.parquet --mass-g 410 --out K7_gait.parquet
```

Then the forces. `--muscles` adds muscle forces, and `--optimise` re-solves the motion:

```bash
mufasa run dynamics K7_3d_heights.parquet --gait K7_gait.parquet --mass-g 410 \
    --muscles rat_hindlimb.osim --optimise
```

**All of it.** `mufasa run all` chains 3d-inference → heights → gait-extrapolation → dynamics for
one video. It names the files from the video, skips steps whose output is newer than their inputs,
and records each step in `<stem>_run.json`:

```bash
mufasa run all --pose sources/pose/K7_camDLC.parquet --camera sources/calibration/camera.toml \
    --mass-g 410 --out-dir derived/K7
mufasa run all ... --dry-run              # say what would run, run nothing
mufasa run all ... --from heights         # start later (e.g. after editing the 3-D file)
mufasa run all ... --to gait-extrapolation
mufasa run all ... --force                # rerun what is up to date
mufasa run all --pose3d K7_3d.parquet     # start from a 3-D file
```

**One animal, several sessions.** Calibrate the body model to the animal once, from a few of its
sessions, then pass it to the rest:

```bash
mufasa run animal derived/K7/*_3d_heights.parquet --name K7 --mass-g 410 --out K7.animal.json
mufasa run all --pose sources/pose/K7_day2DLC.parquet --camera sources/calibration/camera.toml \
    --animal K7.animal.json
```

`mufasa list models --kind body` then shows the calibrated animal.

## `mufasa doctor`

`mufasa doctor` checks the machine mufasa runs on, one line each:

* mufasa's version, Python, the system and the active conda environment;
* the packages, with their versions:
  * numpy, pandas, pyarrow, scipy, scikit-learn, OpenCV, PySide6;
  * PyTorch, with its CUDA version and the GPUs it sees;
  * MuJoCo, numba, ultralytics, DeepLabCut;
* the GPUs `nvidia-smi` lists;
* FFmpeg, and whether it can encode on the GPU (NVENC);
* the display, and the Qt platform the workbench will use;
* the project `mufasa` would open from here.

```text
$ mufasa doctor
  mufasa       1.4.0
  python       3.12.7 (/home/you/miniforge3/envs/mufasa/bin/python)
  system       Linux 6.8.0
  conda env    mufasa
  numpy        2.1.3
  …
  PySide6      6.8.1
  torch        2.5.1, CUDA 12.4: NVIDIA RTX PRO 6000 Blackwell
  mujoco       3.2.6
  ultralytics  not installed
  GPUs         GPU 0: NVIDIA RTX PRO 6000 Blackwell (UUID: …)
  ffmpeg       /usr/bin/ffmpeg, NVENC
  display      X11; Qt platform xcb
  project      /data/K7_project/project.toml (the current folder's)
```

(This shows the format; the versions are illustrative.)

* If a required package (numpy, pandas, pyarrow, scipy, OpenCV, PySide6) is missing or fails to
  load, its line is marked ✗ and the exit code is 1.
* Packages for one feature only (PyTorch for the GPU, MuJoCo for dynamics, ultralytics for YOLO)
  may be "not installed" without failing.
* `torch … no CUDA` on a machine with a GPU means PyTorch was installed without CUDA. The 3-D
  solver then runs on the CPU.

`--json` gives the checks as JSON.

## Using the output in scripts (`--json`)

Every `list`, `info`, `project` and `doctor` takes `--json`. The output is then a JSON list (or
object) with the same fields, full paths included:

```text
$ mufasa list models --kind smoothing --json
[
  {
    "kind": "smoothing",
    "name": "rat_body",
    "path": "/data/K7_project/models/rat_body/model.npz",
    "where": "project",
    "details": "Kalman body model v2, fitted at 30 fps; from rat_body.npz",
    "size": 949,
    "modified": "2026-11-24 07:15"
  },
  …
]
```

With `jq`:

```bash
# the newest project Kalman model's path
mufasa list models --kind smoothing --project-only --json | jq -r 'sort_by(.modified) | last | .path'

# 3-D files with no video found
mufasa list 3d --json | jq -r '.[] | select(.video == null) | .path'

# total video length in frames
mufasa list videos --details --json | jq '[.[].frames] | add'
```

From Python:

```python
import json, subprocess

def mufasa(*args):
    out = subprocess.run(["mufasa", *args, "--json"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)

for f in mufasa("list", "3d", "--project", "/data/K7_project"):
    print(f["path"], "->", f["video"])
```

The same functions are also available directly, without a subprocess:

```python
from mufasa.cli.commands import collect_models, resolve_project
cfg, how = resolve_project("/data/K7_project")
classifiers = [m for m in collect_models(cfg) if m["kind"] == "classifier"]
```

## Recipes

**Set up a new workstation.** Install mufasa, then:

```bash
mufasa doctor            # everything ✓? GPUs listed? torch sees CUDA? ffmpeg has NVENC?
mufasa list tools        # the command-line tools are on the PATH
```

**Look at a project over SSH.** No display is needed for the commands:

```bash
ssh lab-server
cd /data/K7_project && mufasa project && mufasa list runs
```

**Find which model to smooth a new session with.**

```bash
mufasa list models --kind smoothing
mufasa info models/rat_body/model.npz      # fitted at which fps, trained from which file
```

**Check a 3-D file before opening the viewer.**

```bash
mufasa info derived/pose3d/<run>/K7_camDLC_3d.parquet   # labels, box, flare, and its video
```

If `video` says `(none found by name)`, the 3-D viewer won't find it either. Rename the video so
that the 3-D file's name begins with it, or use the viewer's **Video…** button.

**Run the whole pipeline over every session.**

```bash
for p in sources/pose/*DLC*.parquet; do
    mufasa run all --pose "$p" --camera sources/calibration/camera.toml \
        --animal K7.animal.json --out-dir derived/3d || echo "failed: $p"
done
mufasa list 3d
```

`mufasa run all` skips the steps that are up to date, so running the loop again after adding
sessions only processes the new ones.

**Point at a project without `cd`.** Every command that looks into a project takes `-p`:

```bash
mufasa list videos -p /data/K7_project
mufasa project /data/K7_project
```

## Exit codes

| code | meaning |
|---|---|
| 0 | done |
| 1 | `info`: a file was missing; `doctor`: a required package is missing or broken |
| other non-zero | `run`: the failing step's own code |
| 2 | no project found, an unknown command, or wrong options |

## Troubleshooting

* **`mufasa: no project`**: you are outside a project and none was opened in the workbench yet.
  `cd` into the project, or pass `-p /path/to/project`.
* **The wrong project is used**: the current folder wins over the last opened one. The first line
  of each list says which was used; pass `-p` to choose.
* **`mufasa list videos` misses videos**: only `sources/videos` (and its subfolders) is searched. A
  video elsewhere has to be imported into the project first (Data Import page, *Import video*).
* **A network drive (sshfs, NFS) shows nothing**: the mount may be stale. `ls` the project folder.
  If that hangs or is empty, remount.
* **`mufasa help` opens the workbench**: an older mufasa (before these commands) is first on the
  `PATH`, often from another conda environment. Activate the right environment; `which mufasa`
  shows the one that runs.
