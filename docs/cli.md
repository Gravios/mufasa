# The `mufasa` command line

`mufasa` on its own opens the workbench. Followed by a command, it does the job in the terminal
and opens no window, so it works over SSH too.

```text
mufasa help [command]        the commands; a command's options
mufasa list <what>           models, videos, pose, 3d, runs, calibrations, steps, tools, projects
mufasa info FILE…            what a file holds
mufasa project [PATH]        a project's summary
mufasa run <step> …          the 3-D pipeline (see docs/pose3d.md, "mufasa run")
mufasa doctor                check the environment
mufasa version               mufasa's version
```

`mufasa --help` is the same as `mufasa help`. `mufasa --version` is the same as `mufasa version`.

## Which project

Commands that look into a project use the first of these that applies:

* `--project` (`-p`): its `project.toml`, an old `project_config.ini`, or any folder inside it;
* the project the current folder is in (up to six folders up);
* the last project opened in the workbench.

Each list says which project it used and how it was found. With no project, the command says how
to name one and exits with 2.

## `mufasa list`

| what | lists |
|---|---|
| `models` | every model the project can use (below) |
| `videos` | the project's videos, in subfolders too; `--details` adds frames, fps, pixels and length (it opens each video) |
| `pose` | the 2-D pose files in `sources/pose` and their frames |
| `3d` | the 3-D files (runs, motion capture, `mufasa run`'s), each with the video it was made from; not the gait or dynamics files beside them |
| `runs` | the derived runs by stage, with the latest of each; `--stage pose3d` lists one stage's runs |
| `calibrations` | the cameras in `sources/calibration`: image size, focal length, height above the floor, the box and its flare |
| `steps` | the steps of `mufasa run` |
| `tools` | mufasa's other command-line tools |
| `projects` | the recently opened projects |

Every list takes `--json`, for scripts.

### `mufasa list models`

```text
$ mufasa list models
Project: /data/lab (the current folder's)

Kalman body models (smoothing):
  name         where    what                                                   modified          path
  rat_body     project  Kalman body model v2, fitted at 30 fps; from rat_body.npz  2026-11-20 14:02  models/rat_body/model.npz
  lab_default  global   Kalman body model v2, fitted at 60 fps                 2026-10-02 09:15  /home/…/.config/mufasa/models/lab_default.npz

Behaviour classifiers:
  rear         project  sav                                                    …                 models/generated_models/rear.sav
…
```

| kind | what | where it is looked for |
|---|---|---|
| `smoothing` | Kalman body models for the pose smoother, with the file each was trained as | `models/<name>/model.npz` (its `card.toml`); the global library `~/.config/mufasa/models/*.npz` |
| `classifier` | behaviour classifiers | `.sav`, `.onnx`, `.pkl`, `.joblib` under `models/` |
| `detector` | YOLO detectors | `.pt`, `.engine` under `models/` |
| `body` | 3-D bodies: the built-in virtual rodent, rigs (`*.rig.npz`), calibrated animals (`*.animal.json`, with name and mass), muscle sets (`*.muscles.json`) | anywhere in the project, except extracted frames |
| `prior` | the built-in height priors (Rat7M) | mufasa's assets |

* `--kind smoothing` (or another kind) lists one kind only.
* `--project-only` leaves out the global library.

## `mufasa info`

`mufasa info FILE…` says what a file holds:

* **a video**: frames, fps, pixels, length;
* **a 3-D file**: frames, labels, units, fps, the box and its flare, the camera, the 2-D labels it came from;
* **a gait or dynamics file**: its columns;
* **a model**: its kind, and its `card.toml` (where it came from, when it was copied);
* **a camera** (`.toml`): image size, focal length, height above the floor, the box;
* **an animal or muscle file**: its fields;
* **a project file**: as `mufasa project`.

`--json` gives the same fields as JSON. A missing file is reported, and the exit code is 1.

## `mufasa project`

`mufasa project [PATH]` shows a project's summary:

* where it is, and its layout;
* animals, body parts and behaviours, and the format it imports;
* how many videos, 2-D pose files, 3-D files and cameras it has;
* its models by kind;
* its runs by stage.

`--json` gives the same as JSON.

## `mufasa doctor`

`mufasa doctor` checks the environment, one line each:

* mufasa's version, Python, the system and the active conda environment;
* the packages: numpy, pandas, pyarrow, scipy, scikit-learn, OpenCV, PySide6, PyTorch (with CUDA
  and the GPUs it sees), MuJoCo, numba, ultralytics, DeepLabCut;
* the GPUs (`nvidia-smi`);
* FFmpeg, and whether it has NVENC;
* the display and the Qt platform the workbench will use;
* the project it would open.

A required package that is missing or broken is marked ✗, and the exit code is 1. `--json` gives
the checks as JSON.
