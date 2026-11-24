# 13. The command line

Much of Mufasa runs without the workbench: to see what a project holds, to check a machine, or to
run the 3-D pipeline on a server. This chapter is a summary. [`../cli.md`](../cli.md) has every
command in full, with example output, scripting with `--json`, and recipes.

## `mufasa` and its commands

```text
mufasa                       open the workbench
mufasa help [command]        the commands; a command's options
mufasa list <what>           models, videos, pose, 3d, runs, calibrations, steps, tools, projects
mufasa info FILE…            what a file holds
mufasa project [PATH]        a project's summary
mufasa run <step> …          the 3-D pipeline
mufasa doctor                check the environment
mufasa version               mufasa's version
```

The commands use the project the current folder is in. Otherwise, pass `-p PATH`; failing both,
they use the last project opened in the workbench.

A few examples:

```bash
cd /data/K7
mufasa project                         # body parts, behaviours, and how much of each it holds
mufasa list models                     # Kalman models, classifiers, detectors, 3-D bodies, priors
mufasa list 3d                         # 3-D files, each with its video
mufasa list runs --stage pose3d        # one stage's runs
mufasa info derived/pose3d/<run>/K7_camDLC_3d.parquet
mufasa doctor                          # GPU, FFmpeg, packages
```

### `mufasa doctor`

`mufasa doctor` checks the machine, one line each:

* Python, the packages (and their versions), PyTorch's CUDA and GPUs;
* the GPUs `nvidia-smi` lists;
* FFmpeg and NVENC;
* the display.

A ✗ marks a required package that is missing. Run it first on a new machine, and include its
output when you report a problem.

### `mufasa run`

The 3-D pipeline, one step at a time or all of it:

```bash
mufasa list steps
mufasa run calibrate box marks.json --length 400 --width 300 --wall-height 250 --camera-height 500 \
    --wall-tilt auto --out sources/calibration/camera_K7.toml
mufasa run all --pose sources/pose/K7_camDLC.parquet --camera sources/calibration/camera_K7.toml --mass-g 410
mufasa run all ... --dry-run           # say what would run
```

`mufasa run all` chains 3d-inference → heights → gait-extrapolation → dynamics, and skips what is up
to date. See [chapter 5](05-3d-pose.md) and [chapter 6](06-body-model-and-dynamics.md).

## The other tools

| Command | What it does |
|---|---|
| `mufasa-workbench` | the workbench, without the environment check |
| `mufasa-preview` | plays a video with its raw and smoothed markers, matching files by session name (`--video`, or `--folder` to search a folder) |
| `mufasa-landscape` | a map of a project: joint PDFs of feature pairs with several modes, and t-SNE over a frame sample (needs the `analysis` extra) |
| `mufasa-landscape-animate` | a GIF or MP4 of frames moving between the t-SNE map and feature pairs, for presentations |
| `mufasa-pose3d` | the 3-D commands (`mufasa run` calls them by pipeline names): `calibrate-box`, `calibrate-checkerboard`, `infer`, `smooth`, `heights`, `gait`, `body-rig`, `body-fit`, `dynamics`, `animal`, `import-c3d`, `render-animation`, `gpu-check` |
| `mufasa-view3d` | the 3-D viewer: `mufasa-view3d --pose X_3d.parquet` (finds its video), `mufasa-view3d V.mp4` (finds its 3-D file), `--model virtual_rodent` to add a model |
| `mufasa-relink` | makes a project's video links relative, so it survives being moved or mounted elsewhere |

Each takes `--help`. `mufasa list tools` lists them.

The Kalman v2 smoother also runs on its own, on CSV or parquet pose files:

```bash
python -m mufasa.data_processors.kalman_pose_smoother_v2 session.parquet --beside-input
```
