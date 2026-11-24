# 1. Getting started

## Install

Mufasa runs on Linux with Python 3.11 or later.

```bash
git clone https://github.com/Gravios/mufasa.git
cd mufasa
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[qt]"
```

The `qt` extra installs PySide6, which the workbench needs. Mufasa compiles a few native kernels
when it installs, so a C compiler must be present. A conda environment works the same way:
`conda create -n mufasa python=3.12`, activate it, then `pip install -e ".[qt]"`.

Optional extras add features. Install several at once, for example `pip install -e ".[qt,pose3d,biomech]"`:

| Extra | Adds | Needed for |
|---|---|---|
| `pose3d` | PyTorch 2.7 or later | the fast 3-D solvers (CPU and GPU), the 3-D viewer's posing |
| `biomech` | MuJoCo 3.2 or later | joint torques, ground and muscle forces (`dynamics`) |
| `gpu` | CuPy for CUDA 12, PyTorch | GPU acceleration elsewhere |
| `onnx` | ONNX export and onnxruntime | exporting classifiers to ONNX |
| `analysis` | `diptest` | `mufasa-landscape` |
| `dev` | pytest, ruff, mypy, pre-commit | working on Mufasa itself |
| `all` | everything above | |

For video encoding on the GPU, FFmpeg must be built with NVENC; Mufasa uses the system's `ffmpeg`.

**GPUs.** PyTorch 2.7 is the first release with kernels for Blackwell GPUs (RTX 50xx, RTX PRO
6000). On Linux, PyTorch 2.11 and later from PyPI are CUDA 13 builds, which need a driver whose
`nvidia-smi` shows CUDA 13. With an older driver, install PyTorch from
`https://download.pytorch.org/whl/cu128` first. `mufasa-pose3d gpu-check` says what is missing.

Check the installation:

```bash
mufasa doctor
```

Every line should be unmarked; a ✗ marks a required package that is missing. On a GPU machine,
the `torch` line should list your GPU. See [chapter 13](13-command-line.md#mufasa-doctor).

## Open the workbench

```bash
mufasa                         # the project the current folder is in, else the last one opened
mufasa --project /data/K7      # this project
mufasa --no-auto-discover      # ignore the current folder
mufasa --no-recent             # don't fall back to the last one opened
```

`mufasa` checks the environment first (OpenCV, FFmpeg, the display), then opens the workbench.
`mufasa-workbench` opens it without the checks.

## Finding your way round the workbench

```text
┌──────────────┬─────────────────────────────────────────────────────────────┐
│ File Tools   │                                                             │
│ Help         │   Preprocessing                                             │
├──────────────┤   ▸ Preprocess Videos                                       │
│ Projects     │   ▸ Video Calibration                          ✓            │
│ Data Import  │   ▾ Kalman v2 smoothing                        ○            │
│ Data insp.   │       Device: [cuda ▾]   Frame rate: [30]                   │
│ Model mod.   │       …                                                     │
│ Preprocessing│       [ Run ]                                               │
│ 3-D          │   ▸ Run outlier correction                                  │
│ Body model   │   …                                                         │
│ ROI          │                                                             │
│ …            │                                                             │
├──────────────┴─────────────────────────────────────────────────────────────┤
│ /data/K7/project.toml — CPU 32 · CUDA ✓ · NVENC ✓ · x11                    │
└────────────────────────────────────────────────────────────────────────────┘
```

* **The sidebar** lists the pages in pipeline order: Projects, Data Import, Data inspector, Model
  modifications, Preprocessing, 3-D, Body model, ROI, Manage classifiers, Features, Annotation,
  Train, Validate, Inference, YOLO pose, Analysis, Visualizations, Video Processing, Add-ons, Tools.
* **Each page** is a column of sections. Click a section's title to open or close it. Most
  sections are a form with a **Run** button. The work runs in the background with a progress bar
  and can be cancelled, so the window stays usable.
* **Status marks** beside a section's title: ✓ means it has run on the current data; an orange ○
  means something it depends on has run again since, so its output is out of date.
* **Pop out ⇱** on some larger forms (Preprocess Videos, Train, Validate, Inference) opens the
  form in a window of its own.
* **The viewers** (the feature viewer, the 3-D viewer) open in windows of their own, so the main
  window stays as it is.
* **The status bar** shows the open project and what this machine has: CPU cores, CUDA, NVENC and
  the display server.
* **File** › *New project…*, *Open project…*, *Reconfigure project from DLC file…*, *Quit*.
  **Tools** has quick video utilities: *Reverse video…*, *Crossfade two videos…*, *Print video
  metadata…*, *Check video seekable…*, *Calibrate pixels/mm…*, *Change video playback speed…*.

## Your first project in ten steps

This is the shortest route from tracker output to behaviour statistics. Each step links to its
chapter.

1. **Create the project.** **File › New project…**: choose a parent folder and a name. Click
   *Auto-detect from pose file…* and pick one of your tracker's files, so the body parts come from
   it. Set the animal count and the behaviours you will score (one per line, for example `Rear`
   and `Groom`), then click *Create project*. → [chapter 2](02-projects.md)
2. **Import the pose files.** **Data Import › Import pose data**: choose the tracker's format and
   the folder. → [chapter 3](03-importing-data.md)
3. **Import the videos.** **Data Import › Import video**: a folder of videos. By default they are
   linked, not copied. → [chapter 3](03-importing-data.md#import-video)
4. **Calibrate pixels per millimetre.** **Preprocessing › Video Calibration**: measure a known
   distance in one video and apply it to all. Without this, distances come out in pixels.
   → [chapter 4](04-preprocessing.md#video-calibration)
5. **Smooth the pose.** **Preprocessing › Kalman v2 smoothing**: run it with the defaults.
   → [chapter 4](04-preprocessing.md#kalman-v2-smoothing)
6. **Compute features.** **Features › Compute Features**: choose the families (distances, angles,
   movements, …). → [chapter 8](08-features.md)
7. **Label behaviour.** **Annotation › Frame Labeling**: label frames of a few videos for each
   behaviour. → [chapter 9](09-classifiers.md#annotation)
8. **Train.** **Train**: train a classifier per behaviour. → [chapter 9](09-classifiers.md#train)
9. **Run it on every video.** **Inference**. → [chapter 9](09-classifiers.md#inference)
10. **Get the numbers.** **Analysis › Run analysis** › *Classifier descriptive statistics* (bout
    counts, durations), and **Visualizations** for Gantt charts and videos.
    → [chapter 11](11-analysis-and-visualizations.md)

[`../workflow_recipes.md`](../workflow_recipes.md) has variants of this for other setups: two
animals, ROI-bounded behaviour, cue lights, mazes, pup retrieval, third-party annotations and
unsupervised discovery.

## Working without a display

The command line needs no display, so it works over SSH:

```bash
ssh lab-server
cd /data/K7 && mufasa project              # what is in the project
mufasa run all --pose sources/pose/K7DLC.parquet --camera sources/calibration/camera.toml
```

See [chapter 13](13-command-line.md).
