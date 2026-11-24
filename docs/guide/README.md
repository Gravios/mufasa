# Mufasa user guide

This guide explains what Mufasa does and how to use it, page by page, in the order an experiment
goes through them. The last part is for people who change Mufasa's code.

Mufasa takes the output of a pose tracker (DeepLabCut, SLEAP, YOLO, …) and the videos it came
from, and turns them into:

* clean pose;
* 3-D pose;
* gait and forces;
* features;
* behaviour labels;
* statistics, plots and videos.

It runs as a desktop workbench (`mufasa`), and many steps also run from the command line.

## How the pipeline fits together

```text
 tracker output + videos
        │
        ▼
 Projects ─► Data Import ─► Preprocessing ─────────────► Features ─► Annotation ─► Train ─► Validate ─► Inference
 (project)   (pose, video)  (calibrate, interpolate,       │             ▲                                 │
                             Kalman v2 smoothing,          │        Manage classifiers                     ▼
                             outlier correction)           │                                   Analysis · Visualizations
                                   │                       │
                                   ▼                       │
                          3-D ─► Body model / dynamics ────┘   (3-D features, gait, forces)
                   (camera, 3-D pose, gait, viewer)

 Side tracks: ROI (regions of interest) · YOLO pose (train your own tracker)
              Video Processing · Add-ons (task-specific analyses) · Tools (format conversion)
```

You don't need every step. A typical 2-D behaviour project goes Projects → Data Import →
Preprocessing → Features → Annotation → Train → Inference → Analysis. A gait study goes Projects →
Data Import → Preprocessing → 3-D → dynamics.

## The guide

| Chapter | Workbench page(s) | What it covers |
|---|---|---|
| [1. Getting started](01-getting-started.md) | — | Install, open the workbench, find your way round it, your first project in ten steps |
| [2. Projects](02-projects.md) | Projects | Create and open projects, what a project holds on disk, runs |
| [3. Importing data](03-importing-data.md) | Data Import · Data inspector · Model modifications | Pose files from every supported tracker, videos, adding sessions, renaming markers |
| [4. Preprocessing](04-preprocessing.md) | Preprocessing | Video preprocessing and calibration (pixels per mm), interpolation, Kalman v2 smoothing, outlier correction, egocentric alignment |
| [5. 3-D pose](05-3d-pose.md) | 3-D | Camera calibration from the cage, 3-D from one or two cameras, 3-D smoothing, motion capture, gait, the 3-D viewer |
| [6. Body model and dynamics](06-body-model-and-dynamics.md) | Body model · `mufasa run` | Fitting a rigged body to video, joint torques, ground and muscle forces, per-animal calibration |
| [7. Regions of interest](07-roi.md) | ROI | Drawing ROIs, time in and entries into them, ROI features |
| [8. Features](08-features.md) | Features | The feature families, computing them, looking at them (distributions, joint plots, the feature viewer) |
| [9. Behaviour classifiers](09-classifiers.md) | Manage classifiers · Annotation · Train · Validate · Inference | Labelling frames, training random-forest classifiers, checking and running them |
| [10. YOLO pose](10-yolo-pose.md) | YOLO pose | Training and running your own YOLO pose model |
| [11. Analysis and visualizations](11-analysis-and-visualizations.md) | Analysis · Visualizations | Every statistic and every plot or video Mufasa can make |
| [12. Video processing, add-ons and tools](12-video-addons-tools.md) | Video Processing · Add-ons · Tools | Video editing, task-specific analyses (cue lights, alternation, pup retrieval), format conversion |
| [13. The command line](13-command-line.md) | — | `mufasa help`, `list`, `info`, `project`, `doctor`, `run`, and the other tools |
| [14. Troubleshooting](14-troubleshooting.md) | — | Common problems and what to do about them |

For developers:

| Chapter | What it covers |
|---|---|
| [D1. Architecture](developers/architecture.md) | The package's layout, how data moves through it, the project layout API, runs and provenance |
| [D2. Extending Mufasa](developers/extending.md) | Adding a workbench form, a page, a feature family, a command-line command or a pipeline step |
| [D3. Testing and patches](developers/testing-and-patches.md) | The smoke tests, running them, lint rules, invariants, how changes are made and delivered |

## Deeper references

The guide links to these where they apply:

| Document | Topic |
|---|---|
| [`../v1_project_layout.md`](../v1_project_layout.md) | The project layout in full |
| [`../data_source_guides.md`](../data_source_guides.md) | Each tracker's import |
| [`../workflow_recipes.md`](../workflow_recipes.md) | End-to-end recipes for common experiments |
| [`../pose3d.md`](../pose3d.md) | 3-D pose in depth |
| [`../bodymodel.md`](../bodymodel.md) | The body model in depth |
| [`../biomech.md`](../biomech.md) | Dynamics, muscles and optimisation in depth |
| [`../cli.md`](../cli.md) | Every command-line command, with examples |
| [`../kalman_smoother_design.md`](../kalman_smoother_design.md) | How the Kalman v2 smoother works |

## Conventions in this guide

* **Page › Section** names a place in the workbench: for example **Preprocessing › Kalman v2
  smoothing** is the *Kalman v2 smoothing* section of the *Preprocessing* page.
* `code` is a file, folder, command or option.
* Paths such as `derived/features/` are relative to the project's folder.
* Numbers in examples (frame rates, masses, sizes) are examples, not recommendations, unless the
  text says so.
