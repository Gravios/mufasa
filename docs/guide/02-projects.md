# 2. Projects

A project is a folder holding one experiment: its videos, the tracker's pose files, everything
computed from them, and the models. Its settings are in `project.toml` at the top of the folder.

## Create a project

**File › New project…** (or **Projects › Create or open project**, when no project is open) opens
the new-project form:

| Field | What to put |
|---|---|
| Project parent directory | the folder the project's folder goes in |
| Project name | the project's folder name, e.g. `K7_gait_2026` |
| Body-part preset | a standard marker set, or… |
| Or auto-detect | *Auto-detect from pose file…*: pick one of your tracker's files, and its body parts (and skeleton, if the file carries one) are read from it. This is usually the easiest. It sets one animal; multi-animal projects choose a preset or edit the project after. |
| Animal count | how many animals are tracked in each video |
| Classifier names (optional) | the behaviours you'll score, one per line. They can be added later (**Manage classifiers**). |
| Workflow file type | `parquet` (recommended: smaller and faster) or `csv` |

*Create project* writes the folder and `project.toml` and opens it.

## Open a project

* **File › Open project…** and pick its `project.toml`.
* **Projects › Create or open project** has *Open project…* and *Open most recent: …*, which
  reopens the last one.
* From a terminal, `mufasa` in a project's folder (or any folder inside it) opens that project.
  `mufasa --project PATH` opens a given one.

## Project information

**Projects › Project information** summarises the open project:

* its name, location and format;
* its animals, file type and markers;
* its behaviours;
* how many source and derived files it has.

*Edit…* changes the metadata: the file type, body parts, number of animals and their names, and
the behaviours. Its *Auto-detect from pose file* button re-reads the body parts from a pose file,
for when the project was created with one marker set and the tracker's files use another.
*Refresh* re-reads the counts.

`mufasa project` in a terminal prints the same summary ([chapter 13](13-command-line.md)).

## What a project holds

```text
K7_gait_2026/
├── project.toml          the project's settings: animals, body parts, skeleton, behaviours, options
├── sources/              what you brought in; Mufasa never changes it
│   ├── videos/           the videos (usually links to the originals)
│   ├── pose/             the tracker's files, as imported
│   ├── pose3d/           imported motion capture (C3D)
│   ├── calibration/      camera calibrations
│   ├── bodymodel/        rigged body models (*.rig.npz)
│   ├── annotations/      third-party annotations (optional)
│   └── video_info.csv    each video's frame rate, size and pixels per mm
├── derived/              everything computed; safe to delete and compute again
│   ├── pose/<run>/                imports
│   ├── interpolated/<run>/        interpolated pose
│   ├── smoothed/kalman_v2/<run>/  Kalman v2 smoothed pose
│   ├── outlier_corrected/<run>/   the pose the later stages read
│   ├── pose3d/<run>/              3-D pose, with gait and dynamics files beside it
│   ├── features/<family>/<video>.parquet
│   ├── labels/<video>.parquet           your behaviour labels
│   ├── classifications/<video>.parquet  the classifiers' predictions
│   └── plots/ · frames/ · …             plots, videos and frames you made
├── models/               Kalman models (<name>/model.npz + card.toml), classifiers (generated_models/)
├── logs/                 logs; logs/measures/rois.json holds the ROIs
└── backups/              snapshots taken before changes in place (e.g. renaming markers)
```

### Runs

Each step that computes something writes a new **run**: a folder named by the time it ran, such as
`derived/smoothed/kalman_v2/20261124-071559-5a0cca/`. Earlier runs are kept, so you can go back to
an earlier result, and the newest run sorts last. A later step reads the newest run of the step
before it unless you choose another.

* Smoothing and interpolation runs are also *published* to `derived/outlier_corrected/`. That is
  where features and the other later stages read from.
* `mufasa list runs` lists each stage's runs.
* Deleting old runs is safe; delete the folder.

### Moving a project

A project can be moved or opened over a network mount (sshfs, NFS). Imported videos are links,
and Mufasa makes them relative when it opens the project, so they survive the move. To do this by
hand, run `mufasa-relink PROJECT`.

For the layout in full (each folder, the run-id format, `project.toml`'s settings), see
[`../v1_project_layout.md`](../v1_project_layout.md).
