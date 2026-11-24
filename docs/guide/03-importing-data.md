# 3. Importing data

Three pages bring data into a project and check it:

* **Data Import** brings in pose files and videos.
* **Data inspector** shows what landed.
* **Model modifications** renames markers.

## Import pose data

**Data Import › Import pose data**: choose the *Pose file format* and the *Source directory* holding
the tracker's files, then **Run**. The files are copied into `sources/pose/` and recorded as an
import run under `derived/pose/`.

| Format | Notes |
|---|---|
| DLC H5 (single animal), DLC CSV (single animal) | DeepLabCut's standard output |
| FreeDLC parquet (single animal) | FreeDLC's output; carries the skeleton in its metadata |
| DLC H5 (multi-animal / maDLC) | multi-animal DeepLabCut; animals are kept apart by identity |
| SLEAP CSV, SLEAP H5, SLEAP .slp | SLEAP's exported predictions or the project file |
| SuperAnimal-TopView | DeepLabCut's SuperAnimal top-view model |
| YOLO-pose | YOLO keypoint predictions |
| MARS (two-mouse social) | MARS output |
| TRK (Animal Part Tracker) | APT's `.trk` files |
| FaceMap (face / whisker / pupil) | FaceMap keypoints |

Each tracker's quirks (file naming, likelihood columns, multi-animal identities) are in
[`../data_source_guides.md`](../data_source_guides.md).

The video and its pose file are matched by name: `K7_cam.mp4` goes with `K7_camDLC_resnet50….h5`
(the pose file's name begins with the video's). Keep the tracker's naming.

### Adding sessions later

New recordings for a project that already has smoothed data go through **Data Import › Import pose
data › Add sessions**. Choose *File…* or *Folder…*, then:

1. **Check** reports, file by file, whether each file's markers match the project's. Mismatches
   name the exact difference (a marker missing, one extra, a different order). This matters: a
   file whose markers don't match imports "successfully" but gives columns nothing later can find,
   which shows up much later as empty results.
2. **Add sessions** imports the files that match.
3. With *Smooth new sessions with the project's latest model* ticked, it also smooths just those
   files with the Kalman model already fitted, without refitting it.

## Import video

**Data Import › Import video**:

| Field | What it does |
|---|---|
| Source type | one video file, or a directory of videos |
| Source | the file or folder |
| Directory format | which kind of file to take from the folder: `mp4`, `avi`, `mov`, `mkv`, `webm` or `m4v` |
| Search subdirectories | include videos in subfolders |
| Create symbolic links instead of copying | on by default: the project links to your original videos instead of copying them, so it stays small. Links are made relative so the project can move. |

The table below the form lists the videos already imported. A video already in the project is
skipped, not overwritten.

*Remove selected…* takes sessions out of the project. Their videos (the links; your footage is not
touched) and their calibration rows are moved to `backups/` with a manifest to undo it, and
optionally their imported pose files too. Nothing is deleted, and derived results are left as
they are.

Videos go into `sources/videos/`. Subfolders are fine; every page and `mufasa list videos` look in
them.

## Re-import skeleton model

When you retrain the tracker with new markers (eyes, shoulders, …), the project's markers and
skeleton must follow. **Data Import › Re-import skeleton model**:

1. *Browse…* to a pose file from the retrained tracker.
2. The form reports what would change: markers added or removed, edges of the skeleton.
3. *Add new markers to the Kalman v2 segments* also places the new markers in the smoother's body
   segments, so they are smoothed with the body.
4. **Apply** writes the change to `project.toml`.

Re-import the pose files afterwards, and re-run smoothing and features.

## Data inspector

**Data inspector** answers "did that work?" without opening a notebook. It lists the project's
data files by stage:

* imported pose;
* smoothed;
* outlier-corrected;
* features;
* classifications.

Select one to see its first rows, its size and how much of the sample is empty (NaN). Only the
start of the file is read, so even long sessions open at once. *Reload* re-reads the list after a
step has run.

Typical checks:

* after importing: each marker has its `_x`, `_y` and likelihood columns, and they are not empty;
* after smoothing: the smoothed file has no NaN where the raw one had gaps;
* after computing features: the families you chose are there.

From a terminal: `mufasa info FILE` describes a file, and `mufasa list pose` lists the pose files
([chapter 13](13-command-line.md)).

## Model modifications

**Model modifications › Rename markers** renames the project's markers. Edit the *new name* column
and press **Save model**. Mufasa first shows what will change (a dry run), then, once you confirm,
renames the marker in:

* the project's body parts;
* the skeleton, both nodes and edges;
* the imported pose files.

A snapshot is taken in `backups/` first. Features embed marker names, so if features already exist
you are warned; compute them again after a rename. *Reload markers* re-reads the list.

**File › Reconfigure project from DLC file…** is the other route: it rebuilds the project's marker
list from a DeepLabCut file.
