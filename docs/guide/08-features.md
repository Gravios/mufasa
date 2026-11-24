# 8. Features

Features are the numbers computed from the pose for every frame: distances between markers,
angles, the body's area, how far each marker moved. Classifiers learn behaviour from them, and you
can look at them directly to see what an animal is doing.

## Compute features

**Features › Compute Features**: tick the families you want and **Run**. The families come in three
groups:

**Subject features** (within one animal):

| Family | Columns, per frame |
|---|---|
| Two-point body-part distances (mm) | the distance between each pair of markers |
| Within-animal three-point body-part angles (degrees) | the angle at the middle marker of each triplet |
| Within-animal three-point / four-point convex hull perimeters (mm) | the perimeter round each group of three or four markers |
| Entire animal convex hull perimeters (mm) · area (mm²) | the perimeter and area round all the animal's markers |
| Frame-by-frame body-part movements (mm) | how far each marker moved since the last frame |
| Body-part distances to video frame edge (mm) | how close each marker is to the image's edges |
| Head velocity in body frame (mm/s) | the head's velocity along the body→head axis (forward) and across it (left positive). Signed. |

**ROI features** need ROIs drawn first ([chapter 7](07-roi.md)):

| Family | Columns |
|---|---|
| Frame-by-frame body-part distances to ROI centers (mm) | each marker's distance to each ROI's centre |
| Frame-by-frame body-parts inside ROIs (Boolean) | whether each marker is inside each ROI |

**3-D features** need a 3-D pose file for the video ([chapter 5](05-3d-pose.md)):

| Family | Columns |
|---|---|
| 3-D body-part heights (mm) | each marker's height above the floor |
| 3-D two-point body-part distances (mm) | 3-D distances between marker pairs |
| 3-D frame-by-frame body-part movements (mm) | 3-D distance moved since the last frame |
| 3-D body posture angles (degrees) | head pitch and roll, body pitch, the spine's bend at each joint, pelvis pitch, lumbar angle |

*3-D pose from* chooses which 3-D run to read: by default the newest one that has the video, then
imported motion capture.

Other options:

* *Parallel workers*: how many videos are computed at once.
* *Run file checks before computing*: check first that every video has its pose and calibration.

**Output:** `derived/features/<family>/<video>.parquet`, one folder per family and one file per
video, one row per frame. Distances are in mm when the videos are calibrated
([chapter 4](04-preprocessing.md#video-calibration)); otherwise they are in pixels.

The features read the pose from `derived/outlier_corrected/`: the latest smoothing (or
interpolation, or outlier correction) run.

**Which markers count as head and body** for *Head velocity in body frame* is set in
`project.toml` (`head_segment`, `body_segment`). Otherwise it is guessed from the skeleton or the
marker names. See [`../v1_project_layout.md`](../v1_project_layout.md).

## Look at features

**Features › Feature Visualization**: choose a video and tick features in the tree. Each feature
family is a branch: tick a family to take all of its features. The search box filters across
families, and *Group by name parts* groups features by the parts of their names. Then:

| Button | What you see |
|---|---|
| Distributions | a histogram per feature, for this video or pooled over all videos. With *Colour by* set to a behaviour, labelled frames are drawn against the rest, and each panel shows the feature's AUC for that behaviour (how well it alone separates the behaviour) |
| Joint (2 features) | two features as a 2-D density, a scatter, or both, with behaviour frames overlaid. *Swap X ↔ Y* swaps the axes |
| Over time | the whole session per feature, with behaviour bouts shaded; double-click a time to open the video there |
| Open with video ▸ | the feature viewer (below) |
| Landscape… | `mufasa-landscape` on the selection: joint PDFs of feature pairs whose joint distribution has several modes, and t-SNE over all sessions |
| Animate… | `mufasa-landscape-animate`: a GIF or MP4 of frames moving between the t-SNE map and each feature pair |

Your selection (and its order), the video and the plot settings are remembered between sessions
in `logs/measures/feature_view.toml`. Only the selected columns are read, so this stays quick with
hundreds of features.

## The feature viewer

**Open with video ▸** opens a window with the video and, under it, the selected features as
traces that follow the playback:

* **Moving around.** Click or drag on the traces to move through the video; the wheel zooms the
  time axis.
* **Histograms** beside each trace show the feature's distribution in three ways: over the whole
  video, within the window around the current frame (*Window ±*), and per behaviour. Click a bin
  to jump to the next frame with a value in it.
* **JPDF beside the video** shows the joint distribution of two features (*x:*, *y:*) beside the
  picture, with the current frame and a *Trail* of the frames before it.
* **Markers on the video.** *FDLC markers* and *Kalman markers* draw the raw and smoothed pose,
  and *Skeleton* draws the skeleton. *p ≥* hides points below a likelihood, *Pose scale* sets
  their size, and *Style…* sets the colours.
* **3-D…** shows the 3-D view of the same session beside the video, on the same playback, if it
  has a 3-D file.
* **● Record** records what is shown as it plays:
  * with **All panels** (on by default), every panel laid out as on the screen, at the video's
    resolution: the video, the JPDF, the 3-D view, the traces and the histograms;
  * off, the video alone.

  *Keep frames* also keeps the frames as images. Recordings go under `derived/frames/`.
* *Follow selection*: when you tick or untick features in Feature Visualization, the open viewer
  follows.

The window can be made narrow; its toolbars wrap onto more rows.

## Exploring without labels

`mufasa-landscape` (Landscape…) gives a map of a whole project for unsupervised discovery. It loads
the smoothed pose and features of every session, finds feature pairs whose joint distribution has
several modes (dip-test splitting), plots their joint PDFs, and runs t-SNE on a sample of frames.
It needs the `analysis` extra. [`../workflow_recipes.md`](../workflow_recipes.md) recipe R8 is a
worked example.
