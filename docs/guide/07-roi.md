# 7. Regions of interest

A region of interest (ROI) is a named area of the arena: a nest, an open arm, an object, a feeder.
The **ROI** page draws ROIs on the videos, measures time in and entries into them, renders them
over the video, and turns them into features for the classifiers.

## Draw ROIs

**ROI › Definitions** is the ROI editor, inside the page. Choose a video, then a tool:

| Tool | Key | How |
|---|---|---|
| Select | `V` | click an ROI to select it (click again on the same spot to cycle through overlapping ones). Drag the inside to move it; drag a corner, side, rim or vertex to reshape it. On a polygon, double-click an edge to add a vertex and right-click a vertex to remove it. Arrow keys nudge it by 1 px (`Shift`: 10); `Delete` removes it; `Esc` deselects. |
| Rectangle | `R` | press and drag |
| Circle | `C` | press at the centre and drag outward |
| Polygon | `P` | click each vertex; finish by clicking the first vertex, double-clicking, or pressing `Enter`. `Backspace` removes the last vertex. |

The editor also has:

* a list of the video's ROIs, where you edit each one's *Name*, *Colour* and line *Width*.
  *Duplicate* copies an ROI; *Delete* removes it;
* *Labels*, to show the ROI names on the frame, and *Fit* (`F`), to fit the frame to the view;
* *Undo* and *Redo*;
* frame stepping, to find a frame where the arena is clear: `,` and `.` for one frame back or
  forward, `Shift` with them for one second.

When every video has the same arena:

* **Apply selected ROI to all…** copies the selected ROI to every video;
* **Copy all ROIs to chosen videos…** copies this video's ROIs to the videos you pick. That is the
  way to handle two set-ups: draw one condition's ROIs, copy them to its videos, then do the same
  for the other;
* **Import / standardize…** imports ROIs from a CSV, or makes same-named ROIs the same size across
  videos (for cameras that moved slightly);
* **Clear this video…** removes this video's ROIs.

*Save* (or *Save & close*) writes them to `logs/measures/rois.json`, one entry per video.

## Analyse

**ROI › Analyze**:

| Option | What it does |
|---|---|
| Analysis mode | *Simple* (time in each ROI, entries), *Aggregate* (also first and last entry, bout durations), *Time bins* (all of it per time bin) |
| Body-parts | which marker decides whether the animal is in an ROI (often the body centre or the nose) |
| Probability threshold | frames where that marker's likelihood is lower are left out |
| Bin size | the time bin, for *Time bins* |
| Measures | *Total time*, *Entry counts*, *First entry time*, *Last entry time*, *Mean bout time*, *Outside-ROI time* |
| *Include detailed bout data* | every bout, not only the totals |
| *Compute ROI-distance metrics* | each marker's distance to each ROI too |

The results are written as CSV tables, per video and in total.

## Visualise

**ROI › Visualize** renders the ROIs over the videos:

| Option | What it does |
|---|---|
| Overlay type | *ROI tracking* (the ROIs and the animal's position, with running counts) or *ROI features* (the ROI features over the video) |
| Video | one video, or all |
| Body-parts · Probability threshold | which markers to draw, and from what likelihood |
| *Show animal names* · *Show body-part dots* · *Show bounding boxes* | what else to draw |
| Worker cores · *Use GPU (NVENC) if available* | speed |

## ROI features

**ROI › Features** adds per-frame ROI features to the feature files, so classifiers can use them:
for example, the distance to an object, or whether the nose is inside the nest.

| Action | What it does |
|---|---|
| Append (by animal) | the features for each animal's reference marker |
| Append (by body-part) | the features for the markers you pick |
| Remove | take the ROI features out again |

The **Features** page also has two ROI families, *frame-by-frame body-part distances to ROI
centers* and *frame-by-frame body-parts inside ROIs* ([chapter 8](08-features.md)).

[`../workflow_recipes.md`](../workflow_recipes.md) recipe R3 is a worked example:
ROI-bounded behaviour, per-video against reference ROIs.
