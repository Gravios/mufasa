# 11. Analysis and visualizations

**Analysis** computes statistics from the pose, the classifications and the ROIs. **Visualizations**
makes plots and videos from them. Each is a single form: choose what you want from the first
drop-down, and the rest of the form shows only that choice's options.

## Analysis

**Analysis › Run analysis** › *Analysis*:

| Analysis | What it gives |
|---|---|
| Classifier descriptive statistics | per video and behaviour: bout count, total time, mean and median bout length, first occurrence, mean interval between bouts |
| Classifier by time bins | the same per time bin (*Bin length*), to see how behaviour changes over a session |
| Classifier by ROI | each behaviour's time and bouts inside each ROI |
| Severity analysis | how severe each bout of a behaviour was (e.g. attacks), from the animals' movement during it, in a number of severity grades (*Severity mode*) |
| Movement analysis | distance moved and velocity per animal, from the chosen body-parts |
| Movement analysis by time bins | the same per time bin |
| Distance between body-parts | the distance between chosen body-parts (of one or two animals) through the session |
| Distance by time bins | the same per time bin |
| FSTTC (forward spike time-tiling) | how often one behaviour follows another within a time window, beyond chance: the temporal coupling between behaviours |
| Boolean conditional slicing | the time and frames in which a combination of conditions holds (behaviour A and not B, inside an ROI, …) |
| Directing toward other animals — statistics | how long each animal's head points toward each other animal |
| Directing toward body-part — statistics | how long the head points toward a chosen body-part (set in **Add-ons › Directing — body-part settings**) |

The options that appear depending on the choice:

* *Classifiers*, *Body-parts*: what to analyse;
* *Bin length*: for the time-bin analyses;
* *Severity mode*: for the severity analysis;
* *Cores*: how many videos at once;
* *Parameters*: anything else the analysis needs.

The results are CSV tables, per video and in total.

## Visualizations

**Visualizations › Create visualisation** › *Visualisation*. Each makes images, a video, or both
(*Outputs*: *Save frames*, *Save video*, *Save last-frame summary*), for one *Video* or all of them,
into *Save to*:

**Where the animal goes**

| Visualisation | What it shows |
|---|---|
| Heatmap — location | where the animal spent its time, as a heatmap over the arena |
| Heatmap — classifier | where a behaviour happened |
| Path plot | the animal's path through the session, optionally coloured by behaviour |
| Path plot — quick (one file) | a path plot from one pose file, without a project |
| Distance plot | the distance between chosen body-parts over time |

**Behaviour**

| Visualisation | What it shows |
|---|---|
| Gantt (behaviour timeline) | each behaviour's bouts on a timeline |
| Classifier probability plot | each frame's probability for a behaviour, over time |
| Classifier results overlay (sklearn) | the classifications written over the video |
| Validation video (single model) | one model's calls over a video, as in **Validate** |
| Classifier validation clips | short clips of each predicted bout, to check by eye |
| Annotated bouts overlay | your labels written over the video |
| Interactive probability inspector | step through a video with the probability plot beside it |

**Pose and data over the video**

| Visualisation | What it shows |
|---|---|
| Pose tracking (pose over videos) | the markers over the video |
| Skeleton video (project pose) | the skeleton drawn over the video, or on its own |
| Data plotter (feature overlay) | chosen features as numbers over the video |
| Circular feature overlay (per video) | circular (angle) features drawn over the video |
| Geometry overlay (Shapely objects) | geometric shapes (hulls, areas) over the video |
| Directing — toward other animals · toward body-part | lines from each animal's head to what it points at |

**ROIs and tasks**

| Visualisation | What it shows |
|---|---|
| ROI overlay (per video) · ROI feature overlay (per video) | the ROIs, and the ROI features, over the video |
| Cue light visualizer | cue-light states over the video ([chapter 12](12-video-addons-tools.md#add-ons)) |
| Spontaneous alternation plot | arm entries and alternations in a maze |

**Other trackers' output**

| Visualisation | What it shows |
|---|---|
| YOLO pose predictions (external) · YOLO pose+track predictions (external) · YOLO segmentation predictions (external) | a YOLO model's output over the video |
| YOLO annotation visualizer (training data) | a YOLO training set's labels on its images |
| Blob tracker output (external) · Blob plotter (simple) | blob-tracking results |

*Cores* and *Use GPU* speed up the videos (on the GPU, the encoding uses NVENC).

## Looking at a session interactively

For exploring rather than producing figures, the viewers are quicker:

* **the feature viewer** (**Features › Feature Visualization › Open with video ▸**): the video
  with feature traces, histograms and a JPDF ([chapter 8](08-features.md#the-feature-viewer));
* **the 3-D viewer** (**3-D › 3-D viewer**): the 3-D pose with the video, features, gait and
  models ([chapter 5](05-3d-pose.md#the-3-d-viewer)).

Both record what they show as a video (**● Record**).
