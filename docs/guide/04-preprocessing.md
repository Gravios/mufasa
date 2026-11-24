# 4. Preprocessing

The **Preprocessing** page gets videos and pose ready for everything after it. Its sections run in
this order:

```text
Preprocess Videos → Video Calibration → Interpolate missing frames → Kalman v2 smoothing
                                                 (or Run outlier correction) → Egocentric alignment
```

For most projects, the steps that matter are **Video Calibration** and **Kalman v2 smoothing**.

## Preprocess Videos

A step-by-step editor for raw videos, applied before anything else, since it changes the
pixels. For each video, in order:

1. crop;
2. downsample;
3. greyscale;
4. flip or rotate;
5. clip to a time range;
6. change the frame rate;
7. CLAHE contrast enhancement.

* *Apply to all* copies a setting from one video to every video.
* *Reset crop* and *Reset all* undo the edits.
* *Pop out ⇱* opens the editor in its own window.

Do this **before** Video Calibration: downsampling and cropping change the pixel scale.

## Video Calibration

Every distance Mufasa reports (millimetres, mm/s) depends on each video's **pixels per
millimetre**. Without it, distances come out in pixels.

**Preprocessing › Video Calibration** is a table with one row per video: frame rate, resolution,
the known distance in mm, and pixels/mm.

1. *Reload from project* fills the table from the videos.
2. Type the known distance in the row's *Distance (mm)* cell: a cage wall, a ruler in the frame.
3. *Calibrate…* opens a frame of that video. Click the two ends of that distance, and pixels/mm is
   computed.
4. If every video was filmed with the same camera set-up, *Apply row 1 px/mm to all* fills the
   other rows. *Apply row 1 distance to all* copies only the distance, to calibrate each video
   from it.
5. *Save to video_info.csv* stores it in `sources/video_info.csv`.

The menu bar's **Tools** › *Calibrate pixels/mm…* does the same for a single video outside the table.

## Interpolate missing frames

Fills frames where the tracker lost a marker or was unsure:

| Option | Choices |
|---|---|
| Interpolate by | *Body-parts*: fill each missing marker; *Animals*: fill frames where the whole animal is missing |
| Method | *Nearest* (fastest), *Linear*, *Quadratic* |

The result is a run under `derived/interpolated/`, published to `derived/outlier_corrected/` for
the later steps. Kalman v2 smoothing fills gaps by itself, so you need this only if you don't use
it.

## Kalman v2 smoothing

The recommended smoother. It learns your animal's body from your own data and uses it to smooth
every session:

* each body segment's size and how it moves;
* each marker's noise;
* each marker's bias (a tracker that is always a few pixels off).

It fills gaps, discounts low-likelihood points instead of trusting them, and keeps the markers of
a segment together.

**Preprocessing › Kalman v2 smoothing** works with the defaults. Its options:

| Option | What it does |
|---|---|
| Device | `cuda` (the GPU), `cpu`, or `auto` (the GPU if there is one) |
| Input dir / Output dir | the pose to smooth, and where to write (by default a new run in `derived/smoothed/kalman_v2/`) |
| Frame rate | the videos' frame rate |
| Likelihood threshold | points below this likelihood count as missing |
| Worker processes | sessions smoothed in parallel |
| Model | *Train new model*: fit the body model to these sessions with EM, then smooth them. *Load saved model*: smooth with a model fitted before (no fitting; much faster). |
| Model file | the saved model to load, or where to save the new one |
| Files for EM, *Train on a subset of input files* | fit on a few representative sessions instead of all, which is faster |
| EM iterations, EM tolerance | how long the fit runs |
| *Fit perspective model* | correct for the camera's perspective (an animal near the camera looks bigger) |
| *Show advanced fine tuning* | the rest: orientation-drift and constant-acceleration segments, noise ceilings, damping. The defaults suit most data; see the design doc. |

The result is a run under `derived/smoothed/kalman_v2/`, published to `derived/outlier_corrected/`.
A fitted model is saved twice:

* in the project, as `models/<name>/model.npz` with a `card.toml` saying where it came from;
* in your own library, `~/.config/mufasa/models/`, so other projects can load it.

`mufasa list models --kind smoothing` lists them.

**Recommended workflow:**

1. Train a model on 5–10 representative sessions.
2. Smooth every session with *Load saved model*.
3. Smooth new sessions as they come with **Data Import › Import pose data** › *Add sessions* (which reuses the latest
   model).

**Without the workbench:**

```bash
python -m mufasa.data_processors.kalman_pose_smoother_v2 session.parquet --beside-input
python -m mufasa.data_processors.kalman_pose_smoother_v2 new.parquet --load-model models/rat_body/model.npz
```

It takes CSV or parquet pose files and writes `<stem>_smoothed_v2.<hash>.parquet` beside the
input. `--save-model` saves the fitted model.

How it works, and why: [`../kalman_smoother_design.md`](../kalman_smoother_design.md).

## Run outlier correction

The classic SimBA outlier correction, as an alternative to Kalman smoothing. It has two stages,
each with its own tick box:

* **Movement correction**: a marker that jumped further between frames than the movement
  criterion is put back where it was.
* **Location correction**: a marker further from its reference marker than the location criterion
  (a swapped or misplaced marker) is put back.

The criteria are set in **Preprocessing › Advanced / legacy** › *Outlier settings*, as multiples of a reference
distance between two of the animal's markers. Output: `derived/outlier_corrected/<run>/`. With
Kalman-smoothed data, movement correction rarely helps. A generous location criterion still catches
swapped markers.

## Egocentric alignment

Rotates the pose (and, if you choose, the videos) so that every frame is centred on one marker and
faces one direction. This is useful for comparing postures independently of where the animal is
and which way it faces.

| Option | What it does |
|---|---|
| Center anchor | the marker put at the origin (default: the one named like "center") |
| Direction anchor | the marker that sets the heading (default: the one named like "nose") |
| Direction angle | the heading it is turned to, in degrees (0 = right, 90 = up) |
| *Also rotate videos* | rotate the videos frame by frame as well (slow) |
| Border fill color | the colour of the corners a rotated video leaves |
| *Use GPU (CUDA)* | rotate on the GPU |

## Advanced / legacy

* **Smoothing**: Savitzky–Golay or Gaussian smoothing over a time window. It is quick, but it knows
  nothing of the body; use Kalman v2 instead.
* **Outlier settings**: the criteria and reference markers used by *Run outlier correction*.
* **Drop body-parts**: remove markers from every pose file of the project.
